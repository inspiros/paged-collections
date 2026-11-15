//! Conversions from Python objects to keys and values.
//!
//! Keys are Python integers in `0..2**32`. Lookups treat any other object as
//! an absent key (as `dict` does for keys it does not hold); insertions raise
//! `OverflowError` or `TypeError` for it.
//!
//! Batch inputs go through `numpy.asarray`. Every batch is converted and
//! validated in full before a collection changes, so a bad batch leaves the
//! collection untouched.

use numpy::{
    Element, PyArray1, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods, PyReadonlyArray1,
    PyUntypedArray, PyUntypedArrayMethods,
};
use paged_collections::id_alloc::Id;
use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyInt;
use std::fmt;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicI32, AtomicI64, AtomicU8, AtomicU32, AtomicU64};

/// The key type of the integer-keyed collections.
pub(crate) type Key = u32;

/// A key type a Python collection can use: `u32` (integer keys) or
/// [`Id`] (generational ids from `IdAllocator`, as `to_bits` integers).
pub(crate) trait PyKey: paged_collections::key::Key + Copy + Eq + Ord + 'static {
    /// The integer a key is in Python and in NumPy arrays.
    type Bits: Element + Copy + fmt::Display + for<'py> IntoPyObject<'py>;

    /// What is wrong with an invalid key, for messages.
    const INVALID: &'static str;

    /// Whether a plain sequence that NumPy reads as float or object is
    /// converted element by element (ids need all 64 bits).
    const SEQUENCE_FALLBACK: bool = false;

    fn to_bits(self) -> Self::Bits;

    /// The key with these bits, if they are valid.
    fn from_bits(bits: Self::Bits) -> Option<Self>;

    fn from_i64(value: i64) -> Option<Self>;

    fn from_u64(value: u64) -> Option<Self>;

    /// The keys as their bits, without copying.
    fn bits_slice(keys: &[Self]) -> &[Self::Bits];

    /// The error for an integer that is not a valid key, for an insertion.
    fn invalid(message: String) -> PyErr;

    /// The key `obj` names, for a lookup: `None` if `obj` is not an integer
    /// that is a valid key, so the lookup finds nothing.
    fn lookup(obj: &Bound<'_, PyAny>) -> Option<Self> {
        obj.extract::<u64>().ok().and_then(Self::from_u64)
    }

    /// The key `obj` names, for an insertion: `TypeError` if it is not an
    /// integer, else `OverflowError` (u32) or `ValueError` (id) if it is not
    /// a valid key.
    fn insert(obj: &Bound<'_, PyAny>) -> PyResult<Self>;
}

impl PyKey for u32 {
    type Bits = u32;
    const INVALID: &'static str = "out of range for uint32";

    fn to_bits(self) -> u32 {
        self
    }

    fn from_bits(bits: u32) -> Option<Self> {
        Some(bits)
    }

    fn from_i64(value: i64) -> Option<Self> {
        u32::try_from(value).ok()
    }

    fn from_u64(value: u64) -> Option<Self> {
        u32::try_from(value).ok()
    }

    fn bits_slice(keys: &[u32]) -> &[u32] {
        keys
    }

    fn invalid(message: String) -> PyErr {
        PyOverflowError::new_err(message)
    }

    fn lookup(obj: &Bound<'_, PyAny>) -> Option<Self> {
        obj.extract::<u32>().ok()
    }

    fn insert(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        obj.extract::<u32>()
    }
}

impl PyKey for Id {
    type Bits = u64;
    const INVALID: &'static str = "not a valid id";
    const SEQUENCE_FALLBACK: bool = true;

    fn to_bits(self) -> u64 {
        Id::to_bits(self)
    }

    fn from_bits(bits: u64) -> Option<Self> {
        Id::from_bits(bits)
    }

    fn from_i64(value: i64) -> Option<Self> {
        u64::try_from(value).ok().and_then(Id::from_bits)
    }

    fn from_u64(value: u64) -> Option<Self> {
        Id::from_bits(value)
    }

    fn bits_slice(keys: &[Id]) -> &[u64] {
        // SAFETY: `Id` has size 8 and alignment 8 (at least that of `u64`),
        // and its bytes are those of `to_bits` in native order (its
        // documented layout), so every `Id` is a valid `u64`. The lifetime is
        // carried over.
        unsafe { std::slice::from_raw_parts(keys.as_ptr().cast::<u64>(), keys.len()) }
    }

    fn invalid(message: String) -> PyErr {
        PyValueError::new_err(message)
    }

    fn insert(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let bits = obj.extract::<u64>()?;
        Id::from_bits(bits)
            .ok_or_else(|| PyValueError::new_err(format!("{bits} is not a valid id")))
    }
}

/// The key `obj` names, for a lookup in an integer-keyed collection.
pub(crate) fn lookup_key(obj: &Bound<'_, PyAny>) -> Option<Key> {
    Key::lookup(obj)
}

/// The key `obj` names, for an insertion into an integer-keyed collection.
pub(crate) fn insert_key(obj: &Bound<'_, PyAny>) -> PyResult<Key> {
    Key::insert(obj)
}

/// `obj` as a base-class NumPy array, through `numpy.asarray`. Subclasses
/// become plain `ndarray`s, so their overrides (of `astype`, say) cannot
/// change what the bindings read.
pub(crate) fn as_ndarray<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyUntypedArray>> {
    let numpy = obj.py().import("numpy")?;
    Ok(numpy
        .call_method1("asarray", (obj,))?
        .cast_into::<PyUntypedArray>()?)
}

/// A batch of keys converted from a Python object.
pub(crate) struct KeyBatch<'py, K: PyKey = Key> {
    /// The keys as given (after `asarray`), for error messages.
    source: Bound<'py, PyUntypedArray>,
    keys: Keys<'py, K>,
}

enum Keys<'py, K: PyKey> {
    /// An array of the key's bits dtype, read in place.
    Native(PyReadonlyArray1<'py, K::Bits>),
    /// Any other integer array, converted; `None` marks an invalid value.
    Converted(Vec<Option<K>>),
}

impl<'py, K: PyKey> KeyBatch<'py, K> {
    /// Converts `obj` (a 1-D array or sequence of integers) to keys.
    pub(crate) fn new(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        let source = as_ndarray(obj)?;
        if source.ndim() != 1 {
            return Err(PyValueError::new_err(format!(
                "keys must be one-dimensional, got {} dimensions",
                source.ndim()
            )));
        }
        // `asarray([])` is float64: an empty batch is fine whatever its dtype.
        if source.is_empty() {
            return Ok(Self {
                source,
                keys: Keys::Converted(Vec::new()),
            });
        }
        if let Ok(native) = source.cast::<PyArray1<K::Bits>>() {
            if is_aligned(native.as_any())? {
                let keys = Keys::Native(native.try_readonly().map_err(borrow_error)?);
                return Ok(Self { source, keys });
            }
        }
        let dtype = source.dtype();
        // NumPy infers float64 (or object) for a sequence that mixes ints
        // above `i64::MAX` with others, so a plain sequence of ids is read
        // element by element instead, as Python ints.
        if K::SEQUENCE_FALLBACK
            && matches!(dtype.kind(), b'f' | b'O')
            && obj.cast::<PyUntypedArray>().is_err()
        {
            return Self::from_sequence(obj);
        }
        let keys = match dtype.kind() {
            b'i' => Keys::Converted(
                cast_to::<i64>(&source, "int64")?
                    .into_iter()
                    .map(K::from_i64)
                    .collect(),
            ),
            b'u' => Keys::Converted(
                cast_to::<u64>(&source, "uint64")?
                    .into_iter()
                    .map(K::from_u64)
                    .collect(),
            ),
            _ => {
                return Err(PyTypeError::new_err(format!(
                    "keys must be integers, got an array of dtype {dtype}"
                )));
            }
        };
        Ok(Self { source, keys })
    }

    /// Reads a 1-D sequence (not an array) element by element through
    /// `operator.index`; `TypeError` for any element that is not an integer.
    fn from_sequence(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        let source = obj
            .py()
            .import("numpy")?
            .call_method1("array", (obj, "object"))?
            .cast_into::<PyUntypedArray>()?;
        let index = obj.py().import("operator")?.getattr("index")?;
        let mut keys = Vec::with_capacity(source.len());
        for item in obj.try_iter()? {
            // `operator.index` accepts ints and NumPy integer scalars and
            // raises `TypeError` for anything else (floats included).
            let int = index.call1((item?,))?;
            keys.push(int.extract::<u64>().ok().and_then(K::from_u64));
        }
        if keys.len() != source.len() {
            return Err(PyValueError::new_err(
                "keys changed length while being read",
            ));
        }
        Ok(Self {
            source,
            keys: Keys::Converted(keys),
        })
    }

    /// Number of keys in the batch.
    pub(crate) fn len(&self) -> usize {
        match &self.keys {
            Keys::Native(array) => array.len(),
            Keys::Converted(keys) => keys.len(),
        }
    }

    /// Calls `f` with each key in order, or `None` for a value out of range.
    pub(crate) fn for_each(&self, mut f: impl FnMut(Option<K>)) {
        match &self.keys {
            Keys::Native(array) => array.as_array().iter().for_each(|&k| f(K::from_bits(k))),
            Keys::Converted(keys) => keys.iter().for_each(|&k| f(k)),
        }
    }

    /// The batch as keys for insertion: `OverflowError` names the first
    /// value out of range.
    pub(crate) fn to_insert_keys(&self) -> PyResult<Vec<K>> {
        let mut keys = Vec::with_capacity(self.len());
        let mut bad = None;
        self.for_each(|k| match k {
            Some(k) => keys.push(k),
            None => {
                bad.get_or_insert(keys.len());
            }
        });
        match bad {
            None => Ok(keys),
            Some(pos) => Err(match self.item(pos) {
                Ok(value) => K::invalid(format!("key {value} at position {pos} is {}", K::INVALID)),
                Err(err) => err,
            }),
        }
    }

    /// The key at `pos` as given, as a Python scalar (`int` for integer
    /// arrays).
    pub(crate) fn item(&self, pos: usize) -> PyResult<Bound<'py, PyAny>> {
        let item = self.source.as_any().get_item(pos)?;
        // Object arrays hold Python ints already.
        match item.getattr_opt("item")? {
            Some(method) if item.cast::<PyInt>().is_err() => method.call0(),
            _ => Ok(item),
        }
    }
}

/// The elements of `array.astype(dtype)` as `T` (which must match `dtype`).
pub(crate) fn cast_to<T: Element + Copy>(
    array: &Bound<'_, PyUntypedArray>,
    dtype: &str,
) -> PyResult<Vec<T>> {
    let cast = array
        .call_method1("astype", (dtype,))?
        .cast_into::<PyArrayDyn<T>>()?;
    let values = cast
        .try_readonly()
        .map_err(borrow_error)?
        .as_array()
        .iter()
        .copied()
        .collect();
    Ok(values)
}

/// Whether `array`'s data is aligned for its dtype; rust-numpy reads only
/// aligned arrays in place. Unaligned ones are converted with `astype`.
fn is_aligned(array: &Bound<'_, PyAny>) -> PyResult<bool> {
    array.getattr("flags")?.getattr("aligned")?.extract()
}

fn borrow_error(err: numpy::BorrowError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// A value type that typed maps store and expose as a NumPy dtype.
pub(crate) trait TypedValue: Element + Copy + PartialEq + 'static {
    /// The atomic integer that stores a value in a map (see
    /// [`crate::export::ValueCell`]).
    type Atomic: Send + Sync;

    /// `value` as an atomic.
    fn to_atomic(value: Self) -> Self::Atomic;

    /// Reads `cell`.
    fn load(cell: &Self::Atomic) -> Self;

    /// Writes `value` to `cell`.
    fn store(cell: &Self::Atomic, value: Self);

    /// The NumPy dtype name, for messages.
    const DTYPE: &'static str;

    /// Converts an array of any dtype to `Self`, refusing lossy kinds
    /// (floats into an integer map, anything but bool into a bool map) and
    /// integers out of range.
    fn convert(array: &Bound<'_, PyUntypedArray>) -> PyResult<Vec<Self>>;
}

fn wrong_kind(array: &Bound<'_, PyUntypedArray>, target: &str) -> PyErr {
    PyTypeError::new_err(format!(
        "cannot store values of dtype {} as {target}",
        array.dtype()
    ))
}

// Relaxed ordering: the GIL orders every access by Rust code, and NumPy
// accesses the same memory non-atomically anyway.
macro_rules! float_value {
    ($($t:ty => $dtype:literal, $atomic:ty),*) => {$(
        impl TypedValue for $t {
            const DTYPE: &'static str = $dtype;
            type Atomic = $atomic;

            fn to_atomic(value: Self) -> Self::Atomic {
                <$atomic>::new(value.to_bits())
            }

            fn load(cell: &Self::Atomic) -> Self {
                <$t>::from_bits(cell.load(Relaxed))
            }

            fn store(cell: &Self::Atomic, value: Self) {
                cell.store(value.to_bits(), Relaxed);
            }

            fn convert(array: &Bound<'_, PyUntypedArray>) -> PyResult<Vec<Self>> {
                match array.dtype().kind() {
                    b'b' | b'i' | b'u' | b'f' => cast_to::<$t>(array, $dtype),
                    _ => Err(wrong_kind(array, $dtype)),
                }
            }
        }
    )*};
}

macro_rules! int_value {
    ($($t:ty => $dtype:literal, $atomic:ty),*) => {$(
        impl TypedValue for $t {
            const DTYPE: &'static str = $dtype;
            type Atomic = $atomic;

            fn to_atomic(value: Self) -> Self::Atomic {
                <$atomic>::new(value)
            }

            fn load(cell: &Self::Atomic) -> Self {
                cell.load(Relaxed)
            }

            fn store(cell: &Self::Atomic, value: Self) {
                cell.store(value, Relaxed);
            }

            fn convert(array: &Bound<'_, PyUntypedArray>) -> PyResult<Vec<Self>> {
                let out_of_range = |value: &dyn core::fmt::Display| {
                    PyOverflowError::new_err(format!("value {value} is out of range for {}", $dtype))
                };
                match array.dtype().kind() {
                    b'b' | b'i' => cast_to::<i64>(array, "int64")?
                        .into_iter()
                        .map(|v| <$t>::try_from(v).map_err(|_| out_of_range(&v)))
                        .collect(),
                    b'u' => cast_to::<u64>(array, "uint64")?
                        .into_iter()
                        .map(|v| <$t>::try_from(v).map_err(|_| out_of_range(&v)))
                        .collect(),
                    _ => Err(wrong_kind(array, $dtype)),
                }
            }
        }
    )*};
}

float_value!(f64 => "float64", AtomicU64, f32 => "float32", AtomicU32);
int_value!(
    i64 => "int64", AtomicI64,
    i32 => "int32", AtomicI32,
    u64 => "uint64", AtomicU64,
    u32 => "uint32", AtomicU32
);

impl TypedValue for bool {
    const DTYPE: &'static str = "bool";
    /// A byte, read as `byte != 0`: NumPy can write any byte to the memory
    /// of a bool array, and only 0 and 1 are valid Rust `bool`s.
    type Atomic = AtomicU8;

    fn to_atomic(value: Self) -> Self::Atomic {
        AtomicU8::new(u8::from(value))
    }

    fn load(cell: &Self::Atomic) -> Self {
        cell.load(Relaxed) != 0
    }

    fn store(cell: &Self::Atomic, value: Self) {
        cell.store(u8::from(value), Relaxed);
    }

    fn convert(array: &Bound<'_, PyUntypedArray>) -> PyResult<Vec<Self>> {
        match array.dtype().kind() {
            // Read as bytes: a bool array can hold bytes other than 0 and 1
            // (written through a `uint8` view), which are not valid `bool`s.
            b'b' => Ok(cast_to::<u8>(array, "uint8")?
                .into_iter()
                .map(|byte| byte != 0)
                .collect()),
            _ => Err(wrong_kind(array, "bool")),
        }
    }
}

/// Converts `obj` to `n` values: a 1-D array or sequence of length `n`, or a
/// scalar, which is repeated.
pub(crate) fn value_batch<T: TypedValue>(obj: &Bound<'_, PyAny>, n: usize) -> PyResult<Vec<T>> {
    let array = as_ndarray(obj)?;
    match array.ndim() {
        0 => {
            let value = T::convert(&array)?[0];
            Ok(vec![value; n])
        }
        1 if array.len() != n => Err(PyValueError::new_err(format!(
            "got {} values for {n} keys",
            array.len()
        ))),
        // `asarray([])` is float64: allow it whatever the target dtype.
        1 if n == 0 => Ok(Vec::new()),
        1 => T::convert(&array),
        ndim => Err(PyValueError::new_err(format!(
            "values must be a scalar or one-dimensional, got {ndim} dimensions"
        ))),
    }
}
