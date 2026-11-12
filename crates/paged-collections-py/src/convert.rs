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
use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicI32, AtomicI64, AtomicU8, AtomicU32, AtomicU64};

/// The key type every Python collection uses.
pub(crate) type Key = u32;

/// The key `obj` names, for a lookup: `None` if `obj` is not an integer in
/// range, so the lookup finds nothing.
pub(crate) fn lookup_key(obj: &Bound<'_, PyAny>) -> Option<Key> {
    obj.extract::<Key>().ok()
}

/// The key `obj` names, for an insertion: `OverflowError` if it is an
/// integer out of range, `TypeError` if it is not an integer.
pub(crate) fn insert_key(obj: &Bound<'_, PyAny>) -> PyResult<Key> {
    obj.extract::<Key>()
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
pub(crate) struct KeyBatch<'py> {
    /// The keys as given (after `asarray`), for error messages.
    source: Bound<'py, PyUntypedArray>,
    keys: Keys<'py>,
}

enum Keys<'py> {
    /// A `uint32` array, read in place.
    Native(PyReadonlyArray1<'py, Key>),
    /// Any other integer array, converted; `None` marks a value out of range.
    Converted(Vec<Option<Key>>),
}

impl<'py> KeyBatch<'py> {
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
        if let Ok(native) = source.cast::<PyArray1<Key>>() {
            if is_aligned(native.as_any())? {
                let keys = Keys::Native(native.try_readonly().map_err(borrow_error)?);
                return Ok(Self { source, keys });
            }
        }
        let dtype = source.dtype();
        let keys = match dtype.kind() {
            b'i' => Keys::Converted(
                cast_to::<i64>(&source, "int64")?
                    .into_iter()
                    .map(|k| Key::try_from(k).ok())
                    .collect(),
            ),
            b'u' => Keys::Converted(
                cast_to::<u64>(&source, "uint64")?
                    .into_iter()
                    .map(|k| Key::try_from(k).ok())
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

    /// Number of keys in the batch.
    pub(crate) fn len(&self) -> usize {
        match &self.keys {
            Keys::Native(array) => array.len(),
            Keys::Converted(keys) => keys.len(),
        }
    }

    /// Calls `f` with each key in order, or `None` for a value out of range.
    pub(crate) fn for_each(&self, mut f: impl FnMut(Option<Key>)) {
        match &self.keys {
            Keys::Native(array) => array.as_array().iter().for_each(|&k| f(Some(k))),
            Keys::Converted(keys) => keys.iter().for_each(|&k| f(k)),
        }
    }

    /// The batch as keys for insertion: `OverflowError` names the first
    /// value out of range.
    pub(crate) fn to_insert_keys(&self) -> PyResult<Vec<Key>> {
        match &self.keys {
            Keys::Native(array) => Ok(array.as_array().to_vec()),
            Keys::Converted(keys) => keys
                .iter()
                .enumerate()
                .map(|(pos, &k)| {
                    k.ok_or_else(|| match self.item(pos) {
                        Ok(value) => PyOverflowError::new_err(format!(
                            "key {value} at position {pos} is out of range for uint32"
                        )),
                        Err(err) => err,
                    })
                })
                .collect(),
        }
    }

    /// The key at `pos` as given, as a Python scalar (`int` for integer
    /// arrays).
    pub(crate) fn item(&self, pos: usize) -> PyResult<Bound<'py, PyAny>> {
        self.source.as_any().get_item(pos)?.call_method0("item")
    }
}

/// The elements of `array.astype(dtype)` as `T` (which must match `dtype`).
fn cast_to<T: Element + Copy>(array: &Bound<'_, PyUntypedArray>, dtype: &str) -> PyResult<Vec<T>> {
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
