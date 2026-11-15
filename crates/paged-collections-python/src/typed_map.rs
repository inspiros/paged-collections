//! `SparseMapF64` and friends: `PagedSparseMap<u32, T>` for NumPy dtypes `T`;
//! `IdMapF64` and friends: the same keyed by `Id`.

use crate::storage::MapInner;
use numpy::PyArray1;
use paged_collections::sparse_map::PagedSparseMap;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyKeyError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::convert::{Key, KeyBatch, PyKey, TypedValue, value_batch};
use crate::export::{Exports, ValueCell, cell_view, read_only_view};
use crate::iter::key_iterator;
use crate::repr_items;
use paged_collections::id_alloc::Id;

/// Writes `values` for `keys` (duplicates: the last wins) and returns how
/// many keys were new. With views alive, only keys already present may be
/// written, and the call fails before writing anything otherwise.
///
/// A key counts as new if the map did not hold it (exactly) before it was
/// written, so an id that evicts another generation of its slot is new,
/// although the length does not change.
pub(crate) fn write_batch<K: PyKey, V, const PAGE: usize>(
    map: &mut PagedSparseMap<K, V, PAGE>,
    exports: &Exports,
    type_name: &str,
    keys: Vec<K>,
    values: Vec<V>,
) -> PyResult<usize> {
    debug_assert_eq!(keys.len(), values.len());
    let new = keys.iter().any(|&k| !map.contains_key(k));
    if new {
        exports.check(type_name)?;
        map.try_reserve(keys.len()).map_err(crate::reserve_error)?;
    }
    let mut added = 0;
    for (key, value) in keys.into_iter().zip(values) {
        added += usize::from(!map.contains_key(key));
        map.insert(key, value);
    }
    Ok(added)
}

/// As [`write_batch`], for maps whose values are atomic cells that may be
/// exported: existing values are written with `ValueCell::set`, never through a
/// `&mut` (see the invariants in [`crate::export`]).
fn write_cells<K: PyKey, T: TypedValue, const PAGE: usize>(
    map: &mut PagedSparseMap<K, ValueCell<T>, PAGE>,
    exports: &Exports,
    type_name: &str,
    keys: Vec<K>,
    values: Vec<T>,
) -> PyResult<usize> {
    debug_assert_eq!(keys.len(), values.len());
    let new = keys.iter().any(|&k| !map.contains_key(k));
    if new {
        exports.check(type_name)?;
        map.try_reserve(keys.len()).map_err(crate::reserve_error)?;
    }
    let mut added = 0;
    for (key, value) in keys.into_iter().zip(values) {
        match map.get(key) {
            Some(cell) => cell.set(value),
            None => {
                added += 1;
                map.insert(key, ValueCell::new(value));
            }
        }
    }
    Ok(added)
}

/// The optional `default` of `pop`-like methods, given as `*args`.
pub(crate) fn optional_default<'py>(
    default: &Bound<'py, PyTuple>,
    method: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    match default.len() {
        0 => Ok(None),
        1 => Ok(Some(default.get_item(0)?)),
        n => Err(PyTypeError::new_err(format!(
            "{method} expected at most 2 arguments, got {}",
            n + 1
        ))),
    }
}

macro_rules! typed_map {
    ($name:ident, $iter:ident, $pyname:literal, $t:ty, $view:ident) => {
        typed_map!(
            $name, $iter, $pyname, $t, $view,
            Key, "integers in ``0 <= key < 2**32``", "uint32"
        );
    };
    (
        $name:ident, $iter:ident, $pyname:literal, $t:ty, $view:ident,
        $K:ty, $keydoc:literal, $keydtype:literal
    ) => {
        #[doc = concat!(
            "A map from ", $keydoc, " to ``",
            stringify!($t),
            "`` values, with O(1) insert, delete and lookup.\n\n",
            "Keys and values live in two parallel contiguous arrays (in insertion order ",
            "until a removal moves the last entry into the gap); the key index is paged, ",
            "so memory follows the key ranges in use.\n\n",
            "Batch methods (``set_many``, ``get_many``, ``contains_many``, ``discard_many``) ",
            "take NumPy arrays or sequences. ``keys_view()`` and ``values_view()`` return ",
            "zero-copy arrays; while any view is alive, only values of keys already present ",
            "can be written, and methods that would resize or reorder the map raise ",
            "``BufferError``."
        )]
        #[pyclass(module = "paged_collections._core", name = $pyname, subclass)]
        pub(crate) struct $name {
            /// Values are atomic cells so they can be exported to NumPy and still
            /// be written by Rust (see [`crate::export`]).
            map: MapInner<$K, ValueCell<$t>>,
            exports: Exports,
            /// Changes on every structural change, so iterators can detect them.
            mods: u64,
        }

        impl $name {
            fn empty() -> Self {
                Self {
                    map: MapInner::default(),
                    exports: Exports::default(),
                    mods: 0,
                }
            }

            fn changed(&mut self) {
                self.mods = self.mods.wrapping_add(1);
            }

            pub(crate) fn dense_keys(&self) -> &[$K] {
                self.map.key_slice()
            }

            fn set_batch(
                &mut self,
                keys: &Bound<'_, PyAny>,
                values: &Bound<'_, PyAny>,
            ) -> PyResult<usize> {
                let keys = KeyBatch::<$K>::new(keys)?.to_insert_keys()?;
                let values = value_batch::<$t>(values, keys.len())?;
                let new = paged!(MapInner, &mut self.map, map => write_cells(map, &self.exports, $pyname, keys, values))?;
                if new > 0 {
                    self.changed();
                }
                Ok(new)
            }

            fn remove_present(&mut self, key: $K) -> PyResult<$t> {
                self.exports.check($pyname)?;
                let value = self.map.remove(key);
                self.changed();
                Ok(value.expect("caller checked the key is present").get())
            }
        }

        #[pymethods]
        impl $name {
            // `__new__` accepts and ignores any arguments, so subclasses can
            // define their own `__init__`; `__init__` fills the map.
            #[new]
            #[pyo3(signature = (*_args, **_kwargs), text_signature = "(keys=None, values=None, *, page_size=DEFAULT_PAGE_SIZE)")]
            fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
                Self::empty()
            }

            #[pyo3(signature = (keys = None, values = None, *, page_size = crate::DEFAULT_PAGE_SIZE))]
            fn __init__(
                &mut self,
                keys: Option<&Bound<'_, PyAny>>,
                values: Option<&Bound<'_, PyAny>>,
                #[pyo3(from_py_with = crate::extract_page_size)] page_size: usize,
            ) -> PyResult<()> {
                if self.map.page_size() != page_size {
                    self.exports.check($pyname)?;
                    self.map.set_page_size(page_size)?;
                    self.changed();
                }
                match (keys, values) {
                    (None, None) => Ok(()),
                    (Some(keys), Some(values)) => self.set_batch(keys, values).map(drop),
                    _ => Err(PyTypeError::new_err("give both keys and values, or neither")),
                }
            }

            fn __len__(&self) -> usize {
                self.map.len()
            }

            fn __contains__(&self, key: &Bound<'_, PyAny>) -> bool {
                <$K as PyKey>::lookup(key).is_some_and(|k| self.map.contains_key(k))
            }

            fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<$t> {
                <$K as PyKey>::lookup(key)
                    .and_then(|k| self.map.get(k).map(ValueCell::get))
                    .ok_or_else(|| PyKeyError::new_err(key.clone().unbind()))
            }

            fn __setitem__(&mut self, key: &Bound<'_, PyAny>, value: $t) -> PyResult<()> {
                let key = <$K as PyKey>::insert(key)?;
                if let Some(cell) = self.map.get(key) {
                    cell.set(value);
                } else {
                    self.exports.check($pyname)?;
                    self.map.insert(key, ValueCell::new(value));
                    self.changed();
                }
                Ok(())
            }

            fn __delitem__(&mut self, key: &Bound<'_, PyAny>) -> PyResult<()> {
                match <$K as PyKey>::lookup(key).filter(|&k| self.map.contains_key(k)) {
                    Some(k) => self.remove_present(k).map(drop),
                    None => Err(PyKeyError::new_err(key.clone().unbind())),
                }
            }

            fn __iter__(slf: &Bound<'_, Self>) -> PyResult<$iter> {
                $iter::new(slf)
            }

            fn __eq__(&self, other: PyRef<'_, Self>) -> bool {
                self.map == other.map
            }

            fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
                let py = slf.py();
                let this = slf.try_borrow()?;
                let items = repr_items(this.map.iter().map(|(&k, v)| (k.to_bits(), v.get())), py)?;
                Ok(format!("{}({{{items}}}{})", slf.get_type().name()?, crate::page_size_repr(slf.try_borrow()?.map.page_size())))
            }

            fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
                let py = slf.py();
                let this = slf.try_borrow()?;
                let contents = vec![this.keys(py).into_any(), this.values(py).into_any()];
                crate::reduce(slf.as_any(), &py.get_type::<Self>(), contents, None)
            }

            /// Return a copy (it has no views).
            fn copy(&self) -> Self {
                Self {
                    map: self.map.clone(),
                    exports: Exports::default(),
                    mods: 0,
                }
            }

            /// Return the value for ``key``, or ``default`` if it is absent.
            #[pyo3(signature = (key, default = None))]
            fn get<'py>(
                &self,
                py: Python<'py>,
                key: &Bound<'py, PyAny>,
                default: Option<Bound<'py, PyAny>>,
            ) -> PyResult<Bound<'py, PyAny>> {
                match <$K as PyKey>::lookup(key).and_then(|k| self.map.get(k)) {
                    Some(value) => value.get().into_bound_py_any(py),
                    None => Ok(default.unwrap_or_else(|| py.None().into_bound(py))),
                }
            }

            /// Remove ``key`` and return its value. If it is absent, return
            /// ``default`` if given, else raise ``KeyError``.
            #[pyo3(signature = (key, *default))]
            fn pop<'py>(
                &mut self,
                py: Python<'py>,
                key: &Bound<'py, PyAny>,
                default: &Bound<'py, PyTuple>,
            ) -> PyResult<Bound<'py, PyAny>> {
                let default = optional_default(default, "pop")?;
                match <$K as PyKey>::lookup(key).filter(|&k| self.map.contains_key(k)) {
                    Some(k) => {
                        self.remove_present(k)?.into_bound_py_any(py)
                    }
                    None => default.ok_or_else(|| PyKeyError::new_err(key.clone().unbind())),
                }
            }

            /// Remove and return the last ``(key, value)`` pair in dense order;
            /// raise ``KeyError`` if the map is empty.
            fn popitem(&mut self) -> PyResult<(<$K as PyKey>::Bits, $t)> {
                let &key = self.map.key_slice().last().ok_or_else(|| {
                    PyKeyError::new_err(concat!("popitem(): ", $pyname, " is empty"))
                })?;
                Ok((key.to_bits(), self.remove_present(key)?))
            }

            /// Return the value for ``key``, first inserting ``default`` if
            /// ``key`` is absent.
            fn setdefault(&mut self, key: &Bound<'_, PyAny>, default: $t) -> PyResult<$t> {
                let key = <$K as PyKey>::insert(key)?;
                if let Some(value) = self.map.get(key) {
                    return Ok(value.get());
                }
                self.exports.check($pyname)?;
                self.map.insert(key, ValueCell::new(default));
                self.changed();
                Ok(default)
            }

            /// Insert the pairs of ``other``: a mapping, or an iterable of
            /// ``(key, value)`` pairs. Like ``dict.update``, a failure part way
            /// leaves the earlier pairs inserted; use ``set_many`` for an
            /// all-or-nothing batch.
            fn update(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<()> {
                if other.is(slf) {
                    return Ok(());
                }
                let pairs = if other.hasattr("keys")? {
                    other.call_method0("items")?
                } else {
                    other.clone()
                };
                for pair in pairs.try_iter()? {
                    let (key, value): (Bound<'_, PyAny>, $t) = pair?.extract()?;
                    slf.try_borrow_mut()?.__setitem__(&key, value)?;
                }
                Ok(())
            }

            /// Remove every entry. Pages stay allocated for reuse.
            fn clear(&mut self) -> PyResult<()> {
                if !self.map.is_empty() {
                    self.exports.check($pyname)?;
                    self.map.clear();
                    self.changed();
                }
                Ok(())
            }

            #[doc = concat!("Return the keys as a new ``", $keydtype, "`` array, in dense order.")]
            fn keys<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<<$K as PyKey>::Bits>> {
                PyArray1::from_slice(py, <$K as PyKey>::bits_slice(self.map.key_slice()))
            }

            #[doc = concat!("Return the values as a new ``", stringify!($t), "`` array, in dense order.")]
            fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<$t>> {
                PyArray1::from_vec(py, self.map.values().map(ValueCell::get).collect())
            }

            /// Return a list of ``(key, value)`` pairs in dense order.
            fn items(&self) -> Vec<(<$K as PyKey>::Bits, $t)> {
                self.map.iter().map(|(&k, v)| (k.to_bits(), v.get())).collect()
            }

            #[doc = concat!(
                "Return a zero-copy, read-only ``", $keydtype, "`` array of the keys in ",
                "dense order. See the class docs for what views prevent."
            )]
            fn keys_view<'py>(
                slf: &Bound<'py, Self>,
            ) -> PyResult<Bound<'py, PyArray1<<$K as PyKey>::Bits>>> {
                let this = slf.try_borrow()?;
                let keys = <$K as PyKey>::bits_slice(this.map.key_slice());
                // SAFETY: the keys are the map's dense key `Vec` (seen as
                // their bits), and every method that could move them checks
                // `exports` first.
                unsafe { read_only_view(slf.as_any(), &this.exports, keys) }
            }

            /// Return a zero-copy array of the values in dense order:
            /// ``values_view()[i]`` belongs to ``keys_view()[i]``. Writable,
            /// except for bool maps. See the class docs for what views
            /// prevent.
            fn values_view<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyArray1<$t>>> {
                typed_map!(@view $view, slf)
            }

            /// Set the value of each key in ``keys`` to the matching entry of
            /// ``values`` (a sequence of the same length, or a scalar for
            /// all), and return how many keys were new. For a key listed
            /// twice, the last value wins.
            ///
            /// The batch is validated first: a bad key or value raises and
            /// leaves the map unchanged.
            fn set_many(
                &mut self,
                keys: &Bound<'_, PyAny>,
                values: &Bound<'_, PyAny>,
            ) -> PyResult<usize> {
                self.set_batch(keys, values)
            }

            /// Return an array of the values of ``keys``. A missing key takes
            /// ``default`` if given, else raises ``KeyError``.
            #[pyo3(signature = (keys, default = None))]
            fn get_many<'py>(
                &self,
                py: Python<'py>,
                keys: &Bound<'py, PyAny>,
                default: Option<$t>,
            ) -> PyResult<Bound<'py, PyArray1<$t>>> {
                let batch = KeyBatch::<$K>::new(keys)?;
                let mut out = Vec::with_capacity(batch.len());
                let mut missing = None;
                batch.for_each(|k| match (k.and_then(|k| self.map.get(k)), default) {
                    (Some(value), _) => out.push(value.get()),
                    (None, Some(value)) => out.push(value),
                    (None, None) => {
                        missing.get_or_insert(out.len());
                        out.push(<$t>::default());
                    }
                });
                match missing {
                    Some(pos) => Err(PyKeyError::new_err(batch.item(pos)?.unbind())),
                    None => Ok(PyArray1::from_vec(py, out)),
                }
            }

            /// Return a boolean array: whether each key in ``keys`` is present.
            fn contains_many<'py>(
                &self,
                py: Python<'py>,
                keys: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyArray1<bool>>> {
                let batch = KeyBatch::<$K>::new(keys)?;
                let mut out = Vec::with_capacity(batch.len());
                batch.for_each(|k| out.push(k.is_some_and(|k| self.map.contains_key(k))));
                Ok(PyArray1::from_vec(py, out))
            }

            /// Remove every key in ``keys`` that is present and return how
            /// many were removed. Absent and out-of-range keys are ignored.
            fn discard_many(&mut self, keys: &Bound<'_, PyAny>) -> PyResult<usize> {
                let batch = KeyBatch::<$K>::new(keys)?;
                let mut present = Vec::new();
                batch.for_each(|k| present.extend(k.filter(|&k| self.map.contains_key(k))));
                if present.is_empty() {
                    return Ok(0);
                }
                self.exports.check($pyname)?;
                let removed = present
                    .into_iter()
                    .filter(|&k| self.map.remove(k).is_some())
                    .count();
                self.changed();
                Ok(removed)
            }

            /// Sort the entries by key, ascending (ids: by index, then generation).
            fn sort_keys(&mut self) -> PyResult<()> {
                self.exports.check($pyname)?;
                self.map.sort_unstable_keys();
                self.changed();
                Ok(())
            }

            /// Reserve room for at least ``additional`` more entries.
            fn reserve(&mut self, additional: usize) -> PyResult<()> {
                self.exports.check($pyname)?;
                self.map.try_reserve(additional).map_err(crate::reserve_error)
            }

            /// Release unused memory; frees every page if the map is empty.
            fn shrink_to_fit(&mut self) -> PyResult<()> {
                self.exports.check($pyname)?;
                self.map.shrink_to_fit();
                Ok(())
            }

            /// Number of allocated pages of the key index.
            #[getter]
            fn allocated_pages(&self) -> usize {
                self.map.allocated_pages()
            }

            /// Slots per page for this instance.
            #[getter]
            fn page_size(&self) -> usize { self.map.page_size() }

            /// The NumPy dtype name of the values.
            #[classattr]
            fn dtype() -> &'static str {
                <$t as TypedValue>::DTYPE
            }
        }

        key_iterator!($iter, $name, $pyname, $K);
    };

    (@view writeable, $slf:ident) => {{
        let this = $slf.try_borrow()?;
        // SAFETY: the values are the map's dense value `Vec` of cells. Every
        // method that could move them checks `exports` first, Rust writes
        // them only through `ValueCell::set`, and every bit pattern is a valid
        // atomic of this size.
        unsafe { cell_view($slf.as_any(), &this.exports, this.map.value_slice(), true) }
    }};

    (@view read_only, $slf:ident) => {{
        let this = $slf.try_borrow()?;
        // SAFETY: as for the writeable view. The view is read-only so that
        // NumPy cannot store bool bytes other than 0 and 1, which the map
        // would read back as `True` but NumPy would show differently.
        unsafe { cell_view($slf.as_any(), &this.exports, this.map.value_slice(), false) }
    }};
}

typed_map!(
    SparseMapF64,
    SparseMapF64Iterator,
    "SparseMapF64",
    f64,
    writeable
);
typed_map!(
    SparseMapF32,
    SparseMapF32Iterator,
    "SparseMapF32",
    f32,
    writeable
);
typed_map!(
    SparseMapI64,
    SparseMapI64Iterator,
    "SparseMapI64",
    i64,
    writeable
);
typed_map!(
    SparseMapI32,
    SparseMapI32Iterator,
    "SparseMapI32",
    i32,
    writeable
);
typed_map!(
    SparseMapU64,
    SparseMapU64Iterator,
    "SparseMapU64",
    u64,
    writeable
);
typed_map!(
    SparseMapU32,
    SparseMapU32Iterator,
    "SparseMapU32",
    u32,
    writeable
);
typed_map!(
    SparseMapBool,
    SparseMapBoolIterator,
    "SparseMapBool",
    bool,
    read_only
);
typed_map!(
    IdMapF64,
    IdMapF64Iterator,
    "IdMapF64",
    f64,
    writeable,
    Id,
    "generational ids (see ``IdAllocator``)",
    "uint64"
);
typed_map!(
    IdMapF32,
    IdMapF32Iterator,
    "IdMapF32",
    f32,
    writeable,
    Id,
    "generational ids (see ``IdAllocator``)",
    "uint64"
);
typed_map!(
    IdMapI64,
    IdMapI64Iterator,
    "IdMapI64",
    i64,
    writeable,
    Id,
    "generational ids (see ``IdAllocator``)",
    "uint64"
);
typed_map!(
    IdMapI32,
    IdMapI32Iterator,
    "IdMapI32",
    i32,
    writeable,
    Id,
    "generational ids (see ``IdAllocator``)",
    "uint64"
);
typed_map!(
    IdMapU64,
    IdMapU64Iterator,
    "IdMapU64",
    u64,
    writeable,
    Id,
    "generational ids (see ``IdAllocator``)",
    "uint64"
);
typed_map!(
    IdMapU32,
    IdMapU32Iterator,
    "IdMapU32",
    u32,
    writeable,
    Id,
    "generational ids (see ``IdAllocator``)",
    "uint64"
);
typed_map!(
    IdMapBool,
    IdMapBoolIterator,
    "IdMapBool",
    bool,
    read_only,
    Id,
    "generational ids (see ``IdAllocator``)",
    "uint64"
);
