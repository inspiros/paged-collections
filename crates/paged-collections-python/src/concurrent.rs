//! Primitive concurrent containers. Detached loops use owned inputs only;
//! PyO3 keeps the receiver borrowed until the entire Python call returns.

use numpy::{PyArray1, PyArrayDescrMethods, PyUntypedArray, PyUntypedArrayMethods};
use paged_collections::concurrent::{
    ConcurrentDirectMap as RustDirectMap, ConcurrentSparseMap as RustSparseMap,
    ConcurrentSparseSet as RustSparseSet,
};
use paged_collections::sparse_map::PagedSparseMap;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyIterator, PyList, PyTuple};

use crate::convert::{
    Key, KeyBatch, TypedValue, insert_key, lookup_key, query_bounds, value_batch,
};

/// Copy even native NumPy keys before detaching: no array borrow or Python
/// object is accessed by a detached loop. Invalid lookup keys stay absent.
fn lookup_keys(keys: &Bound<'_, PyAny>) -> PyResult<Vec<Option<Key>>> {
    let batch = KeyBatch::<Key>::lookup(keys)?;
    let mut keys = Vec::with_capacity(batch.len());
    batch.for_each(|key| keys.push(key));
    Ok(keys)
}

/// Preserve the original elements for scalar-style membership checks.
fn membership_keys(keys: &Bound<'_, PyAny>) -> PyResult<Vec<Option<Key>>> {
    if let Ok(array) = keys.cast::<PyUntypedArray>() {
        if array.ndim() != 1 {
            return Err(PyValueError::new_err(format!(
                "keys must be one-dimensional, got {} dimensions",
                array.ndim()
            )));
        }
        if matches!(array.dtype().kind(), b'i' | b'u') {
            return lookup_keys(keys);
        }
    }
    keys.try_iter()?
        .map(|key| key.map(|key| lookup_key(&key)))
        .collect()
}

/// The common copied lookup/snapshot interface, hiding direct-map guards.
trait CopyMap<V: Copy> {
    fn copied_get(&self, key: Key) -> Option<V>;
    fn copied_items(&self) -> Vec<(Key, V)>;
}

impl<V: Copy> CopyMap<V> for RustSparseMap<Key, V> {
    fn copied_get(&self, key: Key) -> Option<V> {
        self.get(&key).copied()
    }

    fn copied_items(&self) -> Vec<(Key, V)> {
        self.iter().map(|(&key, &value)| (key, value)).collect()
    }
}

impl<V: Copy> CopyMap<V> for RustDirectMap<Key, V> {
    fn copied_get(&self, key: Key) -> Option<V> {
        self.get(key).map(|value| *value)
    }

    fn copied_items(&self) -> Vec<(Key, V)> {
        let mut items = Vec::new();
        // Copy under one page lock at a time; no guard reaches Python.
        self.for_each(|&key, &value| items.push((key, value)));
        items
    }
}

/// Construction from validated batches, matching the plain maps: later
/// duplicate keys win.
trait Build<V> {
    fn build(keys: Vec<Key>, values: Vec<V>) -> Self;
}

impl<V> Build<V> for RustSparseMap<Key, V> {
    fn build(keys: Vec<Key>, values: Vec<V>) -> Self {
        // The concurrent sparse map is insert-only (first wins), so resolve
        // duplicates in a plain map; `From` keeps its dense order.
        let mut plain = PagedSparseMap::new();
        for (key, value) in keys.into_iter().zip(values) {
            plain.insert(key, value);
        }
        Self::from(plain)
    }
}

impl<V> Build<V> for RustDirectMap<Key, V> {
    fn build(keys: Vec<Key>, values: Vec<V>) -> Self {
        let map = Self::new();
        for (key, value) in keys.into_iter().zip(values) {
            map.insert(key, value);
        }
        map
    }
}

/// A shared-insertion set of uint32 keys with copied snapshots.
/// Batch methods release the GIL; single-key calls do not. Removal and
/// clear require exclusive access and raise RuntimeError ("Already borrowed")
/// if another call holds a shared borrow. Uses the default Rust page size
/// and Flat directory; there are no guards or views. Pickling and copies
/// take a weakly consistent snapshot (see the Rust `snapshot` docs).
#[pyclass(module = "paged_collections._core", weakref)]
pub(crate) struct ConcurrentSparseSet {
    set: RustSparseSet<Key>,
}

#[pymethods]
impl ConcurrentSparseSet {
    /// Build from a batch of keys (array or sequence); duplicates are kept
    /// once, in first-occurrence order. Releases the GIL while inserting.
    #[new]
    #[pyo3(signature = (keys = None))]
    fn new(py: Python<'_>, keys: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let set = RustSparseSet::new();
        if let Some(keys) = keys {
            let keys = KeyBatch::<Key>::new(keys)?.to_insert_keys()?;
            py.detach(|| set.insert_many(keys));
        }
        Ok(Self { set })
    }

    /// Return a snapshot copy. Releases the GIL while copying.
    fn copy(&self, py: Python<'_>) -> Self {
        Self {
            set: py.detach(|| self.set.clone()),
        }
    }

    fn __copy__(&self, py: Python<'_>) -> Self {
        self.copy(py)
    }

    fn __deepcopy__(&self, py: Python<'_>, _memo: &Bound<'_, PyAny>) -> Self {
        self.copy(py)
    }

    fn __len__(&self) -> usize {
        self.set.len()
    }

    fn __contains__(&self, key: &Bound<'_, PyAny>) -> bool {
        lookup_key(key).is_some_and(|key| self.set.contains(key))
    }

    fn insert(&self, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self.set.insert(insert_key(key)?))
    }

    fn position(&self, key: &Bound<'_, PyAny>) -> Option<usize> {
        lookup_key(key).and_then(|key| self.set.position(key))
    }

    /// Return a uint32 array snapshot in dense order.
    fn keys<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Key>> {
        let keys = py.detach(|| self.set.iter().copied().collect());
        PyArray1::from_vec(py, keys)
    }

    /// Iterate over copied Python integers, independent of later mutations.
    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyIterator>> {
        let keys: Vec<Key> = py.detach(|| self.set.iter().copied().collect());
        PyList::new(py, keys)?.as_any().try_iter()
    }

    fn insert_many(&self, py: Python<'_>, keys: &Bound<'_, PyAny>) -> PyResult<usize> {
        let keys = KeyBatch::<Key>::new(keys)?.to_insert_keys()?;
        Ok(py.detach(|| self.set.insert_many(keys)))
    }

    fn contains_many<'py>(
        &self,
        py: Python<'py>,
        keys: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let keys = lookup_keys(keys)?;
        let values = py.detach(|| {
            keys.into_iter()
                .map(|key| key.is_some_and(|key| self.set.contains(key)))
                .collect()
        });
        Ok(PyArray1::from_vec(py, values))
    }

    /// Remove a key, returning whether it was present. Exclusive access.
    fn remove(&mut self, key: &Bound<'_, PyAny>) -> bool {
        lookup_key(key).is_some_and(|key| self.set.remove(key))
    }

    /// Remove present keys once each. Exclusive access; releases the GIL.
    fn remove_many(&mut self, py: Python<'_>, keys: &Bound<'_, PyAny>) -> PyResult<usize> {
        let keys = lookup_keys(keys)?;
        Ok(py.detach(|| {
            keys.into_iter()
                .filter(|key| key.is_some_and(|key| self.set.remove(key)))
                .count()
        }))
    }

    /// Clear with exclusive access; conflicts raise RuntimeError.
    fn clear(&mut self) {
        self.set.clear();
    }

    /// Free pages with no keys. Requires exclusive access; conflicts raise RuntimeError.
    fn shrink_to_fit(&mut self) {
        self.set.shrink_to_fit();
    }

    fn __repr__(&self) -> String {
        let keys = crate::truncated(self.set.iter().map(Key::to_string));
        format!("ConcurrentSparseSet([{keys}])")
    }

    /// Pickle as `(ConcurrentSparseSet, (keys,))` from one snapshot pass.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let keys = slf.try_borrow()?.keys(py);
        (py.get_type::<Self>(), (keys,)).into_pyobject(py)
    }
}

// One pymethods block per class (no PyO3 multiple-pymethods feature needed).
macro_rules! concurrent_map {
    ($name:ident, $t:ty, $inner:ty, $semantics:literal; $($methods:tt)*) => {
        #[doc = concat!(
            $semantics,
            "\n\nKeys are uint32; values and snapshots are copied, with no guards or views. ",
            "Batch methods release the GIL; single-key calls do not. ",
            "clear, shrink_to_fit and sparse remove/remove_many require exclusive access: conflicts ",
            "with shared calls raise RuntimeError (Already borrowed). ",
            "Uses the default Rust page size and Flat directory. ",
            "Pickling and copy() take one weakly consistent snapshot. ",
            "Snapshots can omit insertions in flight; direct snapshots are consistent ",
            "per page only. Batch calls are not transactions."
        )]
        #[pyclass(module = "paged_collections._core", weakref)]
        pub(crate) struct $name {
            map: $inner,
        }

        #[pymethods]
        impl $name {
            /// Build from key and value batches (both or neither); later
            /// duplicate keys win, as for the plain maps. Validates both
            /// batches before writing; releases the GIL while inserting.
            #[new]
            #[pyo3(signature = (keys = None, values = None))]
            fn new(
                py: Python<'_>,
                keys: Option<&Bound<'_, PyAny>>,
                values: Option<&Bound<'_, PyAny>>,
            ) -> PyResult<Self> {
                let (keys, values) = match (keys, values) {
                    (None, None) => (Vec::new(), Vec::new()),
                    (Some(keys), Some(values)) => {
                        let keys = KeyBatch::<Key>::new(keys)?.to_insert_keys()?;
                        let values = value_batch::<$t>(values, keys.len())?;
                        (keys, values)
                    }
                    _ => return Err(PyTypeError::new_err("give both keys and values, or neither")),
                };
                Ok(Self { map: py.detach(|| <$inner as Build<$t>>::build(keys, values)) })
            }

            /// Return a snapshot copy. Releases the GIL while copying.
            fn copy(&self, py: Python<'_>) -> Self {
                Self { map: py.detach(|| self.map.clone()) }
            }

            fn __copy__(&self, py: Python<'_>) -> Self {
                self.copy(py)
            }

            fn __deepcopy__(&self, py: Python<'_>, _memo: &Bound<'_, PyAny>) -> Self {
                self.copy(py)
            }

            #[classattr]
            fn dtype() -> &'static str {
                <$t as TypedValue>::DTYPE
            }

            fn __len__(&self) -> usize {
                self.map.len()
            }

            fn __contains__(&self, key: &Bound<'_, PyAny>) -> bool {
                lookup_key(key).is_some_and(|key| self.map.copied_get(key).is_some())
            }

            fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<$t> {
                lookup_key(key)
                    .and_then(|key| self.map.copied_get(key))
                    .ok_or_else(|| PyKeyError::new_err(key.clone().unbind()))
            }

            #[pyo3(signature = (key, default = None))]
            fn get<'py>(
                &self,
                py: Python<'py>,
                key: &Bound<'_, PyAny>,
                default: Option<Bound<'py, PyAny>>,
            ) -> PyResult<Bound<'py, PyAny>> {
                match lookup_key(key).and_then(|key| self.map.copied_get(key)) {
                    Some(value) => value.into_bound_py_any(py),
                    None => Ok(default.unwrap_or_else(|| py.None().into_bound(py))),
                }
            }

            /// Return copied values, using default for absent/out-of-range keys.
            fn get_many<'py>(
                &self,
                py: Python<'py>,
                keys: &Bound<'_, PyAny>,
                default: $t,
            ) -> PyResult<Bound<'py, PyArray1<$t>>> {
                let keys = lookup_keys(keys)?;
                let values = py.detach(|| {
                    keys.into_iter()
                        .map(|key| key.and_then(|key| self.map.copied_get(key)).unwrap_or(default))
                        .collect()
                });
                Ok(PyArray1::from_vec(py, values))
            }

            fn keys<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Key>> {
                let keys = py.detach(|| self.map.copied_items().into_iter().map(|(key, _)| key).collect());
                PyArray1::from_vec(py, keys)
            }

            /// Iterate over a snapshot of keys, with the same order and consistency as ``keys()``.
            fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyIterator>> {
                let keys: Vec<_> = py.detach(|| self.map.copied_items().into_iter().map(|(key, _)| key).collect());
                PyList::new(py, keys)?.as_any().try_iter()
            }

            /// Return a boolean array of membership tests. Invalid keys are
            /// absent, as in ``key in map``. Releases the GIL during lookup.
            fn contains_many<'py>(
                &self,
                py: Python<'py>,
                keys: &Bound<'_, PyAny>,
            ) -> PyResult<Bound<'py, PyArray1<bool>>> {
                let keys = membership_keys(keys)?;
                let present = py.detach(|| {
                    keys.into_iter()
                        .map(|key| key.is_some_and(|key| self.map.copied_get(key).is_some()))
                        .collect()
                });
                Ok(PyArray1::from_vec(py, present))
            }

            fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<$t>> {
                let values = py.detach(|| self.map.copied_items().into_iter().map(|(_, value)| value).collect());
                PyArray1::from_vec(py, values)
            }

            /// Return a copied list of (key, value) pairs.
            fn items(&self, py: Python<'_>) -> Vec<(Key, $t)> {
                py.detach(|| self.map.copied_items())
            }

            /// Clear with exclusive access; conflicts raise RuntimeError.
            fn clear(&mut self) {
                self.map.clear();
            }

            fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
                // Release every direct-map guard before invoking Python repr.
                let items = crate::repr_items(self.map.copied_items().into_iter(), py)?;
                Ok(format!("{}({{{items}}})", stringify!($name)))
            }

            /// Pickle as `(cls, (keys, values))`. Keys and values come from
            /// one snapshot pass, so they always pair up.
            fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
                let py = slf.py();
                let this = slf.try_borrow()?;
                // Capture the map, not the PyRef, which is not Send.
                let map = &this.map;
                let (keys, values): (Vec<Key>, Vec<$t>) =
                    py.detach(|| map.copied_items().into_iter().unzip());
                let keys = PyArray1::from_vec(py, keys);
                let values = PyArray1::from_vec(py, values);
                (py.get_type::<Self>(), (keys, values)).into_pyobject(py)
            }

            $($methods)*
        }
    };
}

macro_rules! sparse_map {
    ($name:ident, $t:ty) => {
        concurrent_map!($name, $t, RustSparseMap<Key, $t>,
            "An insert-only shared sparse map; existing values are retained.";

            fn try_insert(&self, key: &Bound<'_, PyAny>, value: $t) -> PyResult<bool> {
                Ok(self.map.try_insert(insert_key(key)?, value).is_ok())
            }

            /// Free pages with no entries. Requires exclusive access; conflicts
            /// raise RuntimeError.
            fn shrink_to_fit(&mut self) {
                self.map.shrink_to_fit();
            }

            fn get_or_insert(&self, key: &Bound<'_, PyAny>, value: $t) -> PyResult<$t> {
                let key = insert_key(key)?;
                // u32 keys have only one generation, so this cannot be None.
                Ok(*self.map.get_or_insert_with(key, || value).expect("u32 key has no generation conflict"))
            }

            /// Insert absent keys; existing keys and earlier duplicates win.
            fn insert_many(
                &self,
                py: Python<'_>,
                keys: &Bound<'_, PyAny>,
                values: &Bound<'_, PyAny>,
            ) -> PyResult<usize> {
                let keys = KeyBatch::<Key>::new(keys)?.to_insert_keys()?;
                let values = value_batch::<$t>(values, keys.len())?;
                Ok(py.detach(|| self.map.try_insert_many(keys.into_iter().zip(values))))
            }

            /// Remove with exclusive access, returning the value or None.
            fn remove(&mut self, key: &Bound<'_, PyAny>) -> Option<$t> {
                lookup_key(key).and_then(|key| self.map.remove(&key))
            }

            /// Remove present keys once each. Exclusive access; releases the GIL.
            fn remove_many(&mut self, py: Python<'_>, keys: &Bound<'_, PyAny>) -> PyResult<usize> {
                let keys = lookup_keys(keys)?;
                Ok(py.detach(|| {
                    keys.into_iter()
                        .filter(|key| key.is_some_and(|key| self.map.remove(&key).is_some()))
                        .count()
                }))
            }
        );
    };
}

macro_rules! direct_map {
    ($name:ident, $t:ty) => {
        concurrent_map!($name, $t, RustDirectMap<Key, $t>,
            "A shared mutable direct map, with snapshots in ascending key order.";

            fn __setitem__(&self, key: &Bound<'_, PyAny>, value: $t) -> PyResult<()> {
                self.map.insert(insert_key(key)?, value);
                Ok(())
            }

            fn __delitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<()> {
                lookup_key(key)
                    .and_then(|key| self.map.remove(key))
                    .map(drop)
                    .ok_or_else(|| PyKeyError::new_err(key.clone().unbind()))
            }

            fn insert(&self, key: &Bound<'_, PyAny>, value: $t) -> PyResult<Option<$t>> {
                Ok(self.map.insert(insert_key(key)?, value))
            }

            fn try_insert(&self, key: &Bound<'_, PyAny>, value: $t) -> PyResult<bool> {
                Ok(self.map.try_insert(insert_key(key)?, value).is_ok())
            }

            /// Atomically insert the value if the key is absent. Return the stored value.
            fn get_or_insert(&self, key: &Bound<'_, PyAny>, value: $t) -> PyResult<$t> {
                let key = insert_key(key)?;
                Ok(*self.map.get_or_insert_with(key, || value))
            }

            /// Atomically insert the value if the key is absent and return
            /// the stored value, as in ``get_or_insert()``.
            fn setdefault(&self, key: &Bound<'_, PyAny>, value: $t) -> PyResult<$t> {
                self.get_or_insert(key, value)
            }

            /// Return a weakly consistent snapshot of the entries in
            /// ``[start, stop)``, in ascending key order. Bounds are optional
            /// and clamped to the key domain.
            #[pyo3(signature = (start = None, stop = None))]
            fn items_range(
                &self,
                py: Python<'_>,
                start: Option<&Bound<'_, PyAny>>,
                stop: Option<&Bound<'_, PyAny>>,
            ) -> PyResult<Vec<(Key, $t)>> {
                let bounds = query_bounds(start, stop)?;
                Ok(py.detach(|| {
                    let mut items = Vec::new();
                    self.map.range_for_each(bounds, |&key, &value| items.push((key, value)));
                    items
                }))
            }

            /// Return a new uint32 array of keys in ``[start, stop)``, in key order.
            #[pyo3(signature = (start = None, stop = None))]
            fn keys_range<'py>(
                &self,
                py: Python<'py>,
                start: Option<&Bound<'_, PyAny>>,
                stop: Option<&Bound<'_, PyAny>>,
            ) -> PyResult<Bound<'py, PyArray1<Key>>> {
                let keys = self.items_range(py, start, stop)?.into_iter().map(|(key, _)| key).collect();
                Ok(PyArray1::from_vec(py, keys))
            }

            /// Return a new array of values in ``[start, stop)``, in key order.
            #[pyo3(signature = (start = None, stop = None))]
            fn values_range<'py>(
                &self,
                py: Python<'py>,
                start: Option<&Bound<'_, PyAny>>,
                stop: Option<&Bound<'_, PyAny>>,
            ) -> PyResult<Bound<'py, PyArray1<$t>>> {
                let values = self.items_range(py, start, stop)?.into_iter().map(|(_, value)| value).collect();
                Ok(PyArray1::from_vec(py, values))
            }

            /// Free pages with no live entries. Requires exclusive access.
            fn shrink_to_fit(&mut self) {
                self.map.shrink_to_fit();
            }

            #[pyo3(signature = (key, default = None))]
            fn remove<'py>(
                &self,
                py: Python<'py>,
                key: &Bound<'_, PyAny>,
                default: Option<Bound<'py, PyAny>>,
            ) -> PyResult<Bound<'py, PyAny>> {
                match lookup_key(key).and_then(|key| self.map.remove(key)) {
                    Some(value) => value.into_bound_py_any(py),
                    None => Ok(default.unwrap_or_else(|| py.None().into_bound(py))),
                }
            }

            /// Set values; later duplicates win. Validate before any write.
            fn set_many(
                &self,
                py: Python<'_>,
                keys: &Bound<'_, PyAny>,
                values: &Bound<'_, PyAny>,
            ) -> PyResult<()> {
                let keys = KeyBatch::<Key>::new(keys)?.to_insert_keys()?;
                let values = value_batch::<$t>(values, keys.len())?;
                py.detach(|| {
                    for (key, value) in keys.into_iter().zip(values) {
                        self.map.insert(key, value);
                    }
                });
                Ok(())
            }

            fn remove_many(&self, py: Python<'_>, keys: &Bound<'_, PyAny>) -> PyResult<usize> {
                let keys = lookup_keys(keys)?;
                Ok(py.detach(|| {
                    keys.into_iter()
                        .filter(|key| key.is_some_and(|key| self.map.remove(key).is_some()))
                        .count()
                }))
            }
        );
    };
}

sparse_map!(ConcurrentSparseMapF64, f64);
sparse_map!(ConcurrentSparseMapF32, f32);
sparse_map!(ConcurrentSparseMapI64, i64);
sparse_map!(ConcurrentSparseMapI32, i32);
sparse_map!(ConcurrentSparseMapU64, u64);
sparse_map!(ConcurrentSparseMapU32, u32);
sparse_map!(ConcurrentSparseMapBool, bool);

direct_map!(ConcurrentDirectMapF64, f64);
direct_map!(ConcurrentDirectMapF32, f32);
direct_map!(ConcurrentDirectMapI64, i64);
direct_map!(ConcurrentDirectMapI32, i32);
direct_map!(ConcurrentDirectMapU64, u64);
direct_map!(ConcurrentDirectMapU32, u32);
direct_map!(ConcurrentDirectMapBool, bool);
