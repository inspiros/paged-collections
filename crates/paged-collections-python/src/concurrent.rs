//! Primitive concurrent containers. Detached loops use owned inputs only;
//! PyO3 keeps the receiver borrowed until the entire Python call returns.

use numpy::PyArray1;
use paged_collections::concurrent::{
    ConcurrentDirectMap as RustDirectMap, ConcurrentSparseMap as RustSparseMap,
    ConcurrentSparseSet as RustSparseSet,
};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyKeyError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyIterator, PyList};

use crate::convert::{Key, KeyBatch, TypedValue, insert_key, lookup_key, value_batch};

/// Copy even native NumPy keys before detaching: no array borrow or Python
/// object is accessed by a detached loop. Invalid lookup keys stay absent.
fn lookup_keys(keys: &Bound<'_, PyAny>) -> PyResult<Vec<Option<Key>>> {
    let batch = KeyBatch::<Key>::lookup(keys)?;
    let mut keys = Vec::with_capacity(batch.len());
    batch.for_each(|key| keys.push(key));
    Ok(keys)
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

/// A shared-insertion set of uint32 keys with copied snapshots.
/// Batch methods release the GIL; single-key calls do not. Removal and
/// clear require exclusive access and raise RuntimeError ("Already borrowed")
/// if another call holds a shared borrow. Uses the default Rust page size
/// and Flat directory; there are no guards, views or pickle support.
#[pyclass(module = "paged_collections._core", weakref)]
pub(crate) struct ConcurrentSparseSet {
    set: RustSparseSet<Key>,
}

#[pymethods]
impl ConcurrentSparseSet {
    #[new]
    fn new() -> Self {
        Self {
            set: RustSparseSet::new(),
        }
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

    fn __repr__(&self) -> String {
        let keys = crate::truncated(self.set.iter().map(Key::to_string));
        format!("ConcurrentSparseSet([{keys}])")
    }

    fn __reduce__(&self) -> PyResult<()> {
        Err(PyTypeError::new_err(
            "ConcurrentSparseSet does not support pickling",
        ))
    }
}

// One pymethods block per class (no PyO3 multiple-pymethods feature needed).
macro_rules! concurrent_map {
    ($name:ident, $t:ty, $inner:ty, $semantics:literal; $($methods:tt)*) => {
        #[doc = concat!(
            $semantics,
            "\n\nKeys are uint32; values and snapshots are copied, with no guards or views. ",
            "Batch methods release the GIL; single-key calls do not. ",
            "clear and sparse remove/remove_many require exclusive access: conflicts ",
            "with shared calls raise RuntimeError (Already borrowed). ",
            "Uses the default Rust page size and Flat directory; no pickling. ",
            "Snapshots can omit insertions in flight; direct snapshots are consistent ",
            "per page only. Batch calls are not transactions."
        )]
        #[pyclass(module = "paged_collections._core", weakref)]
        pub(crate) struct $name {
            map: $inner,
        }

        #[pymethods]
        impl $name {
            #[new]
            fn new() -> Self {
                Self { map: <$inner>::new() }
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

            fn __reduce__(&self) -> PyResult<()> {
                Err(PyTypeError::new_err(concat!(stringify!($name), " does not support pickling")))
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
