//! `DirectMap`: `PagedDirectMap<u32, Py<PyAny>>`, for arbitrary values.

use numpy::PyArray1;
use paged_collections::direct_map::PagedDirectMap;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyInt, PyList, PyTuple};
use pyo3::{PyTraverseError, PyVisit};
use std::ops::Bound as SlotBound;

use crate::convert::{Key, KeyBatch, TypedValue, insert_key, lookup_key, value_batch};
use crate::iter::ordered_key_iterator;
use crate::repr_items;
use crate::typed_map::optional_default;

const KEY_LIMIT: u64 = 1_u64 << 32;

/// A Python int as a nonnegative slot index, saturated at the key limit;
/// `None` denotes a negative int. Check the type before extracting so
/// oversized ints are clamped, while non-ints still raise `TypeError`.
fn query_index(index: &Bound<'_, PyAny>) -> PyResult<Option<u64>> {
    let index = index.cast::<PyInt>()?;
    // `operator.index` returns an exact `int` copy of an `int` subclass, so
    // an overridden `__lt__` cannot change the sign test below.
    let index = index
        .py()
        .import("operator")?
        .call_method1("index", (index,))?;
    if index.lt(0)? {
        Ok(None)
    } else {
        Ok(Some(
            index.extract::<u64>().unwrap_or(KEY_LIMIT).min(KEY_LIMIT),
        ))
    }
}

/// Half-open Python bounds as inclusive Rust slot bounds. The inclusive
/// upper bound also represents the full u32 domain on a 32-bit target.
fn query_bounds(
    start: Option<&Bound<'_, PyAny>>,
    stop: Option<&Bound<'_, PyAny>>,
) -> PyResult<(SlotBound<usize>, SlotBound<usize>)> {
    let start = match start {
        Some(index) => query_index(index)?.unwrap_or(0),
        None => 0,
    };
    let stop = match stop {
        Some(index) => query_index(index)?.unwrap_or(0),
        None => KEY_LIMIT,
    };
    if start >= stop {
        Ok((SlotBound::Excluded(0), SlotBound::Excluded(0)))
    } else {
        Ok((
            SlotBound::Included(start as usize),
            SlotBound::Included((stop - 1) as usize),
        ))
    }
}

/// Write a fully validated batch. Duplicate keys keep their last value.
fn write_batch<V>(map: &mut PagedDirectMap<Key, V>, keys: Vec<Key>, values: Vec<V>) -> usize {
    debug_assert_eq!(keys.len(), values.len());
    let before = map.len();
    for (key, value) in keys.into_iter().zip(values) {
        map.insert(key, value);
    }
    map.len() - before
}

/// A map from integers in ``0 <= key < 2**32`` to arbitrary Python objects,
/// with O(1) insert, delete and lookup.
///
/// Values live in pages allocated for the key ranges in use. Entries are
/// always visited in ascending key order. Ordered queries find nearby
/// entries or entries in a half-open range; array results are copies.
#[pyclass(
    module = "paged_collections._core",
    name = "DirectMap",
    subclass,
    weakref
)]
pub(crate) struct DirectMap {
    map: PagedDirectMap<Key, Py<PyAny>>,
    /// Changes on every structural change, so iterators can detect them.
    mods: u64,
}

impl DirectMap {
    fn empty() -> Self {
        Self {
            map: PagedDirectMap::new(),
            mods: 0,
        }
    }

    fn changed(&mut self) {
        self.mods = self.mods.wrapping_add(1);
    }

    fn next_key_from(&self, index: u64) -> Option<Key> {
        if index < KEY_LIMIT {
            self.map.next_occupant(index as usize).map(|(&key, _)| key)
        } else {
            None
        }
    }

    fn set_batch(&mut self, keys: &Bound<'_, PyAny>, values: &Bound<'_, PyAny>) -> PyResult<usize> {
        let keys = KeyBatch::new(keys)?.to_insert_keys()?;
        let values: Vec<Py<PyAny>> = values
            .try_iter()?
            .map(|value| value.map(Bound::unbind))
            .collect::<PyResult<_>>()?;
        if values.len() != keys.len() {
            return Err(PyValueError::new_err(format!(
                "got {} values for {} keys",
                values.len(),
                keys.len()
            )));
        }
        let new = write_batch(&mut self.map, keys, values);
        if new > 0 {
            self.changed();
        }
        Ok(new)
    }

    fn remove_present(&mut self, key: Key) -> PyResult<Py<PyAny>> {
        let value = self.map.remove(key);
        self.changed();
        Ok(value.expect("caller checked the key is present"))
    }
}

#[pymethods]
impl DirectMap {
    // `__new__` accepts and ignores any arguments, so subclasses can define
    // their own `__init__`; `__init__` fills the collection, as for `dict`.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs), text_signature = "(keys=None, values=None)")]
    fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        Self::empty()
    }

    #[pyo3(signature = (keys = None, values = None))]
    fn __init__(
        &mut self,
        keys: Option<&Bound<'_, PyAny>>,
        values: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        match (keys, values) {
            (None, None) => Ok(()),
            (Some(keys), Some(values)) => self.set_batch(keys, values).map(drop),
            _ => Err(PyTypeError::new_err(
                "give both keys and values, or neither",
            )),
        }
    }

    fn __len__(&self) -> usize {
        self.map.len()
    }

    fn __contains__(&self, key: &Bound<'_, PyAny>) -> bool {
        lookup_key(key).is_some_and(|k| self.map.contains_key(k))
    }

    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        lookup_key(key)
            .and_then(|k| self.map.get(k))
            .map(|value| value.clone_ref(py))
            .ok_or_else(|| PyKeyError::new_err(key.clone().unbind()))
    }

    fn __setitem__(&mut self, key: &Bound<'_, PyAny>, value: Py<PyAny>) -> PyResult<()> {
        let key = insert_key(key)?;
        if let Some(slot) = self.map.get_mut(key) {
            // The old value is dropped here; its `__del__` cannot reach this
            // map, which is borrowed.
            *slot = value;
        } else {
            self.map.insert(key, value);
            self.changed();
        }
        Ok(())
    }

    fn __delitem__(&mut self, key: &Bound<'_, PyAny>) -> PyResult<()> {
        match lookup_key(key).filter(|&k| self.map.contains_key(k)) {
            Some(k) => self.remove_present(k).map(drop),
            None => Err(PyKeyError::new_err(key.clone().unbind())),
        }
    }

    fn __iter__(slf: &Bound<'_, Self>) -> PyResult<DirectMapIterator> {
        DirectMapIterator::new(slf)
    }

    /// Equal when both hold the same keys with equal values (compared with
    /// ``is`` first, then ``==``, as ``dict`` does).
    fn __eq__(&self, py: Python<'_>, other: PyRef<'_, Self>) -> PyResult<bool> {
        if self.map.len() != other.map.len() {
            return Ok(false);
        }
        for (&key, value) in self.map.iter() {
            let Some(other_value) = other.map.get(key) else {
                return Ok(false);
            };
            if !value.is(other_value) && !value.bind(py).eq(other_value.bind(py))? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let name = slf.get_type().name()?.to_string();
        crate::repr_guarded(slf.as_any(), &name, || {
            let this = slf.try_borrow()?;
            let items = repr_items(this.map.iter().map(|(&k, v)| (k, v.bind(py).clone())), py)?;
            Ok(format!("{name}({{{items}}})"))
        })
    }

    /// Pickles as an empty map plus its items, so `pickle` and
    /// `copy.deepcopy` create the map before its values and handle maps that
    /// contain themselves. Items are restored with `__setitem__`, as for
    /// `dict`.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let items = PyList::new(py, slf.try_borrow()?.items(py))?;
        crate::reduce(
            slf.as_any(),
            &py.get_type::<Self>(),
            Vec::new(),
            Some(items.try_iter()?),
        )
    }

    /// Return a shallow copy: the values are shared, not copied.
    fn copy(&self, py: Python<'_>) -> Self {
        let mut map = PagedDirectMap::new();
        for (&key, value) in self.map.iter() {
            map.insert(key, value.clone_ref(py));
        }
        Self { map, mods: 0 }
    }

    /// Return the value for ``key``, or ``default`` if it is absent.
    #[pyo3(signature = (key, default = None))]
    fn get(&self, py: Python<'_>, key: &Bound<'_, PyAny>, default: Option<Py<PyAny>>) -> Py<PyAny> {
        match lookup_key(key).and_then(|k| self.map.get(k)) {
            Some(value) => value.clone_ref(py),
            None => default.unwrap_or_else(|| py.None()),
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
        match lookup_key(key).filter(|&k| self.map.contains_key(k)) {
            Some(k) => Ok(self.remove_present(k)?.into_bound(py)),
            None => default.ok_or_else(|| PyKeyError::new_err(key.clone().unbind())),
        }
    }

    /// Remove and return the entry with the largest key; raise
    /// ``KeyError`` if the map is empty.
    fn popitem(&mut self) -> PyResult<(Key, Py<PyAny>)> {
        let item = self
            .map
            .pop_last()
            .ok_or_else(|| PyKeyError::new_err("popitem(): DirectMap is empty"))?;
        self.changed();
        Ok(item)
    }

    /// Return the entry with the smallest key; raise ``KeyError`` if empty.
    fn first_item(&self, py: Python<'_>) -> PyResult<(Key, Py<PyAny>)> {
        self.map
            .first_key_value()
            .map(|(&key, value)| (key, value.clone_ref(py)))
            .ok_or_else(|| PyKeyError::new_err("first_item(): DirectMap is empty"))
    }

    /// Return the entry with the largest key; raise ``KeyError`` if empty.
    fn last_item(&self, py: Python<'_>) -> PyResult<(Key, Py<PyAny>)> {
        self.map
            .last_key_value()
            .map(|(&key, value)| (key, value.clone_ref(py)))
            .ok_or_else(|| PyKeyError::new_err("last_item(): DirectMap is empty"))
    }

    /// Remove and return the entry with the smallest key; raise
    /// ``KeyError`` if the map is empty.
    fn pop_first(&mut self) -> PyResult<(Key, Py<PyAny>)> {
        let item = self
            .map
            .pop_first()
            .ok_or_else(|| PyKeyError::new_err("pop_first(): DirectMap is empty"))?;
        self.changed();
        Ok(item)
    }

    /// Remove and return the entry with the largest key; raise
    /// ``KeyError`` if the map is empty.
    fn pop_last(&mut self) -> PyResult<(Key, Py<PyAny>)> {
        let item = self
            .map
            .pop_last()
            .ok_or_else(|| PyKeyError::new_err("pop_last(): DirectMap is empty"))?;
        self.changed();
        Ok(item)
    }

    /// Return the first entry at or after ``index``, or ``None``.
    fn next_occupant(
        &self,
        py: Python<'_>,
        index: &Bound<'_, PyAny>,
    ) -> PyResult<Option<(Key, Py<PyAny>)>> {
        let index = query_index(index)?.unwrap_or(0);
        Ok(if index < KEY_LIMIT {
            self.map
                .next_occupant(index as usize)
                .map(|(&key, value)| (key, value.clone_ref(py)))
        } else {
            None
        })
    }

    /// Return the last entry at or before ``index``, or ``None``.
    fn prev_occupant(
        &self,
        py: Python<'_>,
        index: &Bound<'_, PyAny>,
    ) -> PyResult<Option<(Key, Py<PyAny>)>> {
        let Some(index) = query_index(index)? else {
            return Ok(None);
        };
        Ok(self
            .map
            .prev_occupant(index.min(KEY_LIMIT - 1) as usize)
            .map(|(&key, value)| (key, value.clone_ref(py))))
    }

    /// Return the entries in ``[start, stop)``, in ascending key order.
    /// ``None`` leaves a bound open; integer bounds are clamped to the key domain.
    #[pyo3(signature = (start = None, stop = None))]
    fn items_range(
        &self,
        py: Python<'_>,
        start: Option<&Bound<'_, PyAny>>,
        stop: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<(Key, Py<PyAny>)>> {
        Ok(self
            .map
            .slot_range(query_bounds(start, stop)?)
            .map(|(&key, value)| (key, value.clone_ref(py)))
            .collect())
    }

    /// Return the keys in ``[start, stop)`` as a new ``uint32`` array.
    #[pyo3(signature = (start = None, stop = None))]
    fn keys_range<'py>(
        &self,
        py: Python<'py>,
        start: Option<&Bound<'_, PyAny>>,
        stop: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray1<Key>>> {
        Ok(PyArray1::from_vec(
            py,
            self.map
                .slot_range(query_bounds(start, stop)?)
                .map(|(&key, _)| key)
                .collect(),
        ))
    }

    /// Return the values in ``[start, stop)`` as a new list, in key order.
    #[pyo3(signature = (start = None, stop = None))]
    fn values_range<'py>(
        &self,
        py: Python<'py>,
        start: Option<&Bound<'_, PyAny>>,
        stop: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyList>> {
        PyList::new(
            py,
            self.map
                .slot_range(query_bounds(start, stop)?)
                .map(|(_, value)| value.bind(py)),
        )
    }

    /// Return the value for ``key``, first inserting ``default`` if ``key``
    /// is absent.
    #[pyo3(signature = (key, default = None))]
    fn setdefault(
        &mut self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        default: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let key = insert_key(key)?;
        if let Some(value) = self.map.get(key) {
            return Ok(value.clone_ref(py));
        }
        let default = default.unwrap_or_else(|| py.None());
        self.map.insert(key, default.clone_ref(py));
        self.changed();
        Ok(default)
    }

    /// Insert the pairs of ``other``: a mapping, or an iterable of
    /// ``(key, value)`` pairs. Like ``dict.update``, a failure part way leaves
    /// the earlier pairs inserted; use ``set_many`` for an all-or-nothing
    /// batch.
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
            let (key, value): (Bound<'_, PyAny>, Py<PyAny>) = pair?.extract()?;
            slf.try_borrow_mut()?.__setitem__(&key, value)?;
        }
        Ok(())
    }

    /// Remove every entry. Pages stay allocated for reuse.
    fn clear(&mut self) -> PyResult<()> {
        if !self.map.is_empty() {
            self.map.clear();
            self.changed();
        }
        Ok(())
    }

    /// Return the keys as a new ``uint32`` array, in ascending key order.
    fn keys<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Key>> {
        PyArray1::from_vec(py, self.map.keys().copied().collect())
    }

    /// Return the values as a new list, in ascending key order.
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, self.map.values().map(|value| value.bind(py)))
    }

    /// Return a list of ``(key, value)`` pairs in ascending key order.
    fn items(&self, py: Python<'_>) -> Vec<(Key, Py<PyAny>)> {
        self.map
            .iter()
            .map(|(&k, v)| (k, v.clone_ref(py)))
            .collect()
    }

    /// Set the value of each key in ``keys`` to the matching item of
    /// ``values`` (an iterable of the same length) and return how many keys
    /// were new. For a key listed twice, the last value wins.
    ///
    /// The batch is validated first: a bad key or a length mismatch raises
    /// and leaves the map unchanged.
    fn set_many(&mut self, keys: &Bound<'_, PyAny>, values: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.set_batch(keys, values)
    }

    /// Return a list of the values of ``keys``. A missing key takes
    /// ``default`` if given, else raises ``KeyError``.
    #[pyo3(signature = (keys, *default))]
    fn get_many<'py>(
        &self,
        py: Python<'py>,
        keys: &Bound<'py, PyAny>,
        default: &Bound<'py, PyTuple>,
    ) -> PyResult<Bound<'py, PyList>> {
        let default = optional_default(default, "get_many")?;
        let batch = KeyBatch::new(keys)?;
        let mut out = Vec::with_capacity(batch.len());
        let mut missing = None;
        batch.for_each(|k| match (k.and_then(|k| self.map.get(k)), &default) {
            (Some(value), _) => out.push(value.bind(py).clone()),
            (None, Some(value)) => out.push(value.clone()),
            (None, None) => {
                missing.get_or_insert(out.len());
                out.push(py.None().into_bound(py));
            }
        });
        match missing {
            Some(pos) => Err(PyKeyError::new_err(batch.item(pos)?.unbind())),
            None => PyList::new(py, out),
        }
    }

    /// Return a boolean array: whether each key in ``keys`` is present.
    fn contains_many<'py>(
        &self,
        py: Python<'py>,
        keys: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let batch = KeyBatch::new(keys)?;
        let mut out = Vec::with_capacity(batch.len());
        batch.for_each(|k| out.push(k.is_some_and(|k| self.map.contains_key(k))));
        Ok(PyArray1::from_vec(py, out))
    }

    /// Remove every key in ``keys`` that is present and return how many were
    /// removed. Absent and out-of-range keys are ignored.
    fn discard_many(&mut self, keys: &Bound<'_, PyAny>) -> PyResult<usize> {
        let batch = KeyBatch::new(keys)?;
        let mut present = Vec::new();
        batch.for_each(|k| present.extend(k.filter(|&k| self.map.contains_key(k))));
        if present.is_empty() {
            return Ok(0);
        }
        // Collect the removed values and drop them after the loop, so the
        // map is consistent before any `__del__` runs.
        let removed: Vec<Py<PyAny>> = present
            .into_iter()
            .filter_map(|k| self.map.remove(k))
            .collect();
        self.changed();
        Ok(removed.len())
    }

    /// Release unused memory; frees every page if the map is empty.
    fn shrink_to_fit(&mut self) -> PyResult<()> {
        self.map.shrink_to_fit();
        Ok(())
    }

    /// Number of allocated pages of entries.
    #[getter]
    fn allocated_pages(&self) -> usize {
        self.map.allocated_pages()
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        for value in self.map.values() {
            visit.call(value)?;
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        self.map.clear();
        self.changed();
    }

    /// Values are arbitrary Python objects.
    #[classattr]
    fn dtype() -> &'static str {
        "object"
    }
}

ordered_key_iterator!(DirectMapIterator, DirectMap, "DirectMap");

macro_rules! typed_direct_map {
    ($name:ident, $iter:ident, $pyname:literal, $t:ty) => {
        #[doc = concat!(
            "A map from integers in ``0 <= key < 2**32`` to ``",
            stringify!($t),
            "`` values, with O(1) insert, delete and lookup.\n\n",
            "Values live in pages allocated for the key ranges in use. Entries are ",
            "always visited in ascending key order. Ordered queries find nearby ",
            "entries or entries in a half-open range; array results are copies."
        )]
        #[pyclass(module = "paged_collections._core", name = $pyname, subclass)]
        pub(crate) struct $name {
            map: PagedDirectMap<Key, $t>,
            /// Changes on every structural change, so iterators can detect them.
            mods: u64,
        }

        impl $name {
            fn empty() -> Self {
                Self {
                    map: PagedDirectMap::new(),
                    mods: 0,
                }
            }

            fn changed(&mut self) {
                self.mods = self.mods.wrapping_add(1);
            }

            fn next_key_from(&self, index: u64) -> Option<Key> {
                if index < KEY_LIMIT {
                    self.map.next_occupant(index as usize).map(|(&key, _)| key)
                } else {
                    None
                }
            }

            fn set_batch(
                &mut self,
                keys: &Bound<'_, PyAny>,
                values: &Bound<'_, PyAny>,
            ) -> PyResult<usize> {
                let keys = KeyBatch::new(keys)?.to_insert_keys()?;
                let values = value_batch::<$t>(values, keys.len())?;
                let new = write_batch(&mut self.map, keys, values);
                if new > 0 {
                    self.changed();
                }
                Ok(new)
            }

            fn remove_present(&mut self, key: Key) -> PyResult<$t> {
                let value = self.map.remove(key);
                self.changed();
                Ok(value.expect("caller checked the key is present"))
            }
        }

        #[pymethods]
        impl $name {
            // `__new__` accepts and ignores any arguments, so subclasses can
            // define their own `__init__`; `__init__` fills the map.
            #[new]
            #[pyo3(signature = (*_args, **_kwargs), text_signature = "(keys=None, values=None)")]
            fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
                Self::empty()
            }

            #[pyo3(signature = (keys = None, values = None))]
            fn __init__(
                &mut self,
                keys: Option<&Bound<'_, PyAny>>,
                values: Option<&Bound<'_, PyAny>>,
            ) -> PyResult<()> {
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
                lookup_key(key).is_some_and(|k| self.map.contains_key(k))
            }

            fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<$t> {
                lookup_key(key)
                    .and_then(|k| self.map.get(k).copied())
                    .ok_or_else(|| PyKeyError::new_err(key.clone().unbind()))
            }

            fn __setitem__(&mut self, key: &Bound<'_, PyAny>, value: $t) -> PyResult<()> {
                let key = insert_key(key)?;
                if let Some(slot) = self.map.get_mut(key) {
                    *slot = value;
                } else {
                    self.map.insert(key, value);
                    self.changed();
                }
                Ok(())
            }

            fn __delitem__(&mut self, key: &Bound<'_, PyAny>) -> PyResult<()> {
                match lookup_key(key).filter(|&k| self.map.contains_key(k)) {
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
                let items = repr_items(this.map.iter().map(|(&k, v)| (k, *v)), py)?;
                Ok(format!("{}({{{items}}})", slf.get_type().name()?))
            }

            fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
                let py = slf.py();
                let this = slf.try_borrow()?;
                let contents = vec![this.keys(py).into_any(), this.values(py).into_any()];
                crate::reduce(slf.as_any(), &py.get_type::<Self>(), contents, None)
            }

            /// Return a copy.
            fn copy(&self) -> Self {
                Self {
                    map: self.map.clone(),
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
                match lookup_key(key).and_then(|k| self.map.get(k)) {
                    Some(value) => (*value).into_bound_py_any(py),
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
                match lookup_key(key).filter(|&k| self.map.contains_key(k)) {
                    Some(k) => {
                        self.remove_present(k)?.into_bound_py_any(py)
                    }
                    None => default.ok_or_else(|| PyKeyError::new_err(key.clone().unbind())),
                }
            }

            /// Remove and return the entry with the largest key; raise
            /// ``KeyError`` if the map is empty.
            fn popitem(&mut self) -> PyResult<(Key, $t)> {
                let item = self.map.pop_last().ok_or_else(|| {
                    PyKeyError::new_err(concat!("popitem(): ", $pyname, " is empty"))
                })?;
                self.changed();
                Ok(item)
            }

            /// Return the entry with the smallest key; raise ``KeyError`` if empty.
            fn first_item(&self) -> PyResult<(Key, $t)> {
                self.map.first_key_value()
                    .map(|(&key, value)| (key, *value))
                    .ok_or_else(|| PyKeyError::new_err(concat!("first_item(): ", $pyname, " is empty")))
            }

            /// Return the entry with the largest key; raise ``KeyError`` if empty.
            fn last_item(&self) -> PyResult<(Key, $t)> {
                self.map.last_key_value()
                    .map(|(&key, value)| (key, *value))
                    .ok_or_else(|| PyKeyError::new_err(concat!("last_item(): ", $pyname, " is empty")))
            }

            /// Remove and return the entry with the smallest key; raise
            /// ``KeyError`` if the map is empty.
            fn pop_first(&mut self) -> PyResult<(Key, $t)> {
                let item = self.map.pop_first().ok_or_else(|| {
                    PyKeyError::new_err(concat!("pop_first(): ", $pyname, " is empty"))
                })?;
                self.changed();
                Ok(item)
            }

            /// Remove and return the entry with the largest key; raise
            /// ``KeyError`` if the map is empty.
            fn pop_last(&mut self) -> PyResult<(Key, $t)> {
                let item = self.map.pop_last().ok_or_else(|| {
                    PyKeyError::new_err(concat!("pop_last(): ", $pyname, " is empty"))
                })?;
                self.changed();
                Ok(item)
            }

            /// Return the first entry at or after ``index``, or ``None``.
            fn next_occupant(&self, index: &Bound<'_, PyAny>) -> PyResult<Option<(Key, $t)>> {
                let index = query_index(index)?.unwrap_or(0);
                Ok(if index < KEY_LIMIT {
                    self.map.next_occupant(index as usize)
                        .map(|(&key, value)| (key, *value))
                } else {
                    None
                })
            }

            /// Return the last entry at or before ``index``, or ``None``.
            fn prev_occupant(&self, index: &Bound<'_, PyAny>) -> PyResult<Option<(Key, $t)>> {
                let Some(index) = query_index(index)? else { return Ok(None) };
                Ok(self.map.prev_occupant(index.min(KEY_LIMIT - 1) as usize)
                    .map(|(&key, value)| (key, *value)))
            }

            /// Return the entries in ``[start, stop)``, in ascending key order.
            /// ``None`` leaves a bound open; integer bounds are clamped to the key domain.
            #[pyo3(signature = (start = None, stop = None))]
            fn items_range(&self, start: Option<&Bound<'_, PyAny>>, stop: Option<&Bound<'_, PyAny>>)
                -> PyResult<Vec<(Key, $t)>> {
                Ok(self.map.slot_range(query_bounds(start, stop)?)
                    .map(|(&key, value)| (key, *value)).collect())
            }

            /// Return the keys in ``[start, stop)`` as a new ``uint32`` array.
            #[pyo3(signature = (start = None, stop = None))]
            fn keys_range<'py>(&self, py: Python<'py>, start: Option<&Bound<'_, PyAny>>, stop: Option<&Bound<'_, PyAny>>)
                -> PyResult<Bound<'py, PyArray1<Key>>> {
                Ok(PyArray1::from_vec(py, self.map.slot_range(query_bounds(start, stop)?)
                    .map(|(&key, _)| key).collect()))
            }

            /// Return the values in ``[start, stop)`` as a new array, in key order.
            #[pyo3(signature = (start = None, stop = None))]
            fn values_range<'py>(&self, py: Python<'py>, start: Option<&Bound<'_, PyAny>>, stop: Option<&Bound<'_, PyAny>>)
                -> PyResult<Bound<'py, PyArray1<$t>>> {
                Ok(PyArray1::from_vec(py, self.map.slot_range(query_bounds(start, stop)?)
                    .map(|(_, &value)| value).collect()))
            }

            /// Return the value for ``key``, first inserting ``default`` if
            /// ``key`` is absent.
            fn setdefault(&mut self, key: &Bound<'_, PyAny>, default: $t) -> PyResult<$t> {
                let key = insert_key(key)?;
                if let Some(value) = self.map.get(key) {
                    return Ok(*value);
                }
                self.map.insert(key, default);
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
                    self.map.clear();
                    self.changed();
                }
                Ok(())
            }

            /// Return the keys as a new ``uint32`` array, in ascending key order.
            fn keys<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Key>> {
                PyArray1::from_vec(py, self.map.keys().copied().collect())
            }

            #[doc = concat!("Return the values as a new ``", stringify!($t), "`` array, in ascending key order.")]
            fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<$t>> {
                PyArray1::from_vec(py, self.map.values().copied().collect())
            }

            /// Return a list of ``(key, value)`` pairs in ascending key order.
            fn items(&self) -> Vec<(Key, $t)> {
                self.map.iter().map(|(&k, v)| (k, *v)).collect()
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
                let batch = KeyBatch::new(keys)?;
                let mut out = Vec::with_capacity(batch.len());
                let mut missing = None;
                batch.for_each(|k| match (k.and_then(|k| self.map.get(k)), default) {
                    (Some(value), _) => out.push(*value),
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
                let batch = KeyBatch::new(keys)?;
                let mut out = Vec::with_capacity(batch.len());
                batch.for_each(|k| out.push(k.is_some_and(|k| self.map.contains_key(k))));
                Ok(PyArray1::from_vec(py, out))
            }

            /// Remove every key in ``keys`` that is present and return how
            /// many were removed. Absent and out-of-range keys are ignored.
            fn discard_many(&mut self, keys: &Bound<'_, PyAny>) -> PyResult<usize> {
                let batch = KeyBatch::new(keys)?;
                let mut present = Vec::new();
                batch.for_each(|k| present.extend(k.filter(|&k| self.map.contains_key(k))));
                if present.is_empty() {
                    return Ok(0);
                }
                let removed = present
                    .into_iter()
                    .filter(|&k| self.map.remove(k).is_some())
                    .count();
                self.changed();
                Ok(removed)
            }

            /// Release unused memory; frees every page if the map is empty.
            fn shrink_to_fit(&mut self) -> PyResult<()> {
                self.map.shrink_to_fit();
                Ok(())
            }

            /// Number of allocated pages of entries.
            #[getter]
            fn allocated_pages(&self) -> usize {
                self.map.allocated_pages()
            }

            /// The NumPy dtype name of the values.
            #[classattr]
            fn dtype() -> &'static str {
                <$t as TypedValue>::DTYPE
            }
        }

        ordered_key_iterator!($iter, $name, $pyname);
    };
}

typed_direct_map!(DirectMapF64, DirectMapF64Iterator, "DirectMapF64", f64);
typed_direct_map!(DirectMapF32, DirectMapF32Iterator, "DirectMapF32", f32);
typed_direct_map!(DirectMapI64, DirectMapI64Iterator, "DirectMapI64", i64);
typed_direct_map!(DirectMapI32, DirectMapI32Iterator, "DirectMapI32", i32);
typed_direct_map!(DirectMapU64, DirectMapU64Iterator, "DirectMapU64", u64);
typed_direct_map!(DirectMapU32, DirectMapU32Iterator, "DirectMapU32", u32);
typed_direct_map!(DirectMapBool, DirectMapBoolIterator, "DirectMapBool", bool);
