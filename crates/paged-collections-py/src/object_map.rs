//! `SparseMap`: `PagedSparseMap<u32, Py<PyAny>>`, for arbitrary values.

use numpy::PyArray1;
use paged_collections::sparse_map::PagedSparseMap;
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::{PyTraverseError, PyVisit};

use crate::convert::{Key, KeyBatch, insert_key, lookup_key};
use crate::export::{Exports, read_only_view};
use crate::iter::key_iterator;
use crate::repr_items;
use crate::typed_map::optional_default;

const NAME: &str = "SparseMap";

/// A map from integers in ``0 <= key < 2**32`` to arbitrary Python objects,
/// with O(1) insert, delete and lookup.
///
/// Keys and values live in two parallel contiguous arrays (in insertion
/// order until a removal moves the last entry into the gap); the key index
/// is paged, so memory follows the key ranges in use. For numeric values,
/// the typed maps (``SparseMapF64`` and friends) store values unboxed and
/// expose them as NumPy arrays.
///
/// ``keys_view()`` returns a zero-copy, read-only array of the keys; while
/// any view is alive, methods that would resize or reorder the map raise
/// ``BufferError``. Do not store a view of a map in that map: NumPy arrays
/// are invisible to the garbage collector, so the cycle is never freed.
#[pyclass(module = "paged_collections._core", name = "SparseMap", subclass)]
pub(crate) struct SparseMap {
    map: PagedSparseMap<Key, Py<PyAny>>,
    exports: Exports,
    /// Changes on every structural change, so iterators can detect them.
    mods: u64,
}

impl SparseMap {
    fn empty() -> Self {
        Self {
            map: PagedSparseMap::new(),
            exports: Exports::default(),
            mods: 0,
        }
    }

    fn changed(&mut self) {
        self.mods = self.mods.wrapping_add(1);
    }

    pub(crate) fn dense_keys(&self) -> &[Key] {
        self.map.key_slice()
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
        let new = crate::typed_map::write_batch(&mut self.map, &self.exports, NAME, keys, values)?;
        if new > 0 {
            self.changed();
        }
        Ok(new)
    }

    fn remove_present(&mut self, key: Key) -> PyResult<Py<PyAny>> {
        self.exports.check(NAME)?;
        let value = self.map.remove(key);
        self.changed();
        Ok(value.expect("caller checked the key is present"))
    }
}

#[pymethods]
impl SparseMap {
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
            self.exports.check(NAME)?;
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

    fn __iter__(slf: &Bound<'_, Self>) -> PyResult<SparseMapIterator> {
        SparseMapIterator::new(slf)
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

    /// Return a shallow copy: the values are shared, not copied. The copy
    /// has no views.
    fn copy(&self, py: Python<'_>) -> Self {
        let mut map = PagedSparseMap::with_capacity(self.map.len());
        for (&key, value) in self.map.iter() {
            map.insert(key, value.clone_ref(py));
        }
        Self {
            map,
            exports: Exports::default(),
            mods: 0,
        }
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

    /// Remove and return the last ``(key, value)`` pair in dense order;
    /// raise ``KeyError`` if the map is empty.
    fn popitem(&mut self) -> PyResult<(Key, Py<PyAny>)> {
        let &key = self
            .map
            .key_slice()
            .last()
            .ok_or_else(|| PyKeyError::new_err("popitem(): SparseMap is empty"))?;
        Ok((key, self.remove_present(key)?))
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
        self.exports.check(NAME)?;
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
            self.exports.check(NAME)?;
            self.map.clear();
            self.changed();
        }
        Ok(())
    }

    /// Return the keys as a new ``uint32`` array, in dense order.
    fn keys<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Key>> {
        PyArray1::from_slice(py, self.map.key_slice())
    }

    /// Return the values as a new list, in dense order.
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, self.map.values().map(|value| value.bind(py)))
    }

    /// Return a list of ``(key, value)`` pairs in dense order.
    fn items(&self, py: Python<'_>) -> Vec<(Key, Py<PyAny>)> {
        self.map
            .iter()
            .map(|(&k, v)| (k, v.clone_ref(py)))
            .collect()
    }

    /// Return a zero-copy, read-only ``uint32`` array of the keys in dense
    /// order. See the class docs for what views prevent.
    fn keys_view<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyArray1<Key>>> {
        let this = slf.try_borrow()?;
        // SAFETY: the keys are the map's dense key `Vec`, and every method
        // that could move them checks `exports` first.
        unsafe { read_only_view(slf.as_any(), &this.exports, this.map.key_slice()) }
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
        self.exports.check(NAME)?;
        // Collect the removed values and drop them after the loop, so the
        // map is consistent before any `__del__` runs.
        let removed: Vec<Py<PyAny>> = present
            .into_iter()
            .filter_map(|k| self.map.remove(k))
            .collect();
        self.changed();
        Ok(removed.len())
    }

    /// Sort the entries by key, ascending.
    fn sort_keys(&mut self) -> PyResult<()> {
        self.exports.check(NAME)?;
        self.map.sort_unstable_keys();
        self.changed();
        Ok(())
    }

    /// Reserve room for at least ``additional`` more entries.
    fn reserve(&mut self, additional: usize) -> PyResult<()> {
        self.exports.check(NAME)?;
        self.map
            .try_reserve(additional)
            .map_err(crate::reserve_error)
    }

    /// Release unused memory; frees every page if the map is empty.
    fn shrink_to_fit(&mut self) -> PyResult<()> {
        self.exports.check(NAME)?;
        self.map.shrink_to_fit();
        Ok(())
    }

    /// Number of allocated pages of the key index.
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
        // Called by the garbage collector to break a cycle. While a keys
        // view is alive the keys must stay put, so only the values are
        // dropped (replaced by `None`).
        if self.exports.active() {
            Python::attach(|py| {
                for value in self.map.values_mut() {
                    *value = py.None();
                }
            });
        } else {
            self.map.clear();
            self.changed();
        }
    }

    /// Values are arbitrary Python objects.
    #[classattr]
    fn dtype() -> &'static str {
        "object"
    }
}

key_iterator!(SparseMapIterator, SparseMap, "SparseMap");
