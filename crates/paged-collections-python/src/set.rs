//! `SparseSet`: Python wrapper of `PagedSparseSet<u32>`.

use crate::storage::SetInner;
use numpy::PyArray1;
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::convert::{Key, KeyBatch, insert_key, lookup_key};
use crate::export::{Exports, read_only_view};
use crate::iter::key_iterator;

const NAME: &str = "SparseSet";

/// A set of integers in ``0 <= key < 2**32`` with O(1) add, discard and
/// membership, and a contiguous array of its keys.
///
/// Memory follows the key ranges in use: keys are indexed through pages of
/// configurable size, allocated on first use. Keys keep insertion order until a
/// removal, which moves the last key into the gap.
///
/// Batch methods (``add_many``, ``discard_many``, ``contains_many``) take
/// NumPy arrays or sequences of integers. ``view()`` returns a zero-copy,
/// read-only array of the keys; while any view is alive, methods that would
/// resize or reorder the set raise ``BufferError``.
#[pyclass(module = "paged_collections._core", name = "SparseSet", subclass)]
pub(crate) struct SparseSet {
    pub(crate) set: SetInner<Key>,
    pub(crate) exports: Exports,
    /// Changes on every structural change, so iterators can detect them.
    pub(crate) mods: u64,
}

impl SparseSet {
    fn empty() -> Self {
        Self {
            set: SetInner::default(),
            exports: Exports::default(),
            mods: 0,
        }
    }

    fn changed(&mut self) {
        self.mods = self.mods.wrapping_add(1);
    }

    pub(crate) fn dense_keys(&self) -> &[Key] {
        self.set.as_slice()
    }

    /// Adds every key of `keys`; returns how many were new.
    fn add_batch(&mut self, keys: &Bound<'_, PyAny>) -> PyResult<usize> {
        let keys = KeyBatch::new(keys)?.to_insert_keys()?;
        if keys.iter().all(|&k| self.set.contains(k)) {
            return Ok(0);
        }
        self.exports.check(NAME)?;
        let before = self.set.len();
        self.set
            .try_reserve(keys.len())
            .map_err(crate::reserve_error)?;
        self.set.extend(keys);
        self.changed();
        Ok(self.set.len() - before)
    }
}

#[pymethods]
impl SparseSet {
    // `__new__` accepts and ignores any arguments, so subclasses can define
    // their own `__init__`; `__init__` fills the collection, as for `dict`.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs), text_signature = "(keys=None, *, page_size=DEFAULT_PAGE_SIZE)")]
    fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        Self::empty()
    }

    #[pyo3(signature = (keys = None, *, page_size = crate::DEFAULT_PAGE_SIZE))]
    fn __init__(
        &mut self,
        keys: Option<&Bound<'_, PyAny>>,
        #[pyo3(from_py_with = crate::extract_page_size)] page_size: usize,
    ) -> PyResult<()> {
        if self.set.page_size() != page_size {
            self.exports.check(NAME)?;
            self.set.set_page_size(page_size)?;
            self.changed();
        }
        if let Some(keys) = keys {
            self.add_batch(keys)?;
        }
        Ok(())
    }

    fn __len__(&self) -> usize {
        self.set.len()
    }

    fn __contains__(&self, key: &Bound<'_, PyAny>) -> bool {
        lookup_key(key).is_some_and(|k| self.set.contains(k))
    }

    fn __iter__(slf: &Bound<'_, Self>) -> PyResult<SparseSetIterator> {
        SparseSetIterator::new(slf)
    }

    fn __eq__(&self, other: PyRef<'_, Self>) -> bool {
        self.set == other.set
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let keys = crate::truncated(slf.try_borrow()?.set.iter().map(Key::to_string));
        Ok(format!(
            "{}([{keys}]{})",
            slf.get_type().name()?,
            crate::page_size_repr(slf.try_borrow()?.set.page_size())
        ))
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let keys = slf.try_borrow()?.to_array(slf.py());
        crate::reduce(
            slf.as_any(),
            &slf.py().get_type::<Self>(),
            vec![keys.into_any()],
            None,
        )
    }

    /// Return a shallow copy (it has no views).
    fn copy(&self) -> Self {
        Self {
            set: self.set.clone(),
            exports: Exports::default(),
            mods: 0,
        }
    }

    /// Add ``key``.
    fn add(&mut self, key: &Bound<'_, PyAny>) -> PyResult<()> {
        let key = insert_key(key)?;
        if !self.set.contains(key) {
            self.exports.check(NAME)?;
            self.set.insert(key);
            self.changed();
        }
        Ok(())
    }

    /// Remove ``key`` if present.
    fn discard(&mut self, key: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Some(k) = lookup_key(key).filter(|&k| self.set.contains(k)) {
            self.exports.check(NAME)?;
            self.set.remove(k);
            self.changed();
        }
        Ok(())
    }

    /// Remove ``key``; raise ``KeyError`` if it is absent.
    fn remove(&mut self, key: &Bound<'_, PyAny>) -> PyResult<()> {
        match lookup_key(key).filter(|&k| self.set.contains(k)) {
            Some(k) => {
                self.exports.check(NAME)?;
                self.set.remove(k);
                self.changed();
                Ok(())
            }
            None => Err(PyKeyError::new_err(key.clone().unbind())),
        }
    }

    /// Remove and return the last key in dense order; raise ``KeyError`` if
    /// the set is empty.
    fn pop(&mut self) -> PyResult<Key> {
        let &key = self
            .set
            .as_slice()
            .last()
            .ok_or_else(|| PyKeyError::new_err("pop from an empty SparseSet"))?;
        self.exports.check(NAME)?;
        self.set.remove(key);
        self.changed();
        Ok(key)
    }

    /// Remove every key. Pages stay allocated for reuse.
    fn clear(&mut self) -> PyResult<()> {
        if !self.set.is_empty() {
            self.exports.check(NAME)?;
            self.set.clear();
            self.changed();
        }
        Ok(())
    }

    /// Add every key in ``keys`` (an array or sequence of integers) and
    /// return how many were not already present.
    ///
    /// The batch is validated first: an out-of-range key raises
    /// ``OverflowError`` and leaves the set unchanged.
    fn add_many(&mut self, keys: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.add_batch(keys)
    }

    /// Remove every key in ``keys`` that is present and return how many were
    /// removed. Absent and out-of-range keys are ignored.
    fn discard_many(&mut self, keys: &Bound<'_, PyAny>) -> PyResult<usize> {
        let batch = KeyBatch::lookup(keys)?;
        let mut present = Vec::new();
        batch.for_each(|k| present.extend(k.filter(|&k| self.set.contains(k))));
        if present.is_empty() {
            return Ok(0);
        }
        self.exports.check(NAME)?;
        // A key listed twice is removed once.
        let removed = present.into_iter().filter(|&k| self.set.remove(k)).count();
        self.changed();
        Ok(removed)
    }

    /// Return a boolean array: whether each key in ``keys`` is present.
    fn contains_many<'py>(
        &self,
        py: Python<'py>,
        keys: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let batch = KeyBatch::lookup(keys)?;
        let mut out = Vec::with_capacity(batch.len());
        batch.for_each(|k| out.push(k.is_some_and(|k| self.set.contains(k))));
        Ok(PyArray1::from_vec(py, out))
    }

    /// Return the keys as a new ``uint32`` array, in dense order.
    fn to_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Key>> {
        PyArray1::from_slice(py, self.set.as_slice())
    }

    /// Return a zero-copy, read-only ``uint32`` array of the keys in dense
    /// order. While it (or an array derived from it) is alive, methods that
    /// would resize or reorder the set raise ``BufferError``.
    fn view<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyArray1<Key>>> {
        let this = slf.try_borrow()?;
        // SAFETY: the keys are the set's dense `Vec`, and every method that
        // could move them checks `exports` first.
        unsafe { read_only_view(slf.as_any(), &this.exports, this.set.as_slice()) }
    }

    /// Sort the keys in ascending order.
    fn sort(&mut self) -> PyResult<()> {
        self.exports.check(NAME)?;
        self.set.sort_unstable();
        self.changed();
        Ok(())
    }

    /// Reserve room for at least ``additional`` more keys.
    fn reserve(&mut self, additional: usize) -> PyResult<()> {
        self.exports.check(NAME)?;
        self.set
            .try_reserve(additional)
            .map_err(crate::reserve_error)
    }

    /// Release unused memory; frees every page if the set is empty.
    fn shrink_to_fit(&mut self) -> PyResult<()> {
        self.exports.check(NAME)?;
        self.set.shrink_to_fit();
        Ok(())
    }

    /// Slots per page for this instance.
    #[getter]
    fn page_size(&self) -> usize {
        self.set.page_size()
    }

    /// Number of allocated pages of the sparse index.
    #[getter]
    fn allocated_pages(&self) -> usize {
        self.set.allocated_pages()
    }
}

key_iterator!(SparseSetIterator, SparseSet, "SparseSet");
