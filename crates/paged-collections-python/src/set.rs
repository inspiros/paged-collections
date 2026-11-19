//! `SparseSet`: Python wrapper of `PagedSparseSet<u32>`.

use crate::storage::SetInner;
use numpy::PyArray1;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyKeyError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;
use pyo3::types::{PyDict, PyInt, PyList, PySet, PyTuple};

use crate::convert::{Key, KeyBatch, insert_key, lookup_key};
use crate::export::{Exports, read_only_view};
use crate::iter::key_iterator;

const NAME: &str = "SparseSet";

/// A set of integers in ``0 <= key < 2**32`` with O(1) add, discard and
/// membership, and a contiguous array of its keys.
///
/// Memory follows the key ranges in use: keys are indexed through pages of
/// configurable size, allocated on first use. Keys keep insertion order until a
/// removal, which moves the last key into the gap. Copied algebra results retain
/// surviving receiver keys in dense order, then append new keys in operand
/// iteration order (duplicates appear once). Results use the receiver's page
/// size and are always base ``SparseSet`` instances, including for subclasses.
/// ``difference_update`` and ``-=`` use the same swap-removal order as
/// ``discard``; other in-place algebra keeps surviving receiver order.
///
/// Batch methods (``add_many``, ``discard_many``, ``contains_many``) take
/// NumPy arrays or sequences of integers. ``view()`` returns a zero-copy,
/// read-only array of the keys; while any view is alive, methods that would
/// resize or reorder the set raise ``BufferError``.
#[pyclass(
    module = "paged_collections._core",
    name = "SparseSet",
    subclass,
    weakref
)]
pub(crate) struct SparseSet {
    pub(crate) set: SetInner<Key>,
    pub(crate) exports: Exports,
    /// Changes on every structural change, so iterators can detect them.
    pub(crate) mods: u64,
}

#[derive(Clone, Copy)]
enum Algebra {
    Union,
    Intersection,
    Difference,
    SymmetricDifference,
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

    // Builtin sets bypass a set subclass's __iter__ for set operands. Snapshot
    // our native storage too; arbitrary iterable callbacks run unborrowed.
    fn operand<'py>(other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        if let Ok(other) = other.cast::<Self>() {
            let keys = other.try_borrow()?.set.as_slice().to_vec();
            Ok(PyList::new(other.py(), keys)?.into_any())
        } else {
            Ok(other.clone())
        }
    }

    fn operand_set<'py>(other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PySet>> {
        let set = PySet::empty(other.py())?;
        set.call_method1("update", (Self::operand(other)?,))?;
        Ok(set)
    }

    fn algebra(
        slf: &Bound<'_, Self>,
        others: &[Bound<'_, PyAny>],
        op: Algebra,
        inplace: bool,
    ) -> PyResult<Self> {
        let (mut keys, page_size, mods) = {
            let this = slf.try_borrow()?;
            if inplace {
                this.exports.check(NAME)?;
            }
            (
                this.set.as_slice().to_vec(),
                this.set.page_size(),
                this.mods,
            )
        };
        for other in others {
            match op {
                Algebra::Union | Algebra::SymmetricDifference => {
                    let mut seen = std::collections::HashSet::new();
                    let mut incoming = Vec::new();
                    for key in Self::operand(other)?.try_iter()? {
                        let key = insert_key(&key?)?;
                        if seen.insert(key) {
                            incoming.push(key);
                        }
                    }
                    let present: std::collections::HashSet<_> = keys.iter().copied().collect();
                    if matches!(op, Algebra::SymmetricDifference) {
                        keys.retain(|key| !seen.contains(key));
                    }
                    keys.extend(incoming.into_iter().filter(|key| !present.contains(key)));
                }
                Algebra::Intersection | Algebra::Difference => {
                    let other = Self::operand_set(other)?;
                    let mut retained = Vec::with_capacity(keys.len());
                    for key in keys {
                        if other.contains(key)? == matches!(op, Algebra::Intersection) {
                            retained.push(key);
                        }
                    }
                    keys = retained;
                }
            }
        }
        let mut out = Self::empty();
        out.set.set_page_size(page_size)?;
        out.set
            .try_reserve(keys.len())
            .map_err(crate::reserve_error)?;
        out.set.extend(keys);
        if inplace {
            let mut this = slf.try_borrow_mut()?;
            this.exports.check(NAME)?;
            if this.mods != mods {
                return Err(PyRuntimeError::new_err(
                    "SparseSet changed during set operation",
                ));
            }
            if this.set.as_slice() != out.set.as_slice() {
                this.set
                    .try_reserve(out.set.len())
                    .map_err(crate::reserve_error)?;
                this.set.clear();
                this.set.extend(out.set.iter().copied());
                this.changed();
            }
        }
        Ok(out)
    }

    // Like set.update/difference_update, commit each yielded key so a failing
    // iterator preserves earlier changes. Advance, hash, convert and compare
    // only outside borrows; callbacks can read or mutate the live receiver.
    fn streaming_update(
        slf: &Bound<'_, Self>,
        others: &[Bound<'_, PyAny>],
        remove: bool,
    ) -> PyResult<()> {
        slf.try_borrow()?.exports.check(NAME)?;
        for other in others {
            for key in Self::operand(other)?.try_iter()? {
                let key = key?;
                if !remove {
                    let key = insert_key(&key)?;
                    let mut this = slf.try_borrow_mut()?;
                    this.exports.check(NAME)?;
                    if !this.set.contains(key) {
                        this.set.try_reserve(1).map_err(crate::reserve_error)?;
                        this.set.insert(key);
                        this.changed();
                    }
                } else {
                    // Exact ints are the usual case. For other hashable keys,
                    // use Python set membership (e.g. 1.0 compares equal to 1).
                    let removed = if key.is_exact_instance_of::<PyInt>() {
                        lookup_key(&key).into_iter().collect::<Vec<_>>()
                    } else {
                        let probe = PySet::new(slf.py(), [&key])?;
                        let keys = slf.try_borrow()?.set.as_slice().to_vec();
                        let mut removed = Vec::new();
                        for key in keys {
                            if probe.contains(key)? {
                                removed.push(key);
                            }
                        }
                        removed
                    };
                    let mut this = slf.try_borrow_mut()?;
                    this.exports.check(NAME)?;
                    for key in removed {
                        if this.set.remove(key) {
                            this.changed();
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn subset_order(&self, other: &Self, op: CompareOp) -> bool {
        let subset = || self.set.iter().all(|&key| other.set.contains(key));
        let superset = || other.set.iter().all(|&key| self.set.contains(key));
        match op {
            CompareOp::Lt => self.set.len() < other.set.len() && subset(),
            CompareOp::Le => self.set.len() <= other.set.len() && subset(),
            CompareOp::Gt => self.set.len() > other.set.len() && superset(),
            CompareOp::Ge => self.set.len() >= other.set.len() && superset(),
            _ => unreachable!(),
        }
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

    /// Return a base SparseSet in surviving receiver order, then new operand
    /// keys in iteration order, with the receiver's page size.
    #[pyo3(signature = (*others))]
    fn union(slf: &Bound<'_, Self>, others: &Bound<'_, PyTuple>) -> PyResult<Self> {
        Self::algebra(
            slf,
            &others.iter().collect::<Vec<_>>(),
            Algebra::Union,
            false,
        )
    }
    /// Return a base SparseSet in surviving receiver order, then new operand
    /// keys in iteration order, with the receiver's page size.
    #[pyo3(signature = (*others))]
    fn intersection(slf: &Bound<'_, Self>, others: &Bound<'_, PyTuple>) -> PyResult<Self> {
        Self::algebra(
            slf,
            &others.iter().collect::<Vec<_>>(),
            Algebra::Intersection,
            false,
        )
    }
    /// Return a base SparseSet in surviving receiver order, then new operand
    /// keys in iteration order, with the receiver's page size.
    #[pyo3(signature = (*others))]
    fn difference(slf: &Bound<'_, Self>, others: &Bound<'_, PyTuple>) -> PyResult<Self> {
        Self::algebra(
            slf,
            &others.iter().collect::<Vec<_>>(),
            Algebra::Difference,
            false,
        )
    }
    /// Return keys in exactly one operand, in surviving receiver order then
    /// new operand iteration order, with the receiver's page size.
    fn symmetric_difference(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::algebra(
            slf,
            std::slice::from_ref(other),
            Algebra::SymmetricDifference,
            false,
        )
    }
    /// Add keys as they are yielded; iterator errors preserve earlier additions.
    #[pyo3(signature = (*others))]
    fn update(slf: &Bound<'_, Self>, others: &Bound<'_, PyTuple>) -> PyResult<()> {
        Self::streaming_update(slf, &others.iter().collect::<Vec<_>>(), false)
    }
    #[pyo3(signature = (*others))]
    fn intersection_update(slf: &Bound<'_, Self>, others: &Bound<'_, PyTuple>) -> PyResult<()> {
        Self::algebra(
            slf,
            &others.iter().collect::<Vec<_>>(),
            Algebra::Intersection,
            true,
        )?;
        Ok(())
    }
    /// Discard keys as yielded, using swap removal. Iterator errors preserve
    /// earlier removals, like builtin set.difference_update.
    #[pyo3(signature = (*others))]
    fn difference_update(slf: &Bound<'_, Self>, others: &Bound<'_, PyTuple>) -> PyResult<()> {
        Self::streaming_update(slf, &others.iter().collect::<Vec<_>>(), true)
    }
    fn symmetric_difference_update(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        Self::algebra(
            slf,
            std::slice::from_ref(other),
            Algebra::SymmetricDifference,
            true,
        )?;
        Ok(())
    }

    fn issubset(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let other = Self::operand_set(other)?;
        let keys = slf.try_borrow()?.set.as_slice().to_vec();
        for key in keys {
            if !other.contains(key)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn issuperset(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let keys = slf.try_borrow()?.set.as_slice().to_vec();
        let own = PySet::new(slf.py(), keys)?;
        own.call_method1("issuperset", (Self::operand(other)?,))?
            .extract()
    }

    fn isdisjoint(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let keys = slf.try_borrow()?.set.as_slice().to_vec();
        let own = PySet::new(slf.py(), keys)?;
        own.call_method1("isdisjoint", (Self::operand(other)?,))?
            .extract()
    }
    fn __or__(slf: &Bound<'_, Self>, other: &Bound<'_, Self>) -> PyResult<Self> {
        Self::algebra(slf, &[other.as_any().clone()], Algebra::Union, false)
    }
    fn __ior__(slf: &Bound<'_, Self>, other: &Bound<'_, Self>) -> PyResult<()> {
        Self::streaming_update(slf, &[other.as_any().clone()], false)
    }
    fn __and__(slf: &Bound<'_, Self>, other: &Bound<'_, Self>) -> PyResult<Self> {
        Self::algebra(slf, &[other.as_any().clone()], Algebra::Intersection, false)
    }
    fn __iand__(slf: &Bound<'_, Self>, other: &Bound<'_, Self>) -> PyResult<()> {
        Self::algebra(slf, &[other.as_any().clone()], Algebra::Intersection, true)?;
        Ok(())
    }
    fn __sub__(slf: &Bound<'_, Self>, other: &Bound<'_, Self>) -> PyResult<Self> {
        Self::algebra(slf, &[other.as_any().clone()], Algebra::Difference, false)
    }
    fn __isub__(slf: &Bound<'_, Self>, other: &Bound<'_, Self>) -> PyResult<()> {
        Self::streaming_update(slf, &[other.as_any().clone()], true)
    }
    fn __xor__(slf: &Bound<'_, Self>, other: &Bound<'_, Self>) -> PyResult<Self> {
        Self::algebra(
            slf,
            &[other.as_any().clone()],
            Algebra::SymmetricDifference,
            false,
        )
    }
    fn __ixor__(slf: &Bound<'_, Self>, other: &Bound<'_, Self>) -> PyResult<()> {
        Self::algebra(
            slf,
            &[other.as_any().clone()],
            Algebra::SymmetricDifference,
            true,
        )?;
        Ok(())
    }
    fn __lt__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let Ok(other) = other.cast::<Self>() else {
            return Ok(py.NotImplemented());
        };
        let other = other.try_borrow()?;
        self.subset_order(&other, CompareOp::Lt).into_py_any(py)
    }
    fn __le__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let Ok(other) = other.cast::<Self>() else {
            return Ok(py.NotImplemented());
        };
        let other = other.try_borrow()?;
        self.subset_order(&other, CompareOp::Le).into_py_any(py)
    }
    fn __gt__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let Ok(other) = other.cast::<Self>() else {
            return Ok(py.NotImplemented());
        };
        let other = other.try_borrow()?;
        self.subset_order(&other, CompareOp::Gt).into_py_any(py)
    }
    fn __ge__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let Ok(other) = other.cast::<Self>() else {
            return Ok(py.NotImplemented());
        };
        let other = other.try_borrow()?;
        self.subset_order(&other, CompareOp::Ge).into_py_any(py)
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

    /// Return the smallest key. Raise KeyError if empty.
    fn first(&self) -> PyResult<Key> {
        self.set
            .first()
            .ok_or_else(|| PyKeyError::new_err("first(): SparseSet is empty"))
    }

    /// Return the largest key. Raise KeyError if empty.
    fn last(&self) -> PyResult<Key> {
        self.set
            .last()
            .ok_or_else(|| PyKeyError::new_err("last(): SparseSet is empty"))
    }

    /// Remove and return the smallest key. Raise KeyError if empty.
    fn pop_first(&mut self) -> PyResult<Key> {
        if !self.set.is_empty() {
            self.exports.check(NAME)?;
        }
        let key = self
            .set
            .pop_first()
            .ok_or_else(|| PyKeyError::new_err("pop_first(): SparseSet is empty"))?;
        self.changed();
        Ok(key)
    }

    /// Remove and return the largest key. Raise KeyError if empty.
    fn pop_last(&mut self) -> PyResult<Key> {
        if !self.set.is_empty() {
            self.exports.check(NAME)?;
        }
        let key = self
            .set
            .pop_last()
            .ok_or_else(|| PyKeyError::new_err("pop_last(): SparseSet is empty"))?;
        self.changed();
        Ok(key)
    }

    /// Return the first key at or after index, or None.
    fn next_occupant(slf: &Bound<'_, Self>, index: &Bound<'_, PyAny>) -> PyResult<Option<Key>> {
        let index = insert_key(index)?;
        Ok(slf.try_borrow()?.set.next_occupant(index as usize))
    }

    /// Return the last key at or before index, or None.
    fn prev_occupant(slf: &Bound<'_, Self>, index: &Bound<'_, PyAny>) -> PyResult<Option<Key>> {
        let index = insert_key(index)?;
        Ok(slf.try_borrow()?.set.prev_occupant(index as usize))
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
