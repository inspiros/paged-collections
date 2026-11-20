//! `BitSet`: Python wrapper of `PagedBitSet<u32>`.

use crate::DEFAULT_BIT_PAGE_SIZE;
use crate::convert::{KEY_LIMIT, Key, KeyBatch, TypedValue, as_ndarray, insert_key, lookup_key};
use crate::storage::{BitSetInner, BitSetOp};
use numpy::{PyArray1, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyKeyError, PyOverflowError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;
use pyo3::types::{PyDict, PyInt, PySet, PyTuple, PyType};
use std::collections::HashSet;
use std::ops::{Range, RangeInclusive};

/// Validate both endpoints before checking whether the range is empty.
fn bounds(start: &Bound<'_, PyAny>, stop: &Bound<'_, PyAny>) -> PyResult<Range<u64>> {
    fn endpoint(value: &Bound<'_, PyAny>) -> PyResult<u64> {
        // Normalize __index__ once, then use the scalar key conversion for
        // errors. The only exception to the key domain is the endpoint 2**32.
        let int = if value.is_instance_of::<PyInt>() {
            value.clone()
        } else {
            value
                .py()
                .import("operator")?
                .call_method1("index", (value,))?
        };
        match int.extract::<u64>() {
            Ok(index) if index <= KEY_LIMIT => Ok(index),
            _ => insert_key(&int).map(u64::from),
        }
    }
    Ok(endpoint(start)?..endpoint(stop)?)
}

/// The inclusive end represents 2**32 - 1 even on a 32-bit target.
fn key_span(bounds: &Range<u64>) -> Option<RangeInclusive<Key>> {
    (!bounds.is_empty()).then(|| bounds.start as Key..=(bounds.end - 1) as Key)
}

fn matching_keys(
    keys: impl IntoIterator<Item = Key>,
    other: &Bound<'_, PySet>,
    matching: bool,
) -> PyResult<Vec<Key>> {
    let mut matches = Vec::new();
    for key in keys {
        if other.contains(key)? == matching {
            matches.push(key);
        }
    }
    Ok(matches)
}

/// An ordered set of integers in ``0 <= key < 2**32``, with one bit per
/// slot in an allocated page. The page size counts 64-bit words.
/// Iteration and NumPy key arrays are in ascending order.
#[pyclass(module = "paged_collections._core", subclass, weakref)]
pub(crate) struct BitSet {
    set: BitSetInner,
    mods: u64,
}

impl BitSet {
    fn with_storage(set: BitSetInner) -> Self {
        Self { set, mods: 0 }
    }

    fn record_change(&mut self, changed: bool) {
        if changed {
            self.mods = self.mods.wrapping_add(1);
        }
    }

    fn insert_keys(&mut self, keys: Vec<Key>) -> usize {
        let before = self.set.len();
        self.set.extend(keys);
        let added = self.set.len() - before;
        self.record_change(added != 0);
        added
    }

    fn apply_native(slf: &Bound<'_, Self>, other: &Bound<'_, Self>, op: BitSetOp) -> PyResult<()> {
        if slf.is(other) {
            if matches!(op, BitSetOp::Difference | BitSetOp::SymmetricDifference) {
                slf.try_borrow_mut()?.clear();
            }
        } else {
            let right = other.try_borrow()?;
            let mut left = slf.try_borrow_mut()?;
            let changed = left.set.assign(&right.set, op);
            left.record_change(changed);
        }
        Ok(())
    }

    fn combine_native(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, Self>,
        op: BitSetOp,
    ) -> PyResult<Self> {
        let left = slf.try_borrow()?;
        let right = other.try_borrow()?;
        Ok(Self::with_storage(left.set.combined(&right.set, op)))
    }

    // The candidate is separately owned so arbitrary iterable callbacks
    // run without a PyO3 borrow of the receiver.
    fn apply_operand(
        set: &mut BitSetInner,
        other: &Bound<'_, PyAny>,
        op: BitSetOp,
    ) -> PyResult<()> {
        if let Ok(other) = other.cast::<Self>() {
            set.assign(&other.try_borrow()?.set, op);
            return Ok(());
        }
        match op {
            BitSetOp::Union => {
                for key in other.try_iter()? {
                    set.insert(insert_key(&key?)?);
                }
            }
            BitSetOp::Intersection | BitSetOp::Difference => {
                let operand = PySet::empty(other.py())?;
                operand.call_method1("update", (other,))?;
                let matching = matches!(op, BitSetOp::Difference);
                for key in matching_keys(set.iter(), &operand, matching)? {
                    set.remove(key);
                }
            }
            BitSetOp::SymmetricDifference => {
                let mut keys = HashSet::new();
                for key in other.try_iter()? {
                    let key = insert_key(&key?)?;
                    if keys.insert(key) && !set.remove(key) {
                        set.insert(key);
                    }
                }
            }
        }
        Ok(())
    }

    fn algebra(slf: &Bound<'_, Self>, others: &[Bound<'_, PyAny>], op: BitSetOp) -> PyResult<Self> {
        if let [other] = others {
            if let Ok(other) = other.cast::<Self>() {
                return Self::combine_native(slf, other, op);
            }
        }
        let mut set = slf.try_borrow()?.set.clone();
        for other in others {
            Self::apply_operand(&mut set, other, op)?;
        }
        Ok(Self::with_storage(set))
    }

    fn staged_update(
        slf: &Bound<'_, Self>,
        others: &[Bound<'_, PyAny>],
        op: BitSetOp,
    ) -> PyResult<()> {
        if others.is_empty() {
            return Ok(());
        }
        if let [other] = others {
            if let Ok(other) = other.cast::<Self>() {
                return Self::apply_native(slf, other, op);
            }
        }
        let (mut set, mods) = {
            let this = slf.try_borrow()?;
            (this.set.clone(), this.mods)
        };
        for other in others {
            Self::apply_operand(&mut set, other, op)?;
        }
        let mut this = slf.try_borrow_mut()?;
        if this.mods != mods {
            return Err(PyRuntimeError::new_err(
                "BitSet changed during set operation",
            ));
        }
        if set.len() != this.set.len() || !set.is_subset(&this.set) {
            this.set = set;
            this.record_change(true);
        }
        Ok(())
    }

    // Like SparseSet, update and difference_update commit each yielded key.
    fn streaming_update(
        slf: &Bound<'_, Self>,
        others: &Bound<'_, PyTuple>,
        op: BitSetOp,
    ) -> PyResult<()> {
        for other in others.iter() {
            if let Ok(other) = other.cast::<Self>() {
                Self::apply_native(slf, other, op)?;
                continue;
            }
            for item in other.try_iter()? {
                let item = item?;
                if matches!(op, BitSetOp::Union) {
                    Self::add(slf, &item)?;
                } else if item.is_exact_instance_of::<PyInt>() {
                    Self::discard(slf, &item)?;
                } else {
                    // Use Python membership for foreign objects that may
                    // compare equal to a key (e.g. 1.0).
                    let probe = PySet::new(slf.py(), [&item])?;
                    let keys: Vec<_> = slf.try_borrow()?.set.iter().collect();
                    let removed = matching_keys(keys, &probe, true)?;
                    let mut this = slf.try_borrow_mut()?;
                    let changed = removed
                        .into_iter()
                        .filter(|&key| this.set.remove(key))
                        .count();
                    this.record_change(changed != 0);
                }
            }
        }
        Ok(())
    }
}

#[pymethods]
impl BitSet {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    // Subclasses can define their own initializer, just as for SparseSet.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs), text_signature = "(keys=None, *, page_size=DEFAULT_BIT_PAGE_SIZE)")]
    fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        Self::with_storage(BitSetInner::default())
    }

    #[pyo3(signature = (keys = None, *, page_size = DEFAULT_BIT_PAGE_SIZE))]
    fn __init__(
        slf: &Bound<'_, Self>,
        keys: Option<&Bound<'_, PyAny>>,
        #[pyo3(from_py_with = crate::extract_page_size)] page_size: usize,
    ) -> PyResult<()> {
        let keys = keys
            .map(|keys| KeyBatch::<Key>::new(keys)?.to_insert_keys())
            .transpose()?;
        let mut this = slf.try_borrow_mut()?;
        if this.set.page_size() != page_size {
            let mut replacement = BitSetInner::new(page_size)?;
            replacement.extend(this.set.iter());
            this.set = replacement;
            this.record_change(true);
        }
        if let Some(keys) = keys {
            this.insert_keys(keys);
        }
        Ok(())
    }

    /// Number of 64-bit words per page.
    #[getter]
    fn page_size(&self) -> usize {
        self.set.page_size()
    }

    fn __len__(&self) -> usize {
        self.set.len()
    }

    fn __contains__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        let Some(key) = lookup_key(key) else {
            return Ok(false);
        };
        Ok(slf.try_borrow()?.set.contains(key))
    }

    fn __iter__(slf: &Bound<'_, Self>) -> PyResult<BitSetIterator> {
        BitSetIterator::new(slf, None, false)
    }

    fn __reversed__(slf: &Bound<'_, Self>) -> PyResult<BitSetIterator> {
        BitSetIterator::new(slf, None, true)
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let (keys, page_size) = {
            let this = slf.try_borrow()?;
            (
                crate::truncated(this.set.iter().map(|key| key.to_string())),
                this.set.page_size(),
            )
        };
        let size = if page_size == DEFAULT_BIT_PAGE_SIZE {
            String::new()
        } else {
            format!(", page_size={page_size}")
        };
        Ok(format!("{}([{keys}]{size})", slf.get_type().name()?))
    }

    fn __richcmp__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        op: CompareOp,
    ) -> PyResult<Py<PyAny>> {
        let Ok(other) = other.cast::<Self>() else {
            return Ok(py.NotImplemented());
        };
        let other = other.try_borrow()?;
        let a = self.set.len();
        let b = other.set.len();
        let result = match op {
            CompareOp::Eq => a == b && self.set.is_subset(&other.set),
            CompareOp::Ne => a != b || !self.set.is_subset(&other.set),
            CompareOp::Lt => a < b && self.set.is_subset(&other.set),
            CompareOp::Le => self.set.is_subset(&other.set),
            CompareOp::Gt => a > b && other.set.is_subset(&self.set),
            CompareOp::Ge => other.set.is_subset(&self.set),
        };
        result.into_py_any(py)
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let (keys, page_size) = {
            let this = slf.try_borrow()?;
            (this.to_numpy(py), this.set.page_size())
        };
        crate::reduce_with_page_size(
            slf.as_any(),
            &py.get_type::<Self>(),
            vec![keys.into_any()],
            None,
            Some(page_size.into_bound_py_any(py)?),
        )
    }

    /// Return a base BitSet with the same keys and page size.
    fn copy(&self) -> Self {
        Self::with_storage(self.set.clone())
    }

    fn add(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<()> {
        let key = insert_key(key)?;
        let mut this = slf.try_borrow_mut()?;
        let added = this.set.insert(key);
        this.record_change(added);
        Ok(())
    }

    /// Flip membership of the key and return whether it is now present.
    fn toggle(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        let key = insert_key(key)?;
        let mut this = slf.try_borrow_mut()?;
        let present = this.set.toggle(key);
        this.record_change(true);
        Ok(present)
    }

    fn discard(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Some(key) = lookup_key(key) {
            let mut this = slf.try_borrow_mut()?;
            let removed = this.set.remove(key);
            this.record_change(removed);
        }
        Ok(())
    }

    fn remove(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Some(index) = lookup_key(key) {
            let mut this = slf.try_borrow_mut()?;
            if this.set.remove(index) {
                this.record_change(true);
                return Ok(());
            }
        }
        Err(PyKeyError::new_err(key.clone().unbind()))
    }

    /// Remove and return the smallest key. Raise KeyError if empty.
    fn pop(&mut self) -> PyResult<Key> {
        self.pop_first()
    }

    /// Remove and return the smallest key. Raise KeyError if empty.
    fn pop_first(&mut self) -> PyResult<Key> {
        let key = self
            .set
            .pop_first()
            .ok_or_else(|| PyKeyError::new_err("pop from an empty BitSet"))?;
        self.record_change(true);
        Ok(key)
    }

    /// Remove and return the largest key. Raise KeyError if empty.
    fn pop_last(&mut self) -> PyResult<Key> {
        let key = self
            .set
            .pop_last()
            .ok_or_else(|| PyKeyError::new_err("pop from an empty BitSet"))?;
        self.record_change(true);
        Ok(key)
    }

    /// Remove every key, retaining pages for reuse.
    fn clear(&mut self) {
        if !self.set.is_empty() {
            self.set.clear();
            self.record_change(true);
        }
    }

    /// Return the smallest key. Raise KeyError if empty.
    fn first(&self) -> PyResult<Key> {
        self.set
            .first()
            .ok_or_else(|| PyKeyError::new_err("first(): BitSet is empty"))
    }

    /// Return the largest key. Raise KeyError if empty.
    fn last(&self) -> PyResult<Key> {
        self.set
            .last()
            .ok_or_else(|| PyKeyError::new_err("last(): BitSet is empty"))
    }

    /// Return the first key at or after index, or None.
    fn next_occupant(slf: &Bound<'_, Self>, index: &Bound<'_, PyAny>) -> PyResult<Option<Key>> {
        let index = insert_key(index)?;
        Ok(slf.try_borrow()?.set.next_occupant(index))
    }

    /// Return the last key at or before index, or None.
    fn prev_occupant(slf: &Bound<'_, Self>, index: &Bound<'_, PyAny>) -> PyResult<Option<Key>> {
        let index = insert_key(index)?;
        Ok(slf.try_borrow()?.set.prev_occupant(index))
    }

    /// Add keys in [start, stop) and return how many were newly added.
    fn add_range(
        slf: &Bound<'_, Self>,
        start: &Bound<'_, PyAny>,
        stop: &Bound<'_, PyAny>,
    ) -> PyResult<usize> {
        let bounds = bounds(start, stop)?;
        let Some(span) = key_span(&bounds) else {
            return Ok(0);
        };
        usize::try_from(bounds.end - bounds.start)
            .map_err(|err| PyOverflowError::new_err(err.to_string()))?;
        let mut this = slf.try_borrow_mut()?;
        let added = this.set.insert_range(span);
        this.record_change(added != 0);
        Ok(added)
    }

    /// Discard keys in [start, stop) and return how many were removed.
    fn discard_range(
        slf: &Bound<'_, Self>,
        start: &Bound<'_, PyAny>,
        stop: &Bound<'_, PyAny>,
    ) -> PyResult<usize> {
        let Some(span) = key_span(&bounds(start, stop)?) else {
            return Ok(0);
        };
        let mut this = slf.try_borrow_mut()?;
        let removed = this.set.remove_range(span);
        this.record_change(removed != 0);
        Ok(removed)
    }

    /// Count the keys in [start, stop).
    fn count_range(
        slf: &Bound<'_, Self>,
        start: &Bound<'_, PyAny>,
        stop: &Bound<'_, PyAny>,
    ) -> PyResult<usize> {
        let Some(span) = key_span(&bounds(start, stop)?) else {
            return Ok(0);
        };
        Ok(slf.try_borrow()?.set.count_range(span))
    }

    /// Iterate over keys in [start, stop) in ascending order.
    fn range(
        slf: &Bound<'_, Self>,
        start: &Bound<'_, PyAny>,
        stop: &Bound<'_, PyAny>,
    ) -> PyResult<BitSetIterator> {
        BitSetIterator::new(slf, Some(bounds(start, stop)?), false)
    }

    /// Add a validated batch of integer keys, returning the number added.
    fn add_many(slf: &Bound<'_, Self>, keys: &Bound<'_, PyAny>) -> PyResult<usize> {
        let keys = KeyBatch::<Key>::new(keys)?.to_insert_keys()?;
        Ok(slf.try_borrow_mut()?.insert_keys(keys))
    }

    /// Discard a batch of keys, returning the number removed.
    /// Integers outside the key domain are ignored.
    fn discard_many(slf: &Bound<'_, Self>, keys: &Bound<'_, PyAny>) -> PyResult<usize> {
        let batch = KeyBatch::<Key>::lookup(keys)?;
        let mut this = slf.try_borrow_mut()?;
        let before = this.set.len();
        batch.for_each(|key| {
            if let Some(key) = key {
                this.set.remove(key);
            }
        });
        let removed = before - this.set.len();
        this.record_change(removed != 0);
        Ok(removed)
    }

    /// Return a NumPy bool array indicating membership for each supplied key.
    fn contains_many<'py>(
        slf: &Bound<'py, Self>,
        keys: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let batch = KeyBatch::<Key>::lookup(keys)?;
        let this = slf.try_borrow()?;
        let mut out = Vec::with_capacity(batch.len());
        batch.for_each(|key| out.push(key.is_some_and(|key| this.set.contains(key))));
        Ok(PyArray1::from_vec(slf.py(), out))
    }

    /// Return a new uint32 NumPy array of the keys in ascending order.
    fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Key>> {
        PyArray1::from_vec(py, self.set.iter().collect())
    }

    /// Build a set from a one-dimensional boolean mask. True at position i
    /// includes key offset + i. The page size counts 64-bit words.
    #[classmethod]
    #[pyo3(signature = (mask, offset = 0, *, page_size = DEFAULT_BIT_PAGE_SIZE))]
    fn from_mask<'py>(
        cls: &Bound<'py, PyType>,
        mask: &Bound<'py, PyAny>,
        #[pyo3(from_py_with = insert_key)] offset: Key,
        #[pyo3(from_py_with = crate::extract_page_size)] page_size: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let array = as_ndarray(mask)?;
        if array.ndim() != 1 {
            return Err(PyValueError::new_err(format!(
                "mask must be one-dimensional, got {} dimensions",
                array.ndim()
            )));
        }
        if array.len() as u64 > KEY_LIMIT - u64::from(offset) {
            return Err(PyOverflowError::new_err(
                "mask extends outside the uint32 key domain",
            ));
        }
        let mut set = BitSetInner::new(page_size)?;
        if !array.is_empty() || mask.cast::<PyUntypedArray>().is_ok() {
            let mask = <bool as TypedValue>::convert(&array)?;
            set.extend(
                mask.into_iter()
                    .enumerate()
                    .filter(|&(_, present)| present)
                    .map(|(i, _)| offset + i as Key),
            );
        }
        let obj = cls.call_method1("__new__", (cls,))?;
        {
            let mut this = obj.cast::<Self>()?.try_borrow_mut()?;
            this.set = set;
            this.record_change(true);
        }
        Ok(obj)
    }

    /// Return a NumPy bool array for the half-open interval [start, stop).
    fn to_mask<'py>(
        slf: &Bound<'py, Self>,
        start: &Bound<'_, PyAny>,
        stop: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let bounds = bounds(start, stop)?;
        let mut mask = Vec::new();
        if let Some(span) = key_span(&bounds) {
            let len = usize::try_from(bounds.end - bounds.start)
                .map_err(|err| PyOverflowError::new_err(err.to_string()))?;
            mask.try_reserve_exact(len).map_err(crate::reserve_error)?;
            mask.resize(len, false);
            for key in slf.try_borrow()?.set.range(span) {
                mask[(u64::from(key) - bounds.start) as usize] = true;
            }
        }
        Ok(PyArray1::from_vec(slf.py(), mask))
    }

    #[pyo3(signature = (*others))]
    fn union(slf: &Bound<'_, Self>, others: &Bound<'_, PyTuple>) -> PyResult<Self> {
        Self::algebra(slf, &others.iter().collect::<Vec<_>>(), BitSetOp::Union)
    }

    #[pyo3(signature = (*others))]
    fn intersection(slf: &Bound<'_, Self>, others: &Bound<'_, PyTuple>) -> PyResult<Self> {
        Self::algebra(
            slf,
            &others.iter().collect::<Vec<_>>(),
            BitSetOp::Intersection,
        )
    }

    #[pyo3(signature = (*others))]
    fn difference(slf: &Bound<'_, Self>, others: &Bound<'_, PyTuple>) -> PyResult<Self> {
        Self::algebra(
            slf,
            &others.iter().collect::<Vec<_>>(),
            BitSetOp::Difference,
        )
    }

    fn symmetric_difference(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::algebra(
            slf,
            std::slice::from_ref(other),
            BitSetOp::SymmetricDifference,
        )
    }

    #[pyo3(signature = (*others))]
    fn update(slf: &Bound<'_, Self>, others: &Bound<'_, PyTuple>) -> PyResult<()> {
        Self::streaming_update(slf, others, BitSetOp::Union)
    }

    #[pyo3(signature = (*others))]
    fn intersection_update(slf: &Bound<'_, Self>, others: &Bound<'_, PyTuple>) -> PyResult<()> {
        Self::staged_update(
            slf,
            &others.iter().collect::<Vec<_>>(),
            BitSetOp::Intersection,
        )
    }

    #[pyo3(signature = (*others))]
    fn difference_update(slf: &Bound<'_, Self>, others: &Bound<'_, PyTuple>) -> PyResult<()> {
        Self::streaming_update(slf, others, BitSetOp::Difference)
    }

    fn symmetric_difference_update(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        Self::staged_update(
            slf,
            std::slice::from_ref(other),
            BitSetOp::SymmetricDifference,
        )
    }

    fn issubset(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        if let Ok(other) = other.cast::<Self>() {
            return Ok(slf.try_borrow()?.set.is_subset(&other.try_borrow()?.set));
        }
        let operand = PySet::empty(slf.py())?;
        operand.call_method1("update", (other,))?;
        let keys: Vec<_> = slf.try_borrow()?.set.iter().collect();
        for key in keys {
            if !operand.contains(key)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn issuperset(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        if let Ok(other) = other.cast::<Self>() {
            return Ok(other.try_borrow()?.set.is_subset(&slf.try_borrow()?.set));
        }
        let keys: Vec<_> = slf.try_borrow()?.set.iter().collect();
        PySet::new(slf.py(), keys)?
            .call_method1("issuperset", (other,))?
            .extract()
    }

    fn isdisjoint(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        if let Ok(other) = other.cast::<Self>() {
            return Ok(slf.try_borrow()?.set.is_disjoint(&other.try_borrow()?.set));
        }
        let keys: Vec<_> = slf.try_borrow()?.set.iter().collect();
        PySet::new(slf.py(), keys)?
            .call_method1("isdisjoint", (other,))?
            .extract()
    }

    fn __or__(slf: &Bound<'_, Self>, other: &Bound<'_, Self>) -> PyResult<Self> {
        Self::combine_native(slf, other, BitSetOp::Union)
    }

    fn __and__(slf: &Bound<'_, Self>, other: &Bound<'_, Self>) -> PyResult<Self> {
        Self::combine_native(slf, other, BitSetOp::Intersection)
    }

    fn __sub__(slf: &Bound<'_, Self>, other: &Bound<'_, Self>) -> PyResult<Self> {
        Self::combine_native(slf, other, BitSetOp::Difference)
    }

    fn __xor__(slf: &Bound<'_, Self>, other: &Bound<'_, Self>) -> PyResult<Self> {
        Self::combine_native(slf, other, BitSetOp::SymmetricDifference)
    }

    fn __ior__(slf: &Bound<'_, Self>, other: &Bound<'_, Self>) -> PyResult<()> {
        Self::apply_native(slf, other, BitSetOp::Union)
    }

    fn __iand__(slf: &Bound<'_, Self>, other: &Bound<'_, Self>) -> PyResult<()> {
        Self::apply_native(slf, other, BitSetOp::Intersection)
    }

    fn __isub__(slf: &Bound<'_, Self>, other: &Bound<'_, Self>) -> PyResult<()> {
        Self::apply_native(slf, other, BitSetOp::Difference)
    }

    fn __ixor__(slf: &Bound<'_, Self>, other: &Bound<'_, Self>) -> PyResult<()> {
        Self::apply_native(slf, other, BitSetOp::SymmetricDifference)
    }
}

/// An owner plus integer bounds, so no borrowed Rust iterator survives
/// across Python calls. Once exhausted, the iterator stays exhausted.
#[pyclass(module = "paged_collections._core")]
pub(crate) struct BitSetIterator {
    owner: Py<BitSet>,
    bounds: Range<u64>,
    reverse: bool,
    remaining: Option<usize>,
    mods: u64,
    done: bool,
}

impl BitSetIterator {
    fn new(owner: &Bound<'_, BitSet>, bounds: Option<Range<u64>>, reverse: bool) -> PyResult<Self> {
        let this = owner.try_borrow()?;
        Ok(Self {
            owner: owner.clone().unbind(),
            remaining: bounds.is_none().then(|| this.set.len()),
            bounds: bounds.unwrap_or(0..KEY_LIMIT),
            reverse,
            mods: this.mods,
            done: false,
        })
    }
}

#[pymethods]
impl BitSetIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Key>> {
        if self.done {
            return Ok(None);
        }
        let owner = self.owner.try_borrow(py)?;
        if owner.mods != self.mods {
            return Err(PyRuntimeError::new_err("BitSet changed during iteration"));
        }
        let key = if self.remaining == Some(0) {
            None
        } else {
            key_span(&self.bounds).and_then(|span| {
                let mut keys = owner.set.range(span);
                if self.reverse {
                    keys.next_back()
                } else {
                    keys.next()
                }
            })
        };
        match key {
            Some(key) => {
                if self.reverse {
                    self.bounds.end = u64::from(key);
                } else {
                    self.bounds.start = u64::from(key) + 1;
                }
                self.remaining = self.remaining.map(|remaining| remaining - 1);
            }
            None => {
                self.done = true;
                self.remaining = Some(0);
            }
        }
        Ok(key)
    }

    fn __length_hint__(&self) -> usize {
        self.remaining.unwrap_or(0)
    }

    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        visit.call(&self.owner)
    }
}
