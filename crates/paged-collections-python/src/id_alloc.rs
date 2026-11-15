//! Generational ids as Python integers, packed as `Id::to_bits()`.

use numpy::{PyArray1, PyArrayDescrMethods, PyUntypedArray, PyUntypedArrayMethods};
use paged_collections::id_alloc::{Id, IdAllocator as Allocator, MAX_INDICES};
use pyo3::exceptions::{PyOverflowError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyInt, PyTuple};
use std::num::NonZeroU32;

use crate::convert::{KeyBatch, TypedValue, as_ndarray};

/// Normalize an int subclass before extracting or comparing it, just as
/// `direct_map::query_index` does. Objects with only `__index__` are not ints.
fn python_int<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let obj = obj.cast::<PyInt>()?;
    obj.py().import("operator")?.call_method1("index", (obj,))
}

fn query_id(obj: &Bound<'_, PyAny>) -> PyResult<Option<Id>> {
    Ok(python_int(obj)?
        .extract::<u64>()
        .ok()
        .and_then(Id::from_bits))
}

fn valid_id(obj: &Bound<'_, PyAny>) -> PyResult<Id> {
    query_id(obj)?.ok_or_else(|| PyValueError::new_err("invalid id bits"))
}

/// Validate the whole batch before a caller mutates anything. Integer
/// arrays use the shared NumPy conversion (including endian/stride handling).
/// Other iterables are read directly so arbitrary Python ints never pass
/// through NumPy's lossy float inference, and generators work too.
fn id_batch(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Option<Id>>> {
    if obj.cast::<PyUntypedArray>().is_ok() {
        let array = as_ndarray(obj)?;
        if array.ndim() != 1 {
            return Err(PyValueError::new_err(format!(
                "ids must be one-dimensional, got {} dimensions",
                array.ndim()
            )));
        }
        if array.is_empty() {
            return Ok(Vec::new());
        }
        return match array.dtype().kind() {
            b'i' => Ok(i64::convert(&array)?
                .into_iter()
                .map(|bits| u64::try_from(bits).ok().and_then(Id::from_bits))
                .collect()),
            b'u' => Ok(u64::convert(&array)?
                .into_iter()
                .map(Id::from_bits)
                .collect()),
            _ => Err(PyTypeError::new_err(format!(
                "ids must be integers, got an array of dtype {}",
                array.dtype()
            ))),
        };
    }
    obj.try_iter()?.map(|obj| query_id(&obj?)).collect()
}

fn exhausted() -> PyErr {
    PyOverflowError::new_err("IdAllocator is exhausted")
}

/// Allocates generational ids, packed as ``generation << 32 | index``.
/// Freed indices are reused last freed first, with a new generation.
/// Iteration and ``ids()`` visit live ids in index order; arrays are copies.
///
/// The optional ``_state`` constructor argument is internal pickle state:
/// a pair of uint32 arrays or lists holding versions and the free list.
#[pyclass(module = "paged_collections._core", subclass, weakref)]
pub(crate) struct IdAllocator {
    allocator: Allocator,
    mods: u64,
}

impl IdAllocator {
    fn changed(&mut self) {
        self.mods = self.mods.wrapping_add(1);
    }

    fn free_batch(&mut self, ids: Vec<Option<Id>>) -> usize {
        let freed = ids
            .into_iter()
            .flatten()
            .filter(|&id| self.allocator.free(id))
            .count();
        if freed != 0 {
            self.changed();
        }
        freed
    }

    /// Use the same reconstruction as Python's copy protocol, including
    /// subclass dict/slots state and the deepcopy memo for cycles.
    fn reconstruct_copy<'py>(
        slf: &Bound<'py, Self>,
        memo: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let reduced = Self::__reduce__(slf)?;
        let args: Vec<_> = [slf.as_any().clone(), memo]
            .into_iter()
            .chain(reduced.iter())
            .collect();
        let args = PyTuple::new(slf.py(), args)?;
        slf.py()
            .import("copy")?
            .getattr("_reconstruct")?
            .call1(args)
    }
}

#[pymethods]
impl IdAllocator {
    // Accept subclass constructor arguments without calling its __init__.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs), text_signature = "()")]
    fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        Self {
            allocator: Allocator::new(),
            mods: 0,
        }
    }

    #[pyo3(signature = (_state = None))]
    fn __init__(&mut self, _state: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let allocator = match _state {
            None => Allocator::new(),
            Some(state) => {
                let parts = state.cast::<PyTuple>().map_err(|_| {
                    PyValueError::new_err("_state must be a (versions, free) tuple")
                })?;
                if parts.len() != 2 {
                    return Err(PyValueError::new_err(
                        "_state must be a (versions, free) tuple",
                    ));
                }
                let raw = |pos| KeyBatch::new(&parts.get_item(pos)?)?.to_insert_keys();
                let versions = raw(0).map_err(|err| PyValueError::new_err(err.to_string()))?;
                let free = raw(1).map_err(|err| PyValueError::new_err(err.to_string()))?;
                Allocator::try_from_raw_parts(versions, free)
                    .map_err(|err| PyValueError::new_err(err.to_string()))?
            }
        };
        self.allocator = allocator;
        self.changed();
        Ok(())
    }

    fn __len__(&self) -> usize {
        self.allocator.len()
    }

    fn __contains__(&self, id: &Bound<'_, PyAny>) -> bool {
        query_id(id)
            .ok()
            .flatten()
            .is_some_and(|id| self.allocator.is_alive(id))
    }

    fn __iter__(slf: &Bound<'_, Self>) -> PyResult<IdAllocatorIterator> {
        IdAllocatorIterator::new(slf)
    }

    fn __eq__(&self, other: PyRef<'_, Self>) -> bool {
        self.allocator == other.allocator
    }

    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let this = slf.try_borrow()?;
        Ok(format!(
            "{}(len={}, index_bound={})",
            slf.get_type().name()?,
            this.allocator.len(),
            this.allocator.index_bound()
        ))
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let state = {
            let this = slf.try_borrow()?;
            let (versions, free) = this.allocator.raw_parts();
            PyTuple::new(
                py,
                [
                    PyArray1::from_slice(py, versions),
                    PyArray1::from_slice(py, free),
                ],
            )?
        };
        crate::reduce(
            slf.as_any(),
            &py.get_type::<Self>(),
            vec![state.into_any()],
            None,
        )
    }

    /// Return an independent copy of the full allocation history.
    fn copy(&self) -> Self {
        Self {
            allocator: self.allocator.clone(),
            mods: 0,
        }
    }

    fn __copy__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        Self::reconstruct_copy(slf, slf.py().None().into_bound(slf.py()))
    }

    fn __deepcopy__<'py>(
        slf: &Bound<'py, Self>,
        memo: &Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Self::reconstruct_copy(slf, memo.clone().into_any())
    }

    /// Issue an id, reusing the most recently freed index first.
    fn alloc(&mut self) -> PyResult<u64> {
        if self.allocator.free_len() == 0 && self.allocator.index_bound() == MAX_INDICES as usize {
            return Err(exhausted());
        }
        self.allocator
            .try_reserve(1)
            .map_err(crate::reserve_error)?;
        let id = self.allocator.try_alloc().ok_or_else(exhausted)?;
        self.changed();
        Ok(id.to_bits())
    }

    /// Issue ``n`` ids as a uint64 array, or change nothing on failure.
    fn alloc_many<'py>(
        &mut self,
        py: Python<'py>,
        n: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<u64>>> {
        let n = python_int(n)?;
        if n.lt(0)? {
            return Err(PyValueError::new_err("n must be nonnegative"));
        }
        let available = self.allocator.free_len() as u64
            + (u64::from(MAX_INDICES) - self.allocator.index_bound() as u64);
        let n = n.extract::<u64>().map_err(|_| exhausted())?;
        if n > available {
            return Err(exhausted());
        }
        let n = n as usize; // At most MAX_INDICES, even on 32-bit targets.
        let mut ids = Vec::new();
        ids.try_reserve(n).map_err(crate::reserve_error)?;
        self.allocator
            .try_reserve(n)
            .map_err(crate::reserve_error)?;
        for _ in 0..n {
            ids.push(
                self.allocator
                    .try_alloc()
                    .expect("batch capacity checked")
                    .to_bits(),
            );
        }
        if n != 0 {
            self.changed();
        }
        Ok(PyArray1::from_vec(py, ids))
    }

    /// Free a live id; return whether it was freed.
    fn free(&mut self, id: &Bound<'_, PyAny>) -> PyResult<bool> {
        let freed = query_id(id)?.is_some_and(|id| self.allocator.free(id));
        if freed {
            self.changed();
        }
        Ok(freed)
    }

    /// Free a fully validated batch; count each live id only once.
    /// The batch is converted before the allocator is borrowed, so it may
    /// be the allocator itself or a generator that reads it.
    fn free_many(slf: &Bound<'_, Self>, ids: &Bound<'_, PyAny>) -> PyResult<usize> {
        let batch = id_batch(ids)?;
        Ok(slf.try_borrow_mut()?.free_batch(batch))
    }

    fn is_alive(&self, id: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(query_id(id)?.is_some_and(|id| self.allocator.is_alive(id)))
    }

    fn is_alive_many<'py>(
        slf: &Bound<'py, Self>,
        ids: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let batch = id_batch(ids)?;
        let this = slf.try_borrow()?;
        let alive = batch
            .into_iter()
            .map(|id| id.is_some_and(|id| this.allocator.is_alive(id)))
            .collect();
        Ok(PyArray1::from_vec(slf.py(), alive))
    }

    /// The live id at an index, or None outside the u32 domain.
    fn occupant(&self, index: &Bound<'_, PyAny>) -> PyResult<Option<u64>> {
        Ok(python_int(index)?
            .extract::<u32>()
            .ok()
            .and_then(|index| self.allocator.occupant(index))
            .map(Id::to_bits))
    }

    /// Return a detached uint64 array of live ids, in index order.
    fn ids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        PyArray1::from_vec(py, self.allocator.iter().map(Id::to_bits).collect())
    }

    /// Keep ids whose predicate is truthy, committing only after all calls
    /// succeed and only if the allocator did not change during those calls.
    fn retain(slf: &Bound<'_, Self>, predicate: &Bound<'_, PyAny>) -> PyResult<()> {
        let (ids, mods) = {
            let this = slf.try_borrow()?;
            (this.allocator.iter().collect::<Vec<_>>(), this.mods)
        };
        let mut discard = Vec::new();
        for id in ids {
            if !predicate.call1((id.to_bits(),))?.is_truthy()? {
                discard.push(Some(id));
            }
        }
        let mut this = slf.try_borrow_mut()?;
        if this.mods != mods {
            return Err(PyRuntimeError::new_err("IdAllocator changed during retain"));
        }
        this.free_batch(discard);
        Ok(())
    }

    /// Free all live ids without forgetting their generations.
    fn clear(&mut self) {
        self.allocator.clear();
        self.changed();
    }

    fn shrink_to_fit(&mut self) {
        self.allocator.shrink_to_fit();
    }

    fn reserve(&mut self, additional: usize) -> PyResult<()> {
        self.allocator
            .try_reserve(additional)
            .map_err(crate::reserve_error)
    }

    #[getter]
    fn index_bound(&self) -> usize {
        self.allocator.index_bound()
    }

    #[getter]
    fn free_len(&self) -> usize {
        self.allocator.free_len()
    }

    #[getter]
    fn retired_len(&self) -> usize {
        self.allocator.retired_len()
    }
}

/// Cursor over live ids in index order, detecting structural changes.
#[pyclass(module = "paged_collections._core")]
pub(crate) struct IdAllocatorIterator {
    owner: Py<IdAllocator>,
    cursor: usize,
    remaining: usize,
    mods: u64,
    done: bool,
}

impl IdAllocatorIterator {
    fn new(owner: &Bound<'_, IdAllocator>) -> PyResult<Self> {
        let this = owner.try_borrow()?;
        Ok(Self {
            owner: owner.clone().unbind(),
            cursor: 0,
            remaining: this.allocator.len(),
            mods: this.mods,
            done: false,
        })
    }
}

#[pymethods]
impl IdAllocatorIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<u64>> {
        if self.done {
            return Ok(None);
        }
        let owner = self.owner.try_borrow(py)?;
        if owner.mods != self.mods {
            return Err(PyRuntimeError::new_err(
                "IdAllocator changed during iteration",
            ));
        }
        while self.cursor < owner.allocator.index_bound() && self.remaining != 0 {
            let index = self.cursor as u32;
            self.cursor += 1;
            if let Some(id) = owner.allocator.occupant(index) {
                self.remaining -= 1;
                return Ok(Some(id.to_bits()));
            }
        }
        self.remaining = 0;
        self.done = true;
        Ok(None)
    }

    fn __length_hint__(&self) -> usize {
        self.remaining
    }

    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        visit.call(&self.owner)
    }
}

/// Extract an id's low 32-bit slot index. Invalid bits raise ValueError.
#[pyfunction]
pub(crate) fn id_index(id: &Bound<'_, PyAny>) -> PyResult<u32> {
    Ok(valid_id(id)?.index())
}

/// Extract an id's nonzero high 32-bit generation.
#[pyfunction]
pub(crate) fn id_generation(id: &Bound<'_, PyAny>) -> PyResult<u32> {
    Ok(valid_id(id)?.generation().get())
}

/// Pack u32 parts into an id. Generations 1..=2**32-1 are valid, though
/// the allocator never issues generations above MAX_GENERATION (2**31).
#[pyfunction]
pub(crate) fn make_id(index: &Bound<'_, PyAny>, generation: &Bound<'_, PyAny>) -> PyResult<u64> {
    let index = python_int(index)?;
    let generation = python_int(generation)?;
    let index = index
        .extract::<u32>()
        .map_err(|_| PyValueError::new_err("index is out of range for uint32"))?;
    let generation = generation
        .extract::<u32>()
        .ok()
        .and_then(NonZeroU32::new)
        .ok_or_else(|| PyValueError::new_err("generation must be in 1..2**32"))?;
    Ok(Id::new(index, generation).to_bits())
}

/// Extract slot indices as a uint32 array; invalid id bits raise ValueError.
#[pyfunction]
pub(crate) fn id_indices<'py>(
    py: Python<'py>,
    ids: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let indices = id_batch(ids)?
        .into_iter()
        .map(|id| {
            id.map(Id::index)
                .ok_or_else(|| PyValueError::new_err("invalid id bits"))
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyArray1::from_vec(py, indices))
}
