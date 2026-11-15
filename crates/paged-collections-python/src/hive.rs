//! Object-valued hive with generational integer handles.

use crate::storage::HiveInner;
use numpy::PyArray1;
use paged_collections::hive::{RawPage, RawSlot};
use paged_collections::id_alloc::{Id, MAX_INDICES};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyKeyError, PyOverflowError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::{PyTraverseError, PyVisit};

use crate::convert::{KeyBatch, PyKey};
use crate::typed_map::optional_default;

fn exhausted() -> PyErr {
    PyOverflowError::new_err("Hive is exhausted")
}

/// Objects stored in pages of reusable slots, addressed by generational
/// integer ids. Removed ids stay dead throughout this hive's allocation history.
/// Iteration visits ids in slot order; array and list results are copies.
/// Ids work with ``id_index``, ``id_generation`` and ``IdMap``.
#[pyclass(module = "paged_collections._core", subclass, weakref)]
pub(crate) struct Hive {
    hive: HiveInner<Py<PyAny>>,
    mods: u64,
}

impl Hive {
    fn changed(&mut self) {
        self.mods = self.mods.wrapping_add(1);
    }

    fn snapshot(&self, py: Python<'_>) -> Vec<(u64, Py<PyAny>)> {
        self.hive
            .iter()
            .map(|(id, value)| (id.to_bits(), value.clone_ref(py)))
            .collect()
    }

    // Fast path: unused indices alone cover the batch. Near the index limit,
    // account for retired slots too; capacity includes those unusable slots.
    fn reserve_insert(&mut self, n: usize) -> PyResult<()> {
        if n != 0 && self.hive.is_exhausted() {
            return Err(exhausted());
        }
        if n > MAX_INDICES as usize - self.hive.index_bound() {
            let retired = self
                .hive
                .raw_pages()
                .flat_map(|page| page.slots)
                .filter(|slot| matches!(slot, RawSlot::Vacant { version: 0 }))
                .count();
            if n > MAX_INDICES as usize - self.hive.len() - retired {
                return Err(exhausted());
            }
        }
        self.hive.try_reserve(n).map_err(crate::reserve_error)
    }

    fn remove(&mut self, id: Option<Id>) -> Option<Py<PyAny>> {
        let value = id.and_then(|id| self.hive.remove(id));
        if value.is_some() {
            self.changed();
        }
        value
    }

    fn remove_ids(&mut self, ids: impl IntoIterator<Item = Id>) -> Vec<Py<PyAny>> {
        let removed: Vec<_> = ids
            .into_iter()
            .filter_map(|id| self.hive.remove(id))
            .collect();
        if !removed.is_empty() {
            self.changed();
        }
        removed
    }

    fn from_state(state: &Bound<'_, PyAny>, page_size: usize) -> PyResult<HiveInner<Py<PyAny>>> {
        let parse = || -> PyResult<HiveInner<Py<PyAny>>> {
            let parts = state.cast::<PyTuple>()?;
            if parts.len() != 3 {
                return Err(PyValueError::new_err(
                    "_state must be a (floors, page_lens, versions) tuple",
                ));
            }
            let raw = |pos| KeyBatch::<u32>::new(&parts.get_item(pos)?)?.to_insert_keys();
            let floors = raw(0)?;
            let lens = raw(1)?;
            let versions = raw(2)?;
            let total = lens.iter().try_fold(0usize, |sum, &n| {
                sum.checked_add(n as usize)
                    .ok_or_else(|| PyValueError::new_err("page lengths overflow"))
            })?;
            if floors.len() != lens.len() || total != versions.len() {
                return Err(PyValueError::new_err("inconsistent Hive state lengths"));
            }
            let mut versions = versions.into_iter();
            let pages = floors.into_iter().zip(lens).map(|(floor, len)| RawPage {
                floor,
                slots: versions
                    .by_ref()
                    .take(len as usize)
                    .map(|version| {
                        if version % 2 == 1 {
                            RawSlot::Occupied {
                                version,
                                value: state.py().None(),
                            }
                        } else {
                            RawSlot::Vacant { version }
                        }
                    })
                    .collect(),
            });
            HiveInner::try_from_raw_pages(page_size, pages)
                .map_err(|err| PyValueError::new_err(err.to_string()))
        };
        parse().map_err(|err| PyValueError::new_err(err.to_string()))
    }

    /// Retile generation history when explicitly reinitializing at a new size.
    /// Missing slots had no live id; materialize their floor so merging pages
    /// cannot resurrect any stale id. At floor zero, skipping generation one
    /// distinguishes a reusable vacancy from a retired (version zero) slot.
    fn repage(&self, py: Python<'_>, page_size: usize) -> PyResult<HiveInner<Py<PyAny>>> {
        let mut pages = Vec::new();
        let mut slots = Vec::new();
        let old_size = self.hive.page_size();
        for (p, page) in self.hive.raw_pages().enumerate() {
            let limit = old_size.min(MAX_INDICES as usize - p * old_size);
            let mut source = page.slots.into_iter();
            for _ in 0..limit {
                slots.push(match source.next() {
                    Some(RawSlot::Occupied { version, value }) => RawSlot::Occupied {
                        version,
                        value: value.clone_ref(py),
                    },
                    Some(RawSlot::Vacant { version }) => RawSlot::Vacant { version },
                    None => RawSlot::Vacant {
                        version: page.floor.max(2),
                    },
                });
                if slots.len() == page_size {
                    pages.push(RawPage {
                        floor: 0,
                        slots: std::mem::take(&mut slots),
                    });
                }
            }
        }
        if !slots.is_empty() {
            pages.push(RawPage { floor: 0, slots });
        }
        HiveInner::try_from_raw_pages(page_size, pages)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

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
impl Hive {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs), text_signature = "(values=None, *, page_size=DEFAULT_PAGE_SIZE)")]
    fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        Self {
            hive: HiveInner::default(),
            mods: 0,
        }
    }

    #[pyo3(signature = (values=None, _state=None, *, page_size=crate::DEFAULT_PAGE_SIZE))]
    fn __init__(
        slf: &Bound<'_, Self>,
        values: Option<&Bound<'_, PyAny>>,
        _state: Option<&Bound<'_, PyAny>>,
        #[pyo3(from_py_with = crate::extract_page_size)] page_size: usize,
    ) -> PyResult<()> {
        // Parse and materialize before borrowing, including on reinitialization.
        let hive = _state
            .map(|state| Self::from_state(state, page_size))
            .transpose()?;
        let values: Vec<_> = match values {
            Some(values) => values
                .try_iter()?
                .map(|v| v.map(Bound::unbind))
                .collect::<PyResult<_>>()?,
            None => Vec::new(),
        };
        let hive = match hive {
            Some(hive) => Some(hive),
            None => {
                let this = slf.try_borrow()?;
                (this.hive.page_size() != page_size)
                    .then(|| this.repage(slf.py(), page_size))
                    .transpose()?
            }
        };
        if let Some(hive) = hive {
            let old = {
                let mut this = slf.try_borrow_mut()?;
                let old = std::mem::replace(&mut this.hive, hive);
                this.changed();
                old
            };
            drop(old);
        }
        // Without _state, insert into the existing history, like map __init__.
        Self::insert_values(slf, values).map(drop)
    }

    fn __len__(&self) -> usize {
        self.hive.len()
    }

    fn __contains__(slf: &Bound<'_, Self>, id: &Bound<'_, PyAny>) -> PyResult<bool> {
        let id = Id::lookup(id);
        let this = slf.try_borrow()?;
        Ok(id.is_some_and(|id| this.hive.contains(id)))
    }

    fn __getitem__(slf: &Bound<'_, Self>, id: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let key = Id::lookup(id);
        slf.try_borrow()?
            .hive
            .get(key.ok_or_else(|| PyKeyError::new_err(id.clone().unbind()))?)
            .map(|v| v.clone_ref(slf.py()))
            .ok_or_else(|| PyKeyError::new_err(id.clone().unbind()))
    }

    fn __setitem__(slf: &Bound<'_, Self>, id: &Bound<'_, PyAny>, value: Py<PyAny>) -> PyResult<()> {
        let key = Id::lookup(id).ok_or_else(|| PyKeyError::new_err(id.clone().unbind()))?;
        let old = {
            let mut this = slf.try_borrow_mut()?;
            let slot = this
                .hive
                .get_mut(key)
                .ok_or_else(|| PyKeyError::new_err(id.clone().unbind()))?;
            std::mem::replace(slot, value)
        };
        drop(old);
        Ok(())
    }

    fn __delitem__(slf: &Bound<'_, Self>, id: &Bound<'_, PyAny>) -> PyResult<()> {
        let key = Id::lookup(id);
        let removed = slf.try_borrow_mut()?.remove(key);
        drop(removed.ok_or_else(|| PyKeyError::new_err(id.clone().unbind()))?);
        Ok(())
    }

    fn __iter__(slf: &Bound<'_, Self>) -> PyResult<HiveIterator> {
        let this = slf.try_borrow()?;
        Ok(HiveIterator {
            owner: slf.clone().unbind(),
            ids: this.hive.ids().map(Id::to_bits).collect(),
            pos: 0,
            mods: this.mods,
            done: false,
        })
    }

    fn __eq__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let Ok(other) = other.cast::<Self>() else {
            return Ok(py.NotImplemented());
        };
        if slf.is(other) {
            return true.into_py_any(py);
        }
        let left = slf.try_borrow()?.snapshot(py);
        let right = other.try_borrow()?.snapshot(py);
        if left.len() != right.len() {
            return false.into_py_any(py);
        }
        for ((a, x), (b, y)) in left.into_iter().zip(right) {
            if a != b || (!x.is(&y) && !x.bind(py).eq(y.bind(py))?) {
                return false.into_py_any(py);
            }
        }
        true.into_py_any(py)
    }

    fn __ne__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let eq = Self::__eq__(slf, other)?;
        if eq.is(slf.py().NotImplemented()) {
            Ok(eq)
        } else {
            (!eq.bind(slf.py()).is_truthy()?).into_py_any(slf.py())
        }
    }

    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let name = slf.get_type().name()?.to_string();
        crate::repr_guarded(slf.as_any(), &name, || {
            let (items, more) = {
                let this = slf.try_borrow()?;
                (
                    this.hive
                        .iter()
                        .take(crate::REPR_ITEMS)
                        .map(|(id, v)| (id.to_bits(), v.clone_ref(slf.py())))
                        .collect::<Vec<_>>(),
                    this.hive.len() > crate::REPR_ITEMS,
                )
            };
            let mut parts = items
                .iter()
                .map(|(id, v)| Ok(format!("{id}: {}", v.bind(slf.py()).repr()?)))
                .collect::<PyResult<Vec<_>>>()?;
            if more {
                parts.push("...".to_owned());
            }
            Ok(format!(
                "{name}({{{}}}{})",
                parts.join(", "),
                crate::page_size_repr(slf.try_borrow()?.hive.page_size())
            ))
        })
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let (floors, lens, versions, items) = {
            let this = slf.try_borrow()?;
            let mut floors = Vec::new();
            let mut lens = Vec::new();
            let mut versions = Vec::new();
            for page in this.hive.raw_pages() {
                floors.push(page.floor);
                lens.push(page.slots.len() as u32);
                versions.extend(page.slots.into_iter().map(|slot| match slot {
                    RawSlot::Occupied { version, .. } | RawSlot::Vacant { version } => version,
                }));
            }
            (floors, lens, versions, this.snapshot(py))
        };
        let state = PyTuple::new(
            py,
            [
                PyArray1::from_vec(py, floors),
                PyArray1::from_vec(py, lens),
                PyArray1::from_vec(py, versions),
            ],
        )?;
        let items = PyList::new(py, items)?;
        crate::reduce(
            slf.as_any(),
            &py.get_type::<Self>(),
            vec![py.None().into_bound(py), state.into_any()],
            Some(items.try_iter()?),
        )
    }

    /// Shallow copy preserving ids and all generation history.
    fn copy(&self, py: Python<'_>) -> Self {
        Self {
            hive: HiveInner::try_from_raw_pages(
                self.hive.page_size(),
                self.hive.raw_pages().map(|page| RawPage {
                    floor: page.floor,
                    slots: page
                        .slots
                        .into_iter()
                        .map(|slot| match slot {
                            RawSlot::Occupied { version, value } => RawSlot::Occupied {
                                version,
                                value: value.clone_ref(py),
                            },
                            RawSlot::Vacant { version } => RawSlot::Vacant { version },
                        })
                        .collect(),
                }),
            )
            .expect("valid Hive state"),
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

    fn insert(slf: &Bound<'_, Self>, value: Py<PyAny>) -> PyResult<u64> {
        // No `reserve_insert` here: its reservation scans the page directory,
        // which would make a run of inserts quadratic. `try_insert` only
        // fails once no slot index is left.
        let result = {
            let mut this = slf.try_borrow_mut()?;
            let result = this.hive.try_insert(value);
            if result.is_ok() {
                this.changed();
            }
            result
        };
        // On failure the value comes back; drop it after the borrow ends.
        result.map(Id::to_bits).map_err(|_| exhausted())
    }

    fn insert_many<'py>(
        slf: &Bound<'py, Self>,
        values: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<u64>>> {
        let values = values
            .try_iter()?
            .map(|v| v.map(Bound::unbind))
            .collect::<PyResult<Vec<_>>>()?;
        let ids = Self::insert_values(slf, values)?;
        Ok(PyArray1::from_vec(slf.py(), ids))
    }

    #[pyo3(signature = (id, default=None))]
    fn get(
        slf: &Bound<'_, Self>,
        id: &Bound<'_, PyAny>,
        default: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let id = Id::lookup(id);
        let value = {
            let this = slf.try_borrow()?;
            id.and_then(|id| this.hive.get(id))
                .map(|v| v.clone_ref(slf.py()))
        };
        Ok(value.unwrap_or_else(|| default.unwrap_or_else(|| slf.py().None())))
    }

    #[pyo3(signature = (id, *default))]
    fn pop<'py>(
        slf: &Bound<'py, Self>,
        id: &Bound<'py, PyAny>,
        default: &Bound<'py, PyTuple>,
    ) -> PyResult<Py<PyAny>> {
        let default = optional_default(default, "pop")?;
        let key = Id::lookup(id);
        let value = slf.try_borrow_mut()?.remove(key);
        value
            .or_else(|| default.map(Bound::unbind))
            .ok_or_else(|| PyKeyError::new_err(id.clone().unbind()))
    }

    fn discard(slf: &Bound<'_, Self>, id: &Bound<'_, PyAny>) -> PyResult<bool> {
        let key = Id::lookup(id);
        let removed = slf.try_borrow_mut()?.remove(key);
        let found = removed.is_some();
        drop(removed);
        Ok(found)
    }

    #[pyo3(signature = (ids, *default))]
    fn get_many<'py>(
        slf: &Bound<'py, Self>,
        ids: &Bound<'py, PyAny>,
        default: &Bound<'py, PyTuple>,
    ) -> PyResult<Bound<'py, PyList>> {
        let default = optional_default(default, "get_many")?;
        let batch = KeyBatch::<Id>::new(ids)?;
        let mut out = Vec::with_capacity(batch.len());
        let mut missing = None;
        {
            let this = slf.try_borrow()?;
            batch.for_each(|id| match id.and_then(|id| this.hive.get(id)) {
                Some(value) => out.push(value.bind(slf.py()).clone()),
                None => {
                    if default.is_none() {
                        missing.get_or_insert(out.len());
                    }
                    out.push(
                        default
                            .clone()
                            .unwrap_or_else(|| slf.py().None().into_bound(slf.py())),
                    );
                }
            });
        }
        match missing {
            Some(pos) => Err(PyKeyError::new_err(batch.item(pos)?.unbind())),
            None => PyList::new(slf.py(), out),
        }
    }

    fn contains_many<'py>(
        slf: &Bound<'py, Self>,
        ids: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let batch = KeyBatch::<Id>::new(ids)?;
        let mut out = Vec::with_capacity(batch.len());
        {
            let this = slf.try_borrow()?;
            batch.for_each(|id| out.push(id.is_some_and(|id| this.hive.contains(id))));
        }
        Ok(PyArray1::from_vec(slf.py(), out))
    }

    fn discard_many(slf: &Bound<'_, Self>, ids: &Bound<'_, PyAny>) -> PyResult<usize> {
        let batch = KeyBatch::<Id>::new(ids)?;
        let mut keys = Vec::new();
        batch.for_each(|id| keys.extend(id));
        let removed = slf.try_borrow_mut()?.remove_ids(keys);
        let count = removed.len();
        drop(removed);
        Ok(count)
    }

    fn ids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        PyArray1::from_vec(py, self.hive.ids().map(Id::to_bits).collect())
    }

    fn values(&self, py: Python<'_>) -> Vec<Py<PyAny>> {
        self.hive.values().map(|v| v.clone_ref(py)).collect()
    }

    fn items(&self, py: Python<'_>) -> Vec<(u64, Py<PyAny>)> {
        self.snapshot(py)
    }

    fn occupant(slf: &Bound<'_, Self>, index: &Bound<'_, PyAny>) -> PyResult<Option<u64>> {
        // Normalize before borrowing: integer extraction can call __index__.
        let index = index.extract::<u32>().ok();
        let this = slf.try_borrow()?;
        Ok(index.and_then(|index| this.hive.occupant(index).map(Id::to_bits)))
    }

    /// Evaluate a snapshot, then remove rejected ids that are still live.
    /// Callback mutations remain in place; newly inserted ids are not visited.
    fn retain(slf: &Bound<'_, Self>, predicate: &Bound<'_, PyAny>) -> PyResult<()> {
        let items = slf.try_borrow()?.snapshot(slf.py());
        let mut rejected = Vec::new();
        for (bits, value) in items {
            if !predicate.call1((bits, value))?.is_truthy()? {
                rejected.push(Id::from_bits(bits).expect("live id"));
            }
        }
        let removed = slf.try_borrow_mut()?.remove_ids(rejected);
        drop(removed);
        Ok(())
    }

    /// Remove all values, preserving the generation history.
    fn clear(slf: &Bound<'_, Self>) -> PyResult<()> {
        let removed: Vec<_> = {
            let mut this = slf.try_borrow_mut()?;
            let removed = this.hive.drain().collect();
            this.changed();
            removed
        };
        drop(removed);
        Ok(())
    }

    fn reserve(slf: &Bound<'_, Self>, additional: &Bound<'_, PyAny>) -> PyResult<()> {
        let additional = additional.extract::<usize>()?;
        slf.try_borrow_mut()?
            .hive
            .try_reserve(additional)
            .map_err(crate::reserve_error)
    }

    fn shrink_to_fit(&mut self) {
        self.hive.shrink_to_fit();
    }

    #[getter]
    fn index_bound(&self) -> usize {
        self.hive.index_bound()
    }

    #[getter]
    fn capacity(&self) -> usize {
        self.hive.capacity()
    }

    #[getter]
    fn allocated_pages(&self) -> usize {
        self.hive.allocated_pages()
    }

    #[getter]
    fn is_exhausted(&self) -> bool {
        self.hive.is_exhausted()
    }

    #[classattr]
    fn dtype() -> &'static str {
        "object"
    }

    #[getter]
    fn page_size(&self) -> usize {
        self.hive.page_size()
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        for value in self.hive.values() {
            visit.call(value)?;
        }
        Ok(())
    }

    fn __clear__(slf: &Bound<'_, Self>) -> PyResult<()> {
        Self::clear(slf)
    }
}

impl Hive {
    fn insert_values(slf: &Bound<'_, Self>, values: Vec<Py<PyAny>>) -> PyResult<Vec<u64>> {
        let mut ids = Vec::new();
        ids.try_reserve(values.len())
            .map_err(crate::reserve_error)?;
        slf.try_borrow_mut()?.reserve_insert(values.len())?;
        let mut this = slf.try_borrow_mut()?;
        for value in values {
            ids.push(
                this.hive
                    .try_insert(value)
                    .expect("capacity checked")
                    .to_bits(),
            );
        }
        if !ids.is_empty() {
            this.changed();
        }
        Ok(ids)
    }
}

/// Snapshot of slot-ordered ids, invalidated by structural changes.
#[pyclass(module = "paged_collections._core")]
pub(crate) struct HiveIterator {
    owner: Py<Hive>,
    ids: Vec<u64>,
    pos: usize,
    mods: u64,
    done: bool,
}

#[pymethods]
impl HiveIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<u64>> {
        if self.done {
            return Ok(None);
        }
        if self.owner.try_borrow(py)?.mods != self.mods {
            return Err(PyRuntimeError::new_err("Hive changed during iteration"));
        }
        let id = self.ids.get(self.pos).copied();
        self.pos += usize::from(id.is_some());
        self.done = id.is_none();
        Ok(id)
    }

    fn __length_hint__(&self) -> usize {
        self.ids.len() - self.pos
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.owner)
    }
}
