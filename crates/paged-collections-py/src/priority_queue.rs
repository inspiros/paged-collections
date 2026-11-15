//! Indexed priority queues keyed by integers in `0..2**32`, with `int64`
//! (`PriorityQueue`) or `float64` (`PriorityQueueF64`) priorities. A queue
//! is a max-queue by default, or a min-queue with `min=True`. Priorities are
//! plain numbers, so no Python code runs while a queue is borrowed.

use std::cmp::Ordering;
use std::panic::{AssertUnwindSafe, catch_unwind};

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyIndexError, PyKeyError, PyMemoryError, PyOverflowError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::convert::{insert_key, lookup_key};

use crate::storage::QueueInner;

fn has_entry_room(len: usize, present: bool) -> bool {
    present || len < u32::MAX as usize
}

fn reserved_len(len: usize, additional: usize) -> Option<usize> {
    len.checked_add(additional)
        .filter(|&total| total <= u32::MAX as usize)
}

/// A number with the queue's direction: every priority in one queue has the
/// same `min`, so the order is total and consistent.
#[derive(Clone, Copy)]
struct Prio<T> {
    value: T,
    min: bool,
}

/// A priority type: `int64` or non-NaN `float64`.
trait Number: Copy + PartialOrd + 'static {
    fn check(self) -> PyResult<Self> {
        Ok(self)
    }
}
impl Number for i64 {}
impl Number for f64 {
    fn check(self) -> PyResult<Self> {
        if self.is_nan() {
            Err(PyValueError::new_err("priority must not be NaN"))
        } else {
            Ok(self)
        }
    }
}

impl<T: Number> PartialEq for Prio<T> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl<T: Number> Eq for Prio<T> {}
impl<T: Number> PartialOrd for Prio<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<T: Number> Ord for Prio<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        // No NaN is ever stored, so `partial_cmp` is total here.
        let ord = self
            .value
            .partial_cmp(&other.value)
            .unwrap_or(Ordering::Equal);
        if self.min { ord.reverse() } else { ord }
    }
}

macro_rules! priority_queue {
    ($name:ident, $pyname:literal, $t:ty, $dtype:literal) => {
        #[doc = concat!(
                            "An indexed priority queue mapping integer keys in ``0..2**32`` to ",
                            $dtype, " priorities. ``pop`` returns the greatest priority, or the ",
                            "least with ``min=True``; ties have unspecified order. Each key ",
                            "appears at most once: pushing a present key changes its priority. ",
                            "Lookups, pushes, priority changes and removals take O(log n)."
                        )]
        #[pyclass(module = "paged_collections._core", name = $pyname, subclass, weakref)]
        pub(crate) struct $name {
            queue: QueueInner<Prio<$t>>,
            min: bool,
        }

        impl $name {
            fn empty(min: bool) -> Self {
                Self {
                    queue: QueueInner::default(),
                    min,
                }
            }
            /// Entries in heap order.
            fn entries(&self) -> Vec<(u32, $t)> {
                self.queue.iter().map(|(k, p)| (*k, p.value)).collect()
            }
            fn push_number(&mut self, key: u32, value: $t) -> PyResult<Option<$t>> {
                if !has_entry_room(self.queue.len(), self.queue.contains_key(&key)) {
                    return Err(PyValueError::new_err(concat!($pyname, " is full")));
                }
                Ok(self.queue.push(key, Prio { value, min: self.min }).map(|old| old.value))
            }
        }

        #[pymethods]
        impl $name {
            #[new]
            #[pyo3(signature = (*_args, **_kwargs), text_signature = "(items=None, min=False, *, page_size=DEFAULT_PAGE_SIZE)")]
            fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
                Self::empty(false)
            }

            #[pyo3(signature = (items=None, min=false, *, page_size=crate::DEFAULT_PAGE_SIZE))]
            fn __init__(
                slf: &Bound<'_, Self>,
                items: Option<&Bound<'_, PyAny>>,
                min: bool,
                #[pyo3(from_py_with = crate::extract_page_size)] page_size: usize,
            ) -> PyResult<()> {
                // Validate both keys and priorities without borrowing or
                // modifying self. On failure the previous contents/direction
                // survive; conversion callbacks retain their own side effects.
                let mut pairs = Vec::new();
                if let Some(items) = items {
                    // As with dict(items), a keys attribute selects the
                    // mapping protocol, including on dict subclasses. An
                    // exact dict cannot override it, so read a snapshot of
                    // its items: conversion callbacks may resize it.
                    if let Ok(d) = items.cast_exact::<PyDict>() {
                        for item in d.items().iter() {
                            let (k, p): (Bound<'_, PyAny>, Bound<'_, PyAny>) = item.extract()?;
                            pairs.push((insert_key(&k)?, p.extract::<$t>()?.check()?));
                        }
                    } else if let Some(keys) = items.getattr_opt("keys")? {
                        for k in keys.call0()?.try_iter()? {
                            let k = k?;
                            let p = items.get_item(&k)?;
                            pairs.push((insert_key(&k)?, p.extract::<$t>()?.check()?));
                        }
                    } else {
                        for item in items.try_iter()? {
                            let item = item?;
                            let (k, p): (Bound<'_, PyAny>, Bound<'_, PyAny>) = item.extract()?;
                            pairs.push((insert_key(&k)?, p.extract::<$t>()?.check()?));
                        }
                    }
                }
                let mut replacement = Self { queue: QueueInner::new(page_size)?, min };
                for (k, p) in pairs {
                    replacement.push_number(k, p)?;
                }
                let old = {
                    let mut this = slf.try_borrow_mut()?;
                    std::mem::replace(&mut *this, replacement)
                };
                drop(old);
                Ok(())
            }

            fn __len__(slf: &Bound<'_, Self>) -> PyResult<usize> {
                Ok(slf.try_borrow()?.queue.len())
            }

            fn __bool__(slf: &Bound<'_, Self>) -> PyResult<bool> {
                Ok(!slf.try_borrow()?.queue.is_empty())
            }

            fn __contains__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
                let key = lookup_key(key);
                let this = slf.try_borrow()?;
                Ok(key.is_some_and(|k| this.queue.contains_key(&k)))
            }

            fn __getitem__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<$t> {
                let k = lookup_key(key);
                let value = {
                    let this = slf.try_borrow()?;
                    k.and_then(|k| this.queue.priority(&k)).map(|p| p.value)
                };
                value.ok_or_else(|| PyKeyError::new_err(key.clone().unbind()))
            }

            fn __setitem__(
                slf: &Bound<'_, Self>,
                key: &Bound<'_, PyAny>,
                priority: &Bound<'_, PyAny>,
            ) -> PyResult<()> {
                Self::push(slf, key, priority).map(drop)
            }

            fn __delitem__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<()> {
                Self::remove(slf, key).map(drop)
            }

            /// The priority of ``key``, or ``default`` if absent.
            #[pyo3(signature = (key, default=None))]
            fn get(
                slf: &Bound<'_, Self>,
                key: &Bound<'_, PyAny>,
                default: Option<Py<PyAny>>,
            ) -> PyResult<Py<PyAny>> {
                let py = slf.py();
                let k = lookup_key(key);
                let value = {
                    let this = slf.try_borrow()?;
                    k.and_then(|k| this.queue.priority(&k)).map(|p| p.value)
                };
                match value {
                    Some(p) => p.into_py_any(py),
                    None => Ok(default.unwrap_or_else(|| py.None())),
                }
            }

            /// Sets the priority of ``key``, adding it if absent. Returns the
            /// previous priority, or ``None``.
            fn push(
                slf: &Bound<'_, Self>,
                key: &Bound<'_, PyAny>,
                priority: &Bound<'_, PyAny>,
            ) -> PyResult<Option<$t>> {
                let k = insert_key(key)?;
                let p = priority.extract::<$t>()?.check()?;
                slf.try_borrow_mut()?.push_number(k, p)
            }

            /// Removes and returns ``(key, priority)`` with the greatest
            /// (least, for a min-queue) priority.
            fn pop(slf: &Bound<'_, Self>) -> PyResult<(u32, $t)> {
                let value = slf.try_borrow_mut()?.queue.pop().map(|(k, p)| (k, p.value));
                value.ok_or_else(|| PyIndexError::new_err(concat!("pop from an empty ", $pyname)))
            }

            /// ``(key, priority)`` that ``pop`` would return, without removing it.
            fn peek(slf: &Bound<'_, Self>) -> PyResult<(u32, $t)> {
                let value = slf.try_borrow()?.queue.peek().map(|(k, p)| (*k, p.value));
                value.ok_or_else(|| PyIndexError::new_err(concat!("peek at an empty ", $pyname)))
            }

            /// Removes ``key`` and returns its priority. Raises ``KeyError`` if absent.
            fn remove(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<$t> {
                let k = lookup_key(key);
                let value = {
                    let mut this = slf.try_borrow_mut()?;
                    k.and_then(|k| this.queue.remove(&k)).map(|(_, p)| p.value)
                };
                value.ok_or_else(|| PyKeyError::new_err(key.clone().unbind()))
            }

            fn clear(slf: &Bound<'_, Self>) -> PyResult<()> {
                slf.try_borrow_mut()?.queue.clear();
                Ok(())
            }

            fn copy(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
                crate::deque::copy_from_reduce(slf.as_any(), Self::__reduce__(slf)?)
            }

            /// Keys in heap order (a snapshot).
            fn keys<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyList>> {
                let keys: Vec<_> = slf.try_borrow()?.queue.keys().copied().collect();
                PyList::new(slf.py(), keys)
            }

            /// ``(key, priority)`` pairs in heap order (a snapshot).
            fn items<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyList>> {
                let entries = slf.try_borrow()?.entries();
                PyList::new(slf.py(), entries)
            }

            /// ``(key, priority)`` pairs in ``pop`` order, without changing the queue.
            fn sorted<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyList>> {
                let queue = slf.try_borrow()?.queue.clone();
                let sorted: Vec<_> = queue
                    .into_sorted_iter()
                    .map(|(k, p)| (k, p.value))
                    .collect();
                PyList::new(slf.py(), sorted)
            }

            /// Iterates over a snapshot of the keys, in heap order.
            fn __iter__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
                Self::keys(slf)?.try_iter().map(Bound::into_any)
            }

            fn __eq__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                let py = slf.py();
                let Ok(other) = other.cast::<Self>() else {
                    return Ok(py.NotImplemented());
                };
                let equal = {
                    let this = slf.try_borrow()?;
                    let other = other.try_borrow()?;
                    this.min == other.min
                    && this.queue.len() == other.queue.len()
                    && this
                        .queue
                        .iter()
                        .all(|(k, p)| other.queue.priority(k).is_some_and(|q| q.value == p.value))
                };
                equal.into_py_any(py)
            }

            #[getter]
            fn is_min(slf: &Bound<'_, Self>) -> PyResult<bool> {
                Ok(slf.try_borrow()?.min)
            }

            fn reserve(slf: &Bound<'_, Self>, additional: &Bound<'_, PyAny>) -> PyResult<()> {
                let additional = additional.extract::<usize>()?;
                let (len, capacity) = {
                    let this = slf.try_borrow()?;
                    (this.queue.len(), this.queue.capacity())
                };
                let total = reserved_len(len, additional).ok_or_else(||
                    PyOverflowError::new_err("reservation exceeds u32::MAX entries"))?;
                if total <= capacity { return Ok(()); }
                // Core has no try_reserve. Probe both dense allocation sizes
                // fallibly before calling it, and translate capacity panics.
                let mut keys = Vec::<u32>::new();
                keys.try_reserve_exact(total).map_err(crate::reserve_error)?;
                let mut priorities = Vec::<Prio<$t>>::new();
                priorities.try_reserve_exact(total).map_err(crate::reserve_error)?;
                drop((keys, priorities));
                let result = catch_unwind(AssertUnwindSafe(|| {
                    slf.try_borrow_mut()?.queue.reserve(additional);
                    Ok(())
                }));
                result.unwrap_or_else(|_| Err(PyMemoryError::new_err("priority queue reservation failed")))
            }

            fn shrink_to_fit(slf: &Bound<'_, Self>) -> PyResult<()> {
                slf.try_borrow_mut()?.queue.shrink_to_fit();
                Ok(())
            }

            #[getter]
            fn capacity(slf: &Bound<'_, Self>) -> PyResult<usize> {
                Ok(slf.try_borrow()?.queue.capacity())
            }

            #[getter]
            fn allocated_pages(slf: &Bound<'_, Self>) -> PyResult<usize> {
                Ok(slf.try_borrow()?.queue.allocated_pages())
            }

            #[classattr]
            fn dtype() -> &'static str {
                $dtype
            }

            #[getter]
            fn page_size(&self) -> usize { self.queue.page_size() }

            fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
                let name = slf.get_type().name()?.to_string();
                let (entries, min) = {
                    let this = slf.try_borrow()?;
                    (this.entries(), this.min)
                };
                let body = crate::repr_items(entries.into_iter(), slf.py())?;
                Ok(if min {
                    format!("{name}({{{body}}}, min=True{})", crate::page_size_repr(slf.try_borrow()?.queue.page_size()))
                } else {
                    format!("{name}({{{body}}}{})", crate::page_size_repr(slf.try_borrow()?.queue.page_size()))
                })
            }

            fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
                let py = slf.py();
                let (entries, min) = {
                    let this = slf.try_borrow()?;
                    (this.entries(), this.min)
                };
                let items = PyList::new(py, entries)?.into_any();
                let min = min.into_bound_py_any(py)?;
                crate::reduce(slf.as_any(), &py.get_type::<Self>(), vec![items, min], None)
            }
        }
    };
}

priority_queue!(PriorityQueue, "PriorityQueue", i64, "int64");
priority_queue!(PriorityQueueF64, "PriorityQueueF64", f64, "float64");

#[cfg(test)]
mod tests {
    use super::{has_entry_room, reserved_len};

    #[test]
    fn full_queue_allows_only_updates() {
        let full = u32::MAX as usize;
        // Both constructor and push use this check; no multi-gigabyte
        // allocation is needed to exercise the entry-count boundary.
        assert!(has_entry_room(full, true));
        assert!(!has_entry_room(full, false));
        assert!(has_entry_room(full - 1, false));
    }

    #[test]
    fn reservation_checks_final_length_and_overflow() {
        let full = u32::MAX as usize;
        assert_eq!(reserved_len(full, 0), Some(full));
        assert_eq!(reserved_len(full - 1, 1), Some(full));
        assert_eq!(reserved_len(1, full), None);
        assert_eq!(reserved_len(1, usize::MAX), None);
    }
}
