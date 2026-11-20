//! Python bindings for the object FIFO and the sharded numeric priority queues.

use paged_collections::concurrent::{
    ConcurrentPriorityQueue as RustPriorityQueue, ConcurrentQueue as RustQueue,
};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyIndexError, PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::convert::{Key, insert_key, lookup_key};
use crate::priority_queue::{Number, Prio, extract_pairs};

fn index_at_least<'py>(
    obj: &Bound<'py, PyAny>,
    minimum: usize,
    message: &'static str,
) -> PyResult<Bound<'py, PyAny>> {
    let index = obj.py().import("operator")?.call_method1("index", (obj,))?;
    if index.lt(minimum)? {
        return Err(PyValueError::new_err(message));
    }
    Ok(index)
}

const QUEUE_COPY_ERROR: &str =
    "ConcurrentQueue cannot be copied or pickled; drain() into a list first";

/// An unbounded FIFO of Python objects for shared pushes and pops.
/// Length and truthiness are approximate under concurrency and exact when
/// no operation is in flight. Clear and drain require exclusive access and
/// raise RuntimeError (Already borrowed) when another call holds a borrow.
/// There is no iteration, peek, copy, or pickle support. Reference cycles
/// through this queue are not collected by Python's garbage collector.
#[pyclass(module = "paged_collections._core", weakref)]
pub(crate) struct ConcurrentQueue {
    queue: RustQueue<Py<PyAny>>,
}

#[pymethods]
impl ConcurrentQueue {
    #[new]
    #[pyo3(signature = (iterable = None))]
    fn new(iterable: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let this = Self {
            queue: RustQueue::new(),
        };
        if let Some(iterable) = iterable {
            this.push_many(iterable)?;
        }
        Ok(this)
    }

    /// Append an object to the queue.
    fn push(&self, obj: Py<PyAny>) {
        self.queue.push(obj);
    }

    /// Append items in iteration order and return the number pushed.
    /// Items already pushed remain queued if iteration raises.
    fn push_many(&self, iterable: &Bound<'_, PyAny>) -> PyResult<usize> {
        let mut count = 0;
        for item in iterable.try_iter()? {
            self.push(item?.unbind());
            count += 1;
        }
        Ok(count)
    }

    /// Remove and return the oldest item. Raises IndexError if empty.
    fn pop(&self) -> PyResult<Py<PyAny>> {
        self.queue
            .pop()
            .ok_or_else(|| PyIndexError::new_err("pop from an empty ConcurrentQueue"))
    }

    /// Remove up to n items in FIFO order, stopping at the first empty pop.
    fn pop_many(&self, n: &Bound<'_, PyAny>) -> PyResult<Vec<Py<PyAny>>> {
        let n = index_at_least(n, 0, "n must be nonnegative")?;
        let n = n.extract::<usize>().unwrap_or(usize::MAX);
        Ok(std::iter::from_fn(|| self.queue.pop()).take(n).collect())
    }

    fn __len__(&self) -> usize {
        self.queue.len()
    }

    fn __bool__(&self) -> bool {
        !self.queue.is_empty()
    }

    /// Remove all items. Requires exclusive access.
    fn clear(&mut self) {
        // Drop Python references while the thread is attached.
        self.queue.clear();
    }

    /// Remove and return all items in FIFO order. Requires exclusive access.
    fn drain(&mut self) -> Vec<Py<PyAny>> {
        self.queue.drain().collect()
    }

    fn __repr__(&self) -> String {
        format!("ConcurrentQueue(len={})", self.queue.len())
    }

    fn __copy__(&self) -> PyResult<()> {
        Err(PyTypeError::new_err(QUEUE_COPY_ERROR))
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> PyResult<()> {
        Err(PyTypeError::new_err(QUEUE_COPY_ERROR))
    }

    fn __reduce__(&self) -> PyResult<()> {
        Err(PyTypeError::new_err(QUEUE_COPY_ERROR))
    }
}

macro_rules! concurrent_priority_queue {
    ($name:ident, $t:ty, $dtype:literal) => {
        #[doc = concat!(
            "A sharded indexed priority queue with ", $dtype, " priorities and integer keys in ",
            "0..2**32. The best priority is the greatest by default, or the least with min=True. ",
            "Floating-point priorities reject NaN. ",
            "Ties have unspecified order. Length and truthiness are approximate under ",
            "concurrency and exact when no write is in flight. Items, sorted, copies and ",
            "pickling use one snapshot, consistent within each shard and weakly consistent ",
            "across shards. Clear requires exclusive access and conflicting calls raise ",
            "RuntimeError (Already borrowed)."
        )]
        #[pyclass(module = "paged_collections._core", weakref)]
        pub(crate) struct $name {
            queue: RustPriorityQueue<Key, Prio<$t>>,
            min: bool,
        }

        #[pymethods]
        impl $name {
            /// Validate all items before inserting them. Later duplicate keys win.
            #[new]
            #[pyo3(signature = (items = None, min = false, *, shards = None))]
            fn new(
                py: Python<'_>,
                items: Option<&Bound<'_, PyAny>>,
                min: bool,
                shards: Option<&Bound<'_, PyAny>>,
            ) -> PyResult<Self> {
                let shards = shards
                    .map(|value| {
                        index_at_least(value, 1, "shards must be at least 1")?.extract::<usize>()
                    })
                    .transpose()?;
                let pairs = extract_pairs::<$t>(items)?;
                Ok(py.detach(move || {
                    let mut queue =
                        shards.map_or_else(RustPriorityQueue::new, RustPriorityQueue::with_shards);
                    queue.extend(pairs.into_iter().map(|(key, value)| (key, Prio { value, min })));
                    Self { queue, min }
                }))
            }

            #[classattr]
            fn dtype() -> &'static str {
                $dtype
            }

            #[classattr]
            const __hash__: Option<Py<PyAny>> = None;

            #[getter]
            fn is_min(&self) -> bool {
                self.min
            }

            #[getter]
            fn shards(&self) -> usize {
                self.queue.shards()
            }

            #[getter]
            fn allocated_pages(&self) -> usize {
                self.queue.allocated_pages()
            }

            /// Set the priority of key and return its previous priority, if any.
            fn push(
                &self,
                key: &Bound<'_, PyAny>,
                priority: &Bound<'_, PyAny>,
            ) -> PyResult<Option<$t>> {
                let key = insert_key(key)?;
                let value = priority.extract::<$t>()?.check()?;
                Ok(self.queue.push(key, Prio { value, min: self.min }).map(|old| old.value))
            }

            fn __setitem__(&self, key: &Bound<'_, PyAny>, priority: &Bound<'_, PyAny>) -> PyResult<()> {
                self.push(key, priority).map(drop)
            }

            fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<$t> {
                lookup_key(key)
                    .and_then(|key| self.queue.priority(&key))
                    .map(|priority| priority.value)
                    .ok_or_else(|| PyKeyError::new_err(key.clone().unbind()))
            }

            #[pyo3(signature = (key, default = None))]
            fn get<'py>(
                &self,
                py: Python<'py>,
                key: &Bound<'_, PyAny>,
                default: Option<Bound<'py, PyAny>>,
            ) -> PyResult<Bound<'py, PyAny>> {
                match lookup_key(key).and_then(|key| self.queue.priority(&key)) {
                    Some(priority) => priority.value.into_bound_py_any(py),
                    None => Ok(default.unwrap_or_else(|| py.None().into_bound(py))),
                }
            }

            fn __contains__(&self, key: &Bound<'_, PyAny>) -> bool {
                lookup_key(key).is_some_and(|key| self.queue.contains_key(&key))
            }

            /// Remove key and return its priority. Raises KeyError if absent.
            fn remove(&self, key: &Bound<'_, PyAny>) -> PyResult<$t> {
                lookup_key(key)
                    .and_then(|key| self.queue.remove(&key))
                    .map(|(_, priority)| priority.value)
                    .ok_or_else(|| PyKeyError::new_err(key.clone().unbind()))
            }

            fn __delitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<()> {
                self.remove(key).map(drop)
            }

            /// Remove and return a (key, priority) pair using relaxed ordering.
            /// With multiple shards, the returned entry can rank below the
            /// global best. Use pop_exact() for the global best. Raises
            /// IndexError if the call finds no entry.
            fn pop(&self) -> PyResult<(Key, $t)> {
                self.queue.pop().map(|(key, priority)| (key, priority.value))
                    .ok_or_else(|| PyIndexError::new_err(concat!("pop from an empty ", stringify!($name))))
            }

            /// Remove and return the global best (key, priority). Raises IndexError if empty.
            fn pop_exact(&self) -> PyResult<(Key, $t)> {
                self.queue.pop_exact().map(|(key, priority)| (key, priority.value))
                    .ok_or_else(|| PyIndexError::new_err(concat!("pop from an empty ", stringify!($name))))
            }

            /// Return the global best (key, priority). Raises IndexError if empty.
            fn peek_exact(&self) -> PyResult<(Key, $t)> {
                self.queue.peek_exact().map(|(key, priority)| (key, priority.value))
                    .ok_or_else(|| PyIndexError::new_err(concat!("peek at an empty ", stringify!($name))))
            }

            fn __len__(&self) -> usize {
                self.queue.len()
            }

            fn __bool__(&self) -> bool {
                !self.queue.is_empty()
            }

            /// Remove all entries. Requires exclusive access.
            fn clear(&mut self) {
                self.queue.clear();
            }

            /// Return (key, priority) pairs from a weakly consistent snapshot.
            fn items(&self, py: Python<'_>) -> Vec<(Key, $t)> {
                py.detach(|| {
                    self.queue.snapshot().iter().map(|(&key, priority)| (key, priority.value)).collect()
                })
            }

            /// Return (key, priority) pairs from one snapshot, best priority first.
            fn sorted(&self, py: Python<'_>) -> Vec<(Key, $t)> {
                py.detach(|| {
                    self.queue.snapshot().into_sorted_iter().map(|(key, priority)| (key, priority.value)).collect()
                })
            }

            /// Return an independent queue rebuilt from a weakly consistent snapshot.
            fn copy(&self, py: Python<'_>) -> Self {
                Self { queue: py.detach(|| self.queue.clone()), min: self.min }
            }

            fn __copy__(&self, py: Python<'_>) -> Self {
                self.copy(py)
            }

            fn __deepcopy__(&self, py: Python<'_>, _memo: &Bound<'_, PyAny>) -> Self {
                self.copy(py)
            }

            fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
                let py = slf.py();
                let (items, min, shards) = {
                    let this = slf.try_borrow()?;
                    (this.items(py), this.min, this.shards())
                };
                // The reduce protocol passes positional arguments. Bind the
                // keyword-only shard count in a picklable callable.
                let kwargs = PyDict::new(py);
                kwargs.set_item("shards", shards)?;
                let constructor = py.import("functools")?.getattr("partial")?
                    .call((py.get_type::<Self>(),), Some(&kwargs))?;
                (constructor, (items, min)).into_pyobject(py)
            }
        }
    };
}

concurrent_priority_queue!(ConcurrentPriorityQueue, i64, "int64");
concurrent_priority_queue!(ConcurrentPriorityQueueF64, f64, "float64");
