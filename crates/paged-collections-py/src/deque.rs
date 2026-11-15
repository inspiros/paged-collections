//! `PagedDeque`: a `collections.deque` of objects stored in pages. Python
//! code (comparisons, `__index__`, destructors) always runs with no borrow of
//! the deque held: elements are cloned out first, and removed elements are
//! dropped after the borrow ends.

use pyo3::exceptions::{
    PyIndexError, PyMemoryError, PyOverflowError, PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;
use pyo3::types::{PyDict, PyInt, PyList, PyTuple};
use pyo3::{IntoPyObjectExt, PyTraverseError, PyVisit};

use crate::storage::DequeInner;

/// Reuse the shallow reconstruction used by copy.copy, including subclass
/// state and slots, without dispatching to a subclass's __init__ or __copy__.
pub(crate) fn copy_from_reduce<'py>(
    obj: &Bound<'py, PyAny>,
    reduced: Bound<'py, PyTuple>,
) -> PyResult<Py<PyAny>> {
    let py = obj.py();
    let mut args = vec![obj.clone(), py.None().into_bound(py)];
    args.extend(reduced.iter());
    py.import("copy")?
        .getattr("_reconstruct")?
        .call1(PyTuple::new(py, args)?)
        .map(Bound::unbind)
}

fn strict_index(obj: &Bound<'_, PyAny>) -> PyResult<isize> {
    obj.py()
        .import("operator")?
        .call_method1("index", (obj,))?
        .extract()
}

fn subscript_index(obj: &Bound<'_, PyAny>) -> PyResult<isize> {
    let int = obj.py().import("operator")?.call_method1("index", (obj,))?;
    int.extract().map_err(|err: PyErr| {
        if err.is_instance_of::<PyOverflowError>(obj.py()) {
            PyIndexError::new_err("PagedDeque index out of range")
        } else {
            err
        }
    })
}

fn mutated(removing: bool) -> PyErr {
    if removing {
        PyIndexError::new_err("PagedDeque mutated during iteration")
    } else {
        PyRuntimeError::new_err("PagedDeque mutated during iteration")
    }
}

/// `operator.index(obj)`, saturated to `isize`. Convert before borrowing:
/// `__index__` can run arbitrary Python code.
fn signed_index(obj: &Bound<'_, PyAny>) -> PyResult<isize> {
    let int = obj.py().import("operator")?.call_method1("index", (obj,))?;
    match int.extract::<isize>() {
        Ok(i) => Ok(i),
        Err(_) => Ok(if int.lt(0)? { isize::MIN } else { isize::MAX }),
    }
}

fn position(i: isize, len: usize) -> PyResult<usize> {
    let i = if i < 0 {
        i.saturating_add(len as isize)
    } else {
        i
    };
    if i < 0 || i as usize >= len {
        Err(PyIndexError::new_err("PagedDeque index out of range"))
    } else {
        Ok(i as usize)
    }
}

/// A slice bound clamped to `0..=len`, as `list.index` and `deque.insert` do.
fn bound(i: isize, len: usize) -> usize {
    if i < 0 {
        i.saturating_add(len as isize).max(0) as usize
    } else {
        (i as usize).min(len)
    }
}

fn equal_objects(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(a.is(b) || a.eq(b)?)
}

/// A double-ended queue of objects stored in pages of configurable size, with
/// the interface of ``collections.deque``. Pushing at either end never moves
/// existing elements. With ``maxlen``, pushing onto a full deque discards an
/// element from the other end.
#[pyclass(module = "paged_collections._core", subclass, weakref)]
pub(crate) struct PagedDeque {
    deque: DequeInner<Py<PyAny>>,
    maxlen: Option<usize>,
    /// CPython's state: endpoint movement invalidates iterators, replacement
    /// and reverse do not. Empty clear and rotations of length <= 1 are inert.
    mods: u64,
}

impl PagedDeque {
    fn empty(maxlen: Option<usize>) -> Self {
        Self {
            deque: DequeInner::default(),
            maxlen,
            mods: 0,
        }
    }
    fn changed(&mut self) {
        self.mods = self.mods.wrapping_add(1);
    }
    fn snapshot(&self, py: Python<'_>) -> Vec<Py<PyAny>> {
        self.deque.iter().map(|v| v.clone_ref(py)).collect()
    }
    /// Snapshot self-extension before pushing anything. Like PySequence_List
    /// in deque_extend, list(self) honors a subclass's iteration protocol.
    fn self_snapshot(slf: &Bound<'_, Self>) -> PyResult<Vec<Py<PyAny>>> {
        if slf.as_any().is_exact_instance_of::<Self>() {
            Ok(slf.try_borrow()?.snapshot(slf.py()))
        } else {
            slf.py().get_type::<PyList>().call1((slf,))?.extract()
        }
    }
    fn push_py(slf: &Bound<'_, Self>, value: Py<PyAny>, front: bool) -> PyResult<()> {
        // Keep ownership outside the borrow, including all error paths.
        let mut value = Some(value);
        let mut evicted = Vec::new();
        {
            let mut this = slf.try_borrow_mut()?;
            if this.maxlen != Some(0) {
                // A front push may need an entire spare page even when the
                // tail has unused slots. Make that allocation fallible too.
                let additional = if front { this.deque.page_size() } else { 1 };
                this.deque
                    .try_reserve(additional)
                    .map_err(crate::reserve_error)?;
                if let Some(limit) = this.maxlen {
                    evicted
                        .try_reserve(this.deque.len().saturating_sub(limit) + 1)
                        .map_err(crate::reserve_error)?;
                    while this.deque.len() >= limit {
                        let old = if front {
                            this.deque.pop_back()
                        } else {
                            this.deque.pop_front()
                        };
                        evicted.push(old.expect("a positive full limit has an element"));
                    }
                }
                if front {
                    this.deque
                        .push_front(value.take().expect("value is present"));
                } else {
                    this.deque
                        .push_back(value.take().expect("value is present"));
                }
            }
            // append at maxlen=0 still moves CPython's endpoints; extend at
            // maxlen=0 is handled separately and only consumes its iterator.
            this.changed();
        }
        drop(evicted);
        drop(value);
        Ok(())
    }
    /// Clones element `i` and releases the borrow before any comparison.
    fn at(slf: &Bound<'_, Self>, i: usize) -> PyResult<Option<Py<PyAny>>> {
        Ok(slf
            .try_borrow()?
            .deque
            .get(i)
            .map(|v| v.clone_ref(slf.py())))
    }
    /// The first position in `start..stop` holding a value equal to `value`.
    /// contains/index return a successful match before checking state;
    /// remove checks state first and raises IndexError on mutation.
    fn find(
        slf: &Bound<'_, Self>,
        value: &Bound<'_, PyAny>,
        start: usize,
        stop: usize,
        removing: bool,
    ) -> PyResult<Option<usize>> {
        let mods = slf.try_borrow()?.mods;
        for i in start..stop {
            let Some(item) = Self::at(slf, i)? else {
                break;
            };
            let equal = equal_objects(item.bind(slf.py()), value);
            drop(item);
            let equal = equal?;
            if equal && !removing {
                return Ok(Some(i));
            }
            let changed = slf.try_borrow()?.mods != mods;
            if changed {
                return Err(mutated(removing));
            }
            if equal {
                return Ok(Some(i));
            }
        }
        Ok(None)
    }

    fn repeat(slf: &Bound<'_, Self>, n: isize) -> PyResult<()> {
        let (size, maxlen) = {
            let this = slf.try_borrow()?;
            (this.deque.len(), this.maxlen)
        };
        if size == 0 || n == 1 {
            return Ok(());
        }
        if n <= 0 {
            return Self::clear(slf);
        }
        let py = slf.py();
        let vals = slf.try_borrow()?.snapshot(py);
        if size == 1 {
            let total = maxlen.map_or(n as usize, |limit| limit.min(n as usize));
            let mut this = slf.try_borrow_mut()?;
            this.deque
                .try_reserve(total.saturating_sub(1))
                .map_err(crate::reserve_error)?;
            this.changed();
            for _ in 1..total {
                this.deque.push_back(vals[0].clone_ref(py));
            }
            return Ok(());
        }
        let total = size
            .checked_mul(n as usize)
            .filter(|&len| len <= isize::MAX as usize)
            .ok_or_else(|| PyMemoryError::new_err("repeated PagedDeque is too large"))?;
        let repeats = match maxlen {
            Some(limit) if total > limit => limit.div_ceil(size),
            _ => n as usize,
        };
        // Retain the original sequence throughout, as deque_inplace_repeat
        // does. Bounded repeats need only enough passes to fill maxlen.
        for _ in 1..repeats {
            for value in &vals {
                Self::push_py(slf, value.clone_ref(py), false)?;
            }
        }
        Ok(())
    }
    fn parse_maxlen(maxlen: Option<&Bound<'_, PyAny>>) -> PyResult<Option<usize>> {
        match maxlen {
            None => Ok(None),
            Some(m) if m.is_none() => Ok(None),
            Some(m) => {
                // PyLong_AsSsize_t in deque_init accepts actual integers,
                // rather than arbitrary __index__ providers.
                let n = m.cast::<PyInt>()?.extract::<isize>()?;
                if n < 0 {
                    Err(PyValueError::new_err("maxlen must be non-negative"))
                } else {
                    Ok(Some(n as usize))
                }
            }
        }
    }
}

#[pymethods]
impl PagedDeque {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs), text_signature = "(iterable=(), maxlen=None, *, page_size=DEFAULT_PAGE_SIZE)")]
    fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        Self::empty(None)
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn __init__(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        // Parse manually to distinguish an omitted iterable from explicit None.
        if args.len() > 2 {
            return Err(PyTypeError::new_err(
                "PagedDeque expected at most 2 arguments",
            ));
        }
        let mut iterable = if !args.is_empty() {
            Some(args.get_item(0)?)
        } else {
            None
        };
        let mut maxlen = if args.len() > 1 {
            Some(args.get_item(1)?)
        } else {
            None
        };
        let mut page_size = crate::DEFAULT_PAGE_SIZE;
        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs.iter() {
                match key.extract::<&str>()? {
                    "iterable" if iterable.is_none() => iterable = Some(value),
                    "maxlen" if maxlen.is_none() => maxlen = Some(value),
                    "page_size" => page_size = crate::extract_page_size(&value)?,
                    "iterable" | "maxlen" => {
                        return Err(PyTypeError::new_err("multiple values for argument"));
                    }
                    _ => return Err(PyTypeError::new_err("unexpected keyword argument")),
                }
            }
        }
        let replacement = DequeInner::new(page_size)?;
        let maxlen = Self::parse_maxlen(maxlen.as_ref())?;
        // Install before releasing old elements: their destructors may append.
        slf.try_borrow_mut()?.maxlen = maxlen;
        Self::clear(slf)?;
        if slf.try_borrow()?.deque.page_size() != page_size {
            let vals = slf.try_borrow()?.snapshot(slf.py());
            let mut this = slf.try_borrow_mut()?;
            this.deque = replacement;
            for value in vals {
                this.deque.push_back(value);
            }
            this.changed();
        }
        match iterable {
            Some(values) => Self::extend(slf, &values),
            None => Ok(()),
        }
    }

    fn __len__(slf: &Bound<'_, Self>) -> PyResult<usize> {
        Ok(slf.try_borrow()?.deque.len())
    }

    fn __getitem__(slf: &Bound<'_, Self>, index: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let i = subscript_index(index)?;
        let this = slf.try_borrow()?;
        let i = position(i, this.deque.len())?;
        Ok(this.deque[i].clone_ref(slf.py()))
    }

    fn __setitem__(
        slf: &Bound<'_, Self>,
        index: &Bound<'_, PyAny>,
        value: Py<PyAny>,
    ) -> PyResult<()> {
        let i = subscript_index(index)?;
        let old = {
            let mut this = slf.try_borrow_mut()?;
            let i = position(i, this.deque.len())?;
            std::mem::replace(&mut this.deque[i], value)
        };
        drop(old);
        Ok(())
    }

    fn __delitem__(slf: &Bound<'_, Self>, index: &Bound<'_, PyAny>) -> PyResult<()> {
        let i = subscript_index(index)?;
        let old = {
            let mut this = slf.try_borrow_mut()?;
            let i = position(i, this.deque.len())?;
            let old = this.deque.remove(i);
            this.changed();
            old
        };
        drop(old);
        Ok(())
    }

    fn __iter__(slf: &Bound<'_, Self>) -> PyResult<PagedDequeIterator> {
        PagedDequeIterator::new(slf, false)
    }

    fn __reversed__(slf: &Bound<'_, Self>) -> PyResult<PagedDequeIterator> {
        PagedDequeIterator::new(slf, true)
    }

    fn __contains__(slf: &Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        let len = slf.try_borrow()?.deque.len();
        Ok(Self::find(slf, value, 0, len, false)?.is_some())
    }

    fn __richcmp__(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
        op: CompareOp,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let Ok(other) = other.cast::<Self>() else {
            return Ok(py.NotImplemented());
        };
        let a_len = slf.try_borrow()?.deque.len();
        let b_len = other.try_borrow()?.deque.len();
        if matches!(op, CompareOp::Eq | CompareOp::Ne) {
            // CPython 3.12 answers `d == d` and `d != d` by identity, without
            // iterating, even for a subclass that overrides `__iter__`.
            if slf.is(other) {
                return matches!(op, CompareOp::Eq).into_py_any(py);
            }
            if a_len != b_len {
                return matches!(op, CompareOp::Ne).into_py_any(py);
            }
        }
        // Use live Python iterators, including subclass overrides, just as
        // deque_richcompare does. All calls and object drops are unborrowed.
        let mut a = slf.as_any().try_iter()?;
        let mut b = other.as_any().try_iter()?;
        loop {
            let x = a.next().transpose()?;
            let y = b.next().transpose()?;
            match (x, y) {
                (Some(x), Some(y)) => {
                    if !equal_objects(&x, &y)? {
                        // Equality was already tested. Do not call an
                        // element's __eq__ again for == or !=.
                        if matches!(op, CompareOp::Eq | CompareOp::Ne) {
                            return matches!(op, CompareOp::Ne).into_py_any(py);
                        }
                        return x.rich_compare(&y, op)?.is_truthy()?.into_py_any(py);
                    }
                }
                (x, y) => {
                    let result = match op {
                        CompareOp::Lt => y.is_some(),
                        CompareOp::Le => x.is_none(),
                        CompareOp::Eq => x.is_none() && y.is_none(),
                        CompareOp::Ne => x.is_some() || y.is_some(),
                        CompareOp::Gt => x.is_some(),
                        CompareOp::Ge => y.is_none(),
                    };
                    return result.into_py_any(py);
                }
            }
        }
    }

    fn __iadd__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::extend(slf, other)
    }

    fn __add__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if other.cast::<Self>().is_err() {
            return Ok(slf.py().NotImplemented());
        }
        let out = Self::copy(slf)?;
        Self::extend(out.bind(slf.py()).cast::<Self>()?, other)?;
        Ok(out)
    }

    fn __mul__(slf: &Bound<'_, Self>, n: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let n = strict_index(n)?;
        let out = Self::copy(slf)?;
        Self::repeat(out.bind(slf.py()).cast::<Self>()?, n)?;
        Ok(out)
    }

    fn __rmul__(slf: &Bound<'_, Self>, n: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Self::__mul__(slf, n)
    }

    fn __imul__(slf: &Bound<'_, Self>, n: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::repeat(slf, strict_index(n)?)
    }

    fn __bool__(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(!slf.try_borrow()?.deque.is_empty())
    }

    fn append(slf: &Bound<'_, Self>, value: Py<PyAny>) -> PyResult<()> {
        Self::push_py(slf, value, false)
    }

    fn appendleft(slf: &Bound<'_, Self>, value: Py<PyAny>) -> PyResult<()> {
        Self::push_py(slf, value, true)
    }

    fn extend(slf: &Bound<'_, Self>, iterable: &Bound<'_, PyAny>) -> PyResult<()> {
        if iterable.is(slf) {
            let vals = Self::self_snapshot(slf)?;
            let consume_only = slf.try_borrow()?.maxlen == Some(0);
            if !consume_only {
                for v in vals {
                    Self::push_py(slf, v, false)?;
                }
            }
            return Ok(());
        }
        let consume_only = slf.try_borrow()?.maxlen == Some(0);
        if consume_only {
            for value in iterable.try_iter()? {
                drop(value?);
            }
            return Ok(());
        }
        // Like deque.extend, a failing iterator leaves earlier values in.
        // The iterator advances only with no borrow held.
        for value in iterable.try_iter()? {
            Self::push_py(slf, value?.unbind(), false)?;
        }
        Ok(())
    }

    fn extendleft(slf: &Bound<'_, Self>, iterable: &Bound<'_, PyAny>) -> PyResult<()> {
        if iterable.is(slf) {
            let vals = Self::self_snapshot(slf)?;
            let consume_only = slf.try_borrow()?.maxlen == Some(0);
            if !consume_only {
                for v in vals {
                    Self::push_py(slf, v, true)?;
                }
            }
            return Ok(());
        }
        let consume_only = slf.try_borrow()?.maxlen == Some(0);
        if consume_only {
            for value in iterable.try_iter()? {
                drop(value?);
            }
            return Ok(());
        }
        for value in iterable.try_iter()? {
            Self::push_py(slf, value?.unbind(), true)?;
        }
        Ok(())
    }

    fn pop(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let value = {
            let mut this = slf.try_borrow_mut()?;
            let value = this.deque.pop_back();
            if value.is_some() {
                this.changed();
            }
            value
        };
        value.ok_or_else(|| PyIndexError::new_err("pop from an empty PagedDeque"))
    }

    fn popleft(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let value = {
            let mut this = slf.try_borrow_mut()?;
            let value = this.deque.pop_front();
            if value.is_some() {
                this.changed();
            }
            value
        };
        value.ok_or_else(|| PyIndexError::new_err("pop from an empty PagedDeque"))
    }

    fn clear(slf: &Bound<'_, Self>) -> PyResult<()> {
        let removed: Vec<_> = {
            let mut this = slf.try_borrow_mut()?;
            let removed: Vec<_> = std::iter::from_fn(|| this.deque.pop_front()).collect();
            if !removed.is_empty() {
                this.changed();
            }
            removed
        };
        drop(removed);
        Ok(())
    }

    fn copy(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        copy_from_reduce(slf.as_any(), Self::__reduce__(slf)?)
    }

    fn count(slf: &Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        let mut n = 0;
        let (len, mods) = {
            let this = slf.try_borrow()?;
            (this.deque.len(), this.mods)
        };
        for i in 0..len {
            let Some(item) = Self::at(slf, i)? else {
                break;
            };
            let equal = equal_objects(item.bind(slf.py()), value);
            drop(item);
            n += usize::from(equal?);
            let changed = slf.try_borrow()?.mods != mods;
            if changed {
                return Err(mutated(false));
            }
        }
        Ok(n)
    }

    #[pyo3(signature = (value, start=0, stop=isize::MAX, /))]
    fn index(
        slf: &Bound<'_, Self>,
        value: &Bound<'_, PyAny>,
        #[pyo3(from_py_with = signed_index)] start: isize,
        #[pyo3(from_py_with = signed_index)] stop: isize,
    ) -> PyResult<usize> {
        let len = slf.try_borrow()?.deque.len();
        let (start, stop) = (bound(start, len), bound(stop, len));
        Self::find(slf, value, start, stop, false)?
            .ok_or_else(|| PyValueError::new_err("value is not in PagedDeque"))
    }

    fn insert(slf: &Bound<'_, Self>, index: &Bound<'_, PyAny>, value: Py<PyAny>) -> PyResult<()> {
        let i = strict_index(index)?;
        let mut this = slf.try_borrow_mut()?;
        if this.maxlen.is_some_and(|limit| this.deque.len() >= limit) {
            return Err(PyIndexError::new_err(
                "PagedDeque already at its maximum size",
            ));
        }
        let i = bound(i, this.deque.len());
        // Core insert shifts toward the shorter side and may push_front.
        let page_size = this.deque.page_size();
        this.deque
            .try_reserve(page_size)
            .map_err(crate::reserve_error)?;
        this.deque.insert(i, value);
        this.changed();
        Ok(())
    }

    fn remove(slf: &Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let len = slf.try_borrow()?.deque.len();
        let i = Self::find(slf, value, 0, len, true)?
            .ok_or_else(|| PyValueError::new_err("PagedDeque.remove(x): x not in deque"))?;
        // `find` raised if a comparison mutated the deque, so `i` is current.
        let old = {
            let mut this = slf.try_borrow_mut()?;
            let old = this.deque.remove(i);
            this.changed();
            old
        };
        drop(old);
        Ok(())
    }

    /// Rotates ``n`` steps to the right (to the left if ``n`` is negative).
    #[pyo3(signature = (n=1, /))]
    fn rotate(
        slf: &Bound<'_, Self>,
        #[pyo3(from_py_with = strict_index)] n: isize,
    ) -> PyResult<()> {
        let mut this = slf.try_borrow_mut()?;
        let len = this.deque.len();
        if len <= 1 {
            return Ok(());
        }
        // Preallocate a spare page for either direction, then move only the
        // shorter end. Moved references are never decrefed. Even rotate(0)
        // invalidates when len > 1.
        let k = n.rem_euclid(len as isize) as usize;
        this.changed();
        if k != 0 {
            let page_size = this.deque.page_size();
            this.deque
                .try_reserve(page_size)
                .map_err(crate::reserve_error)?;
            if k <= len / 2 {
                for _ in 0..k {
                    let value = this.deque.pop_back().expect("rotation is nonempty");
                    this.deque.push_front(value);
                }
            } else {
                for _ in 0..len - k {
                    let value = this.deque.pop_front().expect("rotation is nonempty");
                    this.deque.push_back(value);
                }
            }
        }
        Ok(())
    }

    fn reverse(slf: &Bound<'_, Self>) -> PyResult<()> {
        let mut this = slf.try_borrow_mut()?;
        let len = this.deque.len();
        for i in 0..len / 2 {
            this.deque.swap(i, len - 1 - i);
        }
        Ok(())
    }

    #[getter]
    fn maxlen(slf: &Bound<'_, Self>) -> PyResult<Option<usize>> {
        Ok(slf.try_borrow()?.maxlen)
    }

    fn reserve(slf: &Bound<'_, Self>, additional: &Bound<'_, PyAny>) -> PyResult<()> {
        // Convert before borrowing: `__index__` can run Python code.
        let additional = additional.extract::<usize>()?;
        slf.try_borrow_mut()?
            .deque
            .try_reserve(additional)
            .map_err(crate::reserve_error)
    }

    fn shrink_to_fit(slf: &Bound<'_, Self>) -> PyResult<()> {
        slf.try_borrow_mut()?.deque.shrink_to_fit();
        Ok(())
    }

    #[getter]
    fn capacity(slf: &Bound<'_, Self>) -> PyResult<usize> {
        Ok(slf.try_borrow()?.deque.capacity())
    }

    #[getter]
    fn allocated_pages(slf: &Bound<'_, Self>) -> PyResult<usize> {
        Ok(slf.try_borrow()?.deque.allocated_pages())
    }

    #[getter]
    fn page_size(&self) -> usize {
        self.deque.page_size()
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let name = slf.get_type().name()?.to_string();
        crate::repr_guarded_with(
            slf.as_any(),
            || "[...]".to_owned(),
            || {
                let (vals, maxlen) = {
                    let this = slf.try_borrow()?;
                    (this.snapshot(slf.py()), this.maxlen)
                };
                let parts = vals
                    .iter()
                    .map(|v| Ok(v.bind(slf.py()).repr()?.to_string()))
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(match maxlen {
                    Some(m) => format!(
                        "{name}([{}], maxlen={m}{})",
                        parts.join(", "),
                        crate::page_size_repr(slf.try_borrow()?.deque.page_size())
                    ),
                    None => format!(
                        "{name}([{}]{})",
                        parts.join(", "),
                        crate::page_size_repr(slf.try_borrow()?.deque.page_size())
                    ),
                })
            },
        )
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let (vals, maxlen) = {
            let this = slf.try_borrow()?;
            (this.snapshot(py), this.maxlen)
        };
        // Rebuilt empty with its maxlen, then refilled through `extend`, so a
        // deque that contains itself is memoized before its items are copied.
        let contents = vec![PyTuple::empty(py).into_any(), maxlen.into_bound_py_any(py)?];
        let base = crate::reduce(slf.as_any(), &py.get_type::<Self>(), contents, None)?;
        let mut parts: Vec<_> = base.iter().collect();
        parts.push(PyList::new(py, vals)?.try_iter()?.into_any());
        PyTuple::new(py, parts)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        for value in self.deque.iter() {
            visit.call(value)?;
        }
        Ok(())
    }

    fn __clear__(slf: &Bound<'_, Self>) -> PyResult<()> {
        Self::clear(slf)
    }

    #[classmethod]
    fn __class_getitem__(
        cls: &Bound<'_, pyo3::types::PyType>,
        item: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let py = cls.py();
        py.import("types")?
            .getattr("GenericAlias")?
            .call1((cls, item))?
            .into_py_any(py)
    }
}

/// Iterates a `PagedDeque`; raises `RuntimeError` if the deque is mutated
/// between steps, as `collections.deque` iterators do.
#[pyclass(module = "paged_collections._core")]
pub(crate) struct PagedDequeIterator {
    owner: Py<PagedDeque>,
    mods: u64,
    /// Elements left to yield.
    left: usize,
    /// The next front position (or, reversed, one past the next position).
    pos: usize,
    reverse: bool,
}

impl PagedDequeIterator {
    fn new(owner: &Bound<'_, PagedDeque>, reverse: bool) -> PyResult<Self> {
        let this = owner.try_borrow()?;
        let len = this.deque.len();
        Ok(Self {
            owner: owner.clone().unbind(),
            mods: this.mods,
            left: len,
            pos: if reverse { len } else { 0 },
            reverse,
        })
    }
}

#[pymethods]
impl PagedDequeIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        if self.reverse && self.left == 0 {
            return Ok(None);
        }
        let owner = self.owner.try_borrow(py)?;
        if owner.mods != self.mods {
            self.left = 0;
            drop(owner);
            return Err(mutated(false));
        }
        if self.left == 0 {
            return Ok(None);
        }
        let i = if self.reverse {
            self.pos -= 1;
            self.pos
        } else {
            self.pos += 1;
            self.pos - 1
        };
        self.left -= 1;
        Ok(Some(owner.deque[i].clone_ref(py)))
    }
    fn __length_hint__(&self) -> usize {
        self.left
    }
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.owner)
    }
}
