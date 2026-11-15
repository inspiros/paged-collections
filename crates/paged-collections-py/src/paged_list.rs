//! Python lists backed by stable-address pages. Object callbacks always run
//! outside collection borrows; typed writes use only `ValueCell::set`.

use std::cmp::Ordering;

use numpy::{PyArray1, PyArrayDescrMethods, PyUntypedArray, PyUntypedArrayMethods};
use paged_collections::paged_vec::PagedVec;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyIndexError, PyMemoryError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::slice::PySliceIndices;
use pyo3::types::{PyDict, PyList, PySlice, PyTuple};
use pyo3::{PyTraverseError, PyVisit};

use crate::convert::{TypedValue, as_ndarray, cast_to, value_batch};
use crate::export::{Exports, ValueCell, cell_view};

use crate::storage::ListInner;

/// Convert before borrowing: `__index__` can run arbitrary Python code.
/// Saturation is useful for insertion/search bounds and preserves out-of-range
/// detection for subscripts (Python containers cannot have isize::MAX entries).
fn signed_index(obj: &Bound<'_, PyAny>) -> PyResult<isize> {
    let int = obj.py().import("operator")?.call_method1("index", (obj,))?;
    match int.extract::<isize>() {
        Ok(i) => Ok(i),
        Err(_) => Ok(if int.lt(0)? { isize::MIN } else { isize::MAX }),
    }
}

/// As `list.insert` and `list * n` do: `OverflowError` for an int beyond
/// `isize`. Convert before borrowing.
fn strict_index(obj: &Bound<'_, PyAny>) -> PyResult<isize> {
    obj.py()
        .import("operator")?
        .call_method1("index", (obj,))?
        .extract::<isize>()
}

/// Replaces the `n` elements at `start..start + n` with `vals` and returns
/// the removed ones, in O(len - start) moves (never O(len * n)). The caller
/// reserves room for any growth first.
fn splice<T, const PAGE: usize>(
    list: &mut PagedVec<T, PAGE>,
    start: usize,
    n: usize,
    vals: Vec<T>,
) -> Vec<T> {
    let mut tail = list.split_off(start + n);
    let removed = list.drain(start..).collect();
    list.extend(vals);
    list.append(&mut tail);
    removed
}

/// Removes the elements at `positions` (distinct, any order) and returns
/// them, keeping the order of the rest, in O(len - min(positions)) moves.
fn remove_positions<T, const PAGE: usize>(
    list: &mut PagedVec<T, PAGE>,
    mut positions: Vec<usize>,
) -> Vec<T> {
    positions.sort_unstable();
    let Some(&lo) = positions.first() else {
        return Vec::new();
    };
    let rest: Vec<T> = list.drain(lo..).collect();
    let mut removed = Vec::with_capacity(positions.len());
    let mut next = positions.into_iter().peekable();
    for (k, value) in rest.into_iter().enumerate() {
        if next.next_if_eq(&(lo + k)).is_some() {
            removed.push(value);
        } else {
            // The drained pages are kept, so this allocates nothing.
            list.push(value);
        }
    }
    removed
}

/// `list.sort`'s `reverse`: any int (bools included), by truth value.
fn sort_reverse(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    if obj.cast::<pyo3::types::PyInt>().is_err() {
        return Err(PyTypeError::new_err(format!(
            "'{}' object cannot be interpreted as an integer",
            obj.get_type().name()?
        )));
    }
    obj.is_truthy()
}

fn position(i: isize, len: usize, name: &str) -> PyResult<usize> {
    let i = if i < 0 {
        i.saturating_add(len as isize)
    } else {
        i
    };
    if i < 0 || i as usize >= len {
        Err(PyIndexError::new_err(format!("{name} index out of range")))
    } else {
        Ok(i as usize)
    }
}

fn bound(i: isize, len: usize) -> usize {
    if i < 0 {
        i.saturating_add(len as isize).max(0) as usize
    } else {
        (i as usize).min(len)
    }
}

/// Canonical slices contain only exact ints/None: normalizing them while a
/// collection is borrowed cannot call a user's `__index__`.
enum Subscript<'py> {
    Int(isize),
    Slice(Bound<'py, PySlice>),
}

impl<'py> Subscript<'py> {
    fn new(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        if obj.cast::<PySlice>().is_ok() {
            // As CPython does: the step first, rejected if zero before the
            // bounds' `__index__` runs.
            let convert = |name: &str| -> PyResult<Bound<'py, PyAny>> {
                let part = obj.getattr(name)?;
                if part.is_none() {
                    Ok(part)
                } else {
                    signed_index(&part)?.into_bound_py_any(obj.py())
                }
            };
            let step = convert("step")?;
            if !step.is_none() && step.extract::<isize>()? == 0 {
                return Err(PyValueError::new_err("slice step cannot be zero"));
            }
            let parts = vec![convert("start")?, convert("stop")?, step];
            let slice = obj
                .py()
                .import("builtins")?
                .getattr("slice")?
                .call1(PyTuple::new(obj.py(), parts)?)?
                .cast_into::<PySlice>()?;
            Ok(Self::Slice(slice))
        } else {
            Ok(Self::Int(signed_index(obj)?))
        }
    }
}

fn slice_positions(s: &PySliceIndices) -> impl Iterator<Item = usize> + '_ {
    // Compute in i128: the final unused increment can overflow isize for a
    // very large step even though every selected index is in bounds.
    (0..s.slicelength).map(|n| (s.start as i128 + n as i128 * s.step as i128) as usize)
}

fn extended_size(got: usize, expected: usize) -> PyResult<()> {
    if got != expected {
        Err(PyValueError::new_err(format!(
            "attempt to assign sequence of size {got} to extended slice of size {expected}"
        )))
    } else {
        Ok(())
    }
}

fn repeat_len(len: usize, n: isize) -> PyResult<usize> {
    len.checked_mul(n.max(0) as usize)
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(|| PyMemoryError::new_err("repeated list is too long"))
}

fn objects(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Py<PyAny>>> {
    obj.try_iter()?.map(|x| x.map(Bound::unbind)).collect()
}

/// Iterable value conversion, including generators, without silently accepting
/// scalar inputs. Convert the entire batch before borrowing or changing a list.
fn typed_values<T: TypedValue>(obj: &Bound<'_, PyAny>) -> PyResult<Vec<T>> {
    let mut array = as_ndarray(obj)?;
    if array.ndim() == 0 && obj.cast::<PyUntypedArray>().is_err() {
        let list = obj
            .py()
            .import("builtins")?
            .getattr("list")?
            .call1((obj,))?;
        array = as_ndarray(&list)?;
    }
    if array.ndim() != 1 {
        return Err(PyValueError::new_err("values must be one-dimensional"));
    }
    if array.is_empty() {
        Ok(Vec::new())
    } else {
        T::convert(&array)
    }
}

/// Signed index batches, including unaligned, strided and non-native-endian
/// integer arrays. Read Python integer scalars rather than narrowing uint64 to
/// int64, which would wrap. A mixed huge-int sequence may become object/float
/// in NumPy; sequences then use operator.index element by element.
fn index_batch(obj: &Bound<'_, PyAny>) -> PyResult<Vec<isize>> {
    let array = as_ndarray(obj)?;
    if array.ndim() != 1 {
        return Err(PyValueError::new_err("indices must be one-dimensional"));
    }
    let kind = array.dtype().kind();
    if array.is_empty() {
        // A plain [] gets NumPy's default float dtype; explicit arrays still
        // have to meet the integer-index contract even when empty.
        if obj.cast::<PyUntypedArray>().is_ok() && !matches!(kind, b'i' | b'u') {
            return Err(PyTypeError::new_err("indices must be integers"));
        }
        return Ok(Vec::new());
    }
    if !matches!(kind, b'i' | b'u') {
        if obj.cast::<PyUntypedArray>().is_err() && matches!(kind, b'O' | b'f') {
            return obj.try_iter()?.map(|x| signed_index(&x?)).collect();
        }
        return Err(PyTypeError::new_err("indices must be integers"));
    }
    // Saturate: any index beyond isize is out of range for every list.
    Ok(if kind == b'i' {
        cast_to::<i64>(&array, "int64")?
            .into_iter()
            .map(|i| isize::try_from(i).unwrap_or(if i < 0 { isize::MIN } else { isize::MAX }))
            .collect()
    } else {
        cast_to::<u64>(&array, "uint64")?
            .into_iter()
            .map(|i| isize::try_from(i).unwrap_or(isize::MAX))
            .collect()
    })
}

fn normalized_batch(indices: &[isize], len: usize, name: &str) -> PyResult<Vec<usize>> {
    indices.iter().map(|&i| position(i, len, name)).collect()
}

fn equal_objects(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(a.is(b) || a.eq(b)?)
}

/// Position-based iterators: length is rechecked on each step; once exhausted,
/// they remain exhausted. The owner reference participates in cyclic GC.
macro_rules! list_iterator {
    ($iter:ident, $owner:ident, $t:ty, $read:expr) => {
        #[pyclass(module = "paged_collections._core")]
        pub(crate) struct $iter {
            owner: Py<$owner>,
            pos: usize,
            reverse: bool,
            done: bool,
        }
        impl $iter {
            fn new(owner: &Bound<'_, $owner>, reverse: bool) -> PyResult<Self> {
                let len = owner.try_borrow()?.list.len();
                Ok(Self {
                    owner: owner.clone().unbind(),
                    pos: if reverse { len } else { 0 },
                    reverse,
                    done: false,
                })
            }
        }
        #[pymethods]
        impl $iter {
            fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
                slf
            }
            fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<$t>> {
                if self.done {
                    return Ok(None);
                }
                let owner = self.owner.try_borrow(py)?;
                let index = if self.reverse {
                    self.pos.checked_sub(1)
                } else {
                    Some(self.pos)
                };
                let value = index
                    .and_then(|i| owner.list.get(i))
                    .map(|v| ($read)(v, py));
                if value.is_none() {
                    self.done = true;
                } else if self.reverse {
                    self.pos -= 1;
                } else {
                    self.pos += 1;
                }
                Ok(value)
            }
            fn __length_hint__(&self, py: Python<'_>) -> usize {
                if self.done {
                    return 0;
                }
                self.owner.try_borrow(py).map_or(0, |owner| {
                    if self.reverse {
                        if self.pos <= owner.list.len() {
                            self.pos
                        } else {
                            0
                        }
                    } else {
                        owner.list.len().saturating_sub(self.pos)
                    }
                })
            }
            fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
                visit.call(&self.owner)
            }
        }
    };
}

/// A list stored in pages of configurable size. Appending never moves existing
/// elements. Slices are copies. Typed variants offer zero-copy NumPy access
/// per page through ``page_view``.
#[pyclass(module = "paged_collections._core", subclass, weakref)]
pub(crate) struct PagedList {
    list: ListInner<Py<PyAny>>,
    /// Every mutation, including value replacement, invalidates an in-flight sort.
    mods: u64,
}

impl PagedList {
    fn empty() -> Self {
        Self {
            list: ListInner::default(),
            mods: 0,
        }
    }
    fn changed(&mut self) {
        self.mods = self.mods.wrapping_add(1);
    }
    fn snapshot(&self, py: Python<'_>) -> Vec<Py<PyAny>> {
        self.list.iter().map(|v| v.clone_ref(py)).collect()
    }
    fn from_values(values: Vec<Py<PyAny>>, page_size: usize) -> PyResult<Self> {
        let mut out = Self {
            list: ListInner::new(page_size)?,
            mods: 0,
        };
        out.list
            .try_reserve(values.len())
            .map_err(crate::reserve_error)?;
        out.list.extend(values);
        Ok(out)
    }
    /// Clone one element and release the borrow before any Python comparison.
    fn at(slf: &Bound<'_, Self>, i: usize) -> PyResult<Option<Py<PyAny>>> {
        Ok(slf.try_borrow()?.list.get(i).map(|v| v.clone_ref(slf.py())))
    }
    fn find(
        slf: &Bound<'_, Self>,
        value: &Bound<'_, PyAny>,
        start: usize,
        stop: usize,
    ) -> PyResult<Option<usize>> {
        let mut i = start;
        while i < stop {
            let Some(item) = Self::at(slf, i)? else {
                break;
            };
            if equal_objects(item.bind(slf.py()), value)? {
                return Ok(Some(i));
            }
            i += 1;
        }
        Ok(None)
    }
}

#[pymethods]
impl PagedList {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs), text_signature = "(values=None, *, page_size=DEFAULT_PAGE_SIZE)")]
    fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        Self::empty()
    }

    #[pyo3(signature = (values=None, *, page_size=crate::DEFAULT_PAGE_SIZE))]
    fn __init__(
        slf: &Bound<'_, Self>,
        values: Option<&Bound<'_, PyAny>>,
        #[pyo3(from_py_with = crate::extract_page_size)] page_size: usize,
    ) -> PyResult<()> {
        // As `list.__init__`: always starts from empty.
        let replacement = ListInner::new(page_size)?;
        Self::clear(slf)?;
        if slf.try_borrow()?.list.page_size() != page_size {
            // clear drops objects outside the borrow; callbacks may append.
            // Preserve those values while switching the empty storage policy.
            let values = slf.try_borrow()?.snapshot(slf.py());
            let mut this = slf.try_borrow_mut()?;
            this.list = replacement;
            this.list.extend(values);
            this.changed();
        }
        match values {
            Some(values) => Self::extend(slf, values),
            None => Ok(()),
        }
    }
    fn __len__(&self) -> usize {
        self.list.len()
    }

    fn __getitem__<'py>(
        slf: &Bound<'py, Self>,
        key: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let key = Subscript::new(key)?;
        let py = slf.py();
        match key {
            Subscript::Int(i) => {
                let this = slf.try_borrow()?;
                let i = position(i, this.list.len(), "PagedList")?;
                Ok(this.list.get(i).unwrap().bind(py).clone())
            }
            Subscript::Slice(slice) => {
                let vals = {
                    let this = slf.try_borrow()?;
                    let s = slice.indices(this.list.len() as isize)?;
                    slice_positions(&s)
                        .map(|i| this.list.get(i).unwrap().clone_ref(py))
                        .collect()
                };
                Ok(Bound::new(
                    py,
                    Self::from_values(vals, slf.try_borrow()?.list.page_size())?,
                )?
                .into_any())
            }
        }
    }

    fn __setitem__(
        slf: &Bound<'_, Self>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let key = Subscript::new(key)?;
        match key {
            Subscript::Int(i) => {
                let old = {
                    let mut this = slf.try_borrow_mut()?;
                    let i = position(i, this.list.len(), "PagedList")?;
                    let old =
                        std::mem::replace(this.list.get_mut(i).unwrap(), value.clone().unbind());
                    this.changed();
                    old
                };
                drop(old);
            }
            Subscript::Slice(slice) => {
                let vals = objects(value)?;
                let removed = {
                    let mut this = slf.try_borrow_mut()?;
                    let s = slice.indices(this.list.len() as isize)?;
                    if s.step != 1 {
                        extended_size(vals.len(), s.slicelength)?;
                    }
                    this.list
                        .try_reserve(vals.len().saturating_sub(s.slicelength))
                        .map_err(crate::reserve_error)?;
                    let removed = if s.step == 1 {
                        paged!(ListInner, &mut this.list, list => splice(list, s.start as usize, s.slicelength, vals))
                    } else {
                        slice_positions(&s)
                            .zip(vals)
                            .map(|(i, v)| std::mem::replace(this.list.get_mut(i).unwrap(), v))
                            .collect()
                    };
                    this.changed();
                    removed
                };
                drop(removed);
            }
        }
        Ok(())
    }

    fn __delitem__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<()> {
        let key = Subscript::new(key)?;
        let removed = {
            let mut this = slf.try_borrow_mut()?;
            let removed = match key {
                Subscript::Int(i) => {
                    let i = position(i, this.list.len(), "PagedList")?;
                    vec![this.list.remove(i)]
                }
                Subscript::Slice(slice) => {
                    let s = slice.indices(this.list.len() as isize)?;
                    paged!(ListInner, &mut this.list, list => remove_positions(list, slice_positions(&s).collect()))
                }
            };
            this.changed();
            removed
        };
        drop(removed);
        Ok(())
    }

    fn __iter__(slf: &Bound<'_, Self>) -> PyResult<PagedListIterator> {
        PagedListIterator::new(slf, false)
    }
    fn __reversed__(slf: &Bound<'_, Self>) -> PyResult<PagedListIterator> {
        PagedListIterator::new(slf, true)
    }
    fn __contains__(slf: &Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(Self::find(slf, value, 0, usize::MAX)?.is_some())
    }
    #[pyo3(signature = (value, start=0, stop=isize::MAX))]
    fn index(
        slf: &Bound<'_, Self>,
        value: &Bound<'_, PyAny>,
        #[pyo3(from_py_with = signed_index)] start: isize,
        #[pyo3(from_py_with = signed_index)] stop: isize,
    ) -> PyResult<usize> {
        let len = slf.try_borrow()?.list.len();
        let start = bound(start, len);
        // Positive stops are not clamped: comparisons may append, as with list.index.
        let stop = if stop < 0 {
            bound(stop, len)
        } else {
            stop as usize
        };
        Self::find(slf, value, start, stop)?.ok_or_else(|| match value.repr() {
            Ok(repr) => PyValueError::new_err(format!("{repr} is not in PagedList")),
            Err(err) => err,
        })
    }
    fn count(slf: &Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        let mut count = 0;
        let mut i = 0;
        while let Some(item) = Self::at(slf, i)? {
            count += usize::from(equal_objects(item.bind(slf.py()), value)?);
            i += 1;
        }
        Ok(count)
    }
    fn remove(slf: &Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Some(i) = Self::find(slf, value, 0, usize::MAX)? {
            let removed = {
                let mut this = slf.try_borrow_mut()?;
                // A comparison can remove the matching position itself.
                let removed = if i < this.list.len() {
                    Some(this.list.remove(i))
                } else {
                    None
                };
                this.changed();
                removed
            };
            drop(removed);
            Ok(())
        } else {
            Err(PyValueError::new_err(
                "PagedList.remove(x): x not in PagedList",
            ))
        }
    }
    fn __eq__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let Ok(other) = other.cast::<Self>() else {
            return Ok(py.NotImplemented());
        };
        if slf.try_borrow()?.list.len() != other.try_borrow()?.list.len() {
            return false.into_py_any(py);
        }
        let mut i = 0;
        loop {
            let a = Self::at(slf, i)?;
            let b = Self::at(other, i)?;
            match (a, b) {
                (Some(a), Some(b)) => {
                    if !equal_objects(a.bind(py), b.bind(py))? {
                        return false.into_py_any(py);
                    }
                }
                _ => {
                    return (slf.try_borrow()?.list.len() == other.try_borrow()?.list.len())
                        .into_py_any(py);
                }
            }
            i += 1;
        }
    }
    fn __ne__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let eq = Self::__eq__(slf, other)?;
        if eq.is(slf.py().NotImplemented()) {
            Ok(eq)
        } else {
            (!eq.bind(slf.py()).is_truthy()?).into_py_any(slf.py())
        }
    }
    fn __add__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let Ok(other) = other.cast::<Self>() else {
            return Ok(slf.py().NotImplemented());
        };
        let mut vals = slf.try_borrow()?.snapshot(slf.py());
        vals.extend(other.try_borrow()?.snapshot(slf.py()));
        Ok(Bound::new(
            slf.py(),
            Self::from_values(vals, slf.try_borrow()?.list.page_size())?,
        )?
        .into_any()
        .unbind())
    }
    fn __iadd__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::extend(slf, other)
    }
    fn __mul__(slf: &Bound<'_, Self>, n: &Bound<'_, PyAny>) -> PyResult<Self> {
        let n = strict_index(n)?;
        if n <= 0 {
            return Self::from_values(Vec::new(), slf.try_borrow()?.list.page_size());
        }
        let vals = slf.try_borrow()?.snapshot(slf.py());
        let total = repeat_len(vals.len(), n)?;
        let mut out = Self::from_values(Vec::new(), slf.try_borrow()?.list.page_size())?;
        out.list.try_reserve(total).map_err(crate::reserve_error)?;
        for _ in 0..if vals.is_empty() { 0 } else { n.max(0) } {
            out.list.extend(vals.iter().map(|v| v.clone_ref(slf.py())));
        }
        Ok(out)
    }
    fn __rmul__(slf: &Bound<'_, Self>, n: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::__mul__(slf, n)
    }
    fn __imul__(slf: &Bound<'_, Self>, n: &Bound<'_, PyAny>) -> PyResult<()> {
        let n = strict_index(n)?;
        if n <= 0 {
            return Self::clear(slf);
        }
        if n == 1 {
            return Ok(());
        }
        let py = slf.py();
        let vals = slf.try_borrow()?.snapshot(py);
        let total = repeat_len(vals.len(), n)?;
        let mut this = slf.try_borrow_mut()?;
        this.list
            .try_reserve(total - vals.len())
            .map_err(crate::reserve_error)?;
        for _ in 1..if vals.is_empty() { 1 } else { n } {
            this.list.extend(vals.iter().map(|v| v.clone_ref(py)));
        }
        this.changed();
        Ok(())
    }
    fn append(slf: &Bound<'_, Self>, value: Py<PyAny>) -> PyResult<()> {
        let mut this = slf.try_borrow_mut()?;
        this.list.try_reserve(1).map_err(crate::reserve_error)?;
        this.list.push(value);
        this.changed();
        Ok(())
    }
    fn extend(slf: &Bound<'_, Self>, iterable: &Bound<'_, PyAny>) -> PyResult<()> {
        if iterable.is(slf) {
            let vals = slf.try_borrow()?.snapshot(slf.py());
            let mut this = slf.try_borrow_mut()?;
            this.list
                .try_reserve(vals.len())
                .map_err(crate::reserve_error)?;
            this.list.extend(vals);
            this.changed();
        } else {
            // Like list.extend, a failing iterator leaves its earlier values
            // appended. The iterator is advanced only with no borrow held.
            for value in iterable.try_iter()? {
                Self::append(slf, value?.unbind())?;
            }
        }
        Ok(())
    }
    fn insert(slf: &Bound<'_, Self>, index: &Bound<'_, PyAny>, value: Py<PyAny>) -> PyResult<()> {
        let index = strict_index(index)?;
        let mut this = slf.try_borrow_mut()?;
        let i = bound(index, this.list.len());
        this.list.try_reserve(1).map_err(crate::reserve_error)?;
        this.list.insert(i, value);
        this.changed();
        Ok(())
    }
    #[pyo3(signature = (index=-1))]
    fn pop(
        slf: &Bound<'_, Self>,
        #[pyo3(from_py_with = signed_index)] index: isize,
    ) -> PyResult<Py<PyAny>> {
        let i = index;
        let mut this = slf.try_borrow_mut()?;
        if this.list.is_empty() {
            return Err(PyIndexError::new_err("pop from empty PagedList"));
        }
        let i = position(i, this.list.len(), "PagedList")
            .map_err(|_| PyIndexError::new_err("pop index out of range"))?;
        let value = this.list.remove(i);
        this.changed();
        Ok(value)
    }
    fn clear(slf: &Bound<'_, Self>) -> PyResult<()> {
        let removed: Vec<_> = {
            let mut this = slf.try_borrow_mut()?;
            let removed = std::iter::from_fn(|| this.list.pop()).collect();
            this.changed();
            removed
        };
        drop(removed);
        Ok(())
    }
    fn reverse(&mut self) {
        self.list.reverse();
        self.changed();
    }
    fn copy(&self, py: Python<'_>) -> PyResult<Self> {
        Self::from_values(self.snapshot(py), self.list.page_size())
    }
    #[pyo3(signature = (*, key=None, reverse=false))]
    fn sort(
        slf: &Bound<'_, Self>,
        key: Option<&Bound<'_, PyAny>>,
        #[pyo3(from_py_with = sort_reverse)] reverse: bool,
    ) -> PyResult<()> {
        let py = slf.py();
        let (vals, mods) = {
            let this = slf.try_borrow()?;
            (this.snapshot(py), this.mods)
        };
        let work = PyList::new(py, vals)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("key", key)?;
        kwargs.set_item("reverse", reverse)?;
        work.call_method("sort", (), Some(&kwargs))?;
        let vals = objects(work.as_any())?;
        let removed = {
            let mut this = slf.try_borrow_mut()?;
            if this.mods != mods {
                return Err(PyValueError::new_err("PagedList modified during sort"));
            }
            let removed: Vec<_> = vals
                .into_iter()
                .enumerate()
                .map(|(i, v)| std::mem::replace(this.list.get_mut(i).unwrap(), v))
                .collect();
            this.changed();
            removed
        };
        drop(removed);
        Ok(())
    }
    fn reserve(slf: &Bound<'_, Self>, additional: &Bound<'_, PyAny>) -> PyResult<()> {
        // Convert before borrowing: `__index__` can run Python code.
        let additional = additional.extract::<usize>()?;
        slf.try_borrow_mut()?
            .list
            .try_reserve(additional)
            .map_err(crate::reserve_error)
    }
    fn shrink_to_fit(&mut self) {
        self.list.shrink_to_fit();
        self.changed();
    }
    #[getter]
    fn capacity(&self) -> usize {
        self.list.capacity()
    }
    #[getter]
    fn allocated_pages(&self) -> usize {
        self.list.allocated_pages()
    }
    #[classattr]
    fn dtype() -> &'static str {
        "object"
    }
    #[getter]
    fn page_size(&self) -> usize {
        self.list.page_size()
    }
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let name = slf.get_type().name()?.to_string();
        crate::repr_guarded(slf.as_any(), &name, || {
            let (vals, more) = {
                let this = slf.try_borrow()?;
                (
                    this.list
                        .iter()
                        .take(crate::REPR_ITEMS)
                        .map(|v| v.clone_ref(slf.py()))
                        .collect::<Vec<_>>(),
                    this.list.len() > crate::REPR_ITEMS,
                )
            };
            let mut parts = vals
                .iter()
                .map(|v| Ok(v.bind(slf.py()).repr()?.to_string()))
                .collect::<PyResult<Vec<_>>>()?;
            if more {
                parts.push("...".to_owned());
            }
            Ok(format!(
                "{name}([{}]{})",
                parts.join(", "),
                crate::page_size_repr(slf.try_borrow()?.list.page_size())
            ))
        })
    }
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let vals = slf.try_borrow()?.snapshot(slf.py());
        let items = PyList::new(slf.py(), vals)?;
        crate::reduce_list(
            slf.as_any(),
            &slf.py().get_type::<Self>(),
            items.try_iter()?,
        )
    }
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        for value in self.list.iter() {
            visit.call(value)?;
        }
        Ok(())
    }
    fn __clear__(slf: &Bound<'_, Self>) -> PyResult<()> {
        Self::clear(slf)
    }
}

list_iterator!(
    PagedListIterator,
    PagedList,
    Py<PyAny>,
    |v: &Py<PyAny>, py: Python<'_>| v.clone_ref(py)
);

/// NumPy's floating sort order: NaNs after numbers; signed zeros compare equal.
/// For all other supported values partial_cmp is a total order.
fn numeric_order<T: PartialOrd>(a: T, b: T) -> Ordering {
    a.partial_cmp(&b).unwrap_or_else(|| {
        match (a.partial_cmp(&a).is_none(), b.partial_cmp(&b).is_none()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            _ => Ordering::Less,
        }
    })
}

macro_rules! typed_list {
    ($name:ident, $iter:ident, $pyname:literal, $t:ty, $writeable:literal) => {
        #[doc = concat!("A list of ``", stringify!($t), "`` stored in pages of configurable size. ",
            "Appending never moves existing elements; slices are copies. ",
            "Zero-copy NumPy access is per page via ``page_view``. While views exist, ",
            "appending and value writes are allowed; removing or moving elements raises BufferError.")]
        #[pyclass(module = "paged_collections._core", name = $pyname, subclass)]
        pub(crate) struct $name {
            list: ListInner<ValueCell<$t>>,
            exports: Exports,
        }
        impl $name {
            fn empty() -> Self { Self { list: ListInner::default(), exports: Exports::default() } }
            fn from_values(vals: Vec<$t>, page_size: usize) -> PyResult<Self> {
                let mut out = Self { list: ListInner::new(page_size)?, exports: Exports::default() };
                out.extend_values(vals)?;
                Ok(out)
            }
            fn extend_values(&mut self, vals: Vec<$t>) -> PyResult<()> {
                self.list.try_reserve(vals.len()).map_err(crate::reserve_error)?;
                // push only touches the new last element; existing cells stay put.
                for value in vals { self.list.push(ValueCell::new(value)); }
                Ok(())
            }
            fn snapshot(&self) -> Vec<$t> { self.list.iter().map(ValueCell::get).collect() }
            fn find(&self, value: $t, start: usize, stop: usize) -> Option<usize> {
                (start..stop.min(self.list.len())).find(|&i| self.list.get(i).unwrap().get() == value)
            }
        }
        #[pymethods]
        impl $name {
            #[new]
            #[pyo3(signature = (*_args, **_kwargs), text_signature = "(values=None, *, page_size=DEFAULT_PAGE_SIZE)")]
            fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self { Self::empty() }
            #[pyo3(signature = (values=None, *, page_size=crate::DEFAULT_PAGE_SIZE))]
            fn __init__(slf: &Bound<'_, Self>, values: Option<&Bound<'_, PyAny>>, #[pyo3(from_py_with = crate::extract_page_size)] page_size: usize) -> PyResult<()> {
                // As `list.__init__`: always starts from empty.
                let vals = match values {
                    Some(values) => typed_values::<$t>(values)?,
                    None => Vec::new(),
                };
                {
                    let mut this = slf.try_borrow_mut()?;
                    if !this.list.is_empty() || this.list.page_size() != page_size { this.exports.check($pyname)?; }
                    if this.list.page_size() != page_size { this.list = ListInner::new(page_size)?; }
                    let additional = vals.len().saturating_sub(this.list.len());
                    this.list.try_reserve(additional).map_err(crate::reserve_error)?;
                    this.list.clear();
                    this.extend_values(vals)?;
                }
                Ok(())
            }
            fn __len__(&self) -> usize { self.list.len() }
            fn __getitem__<'py>(slf: &Bound<'py, Self>, key: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
                let key = Subscript::new(key)?;
                match key {
                    Subscript::Int(i) => {
                        let value = { let this = slf.try_borrow()?;
                            this.list.get(position(i, this.list.len(), $pyname)?).unwrap().get() };
                        value.into_bound_py_any(slf.py())
                    }
                    Subscript::Slice(slice) => {
                        let vals = { let this = slf.try_borrow()?;
                            let s = slice.indices(this.list.len() as isize)?;
                            slice_positions(&s).map(|i| this.list.get(i).unwrap().get()).collect() };
                        Ok(Bound::new(slf.py(), Self::from_values(vals, slf.try_borrow()?.list.page_size())?)?.into_any())
                    }
                }
            }
            fn __setitem__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
                let key = Subscript::new(key)?;
                match key {
                    Subscript::Int(i) => {
                        let value = value.extract::<$t>()?;
                        let this = slf.try_borrow()?;
                        this.list.get(position(i, this.list.len(), $pyname)?).unwrap().set(value);
                    }
                    Subscript::Slice(slice) => {
                        let vals = typed_values::<$t>(value)?;
                        let mut this = slf.try_borrow_mut()?;
                        let s = slice.indices(this.list.len() as isize)?;
                        if s.step != 1 { extended_size(vals.len(), s.slicelength)?; }
                        if vals.len() == s.slicelength {
                            for (i, value) in slice_positions(&s).zip(vals) { this.list.get(i).unwrap().set(value); }
                        } else {
                            this.exports.check($pyname)?;
                            this.list.try_reserve(vals.len().saturating_sub(s.slicelength)).map_err(crate::reserve_error)?;
                            let vals = vals.into_iter().map(ValueCell::new).collect();
                            paged!(ListInner, &mut this.list, list => splice(list, s.start as usize, s.slicelength, vals));
                        }
                    }
                }
                Ok(())
            }
            fn __delitem__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<()> {
                let key = Subscript::new(key)?;
                let mut this = slf.try_borrow_mut()?;
                this.exports.check($pyname)?;
                match key {
                    Subscript::Int(i) => {
                        let i = position(i, this.list.len(), $pyname)?;
                        this.list.remove(i);
                    }
                    Subscript::Slice(slice) => {
                        let s = slice.indices(this.list.len() as isize)?;
                        paged!(ListInner, &mut this.list, list => remove_positions(list, slice_positions(&s).collect()));
                    }
                }
                Ok(())
            }
            fn __iter__(slf: &Bound<'_, Self>) -> PyResult<$iter> { $iter::new(slf, false) }
            fn __reversed__(slf: &Bound<'_, Self>) -> PyResult<$iter> { $iter::new(slf, true) }
            fn __contains__(slf: &Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
                let Ok(value) = value.extract::<$t>() else { return Ok(false); };
                let this = slf.try_borrow()?;
                Ok(this.find(value, 0, this.list.len()).is_some())
            }
            #[pyo3(signature = (value, start=0, stop=isize::MAX))]
            fn index(slf: &Bound<'_, Self>, value: &Bound<'_, PyAny>,
                #[pyo3(from_py_with = signed_index)] start: isize,
                #[pyo3(from_py_with = signed_index)] stop: isize) -> PyResult<usize> {
                let found = match value.extract::<$t>() {
                    Ok(v) => { let this = slf.try_borrow()?;
                        this.find(v, bound(start, this.list.len()), bound(stop, this.list.len())) },
                    Err(_) => None,
                };
                match found { Some(i) => Ok(i), None => Err(PyValueError::new_err(format!("{} is not in {}", value.repr()?, $pyname))) }
            }
            fn count(slf: &Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<usize> {
                let Ok(value) = value.extract::<$t>() else { return Ok(0); };
                Ok(slf.try_borrow()?.list.iter().filter(|v| v.get() == value).count())
            }
            fn remove(slf: &Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<()> {
                let value = value.extract::<$t>().ok();
                let mut this = slf.try_borrow_mut()?;
                this.exports.check($pyname)?;
                let found = value.and_then(|v| this.find(v, 0, this.list.len()));
                match found {
                    Some(i) => { this.list.remove(i); Ok(()) },
                    None => Err(PyValueError::new_err(concat!($pyname, ".remove(x): x not in ", $pyname))),
                }
            }
            fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                let Ok(other) = other.cast::<Self>() else { return Ok(py.NotImplemented()); };
                let other = other.try_borrow()?;
                (self.list.len() == other.list.len() && self.list.iter().zip(other.list.iter()).all(|(a,b)| a.get() == b.get())).into_py_any(py)
            }
            fn __ne__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                let eq = self.__eq__(py, other)?;
                if eq.is(py.NotImplemented()) { Ok(eq) } else { (!eq.bind(py).is_truthy()?).into_py_any(py) }
            }
            fn __add__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                let Ok(other) = other.cast::<Self>() else { return Ok(slf.py().NotImplemented()); };
                let mut vals = slf.try_borrow()?.snapshot();
                vals.extend(other.try_borrow()?.snapshot());
                Ok(Bound::new(slf.py(), Self::from_values(vals, slf.try_borrow()?.list.page_size())?)?.into_any().unbind())
            }
            fn __iadd__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<()> { Self::extend(slf, other) }
            fn __mul__(slf: &Bound<'_, Self>, n: &Bound<'_, PyAny>) -> PyResult<Self> {
                let n = strict_index(n)?;
                if n <= 0 { return Self::from_values(Vec::new(), slf.try_borrow()?.list.page_size()); }
                let vals = slf.try_borrow()?.snapshot();
                let total = repeat_len(vals.len(), n)?;
                let mut out = Self::from_values(Vec::new(), slf.try_borrow()?.list.page_size())?;
                out.list.try_reserve(total).map_err(crate::reserve_error)?;
                for _ in 0..if vals.is_empty() { 0 } else { n.max(0) } {
                    for &v in &vals { out.list.push(ValueCell::new(v)); }
                }
                Ok(out)
            }
            fn __rmul__(slf: &Bound<'_, Self>, n: &Bound<'_, PyAny>) -> PyResult<Self> { Self::__mul__(slf, n) }
            fn __imul__(slf: &Bound<'_, Self>, n: &Bound<'_, PyAny>) -> PyResult<()> {
                let n = strict_index(n)?;
                let mut this = slf.try_borrow_mut()?;
                if n <= 0 { this.exports.check($pyname)?; this.list.clear(); return Ok(()); }
                if n == 1 { return Ok(()); }
                let vals = this.snapshot();
                let total = repeat_len(vals.len(), n)?;
                this.list.try_reserve(total - vals.len()).map_err(crate::reserve_error)?;
                for _ in 1..if vals.is_empty() { 1 } else { n } {
                    for &v in &vals { this.list.push(ValueCell::new(v)); }
                }
                Ok(())
            }
            fn append(slf: &Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<()> {
                let value = value.extract::<$t>()?;
                slf.try_borrow_mut()?.extend_values(vec![value])
            }
            fn extend(slf: &Bound<'_, Self>, iterable: &Bound<'_, PyAny>) -> PyResult<()> {
                let vals = typed_values::<$t>(iterable)?;
                slf.try_borrow_mut()?.extend_values(vals)
            }
            fn insert(slf: &Bound<'_, Self>, index: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
                let i = strict_index(index)?;
                let value = value.extract::<$t>()?;
                let mut this = slf.try_borrow_mut()?;
                let i = bound(i, this.list.len());
                if i < this.list.len() { this.exports.check($pyname)?; }
                this.list.try_reserve(1).map_err(crate::reserve_error)?;
                if i == this.list.len() { this.list.push(ValueCell::new(value)); }
                else { this.list.insert(i, ValueCell::new(value)); }
                Ok(())
            }
            #[pyo3(signature = (index=-1))]
            fn pop(slf: &Bound<'_, Self>, #[pyo3(from_py_with = signed_index)] index: isize) -> PyResult<$t> {
                let i = index;
                let mut this = slf.try_borrow_mut()?;
                this.exports.check($pyname)?;
                if this.list.is_empty() { return Err(PyIndexError::new_err(concat!("pop from empty ", $pyname))); }
                let i = position(i, this.list.len(), $pyname).map_err(|_| PyIndexError::new_err("pop index out of range"))?;
                Ok(this.list.remove(i).get())
            }
            fn clear(&mut self) -> PyResult<()> { self.exports.check($pyname)?; self.list.clear(); Ok(()) }
            fn reverse(&mut self) -> PyResult<()> { self.exports.check($pyname)?; self.list.reverse(); Ok(()) }
            fn copy(&self) -> PyResult<Self> { Self::from_values(self.snapshot(), self.list.page_size()) }
            #[pyo3(signature = (*, reverse=false))]
            fn sort(slf: &Bound<'_, Self>, #[pyo3(from_py_with = sort_reverse)] reverse: bool) -> PyResult<()> {
                let mut this = slf.try_borrow_mut()?;
                this.exports.check($pyname)?;
                this.list.sort_by(|a,b| if reverse { numeric_order(b.get(), a.get()) } else { numeric_order(a.get(), b.get()) });
                Ok(())
            }
            fn reserve(slf: &Bound<'_, Self>, additional: &Bound<'_, PyAny>) -> PyResult<()> {
                // Convert before borrowing: `__index__` can run Python code.
                let additional = additional.extract::<usize>()?;
                slf.try_borrow_mut()?.list.try_reserve(additional).map_err(crate::reserve_error)
            }
            fn shrink_to_fit(&mut self) -> PyResult<()> { self.exports.check($pyname)?; self.list.shrink_to_fit(); Ok(()) }
            #[getter]
            fn capacity(&self) -> usize { self.list.capacity() }
            #[getter]
            fn allocated_pages(&self) -> usize { self.list.allocated_pages() }
            #[classattr]
            fn dtype() -> &'static str { <$t as TypedValue>::DTYPE }
            #[getter]
            fn page_size(&self) -> usize { self.list.page_size() }
            fn tolist(&self) -> Vec<$t> { self.snapshot() }
            fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<$t>> { PyArray1::from_vec(py, self.snapshot()) }
            /// NumPy conversion (`numpy.asarray(lst)`): always a copy, since
            /// the pages are not contiguous.
            #[pyo3(signature = (dtype=None, copy=None))]
            fn __array__<'py>(slf: &Bound<'py, Self>, dtype: Option<&Bound<'py, PyAny>>, copy: Option<bool>) -> PyResult<Bound<'py, PyAny>> {
                let py = slf.py();
                if copy == Some(false) {
                    return Err(PyValueError::new_err(concat!(
                        "a ", $pyname, " cannot be converted to a NumPy array without copying"
                    )));
                }
                // Release the borrow before `astype`, which can run Python code.
                let array = slf.try_borrow()?.to_numpy(py).into_any();
                match dtype {
                    Some(dtype) if !dtype.is_none() => array.call_method1("astype", (dtype,)),
                    _ => Ok(array),
                }
            }
            fn get_many<'py>(slf: &Bound<'py, Self>, indices: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyArray1<$t>>> {
                let indices = index_batch(indices)?;
                let vals = { let this = slf.try_borrow()?;
                    normalized_batch(&indices, this.list.len(), $pyname)?.into_iter().map(|i| this.list.get(i).unwrap().get()).collect() };
                Ok(PyArray1::from_vec(slf.py(), vals))
            }
            fn set_many(slf: &Bound<'_, Self>, indices: &Bound<'_, PyAny>, values: &Bound<'_, PyAny>) -> PyResult<()> {
                let indices = index_batch(indices)?;
                let vals = value_batch::<$t>(values, indices.len())?;
                let this = slf.try_borrow()?;
                let indices = normalized_batch(&indices, this.list.len(), $pyname)?;
                for (i,v) in indices.into_iter().zip(vals) { this.list.get(i).unwrap().set(v); }
                Ok(())
            }
            fn page_view<'py>(slf: &Bound<'py, Self>, page: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyArray1<$t>>> {
                let page = signed_index(page)?;
                let this = slf.try_borrow()?;
                let page = position(page, this.list.pages().len(), $pyname)?;
                let data = this.list.pages().nth(page).unwrap();
                // SAFETY: data is the current prefix of an owner-held page Vec.
                // Its capacity is >= this instance's page_size and never grows beyond it, so
                // appends/reserve cannot reallocate or move existing cells.
                // Every removal/reorder/free checks exports; all value writes
                // use ValueCell::set, never mutable references into the prefix.
                unsafe { cell_view(slf.as_any(), &this.exports, data, $writeable) }
            }
            fn page_views<'py>(slf: &Bound<'py, Self>) -> PyResult<Vec<Bound<'py, PyArray1<$t>>>> {
                let count = slf.try_borrow()?.list.pages().len();
                (0..count).map(|i| Self::page_view(slf, &i.into_bound_py_any(slf.py())?)).collect()
            }
            fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
                let vals = { let this = slf.try_borrow()?;
                    this.list.iter().take(crate::REPR_ITEMS + 1).map(ValueCell::get).collect::<Vec<_>>() };
                let parts = vals.into_iter().map(|v| Ok(v.into_bound_py_any(slf.py())?.repr()?.to_string())).collect::<PyResult<Vec<_>>>()?;
                Ok(format!("{}([{}]{})", slf.get_type().name()?, crate::truncated(parts.into_iter()), crate::page_size_repr(slf.try_borrow()?.list.page_size())))
            }
            fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
                let contents = slf.try_borrow()?.to_numpy(slf.py()).into_any();
                crate::reduce(slf.as_any(), &slf.py().get_type::<Self>(), vec![contents], None)
            }
        }
        list_iterator!($iter, $name, $t, |v: &ValueCell<$t>, _py: Python<'_>| v.get());
    };
}

typed_list!(
    PagedListF64,
    PagedListF64Iterator,
    "PagedListF64",
    f64,
    true
);
typed_list!(
    PagedListF32,
    PagedListF32Iterator,
    "PagedListF32",
    f32,
    true
);
typed_list!(
    PagedListI64,
    PagedListI64Iterator,
    "PagedListI64",
    i64,
    true
);
typed_list!(
    PagedListI32,
    PagedListI32Iterator,
    "PagedListI32",
    i32,
    true
);
typed_list!(
    PagedListU64,
    PagedListU64Iterator,
    "PagedListU64",
    u64,
    true
);
typed_list!(
    PagedListU32,
    PagedListU32Iterator,
    "PagedListU32",
    u32,
    true
);
typed_list!(
    PagedListBool,
    PagedListBoolIterator,
    "PagedListBool",
    bool,
    false
);
