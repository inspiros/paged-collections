//! Python extension module `paged_collections._core`.
//!
//! Classes are concrete (monomorphized) wrappers over the Rust collections,
//! because Rust generics cannot be exposed to Python directly. Keys are
//! integers in `0..2**32` (`u32`, flat page directory, so the directory
//! never exceeds 32 MiB).

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyIterator, PyTuple, PyType};

include!(concat!(env!("OUT_DIR"), "/page_sizes.rs"));

mod concurrent;
mod convert;
mod deque;
mod direct_map;
mod export;
mod hive;
mod id_alloc;
mod iter;
mod object_map;
mod paged_list;
mod priority_queue;
mod set;
mod storage;
mod typed_map;

/// Creates an instance of `cls`, a subclass of `base` (one of this module's
/// classes), and fills it with `base.__init__(obj, *args)`. Pickles use it,
/// so a subclass's own `__init__` is never called with the base's
/// arguments, as `copyreg.__reduce_ex__` does for `dict` subclasses.
#[pyfunction]
#[pyo3(signature = (cls, base, *args, **kwargs))]
fn _rebuild<'py>(
    cls: &Bound<'py, PyType>,
    base: &Bound<'py, PyType>,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    if !cls.is_subclass(base)? {
        return Err(PyTypeError::new_err(format!(
            "{} is not a subclass of {}",
            cls.name()?,
            base.name()?
        )));
    }
    // A pickle without a size predates `page_size`, when every paged class
    // used 1024. Reading it at another build's default would misplace Hive
    // generation floors and could revive stale ids, so ask for 1024 itself.
    let legacy;
    let kwargs = match kwargs {
        None if base.getattr_opt("page_size")?.is_some() => {
            if !PAGE_SIZES.contains(&LEGACY_PAGE_SIZE) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "this pickle predates page_size and needs page size {LEGACY_PAGE_SIZE}, \
                     which this build lacks; rebuild with it in PAGED_COLLECTIONS_PAGE_SIZES"
                )));
            }
            legacy = PyDict::new(cls.py());
            legacy.set_item("page_size", LEGACY_PAGE_SIZE)?;
            Some(&legacy)
        }
        kwargs => kwargs,
    };
    let obj = cls.call_method1("__new__", (cls,))?;
    let init_args: Vec<_> = std::iter::once(obj.clone()).chain(args.iter()).collect();
    let init_args = PyTuple::new(cls.py(), init_args)?;
    base.getattr("__init__")?.call(init_args, kwargs)?;
    Ok(obj)
}

/// Keyword-aware pickle entry point. The final positional value is the size,
/// keeping the contents at their original offsets in the reduce arguments.
#[pyfunction]
#[pyo3(signature = (cls, base, *args))]
fn _rebuild_paged<'py>(
    cls: &Bound<'py, PyType>,
    base: &Bound<'py, PyType>,
    args: &Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyAny>> {
    let count = args
        .len()
        .checked_sub(1)
        .ok_or_else(|| PyTypeError::new_err("missing pickle page_size"))?;
    let page_size = extract_page_size(&args.get_item(count)?)?;
    let kwargs = PyDict::new(cls.py());
    kwargs.set_item("page_size", page_size)?;
    let contents = PyTuple::new(cls.py(), args.iter().take(count))?;
    _rebuild(cls, base, &contents, Some(&kwargs))
}

/// The page size of every paged class before `page_size` was selectable.
const LEGACY_PAGE_SIZE: usize = 1024;

/// Error for a page size not compiled into the extension.
fn page_size_error(page_size: impl std::fmt::Display) -> PyErr {
    let sizes = PAGE_SIZES
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    pyo3::exceptions::PyValueError::new_err(format!(
        "page_size must be one of ({sizes}), got {page_size}; rebuild with PAGED_COLLECTIONS_PAGE_SIZES to add sizes"
    ))
}

/// Extract and validate an instance size; all unsupported ints
/// (including negative and arbitrarily large ones) raise the same ValueError.
fn extract_page_size(value: &Bound<'_, PyAny>) -> PyResult<usize> {
    let value = value.cast::<pyo3::types::PyInt>()?;
    // Normalize int subclasses without invoking their formatting overrides.
    let integer = value
        .py()
        .import("operator")?
        .call_method1("index", (value,))?;
    if let Ok(size) = integer.extract::<usize>() {
        if PAGE_SIZES.contains(&size) {
            return Ok(size);
        }
        return Err(page_size_error(size));
    }
    Err(page_size_error(integer))
}

/// Optional keyword in a collection's call-style repr.
fn page_size_repr(page_size: usize) -> String {
    if page_size == DEFAULT_PAGE_SIZE {
        String::new()
    } else {
        format!(", page_size={page_size}")
    }
}

/// The `__reduce__` value of `obj`, an instance of `base` or a subclass:
/// rebuilt with [`_rebuild`] from `contents`, then given the extra state of
/// a subclass instance (its `__getstate__()` on Python 3.11+, else its
/// `__dict__`), and, if given, the `(key, value)` pairs of `items`.
fn reduce<'py>(
    obj: &Bound<'py, PyAny>,
    base: &Bound<'py, PyType>,
    contents: Vec<Bound<'py, PyAny>>,
    items: Option<Bound<'py, PyIterator>>,
) -> PyResult<Bound<'py, PyTuple>> {
    let py = obj.py();
    let module = py.import("paged_collections._core")?;
    let page_size = base.getattr_opt("page_size")?;
    let rebuild = module.getattr(if page_size.is_some() {
        "_rebuild_paged"
    } else {
        "_rebuild"
    })?;
    let cls = obj.get_type();
    let mut args = vec![cls.clone().into_any(), base.clone().into_any()];
    args.extend(contents);
    if let Some(descriptor) = page_size {
        // Read the base descriptor, bypassing subclass property overrides.
        args.push(descriptor.call_method1("__get__", (obj, base))?);
    }
    let args = PyTuple::new(py, args)?;
    let state = if cls.is(base) {
        None
    } else {
        extra_state(obj)?
    };
    let none = || py.None().into_bound(py);
    let mut parts = vec![rebuild, args.into_any(), state.unwrap_or_else(none)];
    if let Some(items) = items {
        parts.extend([none(), items.into_any()]);
    }
    PyTuple::new(py, parts)
}

/// List-item form of `reduce`: create the empty list before restoring its
/// elements, allowing pickle/deepcopy to memoize self-containing lists. Map
/// reduce tuples keep their existing dictionary-item form.
fn reduce_list<'py>(
    obj: &Bound<'py, PyAny>,
    base: &Bound<'py, PyType>,
    items: Bound<'py, PyIterator>,
) -> PyResult<Bound<'py, PyTuple>> {
    let initial = reduce(obj, base, Vec::new(), None)?;
    let mut parts: Vec<_> = initial.iter().collect();
    parts.push(items.into_any());
    PyTuple::new(obj.py(), parts)
}

/// The state of a subclass instance beyond its contents, or `None` if it
/// has none: `__getstate__()` where defined (Python 3.11+ defines it for
/// every object, covering `__slots__`), else what Python 3.10's
/// `object.__reduce_ex__` builds: the instance `__dict__`, or
/// `(__dict__ or None, {slot: value})` when set slots exist. Pickle and
/// `copy` restore both forms without a `__setstate__`.
fn extra_state<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    let state = match obj.getattr_opt("__getstate__")? {
        Some(getstate) => getstate.call0()?,
        None => {
            let py = obj.py();
            let dict = obj.getattr_opt("__dict__")?;
            let names = py
                .import("copyreg")?
                .call_method1("_slotnames", (obj.get_type(),))?;
            let slots = PyDict::new(py);
            for name in names.try_iter()? {
                let name = name?.cast_into::<pyo3::types::PyString>()?;
                if let Some(value) = obj.getattr_opt(&name)? {
                    slots.set_item(name, value)?;
                }
            }
            match (dict, slots.is_empty()) {
                (Some(dict), true) => dict,
                (None, true) => return Ok(None),
                (dict, false) => {
                    let dict = dict.unwrap_or_else(|| py.None().into_bound(py));
                    PyTuple::new(py, [dict, slots.into_any()])?.into_any()
                }
            }
        }
    };
    let empty = state.is_none() || state.cast::<PyDict>().is_ok_and(|d| d.is_empty());
    Ok((!empty).then_some(state))
}

/// The Python error for a failed reservation.
fn reserve_error(err: std::collections::TryReserveError) -> PyErr {
    pyo3::exceptions::PyMemoryError::new_err(err.to_string())
}

/// `f()`, or `"{name}(...)"` if a `repr` of `obj` is already in progress on
/// this thread (a collection that contains itself), as `dict` does.
fn repr_guarded(
    obj: &Bound<'_, PyAny>,
    name: &str,
    f: impl FnOnce() -> PyResult<String>,
) -> PyResult<String> {
    repr_guarded_with(obj, || format!("{name}(...)"), f)
}

/// A guarded repr with a collection-specific recursion placeholder.
fn repr_guarded_with(
    obj: &Bound<'_, PyAny>,
    recursive: impl FnOnce() -> String,
    f: impl FnOnce() -> PyResult<String>,
) -> PyResult<String> {
    /// Leaves the repr of `obj` on drop, also when `f` unwinds.
    struct Leave<'a, 'py>(&'a Bound<'py, PyAny>);
    impl Drop for Leave<'_, '_> {
        fn drop(&mut self) {
            // SAFETY: the object is live, the thread is attached, and this
            // balances the successful `Py_ReprEnter` below.
            unsafe { pyo3::ffi::Py_ReprLeave(self.0.as_ptr()) };
        }
    }

    // SAFETY: `obj` is a live object and the thread is attached.
    match unsafe { pyo3::ffi::Py_ReprEnter(obj.as_ptr()) } {
        0 => {
            let _leave = Leave(obj);
            f()
        }
        n if n > 0 => Ok(recursive()),
        _ => Err(PyErr::fetch(obj.py())),
    }
}

/// Most items shown by `repr` before it ends with `...`.
const REPR_ITEMS: usize = 20;

/// Joins `items` with `", "`, keeping at most [`REPR_ITEMS`] of them.
fn truncated(items: impl ExactSizeIterator<Item = String>) -> String {
    let more = items.len() > REPR_ITEMS;
    let mut out: Vec<String> = items.take(REPR_ITEMS).collect();
    if more {
        out.push("...".to_owned());
    }
    out.join(", ")
}

/// `key: repr(value)` for each entry, truncated as by [`truncated`].
fn repr_items<'py, K: std::fmt::Display, V: IntoPyObject<'py>>(
    entries: impl ExactSizeIterator<Item = (K, V)>,
    py: Python<'py>,
) -> PyResult<String> {
    let more = entries.len() > REPR_ITEMS;
    let mut out = Vec::with_capacity(entries.len().min(REPR_ITEMS + 1));
    for (key, value) in entries.take(REPR_ITEMS) {
        out.push(format!("{key}: {}", value.into_bound_py_any(py)?.repr()?));
    }
    if more {
        out.push("...".to_owned());
    }
    Ok(out.join(", "))
}

/// Native core of the `paged_collections` Python package.
///
/// The module runs without the GIL on free-threaded builds: every class is
/// `Sync`, PyO3's borrow flags serialize `&mut self` methods against other
/// calls on the same object, and no state is shared between objects except
/// through atomics and locks.
#[pymodule(gil_used = false)]
mod _core {
    use super::*;

    #[pymodule_export]
    use super::{_rebuild, _rebuild_paged};
    #[pymodule_export]
    use crate::concurrent::{
        ConcurrentDirectMapBool, ConcurrentDirectMapF32, ConcurrentDirectMapF64,
        ConcurrentDirectMapI32, ConcurrentDirectMapI64, ConcurrentDirectMapU32,
        ConcurrentDirectMapU64, ConcurrentSparseMapBool, ConcurrentSparseMapF32,
        ConcurrentSparseMapF64, ConcurrentSparseMapI32, ConcurrentSparseMapI64,
        ConcurrentSparseMapU32, ConcurrentSparseMapU64, ConcurrentSparseSet,
    };
    #[pymodule_export]
    use crate::deque::{PagedDeque, PagedDequeIterator};
    #[pymodule_export]
    use crate::direct_map::{
        DirectMap, DirectMapBool, DirectMapBoolIterator, DirectMapF32, DirectMapF32Iterator,
        DirectMapF64, DirectMapF64Iterator, DirectMapI32, DirectMapI32Iterator, DirectMapI64,
        DirectMapI64Iterator, DirectMapIterator, DirectMapU32, DirectMapU32Iterator, DirectMapU64,
        DirectMapU64Iterator,
    };
    #[pymodule_export]
    use crate::export::ExportGuard;
    #[pymodule_export]
    use crate::hive::{Hive, HiveIterator};
    #[pymodule_export]
    use crate::id_alloc::{
        IdAllocator, IdAllocatorIterator, id_generation, id_index, id_indices, make_id,
    };
    #[pymodule_export]
    use crate::object_map::{IdMap, IdMapIterator, SparseMap, SparseMapIterator};
    #[pymodule_export]
    use crate::paged_list::{
        PagedList, PagedListBool, PagedListBoolIterator, PagedListF32, PagedListF32Iterator,
        PagedListF64, PagedListF64Iterator, PagedListI32, PagedListI32Iterator, PagedListI64,
        PagedListI64Iterator, PagedListIterator, PagedListU32, PagedListU32Iterator, PagedListU64,
        PagedListU64Iterator,
    };
    #[pymodule_export]
    use crate::priority_queue::{PriorityQueue, PriorityQueueF64};
    #[pymodule_export]
    use crate::set::{SparseSet, SparseSetIterator};
    #[pymodule_export]
    use crate::typed_map::{
        IdMapBool, IdMapBoolIterator, IdMapF32, IdMapF32Iterator, IdMapF64, IdMapF64Iterator,
        IdMapI32, IdMapI32Iterator, IdMapI64, IdMapI64Iterator, IdMapU32, IdMapU32Iterator,
        IdMapU64, IdMapU64Iterator,
    };
    #[pymodule_export]
    use crate::typed_map::{
        SparseMapBool, SparseMapBoolIterator, SparseMapF32, SparseMapF32Iterator, SparseMapF64,
        SparseMapF64Iterator, SparseMapI32, SparseMapI32Iterator, SparseMapI64,
        SparseMapI64Iterator, SparseMapU32, SparseMapU32Iterator, SparseMapU64,
        SparseMapU64Iterator,
    };

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add("__version__", env!("CARGO_PKG_VERSION"))?;
        m.add(
            "PAGE_SIZES",
            PyTuple::new(m.py(), PAGE_SIZES.iter().copied())?,
        )?;
        m.add("DEFAULT_PAGE_SIZE", DEFAULT_PAGE_SIZE)?;
        m.add(
            "MAX_GENERATION",
            paged_collections::id_alloc::MAX_GENERATION,
        )?;
        m.add("MAX_INDICES", paged_collections::id_alloc::MAX_INDICES)?;
        m.add("KEY_LIMIT", 1_u64 << 32)
    }
}
