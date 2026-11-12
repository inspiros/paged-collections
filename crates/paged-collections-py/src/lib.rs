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

mod convert;
mod direct_map;
mod export;
mod iter;
mod object_map;
mod set;
mod typed_map;

/// Creates an instance of `cls`, a subclass of `base` (one of this module's
/// classes), and fills it with `base.__init__(obj, *args)`. Pickles use it,
/// so a subclass's own `__init__` is never called with the base's
/// arguments, as `copyreg.__reduce_ex__` does for `dict` subclasses.
#[pyfunction]
#[pyo3(signature = (cls, base, *args))]
fn _rebuild<'py>(
    cls: &Bound<'py, PyType>,
    base: &Bound<'py, PyType>,
    args: &Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyAny>> {
    if !cls.is_subclass(base)? {
        return Err(PyTypeError::new_err(format!(
            "{} is not a subclass of {}",
            cls.name()?,
            base.name()?
        )));
    }
    let obj = cls.call_method1("__new__", (cls,))?;
    let init_args: Vec<_> = std::iter::once(obj.clone()).chain(args.iter()).collect();
    let init_args = PyTuple::new(cls.py(), init_args)?;
    base.getattr("__init__")?.call1(init_args)?;
    Ok(obj)
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
    let rebuild = py.import("paged_collections._core")?.getattr("_rebuild")?;
    let cls = obj.get_type();
    let mut args = vec![cls.clone().into_any(), base.clone().into_any()];
    args.extend(contents);
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

/// The state of a subclass instance beyond its contents, or `None` if it
/// has none: `__getstate__()` where defined (Python 3.11+ defines it for
/// every object, covering `__slots__`), else the instance `__dict__`.
fn extra_state<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    let state = match obj.getattr_opt("__getstate__")? {
        Some(getstate) => getstate.call0()?,
        None => match obj.getattr_opt("__dict__")? {
            Some(dict) => dict,
            None => return Ok(None),
        },
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
        n if n > 0 => Ok(format!("{name}(...)")),
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
fn repr_items<'py, V: IntoPyObject<'py>>(
    entries: impl ExactSizeIterator<Item = (convert::Key, V)>,
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
#[pymodule]
mod _core {
    use super::*;

    #[pymodule_export]
    use super::_rebuild;
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
    use crate::object_map::{SparseMap, SparseMapIterator};
    #[pymodule_export]
    use crate::set::{SparseSet, SparseSetIterator};
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
        m.add("KEY_LIMIT", 1_u64 << 32)
    }
}
