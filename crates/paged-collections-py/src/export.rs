//! Zero-copy NumPy views of a collection's dense arrays.
//!
//! A view is a NumPy array whose data pointer points into a `Vec` owned by a
//! collection. The array's `base` is the collection's [`ExportGuard`], which
//! keeps the collection alive. All live views of one collection share one
//! guard (cached through a weak reference), so tools that group arrays by
//! base object, such as rust-numpy's borrow checker, see that they may
//! overlap. The guard counts as an export until NumPy drops the last array
//! that refers to it.
//!
//! # Invariants
//!
//! While a collection's export count is non-zero:
//!
//! 1. Elements covered by a view are not moved, dropped or freed, and their
//!    buffer is not reallocated. Methods that could do so call
//!    [`Exports::check`] first and raise `BufferError`. Maps keep their dense
//!    lengths fixed. PagedList pages may grow: each buffer has capacity at
//!    least PAGE, never grows past PAGE, and push touches only the new last
//!    element. Reserving pages moves only directory headers, not buffers.
//! 2. Rust code creates no `&mut` reference into an exported `Vec`'s
//!    elements. Exported key arrays are only read. Exported value arrays
//!    hold [`ValueCell`]s (atomics), are exported from a shared
//!    `&[ValueCell<T>]`, and are read and written by Rust only through
//!    [`ValueCell::get`] and [`ValueCell::set`], which the aliasing model
//!    allows alongside the raw pointer NumPy holds.
//!
//! # Threads
//!
//! NumPy releases the GIL inside many operations. As with any NumPy array
//! shared between threads, the caller must not write to a view in one thread
//! while another thread reads or writes the same collection; the bindings
//! cannot detect such races.
//!
//! The count lives in an `Arc` shared with the guard, so the guard releases
//! its export without borrowing the collection (which may be borrowed when
//! NumPy drops the last view, for example during garbage collection
//! triggered inside a method).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use numpy::ndarray::ArrayView1;
use numpy::{Element, PyArray1};
use pyo3::exceptions::{PyBufferError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::PyWeakrefReference;

use crate::convert::TypedValue;

/// One value of a typed map: an atomic with the size of `T`, so that NumPy
/// may write it through a raw pointer while Rust holds shared references,
/// and so that the map is `Sync`, as PyO3 requires of a class.
#[repr(transparent)]
pub(crate) struct ValueCell<T: TypedValue>(T::Atomic);

impl<T: TypedValue> ValueCell<T> {
    /// `ValueCell<T>` can be viewed as `T`: same size, alignment at least
    /// `T`'s.
    const LAYOUT: () = assert!(
        size_of::<T::Atomic>() == size_of::<T>() && align_of::<T::Atomic>() >= align_of::<T>()
    );

    pub(crate) fn new(value: T) -> Self {
        Self(T::to_atomic(value))
    }

    pub(crate) fn get(&self) -> T {
        T::load(&self.0)
    }

    pub(crate) fn set(&self, value: T) {
        T::store(&self.0, value);
    }
}

impl<T: TypedValue> Clone for ValueCell<T> {
    fn clone(&self) -> Self {
        Self::new(self.get())
    }
}

impl<T: TypedValue> PartialEq for ValueCell<T> {
    fn eq(&self, other: &Self) -> bool {
        self.get() == other.get()
    }
}

/// The export state of one collection.
#[derive(Default)]
pub(crate) struct Exports {
    /// Number of live guards: 0 or 1, briefly 2 while a dying guard is
    /// replaced.
    count: Arc<AtomicUsize>,
    /// The live guard, if any, so new views share it.
    cached: Mutex<Option<Py<PyWeakrefReference>>>,
}

impl Exports {
    /// Whether any view of the collection is alive.
    pub(crate) fn active(&self) -> bool {
        self.count.load(Ordering::Acquire) != 0
    }

    /// `BufferError` if any view is alive, naming the collection type.
    pub(crate) fn check(&self, type_name: &str) -> PyResult<()> {
        if self.active() {
            Err(PyBufferError::new_err(format!(
                "cannot resize or reorder this {type_name} while NumPy views of it exist"
            )))
        } else {
            Ok(())
        }
    }

    /// The guard of `owner`'s live views, creating it if there is none.
    ///
    /// The lock is never held while Python allocates: allocation can run the
    /// garbage collector, whose finalizers may ask for a view of the same
    /// collection and would deadlock on it.
    fn guard<'py>(&self, owner: &Bound<'py, PyAny>) -> PyResult<Bound<'py, ExportGuard>> {
        let py = owner.py();
        if let Some(live) = self.live_guard(py)? {
            return Ok(live);
        }
        // Count before the guard exists: if `Bound::new` fails, the guard
        // value is dropped and its `Drop` undoes the count.
        self.count.fetch_add(1, Ordering::AcqRel);
        let candidate = Bound::new(
            py,
            ExportGuard {
                owner: owner.clone().unbind(),
                exports: Arc::clone(&self.count),
            },
        )?;
        let weak = PyWeakrefReference::new(candidate.as_any())?.unbind();
        // A reentrant call may have published a guard meanwhile: share it,
        // so all live views keep one base. The unused candidate and weakref
        // are dropped after the lock is released.
        let mut cached = self.lock()?;
        if let Some(live) = cached.as_ref().and_then(|w| w.bind(py).upgrade()) {
            drop(cached);
            drop(weak);
            drop(candidate);
            return Ok(live.cast_into::<ExportGuard>()?);
        }
        let old = cached.replace(weak);
        drop(cached);
        drop(old);
        Ok(candidate)
    }

    /// The cached guard, if it is still alive. Upgrading allocates nothing.
    fn live_guard<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, ExportGuard>>> {
        let cached = self.lock()?;
        let live = cached.as_ref().and_then(|weak| weak.bind(py).upgrade());
        drop(cached);
        live.map(|guard| guard.cast_into::<ExportGuard>().map_err(Into::into))
            .transpose()
    }

    fn lock(&self) -> PyResult<std::sync::MutexGuard<'_, Option<Py<PyWeakrefReference>>>> {
        self.cached
            .lock()
            .map_err(|_| PyRuntimeError::new_err("export state is poisoned"))
    }
}

/// The `base` of every NumPy view of one collection: keeps the collection
/// alive and holds its export until it is dropped.
#[pyclass(module = "paged_collections._core", frozen, weakref)]
pub(crate) struct ExportGuard {
    owner: Py<PyAny>,
    exports: Arc<AtomicUsize>,
}

#[pymethods]
impl ExportGuard {
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        visit.call(&self.owner)
    }
}

impl Drop for ExportGuard {
    fn drop(&mut self) {
        self.exports.fetch_sub(1, Ordering::AcqRel);
    }
}

/// A read-only view of `data`, which `owner` owns and whose export state is
/// `exports`.
///
/// # Safety
///
/// `data` must be the contents of a `Vec` owned by `owner` (directly or
/// through the collection it holds), and that `Vec` must keep the
/// [invariants](self) for as long as `exports` is active: Rust code only
/// reads it through shared references meanwhile.
pub(crate) unsafe fn read_only_view<'py, T: Element>(
    owner: &Bound<'py, PyAny>,
    exports: &Exports,
    data: &[T],
) -> PyResult<Bound<'py, PyArray1<T>>> {
    let guard = exports.guard(owner)?;
    let view = ArrayView1::from(data);
    // SAFETY: the guard is the array's base, so `owner` (and the `Vec` it
    // owns) outlives the array. The caller guarantees the `Vec` keeps the
    // invariants while the guard's export is counted, which lasts until the
    // array and everything derived from it are gone.
    let array = unsafe { PyArray1::borrow_from_array(&view, guard.into_any()) };
    // The data came from a shared borrow: NumPy must not write through it.
    // NumPy refuses to turn the flag back on, because the base is not a
    // writeable buffer.
    array.getattr("flags")?.setattr("writeable", false)?;
    Ok(array)
}

/// A view of the cells in `data` as plain `T`s, writeable if `writeable`.
/// Writes through the view change the collection's values.
///
/// # Safety
///
/// `data` must be the contents of a `Vec`, or a prefix of a page `Vec`,
/// owned by `owner`.
/// Elements in that prefix and their buffer must keep the [invariants](self)
/// while exports are active; the page may append beyond the prefix without
/// reallocating or creating mutable references into the prefix. Rust code may
/// also write the cells, through [`ValueCell::set`] only. If `writeable`,
/// every bit pattern of `size_of::<T>()` bytes must be a valid `T::Atomic`: NumPy code can
/// reinterpret the view as another dtype (`view.view(np.uint8)`) and write
/// any bytes. That holds for every [`TypedValue`].
pub(crate) unsafe fn cell_view<'py, T: TypedValue>(
    owner: &Bound<'py, PyAny>,
    exports: &Exports,
    data: &[ValueCell<T>],
    writeable: bool,
) -> PyResult<Bound<'py, PyArray1<T>>> {
    let () = ValueCell::<T>::LAYOUT;
    let guard = exports.guard(owner)?;
    // The pointer comes from a shared reference to atomics, whose interior
    // mutability allows writes through it while other shared references to
    // the cells exist.
    let ptr = data.as_ptr().cast::<T>();
    // SAFETY: `ptr` is non-null, aligned and valid for `data.len()` reads of
    // `T` (`ValueCell<T>` is `repr(transparent)` over an atomic of `T`'s size
    // and at least its alignment, checked by `LAYOUT`), and nothing writes
    // the cells through a `&mut` while the view exists (invariant 2).
    let view = unsafe { ArrayView1::from_shape_ptr(data.len(), ptr) };
    // SAFETY: as in `read_only_view`. NumPy writes go through the pointer
    // derived from `&[ValueCell<T>]`, which permits them.
    let array = unsafe { PyArray1::borrow_from_array(&view, guard.into_any()) };
    if !writeable {
        array.getattr("flags")?.setattr("writeable", false)?;
    }
    Ok(array)
}
