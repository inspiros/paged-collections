//! Atomic operations shared by the production implementation and loom models.

#[cfg(not(loom))]
pub(super) use core::hint::spin_loop;
#[cfg(not(loom))]
pub(super) use core::sync::atomic::{
    AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering, fence,
};
#[cfg(loom)]
pub(super) use loom::hint::spin_loop;
// Loom models SeqCst accesses as AcqRel but supports SeqCst fences.
#[cfg(loom)]
pub(super) use loom::sync::atomic::fence;
#[cfg(loom)]
pub(super) use loom::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering};

/// Keeps a contended word away from neighbouring fields' cache lines.
#[repr(align(128))]
pub(super) struct CachePadded<T>(pub(super) T);

/// A value cell with the same checked access interface in loom and production.
// Production append pages expose contiguous initialized prefixes. Preserve the
// layout of T through both this wrapper and core::cell::UnsafeCell. Loom cells
// contain tracking metadata, so their prefixes are visited one cell at a time.
#[cfg_attr(not(loom), repr(transparent))]
pub(super) struct UnsafeCell<T> {
    #[cfg(loom)]
    inner: loom::cell::UnsafeCell<T>,
    #[cfg(not(loom))]
    inner: core::cell::UnsafeCell<T>,
}

impl<T> UnsafeCell<T> {
    pub(super) fn new(value: T) -> Self {
        Self {
            #[cfg(loom)]
            inner: loom::cell::UnsafeCell::new(value),
            #[cfg(not(loom))]
            inner: core::cell::UnsafeCell::new(value),
        }
    }

    pub(super) fn with<R>(&self, f: impl FnOnce(*const T) -> R) -> R {
        #[cfg(loom)]
        {
            self.inner.with(f)
        }
        #[cfg(not(loom))]
        {
            f(self.inner.get().cast_const())
        }
    }

    pub(super) fn with_mut<R>(&self, f: impl FnOnce(*mut T) -> R) -> R {
        #[cfg(loom)]
        {
            self.inner.with_mut(f)
        }
        #[cfg(not(loom))]
        {
            f(self.inner.get())
        }
    }

    #[cfg(any(not(feature = "std"), loom))]
    pub(super) fn into_inner(self) -> T {
        self.inner.into_inner()
    }

    #[cfg(loom)]
    pub(super) fn write(&self) -> loom::cell::MutPtr<T> {
        self.inner.get_mut()
    }

    // Keep loom's read tracking alive for the entire Ref, including references
    // returned by Deref that escape the shorter `with` callback.
    #[cfg(loom)]
    pub(super) fn read(&self) -> loom::cell::ConstPtr<T> {
        self.inner.get()
    }
}
