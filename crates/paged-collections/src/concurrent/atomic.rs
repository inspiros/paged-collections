//! Atomic operations shared by the production implementation and loom models.

#[cfg(not(loom))]
pub(super) use core::hint::spin_loop;
#[cfg(not(loom))]
pub(super) use core::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering};
#[cfg(loom)]
pub(super) use loom::hint::spin_loop;
#[cfg(loom)]
pub(super) use loom::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering};

/// Keeps a contended word away from neighbouring fields' cache lines.
#[repr(align(128))]
pub(super) struct CachePadded<T>(pub(super) T);
