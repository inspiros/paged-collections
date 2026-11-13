//! Concurrent containers using the same generational ids as the plain types.
//!
//! **Grow shared, shrink exclusive:** shared access may install pages and
//! recycle slots, but allocated pages and directory chunks remain at stable
//! addresses until exclusive access or destruction. Memory reclamation needs
//! no epochs or hazard pointers. Slot state is synchronized separately.
//!
//! Available with the `concurrent` feature on supported atomic targets.
//! The implementation uses `core` atomics and `alloc`, without requiring `std`.

mod atomic;
mod free_stack;
mod id_alloc;
mod spine;

pub use id_alloc::ConcurrentIdAllocator;

#[cfg(all(test, loom))]
mod loom_tests;
