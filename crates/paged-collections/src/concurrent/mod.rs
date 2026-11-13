//! Concurrent containers using the same generational ids as the plain types.
//!
//! **Grow shared, shrink exclusive:** shared access may install pages and
//! recycle slots, but allocated pages and directory chunks remain at stable
//! addresses until exclusive access or destruction. Memory reclamation needs
//! no epochs or hazard pointers. Slot state is synchronized separately.
//!
//! Available with the `concurrent` feature on supported atomic targets.
//! The implementation uses `core` atomics and `alloc`, without requiring `std`.
//!
//! [`ConcurrentHive`] supports shared insertion and removal. A lookup returns
//! a [`Ref`] guard, which pins its value until dropped. A successful lookup
//! costs two atomic writes: acquiring the guard and releasing it. Hot values
//! can therefore contend even between readers. Prefer the plain
//! [`Hive`](crate::hive::Hive) when shared mutation is unnecessary.

mod atomic;
mod free_stack;
mod hive;
mod id_alloc;
mod spine;

pub use hive::{ConcurrentHive, Drain, Iter, IterMut, Ref};
pub use id_alloc::ConcurrentIdAllocator;

#[cfg(all(test, loom))]
mod loom_hive;

#[cfg(all(test, loom))]
mod loom_tests;
