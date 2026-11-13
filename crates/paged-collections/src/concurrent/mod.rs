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
//!
//! [`ConcurrentPagedVec`] appends without waiting for earlier writers. Its
//! point lookups see completed pushes; `len`, iteration and page slices use a
//! hole-free prefix which can lag while an earlier push is in flight. When
//! no push is in flight, the prefix equals the number of reservations.
//!
//! [`ConcurrentSparseSet`] and [`ConcurrentSparseMap`] allow shared insertion
//! and return ordinary shared references. Entries cannot be removed or
//! changed while shared. Insertions of the same slot wait for its writer;
//! unrelated slots proceed independently. The sparse directory is Flat-like,
//! with a finite index range, and an occupied slot refuses other generations.
//! All three types support removal and conversion to their plain equivalents
//! under exclusive ownership. Conversion moves elements in O(len) time.
//!
//! [`ConcurrentDirectMap`] permits shared replacement and removal through one
//! reader-writer lock per page. Neighbouring keys contend; guards and entries
//! must be released before locking the same page again. Ordered traversals
//! are consistent per page only. Pages are freed only with exclusive access.
//!
//! [`AtomicDirectMap`] and [`AtomicIdMap`] provide lock-free shared updates
//! of one-word values. Prefer them over [`ConcurrentDirectMap`] when values
//! fit their encodings: `usize` keys with `u64` values below `u64::MAX`, or
//! generational ids with any `u32` value. Traversals copy values in index order.

mod append;
mod atomic;
mod atomic_map;
mod direct;
mod free_stack;
mod hive;
mod id_alloc;
mod index;
mod lock;
mod sparse;
mod spine;
mod vec;

pub use atomic_map::{AtomicDirectMap, AtomicIdMap};
pub use direct::{
    ConcurrentDirectMap, DirectMapRef, DirectMapRefMut, Entry, OccupiedEntry, VacantEntry,
};
pub use hive::{ConcurrentHive, Drain, Iter, IterMut, Ref};
pub use id_alloc::ConcurrentIdAllocator;
pub use sparse::{ConcurrentSparseMap, ConcurrentSparseSet};
pub use vec::ConcurrentPagedVec;

#[cfg(all(test, loom))]
mod loom_append;

#[cfg(all(test, loom))]
mod loom_atomic_map;

#[cfg(all(test, loom))]
mod loom_direct;

#[cfg(all(test, loom))]
mod loom_hive;

#[cfg(all(test, loom))]
mod loom_tests;
