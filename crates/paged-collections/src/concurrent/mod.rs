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
//! [`ConcurrentAppendHive`] supports shared insertion and ordinary `&T`
//! lookups with one Acquire slot-state load and no guard or reference count.
//! Removal and mutable access require exclusive ownership. Iteration may
//! miss insertions in flight; conversions to and from [`Hive`](crate::hive::Hive)
//! preserve live ids and stale-id history in O(index bound) time.
//!
//! [`ConcurrentPagedVec`] appends without waiting for earlier writers. Its
//! point lookups see completed pushes; `len`, iteration and page slices use a
//! hole-free prefix which can lag while an earlier push is in flight. When
//! no push is in flight, the prefix equals the number of reservations.
//!
//! [`ConcurrentSparseSet`] and [`ConcurrentSparseMap`] allow shared insertion
//! and return ordinary shared references. Entries cannot be removed or
//! changed while shared. Insertions of the same slot wait for its writer;
//! unrelated slots proceed independently. An occupied slot refuses other
//! generations. The vector and both sparse types support removal and
//! conversion to their plain equivalents
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
//!
//! The sparse types, [`ConcurrentDirectMap`] and [`AtomicDirectMap`] accept a
//! [`SharedDirectory`] parameter. The default [`Flat`](crate::page::Flat)
//! has O(1) page lookups, directory memory following the highest page, and a
//! finite index range. [`Radix`](crate::page::Radix) accepts any `usize` index,
//! uses directory memory proportional to installed pages, and has O(depth)
//! page lookups. [`AtomicIdMap`] remains Flat-only; `Tree` has no concurrent form.
//!
//! # Element invariance
//!
//! Shared insertion requires invariance in every element type: shortening a
//! stored reference's lifetime would allow inserting a reference that expires
//! while the original owner still expects a `'static` reference.
//! The following identity functions typecheck; changing only their returned
//! element lifetimes is rejected, for either directory.
//!
//! ```
//! # #[cfg(feature = "concurrent")] {
//! use paged_collections::concurrent::*;
//! use paged_collections::page::{Flat, Radix};
//! use paged_collections::key::{DirectKey, Key};
//! #[derive(PartialEq, Eq)]
//! struct K(usize);
//! impl Key for &K { fn index(self) -> usize { self.0 } }
//! impl DirectKey for &K { type Tag = Self; fn tag(self) -> Self { self } fn from_parts(_: usize, k: Self) -> Self { k } }
//! fn identity_0(x: ConcurrentHive<&'static u8>) -> ConcurrentHive<&'static u8> { x }
//! fn identity_1(x: ConcurrentAppendHive<&'static u8>) -> ConcurrentAppendHive<&'static u8> { x }
//! fn identity_2(x: ConcurrentPagedVec<&'static u8>) -> ConcurrentPagedVec<&'static u8> { x }
//! fn identity_3(x: ConcurrentSparseSet<&'static K, 4, Flat>) -> ConcurrentSparseSet<&'static K, 4, Flat> { x }
//! fn identity_4(x: ConcurrentSparseMap<&'static K, (), 4, Flat>) -> ConcurrentSparseMap<&'static K, (), 4, Flat> { x }
//! fn identity_5(x: ConcurrentSparseMap<usize, &'static u8, 4, Flat>) -> ConcurrentSparseMap<usize, &'static u8, 4, Flat> { x }
//! fn identity_6(x: ConcurrentDirectMap<&'static K, (), 4, Flat>) -> ConcurrentDirectMap<&'static K, (), 4, Flat> { x }
//! fn identity_7(x: ConcurrentDirectMap<usize, &'static u8, 4, Flat>) -> ConcurrentDirectMap<usize, &'static u8, 4, Flat> { x }
//! fn identity_8(x: ConcurrentSparseSet<&'static K, 4, Radix>) -> ConcurrentSparseSet<&'static K, 4, Radix> { x }
//! fn identity_9(x: ConcurrentSparseMap<&'static K, (), 4, Radix>) -> ConcurrentSparseMap<&'static K, (), 4, Radix> { x }
//! fn identity_10(x: ConcurrentSparseMap<usize, &'static u8, 4, Radix>) -> ConcurrentSparseMap<usize, &'static u8, 4, Radix> { x }
//! fn identity_11(x: ConcurrentDirectMap<&'static K, (), 4, Radix>) -> ConcurrentDirectMap<&'static K, (), 4, Radix> { x }
//! fn identity_12(x: ConcurrentDirectMap<usize, &'static u8, 4, Radix>) -> ConcurrentDirectMap<usize, &'static u8, 4, Radix> { x }
//! # }
//! ```
//!
//! ```compile_fail
//! # #[cfg(feature = "concurrent")] {
//! use paged_collections::concurrent::ConcurrentHive;
//! fn shorten<'a>(x: ConcurrentHive<&'static u8>) -> ConcurrentHive<&'a u8> { x }
//! # }
//! ```
//!
//! ```compile_fail
//! # #[cfg(feature = "concurrent")] {
//! use paged_collections::concurrent::ConcurrentAppendHive;
//! fn shorten<'a>(x: ConcurrentAppendHive<&'static u8>) -> ConcurrentAppendHive<&'a u8> { x }
//! # }
//! ```
//!
//! ```compile_fail
//! # #[cfg(feature = "concurrent")] {
//! use paged_collections::concurrent::ConcurrentPagedVec;
//! fn shorten<'a>(x: ConcurrentPagedVec<&'static u8>) -> ConcurrentPagedVec<&'a u8> { x }
//! # }
//! ```
//!
//! ```compile_fail
//! # #[cfg(feature = "concurrent")] {
//! use paged_collections::concurrent::ConcurrentSparseSet;
//! use paged_collections::page::Flat;
//! use paged_collections::key::Key;
//! #[derive(PartialEq, Eq)]
//! struct K(usize);
//! impl Key for &K { fn index(self) -> usize { self.0 } }
//! fn shorten<'a>(x: ConcurrentSparseSet<&'static K, 4, Flat>) -> ConcurrentSparseSet<&'a K, 4, Flat> { x }
//! # }
//! ```
//!
//! ```compile_fail
//! # #[cfg(feature = "concurrent")] {
//! use paged_collections::concurrent::ConcurrentSparseMap;
//! use paged_collections::page::Flat;
//! use paged_collections::key::Key;
//! #[derive(PartialEq, Eq)]
//! struct K(usize);
//! impl Key for &K { fn index(self) -> usize { self.0 } }
//! fn shorten<'a>(x: ConcurrentSparseMap<&'static K, (), 4, Flat>) -> ConcurrentSparseMap<&'a K, (), 4, Flat> { x }
//! # }
//! ```
//!
//! ```compile_fail
//! # #[cfg(feature = "concurrent")] {
//! use paged_collections::concurrent::ConcurrentSparseMap;
//! use paged_collections::page::Flat;
//! fn shorten<'a>(x: ConcurrentSparseMap<usize, &'static u8, 4, Flat>) -> ConcurrentSparseMap<usize, &'a u8, 4, Flat> { x }
//! # }
//! ```
//!
//! ```compile_fail
//! # #[cfg(feature = "concurrent")] {
//! use paged_collections::concurrent::ConcurrentDirectMap;
//! use paged_collections::page::Flat;
//! use paged_collections::key::{DirectKey, Key};
//! #[derive(PartialEq, Eq)]
//! struct K(usize);
//! impl Key for &K { fn index(self) -> usize { self.0 } }
//! impl DirectKey for &K { type Tag = Self; fn tag(self) -> Self { self } fn from_parts(_: usize, k: Self) -> Self { k } }
//! fn shorten<'a>(x: ConcurrentDirectMap<&'static K, (), 4, Flat>) -> ConcurrentDirectMap<&'a K, (), 4, Flat> { x }
//! # }
//! ```
//!
//! ```compile_fail
//! # #[cfg(feature = "concurrent")] {
//! use paged_collections::concurrent::ConcurrentDirectMap;
//! use paged_collections::page::Flat;
//! fn shorten<'a>(x: ConcurrentDirectMap<usize, &'static u8, 4, Flat>) -> ConcurrentDirectMap<usize, &'a u8, 4, Flat> { x }
//! # }
//! ```
//!
//! ```compile_fail
//! # #[cfg(feature = "concurrent")] {
//! use paged_collections::concurrent::ConcurrentSparseSet;
//! use paged_collections::page::Radix;
//! use paged_collections::key::Key;
//! #[derive(PartialEq, Eq)]
//! struct K(usize);
//! impl Key for &K { fn index(self) -> usize { self.0 } }
//! fn shorten<'a>(x: ConcurrentSparseSet<&'static K, 4, Radix>) -> ConcurrentSparseSet<&'a K, 4, Radix> { x }
//! # }
//! ```
//!
//! ```compile_fail
//! # #[cfg(feature = "concurrent")] {
//! use paged_collections::concurrent::ConcurrentSparseMap;
//! use paged_collections::page::Radix;
//! use paged_collections::key::Key;
//! #[derive(PartialEq, Eq)]
//! struct K(usize);
//! impl Key for &K { fn index(self) -> usize { self.0 } }
//! fn shorten<'a>(x: ConcurrentSparseMap<&'static K, (), 4, Radix>) -> ConcurrentSparseMap<&'a K, (), 4, Radix> { x }
//! # }
//! ```
//!
//! ```compile_fail
//! # #[cfg(feature = "concurrent")] {
//! use paged_collections::concurrent::ConcurrentSparseMap;
//! use paged_collections::page::Radix;
//! fn shorten<'a>(x: ConcurrentSparseMap<usize, &'static u8, 4, Radix>) -> ConcurrentSparseMap<usize, &'a u8, 4, Radix> { x }
//! # }
//! ```
//!
//! ```compile_fail
//! # #[cfg(feature = "concurrent")] {
//! use paged_collections::concurrent::ConcurrentDirectMap;
//! use paged_collections::page::Radix;
//! use paged_collections::key::{DirectKey, Key};
//! #[derive(PartialEq, Eq)]
//! struct K(usize);
//! impl Key for &K { fn index(self) -> usize { self.0 } }
//! impl DirectKey for &K { type Tag = Self; fn tag(self) -> Self { self } fn from_parts(_: usize, k: Self) -> Self { k } }
//! fn shorten<'a>(x: ConcurrentDirectMap<&'static K, (), 4, Radix>) -> ConcurrentDirectMap<&'a K, (), 4, Radix> { x }
//! # }
//! ```
//!
//! ```compile_fail
//! # #[cfg(feature = "concurrent")] {
//! use paged_collections::concurrent::ConcurrentDirectMap;
//! use paged_collections::page::Radix;
//! fn shorten<'a>(x: ConcurrentDirectMap<usize, &'static u8, 4, Radix>) -> ConcurrentDirectMap<usize, &'a u8, 4, Radix> { x }
//! # }
//! ```
//!

mod append;
mod append_hive;
mod atomic;
mod atomic_map;
mod count;
mod direct;
mod directory;
mod free_stack;
mod hive;
mod id_alloc;
mod index;
mod lock;
mod radix;
mod recycle;
mod sparse;
mod spine;
mod vec;

pub use append_hive::{AppendDrain, AppendIter, AppendIterMut, ConcurrentAppendHive};
pub use atomic_map::{AtomicDirectMap, AtomicIdMap};
pub use direct::{
    ConcurrentDirectMap, DirectMapRef, DirectMapRefMut, Entry, OccupiedEntry, VacantEntry,
};
pub use directory::SharedDirectory;
pub use hive::{ConcurrentHive, Drain, Iter, IterMut, Ref};
pub use id_alloc::ConcurrentIdAllocator;
pub use sparse::{ConcurrentSparseMap, ConcurrentSparseSet};
pub use vec::ConcurrentPagedVec;

#[cfg(all(test, loom))]
mod loom_append;

#[cfg(all(test, loom))]
mod loom_append_hive;

#[cfg(all(test, loom))]
mod loom_atomic_map;

#[cfg(all(test, loom))]
mod loom_direct;

#[cfg(all(test, loom))]
mod loom_hive;

#[cfg(all(test, loom))]
mod loom_radix;

#[cfg(all(test, loom))]
mod loom_shards;

#[cfg(all(test, loom))]
mod loom_tests;
