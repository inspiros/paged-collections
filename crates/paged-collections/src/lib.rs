//! Paged collections: containers whose memory is split into fixed-size pages.
//!
//! Paging serves two separate purposes, and each collection states which it uses:
//!
//! - **Paged index.** A lookup table indexed by key is split into pages that are
//!   allocated on first use, so memory follows the occupied key ranges rather than
//!   the largest key ever inserted.
//! - **Paged storage.** Elements live in fixed-size pages that never move, so
//!   references and pointers to elements stay valid while the collection grows.
//!
//! # Collections
//!
//! | Module | Type | Paging |
//! |---|---|---|
//! | [`sparse_set`] | `PagedSparseSet<K>` | index; storage optional |
//! | [`sparse_map`] | `PagedSparseMap<K, V>` | index; storage optional |
//! | [`direct_map`] | `PagedDirectMap<K, V>` | index (values in place, slot order) |
//! | [`paged_vec`] | `PagedVec<T>` | storage (stable addresses) |
//! | [`hive`] | `Hive<T>` | storage (stable addresses, reusable slots, generational ids) |
//! | [`id_alloc`] | `IdAllocator`, `Id` | none (free list of generational ids) |
//!
//! Shared building blocks live in [`key`] (how a key maps to an index) and
//! [`page`] (the page directory).
//!
//! # Features
//!
//! - `std` (default): `std`-only conveniences. Without it the crate is `no_std`
//!   and needs only `alloc`.
//! - `serde`: serialization support.

#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

pub mod direct_map;
// Slot indices are `u32` and must convert to `usize` without loss.
#[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
pub mod hive;
// Indices are `u32` and must convert to `usize` without loss.
#[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
pub mod id_alloc;
pub mod key;
pub mod page;
pub mod paged_vec;
pub mod sparse_map;
pub mod sparse_set;

// Runs the README's Rust examples as doctests.
#[cfg(all(doctest, any(target_pointer_width = "32", target_pointer_width = "64")))]
#[doc = include_str!("../../../README.md")]
struct ReadmeDoctests;
