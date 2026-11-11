//! `PagedSparseMap<K, V>`: a sparse set that also stores one value per key.
//!
//! Planned: values kept in a dense array parallel to the dense keys, so both
//! can be iterated or exposed as contiguous slices.
