//! `PagedSparseSet<K>`: a sparse set whose sparse index is paged.
//!
//! Planned: Briggs–Torczon layout with a dense key array and a paged sparse
//! index; O(1) insert, remove (swap-remove), contains and clear; iteration over
//! the dense array; an option to page the dense side for stable addresses.
