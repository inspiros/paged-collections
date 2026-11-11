//! `IdAllocator`: hands out generational ids and recycles freed ones.
//!
//! Planned: a free list that bumps the generation of each recycled index, for
//! use as keys in the other collections.
