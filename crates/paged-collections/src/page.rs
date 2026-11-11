//! The page directory shared by all paged-index collections.
//!
//! To be designed: a directory mapping page number to a lazily allocated,
//! fixed-size page (`PAGE` slots, a const generic). The flat form
//! (`Vec<Option<Box<[T; PAGE]>>>`) still grows with `max_key / PAGE`, which is
//! too large for 64-bit keys spread over a wide range, so a two-level or
//! hash-based directory is planned as an alternative behind the same interface.
