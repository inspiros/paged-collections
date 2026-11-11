//! Keys and how they map to slot indices.
//!
//! To be designed: a `Key` trait that turns a key into a dense slot index,
//! stripping any generation or version bits first so that generational ids
//! (index plus version) index the same slot. Implementations are planned for
//! `u16`, `u32`, `u64` and `usize`, plus a derive-free way to opt in newtypes.
