//! `PagedVecMap<K, V>`: a direct-indexed map with one optional value per slot.
//!
//! Planned: values stored in place in the paged index (no dense array), which
//! suits keys that cluster into a few dense ranges.
