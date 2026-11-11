//! `PagedVec<T>`: a growable vector stored in fixed-size pages.
//!
//! Planned: pushing never moves existing elements, so references stay valid
//! and growth never copies the whole buffer.
