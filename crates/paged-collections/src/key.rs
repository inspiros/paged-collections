//! Keys and how they map to slot indices.
//!
//! Every paged-index collection stores at most one key per *slot*, and
//! [`Key::index`] names the slot. For plain integers the slot is the integer
//! itself. For generational ids (an index plus a version or generation
//! counter) the slot is the index part only, so successive generations of the
//! same id share a slot.
//!
//! Collections compare the stored key with `==` on every lookup, so a key whose
//! slot is held by a different key (for example an older generation) is not
//! found. Methods that insert document what happens when the slot is held by a
//! different key.
//!
//! # Implementing `Key` for an id type
//!
//! ```
//! use paged_collections::key::Key;
//!
//! /// 32-bit entity id: low 24 bits are the index, high 8 bits the generation.
//! #[derive(Clone, Copy, PartialEq, Eq, Debug)]
//! struct Entity(u32);
//!
//! impl Key for Entity {
//!     fn index(self) -> usize {
//!         (self.0 & 0x00ff_ffff) as usize
//!     }
//! }
//!
//! let a = Entity(5);
//! let a_next_generation = Entity(5 | 1 << 24);
//! assert_eq!(a.index(), a_next_generation.index());
//! assert_ne!(a, a_next_generation);
//! ```

/// A value that names a slot in a paged collection.
///
/// # Contract
///
/// - `index` must be a pure function of the key: equal keys have equal
///   indices, and calling it twice gives the same result.
/// - `index` must not panic. Collections call it while they rewrite their
///   index (after removing or sorting), where a panic would leave them
///   inconsistent.
/// - Different keys may share an index (generational ids do); the collections
///   then treat them as competing for one slot.
/// - Memory use follows the indices in use, not the number of keys, so keys
///   should map to indices that cluster. A flat page directory allocates
///   `max_index / PAGE` directory entries; see [`crate::page`].
///
/// Breaking the contract cannot cause undefined behaviour, but collections
/// may then return wrong results or panic.
pub trait Key: Copy + Eq {
    /// The slot index of this key.
    fn index(self) -> usize;
}

macro_rules! impl_key_lossless {
    ($($t:ty),* $(,)?) => {$(
        impl Key for $t {
            #[inline]
            fn index(self) -> usize {
                usize::from(self)
            }
        }
    )*};
}

impl_key_lossless!(u8, u16);

impl Key for usize {
    #[inline]
    fn index(self) -> usize {
        self
    }
}

// `u32` and `u64` convert to `usize` without loss only on targets whose
// pointers are at least that wide, so the impls are gated on pointer width.
#[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
impl Key for u32 {
    #[inline]
    fn index(self) -> usize {
        self as usize
    }
}

#[cfg(target_pointer_width = "64")]
impl Key for u64 {
    #[inline]
    fn index(self) -> usize {
        self as usize
    }
}
