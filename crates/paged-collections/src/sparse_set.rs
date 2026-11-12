//! [`PagedSparseSet`]: a sparse set whose sparse index is paged.
//!
//! A sparse set keeps its keys in a contiguous *dense* array and finds them
//! through a *sparse* index from slot index to dense position. Here the sparse
//! index is split into pages that are allocated on first use, so memory
//! follows the key ranges actually used instead of the largest key.
//!
//! | Operation | Cost |
//! |---|---|
//! | [`insert`](PagedSparseSet::insert), [`remove`](PagedSparseSet::remove), [`contains`](PagedSparseSet::contains) | O(1) (O(log pages) with [`Tree`](crate::page::Tree)) |
//! | [`clear`](PagedSparseSet::clear) | O(1): pages are kept and old slots are ignored |
//! | iteration, [`as_slice`](PagedSparseSet::as_slice) | over the dense array only |
//!
//! Lookups follow Briggs and Torczon: a slot is trusted only if it points
//! inside the dense array at a key with the same slot index. Stale slots, left
//! by [`clear`](PagedSparseSet::clear), are therefore harmless.
//!
//! ```
//! use paged_collections::sparse_set::PagedSparseSet;
//!
//! let mut set: PagedSparseSet<u32> = PagedSparseSet::new();
//! assert!(set.insert(3_000_000));
//! assert!(set.insert(7));
//! assert!(!set.insert(7));
//! assert!(set.contains(3_000_000));
//! // Two 4 KiB pages instead of a 12 MB flat sparse array.
//! assert_eq!(set.allocated_pages(), 2);
//!
//! assert!(set.remove(7));
//! assert_eq!(set.as_slice(), &[3_000_000]);
//! ```

use alloc::collections::TryReserveError;
use alloc::vec::Vec;
use core::fmt;
use core::num::NonZeroU32;

use crate::key::Key;
use crate::page::{DEFAULT_PAGE, Directory, Flat, PageStore, split};

/// One slot of the sparse index: the dense position plus one, or `None`.
///
/// Storing the position plus one in a [`NonZeroU32`] keeps the slot at four
/// bytes and makes `None` the all-zero default. It limits a set to
/// `u32::MAX` keys.
pub type Slot = Option<NonZeroU32>;

/// A set of [`Key`]s with O(1) insert, remove and lookup, and a contiguous
/// array of its keys.
///
/// - `PAGE` is the number of slots per page of the sparse index; it must be
///   non-zero. See [`DEFAULT_PAGE`].
/// - `D` picks the page directory: [`Flat`] (default) or
///   [`Tree`](crate::page::Tree) for keys spread over the 64-bit range.
///
/// The set holds at most one key per slot index. With plain integer keys that
/// is the usual set semantics. With generational keys, [`insert`] refuses a
/// key whose slot is held by another generation, and [`replace`] overwrites
/// it.
///
/// Keys are kept in insertion order until a removal, which moves the last key
/// into the gap.
///
/// [`insert`]: PagedSparseSet::insert
/// [`replace`]: PagedSparseSet::replace
pub struct PagedSparseSet<K, const PAGE: usize = DEFAULT_PAGE, D: Directory = Flat> {
    dense: Vec<K>,
    sparse: D::Store<Slot, PAGE>,
}

/// Encodes a dense position that is already in the set.
#[inline]
fn encode(pos: usize) -> Slot {
    pos.checked_add(1)
        .and_then(|p| u32::try_from(p).ok())
        .and_then(NonZeroU32::new)
}

impl<K: Key, const PAGE: usize, D: Directory> PagedSparseSet<K, PAGE, D> {
    /// Number of slots per page of the sparse index.
    pub const PAGE_SIZE: usize = PAGE;

    /// Creates an empty set. No memory is allocated until the first insert.
    pub fn new() -> Self {
        Self {
            dense: Vec::new(),
            sparse: Default::default(),
        }
    }

    /// Creates an empty set whose dense array can hold `capacity` keys
    /// without reallocating. Pages are still allocated on demand.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            dense: Vec::with_capacity(capacity),
            sparse: Default::default(),
        }
    }

    /// Number of keys in the set.
    #[inline]
    pub fn len(&self) -> usize {
        self.dense.len()
    }

    /// Whether the set has no keys.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.dense.is_empty()
    }

    /// Number of keys the dense array can hold without reallocating.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.dense.capacity()
    }

    /// Number of allocated pages in the sparse index.
    #[inline]
    pub fn allocated_pages(&self) -> usize {
        self.sparse.allocated()
    }

    /// Reserves room in the dense array for at least `additional` more keys.
    pub fn reserve(&mut self, additional: usize) {
        self.dense.reserve(additional);
    }

    /// Like [`reserve`](Self::reserve), but returns an error instead of
    /// panicking or aborting when the capacity overflows or the allocation
    /// fails.
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.dense.try_reserve(additional)
    }

    /// Dense position of the key occupying slot `index`, if any.
    #[inline]
    pub(crate) fn occupant_pos(&self, index: usize) -> Option<usize> {
        let (page, offset) = split::<PAGE>(index);
        let slot = self.sparse.get(page)?[offset]?;
        // Stored slots never exceed the dense length, which fits in `usize`.
        let pos = slot.get() as usize - 1;
        match self.dense.get(pos) {
            Some(key) if key.index() == index => Some(pos),
            _ => None,
        }
    }

    /// Writes slot `index`, whose page must already be allocated.
    #[inline]
    fn set_slot(&mut self, index: usize, slot: Slot) {
        let (page, offset) = split::<PAGE>(index);
        let page = self.sparse.get_mut(page);
        debug_assert!(page.is_some(), "slot {index} has no page");
        if let Some(page) = page {
            page[offset] = slot;
        }
    }

    /// Dense position of `key`, if it is in the set.
    ///
    /// Positions index [`as_slice`](Self::as_slice) and change when keys are
    /// removed or the set is sorted.
    #[inline]
    pub fn position(&self, key: K) -> Option<usize> {
        self.occupant_pos(key.index())
            .filter(|&pos| self.dense[pos] == key)
    }

    /// Whether `key` is in the set.
    #[inline]
    pub fn contains(&self, key: K) -> bool {
        self.position(key).is_some()
    }

    /// The key occupying slot `index`, if any.
    ///
    /// With generational keys this finds whichever generation is present.
    #[inline]
    pub fn occupant(&self, index: usize) -> Option<&K> {
        self.occupant_pos(index).map(|pos| &self.dense[pos])
    }

    /// Adds `key` and returns `true`, or returns `false` and leaves the set
    /// unchanged if its slot is already held (by `key` or by another key with
    /// the same index).
    ///
    /// # Panics
    ///
    /// Panics if the set already holds `u32::MAX` keys, or if a page or the
    /// directory cannot be allocated.
    pub fn insert(&mut self, key: K) -> bool {
        let index = key.index();
        if self.occupant_pos(index).is_some() {
            return false;
        }
        self.insert_vacant(key, index);
        true
    }

    /// Appends `key`, whose slot `index` the caller has checked is vacant.
    ///
    /// If this panics, the set is unchanged as far as lookups can tell, and
    /// `len` is unchanged.
    pub(crate) fn insert_vacant(&mut self, key: K, index: usize) {
        let pos = self.dense.len();
        let slot = encode(pos).expect("PagedSparseSet cannot hold more than u32::MAX keys");
        let (page, offset) = split::<PAGE>(index);
        self.sparse.get_or_alloc(page)[offset] = Some(slot);
        // If this push panics, the slot points one past the dense array and is
        // ignored by every lookup.
        self.dense.push(key);
    }

    /// Adds `key`, overwriting any key that holds its slot, and returns the
    /// key it overwrote.
    ///
    /// An overwritten key keeps its dense position.
    ///
    /// # Panics
    ///
    /// As [`insert`](Self::insert).
    pub fn replace(&mut self, key: K) -> Option<K> {
        let index = key.index();
        match self.occupant_pos(index) {
            Some(pos) => Some(self.overwrite_at(pos, key)),
            None => {
                self.insert_vacant(key, index);
                None
            }
        }
    }

    /// Replaces the key at dense position `pos` with `key`, which must have
    /// the same slot index, and returns the old key.
    #[inline]
    pub(crate) fn overwrite_at(&mut self, pos: usize, key: K) -> K {
        debug_assert_eq!(self.dense[pos].index(), key.index());
        core::mem::replace(&mut self.dense[pos], key)
    }

    /// Removes the key at dense position `pos` by moving the last key into
    /// its place, as `Vec::swap_remove` does.
    pub(crate) fn remove_at(&mut self, pos: usize) -> K {
        // Read both slot indices before changing anything, so that nothing
        // after the first mutation calls into `Key`.
        let index = self.dense[pos].index();
        let moved_index = match self.dense.last() {
            Some(last) if self.dense.len() - 1 != pos => Some(last.index()),
            _ => None,
        };
        let key = self.dense.swap_remove(pos);
        self.set_slot(index, None);
        if let Some(moved_index) = moved_index {
            self.set_slot(moved_index, encode(pos));
        }
        key
    }

    /// Removes `key` and returns whether it was present.
    ///
    /// The last key moves into the removed key's position.
    pub fn remove(&mut self, key: K) -> bool {
        match self.position(key) {
            Some(pos) => {
                self.remove_at(pos);
                true
            }
            None => false,
        }
    }

    /// Removes whichever key occupies slot `index` and returns it.
    pub fn remove_occupant(&mut self, index: usize) -> Option<K> {
        self.occupant_pos(index).map(|pos| self.remove_at(pos))
    }

    /// Keeps only the keys for which `keep` returns `true`.
    ///
    /// `keep` is called once per key. Kept keys may change position.
    pub fn retain(&mut self, mut keep: impl FnMut(&K) -> bool) {
        let mut pos = 0;
        while pos < self.dense.len() {
            if keep(&self.dense[pos]) {
                pos += 1;
            } else {
                // The key moved into `pos` has not been visited yet.
                self.remove_at(pos);
            }
        }
    }

    /// Removes every key in O(1).
    ///
    /// Pages stay allocated for reuse; use [`reset`](Self::reset) to free
    /// them.
    pub fn clear(&mut self) {
        self.dense.clear();
    }

    /// Removes every key and frees every page. The dense array keeps its
    /// capacity.
    pub fn reset(&mut self) {
        self.dense.clear();
        self.sparse.clear();
    }

    /// Frees every page without touching the dense array, which the caller
    /// must have emptied.
    pub(crate) fn free_pages(&mut self) {
        debug_assert!(self.dense.is_empty());
        self.sparse.clear();
    }

    /// Shrinks the dense array and the page directory. Frees every page if
    /// the set is empty.
    pub fn shrink_to_fit(&mut self) {
        self.dense.shrink_to_fit();
        if self.dense.is_empty() {
            self.sparse.clear();
        }
        self.sparse.shrink_to_fit();
    }

    /// The keys as a slice, in dense order.
    #[inline]
    pub fn as_slice(&self) -> &[K] {
        &self.dense
    }

    /// Iterates over the keys in dense order.
    #[inline]
    pub fn iter(&self) -> core::slice::Iter<'_, K> {
        self.dense.iter()
    }

    /// Rewrites every slot after the dense array was reordered.
    fn reindex(&mut self) {
        for pos in 0..self.dense.len() {
            let index = self.dense[pos].index();
            self.set_slot(index, encode(pos));
        }
    }

    /// Reorders the dense array with `reorder`, then rewrites the index.
    ///
    /// The index is rewritten even if `reorder` panics, so a caught panic in
    /// a user comparator leaves the set consistent (in some order).
    pub(crate) fn reorder_dense(&mut self, reorder: impl FnOnce(&mut [K])) {
        struct Reindex<'a, K: Key, const PAGE: usize, D: Directory>(
            &'a mut PagedSparseSet<K, PAGE, D>,
        );
        impl<K: Key, const PAGE: usize, D: Directory> Drop for Reindex<'_, K, PAGE, D> {
            fn drop(&mut self) {
                self.0.reindex();
            }
        }
        let guard = Reindex(self);
        reorder(&mut guard.0.dense);
    }

    /// Sorts the dense array by `compare` (unstable) and updates the index.
    ///
    /// If `compare` panics, the keys are left in an unspecified order but the
    /// set stays consistent.
    pub fn sort_unstable_by(&mut self, compare: impl FnMut(&K, &K) -> core::cmp::Ordering) {
        self.reorder_dense(|dense| dense.sort_unstable_by(compare));
    }

    /// Sorts the dense array by a key function (unstable) and updates the
    /// index.
    ///
    /// If `f` panics, the keys are left in an unspecified order but the set
    /// stays consistent.
    pub fn sort_unstable_by_key<T: Ord>(&mut self, f: impl FnMut(&K) -> T) {
        self.reorder_dense(|dense| dense.sort_unstable_by_key(f));
    }

    /// Sorts the dense array in ascending order (unstable) and updates the
    /// index.
    pub fn sort_unstable(&mut self)
    where
        K: Ord,
    {
        self.reorder_dense(<[K]>::sort_unstable);
    }
}

impl<K: Key, const PAGE: usize, D: Directory> Default for PagedSparseSet<K, PAGE, D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Key, const PAGE: usize, D: Directory> Clone for PagedSparseSet<K, PAGE, D>
where
    D::Store<Slot, PAGE>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            dense: self.dense.clone(),
            sparse: self.sparse.clone(),
        }
    }
}

impl<K: Key + fmt::Debug, const PAGE: usize, D: Directory> fmt::Debug
    for PagedSparseSet<K, PAGE, D>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.dense.iter()).finish()
    }
}

/// Sets are equal when they hold the same keys, in any order.
impl<K: Key, const PAGE: usize, D: Directory> PartialEq for PagedSparseSet<K, PAGE, D> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().all(|&key| other.contains(key))
    }
}

impl<K: Key, const PAGE: usize, D: Directory> Eq for PagedSparseSet<K, PAGE, D> {}

impl<K: Key, const PAGE: usize, D: Directory> Extend<K> for PagedSparseSet<K, PAGE, D> {
    fn extend<I: IntoIterator<Item = K>>(&mut self, iter: I) {
        for key in iter {
            self.insert(key);
        }
    }
}

impl<'a, K: Key + 'a, const PAGE: usize, D: Directory> Extend<&'a K>
    for PagedSparseSet<K, PAGE, D>
{
    fn extend<I: IntoIterator<Item = &'a K>>(&mut self, iter: I) {
        self.extend(iter.into_iter().copied());
    }
}

impl<K: Key, const PAGE: usize, D: Directory> FromIterator<K> for PagedSparseSet<K, PAGE, D> {
    fn from_iter<I: IntoIterator<Item = K>>(iter: I) -> Self {
        let mut set = Self::new();
        set.extend(iter);
        set
    }
}

impl<'a, K: Key, const PAGE: usize, D: Directory> IntoIterator for &'a PagedSparseSet<K, PAGE, D> {
    type Item = &'a K;
    type IntoIter = core::slice::Iter<'a, K>;

    fn into_iter(self) -> Self::IntoIter {
        self.dense.iter()
    }
}

impl<K: Key, const PAGE: usize, D: Directory> IntoIterator for PagedSparseSet<K, PAGE, D> {
    type Item = K;
    type IntoIter = alloc::vec::IntoIter<K>;

    fn into_iter(self) -> Self::IntoIter {
        self.dense.into_iter()
    }
}

#[cfg(feature = "serde")]
mod serde_impl {
    use super::*;
    use core::marker::PhantomData;
    use serde::de::{Deserialize, Deserializer, SeqAccess, Visitor};
    use serde::ser::{Serialize, Serializer};

    /// Serializes as a sequence of keys in dense order.
    impl<K: Key + Serialize, const PAGE: usize, D: Directory> Serialize for PagedSparseSet<K, PAGE, D> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            serializer.collect_seq(self.dense.iter())
        }
    }

    /// Deserializes from a sequence of keys. A key whose slot is already
    /// taken is skipped, as `HashSet` skips duplicates.
    ///
    /// Memory follows the key indices, not the input length: with the
    /// [`Flat`] directory one large index allocates a directory entry for
    /// every page below it, and an index near `usize::MAX` panics. For
    /// untrusted input, use [`Tree`](crate::page::Tree) or validate the
    /// keys first.
    impl<'de, K: Key + Deserialize<'de>, const PAGE: usize, D: Directory> Deserialize<'de>
        for PagedSparseSet<K, PAGE, D>
    {
        fn deserialize<De: Deserializer<'de>>(deserializer: De) -> Result<Self, De::Error> {
            struct SeqVisitor<K, const PAGE: usize, D>(PhantomData<(K, D)>);

            impl<'de, K: Key + Deserialize<'de>, const PAGE: usize, D: Directory> Visitor<'de>
                for SeqVisitor<K, PAGE, D>
            {
                type Value = PagedSparseSet<K, PAGE, D>;

                fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    f.write_str("a sequence of keys")
                }

                fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                    // Cap the hint so a hostile length cannot force a huge
                    // allocation.
                    let hint = seq.size_hint().unwrap_or(0).min(4096);
                    let mut set = PagedSparseSet::with_capacity(hint);
                    while let Some(key) = seq.next_element()? {
                        set.insert(key);
                    }
                    Ok(set)
                }
            }

            deserializer.deserialize_seq(SeqVisitor::<K, PAGE, D>(PhantomData))
        }
    }
}

// The tests use `u32` keys, which implement `Key` only on 32- and 64-bit
// targets.
#[cfg(all(test, any(target_pointer_width = "32", target_pointer_width = "64")))]
mod tests {
    use super::*;
    #[cfg(target_pointer_width = "64")]
    use crate::page::Tree;

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    struct Gen(u32);

    impl Key for Gen {
        fn index(self) -> usize {
            (self.0 & 0xffff) as usize
        }
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn wide_keys_allocate_few_pages() {
        let mut set: PagedSparseSet<u64, 64, Tree> = PagedSparseSet::new();
        for k in [0, u64::MAX, u64::MAX / 2, 1 << 40] {
            assert!(set.insert(k));
        }
        assert_eq!(set.allocated_pages(), 4);
        assert!(set.contains(u64::MAX));
        assert!(!set.contains(u64::MAX - 1));
    }

    #[test]
    fn remove_moves_last_key() {
        let mut set: PagedSparseSet<u32, 4> = [1, 2, 3, 40].into_iter().collect();
        assert!(set.remove(1));
        assert_eq!(set.as_slice(), &[40, 2, 3]);
        assert_eq!(set.position(40), Some(0));
        assert!(!set.remove(1));
        assert!(!set.contains(1));
    }

    #[test]
    fn clear_is_lazy_but_correct() {
        let mut set: PagedSparseSet<u32, 4> = (0..10).collect();
        set.clear();
        assert!(set.is_empty());
        assert_eq!(set.allocated_pages(), 3);
        for k in 0..10 {
            assert!(!set.contains(k));
        }
        assert!(set.insert(9));
        assert!(!set.contains(0));
        assert_eq!(set.position(9), Some(0));
        set.reset();
        assert_eq!(set.allocated_pages(), 0);
    }

    #[test]
    fn generations_compete_for_one_slot() {
        let old = Gen(7);
        let new = Gen(7 | 1 << 16);
        let mut set: PagedSparseSet<Gen> = PagedSparseSet::new();
        assert!(set.insert(old));
        assert!(!set.insert(new));
        assert!(!set.contains(new));
        assert_eq!(set.occupant(7), Some(&old));
        assert_eq!(set.replace(new), Some(old));
        assert!(set.contains(new) && !set.contains(old));
        assert!(!set.remove(old));
        assert_eq!(set.remove_occupant(7), Some(new));
        assert!(set.is_empty());
    }

    #[test]
    fn retain_and_sort_keep_index_consistent() {
        let mut set: PagedSparseSet<u32, 8> = [9, 3, 7, 1, 8, 2].into_iter().collect();
        set.retain(|&k| k % 3 != 0);
        set.sort_unstable();
        assert_eq!(set.as_slice(), &[1, 2, 7, 8]);
        for (pos, &k) in set.as_slice().iter().enumerate() {
            assert_eq!(set.position(k), Some(pos));
        }
        assert!(!set.contains(3) && !set.contains(9));
    }

    #[test]
    fn equality_ignores_order() {
        let a: PagedSparseSet<u32> = [1, 2, 3].into_iter().collect();
        let b: PagedSparseSet<u32> = [3, 1, 2].into_iter().collect();
        let c: PagedSparseSet<u32> = [1, 2].into_iter().collect();
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
