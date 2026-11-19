//! [`PagedSparseSet`]: a sparse set whose sparse index is paged.
//!
//! A sparse set keeps its keys in a contiguous *dense* array and finds them
//! through a *sparse* index from slot index to dense position. Here the sparse
//! index is split into pages that are allocated on first use, so memory
//! follows the key ranges actually used instead of the largest key.
//!
//! | Operation | Cost |
//! |---|---|
//! | [`insert`](PagedSparseSet::insert), [`remove`](PagedSparseSet::remove), [`contains`](PagedSparseSet::contains) | O(1) (O(depth) with [`Radix`](crate::page::Radix), O(log pages) with [`Tree`](crate::page::Tree)) |
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
//!
//! With the `rayon` feature, the borrowed parallel iterator is indexed and
//! follows dense order.
//!
//! ```rust
//! # #[cfg(feature = "rayon")] {
//! use paged_collections::sparse_set::PagedSparseSet;
//! use rayon::prelude::*;
//!
//! let set: PagedSparseSet<u32> = [7, 2, 9].into_iter().collect();
//! assert_eq!(set.par_iter().copied().collect::<Vec<_>>(), [7, 2, 9]);
//! # }
//! ```

use alloc::collections::TryReserveError;
use alloc::vec::Vec;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::iter::{Chain, FusedIterator};
use core::num::NonZeroU32;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Sub, SubAssign};
use core::slice;

use crate::hash::hash_unordered;
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
/// - `D` picks the page directory: [`Flat`] (default) for dense keys,
///   [`Radix`](crate::page::Radix) for keys spread over the 64-bit range,
///   or [`Tree`](crate::page::Tree). See [`crate::page`].
///
/// The set holds at most one key per slot index. With plain integer keys that
/// is the usual set semantics. With generational keys, [`insert`] refuses a
/// key whose slot is held by another generation, and [`replace`] overwrites
/// it.
///
/// Keys are kept in insertion order until a removal, which moves the last key
/// into the gap.
///
/// # Set operations
///
/// [`union`](Self::union), [`intersection`](Self::intersection),
/// [`difference`](Self::difference), and
/// [`symmetric_difference`](Self::symmetric_difference) iterate over borrowed
/// keys. The corresponding operators `|`, `&`, `-`, and `^` on borrowed sets
/// create a set with the left operand's page size and directory. The assignment
/// operators `|=`, `&=`, `-=`, and `^=` update a set in place. Removals can
/// reorder keys by moving the last key into each gap.
/// [`is_disjoint`](Self::is_disjoint), [`is_subset`](Self::is_subset), and
/// [`is_superset`](Self::is_superset) test relationships between sets.
///
/// ```
/// use paged_collections::page::Radix;
/// use paged_collections::sparse_set::PagedSparseSet;
///
/// let mut a: PagedSparseSet<u16, 4> = [1, 2].into_iter().collect();
/// let b: PagedSparseSet<u16, 64, Radix> = [2, 3].into_iter().collect();
/// assert_eq!((&a | &b).as_slice(), &[1, 2, 3]);
/// a &= &b;
/// assert!(a.is_subset(&b));
/// ```
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

    /// Iterates over keys in `self` absent from `other`, in `self`'s dense order.
    ///
    /// ```
    /// use paged_collections::sparse_set::PagedSparseSet;
    ///
    /// let a: PagedSparseSet<u16> = [3, 1, 2].into_iter().collect();
    /// let b: PagedSparseSet<u16> = [1, 4].into_iter().collect();
    /// assert!(a.difference(&b).copied().eq([3, 2]));
    /// ```
    pub fn difference<'a, const P2: usize, D2: Directory>(
        &'a self,
        other: &'a PagedSparseSet<K, P2, D2>,
    ) -> Difference<'a, K, P2, D2> {
        Difference {
            iter: self.iter(),
            other,
        }
    }

    /// Iterates over keys present in both sets.
    ///
    /// Order follows the smaller set's dense order, or `self`'s dense order
    /// if the sets contain the same number of keys.
    ///
    /// ```
    /// use paged_collections::sparse_set::PagedSparseSet;
    ///
    /// let a: PagedSparseSet<u16> = [1, 2, 3].into_iter().collect();
    /// let b: PagedSparseSet<u16> = [2, 1].into_iter().collect();
    /// assert!(a.intersection(&b).copied().eq([2, 1]));
    /// ```
    pub fn intersection<'a, const P2: usize, D2: Directory>(
        &'a self,
        other: &'a PagedSparseSet<K, P2, D2>,
    ) -> Intersection<'a, K, PAGE, D, P2, D2> {
        let (iter, probe) = if self.len() <= other.len() {
            (self.iter(), IntersectionProbe::Right(other))
        } else {
            (other.iter(), IntersectionProbe::Left(self))
        };
        Intersection { iter, probe }
    }

    /// Iterates over keys present in exactly one set.
    ///
    /// Yields [`self.difference(other)`](Self::difference), then
    /// `other.difference(self)`. Different generations of the same slot are
    /// unequal, so both are yielded. Collecting into a `PagedSparseSet` keeps
    /// `self`'s key because [`insert`](Self::insert) refuses an occupied slot.
    ///
    /// ```
    /// use paged_collections::sparse_set::PagedSparseSet;
    ///
    /// let a: PagedSparseSet<u16> = [3, 1, 2].into_iter().collect();
    /// let b: PagedSparseSet<u16> = [2, 5, 4].into_iter().collect();
    /// assert!(a.symmetric_difference(&b).copied().eq([3, 1, 5, 4]));
    /// ```
    pub fn symmetric_difference<'a, const P2: usize, D2: Directory>(
        &'a self,
        other: &'a PagedSparseSet<K, P2, D2>,
    ) -> SymmetricDifference<'a, K, PAGE, D, P2, D2> {
        SymmetricDifference {
            iter: self.difference(other).chain(other.difference(self)),
        }
    }

    /// Iterates over keys present in either set.
    ///
    /// Yields `self`'s keys in dense order, then `other.difference(self)`.
    /// Different generations of the same slot are unequal, so both are
    /// yielded. Collecting into a `PagedSparseSet` keeps `self`'s key because
    /// [`insert`](Self::insert) refuses an occupied slot.
    ///
    /// ```
    /// use paged_collections::sparse_set::PagedSparseSet;
    ///
    /// let a: PagedSparseSet<u16> = [3, 1, 2].into_iter().collect();
    /// let b: PagedSparseSet<u16> = [2, 5, 4].into_iter().collect();
    /// assert!(a.union(&b).copied().eq([3, 1, 2, 5, 4]));
    /// ```
    pub fn union<'a, const P2: usize, D2: Directory>(
        &'a self,
        other: &'a PagedSparseSet<K, P2, D2>,
    ) -> Union<'a, K, PAGE, D> {
        Union {
            iter: self.iter().chain(other.difference(self)),
        }
    }

    /// Whether the sets have no keys in common. Iterates over the smaller set.
    ///
    /// ```
    /// use paged_collections::sparse_set::PagedSparseSet;
    ///
    /// let a: PagedSparseSet<u16> = [1, 2].into_iter().collect();
    /// let b: PagedSparseSet<u16> = [3].into_iter().collect();
    /// assert!(a.is_disjoint(&b));
    /// ```
    pub fn is_disjoint<const P2: usize, D2: Directory>(
        &self,
        other: &PagedSparseSet<K, P2, D2>,
    ) -> bool {
        self.intersection(other).next().is_none()
    }

    /// Whether every key in `self` is also in `other`.
    ///
    /// ```
    /// use paged_collections::sparse_set::PagedSparseSet;
    ///
    /// let a: PagedSparseSet<u16> = [2].into_iter().collect();
    /// let b: PagedSparseSet<u16> = [1, 2].into_iter().collect();
    /// assert!(a.is_subset(&b));
    /// ```
    pub fn is_subset<const P2: usize, D2: Directory>(
        &self,
        other: &PagedSparseSet<K, P2, D2>,
    ) -> bool {
        self.len() <= other.len() && self.iter().all(|&key| other.contains(key))
    }

    /// Whether every key in `other` is also in `self`.
    ///
    /// ```
    /// use paged_collections::sparse_set::PagedSparseSet;
    ///
    /// let a: PagedSparseSet<u16> = [1, 2].into_iter().collect();
    /// let b: PagedSparseSet<u16> = [2].into_iter().collect();
    /// assert!(a.is_superset(&b));
    /// ```
    pub fn is_superset<const P2: usize, D2: Directory>(
        &self,
        other: &PagedSparseSet<K, P2, D2>,
    ) -> bool {
        other.is_subset(self)
    }

    /// Swaps the keys at dense positions `a` and `b` and fixes their index
    /// slots. Panics if either is out of bounds.
    pub(crate) fn swap_positions(&mut self, a: usize, b: usize) {
        // Validate both positions and read indices before changing anything.
        let a_index = self.dense[a].index();
        let b_index = self.dense[b].index();
        self.dense.swap(a, b);
        self.set_slot(a_index, encode(b));
        self.set_slot(b_index, encode(a));
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

    /// Sorts the dense array by `compare`, keeping the order of equal keys,
    /// and updates the index.
    ///
    /// If `compare` panics, the keys are left in an unspecified order but the
    /// set stays consistent.
    pub fn sort_by(&mut self, compare: impl FnMut(&K, &K) -> core::cmp::Ordering) {
        self.reorder_dense(|dense| dense.sort_by(compare));
    }

    /// Sorts the dense array by a key function, keeping the order of keys
    /// with equal sort keys, and updates the index.
    ///
    /// If `f` panics, the keys are left in an unspecified order but the set
    /// stays consistent.
    pub fn sort_by_key<T: Ord>(&mut self, f: impl FnMut(&K) -> T) {
        self.reorder_dense(|dense| dense.sort_by_key(f));
    }

    /// Sorts the dense array in ascending order and updates the index.
    ///
    /// Stable, which matters only if equal keys can be told apart; prefer
    /// [`sort_unstable`](Self::sort_unstable) otherwise: it allocates nothing.
    pub fn sort(&mut self)
    where
        K: Ord,
    {
        self.reorder_dense(<[K]>::sort);
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

/// Compares exact keys with [`contains`](PagedSparseSet::contains), in any
/// order and across page sizes and directories.
impl<K: Key, const PAGE: usize, D: Directory, const P2: usize, D2: Directory>
    PartialEq<PagedSparseSet<K, P2, D2>> for PagedSparseSet<K, PAGE, D>
{
    fn eq(&self, other: &PagedSparseSet<K, P2, D2>) -> bool {
        self.len() == other.len() && self.iter().all(|&key| other.contains(key))
    }
}

/// Exact key equality gives sets an equivalence relation under [`PartialEq`].
impl<K: Key, const PAGE: usize, D: Directory> Eq for PagedSparseSet<K, PAGE, D> {}

/// Hashes the keys in any order, consistent with [`PartialEq`].
impl<K: Key + Hash, const PAGE: usize, D: Directory> Hash for PagedSparseSet<K, PAGE, D> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_unordered(self.iter(), state);
    }
}

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

/// An iterator over the keys yielded by [`PagedSparseSet::difference`].
pub struct Difference<'a, K, const P2: usize, D2: Directory> {
    iter: slice::Iter<'a, K>,
    other: &'a PagedSparseSet<K, P2, D2>,
}

impl<'a, K: Key, const P2: usize, D2: Directory> Iterator for Difference<'a, K, P2, D2> {
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.find(|&&key| !self.other.contains(key))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.iter.len()))
    }
}

impl<K: Key, const P2: usize, D2: Directory> FusedIterator for Difference<'_, K, P2, D2> {}

impl<K, const P2: usize, D2: Directory> Clone for Difference<'_, K, P2, D2> {
    fn clone(&self) -> Self {
        Self {
            iter: self.iter.clone(),
            other: self.other,
        }
    }
}

impl<K: Key + fmt::Debug, const P2: usize, D2: Directory> fmt::Debug for Difference<'_, K, P2, D2> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

enum IntersectionProbe<'a, K, const PAGE: usize, D: Directory, const P2: usize, D2: Directory> {
    Left(&'a PagedSparseSet<K, PAGE, D>),
    Right(&'a PagedSparseSet<K, P2, D2>),
}

// Manual impls: a derive would require `K: Clone`.
impl<K, const PAGE: usize, D: Directory, const P2: usize, D2: Directory> Clone
    for IntersectionProbe<'_, K, PAGE, D, P2, D2>
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<K, const PAGE: usize, D: Directory, const P2: usize, D2: Directory> Copy
    for IntersectionProbe<'_, K, PAGE, D, P2, D2>
{
}

/// An iterator over the keys yielded by [`PagedSparseSet::intersection`].
pub struct Intersection<'a, K, const PAGE: usize, D: Directory, const P2: usize, D2: Directory> {
    iter: slice::Iter<'a, K>,
    probe: IntersectionProbe<'a, K, PAGE, D, P2, D2>,
}

impl<'a, K: Key, const PAGE: usize, D: Directory, const P2: usize, D2: Directory> Iterator
    for Intersection<'a, K, PAGE, D, P2, D2>
{
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        match self.probe {
            IntersectionProbe::Left(set) => self.iter.find(|&&key| set.contains(key)),
            IntersectionProbe::Right(set) => self.iter.find(|&&key| set.contains(key)),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.iter.len()))
    }
}

impl<K: Key, const PAGE: usize, D: Directory, const P2: usize, D2: Directory> FusedIterator
    for Intersection<'_, K, PAGE, D, P2, D2>
{
}

impl<K, const PAGE: usize, D: Directory, const P2: usize, D2: Directory> Clone
    for Intersection<'_, K, PAGE, D, P2, D2>
{
    fn clone(&self) -> Self {
        Self {
            iter: self.iter.clone(),
            probe: self.probe,
        }
    }
}

impl<K: Key + fmt::Debug, const PAGE: usize, D: Directory, const P2: usize, D2: Directory>
    fmt::Debug for Intersection<'_, K, PAGE, D, P2, D2>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

/// An iterator over the keys yielded by [`PagedSparseSet::symmetric_difference`].
pub struct SymmetricDifference<
    'a,
    K,
    const PAGE: usize,
    D: Directory,
    const P2: usize,
    D2: Directory,
> {
    iter: Chain<Difference<'a, K, P2, D2>, Difference<'a, K, PAGE, D>>,
}

impl<'a, K: Key, const PAGE: usize, D: Directory, const P2: usize, D2: Directory> Iterator
    for SymmetricDifference<'a, K, PAGE, D, P2, D2>
{
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<K: Key, const PAGE: usize, D: Directory, const P2: usize, D2: Directory> FusedIterator
    for SymmetricDifference<'_, K, PAGE, D, P2, D2>
{
}

impl<K, const PAGE: usize, D: Directory, const P2: usize, D2: Directory> Clone
    for SymmetricDifference<'_, K, PAGE, D, P2, D2>
{
    fn clone(&self) -> Self {
        Self {
            iter: self.iter.clone(),
        }
    }
}

impl<K: Key + fmt::Debug, const PAGE: usize, D: Directory, const P2: usize, D2: Directory>
    fmt::Debug for SymmetricDifference<'_, K, PAGE, D, P2, D2>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

/// An iterator over the keys yielded by [`PagedSparseSet::union`].
pub struct Union<'a, K, const PAGE: usize, D: Directory> {
    iter: Chain<slice::Iter<'a, K>, Difference<'a, K, PAGE, D>>,
}

impl<'a, K: Key, const PAGE: usize, D: Directory> Iterator for Union<'a, K, PAGE, D> {
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<K: Key, const PAGE: usize, D: Directory> FusedIterator for Union<'_, K, PAGE, D> {}

impl<K, const PAGE: usize, D: Directory> Clone for Union<'_, K, PAGE, D> {
    fn clone(&self) -> Self {
        Self {
            iter: self.iter.clone(),
        }
    }
}

impl<K: Key + fmt::Debug, const PAGE: usize, D: Directory> fmt::Debug for Union<'_, K, PAGE, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

/// Collects the [`union`](PagedSparseSet::union) into a new set.
impl<K: Key, const PAGE: usize, D: Directory, const P2: usize, D2: Directory>
    BitOr<&PagedSparseSet<K, P2, D2>> for &PagedSparseSet<K, PAGE, D>
{
    type Output = PagedSparseSet<K, PAGE, D>;

    fn bitor(self, other: &PagedSparseSet<K, P2, D2>) -> Self::Output {
        let mut set = PagedSparseSet::with_capacity(self.len());
        set.extend(self.union(other));
        set
    }
}

/// Collects the [`intersection`](PagedSparseSet::intersection) into a new set.
impl<K: Key, const PAGE: usize, D: Directory, const P2: usize, D2: Directory>
    BitAnd<&PagedSparseSet<K, P2, D2>> for &PagedSparseSet<K, PAGE, D>
{
    type Output = PagedSparseSet<K, PAGE, D>;

    fn bitand(self, other: &PagedSparseSet<K, P2, D2>) -> Self::Output {
        let mut set = PagedSparseSet::with_capacity(self.len().min(other.len()));
        set.extend(self.intersection(other));
        set
    }
}

/// Collects the [`difference`](PagedSparseSet::difference) into a new set.
impl<K: Key, const PAGE: usize, D: Directory, const P2: usize, D2: Directory>
    Sub<&PagedSparseSet<K, P2, D2>> for &PagedSparseSet<K, PAGE, D>
{
    type Output = PagedSparseSet<K, PAGE, D>;

    fn sub(self, other: &PagedSparseSet<K, P2, D2>) -> Self::Output {
        let mut set = PagedSparseSet::with_capacity(self.len().saturating_sub(other.len()));
        set.extend(self.difference(other));
        set
    }
}

/// Collects the [`symmetric difference`](PagedSparseSet::symmetric_difference) into a new set.
impl<K: Key, const PAGE: usize, D: Directory, const P2: usize, D2: Directory>
    BitXor<&PagedSparseSet<K, P2, D2>> for &PagedSparseSet<K, PAGE, D>
{
    type Output = PagedSparseSet<K, PAGE, D>;

    fn bitxor(self, other: &PagedSparseSet<K, P2, D2>) -> Self::Output {
        let mut set = PagedSparseSet::with_capacity(self.len().abs_diff(other.len()));
        set.extend(self.symmetric_difference(other));
        set
    }
}

/// Inserts the right operand's keys in dense order, keeping occupied slots.
impl<K: Key, const PAGE: usize, D: Directory, const P2: usize, D2: Directory>
    BitOrAssign<&PagedSparseSet<K, P2, D2>> for PagedSparseSet<K, PAGE, D>
{
    fn bitor_assign(&mut self, other: &PagedSparseSet<K, P2, D2>) {
        self.extend(other);
    }
}

/// Keeps the keys present in both sets. Swap removal can reorder keys.
impl<K: Key, const PAGE: usize, D: Directory, const P2: usize, D2: Directory>
    BitAndAssign<&PagedSparseSet<K, P2, D2>> for PagedSparseSet<K, PAGE, D>
{
    fn bitand_assign(&mut self, other: &PagedSparseSet<K, P2, D2>) {
        self.retain(|&key| other.contains(key));
    }
}

/// Removes keys present in the right operand. Swap removal can reorder keys.
impl<K: Key, const PAGE: usize, D: Directory, const P2: usize, D2: Directory>
    SubAssign<&PagedSparseSet<K, P2, D2>> for PagedSparseSet<K, PAGE, D>
{
    fn sub_assign(&mut self, other: &PagedSparseSet<K, P2, D2>) {
        if other.len() < self.len() {
            for &key in other {
                self.remove(key);
            }
        } else {
            self.retain(|&key| !other.contains(key));
        }
    }
}

/// Toggles each key in the right operand. Swap removal can reorder keys.
/// When the operands hold different generations of the same slot, the left
/// operand keeps its key and skips the right operand's key.
impl<K: Key, const PAGE: usize, D: Directory, const P2: usize, D2: Directory>
    BitXorAssign<&PagedSparseSet<K, P2, D2>> for PagedSparseSet<K, PAGE, D>
{
    fn bitxor_assign(&mut self, other: &PagedSparseSet<K, P2, D2>) {
        for &key in other {
            if !self.remove(key) {
                self.insert(key);
            }
        }
    }
}

#[cfg(feature = "rayon")]
mod rayon_impl {
    use super::*;
    use rayon::prelude::*;

    impl<'a, K: Key + Sync, const PAGE: usize, D: Directory> IntoParallelIterator
        for &'a PagedSparseSet<K, PAGE, D>
    {
        type Item = &'a K;
        type Iter = rayon::slice::Iter<'a, K>;

        fn into_par_iter(self) -> Self::Iter {
            self.as_slice().par_iter()
        }
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
    /// untrusted input, use [`Radix`](crate::page::Radix) or
    /// [`Tree`](crate::page::Tree), or validate the keys first.
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
    use crate::page::Radix;
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
    fn swap_positions_reindexes_both_keys() {
        let old = Gen(2);
        let new = Gen(2 | 1 << 16);
        let mut set: PagedSparseSet<Gen, 4> = [Gen(1), old, Gen(40)].into_iter().collect();
        assert_eq!(set.replace(new), Some(old));
        set.swap_positions(1, 2);
        assert_eq!(set.as_slice(), &[Gen(1), Gen(40), new]);
        for (pos, &key) in set.iter().enumerate() {
            assert_eq!(set.position(key), Some(pos));
        }
        assert!(!set.contains(old));
        set.swap_positions(0, 0);
        assert_eq!(set.position(Gen(1)), Some(0));
        assert!(set.remove(Gen(40)));
        assert_eq!(set.position(new), Some(1));
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
    fn stable_sorts_keep_ties_in_order() {
        let mut set: PagedSparseSet<u32, 4> = [5, 9, 2, 7, 40, 12].into_iter().collect();
        set.sort_by_key(|&k| k % 2);
        assert_eq!(set.as_slice(), &[2, 40, 12, 5, 9, 7]);
        set.sort_by(|a, b| (b / 10).cmp(&(a / 10)));
        assert_eq!(set.as_slice(), &[40, 12, 2, 5, 9, 7]);
        for (pos, &key) in set.as_slice().iter().enumerate() {
            assert_eq!(set.position(key), Some(pos));
        }
        set.sort();
        assert_eq!(set.as_slice(), &[2, 5, 7, 9, 12, 40]);
        assert_eq!(set.position(12), Some(4));
        assert_eq!(set.as_slice().binary_search(&9), Ok(3));
    }

    #[test]
    fn equality_ignores_order() {
        let a: PagedSparseSet<u32> = [1, 2, 3].into_iter().collect();
        let b: PagedSparseSet<u32> = [3, 1, 2].into_iter().collect();
        let c: PagedSparseSet<u32> = [1, 2].into_iter().collect();
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn set_algebra_preserves_the_left_generation_when_stored() {
        let old = Gen(7);
        let new = Gen(7 | 1 << 16);
        let common = Gen(42);
        let left_only = Gen(13);
        let right_only = Gen(21);
        let left: PagedSparseSet<Gen, 4> = [old, common, left_only].into_iter().collect();
        let right: PagedSparseSet<Gen, 64, Radix> = [new, common, right_only].into_iter().collect();

        let union = left.union(&right).copied();
        assert!(union.clone().eq([old, common, left_only, new, right_only]));
        assert_eq!(
            union.collect::<PagedSparseSet<Gen>>().occupant(7),
            Some(&old)
        );
        assert_eq!(
            (&left | &right).as_slice(),
            &[old, common, left_only, right_only]
        );
        assert_eq!((&right | &left).occupant(7), Some(&new));

        let symmetric = left.symmetric_difference(&right).copied();
        assert!(symmetric.clone().eq([old, left_only, new, right_only]));
        assert_eq!(
            symmetric.collect::<PagedSparseSet<Gen>>().occupant(7),
            Some(&old)
        );
        assert_eq!((&left ^ &right).as_slice(), &[old, left_only, right_only]);
        assert_eq!((&right ^ &left).occupant(7), Some(&new));

        let mut assigned = left;
        assigned ^= &right;
        assert_eq!(assigned.as_slice(), &[old, left_only, right_only]);
        assert_eq!(assigned.occupant(7), Some(&old));
        assert!(!assigned.contains(new));
        assert!(!assigned.contains(common));
    }

    #[test]
    fn set_algebra_across_page_sizes_and_directories() {
        let left: PagedSparseSet<u32, 4> = [90, 7, 1, 4].into_iter().collect();
        let right: PagedSparseSet<u32, 64, Radix> = [1, 7, 128].into_iter().collect();
        assert!(left.difference(&right).copied().eq([90, 4]));
        assert!(left.intersection(&right).copied().eq([1, 7]));
        assert!(right.intersection(&left).copied().eq([1, 7]));
        assert!(left.union(&right).copied().eq([90, 7, 1, 4, 128]));
        assert!(left.symmetric_difference(&right).copied().eq([90, 4, 128]));

        let tie: PagedSparseSet<u32, 64, Radix> = [1, 7, 128, 256].into_iter().collect();
        assert!(left.intersection(&tie).copied().eq([7, 1]));

        let union: PagedSparseSet<u32, 4> = &left | &right;
        let intersection: PagedSparseSet<u32, 4> = &left & &right;
        let difference: PagedSparseSet<u32, 4> = &left - &right;
        let symmetric: PagedSparseSet<u32, 4> = &left ^ &right;
        assert_eq!(union.as_slice(), &[90, 7, 1, 4, 128]);
        assert_eq!(intersection.as_slice(), &[1, 7]);
        assert_eq!(difference.as_slice(), &[90, 4]);
        assert_eq!(symmetric.as_slice(), &[90, 4, 128]);

        let mut assigned = left.clone();
        assigned |= &right;
        assert_eq!(assigned, union);
        let mut assigned = left.clone();
        assigned &= &right;
        assert_eq!(assigned, intersection);
        let mut assigned = left.clone();
        assigned -= &right;
        assert_eq!(assigned, difference);
        let mut assigned = left;
        assigned ^= &right;
        assert_eq!(assigned, symmetric);
    }

    #[test]
    fn empty_sets_and_set_relationships() {
        let empty = PagedSparseSet::<u32, 4>::new();
        let set: PagedSparseSet<u32, 64, Radix> = [2, 1].into_iter().collect();
        let subset: PagedSparseSet<u32, 4> = [1].into_iter().collect();
        let disjoint: PagedSparseSet<u32, 4> = [3, 4, 5].into_iter().collect();

        assert_eq!(empty.difference(&set).next(), None);
        assert!(set.difference(&empty).copied().eq([2, 1]));
        assert_eq!(set.intersection(&empty).next(), None);
        assert!(empty.union(&set).copied().eq([2, 1]));
        assert!(set.symmetric_difference(&empty).copied().eq([2, 1]));
        assert_eq!(empty.union(&empty).next(), None);
        assert_eq!(empty.symmetric_difference(&empty).next(), None);

        assert!(empty.is_subset(&empty) && empty.is_superset(&empty));
        assert!(empty.is_disjoint(&empty));
        assert!(empty.is_subset(&set) && set.is_superset(&empty));
        assert!(!set.is_subset(&empty) && !empty.is_superset(&set));
        assert!(empty.is_disjoint(&set) && set.is_disjoint(&empty));
        assert!(subset.is_subset(&set) && set.is_superset(&subset));
        assert!(!set.is_subset(&subset) && !subset.is_superset(&set));
        assert!(!subset.is_disjoint(&set) && !set.is_disjoint(&subset));
        assert!(set.is_subset(&set) && set.is_superset(&set));
        assert!(!set.is_disjoint(&set));
        assert!(set.is_disjoint(&disjoint) && disjoint.is_disjoint(&set));
        assert!(!set.is_subset(&disjoint) && !disjoint.is_superset(&set));
    }

    #[test]
    fn set_iterators_report_remaining_bounds_and_keys() {
        fn assert_exhausted(mut iter: impl FusedIterator) {
            assert!(iter.next().is_none());
            assert_eq!(iter.size_hint(), (0, Some(0)));
            assert!(iter.next().is_none());
        }

        let left: PagedSparseSet<u32, 4> = [4, 2, 7, 3].into_iter().collect();
        let right: PagedSparseSet<u32, 64, Radix> = [7, 2, 5].into_iter().collect();

        let mut difference = left.difference(&right);
        assert_eq!(difference.size_hint(), (0, Some(4)));
        assert_eq!(difference.next(), Some(&4));
        assert_eq!(difference.size_hint(), (0, Some(3)));
        assert_eq!(alloc::format!("{difference:?}"), "[3]");
        assert_eq!(difference.next(), Some(&3));
        assert_exhausted(difference);

        let mut intersection = left.intersection(&right);
        assert_eq!(intersection.size_hint(), (0, Some(3)));
        assert_eq!(intersection.next(), Some(&7));
        assert_eq!(intersection.size_hint(), (0, Some(2)));
        assert_eq!(alloc::format!("{intersection:?}"), "[2]");
        assert_eq!(intersection.next(), Some(&2));
        assert_exhausted(intersection);

        let mut union = left.union(&right);
        assert_eq!(union.size_hint(), (4, Some(7)));
        assert_eq!(union.next(), Some(&4));
        assert_eq!(union.size_hint(), (3, Some(6)));
        assert_eq!(alloc::format!("{union:?}"), "[2, 7, 3, 5]");
        assert!(union.by_ref().take(3).copied().eq([2, 7, 3]));
        assert_eq!(union.size_hint(), (0, Some(3)));
        assert_eq!(union.next(), Some(&5));
        assert_exhausted(union);

        let mut symmetric = left.symmetric_difference(&right);
        assert_eq!(symmetric.size_hint(), (0, Some(7)));
        assert_eq!(symmetric.next(), Some(&4));
        assert_eq!(symmetric.size_hint(), (0, Some(6)));
        assert_eq!(alloc::format!("{symmetric:?}"), "[3, 5]");
        assert_eq!(symmetric.next(), Some(&3));
        assert_eq!(symmetric.size_hint(), (0, Some(3)));
        assert_eq!(symmetric.next(), Some(&5));
        assert_exhausted(symmetric);
    }
}
