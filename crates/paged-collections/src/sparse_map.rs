//! [`PagedSparseMap`]: a paged sparse set that stores one value per key.
//!
//! The map is a [`PagedSparseSet`] of keys plus a dense array of values kept
//! parallel to the dense keys: the value at position `i` belongs to the key at
//! position `i`. Both arrays are contiguous, so keys and values can be
//! iterated, or handed out as slices, without touching the sparse index.
//!
//! | Operation | Cost |
//! |---|---|
//! | [`get`](PagedSparseMap::get), [`remove`](PagedSparseMap::remove) | O(1) (O(log pages) with [`Tree`](crate::page::Tree)) |
//! | [`insert`](PagedSparseMap::insert) | as `get`, amortized over dense-array growth; the first key in a page also allocates the page (and, with [`Flat`], grows the directory up to it) |
//! | [`clear`](PagedSparseMap::clear) | O(1) for the index, plus dropping the values |
//! | iteration, [`value_slice`](PagedSparseMap::value_slice) | over the dense arrays only |
//! | sorting | O(n log n) comparisons, O(n) swaps, then one index write per entry; allocates `n` indices |
//!
//! # Slots and generational keys
//!
//! Like the set, the map holds at most one key per slot index (see
//! [`crate::key`]). With plain integer keys the map behaves like `HashMap`.
//! With generational keys the methods differ in what they do when the slot is
//! held by a *different* key:
//!
//! | Method | Slot held by another key |
//! |---|---|
//! | [`insert`](PagedSparseMap::insert), [`get_or_insert_with`](PagedSparseMap::get_or_insert_with) | evicts it |
//! | [`replace`](PagedSparseMap::replace) | evicts it and returns the evicted entry |
//! | [`try_insert`](PagedSparseMap::try_insert) | refuses, like [`PagedSparseSet::insert`] |
//!
//! `insert` evicts so that, as with `HashMap`, `get(key)` returns the new
//! value afterwards. An evicted entry keeps its dense position.
//!
//! ```
//! use paged_collections::sparse_map::PagedSparseMap;
//!
//! let mut map: PagedSparseMap<u32, &str> = PagedSparseMap::new();
//! assert_eq!(map.insert(3_000_000, "far"), None);
//! assert_eq!(map.insert(7, "near"), None);
//! assert_eq!(map.insert(7, "nearer"), Some("near"));
//! assert_eq!(map.get(3_000_000), Some(&"far"));
//! assert_eq!(map.allocated_pages(), 2);
//!
//! assert_eq!(map.remove(3_000_000), Some("far"));
//! assert_eq!(map.key_slice(), &[7]);
//! assert_eq!(map.value_slice(), &["nearer"]);
//! ```

use alloc::vec::Vec;
use core::cmp::Ordering;
use core::fmt;
use core::iter::{FusedIterator, Zip};
use core::ops::Index;
use core::slice;

use crate::key::Key;
use crate::page::{DEFAULT_PAGE, Directory, Flat};
use crate::sparse_set::{PagedSparseSet, Slot};

/// A map from [`Key`]s to values with O(1) insert, remove and lookup, and
/// contiguous arrays of its keys and values.
///
/// `PAGE` and `D` are as for [`PagedSparseSet`]: slots per page of the
/// sparse index, and the page directory ([`Flat`] or
/// [`Tree`](crate::page::Tree)).
///
/// Entries are kept in insertion order until a removal, which moves the last
/// entry into the gap. See the [module docs](self) for how methods treat a
/// slot held by another generation of a key.
pub struct PagedSparseMap<K, V, const PAGE: usize = DEFAULT_PAGE, D: Directory = Flat> {
    // Invariant: `values.len() == keys.len()`, and `values[i]` belongs to
    // `keys.as_slice()[i]`. Every mutation below keeps it even when a
    // user-supplied closure, `V::drop` or an allocation panics.
    keys: PagedSparseSet<K, PAGE, D>,
    values: Vec<V>,
}

impl<K: Key, V, const PAGE: usize, D: Directory> PagedSparseMap<K, V, PAGE, D> {
    /// Number of slots per page of the sparse index.
    pub const PAGE_SIZE: usize = PAGE;

    /// Creates an empty map. No memory is allocated until the first insert.
    pub fn new() -> Self {
        Self {
            keys: PagedSparseSet::new(),
            values: Vec::new(),
        }
    }

    /// Creates an empty map whose dense arrays can hold `capacity` entries
    /// without reallocating. Pages are still allocated on demand.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            keys: PagedSparseSet::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
        }
    }

    /// Number of entries in the map.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the map has no entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Number of entries the dense arrays can hold without reallocating.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.keys.capacity().min(self.values.capacity())
    }

    /// Number of allocated pages in the sparse index.
    #[inline]
    pub fn allocated_pages(&self) -> usize {
        self.keys.allocated_pages()
    }

    /// Reserves room in the dense arrays for at least `additional` more
    /// entries.
    pub fn reserve(&mut self, additional: usize) {
        self.keys.reserve(additional);
        self.values.reserve(additional);
    }

    /// The keys as a set.
    ///
    /// Gives read access to the set's lookups, such as
    /// [`occupant`](PagedSparseSet::occupant).
    #[inline]
    pub fn as_set(&self) -> &PagedSparseSet<K, PAGE, D> {
        &self.keys
    }

    /// Dense position of `key`, if it is in the map.
    ///
    /// Positions index [`key_slice`](Self::key_slice) and
    /// [`value_slice`](Self::value_slice), and change when entries are removed
    /// or the map is sorted.
    #[inline]
    pub fn position(&self, key: K) -> Option<usize> {
        self.keys.position(key)
    }

    /// Whether `key` is in the map.
    #[inline]
    pub fn contains_key(&self, key: K) -> bool {
        self.keys.contains(key)
    }

    /// The value for `key`, if it is in the map.
    #[inline]
    pub fn get(&self, key: K) -> Option<&V> {
        self.position(key).map(|pos| &self.values[pos])
    }

    /// Mutable access to the value for `key`, if it is in the map.
    #[inline]
    pub fn get_mut(&mut self, key: K) -> Option<&mut V> {
        self.position(key).map(|pos| &mut self.values[pos])
    }

    /// The stored key equal to `key` and its value, if it is in the map.
    #[inline]
    pub fn get_key_value(&self, key: K) -> Option<(&K, &V)> {
        self.position(key).map(|pos| self.entry_at(pos))
    }

    /// The entry occupying slot `index`, if any.
    ///
    /// With generational keys this finds whichever generation is present.
    #[inline]
    pub fn occupant(&self, index: usize) -> Option<(&K, &V)> {
        self.keys.occupant_pos(index).map(|pos| self.entry_at(pos))
    }

    /// The entry occupying slot `index`, with mutable access to its value.
    #[inline]
    pub fn occupant_mut(&mut self, index: usize) -> Option<(&K, &mut V)> {
        let pos = self.keys.occupant_pos(index)?;
        Some((&self.keys.as_slice()[pos], &mut self.values[pos]))
    }

    #[inline]
    fn entry_at(&self, pos: usize) -> (&K, &V) {
        (&self.keys.as_slice()[pos], &self.values[pos])
    }

    /// Appends an entry whose slot `index` the caller has checked is vacant,
    /// and returns its dense position.
    fn push_vacant(&mut self, key: K, index: usize, value: V) -> usize {
        // Reserve first so the final push cannot fail: if anything before it
        // panics, neither array has grown.
        self.values.reserve(1);
        self.keys.insert_vacant(key, index);
        self.values.push(value);
        self.values.len() - 1
    }

    /// Stores `value` for `key` and returns the value previously in `key`'s
    /// slot.
    ///
    /// If `key` is present its value is replaced and the key is kept. If the
    /// slot is held by a different key (another generation), that entry is
    /// evicted and its value returned; use [`replace`](Self::replace) to get
    /// the evicted key too, or [`try_insert`](Self::try_insert) to refuse.
    ///
    /// # Panics
    ///
    /// Panics if the map already holds `u32::MAX` entries, or if a page, the
    /// directory or a dense array cannot be allocated. The entries are then
    /// unchanged, though capacity or allocated pages may have grown.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let index = key.index();
        match self.keys.occupant_pos(index) {
            Some(pos) => {
                // Keep a stored key that is equal to `key`, as `HashMap` does;
                // overwrite only another generation.
                if self.keys.as_slice()[pos] != key {
                    self.keys.overwrite_at(pos, key);
                }
                Some(core::mem::replace(&mut self.values[pos], value))
            }
            None => {
                self.push_vacant(key, index, value);
                None
            }
        }
    }

    /// Stores `value` for `key` and returns the entry previously in `key`'s
    /// slot, whether it was `key` itself or another generation.
    ///
    /// The new entry, key included, takes the old entry's dense position.
    /// Unlike [`insert`](Self::insert), this stores `key` even when the old
    /// key is equal to it.
    ///
    /// # Panics
    ///
    /// As [`insert`](Self::insert).
    pub fn replace(&mut self, key: K, value: V) -> Option<(K, V)> {
        let index = key.index();
        match self.keys.occupant_pos(index) {
            Some(pos) => {
                let old_key = self.keys.overwrite_at(pos, key);
                let old_value = core::mem::replace(&mut self.values[pos], value);
                Some((old_key, old_value))
            }
            None => {
                self.push_vacant(key, index, value);
                None
            }
        }
    }

    /// Adds an entry if `key`'s slot is vacant and returns its value, or
    /// returns `value` back and leaves the map unchanged if the slot is held
    /// (by `key` or by another key with the same index).
    ///
    /// # Panics
    ///
    /// As [`insert`](Self::insert).
    pub fn try_insert(&mut self, key: K, value: V) -> Result<&mut V, V> {
        let index = key.index();
        if self.keys.occupant_pos(index).is_some() {
            return Err(value);
        }
        let pos = self.push_vacant(key, index, value);
        Ok(&mut self.values[pos])
    }

    /// The value for `key`, inserting `make()` first if `key` is absent.
    ///
    /// If the slot is held by a different key, that entry is evicted, as by
    /// [`insert`](Self::insert). `make` runs before the map changes, so if it
    /// panics the entries are unchanged.
    ///
    /// # Panics
    ///
    /// As [`insert`](Self::insert).
    pub fn get_or_insert_with(&mut self, key: K, make: impl FnOnce() -> V) -> &mut V {
        let index = key.index();
        let pos = match self.keys.occupant_pos(index) {
            Some(pos) if self.keys.as_slice()[pos] == key => pos,
            Some(pos) => {
                let value = make();
                self.keys.overwrite_at(pos, key);
                // Drop the evicted value only once the entry is consistent:
                // `V::drop` may panic.
                drop(core::mem::replace(&mut self.values[pos], value));
                pos
            }
            None => {
                let value = make();
                self.push_vacant(key, index, value)
            }
        };
        &mut self.values[pos]
    }

    /// Removes the entry at dense position `pos` by moving the last entry
    /// into its place.
    fn remove_at(&mut self, pos: usize) -> (K, V) {
        let key = self.keys.remove_at(pos);
        let value = self.values.swap_remove(pos);
        (key, value)
    }

    /// Removes `key` and returns its value, if it was present.
    ///
    /// The last entry moves into the removed entry's position.
    pub fn remove(&mut self, key: K) -> Option<V> {
        self.remove_entry(key).map(|(_, value)| value)
    }

    /// Removes `key` and returns the stored key and its value, if it was
    /// present.
    pub fn remove_entry(&mut self, key: K) -> Option<(K, V)> {
        self.position(key).map(|pos| self.remove_at(pos))
    }

    /// Removes whichever entry occupies slot `index` and returns it.
    pub fn remove_occupant(&mut self, index: usize) -> Option<(K, V)> {
        self.keys.occupant_pos(index).map(|pos| self.remove_at(pos))
    }

    /// Keeps only the entries for which `keep` returns `true`.
    ///
    /// `keep` is called once per entry. Kept entries may change position. If
    /// `keep` or a value's `drop` panics, the entries not yet visited are
    /// kept and the map stays consistent.
    pub fn retain(&mut self, mut keep: impl FnMut(&K, &mut V) -> bool) {
        let mut pos = 0;
        while pos < self.values.len() {
            let key = self.keys.as_slice()[pos];
            if keep(&key, &mut self.values[pos]) {
                pos += 1;
            } else {
                // The entry moved into `pos` has not been visited yet. The
                // removed value drops at the end of this statement, after
                // both arrays are updated.
                self.remove_at(pos);
            }
        }
    }

    /// Removes every entry. The sparse index is cleared in O(1); the values
    /// are dropped.
    ///
    /// Pages stay allocated for reuse; use [`reset`](Self::reset) to free
    /// them.
    pub fn clear(&mut self) {
        // Keys first: `values.clear()` runs `V::drop`, which may panic, and
        // `Vec::clear` sets the length to zero before dropping.
        self.keys.clear();
        self.values.clear();
    }

    /// Removes every entry and frees every page. The dense arrays keep their
    /// capacity.
    pub fn reset(&mut self) {
        // Empty both dense arrays before freeing pages, so a panic in
        // `V::drop` or in the page store cannot leave them out of step.
        self.keys.clear();
        self.values.clear();
        self.keys.free_pages();
    }

    /// Shrinks the dense arrays and the page directory. Frees every page if
    /// the map is empty.
    pub fn shrink_to_fit(&mut self) {
        self.keys.shrink_to_fit();
        self.values.shrink_to_fit();
    }

    /// The keys as a slice, in dense order.
    #[inline]
    pub fn key_slice(&self) -> &[K] {
        self.keys.as_slice()
    }

    /// The values as a slice, in dense order: `value_slice()[i]` belongs to
    /// `key_slice()[i]`.
    #[inline]
    pub fn value_slice(&self) -> &[V] {
        &self.values
    }

    /// The values as a mutable slice, in dense order.
    #[inline]
    pub fn value_slice_mut(&mut self) -> &mut [V] {
        &mut self.values
    }

    /// Iterates over the keys in dense order.
    #[inline]
    pub fn keys(&self) -> slice::Iter<'_, K> {
        self.keys.iter()
    }

    /// Iterates over the values in dense order.
    #[inline]
    pub fn values(&self) -> slice::Iter<'_, V> {
        self.values.iter()
    }

    /// Iterates mutably over the values in dense order.
    #[inline]
    pub fn values_mut(&mut self) -> slice::IterMut<'_, V> {
        self.values.iter_mut()
    }

    /// Iterates over the entries in dense order.
    #[inline]
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter(self.keys.iter().zip(self.values.iter()))
    }

    /// Iterates over the entries in dense order, with mutable access to the
    /// values.
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        IterMut(self.keys.iter().zip(self.values.iter_mut()))
    }

    /// Consumes the map and returns its keys in dense order.
    pub fn into_keys(self) -> alloc::vec::IntoIter<K> {
        self.keys.into_iter()
    }

    /// Consumes the map and returns its values in dense order.
    pub fn into_values(self) -> alloc::vec::IntoIter<V> {
        self.values.into_iter()
    }

    /// Reorders both dense arrays so that the entry at old position
    /// `perm[i]` moves to position `i`, where `perm` is the identity sorted by
    /// `order`. Then rewrites the index.
    ///
    /// `order` sees the entries before any of them move, so if it panics the
    /// dense order and the key-value pairing are unchanged.
    fn reorder_by(&mut self, order: impl FnOnce(&mut [usize], &[K], &[V])) {
        let mut perm: Vec<usize> = (0..self.len()).collect();
        order(&mut perm, self.keys.as_slice(), &self.values);
        let values = &mut self.values;
        self.keys
            .reorder_dense(|keys| permute(&mut perm, keys, values));
    }

    /// Sorts the entries by `compare` (unstable) and updates the index.
    ///
    /// If `compare` panics, the dense order and the key-value pairing are
    /// unchanged (values that `compare` changed through interior mutability
    /// stay changed).
    pub fn sort_unstable_by(&mut self, mut compare: impl FnMut((&K, &V), (&K, &V)) -> Ordering) {
        self.reorder_by(|perm, keys, values| {
            perm.sort_unstable_by(|&a, &b| compare((&keys[a], &values[a]), (&keys[b], &values[b])));
        });
    }

    /// Sorts the entries by a key function (unstable) and updates the index.
    ///
    /// `f` may be called more than once per entry. If it panics, the dense
    /// order and the key-value pairing are unchanged.
    pub fn sort_unstable_by_key<T: Ord>(&mut self, mut f: impl FnMut(&K, &V) -> T) {
        self.reorder_by(|perm, keys, values| {
            perm.sort_unstable_by_key(|&i| f(&keys[i], &values[i]));
        });
    }

    /// Sorts the entries by key in ascending order (unstable) and updates
    /// the index.
    pub fn sort_unstable_keys(&mut self)
    where
        K: Ord,
    {
        self.reorder_by(|perm, keys, _| perm.sort_unstable_by_key(|&i| keys[i]));
    }
}

/// Applies `perm` to `keys` and `values` in place: the elements at old
/// position `perm[i]` end up at position `i`. Uses O(n) swaps and leaves
/// `perm` as the identity.
///
/// `perm` must be a permutation of `0..n`, where `n` is the length of all
/// three slices. Nothing here runs user code, so it cannot panic for a valid
/// permutation.
fn permute<K, V>(perm: &mut [usize], keys: &mut [K], values: &mut [V]) {
    debug_assert!(perm.len() == keys.len() && perm.len() == values.len());
    for start in 0..perm.len() {
        // Follow the cycle through `start`, carrying the element that started
        // there; positions already placed hold `perm[i] == i`.
        let mut cur = start;
        while perm[cur] != start {
            let next = perm[cur];
            keys.swap(cur, next);
            values.swap(cur, next);
            perm[cur] = cur;
            cur = next;
        }
        perm[cur] = cur;
    }
}

impl<K: Key, V, const PAGE: usize, D: Directory> Default for PagedSparseMap<K, V, PAGE, D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Key, V: Clone, const PAGE: usize, D: Directory> Clone for PagedSparseMap<K, V, PAGE, D>
where
    D::Store<Slot, PAGE>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            keys: self.keys.clone(),
            values: self.values.clone(),
        }
    }
}

impl<K: Key + fmt::Debug, V: fmt::Debug, const PAGE: usize, D: Directory> fmt::Debug
    for PagedSparseMap<K, V, PAGE, D>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

/// Maps are equal when they hold the same entries, in any order.
impl<K: Key, V: PartialEq, const PAGE: usize, D: Directory> PartialEq
    for PagedSparseMap<K, V, PAGE, D>
{
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len()
            && self
                .iter()
                .all(|(&key, value)| other.get(key) == Some(value))
    }
}

impl<K: Key, V: Eq, const PAGE: usize, D: Directory> Eq for PagedSparseMap<K, V, PAGE, D> {}

/// `map[key]` returns the value for `key`.
///
/// # Panics
///
/// Panics if `key` is not in the map.
impl<K: Key, V, const PAGE: usize, D: Directory> Index<K> for PagedSparseMap<K, V, PAGE, D> {
    type Output = V;

    fn index(&self, key: K) -> &V {
        self.get(key).expect("key not in PagedSparseMap")
    }
}

/// Inserts each entry with [`insert`](PagedSparseMap::insert), so later
/// entries overwrite earlier ones in the same slot.
impl<K: Key, V, const PAGE: usize, D: Directory> Extend<(K, V)> for PagedSparseMap<K, V, PAGE, D> {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        for (key, value) in iter {
            self.insert(key, value);
        }
    }
}

impl<'a, K: Key + 'a, V: Copy + 'a, const PAGE: usize, D: Directory> Extend<(&'a K, &'a V)>
    for PagedSparseMap<K, V, PAGE, D>
{
    fn extend<I: IntoIterator<Item = (&'a K, &'a V)>>(&mut self, iter: I) {
        self.extend(iter.into_iter().map(|(&key, &value)| (key, value)));
    }
}

impl<K: Key, V, const PAGE: usize, D: Directory> FromIterator<(K, V)>
    for PagedSparseMap<K, V, PAGE, D>
{
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut map = Self::new();
        map.extend(iter);
        map
    }
}

impl<'a, K: Key, V, const PAGE: usize, D: Directory> IntoIterator
    for &'a PagedSparseMap<K, V, PAGE, D>
{
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K: Key, V, const PAGE: usize, D: Directory> IntoIterator
    for &'a mut PagedSparseMap<K, V, PAGE, D>
{
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<K: Key, V, const PAGE: usize, D: Directory> IntoIterator for PagedSparseMap<K, V, PAGE, D> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter(self.keys.into_iter().zip(self.values))
    }
}

macro_rules! zip_iterator {
    ($(#[$doc:meta])* $name:ident<$($lt:lifetime,)? $k:ident, $v:ident>, $inner:ty, $item:ty) => {
        $(#[$doc])*
        #[derive(Debug)]
        pub struct $name<$($lt,)? $k, $v>($inner);

        impl<$($lt,)? $k, $v> Iterator for $name<$($lt,)? $k, $v> {
            type Item = $item;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                self.0.next()
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                self.0.size_hint()
            }
        }

        impl<$($lt,)? $k, $v> DoubleEndedIterator for $name<$($lt,)? $k, $v> {
            #[inline]
            fn next_back(&mut self) -> Option<Self::Item> {
                self.0.next_back()
            }
        }

        impl<$($lt,)? $k, $v> ExactSizeIterator for $name<$($lt,)? $k, $v> {}

        impl<$($lt,)? $k, $v> FusedIterator for $name<$($lt,)? $k, $v> {}
    };
}

zip_iterator!(
    /// Iterator over the entries of a [`PagedSparseMap`], from
    /// [`iter`](PagedSparseMap::iter).
    Iter<'a, K, V>,
    Zip<slice::Iter<'a, K>, slice::Iter<'a, V>>,
    (&'a K, &'a V)
);

zip_iterator!(
    /// Iterator over the entries of a [`PagedSparseMap`] with mutable
    /// values, from [`iter_mut`](PagedSparseMap::iter_mut).
    IterMut<'a, K, V>,
    Zip<slice::Iter<'a, K>, slice::IterMut<'a, V>>,
    (&'a K, &'a mut V)
);

zip_iterator!(
    /// Owning iterator over the entries of a [`PagedSparseMap`].
    IntoIter<K, V>,
    Zip<alloc::vec::IntoIter<K>, alloc::vec::IntoIter<V>>,
    (K, V)
);

impl<K, V> Clone for Iter<'_, K, V> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<K: Clone, V: Clone> Clone for IntoIter<K, V> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

#[cfg(feature = "serde")]
mod serde_impl {
    use super::*;
    use core::marker::PhantomData;
    use serde::de::{Deserialize, Deserializer, MapAccess, Visitor};
    use serde::ser::{Serialize, Serializer};

    /// Most entries to reserve from an untrusted size hint: 4096, and at
    /// most 1 MiB of dense storage.
    fn max_prealloc<K, V>() -> usize {
        let entry = core::mem::size_of::<K>() + core::mem::size_of::<V>();
        // `1 << 20` does not fit a 16-bit `usize`; the cap is then `usize::MAX`.
        usize::try_from(1_u32 << 20)
            .unwrap_or(usize::MAX)
            .checked_div(entry)
            .unwrap_or(usize::MAX)
            .min(4096)
    }

    /// Serializes as a map from keys to values, in dense order.
    impl<K: Key + Serialize, V: Serialize, const PAGE: usize, D: Directory> Serialize
        for PagedSparseMap<K, V, PAGE, D>
    {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            serializer.collect_map(self.iter())
        }
    }

    /// Deserializes from a map. Entries are inserted in input order with
    /// [`insert`](PagedSparseMap::insert), so a later entry overwrites an
    /// earlier one in the same slot, as `HashMap` keeps the last duplicate.
    ///
    /// Memory follows the key indices, not the input length: see the
    /// warning on [`PagedSparseSet`]'s `Deserialize` impl. For untrusted
    /// input, use [`Tree`](crate::page::Tree) or validate the keys first.
    impl<'de, K, V, const PAGE: usize, D> Deserialize<'de> for PagedSparseMap<K, V, PAGE, D>
    where
        K: Key + Deserialize<'de>,
        V: Deserialize<'de>,
        D: Directory,
    {
        fn deserialize<De: Deserializer<'de>>(deserializer: De) -> Result<Self, De::Error> {
            struct MapVisitor<K, V, const PAGE: usize, D>(PhantomData<(K, V, D)>);

            impl<'de, K, V, const PAGE: usize, D> Visitor<'de> for MapVisitor<K, V, PAGE, D>
            where
                K: Key + Deserialize<'de>,
                V: Deserialize<'de>,
                D: Directory,
            {
                type Value = PagedSparseMap<K, V, PAGE, D>;

                fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    f.write_str("a map of keys to values")
                }

                fn visit_map<A: MapAccess<'de>>(
                    self,
                    mut access: A,
                ) -> Result<Self::Value, A::Error> {
                    // Cap the hint so a hostile length cannot force a huge
                    // allocation.
                    let hint = access.size_hint().unwrap_or(0).min(max_prealloc::<K, V>());
                    let mut map = PagedSparseMap::with_capacity(hint);
                    while let Some((key, value)) = access.next_entry()? {
                        map.insert(key, value);
                    }
                    Ok(map)
                }
            }

            deserializer.deserialize_map(MapVisitor::<K, V, PAGE, D>(PhantomData))
        }
    }
}

// The tests use `u32` keys, which implement `Key` only on 32- and 64-bit
// targets.
#[cfg(all(test, any(target_pointer_width = "32", target_pointer_width = "64")))]
mod tests {
    use super::*;
    use alloc::string::{String, ToString};

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    struct Gen(u32);

    impl Key for Gen {
        fn index(self) -> usize {
            (self.0 & 0xffff) as usize
        }
    }

    fn assert_parallel<K: Key + fmt::Debug, V, const PAGE: usize, D: Directory>(
        map: &PagedSparseMap<K, V, PAGE, D>,
    ) {
        assert_eq!(map.key_slice().len(), map.value_slice().len());
        for (pos, &key) in map.key_slice().iter().enumerate() {
            assert_eq!(map.position(key), Some(pos), "{key:?}");
        }
    }

    #[test]
    fn remove_moves_last_entry_with_its_value() {
        let mut map: PagedSparseMap<u32, char, 4> = [(1, 'a'), (2, 'b'), (3, 'c'), (40, 'd')]
            .into_iter()
            .collect();
        assert_eq!(map.remove(1), Some('a'));
        assert_eq!(map.key_slice(), &[40, 2, 3]);
        assert_eq!(map.value_slice(), &['d', 'b', 'c']);
        assert_eq!(map.remove(1), None);
        assert_eq!(map.remove_entry(3), Some((3, 'c')));
        assert_eq!(map[40], 'd');
        assert_parallel(&map);
    }

    #[test]
    fn insert_updates_in_place() {
        let mut map: PagedSparseMap<u32, u32> = (0..5).map(|k| (k, k * 10)).collect();
        assert_eq!(map.insert(2, 99), Some(20));
        assert_eq!(map.position(2), Some(2));
        assert_eq!(map.value_slice(), &[0, 10, 99, 30, 40]);
        *map.get_mut(4).unwrap() += 1;
        assert_eq!(map.get(4), Some(&41));
    }

    /// Equal keys that are not identical: `tag` is ignored by `==`.
    #[derive(Clone, Copy, Debug)]
    struct Tagged {
        id: u32,
        tag: u8,
    }

    impl PartialEq for Tagged {
        fn eq(&self, other: &Self) -> bool {
            self.id == other.id
        }
    }

    impl Eq for Tagged {}

    impl Key for Tagged {
        fn index(self) -> usize {
            self.id as usize
        }
    }

    #[test]
    fn insert_keeps_an_equal_stored_key_and_replace_does_not() {
        let mut map: PagedSparseMap<Tagged, u8> = PagedSparseMap::new();
        map.insert(Tagged { id: 3, tag: 1 }, 10);
        assert_eq!(map.insert(Tagged { id: 3, tag: 2 }, 20), Some(10));
        assert_eq!(map.key_slice()[0].tag, 1);
        let (old, _) = map.replace(Tagged { id: 3, tag: 3 }, 30).unwrap();
        assert_eq!(old.tag, 1);
        assert_eq!(map.key_slice()[0].tag, 3);
        assert_eq!(map.clone().into_keys().next().map(|k| k.tag), Some(3));
        assert_eq!(map.into_values().collect::<Vec<_>>(), [30]);
    }

    #[test]
    fn generations_by_method() {
        let old = Gen(7);
        let new = Gen(7 | 1 << 16);
        let mut map: PagedSparseMap<Gen, &str> = PagedSparseMap::new();
        assert_eq!(map.try_insert(old, "old").copied(), Ok("old"));
        assert_eq!(map.try_insert(new, "new"), Err("new"));
        assert_eq!(map.try_insert(old, "again"), Err("again"));
        assert_eq!(map.get(new), None);
        assert_eq!(map.occupant(7), Some((&old, &"old")));

        assert_eq!(map.replace(new, "new"), Some((old, "old")));
        assert_eq!(map.get(old), None);
        assert_eq!(map.insert(old, "back"), Some("new"));
        assert_eq!(map.get_or_insert_with(new, || "made"), &"made");
        assert_eq!(map.get_or_insert_with(new, || unreachable!()), &"made");
        assert_eq!(map.len(), 1);

        assert_eq!(map.remove(old), None);
        if let Some((_, v)) = map.occupant_mut(7) {
            *v = "edited";
        }
        assert_eq!(map.remove_occupant(7), Some((new, "edited")));
        assert!(map.is_empty());
    }

    #[test]
    fn clear_is_lazy_but_correct() {
        let mut map: PagedSparseMap<u32, String, 4> = (0..10).map(|k| (k, k.to_string())).collect();
        map.clear();
        assert!(map.is_empty());
        assert_eq!(map.allocated_pages(), 3);
        assert!((0..10).all(|k| !map.contains_key(k)));
        map.insert(9, "nine".into());
        assert_eq!(map.get(0), None);
        assert_eq!(map.position(9), Some(0));
        map.reset();
        assert_eq!(map.allocated_pages(), 0);
    }

    #[test]
    fn retain_and_sort_move_values_with_keys() {
        let mut map: PagedSparseMap<u32, u32, 8> = [9, 3, 7, 1, 8, 2]
            .into_iter()
            .map(|k| (k, k * 100))
            .collect();
        map.retain(|&k, v| {
            *v += 1;
            k % 3 != 0
        });
        map.sort_unstable_keys();
        assert_eq!(map.key_slice(), &[1, 2, 7, 8]);
        assert_eq!(map.value_slice(), &[101, 201, 701, 801]);
        assert_parallel(&map);

        map.sort_unstable_by(|(_, a), (_, b)| b.cmp(a));
        assert_eq!(map.key_slice(), &[8, 7, 2, 1]);
        map.sort_unstable_by_key(|&k, _| k % 7);
        assert_eq!(map.key_slice(), &[7, 8, 1, 2]);
        assert_eq!(map.value_slice(), &[701, 801, 101, 201]);
        assert_parallel(&map);
    }

    #[test]
    fn permute_handles_every_permutation_of_four() {
        // Brute force: keep the 24 of the 256 index arrays that are
        // permutations.
        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    for d in 0..4 {
                        let p = [a, b, c, d];
                        if (0..4).any(|i| !p.contains(&i)) {
                            continue;
                        }
                        let mut perm = p;
                        let mut keys = [10, 11, 12, 13];
                        let mut values = ['a', 'b', 'c', 'd'];
                        permute(&mut perm, &mut keys, &mut values);
                        assert_eq!(keys, p.map(|i| 10 + i), "{p:?}");
                        assert_eq!(values, p.map(|i| ['a', 'b', 'c', 'd'][i]), "{p:?}");
                        assert_eq!(perm, [0, 1, 2, 3]);
                    }
                }
            }
        }
    }

    #[test]
    fn equality_and_iterators() {
        let a: PagedSparseMap<u32, i32> = [(1, -1), (2, -2)].into_iter().collect();
        let mut b: PagedSparseMap<u32, i32> = [(2, -2), (1, -1)].into_iter().collect();
        assert_eq!(a, b);
        for (_, v) in &mut b {
            *v *= 2;
        }
        assert_ne!(a, b);
        assert_eq!(b.iter().rev().collect::<Vec<_>>(), [(&1, &-2), (&2, &-4)]);
        assert_eq!(b.iter().len(), 2);
        assert_eq!(b.into_iter().collect::<Vec<_>>(), [(2, -4), (1, -2)]);
    }
}
