//! [`PagedDirectMap`]: a direct-address map whose values live in place in a
//! paged index.
//!
//! Each key's entry is stored in the slot [`Key::index`] names, in pages
//! allocated on first use. There is no dense array, so a lookup is one page
//! lookup and one slot read, entries never move, and iteration visits the
//! entries in ascending slot-index order (for plain integer keys, key
//! order). The cost is that iteration and
//! [`clear`](PagedDirectMap::clear) scan every allocated page, occupied or
//! not, and every allocated slot takes the space of an entry.
//!
//! Prefer it to [`PagedSparseMap`](crate::sparse_map::PagedSparseMap) when
//! keys fill a few dense ranges, when you want slot order, or when lookups
//! dominate; prefer the sparse map when keys are scattered thinly or when you
//! need the entries as contiguous slices.
//!
//! | Operation | Cost |
//! |---|---|
//! | [`get`](PagedDirectMap::get), [`insert`](PagedDirectMap::insert), [`remove`](PagedDirectMap::remove) | O(1) (O(depth) with [`Radix`](crate::page::Radix), O(log pages) with [`Tree`](crate::page::Tree)); the first key in a page also allocates the page (and, with [`Flat`], grows the directory up to it) |
//! | iteration, [`clear`](PagedDirectMap::clear), [`retain`](PagedDirectMap::retain) | O(allocated pages × `PAGE`), plus with [`Flat`] O(directory length) to skip unallocated pages |
//! | [`range`](PagedDirectMap::range), [`next_occupant`](PagedDirectMap::next_occupant), [`first_key_value`](PagedDirectMap::first_key_value) and the other [ordered queries](self#ordered-queries) | reaching the first page of the range is O(1) (O(depth) with [`Radix`](crate::page::Radix), O(log pages) with [`Tree`](crate::page::Tree)); then the slots of the allocated pages in the range are scanned up to the entry wanted, plus with [`Flat`] the unallocated directory entries between them |
//! | [`len`](PagedDirectMap::len) | O(1) |
//!
//! Pages are not freed when they empty, as `HashMap` keeps its capacity;
//! [`shrink_to_fit`](PagedDirectMap::shrink_to_fit) frees empty pages and
//! [`reset`](PagedDirectMap::reset) frees all of them.
//!
//! # Slots and generational keys
//!
//! The map holds at most one key per slot index (see [`crate::key`]), and the
//! methods treat a slot held by a *different* key (another generation) as
//! [`PagedSparseMap`](crate::sparse_map::PagedSparseMap) does:
//! [`insert`](PagedDirectMap::insert) and
//! [`get_or_insert_with`](PagedDirectMap::get_or_insert_with) evict it,
//! [`replace`](PagedDirectMap::replace) evicts it and returns it, and
//! [`try_insert`](PagedDirectMap::try_insert) refuses. An
//! [`entry`](PagedDirectMap::entry) is occupied only by the key itself; a
//! vacant entry's inserts evict another generation.
//!
//! # What a slot stores
//!
//! A slot holds the value and the key's [`DirectKey::Tag`]: the part of the
//! key its slot index does not give. Plain integer keys have the tag `()`, so
//! a slot of a `u32 -> u64` map is an `Option<((), u64)>` of 16 bytes, and
//! one of a `u32 -> Box<T>` map takes only the pointer's 8 bytes. Iterators
//! and lookups rebuild each key from its slot index and tag, so they yield
//! keys by value (`(K, &V)`) rather than by reference.
//!
//! # Ordered queries
//!
//! Entries are ordered by slot index, so the map answers the questions an
//! ordered map does, without keeping a tree:
//! [`range`](PagedDirectMap::range) and
//! [`range_mut`](PagedDirectMap::range_mut) iterate the entries between two
//! keys, [`first_key_value`](PagedDirectMap::first_key_value),
//! [`last_key_value`](PagedDirectMap::last_key_value),
//! [`pop_first`](PagedDirectMap::pop_first) and
//! [`pop_last`](PagedDirectMap::pop_last) reach the ends, and
//! [`next_occupant`](PagedDirectMap::next_occupant) and
//! [`prev_occupant`](PagedDirectMap::prev_occupant) find the nearest entry at
//! or after, or at or before, a slot index. The order is that of
//! [`Key::index`]: key order for plain integers, index order (whatever the
//! generation) for generational keys.
//!
//! ```
//! use paged_collections::direct_map::PagedDirectMap;
//!
//! let map: PagedDirectMap<u32, char> =
//!     [(2, 'a'), (5, 'b'), (70_000, 'c')].into_iter().collect();
//! let keys: Vec<u32> = map.range(3..=70_000).map(|(k, _)| k).collect();
//! assert_eq!(keys, [5, 70_000]);
//! assert_eq!(map.next_occupant(6), Some((70_000, &'c')));
//! assert_eq!(map.prev_occupant(4), Some((2, &'a')));
//! assert_eq!(map.last_key_value(), Some((70_000, &'c')));
//! ```
//!
//! ```
//! use paged_collections::direct_map::PagedDirectMap;
//!
//! let mut map: PagedDirectMap<u32, &str> = PagedDirectMap::new();
//! map.insert(3_000_000, "far");
//! map.insert(7, "near");
//! assert_eq!(map.insert(7, "nearer"), Some("near"));
//! assert_eq!(map.get(3_000_000), Some(&"far"));
//! assert_eq!(map.allocated_pages(), 2);
//!
//! // Integer keys iterate in key order, whatever the insertion order.
//! let keys: Vec<u32> = map.keys().collect();
//! assert_eq!(keys, [7, 3_000_000]);
//!
//! map.remove(3_000_000);
//! map.shrink_to_fit(); // frees the now-empty page
//! assert_eq!(map.allocated_pages(), 1);
//! ```
//!
//! With the `rayon` feature, borrowed parallel iterators are unindexed and
//! visit entries page by page. Collecting into a `Vec` preserves slot-index
//! order. The order of `for_each` is unspecified.
//!
//! ```rust
//! # #[cfg(feature = "rayon")] {
//! use paged_collections::direct_map::PagedDirectMap;
//! use rayon::prelude::*;
//!
//! let mut map: PagedDirectMap<u32, i32, 4> =
//!     [(9, 10), (2, 20)].into_iter().collect();
//! map.par_iter_mut().for_each(|(_, value)| *value += 1);
//! assert_eq!(map.par_iter().map(|(k, &v)| (k, v)).collect::<Vec<_>>(), [(2, 21), (9, 11)]);
//! # }
//! ```

use alloc::boxed::Box;
use alloc::vec::{self, Vec};
use core::fmt;
use core::hash::{Hash, Hasher};
use core::iter::{Fuse, FusedIterator};
use core::marker::PhantomData;
use core::mem;
use core::ops::{Bound, Index, RangeBounds};
use core::slice;

use crate::key::{DirectKey, Key};
use crate::page::{DEFAULT_PAGE, Directory, Flat, PageStore, split};

/// One slot of a [`PagedDirectMap`] page: the tag of the key held there and
/// its value, if any.
pub type Slot<K, V> = Option<(<K as DirectKey>::Tag, V)>;

type Store<K, V, const PAGE: usize, D> = <D as Directory>::Store<Slot<K, V>, PAGE>;
type Pages<'a, K, V, const PAGE: usize, D> =
    <Store<K, V, PAGE, D> as PageStore<Slot<K, V>, PAGE>>::Pages<'a>;
type PagesMut<'a, K, V, const PAGE: usize, D> =
    <Store<K, V, PAGE, D> as PageStore<Slot<K, V>, PAGE>>::PagesMut<'a>;
type IntoPages<K, V, const PAGE: usize, D> =
    <Store<K, V, PAGE, D> as PageStore<Slot<K, V>, PAGE>>::IntoPages;

/// A map from [`DirectKey`]s to values stored in place at each key's slot
/// index.
///
/// - `PAGE` is the number of slots per page; it must be non-zero. Each slot
///   takes `size_of::<Option<(K::Tag, V)>>()` bytes whether or not it is
///   occupied; see [what a slot stores](self#what-a-slot-stores).
/// - `D` is the page directory: [`Flat`] (default),
///   [`Radix`](crate::page::Radix) for wide keys, or
///   [`Tree`](crate::page::Tree). See [`crate::page`].
///
/// Entries are ordered by slot index. See the [module docs](self) for costs
/// and for how methods treat a slot held by another generation of a key.
pub struct PagedDirectMap<K: DirectKey, V, const PAGE: usize = DEFAULT_PAGE, D: Directory = Flat> {
    // Invariant: `len` is the number of occupied slots across all pages.
    // Every mutation below keeps it even when a user closure, `V::drop` or an
    // allocation panics.
    pages: Store<K, V, PAGE, D>,
    len: usize,
}

impl<K: DirectKey, V, const PAGE: usize, D: Directory> PagedDirectMap<K, V, PAGE, D> {
    /// Number of slots per page.
    pub const PAGE_SIZE: usize = PAGE;

    /// Creates an empty map. No memory is allocated until the first insert.
    pub fn new() -> Self {
        Self {
            pages: Default::default(),
            len: 0,
        }
    }

    /// Number of entries in the map.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the map has no entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of allocated pages.
    #[inline]
    pub fn allocated_pages(&self) -> usize {
        self.pages.allocated()
    }

    #[inline]
    fn slot(&self, index: usize) -> Option<&Slot<K, V>> {
        let (page, offset) = split::<PAGE>(index);
        self.pages.get(page).map(|page| &page[offset])
    }

    #[inline]
    fn slot_mut(&mut self, index: usize) -> Option<&mut Slot<K, V>> {
        let (page, offset) = split::<PAGE>(index);
        self.pages.get_mut(page).map(|page| &mut page[offset])
    }

    /// The slot for `index`, allocating its page if needed, and the entry
    /// count.
    #[inline]
    fn slot_or_alloc(&mut self, index: usize) -> (&mut Slot<K, V>, &mut usize) {
        let (page, offset) = split::<PAGE>(index);
        (&mut self.pages.get_or_alloc(page)[offset], &mut self.len)
    }

    /// Whether `key` is in the map.
    #[inline]
    pub fn contains_key(&self, key: K) -> bool {
        self.get(key).is_some()
    }

    /// The value for `key`, if it is in the map.
    #[inline]
    pub fn get(&self, key: K) -> Option<&V> {
        match self.slot(key.index())? {
            Some((tag, value)) if *tag == key.tag() => Some(value),
            _ => None,
        }
    }

    /// Mutable access to the value for `key`, if it is in the map.
    #[inline]
    pub fn get_mut(&mut self, key: K) -> Option<&mut V> {
        match self.slot_mut(key.index())? {
            Some((tag, value)) if *tag == key.tag() => Some(value),
            _ => None,
        }
    }

    /// The stored key equal to `key` and its value, if it is in the map.
    #[inline]
    pub fn get_key_value(&self, key: K) -> Option<(K, &V)> {
        let index = key.index();
        match self.slot(index)? {
            Some((tag, value)) if *tag == key.tag() => Some((K::from_parts(index, *tag), value)),
            _ => None,
        }
    }

    /// The entry occupying slot `index`, if any.
    ///
    /// With generational keys this finds whichever generation is present.
    #[inline]
    pub fn occupant(&self, index: usize) -> Option<(K, &V)> {
        let (tag, value) = self.slot(index)?.as_ref()?;
        Some((K::from_parts(index, *tag), value))
    }

    /// The entry occupying slot `index`, with mutable access to its value.
    #[inline]
    pub fn occupant_mut(&mut self, index: usize) -> Option<(K, &mut V)> {
        let (tag, value) = self.slot_mut(index)?.as_mut()?;
        Some((K::from_parts(index, *tag), value))
    }

    /// Stores `value` for `key` and returns the value previously in `key`'s
    /// slot.
    ///
    /// If `key` is present its value is replaced and the stored key is kept.
    /// If the slot is held by a different key (another generation), that
    /// entry is evicted and its value returned; use
    /// [`replace`](Self::replace) to get the evicted key too, or
    /// [`try_insert`](Self::try_insert) to refuse.
    ///
    /// # Panics
    ///
    /// Panics if the page or the directory cannot be allocated. The entries
    /// are then unchanged, though the directory may have grown.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let tag = key.tag();
        let (slot, len) = self.slot_or_alloc(key.index());
        if let Some((stored, old)) = slot {
            if *stored != tag {
                *stored = tag;
            }
            return Some(mem::replace(old, value));
        }
        *slot = Some((tag, value));
        *len += 1;
        None
    }

    /// Stores `value` for `key` and returns the entry previously in `key`'s
    /// slot, whether it was `key` itself or another generation.
    ///
    /// Unlike [`insert`](Self::insert), this stores `key` even when the old
    /// key is equal to it.
    ///
    /// # Panics
    ///
    /// As [`insert`](Self::insert).
    pub fn replace(&mut self, key: K, value: V) -> Option<(K, V)> {
        let index = key.index();
        let (slot, len) = self.slot_or_alloc(index);
        let old = slot.replace((key.tag(), value));
        if old.is_none() {
            *len += 1;
        }
        old.map(|(tag, value)| (K::from_parts(index, tag), value))
    }

    /// Adds an entry if `key`'s slot is vacant and returns its value, or
    /// returns `value` back and leaves the map unchanged if the slot is held
    /// (by `key` or by another key with the same index).
    ///
    /// # Panics
    ///
    /// As [`insert`](Self::insert).
    pub fn try_insert(&mut self, key: K, value: V) -> Result<&mut V, V> {
        let (slot, len) = self.slot_or_alloc(key.index());
        if slot.is_some() {
            return Err(value);
        }
        *len += 1;
        let (_, value) = slot.insert((key.tag(), value));
        Ok(value)
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
        let (index, tag) = (key.index(), key.tag());
        let held_by_key = matches!(self.slot(index), Some(Some((stored, _))) if *stored == tag);
        if !held_by_key {
            // `make` runs before anything is allocated or changed.
            let value = make();
            let (slot, len) = self.slot_or_alloc(index);
            if slot.is_none() {
                *len += 1;
            }
            // Drop an evicted entry only once the slot is consistent:
            // `V::drop` may panic.
            drop(slot.replace((tag, value)));
        }
        // The page exists now, so this allocates nothing.
        let (slot, _) = self.slot_or_alloc(index);
        match slot {
            Some((_, value)) => value,
            None => unreachable!("slot was just filled"),
        }
    }

    /// Removes `key` and returns its value, if it was present.
    pub fn remove(&mut self, key: K) -> Option<V> {
        self.remove_entry(key).map(|(_, value)| value)
    }

    /// Removes `key` and returns the stored key and its value, if it was
    /// present.
    pub fn remove_entry(&mut self, key: K) -> Option<(K, V)> {
        let index = key.index();
        let slot = self.slot_mut(index)?;
        if !matches!(slot, Some((stored, _)) if *stored == key.tag()) {
            return None;
        }
        let (tag, value) = slot.take()?;
        self.len -= 1;
        Some((K::from_parts(index, tag), value))
    }

    /// Removes whichever entry occupies slot `index` and returns it.
    pub fn remove_occupant(&mut self, index: usize) -> Option<(K, V)> {
        let (tag, value) = self.slot_mut(index)?.take()?;
        self.len -= 1;
        Some((K::from_parts(index, tag), value))
    }

    /// Keeps only the entries for which `keep` returns `true`, visiting them
    /// in slot-index order.
    ///
    /// If `keep` or a value's `drop` panics, the entries not yet visited are
    /// kept and the map stays consistent.
    pub fn retain(&mut self, mut keep: impl FnMut(&K, &mut V) -> bool) {
        let mut remaining = self.len;
        for (number, page) in self.pages.pages_mut() {
            for (offset, slot) in page.iter_mut().enumerate() {
                if remaining == 0 {
                    return;
                }
                let Some((tag, value)) = slot else { continue };
                remaining -= 1;
                let key = K::from_parts(number * PAGE + offset, *tag);
                if !keep(&key, value) {
                    let removed = slot.take();
                    self.len -= 1;
                    drop(removed);
                }
            }
        }
    }

    /// Removes every entry, dropping the values in slot-index order. Pages
    /// stay allocated for reuse; use [`reset`](Self::reset) to free them.
    ///
    /// If a value's `drop` panics, the entries not yet dropped stay in the
    /// map.
    pub fn clear(&mut self) {
        self.retain(|_, _| false);
    }

    /// Removes every entry and frees every page.
    pub fn reset(&mut self) {
        // Detach the pages before dropping them: `V::drop` may panic, and the
        // map must be empty and consistent whatever the store's drop order.
        let pages = mem::take(&mut self.pages);
        self.len = 0;
        drop(pages);
    }

    /// Frees every page that holds no entry and shrinks the page directory.
    pub fn shrink_to_fit(&mut self) {
        if self.len == 0 {
            self.pages.clear();
        } else {
            let empty: Vec<usize> = self
                .pages
                .pages()
                .filter(|(_, page)| page.iter().all(Option::is_none))
                .map(|(number, _)| number)
                .collect();
            for number in empty {
                self.pages.remove(number);
            }
        }
        self.pages.shrink_to_fit();
    }

    /// Iterates over the entries in slot-index order.
    #[inline]
    pub fn iter(&self) -> Iter<'_, K, V, PAGE, D> {
        Iter(Cursor::new(self.pages.pages(), self.len))
    }

    /// Iterates over the entries in slot-index order, with mutable access to
    /// the values.
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V, PAGE, D> {
        IterMut(Cursor::new(self.pages.pages_mut(), self.len))
    }

    /// Iterates over the keys in slot-index order.
    #[inline]
    pub fn keys(&self) -> Keys<'_, K, V, PAGE, D> {
        Keys(self.iter())
    }

    /// Iterates over the values in slot-index order.
    #[inline]
    pub fn values(&self) -> Values<'_, K, V, PAGE, D> {
        Values(self.iter())
    }

    /// Iterates mutably over the values in slot-index order.
    #[inline]
    pub fn values_mut(&mut self) -> ValuesMut<'_, K, V, PAGE, D> {
        ValuesMut(self.iter_mut())
    }

    /// Iterates over the entries whose slot index lies between the slot
    /// indices of the bounds, in slot-index order.
    ///
    /// The bounds are keys, compared by [`Key::index`]: for plain integer
    /// keys this is the usual key range. A generational key bounds by its
    /// index only, so `range(id..=id)` yields whichever generation holds
    /// `id`'s slot. An empty or inverted range yields nothing.
    ///
    /// See the [module docs](self) for the cost.
    pub fn range(&self, keys: impl RangeBounds<K>) -> Range<'_, K, V, PAGE, D> {
        self.slot_range(index_bounds(&keys))
    }

    /// Like [`range`](Self::range), with mutable access to the values.
    pub fn range_mut(&mut self, keys: impl RangeBounds<K>) -> RangeMut<'_, K, V, PAGE, D> {
        self.slot_range_mut(index_bounds(&keys))
    }

    /// Iterates over the entries whose slot index is in `indices`, in
    /// slot-index order. An empty or inverted range yields nothing.
    pub fn slot_range(&self, indices: impl RangeBounds<usize>) -> Range<'_, K, V, PAGE, D> {
        let span = Span::new::<PAGE>(&indices, self.len);
        Range(Cursor::new(
            Trim {
                pages: self.pages.pages_in(span.pages.clone()),
                first: span.first,
                last: span.last,
            },
            span.most,
        ))
    }

    /// Like [`slot_range`](Self::slot_range), with mutable access to the
    /// values.
    pub fn slot_range_mut(
        &mut self,
        indices: impl RangeBounds<usize>,
    ) -> RangeMut<'_, K, V, PAGE, D> {
        let span = Span::new::<PAGE>(&indices, self.len);
        RangeMut(Cursor::new(
            Trim {
                pages: self.pages.pages_in_mut(span.pages.clone()),
                first: span.first,
                last: span.last,
            },
            span.most,
        ))
    }

    /// The first entry at slot `index` or after it, in slot-index order.
    ///
    /// For plain integer keys, the entry with the smallest key `>= index`.
    #[inline]
    pub fn next_occupant(&self, index: usize) -> Option<(K, &V)> {
        self.slot_range(index..).next()
    }

    /// The last entry at slot `index` or before it, in slot-index order.
    ///
    /// For plain integer keys, the entry with the largest key `<= index`.
    #[inline]
    pub fn prev_occupant(&self, index: usize) -> Option<(K, &V)> {
        self.slot_range(..=index).next_back()
    }

    /// The entry with the smallest slot index.
    #[inline]
    pub fn first_key_value(&self) -> Option<(K, &V)> {
        self.iter().next()
    }

    /// The entry with the largest slot index.
    #[inline]
    pub fn last_key_value(&self) -> Option<(K, &V)> {
        self.iter().next_back()
    }

    /// Removes and returns the entry with the smallest slot index.
    pub fn pop_first(&mut self) -> Option<(K, V)> {
        let index = self.first_key_value()?.0.index();
        self.remove_occupant(index)
    }

    /// Removes and returns the entry with the largest slot index.
    pub fn pop_last(&mut self) -> Option<(K, V)> {
        let index = self.last_key_value()?.0.index();
        self.remove_occupant(index)
    }

    /// Removes every entry and returns them in slot-index order. Pages stay
    /// allocated, as with [`clear`](Self::clear).
    ///
    /// Entries are removed as the iterator yields them; dropping it removes
    /// the rest. If it is leaked instead (`mem::forget`), or a value's `drop`
    /// panics while the rest are removed, the entries not yet reached stay in
    /// the map, which stays consistent.
    ///
    /// ```
    /// use paged_collections::direct_map::PagedDirectMap;
    ///
    /// let mut map: PagedDirectMap<u32, char> = [(5, 'b'), (1, 'a')].into_iter().collect();
    /// assert_eq!(map.drain().collect::<Vec<_>>(), [(1, 'a'), (5, 'b')]);
    /// assert!(map.is_empty());
    /// ```
    pub fn drain(&mut self) -> Drain<'_, K, V, PAGE, D> {
        Drain {
            map: self,
            next: Some(0),
        }
    }

    /// Removes the entries for which `pred` returns `true` and yields them,
    /// visiting the entries in slot-index order.
    ///
    /// Entries are tested and removed as the iterator advances: those it has
    /// not reached when it is dropped stay in the map. Use
    /// [`retain`](Self::retain) if the removed entries are not needed.
    ///
    /// ```
    /// use paged_collections::direct_map::PagedDirectMap;
    ///
    /// let mut map: PagedDirectMap<u32, u32> = (0..6).map(|k| (k, k * 10)).collect();
    /// let odd: Vec<(u32, u32)> = map.extract_if(|k, _| k % 2 == 1).collect();
    /// assert_eq!(odd, [(1, 10), (3, 30), (5, 50)]);
    /// assert_eq!(map.len(), 3);
    /// ```
    pub fn extract_if<F: FnMut(&K, &mut V) -> bool>(
        &mut self,
        pred: F,
    ) -> ExtractIf<'_, K, V, F, PAGE, D> {
        ExtractIf {
            map: self,
            next: Some(0),
            pred,
        }
    }

    /// The values of several keys at once, mutably: `None` for each key that
    /// is not in the map. O(N log N) for N keys, plus stepping over the pages
    /// between the smallest and largest key's.
    ///
    /// ```
    /// use paged_collections::direct_map::PagedDirectMap;
    ///
    /// let mut map: PagedDirectMap<u32, u32> = [(1, 10), (9000, 20)].into_iter().collect();
    /// if let [Some(a), None, Some(b)] = map.get_disjoint_mut([1, 2, 9000]) {
    ///     core::mem::swap(a, b);
    /// }
    /// assert_eq!((map[1], map[9000]), (20, 10));
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if a key in the map is given twice.
    pub fn get_disjoint_mut<const N: usize>(&mut self, keys: [K; N]) -> [Option<&mut V>; N] {
        // The present keys as (slot index, position in `keys`).
        let mut order = [(0, 0); N];
        let mut n = 0;
        for (i, &key) in keys.iter().enumerate() {
            if self.get(key).is_some() {
                order[n] = (key.index(), i);
                n += 1;
            }
        }
        let order = &mut order[..n];
        order.sort_unstable();
        for pair in order.windows(2) {
            assert!(
                pair[0].0 != pair[1].0,
                "PagedDirectMap::get_disjoint_mut: key at positions {} and {} given twice",
                pair[0].1,
                pair[1].1
            );
        }
        let mut out = [const { None }; N];
        let (Some(&(lo, _)), Some(&(hi, _))) = (order.first(), order.last()) else {
            return out;
        };
        // Walk the pages in order (each is borrowed once), splitting each
        // wanted slot off the front of what is left of its page.
        let mut want = order.iter().peekable();
        for (number, page) in self
            .pages
            .pages_in_mut(split::<PAGE>(lo).0..=split::<PAGE>(hi).0)
        {
            let mut rest: &mut [Slot<K, V>] = page;
            let mut start = 0;
            while let Some(&&(index, i)) = want.peek() {
                let (wanted, offset) = split::<PAGE>(index);
                if wanted > number {
                    break;
                }
                want.next();
                if wanted < number {
                    // Only a store that breaks the page contract skips a
                    // page holding an entry just found.
                    continue;
                }
                let (_, tail) = mem::take(&mut rest).split_at_mut(offset - start);
                let (slot, tail) = tail.split_first_mut().expect("offsets are inside the page");
                if let Some((_, value)) = slot {
                    out[i] = Some(value);
                }
                rest = tail;
                start = offset + 1;
            }
        }
        out
    }

    /// The entry for `key`, for in-place access or insertion.
    ///
    /// The entry is [`Occupied`](Entry::Occupied) only if `key` itself is in
    /// the map. A slot held by a different key (another generation) gives a
    /// [`Vacant`](Entry::Vacant) entry, whose inserts evict that key as
    /// [`insert`](Self::insert) does; [`VacantEntry::occupant`] shows it.
    /// No page is allocated until a vacant entry is inserted into.
    ///
    /// ```
    /// use paged_collections::direct_map::PagedDirectMap;
    ///
    /// let mut counts: PagedDirectMap<u32, u32> = PagedDirectMap::new();
    /// for word in [3, 1, 3, 3] {
    ///     *counts.entry(word).or_insert(0) += 1;
    /// }
    /// assert_eq!((counts[3], counts[1]), (3, 1));
    /// ```
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V, PAGE, D> {
        let index = key.index();
        let stored = match self.slot(index) {
            Some(Some((stored, _))) if *stored == key.tag() => *stored,
            _ => return Entry::Vacant(VacantEntry { map: self, key }),
        };
        let (page, offset) = split::<PAGE>(index);
        match self.pages.get_mut(page) {
            Some(page) => Entry::Occupied(OccupiedEntry {
                key: K::from_parts(index, stored),
                slot: &mut page[offset],
                len: &mut self.len,
            }),
            None => unreachable!("the slot was just found"),
        }
    }

    /// Consumes the map and returns its keys in slot-index order.
    pub fn into_keys(self) -> IntoKeys<K, V, PAGE, D> {
        IntoKeys(self.into_iter())
    }

    /// Consumes the map and returns its values in slot-index order.
    pub fn into_values(self) -> IntoValues<K, V, PAGE, D> {
        IntoValues(self.into_iter())
    }
}

impl<K: DirectKey, V, const PAGE: usize, D: Directory> Default for PagedDirectMap<K, V, PAGE, D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: DirectKey, V: Clone, const PAGE: usize, D: Directory> Clone
    for PagedDirectMap<K, V, PAGE, D>
where
    Store<K, V, PAGE, D>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            pages: self.pages.clone(),
            len: self.len,
        }
    }
}

impl<K: DirectKey + fmt::Debug, V: fmt::Debug, const PAGE: usize, D: Directory> fmt::Debug
    for PagedDirectMap<K, V, PAGE, D>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

/// Compares entries in [`iter`](PagedDirectMap::iter)'s slot-index order,
/// across page sizes and directories.
impl<
    K: DirectKey,
    V: PartialEq<W>,
    W,
    const PAGE: usize,
    D: Directory,
    const P2: usize,
    D2: Directory,
> PartialEq<PagedDirectMap<K, W, P2, D2>> for PagedDirectMap<K, V, PAGE, D>
{
    fn eq(&self, other: &PagedDirectMap<K, W, P2, D2>) -> bool {
        self.len == other.len
            && self
                .iter()
                .zip(other.iter())
                .all(|((key, value), (other_key, other_value))| {
                    key == other_key && value == other_value
                })
    }
}

/// Map [`PartialEq`] is an equivalence relation when the values implement [`Eq`].
impl<K: DirectKey, V: Eq, const PAGE: usize, D: Directory> Eq for PagedDirectMap<K, V, PAGE, D> {}

/// Hashes the length, then the entries in slot-index order, consistent with
/// [`PartialEq`].
impl<K: DirectKey + Hash, V: Hash, const PAGE: usize, D: Directory> Hash
    for PagedDirectMap<K, V, PAGE, D>
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.len);
        for entry in self {
            entry.hash(state);
        }
    }
}

/// `map[key]` returns the value for `key`.
///
/// # Panics
///
/// Panics if `key` is not in the map.
impl<K: DirectKey, V, const PAGE: usize, D: Directory> Index<K> for PagedDirectMap<K, V, PAGE, D> {
    type Output = V;

    fn index(&self, key: K) -> &V {
        self.get(key).expect("key not in PagedDirectMap")
    }
}

/// Inserts each entry with [`insert`](PagedDirectMap::insert), so later
/// entries overwrite earlier ones in the same slot.
impl<K: DirectKey, V, const PAGE: usize, D: Directory> Extend<(K, V)>
    for PagedDirectMap<K, V, PAGE, D>
{
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        for (key, value) in iter {
            self.insert(key, value);
        }
    }
}

impl<'a, K: DirectKey + 'a, V: Copy + 'a, const PAGE: usize, D: Directory> Extend<(&'a K, &'a V)>
    for PagedDirectMap<K, V, PAGE, D>
{
    fn extend<I: IntoIterator<Item = (&'a K, &'a V)>>(&mut self, iter: I) {
        self.extend(iter.into_iter().map(|(&key, &value)| (key, value)));
    }
}

impl<K: DirectKey, V, const PAGE: usize, D: Directory> FromIterator<(K, V)>
    for PagedDirectMap<K, V, PAGE, D>
{
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut map = Self::new();
        map.extend(iter);
        map
    }
}

impl<'a, K: DirectKey, V, const PAGE: usize, D: Directory> IntoIterator
    for &'a PagedDirectMap<K, V, PAGE, D>
{
    type Item = (K, &'a V);
    type IntoIter = Iter<'a, K, V, PAGE, D>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K: DirectKey, V, const PAGE: usize, D: Directory> IntoIterator
    for &'a mut PagedDirectMap<K, V, PAGE, D>
{
    type Item = (K, &'a mut V);
    type IntoIter = IterMut<'a, K, V, PAGE, D>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<K: DirectKey, V, const PAGE: usize, D: Directory> IntoIterator
    for PagedDirectMap<K, V, PAGE, D>
{
    type Item = (K, V);
    type IntoIter = IntoIter<K, V, PAGE, D>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter(Cursor::new(self.pages.into_pages(), self.len))
    }
}

// Iteration: a cursor flattens the page iterator into indexed slots, like
// `Iterator::flatten`, rebuilds each entry's key from its slot index and tag,
// and stops once it has yielded `remaining` entries so trailing empty slots
// are not scanned.

/// The slots of one page (or part of one), each with its slot index.
#[derive(Clone, Default)]
struct Indexed<S> {
    /// The slot index of the next slot from the front.
    start: usize,
    slots: S,
}

impl<S: ExactSizeIterator + DoubleEndedIterator> Iterator for Indexed<S> {
    type Item = (usize, S::Item);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let slot = self.slots.next()?;
        let index = self.start;
        // Wraps only past a slot at `usize::MAX`, the last one there is.
        self.start = index.wrapping_add(1);
        Some((index, slot))
    }
}

impl<S: ExactSizeIterator + DoubleEndedIterator> DoubleEndedIterator for Indexed<S> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let slot = self.slots.next_back()?;
        Some((self.start.wrapping_add(self.slots.len()), slot))
    }
}

/// Turns a page from a page iterator into an iterator over its indexed
/// slots.
trait PageSlots {
    type Slots: DoubleEndedIterator + Default;
    fn slots(self) -> Self::Slots;
}

/// The first slot index of page `number`. Only a store that breaks the page
/// contract can make this wrap, which gives wrong keys but nothing worse.
#[inline]
fn page_start<const PAGE: usize>(number: usize) -> usize {
    number.wrapping_mul(PAGE)
}

/// A part of a page cut by [`Trim`]: its first slot index and its slots.
impl<'a, T> PageSlots for (usize, &'a [T]) {
    type Slots = Indexed<slice::Iter<'a, T>>;
    fn slots(self) -> Self::Slots {
        Indexed {
            start: self.0,
            slots: self.1.iter(),
        }
    }
}

impl<'a, T> PageSlots for (usize, &'a mut [T]) {
    type Slots = Indexed<slice::IterMut<'a, T>>;
    fn slots(self) -> Self::Slots {
        Indexed {
            start: self.0,
            slots: self.1.iter_mut(),
        }
    }
}

/// A whole page: its page number and its slots.
impl<'a, T, const PAGE: usize> PageSlots for (usize, &'a [T; PAGE]) {
    type Slots = Indexed<slice::Iter<'a, T>>;
    fn slots(self) -> Self::Slots {
        Indexed {
            start: page_start::<PAGE>(self.0),
            slots: self.1.iter(),
        }
    }
}

impl<'a, T, const PAGE: usize> PageSlots for (usize, &'a mut [T; PAGE]) {
    type Slots = Indexed<slice::IterMut<'a, T>>;
    fn slots(self) -> Self::Slots {
        Indexed {
            start: page_start::<PAGE>(self.0),
            slots: self.1.iter_mut(),
        }
    }
}

impl<T, const PAGE: usize> PageSlots for (usize, Box<[T; PAGE]>) {
    type Slots = Indexed<vec::IntoIter<T>>;
    fn slots(self) -> Self::Slots {
        let page: Box<[T]> = self.1;
        Indexed {
            start: page_start::<PAGE>(self.0),
            slots: Vec::from(page).into_iter(),
        }
    }
}

/// Turns the slot at a slot index into the entry it holds, if any, with the
/// key of type `K` rebuilt.
trait SlotEntry<K> {
    type Entry;
    fn entry(self, index: usize) -> Option<Self::Entry>;
}

impl<'a, K: DirectKey, V: 'a> SlotEntry<K> for &'a Option<(K::Tag, V)> {
    type Entry = (K, &'a V);
    #[inline]
    fn entry(self, index: usize) -> Option<Self::Entry> {
        let (tag, value) = self.as_ref()?;
        Some((K::from_parts(index, *tag), value))
    }
}

impl<'a, K: DirectKey, V: 'a> SlotEntry<K> for &'a mut Option<(K::Tag, V)> {
    type Entry = (K, &'a mut V);
    #[inline]
    fn entry(self, index: usize) -> Option<Self::Entry> {
        let (tag, value) = self.as_mut()?;
        Some((K::from_parts(index, *tag), value))
    }
}

impl<K: DirectKey, V> SlotEntry<K> for Option<(K::Tag, V)> {
    type Entry = (K, V);
    #[inline]
    fn entry(self, index: usize) -> Option<Self::Entry> {
        let (tag, value) = self?;
        Some((K::from_parts(index, tag), value))
    }
}

struct Cursor<K, P, S> {
    pages: Fuse<P>,
    front: S,
    back: S,
    remaining: usize,
    key: PhantomData<fn() -> K>,
}

impl<K, P: Clone, S: Clone> Clone for Cursor<K, P, S> {
    fn clone(&self) -> Self {
        Self {
            pages: self.pages.clone(),
            front: self.front.clone(),
            back: self.back.clone(),
            remaining: self.remaining,
            key: PhantomData,
        }
    }
}

impl<K, P, S, X> Cursor<K, P, S>
where
    P: DoubleEndedIterator,
    P::Item: PageSlots<Slots = S>,
    S: DoubleEndedIterator<Item = (usize, X)> + Default,
    X: SlotEntry<K>,
{
    #[inline]
    fn len(&self) -> usize {
        self.remaining
    }

    fn new(pages: P, remaining: usize) -> Self {
        Self {
            pages: pages.fuse(),
            front: S::default(),
            back: S::default(),
            remaining,
            key: PhantomData,
        }
    }

    fn next(&mut self) -> Option<X::Entry> {
        while self.remaining > 0 {
            let (index, slot) = match self.front.next() {
                Some(slot) => slot,
                None => match self.pages.next() {
                    Some(page) => {
                        self.front = page.slots();
                        continue;
                    }
                    // The pages are used up; the rest is in `back`.
                    None => match self.back.next() {
                        Some(slot) => slot,
                        None => break,
                    },
                },
            };
            if let Some(entry) = slot.entry(index) {
                self.remaining -= 1;
                return Some(entry);
            }
        }
        // Only a store that breaks the page contract can end here with
        // `remaining > 0`; zero it so `len` stays truthful.
        self.remaining = 0;
        None
    }

    fn next_back(&mut self) -> Option<X::Entry> {
        while self.remaining > 0 {
            let (index, slot) = match self.back.next_back() {
                Some(slot) => slot,
                None => match self.pages.next_back() {
                    Some(page) => {
                        self.back = page.slots();
                        continue;
                    }
                    None => match self.front.next_back() {
                        Some(slot) => slot,
                        None => break,
                    },
                },
            };
            if let Some(entry) = slot.entry(index) {
                self.remaining -= 1;
                return Some(entry);
            }
        }
        // Only a store that breaks the page contract can end here with
        // `remaining > 0`; zero it so `len` stays truthful.
        self.remaining = 0;
        None
    }
}

/// The entries removed by [`PagedDirectMap::drain`], in slot-index order.
pub struct Drain<'a, K: DirectKey, V, const PAGE: usize = DEFAULT_PAGE, D: Directory = Flat> {
    map: &'a mut PagedDirectMap<K, V, PAGE, D>,
    // The slot index to resume from; `None` once the last slot is passed.
    next: Option<usize>,
}

impl<K: DirectKey, V, const PAGE: usize, D: Directory> Iterator for Drain<'_, K, V, PAGE, D> {
    type Item = (K, V);

    fn next(&mut self) -> Option<(K, V)> {
        let start = self.next?;
        let Some((key, _)) = self.map.slot_range(start..).next() else {
            self.next = None;
            return None;
        };
        let index = key.index();
        self.next = index.checked_add(1);
        self.map.remove_occupant(index)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // Every entry before the cursor has been removed.
        (self.map.len, Some(self.map.len))
    }
}

impl<K: DirectKey, V, const PAGE: usize, D: Directory> ExactSizeIterator
    for Drain<'_, K, V, PAGE, D>
{
}

impl<K: DirectKey, V, const PAGE: usize, D: Directory> FusedIterator for Drain<'_, K, V, PAGE, D> {}

impl<K: DirectKey, V, const PAGE: usize, D: Directory> Drop for Drain<'_, K, V, PAGE, D> {
    fn drop(&mut self) {
        self.for_each(drop);
    }
}

impl<K: DirectKey + fmt::Debug, V: fmt::Debug, const PAGE: usize, D: Directory> fmt::Debug
    for Drain<'_, K, V, PAGE, D>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Every entry still in the map is yet to be yielded.
        f.debug_list().entries(self.map.iter()).finish()
    }
}

/// The entries removed by [`PagedDirectMap::extract_if`], in slot-index
/// order.
pub struct ExtractIf<'a, K: DirectKey, V, F, const PAGE: usize = DEFAULT_PAGE, D: Directory = Flat>
{
    map: &'a mut PagedDirectMap<K, V, PAGE, D>,
    next: Option<usize>,
    pred: F,
}

impl<K: DirectKey, V, F: FnMut(&K, &mut V) -> bool, const PAGE: usize, D: Directory> Iterator
    for ExtractIf<'_, K, V, F, PAGE, D>
{
    type Item = (K, V);

    fn next(&mut self) -> Option<(K, V)> {
        loop {
            let start = self.next?;
            let Some((key, value)) = self.map.slot_range_mut(start..).next() else {
                self.next = None;
                return None;
            };
            let index = key.index();
            // Advance first: if `pred` panics, this entry stays.
            self.next = index.checked_add(1);
            if (self.pred)(&key, value) {
                return self.map.remove_occupant(index);
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(if self.next.is_some() { self.map.len } else { 0 }))
    }
}

impl<K: DirectKey, V, F: FnMut(&K, &mut V) -> bool, const PAGE: usize, D: Directory> FusedIterator
    for ExtractIf<'_, K, V, F, PAGE, D>
{
}

impl<K: DirectKey + fmt::Debug, V: fmt::Debug, F, const PAGE: usize, D: Directory> fmt::Debug
    for ExtractIf<'_, K, V, F, PAGE, D>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExtractIf")
            .field("next", &self.next)
            .finish_non_exhaustive()
    }
}

/// The slot-index bounds of a key range.
fn index_bounds<K: Key>(keys: &impl RangeBounds<K>) -> (Bound<usize>, Bound<usize>) {
    (
        keys.start_bound().map(|key| key.index()),
        keys.end_bound().map(|key| key.index()),
    )
}

/// A range of slot indices, resolved: the slots `first..=last`, the pages
/// that hold them, and the most entries they can hold. An empty range has
/// `first > last`, an empty page range and `most == 0`.
struct Span {
    first: usize,
    last: usize,
    pages: core::ops::RangeInclusive<usize>,
    most: usize,
}

impl Span {
    /// Resolves `indices` for a map of `len` entries.
    fn new<const PAGE: usize>(indices: &impl RangeBounds<usize>, len: usize) -> Self {
        let first = match indices.start_bound() {
            Bound::Included(&index) => Some(index),
            Bound::Excluded(&index) => index.checked_add(1),
            Bound::Unbounded => Some(0),
        };
        let last = match indices.end_bound() {
            Bound::Included(&index) => Some(index),
            Bound::Excluded(&index) => index.checked_sub(1),
            Bound::Unbounded => Some(usize::MAX),
        };
        match (first, last) {
            (Some(first), Some(last)) if first <= last => Self {
                first,
                last,
                pages: split::<PAGE>(first).0..=split::<PAGE>(last).0,
                most: (last - first).saturating_add(1).min(len),
            },
            #[allow(clippy::reversed_empty_ranges)]
            _ => Self {
                first: 1,
                last: 0,
                pages: 1..=0,
                most: 0,
            },
        }
    }
}

/// Cuts a page from a page iterator down to its slots in `first..=last`, with
/// the slot index of the first one.
trait Cut<const PAGE: usize> {
    type Part;
    fn cut(self, first: usize, last: usize) -> Self::Part;
}

/// The offsets within page `number` of the slot indices `first..=last`.
fn offsets<const PAGE: usize>(number: usize, first: usize, last: usize) -> core::ops::Range<usize> {
    let base = number.saturating_mul(PAGE);
    let end = match last.checked_sub(base) {
        Some(offset) => offset.saturating_add(1).min(PAGE),
        // Only a store that breaks the page contract yields a page past
        // `last`.
        None => 0,
    };
    first.saturating_sub(base).min(end)..end
}

impl<'a, T, const PAGE: usize> Cut<PAGE> for (usize, &'a [T; PAGE]) {
    type Part = (usize, &'a [T]);
    fn cut(self, first: usize, last: usize) -> Self::Part {
        let offsets = offsets::<PAGE>(self.0, first, last);
        let start = page_start::<PAGE>(self.0).wrapping_add(offsets.start);
        (start, &self.1[offsets])
    }
}

impl<'a, T, const PAGE: usize> Cut<PAGE> for (usize, &'a mut [T; PAGE]) {
    type Part = (usize, &'a mut [T]);
    fn cut(self, first: usize, last: usize) -> Self::Part {
        let offsets = offsets::<PAGE>(self.0, first, last);
        let start = page_start::<PAGE>(self.0).wrapping_add(offsets.start);
        (start, &mut self.1[offsets])
    }
}

/// A page iterator whose pages are cut down to the slots `first..=last`.
#[derive(Clone)]
struct Trim<P, const PAGE: usize> {
    pages: P,
    first: usize,
    last: usize,
}

impl<P: Iterator, const PAGE: usize> Iterator for Trim<P, PAGE>
where
    P::Item: Cut<PAGE>,
{
    type Item = <P::Item as Cut<PAGE>>::Part;

    fn next(&mut self) -> Option<Self::Item> {
        let page = self.pages.next()?;
        Some(page.cut(self.first, self.last))
    }
}

impl<P: DoubleEndedIterator, const PAGE: usize> DoubleEndedIterator for Trim<P, PAGE>
where
    P::Item: Cut<PAGE>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        let page = self.pages.next_back()?;
        Some(page.cut(self.first, self.last))
    }
}

macro_rules! range_iterator {
    (
        $(#[$doc:meta])*
        $name:ident, $pages:ident, $slots:ident, $item:ty
    ) => {
        $(#[$doc])*
        pub struct $name<'a, K: DirectKey, V, const PAGE: usize = DEFAULT_PAGE, D: Directory = Flat>(
            Cursor<K, Trim<$pages<'a, K, V, PAGE, D>, PAGE>, Indexed<slice::$slots<'a, Slot<K, V>>>>,
        )
        where
            K: 'a,
            V: 'a,
            D: 'a;

        impl<'a, K: DirectKey, V, const PAGE: usize, D: Directory> Iterator for $name<'a, K, V, PAGE, D> {
            type Item = $item;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                self.0.next()
            }

            /// The upper bound is loose: the smaller of the map's length
            /// and the number of slots in the range, less the entries
            /// already yielded.
            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                (0, Some(self.0.remaining))
            }
        }

        impl<'a, K: DirectKey, V, const PAGE: usize, D: Directory> DoubleEndedIterator
            for $name<'a, K, V, PAGE, D>
        {
            #[inline]
            fn next_back(&mut self) -> Option<Self::Item> {
                self.0.next_back()
            }
        }

        impl<'a, K: DirectKey, V, const PAGE: usize, D: Directory> FusedIterator
            for $name<'a, K, V, PAGE, D>
        {
        }

        impl<'a, K: DirectKey, V, const PAGE: usize, D: Directory> fmt::Debug for $name<'a, K, V, PAGE, D> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct(stringify!($name)).finish_non_exhaustive()
            }
        }
    };
}

range_iterator!(
    /// Iterator over the entries of a [`PagedDirectMap`] in a range of slot
    /// indices, from [`range`](PagedDirectMap::range) or
    /// [`slot_range`](PagedDirectMap::slot_range).
    Range,
    Pages,
    Iter,
    (K, &'a V)
);

range_iterator!(
    /// Iterator over the entries of a [`PagedDirectMap`] in a range of slot
    /// indices, with mutable values, from
    /// [`range_mut`](PagedDirectMap::range_mut) or
    /// [`slot_range_mut`](PagedDirectMap::slot_range_mut).
    RangeMut,
    PagesMut,
    IterMut,
    (K, &'a mut V)
);

impl<K: DirectKey, V, const PAGE: usize, D: Directory> Clone for Range<'_, K, V, PAGE, D> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

macro_rules! entry_iterator {
    (
        $(#[$doc:meta])*
        $name:ident<$($lt:lifetime,)? K, V, PAGE, D>,
        $inner:ty,
        $item:ty,
        |$entry:pat_param| $map:expr
    ) => {
        $(#[$doc])*
        pub struct $name<$($lt,)? K: DirectKey, V, const PAGE: usize = DEFAULT_PAGE, D: Directory = Flat>($inner)
        where
            $(K: $lt, V: $lt, D: $lt)?;

        impl<$($lt,)? K: DirectKey, V, const PAGE: usize, D: Directory> Iterator for $name<$($lt,)? K, V, PAGE, D>
        where
            $(K: $lt, V: $lt, D: $lt)?
        {
            type Item = $item;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                self.0.next().map(|$entry| $map)
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                let len = self.len();
                (len, Some(len))
            }
        }

        impl<$($lt,)? K: DirectKey, V, const PAGE: usize, D: Directory> DoubleEndedIterator
            for $name<$($lt,)? K, V, PAGE, D>
        where
            $(K: $lt, V: $lt, D: $lt)?
        {
            #[inline]
            fn next_back(&mut self) -> Option<Self::Item> {
                self.0.next_back().map(|$entry| $map)
            }
        }

        impl<$($lt,)? K: DirectKey, V, const PAGE: usize, D: Directory> ExactSizeIterator
            for $name<$($lt,)? K, V, PAGE, D>
        where
            $(K: $lt, V: $lt, D: $lt)?
        {
            #[inline]
            fn len(&self) -> usize {
                self.0.len()
            }
        }

        impl<$($lt,)? K: DirectKey, V, const PAGE: usize, D: Directory> FusedIterator
            for $name<$($lt,)? K, V, PAGE, D>
        where
            $(K: $lt, V: $lt, D: $lt)?
        {
        }

        impl<$($lt,)? K: DirectKey, V, const PAGE: usize, D: Directory> fmt::Debug
            for $name<$($lt,)? K, V, PAGE, D>
        where
            $(K: $lt, V: $lt, D: $lt)?
        {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct(stringify!($name))
                    .field("remaining", &self.len())
                    .finish_non_exhaustive()
            }
        }
    };
}

entry_iterator!(
    /// Iterator over the entries of a [`PagedDirectMap`] in slot-index
    /// order, from [`iter`](PagedDirectMap::iter).
    Iter<'a, K, V, PAGE, D>,
    Cursor<K, Pages<'a, K, V, PAGE, D>, Indexed<slice::Iter<'a, Slot<K, V>>>>,
    (K, &'a V),
    |entry| entry
);

entry_iterator!(
    /// Iterator over the entries of a [`PagedDirectMap`] with mutable
    /// values, from [`iter_mut`](PagedDirectMap::iter_mut).
    IterMut<'a, K, V, PAGE, D>,
    Cursor<K, PagesMut<'a, K, V, PAGE, D>, Indexed<slice::IterMut<'a, Slot<K, V>>>>,
    (K, &'a mut V),
    |entry| entry
);

entry_iterator!(
    /// Owning iterator over the entries of a [`PagedDirectMap`] in
    /// slot-index order.
    IntoIter<K, V, PAGE, D>,
    Cursor<K, IntoPages<K, V, PAGE, D>, Indexed<vec::IntoIter<Slot<K, V>>>>,
    (K, V),
    |entry| entry
);

entry_iterator!(
    /// Iterator over the keys of a [`PagedDirectMap`], from
    /// [`keys`](PagedDirectMap::keys).
    Keys<'a, K, V, PAGE, D>,
    Iter<'a, K, V, PAGE, D>,
    K,
    |(key, _)| key
);

entry_iterator!(
    /// Iterator over the values of a [`PagedDirectMap`], from
    /// [`values`](PagedDirectMap::values).
    Values<'a, K, V, PAGE, D>,
    Iter<'a, K, V, PAGE, D>,
    &'a V,
    |(_, value)| value
);

entry_iterator!(
    /// Mutable iterator over the values of a [`PagedDirectMap`], from
    /// [`values_mut`](PagedDirectMap::values_mut).
    ValuesMut<'a, K, V, PAGE, D>,
    IterMut<'a, K, V, PAGE, D>,
    &'a mut V,
    |(_, value)| value
);

entry_iterator!(
    /// Owning iterator over the keys of a [`PagedDirectMap`], from
    /// [`into_keys`](PagedDirectMap::into_keys).
    IntoKeys<K, V, PAGE, D>,
    IntoIter<K, V, PAGE, D>,
    K,
    |(key, _)| key
);

entry_iterator!(
    /// Owning iterator over the values of a [`PagedDirectMap`], from
    /// [`into_values`](PagedDirectMap::into_values).
    IntoValues<K, V, PAGE, D>,
    IntoIter<K, V, PAGE, D>,
    V,
    |(_, value)| value
);

impl<K: DirectKey, V, const PAGE: usize, D: Directory> Clone for Iter<'_, K, V, PAGE, D> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<K: DirectKey, V, const PAGE: usize, D: Directory> Clone for Keys<'_, K, V, PAGE, D> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<K: DirectKey, V, const PAGE: usize, D: Directory> Clone for Values<'_, K, V, PAGE, D> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

#[cfg(feature = "rayon")]
pub use rayon_impl::{ParIter, ParIterMut};

#[cfg(feature = "rayon")]
mod rayon_impl {
    use super::*;
    use rayon::iter::plumbing::UnindexedConsumer;
    use rayon::prelude::*;

    /// Unindexed parallel iterator over the entries of a [`PagedDirectMap`].
    pub struct ParIter<'a, K: DirectKey, V, const PAGE: usize = DEFAULT_PAGE>(
        Vec<(usize, &'a [Slot<K, V>; PAGE])>,
    );

    /// Unindexed parallel iterator over the entries of a [`PagedDirectMap`], with mutable values.
    pub struct ParIterMut<'a, K: DirectKey, V, const PAGE: usize = DEFAULT_PAGE>(
        Vec<(usize, &'a mut [Slot<K, V>; PAGE])>,
    );

    impl<'a, K: DirectKey + Send, V: Sync, const PAGE: usize, D: Directory> IntoParallelIterator
        for &'a PagedDirectMap<K, V, PAGE, D>
    where
        K::Tag: Sync,
        Store<K, V, PAGE, D>: Sync,
    {
        type Item = (K, &'a V);
        type Iter = ParIter<'a, K, V, PAGE>;

        fn into_par_iter(self) -> Self::Iter {
            ParIter(self.pages.pages().collect())
        }
    }

    impl<'a, K: DirectKey + Send, V: Send, const PAGE: usize, D: Directory> IntoParallelIterator
        for &'a mut PagedDirectMap<K, V, PAGE, D>
    where
        K::Tag: Send,
    {
        type Item = (K, &'a mut V);
        type Iter = ParIterMut<'a, K, V, PAGE>;

        fn into_par_iter(self) -> Self::Iter {
            ParIterMut(self.pages.pages_mut().collect())
        }
    }

    impl<'a, K: DirectKey + Send, V: Sync, const PAGE: usize> ParallelIterator
        for ParIter<'a, K, V, PAGE>
    where
        K::Tag: Sync,
    {
        type Item = (K, &'a V);

        fn drive_unindexed<C: UnindexedConsumer<Self::Item>>(self, consumer: C) -> C::Result {
            self.0
                .into_par_iter()
                .flat_map_iter(|page| {
                    page.slots()
                        .filter_map(|(index, slot)| SlotEntry::<K>::entry(slot, index))
                })
                .drive_unindexed(consumer)
        }
    }

    impl<'a, K: DirectKey + Send, V: Send, const PAGE: usize> ParallelIterator
        for ParIterMut<'a, K, V, PAGE>
    where
        K::Tag: Send,
    {
        type Item = (K, &'a mut V);

        fn drive_unindexed<C: UnindexedConsumer<Self::Item>>(self, consumer: C) -> C::Result {
            self.0
                .into_par_iter()
                .flat_map_iter(|page| {
                    page.slots()
                        .filter_map(|(index, slot)| SlotEntry::<K>::entry(slot, index))
                })
                .drive_unindexed(consumer)
        }
    }
}

#[cfg(feature = "serde")]
mod serde_impl {
    use super::*;
    use core::marker::PhantomData;
    use serde::de::{Deserialize, Deserializer, MapAccess, Visitor};
    use serde::ser::{Serialize, Serializer};

    /// Serializes as a map from keys to values, in slot-index order.
    impl<K: DirectKey + Serialize, V: Serialize, const PAGE: usize, D: Directory> Serialize
        for PagedDirectMap<K, V, PAGE, D>
    {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            serializer.collect_map(self.iter())
        }
    }

    /// Deserializes from a map. Entries are inserted in input order with
    /// [`insert`](PagedDirectMap::insert), so a later entry overwrites an
    /// earlier one in the same slot.
    ///
    /// Memory follows the key indices, not the input length: one key can
    /// allocate a page, and with [`Flat`] a directory up to that page. For
    /// untrusted input, use [`Radix`](crate::page::Radix) or
    /// [`Tree`](crate::page::Tree), or validate the keys first.
    impl<'de, K, V, const PAGE: usize, D> Deserialize<'de> for PagedDirectMap<K, V, PAGE, D>
    where
        K: DirectKey + Deserialize<'de>,
        V: Deserialize<'de>,
        D: Directory,
    {
        fn deserialize<De: Deserializer<'de>>(deserializer: De) -> Result<Self, De::Error> {
            struct MapVisitor<K, V, const PAGE: usize, D>(PhantomData<(K, V, D)>);

            impl<'de, K, V, const PAGE: usize, D> Visitor<'de> for MapVisitor<K, V, PAGE, D>
            where
                K: DirectKey + Deserialize<'de>,
                V: Deserialize<'de>,
                D: Directory,
            {
                type Value = PagedDirectMap<K, V, PAGE, D>;

                fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    f.write_str("a map of keys to values")
                }

                fn visit_map<A: MapAccess<'de>>(
                    self,
                    mut access: A,
                ) -> Result<Self::Value, A::Error> {
                    let mut map = PagedDirectMap::new();
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

/// A view into one key's place in a [`PagedDirectMap`], from
/// [`entry`](PagedDirectMap::entry).
pub enum Entry<'a, K: DirectKey, V, const PAGE: usize = DEFAULT_PAGE, D: Directory = Flat> {
    /// The key is in the map.
    Occupied(OccupiedEntry<'a, K, V>),
    /// The key is not in the map; its slot is empty or held by another
    /// generation of the key.
    Vacant(VacantEntry<'a, K, V, PAGE, D>),
}

/// An [`Entry`] for a key that is in the map.
pub struct OccupiedEntry<'a, K: DirectKey, V> {
    // Invariant: `*slot` is `Some`, `key` is the stored key (rebuilt from the
    // slot's index and tag), and `len` is the map's entry count.
    key: K,
    slot: &'a mut Slot<K, V>,
    len: &'a mut usize,
}

/// An [`Entry`] for a key that is not in the map.
pub struct VacantEntry<'a, K: DirectKey, V, const PAGE: usize = DEFAULT_PAGE, D: Directory = Flat> {
    map: &'a mut PagedDirectMap<K, V, PAGE, D>,
    key: K,
}

impl<'a, K: DirectKey, V, const PAGE: usize, D: Directory> Entry<'a, K, V, PAGE, D> {
    /// The entry's key: the stored key if occupied, else the key given to
    /// [`entry`](PagedDirectMap::entry).
    #[inline]
    pub fn key(&self) -> &K {
        match self {
            Entry::Occupied(entry) => entry.key(),
            Entry::Vacant(entry) => entry.key(),
        }
    }

    /// The value, inserting `default` first if the key is absent (evicting
    /// another generation in its slot).
    #[inline]
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(default),
        }
    }

    /// The value, inserting `make()` first if the key is absent. `make` runs
    /// before the map changes, so if it panics the map is unchanged.
    #[inline]
    pub fn or_insert_with(self, make: impl FnOnce() -> V) -> &'a mut V {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(make()),
        }
    }

    /// Like [`or_insert_with`](Self::or_insert_with), but `make` is given the
    /// key.
    #[inline]
    pub fn or_insert_with_key(self, make: impl FnOnce(&K) -> V) -> &'a mut V {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let value = make(&entry.key);
                entry.insert(value)
            }
        }
    }

    /// The value, inserting `V::default()` first if the key is absent.
    #[inline]
    pub fn or_default(self) -> &'a mut V
    where
        V: Default,
    {
        self.or_insert_with(V::default)
    }

    /// Calls `f` on the value if the key is present, and returns the entry.
    #[inline]
    pub fn and_modify(mut self, f: impl FnOnce(&mut V)) -> Self {
        if let Entry::Occupied(entry) = &mut self {
            f(entry.get_mut());
        }
        self
    }

    /// Stores `value` for the key and returns the occupied entry. An occupied
    /// entry keeps its stored key; a vacant one evicts any other generation.
    #[inline]
    pub fn insert_entry(self, value: V) -> OccupiedEntry<'a, K, V> {
        match self {
            Entry::Occupied(mut entry) => {
                entry.insert(value);
                entry
            }
            Entry::Vacant(entry) => entry.insert_entry(value),
        }
    }
}

impl<'a, K: DirectKey, V> OccupiedEntry<'a, K, V> {
    #[inline]
    fn pair(&self) -> &(K::Tag, V) {
        match &*self.slot {
            Some(pair) => pair,
            None => unreachable!("an occupied entry's slot is full"),
        }
    }

    #[inline]
    fn pair_mut(&mut self) -> &mut (K::Tag, V) {
        match &mut *self.slot {
            Some(pair) => pair,
            None => unreachable!("an occupied entry's slot is full"),
        }
    }

    /// The stored key.
    #[inline]
    pub fn key(&self) -> &K {
        &self.key
    }

    /// The value.
    #[inline]
    pub fn get(&self) -> &V {
        &self.pair().1
    }

    /// Mutable access to the value.
    #[inline]
    pub fn get_mut(&mut self) -> &mut V {
        &mut self.pair_mut().1
    }

    /// Mutable access to the value, for the lifetime of the map borrow.
    #[inline]
    pub fn into_mut(self) -> &'a mut V {
        match self.slot {
            Some((_, value)) => value,
            None => unreachable!("an occupied entry's slot is full"),
        }
    }

    /// Replaces the value and returns the old one; the stored key is kept.
    #[inline]
    pub fn insert(&mut self, value: V) -> V {
        mem::replace(self.get_mut(), value)
    }

    /// Removes the entry and returns its value.
    #[inline]
    pub fn remove(self) -> V {
        self.remove_entry().1
    }

    /// Removes the entry and returns the stored key and its value.
    pub fn remove_entry(self) -> (K, V) {
        match self.slot.take() {
            Some((_, value)) => {
                *self.len -= 1;
                (self.key, value)
            }
            None => unreachable!("an occupied entry's slot is full"),
        }
    }
}

impl<'a, K: DirectKey, V, const PAGE: usize, D: Directory> VacantEntry<'a, K, V, PAGE, D> {
    /// The key given to [`entry`](PagedDirectMap::entry).
    #[inline]
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Gives the key back.
    #[inline]
    pub fn into_key(self) -> K {
        self.key
    }

    /// The entry of another generation holding the key's slot, which an
    /// insert would evict.
    #[inline]
    pub fn occupant(&self) -> Option<(K, &V)> {
        self.map.occupant(self.key.index())
    }

    /// Stores `value` for the key and returns it, evicting (and dropping)
    /// any other generation in the slot.
    ///
    /// # Panics
    ///
    /// As [`PagedDirectMap::insert`]. If an evicted value's `drop` panics,
    /// the new entry is already in place.
    #[inline]
    pub fn insert(self, value: V) -> &'a mut V {
        self.insert_entry(value).into_mut()
    }

    /// Like [`insert`](Self::insert), but returns the occupied entry.
    pub fn insert_entry(self, value: V) -> OccupiedEntry<'a, K, V> {
        let VacantEntry { map, key } = self;
        let (slot, len) = map.slot_or_alloc(key.index());
        if slot.is_none() {
            *len += 1;
        }
        // Drop an evicted entry only once the slot is consistent: `V::drop`
        // may panic.
        drop(slot.replace((key.tag(), value)));
        OccupiedEntry { key, slot, len }
    }
}

impl<K: DirectKey + fmt::Debug, V: fmt::Debug, const PAGE: usize, D: Directory> fmt::Debug
    for Entry<'_, K, V, PAGE, D>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Entry::Occupied(entry) => f.debug_tuple("Occupied").field(entry).finish(),
            Entry::Vacant(entry) => f.debug_tuple("Vacant").field(entry).finish(),
        }
    }
}

impl<K: DirectKey + fmt::Debug, V: fmt::Debug> fmt::Debug for OccupiedEntry<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OccupiedEntry")
            .field("key", self.key())
            .field("value", self.get())
            .finish()
    }
}

impl<K: DirectKey + fmt::Debug, V: fmt::Debug, const PAGE: usize, D: Directory> fmt::Debug
    for VacantEntry<'_, K, V, PAGE, D>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VacantEntry")
            .field("key", &self.key)
            .field("occupant", &self.occupant())
            .finish()
    }
}

// The tests use `u32` keys, which implement `Key` only on 32- and 64-bit
// targets.
#[cfg(all(test, any(target_pointer_width = "32", target_pointer_width = "64")))]
mod tests {
    use super::*;
    use crate::page::Tree;
    use alloc::vec;

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    struct Gen(u32);

    impl Key for Gen {
        fn index(self) -> usize {
            (self.0 & 0xffff) as usize
        }
    }

    impl DirectKey for Gen {
        type Tag = u16;
        fn tag(self) -> u16 {
            (self.0 >> 16) as u16
        }
        fn from_parts(index: usize, tag: u16) -> Self {
            Gen(index as u32 | u32::from(tag) << 16)
        }
    }

    fn entries<K: DirectKey, V: Clone, const PAGE: usize, D: Directory>(
        map: &PagedDirectMap<K, V, PAGE, D>,
    ) -> Vec<(K, V)> {
        map.iter().map(|(k, v)| (k, v.clone())).collect()
    }

    #[test]
    fn iterates_in_key_order_from_both_ends() {
        let map: PagedDirectMap<u32, u32, 4> =
            [(9, 90), (1, 10), (40, 400), (2, 20)].into_iter().collect();
        assert_eq!(map.len(), 4);
        assert_eq!(map.allocated_pages(), 3);
        assert_eq!(entries(&map), [(1, 10), (2, 20), (9, 90), (40, 400)]);
        let back: Vec<u32> = map.keys().rev().collect();
        assert_eq!(back, [40, 9, 2, 1]);

        // Meeting in the middle yields each entry once.
        let mut it = map.iter();
        assert_eq!(it.len(), 4);
        assert_eq!(it.next(), Some((1, &10)));
        assert_eq!(it.next_back(), Some((40, &400)));
        assert_eq!(it.next_back(), Some((9, &90)));
        assert_eq!(it.next(), Some((2, &20)));
        assert_eq!(it.next(), None);
        assert_eq!(it.next_back(), None);

        let owned: Vec<(u32, u32)> = map.clone().into_iter().rev().collect();
        assert_eq!(owned, [(40, 400), (9, 90), (2, 20), (1, 10)]);
        let values: Vec<u32> = map.into_values().collect();
        assert_eq!(values, [10, 20, 90, 400]);
    }

    #[test]
    fn same_order_with_tree_directory() {
        let keys = [u32::MAX - 1, 5, 1 << 31, 6];
        let map: PagedDirectMap<u32, (), 1024, Tree> = keys.iter().map(|&k| (k, ())).collect();
        let got: Vec<u32> = map.keys().collect();
        assert_eq!(got, [5, 6, 1 << 31, u32::MAX - 1]);
        assert_eq!(map.allocated_pages(), 3);
    }

    #[test]
    fn generations_compete_for_one_slot() {
        let mut map: PagedDirectMap<Gen, &str> = PagedDirectMap::new();
        let old = Gen(3);
        let new = Gen(3 | 1 << 16);
        assert_eq!(map.insert(old, "old"), None);
        assert_eq!(map.try_insert(new, "x"), Err("x"));
        assert_eq!(map.get(new), None);
        assert_eq!(map.occupant(3), Some((old, &"old")));
        // `insert` evicts the other generation.
        assert_eq!(map.insert(new, "new"), Some("old"));
        assert_eq!(map.get(old), None);
        assert_eq!(map.get(new), Some(&"new"));
        assert_eq!(map.len(), 1);
        // `replace` returns the evicted key too.
        assert_eq!(map.replace(old, "back"), Some((new, "new")));
        assert_eq!(map.remove(new), None);
        assert_eq!(map.remove_occupant(3), Some((old, "back")));
        assert!(map.is_empty());
    }

    #[test]
    fn entry_occupied_and_vacant() {
        let mut map: PagedDirectMap<u32, i32> = PagedDirectMap::new();
        assert!(matches!(map.entry(9), Entry::Vacant(_)));
        assert_eq!(map.allocated_pages(), 0, "a vacant entry allocates nothing");
        *map.entry(9).or_insert(1) += 10;
        map.entry(9).and_modify(|v| *v *= 2).or_insert(-1);
        assert_eq!(map.get(9), Some(&22));
        assert_eq!(*map.entry(70_000).or_default(), 0);
        assert_eq!(*map.entry(5).or_insert_with_key(|&k| k as i32), 5);
        assert_eq!(map.len(), 3);
        match map.entry(9) {
            Entry::Occupied(mut e) => {
                assert_eq!((e.key(), e.get()), (&9, &22));
                assert_eq!(e.insert(4), 22);
                assert_eq!(e.remove_entry(), (9, 4));
            }
            Entry::Vacant(_) => panic!("9 is present"),
        }
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(9), None);
        let e = map.entry(9).insert_entry(8);
        assert_eq!(e.remove(), 8);
        assert_eq!(entries(&map), [(5, 5), (70_000, 0)]);
    }

    #[test]
    fn entry_treats_another_generation_as_vacant_and_evicts_it() {
        let mut map: PagedDirectMap<Gen, &str, 4, Tree> = PagedDirectMap::new();
        let (old, new) = (Gen(3), Gen(3 | 1 << 16));
        map.insert(old, "old");
        match map.entry(new) {
            Entry::Vacant(e) => {
                assert_eq!(e.occupant(), Some((old, &"old")));
                assert_eq!(*e.insert("new"), "new");
            }
            Entry::Occupied(_) => panic!("only `old` is present"),
        }
        assert_eq!(map.len(), 1);
        assert_eq!(entries(&map), [(new, "new")]);
        // An occupied entry keeps its stored key.
        assert_eq!(*map.entry(new).or_insert("x"), "new");
    }

    #[cfg(feature = "std")]
    #[test]
    fn entry_or_insert_with_panic_leaves_map_unchanged() {
        let mut map: PagedDirectMap<u32, i32> = PagedDirectMap::new();
        let result = std::panic::catch_unwind(core::panic::AssertUnwindSafe(|| {
            map.entry(3).or_insert_with(|| panic!("make"));
        }));
        assert!(result.is_err());
        assert!(map.is_empty());
        assert_eq!(map.allocated_pages(), 0);
    }

    #[test]
    fn get_or_insert_with_runs_make_only_when_absent() {
        let mut map: PagedDirectMap<u32, i32> = PagedDirectMap::new();
        *map.get_or_insert_with(4, || 1) += 10;
        *map.get_or_insert_with(4, || panic!("present")) += 100;
        assert_eq!(map[4], 111);
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn retain_clear_shrink_and_reset() {
        let mut map: PagedDirectMap<u32, u32, 4> = (0..12).map(|k| (k, k)).collect();
        map.retain(|&k, v| {
            *v *= 2;
            !(4..8).contains(&k)
        });
        assert_eq!(map.len(), 8);
        assert_eq!(map.get(9), Some(&18));
        assert_eq!(map.allocated_pages(), 3);
        map.shrink_to_fit();
        assert_eq!(map.allocated_pages(), 2);
        assert_eq!(map.get(9), Some(&18));

        map.clear();
        assert!(map.is_empty());
        assert_eq!(map.iter().next(), None);
        assert_eq!(map.allocated_pages(), 2);
        map.reset();
        assert_eq!(map.allocated_pages(), 0);
    }

    fn ordered_queries<D: Directory>() {
        let keys = [1_u32, 2, 9, 40, 41, 43];
        let mut map: PagedDirectMap<u32, u32, 4, D> = keys.iter().map(|&k| (k, k * 10)).collect();
        // Leave an allocated page with no entry in the middle.
        map.insert(20, 0);
        map.remove(20);
        let range = |r: (Bound<u32>, Bound<u32>)| -> Vec<u32> {
            let got: Vec<u32> = map.range(r).map(|(k, _)| k).collect();
            let mut back: Vec<u32> = map.range(r).rev().map(|(k, _)| k).collect();
            back.reverse();
            assert_eq!(got, back);
            got
        };
        use Bound::{Excluded as Ex, Included as In, Unbounded as Un};
        assert_eq!(range((Un, Un)), keys);
        assert_eq!(range((In(2), Ex(41))), [2, 9, 40]);
        assert_eq!(range((Ex(2), In(41))), [9, 40, 41]);
        assert_eq!(range((In(3), In(8))), [0_u32; 0]);
        assert_eq!(range((In(9), In(9))), [9]);
        assert_eq!(range((In(42), Un)), [43]);
        assert_eq!(range((In(44), Un)), [0_u32; 0]);
        assert_eq!(range((In(41), In(2))), [0_u32; 0]);
        assert_eq!(range((Ex(u32::MAX), Un)), [0_u32; 0]);
        assert_eq!(range((Un, Ex(0))), [0_u32; 0]);
        assert_eq!(map.slot_range(..).count(), keys.len());
        assert_eq!(map.slot_range((Ex(usize::MAX), Un)).next(), None);
        assert_eq!(map.slot_range(usize::MAX..).next(), None);

        // Both ends meet inside one page without repeating an entry.
        {
            let mut it = map.range(40..=43);
            assert_eq!(it.size_hint(), (0, Some(4)));
            assert_eq!(it.next(), Some((40, &400)));
            assert_eq!(it.next_back(), Some((43, &430)));
            assert_eq!(it.clone().next_back(), Some((41, &410)));
            assert_eq!(it.next(), Some((41, &410)));
            assert_eq!(it.next(), None);
            assert_eq!(it.next_back(), None);
        }

        assert_eq!(map.next_occupant(0), Some((1, &10)));
        assert_eq!(map.next_occupant(9), Some((9, &90)));
        assert_eq!(map.next_occupant(10), Some((40, &400)));
        assert_eq!(map.next_occupant(44), None);
        assert_eq!(map.prev_occupant(0), None);
        assert_eq!(map.prev_occupant(39), Some((9, &90)));
        assert_eq!(map.prev_occupant(usize::MAX), Some((43, &430)));
        assert_eq!(map.first_key_value(), Some((1, &10)));
        assert_eq!(map.last_key_value(), Some((43, &430)));

        for (_, value) in map.range_mut(2..41) {
            *value += 1;
        }
        assert_eq!(
            entries(&map),
            [(1, 10), (2, 21), (9, 91), (40, 401), (41, 410), (43, 430)]
        );
        assert_eq!(map.slot_range_mut(40..).next_back(), Some((43, &mut 430)));

        assert_eq!(map.pop_first(), Some((1, 10)));
        assert_eq!(map.pop_last(), Some((43, 430)));
        assert_eq!(map.len(), 4);
        assert_eq!(map.first_key_value(), Some((2, &21)));
        map.clear();
        assert_eq!(map.pop_first(), None);
        assert_eq!(map.pop_last(), None);
        assert_eq!(map.next_occupant(0), None);
        assert_eq!(map.range(..).next(), None);
    }

    #[test]
    fn ordered_queries_with_both_directories() {
        ordered_queries::<Flat>();
        ordered_queries::<Tree>();
    }

    #[test]
    fn a_range_bounds_generational_keys_by_index() {
        let mut map: PagedDirectMap<Gen, u8> = PagedDirectMap::new();
        let held = Gen(3 | 2 << 16);
        map.insert(held, 1);
        map.insert(Gen(7), 2);
        let other = Gen(3);
        assert_eq!(map.range(other..=other).next(), Some((held, &1)));
        assert_eq!(map.range(Gen(4 | 9 << 16)..).next(), Some((Gen(7), &2)));
    }

    #[cfg(feature = "std")]
    #[test]
    fn eq_and_hash_ignore_pages() {
        use core::hash::BuildHasher;
        let a: PagedDirectMap<u32, u8, 4> = [(1, 1), (9, 9)].into_iter().collect();
        let mut b: PagedDirectMap<u32, u8, 4> = [(9, 9), (5, 5), (1, 1)].into_iter().collect();
        assert_ne!(a, b);
        b.remove(5);
        assert_eq!(a, b);
        let s = std::hash::RandomState::new();
        assert_eq!(s.hash_one(&a), s.hash_one(&b));
        assert_eq!(vec![(1, 1), (9, 9)], entries(&b));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_round_trip() {
        let map: PagedDirectMap<u32, u8> = [(70_000, 7), (2, 2)].into_iter().collect();
        let json = serde_json::to_string(&map).unwrap();
        assert_eq!(json, r#"{"2":2,"70000":7}"#);
        let back: PagedDirectMap<u32, u8> = serde_json::from_str(&json).unwrap();
        assert_eq!(back, map);
    }
}
