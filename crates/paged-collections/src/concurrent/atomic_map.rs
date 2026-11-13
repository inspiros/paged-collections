//! Lock-free direct-address maps whose values fit in one atomic word.
//!
//! Slots use acquire loads and acquire-release read-modify-write operations;
//! failed compare-exchanges use acquire ordering. A value can therefore
//! publish data indexed by that value. Pages remain allocated during shared
//! access, and slot operations never wait for another thread to release a lock.

use alloc::boxed::Box;
use core::fmt;
use core::mem;
use core::num::NonZeroU32;
use core::ops::{Bound, RangeBounds};

use super::atomic::{AtomicU64, AtomicUsize, Ordering};
use super::index::{INDEX_CHUNKS, page_number};
use super::spine::Spine;
use crate::id_alloc::Id;
use crate::page::DEFAULT_PAGE;

struct Page {
    slots: Box<[AtomicU64]>,
}

struct Words<const PAGE: usize> {
    pages: Spine<Page, INDEX_CHUNKS>,
    len: AtomicUsize,
}

impl<const PAGE: usize> Words<PAGE> {
    fn new(name: &str) -> Self {
        assert!(
            PAGE > 0 && PAGE <= isize::MAX as usize / mem::size_of::<AtomicU64>(),
            "{name}: invalid page capacity"
        );
        Self {
            pages: Spine::new(),
            len: AtomicUsize::new(0),
        }
    }

    fn slot(&self, index: usize) -> Option<&AtomicU64> {
        self.pages
            .get(page_number::<Page, PAGE>(index)?)
            .map(|page| &page.slots[index % PAGE])
    }

    fn ensure_slot(&self, index: usize, name: &str) -> &AtomicU64 {
        let number = page_number::<Page, PAGE>(index)
            .unwrap_or_else(|| panic!("{name}: key index exceeds direct spine range"));
        let page = self.pages.get_or_insert_with(number, || {
            Box::new(Page {
                slots: (0..PAGE).map(|_| AtomicU64::new(0)).collect(),
            })
        });
        &page.slots[index % PAGE]
    }

    fn load(&self, index: usize) -> u64 {
        self.slot(index)
            .map_or(0, |slot| slot.load(Ordering::Acquire))
    }

    fn count_change(&self, old: u64, new: u64) {
        // Count after the slot's linearization point. A removal can reach
        // this counter before the insertion it undoes, so wrap deliberately.
        match (old == 0, new == 0) {
            (true, false) => {
                self.len.fetch_add(1, Ordering::Relaxed);
            }
            (false, true) => {
                self.len.fetch_sub(1, Ordering::Relaxed);
            }
            _ => {}
        }
    }

    fn len(&self) -> usize {
        (self.len.load(Ordering::Relaxed) as isize).max(0) as usize
    }

    fn swap(&self, index: usize, new: u64, name: &str) -> u64 {
        let slot = if new == 0 {
            let Some(slot) = self.slot(index) else {
                return 0;
            };
            slot
        } else {
            self.ensure_slot(index, name)
        };
        let old = slot.swap(new, Ordering::AcqRel);
        self.count_change(old, new);
        old
    }

    fn try_insert(&self, index: usize, new: u64, name: &str) -> Result<(), u64> {
        let slot = self.ensure_slot(index, name);
        match slot.compare_exchange(0, new, Ordering::AcqRel, Ordering::Acquire) {
            Ok(old) => {
                self.count_change(old, new);
                Ok(())
            }
            Err(actual) => Err(actual),
        }
    }

    fn update(
        &self,
        index: usize,
        name: &str,
        install: bool,
        mut f: impl FnMut(u64) -> Option<u64>,
    ) -> Result<u64, u64> {
        let mut slot = if install {
            Some(self.ensure_slot(index, name))
        } else {
            self.slot(index)
        };
        let mut old = slot.map_or(0, |slot| slot.load(Ordering::Acquire));
        loop {
            let Some(new) = f(old) else {
                return Err(old);
            };
            let target = match slot {
                Some(slot) => slot,
                None => {
                    // Recheck after the callback: it may itself have installed
                    // this page. A no-op on a still absent page needs no CAS
                    // and linearizes at this directory lookup.
                    let found = if new == 0 {
                        self.slot(index)
                    } else {
                        Some(self.ensure_slot(index, name))
                    };
                    let Some(found) = found else {
                        return Ok(0);
                    };
                    slot = Some(found);
                    found
                }
            };
            match target.compare_exchange(old, new, Ordering::AcqRel, Ordering::Acquire) {
                Ok(previous) => {
                    self.count_change(previous, new);
                    return Ok(previous);
                }
                Err(actual) => old = actual,
            }
        }
    }

    fn range(&self, start: usize, end: usize) -> impl Iterator<Item = (usize, u64)> {
        self.pages
            .iter()
            .skip_while(move |(number, _)| *number < start / PAGE)
            .take_while(move |(number, _)| start <= end && *number <= end / PAGE)
            .flat_map(move |(number, page)| {
                let first = if number == start / PAGE {
                    start % PAGE
                } else {
                    0
                };
                let last = if number == end / PAGE {
                    end % PAGE + 1
                } else {
                    PAGE
                };
                // The end bound also prevents overflow in the final page
                // when PAGE does not divide the usize index space evenly.
                (first..last).filter_map(move |offset| {
                    let word = page.slots[offset].load(Ordering::Acquire);
                    (word != 0).then_some((number * PAGE + offset, word))
                })
            })
    }

    fn clear(&mut self) {
        for (_, page) in self.pages.iter_mut() {
            for slot in page.slots.iter() {
                slot.store(0, Ordering::Relaxed);
            }
        }
        self.len.store(0, Ordering::Relaxed);
    }

    fn reset(&mut self) {
        self.pages.clear();
        self.len.store(0, Ordering::Relaxed);
    }
}

fn index_bounds(keys: impl RangeBounds<usize>) -> (usize, usize) {
    let start = match keys.start_bound() {
        Bound::Included(&index) => index,
        Bound::Excluded(&index) => match index.checked_add(1) {
            Some(index) => index,
            None => return (1, 0),
        },
        Bound::Unbounded => 0,
    };
    let end = match keys.end_bound() {
        Bound::Included(&index) => index,
        Bound::Excluded(&index) => match index.checked_sub(1) {
            Some(index) => index,
            None => return (1, 0),
        },
        Bound::Unbounded => usize::MAX,
    };
    (start, end)
}

fn encode_direct(value: u64) -> u64 {
    value
        .checked_add(1)
        .expect("AtomicDirectMap: u64::MAX cannot be stored")
}

fn decode_direct(word: u64) -> Option<u64> {
    word.checked_sub(1)
}

/// A lock-free direct-address map from slot indices to one-word values.
///
/// The key is the slot index itself; callers with `u32` keys pass `key as
/// usize`. No key or generation is stored. Use [`AtomicIdMap`] for
/// generational keys. Values are encoded as `value + 1`, reserving zero for
/// an empty slot: `u64::MAX` cannot be stored; see [`MAX_VALUE`](Self::MAX_VALUE).
///
/// Slot loads use acquire ordering, successful read-modify-write operations
/// use acquire-release, and failed compare-exchanges use acquire. Values may
/// publish other data indexed by them. Slot operations are lock-free; page
/// installation can allocate. The finite, Flat-like spine retains pages until
/// [`reset`](Self::reset) or destruction. Out-of-range lookups and removals
/// return absent; inserting outside the spine range panics. `PAGE` must be
/// nonzero and its atomic slot storage must fit in `isize::MAX` bytes.
///
/// Length uses a wrapping counter updated after each occupancy change. A
/// removal can be counted before the insertion it undoes; [`len`](Self::len)
/// interprets the counter as signed and clamps negatives to zero. It is exact
/// when no write is in flight. Traversals are weakly consistent, visit ascending
/// slots and load each visited slot once; they do not provide a snapshot.
///
/// ```
/// # #[cfg(feature = "concurrent")] {
/// use paged_collections::concurrent::AtomicDirectMap;
/// let counters = AtomicDirectMap::<4>::new();
/// // Increment a counter, initializing an absent counter to zero first.
/// assert_eq!(counters.fetch_update(7, |old| Some(Some(old.unwrap_or(0) + 1))), Ok(None));
/// assert_eq!(counters.fetch_update(7, |old| Some(Some(old.unwrap_or(0) + 1))), Ok(Some(1)));
/// assert_eq!(counters.get(7), Some(2));
/// # }
/// ```
pub struct AtomicDirectMap<const PAGE: usize = DEFAULT_PAGE> {
    words: Words<PAGE>,
}

impl<const PAGE: usize> AtomicDirectMap<PAGE> {
    /// Number of atomic slots per page.
    pub const PAGE_SIZE: usize = PAGE;

    /// Largest storable value; `u64::MAX` is reserved by the encoding.
    pub const MAX_VALUE: u64 = u64::MAX - 1;

    /// Creates an empty map without allocating.
    ///
    /// Panics if `PAGE` is zero or its atomic storage exceeds `isize::MAX` bytes.
    pub fn new() -> Self {
        Self {
            words: Words::new("AtomicDirectMap"),
        }
    }

    /// Number of entries; exact when no write is in flight.
    /// Interprets the wrapping counter as signed and clamps negatives to zero.
    pub fn len(&self) -> usize {
        self.words.len()
    }

    /// Whether `len()` is zero, subject to concurrent writes.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Counts installed pages; weakly consistent during page installation.
    pub fn allocated_pages(&self) -> usize {
        self.words.pages.iter().count()
    }

    /// Loads a value with acquire ordering. Out-of-range keys return `None`.
    pub fn get(&self, key: usize) -> Option<u64> {
        decode_direct(self.words.load(key))
    }

    /// Whether this slot is occupied.
    pub fn contains_key(&self, key: usize) -> bool {
        self.get(key).is_some()
    }

    /// Swaps in `value`, returning the previous value.
    /// Panics if `value == u64::MAX` or the key exceeds the spine range.
    pub fn insert(&self, key: usize, value: u64) -> Option<u64> {
        decode_direct(
            self.words
                .swap(key, encode_direct(value), "AtomicDirectMap"),
        )
    }

    /// Inserts into an empty slot, or returns `Err(current)`.
    /// Panics if `value == u64::MAX` or the key exceeds the spine range.
    pub fn try_insert(&self, key: usize, value: u64) -> Result<(), u64> {
        self.words
            .try_insert(key, encode_direct(value), "AtomicDirectMap")
            .map_err(|word| decode_direct(word).unwrap())
    }

    /// Swaps the slot to empty, returning its previous value.
    /// Out-of-range keys return `None` without allocating.
    pub fn remove(&self, key: usize) -> Option<u64> {
        decode_direct(self.words.swap(key, 0, "AtomicDirectMap"))
    }

    /// Stores `new` only if the current value equals `current`, without
    /// spurious failure. `None` means empty. Returns `Ok(previous)` on success
    /// or `Err(actual)` on mismatch. A `None` write never allocates a page.
    /// Panics if `new == Some(u64::MAX)` or an inserting key exceeds the spine
    /// range. A nonempty write installs its page before checking `current`.
    pub fn compare_exchange(
        &self,
        key: usize,
        current: Option<u64>,
        new: Option<u64>,
    ) -> Result<Option<u64>, Option<u64>> {
        let word = new.map_or(0, encode_direct);
        self.words
            .update(key, "AtomicDirectMap", new.is_some(), |old| {
                (decode_direct(old) == current).then_some(word)
            })
            .map(decode_direct)
            .map_err(decode_direct)
    }

    /// Repeatedly calls `f` with the current value until a change succeeds or
    /// `f` returns `None`. Returns `Ok(previous)` after storing `Some(new)`;
    /// `Some(None)` removes the entry. Returning `None` stops with `Err(current)`.
    /// The callback may run several times, but only one change is committed.
    /// Removing from an absent page does not allocate it.
    /// Panics if `f` requests storing `u64::MAX`, or an insertion exceeds the
    /// spine range. A panicking callback leaves the slot unchanged by this call.
    pub fn fetch_update(
        &self,
        key: usize,
        mut f: impl FnMut(Option<u64>) -> Option<Option<u64>>,
    ) -> Result<Option<u64>, Option<u64>> {
        self.words
            .update(key, "AtomicDirectMap", false, |old| {
                f(decode_direct(old)).map(|new| new.map_or(0, encode_direct))
            })
            .map(decode_direct)
            .map_err(decode_direct)
    }

    /// Iterates in ascending slot order, loading each slot once with acquire
    /// ordering. Weakly consistent under mutation; not a snapshot.
    pub fn iter(&self) -> impl Iterator<Item = (usize, u64)> {
        self.range(..)
    }

    /// Iterates over a key range in ascending order, loading each visited slot
    /// once with acquire ordering. Weakly consistent under mutation. Empty or
    /// inverted ranges yield nothing.
    pub fn range(&self, keys: impl RangeBounds<usize>) -> impl Iterator<Item = (usize, u64)> {
        let (start, end) = index_bounds(keys);
        self.words
            .range(start, end)
            .map(|(key, word)| (key, word - 1))
    }

    /// Empties the map, retaining pages for reuse. Requires exclusive access.
    pub fn clear(&mut self) {
        self.words.clear();
    }

    /// Frees all pages and directory chunks, leaving an empty map.
    pub fn reset(&mut self) {
        self.words.reset();
    }
}

fn encode_id(id: Id, value: u32) -> u64 {
    (u64::from(id.generation().get()) << 32) | u64::from(value)
}

fn decode_id(index: u32, word: u64) -> Option<(Id, u32)> {
    let generation = NonZeroU32::new((word >> 32) as u32)?;
    Some((Id::new(index, generation), word as u32))
}

fn id_value(id: Id, word: u64) -> Option<u32> {
    ((word >> 32) as u32 == id.generation().get()).then_some(word as u32)
}

fn id_update(id: Id, old: u64, new: Option<u32>) -> u64 {
    match new {
        Some(value) => encode_id(id, value),
        None if id_value(id, old).is_some() => 0,
        None => old,
    }
}

/// A lock-free direct-address map from generational [`Id`]s to `u32` values.
///
/// Each slot at `id.index()` stores `generation << 32 | value`. Generation is
/// nonzero, so zero denotes an empty slot and all `u32` values are storable.
/// Lookups and [`remove`](Self::remove) match the generation. Unconditional
/// insertion evicts any generation, while [`try_insert`](Self::try_insert)
/// refuses any occupant.
///
/// Conditional updates view another generation as absent. Writing `Some(v)`
/// installs the requested id and evicts any other generation; writing `None`
/// removes only the requested id and succeeds without changing another
/// generation. Filling an empty slot adds one to length, eviction leaves it
/// unchanged, and removing this id subtracts one.
///
/// Slot loads use acquire ordering, successful read-modify-write operations
/// use acquire-release, and failed compare-exchanges use acquire. Values may
/// publish other data indexed by them. Slot operations are lock-free; page
/// installation can allocate. The finite, Flat-like spine retains pages until
/// [`reset`](Self::reset) or destruction. Out-of-range lookups and removals
/// return absent; inserting outside the spine range panics. `PAGE` must be
/// nonzero and its atomic slot storage must fit in `isize::MAX` bytes.
///
/// Length uses a wrapping counter updated after each occupancy change. A
/// removal can be counted before the insertion it undoes; [`len`](Self::len)
/// interprets the counter as signed and clamps negatives to zero. It is exact
/// when no write is in flight. Traversals are weakly consistent, visit ascending
/// indices and load each visited slot once; they do not provide a snapshot.
///
/// ```
/// # #[cfg(feature = "concurrent")] {
/// use core::num::NonZeroU32;
/// use paged_collections::{concurrent::AtomicIdMap, id_alloc::Id};
/// let map = AtomicIdMap::<4>::new();
/// let old = Id::new(3, NonZeroU32::new(1).unwrap());
/// let new = Id::new(3, NonZeroU32::new(2).unwrap());
/// map.insert(old, 10);
/// assert_eq!(map.replace(new, u32::MAX), Some((old, 10)));
/// assert_eq!(map.remove(old), None);
/// assert_eq!(map.get(new), Some(u32::MAX));
/// # }
/// ```
pub struct AtomicIdMap<const PAGE: usize = DEFAULT_PAGE> {
    words: Words<PAGE>,
}

impl<const PAGE: usize> AtomicIdMap<PAGE> {
    /// Number of atomic slots per page.
    pub const PAGE_SIZE: usize = PAGE;

    /// Creates an empty map without allocating.
    ///
    /// Panics if `PAGE` is zero or its atomic storage exceeds `isize::MAX` bytes.
    pub fn new() -> Self {
        Self {
            words: Words::new("AtomicIdMap"),
        }
    }

    /// Number of entries; exact when no write is in flight.
    /// Interprets the wrapping counter as signed and clamps negatives to zero.
    pub fn len(&self) -> usize {
        self.words.len()
    }

    /// Whether `len()` is zero, subject to concurrent writes.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Counts installed pages; weakly consistent during page installation.
    pub fn allocated_pages(&self) -> usize {
        self.words.pages.iter().count()
    }

    /// Loads the value only if this generation holds the slot.
    /// Uses acquire ordering; out-of-range ids return `None`.
    pub fn get(&self, id: Id) -> Option<u32> {
        id_value(id, self.words.load(id.index() as usize))
    }

    /// Whether this id's generation holds the slot.
    pub fn contains_key(&self, id: Id) -> bool {
        self.get(id).is_some()
    }

    /// Loads whichever generation occupies `index`, with acquire ordering.
    /// Out-of-range indices return `None`.
    pub fn occupant(&self, index: u32) -> Option<(Id, u32)> {
        decode_id(index, self.words.load(index as usize))
    }

    /// Swaps in this id and value, returning the previous value from any
    /// generation. Panics if the index exceeds the spine range.
    pub fn insert(&self, id: Id, value: u32) -> Option<u32> {
        self.replace(id, value).map(|(_, value)| value)
    }

    /// Swaps in this id and value, returning the previous occupant from any
    /// generation. Panics if the index exceeds the spine range.
    pub fn replace(&self, id: Id, value: u32) -> Option<(Id, u32)> {
        let old = self
            .words
            .swap(id.index() as usize, encode_id(id, value), "AtomicIdMap");
        decode_id(id.index(), old)
    }

    /// Inserts only into an empty slot, returning any occupant on failure.
    /// Panics if the index exceeds the spine range.
    pub fn try_insert(&self, id: Id, value: u32) -> Result<(), (Id, u32)> {
        self.words
            .try_insert(id.index() as usize, encode_id(id, value), "AtomicIdMap")
            .map_err(|word| decode_id(id.index(), word).unwrap())
    }

    /// Removes only this generation, returning its value. A stale id never
    /// removes another generation. Out-of-range ids return `None`.
    pub fn remove(&self, id: Id) -> Option<u32> {
        self.words
            .update(id.index() as usize, "AtomicIdMap", false, |old| {
                id_value(id, old).map(|_| 0)
            })
            .ok()
            .and_then(|old| id_value(id, old))
    }

    /// Swaps the slot to empty and returns any occupant.
    /// Out-of-range indices return `None` without allocating.
    pub fn remove_occupant(&self, index: u32) -> Option<(Id, u32)> {
        decode_id(index, self.words.swap(index as usize, 0, "AtomicIdMap"))
    }

    /// Stores `new` if this id's value equals `current`, without spurious
    /// failure. Returns `Ok(previous)` or `Err(actual)` for this id only.
    /// `None` matches both an empty slot and another generation. `Some(v)`
    /// evicts any generation; `None` removes only this id and otherwise
    /// succeeds without changing the occupant. Eviction preserves length.
    /// A `None` write never allocates a page. A nonempty write installs its
    /// page before checking `current`; an out-of-range insertion panics.
    pub fn compare_exchange(
        &self,
        id: Id,
        current: Option<u32>,
        new: Option<u32>,
    ) -> Result<Option<u32>, Option<u32>> {
        self.words
            .update(id.index() as usize, "AtomicIdMap", new.is_some(), |old| {
                (id_value(id, old) == current).then(|| id_update(id, old, new))
            })
            .map(|old| id_value(id, old))
            .map_err(|actual| id_value(id, actual))
    }

    /// Repeatedly calls `f` with this id's value until a change succeeds or
    /// `f` returns `None`. Another generation is presented as absent.
    /// `Some(Some(v))` stores this id and evicts any generation; `Some(None)`
    /// removes only this id, leaving another generation unchanged. Eviction
    /// preserves length. Returns `Ok(previous)` on success or `Err(current)`
    /// if `f` stops. The callback may run several times, even if its argument
    /// remains `None` while other generations change; only one change commits.
    /// Removing from an absent page does not allocate it. An out-of-range
    /// insertion panics. A panicking callback leaves the slot unchanged by
    /// this call.
    pub fn fetch_update(
        &self,
        id: Id,
        mut f: impl FnMut(Option<u32>) -> Option<Option<u32>>,
    ) -> Result<Option<u32>, Option<u32>> {
        self.words
            .update(id.index() as usize, "AtomicIdMap", false, |old| {
                f(id_value(id, old)).map(|new| id_update(id, old, new))
            })
            .map(|old| id_value(id, old))
            .map_err(|actual| id_value(id, actual))
    }

    /// Iterates in ascending index order, loading each slot once with acquire
    /// ordering. Weakly consistent under mutation; not a snapshot.
    pub fn iter(&self) -> impl Iterator<Item = (Id, u32)> {
        self.range(..)
    }

    /// Iterates over an index range in ascending order, independent of
    /// generation. Loads each visited slot once with acquire ordering;
    /// weakly consistent under mutation. Empty or inverted ranges yield nothing.
    pub fn range(&self, indices: impl RangeBounds<u32>) -> impl Iterator<Item = (Id, u32)> {
        let bounds = (
            indices.start_bound().map(|&index| index as usize),
            indices.end_bound().map(|&index| index as usize),
        );
        let (start, end) = index_bounds(bounds);
        self.words
            .range(start, end)
            .filter_map(|(index, word)| decode_id(index as u32, word))
    }

    /// Empties the map, retaining pages for reuse. Requires exclusive access.
    pub fn clear(&mut self) {
        self.words.clear();
    }

    /// Frees all pages and directory chunks, leaving an empty map.
    pub fn reset(&mut self) {
        self.words.reset();
    }
}

impl<const PAGE: usize> Default for AtomicDirectMap<PAGE> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const PAGE: usize> fmt::Debug for AtomicDirectMap<PAGE> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<const PAGE: usize> FromIterator<(usize, u64)> for AtomicDirectMap<PAGE> {
    /// Collects entries, panicking if a value is `u64::MAX` or a key exceeds
    /// the spine range. Later entries replace earlier values at the same key.
    fn from_iter<T: IntoIterator<Item = (usize, u64)>>(iter: T) -> Self {
        let mut map = Self::new();
        map.extend(iter);
        map
    }
}

impl<const PAGE: usize> Extend<(usize, u64)> for AtomicDirectMap<PAGE> {
    /// Inserts entries, panicking if a value is `u64::MAX` or a key exceeds
    /// the spine range. Entries inserted before a panic remain in the map.
    fn extend<T: IntoIterator<Item = (usize, u64)>>(&mut self, iter: T) {
        for (key, value) in iter {
            self.insert(key, value);
        }
    }
}

impl<const PAGE: usize> Default for AtomicIdMap<PAGE> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const PAGE: usize> fmt::Debug for AtomicIdMap<PAGE> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<const PAGE: usize> FromIterator<(Id, u32)> for AtomicIdMap<PAGE> {
    fn from_iter<T: IntoIterator<Item = (Id, u32)>>(iter: T) -> Self {
        let mut map = Self::new();
        map.extend(iter);
        map
    }
}

impl<const PAGE: usize> Extend<(Id, u32)> for AtomicIdMap<PAGE> {
    fn extend<T: IntoIterator<Item = (Id, u32)>>(&mut self, iter: T) {
        for (id, value) in iter {
            self.insert(id, value);
        }
    }
}
