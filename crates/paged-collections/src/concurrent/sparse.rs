//! Insert-only shared sparse collections; editing requires exclusive access.

use alloc::boxed::Box;
use core::fmt;
use core::iter::FusedIterator;

use super::append::AppendLog;
use super::atomic::{AtomicU32, Ordering};
use super::directory::{SharedDirectory, SharedPages};
use super::index::retry;
use crate::key::Key;
use crate::page::{DEFAULT_PAGE, Flat};
use crate::sparse_map::PagedSparseMap;
use crate::sparse_set::PagedSparseSet;

const RESERVED: u32 = u32::MAX;
// 0 is empty and u32::MAX is RESERVED: encodings 1..u32::MAX are positions.
const DENSE_LIMIT: usize = (u32::MAX - 1) as usize;

struct IndexPage {
    slots: Box<[AtomicU32]>,
}

struct Index<const PAGE: usize, D: SharedDirectory = Flat> {
    pages: D::Pages<IndexPage>,
}

/// Only a successful 0 -> RESERVED CAS creates this guard. It restores 0 on
/// any unwind, including factory/allocation panics, or publishes one position.
struct Reservation<'a> {
    slot: &'a AtomicU32,
    armed: bool,
}

impl Reservation<'_> {
    fn publish(mut self, pos: usize) {
        // Release publishes the dense write AND its ready bit. Point lookups
        // acquire this slot and need no check against the committed prefix.
        self.slot.store(pos as u32 + 1, Ordering::Release);
        self.armed = false;
    }
}

impl Drop for Reservation<'_> {
    fn drop(&mut self) {
        if self.armed {
            // Release ends this claim. Waiters must retry 0, not spin forever
            // waiting for a position after a panicking factory/full log.
            self.slot.store(0, Ordering::Release);
        }
    }
}

impl<const PAGE: usize, D: SharedDirectory> Index<PAGE, D> {
    fn new() -> Self {
        assert!(
            PAGE > 0 && PAGE <= isize::MAX as usize / core::mem::size_of::<AtomicU32>(),
            "ConcurrentSparse: invalid page capacity"
        );
        Self {
            pages: D::Pages::<IndexPage>::new(),
        }
    }

    fn page_number(index: usize) -> usize {
        D::Pages::<IndexPage>::page_number::<PAGE>(index)
            .expect("ConcurrentSparse: key index exceeds sparse spine range")
    }

    fn slot(&self, index: usize) -> Option<&AtomicU32> {
        self.pages
            .get(Self::page_number(index))
            .map(|page| &page.slots[index % PAGE])
    }

    fn ensure_slot(&self, index: usize) -> &AtomicU32 {
        let page = self.pages.get_or_insert_with(Self::page_number(index), || {
            Box::new(IndexPage {
                slots: (0..PAGE).map(|_| AtomicU32::new(0)).collect(),
            })
        });
        &page.slots[index % PAGE]
    }

    fn position(&self, index: usize) -> Option<usize> {
        // Acquire observes the dense entry written before position publication.
        match self.slot(index)?.load(Ordering::Acquire) {
            0 | RESERVED => None,
            word => Some(word as usize - 1),
        }
    }

    fn claim(&self, index: usize) -> Result<Reservation<'_>, usize> {
        let slot = self.ensure_slot(index);
        let mut spins = 0;
        loop {
            // Acquire sees the winning entry or a rollback from RESERVED to 0.
            match slot.load(Ordering::Acquire) {
                0 => {
                    // AcqRel grants one caller ownership and observes any
                    // previous rollback; failure acquires the competing state.
                    if slot
                        .compare_exchange(0, RESERVED, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok()
                    {
                        return Ok(Reservation { slot, armed: true });
                    }
                }
                RESERVED => {}
                word => return Err(word as usize - 1),
            }
            // Same-slot insertions wait for publication or rollback. Other
            // slots proceed independently. A factory must not recursively
            // insert into its own reserved slot.
            retry(&mut spins);
        }
    }
}

/// A paged sparse set with shared insertion and stable shared references.
///
/// At most one key occupies each [`Key::index`]. An occupied slot refuses
/// both duplicates and other generations. Same-slot insertions wait for the
/// winning insertion; lookups do not wait. Removing or clearing requires
/// exclusive access and can move the last dense key into a removed position.
///
/// Iteration and pages snapshot the hole-free committed prefix. Its length
/// can lag behind completed insertions while an earlier append is in flight;
/// lookups see every completed insertion. With no insertion in flight the
/// prefix contains every entry.
///
/// The default [`Flat`] directory has O(1) page lookups and memory following
/// the highest page; indices outside its finite spine range panic.
/// [`Radix`](crate::page::Radix) accepts any `usize` index, uses directory
/// memory proportional to installed pages, and looks up pages in O(depth). At most
/// `u32::MAX - 1` entries fit because one encoding is reserved for writers.
/// Pages are retained until destruction. Uses only `core` and `alloc`.
///
/// ```
/// # #[cfg(feature = "concurrent")] {
/// use paged_collections::{concurrent::ConcurrentSparseSet, page::Radix};
/// let set = ConcurrentSparseSet::<usize, 4, Radix>::new();
/// assert!(set.insert(usize::MAX - 1));
/// assert!(set.contains(usize::MAX - 1));
/// # }
/// ```
pub struct ConcurrentSparseSet<K: Key, const PAGE: usize = DEFAULT_PAGE, D: SharedDirectory = Flat>
{
    index: Index<PAGE, D>,
    // AppendLog stores keys in UnsafeCell, keeping K invariant.
    dense: AppendLog<K, PAGE>,
}

// SAFETY: moving the set transfers owned keys; no reference outlives its borrow.
unsafe impl<K: Key + Send, const PAGE: usize, D: SharedDirectory> Send
    for ConcurrentSparseSet<K, PAGE, D>
{
}
// SAFETY: RESERVED gives each slot one writer, the append log owns distinct
// writes, and shared lookups acquire publication and expose only shared keys.
unsafe impl<K: Key + Send + Sync, const PAGE: usize, D: SharedDirectory> Sync
    for ConcurrentSparseSet<K, PAGE, D>
{
}

impl<K: Key, const PAGE: usize, D: SharedDirectory> ConcurrentSparseSet<K, PAGE, D> {
    /// Number of slots per sparse page and keys per dense page.
    pub const PAGE_SIZE: usize = PAGE;

    /// Creates an empty set without allocating.
    ///
    /// # Panics
    ///
    /// Panics if `PAGE` is zero or page storage exceeds `isize::MAX` bytes.
    pub fn new() -> Self {
        Self {
            index: Index::new(),
            dense: AppendLog::new(),
        }
    }

    /// Adds `key`, returning `false` if any key already holds its slot.
    /// Waits for an insertion already reserving the same slot to finish.
    ///
    /// # Panics
    ///
    /// Panics at `u32::MAX - 1` entries, for indices outside the `Flat` spine
    /// range, or on allocation capacity overflow. Allocation can also fail.
    pub fn insert(&self, key: K) -> bool {
        let Ok(owner) = self.index.claim(key.index()) else {
            return false;
        };
        match self.dense.push(key, DENSE_LIMIT) {
            Ok(pos) => owner.publish(pos),
            Err(_) => {
                drop(owner);
                panic!("ConcurrentSparseSet cannot hold more than u32::MAX - 1 keys");
            }
        }
        true
    }

    /// Dense position of this key, including a completed insertion beyond
    /// the committed prefix. Positions can change under exclusive removal.
    /// With `Flat`, panics if the key index exceeds the sparse spine range.
    pub fn position(&self, key: K) -> Option<usize> {
        let pos = self.index.position(key.index())?;
        // The acquired sparse position follows the write and ready bit. get
        // checks readiness independently, never bounds it by committed.
        self.dense
            .get(pos)
            .filter(|&&stored| stored == key)
            .map(|_| pos)
    }

    /// Whether this key is present. Does not wait for an in-flight insertion.
    /// With `Flat`, panics if the key index exceeds the sparse spine range.
    pub fn contains(&self, key: K) -> bool {
        self.position(key).is_some()
    }

    /// Length of the committed prefix; can lag while an earlier append runs.
    pub fn len(&self) -> usize {
        self.dense.len()
    }

    /// Whether the committed prefix is empty, subject to the lag of `len`.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterates over a snapshot of the committed keys in dense order.
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &K> + ExactSizeIterator + FusedIterator {
        self.dense.iter()
    }

    /// Nonempty page slices of a snapshot of the committed dense prefix.
    /// Every page but the last is full.
    pub fn pages(
        &self,
    ) -> impl DoubleEndedIterator<Item = &[K]> + ExactSizeIterator + FusedIterator {
        self.dense.pages()
    }

    /// Removes a matching key, moving the last key into its position.
    /// Requires exclusive access. With `Flat`, panics for an out-of-range key index.
    pub fn remove(&mut self, key: K) -> bool {
        let Some(pos) = self.position(key) else {
            return false;
        };
        let last = self.dense.len() - 1;
        // Read all user-supplied indices and find slots before any mutation.
        let removed = self
            .index
            .slot(key.index())
            .expect("present key has a slot");
        let moved = if pos == last {
            None
        } else {
            Some(
                self.index
                    .slot(self.dense.get(last).expect("last key exists").index())
                    .expect("moved key has a slot"),
            )
        };
        self.dense.swap_remove(pos);
        // Relaxed: exclusive access excludes readers and slot writers.
        removed.store(0, Ordering::Relaxed);
        if let Some(slot) = moved {
            // Relaxed: moved key is already written, and access is exclusive.
            slot.store(pos as u32 + 1, Ordering::Relaxed);
        }
        true
    }

    /// Removes every key and clears readiness, retaining allocated pages.
    /// Takes O(len) time.
    pub fn clear(&mut self) {
        while let Some(last) = self.dense.len().checked_sub(1) {
            let key = *self.dense.get(last).expect("last key exists");
            let slot = self
                .index
                .slot(key.index())
                .expect("present key has a slot");
            let _ = self.dense.pop();
            // Relaxed: exclusive access; no lookup can observe the change.
            slot.store(0, Ordering::Relaxed);
        }
    }

    /// Moves keys into the plain set with the same directory in dense order, in O(len).
    pub fn into_inner(self) -> PagedSparseSet<K, PAGE, D> {
        self.dense.into_inner().into_iter().collect()
    }
}

impl<K: Key, const PAGE: usize, D: SharedDirectory> Default for ConcurrentSparseSet<K, PAGE, D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Key + fmt::Debug, const PAGE: usize, D: SharedDirectory> fmt::Debug
    for ConcurrentSparseSet<K, PAGE, D>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl<K: Key, const PAGE: usize, D: SharedDirectory> FromIterator<K>
    for ConcurrentSparseSet<K, PAGE, D>
{
    fn from_iter<I: IntoIterator<Item = K>>(iter: I) -> Self {
        let result = Self::new();
        for key in iter {
            result.insert(key);
        }
        result
    }
}

impl<K: Key, const PAGE: usize, D: SharedDirectory> From<PagedSparseSet<K, PAGE, D>>
    for ConcurrentSparseSet<K, PAGE, D>
{
    fn from(inner: PagedSparseSet<K, PAGE, D>) -> Self {
        inner.into_iter().collect()
    }
}

/// A paged sparse map with insert-only shared access and exclusive editing.
///
/// Slots, prefix snapshots and directory selection follow
/// [`ConcurrentSparseSet`]. An occupied slot refuses a new entry, including
/// another generation. Existing values are never replaced while shared, so
/// lookups return stable `&V` references without guards. Shared insertion can
/// interleave dense order; exclusive removal uses swap-remove.
///
/// Same-slot insertions wait for the winning writer, including its factory.
/// Factories must not recursively insert into their own reserved slot, and
/// factories on two threads that each insert the other's pending key
/// deadlock (each waits on the other's reservation). `Sync`
/// requires both `K` and `V` to be `Send + Sync`. Uses `core` and `alloc` only.
///
/// With [`Flat`], memory follows the highest page and lookups cost O(1);
/// its finite index range can panic. [`Radix`](crate::page::Radix) accepts any
/// `usize` index, uses directory memory proportional to installed pages, and
/// looks up pages in O(depth).
///
/// ```
/// # #[cfg(feature = "concurrent")] {
/// use paged_collections::{concurrent::ConcurrentSparseMap, page::Radix};
/// let map = ConcurrentSparseMap::<usize, _, 4, Radix>::new();
/// assert_eq!(map.try_insert(usize::MAX - 1, "wide"), Ok(&"wide"));
/// assert_eq!(map.get(&(usize::MAX - 1)), Some(&"wide"));
/// # }
/// ```
pub struct ConcurrentSparseMap<
    K: Key,
    V,
    const PAGE: usize = DEFAULT_PAGE,
    D: SharedDirectory = Flat,
> {
    index: Index<PAGE, D>,
    // AppendLog stores entries in UnsafeCell, keeping K and V invariant.
    dense: AppendLog<(K, V), PAGE>,
}

// SAFETY: moving the map transfers all owned keys/values and their drops;
// no reference can outlive the borrow from which it was obtained.
unsafe impl<K: Key + Send, V: Send, const PAGE: usize, D: SharedDirectory> Send
    for ConcurrentSparseMap<K, V, PAGE, D>
{
}
// SAFETY: RESERVED uniquely owns each insertion; acquiring sparse publication
// exposes initialized immutable entries, and removal/mutation is exclusive.
unsafe impl<K: Key + Send + Sync, V: Send + Sync, const PAGE: usize, D: SharedDirectory> Sync
    for ConcurrentSparseMap<K, V, PAGE, D>
{
}

impl<K: Key, V, const PAGE: usize, D: SharedDirectory> ConcurrentSparseMap<K, V, PAGE, D> {
    /// Number of slots per sparse page and entries per dense page.
    pub const PAGE_SIZE: usize = PAGE;

    /// Creates an empty map without allocating.
    ///
    /// # Panics
    ///
    /// Panics if `PAGE` is zero or page storage exceeds `isize::MAX` bytes.
    pub fn new() -> Self {
        Self {
            index: Index::new(),
            dense: AppendLog::new(),
        }
    }

    fn position(&self, key: &K) -> Option<usize> {
        let pos = self.index.position(key.index())?;
        self.dense
            .get(pos)
            .filter(|(stored, _)| stored == key)
            .map(|_| pos)
    }

    fn append(&self, owner: Reservation<'_>, key: K, value: V) -> &V {
        let pos = match self.dense.push((key, value), DENSE_LIMIT) {
            Ok(pos) => pos,
            Err(entry) => {
                drop(owner);
                drop(entry);
                panic!("ConcurrentSparseMap cannot hold more than u32::MAX - 1 entries");
            }
        };
        owner.publish(pos);
        &self.dense.get(pos).expect("published entry is ready").1
    }

    /// Adds an entry if its slot is vacant, returning a shared value reference.
    /// Returns `Err((key, value))` for a duplicate or another generation.
    /// Waits for a writer already reserving that slot to publish or roll back.
    ///
    /// # Panics
    ///
    /// Panics at `u32::MAX - 1` entries, for indices outside the `Flat` spine
    /// range, or on allocation capacity overflow. Allocation can also fail.
    pub fn try_insert(&self, key: K, value: V) -> Result<&V, (K, V)> {
        match self.index.claim(key.index()) {
            Ok(owner) => Ok(self.append(owner, key, value)),
            Err(_) => Err((key, value)),
        }
    }

    /// The value for this key, including completed appends beyond `len`.
    /// Does not wait for in-flight insertions. With `Flat`, panics for an out-of-range index.
    pub fn get(&self, key: &K) -> Option<&V> {
        self.position(key)
            .map(|pos| &self.dense.get(pos).expect("published entry is ready").1)
    }

    /// Whether this exact key is present, without waiting for insertions.
    /// With `Flat`, panics for an out-of-range key index.
    pub fn contains_key(&self, key: &K) -> bool {
        self.position(key).is_some()
    }

    /// Gets this key's value, or inserts `make()` if its slot is vacant.
    /// Returns `None` only if another generation occupies the slot.
    ///
    /// Only the RESERVED winner calls `make`, at most once per call. If it
    /// panics, a guard restores the slot to empty and other writers may retry.
    /// The factory must not insert recursively into this same slot: it would
    /// wait on its own reservation. Nor may two threads' factories insert
    /// each other's pending keys: each would wait on the other's
    /// reservation. Factories run before dense reservation.
    ///
    /// # Panics
    ///
    /// As [`try_insert`](Self::try_insert), or if `make` panics.
    pub fn get_or_insert_with(&self, key: K, make: impl FnOnce() -> V) -> Option<&V> {
        match self.index.claim(key.index()) {
            Ok(owner) => {
                let value = make();
                Some(self.append(owner, key, value))
            }
            Err(pos) => {
                let (stored, value) = self.dense.get(pos).expect("published entry is ready");
                (stored == &key).then_some(value)
            }
        }
    }

    /// Length of the committed prefix; can lag while an earlier append runs.
    pub fn len(&self) -> usize {
        self.dense.len()
    }

    /// Whether the committed prefix is empty, subject to the lag of `len`.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterates over a snapshot of committed `(&key, &value)` pairs.
    pub fn iter(
        &self,
    ) -> impl DoubleEndedIterator<Item = (&K, &V)> + ExactSizeIterator + FusedIterator {
        self.dense.iter().map(|(key, value)| (key, value))
    }

    /// Mutable access to a matching key's value under exclusive access.
    /// With `Flat`, panics for an out-of-range key index.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let pos = self.position(key)?;
        self.dense.get_mut(pos).map(|(_, value)| value)
    }

    /// Removes a matching entry and returns its value, moving the last entry
    /// into its position. With `Flat`, panics for an out-of-range key index.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let pos = self.position(key)?;
        let last = self.dense.len() - 1;
        let removed = self
            .index
            .slot(key.index())
            .expect("present key has a slot");
        let moved = if pos == last {
            None
        } else {
            Some(
                self.index
                    .slot(self.dense.get(last).expect("last entry exists").0.index())
                    .expect("moved key has a slot"),
            )
        };
        let (_, value) = self.dense.swap_remove(pos);
        // Relaxed: exclusive access excludes readers and slot writers.
        removed.store(0, Ordering::Relaxed);
        if let Some(slot) = moved {
            // Relaxed: the moved entry is written and access is exclusive.
            slot.store(pos as u32 + 1, Ordering::Relaxed);
        }
        Some(value)
    }

    /// Iterates over `(&key, &mut value)` pairs under exclusive access.
    /// Keys cannot be changed, preserving the sparse index.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&K, &mut V)> {
        self.dense.iter_mut().map(|(key, value)| (&*key, value))
    }

    /// Drops every entry and clears readiness, retaining pages. Takes O(len).
    /// If a destructor panics, that entry is gone; remaining entries keep
    /// their sparse positions and may still be used.
    pub fn clear(&mut self) {
        while let Some(last) = self.dense.len().checked_sub(1) {
            let key = self.dense.get(last).expect("last entry exists").0;
            let slot = self
                .index
                .slot(key.index())
                .expect("present key has a slot");
            let entry = self.dense.pop();
            // Relaxed: exclusive access; clear the slot before user drop code.
            slot.store(0, Ordering::Relaxed);
            drop(entry);
        }
    }

    /// Moves entries into a plain map with the same directory in dense order, in O(len).
    /// Values move without cloning; their addresses can change.
    pub fn into_inner(self) -> PagedSparseMap<K, V, PAGE, D> {
        self.dense.into_inner().into_iter().collect()
    }
}

impl<K: Key, V, const PAGE: usize, D: SharedDirectory> Default
    for ConcurrentSparseMap<K, V, PAGE, D>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Key + fmt::Debug, V: fmt::Debug, const PAGE: usize, D: SharedDirectory> fmt::Debug
    for ConcurrentSparseMap<K, V, PAGE, D>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K: Key, V, const PAGE: usize, D: SharedDirectory> From<PagedSparseMap<K, V, PAGE, D>>
    for ConcurrentSparseMap<K, V, PAGE, D>
{
    fn from(inner: PagedSparseMap<K, V, PAGE, D>) -> Self {
        let result = Self::new();
        for (key, value) in inner {
            let _ = result.try_insert(key, value);
        }
        result
    }
}

#[cfg(all(test, not(loom), feature = "std"))]
mod tests {
    use super::*;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    #[test]
    fn bounded_log_failure_rolls_back_sparse_reservation() {
        let index = Index::<2>::new();
        let dense = AppendLog::<u32, 2>::new();
        let owner = index.claim(1).ok().unwrap();
        assert_eq!(dense.push(1, 0), Err(1));
        drop(owner);
        assert_eq!(index.position(1), None);
        let owner = index.claim(1).ok().unwrap();
        owner.publish(dense.push(1, 1).unwrap());
        assert_eq!(index.position(1), Some(0));
    }

    #[test]
    fn unwinding_before_dense_claim_does_not_leave_reserved() {
        let index = Index::<2>::new();
        let result = catch_unwind(AssertUnwindSafe(|| {
            let _owner = index.claim(1).ok().unwrap();
            panic!("simulated allocation failure before dense claim");
        }));
        assert!(result.is_err());
        assert!(index.claim(1).is_ok());
    }
}
