//! Insert-only shared sparse collections; editing requires exclusive access.

use alloc::boxed::Box;
use alloc::vec::{IntoIter, Vec};
use core::fmt;
use core::iter::FusedIterator;

use super::append::{AppendLog, DenseIter};
use super::atomic::{AtomicU32, Ordering};
use super::directory::{SharedDirectory, SharedPages};
use super::index::retry;
use super::vec::VecIterMut;
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
    fn valid_page_capacity() -> bool {
        PAGE > 0 && PAGE <= isize::MAX as usize / core::mem::size_of::<AtomicU32>()
    }

    fn new() -> Self {
        assert!(
            Self::valid_page_capacity(),
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

    /// One attempt to reserve the slot, never waiting.
    fn try_claim(slot: &AtomicU32) -> Claim<'_> {
        // Acquire sees the winning entry or a rollback from RESERVED to 0.
        match slot.load(Ordering::Acquire) {
            0 => {
                // AcqRel grants one caller ownership and observes any
                // previous rollback; failure acquires the competing state.
                match slot.compare_exchange(0, RESERVED, Ordering::AcqRel, Ordering::Acquire) {
                    Ok(_) => Claim::Owned(Reservation { slot, armed: true }),
                    // Failure means the slot is no longer 0.
                    Err(RESERVED) => Claim::Busy,
                    Err(word) => Claim::Present(word as usize - 1),
                }
            }
            RESERVED => Claim::Busy,
            word => Claim::Present(word as usize - 1),
        }
    }

    fn claim(&self, index: usize) -> Result<Reservation<'_>, usize> {
        let slot = self.ensure_slot(index);
        let mut spins = 0;
        loop {
            match Self::try_claim(slot) {
                Claim::Owned(owner) => return Ok(owner),
                Claim::Present(pos) => return Err(pos),
                // Same-slot insertions wait for publication or rollback.
                // Other slots proceed independently. A factory must not
                // recursively insert into its own reserved slot.
                Claim::Busy => retry(&mut spins),
            }
        }
    }
}

enum Claim<'a> {
    Owned(Reservation<'a>),
    Present(usize),
    Busy,
}

/// Keys per batch reservation in [`ConcurrentSparseSet::insert_many`] and
/// [`ConcurrentSparseMap::try_insert_many`]: large enough to amortize the
/// shared append counters, small enough to bound held reservations.
const BATCH: usize = 256;

/// Loom coverage: counts batch items deferred because another writer held
/// their slot. Plain std atomic, outside the model.
#[cfg(all(test, loom))]
pub(super) static BATCH_BUSY: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Inserts `items` in batches. Each batch claims vacant slots without
/// waiting, appends all claimed entries with one reservation, then publishes
/// their positions. Slots another writer holds (including an earlier
/// duplicate in this batch) are deferred and inserted one at a time with the
/// waiting path once this batch holds no reservations, so two batches never
/// wait on each other. Returns the number inserted.
fn insert_batched<'a, K: Key, T, const PAGE: usize, D: SharedDirectory>(
    index: &'a Index<PAGE, D>,
    dense: &AppendLog<T, PAGE>,
    items: impl IntoIterator<Item = T>,
    key_of: impl Fn(&T) -> K,
    insert_one: impl Fn(T) -> bool,
    full: &str,
) -> usize {
    let mut items = items.into_iter();
    let mut inserted = 0;
    // Keep user values in buffers declared before the reservation guards,
    // including while the dense log prepares storage. A value's destructor
    // may insert into a slot that this batch reserved.
    let mut pending = Vec::with_capacity(BATCH);
    let mut deferred = Vec::new();
    let mut entries: Vec<T> = Vec::with_capacity(BATCH);
    let mut owners: Vec<Reservation<'a>> = Vec::with_capacity(BATCH);
    loop {
        // Run the iterator, validate keys and allocate sparse slots before
        // claiming anything. A panic here leaves earlier batches intact.
        for item in items.by_ref().take(BATCH) {
            let key_index = key_of(&item).index();
            let slot = index.ensure_slot(key_index);
            pending.push((key_index, slot, Some(item)));
        }
        let taken = pending.len();
        // Reserve space before claiming so that moving values cannot panic
        // on a buffer allocation. Rejected values stay in `pending`.
        deferred.reserve(taken);
        for &mut (key_index, slot, ref mut item) in &mut pending {
            // A later copy of a deferred key is deferred too: if the other
            // writer rolls back, the earlier copy must still win.
            if deferred.iter().any(|(key, _)| *key == key_index) {
                deferred.push((key_index, item.take().expect("prepared item")));
            } else {
                match Index::<PAGE, D>::try_claim(slot) {
                    Claim::Owned(owner) => {
                        owners.push(owner);
                        entries.push(item.take().expect("prepared item"));
                    }
                    Claim::Present(_) => {}
                    Claim::Busy => {
                        #[cfg(all(test, loom))]
                        BATCH_BUSY.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        deferred.push((key_index, item.take().expect("prepared item")));
                    }
                }
            }
        }
        if !owners.is_empty() {
            let Some(start) = dense.push_many_from(&mut entries, DENSE_LIMIT) else {
                panic!("{full}");
            };
            inserted += owners.len();
            for (i, owner) in owners.drain(..).enumerate() {
                owner.publish(start + i);
            }
        }
        // No reservation is held from here on.
        pending.clear();
        for (_, item) in deferred.drain(..) {
            inserted += usize::from(insert_one(item));
        }
        // Only an exhausted input ends the loop: a batch of present keys
        // claims nothing but more input may follow.
        if taken < BATCH {
            return inserted;
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

    /// Adds every key whose slot is vacant, returning how many were added.
    /// Earlier duplicates win. Batches share one dense reservation, which
    /// scales across threads far better than repeated [`insert`](Self::insert).
    /// Positions need not follow input order.
    ///
    /// Each batch is read and its keys validated before any slots are reserved.
    /// If the iterator panics, earlier batches stay inserted.
    ///
    /// # Panics
    ///
    /// As [`insert`](Self::insert).
    pub fn insert_many(&self, keys: impl IntoIterator<Item = K>) -> usize {
        insert_batched(
            &self.index,
            &self.dense,
            keys,
            |&key| key,
            |key| self.insert(key),
            "ConcurrentSparseSet cannot hold more than u32::MAX - 1 keys",
        )
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
    pub fn iter(&self) -> DenseIter<'_, K, PAGE> {
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
        while self.pop_key().is_some() {}
    }

    /// Removes all keys and returns them in dense order. Keeps allocated pages.
    /// Takes O(len) time and temporary storage.
    pub fn drain(&mut self) -> IntoIter<K> {
        let mut keys = Vec::with_capacity(self.len());
        while let Some(key) = self.pop_key() {
            keys.push(key);
        }
        keys.reverse();
        keys.into_iter()
    }

    fn pop_key(&mut self) -> Option<K> {
        let last = self.dense.len().checked_sub(1)?;
        let key = *self.dense.get(last).expect("last key exists");
        let slot = self
            .index
            .slot(key.index())
            .expect("present key has a slot");
        let key = self.dense.pop();
        // Relaxed: exclusive access; no lookup can observe the change.
        slot.store(0, Ordering::Relaxed);
        key
    }

    /// Copies the committed prefix into a plain set in O(prefix length).
    /// Like [`len`](Self::len) and [`iter`](Self::iter), this can omit
    /// completed inserts beyond an earlier unpublished reservation. The keys
    /// form a valid plain set.
    pub fn snapshot(&self) -> PagedSparseSet<K, PAGE, D> {
        self.iter().copied().collect()
    }

    /// Moves keys into the plain set with the same directory in dense order, in O(len).
    pub fn into_inner(self) -> PagedSparseSet<K, PAGE, D> {
        self.dense.into_inner().into_iter().collect()
    }
}

impl<'a, K: Key, const PAGE: usize, D: SharedDirectory> IntoIterator
    for &'a ConcurrentSparseSet<K, PAGE, D>
{
    type Item = &'a K;
    type IntoIter = DenseIter<'a, K, PAGE>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Copies the committed prefix via [`ConcurrentSparseSet::snapshot`],
/// which can omit completed inserts beyond an unpublished reservation.
impl<K: Key, const PAGE: usize, D: SharedDirectory> Clone for ConcurrentSparseSet<K, PAGE, D> {
    fn clone(&self) -> Self {
        Self::from(self.snapshot())
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
        let mut result = Self::new();
        result.extend(iter);
        result
    }
}

/// Inserts keys in input order, keeping the first occupant of each slot.
impl<K: Key, const PAGE: usize, D: SharedDirectory> Extend<K> for ConcurrentSparseSet<K, PAGE, D> {
    fn extend<I: IntoIterator<Item = K>>(&mut self, iter: I) {
        for key in iter {
            self.insert(key);
        }
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

    fn append(&self, reservation: Reservation<'_>, key: K, value: V) -> &V {
        let mut entry = Some((key, value));
        // This guard must unwind before the entry if dense preparation panics.
        let owner = reservation;
        let Some(pos) = self.dense.push_from(&mut entry, DENSE_LIMIT) else {
            panic!("ConcurrentSparseMap cannot hold more than u32::MAX - 1 entries");
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

    /// Adds every entry whose slot is vacant, dropping the others, and returns
    /// how many were added. Existing entries and earlier duplicates win.
    /// Batches share one dense reservation, which scales across threads far
    /// better than repeated [`try_insert`](Self::try_insert).
    ///
    /// Each batch is read and its keys validated before any slots are reserved.
    /// If the iterator panics, earlier batches stay inserted.
    ///
    /// # Panics
    ///
    /// As [`try_insert`](Self::try_insert).
    pub fn try_insert_many(&self, entries: impl IntoIterator<Item = (K, V)>) -> usize {
        insert_batched(
            &self.index,
            &self.dense,
            entries,
            |(key, _)| *key,
            |(key, value)| self.try_insert(key, value).is_ok(),
            "ConcurrentSparseMap cannot hold more than u32::MAX - 1 entries",
        )
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
    pub fn iter(&self) -> SparseMapIter<'_, K, V, PAGE> {
        SparseMapIter {
            inner: self.dense.iter(),
        }
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
    pub fn iter_mut(&mut self) -> SparseMapIterMut<'_, K, V> {
        SparseMapIterMut {
            inner: VecIterMut::new(&mut self.dense),
        }
    }

    /// Drops every entry and clears readiness, retaining pages. Takes O(len).
    /// If a destructor panics, that entry is gone; remaining entries keep
    /// their sparse positions and may still be used.
    pub fn clear(&mut self) {
        while let Some(entry) = self.pop_entry() {
            drop(entry);
        }
    }

    /// Removes all entries and returns them as `(key, value)` pairs in dense order.
    /// Keeps allocated pages. Takes O(len) time and temporary storage.
    pub fn drain(&mut self) -> IntoIter<(K, V)> {
        let mut entries = Vec::with_capacity(self.len());
        while let Some(entry) = self.pop_entry() {
            entries.push(entry);
        }
        entries.reverse();
        entries.into_iter()
    }

    fn pop_entry(&mut self) -> Option<(K, V)> {
        let last = self.dense.len().checked_sub(1)?;
        let key = self.dense.get(last).expect("last entry exists").0;
        let slot = self
            .index
            .slot(key.index())
            .expect("present key has a slot");
        let entry = self.dense.pop();
        // Relaxed: exclusive access; clear the slot before user drop code.
        slot.store(0, Ordering::Relaxed);
        entry
    }

    /// Clones the committed prefix into a plain map in O(prefix length).
    /// Like [`len`](Self::len) and [`iter`](Self::iter), this can omit
    /// completed inserts beyond an earlier unpublished reservation. The
    /// entries form a valid plain map.
    pub fn snapshot(&self) -> PagedSparseMap<K, V, PAGE, D>
    where
        V: Clone,
    {
        self.iter()
            .map(|(key, value)| (*key, value.clone()))
            .collect()
    }

    /// Moves entries into a plain map with the same directory in dense order, in O(len).
    /// Values move without cloning; their addresses can change.
    pub fn into_inner(self) -> PagedSparseMap<K, V, PAGE, D> {
        self.dense.into_inner().into_iter().collect()
    }
}

impl<'a, K: Key, V, const PAGE: usize, D: SharedDirectory> IntoIterator
    for &'a ConcurrentSparseMap<K, V, PAGE, D>
{
    type Item = (&'a K, &'a V);
    type IntoIter = SparseMapIter<'a, K, V, PAGE>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K: Key, V, const PAGE: usize, D: SharedDirectory> IntoIterator
    for &'a mut ConcurrentSparseMap<K, V, PAGE, D>
{
    type Item = (&'a K, &'a mut V);
    type IntoIter = SparseMapIterMut<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

/// Iterates over the committed entries of a [`ConcurrentSparseMap`] in dense order.
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct SparseMapIter<'a, K, V, const PAGE: usize = DEFAULT_PAGE> {
    inner: DenseIter<'a, (K, V), PAGE>,
}

impl<'a, K, V, const PAGE: usize> Iterator for SparseMapIter<'a, K, V, PAGE> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        self.nth(0)
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.inner.nth(n).map(|(key, value)| (key, value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K, V, const PAGE: usize> DoubleEndedIterator for SparseMapIter<'_, K, V, PAGE> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.nth_back(0)
    }

    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.inner.nth_back(n).map(|(key, value)| (key, value))
    }
}

impl<K, V, const PAGE: usize> ExactSizeIterator for SparseMapIter<'_, K, V, PAGE> {}
impl<K, V, const PAGE: usize> FusedIterator for SparseMapIter<'_, K, V, PAGE> {}

impl<K, V, const PAGE: usize> fmt::Debug for SparseMapIter<'_, K, V, PAGE> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SparseMapIter")
            .field("inner", &self.inner)
            .finish()
    }
}

/// Iterates over the entries of a [`ConcurrentSparseMap`] in dense order,
/// allowing each value to be mutated.
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct SparseMapIterMut<'a, K, V> {
    inner: VecIterMut<'a, (K, V)>,
}

impl<'a, K, V> Iterator for SparseMapIterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(key, value)| (&*key, value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K, V> fmt::Debug for SparseMapIterMut<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SparseMapIterMut").finish_non_exhaustive()
    }
}

/// Clones the committed prefix via [`ConcurrentSparseMap::snapshot`],
/// which can omit completed inserts beyond an unpublished reservation.
impl<K: Key, V: Clone, const PAGE: usize, D: SharedDirectory> Clone
    for ConcurrentSparseMap<K, V, PAGE, D>
{
    fn clone(&self) -> Self {
        Self::from(self.snapshot())
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
        inner.into_iter().collect()
    }
}

/// Collects entries in input order, keeping the first occupant of each slot.
impl<K: Key, V, const PAGE: usize, D: SharedDirectory> FromIterator<(K, V)>
    for ConcurrentSparseMap<K, V, PAGE, D>
{
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut result = Self::new();
        result.extend(iter);
        result
    }
}

/// Inserts entries in input order, keeping the first occupant of each slot.
impl<K: Key, V, const PAGE: usize, D: SharedDirectory> Extend<(K, V)>
    for ConcurrentSparseMap<K, V, PAGE, D>
{
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        for (key, value) in iter {
            let _ = self.try_insert(key, value);
        }
    }
}

#[cfg(feature = "serde")]
mod serde_impl {
    use super::*;
    use crate::concurrent::serde::deserialize_map;
    use core::marker::PhantomData;
    use serde::de::{Deserialize, Deserializer, Error, SeqAccess, Visitor};
    use serde::ser::{Serialize, Serializer};

    fn check_index<const PAGE: usize, D: SharedDirectory>(
        index: usize,
    ) -> Result<(), &'static str> {
        if D::Pages::<IndexPage>::page_number::<PAGE>(index).is_none() {
            return Err("ConcurrentSparse: key index exceeds sparse spine range");
        }
        Ok(())
    }

    fn can_append_at<T, const PAGE: usize>(len: usize) -> bool {
        len < DENSE_LIMIT && AppendLog::<T, PAGE>::can_append_at(len)
    }

    /// Serializes the committed keys in dense order as a valid plain set
    /// sequence in O(prefix length). The prefix can omit completed inserts
    /// beyond an earlier unpublished reservation, as in [`ConcurrentSparseSet::iter`].
    impl<K: Key + Serialize, const PAGE: usize, D: SharedDirectory> Serialize
        for ConcurrentSparseSet<K, PAGE, D>
    {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            serializer.collect_seq(self.iter())
        }
    }

    /// Reads the plain [`PagedSparseSet`] format, rejecting keys and lengths
    /// outside the concurrent set's capacity.
    impl<'de, K: Key + Deserialize<'de>, const PAGE: usize, D: SharedDirectory> Deserialize<'de>
        for ConcurrentSparseSet<K, PAGE, D>
    {
        fn deserialize<De: Deserializer<'de>>(deserializer: De) -> Result<Self, De::Error> {
            struct SetVisitor<K, const PAGE: usize, D>(PhantomData<(K, D)>);

            impl<'de, K: Key + Deserialize<'de>, const PAGE: usize, D: SharedDirectory> Visitor<'de>
                for SetVisitor<K, PAGE, D>
            {
                type Value = ConcurrentSparseSet<K, PAGE, D>;

                fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    f.write_str("a sequence of keys")
                }

                fn visit_seq<A: SeqAccess<'de>>(
                    self,
                    mut access: A,
                ) -> Result<Self::Value, A::Error> {
                    let set = ConcurrentSparseSet::new();
                    while let Some(key) = access.next_element::<K>()? {
                        let index = key.index();
                        check_index::<PAGE, D>(index).map_err(A::Error::custom)?;
                        if !can_append_at::<K, PAGE>(set.len())
                            && set.index.position(index).is_none()
                        {
                            return Err(A::Error::custom(
                                "ConcurrentSparseSet: dense capacity exceeded",
                            ));
                        }
                        set.insert(key);
                    }
                    Ok(set)
                }
            }

            if !Index::<PAGE, D>::valid_page_capacity()
                || !AppendLog::<K, PAGE>::valid_page_capacity()
            {
                return Err(De::Error::custom(
                    "ConcurrentSparseSet: invalid page capacity",
                ));
            }
            deserializer.deserialize_seq(SetVisitor::<K, PAGE, D>(PhantomData))
        }
    }

    /// Serializes the committed entries in dense order as a valid plain map
    /// in O(prefix length). The prefix can omit completed inserts beyond an
    /// earlier unpublished reservation, as in [`ConcurrentSparseMap::iter`].
    impl<K: Key + Serialize, V: Serialize, const PAGE: usize, D: SharedDirectory> Serialize
        for ConcurrentSparseMap<K, V, PAGE, D>
    {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            serializer.collect_map(self.iter())
        }
    }

    /// Reads the plain [`PagedSparseMap`] format, rejecting keys and lengths
    /// outside the concurrent map's capacity. Later entries replace earlier
    /// entries at the same index in place, as in the plain map.
    impl<'de, K: Key + Deserialize<'de>, V: Deserialize<'de>, const PAGE: usize, D: SharedDirectory>
        Deserialize<'de> for ConcurrentSparseMap<K, V, PAGE, D>
    {
        fn deserialize<De: Deserializer<'de>>(deserializer: De) -> Result<Self, De::Error> {
            if !Index::<PAGE, D>::valid_page_capacity()
                || !AppendLog::<(K, V), PAGE>::valid_page_capacity()
            {
                return Err(De::Error::custom(
                    "ConcurrentSparseMap: invalid page capacity",
                ));
            }
            deserialize_map(deserializer, Self::new(), |map, key: K, value: V| {
                let index = key.index();
                check_index::<PAGE, D>(index)?;
                if let Some(pos) = map.index.position(index) {
                    let (stored, old) = map.dense.get_mut(pos).expect("published entry is ready");
                    if *stored != key {
                        *stored = key;
                    }
                    *old = value;
                } else {
                    if !can_append_at::<(K, V), PAGE>(map.len()) {
                        return Err("ConcurrentSparseMap: dense capacity exceeded");
                    }
                    let _ = map.try_insert(key, value);
                }
                Ok(())
            })
        }
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
