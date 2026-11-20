//! A sharded indexed max-priority queue.

use alloc::boxed::Box;
use alloc::vec::{IntoIter, Vec};
#[cfg(all(feature = "std", not(loom)))]
use core::cell::Cell;
use core::fmt;

use super::atomic::{AtomicU64, AtomicUsize, CachePadded, Ordering};
use super::lock::{ReadGuard, RwLock};
use crate::key::Key;
use crate::page::{DEFAULT_PAGE, Directory, Flat};
use crate::priority_queue::PagedPriorityQueue;

const SPLITMIX_INCREMENT: u64 = 0x9e37_79b9_7f4a_7c15;

fn splitmix64_finalizer(mut word: u64) -> u64 {
    word = (word ^ (word >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    word = (word ^ (word >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    word ^ (word >> 31)
}

#[cfg(all(feature = "std", not(loom)))]
fn random_word() -> u64 {
    static SEED: AtomicU64 = AtomicU64::new(0);
    std::thread_local! {
        static STATE: Cell<u64> = const { Cell::new(0) };
    }

    STATE.with(|state| {
        let mut word = state.get();
        if word == 0 {
            let counter = SEED.fetch_add(SPLITMIX_INCREMENT, Ordering::Relaxed);
            let address = core::ptr::from_ref(state).addr() as u64;
            word = splitmix64_finalizer(counter.wrapping_add(address)).max(1);
        }
        // xorshift64 with a nonzero state.
        word ^= word << 13;
        word ^= word >> 7;
        word ^= word << 17;
        state.set(word);
        word
    })
}

fn default_shards() -> usize {
    #[cfg(feature = "std")]
    {
        std::thread::available_parallelism()
            .map_or(1, core::num::NonZeroUsize::get)
            .saturating_mul(4)
            .min(256)
    }
    #[cfg(not(feature = "std"))]
    {
        8
    }
}

struct Shard<K, P, const PAGE: usize, D: Directory> {
    queue: RwLock<PagedPriorityQueue<K, P, PAGE, D>>,
    len: AtomicUsize,
}

struct UpdateLen<'a, K: Key, P, const PAGE: usize, D: Directory> {
    queue: &'a mut PagedPriorityQueue<K, P, PAGE, D>,
    len: &'a AtomicUsize,
}

impl<K: Key, P, const PAGE: usize, D: Directory> Drop for UpdateLen<'_, K, P, PAGE, D> {
    fn drop(&mut self) {
        self.len.store(self.queue.len(), Ordering::Relaxed);
    }
}

impl<K: Key, P, const PAGE: usize, D: Directory> Shard<K, P, PAGE, D> {
    fn update<R>(&self, f: impl FnOnce(&mut PagedPriorityQueue<K, P, PAGE, D>) -> R) -> R {
        self.update_locked(&mut self.queue.write(), f)
    }

    // Publish the length before releasing the caller's write lock, even if
    // user code unwinds after changing membership.
    fn update_locked<R>(
        &self,
        queue: &mut PagedPriorityQueue<K, P, PAGE, D>,
        f: impl FnOnce(&mut PagedPriorityQueue<K, P, PAGE, D>) -> R,
    ) -> R {
        let update = UpdateLen {
            queue,
            len: &self.len,
        };
        f(update.queue)
    }
}

/// A concurrent indexed max-priority queue with relaxed and exact removal.
///
/// Each shard wraps a [`PagedPriorityQueue`]. The key's index page determines
/// its shard, keeping all generations of a slot together and allocating that
/// index page in just one shard. `PAGE` must be nonzero. Workloads with fewer
/// key pages than shards can distribute entries unevenly and reduce the
/// effectiveness of relaxed popping. `D` selects the plain queue's directory.
/// Each shard can hold at most `u32::MAX` entries.
///
/// [`pop`](Self::pop) returns the greatest of up to two sampled shard tops,
/// with a fallback scan for nonempty shards. It has no rank guarantee with
/// multiple shards. A key present throughout the call, with its insertion
/// happening-before the call, prevents `None`. [`pop_exact`](Self::pop_exact)
/// and [`peek_exact`](Self::peek_exact) select the global maximum. Ties have
/// unspecified order.
///
/// # Locking and panics
///
/// Per-key operations lock one shard. Operations holding several shard locks
/// at once acquire them in ascending shard order. Callbacks and user trait
/// implementations such as `Ord`, `Clone`, `Debug` and `Serialize` run under
/// shard locks and must avoid calling back into this queue, which can deadlock
/// even on reads. The spin lock used without `std` permits writer starvation.
///
/// Locks ignore poisoning. If a priority comparison or a mutation callback
/// panics, mutations already made remain in effect and all remaining entries
/// stay findable with their priorities. Heap order may be broken in the
/// modified shard, so subsequent maximum results may be unspecified.
/// See [`PagedPriorityQueue`] for its panic guarantees.
///
/// ```
/// use paged_collections::concurrent::ConcurrentPriorityQueue;
///
/// let queue = ConcurrentPriorityQueue::<u32, i32, 4>::with_shards(3);
/// assert_eq!(queue.push(1, 10), None);
/// assert_eq!(queue.push(5, 20), None);
/// assert_eq!(queue.change_priority(&1, 30), Some(10));
/// assert_eq!(queue.pop_exact(), Some((1, 30)));
/// assert_eq!(queue.pop(), Some((5, 20)));
/// assert_eq!(queue.pop(), None);
/// ```
pub struct ConcurrentPriorityQueue<K, P, const PAGE: usize = DEFAULT_PAGE, D: Directory = Flat> {
    shards: Box<[CachePadded<Shard<K, P, PAGE, D>>]>,
    #[cfg(any(not(feature = "std"), loom))]
    random: AtomicU64,
}

impl<K: Key, P, const PAGE: usize, D: Directory> ConcurrentPriorityQueue<K, P, PAGE, D> {
    /// Creates an empty queue with the default shard count.
    ///
    /// With `std`, the count is four times the available parallelism, capped
    /// at 256, or four if parallelism is unknown. Without `std` it is eight.
    ///
    /// # Panics
    ///
    /// Panics if `PAGE` is zero.
    pub fn new() -> Self {
        Self::with_shards(default_shards())
    }

    /// Creates an empty queue with `shards` shards.
    ///
    /// # Panics
    ///
    /// Panics if `shards` or `PAGE` is zero.
    pub fn with_shards(shards: usize) -> Self {
        Self::with_shards_and_capacity(shards, 0)
    }

    /// Creates an empty queue with dense capacity distributed across `shards`
    /// shards, rounding up per shard. Index pages are allocated on demand.
    ///
    /// # Panics
    ///
    /// Panics if `shards` or `PAGE` is zero, or the capacity cannot be allocated.
    pub fn with_shards_and_capacity(shards: usize, capacity: usize) -> Self {
        assert!(
            shards > 0,
            "ConcurrentPriorityQueue requires at least one shard"
        );
        assert!(
            PAGE > 0,
            "ConcurrentPriorityQueue requires a nonzero page size"
        );
        let per_shard = capacity.div_ceil(shards);
        Self {
            shards: (0..shards)
                .map(|_| {
                    CachePadded(Shard {
                        queue: RwLock::new(PagedPriorityQueue::with_capacity(per_shard)),
                        len: AtomicUsize::new(0),
                    })
                })
                .collect(),
            #[cfg(any(not(feature = "std"), loom))]
            random: AtomicU64::new(SPLITMIX_INCREMENT),
        }
    }

    /// Returns the number of shards.
    pub fn shards(&self) -> usize {
        self.shards.len()
    }

    /// Returns the sum of shard lengths, exact when no write is in flight.
    pub fn len(&self) -> usize {
        self.shards
            .iter()
            .map(|shard| shard.0.len.load(Ordering::Relaxed))
            .sum()
    }

    /// Returns whether the length hints report an empty queue.
    ///
    /// Like [`len`](Self::len), this is exact when no write is in flight.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the total number of allocated index pages, reading one shard
    /// at a time. The total is weakly consistent across shards.
    pub fn allocated_pages(&self) -> usize {
        self.read_shards()
            .map(|queue| queue.allocated_pages())
            .sum()
    }

    /// Looks up the exact key, including its generation, under a read lock.
    pub fn contains_key(&self, key: &K) -> bool {
        self.shard(key).queue.read().contains_key(key)
    }

    /// Clones the exact key's priority under a read lock, or returns `None`
    /// if absent. The clone must avoid calling back into this queue.
    pub fn priority(&self, key: &K) -> Option<P>
    where
        P: Clone,
    {
        self.with_priority(key, P::clone)
    }

    /// Calls `f` with the exact key's priority under a read lock.
    ///
    /// Returns `None` without calling `f` if absent. The callback must avoid
    /// calling back into this queue.
    pub fn with_priority<R>(&self, key: &K, f: impl FnOnce(&P) -> R) -> Option<R> {
        self.shard(key).queue.read().priority(key).map(f)
    }

    /// Removes all entries, retaining index pages and dense capacity.
    pub fn clear(&mut self) {
        for CachePadded(shard) in self.shards.iter_mut() {
            let update = UpdateLen {
                queue: shard.queue.get_mut(),
                len: &shard.len,
            };
            update.queue.clear();
        }
    }

    /// Shrinks the dense arrays and directories, freeing index pages of
    /// empty shards.
    pub fn shrink_to_fit(&mut self) {
        for CachePadded(shard) in self.shards.iter_mut() {
            shard.queue.get_mut().shrink_to_fit();
        }
    }

    /// Eagerly removes all entries and returns them in arbitrary order.
    ///
    /// Index pages and dense capacity stay allocated. Dropping the iterator
    /// drops its remaining entries.
    pub fn drain(&mut self) -> IntoIter<(K, P)> {
        let mut entries = Vec::with_capacity(self.len());
        for CachePadded(shard) in self.shards.iter_mut() {
            entries.extend(shard.queue.get_mut().drain());
            shard.len.store(0, Ordering::Relaxed);
        }
        entries.into_iter()
    }

    fn shard_of(&self, key: &K) -> usize {
        (splitmix64_finalizer((key.index() / PAGE) as u64) % self.shards() as u64) as usize
    }

    fn shard(&self, key: &K) -> &Shard<K, P, PAGE, D> {
        &self.shards[self.shard_of(key)].0
    }

    fn read_shards(
        &self,
    ) -> impl Iterator<Item = ReadGuard<'_, PagedPriorityQueue<K, P, PAGE, D>>> {
        self.shards.iter().map(|shard| shard.0.queue.read())
    }

    fn into_entries(self) -> impl Iterator<Item = (K, P)> {
        self.shards
            .into_vec()
            .into_iter()
            .flat_map(|shard| shard.0.queue.into_inner())
    }
}

impl<K: Key, P: Ord, const PAGE: usize, D: Directory> ConcurrentPriorityQueue<K, P, PAGE, D> {
    /// Inserts a key or updates its priority under one shard write lock.
    ///
    /// Returns the previous priority of the same key, keeping the stored
    /// key. Another generation occupying the slot is evicted and yields
    /// `None`.
    pub fn push(&self, key: K, priority: P) -> Option<P> {
        self.shard(&key).update(|queue| queue.push(key, priority))
    }

    /// Inserts or updates under one shard write lock, refusing an occupied
    /// slot held by another generation.
    ///
    /// Returns the previous priority on update, or the supplied pair on
    /// refusal.
    pub fn try_push(&self, key: K, priority: P) -> Result<Option<P>, (K, P)> {
        self.shard(&key)
            .update(|queue| queue.try_push(key, priority))
    }

    /// Stores the supplied key and priority under one shard write lock.
    ///
    /// Returns the former occupant of the slot, including an evicted
    /// generation or an equal key.
    pub fn replace(&self, key: K, priority: P) -> Option<(K, P)> {
        self.shard(&key)
            .update(|queue| queue.replace(key, priority))
    }

    /// Inserts an absent key or raises its priority if strictly greater,
    /// returning whether the queue changed. An absent generation evicts the
    /// slot's occupant as in [`push`](Self::push).
    pub fn push_increase(&self, key: K, priority: P) -> bool {
        self.shard(&key)
            .update(|queue| queue.push_increase(key, priority))
    }

    /// Inserts an absent key or lowers its priority if strictly smaller,
    /// returning whether the queue changed. An absent generation evicts the
    /// slot's occupant as in [`push`](Self::push).
    pub fn push_decrease(&self, key: K, priority: P) -> bool {
        self.shard(&key)
            .update(|queue| queue.push_decrease(key, priority))
    }

    /// Changes the priority of the exact key under a write lock, returning
    /// its previous priority or `None` if absent.
    pub fn change_priority(&self, key: &K, priority: P) -> Option<P> {
        self.shard(key)
            .update(|queue| queue.change_priority(key, priority))
    }

    /// Mutates the exact key's priority under a write lock and repairs heap
    /// order. Returns whether the key was present.
    ///
    /// The callback runs only for a present key and must avoid calling back
    /// into this queue. If it panics, its changes remain and heap order may
    /// be broken.
    pub fn change_priority_with(&self, key: &K, change: impl FnOnce(&mut P)) -> bool {
        self.shard(key)
            .update(|queue| queue.change_priority_with(key, change))
    }

    /// Removes and returns the stored pair for the exact key, or `None` if
    /// absent, under one shard write lock.
    pub fn remove(&self, key: &K) -> Option<(K, P)> {
        self.shard(key).update(|queue| queue.remove(key))
    }

    /// Removes an entry using relaxed priority ordering.
    ///
    /// Samples two distinct shards, preferring nonzero length hints. Locks
    /// both in ascending shard order and removes the greater top. If both
    /// are empty, visits every shard once from a random start, locking one
    /// at a time, and removes the first top it finds.
    ///
    /// With one shard this removes the maximum. With multiple shards there
    /// is no rank guarantee. Rihani, Sanders and Dementiev (2015) report
    /// empirical expected rank O(shards) for [MultiQueues].
    ///
    /// Returns `None` only if every shard was empty when visited. An entry
    /// present throughout the call, with its insertion happening-before the
    /// call, therefore prevents `None` because entries never migrate.
    ///
    /// [MultiQueues]: https://arxiv.org/abs/1411.1209
    pub fn pop(&self) -> Option<(K, P)> {
        if self.shards() == 1 {
            return self.shards[0].0.update(PagedPriorityQueue::pop);
        }

        let [a, b] = self.choose_shards();
        if let Some(entry) = self.pop_pair(a.min(b), a.max(b)) {
            return Some(entry);
        }

        let start = self.random_index(self.shards());
        let found = self
            .shard_indices_from(start)
            .find_map(|index| self.shards[index].0.update(PagedPriorityQueue::pop));
        #[cfg(all(test, loom))]
        if found.is_some() {
            SCAN_FOUND.store(true, core::sync::atomic::Ordering::Relaxed);
        }
        found
    }

    /// Removes the global maximum, or returns `None` if empty.
    ///
    /// Acquires all shard write locks in ascending order. This operation
    /// is linearizable.
    pub fn pop_exact(&self) -> Option<(K, P)> {
        let mut queues: Vec<_> = self
            .shards
            .iter()
            .map(|shard| shard.0.queue.write())
            .collect();
        let best = queues
            .iter()
            .enumerate()
            .filter_map(|(index, queue)| queue.peek().map(|(_, priority)| (index, priority)))
            .max_by(|(_, left), (_, right)| left.cmp(right))?
            .0;
        self.shards[best]
            .0
            .update_locked(&mut queues[best], PagedPriorityQueue::pop)
    }

    /// Copies the key and clones the priority of the global maximum, or
    /// returns `None` if empty.
    ///
    /// Acquires all shard read locks in ascending order for a linearizable
    /// view. The priority's `Clone` must avoid calling back into this queue.
    pub fn peek_exact(&self) -> Option<(K, P)>
    where
        P: Clone,
    {
        let queues: Vec<_> = self.read_shards().collect();
        queues
            .iter()
            .filter_map(|queue| queue.peek())
            .max_by(|(_, left), (_, right)| left.cmp(right))
            .map(|(key, priority)| (*key, priority.clone()))
    }

    /// Consumes the queue into entries in ascending priority order.
    pub fn into_sorted_vec(self) -> Vec<(K, P)> {
        let mut entries: Vec<_> = self.into_iter().collect();
        entries.sort_unstable_by(|(_, left), (_, right)| left.cmp(right));
        entries
    }

    /// Consumes the queue and merges its entries into one plain queue.
    ///
    /// # Panics
    ///
    /// Panics if the entries exceed the plain queue's `u32::MAX` limit or
    /// if comparing priorities panics.
    pub fn into_queue(self) -> PagedPriorityQueue<K, P, PAGE, D> {
        let mut queue = PagedPriorityQueue::with_capacity(self.len());
        queue.extend(self.into_entries());
        queue
    }

    /// Copies entries into a plain queue using one shard read lock at a time.
    ///
    /// The snapshot is consistent within each shard and weakly consistent
    /// across shards. An entry that stays unchanged throughout the call
    /// is included if its insertion happens-before the call. Priority
    /// comparisons and clones must avoid calling back into this queue.
    ///
    /// # Panics
    ///
    /// Panics if the entries copied exceed the plain queue's `u32::MAX`
    /// limit or if cloning or comparing priorities panics.
    pub fn snapshot(&self) -> PagedPriorityQueue<K, P, PAGE, D>
    where
        P: Clone,
    {
        let mut snapshot = PagedPriorityQueue::with_capacity(self.len());
        for shard in self.read_shards() {
            snapshot.extend(shard.iter().map(|(key, priority)| (*key, priority.clone())));
        }
        snapshot
    }

    fn random_index(&self, bound: usize) -> usize {
        #[cfg(all(feature = "std", not(loom)))]
        let word = random_word();
        #[cfg(any(not(feature = "std"), loom))]
        let word =
            splitmix64_finalizer(self.random.fetch_add(SPLITMIX_INCREMENT, Ordering::Relaxed));
        (word % bound as u64) as usize
    }

    fn shard_indices_from(&self, start: usize) -> impl Iterator<Item = usize> {
        (start..self.shards()).chain(0..start)
    }

    fn choose_shards(&self) -> [usize; 2] {
        let count = self.shards();
        let first = self.random_index(count);
        let second = self.random_index(count - 1);
        let mut chosen = [first, second + usize::from(second >= first)];
        let has_hint = |index: usize| self.shards[index].0.len.load(Ordering::Relaxed) != 0;
        let mut hinted = chosen.map(has_hint);

        // Keep the random pair when both hints are nonzero. Otherwise try
        // to replace the empty candidates in one pass, without locking.
        for index in self.shard_indices_from(first) {
            let Some(slot) = hinted.iter().position(|&nonempty| !nonempty) else {
                break;
            };
            if !chosen.contains(&index) && has_hint(index) {
                chosen[slot] = index;
                hinted[slot] = true;
            }
        }
        chosen
    }

    fn pop_pair(&self, lower: usize, higher: usize) -> Option<(K, P)> {
        let left = &self.shards[lower].0;
        let right = &self.shards[higher].0;
        let mut left_queue = left.queue.write();
        let mut right_queue = right.queue.write();
        let left_top = left_queue.peek().map(|(_, priority)| priority);
        let right_top = right_queue.peek().map(|(_, priority)| priority);
        if left_top.cmp(&right_top).is_ge() {
            left.update_locked(&mut left_queue, PagedPriorityQueue::pop)
        } else {
            right.update_locked(&mut right_queue, PagedPriorityQueue::pop)
        }
    }
}

impl<K: Key, P, const PAGE: usize, D: Directory> Default
    for ConcurrentPriorityQueue<K, P, PAGE, D>
{
    fn default() -> Self {
        Self::new()
    }
}

/// Rebuilds the queue from a weakly consistent snapshot with the same shard
/// count. See [`ConcurrentPriorityQueue::snapshot`] for locking.
impl<K: Key, P: Ord + Clone, const PAGE: usize, D: Directory> Clone
    for ConcurrentPriorityQueue<K, P, PAGE, D>
{
    fn clone(&self) -> Self {
        let mut cloned = Self::with_shards(self.shards());
        cloned.extend(self.snapshot());
        cloned
    }
}

/// Formats entries as a map using one shard read lock at a time, with the
/// same consistency as [`ConcurrentPriorityQueue::snapshot`]. User formatting
/// must avoid calling back into the queue.
impl<K: Key + fmt::Debug, P: fmt::Debug, const PAGE: usize, D: Directory> fmt::Debug
    for ConcurrentPriorityQueue<K, P, PAGE, D>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut map = f.debug_map();
        for shard in self.read_shards() {
            map.entries(shard.iter());
        }
        map.finish()
    }
}

/// Moves a plain queue into the default number of shards by re-pushing its
/// entries in O(n log n) time.
impl<K: Key, P: Ord, const PAGE: usize, D: Directory> From<PagedPriorityQueue<K, P, PAGE, D>>
    for ConcurrentPriorityQueue<K, P, PAGE, D>
{
    fn from(queue: PagedPriorityQueue<K, P, PAGE, D>) -> Self {
        queue.into_iter().collect()
    }
}

/// Consumes the queue and yields entries in arbitrary order.
impl<K: Key, P, const PAGE: usize, D: Directory> IntoIterator
    for ConcurrentPriorityQueue<K, P, PAGE, D>
{
    type Item = (K, P);
    type IntoIter = IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        let mut entries = Vec::with_capacity(self.len());
        entries.extend(self.into_entries());
        entries.into_iter()
    }
}

/// Pushes entries under exclusive access. Later entries in a slot win.
impl<K: Key, P: Ord, const PAGE: usize, D: Directory> Extend<(K, P)>
    for ConcurrentPriorityQueue<K, P, PAGE, D>
{
    fn extend<I: IntoIterator<Item = (K, P)>>(&mut self, iter: I) {
        for (key, priority) in iter {
            let index = self.shard_of(&key);
            let shard = &mut self.shards[index].0;
            let update = UpdateLen {
                queue: shard.queue.get_mut(),
                len: &shard.len,
            };
            update.queue.push(key, priority);
        }
    }
}

/// Collects entries into the default number of shards. Later entries in a
/// slot win.
impl<K: Key, P: Ord, const PAGE: usize, D: Directory> FromIterator<(K, P)>
    for ConcurrentPriorityQueue<K, P, PAGE, D>
{
    fn from_iter<I: IntoIterator<Item = (K, P)>>(iter: I) -> Self {
        let mut queue = Self::new();
        queue.extend(iter);
        queue
    }
}

#[cfg(feature = "serde")]
mod serde_impl {
    use super::*;
    use serde::de::{Deserialize, Deserializer};
    use serde::ser::{Serialize, SerializeSeq, Serializer};

    /// Serializes a sequence of key-priority pairs in the plain queue format,
    /// holding one shard read lock at a time. User serialization must avoid
    /// calling back into the queue.
    ///
    /// The length of the weakly consistent view is unknown upfront. For a
    /// serializer that requires a sequence length, serialize
    /// [`ConcurrentPriorityQueue::snapshot`] instead.
    impl<K: Key + Serialize, P: Serialize, const PAGE: usize, D: Directory> Serialize
        for ConcurrentPriorityQueue<K, P, PAGE, D>
    {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut seq = serializer.serialize_seq(None)?;
            for shard in self.read_shards() {
                for entry in shard.iter() {
                    seq.serialize_element(&entry)?;
                }
            }
            seq.end()
        }
    }

    /// Reads the plain queue format and redistributes entries into the
    /// default number of shards.
    impl<'de, K: Key + Deserialize<'de>, P: Ord + Deserialize<'de>, const PAGE: usize, D: Directory>
        Deserialize<'de> for ConcurrentPriorityQueue<K, P, PAGE, D>
    {
        fn deserialize<De: Deserializer<'de>>(deserializer: De) -> Result<Self, De::Error> {
            PagedPriorityQueue::deserialize(deserializer).map(Self::from)
        }
    }
}

/// Set when `pop` finds an entry only in its fallback scan (a plain atomic
/// outside the loom model): the scan model asserts it is reached.
#[cfg(all(test, loom))]
static SCAN_FOUND: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

#[cfg(all(test, loom))]
#[path = "loom_priority_queue.rs"]
mod loom_tests;
