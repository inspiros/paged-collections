//! [`PagedPriorityQueue`]: an indexed max-heap with a paged key index.
//!
//! The greatest priority is returned first. Use [`core::cmp::Reverse`] for a
//! min-queue. Equal priorities have unspecified order. Keys and priorities
//! occupy parallel contiguous arrays in heap order; a four-way heap keeps
//! insertion, priority changes and removal to O(log n) comparisons.
//!
//! Lookups use [`PagedSparseSet`]'s paged index. With [`Flat`] they are O(1);
//! [`Radix`](crate::page::Radix) and [`Tree`](crate::page::Tree) add their
//! directory lookup costs to each index access. Only occupied key ranges
//! allocate index pages.
//!
//! A slot holds one key. [`push`](PagedPriorityQueue::push) evicts another
//! generation of that key; [`try_push`](PagedPriorityQueue::try_push) refuses
//! it. Lookups always check the whole key, including its generation.
//!
//! # Panic safety and ordering
//!
//! If a priority comparison or a closure passed to
//! [`change_priority_with`](PagedPriorityQueue::change_priority_with) panics,
//! all remaining entries stay findable and paired with their priorities.
//! A mutation already made remains made, but heap order may be broken, so
//! subsequent maximum and sorted-order results are unspecified. Changing a
//! stored priority's ordering through interior mutability is a logic error
//! with the same consequence. These errors cannot cause undefined behaviour.
//!
//! ```
//! use paged_collections::priority_queue::PagedPriorityQueue;
//!
//! let mut queue: PagedPriorityQueue<u32, i32> = PagedPriorityQueue::new();
//! assert_eq!(queue.push(7, 10), None);
//! assert_eq!(queue.push(3_000_000, 20), None);
//! assert_eq!(queue.change_priority(&7, 30), Some(10));
//! assert_eq!(queue.pop(), Some((7, 30)));
//! assert_eq!(queue.pop(), Some((3_000_000, 20)));
//! ```

use alloc::vec::Vec;
use core::fmt;
use core::iter::{FusedIterator, Zip};
use core::marker::PhantomData;
use core::slice;

use crate::key::Key;
use crate::page::{DEFAULT_PAGE, Directory, Flat};
use crate::sparse_set::{PagedSparseSet, Slot};

const ARITY: usize = 4;

/// An indexed max-priority queue with a paged index and dense heap storage.
///
/// `PAGE` is the non-zero number of index slots per page. `D` selects the
/// directory: [`Flat`] (default), [`Radix`](crate::page::Radix), or
/// [`Tree`](crate::page::Tree). See [`crate::page`]. At most `u32::MAX`
/// entries can be stored. The greatest priority is at position zero; ties
/// have unspecified order. See the [module docs](self) for panic safety.
pub struct PagedPriorityQueue<K, P, const PAGE: usize = DEFAULT_PAGE, D: Directory = Flat> {
    // Invariants: equal lengths; priorities[i] belongs to keys.as_slice()[i].
    // For i > 0, priorities[(i - 1) / ARITY] >= priorities[i], unless user
    // code panicked or changed ordering through interior mutability.
    keys: PagedSparseSet<K, PAGE, D>,
    priorities: Vec<P>,
}

impl<K: Key, P, const PAGE: usize, D: Directory> PagedPriorityQueue<K, P, PAGE, D> {
    /// Creates an empty queue without allocating.
    pub fn new() -> Self {
        Self {
            keys: PagedSparseSet::new(),
            priorities: Vec::new(),
        }
    }

    /// Creates an empty queue with room for at least `capacity` entries in
    /// its dense arrays. Index pages are allocated on demand.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            keys: PagedSparseSet::with_capacity(capacity),
            priorities: Vec::with_capacity(capacity),
        }
    }

    /// Number of entries in the queue.
    pub fn len(&self) -> usize {
        self.priorities.len()
    }

    /// Whether the queue has no entries.
    pub fn is_empty(&self) -> bool {
        self.priorities.is_empty()
    }

    /// Number of entries the dense arrays can hold without reallocating.
    pub fn capacity(&self) -> usize {
        self.keys.capacity().min(self.priorities.capacity())
    }

    /// Reserves room for at least `additional` more entries in both dense arrays.
    pub fn reserve(&mut self, additional: usize) {
        self.keys.reserve(additional);
        self.priorities.reserve(additional);
    }

    /// Shrinks the dense arrays and index directory. Frees every index page
    /// if the queue is empty.
    pub fn shrink_to_fit(&mut self) {
        self.keys.shrink_to_fit();
        self.priorities.shrink_to_fit();
    }

    /// Removes every entry, dropping its priority. Index pages and dense
    /// capacity stay allocated for reuse.
    pub fn clear(&mut self) {
        // Vec::clear sets its length to zero before running P::drop. Clear
        // the keys first, so even a panicking destructor leaves equal lengths.
        self.keys.clear();
        self.priorities.clear();
    }

    /// Number of allocated pages in the key index.
    pub fn allocated_pages(&self) -> usize {
        self.keys.allocated_pages()
    }

    /// The stored key and greatest priority, or `None` if empty.
    pub fn peek(&self) -> Option<(&K, &P)> {
        self.keys.as_slice().first().zip(self.priorities.first())
    }

    /// Whether the exact key is present, including its generation.
    pub fn contains_key(&self, key: &K) -> bool {
        self.keys.contains(*key)
    }

    /// The priority of the exact key, or `None` if absent.
    pub fn priority(&self, key: &K) -> Option<&P> {
        self.position(key).map(|pos| &self.priorities[pos])
    }

    /// Heap position of the exact key, or `None` if absent.
    ///
    /// Positions index both slices and can change on any heap mutation.
    pub fn position(&self, key: &K) -> Option<usize> {
        self.keys.position(*key)
    }

    /// Iterates over entries in heap order, which is otherwise unspecified.
    pub fn iter(&self) -> Iter<'_, K, P> {
        self.keys.iter().zip(self.priorities.iter())
    }

    /// Iterates over keys in heap order.
    pub fn keys(&self) -> slice::Iter<'_, K> {
        self.keys.iter()
    }

    /// Iterates over priorities in heap order.
    pub fn priorities(&self) -> slice::Iter<'_, P> {
        self.priorities.iter()
    }

    /// Keys in heap order; position zero has the greatest priority.
    pub fn key_slice(&self) -> &[K] {
        self.keys.as_slice()
    }

    /// Priorities in heap order, parallel to [`key_slice`](Self::key_slice).
    pub fn priority_slice(&self) -> &[P] {
        &self.priorities
    }

    /// Removes all entries eagerly and iterates over them in arbitrary order.
    ///
    /// Index pages and dense capacity are kept. Dropping the iterator drops
    /// its remaining entries; forgetting it still leaves the queue empty.
    /// The queue stays mutably borrowed until the iterator is released.
    pub fn drain(&mut self) -> Drain<'_, K, P, PAGE, D> {
        // Allocate before either length changes. All following pushes fit;
        // copying keys and moving priorities runs no user code.
        let mut entries = Vec::with_capacity(self.len());
        for (key, priority) in self.keys.iter().copied().zip(self.priorities.drain(..)) {
            entries.push((key, priority));
        }
        self.keys.clear();
        Drain {
            inner: entries.into_iter(),
            _borrow: PhantomData,
        }
    }

    /// Swaps complete entries without comparisons or allocation.
    fn swap_positions(&mut self, a: usize, b: usize) {
        self.keys.swap_positions(a, b);
        self.priorities.swap(a, b);
    }
}

impl<K: Key, P: Ord, const PAGE: usize, D: Directory> PagedPriorityQueue<K, P, PAGE, D> {
    /// Inserts a key or updates its priority, returning the previous priority
    /// of that same key. An equal stored key is kept.
    ///
    /// If another generation holds the slot, it is evicted and `None` is
    /// returned. The replacement starts at the old entry's position and is
    /// then sifted to restore heap order.
    ///
    /// # Panics
    ///
    /// Panics if a new entry would exceed `u32::MAX` entries, if storage
    /// cannot be allocated, or if comparing priorities panics. Allocation
    /// failure leaves entries unchanged; a comparison panic has the
    /// guarantees described in the [module docs](self).
    pub fn push(&mut self, key: K, priority: P) -> Option<P> {
        if let Some(pos) = self.keys.occupant_pos(key.index()) {
            let stored = self.keys.as_slice()[pos];
            // Equality is user code too: finish it before changing either array.
            let same = stored == key;
            if !same {
                self.keys.replace(key);
            }
            let old = core::mem::replace(&mut self.priorities[pos], priority);
            self.repair(pos);
            if same { Some(old) } else { None }
        } else {
            // Reserve priorities first: once insert appends the key, push
            // cannot allocate or call user code, so lengths stay in step.
            self.priorities.reserve(1);
            self.keys.insert(key);
            self.priorities.push(priority);
            self.sift_up(self.len() - 1);
            None
        }
    }

    /// Inserts or updates as [`push`](Self::push), but returns the supplied
    /// pair unchanged if another generation holds the slot.
    ///
    /// Updating the same key is allowed and returns its previous priority.
    /// Panics under the same conditions as `push`.
    pub fn try_push(&mut self, key: K, priority: P) -> Result<Option<P>, (K, P)> {
        if self
            .keys
            .occupant(key.index())
            .is_some_and(|stored| *stored != key)
        {
            return Err((key, priority));
        }
        Ok(self.push(key, priority))
    }

    /// Inserts an absent key, or updates it only if `priority` is strictly
    /// greater. Returns whether the queue changed.
    ///
    /// An absent generation evicts another generation as [`push`](Self::push)
    /// does, regardless of the evicted priority. Panics as `push` does.
    pub fn push_increase(&mut self, key: K, priority: P) -> bool {
        if self.priority(&key).is_some_and(|old| priority <= *old) {
            return false;
        }
        self.push(key, priority);
        true
    }

    /// Inserts an absent key, or updates it only if `priority` is strictly
    /// smaller. Returns whether the queue changed.
    ///
    /// An absent generation evicts another generation as [`push`](Self::push)
    /// does, regardless of the evicted priority. Panics as `push` does.
    pub fn push_decrease(&mut self, key: K, priority: P) -> bool {
        if self.priority(&key).is_some_and(|old| priority >= *old) {
            return false;
        }
        self.push(key, priority);
        true
    }

    /// Changes the priority of a present key, returning its old priority.
    /// Returns `None` without changing the queue if the exact key is absent.
    ///
    /// A comparison panic has the guarantees in the [module docs](self).
    pub fn change_priority(&mut self, key: &K, priority: P) -> Option<P> {
        let pos = self.position(key)?;
        let old = core::mem::replace(&mut self.priorities[pos], priority);
        self.repair(pos);
        Some(old)
    }

    /// Calls `change` on a present key's priority and restores heap order.
    /// Returns whether the exact key was present; an absent key does not
    /// call the closure.
    ///
    /// If the closure or a comparison panics, its changes remain and heap
    /// order may be broken, but entries stay findable and correctly paired.
    pub fn change_priority_with(&mut self, key: &K, change: impl FnOnce(&mut P)) -> bool {
        let Some(pos) = self.position(key) else {
            return false;
        };
        change(&mut self.priorities[pos]);
        self.repair(pos);
        true
    }

    /// Removes and returns the entry with the greatest priority.
    ///
    /// Returns `None` if empty. If comparing priorities panics, the removed
    /// entry is dropped and the remaining entries stay findable and paired,
    /// though heap order may be broken.
    pub fn pop(&mut self) -> Option<(K, P)> {
        if self.is_empty() {
            None
        } else {
            Some(self.remove_at(0))
        }
    }

    /// Removes and returns the stored key and priority for the exact key.
    ///
    /// Returns `None` if absent. A comparison panic has the same guarantees
    /// as [`pop`](Self::pop).
    pub fn remove(&mut self, key: &K) -> Option<(K, P)> {
        let pos = self.position(key)?;
        Some(self.remove_at(pos))
    }

    fn remove_at(&mut self, pos: usize) -> (K, P) {
        // Remove by position, not by key equality (user code): the key and
        // priority arrays stay in step even if `K: Eq` misbehaves. Neither
        // swap_remove can allocate or run P code.
        let key = self.keys.remove_at(pos);
        let priority = self.priorities.swap_remove(pos);
        if pos < self.len() {
            self.repair(pos);
        }
        (key, priority)
    }

    /// Consumes the queue into entries in ascending priority order.
    /// Equal priorities have unspecified order.
    pub fn into_sorted_vec(self) -> Vec<(K, P)> {
        let mut entries: Vec<_> = self.into_sorted_iter().collect();
        entries.reverse();
        entries
    }

    /// Consumes the queue and repeatedly pops entries in descending priority
    /// order. Equal priorities have unspecified order.
    pub fn into_sorted_iter(self) -> IntoSortedIter<K, P, PAGE, D> {
        IntoSortedIter { queue: self }
    }

    fn repair(&mut self, pos: usize) {
        if pos > 0 && self.priorities[pos] > self.priorities[(pos - 1) / ARITY] {
            self.sift_up(pos);
        } else {
            self.sift_down(pos);
        }
    }

    fn sift_up(&mut self, mut pos: usize) {
        while pos > 0 {
            let parent = (pos - 1) / ARITY;
            if self.priorities[parent] >= self.priorities[pos] {
                break;
            }
            // Every completed swap restores pairing and the key index
            // before the next comparison can panic.
            self.swap_positions(parent, pos);
            pos = parent;
        }
    }

    fn sift_down(&mut self, mut pos: usize) {
        // Checked arithmetic also handles a leaf near usize::MAX on 32-bit
        // targets, where multiplying its position by ARITY could overflow.
        while let Some(first) = pos.checked_mul(ARITY).and_then(|n| n.checked_add(1)) {
            if first >= self.len() {
                break;
            }
            let end = first.saturating_add(ARITY).min(self.len());
            let mut greatest = first;
            for child in first + 1..end {
                if self.priorities[child] > self.priorities[greatest] {
                    greatest = child;
                }
            }
            if self.priorities[pos] >= self.priorities[greatest] {
                break;
            }
            self.swap_positions(pos, greatest);
            pos = greatest;
        }
    }

    #[cfg(test)]
    fn check(&self) {
        assert_eq!(self.keys.len(), self.priorities.len());
        for (pos, key) in self.keys.iter().enumerate() {
            assert_eq!(self.position(key), Some(pos));
            if pos > 0 {
                assert!(self.priorities[(pos - 1) / ARITY] >= self.priorities[pos]);
            }
        }
    }
}

impl<K: Key, P, const PAGE: usize, D: Directory> Default for PagedPriorityQueue<K, P, PAGE, D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Key, P: Clone, const PAGE: usize, D: Directory> Clone for PagedPriorityQueue<K, P, PAGE, D>
where
    D::Store<Slot, PAGE>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            keys: self.keys.clone(),
            priorities: self.priorities.clone(),
        }
    }
}

impl<K: Key + fmt::Debug, P: fmt::Debug, const PAGE: usize, D: Directory> fmt::Debug
    for PagedPriorityQueue<K, P, PAGE, D>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

/// Repeatedly pushes entries, so later entries in a slot win.
impl<K: Key, P: Ord, const PAGE: usize, D: Directory> Extend<(K, P)>
    for PagedPriorityQueue<K, P, PAGE, D>
{
    fn extend<I: IntoIterator<Item = (K, P)>>(&mut self, iter: I) {
        for (key, priority) in iter {
            self.push(key, priority);
        }
    }
}

impl<K: Key, P: Ord, const PAGE: usize, D: Directory> FromIterator<(K, P)>
    for PagedPriorityQueue<K, P, PAGE, D>
{
    fn from_iter<I: IntoIterator<Item = (K, P)>>(iter: I) -> Self {
        let mut queue = Self::new();
        queue.extend(iter);
        queue
    }
}

impl<'a, K: Key, P, const PAGE: usize, D: Directory> IntoIterator
    for &'a PagedPriorityQueue<K, P, PAGE, D>
{
    type Item = (&'a K, &'a P);
    type IntoIter = Iter<'a, K, P>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<K: Key, P, const PAGE: usize, D: Directory> IntoIterator
    for PagedPriorityQueue<K, P, PAGE, D>
{
    type Item = (K, P);
    type IntoIter = IntoIter<K, P>;

    fn into_iter(self) -> Self::IntoIter {
        self.keys.into_iter().zip(self.priorities)
    }
}

/// Borrowed iterator over entries in heap order, from
/// [`iter`](PagedPriorityQueue::iter).
pub type Iter<'a, K, P> = Zip<slice::Iter<'a, K>, slice::Iter<'a, P>>;

/// Owning iterator over entries in arbitrary heap order.
pub type IntoIter<K, P> = Zip<alloc::vec::IntoIter<K>, alloc::vec::IntoIter<P>>;

/// Owning iterator over entries in descending priority order, from
/// [`into_sorted_iter`](PagedPriorityQueue::into_sorted_iter).
pub struct IntoSortedIter<K, P, const PAGE: usize = DEFAULT_PAGE, D: Directory = Flat> {
    queue: PagedPriorityQueue<K, P, PAGE, D>,
}

impl<K: Key, P: Ord, const PAGE: usize, D: Directory> Iterator for IntoSortedIter<K, P, PAGE, D> {
    type Item = (K, P);

    fn next(&mut self) -> Option<Self::Item> {
        self.queue.pop()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.queue.len();
        (len, Some(len))
    }
}

impl<K: Key, P: Ord, const PAGE: usize, D: Directory> ExactSizeIterator
    for IntoSortedIter<K, P, PAGE, D>
{
}
impl<K: Key, P: Ord, const PAGE: usize, D: Directory> FusedIterator
    for IntoSortedIter<K, P, PAGE, D>
{
}

/// Iterator over eagerly removed entries, from [`drain`](PagedPriorityQueue::drain).
/// Dropping it drops any remaining entries; the queue is already empty.
pub struct Drain<'a, K, P, const PAGE: usize = DEFAULT_PAGE, D: Directory = Flat> {
    inner: alloc::vec::IntoIter<(K, P)>,
    _borrow: PhantomData<&'a mut PagedPriorityQueue<K, P, PAGE, D>>,
}

impl<K, P, const PAGE: usize, D: Directory> Iterator for Drain<'_, K, P, PAGE, D> {
    type Item = (K, P);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K, P, const PAGE: usize, D: Directory> DoubleEndedIterator for Drain<'_, K, P, PAGE, D> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back()
    }
}

impl<K, P, const PAGE: usize, D: Directory> ExactSizeIterator for Drain<'_, K, P, PAGE, D> {}
impl<K, P, const PAGE: usize, D: Directory> FusedIterator for Drain<'_, K, P, PAGE, D> {}

#[cfg(feature = "serde")]
mod serde_impl {
    use super::*;
    use serde::de::{Deserialize, Deserializer, SeqAccess, Visitor};
    use serde::ser::{Serialize, Serializer};

    /// Serializes as a sequence of `(key, priority)` pairs in heap order.
    impl<K: Key + Serialize, P: Serialize, const PAGE: usize, D: Directory> Serialize
        for PagedPriorityQueue<K, P, PAGE, D>
    {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            serializer.collect_seq(self.iter())
        }
    }

    /// Deserializes pairs by repeated `push`, so later entries in a slot win.
    /// Memory follows key indices: use Radix or Tree for widely spread keys.
    impl<'de, K, P, const PAGE: usize, D> Deserialize<'de> for PagedPriorityQueue<K, P, PAGE, D>
    where
        K: Key + Deserialize<'de>,
        P: Ord + Deserialize<'de>,
        D: Directory,
    {
        fn deserialize<De: Deserializer<'de>>(deserializer: De) -> Result<Self, De::Error> {
            struct SeqVisitor<K, P, const PAGE: usize, D>(PhantomData<(K, P, D)>);

            impl<'de, K, P, const PAGE: usize, D> Visitor<'de> for SeqVisitor<K, P, PAGE, D>
            where
                K: Key + Deserialize<'de>,
                P: Ord + Deserialize<'de>,
                D: Directory,
            {
                type Value = PagedPriorityQueue<K, P, PAGE, D>;

                fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    f.write_str("a sequence of key-priority pairs")
                }

                fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                    // Cap both entry count and bytes from an untrusted hint.
                    let entry = core::mem::size_of::<K>().saturating_add(core::mem::size_of::<P>());
                    let bytes = usize::try_from(1_u32 << 20).unwrap_or(usize::MAX);
                    let cap = bytes.checked_div(entry).unwrap_or(usize::MAX).min(4096);
                    let mut queue =
                        PagedPriorityQueue::with_capacity(seq.size_hint().unwrap_or(0).min(cap));
                    while let Some((key, priority)) = seq.next_element()? {
                        queue.push(key, priority);
                    }
                    Ok(queue)
                }
            }

            deserializer.deserialize_seq(SeqVisitor::<K, P, PAGE, D>(PhantomData))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cmp::Reverse;

    #[test]
    fn updates_remove_and_pop_restore_heap_order() {
        let mut queue: PagedPriorityQueue<u16, i32, 4> = PagedPriorityQueue::new();
        queue.check();
        for key in 0..80 {
            assert_eq!(queue.push(key, i32::from(key)), None);
            queue.check();
        }
        assert_eq!(queue.push(3, 100), Some(3));
        queue.check();
        assert_eq!(queue.peek(), Some((&3, &100)));
        assert_eq!(queue.try_push(3, -100), Ok(Some(100)));
        queue.check();
        assert_eq!(queue.change_priority(&1, 120), Some(1));
        queue.check();
        assert!(queue.change_priority_with(&1, |p| *p = -120));
        queue.check();
        assert_eq!(queue.change_priority(&100, 0), None);
        queue.check();
        assert!(!queue.change_priority_with(&100, |_| panic!("absent")));
        queue.check();
        // Remove from the middle, the end, and the root.
        let middle = queue.key_slice()[queue.len() / 2];
        assert_eq!(queue.remove(&middle).map(|e| e.0), Some(middle));
        queue.check();
        let last = *queue.key_slice().last().unwrap();
        assert_eq!(queue.remove(&last).map(|e| e.0), Some(last));
        queue.check();
        assert_eq!(queue.remove(&100), None);
        queue.check();
        let mut previous = i32::MAX;
        while let Some((_, priority)) = queue.pop() {
            assert!(priority <= previous);
            previous = priority;
            queue.check();
        }
        assert_eq!(queue.pop(), None);
        queue.check();
    }

    #[test]
    fn strict_changes_ties_and_min_queue() {
        let mut queue: PagedPriorityQueue<u16, i32, 1> = PagedPriorityQueue::new();
        assert!(queue.push_increase(1, 10));
        queue.check();
        for priority in [9, 10] {
            assert!(!queue.push_increase(1, priority));
            queue.check();
        }
        assert!(queue.push_increase(1, 11));
        queue.check();
        for priority in [11, 12] {
            assert!(!queue.push_decrease(1, priority));
            queue.check();
        }
        assert!(queue.push_decrease(1, 10));
        queue.check();
        assert!(queue.push_decrease(2, 10));
        queue.check();
        assert_eq!(queue.clone().into_sorted_vec().len(), 2);
        let mut min: PagedPriorityQueue<u16, Reverse<i32>> = PagedPriorityQueue::new();
        for (key, value) in [(1, 10), (2, -2), (3, 10)] {
            min.push(key, Reverse(value));
            min.check();
        }
        assert_eq!(min.pop(), Some((2, Reverse(-2))));
        min.check();
    }

    #[test]
    fn iteration_sorting_and_eager_drain() {
        let mut queue: PagedPriorityQueue<u16, i32, 4> = PagedPriorityQueue::default();
        for entry in [(1, 5), (9, 2), (2, 5), (1, 7)] {
            queue.extend([entry]);
            queue.check();
        }
        assert_eq!(queue.len(), 3);
        assert_eq!(queue.iter().len(), 3);
        assert_eq!(queue.keys().copied().collect::<Vec<_>>(), queue.key_slice());
        assert_eq!(
            queue.priorities().copied().collect::<Vec<_>>(),
            queue.priority_slice()
        );
        assert_eq!(
            queue.iter().next_back(),
            queue.key_slice().last().zip(queue.priority_slice().last())
        );
        let entries: Vec<_> = queue.iter().map(|(&k, &p)| (k, p)).collect();
        assert_eq!(queue.clone().into_iter().collect::<Vec<_>>(), entries);
        assert_eq!((&queue).into_iter().count(), 3);
        assert_eq!(
            queue
                .clone()
                .into_sorted_vec()
                .iter()
                .map(|e| e.1)
                .collect::<Vec<_>>(),
            [2, 5, 7]
        );
        let mut sorted = queue.clone().into_sorted_iter();
        assert_eq!(sorted.len(), 3);
        assert_eq!(sorted.next(), Some((1, 7)));
        assert_eq!(sorted.len(), 2);
        assert_eq!(sorted.by_ref().map(|e| e.1).collect::<Vec<_>>(), [5, 2]);
        assert_eq!(sorted.next(), None);
        let capacity = queue.capacity();
        let pages = queue.allocated_pages();
        let mut drain = queue.drain();
        assert_eq!(drain.len(), 3);
        assert_eq!(drain.next(), entries.first().copied());
        assert_eq!(drain.next_back(), entries.last().copied());
        drop(drain);
        queue.check();
        assert!(queue.is_empty());
        assert_eq!(queue.capacity(), capacity);
        assert_eq!(queue.allocated_pages(), pages);
        queue.push(1, 1);
        queue.check();
        core::mem::forget(queue.drain());
        queue.check();
        assert!(queue.is_empty());
        assert_eq!(queue.pop(), None);
        queue.check();
    }

    #[test]
    fn capacity_pages_and_non_ordered_priorities() {
        let mut queue: PagedPriorityQueue<u16, i32, 4> = PagedPriorityQueue::with_capacity(5);
        queue.check();
        assert!(queue.capacity() >= 5);
        assert_eq!(queue.allocated_pages(), 0);
        for key in [1, 2, 40] {
            queue.push(key, 0);
            queue.check();
        }
        queue.reserve(20);
        queue.check();
        assert!(queue.capacity() >= 23);
        assert_eq!(queue.allocated_pages(), 2);
        let capacity = queue.capacity();
        queue.clear();
        queue.check();
        assert_eq!(queue.capacity(), capacity);
        assert_eq!(queue.allocated_pages(), 2);
        for key in [1, 2, 40] {
            assert!(!queue.contains_key(&key));
        }
        queue.push(40, 1);
        queue.check();
        assert!(!queue.contains_key(&1));
        queue.shrink_to_fit();
        queue.check();
        assert_eq!(queue.allocated_pages(), 2);
        queue.clear();
        queue.check();
        queue.shrink_to_fit();
        queue.check();
        assert_eq!(queue.allocated_pages(), 0);

        // Construction, inspection, clearing and arbitrary iteration do not
        // require P: Ord, even though heap mutations do.
        let mut unordered: PagedPriorityQueue<u16, f32> = PagedPriorityQueue::new();
        unordered.reserve(1);
        assert_eq!(unordered.peek(), None);
        assert_eq!(unordered.priority(&0), None);
        assert_eq!(unordered.clone().into_iter().count(), 0);
        assert_eq!(unordered.drain().count(), 0);
        unordered.clear();
        unordered.shrink_to_fit();
        assert!(unordered.key_slice().is_empty());
        assert!(unordered.priority_slice().is_empty());
    }
}
