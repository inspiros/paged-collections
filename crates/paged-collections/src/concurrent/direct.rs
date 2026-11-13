//! A direct-address map with one reader-writer lock per page.

use alloc::boxed::Box;
use core::fmt;
use core::mem;
use core::ops::{Bound, Deref, DerefMut, RangeBounds};

use super::atomic::{AtomicUsize, Ordering};
use super::directory::{SharedDirectory, SharedPages};
use super::lock::{ReadGuard, RwLock, WriteGuard};
use crate::direct_map::PagedDirectMap;
use crate::key::Key;
use crate::page::{DEFAULT_PAGE, Flat};

struct Page<K, V> {
    data: RwLock<PageData<K, V>>,
}

struct PageData<K, V> {
    slots: Box<[Option<(K, V)>]>,
    occupied: usize,
}

impl<K, V> PageData<K, V> {
    fn replace(&mut self, offset: usize, pair: (K, V), len: &AtomicUsize) -> Option<(K, V)> {
        let old = self.slots[offset].replace(pair);
        if old.is_none() {
            self.occupied += 1;
            len.fetch_add(1, Ordering::Relaxed);
        }
        old
    }

    fn remove(&mut self, offset: usize, len: &AtomicUsize) -> Option<(K, V)> {
        let old = self.slots[offset].take();
        if old.is_some() {
            self.occupied -= 1;
            len.fetch_sub(1, Ordering::Relaxed);
        }
        old
    }
}

/// A shared direct-address map with one reader-writer lock per page.
///
/// Values live at [`Key::index`]. Neighbouring keys on the same page contend;
/// operations on different pages lock independently. Pages remain allocated
/// until [`reset`](Self::reset) or destruction. The default [`Flat`] directory
/// has O(1) page lookups and memory following the highest page. Its finite
/// range makes out-of-range insertions panic; lookups and removals return
/// absent. [`Radix`](crate::page::Radix) accepts any `usize` index, uses directory
/// memory proportional to installed pages, and looks up pages in O(depth).
/// `PAGE` must be nonzero.
///
/// Holding a guard or entry and calling any method that locks the same page
/// from the same thread deadlocks (or panics on `std`). This also applies to
/// `make`, `with`, `retain`, formatting and traversal callbacks touching the
/// same map. Release guards before making another call on their page.
/// Holding locks on two pages at once can also deadlock across threads: if
/// one thread holds page A and waits for page B while another holds B and
/// waits for A (for example, two factories each looking up the other's key).
/// Nested calls must lock pages in one global order (such as ascending
/// [`Key::index`]), or callbacks must not touch the map at all.
///
/// Traversals visit ascending slot indices, holding one page lock at a time.
/// They are consistent per page only and weakly consistent across pages.
/// Length is exact when no write is in flight. The `no_std` spin lock permits
/// writer starvation; the `std` lock ignores poisoning because mutations keep
/// slots and counts valid even when user code panics.
///
/// ```
/// # #[cfg(feature = "concurrent")] {
/// use paged_collections::{concurrent::ConcurrentDirectMap, page::Radix};
/// let map = ConcurrentDirectMap::<usize, _, 4, Radix>::new();
/// assert_eq!(map.insert(usize::MAX - 1, "wide"), None);
/// assert_eq!(map.get_cloned(usize::MAX - 1), Some("wide"));
/// # }
/// ```
pub struct ConcurrentDirectMap<
    K: Key,
    V,
    const PAGE: usize = DEFAULT_PAGE,
    D: SharedDirectory = Flat,
> {
    // A GAT projection is invariant in K and V, as shared insertion requires.
    pages: D::Pages<Page<K, V>>,
    len: AtomicUsize,
}

// SAFETY: both sealed directories own stable pages and are Send when their
// pages are Send. Moving the map transfers its keys, values and page locks.
unsafe impl<K: Key + Send, V: Send, const PAGE: usize, D: SharedDirectory> Send
    for ConcurrentDirectMap<K, V, PAGE, D>
{
}
// SAFETY: both sealed directories publish pages safely; each page lock
// synchronizes access to its keys and values. Shared access cannot free pages.
unsafe impl<K: Key + Send + Sync, V: Send + Sync, const PAGE: usize, D: SharedDirectory> Sync
    for ConcurrentDirectMap<K, V, PAGE, D>
{
}

/// A value reference holding its page's read lock.
///
/// Calling another method that locks this page while holding the guard may
/// deadlock or panic. See [`ConcurrentDirectMap`].
pub struct DirectMapRef<'a, K, V> {
    guard: ReadGuard<'a, PageData<K, V>>,
    offset: usize,
}

/// A mutable value reference holding its page's write lock.
///
/// Calling another method that locks this page while holding the guard
/// deadlocks or panics. See [`ConcurrentDirectMap`].
pub struct DirectMapRefMut<'a, K, V> {
    guard: WriteGuard<'a, PageData<K, V>>,
    offset: usize,
}

/// A page-write-locked view of a key, occupied only by that key itself.
pub enum Entry<'a, K, V> {
    /// The requested key is present.
    Occupied(OccupiedEntry<'a, K, V>),
    /// The slot is empty or holds another generation.
    Vacant(VacantEntry<'a, K, V>),
}

/// An occupied entry holding its page's write lock.
pub struct OccupiedEntry<'a, K, V> {
    value: DirectMapRefMut<'a, K, V>,
    len: &'a AtomicUsize,
}

/// A vacant entry holding its page's write lock, possibly over another generation.
pub struct VacantEntry<'a, K, V> {
    key: K,
    value: DirectMapRefMut<'a, K, V>,
    len: &'a AtomicUsize,
}

impl<K: Key, V, const PAGE: usize, D: SharedDirectory> ConcurrentDirectMap<K, V, PAGE, D> {
    /// Number of slots protected by each page lock.
    pub const PAGE_SIZE: usize = PAGE;

    /// Creates an empty map without allocating.
    ///
    /// # Panics
    ///
    /// Panics if `PAGE` is zero or its slot storage exceeds `isize::MAX` bytes.
    pub fn new() -> Self {
        assert!(
            PAGE > 0 && PAGE <= isize::MAX as usize / mem::size_of::<Option<(K, V)>>(),
            "ConcurrentDirectMap: invalid page capacity"
        );
        Self {
            pages: D::Pages::<Page<K, V>>::new(),
            len: AtomicUsize::new(0),
        }
    }

    /// Number of entries; exact when no write is in flight.
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

    /// Whether `len()` is zero, subject to concurrent writes.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Counts installed pages by scanning the directory without page locks.
    /// Weakly consistent while pages are being installed.
    pub fn allocated_pages(&self) -> usize {
        self.pages.iter().count()
    }

    fn page(&self, index: usize) -> Option<&Page<K, V>> {
        self.pages
            .get(D::Pages::<Page<K, V>>::page_number::<PAGE>(index)?)
    }

    fn ensure_page(&self, index: usize) -> &Page<K, V> {
        let number = D::Pages::<Page<K, V>>::page_number::<PAGE>(index)
            .expect("ConcurrentDirectMap: key index exceeds direct spine range");
        self.pages.get_or_insert_with(number, || {
            Box::new(Page {
                data: RwLock::new(PageData {
                    slots: (0..PAGE).map(|_| None).collect(),
                    occupied: 0,
                }),
            })
        })
    }

    /// Whether the requested key itself is present.
    pub fn contains_key(&self, key: K) -> bool {
        self.get(key).is_some()
    }

    /// Acquires a read guard if this key itself is present.
    /// An out-of-range index returns `None`.
    pub fn get(&self, key: K) -> Option<DirectMapRef<'_, K, V>> {
        let index = key.index();
        let guard = self.page(index)?.data.read();
        let offset = index % PAGE;
        if !matches!(&guard.slots[offset], Some((stored, _)) if *stored == key) {
            return None;
        }
        Some(DirectMapRef { guard, offset })
    }

    /// Acquires a write guard if this key itself is present.
    /// An out-of-range index returns `None`.
    pub fn get_mut(&self, key: K) -> Option<DirectMapRefMut<'_, K, V>> {
        let index = key.index();
        let guard = self.page(index)?.data.write();
        let offset = index % PAGE;
        if !matches!(&guard.slots[offset], Some((stored, _)) if *stored == key) {
            return None;
        }
        Some(DirectMapRefMut { guard, offset })
    }

    /// Calls `f` under the page's read lock if this key is present.
    pub fn with<R>(&self, key: K, f: impl FnOnce(&V) -> R) -> Option<R> {
        self.get(key).map(|value| f(&value))
    }

    /// Clones a present value under its page's read lock.
    pub fn get_cloned(&self, key: K) -> Option<V>
    where
        V: Clone,
    {
        self.with(key, V::clone)
    }

    /// Inserts a value, returning the old value from this slot, if any.
    /// Keeps an equal stored key; evicts a different generation.
    /// Panics for indices outside the Flat spine range or allocation overflow.
    pub fn insert(&self, key: K, value: V) -> Option<V> {
        let index = key.index();
        let mut guard = self.ensure_page(index).data.write();
        let offset = index % PAGE;
        if let Some((stored, old)) = &mut guard.slots[offset] {
            if *stored != key {
                *stored = key;
            }
            let old = mem::replace(old, value);
            drop(guard);
            return Some(old);
        }
        let old = guard.replace(offset, (key, value), &self.len);
        drop(guard);
        old.map(|(_, value)| value)
    }

    /// Replaces this slot, always storing `key`, and returns its previous pair.
    /// Evicts another generation. With `Flat`, panics for an out-of-range insertion.
    pub fn replace(&self, key: K, value: V) -> Option<(K, V)> {
        let index = key.index();
        let mut guard = self.ensure_page(index).data.write();
        let old = guard.replace(index % PAGE, (key, value), &self.len);
        drop(guard);
        old
    }

    /// Inserts only into an empty slot; returns `value` if any generation holds it.
    /// With `Flat`, panics for an out-of-range insertion.
    pub fn try_insert(&self, key: K, value: V) -> Result<(), V> {
        let index = key.index();
        let mut guard = self.ensure_page(index).data.write();
        let offset = index % PAGE;
        if guard.slots[offset].is_some() {
            drop(guard);
            return Err(value);
        }
        let old = guard.replace(offset, (key, value), &self.len);
        drop(guard);
        drop(old);
        Ok(())
    }

    /// Removes this key itself, returning its value after releasing the lock.
    /// An out-of-range index returns `None`.
    pub fn remove(&self, key: K) -> Option<V> {
        self.remove_entry(key).map(|(_, value)| value)
    }

    /// Removes this key itself, returning its stored pair after releasing the lock.
    /// An out-of-range index returns `None`.
    pub fn remove_entry(&self, key: K) -> Option<(K, V)> {
        let index = key.index();
        let mut guard = self.page(index)?.data.write();
        let offset = index % PAGE;
        if !matches!(&guard.slots[offset], Some((stored, _)) if *stored == key) {
            return None;
        }
        let old = guard.remove(offset, &self.len);
        drop(guard);
        old
    }

    /// Removes whichever generation occupies `index`, releasing the lock first.
    /// An out-of-range index returns `None`.
    pub fn remove_occupant(&self, index: usize) -> Option<(K, V)> {
        let mut guard = self.page(index)?.data.write();
        let old = guard.remove(index % PAGE, &self.len);
        drop(guard);
        old
    }

    /// Returns this key's value, constructing it under the page write lock if absent.
    /// Another generation is evicted. A panicking factory leaves entries unchanged.
    /// At most one competing factory runs before the key is removed or evicted.
    /// With `Flat`, panics for an out-of-range insertion.
    pub fn get_or_insert_with(
        &self,
        key: K,
        make: impl FnOnce() -> V,
    ) -> DirectMapRefMut<'_, K, V> {
        self.entry(key).or_insert_with(make)
    }

    /// Locks the key's page for an entry operation, installing an empty page if needed.
    /// Another generation makes the entry vacant; inserting through it evicts that pair.
    /// With `Flat`, panics for an out-of-range index.
    pub fn entry(&self, key: K) -> Entry<'_, K, V> {
        let index = key.index();
        let guard = self.ensure_page(index).data.write();
        let offset = index % PAGE;
        let occupied = matches!(&guard.slots[offset], Some((stored, _)) if *stored == key);
        let value = DirectMapRefMut { guard, offset };
        if occupied {
            Entry::Occupied(OccupiedEntry {
                value,
                len: &self.len,
            })
        } else {
            Entry::Vacant(VacantEntry {
                key,
                value,
                len: &self.len,
            })
        }
    }

    /// Keeps entries for which `keep` returns true, in ascending slot order.
    /// Holds one page write lock at a time; atomic only within each page.
    /// Counts remain correct if the callback or a removed value's destructor panics.
    pub fn retain(&self, mut keep: impl FnMut(&K, &mut V) -> bool) {
        for (_, page) in self.pages.iter() {
            let mut guard = page.data.write();
            for offset in 0..guard.slots.len() {
                let remove = match &mut guard.slots[offset] {
                    Some((key, value)) => !keep(key, value),
                    None => false,
                };
                if remove {
                    let old = guard.remove(offset, &self.len);
                    drop(old);
                }
            }
        }
    }

    /// Calls `f` in ascending slot order, holding one page read lock at a time.
    /// Consistent per page only, weakly consistent across pages.
    pub fn for_each(&self, f: impl FnMut(&K, &V)) {
        self.range_for_each(.., f);
    }

    /// Calls `f` for a range bounded by [`Key::index`], in ascending slot order.
    /// Empty or inverted ranges visit nothing. Holds one page read lock at a
    /// time; consistent per page only, weakly consistent across pages.
    pub fn range_for_each(&self, keys: impl RangeBounds<K>, mut f: impl FnMut(&K, &V)) {
        let start = match keys.start_bound() {
            Bound::Included(key) => key.index(),
            Bound::Excluded(key) => match key.index().checked_add(1) {
                Some(index) => index,
                None => return,
            },
            Bound::Unbounded => 0,
        };
        let end = match keys.end_bound() {
            Bound::Included(key) => key.index(),
            Bound::Excluded(key) => match key.index().checked_sub(1) {
                Some(index) => index,
                None => return,
            },
            Bound::Unbounded => usize::MAX,
        };
        if start > end {
            return;
        }
        for (number, page) in self.pages.iter() {
            if number < start / PAGE {
                continue;
            }
            if number > end / PAGE {
                break;
            }
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
            let guard = page.data.read();
            for (key, value) in guard.slots[first..last].iter().flatten() {
                f(key, value);
            }
        }
    }

    /// Iterates exclusively in ascending slot order without acquiring page locks.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&K, &mut V)> {
        self.pages.iter_mut().flat_map(|(_, page)| {
            page.data
                .get_mut()
                .slots
                .iter_mut()
                .filter_map(|slot| slot.as_mut().map(|(key, value)| (&*key, value)))
        })
    }

    /// Removes entries without locking, retaining pages for reuse.
    /// If a destructor panics, unvisited entries remain and counts stay correct.
    pub fn clear(&mut self) {
        for (_, page) in self.pages.iter_mut() {
            let data = page.data.get_mut();
            for offset in 0..data.slots.len() {
                let old = data.remove(offset, &self.len);
                drop(old);
            }
        }
    }

    /// Detaches and frees all pages. The map is empty even if a destructor panics.
    pub fn reset(&mut self) {
        let pages = mem::replace(&mut self.pages, D::Pages::<Page<K, V>>::new());
        self.len.store(0, Ordering::Relaxed);
        drop(pages);
    }

    /// Moves entries into a plain map in slot order, scanning allocated slots.
    /// Each occupied entry is moved once; no values are cloned.
    pub fn into_inner(mut self) -> PagedDirectMap<K, V, PAGE, D> {
        let mut map = PagedDirectMap::new();
        for (_, page) in self.pages.iter_mut() {
            let lock = mem::replace(
                &mut page.data,
                RwLock::new(PageData {
                    slots: Box::default(),
                    occupied: 0,
                }),
            );
            for (key, value) in lock.into_inner().slots.into_vec().into_iter().flatten() {
                map.insert(key, value);
            }
        }
        map
    }
}

impl<K, V> DirectMapRef<'_, K, V> {
    /// The stored key, which may differ in representation from an equal query key.
    pub fn key(&self) -> &K {
        &self.guard.slots[self.offset].as_ref().unwrap().0
    }
}

impl<K, V> Deref for DirectMapRef<'_, K, V> {
    type Target = V;

    fn deref(&self) -> &V {
        &self.guard.slots[self.offset].as_ref().unwrap().1
    }
}

impl<K, V> DirectMapRefMut<'_, K, V> {
    /// The stored key, which cannot be changed through the guard.
    pub fn key(&self) -> &K {
        &self.guard.slots[self.offset].as_ref().unwrap().0
    }
}

impl<K, V> Deref for DirectMapRefMut<'_, K, V> {
    type Target = V;

    fn deref(&self) -> &V {
        &self.guard.slots[self.offset].as_ref().unwrap().1
    }
}

impl<K, V> DerefMut for DirectMapRefMut<'_, K, V> {
    fn deref_mut(&mut self) -> &mut V {
        &mut self.guard.slots[self.offset].as_mut().unwrap().1
    }
}

impl<'a, K, V> Entry<'a, K, V> {
    /// The occupied stored key, or the query key for a vacant entry.
    pub fn key(&self) -> &K {
        match self {
            Self::Occupied(entry) => entry.key(),
            Self::Vacant(entry) => entry.key(),
        }
    }

    /// Returns the value, inserting `default` if this key is absent.
    pub fn or_insert(self, default: V) -> DirectMapRefMut<'a, K, V> {
        match self {
            Self::Occupied(entry) => entry.into_ref(),
            Self::Vacant(entry) => entry.insert(default),
        }
    }

    /// Returns the value, calling `make` under the page write lock only if absent.
    /// A panicking factory leaves the slot unchanged.
    pub fn or_insert_with(self, make: impl FnOnce() -> V) -> DirectMapRefMut<'a, K, V> {
        match self {
            Self::Occupied(entry) => entry.into_ref(),
            Self::Vacant(entry) => {
                let value = make();
                entry.insert(value)
            }
        }
    }

    /// Like [`or_insert_with`](Self::or_insert_with), passing the query key to `make`.
    pub fn or_insert_with_key(self, make: impl FnOnce(&K) -> V) -> DirectMapRefMut<'a, K, V> {
        match self {
            Self::Occupied(entry) => entry.into_ref(),
            Self::Vacant(entry) => {
                let value = make(entry.key());
                entry.insert(value)
            }
        }
    }

    /// Returns the value, inserting its default under the lock if absent.
    pub fn or_default(self) -> DirectMapRefMut<'a, K, V>
    where
        V: Default,
    {
        self.or_insert_with(V::default)
    }

    /// Modifies an occupied value and returns this entry. A panic keeps it occupied.
    pub fn and_modify(mut self, f: impl FnOnce(&mut V)) -> Self {
        if let Self::Occupied(entry) = &mut self {
            f(entry.get_mut());
        }
        self
    }
}

impl<'a, K, V> OccupiedEntry<'a, K, V> {
    /// The stored key.
    pub fn key(&self) -> &K {
        self.value.key()
    }

    /// The stored value.
    pub fn get(&self) -> &V {
        &self.value
    }

    /// Mutable access to the value under the page write lock.
    pub fn get_mut(&mut self) -> &mut V {
        &mut self.value
    }

    /// Converts the entry into a mutable guard, retaining the page write lock.
    pub fn into_ref(self) -> DirectMapRefMut<'a, K, V> {
        self.value
    }

    /// Replaces only the value, returning the previous value and keeping the stored key.
    pub fn insert(&mut self, value: V) -> V {
        mem::replace(self.get_mut(), value)
    }

    /// Removes the pair and returns its value after releasing the page lock.
    pub fn remove(self) -> V {
        self.remove_entry().1
    }

    /// Removes the pair and returns it after releasing the page lock.
    pub fn remove_entry(mut self) -> (K, V) {
        let offset = self.value.offset;
        let old = self.value.guard.remove(offset, self.len).unwrap();
        drop(self);
        old
    }
}

impl<'a, K, V> VacantEntry<'a, K, V> {
    /// The query key, which may share its slot with another generation.
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Releases the lock and returns the query key without changing the slot.
    pub fn into_key(self) -> K {
        self.key
    }

    /// The pair from another generation that insertion would evict.
    pub fn occupant(&self) -> Option<(&K, &V)> {
        self.value.guard.slots[self.value.offset]
            .as_ref()
            .map(|(key, value)| (key, value))
    }

    /// Inserts this key and returns a mutable guard, evicting another generation.
    /// The slot and counts are consistent before an evicted value is dropped.
    pub fn insert(mut self, value: V) -> DirectMapRefMut<'a, K, V> {
        let offset = self.value.offset;
        let old = self
            .value
            .guard
            .replace(offset, (self.key, value), self.len);
        drop(old);
        self.value
    }
}

impl<K: Key, V, const PAGE: usize, D: SharedDirectory> Default
    for ConcurrentDirectMap<K, V, PAGE, D>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Key + fmt::Debug, V: fmt::Debug, const PAGE: usize, D: SharedDirectory> fmt::Debug
    for ConcurrentDirectMap<K, V, PAGE, D>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut map = f.debug_map();
        self.for_each(|key, value| {
            map.entry(key, value);
        });
        map.finish()
    }
}

impl<K: Key, V, const PAGE: usize, D: SharedDirectory> FromIterator<(K, V)>
    for ConcurrentDirectMap<K, V, PAGE, D>
{
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut map = Self::new();
        map.extend(iter);
        map
    }
}

impl<K: Key, V, const PAGE: usize, D: SharedDirectory> Extend<(K, V)>
    for ConcurrentDirectMap<K, V, PAGE, D>
{
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        for (key, value) in iter {
            self.insert(key, value);
        }
    }
}

impl<K: Key, V, const PAGE: usize, D: SharedDirectory> From<PagedDirectMap<K, V, PAGE, D>>
    for ConcurrentDirectMap<K, V, PAGE, D>
{
    fn from(map: PagedDirectMap<K, V, PAGE, D>) -> Self {
        map.into_iter().collect()
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::ConcurrentDirectMap;
    use alloc::string::String;

    #[test]
    fn map_is_send_and_sync() {
        fn assert_traits<T: Send + Sync>() {}
        assert_traits::<ConcurrentDirectMap<u32, String>>();
    }

    #[test]
    #[should_panic(expected = "invalid page capacity")]
    fn zero_page_is_rejected() {
        let _ = ConcurrentDirectMap::<usize, (), 0>::new();
    }

    #[test]
    #[should_panic(expected = "key index exceeds direct spine range")]
    fn out_of_range_insertion_panics() {
        ConcurrentDirectMap::<usize, (), 1>::new().insert(usize::MAX, ());
    }

    #[test]
    fn out_of_range_lookups_are_absent() {
        let map = ConcurrentDirectMap::<usize, (), 1>::new();
        assert!(!map.contains_key(usize::MAX));
        assert!(map.get(usize::MAX).is_none());
        assert!(map.get_mut(usize::MAX).is_none());
        assert_eq!(map.with(usize::MAX, |_| ()), None);
        assert_eq!(map.get_cloned(usize::MAX), None);
        assert_eq!(map.remove(usize::MAX), None);
        assert_eq!(map.remove_entry(usize::MAX), None);
        assert_eq!(map.remove_occupant(usize::MAX), None);
        assert_eq!(map.allocated_pages(), 0);
    }
}
