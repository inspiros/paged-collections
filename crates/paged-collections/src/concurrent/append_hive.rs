//! Stable value cells with shared insertion and exclusive removal.
//!
//! The low 32 state bits are the id version. BUSY, in bit 32, owns the cell
//! exclusively while inserting. A published live state changes only under
//! exclusive access, so one Acquire state load permits an ordinary `&T`.
//! Pages do not drop values: an exclusive hive scan does that.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::fmt;
use core::iter::FusedIterator;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ops::Index;
use core::ptr;

use super::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering, UnsafeCell};
use super::free_stack::FreeStack;
use super::recycle::PageStack;
use super::spine::Spine;
use crate::hive::{Hive, RawPage, RawSlot};
use crate::id_alloc::{Id, MAX_INDICES, generation_of, live_version};
use crate::page::DEFAULT_PAGE;

#[cfg(not(loom))]
const CHUNKS: usize = 28;
#[cfg(loom)]
const CHUNKS: usize = 32;

const VERSION: u64 = u32::MAX as u64;
const BUSY: u64 = 1 << 32;

struct Slot<T> {
    state: AtomicU64,
    next: AtomicU32,
    value: UnsafeCell<MaybeUninit<T>>,
}

impl<T> Slot<T> {
    fn new() -> Self {
        Self::with_state(0)
    }

    /// A vacant slot at `version` (even), for a recycled page.
    fn with_state(version: u32) -> Self {
        Self {
            state: AtomicU64::new(u64::from(version)),
            next: AtomicU32::new(0),
            value: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }

    // Called only under an exclusive hive borrow (including exclusive
    // iterator construction and traversal). No insertion can be in flight.
    fn exclusive_version(&self) -> u32 {
        // Relaxed suffices: exclusivity rules out concurrent publication.
        let word = self.state.load(Ordering::Relaxed);
        debug_assert_eq!(word & !VERSION, 0, "exclusive slot is BUSY");
        word as u32
    }
}

struct Page<T> {
    // MaybeUninit deliberately has no value destructor. The hive scans cells.
    slots: Box<[Slot<T>]>,
}

fn vacant(live: u32) -> u64 {
    u64::from(live.checked_add(1).unwrap_or(0))
}

/// A paged container with stable addresses and shared insertion.
///
/// [`get`](Self::get) returns an ordinary `&T` after one Acquire slot-state
/// load, with no guard or reference count. Published values can be changed,
/// removed or moved only under `&mut self`, which excludes every reference
/// borrowed from shared access. Prefer [`ConcurrentHive`](super::ConcurrentHive)
/// when removal must also be shared.
///
/// Pages stay allocated until [`shrink_to_fit`](Self::shrink_to_fit) frees
/// the empty ones; shared inserts reuse freed pages later. Shared
/// iteration is weakly consistent and may miss insertions in flight. `len`
/// is approximate during insertion and exact when insertions have finished.
/// Exclusive scans and conversions take O([`index_bound`](Self::index_bound))
/// time. Conversions to and from [`Hive`] preserve live ids and stale-id
/// history, but may change allocation and slot reuse order.
///
/// As with [`Hive`], ids carry no container identity, generations never wrap,
/// and stale ids stay dead within one allocation history. A panicking
/// [`insert_with`](Self::insert_with) spends the id shown to its factory.
///
/// `PAGE` must be non-zero, fit in a `u32`, and fit a page in `isize::MAX`
/// bytes. Uses only `core` and `alloc`.
///
/// ```
/// use paged_collections::concurrent::ConcurrentAppendHive;
///
/// let mut hive = ConcurrentAppendHive::<&str, 4>::new();
/// let first = hive.insert("first");
/// let value = hive.get(first).unwrap();
/// let second = hive.insert("second");
/// assert_eq!(*value, "first");
/// assert_eq!(hive[second], "second");
/// assert_eq!(hive.remove(first), Some("first"));
/// assert!(hive.get(first).is_none());
/// ```
///
/// Sharing requires both `Send` and `Sync` elements:
///
/// ```compile_fail
/// use paged_collections::concurrent::ConcurrentAppendHive;
/// use std::rc::Rc;
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<ConcurrentAppendHive<Rc<()>>>();
/// ```
pub struct ConcurrentAppendHive<T, const PAGE: usize = DEFAULT_PAGE> {
    pages: Spine<Page<T>, CHUNKS>,
    free: FreeStack,
    /// Pages freed by `shrink_to_fit`, below the reservation bound.
    recycled: PageStack,
    len: AtomicUsize,
}

// SAFETY: moving the hive transfers its owned cells; no borrowed reference
// outlives it. Exclusive removal or destruction may drop T on the receiver.
unsafe impl<T: Send, const PAGE: usize> Send for ConcurrentAppendHive<T, PAGE> {}
// SAFETY: the claim CAS grants exclusive cell ownership until publication;
// published T is shared immutably. Shared insertion also transfers T between
// threads, so both Send and Sync are required.
unsafe impl<T: Send + Sync, const PAGE: usize> Sync for ConcurrentAppendHive<T, PAGE> {}

/// Owns an uninitialized cell already claimed by CAS. Unwinding spends the
/// factory's id and publishes a reusable vacancy, or retires the slot.
struct Busy<'a, T, const PAGE: usize> {
    hive: &'a ConcurrentAppendHive<T, PAGE>,
    slot: &'a Slot<T>,
    index: u32,
    vacant: u64,
    armed: bool,
}

impl<T, const PAGE: usize> Drop for Busy<'_, T, PAGE> {
    fn drop(&mut self) {
        if self.armed {
            // Release publishes the end of all factory/cell access before
            // reuse. A claim acquires this even with a stale stack hint.
            self.slot.state.store(self.vacant, Ordering::Release);
            if self.vacant != 0 {
                self.hive.free.push(self.index, &self.slot.next);
            }
        }
    }
}

impl<T, const PAGE: usize> ConcurrentAppendHive<T, PAGE> {
    /// Number of slots per page.
    pub const PAGE: usize = PAGE;

    /// An empty hive, without allocating.
    ///
    /// # Panics
    ///
    /// Panics if the page capacity is invalid; see the type's documentation.
    pub fn new() -> Self {
        assert!(
            PAGE > 0
                && PAGE <= u32::MAX as usize
                && PAGE <= isize::MAX as usize / core::mem::size_of::<Slot<T>>(),
            "ConcurrentAppendHive: invalid page capacity"
        );
        Self {
            pages: Spine::new(),
            free: FreeStack::new(),
            recycled: PageStack::new(),
            len: AtomicUsize::new(0),
        }
    }

    fn slot(&self, index: u32) -> Option<&Slot<T>> {
        if index == MAX_INDICES {
            return None;
        }
        self.pages
            .get(index as usize / PAGE)
            .map(|page| &page.slots[index as usize % PAGE])
    }

    fn ensure_slot(&self, index: u32) -> &Slot<T> {
        let page = self.pages.get_or_insert_with(index as usize / PAGE, || {
            Box::new(Page {
                slots: (0..PAGE).map(|_| Slot::new()).collect(),
            })
        });
        &page.slots[index as usize % PAGE]
    }

    /// Reinstalls a page freed by `shrink_to_fit`, with every usable slot
    /// vacant at the page's floor. Pushes all its slots but the first to the
    /// free stack and returns the first, for the caller to claim by CAS.
    ///
    /// The popper is the page's only installer: the free stack holds none of
    /// its indices (it was rebuilt when the page was freed) and fresh
    /// reservation never goes below the bound. Its slots are unreachable
    /// until pushed, so this thread owns them while pushing.
    fn recycle(&self) -> Option<u32> {
        let (page, floor) = self.recycled.pop()?;
        let base = page * PAGE;
        let limit = PAGE.min(MAX_INDICES as usize - base);
        let installed = self.pages.get_or_insert_with(page, || {
            Box::new(Page {
                slots: (0..PAGE)
                    .map(|offset| Slot::with_state(if offset < limit { floor } else { 0 }))
                    .collect(),
            })
        });
        for offset in (1..limit).rev() {
            self.free
                .push((base + offset) as u32, &installed.slots[offset].next);
        }
        Some(base as u32)
    }

    fn claim(&self) -> Option<(Busy<'_, T, PAGE>, u32)> {
        loop {
            let popped = self
                .free
                .pop(|index| &self.slot(index).expect("free index has a page").next);
            let reused = popped.or_else(|| self.recycle());
            let index = match reused {
                Some(index) => index,
                None => self.free.fresh(MAX_INDICES)?,
            };
            let slot = self.ensure_slot(index);
            // Acquire observes the completed vacancy, independently of the
            // free-stack hint. Only even versions without BUSY may be claimed.
            let old = slot.state.load(Ordering::Acquire);
            if old & !VERSION != 0 || old & 1 != 0 || (reused.is_some() && old == 0) {
                continue;
            }
            // AcqRel claims sole ownership and observes the preceding vacant
            // Release. A failed CAS discards the hint, never retries that word.
            // Fresh zero is claimable only through its unique reservation;
            // zero popped from the stack is retired and was rejected above.
            if slot
                .state
                .compare_exchange(old, old | BUSY, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                continue;
            }
            let live = old as u32 + 1;
            return Some((
                Busy {
                    hive: self,
                    slot,
                    index,
                    vacant: vacant(live),
                    armed: true,
                },
                live,
            ));
        }
    }

    // The caller owns this empty cell through a claim or exclusive conversion.
    fn write_value(&self, slot: &Slot<T>, live: u32, value: T) {
        slot.value.with_mut(|cell| {
            // SAFETY: the claim CAS or exclusive construction grants sole
            // ownership of this uninitialized cell. MaybeUninit has T's layout.
            unsafe { ptr::write(cell.cast::<T>(), value) };
        });
        // Relaxed accounting precedes publication; len is exact once no
        // insertion is in flight, but does not itself grant value access.
        self.len.fetch_add(1, Ordering::Relaxed);
        // Release publishes the initialized value to matching Acquire reads.
        slot.state.store(u64::from(live), Ordering::Release);
    }

    fn publish(&self, mut owner: Busy<'_, T, PAGE>, live: u32, value: T) -> Id {
        self.write_value(owner.slot, live, value);
        owner.armed = false;
        Id::new(owner.index, generation_of(live))
    }

    /// Inserts `value` and returns its id.
    ///
    /// # Panics
    ///
    /// Panics if all [`MAX_INDICES`] indices are occupied or retired, or a
    /// page cannot be allocated. A panicking concurrent factory may make room.
    pub fn insert(&self, value: T) -> Id {
        self.try_insert(value).unwrap_or_else(|_| {
            panic!("ConcurrentAppendHive: all {MAX_INDICES} slot indices are occupied or retired")
        })
    }

    /// Like [`insert`](Self::insert), returning the value on index exhaustion.
    /// Allocation failures may still panic or abort.
    pub fn try_insert(&self, value: T) -> Result<Id, T> {
        let Some((owner, live)) = self.claim() else {
            return Err(value);
        };
        Ok(self.publish(owner, live, value))
    }

    /// Inserts the value `f` makes from its id.
    ///
    /// If `f` panics, its id's generation is spent and that id stays dead.
    /// The slot becomes reusable unless its last generation was spent.
    ///
    /// # Panics
    ///
    /// Panics on index exhaustion or if `f` panics; allocation may also fail.
    pub fn insert_with(&self, f: impl FnOnce(Id) -> T) -> Id {
        self.try_insert_with(f).unwrap_or_else(|| {
            panic!("ConcurrentAppendHive: all {MAX_INDICES} slot indices are occupied or retired")
        })
    }

    /// Like [`insert_with`](Self::insert_with), returning `None` without
    /// calling `f` when no index is available.
    pub fn try_insert_with(&self, f: impl FnOnce(Id) -> T) -> Option<Id> {
        let (owner, live) = self.claim()?;
        let value = f(Id::new(owner.index, generation_of(live)));
        Some(self.publish(owner, live, value))
    }

    /// The value of a matching live id, without a guard or reference count.
    /// Returns `None` for stale, invalid or unpublished ids.
    pub fn get(&self, id: Id) -> Option<&T> {
        let live = live_version(id.generation())?;
        let slot = self.slot(id.index())?;
        // Acquire observes the initialized value; exact equality rejects BUSY.
        if slot.state.load(Ordering::Acquire) != u64::from(live) {
            return None;
        }
        Some(slot.value.with(|cell| {
            // SAFETY: the matching published live state changes only under
            // &mut self, which cannot coexist with this shared borrow. The
            // value remains initialized and immovable for the returned lifetime.
            unsafe { &*cell.cast::<T>() }
        }))
    }

    /// The id of the element at `value`'s address, if `value` is a live
    /// element of this hive: the inverse of [`get`](Self::get).
    ///
    /// Takes time linear in the directory's page cells up to the highest
    /// installed page (with `Flat`, O(highest page number)): it compares the
    /// address with each installed page's slot range, then checks that the slot found is published live
    /// and holds exactly that element. Anything else gives `None`. Uses
    /// address comparison only; no pointer is dereferenced.
    ///
    /// ```
    /// use paged_collections::concurrent::ConcurrentAppendHive;
    ///
    /// let hive = ConcurrentAppendHive::<u32, 4>::new();
    /// let id = hive.insert(7);
    /// let element = hive.get(id).unwrap();
    /// assert_eq!(hive.id_of(element), Some(id));
    /// assert_eq!(hive.id_of(&7), None);
    /// ```
    pub fn id_of(&self, value: &T) -> Option<Id> {
        let target = ptr::from_ref(value);
        let size = core::mem::size_of::<Slot<T>>();
        for (number, page) in self.pages.iter() {
            let start = page.slots.as_ptr().addr();
            let Some(offset) = target.addr().checked_sub(start).map(|bytes| bytes / size) else {
                continue;
            };
            let Some(slot) = page.slots.get(offset) else {
                continue;
            };
            if !slot.value.with(|cell| ptr::eq(cell.cast::<T>(), target)) {
                return None;
            }
            // Acquire observes publication; only an exact odd state is live.
            let word = slot.state.load(Ordering::Acquire);
            return (word & 1 == 1 && word & !VERSION == 0)
                .then(|| Id::new((number * PAGE + offset) as u32, generation_of(word as u32)));
        }
        None
    }

    /// Whether this id is live at the instant of one state load.
    pub fn contains(&self, id: Id) -> bool {
        let Some(live) = live_version(id.generation()) else {
            return false;
        };
        // Acquire observes publication; exact equality excludes BUSY.
        self.slot(id.index())
            .is_some_and(|slot| slot.state.load(Ordering::Acquire) == u64::from(live))
    }

    /// Number of live values; approximate during shared insertion.
    /// Exact when no insertion is in flight; this is no publication barrier.
    pub fn len(&self) -> usize {
        // Relaxed: accounting does not grant page or value access.
        self.len.load(Ordering::Relaxed)
    }

    /// Whether the hive is empty; approximate during shared insertion.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// One past the highest reserved index, including insertions in flight.
    /// A reserved index's page may not have been installed yet.
    pub fn index_bound(&self) -> u32 {
        self.free.index_bound()
    }

    /// Iterates over `(id, &value)` in slot order, weakly consistently.
    /// Scans only indices below the reservation bound at iterator creation;
    /// may miss insertions in flight, but never yields an unpublished value.
    pub fn iter(&self) -> AppendIter<'_, T, PAGE> {
        AppendIter {
            hive: self,
            indices: 0..self.index_bound(),
        }
    }

    /// The matching live value, mutably, under exclusive access.
    pub fn get_mut(&mut self, id: Id) -> Option<&mut T> {
        let live = live_version(id.generation())?;
        let slot = self.slot(id.index())?;
        if slot.exclusive_version() != live {
            return None;
        }
        Some(slot.value.with_mut(|cell| {
            // SAFETY: &mut self excludes all other accesses; the matching
            // live state proves the cell holds an initialized T.
            unsafe { &mut *cell.cast::<T>() }
        }))
    }

    /// Removes and returns a matching live value under exclusive access.
    /// The id stays dead; its slot is reused with a new generation or retired.
    pub fn remove(&mut self, id: Id) -> Option<T> {
        let live = live_version(id.generation())?;
        let slot = self.slot(id.index())?;
        if slot.exclusive_version() != live {
            return None;
        }
        let value = slot.value.with_mut(|cell| {
            // SAFETY: exclusive access excludes all borrowed values and
            // inserters; this matching live value is moved out exactly once.
            unsafe { ptr::read(cell.cast::<T>()) }
        });
        let vacant = vacant(live);
        // Release records the empty cell before any future shared claim.
        slot.state.store(vacant, Ordering::Release);
        if vacant != 0 {
            self.free.push(id.index(), &slot.next);
        }
        // Relaxed accounting completes before user destruction can run.
        self.len.fetch_sub(1, Ordering::Relaxed);
        Some(value)
    }

    /// Iterates over live `(id, &mut value)` pairs in slot order.
    pub fn iter_mut(&mut self) -> AppendIterMut<'_, T, PAGE> {
        AppendIterMut {
            remaining: self.len(),
            indices: 0..self.index_bound(),
            hive: self,
            exclusive: PhantomData,
        }
    }

    /// Eagerly removes every live value, yielding owned `(id, value)` pairs
    /// in slot order. Dropping or forgetting the iterator leaves the hive
    /// empty. Pages and generation history stay allocated.
    pub fn drain(&mut self) -> AppendDrain<'_, T, PAGE> {
        let mut values = Vec::with_capacity(self.len());
        for index in 0..self.index_bound() {
            let Some(slot) = self.slot(index) else {
                continue;
            };
            let version = slot.exclusive_version();
            if version & 1 != 0 {
                let id = Id::new(index, generation_of(version));
                values.push((id, self.remove(id).expect("exclusive live value")));
            }
        }
        AppendDrain {
            inner: values.into_iter(),
            exclusive: PhantomData,
        }
    }

    /// Keeps values for which `keep(id, value)` returns true, in slot order.
    /// If the predicate or a destructor panics, prior changes remain and
    /// `len` stays exact.
    pub fn retain(&mut self, mut keep: impl FnMut(Id, &mut T) -> bool) {
        for index in 0..self.index_bound() {
            let Some(slot) = self.slot(index) else {
                continue;
            };
            let version = slot.exclusive_version();
            if version & 1 == 0 {
                continue;
            }
            let id = Id::new(index, generation_of(version));
            if !keep(id, self.get_mut(id).expect("exclusive live value")) {
                drop(self.remove(id));
            }
        }
    }

    /// Removes and drops every value, retaining pages and generation history.
    pub fn clear(&mut self) {
        self.retain(|_, _| false);
    }

    /// Frees every page that holds no value, under exclusive access. Takes
    /// O([`index_bound`](Self::index_bound)) time.
    ///
    /// A page is freed if it lies wholly below the reservation bound and all
    /// its slots are vacant and reusable (a page with a retired slot is
    /// kept). It remembers its highest version, so ids it issued stay dead:
    /// shared inserts reuse freed pages, lowest first, after the free slots
    /// of the remaining pages, with generations above every earlier one.
    /// That order is exact only for one inserter: while one insert is
    /// relinking a reused page, a concurrent insert can take a fresh index
    /// instead (and fail at the index limit even though slots are about to
    /// be freed). The free slots are relinked in index order. The page directory's own
    /// small arrays stay allocated.
    ///
    /// ```
    /// use paged_collections::concurrent::ConcurrentAppendHive;
    ///
    /// let mut hive = ConcurrentAppendHive::<u32, 4>::new();
    /// let ids: Vec<_> = (0..12).map(|v| hive.insert(v)).collect();
    /// for &id in &ids[..8] {
    ///     hive.remove(id);
    /// }
    /// hive.shrink_to_fit(); // frees the first two pages
    /// assert_eq!(hive[ids[8]], 8);
    /// let new = hive.insert(99); // reuses a freed page
    /// assert!(new.index() < 8 && !ids.contains(&new));
    /// assert!(hive.get(ids[0]).is_none());
    /// ```
    pub fn shrink_to_fit(&mut self) {
        let bound = self.index_bound() as usize;
        let mut freed = self.recycled.pending().to_vec();
        let installed: Vec<usize> = self.pages.iter().map(|(number, _)| number).collect();
        for number in installed {
            let base = number * PAGE;
            let end = (base + PAGE).min(MAX_INDICES as usize);
            if end > bound {
                continue;
            }
            let page = self.pages.get(number).expect("page was just listed");
            let mut floor = 0;
            let empty = page.slots[..end - base].iter().all(|slot| {
                let version = slot.exclusive_version();
                floor = floor.max(version);
                // Live (odd) or retired (zero below the bound) slots stay.
                version & 1 == 0 && version != 0
            });
            if empty {
                // The page's cells are all vacant: dropping it drops no T.
                drop(self.pages.remove(number));
                freed.push((number, floor));
            }
        }
        // The free stack linked through freed slots: relink the rest,
        // pushed from the top so the lowest index pops first.
        self.free.reset();
        for index in (0..bound as u32).rev() {
            if let Some(slot) = self.slot(index) {
                let version = slot.exclusive_version();
                if version & 1 == 0 && version != 0 {
                    self.free.push(index, &slot.next);
                }
            }
        }
        self.recycled.set(freed);
    }

    /// Moves into a plain [`Hive`], preserving live ids and stale-id history.
    /// Takes O([`index_bound`](Self::index_bound)) time.
    ///
    /// Every slot below the bound is represented explicitly: zero states
    /// become retired vacancies, even versions become reusable vacancies,
    /// and odd versions keep their values. A page freed by
    /// [`shrink_to_fit`](Self::shrink_to_fit) becomes an empty raw page with
    /// its floor; other floors are zero. Offsets beyond the bound have never
    /// issued ids and need no history.
    ///
    /// Raw-page validation cannot fail by construction: all versions have
    /// the required parity, all pages fit `PAGE`, and indices are below
    /// [`MAX_INDICES`]. Allocation may still panic or abort. Values already
    /// moved are owned by the raw pages; this hive's empty cells cannot drop
    /// them again, including if allocation unwinds.
    pub fn into_hive(mut self) -> Hive<T, PAGE> {
        let bound = self.index_bound() as usize;
        let mut pages = Vec::with_capacity(bound.div_ceil(PAGE));
        for base in (0..bound).step_by(PAGE) {
            if let Some(floor) = self.recycled.floor_of(base / PAGE) {
                // A freed page: every slot restarts above its floor.
                pages.push(RawPage {
                    floor,
                    slots: Vec::new(),
                });
                continue;
            }
            let end = base + PAGE.min(bound - base);
            let mut slots = Vec::with_capacity(end - base);
            for index in base..end {
                let version = self.slot(index as u32).map_or(0, Slot::exclusive_version);
                slots.push(if version & 1 != 0 {
                    let id = Id::new(index as u32, generation_of(version));
                    RawSlot::Occupied {
                        version,
                        value: self.remove(id).expect("exclusive live value"),
                    }
                } else {
                    RawSlot::Vacant { version }
                });
            }
            pages.push(RawPage { floor: 0, slots });
        }
        Hive::try_from_raw_pages(pages).expect("append hive raw pages are valid by construction")
    }

    #[cfg(all(test, loom))]
    pub(super) fn prepare_fresh_page(&self) {
        // Install before the publication litmus starts: otherwise the page
        // directory's Release could hide a missing slot-state Release.
        self.ensure_slot(0);
    }

    #[cfg(all(test, loom))]
    pub(super) fn inject_busy_hint(&self, index: u32) {
        let slot = self.slot(index).unwrap();
        assert_ne!(slot.state.load(Ordering::Acquire) & BUSY, 0);
        // Models a stale link installed after tag wrap, without 2^32 pops.
        self.free.push(index, &slot.next);
    }
}

impl<T, const PAGE: usize> Default for ConcurrentAppendHive<T, PAGE> {
    fn default() -> Self {
        Self::new()
    }
}

/// Moves a plain hive into shared append storage without changing live ids.
/// Takes O(the source hive's index bound) time; pages and reuse order may change.
///
/// The new reservation bound ends after the last explicit raw slot or the
/// last page with a nonzero floor, whichever is later. The untouched suffix
/// of floor-zero pages is left to fresh reservation. Omitted offsets below
/// this bound become reusable vacancies at their page floor. For floor-zero
/// holes only, generation 1 is spent and the vacancy starts at version 2:
/// a popped zero state must always mean retired. Nonzero floors retain their
/// exact next generation, `floor / 2 + 1`. Explicit retired zeros are never
/// pushed. Thus no previously issued dead id can revive.
///
/// Versions are copied before draining values, since draining advances the
/// source versions. Occupied states are published only after their values
/// have moved; unwinding during allocation cannot expose uninitialized cells
/// to this hive's destructor.
impl<T, const PAGE: usize> From<Hive<T, PAGE>> for ConcurrentAppendHive<T, PAGE> {
    fn from(mut hive: Hive<T, PAGE>) -> Self {
        let mut out = Self::new();
        // Copy only floors and versions; no borrowed value escapes raw_pages.
        let metadata: Vec<(u32, Vec<u32>)> = hive
            .raw_pages()
            .map(|raw| {
                let versions = raw
                    .slots
                    .into_iter()
                    .map(|slot| match slot {
                        RawSlot::Occupied { version, .. } | RawSlot::Vacant { version } => version,
                    })
                    .collect();
                (raw.floor, versions)
            })
            .collect();
        let mut bound = 0;
        for (page, (floor, versions)) in metadata.iter().enumerate() {
            let base = page * PAGE;
            if *floor != 0 {
                // Include the entire usable page: omitted offsets inherit
                // this floor even when no raw slot remains after shrinking.
                bound = base + PAGE.min(MAX_INDICES as usize - base);
            } else if !versions.is_empty() {
                bound = base + versions.len();
            }
        }
        out.free.set_next_fresh(bound as u32);
        for index in 0..bound {
            let (floor, versions) = &metadata[index / PAGE];
            let version = versions
                .get(index % PAGE)
                .copied()
                .unwrap_or((*floor).max(2));
            let slot = out.ensure_slot(index as u32);
            if version & 1 == 0 {
                // Relaxed suffices during exclusive construction. Nonzero
                // vacancies are reusable; explicit zero states stay retired.
                slot.state.store(u64::from(version), Ordering::Relaxed);
                if version != 0 {
                    out.free.push(index as u32, &slot.next);
                }
            }
            // Occupied metadata stays at empty state zero until its value
            // is moved below; no panic can make Drop read uninitialized T.
        }
        for (id, value) in hive.drain() {
            let live = live_version(id.generation()).expect("Hive issues valid generations");
            let slot = out.slot(id.index()).expect("copied slot has a page");
            out.write_value(slot, live, value);
        }
        out
    }
}

impl<T, const PAGE: usize> fmt::Debug for ConcurrentAppendHive<T, PAGE> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConcurrentAppendHive")
            .field("len", &self.len())
            .field("index_bound", &self.index_bound())
            .finish_non_exhaustive()
    }
}

impl<T, const PAGE: usize> Index<Id> for ConcurrentAppendHive<T, PAGE> {
    type Output = T;

    /// # Panics
    ///
    /// Panics if `id` is not live.
    fn index(&self, id: Id) -> &T {
        self.get(id)
            .unwrap_or_else(|| panic!("ConcurrentAppendHive: {id:?} is not live"))
    }
}

impl<T, const PAGE: usize> Drop for ConcurrentAppendHive<T, PAGE> {
    fn drop(&mut self) {
        // If a value's destructor panics, keep dropping the remaining values
        // while unwinding (a second panic then aborts, as usual).
        struct Rest<'a, T, const PAGE: usize>(&'a mut ConcurrentAppendHive<T, PAGE>);
        impl<T, const PAGE: usize> Drop for Rest<'_, T, PAGE> {
            fn drop(&mut self) {
                self.0.clear();
            }
        }
        let rest = Rest(self);
        rest.0.clear();
        core::mem::forget(rest);
    }
}

/// Weakly consistent iterator over `(id, &value)` pairs, in slot order.
/// Each slot is visited once, up to the initial reservation bound.
pub struct AppendIter<'a, T, const PAGE: usize = DEFAULT_PAGE> {
    hive: &'a ConcurrentAppendHive<T, PAGE>,
    indices: core::ops::Range<u32>,
}

impl<'a, T, const PAGE: usize> Iterator for AppendIter<'a, T, PAGE> {
    type Item = (Id, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        for index in self.indices.by_ref() {
            let Some(slot) = self.hive.slot(index) else {
                continue;
            };
            // Acquire observes value publication; BUSY is never yielded.
            let word = slot.state.load(Ordering::Acquire);
            if word & !VERSION != 0 || word & 1 == 0 {
                continue;
            }
            let value = slot.value.with(|cell| {
                // SAFETY: this published live state changes only under
                // &mut self; the iterator's shared borrow excludes removal
                // and keeps the initialized value alive for 'a.
                unsafe { &*cell.cast::<T>() }
            });
            return Some((Id::new(index, generation_of(word as u32)), value));
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some((self.indices.end - self.indices.start) as usize))
    }
}

impl<T, const PAGE: usize> FusedIterator for AppendIter<'_, T, PAGE> {}

/// Exclusive iterator over `(id, &mut value)` pairs, in slot order.
/// Holds the exclusive hive borrow even though traversal reads its metadata.
pub struct AppendIterMut<'a, T, const PAGE: usize = DEFAULT_PAGE> {
    hive: &'a ConcurrentAppendHive<T, PAGE>,
    indices: core::ops::Range<u32>,
    remaining: usize,
    exclusive: PhantomData<&'a mut ConcurrentAppendHive<T, PAGE>>,
}

impl<'a, T, const PAGE: usize> Iterator for AppendIterMut<'a, T, PAGE> {
    type Item = (Id, &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        for index in self.indices.by_ref() {
            let Some(slot) = self.hive.slot(index) else {
                continue;
            };
            let version = slot.exclusive_version();
            if version & 1 == 0 {
                continue;
            }
            self.remaining -= 1;
            let value = slot.value.with_mut(|cell| {
                // SAFETY: exclusive iterator construction excludes every
                // other hive access. Each distinct slot is yielded once;
                // metadata traversal never reborrows any yielded value.
                unsafe { &mut *cell.cast::<T>() }
            });
            return Some((Id::new(index, generation_of(version)), value));
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T, const PAGE: usize> ExactSizeIterator for AppendIterMut<'_, T, PAGE> {}
impl<T, const PAGE: usize> FusedIterator for AppendIterMut<'_, T, PAGE> {}

/// Eager draining iterator of owned `(id, value)` pairs.
/// All values have already been removed from the hive when this is returned.
pub struct AppendDrain<'a, T, const PAGE: usize = DEFAULT_PAGE> {
    inner: alloc::vec::IntoIter<(Id, T)>,
    exclusive: PhantomData<&'a mut ConcurrentAppendHive<T, PAGE>>,
}

impl<T, const PAGE: usize> Iterator for AppendDrain<'_, T, PAGE> {
    type Item = (Id, T);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T, const PAGE: usize> DoubleEndedIterator for AppendDrain<'_, T, PAGE> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back()
    }
}

impl<T, const PAGE: usize> ExactSizeIterator for AppendDrain<'_, T, PAGE> {}
impl<T, const PAGE: usize> FusedIterator for AppendDrain<'_, T, PAGE> {}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use crate::id_alloc::MAX_GENERATION;
    use core::num::NonZeroU32;

    #[test]
    fn last_live_version_retires_and_even_a_stale_hint_cannot_revive_it() {
        let plain = Hive::<u8, 2>::try_from_raw_pages([RawPage {
            floor: 0,
            slots: alloc::vec![
                RawSlot::Occupied {
                    version: u32::MAX,
                    value: 7,
                },
                RawSlot::Vacant { version: 0 },
            ],
        }])
        .unwrap();
        let mut hive = ConcurrentAppendHive::from(plain);
        let last = Id::new(0, NonZeroU32::new(MAX_GENERATION).unwrap());
        assert_eq!(hive.get(last), Some(&7));
        assert_eq!(hive.iter().next(), Some((last, &7)));
        assert_eq!(hive.remove(last), Some(7));
        let slot = hive.slot(0).unwrap();
        assert_eq!(slot.state.load(Ordering::Acquire), 0);
        hive.free.push(0, &slot.next);
        let next = hive.insert(8);
        assert_eq!(next.index(), 2);
        assert!(hive.get(last).is_none());
        assert_eq!(hive.len(), 1);
    }

    #[test]
    fn a_live_stack_hint_is_discarded() {
        let hive = ConcurrentAppendHive::<u8, 2>::new();
        let a = hive.insert(1);
        let borrowed = hive.get(a).unwrap();
        let slot = hive.slot(a.index()).unwrap();
        hive.free.push(a.index(), &slot.next);
        let b = hive.insert(2);
        assert_ne!(a.index(), b.index());
        assert_eq!(*borrowed, 1);
        assert_eq!(hive.get(b), Some(&2));
        assert_eq!(hive.len(), 2);
    }
}
