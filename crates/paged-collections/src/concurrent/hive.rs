//! Stable value cells, pinned by guards and claimed by CAS before every reuse.
//!
//! The low 32 state bits are the id version. BUSY owns the cell exclusively
//! while inserting or finalizing; MARKED prevents new guards; TAKE reserves a
//! removed value for its taker. The remaining 29 bits count guards. Pages do
//! not drop values: only a BUSY owner, or an exclusive hive scan, may do that.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::fmt;
use core::iter::FusedIterator;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ops::Deref;
use core::ptr;

use super::atomic::{AtomicU32, AtomicU64, Ordering, UnsafeCell, spin_loop};
use super::count::StripedLen;
use super::free_stack::{FreeStack, Unissued, block_for};
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
const MARKED: u64 = 1 << 33;
const TAKE: u64 = 1 << 34;
const REF: u64 = 1 << 35;
const REFS: u64 = !(REF - 1);

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
}

struct Page<T> {
    // MaybeUninit deliberately has no value destructor. The hive scans cells.
    slots: Box<[Slot<T>]>,
}

fn vacant(live: u32) -> u64 {
    u64::from(live.checked_add(1).unwrap_or(0))
}

fn matches(word: u64, version: u32) -> bool {
    word & VERSION == u64::from(version) && word & (BUSY | MARKED) == 0
}

fn retry(spins: &mut u32) {
    *spins = spins.saturating_add(1);
    #[cfg(all(feature = "std", not(loom)))]
    if *spins >= 64 {
        std::thread::yield_now();
        return;
    }
    spin_loop();
}

/// A paged container with stable addresses and shared insertion and removal.
///
/// [`get`](Self::get) returns a guard instead of a bare reference. Removal
/// invalidates the id immediately, but existing guards keep the value alive.
/// Taking a value waits for its guards. A forgotten guard pins its slot until
/// exclusive access or destruction. There is no mutable access while shared.
///
/// Pages stay allocated until [`shrink_to_fit`](Self::shrink_to_fit) frees
/// the empty ones; shared inserts reuse freed pages later. Shared iteration is weakly
/// consistent: it may miss concurrent insertions or removals. `len` is
/// approximate during shared operations and exact when they have finished.
/// Exclusive scans take O([`index_bound`](Self::index_bound)) time.
///
/// Each successful lookup costs two atomic writes (guard acquisition and
/// release). Prefer [`Hive`](crate::hive::Hive) when shared mutation is not
/// needed. As with that type, ids carry no container identity, generations
/// never wrap, and stale ids stay dead within one allocation history.
///
/// `PAGE` must be non-zero, fit in a `u32`, and fit a page in `isize::MAX`
/// bytes. Uses only `core` and `alloc`; `std` enables yielding while waiting.
pub struct ConcurrentHive<T, const PAGE: usize = DEFAULT_PAGE> {
    pages: Spine<Page<T>, CHUNKS>,
    free: FreeStack,
    /// Pages freed by `shrink_to_fit`, below the reservation bound.
    recycled: PageStack,
    len: StripedLen,
}

// SAFETY: moving the hive transfers its owned cells; no guard outlives its
// borrow. Exclusive finalization may drop T on the receiving thread.
unsafe impl<T: Send, const PAGE: usize> Send for ConcurrentHive<T, PAGE> {}
// SAFETY: BUSY exclusively owns writes/drops; matching counted guards alone
// share live T. Shared removal/take can also transfer or drop T across threads.
unsafe impl<T: Send + Sync, const PAGE: usize> Sync for ConcurrentHive<T, PAGE> {}

/// Owns a cell already claimed by CAS (or by an exclusive scan). Releasing it
/// publishes completion before making it reusable. Also runs during unwinding
/// after a factory or value destructor panics; value drop glue finishes before
/// this owner's cleanup runs.
struct Busy<'a, T, const PAGE: usize> {
    hive: &'a ConcurrentHive<T, PAGE>,
    slot: &'a Slot<T>,
    index: u32,
    vacant: u64,
    armed: bool,
}

impl<T, const PAGE: usize> Drop for Busy<'_, T, PAGE> {
    fn drop(&mut self) {
        if self.armed {
            // Release publishes the end of ALL value access, including drop
            // glue during unwinding. A claim's Acquire observes this even if
            // its stack hint did not come from this particular push.
            self.slot.state.store(self.vacant, Ordering::Release);
            if self.vacant != 0 {
                self.hive.free.push(self.index, &self.slot.next);
            }
        }
    }
}

impl<T, const PAGE: usize> ConcurrentHive<T, PAGE> {
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
            "ConcurrentHive: invalid page capacity"
        );
        Self {
            pages: Spine::new(),
            free: FreeStack::new(),
            recycled: PageStack::new(),
            len: StripedLen::new(),
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

    fn busy<'a>(&'a self, slot: &'a Slot<T>, index: u32, vacant: u64) -> Busy<'a, T, PAGE> {
        Busy {
            hive: self,
            slot,
            index,
            vacant,
            armed: true,
        }
    }

    /// Reinstalls a page freed by `shrink_to_fit`, with every usable slot
    /// vacant at the page's floor. Pushes all its slots but the first to the
    /// free stack and returns the first, for the caller to claim by CAS.
    ///
    /// The popper is the page's only installer: the free stack holds none of
    /// its indices (it was rebuilt when the page was freed), and fresh
    /// reservation never hands out an index in a freed page: the counter
    /// never goes below the bound, and indices reserved in a thread's block
    /// but not handed out keep state 0, which stops their page from being
    /// freed (uninstalled pages are never freed). Its slots are unreachable
    /// until pushed (a guessed id finds a vacant state), so this thread owns
    /// them while pushing.
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
            let (index, reused) = self.free.next_index(
                |index| &self.slot(index).expect("free index has a page").next,
                || self.recycle(),
                MAX_INDICES,
                block_for(PAGE),
            )?;
            let slot = self.ensure_slot(index);
            // Acquire observes a completed finalizer, independently of the
            // free-stack hint. No flags or references may remain at claim.
            let old = slot.state.load(Ordering::Acquire);
            if old & !VERSION != 0 || old & 1 != 0 || (reused && old == 0) {
                continue;
            }
            // AcqRel claims sole ownership and observes the preceding vacant
            // Release. A failed CAS discards the hint, never retries that word.
            // Fresh 0|BUSY and retiring 0|BUSY overlap harmlessly: both are
            // owned, unclaimable states; fresh indices are uniquely reserved.
            if slot
                .state
                .compare_exchange(old, old | BUSY, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                continue;
            }
            // If the insertion unwinds (a panicking factory), the id it was
            // shown is spent: the slot returns as the vacancy after `live`,
            // so that id can never name a later value.
            let live = old as u32 + 1;
            return Some((self.busy(slot, index, vacant(live)), live));
        }
    }

    // The caller owns this empty cell through a claim or exclusive conversion.
    fn write_value(&self, slot: &Slot<T>, index: u32, live: u32, value: T) {
        slot.value.with_mut(|cell| {
            // SAFETY: the claim CAS or exclusive construction grants sole
            // ownership of this uninitialized cell. MaybeUninit<T> has T's
            // layout/alignment.
            unsafe { ptr::write(cell.cast::<T>(), value) };
        });
        // Relaxed accounting precedes publication, so any removal of this
        // value, even through a guessed id, is counted after it.
        self.len.increment(index as usize / PAGE);
        // Release publishes the initialized value and preceding accounting;
        // every successful reader/remover CAS acquires that publication.
        slot.state.store(u64::from(live), Ordering::Release);
    }

    fn publish(&self, mut owner: Busy<'_, T, PAGE>, live: u32, value: T) -> Id {
        self.write_value(owner.slot, owner.index, live, value);
        owner.armed = false;
        Id::new(owner.index, generation_of(live))
    }

    /// Inserts `value` and returns its id.
    ///
    /// # Panics
    ///
    /// Panics if all [`MAX_INDICES`] indices are occupied or retired, or a
    /// page cannot be allocated. A concurrent finalizer may make room later.
    pub fn insert(&self, value: T) -> Id {
        self.try_insert(value).unwrap_or_else(|_| {
            panic!("ConcurrentHive: all {MAX_INDICES} slot indices are occupied or retired")
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
    /// If `f` panics, the slot is returned to the free stack and the
    /// generation of the id passed to `f` is spent, so that id stays dead.
    ///
    /// # Panics
    ///
    /// Panics on index exhaustion or if `f` panics; allocation may also fail.
    pub fn insert_with(&self, f: impl FnOnce(Id) -> T) -> Id {
        self.try_insert_with(f).unwrap_or_else(|| {
            panic!("ConcurrentHive: all {MAX_INDICES} slot indices are occupied or retired")
        })
    }

    /// Like [`insert_with`](Self::insert_with), returning `None` without
    /// calling `f` when no index is available.
    pub fn try_insert_with(&self, f: impl FnOnce(Id) -> T) -> Option<Id> {
        let (owner, live) = self.claim()?;
        let value = f(Id::new(owner.index, generation_of(live)));
        Some(self.publish(owner, live, value))
    }

    fn acquire(&self, index: u32, expected: Option<u32>) -> Option<Ref<'_, T, PAGE>> {
        let slot = self.slot(index)?;
        self.acquire_slot(slot, index, expected).ok()
    }

    // Returns the word that ruled out a guard, so a scan can map that state
    // without loading again after a concurrent transition.
    fn acquire_slot<'a>(
        &'a self,
        slot: &'a Slot<T>,
        index: u32,
        expected: Option<u32>,
    ) -> Result<Ref<'a, T, PAGE>, u64> {
        // Acquire observes publication before inspecting the version/flags.
        let mut word = slot.state.load(Ordering::Acquire);
        loop {
            let live = word as u32;
            if live & 1 == 0
                || word & (BUSY | MARKED) != 0
                || expected.is_some_and(|version| version != live)
            {
                return Err(word);
            }
            assert!(word & REFS != REFS, "ConcurrentHive: guard count overflow");
            // Acquire sees the value published by the inserter; Release makes
            // this guard count visible to removers and subsequent guard drops.
            match slot.state.compare_exchange_weak(
                word,
                word + REF,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    return Ok(Ref {
                        hive: self,
                        slot,
                        index,
                        version: live,
                        #[cfg(loom)]
                        access: Some(slot.value.read()),
                    });
                }
                Err(actual) => word = actual,
            }
        }
    }

    fn snapshot_slot(&self, index: u32, unissued: &Unissued) -> RawSlot<Ref<'_, T, PAGE>> {
        let Some(slot) = self.slot(index) else {
            // A reservation or recycled page may still be uninstalled.
            return RawSlot::Vacant { version: 0 };
        };
        let word = match self.acquire_slot(slot, index, None) {
            Ok(value) => {
                return RawSlot::Occupied {
                    version: value.version,
                    value,
                };
            }
            Err(word) => word,
        };
        let mut version = word as u32;
        if word & BUSY != 0 {
            #[cfg(all(test, loom))]
            super::loom_hive::record_snapshot_busy(index);
            // A claimer keeps the old vacancy; a finalizer has already
            // advanced to the next vacancy (or to zero on retirement).
            debug_assert_eq!(version & 1, 0);
        } else if word & MARKED != 0 {
            #[cfg(all(test, loom))]
            super::loom_hive::record_snapshot_marked();
            // Logical removal precedes finalization, including for TAKE.
            version = vacant(version) as u32;
        } else if version == 0 && unissued.contains_shared(index) {
            version = 2;
        }
        RawSlot::Vacant { version }
    }

    /// Pins the value of a matching live id until the returned guard drops.
    ///
    /// Returns `None` for a stale or removed id. Panics if the slot already
    /// has the maximum of 2²⁹ − 1 guards. Guards cannot be cloned.
    pub fn get(&self, id: Id) -> Option<Ref<'_, T, PAGE>> {
        self.acquire(id.index(), Some(live_version(id.generation())?))
    }

    /// Whether this id is live at the instant of one state load.
    /// Does not pin the value against a subsequent removal.
    pub fn contains(&self, id: Id) -> bool {
        let Some(live) = live_version(id.generation()) else {
            return false;
        };
        // Acquire observes publication/invalidation, but grants no cell access.
        self.slot(id.index())
            .is_some_and(|slot| matches(slot.state.load(Ordering::Acquire), live))
    }

    /// Logically removes a matching live id. Exactly one remover succeeds.
    /// Existing guards defer destruction until the last guard drops.
    ///
    /// If immediate value destruction panics, the id is still removed and its
    /// slot is returned after drop glue finishes unwinding.
    pub fn remove(&self, id: Id) -> bool {
        let Some(live) = live_version(id.generation()) else {
            return false;
        };
        let Some(slot) = self.slot(id.index()) else {
            return false;
        };
        // Acquire observes publication and competing guard/removal operations.
        let mut word = slot.state.load(Ordering::Acquire);
        loop {
            if !matches(word, live) {
                return false;
            }
            let direct = word & REFS == 0;
            let new = if direct {
                vacant(live) | BUSY
            } else {
                word | MARKED
            };
            // AcqRel acquires value publication and prior guard releases, and
            // publishes invalidation. Only the winner may finalize or mark.
            match slot
                .state
                .compare_exchange_weak(word, new, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => {
                    // Relaxed accounting: logically removed now, before any
                    // user destructor (which may panic or reenter the hive).
                    self.len.decrement(id.index() as usize / PAGE);
                    if direct {
                        let owner = self.busy(slot, id.index(), vacant(live));
                        self.drop_owned(&owner);
                    }
                    return true;
                }
                Err(actual) => word = actual,
            }
        }
    }

    fn drop_owned(&self, owner: &Busy<'_, T, PAGE>) {
        owner.slot.value.with_mut(|cell| {
            // SAFETY: the finalizing CAS (or exclusive scan) granted this
            // BUSY owner an initialized value, with no remaining guard holders.
            unsafe { ptr::drop_in_place(cell.cast::<T>()) };
        });
    }

    /// Removes and returns a matching live value, waiting for its guards.
    ///
    /// Blocks while any guard on this id lives; **deadlocks if the calling
    /// thread holds one**. A forgotten guard can block this call forever.
    /// Competing removes or takes return `false`/`None` once this call marks
    /// the id. The last guard never drops a value owned by a taker.
    pub fn take(&self, id: Id) -> Option<T> {
        let live = live_version(id.generation())?;
        let slot = self.slot(id.index())?;
        // Acquire observes the initialized live value and current guard count.
        let mut word = slot.state.load(Ordering::Acquire);
        loop {
            if !matches(word, live) {
                return None;
            }
            // AcqRel reserves the value for this taker, observes publication,
            // and prevents any new guard from being acquired.
            match slot.state.compare_exchange_weak(
                word,
                word | MARKED | TAKE,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(actual) => word = actual,
            }
        }
        // Relaxed accounting: the mark is the logical removal point.
        self.len.decrement(id.index() as usize / PAGE);
        let mut spins = 0;
        loop {
            // Acquire observes guard releases before exclusive extraction.
            let word = slot.state.load(Ordering::Acquire);
            // AcqRel transfers the cell from the marked taker to BUSY;
            // its Acquire sees the last guard's Release, even on retries.
            if word & REFS == 0
                && slot
                    .state
                    .compare_exchange(
                        u64::from(live) | MARKED | TAKE,
                        vacant(live) | BUSY,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok()
            {
                let owner = self.busy(slot, id.index(), vacant(live));
                let value = slot.value.with_mut(|cell| {
                    // SAFETY: the taker's final CAS owns BUSY after all
                    // guards released; no finalizer can drop this value.
                    unsafe { ptr::read(cell.cast::<T>()) }
                });
                drop(owner);
                return Some(value);
            }
            retry(&mut spins);
        }
    }

    /// Number of logically live values; approximate during shared operations.
    /// Exact when no operation is in flight; this is no publication barrier.
    ///
    /// It sums per-thread counters (per page group without `std`). While
    /// operations are in flight it may combine counters read at different
    /// moments and differ from every count the hive held; it never
    /// exceeds the number of insertions.
    pub fn len(&self) -> usize {
        self.len.get()
    }

    /// Whether the hive is empty; approximate during shared operations.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// One past the highest reserved index, including insertions in flight.
    /// A reserved index's page may not have been installed yet.
    ///
    /// Threads that contend for fresh indices reserve them in blocks of
    /// about one page, so the bound may exceed the indices handed out by up
    /// to one block per thread.
    pub fn index_bound(&self) -> u32 {
        self.free.index_bound()
    }

    /// Iterates over `(id, guard)` in slot order, weakly consistently.
    /// Scans only indices below the reservation bound at iterator creation.
    pub fn iter(&self) -> Iter<'_, T, PAGE> {
        Iter {
            hive: self,
            indices: 0..self.index_bound(),
        }
    }

    // &mut self proves every outstanding count belongs to a forgotten Ref.
    // Such a guard can never run again. MARKED values are already absent from
    // len; finalize them without decrementing it a second time.
    fn normalize_slot(&mut self, index: u32) {
        let Some(slot) = self.slot(index) else {
            return;
        };
        // Acquire is conservative; exclusivity alone rules out concurrent use.
        let word = slot.state.load(Ordering::Acquire);
        if word & 1 == 0 {
            return;
        }
        if word & MARKED != 0 {
            let vacant = vacant(word as u32);
            // Release records exclusive BUSY ownership before user drop code.
            slot.state.store(vacant | BUSY, Ordering::Release);
            let owner = self.busy(slot, index, vacant);
            self.drop_owned(&owner);
        } else if word & REFS != 0 {
            // Release resets forgotten counts; no actual guard can still exist.
            slot.state.store(word & !REFS, Ordering::Release);
        }
    }

    fn normalize(&mut self) {
        for index in 0..self.index_bound() {
            self.normalize_slot(index);
        }
    }

    /// The matching live value, mutably, under exclusive access.
    /// Forgotten guard counts at this slot are reset. A marked value is
    /// finalized and is never returned, even for a different requested id.
    pub fn get_mut(&mut self, id: Id) -> Option<&mut T> {
        self.normalize_slot(id.index());
        let live = live_version(id.generation())?;
        let slot = self.slot(id.index())?;
        // Acquire observes the normalized state; exclusivity excludes writers.
        if !matches(slot.state.load(Ordering::Acquire), live) {
            return None;
        }
        Some(slot.value.with_mut(|cell| {
            // SAFETY: &mut self excludes all usable guards and other accesses;
            // normalization removed marked values and reset forgotten counts.
            unsafe { &mut *cell.cast::<T>() }
        }))
    }

    /// Removes and returns a live value without waiting for forgotten guards.
    /// Finalizes a marked value at this slot without returning it.
    pub fn take_mut(&mut self, id: Id) -> Option<T> {
        self.normalize_slot(id.index());
        self.take(id)
    }

    /// Iterates over live `(id, &mut value)` pairs in slot order.
    /// Finalizes all marked values and resets forgotten counts before yielding.
    pub fn iter_mut(&mut self) -> IterMut<'_, T, PAGE> {
        self.normalize();
        IterMut {
            remaining: self.len(),
            indices: 0..self.index_bound(),
            hive: self,
            exclusive: PhantomData,
        }
    }

    /// Removes and drops every value, retaining pages and generation history.
    /// Forgotten guards do not pin values under exclusive access.
    pub fn clear(&mut self) {
        self.retain(|_, _| false);
    }

    /// Eagerly removes every live value, yielding owned `(id, value)` pairs in
    /// slot order. Dropping or forgetting the iterator leaves the hive empty.
    /// Marked values are finalized and not yielded. Pages and generations stay.
    pub fn drain(&mut self) -> Drain<'_, T, PAGE> {
        self.normalize();
        let mut values = Vec::with_capacity(self.len());
        for index in 0..self.index_bound() {
            let Some(slot) = self.slot(index) else {
                continue;
            };
            // Acquire observes normalized live state; access is exclusive.
            let word = slot.state.load(Ordering::Acquire);
            if word & 1 != 0 {
                let id = Id::new(index, generation_of(word as u32));
                values.push((id, self.take(id).expect("normalized live value")));
            }
        }
        Drain {
            inner: values.into_iter(),
            exclusive: PhantomData,
        }
    }

    /// Keeps values for which `keep(id, value)` returns true, in slot order.
    /// Finalizes marked values and resets forgotten counts before calling
    /// `keep`. If the predicate or a destructor panics, prior changes remain
    /// and `len` stays exact.
    pub fn retain(&mut self, mut keep: impl FnMut(Id, &mut T) -> bool) {
        self.normalize();
        for index in 0..self.index_bound() {
            let Some(slot) = self.slot(index) else {
                continue;
            };
            // Acquire observes normalized state; exclusivity rules out races.
            let word = slot.state.load(Ordering::Acquire);
            if word & 1 == 0 {
                continue;
            }
            let id = Id::new(index, generation_of(word as u32));
            let keep_value = keep(id, self.get_mut(id).expect("normalized live value"));
            if !keep_value {
                drop(self.take(id));
            }
        }
    }

    /// Frees every page that holds no value, under exclusive access. Takes
    /// O([`index_bound`](Self::index_bound)) time.
    ///
    /// Marked values are finalized and forgotten guard counts reset first.
    /// A page is then freed if it lies wholly below the reservation bound and
    /// all its slots are vacant and reusable (a page with a retired slot is
    /// kept). It remembers its highest version, so ids it issued stay dead:
    /// shared inserts reuse freed pages, lowest first, after the free slots
    /// of the remaining pages, with generations above every earlier one.
    /// That order is exact only for one inserter that has not contended
    /// with others: a thread first uses up a block of fresh indices it
    /// reserved under contention, and while one insert is relinking a
    /// reused page, a concurrent insert can take a fresh index
    /// instead (and fail at the index limit even though slots are about to
    /// be freed). The free slots are relinked in index order. The page directory's own
    /// small arrays stay allocated.
    ///
    /// ```
    /// use paged_collections::concurrent::ConcurrentHive;
    ///
    /// let mut hive = ConcurrentHive::<u32, 4>::new();
    /// let ids: Vec<_> = (0..12).map(|v| hive.insert(v)).collect();
    /// for &id in &ids[..8] {
    ///     hive.remove(id);
    /// }
    /// hive.shrink_to_fit(); // frees the first two pages
    /// assert_eq!(*hive.get(ids[8]).unwrap(), 8);
    /// let new = hive.insert(99); // reuses a freed page
    /// assert!(new.index() < 8 && !ids.contains(&new));
    /// assert!(hive.get(ids[0]).is_none());
    /// ```
    pub fn shrink_to_fit(&mut self) {
        self.normalize();
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
                // Relaxed: exclusive access, and normalization left no flags.
                let word = slot.state.load(Ordering::Relaxed);
                debug_assert_eq!(word & !VERSION, 0, "normalized slot has flags");
                let version = word as u32;
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
                // Relaxed: exclusive access rules out concurrent changes.
                let version = slot.state.load(Ordering::Relaxed) as u32;
                if version & 1 == 0 && version != 0 {
                    self.free.push(index, &slot.next);
                }
            }
        }
        self.recycled.set(freed);
    }

    /// Clones a weakly consistent snapshot into a plain [`Hive`].
    /// Insertions that happen-before the call are included if the entry
    /// stays live throughout it. Ids whose removal happens-before the
    /// call stay dead. The raw pages always validate. Scans up to
    /// [`index_bound`](Self::index_bound) with a floor lookup per page.
    ///
    /// Each value stays guarded while its `Clone` runs. Taking that id in
    /// the callback, or waiting there for another thread to take it, deadlocks.
    pub fn snapshot(&self) -> Hive<T, PAGE>
    where
        T: Clone,
    {
        let bound = self.index_bound();
        let unissued = self.free.unissued_shared();
        let pages = self
            .recycled
            .snapshot_pages(bound, PAGE)
            .map(|(floor, indices)| RawPage {
                floor,
                slots: indices
                    .map(|index| match self.snapshot_slot(index, &unissued) {
                        RawSlot::Occupied { version, value } => RawSlot::Occupied {
                            version,
                            value: (*value).clone(),
                        },
                        RawSlot::Vacant { version } => RawSlot::Vacant { version },
                    })
                    .collect(),
            });
        Hive::try_from_raw_pages(pages).expect("hive snapshot has valid raw pages")
    }

    /// Moves into a plain [`Hive`], preserving live ids and stale-id history.
    /// Takes O([`index_bound`](Self::index_bound)) time; pages and reuse order
    /// may change.
    ///
    /// Marked values are finalized and not carried over. Forgotten guards are
    /// released: their counts are reset before any live value is moved.
    /// Every slot below the bound is represented explicitly: zero states
    /// become retired vacancies, even versions become reusable vacancies,
    /// and odd versions keep their values. Indices reserved for a thread's
    /// block but never handed out become reusable vacancies at version 2. A page freed by
    /// [`shrink_to_fit`](Self::shrink_to_fit) becomes an empty raw page with
    /// its floor; other floors are zero. Offsets beyond the bound have never
    /// issued ids and need no history.
    ///
    /// Raw-page validation cannot fail by construction: all versions have
    /// the required parity, all pages fit `PAGE`, and indices are below
    /// [`MAX_INDICES`]. Allocation or a marked value's destructor may panic;
    /// allocation may also abort. Values already moved are owned by the raw
    /// pages; this hive's empty cells cannot drop them again, including if
    /// allocation unwinds.
    ///
    /// # Examples
    ///
    /// ```
    /// use paged_collections::concurrent::ConcurrentHive;
    ///
    /// let hive = ConcurrentHive::<String, 4>::new();
    /// let id = hive.insert(String::from("hello"));
    /// let plain = hive.into_hive();
    /// assert_eq!(plain.get(id).map(String::as_str), Some("hello"));
    /// assert_eq!(plain.len(), 1);
    /// ```
    pub fn into_hive(mut self) -> Hive<T, PAGE> {
        self.normalize();
        let bound = self.index_bound() as usize;
        let mut unissued = self.free.unissued();
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
                let mut version = self.slot(index as u32).map_or(0, |slot| {
                    // Acquire is conservative: normalization and exclusive
                    // ownership rule out concurrent flags or guard counts.
                    let word = slot.state.load(Ordering::Acquire);
                    debug_assert!(word & !VERSION == 0);
                    word as u32
                });
                if unissued.contains(index as u32) {
                    // Reserved in a block, never handed out: reusable, with
                    // generation 1 spent (a zero would read as retired).
                    debug_assert_eq!(version, 0, "unissued index was used");
                    version = 2;
                }
                slots.push(if version & 1 != 0 {
                    let id = Id::new(index as u32, generation_of(version));
                    RawSlot::Occupied {
                        version,
                        value: self.take(id).expect("normalized live value"),
                    }
                } else {
                    RawSlot::Vacant { version }
                });
            }
            pages.push(RawPage { floor: 0, slots });
        }
        Hive::try_from_raw_pages(pages)
            .expect("concurrent hive raw pages are valid by construction")
    }

    #[cfg(all(test, loom))]
    pub(super) fn inject_busy_hint(&self, index: u32) {
        let slot = self.slot(index).unwrap();
        assert_ne!(slot.state.load(Ordering::Acquire) & BUSY, 0);
        // Models a stale link installed after tag wrap, without 2^32 pops.
        self.free.push(index, &slot.next);
    }
}

/// Clones via [`ConcurrentHive::snapshot`], preserving generations.
/// Each value stays guarded while its `Clone` runs, so taking that id
/// or waiting for its taker from the callback deadlocks.
impl<T: Clone, const PAGE: usize> Clone for ConcurrentHive<T, PAGE> {
    fn clone(&self) -> Self {
        Self::from(self.snapshot())
    }
}

impl<'a, T, const PAGE: usize> IntoIterator for &'a ConcurrentHive<T, PAGE> {
    type Item = (Id, Ref<'a, T, PAGE>);
    type IntoIter = Iter<'a, T, PAGE>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, const PAGE: usize> IntoIterator for &'a mut ConcurrentHive<T, PAGE> {
    type Item = (Id, &'a mut T);
    type IntoIter = IterMut<'a, T, PAGE>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T, const PAGE: usize> Default for ConcurrentHive<T, PAGE> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "serde")]
mod serde_impl {
    use super::*;
    use crate::hive::serde_impl::raw_pages;
    use serde::de::{Deserialize, Deserializer};
    use serde::ser::{Serialize, Serializer};

    /// Serializes a weakly consistent snapshot in the plain [`Hive`] format.
    /// Uses the same consistency as [`ConcurrentHive::snapshot`] and scans
    /// to the initial index bound with one floor lookup per page.
    /// The resulting pages always pass [`Hive`]'s validation.
    ///
    /// Each value stays guarded while its `Serialize` runs. Taking that id
    /// in the callback, or waiting there for another thread to take it, deadlocks.
    impl<T: Serialize, const PAGE: usize> Serialize for ConcurrentHive<T, PAGE> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let bound = self.index_bound();
            let unissued = self.free.unissued_shared();
            raw_pages::serialize(
                self.recycled.snapshot_pages(bound, PAGE),
                |index| self.snapshot_slot(index, &unissued),
                serializer,
            )
        }
    }

    /// Validates the plain [`Hive`] format and converts in O(index bound).
    impl<'de, T: Deserialize<'de>, const PAGE: usize> Deserialize<'de> for ConcurrentHive<T, PAGE> {
        fn deserialize<De: Deserializer<'de>>(deserializer: De) -> Result<Self, De::Error> {
            Hive::deserialize(deserializer).map(Self::from)
        }
    }
}

/// Moves a plain hive into shared storage without changing live ids.
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
impl<T, const PAGE: usize> From<Hive<T, PAGE>> for ConcurrentHive<T, PAGE> {
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
            out.write_value(slot, id.index(), live, value);
        }
        out
    }
}

impl<T, const PAGE: usize> fmt::Debug for ConcurrentHive<T, PAGE> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConcurrentHive")
            .field("len", &self.len())
            .field("index_bound", &self.index_bound())
            .finish_non_exhaustive()
    }
}

impl<T, const PAGE: usize> Drop for ConcurrentHive<T, PAGE> {
    fn drop(&mut self) {
        // If a value's destructor panics, keep dropping the remaining values
        // while unwinding (a second panic then aborts, as usual).
        struct Rest<'a, T, const PAGE: usize>(&'a mut ConcurrentHive<T, PAGE>);
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

/// A guard pinning one live value in a [`ConcurrentHive`].
///
/// Dereferences to `T`; no mutable access or cloning is provided. The value
/// stays alive after logical removal until all its guards are dropped, unless
/// a taker owns it. Forgetting a guard pins its slot during shared access.
pub struct Ref<'a, T, const PAGE: usize = DEFAULT_PAGE> {
    hive: &'a ConcurrentHive<T, PAGE>,
    slot: &'a Slot<T>,
    index: u32,
    version: u32,
    #[cfg(loom)]
    access: Option<loom::cell::ConstPtr<MaybeUninit<T>>>,
}

// SAFETY: the counted guard pins a shared T; sending the guard can make this
// thread the finalizer, so both sharing and transferring/dropping T are needed.
unsafe impl<T: Send + Sync, const PAGE: usize> Send for Ref<'_, T, PAGE> {}
// SAFETY: dereferencing shares only a pinned T, and the hive can be shared
// because T: Send + Sync; no shared operation yields mutable value access.
unsafe impl<T: Send + Sync, const PAGE: usize> Sync for Ref<'_, T, PAGE> {}

impl<T, const PAGE: usize> Ref<'_, T, PAGE> {
    /// The id whose value this guard pins, even after logical removal.
    pub fn id(&self) -> Id {
        Id::new(self.index, generation_of(self.version))
    }
}

impl<T, const PAGE: usize> Deref for Ref<'_, T, PAGE> {
    type Target = T;

    fn deref(&self) -> &T {
        self.slot.value.with(|cell| {
            // SAFETY: this matching-version guard holds a counted reference;
            // no finalizer or taker can access the cell until it is released.
            // Under loom, access also tracks this read for the entire guard.
            unsafe { &*cell.cast::<T>() }
        })
    }
}

impl<T: fmt::Debug, const PAGE: usize> fmt::Debug for Ref<'_, T, PAGE> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ref")
            .field("id", &self.id())
            .field("value", &**self)
            .finish()
    }
}

impl<T, const PAGE: usize> Drop for Ref<'_, T, PAGE> {
    fn drop(&mut self) {
        // End loom's tracked read BEFORE releasing the count or finalizing.
        #[cfg(loom)]
        drop(self.access.take());
        // Acquire sees removal/take marks and prior guard releases.
        let mut word = self.slot.state.load(Ordering::Acquire);
        loop {
            debug_assert!(word & REFS != 0);
            let finalize = word & REFS == REF && word & MARKED != 0 && word & TAKE == 0;
            let new = if finalize {
                vacant(self.version) | BUSY
            } else {
                word - REF
            };
            // Release ends this guard's value access. Acquire observes the
            // preceding releases if this is the last guard and must finalize.
            match self.slot.state.compare_exchange_weak(
                word,
                new,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    if finalize {
                        let owner = self.hive.busy(self.slot, self.index, vacant(self.version));
                        self.hive.drop_owned(&owner);
                    }
                    return;
                }
                Err(actual) => word = actual,
            }
        }
    }
}

/// Weakly consistent iterator over `(id, guard)` pairs, in slot order.
/// Each slot is visited once, up to the initial reservation bound.
pub struct Iter<'a, T, const PAGE: usize = DEFAULT_PAGE> {
    hive: &'a ConcurrentHive<T, PAGE>,
    indices: core::ops::Range<u32>,
}

impl<'a, T, const PAGE: usize> Iterator for Iter<'a, T, PAGE> {
    type Item = (Id, Ref<'a, T, PAGE>);

    fn next(&mut self) -> Option<Self::Item> {
        for index in self.indices.by_ref() {
            if let Some(value) = self.hive.acquire(index, None) {
                return Some((value.id(), value));
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some((self.indices.end - self.indices.start) as usize))
    }
}

impl<T, const PAGE: usize> FusedIterator for Iter<'_, T, PAGE> {}

/// Exclusive iterator over `(id, &mut value)` pairs, in slot order.
/// Holds the exclusive hive borrow even though traversal reads its metadata.
pub struct IterMut<'a, T, const PAGE: usize = DEFAULT_PAGE> {
    hive: &'a ConcurrentHive<T, PAGE>,
    indices: core::ops::Range<u32>,
    remaining: usize,
    exclusive: PhantomData<&'a mut ConcurrentHive<T, PAGE>>,
}

impl<'a, T, const PAGE: usize> Iterator for IterMut<'a, T, PAGE> {
    type Item = (Id, &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        for index in self.indices.by_ref() {
            let Some(slot) = self.hive.slot(index) else {
                continue;
            };
            // Acquire is conservative: exclusive construction normalized every
            // slot and the iterator alone can access this hive until it drops.
            let word = slot.state.load(Ordering::Acquire);
            if word & 1 == 0 {
                continue;
            }
            self.remaining -= 1;
            let value = slot.value.with_mut(|cell| {
                // SAFETY: exclusive iterator construction removed marked values
                // and reset forgotten refs. Each distinct slot is yielded once;
                // metadata traversal never reborrows any yielded value.
                unsafe { &mut *cell.cast::<T>() }
            });
            return Some((Id::new(index, generation_of(word as u32)), value));
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T, const PAGE: usize> ExactSizeIterator for IterMut<'_, T, PAGE> {}
impl<T, const PAGE: usize> FusedIterator for IterMut<'_, T, PAGE> {}

/// Eager draining iterator of owned `(id, value)` pairs.
/// All values have already been removed from the hive when this is returned.
pub struct Drain<'a, T, const PAGE: usize = DEFAULT_PAGE> {
    inner: alloc::vec::IntoIter<(Id, T)>,
    exclusive: PhantomData<&'a mut ConcurrentHive<T, PAGE>>,
}

impl<T, const PAGE: usize> Iterator for Drain<'_, T, PAGE> {
    type Item = (Id, T);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T, const PAGE: usize> DoubleEndedIterator for Drain<'_, T, PAGE> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back()
    }
}

impl<T, const PAGE: usize> ExactSizeIterator for Drain<'_, T, PAGE> {}
impl<T, const PAGE: usize> FusedIterator for Drain<'_, T, PAGE> {}

#[cfg(all(test, not(loom), feature = "std"))]
mod tests {
    use super::*;
    use crate::id_alloc::MAX_GENERATION;
    use core::num::NonZeroU32;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    /// Versions of a converted hive's first page-group, slot by slot.
    fn raw_versions<T>(plain: &Hive<T, 4>) -> Vec<u32> {
        plain
            .raw_pages()
            .flat_map(|page| page.slots)
            .map(|slot| match slot {
                RawSlot::Occupied { version, .. } | RawSlot::Vacant { version } => version,
            })
            .collect()
    }

    #[test]
    fn unissued_block_tail_becomes_reusable_vacancies() {
        let mut hive = ConcurrentHive::<u32, 4>::new();
        let a = hive.insert(0);
        let b = hive.insert(1);
        hive.free.force_blocks();
        // Block mode reserves [2, 64) (blocks of 64, aligned) and hands out 2.
        let c = hive.insert(2);
        assert_eq!((a.index(), b.index(), c.index()), (0, 1, 2));
        assert_eq!(hive.index_bound(), 64);
        assert!(hive.remove(b));
        // State-0 tail slots keep their pages: shrinking frees nothing here.
        hive.shrink_to_fit();
        let versions = raw_versions(&hive.snapshot());
        assert_eq!(&versions[..4], &[1, 2, 1, 2]);
        // Index 3 is on the installed page. The rest have no page yet.
        assert!(versions[4..].iter().all(|&v| v == 0));
        #[cfg(feature = "serde")]
        {
            let json = serde_json::to_string(&hive).unwrap();
            let decoded: Hive<u32, 4> = serde_json::from_str(&json).unwrap();
            assert_eq!(raw_versions(&decoded), versions);
        }
        let plain = hive.into_hive();
        let versions = raw_versions(&plain);
        assert_eq!(versions.len(), 64);
        assert_eq!(&versions[..3], &[1, 2, 1]);
        assert!(versions[3..].iter().all(|&v| v == 2), "{versions:?}");
        assert!(plain.get(b).is_none());
        assert_eq!((plain.get(a), plain.get(c)), (Some(&0), Some(&2)));
        // Round trip: the holes stay reusable and b stays dead.
        let back = ConcurrentHive::from(plain);
        let fresh: Vec<Id> = (0..62).map(|v| back.insert(v)).collect();
        assert!(fresh.iter().all(|id| id.index() < 64 && *id != b));
        assert!(back.get(b).is_none());
        assert_eq!(back.index_bound(), 64);
    }

    #[test]
    fn last_generation_retires_to_plain_zero() {
        let hive = ConcurrentHive::<u8, 2>::new();
        let old = hive.insert(7);
        let slot = hive.slot(old.index()).unwrap();
        // Single-threaded fixture: jump this initialized value to its final
        // live version without waiting for billions of allocation cycles.
        slot.state.store(u64::from(u32::MAX), Ordering::Release);
        let last = Id::new(old.index(), NonZeroU32::new(MAX_GENERATION).unwrap());
        assert!(hive.remove(last));
        assert_eq!(slot.state.load(Ordering::Acquire), 0);
        // Even a corrupted stack must never resurrect a retired zero slot.
        hive.free.push(old.index(), &slot.next);
        let new = hive.insert(8);
        assert_ne!(old.index(), new.index());
        assert!(!hive.contains(last));
        assert_eq!(hive.len(), 1);
    }

    #[test]
    fn overflow_panics_without_wrapping_the_count() {
        let mut hive = ConcurrentHive::<u8, 2>::new();
        let id = hive.insert(7);
        let slot = hive.slot(id.index()).unwrap();
        // Single-threaded fixture models counts from forgotten guards. No
        // actual reference exists, so exclusive recovery may discard them.
        slot.state.store(1 | REFS, Ordering::Release);
        assert!(catch_unwind(AssertUnwindSafe(|| hive.get(id))).is_err());
        assert_eq!(slot.state.load(Ordering::Acquire), 1 | REFS);
        assert_eq!(hive.take_mut(id), Some(7));
        assert!(hive.is_empty());
    }

    // Leaks the hive: exclusive cleanup (drop, clear, shrink) walks every
    // index below the bound, ~4.3e9 here.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn the_last_slots_below_u32_max_hold_values_and_exhaust_exactly() {
        // The last page is [2^32 - 4096, 2^32); its top slot (index
        // u32::MAX) is NONE and never usable.
        let mut hive = ConcurrentHive::<u64, 4096>::new();
        hive.free.set_next_fresh(MAX_INDICES - 2);
        let a = hive.insert(1);
        let b = hive.insert(2);
        assert_eq!((a.index(), b.index()), (MAX_INDICES - 2, MAX_INDICES - 1));
        assert_eq!(hive.try_insert(3), Err(3));
        assert_eq!(hive.index_bound(), MAX_INDICES);
        assert_eq!((*hive.get(a).unwrap(), *hive.get(b).unwrap()), (1, 2));
        let top = Id::new(MAX_INDICES, b.generation());
        assert!(hive.get(top).is_none() && !hive.remove(top));
        assert_eq!(hive.take(top), None);
        // The top usable slot is reused with a new generation.
        assert_eq!(hive.take(b), Some(2));
        let c = hive.insert(4);
        assert_eq!(c.index(), MAX_INDICES - 1);
        assert!(c != b && hive.get(b).is_none());
        assert_eq!(*hive.get(c).unwrap(), 4);
        assert_eq!(hive.try_insert(5), Err(5));
        assert_eq!(hive.len(), 2);
        // u64 values need no drop; the pages are freed with the process.
        core::mem::forget(hive);
    }

    #[test]
    fn a_live_stack_hint_is_discarded() {
        let hive = ConcurrentHive::<u8, 2>::new();
        let a = hive.insert(1);
        let slot = hive.slot(a.index()).unwrap();
        hive.free.push(a.index(), &slot.next);
        let b = hive.insert(2);
        assert_ne!(a.index(), b.index());
        assert_eq!(*hive.get(a).unwrap(), 1);
        assert_eq!(*hive.get(b).unwrap(), 2);
        assert_eq!(hive.len(), 2);
    }
}
