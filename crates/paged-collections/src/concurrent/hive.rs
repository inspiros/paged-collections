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

use super::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering, UnsafeCell, spin_loop};
use super::free_stack::FreeStack;
use super::spine::Spine;
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
        Self {
            state: AtomicU64::new(0),
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
/// Pages stay allocated until destruction. Shared iteration is weakly
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
    len: AtomicUsize,
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

    fn busy<'a>(&'a self, slot: &'a Slot<T>, index: u32, vacant: u64) -> Busy<'a, T, PAGE> {
        Busy {
            hive: self,
            slot,
            index,
            vacant,
            armed: true,
        }
    }

    fn claim(&self) -> Option<(Busy<'_, T, PAGE>, u32)> {
        loop {
            let reused = self
                .free
                .pop(|index| &self.slot(index).expect("free index has a page").next);
            let index = match reused {
                Some(index) => index,
                None => self.free.fresh(MAX_INDICES)?,
            };
            let slot = self.ensure_slot(index);
            // Acquire observes a completed finalizer, independently of the
            // free-stack hint. No flags or references may remain at claim.
            let old = slot.state.load(Ordering::Acquire);
            if old & !VERSION != 0 || old & 1 != 0 || (reused.is_some() && old == 0) {
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

    fn publish(&self, mut owner: Busy<'_, T, PAGE>, live: u32, value: T) -> Id {
        owner.slot.value.with_mut(|cell| {
            // SAFETY: the claim CAS gave owner sole BUSY ownership of this
            // uninitialized cell. MaybeUninit<T> has T's layout/alignment.
            unsafe { ptr::write(cell.cast::<T>(), value) };
        });
        // Relaxed accounting precedes publication so even a guessed id removed
        // immediately after publication cannot decrement an uncounted value.
        self.len.fetch_add(1, Ordering::Relaxed);
        // Release publishes the initialized value and preceding accounting;
        // every successful reader/remover CAS acquires that publication.
        owner.slot.state.store(u64::from(live), Ordering::Release);
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
        // Acquire observes publication before inspecting the version/flags.
        let mut word = slot.state.load(Ordering::Acquire);
        loop {
            let live = word as u32;
            if live & 1 == 0
                || word & (BUSY | MARKED) != 0
                || expected.is_some_and(|version| version != live)
            {
                return None;
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
                    return Some(Ref {
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
                    self.len.fetch_sub(1, Ordering::Relaxed);
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
        self.len.fetch_sub(1, Ordering::Relaxed);
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
    pub fn len(&self) -> usize {
        // Relaxed: accounting does not grant page or value access.
        self.len.load(Ordering::Relaxed)
    }

    /// Whether the hive is empty; approximate during shared operations.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// One past the highest reserved index, including insertions in flight.
    /// A reserved index's page may not have been installed yet.
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

    #[cfg(all(test, loom))]
    pub(super) fn inject_busy_hint(&self, index: u32) {
        let slot = self.slot(index).unwrap();
        assert_ne!(slot.state.load(Ordering::Acquire) & BUSY, 0);
        // Models a stale link installed after tag wrap, without 2^32 pops.
        self.free.push(index, &slot.next);
    }
}

impl<T, const PAGE: usize> Default for ConcurrentHive<T, PAGE> {
    fn default() -> Self {
        Self::new()
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
