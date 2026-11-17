//! Paged generational ids with shared allocation and freeing.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::fmt;

use super::atomic::{AtomicU32, Ordering};
use super::count::StripedLen;
use super::free_stack::{FreeStack, block_for};
use super::spine::Spine;
use crate::id_alloc::{Id, IdAllocator, MAX_INDICES, generation_of, live_version};
use crate::page::DEFAULT_PAGE;

// Include the last partial range of u32 page numbers, even for PAGE = 1.
// The design's 27 normal chunks cover only 2^32 - 32 pages.
#[cfg(not(loom))]
const CHUNKS: usize = 28;
#[cfg(loom)]
const CHUNKS: usize = 32;

#[derive(Default)]
struct Slot {
    // Zero is never used for a fresh index, retired for an issued index.
    version: AtomicU32,
    next: AtomicU32,
}

struct Page {
    slots: Box<[Slot]>,
}

/// A thread-safe, paged allocator of generational [`Id`]s.
///
/// Shared access can allocate, free, and check ids. Freed indices are reused
/// through tagged LIFO stacks, one per thread group (a thread with none
/// steals another's); their generations never wrap. As with
/// [`IdAllocator`], ids carry no allocator identity, and the stale-id
/// guarantee applies within one allocation history. Conversions preserve it.
///
/// Pages stay allocated until destruction. [`clear`](Self::clear) requires
/// exclusive access and preserves every index's generation history.
/// Allocation and freeing use CAS retry loops; liveness checks do not retry.
/// The free stack assumes fewer than 2^32 pops during any stalled operation.
pub struct ConcurrentIdAllocator<const PAGE: usize = DEFAULT_PAGE> {
    pages: Spine<Page, CHUNKS>,
    free: FreeStack,
    len: StripedLen,
}

impl<const PAGE: usize> ConcurrentIdAllocator<PAGE> {
    /// An empty allocator, without allocating memory.
    ///
    /// # Panics
    ///
    /// Panics if `PAGE` is zero or a page's slot storage exceeds `isize::MAX`.
    pub fn new() -> Self {
        assert!(
            PAGE > 0 && PAGE <= isize::MAX as usize / core::mem::size_of::<Slot>(),
            "ConcurrentIdAllocator: invalid page capacity"
        );
        Self {
            pages: Spine::new(),
            free: FreeStack::new(),
            len: StripedLen::new(),
        }
    }

    fn slot(&self, index: u32) -> Option<&Slot> {
        self.pages
            .get(index as usize / PAGE)
            .map(|page| &page.slots[index as usize % PAGE])
    }

    fn ensure_slot(&self, index: u32) -> &Slot {
        let page = self.pages.get_or_insert_with(index as usize / PAGE, || {
            Box::new(Page {
                slots: (0..PAGE).map(|_| Slot::default()).collect(),
            })
        });
        &page.slots[index as usize % PAGE]
    }

    /// Issues an id, reusing a free index or reserving a fresh one.
    ///
    /// # Panics
    ///
    /// Panics when all [`MAX_INDICES`] indices are live or retired.
    pub fn allocate(&self) -> Id {
        self.try_allocate().unwrap_or_else(|| {
            panic!("ConcurrentIdAllocator: all {MAX_INDICES} indices are live or retired")
        })
    }

    /// Like [`allocate`](Self::allocate), returning `None` on exhaustion.
    /// A concurrent free may make room immediately after the stack is checked.
    pub fn try_allocate(&self) -> Option<Id> {
        loop {
            let (index, reused) = self.free.next_index(
                |index| &self.slot(index).expect("free index has a page").next,
                || None,
                MAX_INDICES,
                block_for(PAGE),
            )?;
            let slot = self.ensure_slot(index);
            // Acquire observes the previous owner's vacant version after pop.
            // A fresh reservation uniquely owns its never-used zero version.
            let previous = slot.version.load(Ordering::Acquire);
            // The free stack is only a hint: owning an index means winning
            // the version CAS below. If the stack's ABA tag ever wrapped
            // (2^32 pops inside one stalled pop), it could hand out a live
            // or retired index; that index is skipped here, so a wrap can
            // leak indices but can never issue one id twice.
            if previous % 2 == 1 || (reused && previous == 0) {
                continue;
            }
            // Relaxed: accounting only. Increment before publication, so any
            // free of this id is counted after it; undone if the claim is lost.
            self.len.increment(index as usize / PAGE);
            let version = previous + 1;
            // Release publishes this generation; a successful free's Acquire
            // observes the preceding count increment. Acquire on failure:
            // another claimant won, so drop this candidate and retry.
            if slot
                .version
                .compare_exchange(previous, version, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Some(Id::new(index, generation_of(version)));
            }
            self.len.decrement(index as usize / PAGE);
        }
    }

    /// Frees a matching live id, returning whether this call won the free.
    /// Stale, never-issued, and already-freed ids return `false`.
    pub fn free(&self, id: Id) -> bool {
        let Some(live) = live_version(id.generation()) else {
            return false;
        };
        let Some(slot) = self.slot(id.index()) else {
            return false;
        };
        let vacant = live.checked_add(1).unwrap_or(0);
        // AcqRel observes publication (including accounting) and publishes
        // invalidation. Exactly one CAS can win for this live generation.
        // Acquire on failure observes the competing state transition.
        if slot
            .version
            .compare_exchange(live, vacant, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return false;
        }
        // Relaxed: accounting only; decrement before push so a later owner
        // cannot accumulate arbitrarily many pending decrements on this index.
        self.len.decrement(id.index() as usize / PAGE);
        if vacant != 0 {
            self.free.push(id.index(), &slot.next);
        }
        true
    }

    /// Whether this generation is live at the instant its version is read.
    pub fn is_live(&self, id: Id) -> bool {
        let Some(live) = live_version(id.generation()) else {
            return false;
        };
        // Acquire observes generation publication/invalidation. A slot's
        // version never returns to an earlier live value in this history.
        self.slot(id.index())
            .is_some_and(|slot| slot.version.load(Ordering::Acquire) == live)
    }

    /// Number of live ids; approximate while other threads allocate or free.
    /// Exact when no operation is in flight. This is not a publication barrier.
    ///
    /// It sums per-thread counters (per page group without `std`). While
    /// operations are in flight it may combine counters read at different
    /// moments and differ from every count the allocator held; it never
    /// exceeds the number of insertions.
    pub fn len(&self) -> usize {
        self.len.get()
    }

    /// Whether no ids are live; approximate while other threads run.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// One past the highest reserved index, including reservations in flight.
    /// This does not imply that every reserved index's page is installed yet.
    ///
    /// Threads that contend for fresh indices reserve them in blocks of
    /// about one page, so the bound may exceed the indices handed out by up
    /// to one block per thread.
    pub fn index_bound(&self) -> u32 {
        self.free.index_bound()
    }

    /// Frees all live ids, preserving generations so stale ids stay dead.
    /// Keeps pages and retired indices; requires exclusive access.
    pub fn clear(&mut self) {
        for index in 0..self.index_bound() {
            // A reservation whose page allocation unwound has no page and
            // never issued an id.
            let Some(slot) = self.slot(index) else {
                continue;
            };
            // Relaxed: exclusive access excludes all other version writers.
            let version = slot.version.load(Ordering::Relaxed);
            if version % 2 == 1 {
                let freed = self.free(Id::new(index, generation_of(version)));
                debug_assert!(freed);
            }
        }
    }

    /// Converts to the plain allocator, preserving ids, and LIFO reuse order
    /// within each thread group's stack. Indices reserved in a block but
    /// never handed out become free, with generation 1 spent.
    /// Takes O(number of indices ever reserved) time and temporary storage.
    pub fn into_inner(mut self) -> IdAllocator {
        let bound = self.index_bound() as usize;
        let mut unissued = self.free.unissued();
        let mut spare = Vec::new();
        // By index, not by installed page: a reservation whose page
        // allocation unwound has no page and keeps version 0 (retired, as it
        // cannot be told apart from a retired index). Block indices never
        // handed out become vacancies with generation 1 spent.
        let versions: Vec<u32> = (0..bound as u32)
            .map(|index| {
                // Relaxed: self is exclusively owned, with no operations in flight.
                let version = self
                    .slot(index)
                    .map_or(0, |slot| slot.version.load(Ordering::Relaxed));
                if unissued.contains(index) {
                    debug_assert_eq!(version, 0, "unissued index was used");
                    spare.push(index);
                    2
                } else {
                    version
                }
            })
            .collect();
        // The plain allocator pops from the end: stack tops last, spare
        // block indices first (lowest last among them).
        let mut free = self
            .free
            .drain(|index| &self.slot(index).expect("free index has a page").next);
        free.reverse();
        spare.reverse();
        spare.append(&mut free);
        let free = spare;
        IdAllocator::try_from_raw_parts(versions, free)
            .expect("concurrent allocator maintains valid raw parts")
    }
}

impl<const PAGE: usize> From<IdAllocator> for ConcurrentIdAllocator<PAGE> {
    fn from(inner: IdAllocator) -> Self {
        let mut result = Self::new();
        let (versions, free) = inner.raw_parts();
        for &version in versions {
            let index = result
                .free
                .fresh(MAX_INDICES, block_for(PAGE))
                .expect("valid index bound");
            // Release publishes the imported generation, just as allocation does.
            result
                .ensure_slot(index)
                .version
                .store(version, Ordering::Release);
        }
        for &index in free {
            let slot = result.slot(index).expect("imported index has a page");
            result.free.push(index, &slot.next);
        }
        // Relaxed: result is not shared until the conversion returns.
        result.len.set(inner.len());
        result
    }
}

impl<const PAGE: usize> Default for ConcurrentIdAllocator<PAGE> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const PAGE: usize> fmt::Debug for ConcurrentIdAllocator<PAGE> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConcurrentIdAllocator")
            .field("len", &self.len())
            .field("index_bound", &self.index_bound())
            .finish_non_exhaustive()
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;

    #[test]
    fn unissued_block_tail_becomes_free_with_generation_one_spent() {
        let ids: ConcurrentIdAllocator<4> = ConcurrentIdAllocator::new();
        let a = ids.allocate();
        let b = ids.allocate();
        ids.free.force_blocks();
        let c = ids.allocate();
        assert_eq!((a.index(), b.index(), c.index()), (0, 1, 2));
        assert_eq!(ids.index_bound(), 64);
        assert!(ids.free(b));
        let plain = ids.into_inner();
        let (versions, free) = plain.raw_parts();
        assert_eq!(&versions[..3], &[1, 2, 1]);
        assert!(versions[3..].iter().all(|&v| v == 2));
        assert_eq!(free.len(), 62);
        assert_eq!(plain.retired_len(), 0);
        assert!(!plain.is_alive(b) && plain.is_alive(a) && plain.is_alive(c));
        let back = ConcurrentIdAllocator::<4>::from(plain);
        for _ in 0..62 {
            let id = back.allocate();
            assert!(id.index() < 64 && id != b);
        }
        assert_eq!(back.index_bound(), 64);
    }

    #[test]
    fn the_last_indices_below_u32_max_are_issued_freed_and_reused() {
        // Reservation resumes 3 below the limit: the last page is
        // [2^32 - 4096, 2^32), and its top slot (index u32::MAX) is NONE.
        let mut ids: ConcurrentIdAllocator<4096> = ConcurrentIdAllocator::new();
        ids.free.set_next_fresh(MAX_INDICES - 3);
        let got: Vec<Id> = (0..3).map(|_| ids.allocate()).collect();
        let indices: Vec<u32> = got.iter().map(|id| id.index()).collect();
        assert_eq!(indices, [MAX_INDICES - 3, MAX_INDICES - 2, MAX_INDICES - 1]);
        assert_eq!(ids.try_allocate(), None);
        assert_eq!(ids.index_bound(), MAX_INDICES);
        assert!(!ids.is_live(Id::new(MAX_INDICES, got[0].generation())));
        assert!(!ids.free(Id::new(MAX_INDICES, got[0].generation())));
        // The top index goes back on a stack and comes out with a new
        // generation; nothing else is issued.
        assert!(ids.free(got[2]));
        let again = ids.allocate();
        assert_eq!(again.index(), MAX_INDICES - 1);
        assert_ne!(again, got[2]);
        assert!(ids.is_live(again) && !ids.is_live(got[2]));
        assert_eq!(ids.try_allocate(), None);
        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn a_live_index_on_the_free_stack_is_never_reissued() {
        // Simulates the stack corruption an ABA tag wrap could cause: a
        // live index sits on the free stack.
        let ids: ConcurrentIdAllocator<2> = ConcurrentIdAllocator::new();
        let a = ids.allocate();
        let slot = ids.slot(a.index()).unwrap();
        ids.free.push(a.index(), &slot.next);
        let b = ids.allocate();
        let c = ids.allocate();
        assert!(ids.is_live(a) && ids.is_live(b) && ids.is_live(c));
        assert_ne!(a.index(), b.index());
        assert_ne!(a.index(), c.index());
        assert_ne!(b.index(), c.index());
        assert_eq!(ids.len(), 3);
    }
}
