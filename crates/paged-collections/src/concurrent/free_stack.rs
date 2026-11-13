//! Intrusive free indices and bounded fresh-index reservation.
//!
//! Callers must push an index only while owning it, and at most once before
//! it is popped again. Links must stay allocated while shared access exists.
//! A tag increments on every pop, which defeats ABA unless 2^32 pops happen
//! inside one stalled pop. After such a wrap a pop may return an index that
//! is not free. So **a pop is only a hint**: the caller must claim the index
//! with a compare-exchange on that slot's own state and discard the index if
//! the claim fails. Soundness must never rest on the tag.

use super::atomic::{AtomicU32, AtomicU64, CachePadded, Ordering, spin_loop};

pub(super) const NONE: u32 = u32::MAX;

fn retry(spins: &mut u32) {
    *spins = spins.saturating_add(1);
    #[cfg(all(feature = "std", not(loom)))]
    if *spins >= 64 {
        std::thread::yield_now();
        return;
    }
    spin_loop();
}

pub(crate) struct FreeStack {
    head: CachePadded<AtomicU64>,
    next_fresh: AtomicU32,
}

impl FreeStack {
    #[cfg(not(loom))]
    pub(crate) const fn new() -> Self {
        Self {
            head: CachePadded(AtomicU64::new(NONE as u64)),
            next_fresh: AtomicU32::new(0),
        }
    }

    #[cfg(loom)]
    pub(crate) fn new() -> Self {
        Self {
            head: CachePadded(AtomicU64::new(NONE as u64)),
            next_fresh: AtomicU32::new(0),
        }
    }

    pub(crate) fn push(&self, index: u32, next: &AtomicU32) {
        debug_assert_ne!(index, NONE);
        // Acquire observes the previously published stack and its links.
        let mut head = self.head.0.load(Ordering::Acquire);
        let mut spins = 0;
        loop {
            // Release publishes the link; only the owning pusher writes it.
            // Stalled poppers may still read it, so it must remain atomic.
            next.store(head as u32, Ordering::Release);
            let new = (head & !(u32::MAX as u64)) | u64::from(index);
            // Release publishes the link and prior slot state. Acquire (also
            // on failure) observes the new head before preparing another link.
            match self
                .head
                .0
                .compare_exchange_weak(head, new, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => return,
                Err(actual) => head = actual,
            }
            retry(&mut spins);
        }
    }

    pub(crate) fn pop<'a>(&self, next: impl Fn(u32) -> &'a AtomicU32) -> Option<u32> {
        // Acquire observes links and slot state published by push.
        let mut head = self.head.0.load(Ordering::Acquire);
        let mut spins = 0;
        loop {
            let index = head as u32;
            if index == NONE {
                return None;
            }
            // Acquire reads the published link. It may already be stale after
            // another pop/push, but the changed tag makes our CAS fail.
            let following = next(index).load(Ordering::Acquire);
            let tag = ((head >> 32) as u32).wrapping_add(1);
            let new = (u64::from(tag) << 32) | u64::from(following);
            // Acquire transfers ownership and sees the pusher's slot writes;
            // Release carries that publication to later head readers/poppers.
            match self
                .head
                .0
                .compare_exchange_weak(head, new, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => return Some(index),
                Err(actual) => head = actual,
            }
            retry(&mut spins);
        }
    }

    pub(crate) fn fresh(&self, limit: u32) -> Option<u32> {
        // Relaxed suffices: this counter only assigns distinct indices; page
        // and version publication use separate synchronization.
        let mut index = self.next_fresh.load(Ordering::Relaxed);
        let mut spins = 0;
        loop {
            if index >= limit {
                return None;
            }
            match self.next_fresh.compare_exchange_weak(
                index,
                index + 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Some(index),
                Err(actual) => index = actual,
            }
            retry(&mut spins);
        }
    }

    pub(crate) fn index_bound(&self) -> u32 {
        // Relaxed: a reservation bound, not a page-initialization barrier.
        self.next_fresh.load(Ordering::Relaxed)
    }

    /// Sets the fresh-index boundary during exclusive construction only.
    pub(crate) fn set_next_fresh(&mut self, bound: u32) {
        debug_assert_eq!(self.index_bound(), 0);
        // Relaxed: the newly built owner has not been shared or reserved from.
        self.next_fresh.store(bound, Ordering::Relaxed);
    }
}
