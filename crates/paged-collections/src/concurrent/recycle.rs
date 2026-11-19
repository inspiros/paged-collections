//! Freed pages awaiting reuse, each with the version its slots restart at.
//!
//! Entries are written only under exclusive access (`set`); shared claims
//! only pop them. An entry, once popped, never returns while shared access
//! lasts, so the stack needs neither tags nor links: an atomic cursor over
//! a fixed list hands each entry to exactly one claimer.

use alloc::vec::Vec;
use core::ops::Range;

use super::atomic::{AtomicUsize, Ordering};

pub(crate) struct PageStack {
    /// `(page, floor)`, popped from the end; `floor` is even and non-zero.
    entries: Vec<(usize, u32)>,
    /// Number of entries not yet popped.
    len: AtomicUsize,
}

impl PageStack {
    #[cfg(not(loom))]
    pub(crate) const fn new() -> Self {
        Self {
            entries: Vec::new(),
            len: AtomicUsize::new(0),
        }
    }

    #[cfg(loom)]
    pub(crate) fn new() -> Self {
        Self {
            entries: Vec::new(),
            len: AtomicUsize::new(0),
        }
    }

    /// Hands out one freed page and its floor, or `None` if none is left.
    pub(crate) fn pop(&self) -> Option<(usize, u32)> {
        // Relaxed suffices: the cursor only assigns distinct entries. The
        // entries were written under `&mut self` before this shared borrow
        // began; whatever shared the borrow across threads ordered them.
        let mut n = self.len.load(Ordering::Relaxed);
        loop {
            let rest = n.checked_sub(1)?;
            match self
                .len
                .compare_exchange_weak(n, rest, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => return Some(self.entries[rest]),
                Err(actual) => n = actual,
            }
        }
    }

    /// The entries not yet popped, under exclusive access.
    pub(crate) fn pending(&mut self) -> &[(usize, u32)] {
        // Relaxed: exclusive access rules out concurrent pops.
        let n = self.len.load(Ordering::Relaxed);
        &self.entries[..n]
    }

    /// The floor of freed page `page`, if it is waiting for reuse.
    /// O(log n): entries stay sorted by descending page.
    pub(crate) fn floor_of(&mut self, page: usize) -> Option<u32> {
        let pending = self.pending();
        pending
            .binary_search_by(|&(p, _)| page.cmp(&p))
            .ok()
            .map(|i| pending[i].1)
    }

    /// Reads a pending page's floor in O(log pending pages). A concurrent
    /// pop may race the read.
    pub(crate) fn floor_of_shared(&self, page: usize) -> Option<u32> {
        // Acquire reads the pending prefix. The entries are immutable for
        // this borrow and a shared pop can only shorten that prefix.
        let n = self.len.load(Ordering::Acquire);
        let pending = &self.entries[..n];
        pending
            .binary_search_by(|&(p, _)| page.cmp(&p))
            .ok()
            .map(|i| pending[i].1)
    }

    /// Lists snapshot pages up to `bound`, with an empty slot range for a
    /// page still pending reuse. Each floor is read as its page is visited.
    pub(crate) fn snapshot_pages(
        &self,
        bound: u32,
        page_size: usize,
    ) -> impl ExactSizeIterator<Item = (u32, Range<u32>)> + '_ {
        (0..bound as usize).step_by(page_size).map(move |base| {
            if let Some(floor) = self.floor_of_shared(base / page_size) {
                (floor, 0..0)
            } else {
                let end = base + page_size.min(bound as usize - base);
                (0, base as u32..end as u32)
            }
        })
    }

    /// Replaces the entries under exclusive access; the lowest page pops
    /// first.
    pub(crate) fn set(&mut self, mut entries: Vec<(usize, u32)>) {
        debug_assert!(
            entries
                .iter()
                .all(|&(_, floor)| floor != 0 && floor % 2 == 0)
        );
        entries.sort_unstable_by_key(|&(page, _)| core::cmp::Reverse(page));
        // Relaxed: exclusive access rules out concurrent pops.
        self.len.store(entries.len(), Ordering::Relaxed);
        self.entries = entries;
    }
}
