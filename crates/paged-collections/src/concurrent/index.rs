//! Flat-like index bounds and contention backoff shared by concurrent maps.

use super::atomic::{AtomicPtr, spin_loop};

#[cfg(not(loom))]
pub(super) const INDEX_CHUNKS: usize = 28;
#[cfg(loom)]
pub(super) const INDEX_CHUNKS: usize = 32;
#[cfg(not(loom))]
const SPINE_BASE: usize = 32;
#[cfg(loom)]
const SPINE_BASE: usize = 2;

/// Checks both the spine range and the allocation layout of its chunk.
pub(super) fn page_number<P, const PAGE: usize>(index: usize) -> Option<usize> {
    let page = index / PAGE;
    let k = ((page / SPINE_BASE) + 1).ilog2() as usize;
    let cells = 1usize
        .checked_shl(k as u32)
        .and_then(|factor| SPINE_BASE.checked_mul(factor));
    (k < INDEX_CHUNKS
        && cells.is_some_and(|n| n <= isize::MAX as usize / core::mem::size_of::<AtomicPtr<P>>()))
    .then_some(page)
}

pub(super) fn retry(spins: &mut u32) {
    *spins = spins.saturating_add(1);
    #[cfg(all(feature = "std", not(loom)))]
    if *spins >= 64 {
        std::thread::yield_now();
        return;
    }
    spin_loop();
}
