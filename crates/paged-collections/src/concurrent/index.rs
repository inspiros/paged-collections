//! Flat-like index bounds and contention backoff shared by concurrent maps.

use super::atomic::spin_loop;
use super::spine::Spine;

#[cfg(not(loom))]
pub(super) const INDEX_CHUNKS: usize = 28;
#[cfg(loom)]
pub(super) const INDEX_CHUNKS: usize = 32;
/// Checks both the spine range and the allocation layout of its chunk.
pub(super) fn page_number<P, const PAGE: usize>(index: usize) -> Option<usize> {
    let page = index / PAGE;
    Spine::<P, INDEX_CHUNKS>::can_allocate_page(page).then_some(page)
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
