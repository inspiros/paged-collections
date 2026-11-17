//! A length counter split over cache-line-padded stripes.
//!
//! One shared counter serializes every insertion of a new key on one cache
//! line. Here each update goes to the stripe chosen by its page number, so
//! writers on different pages usually touch different lines; `get` sums all
//! stripes.
//!
//! Callers must count each entry's insertion and removal on the SAME stripe
//! (pass the same page number). Each stripe then counts entries that exist
//! in its pages, and its modification order follows the page lock (or slot
//! CAS) that orders an insertion before its removal, so no stripe ever goes
//! below zero and the sum of any loads cannot wrap. Like a single Relaxed
//! counter, `get` is exact when no update is in flight and otherwise weakly
//! consistent: it may combine stripes read at different moments.

use super::atomic::{AtomicUsize, Ordering};

#[cfg(not(loom))]
const STRIPES: usize = 32;
// Two stripes keep loom models small while still splitting.
#[cfg(loom)]
const STRIPES: usize = 2;

// 128 bytes: adjacent-line prefetching pairs 64-byte lines on x86.
#[repr(align(128))]
struct Stripe(AtomicUsize);

pub(crate) struct StripedLen {
    stripes: [Stripe; STRIPES],
}

impl StripedLen {
    pub(crate) fn new() -> Self {
        Self {
            stripes: core::array::from_fn(|_| Stripe(AtomicUsize::new(0))),
        }
    }

    fn stripe(&self, page: usize) -> &AtomicUsize {
        &self.stripes[page % STRIPES].0
    }

    pub(crate) fn increment(&self, page: usize) {
        // Relaxed: a count grants no access to entries.
        self.stripe(page).fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn decrement(&self, page: usize) {
        self.stripe(page).fetch_sub(1, Ordering::Relaxed);
    }

    pub(crate) fn get(&self) -> usize {
        self.stripes
            .iter()
            .map(|stripe| stripe.0.load(Ordering::Relaxed))
            .fold(0, usize::wrapping_add)
    }

    pub(crate) fn reset(&mut self) {
        for stripe in &mut self.stripes {
            // Relaxed: exclusive access.
            stripe.0.store(0, Ordering::Relaxed);
        }
    }
}
