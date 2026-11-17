//! A length counter split over cache-line-padded stripes.
//!
//! One shared counter serializes every counted update on one cache line.
//! Here each update goes to the calling thread's stripe, so a thread keeps
//! updating a line it already owns; `get` sums all stripes. Threads take
//! stripes round-robin on first use, so threads that start one after another
//! share a stripe only after `STRIPES` assignments. Without `std`, and under
//! loom (whose models must be deterministic), the caller's hint (a page
//! number) picks the stripe instead.
//!
//! An entry's removal may be counted on another stripe than its insertion,
//! or even before it (the atomic maps count after their slot CAS), so a
//! stripe may go below zero. Stripes therefore wrap, and `get` reads their
//! wrapping sum as signed, clamped at zero. Stripes are 64-bit on every
//! target, so the signed range covers any real count, even of a 32-bit
//! direct map with zero-sized values. The sum is exact when no update
//! is in flight. Otherwise it may combine stripes read at different moments
//! and so differ from every count the container held, though it stays
//! between 0 and the number of insertions counted by the end of the call.

use super::atomic::{AtomicU64, Ordering};

#[cfg(not(loom))]
const STRIPES: usize = 32;
// Two stripes keep loom models small while still splitting.
#[cfg(loom)]
const STRIPES: usize = 2;

// 128 bytes: adjacent-line prefetching pairs 64-byte lines on x86.
#[repr(align(128))]
struct Stripe(AtomicU64);

pub(crate) struct StripedLen {
    stripes: [Stripe; STRIPES],
}

/// The calling thread's stripe, assigned round-robin on its first update.
#[cfg(all(feature = "std", not(loom)))]
#[inline]
fn stripe_index(_hint: usize) -> usize {
    use core::cell::Cell;
    use core::sync::atomic::AtomicUsize;
    // Shared by all counters: a thread uses the same stripe in each of them.
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    std::thread_local! {
        static INDEX: Cell<usize> = const { Cell::new(usize::MAX) };
    }
    INDEX
        .try_with(|index| {
            if index.get() == usize::MAX {
                // Relaxed: only distinctness matters, not ordering.
                index.set(NEXT.fetch_add(1, Ordering::Relaxed) % STRIPES);
            }
            index.get()
        })
        // During thread teardown any stripe is correct, just slower.
        .unwrap_or(0)
}

#[cfg(not(all(feature = "std", not(loom))))]
#[inline]
fn stripe_index(hint: usize) -> usize {
    hint % STRIPES
}

impl StripedLen {
    pub(crate) fn new() -> Self {
        Self {
            stripes: core::array::from_fn(|_| Stripe(AtomicU64::new(0))),
        }
    }

    /// Counts one insertion; `hint` picks the stripe only without `std`.
    #[inline]
    pub(crate) fn increment(&self, hint: usize) {
        // Relaxed: a count grants no access to entries.
        self.stripes[stripe_index(hint)]
            .0
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Counts one removal; any stripe is correct, as stripes wrap.
    #[inline]
    pub(crate) fn decrement(&self, hint: usize) {
        self.stripes[stripe_index(hint)]
            .0
            .fetch_sub(1, Ordering::Relaxed);
    }

    pub(crate) fn get(&self) -> usize {
        let sum = self
            .stripes
            .iter()
            .map(|stripe| stripe.0.load(Ordering::Relaxed))
            .fold(0, u64::wrapping_add);
        // A sum past i64::MAX can only be a transient deficit: no container
        // holds 2^63 entries.
        usize::try_from((sum as i64).max(0)).unwrap_or(usize::MAX)
    }

    /// Sets the count under exclusive access.
    pub(crate) fn set(&mut self, len: usize) {
        for (i, stripe) in self.stripes.iter_mut().enumerate() {
            // Relaxed: exclusive access.
            stripe
                .0
                .store(if i == 0 { len as u64 } else { 0 }, Ordering::Relaxed);
        }
    }

    pub(crate) fn reset(&mut self) {
        self.set(0);
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;

    fn add(len: &StripedLen, stripe: usize, delta: isize) {
        len.stripes[stripe]
            .0
            .fetch_add(delta as u64, Ordering::Relaxed);
    }

    #[test]
    fn stripes_below_zero_sum_exactly() {
        let mut len = StripedLen::new();
        // Inserted on stripe 0, removed on stripe 1: stripe 1 wraps.
        add(&len, 0, 3);
        add(&len, 1, -2);
        assert_eq!(len.get(), 1);
        add(&len, 0, -1);
        assert_eq!(len.get(), 0);
        len.set(7);
        assert_eq!(len.get(), 7);
        // Above isize::MAX on 32-bit targets, where a direct map of
        // zero-sized values can hold this many entries; never a deficit.
        len.set(u32::MAX as usize);
        assert_eq!(len.get(), u32::MAX as usize);
        len.increment(0);
        assert_eq!(len.get() as u64, u64::from(u32::MAX) + 1);
        len.reset();
        assert_eq!(len.get(), 0);
    }

    #[test]
    fn transient_deficit_reads_as_zero() {
        let len = StripedLen::new();
        // A removal counted before its insertion, or a snapshot that saw a
        // removal but missed the matching insertion on another stripe.
        add(&len, 1, -1);
        assert_eq!(len.get(), 0);
        add(&len, 0, 1);
        assert_eq!(len.get(), 0);
    }

    #[cfg(feature = "std")]
    #[test]
    fn cross_thread_removals_cancel() {
        let len = StripedLen::new();
        let run = |n: usize, f: fn(&StripedLen, usize)| {
            std::thread::scope(|scope| {
                scope
                    .spawn(|| {
                        (0..n).for_each(|_| f(&len, 0));
                        stripe_index(0)
                    })
                    .join()
                    .unwrap()
            })
        };
        let inserter = run(1000, StripedLen::increment);
        let remover = run(400, StripedLen::decrement);
        assert_eq!(len.get(), 600);
        // Other tests' threads also take stripes, so the two may coincide.
        if inserter != remover {
            assert_eq!(
                len.stripes[remover].0.load(Ordering::Relaxed),
                400u64.wrapping_neg()
            );
        }
    }
}
