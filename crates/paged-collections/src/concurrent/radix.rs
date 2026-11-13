//! Sparse stable pages, published once and reclaimed only under exclusive access.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::marker::PhantomData;
use core::{mem, ptr, slice};

use super::atomic::{AtomicPtr, Ordering};

#[cfg(not(loom))]
const DIGIT: u32 = 6;
#[cfg(loom)]
const DIGIT: u32 = 2;
const FANOUT: usize = 1 << DIGIT;
const MASK: usize = FANOUT - 1;

pub(crate) struct RadixSpine<P> {
    root: AtomicPtr<()>,
    owner: PhantomData<Box<P>>,
    // The erased `AtomicPtr<()>` would leave P covariant, and `&self`
    // inserts P: a `RadixSpine<&'static T>` could then be shortened to
    // `RadixSpine<&'a T>`, store a short-lived reference, and later hand it
    // out as `&'static T` (use after free). Spine is invariant through
    // `AtomicPtr<AtomicPtr<P>>`; this marker restores the same here.
    invariant: PhantomData<fn(P) -> P>,
}

struct Leaf<P> {
    number: usize,
    // Inline storage saves one indirection on every successful lookup. The
    // factory's Box is consumed before publication; installed pages never move.
    page: P,
}

struct Branch {
    prefix: usize,
    shift: u32,
    children: [AtomicPtr<()>; FANOUT],
}

fn digit(number: usize, shift: u32) -> usize {
    (number >> shift) & MASK
}

fn prefix(number: usize, shift: u32) -> usize {
    // The highest digit may contain fewer than DIGIT bits.
    number.checked_shr(shift + DIGIT).unwrap_or(0)
}

fn is_leaf(raw: *mut ()) -> bool {
    raw.addr() & 1 != 0
}

fn leaf_ptr<P>(raw: *mut ()) -> *mut Leaf<P> {
    // Keep the allocation's provenance while removing the tag.
    raw.map_addr(|addr| addr & !1).cast()
}

// SAFETY: the directory owns its pages; moving it transfers that ownership,
// and no page reference can outlive the borrow of the directory.
unsafe impl<P: Send> Send for RadixSpine<P> {}
// SAFETY: shared operations only publish immutable ownership links; pages
// are shared as &P, and a losing installer may drop its own unpublished P.
unsafe impl<P: Send + Sync> Sync for RadixSpine<P> {}

impl<P> RadixSpine<P> {
    #[cfg(not(loom))]
    pub(crate) const fn new() -> Self {
        const {
            assert!(mem::align_of::<Leaf<P>>() >= 2);
            assert!(mem::align_of::<Branch>() >= 2);
        }
        Self {
            root: AtomicPtr::new(ptr::null_mut()),
            owner: PhantomData,
            invariant: PhantomData,
        }
    }

    #[cfg(loom)]
    pub(crate) fn new() -> Self {
        const {
            assert!(mem::align_of::<Leaf<P>>() >= 2);
            assert!(mem::align_of::<Branch>() >= 2);
        }
        Self {
            root: AtomicPtr::new(ptr::null_mut()),
            owner: PhantomData,
            invariant: PhantomData,
        }
    }

    pub(crate) fn get(&self, page: usize) -> Option<&P> {
        let mut slot = &self.root;
        loop {
            // Acquire observes the node, page and old-child links published
            // by an installer. Shared access never reclaims any of them.
            let raw = slot.load(Ordering::Acquire);
            if raw.is_null() {
                return None;
            }
            if is_leaf(raw) {
                // SAFETY: the tag identifies an initialized Leaf<P>; its
                // allocation remains live throughout this shared borrow.
                let leaf = unsafe { &*leaf_ptr::<P>(raw) };
                return (leaf.number == page).then_some(&leaf.page);
            }
            // SAFETY: an untagged non-null pointer owns an initialized Branch,
            // which cannot be freed during this shared borrow.
            let branch = unsafe { &*raw.cast::<Branch>() };
            if prefix(page, branch.shift) != branch.prefix {
                return None;
            }
            slot = &branch.children[digit(page, branch.shift)];
        }
    }

    pub(crate) fn get_or_insert_with(&self, page: usize, make: impl FnOnce() -> Box<P>) -> &P {
        let mut slot = &self.root;
        let mut make = Some(make);
        let mut fresh = None;
        // Acquire observes any node already installed in the root slot.
        let mut old = slot.load(Ordering::Acquire);
        loop {
            let representative = if old.is_null() {
                None
            } else if is_leaf(old) {
                // SAFETY: this acquired pointer identifies a live Leaf<P>;
                // published nodes cannot be reclaimed during shared access.
                let leaf = unsafe { &*leaf_ptr::<P>(old) };
                if leaf.number == page {
                    // The guard drops our leaf, if made, when another thread
                    // has installed the same page number first.
                    drop(fresh);
                    return &leaf.page;
                }
                Some(leaf.number)
            } else {
                // SAFETY: the acquired untagged pointer is a live Branch;
                // its prefix and shift never change after publication.
                let branch = unsafe { &*old.cast::<Branch>() };
                if prefix(page, branch.shift) == branch.prefix {
                    slot = &branch.children[digit(page, branch.shift)];
                    // Acquire observes publication into this child slot.
                    old = slot.load(Ordering::Acquire);
                    continue;
                }
                Some(branch.prefix.checked_shl(branch.shift + DIGIT).unwrap_or(0))
            };

            let leaf = fresh.get_or_insert_with(|| {
                let make = make.take().expect("factory is called at most once");
                UnpublishedLeaf(Box::into_raw(Box::new(Leaf {
                    number: page,
                    page: *make(),
                })))
            });
            let new_leaf = leaf.0.cast::<()>().map_addr(|addr| addr | 1);
            let replacement = if let Some(existing) = representative {
                // A prefix mismatch differs above the old branch's digit,
                // so this split also strictly decreases shifts down the tree.
                let shift = (usize::BITS - 1 - (existing ^ page).leading_zeros()) / DIGIT * DIGIT;
                let old_digit = digit(existing, shift);
                let new_digit = digit(page, shift);
                let branch = Box::new(Branch {
                    prefix: prefix(page, shift),
                    shift,
                    children: core::array::from_fn(|i| {
                        AtomicPtr::new(if i == old_digit {
                            old
                        } else if i == new_digit {
                            new_leaf
                        } else {
                            ptr::null_mut()
                        })
                    }),
                });
                Box::into_raw(branch).cast::<()>()
            } else {
                new_leaf
            };

            // Release publishes the initialized leaf/branch and transitively
            // the old child's links. Acquire on failure observes the winner.
            // Shared changes are only null -> child or child -> a branch
            // containing that child, so every installed page stays reachable.
            match slot.compare_exchange(old, replacement, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => {
                    let raw = fresh.take().expect("leaf was just made").into_raw();
                    // SAFETY: the CAS published this initialized leaf, directly
                    // or inside the new branch; its guard relinquished ownership
                    // and shared access prevents reclamation.
                    return unsafe { &(*raw).page };
                }
                Err(winner) => {
                    if representative.is_some() {
                        // SAFETY: this branch was never published. Its atomic
                        // pointers do not own/drop the old child or our leaf.
                        drop(unsafe { Box::from_raw(replacement.cast::<Branch>()) });
                    }
                    // Retry this very slot: it may now hold a branch that we
                    // must descend into. Keep our made leaf across all retries.
                    // A failed strong CAS means another installer progressed;
                    // nothing is removed, so this loop is lock-free.
                    old = winner;
                }
            }
        }
    }

    /// Installed pages in ascending order, weakly consistent across pages.
    /// May miss concurrent installs; every page is yielded at most once.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (usize, &P)> {
        SharedPages {
            inner: Walk::new(&self.root, Ordering::Acquire),
            marker: PhantomData,
        }
    }

    /// Installed pages in ascending order, with exclusive access to each.
    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = (usize, &mut P)> {
        ExclusivePages {
            inner: Walk::new(&self.root, Ordering::Relaxed),
            marker: PhantomData,
        }
    }

    /// Frees every node and page. The root is detached first, and a cleanup
    /// guard keeps freeing the remaining nodes if one page destructor panics.
    /// A second destructor panic during unwinding aborts as usual.
    pub(crate) fn clear(&mut self) {
        // Relaxed suffices: exclusive access rules out readers/installers.
        let raw = self.root.swap(ptr::null_mut(), Ordering::Relaxed);
        let mut reclaim = Reclaim::<P> {
            nodes: Vec::new(),
            owner: PhantomData,
        };
        if !raw.is_null() {
            reclaim.nodes.push(raw);
        }
        reclaim.drain();
    }
}

/// Owns the made leaf until publication, including across failed splits.
struct UnpublishedLeaf<P>(*mut Leaf<P>);

impl<P> UnpublishedLeaf<P> {
    fn into_raw(self) -> *mut Leaf<P> {
        let raw = self.0;
        mem::forget(self);
        raw
    }
}

impl<P> Drop for UnpublishedLeaf<P> {
    fn drop(&mut self) {
        // SAFETY: this guard uniquely owns a Box that was never published.
        drop(unsafe { Box::from_raw(self.0) });
    }
}

/// DFS frames borrow slots, not their contents. Each slot is loaded once.
/// If a split happens after that load, this walk keeps the old child and
/// never visits its new ancestor. If before, it visits the new branch and
/// then its old child once. Parent slots have disjoint ordered ranges, so
/// neither case can duplicate a leaf or put it out of ascending order.
struct Walk<'a> {
    levels: Vec<slice::Iter<'a, AtomicPtr<()>>>,
    ordering: Ordering,
}

impl<'a> Walk<'a> {
    fn new(root: &'a AtomicPtr<()>, ordering: Ordering) -> Self {
        let levels = alloc::vec![slice::from_ref(root).iter()];
        Self { levels, ordering }
    }
}

impl Iterator for Walk<'_> {
    type Item = *mut ();

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let level = self.levels.last_mut()?;
            let Some(slot) = level.next() else {
                self.levels.pop();
                continue;
            };
            // Acquire observes node/page publication for shared iteration;
            // Relaxed suffices when an exclusive iterator borrows the directory.
            let raw = slot.load(self.ordering);
            if raw.is_null() {
                continue;
            }
            if is_leaf(raw) {
                return Some(raw);
            }
            // SAFETY: the owning iterator borrows the directory, preventing
            // reclamation of this installed Branch and its initialized slots.
            let branch = unsafe { &*raw.cast::<Branch>() };
            self.levels.push(branch.children.iter());
        }
    }
}

struct SharedPages<'a, P> {
    inner: Walk<'a>,
    // Match the auto traits of a shared directory borrow, like Spine::iter.
    marker: PhantomData<&'a RadixSpine<P>>,
}

impl<'a, P> Iterator for SharedPages<'a, P> {
    type Item = (usize, &'a P);

    fn next(&mut self) -> Option<Self::Item> {
        let raw = self.inner.next()?;
        // SAFETY: Walk yields only initialized installed leaves, whose
        // allocations remain live for the iterator's shared directory borrow.
        let leaf = unsafe { &*leaf_ptr::<P>(raw) };
        Some((leaf.number, &leaf.page))
    }
}

/// Carries `&mut P`'s auto traits for RadixSpine::iter_mut, like Spine's adapter.
struct ExclusivePages<'a, P> {
    inner: Walk<'a>,
    marker: PhantomData<&'a mut P>,
}

impl<'a, P> Iterator for ExclusivePages<'a, P> {
    type Item = (usize, &'a mut P);

    fn next(&mut self) -> Option<Self::Item> {
        let raw = self.inner.next()?;
        // SAFETY: exclusive access prevents other page borrows or installers;
        // Walk visits each distinct owned leaf once, even across next calls.
        let leaf = unsafe { &mut *leaf_ptr::<P>(raw) };
        Some((leaf.number, &mut leaf.page))
    }
}

/// Detached ownership, with a second drain on unwinding after a page panic.
struct Reclaim<P> {
    nodes: Vec<*mut ()>,
    owner: PhantomData<Box<P>>,
}

impl<P> Reclaim<P> {
    fn drain(&mut self) {
        while let Some(raw) = self.nodes.pop() {
            if is_leaf(raw) {
                // SAFETY: the detached tree uniquely owns every leaf, and
                // popping first prevents a second drop if P::drop panics.
                drop(unsafe { Box::from_raw(leaf_ptr::<P>(raw)) });
            } else {
                // SAFETY: each untagged pointer uniquely owns a Branch in the
                // detached tree; no directory borrows or other owners remain.
                let branch = unsafe { Box::from_raw(raw.cast::<Branch>()) };
                for slot in &branch.children {
                    // Relaxed suffices: this entire tree is exclusively owned.
                    let child = slot.load(Ordering::Relaxed);
                    if !child.is_null() {
                        self.nodes.push(child);
                    }
                }
            }
        }
    }
}

impl<P> Drop for Reclaim<P> {
    fn drop(&mut self) {
        self.drain();
    }
}

impl<P> Drop for RadixSpine<P> {
    fn drop(&mut self) {
        self.clear();
    }
}

#[cfg(test)]
impl<P> RadixSpine<P> {
    /// Checks a quiescent tree; callers must first join any installers.
    pub(super) fn check(&self) {
        let mut nodes = Vec::new();
        let mut numbers = Vec::new();
        // Acquire observes the installed root's initialization.
        let root = self.root.load(Ordering::Acquire);
        if !root.is_null() {
            nodes.push((root, None));
        }
        while let Some((raw, parent)) = nodes.pop() {
            let (number, shift) = if is_leaf(raw) {
                // SAFETY: this acquired pointer identifies an installed leaf,
                // which stays allocated throughout the check's shared borrow.
                let leaf = unsafe { &*leaf_ptr::<P>(raw) };
                numbers.push(leaf.number);
                (leaf.number, None)
            } else {
                // SAFETY: the acquired untagged pointer is an installed Branch
                // and its allocation stays live during this shared borrow.
                let branch = unsafe { &*raw.cast::<Branch>() };
                assert_eq!(branch.shift % DIGIT, 0);
                assert!(branch.shift < usize::BITS);
                let number = branch.prefix.checked_shl(branch.shift + DIGIT).unwrap_or(0);
                assert_eq!(prefix(number, branch.shift), branch.prefix);
                let mut occupied = 0;
                for (i, slot) in branch.children.iter().enumerate().rev() {
                    // Acquire observes the child's initialization.
                    let child = slot.load(Ordering::Acquire);
                    if !child.is_null() {
                        occupied += 1;
                        nodes.push((child, Some((branch.prefix, branch.shift, i))));
                    }
                }
                assert!(occupied >= 2);
                (number, Some(branch.shift))
            };
            if let Some((parent_prefix, parent_shift, index)) = parent {
                assert_eq!(prefix(number, parent_shift), parent_prefix);
                assert_eq!(digit(number, parent_shift), index);
                if let Some(shift) = shift {
                    assert!(shift < parent_shift);
                }
            }
        }
        assert!(numbers.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(self.iter().map(|(n, _)| n).collect::<Vec<_>>(), numbers);
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use alloc::collections::BTreeMap;
    use alloc::sync::Arc;
    use core::cell::Cell;
    use core::sync::atomic::AtomicUsize;
    use proptest::prelude::*;

    fn assert_contents(spine: &RadixSpine<u32>, expected: &BTreeMap<usize, u32>, probe: usize) {
        for (&number, value) in expected {
            assert_eq!(spine.get(number), Some(value));
        }
        for number in [0, 1, usize::MAX, probe, probe.wrapping_add(1), probe ^ 1] {
            assert_eq!(spine.get(number), expected.get(&number));
        }
        let absent = (0..).find(|n| !expected.contains_key(n)).unwrap();
        assert!(spine.get(absent).is_none());
        assert_eq!(
            spine
                .iter()
                .map(|(n, value)| (n, *value))
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|(&n, &value)| (n, value))
                .collect::<Vec<_>>()
        );
        spine.check();
    }

    #[test]
    fn random_operations_match_btree_map() {
        let config = proptest::test_runner::Config {
            cases: if cfg!(miri) { 4 } else { 256 },
            failure_persistence: None,
            ..proptest::test_runner::Config::default()
        };
        let operations = proptest::collection::vec(
            (0_u8..8, any::<usize>(), 0_u8..5, any::<u32>()),
            1..if cfg!(miri) { 16 } else { 160 },
        );
        proptest::test_runner::TestRunner::new(config)
            .run(&operations, |operations| {
                let mut spine = RadixSpine::new();
                let mut expected = BTreeMap::new();
                for (op, raw, region, value) in operations {
                    let number = match region {
                        0 => raw % 256,
                        1 => usize::MAX - raw % 256,
                        2 => ((usize::MAX / 3) & !MASK) | (raw & MASK),
                        3 => (usize::MAX & !0xfff) | (raw & 0xfff),
                        _ => raw,
                    };
                    match op {
                        0..=4 => {
                            let called = Cell::new(false);
                            let present = expected.contains_key(&number);
                            // Moving the Box into the closure also exercises a
                            // factory that only implements FnOnce.
                            let made = Box::new(value);
                            let page = spine.get_or_insert_with(number, || {
                                assert!(!called.replace(true));
                                made
                            });
                            assert_eq!(*page, *expected.entry(number).or_insert(value));
                            assert_eq!(called.get(), !present);
                        }
                        5 => assert_eq!(spine.get(number), expected.get(&number)),
                        6 => {
                            for (_, page) in spine.iter_mut() {
                                *page ^= value;
                            }
                            for page in expected.values_mut() {
                                *page ^= value;
                            }
                        }
                        _ => {
                            spine.clear();
                            expected.clear();
                        }
                    }
                    assert_contents(&spine, &expected, number);
                }
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn every_digit_and_the_partial_top_digit() {
        let spine = RadixSpine::new();
        let mut expected = BTreeMap::new();
        let numbers: Vec<_> = (0..usize::BITS)
            .step_by(DIGIT as usize)
            .flat_map(|shift| {
                let boundary = 1_usize << shift;
                [boundary - 1, boundary, boundary + 1]
            })
            .chain([usize::MAX, usize::MAX - 1])
            .collect();
        // The reversed order exercises splitting above branches as well as
        // extending their subtrees at progressively lower digits.
        for (i, number) in numbers.into_iter().rev().enumerate() {
            let value = i as u32;
            assert_eq!(
                *spine.get_or_insert_with(number, || Box::new(value)),
                *expected.entry(number).or_insert(value)
            );
            assert_contents(&spine, &expected, number);
        }
    }

    #[test]
    fn exclusive_iteration_clear_and_reuse() {
        const fn empty() -> RadixSpine<u32> {
            RadixSpine::new()
        }

        let mut spine = empty();
        assert!(spine.get(usize::MAX).is_none());
        assert_eq!(spine.iter().count(), 0);
        assert_eq!(spine.iter_mut().count(), 0);
        spine.clear();
        let numbers = [0, 1, 64, 1 << (usize::BITS - 1), usize::MAX];
        for number in numbers.into_iter().rev() {
            spine.get_or_insert_with(number, || Box::new(1));
        }
        let addresses: Vec<_> = spine.iter().map(|(_, p)| ptr::from_ref(p)).collect();
        // Hold all mutable references together to exercise disjointness.
        let pages: Vec<_> = spine.iter_mut().collect();
        for (i, (number, page)) in pages.into_iter().enumerate() {
            assert_eq!(number, numbers[i]);
            assert_eq!(ptr::from_mut(page).cast_const(), addresses[i]);
            *page += i as u32;
        }
        assert_eq!(
            spine.iter().map(|(_, page)| *page).collect::<Vec<_>>(),
            [1, 2, 3, 4, 5]
        );
        spine.check();
        spine.clear();
        spine.check();
        for number in numbers {
            assert!(spine.get(number).is_none());
        }
        assert_eq!(*spine.get_or_insert_with(usize::MAX, || Box::new(9)), 9);
        spine.check();
    }

    #[test]
    fn zero_sized_pages_and_auto_traits() {
        fn assert_send<T: Send>(_: T) {}
        fn assert_sync<T: Sync>(_: &T) {}

        let spine = RadixSpine::new();
        for number in [0, usize::MAX] {
            spine.get_or_insert_with(number, || Box::new(()));
        }
        assert_eq!(
            spine.iter().map(|(n, _)| n).collect::<Vec<_>>(),
            [0, usize::MAX]
        );
        spine.check();
        assert_sync(&spine);
        assert_sync(&spine.iter());
        assert_send(spine.iter());
        assert_send(spine);

        // A Cell page is Send but not Sync. Exclusive iteration must retain
        // the auto traits of &mut P rather than those of a shared directory.
        let mut spine = RadixSpine::<Cell<u32>>::new();
        spine.get_or_insert_with(0, || Box::new(Cell::new(7)));
        assert_send(spine.iter_mut());
        assert_send(spine);
    }

    struct CountedPage {
        number: usize,
        dropped: Arc<AtomicUsize>,
    }

    impl Drop for CountedPage {
        fn drop(&mut self) {
            assert_eq!(self.dropped.fetch_add(1, Ordering::Relaxed), 0);
        }
    }

    #[test]
    fn each_installed_page_is_dropped_once() {
        let mut spine = RadixSpine::new();
        let mut counters = Vec::new();
        for number in [0, 64, usize::MAX, usize::MAX - 1] {
            let dropped = Arc::new(AtomicUsize::new(0));
            counters.push(dropped.clone());
            spine.get_or_insert_with(number, || Box::new(CountedPage { number, dropped }));
            let page = spine.get_or_insert_with(number, || panic!("factory must stay lazy"));
            assert_eq!(page.number, number);
        }
        assert!(counters.iter().all(|n| n.load(Ordering::Relaxed) == 0));
        spine.clear();
        assert!(counters.iter().all(|n| n.load(Ordering::Relaxed) == 1));
        let dropped = Arc::new(AtomicUsize::new(0));
        let reused = dropped.clone();
        spine.get_or_insert_with(usize::MAX, || {
            Box::new(CountedPage {
                number: usize::MAX,
                dropped: reused,
            })
        });
        drop(spine);
        assert_eq!(dropped.load(Ordering::Relaxed), 1);
        assert!(counters.iter().all(|n| n.load(Ordering::Relaxed) == 1));
    }

    #[cfg(feature = "std")]
    #[test]
    fn clear_keeps_freeing_after_a_page_panics() {
        use std::panic::{AssertUnwindSafe, catch_unwind};

        struct PanickingPage {
            dropped: Arc<AtomicUsize>,
            panic: bool,
        }

        impl Drop for PanickingPage {
            fn drop(&mut self) {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                assert!(!self.panic, "page destructor failed");
            }
        }

        let mut spine = RadixSpine::new();
        let counters: Vec<_> = (0..3).map(|_| Arc::new(AtomicUsize::new(0))).collect();
        for (i, number) in [0, 1, usize::MAX].into_iter().enumerate() {
            spine.get_or_insert_with(number, || {
                Box::new(PanickingPage {
                    dropped: counters[i].clone(),
                    // Reclamation visits the highest child first, leaving
                    // both other leaves for the unwind cleanup guard.
                    panic: number == usize::MAX,
                })
            });
        }
        assert!(catch_unwind(AssertUnwindSafe(|| spine.clear())).is_err());
        assert!(counters.iter().all(|n| n.load(Ordering::Relaxed) == 1));
        assert_eq!(spine.iter().count(), 0);
        spine.check();
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                spine.get_or_insert_with(3, || panic!("factory failed"));
            }))
            .is_err()
        );
        spine.check();
        spine.get_or_insert_with(4, || {
            Box::new(PanickingPage {
                dropped: counters[0].clone(),
                panic: false,
            })
        });
        drop(spine);
        assert_eq!(counters[0].load(Ordering::Relaxed), 2);
    }

    #[cfg(feature = "std")]
    #[test]
    fn overlapping_thread_installs_keep_one_page_per_number() {
        use std::sync::Barrier;
        use std::thread;

        let spine = Arc::new(RadixSpine::new());
        let gate = Arc::new(Barrier::new(8));
        let numbers: Arc<Vec<_>> = Arc::new(
            (0..64)
                .flat_map(|n| [n, usize::MAX - n, ((usize::MAX / 3) & !MASK) | n])
                .collect(),
        );
        let threads: Vec<_> = (0..8)
            .map(|thread_number| {
                let spine = spine.clone();
                let gate = gate.clone();
                let numbers = numbers.clone();
                thread::spawn(move || {
                    let mut made = Vec::new();
                    let mut install = |number, first| {
                        let page = spine.get_or_insert_with(number, || {
                            let dropped = Arc::new(AtomicUsize::new(0));
                            made.push(dropped.clone());
                            if first {
                                // Force eight factories for the first page,
                                // so losing allocations are always exercised.
                                gate.wait();
                            }
                            Box::new(CountedPage { number, dropped })
                        });
                        assert_eq!(page.number, number);
                        (number, ptr::from_ref(page).addr())
                    };
                    let mut addresses = alloc::vec![install(numbers[0], true)];
                    for offset in 1..numbers.len() {
                        let i = 1 + (offset - 1 + thread_number * 17) % (numbers.len() - 1);
                        addresses.push(install(numbers[i], false));
                    }
                    addresses.sort_unstable();
                    (addresses, made)
                })
            })
            .collect();
        let results: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
        for (addresses, _) in &results {
            assert_eq!(*addresses, results[0].0);
        }
        assert_eq!(spine.iter().count(), numbers.len());
        spine.check();
        let counters: Vec<_> = results.into_iter().flat_map(|(_, made)| made).collect();
        assert!(counters.len() >= numbers.len() + 7);
        let dropped: usize = counters.iter().map(|n| n.load(Ordering::Relaxed)).sum();
        assert_eq!(counters.len() - dropped, numbers.len());
        assert!(counters.iter().all(|n| n.load(Ordering::Relaxed) <= 1));
        let mut spine = Arc::try_unwrap(spine).ok().unwrap();
        spine.clear();
        assert!(counters.iter().all(|n| n.load(Ordering::Relaxed) == 1));
        drop(spine);
        assert!(counters.iter().all(|n| n.load(Ordering::Relaxed) == 1));
    }
}
