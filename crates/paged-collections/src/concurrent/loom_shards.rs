//! Bounded explorations of the sharded free stacks: stealing, the nonempty
//! mask, and fresh blocks.

#[cfg(not(feature = "std"))]
extern crate std;

use alloc::vec::Vec;
use loom::sync::Arc;
use loom::thread;
use std::sync::atomic::{AtomicBool as StdAtomicBool, Ordering as StdOrdering};

use super::ConcurrentIdAllocator;
use super::atomic::AtomicU32;
use super::free_stack::{FreeStack, set_loom_shard};

fn model(f: impl Fn() + Send + Sync + 'static) {
    let mut builder = loom::model::Builder::new();
    if builder.preemption_bound.is_none() {
        builder.preemption_bound = Some(2);
    }
    builder.max_branches = 10_000;
    builder.check(f);
}

type Links = Arc<[AtomicU32; 3]>;

fn links() -> Links {
    Arc::new([AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0)])
}

/// Pops the calling thread's shard, then steals, until both are empty.
fn take_all(stack: &FreeStack, links: &Links) -> Vec<u32> {
    let next = |i: u32| &links[i as usize];
    let mut got = Vec::new();
    while let Some(i) = stack.pop(next).or_else(|| stack.steal(next)) {
        got.push(i);
    }
    got
}

#[test]
fn a_thief_and_the_owner_split_one_stack() {
    static SPLIT: StdAtomicBool = StdAtomicBool::new(false);
    SPLIT.store(false, StdOrdering::Relaxed);
    model(|| {
        let stack = Arc::new(FreeStack::new());
        let links = links();
        // Reserve the indices, as callers always do before pushing.
        for _ in 0..3 {
            stack.fresh(3, 1);
        }
        for i in 0..3 {
            stack.push(i, &links[i as usize]);
        }
        let (s, l) = (stack.clone(), links.clone());
        let thief = thread::spawn(move || {
            set_loom_shard(1);
            let next = |i: u32| &l[i as usize];
            let mut got: Vec<u32> = s.steal(next).into_iter().collect();
            while let Some(i) = s.pop(next) {
                got.push(i);
            }
            got
        });
        let (s, l) = (stack.clone(), links.clone());
        let owner = thread::spawn(move || {
            let next = |i: u32| &l[i as usize];
            let mut got = Vec::new();
            while let Some(i) = s.pop(next) {
                got.push(i);
            }
            got
        });
        let stolen = thief.join().unwrap();
        let popped = owner.join().unwrap();
        if !stolen.is_empty() && !popped.is_empty() {
            SPLIT.store(true, StdOrdering::Relaxed);
        }
        let mut all = stolen;
        all.extend(popped);
        all.extend(take_all(&stack, &links));
        all.sort_unstable();
        assert_eq!(all, [0, 1, 2]);
    });
    assert!(
        SPLIT.load(StdOrdering::Relaxed),
        "no schedule split the stack"
    );
}

#[test]
fn a_push_and_a_clearing_thief_never_strand_an_index() {
    static CLEARED_EMPTY: StdAtomicBool = StdAtomicBool::new(false);
    CLEARED_EMPTY.store(false, StdOrdering::Relaxed);
    model(|| {
        let stack = Arc::new(FreeStack::new());
        let links = links();
        stack.fresh(1, 1);
        // Leave shard 1's bit set over an empty stack, so the pusher skips
        // setting it and the thief clears it.
        set_loom_shard(1);
        stack.push(0, &links[0]);
        assert_eq!(stack.pop(|i| &links[i as usize]), Some(0));
        set_loom_shard(0);
        let (s, l) = (stack.clone(), links.clone());
        let pusher = thread::spawn(move || {
            set_loom_shard(1);
            s.push(0, &l[0]);
        });
        let (s, l) = (stack.clone(), links.clone());
        let thief = thread::spawn(move || s.steal(|i| &l[i as usize]));
        pusher.join().unwrap();
        let stolen = thief.join().unwrap();
        if stolen.is_none() {
            CLEARED_EMPTY.store(true, StdOrdering::Relaxed);
            // The thief cleared the bit and missed the push: then the
            // pusher must have seen the clear bit and set it again.
            assert_eq!(stack.steal(|i| &links[i as usize]), Some(0));
        } else {
            assert_eq!(stolen, Some(0));
        }
    });
    assert!(CLEARED_EMPTY.load(StdOrdering::Relaxed));
}

#[test]
fn a_stolen_chain_carries_links_published_by_its_pusher() {
    model(|| {
        let stack = Arc::new(FreeStack::new());
        let links = links();
        for _ in 0..3 {
            stack.fresh(3, 1);
        }
        let (s, l) = (stack.clone(), links.clone());
        let pusher = thread::spawn(move || {
            set_loom_shard(1);
            // Links 1 -> NONE, 2 -> 1; both start at 0, so a thief that
            // missed the publication would follow a link to index 0.
            s.push(1, &l[1]);
            s.push(2, &l[2]);
        });
        let (s, l) = (stack.clone(), links.clone());
        let thief = thread::spawn(move || {
            let next = |i: u32| &l[i as usize];
            let mut got: Vec<u32> = s.steal(next).into_iter().collect();
            while let Some(i) = s.pop(next) {
                got.push(i);
            }
            got
        });
        thread::yield_now();
        pusher.join().unwrap();
        let mut all = thief.join().unwrap();
        all.extend(take_all(&stack, &links));
        all.sort_unstable();
        assert_eq!(all, [1, 2]);
    });
}

/// Every index below the bound is either handed out or listed as unissued.
fn assert_no_lost_reservation(stack: Arc<FreeStack>, mut handed: Vec<u32>, limit: u32) {
    let mut stack = Arc::try_unwrap(stack).ok().expect("threads joined");
    let bound = stack.index_bound();
    assert!(bound <= limit);
    handed.sort_unstable();
    let before = handed.len();
    handed.dedup();
    assert_eq!(handed.len(), before, "an index was handed out twice");
    let mut unissued = stack.unissued();
    for index in 0..bound {
        let listed = unissued.contains(index);
        assert_ne!(
            handed.binary_search(&index).is_ok(),
            listed,
            "index {index}"
        );
    }
}

#[test]
fn contended_single_reservations_switch_to_blocks_without_loss() {
    model(|| {
        let stack = Arc::new(FreeStack::new());
        let threads: Vec<_> = (0..2)
            .map(|_| {
                let s = stack.clone();
                thread::spawn(move || [s.fresh(8, 4), s.fresh(8, 4)])
            })
            .collect();
        let handed: Vec<u32> = threads
            .into_iter()
            .flat_map(|t| t.join().unwrap())
            .map(|i| i.expect("below the limit"))
            .collect();
        assert_no_lost_reservation(stack, handed, 8);
    });
}

#[test]
fn shard_mates_refill_one_block_at_a_time() {
    model(|| {
        let stack = Arc::new(FreeStack::new());
        stack.force_blocks();
        let threads: Vec<_> = (0..2)
            .map(|_| {
                let s = stack.clone();
                thread::spawn(move || [s.fresh(5, 2), s.fresh(5, 2)])
            })
            .collect();
        let handed: Vec<u32> = threads
            .into_iter()
            .flat_map(|t| t.join().unwrap())
            .map(|i| i.expect("below the limit"))
            .collect();
        assert_no_lost_reservation(stack, handed, 5);
    });
}

#[test]
fn exhaustion_is_reported_only_when_every_index_is_handed_out() {
    model(|| {
        let stack = Arc::new(FreeStack::new());
        let s = stack.clone();
        // Shard 1 reserves blocks of 2; shard 0 drains them at the limit.
        let blocks = thread::spawn(move || {
            set_loom_shard(1);
            s.force_blocks();
            [s.fresh(4, 2), s.fresh(4, 2)]
        });
        let s = stack.clone();
        let single = thread::spawn(move || [s.fresh(4, 2), s.fresh(4, 2), s.fresh(4, 2)]);
        let blocks = blocks.join().unwrap();
        let single = single.join().unwrap();
        // A None means all four indices were handed out, so the same thread
        // cannot get one afterwards. (A scan that misses a block published
        // mid-scan reports None early and finds the index on the next call.)
        for thread in [&blocks[..], &single[..]] {
            let first_none = thread.iter().position(Option::is_none);
            if let Some(at) = first_none {
                assert!(thread[at..].iter().all(Option::is_none), "{thread:?}");
            }
        }
        let mut results: Vec<Option<u32>> = blocks.into();
        results.extend(single);
        // Five requests, four indices: a None only once all four are taken.
        assert_eq!(results.iter().filter(|r| r.is_none()).count(), 1);
        let handed = results.into_iter().flatten().collect();
        assert_no_lost_reservation(stack, handed, 4);
    });
}

#[test]
fn the_single_fast_path_switches_to_blocks_and_exhausts_exactly() {
    static SWITCHED: StdAtomicBool = StdAtomicBool::new(false);
    SWITCHED.store(false, StdOrdering::Relaxed);
    model(|| {
        let stack = Arc::new(FreeStack::new());
        let links = links();
        // Both shards start SINGLE, so claims go through next_index's inline
        // path; a failed counter CAS there must switch to blocks at once.
        let claim = |s: &FreeStack, l: &Links| {
            s.next_index(|i| &l[i as usize], || None, 4, 2)
                .map(|(index, reused)| {
                    assert!(!reused);
                    index
                })
        };
        let (s, l) = (stack.clone(), links.clone());
        let other = thread::spawn(move || {
            set_loom_shard(1);
            [claim(&s, &l), claim(&s, &l)]
        });
        // Lets the other shard go first without spending a preemption: it
        // must be able to refill a block before this thread hits the limit.
        thread::yield_now();
        let mine = [
            claim(&stack, &links),
            claim(&stack, &links),
            claim(&stack, &links),
        ];
        let other = other.join().unwrap();
        for thread in [&other[..], &mine[..]] {
            if let Some(at) = thread.iter().position(Option::is_none) {
                assert!(thread[at..].iter().all(Option::is_none), "{thread:?}");
            }
        }
        if stack.index_bound() > 0 && stack.shards_in_block_mode() > 0 {
            SWITCHED.store(true, StdOrdering::Relaxed);
        }
        let mut results: Vec<Option<u32>> = other.into();
        results.extend(mine);
        assert_eq!(results.iter().filter(|r| r.is_none()).count(), 1);
        let handed = results.into_iter().flatten().collect();
        assert_no_lost_reservation(stack, handed, 4);
    });
    assert!(
        SWITCHED.load(StdOrdering::Relaxed),
        "no schedule switched a shard to blocks"
    );
}

#[test]
fn a_free_on_another_shard_is_stolen_with_a_new_generation() {
    model(|| {
        let ids = Arc::new(ConcurrentIdAllocator::<2>::new());
        let a = ids.allocate();
        let b = ids.allocate();
        let i = ids.clone();
        let freer = thread::spawn(move || {
            set_loom_shard(1);
            assert!(i.free(a));
        });
        let i = ids.clone();
        let allocator = thread::spawn(move || i.allocate());
        thread::yield_now();
        freer.join().unwrap();
        let c = allocator.join().unwrap();
        assert!(ids.is_live(b) && ids.is_live(c) && !ids.is_live(a));
        assert_ne!(c, a);
        assert_ne!(c.index(), b.index());
        // Index 0 is either reused by c or still free for the next caller.
        let d = ids.allocate();
        assert!(c.index() == a.index() || d.index() == a.index());
        assert_eq!(ids.len(), 3);
    });
}
