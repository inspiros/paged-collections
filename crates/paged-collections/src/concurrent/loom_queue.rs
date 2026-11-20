//! Small position, destruction and pool-claim schedule models.

#[cfg(not(feature = "std"))]
extern crate std;

use alloc::vec::Vec;
use loom::sync::{Arc, mpsc};
use loom::thread;
use std::sync::atomic::{AtomicBool, Ordering as CoverageOrdering};

use super::super::free_stack::set_loom_shard;
use super::{
    BOUNDARY_WAITED, ConcurrentQueue, INSTALL_LOST, Ordering, PopStep, REUSED_BLOCK, SHIFT,
};

fn model(f: impl Fn() + Send + Sync + 'static) {
    model_with_branches(10_000, f);
}

// Spin waits (a pop waiting for WRITE, a push at the boundary marker) add a
// branch per loop turn, so models where the main thread races producers
// need a higher limit.
fn model_with_branches(max_branches: usize, f: impl Fn() + Send + Sync + 'static) {
    let mut builder = loom::model::Builder::new();
    if builder.preemption_bound.is_none() {
        builder.preemption_bound = Some(2);
    }
    builder.max_branches = max_branches;
    builder.check(f);
}

fn check_producers<const PAGE: usize, const PUSHES: usize, const POP_EARLY: bool>() {
    static POPPED_BEFORE_JOIN: AtomicBool = AtomicBool::new(false);
    POPPED_BEFORE_JOIN.store(false, CoverageOrdering::Relaxed);
    model_with_branches(200_000, || {
        let queue = Arc::new(ConcurrentQueue::<_, PAGE>::with_capacity(5 * PAGE));
        // Install the first block up front to keep the model focused on
        // position claims and value publication.
        queue.push((2, 0));
        assert_eq!(queue.pop(), Some((2, 0)));

        let a = Arc::clone(&queue);
        let b = Arc::clone(&queue);
        let first = thread::spawn(move || {
            for sequence in 0..PUSHES {
                a.push((0, sequence));
            }
        });
        let second = thread::spawn(move || {
            for sequence in 0..PUSHES {
                b.push((1, sequence));
            }
        });

        let mut observed = Vec::new();
        // Without a yield, loom may never schedule a producer before this
        // pop, and the coverage flag below stays unset.
        if POP_EARLY {
            thread::yield_now();
            if let Some(value) = queue.pop() {
                POPPED_BEFORE_JOIN.store(true, CoverageOrdering::Relaxed);
                observed.push(value);
            }
        }
        first.join().unwrap();
        second.join().unwrap();
        observed.extend(core::iter::from_fn(|| queue.pop()));

        let mut next = [0, 0];
        for (producer, sequence) in observed {
            assert_eq!(sequence, next[producer]);
            next[producer] += 1;
        }
        assert_eq!(next, [PUSHES, PUSHES]);
    });
    // One static serves every instantiation and tests run in parallel, so
    // only the early-pop variant may assert it.
    if POP_EARLY {
        assert!(POPPED_BEFORE_JOIN.load(CoverageOrdering::Relaxed));
    }
}

#[test]
fn two_producers_in_one_page_preserve_fifo() {
    // Two pushes each inside one page: position claims and publication.
    check_producers::<8, 2, true>();
}

#[test]
fn two_producers_race_for_the_last_slot() {
    // One push each racing for a page's last slot: one installs the
    // successor while the other waits at the boundary marker. A concurrent
    // pop here makes two spin waits race, which loom could not finish in
    // 40 minutes, so the consumer pops only after the joins.
    BOUNDARY_WAITED.store(false, CoverageOrdering::Relaxed);
    check_producers::<2, 1, false>();
    assert!(BOUNDARY_WAITED.load(CoverageOrdering::Relaxed));
}

fn check_consumers<const PAGE: usize>() {
    static HANDED_OFF: AtomicBool = AtomicBool::new(false);
    static RECYCLED: AtomicBool = AtomicBool::new(false);
    HANDED_OFF.store(false, CoverageOrdering::Relaxed);
    RECYCLED.store(false, CoverageOrdering::Relaxed);
    model(|| {
        let queue = Arc::new(ConcurrentQueue::<_, PAGE>::new());
        queue.push(1);
        queue.push(2);
        let allocated = queue.allocated_pages();

        let pop = |queue: &ConcurrentQueue<usize, PAGE>| {
            queue.pop_inner(|step| {
                if let PopStep::DestroyHandOff = step {
                    HANDED_OFF.store(true, CoverageOrdering::Relaxed);
                }
            })
        };
        let reader = Arc::clone(&queue);
        let worker = thread::spawn(move || pop(&reader));
        let main_value = pop(&queue).unwrap();
        let mut values = [main_value, worker.join().unwrap().unwrap()];
        values.sort_unstable();
        assert_eq!(values, [1, 2]);

        // The producer installed a fresh empty tail at the boundary. Fill
        // it too: installing its successor must reuse a consumed block.
        for value in 0..PAGE {
            queue.push(value);
        }
        assert_eq!(queue.allocated_pages(), allocated);
        RECYCLED.store(true, CoverageOrdering::Relaxed);
        for value in 0..PAGE {
            assert_eq!(queue.pop(), Some(value));
        }
        assert_eq!(queue.pop(), None);
    });
    // PAGE == 1 has no earlier reader to inherit destruction.
    assert_eq!(HANDED_OFF.load(CoverageOrdering::Relaxed), PAGE > 1);
    assert!(RECYCLED.load(CoverageOrdering::Relaxed));
}

#[test]
fn two_consumers_cross_the_boundary_and_recycle() {
    check_consumers::<1>();
    check_consumers::<2>();
}

fn check_stale_position<const PAGE: usize>() {
    static REJECTED_STALE: AtomicBool = AtomicBool::new(false);
    REJECTED_STALE.store(false, CoverageOrdering::Relaxed);
    model(|| {
        let queue = Arc::new(ConcurrentQueue::<_, PAGE>::new());
        for value in 0..PAGE {
            queue.push(value);
        }
        let original = queue.head.0.block.load(Ordering::Relaxed);
        let address = original.addr();
        let reader = Arc::clone(&queue);
        let (cached, wait_cached) = mpsc::channel();
        let (resume, wait_resume) = mpsc::channel();
        let stale = thread::spawn(move || {
            let mut cached_old = false;
            reader.pop_inner(|step| {
                if let PopStep::BeforeClaim { head, block } = step {
                    assert_eq!(block.addr(), address);
                    if cached_old {
                        assert_eq!(head >> SHIFT, 2 * ConcurrentQueue::<usize, PAGE>::LAP);
                        REJECTED_STALE.store(true, CoverageOrdering::Relaxed);
                    } else {
                        assert_eq!(head >> SHIFT, 0);
                        cached_old = true;
                        cached.send(()).unwrap();
                        wait_resume.recv().unwrap();
                    }
                }
            })
        });

        wait_cached.recv().unwrap();
        for value in 0..PAGE {
            assert_eq!(queue.pop(), Some(value));
        }
        // Steal the retired block from the recycler's shard, install it as
        // the successor, then advance the head to the reused address.
        set_loom_shard(1);
        for value in PAGE..2 * PAGE {
            queue.push(value);
        }
        for value in PAGE..2 * PAGE {
            assert_eq!(queue.pop(), Some(value));
        }
        assert_eq!(queue.head.0.block.load(Ordering::Relaxed), original);
        queue.push(2 * PAGE);
        resume.send(()).unwrap();

        assert_eq!(stale.join().unwrap(), Some(2 * PAGE));
        assert_eq!(queue.pop(), None);
        assert_eq!(queue.allocated_pages(), 2);
    });
    assert!(REJECTED_STALE.load(CoverageOrdering::Relaxed));
}

#[test]
fn recycled_address_cannot_make_a_stale_position_win() {
    check_stale_position::<1>();
    check_stale_position::<2>();
}

fn check_duplicate_hint<const PAGE: usize>() {
    static WON: [AtomicBool; 2] = [AtomicBool::new(false), AtomicBool::new(false)];
    for flag in &WON {
        flag.store(false, CoverageOrdering::Relaxed);
    }
    model(|| {
        let queue = Arc::new(ConcurrentQueue::<usize, PAGE>::with_capacity(PAGE));
        let number = queue.pool.pop(|n| &queue.block(n).pool_next).unwrap();

        // A wrapped tag can yield the same hint to two takers. Inject the
        // duplicate here and exercise the real claim CAS from both threads.
        let other = Arc::clone(&queue);
        let contender = thread::spawn(move || {
            let claimed = other.claim_pooled(number).is_some();
            if claimed {
                WON[1].store(true, CoverageOrdering::Relaxed);
            }
            claimed
        });
        let claimed = queue.claim_pooled(number).is_some();
        if claimed {
            WON[0].store(true, CoverageOrdering::Relaxed);
        }
        assert_ne!(claimed, contender.join().unwrap());
    });
    assert!(WON.iter().all(|flag| flag.load(CoverageOrdering::Relaxed)));
}

#[test]
fn duplicate_pool_hint_grants_only_one_claim() {
    check_duplicate_hint::<1>();
    check_duplicate_hint::<2>();
}

#[test]
fn a_block_retired_by_one_thread_is_reused_by_another() {
    REUSED_BLOCK.store(false, CoverageOrdering::Relaxed);
    model(|| {
        // PAGE 1: the first push installs block A and pre-installs B.
        let queue = Arc::new(ConcurrentQueue::<usize, 1>::new());
        queue.push(0);
        let consumer = Arc::clone(&queue);
        // Popping 0 finishes A, so this thread recycles it into the pool.
        let retire = thread::spawn(move || consumer.pop());
        thread::yield_now();
        // Pushing 1 fills B and takes its successor from the pool (A, if
        // already recycled); pushing 2 then writes into that block's slot,
        // with no join ordering it after the retiring thread's read.
        queue.push(1);
        queue.push(2);
        assert_eq!(retire.join().unwrap(), Some(0));
        assert_eq!(queue.pop(), Some(1));
        assert_eq!(queue.pop(), Some(2));
        assert_eq!(queue.pop(), None);
    });
    // Other models reuse blocks too, so this flag alone does not prove the
    // concurrent case; the model's value is the race check on the reuse.
    assert!(REUSED_BLOCK.load(CoverageOrdering::Relaxed));
}

#[test]
fn losing_the_first_install_pools_the_block_for_reuse() {
    INSTALL_LOST.store(false, CoverageOrdering::Relaxed);
    model(|| {
        // An empty queue has no block: both first pushes try to install one.
        let queue = Arc::new(ConcurrentQueue::<usize, 2>::new());
        let other = Arc::clone(&queue);
        let rival = thread::spawn(move || other.push(1));
        queue.push(0);
        rival.join().unwrap();
        // The CAS loser pools its block instead of installing it.
        let installed = queue.allocated_pages();
        assert!(installed == 1 || installed == 2);

        // Filling the page needs a successor. A lost block sits in the
        // rival's free-stack shard, so this thread may allocate a fresh one
        // rather than steal it; either way no block may be linked twice.
        queue.push(2);
        queue.push(3);
        assert!(queue.allocated_pages() <= installed + 1);
        let mut first = [queue.pop().unwrap(), queue.pop().unwrap()];
        first.sort_unstable();
        assert_eq!(first, [0, 1]);
        assert_eq!(queue.pop(), Some(2));
        assert_eq!(queue.pop(), Some(3));
        assert_eq!(queue.pop(), None);
    });
    assert!(INSTALL_LOST.load(CoverageOrdering::Relaxed));
}
