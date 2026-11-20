//! Models relaxed popping while other threads push or pop.

#[cfg(not(feature = "std"))]
extern crate std;

use loom::sync::Arc;
use loom::thread;
use std::sync::atomic::Ordering as CoverageOrdering;

use super::{ConcurrentPriorityQueue, SCAN_FOUND};

fn model(f: impl Fn() + Send + Sync + 'static) {
    let mut builder = loom::model::Builder::new();
    if builder.preemption_bound.is_none() {
        builder.preemption_bound = Some(2);
    }
    builder.max_branches = 10_000;
    builder.check(f);
}

#[test]
fn pop_racing_push_finds_an_entry_present_before_the_call() {
    model(|| {
        let queue = Arc::new(ConcurrentPriorityQueue::<u32, u32, 1>::with_shards(3));
        // The deterministic sampler starts with shards 1 and 0. The
        // entry in shard 2 must still be found if the producer is late.
        assert_eq!(queue.shard_of(&3), 2);
        assert_eq!(queue.shard_of(&1), 1);
        assert_eq!(queue.push(3, 10), None);
        let producer = Arc::clone(&queue);
        let writer = thread::spawn(move || assert_eq!(producer.push(1, 20), None));

        let first = queue.pop();
        writer.join().unwrap();
        match first {
            Some((3, 10)) => assert_eq!(queue.pop(), Some((1, 20))),
            Some((1, 20)) => assert_eq!(queue.pop(), Some((3, 10))),
            other => panic!("pop missed the entry present before the call: {other:?}"),
        }
        assert_eq!(queue.pop(), None);
    });
}

#[test]
fn fallback_scan_finds_an_entry_pushed_after_sampling() {
    SCAN_FOUND.store(false, CoverageOrdering::Relaxed);
    model(|| {
        let queue = Arc::new(ConcurrentPriorityQueue::<u32, u32, 1>::with_shards(3));
        // The sampler starts with shards 1 and 0, so an entry in shard 2
        // that arrives after the length hints were read is found only by
        // the scan.
        assert_eq!(queue.shard_of(&3), 2);
        let producer = Arc::clone(&queue);
        let writer = thread::spawn(move || assert_eq!(producer.push(3, 10), None));

        let first = queue.pop();
        writer.join().unwrap();
        // The push races the call, so either outcome is allowed, but the
        // entry is popped exactly once.
        match first {
            Some(entry) => {
                assert_eq!(entry, (3, 10));
                assert_eq!(queue.pop(), None);
            }
            None => assert_eq!(queue.pop(), Some((3, 10))),
        }
        assert!(queue.is_empty());
    });
    assert!(SCAN_FOUND.load(CoverageOrdering::Relaxed));
}

#[test]
fn two_pops_take_one_entry_exactly_once() {
    model(|| {
        // Shard pairs lock in ascending order, so two pops whose pairs
        // overlap must not deadlock (loom reports one if they can).
        let queue = Arc::new(ConcurrentPriorityQueue::<u32, u32, 1>::with_shards(3));
        assert_eq!(queue.push(3, 10), None);
        let consumer = Arc::clone(&queue);
        let other = thread::spawn(move || consumer.pop());
        let mine = queue.pop();
        let theirs = other.join().unwrap();
        assert!(matches!(
            (mine, theirs),
            (Some((3, 10)), None) | (None, Some((3, 10)))
        ));
        assert!(queue.is_empty());
    });
}
