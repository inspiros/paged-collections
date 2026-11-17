//! Small append, prefix-publication and sparse-reservation schedule models.

#[cfg(not(feature = "std"))]
extern crate std;

use alloc::vec::Vec;
use loom::sync::atomic::{AtomicUsize, Ordering};
use loom::sync::{Arc, Condvar, Mutex};
use loom::thread;
use std::sync::atomic::{AtomicBool as CoverageFlag, Ordering as CoverageOrdering};

use super::append::AppendLog;
use super::{ConcurrentSparseMap, ConcurrentSparseSet};

fn model(f: impl Fn() + Send + Sync + 'static) {
    let mut builder = loom::model::Builder::new();
    if builder.preemption_bound.is_none() {
        builder.preemption_bound = Some(2);
    }
    builder.max_branches = 10_000;
    builder.check(f);
}

struct Value {
    tag: usize,
    complement: usize,
}

impl Value {
    fn new(tag: usize) -> Self {
        Self {
            tag,
            complement: !tag,
        }
    }
    fn check(&self) {
        assert!(self.tag == 1 || self.tag == 2);
        assert_eq!(self.complement, !self.tag);
    }
}

#[test]
fn two_pushes_end_with_equal_watermarks_and_distinct_positions() {
    model(|| {
        let log = Arc::new(AppendLog::<Value, 2>::new());
        let mut writers = Vec::new();
        for tag in 1..=2 {
            let log = log.clone();
            writers.push(thread::spawn(move || {
                let pos = log.push(Value::new(tag), 2).ok().unwrap();
                assert_eq!(log.get(pos).unwrap().tag, tag);
                pos
            }));
        }
        let positions: Vec<_> = writers.into_iter().map(|w| w.join().unwrap()).collect();
        assert_ne!(positions[0], positions[1]);
        assert_eq!(log.len(), 2);
        assert_eq!(log.reserved_len(), 2);
        for value in log.iter() {
            value.check();
        }
        assert_eq!(log.get(positions[0]).unwrap().tag, 1);
        assert_eq!(log.get(positions[1]).unwrap().tag, 2);
    });
}

#[test]
fn two_pushes_and_reader_observe_only_written_prefix_and_finished_points() {
    model(|| {
        let log = Arc::new(AppendLog::<Value, 2>::new());
        let finished = Arc::new([AtomicUsize::new(0), AtomicUsize::new(0)]);
        let mut writers = Vec::new();
        for tag in 1..=2 {
            let log = log.clone();
            let finished = finished.clone();
            writers.push(thread::spawn(move || {
                let pos = log.push(Value::new(tag), 2).ok().unwrap();
                // Release tells the reader that push returned, including its
                // index, but the prefix may still lag behind another writer.
                finished[tag - 1].store(pos + 1, Ordering::Release);
            }));
        }
        let reader = log.clone();
        let observed = finished.clone();
        let observer = thread::spawn(move || {
            // These lookups have no finished-index handshake: any Some must
            // acquire publication through the log itself, including readiness
            // above a still-stalled committed prefix.
            for pos in 0..2 {
                if let Some(value) = reader.get(pos) {
                    value.check();
                }
            }
            for (tag, finished) in observed.iter().enumerate() {
                // Acquire receives a returned index from its writer.
                let encoded = finished.load(Ordering::Acquire);
                if encoded != 0 {
                    let value = reader.get(encoded - 1).expect("finished push is visible");
                    value.check();
                    assert_eq!(value.tag, tag + 1);
                }
            }
            for page in reader.pages() {
                for value in page {
                    value.check();
                }
            }
        });
        for writer in writers {
            writer.join().unwrap();
        }
        observer.join().unwrap();
        assert_eq!((log.len(), log.reserved_len()), (2, 2));
    });
}

#[test]
fn later_position_finishes_first_then_earlier_writer_helps_over_it() {
    model(|| {
        let log = Arc::new(AppendLog::<Value, 2>::new());
        // Test-only split claim/publication creates the precise stalled-writer
        // schedule without running a callback inside production push.
        let p = log.reserve_for_test(2);
        let q = log.reserve_for_test(2);
        assert_eq!((p, q), (0, 1));
        let later = log.clone();
        let b = thread::spawn(move || {
            later.finish_for_test(q, Value::new(2));
            assert!(later.get(p).is_none());
            assert_eq!(later.get(q).unwrap().tag, 2);
            assert_eq!(later.len(), 0);
            assert!(later.pages().next().is_none());
        });
        b.join().unwrap();
        let earlier = log.clone();
        thread::spawn(move || earlier.finish_for_test(p, Value::new(1)))
            .join()
            .unwrap();
        assert_eq!((log.len(), log.reserved_len()), (2, 2));
        assert_eq!(log.iter().map(|v| v.tag).collect::<Vec<_>>(), [1, 2]);
    });
}

#[test]
fn helpers_cross_a_page_installation_boundary_without_losing_advance() {
    model(|| {
        let log = Arc::new(AppendLog::<usize, 2>::new());
        assert_eq!(log.push(7, 3), Ok(0));
        // Position 1 is in the old page; position 2 installs a new page.
        // There is no join/handshake between the two publishers' help loops.
        let a = log.clone();
        let a = thread::spawn(move || a.push(8, 3).unwrap());
        let b = log.clone();
        let b = thread::spawn(move || b.push(9, 3).unwrap());
        assert_ne!(a.join().unwrap(), b.join().unwrap());
        assert_eq!((log.len(), log.reserved_len()), (3, 3));
        let mut values: Vec<_> = log.pages().flatten().copied().collect();
        values.sort_unstable();
        assert_eq!(values, [7, 8, 9]);
    });
}

#[test]
fn bounded_reservation_never_claims_past_limit() {
    model(|| {
        let log = Arc::new(AppendLog::<usize, 2>::new());
        let a = log.clone();
        let a = thread::spawn(move || a.push(1, 1));
        let b = log.clone();
        let b = thread::spawn(move || b.push(2, 1));
        let results = [a.join().unwrap(), b.join().unwrap()];
        assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
        assert_eq!(results.iter().filter(|r| r.is_err()).count(), 1);
        assert_eq!((log.len(), log.reserved_len()), (1, 1));
    });
}

#[test]
fn two_inserts_of_one_key_have_one_winner() {
    model(|| {
        let set = Arc::new(ConcurrentSparseSet::<u32, 2>::new());
        let a = set.clone();
        let a = thread::spawn(move || a.insert(1));
        let b = set.clone();
        let b = thread::spawn(move || b.insert(1));
        assert_eq!(
            usize::from(a.join().unwrap()) + usize::from(b.join().unwrap()),
            1
        );
        assert_eq!(set.len(), 1);
        assert_eq!(set.iter().copied().collect::<Vec<_>>(), [1]);
        assert!(set.contains(1));
    });
}

#[test]
fn sparse_reader_racing_insert_never_resolves_the_wrong_key() {
    model(|| {
        let set = Arc::new(ConcurrentSparseSet::<u32, 2>::new());
        assert!(set.insert(0));
        let done = Arc::new(AtomicUsize::new(0));
        let writer = set.clone();
        let completed = done.clone();
        let a = thread::spawn(move || {
            assert!(writer.insert(1));
            assert!(writer.contains(1));
            // Release notifies the reader after insert returns.
            completed.store(1, Ordering::Release);
        });
        let reader = set.clone();
        let b = thread::spawn(move || {
            if let Some(pos) = reader.position(1) {
                assert_eq!(pos, 1);
            }
            assert!(!reader.contains(2));
            // Acquire proves the writer has returned before the second read.
            if done.load(Ordering::Acquire) != 0 {
                assert!(reader.contains(1));
            }
        });
        a.join().unwrap();
        b.join().unwrap();
        assert!(set.contains(0) && set.contains(1));
        assert_eq!(set.len(), 2);
    });
}

#[test]
fn disjoint_sparse_slots_both_publish_into_one_prefix() {
    model(|| {
        let set = Arc::new(ConcurrentSparseSet::<u32, 2>::new());
        let a = set.clone();
        let a = thread::spawn(move || assert!(a.insert(0)));
        let b = set.clone();
        let b = thread::spawn(move || assert!(b.insert(1)));
        a.join().unwrap();
        b.join().unwrap();
        assert_eq!(set.len(), 2);
        assert!(set.contains(0) && set.contains(1));
        let mut keys: Vec<_> = set.pages().flatten().copied().collect();
        keys.sort_unstable();
        assert_eq!(keys, [0, 1]);
    });
}

#[test]
fn two_factories_of_one_key_run_once_and_share_one_value_address() {
    model(|| {
        let map = Arc::new(ConcurrentSparseMap::<u32, usize, 2>::new());
        let calls = Arc::new(AtomicUsize::new(0));
        let mut writers = Vec::new();
        for _ in 0..2 {
            let map = map.clone();
            let calls = calls.clone();
            writers.push(thread::spawn(move || {
                let value = map
                    .get_or_insert_with(1, || {
                        // SeqCst counts all calls independently of the winner.
                        calls.fetch_add(1, Ordering::SeqCst);
                        42
                    })
                    .unwrap();
                assert_eq!(*value, 42);
                value as *const usize as usize
            }));
        }
        let addresses: Vec<_> = writers.into_iter().map(|w| w.join().unwrap()).collect();
        assert_eq!(addresses[0], addresses[1]);
        // SeqCst observes the completed factory count.
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(map.get(&1), Some(&42));
        assert_eq!(map.len(), 1);
    });
}

#[test]
fn rollback_wakes_a_same_slot_waiter() {
    extern crate std;

    model(|| {
        let map = Arc::new(ConcurrentSparseMap::<u32, usize, 2>::new());
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let writer = map.clone();
        let waiting = gate.clone();
        let a = thread::spawn(move || {
            // Catching a factory panic runs the real RESERVED rollback guard.
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    writer.get_or_insert_with(1, || {
                        let mut started = waiting.0.lock().unwrap();
                        *started = true;
                        waiting.1.notify_all();
                        drop(started);
                        thread::yield_now();
                        panic!("factory failed");
                    });
                }))
                .is_err()
            );
        });
        let mut started = gate.0.lock().unwrap();
        while !*started {
            started = gate.1.wait(started).unwrap();
        }
        drop(started);
        let b = thread::spawn(move || *map.get_or_insert_with(1, || 9).unwrap());
        a.join().unwrap();
        assert_eq!(b.join().unwrap(), 9);
    });
}

#[test]
fn radix_sparse_set_split_races_install_and_reader() {
    // PAGE = 4 keeps all four dense entries in a preinstalled append page.
    // Sparse pages 0 and 16 seed a shift-4 root (loom DIGIT = 2); inserting
    // keys 4 and 8 sends both writers to leaf 0, racing a split with install.
    // Yield allows a split/publication before or during the main-thread reads.
    model(|| {
        let set = Arc::new(ConcurrentSparseSet::<usize, 4, crate::page::Radix>::new());
        assert!(set.insert(0));
        assert!(set.insert(64));
        let first = set.clone();
        let a = thread::spawn(move || assert!(first.insert(4)));
        let second = set.clone();
        let b = thread::spawn(move || assert!(second.insert(8)));
        thread::yield_now();
        assert!(set.contains(0));
        let first = set.position(4);
        let second = set.position(8);
        if let Some(pos) = first {
            assert!(pos == 2 || pos == 3);
            assert!(set.contains(4));
        }
        if let Some(pos) = second {
            assert!(pos == 2 || pos == 3);
            assert!(set.contains(8));
        }
        if first.is_some() && second.is_some() {
            assert_ne!(first, second);
        }
        a.join().unwrap();
        b.join().unwrap();
        let mut keys: Vec<_> = set.iter().copied().collect();
        keys.sort_unstable();
        assert_eq!(keys, [0, 4, 8, 64]);
        assert_eq!(set.len(), 4);
        for key in keys {
            assert!(set.contains(key));
        }
    });
}

#[test]
fn radix_sparse_map_split_races_install_and_reader() {
    // As above: sparse pages 1 and 2 both split leaf 0 under a shift-4 root;
    // loom can pause either installer between loading it and publishing CAS.
    // Dense storage is preinstalled; yield makes the reader race publication.
    model(|| {
        let map = Arc::new(ConcurrentSparseMap::<
            usize,
            (usize, usize),
            4,
            crate::page::Radix,
        >::new());
        assert!(map.try_insert(0, (0, !0)).is_ok());
        assert!(map.try_insert(64, (64, !64)).is_ok());
        let first = map.clone();
        let a = thread::spawn(move || assert_eq!(first.try_insert(4, (4, !4)), Ok(&(4, !4))));
        let second = map.clone();
        let b = thread::spawn(move || assert_eq!(second.try_insert(8, (8, !8)), Ok(&(8, !8))));
        thread::yield_now();
        assert_eq!(map.get(&0), Some(&(0, !0)));
        for key in [4, 8] {
            if let Some(value) = map.get(&key) {
                assert_eq!(*value, (key, !key));
                assert_eq!(map.get(&key), Some(value));
            }
        }
        a.join().unwrap();
        b.join().unwrap();
        for key in [0, 4, 8, 64] {
            assert_eq!(map.get(&key), Some(&(key, !key)));
        }
        assert_eq!(map.len(), 4);
    });
}

#[test]
fn batch_and_single_publish_ready_payloads_across_pages() {
    static BATCH_READ: CoverageFlag = CoverageFlag::new(false);
    static SINGLE_READ: CoverageFlag = CoverageFlag::new(false);
    static CROSSED_PAGE: CoverageFlag = CoverageFlag::new(false);
    BATCH_READ.store(false, CoverageOrdering::Relaxed);
    SINGLE_READ.store(false, CoverageOrdering::Relaxed);
    CROSSED_PAGE.store(false, CoverageOrdering::Relaxed);
    model(|| {
        // Preinstall storage before the side payloads are written, so page
        // publication cannot supply their happens-before edge to the reader.
        let mut log = AppendLog::<usize, 2>::new();
        for _ in 0..5 {
            log.push(0, 5).unwrap();
        }
        log.clear();
        // Keep committed at zero until AFTER the reader and both joins. Every
        // successful point read must acquire a ready bit, not the watermark.
        assert_eq!(log.reserve_for_test(5), 0);
        let log = Arc::new(log);
        let payload = Arc::new([AtomicUsize::new(0), AtomicUsize::new(0)]);
        let batch = log.clone();
        let batch_payload = payload.clone();
        let a = thread::spawn(move || {
            batch_payload[0].store(42, Ordering::Relaxed);
            batch.push_many(alloc::vec![10, 11, 12], 5).unwrap()
        });
        let single = log.clone();
        let single_payload = payload.clone();
        let b = thread::spawn(move || {
            single_payload[1].store(84, Ordering::Relaxed);
            single.push(20, 5).unwrap()
        });
        thread::yield_now();
        for pos in 1..5 {
            if let Some(&value) = log.get(pos) {
                match value {
                    10..=12 => {
                        assert_eq!(payload[0].load(Ordering::Relaxed), 42);
                        BATCH_READ.store(true, CoverageOrdering::Relaxed);
                    }
                    20 => {
                        assert_eq!(payload[1].load(Ordering::Relaxed), 84);
                        SINGLE_READ.store(true, CoverageOrdering::Relaxed);
                    }
                    _ => panic!("unwritten or wrong batch payload"),
                }
            }
        }
        let start = a.join().unwrap();
        let single = b.join().unwrap();
        assert!(!(start..start + 3).contains(&single));
        assert_ne!(start / 2, (start + 2) / 2);
        CROSSED_PAGE.store(true, CoverageOrdering::Relaxed);
        for (offset, value) in [10, 11, 12].into_iter().enumerate() {
            assert_eq!(log.get(start + offset), Some(&value));
        }
        assert_eq!(log.get(single), Some(&20));
        assert_eq!((log.len(), log.reserved_len()), (0, 5));
        log.finish_for_test(0, 0);
        assert_eq!((log.len(), log.reserved_len()), (5, 5));
        assert_eq!(log.iter().count(), 5);
    });
    assert!(BATCH_READ.load(CoverageOrdering::Relaxed));
    assert!(SINGLE_READ.load(CoverageOrdering::Relaxed));
    assert!(CROSSED_PAGE.load(CoverageOrdering::Relaxed));
}

#[test]
fn two_batches_cross_pages_and_help_the_entire_ready_run() {
    static FIRST_READ: CoverageFlag = CoverageFlag::new(false);
    static SECOND_READ: CoverageFlag = CoverageFlag::new(false);
    static BOTH_CROSSED: CoverageFlag = CoverageFlag::new(false);
    FIRST_READ.store(false, CoverageOrdering::Relaxed);
    SECOND_READ.store(false, CoverageOrdering::Relaxed);
    BOTH_CROSSED.store(false, CoverageOrdering::Relaxed);
    model(|| {
        let mut log = AppendLog::<usize, 2>::new();
        for _ in 0..5 {
            log.push(0, 5).unwrap();
        }
        log.truncate(1);
        let log = Arc::new(log);
        let payload = Arc::new([AtomicUsize::new(0), AtomicUsize::new(0)]);
        let mut writers = Vec::new();
        for tag in 1..=2 {
            let log = log.clone();
            let payload = payload.clone();
            writers.push(thread::spawn(move || {
                payload[tag - 1].store(tag + 40, Ordering::Relaxed);
                log.push_many(alloc::vec![tag * 10, tag * 10 + 1], 5)
                    .unwrap()
            }));
        }
        thread::yield_now();
        // pages() acquires a prefix snapshot: run-CAS publication must carry
        // all writers' payloads, including ones another helper skipped over.
        for &value in log.pages().flatten().filter(|&&v| v != 0) {
            let tag = value / 10;
            assert!((1..=2).contains(&tag));
            assert_eq!(payload[tag - 1].load(Ordering::Relaxed), tag + 40);
            if tag == 1 {
                FIRST_READ.store(true, CoverageOrdering::Relaxed);
            } else {
                SECOND_READ.store(true, CoverageOrdering::Relaxed);
            }
        }
        let starts: Vec<_> = writers.into_iter().map(|w| w.join().unwrap()).collect();
        assert!(starts == [1, 3] || starts == [3, 1]);
        assert!(starts.iter().all(|&s| s / 2 != (s + 1) / 2));
        BOTH_CROSSED.store(true, CoverageOrdering::Relaxed);
        assert_eq!((log.len(), log.reserved_len()), (5, 5));
        for (tag, start) in starts.into_iter().enumerate() {
            assert_eq!(log.get(start), Some(&((tag + 1) * 10)));
            assert_eq!(log.get(start + 1), Some(&((tag + 1) * 10 + 1)));
        }
    });
    assert!(FIRST_READ.load(CoverageOrdering::Relaxed));
    assert!(SECOND_READ.load(CoverageOrdering::Relaxed));
    assert!(BOTH_CROSSED.load(CoverageOrdering::Relaxed));
}

// A two-round rendezvous in iterator::next: first both calls own their
// initial slot, then both have attempted the other's slot BEFORE either
// publishes. No production insertion is called from this iterator.
fn reciprocal_busy_items<T: Copy>(
    first: T,
    second: T,
    gate: Arc<(Mutex<usize>, Condvar)>,
    reached: &'static CoverageFlag,
) -> impl Iterator<Item = T> {
    let mut step = 0;
    core::iter::from_fn(move || {
        let current = step;
        step += 1;
        if current == 1 || current == 2 {
            let mut arrivals = gate.0.lock().unwrap();
            *arrivals += 1;
            gate.1.notify_all();
            while *arrivals < current * 2 {
                arrivals = gate.1.wait(arrivals).unwrap();
            }
            if current == 2 {
                reached.store(true, CoverageOrdering::Relaxed);
            }
        }
        match current {
            0 => Some(first),
            1 => Some(second),
            _ => None,
        }
    })
}

#[test]
fn overlapping_set_batches_defer_reciprocal_busy_slots() {
    static BOTH_BUSY: CoverageFlag = CoverageFlag::new(false);
    static PUBLISHED_READ: CoverageFlag = CoverageFlag::new(false);
    BOTH_BUSY.store(false, CoverageOrdering::Relaxed);
    PUBLISHED_READ.store(false, CoverageOrdering::Relaxed);
    model(|| {
        let mut set = ConcurrentSparseSet::<u32, 2>::new();
        assert_eq!(set.insert_many([0, 1]), 2);
        set.clear();
        let set = Arc::new(set);
        let gate = Arc::new((Mutex::new(0), Condvar::new()));
        let mut writers = Vec::new();
        for first in 0..2 {
            let set = set.clone();
            let gate = gate.clone();
            writers.push(thread::spawn(move || {
                set.insert_many(reciprocal_busy_items(first, 1 - first, gate, &BOTH_BUSY))
            }));
        }
        thread::yield_now();
        for key in 0..2 {
            if let Some(pos) = set.position(key) {
                assert!(pos < 2);
                assert!(set.contains(key));
                PUBLISHED_READ.store(true, CoverageOrdering::Relaxed);
            }
        }
        let counts: Vec<_> = writers.into_iter().map(|w| w.join().unwrap()).collect();
        assert_eq!(counts, [1, 1]);
        assert_eq!(counts.iter().sum::<usize>(), set.len());
        let mut keys: Vec<_> = set.iter().copied().collect();
        keys.sort_unstable();
        assert_eq!(keys, [0, 1]);
        assert_ne!(set.position(0), set.position(1));
    });
    assert!(BOTH_BUSY.load(CoverageOrdering::Relaxed));
    assert!(PUBLISHED_READ.load(CoverageOrdering::Relaxed));
}

#[test]
fn overlapping_map_batches_defer_busy_and_publish_payloads() {
    static BOTH_BUSY: CoverageFlag = CoverageFlag::new(false);
    static PUBLISHED_READ: CoverageFlag = CoverageFlag::new(false);
    BOTH_BUSY.store(false, CoverageOrdering::Relaxed);
    PUBLISHED_READ.store(false, CoverageOrdering::Relaxed);
    model(|| {
        let mut map = ConcurrentSparseMap::<u32, usize, 2>::new();
        assert_eq!(map.try_insert_many([(0, 0), (1, 0)]), 2);
        map.clear();
        let map = Arc::new(map);
        let payload = Arc::new([AtomicUsize::new(0), AtomicUsize::new(0)]);
        let gate = Arc::new((Mutex::new(0), Condvar::new()));
        let mut writers = Vec::new();
        for first in 0..2 {
            let map = map.clone();
            let payload = payload.clone();
            let gate = gate.clone();
            writers.push(thread::spawn(move || {
                payload[first as usize].store(first as usize + 40, Ordering::Relaxed);
                map.try_insert_many(reciprocal_busy_items(
                    (first, first as usize + 10),
                    (1 - first, 99),
                    gate,
                    &BOTH_BUSY,
                ))
            }));
        }
        thread::yield_now();
        for key in 0..2 {
            if let Some(&value) = map.get(&key) {
                assert_eq!(value, key as usize + 10);
                assert_eq!(
                    payload[key as usize].load(Ordering::Relaxed),
                    key as usize + 40
                );
                PUBLISHED_READ.store(true, CoverageOrdering::Relaxed);
            }
        }
        let counts: Vec<_> = writers.into_iter().map(|w| w.join().unwrap()).collect();
        assert_eq!(counts, [1, 1]);
        assert_eq!(counts.iter().sum::<usize>(), map.len());
        let mut entries: Vec<_> = map.iter().map(|(&k, &v)| (k, v)).collect();
        entries.sort_unstable();
        assert_eq!(entries, [(0, 10), (1, 11)]);
    });
    assert!(BOTH_BUSY.load(CoverageOrdering::Relaxed));
    assert!(PUBLISHED_READ.load(CoverageOrdering::Relaxed));
}
