//! Small append, prefix-publication and sparse-reservation schedule models.

use alloc::vec::Vec;
use loom::sync::atomic::{AtomicUsize, Ordering};
use loom::sync::{Arc, Condvar, Mutex};
use loom::thread;

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
