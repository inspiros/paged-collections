//! Two-thread models of one-word updates and ordered page traversal.

use core::num::NonZeroU32;

use loom::sync::Arc;
use loom::thread;

use super::{AtomicDirectMap, AtomicIdMap};
use crate::id_alloc::Id;

fn model(f: impl Fn() + Send + Sync + 'static) {
    let mut builder = loom::model::Builder::new();
    if builder.preemption_bound.is_none() {
        builder.preemption_bound = Some(2);
    }
    builder.max_branches = 10_000;
    builder.check(f);
}

fn id(index: u32, generation: u32) -> Id {
    Id::new(index, NonZeroU32::new(generation).unwrap())
}

#[test]
fn direct_two_fetch_updates() {
    model(|| {
        let map = Arc::new(AtomicDirectMap::<1>::new());
        let writer = map.clone();
        let worker = thread::spawn(move || {
            writer
                .fetch_update(0, |old| Some(Some(old.unwrap_or(0) + 1)))
                .unwrap()
        });
        let previous = map
            .fetch_update(0, |old| Some(Some(old.unwrap_or(0) + 1)))
            .unwrap();
        let other = worker.join().unwrap();
        assert!(matches!(
            (previous, other),
            (None, Some(1)) | (Some(1), None)
        ));
        assert_eq!(map.get(0), Some(2));
        assert_eq!(map.len(), 1);
    });
}

#[test]
fn id_two_fetch_updates() {
    model(|| {
        let map = Arc::new(AtomicIdMap::<1>::new());
        let key = id(0, 1);
        let writer = map.clone();
        let worker = thread::spawn(move || {
            writer
                .fetch_update(key, |old| Some(Some(old.unwrap_or(0) + 1)))
                .unwrap()
        });
        let previous = map
            .fetch_update(key, |old| Some(Some(old.unwrap_or(0) + 1)))
            .unwrap();
        let other = worker.join().unwrap();
        assert!(matches!(
            (previous, other),
            (None, Some(1)) | (Some(1), None)
        ));
        assert_eq!(map.get(key), Some(2));
        assert_eq!(map.len(), 1);
    });
}

#[test]
fn direct_insert_races_remove() {
    model(|| {
        let map = Arc::new(AtomicDirectMap::<1>::new());
        let writer = map.clone();
        let worker = thread::spawn(move || assert_eq!(writer.insert(0, 7), None));
        // A yield costs no preemption and lets loom run the insert first;
        // without it the remove always won and the Some(7) arm never ran.
        thread::yield_now();
        let removed = map.remove(0);
        worker.join().unwrap();
        match removed {
            Some(7) => {
                assert_eq!(map.get(0), None);
                assert_eq!(map.len(), 0);
            }
            None => {
                assert_eq!(map.get(0), Some(7));
                assert_eq!(map.len(), 1);
            }
            _ => panic!("unexpected removed value"),
        }
    });
}

#[test]
fn id_insert_races_remove() {
    model(|| {
        let map = Arc::new(AtomicIdMap::<1>::new());
        let key = id(0, 1);
        let writer = map.clone();
        let worker = thread::spawn(move || assert_eq!(writer.insert(key, 7), None));
        // Lets loom run the insert first (see direct_insert_races_remove).
        thread::yield_now();
        let removed = map.remove(key);
        worker.join().unwrap();
        match removed {
            Some(7) => {
                assert_eq!(map.get(key), None);
                assert_eq!(map.len(), 0);
            }
            None => {
                assert_eq!(map.get(key), Some(7));
                assert_eq!(map.len(), 1);
            }
            _ => panic!("unexpected removed value"),
        }
    });
}

#[test]
fn direct_two_try_inserts() {
    model(|| {
        let map = Arc::new(AtomicDirectMap::<1>::new());
        let writer = map.clone();
        let worker = thread::spawn(move || writer.try_insert(0, 10));
        let second = map.try_insert(0, 20);
        let first = worker.join().unwrap();
        match (first, second) {
            (Ok(()), Err(10)) => assert_eq!(map.get(0), Some(10)),
            (Err(20), Ok(())) => assert_eq!(map.get(0), Some(20)),
            _ => panic!("exactly one insertion must succeed"),
        }
        assert_eq!(map.len(), 1);
    });
}

#[test]
fn id_two_try_inserts() {
    model(|| {
        let map = Arc::new(AtomicIdMap::<1>::new());
        let first_id = id(0, 1);
        let second_id = id(0, 2);
        let writer = map.clone();
        let worker = thread::spawn(move || writer.try_insert(first_id, 10));
        let second = map.try_insert(second_id, 20);
        let first = worker.join().unwrap();
        match (first, second) {
            (Ok(()), Err(pair)) => {
                assert_eq!(pair, (first_id, 10));
                assert_eq!(map.occupant(0), Some(pair));
            }
            (Err(pair), Ok(())) => {
                assert_eq!(pair, (second_id, 20));
                assert_eq!(map.occupant(0), Some(pair));
            }
            _ => panic!("exactly one generation must win"),
        }
        assert_eq!(map.len(), 1);
    });
}

#[test]
fn stale_remove_races_new_generation_insert() {
    model(|| {
        let map = Arc::new(AtomicIdMap::<1>::new());
        let old = id(0, 1);
        let new = id(0, 2);
        map.insert(old, 10);
        let writer = map.clone();
        let worker = thread::spawn(move || {
            let evicted = writer.insert(new, 20);
            assert!(evicted.is_none() || evicted == Some(10));
        });
        let removed = map.remove(old);
        assert!(removed.is_none() || removed == Some(10));
        worker.join().unwrap();
        assert_eq!(map.get(old), None);
        assert_eq!(map.occupant(0), Some((new, 20)));
        assert_eq!(map.len(), 1);
    });
}

#[test]
fn conditional_remove_races_new_generation_insert() {
    model(|| {
        let map = Arc::new(AtomicIdMap::<1>::new());
        let old = id(0, 1);
        let new = id(0, 2);
        map.insert(old, 10);
        let writer = map.clone();
        let worker = thread::spawn(move || writer.insert(new, 20));
        let removed = map.fetch_update(old, |_| Some(None));
        assert!(removed == Ok(Some(10)) || removed == Ok(None));
        worker.join().unwrap();
        assert_eq!(map.occupant(0), Some((new, 20)));
        assert_eq!(map.len(), 1);
    });
}

#[test]
fn absent_generation_compare_exchange_preserves_other_occupants() {
    model(|| {
        let map = Arc::new(AtomicIdMap::<1>::new());
        let absent = id(0, 1);
        let old = id(0, 2);
        let new = id(0, 3);
        map.insert(old, 10);
        let writer = map.clone();
        let worker = thread::spawn(move || assert_eq!(writer.insert(new, 20), Some(10)));
        assert_eq!(map.compare_exchange(absent, None, None), Ok(None));
        worker.join().unwrap();
        assert_eq!(map.occupant(0), Some((new, 20)));
        assert_eq!(map.len(), 1);
    });
}

#[test]
fn direct_traversal_races_old_and_new_chunks() {
    // Loom BASE = 2: page 1 joins chunk 0; page 2 installs chunk 1.
    model(|| {
        let map = Arc::new(AtomicDirectMap::<1>::new());
        map.insert(0, 100);
        let writer = map.clone();
        let worker = thread::spawn(move || {
            writer.insert(1, 101);
            writer.insert(2, 102);
        });
        // Yields cost no preemption and let loom run the writer before and
        // during the traversal; without them every traversal saw only key 0.
        thread::yield_now();
        let seen: alloc::vec::Vec<_> = map.iter().inspect(|_| thread::yield_now()).collect();
        assert_eq!(seen.first(), Some(&(0, 100)));
        assert!(seen.windows(2).all(|pair| pair[0].0 < pair[1].0));
        assert!(seen.iter().all(|&(key, value)| value == 100 + key as u64));
        worker.join().unwrap();
        assert_eq!(
            map.iter().collect::<alloc::vec::Vec<_>>(),
            [(0, 100), (1, 101), (2, 102)]
        );
        assert_eq!(map.len(), 3);
        assert_eq!(map.allocated_pages(), 3);
    });
}

#[test]
fn id_traversal_races_old_and_new_chunks() {
    model(|| {
        let map = Arc::new(AtomicIdMap::<1>::new());
        map.insert(id(0, 1), 100);
        let writer = map.clone();
        let worker = thread::spawn(move || {
            writer.insert(id(1, 2), 101);
            writer.insert(id(2, 3), 102);
        });
        // Yields cost no preemption and let loom run the writer before and
        // during the traversal; without them every traversal saw only key 0.
        thread::yield_now();
        let seen: alloc::vec::Vec<_> = map.iter().inspect(|_| thread::yield_now()).collect();
        assert_eq!(seen.first(), Some(&(id(0, 1), 100)));
        assert!(
            seen.windows(2)
                .all(|pair| pair[0].0.index() < pair[1].0.index())
        );
        assert!(seen.iter().all(|&(key, value)| {
            key.generation().get() == key.index() + 1 && value == 100 + key.index()
        }));
        worker.join().unwrap();
        assert_eq!(
            map.iter().collect::<alloc::vec::Vec<_>>(),
            [(id(0, 1), 100), (id(1, 2), 101), (id(2, 3), 102)]
        );
        assert_eq!(map.len(), 3);
        assert_eq!(map.allocated_pages(), 3);
    });
}

// Publication litmus tests: a payload written Relaxed before the slot write
// must be visible to a thread that observes the slot value. The page is
// installed before spawning, so the spine's own publication cannot supply
// the synchronization the slot operations are meant to provide.

#[test]
fn direct_insert_publishes_payload_to_get() {
    use loom::sync::atomic::{AtomicU64, Ordering};
    model(|| {
        let map = Arc::new(AtomicDirectMap::<1>::new());
        map.insert(0, 0);
        assert_eq!(map.remove(0), Some(0));
        let payload = Arc::new(AtomicU64::new(0));
        let (writer, data) = (map.clone(), payload.clone());
        let producer = thread::spawn(move || {
            data.store(42, Ordering::Relaxed);
            writer.insert(0, 1);
        });
        if map.get(0) == Some(1) {
            assert_eq!(payload.load(Ordering::Relaxed), 42);
        }
        producer.join().unwrap();
    });
}

// The producer swaps rather than running fetch_update: with every slot
// ordering weakened to Relaxed, loom (0.7, preemption bound 2) finds the
// stale payload through a swap but not through fetch_update's load + CAS
// loop, so a fetch_update producer would not test the orderings at all.
#[test]
fn id_insert_publishes_payload_to_failed_compare_exchange() {
    use loom::sync::atomic::{AtomicU64, Ordering};
    model(|| {
        let map = Arc::new(AtomicIdMap::<1>::new());
        let key = id(0, 1);
        map.insert(key, 0);
        let payload = Arc::new(AtomicU64::new(0));
        let (writer, data) = (map.clone(), payload.clone());
        let producer = thread::spawn(move || {
            data.store(42, Ordering::Relaxed);
            writer.insert(key, 1);
        });
        // A failed compare-exchange returns the observed value: it must
        // synchronize with the write that stored it.
        if map.compare_exchange(key, Some(7), Some(8)) == Err(Some(1)) {
            assert_eq!(payload.load(Ordering::Relaxed), 42);
        }
        producer.join().unwrap();
    });
}
