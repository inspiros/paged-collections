//! Regression tests for the 2026-11 pre-release review findings.

#![cfg(feature = "concurrent")]
#![cfg(not(loom))]
#![cfg(all(
    any(target_pointer_width = "32", target_pointer_width = "64"),
    target_has_atomic = "ptr",
    target_has_atomic = "64"
))]

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Weak, mpsc};
use std::thread;
use std::time::Duration;

use paged_collections::concurrent::{ConcurrentDirectMap, ConcurrentSparseMap};
use paged_collections::page::Flat;

type ReentrantMap = ConcurrentSparseMap<usize, ReinsertOnDrop, 1, Flat>;

struct ReinsertOnDrop {
    map: Option<Weak<ReentrantMap>>,
    completed: Arc<AtomicUsize>,
}

impl Drop for ReinsertOnDrop {
    fn drop(&mut self) {
        if let Some(map) = self.map.take().and_then(|map| map.upgrade()) {
            let _ = map.try_insert(
                0,
                Self {
                    map: None,
                    completed: self.completed.clone(),
                },
            );
            self.completed.fetch_add(1, Ordering::SeqCst);
        }
    }
}

#[test]
fn sparse_batch_range_panic_releases_reservations_before_value_drop() {
    // Finding 9: an invalid batch item's destructor must not wait on an earlier reservation.
    let (finished_tx, finished_rx) = mpsc::channel();
    let worker = thread::spawn(move || {
        let map = Arc::new(ReentrantMap::new());
        let completed = Arc::new(AtomicUsize::new(0));
        let result = catch_unwind(AssertUnwindSafe(|| {
            map.try_insert_many([
                (
                    0,
                    ReinsertOnDrop {
                        map: None,
                        completed: completed.clone(),
                    },
                ),
                (
                    usize::MAX,
                    ReinsertOnDrop {
                        map: Some(Arc::downgrade(&map)),
                        completed: completed.clone(),
                    },
                ),
            ]);
        }));
        let outcome = (
            result.is_err(),
            completed.load(Ordering::SeqCst),
            map.contains_key(&0),
        );
        drop(map);
        finished_tx.send(outcome).unwrap();
    });

    // Do not join a deadlocked worker: dropping its handle lets the test fail on time.
    let outcome = finished_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("batch unwind deadlocked in a destructor inserting an already-reserved key");
    worker
        .join()
        .expect("batch worker panicked outside catch_unwind");
    assert_eq!(outcome, (true, 1, true));
}

struct CountedDrop {
    id: usize,
    drops: Arc<Vec<AtomicUsize>>,
}

impl Drop for CountedDrop {
    fn drop(&mut self) {
        self.drops[self.id].fetch_add(1, Ordering::SeqCst);
        assert_ne!(self.id, 0, "intentional value destructor panic");
    }
}

#[test]
fn flat_direct_map_drop_panic_still_drops_all_other_pages() {
    // Finding 12: a panicking page destructor must not leak the remaining Flat directory pages.
    let keys = [0, 1, 2, 31, 32, 96];
    let drops = Arc::new(keys.iter().map(|_| AtomicUsize::new(0)).collect::<Vec<_>>());
    let map = ConcurrentDirectMap::<usize, CountedDrop, 1, Flat>::new();
    // PAGE=1 puts every value in its own page; the keys span three spine chunks.
    for (id, key) in keys.into_iter().enumerate() {
        assert!(
            map.insert(
                key,
                CountedDrop {
                    id,
                    drops: drops.clone(),
                },
            )
            .is_none()
        );
    }

    let result = catch_unwind(AssertUnwindSafe(|| drop(map)));
    assert!(result.is_err(), "the intentional destructor panic was lost");
    let actual: Vec<_> = drops.iter().map(|n| n.load(Ordering::SeqCst)).collect();
    assert_eq!(
        actual,
        vec![1; keys.len()],
        "some pages' values were leaked"
    );
}
