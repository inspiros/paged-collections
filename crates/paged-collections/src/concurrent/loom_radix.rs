//! Two-thread explorations of radix splits, page ownership and publication.

#[cfg(not(feature = "std"))]
extern crate std;

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::ptr;
use loom::sync::{Arc, Condvar, Mutex};
use loom::thread;
use std::sync::Arc as StdArc;
use std::sync::atomic::{AtomicBool as CoverageFlag, Ordering as CoverageOrdering};

use super::atomic::{AtomicUsize, Ordering};
use super::radix::RadixSpine;

fn model(f: impl Fn() + Send + Sync + 'static) {
    let mut builder = loom::model::Builder::new();
    if builder.preemption_bound.is_none() {
        builder.preemption_bound = Some(2);
    }
    builder.max_branches = 10_000;
    builder.check(f);
}

// These unmodelled flags accumulate coverage across executions. Only the
// reader writes them, after its assertions; they cannot publish page data.
fn coverage_flag() -> StdArc<CoverageFlag> {
    StdArc::new(CoverageFlag::new(false))
}

#[test]
fn colliding_installs_remain_reachable() {
    model(|| {
        let spine = Arc::new(RadixSpine::new());
        for number in [0, 16] {
            spine.get_or_insert_with(number, || Box::new(number));
        }
        let writer = spine.clone();
        let producer = thread::spawn(move || {
            *writer.get_or_insert_with(2, || {
                thread::yield_now();
                Box::new(2)
            })
        });
        // Both numbers route to the slot holding leaf 0 in the root branch.
        thread::yield_now();
        assert_eq!(
            *spine.get_or_insert_with(1, || {
                thread::yield_now();
                Box::new(1)
            }),
            1
        );
        assert_eq!(producer.join().unwrap(), 2);
        for number in [0, 1, 2, 16] {
            assert_eq!(spine.get(number), Some(&number));
        }
        spine.check();
    });
}

struct Gate {
    arrived: Mutex<usize>,
    changed: Condvar,
}

impl Gate {
    fn meet(&self) {
        let mut arrived = self.arrived.lock().unwrap();
        *arrived += 1;
        self.changed.notify_all();
        while *arrived != 2 {
            arrived = self.changed.wait(arrived).unwrap();
        }
    }
}

struct CountedPage {
    value: usize,
    dropped: Arc<AtomicUsize>,
}

impl Drop for CountedPage {
    fn drop(&mut self) {
        assert_eq!(self.dropped.fetch_add(1, Ordering::SeqCst), 0);
    }
}

#[test]
fn same_number_keeps_one_page_and_drops_the_loser_once() {
    model(|| {
        let spine = Arc::new(RadixSpine::new());
        let drops = [Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0))];
        let gate = Arc::new(Gate {
            arrived: Mutex::new(0),
            changed: Condvar::new(),
        });
        let writer = spine.clone();
        let other_gate = gate.clone();
        let other_drops = drops[1].clone();
        let producer = thread::spawn(move || {
            let page = writer.get_or_insert_with(3, || {
                // Neither factory can finish until both have run. This makes
                // a losing made page mandatory in every execution.
                other_gate.meet();
                Box::new(CountedPage {
                    value: 1,
                    dropped: other_drops,
                })
            });
            ptr::from_ref(page).addr()
        });
        let page = spine.get_or_insert_with(3, || {
            gate.meet();
            Box::new(CountedPage {
                value: 0,
                dropped: drops[0].clone(),
            })
        });
        let winner = page.value;
        assert_eq!(ptr::from_ref(page).addr(), producer.join().unwrap());
        assert_eq!(drops[winner].load(Ordering::SeqCst), 0);
        assert_eq!(drops[1 - winner].load(Ordering::SeqCst), 1);
        spine.check();
        let mut spine = Arc::try_unwrap(spine).ok().unwrap();
        for (number, page) in spine.iter_mut() {
            assert_eq!(number, 3);
            page.value += 10;
        }
        assert_eq!(spine.get(3).unwrap().value, winner + 10);
        spine.clear();
        assert!(spine.get(3).is_none());
        assert_eq!(spine.iter_mut().count(), 0);
        spine.check();
        drop(spine);
        assert!(drops.iter().all(|n| n.load(Ordering::SeqCst) == 1));
    });
}

#[test]
fn splitting_a_leaf_never_hides_its_page() {
    let seen = coverage_flag();
    let reached = seen.clone();
    model(move || {
        let spine = Arc::new(RadixSpine::new());
        let old = spine.get_or_insert_with(0, || Box::new(41));
        let writer = spine.clone();
        let producer = thread::spawn(move || {
            assert_eq!(*writer.get_or_insert_with(1, || Box::new(42)), 42);
        });
        // Main-thread readers must yield before reading after spawn; otherwise
        // loom can miss every execution in which publication precedes a read.
        thread::yield_now();
        for _ in 0..3 {
            let page = spine.get(0).expect("an installed page never disappears");
            assert!(ptr::eq(page, old));
            assert_eq!(*page, 41);
            if spine.get(1).is_some() {
                reached.store(true, CoverageOrdering::Relaxed);
            }
            thread::yield_now();
        }
        producer.join().unwrap();
        assert_eq!(spine.get(1), Some(&42));
        spine.check();
    });
    assert!(seen.load(CoverageOrdering::Relaxed));
}

#[test]
fn splitting_above_a_branch_preserves_a_subtree_install() {
    model(|| {
        let spine = Arc::new(RadixSpine::new());
        for number in [0, 1] {
            spine.get_or_insert_with(number, || Box::new(number));
        }
        let writer = spine.clone();
        let producer = thread::spawn(move || {
            // 4 mismatches the prefix of the existing shift-0 branch.
            assert_eq!(*writer.get_or_insert_with(4, || Box::new(4)), 4);
        });
        thread::yield_now();
        assert_eq!(
            *spine.get_or_insert_with(2, || {
                // The walk has already selected a slot in the old branch;
                // allow the other installer to wrap it before our CAS.
                thread::yield_now();
                Box::new(2)
            }),
            2
        );
        producer.join().unwrap();
        for number in [0, 1, 2, 4] {
            assert_eq!(spine.get(number), Some(&number));
        }
        spine.check();
    });
}

#[test]
fn get_acquires_contents_written_in_the_factory() {
    // Cover direct publication, a root split, and a split below a root that
    // predates the payload write (where the child load must acquire it).
    for mode in 0..3 {
        let seen = coverage_flag();
        let reached = seen.clone();
        model(move || {
            let spine = Arc::new(RadixSpine::new());
            if mode > 0 {
                spine.get_or_insert_with(0, || Box::new(Arc::new(AtomicUsize::new(7))));
            }
            if mode > 1 {
                spine.get_or_insert_with(4, || Box::new(Arc::new(AtomicUsize::new(8))));
            }
            // This atomic predates the spawn, so a missing publication edge
            // can expose its original zero through the page's relaxed read.
            let payload = Arc::new(AtomicUsize::new(0));
            let writer = spine.clone();
            let producer = thread::spawn(move || {
                writer.get_or_insert_with(1, || {
                    payload.store(42, Ordering::Relaxed);
                    Box::new(payload)
                });
            });
            thread::yield_now();
            for _ in 0..3 {
                if let Some(page) = spine.get(1) {
                    // No join or payload-transfer handshake precedes this read.
                    assert_eq!(page.load(Ordering::Relaxed), 42);
                    reached.store(true, CoverageOrdering::Relaxed);
                    break;
                }
                thread::yield_now();
            }
            producer.join().unwrap();
            assert_eq!(spine.get(1).unwrap().load(Ordering::Relaxed), 42);
            spine.check();
        });
        assert!(seen.load(CoverageOrdering::Relaxed));
    }
}

#[test]
fn shared_iteration_stays_ordered_and_sees_preexisting_pages() {
    let seen = coverage_flag();
    let reached = seen.clone();
    model(move || {
        let spine = Arc::new(RadixSpine::new());
        let preexisting = [0, 1, 16];
        for number in preexisting {
            spine.get_or_insert_with(number, || Box::new(Arc::new(AtomicUsize::new(number + 10))));
        }
        let new_numbers = [64, 2, 4];
        let payloads: Vec<_> = new_numbers
            .iter()
            .map(|_| Arc::new(AtomicUsize::new(0)))
            .collect();
        let iter = spine.iter();
        let writer = spine.clone();
        let producer = thread::spawn(move || {
            for (number, payload) in new_numbers.into_iter().zip(payloads) {
                writer.get_or_insert_with(number, || {
                    payload.store(number + 10, Ordering::Relaxed);
                    Box::new(payload)
                });
                thread::yield_now();
            }
        });
        thread::yield_now();
        let mut numbers = Vec::new();
        for (number, page) in iter {
            assert!(preexisting.contains(&number) || new_numbers.contains(&number));
            assert_eq!(page.load(Ordering::Relaxed), number + 10);
            if new_numbers.contains(&number) {
                reached.store(true, CoverageOrdering::Relaxed);
            }
            numbers.push(number);
            thread::yield_now();
        }
        assert!(numbers.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(preexisting.iter().all(|number| numbers.contains(number)));
        producer.join().unwrap();
        assert_eq!(spine.iter().count(), preexisting.len() + new_numbers.len());
        spine.check();
    });
    assert!(seen.load(CoverageOrdering::Relaxed));
}
