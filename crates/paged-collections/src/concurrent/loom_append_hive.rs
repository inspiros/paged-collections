//! Two-thread explorations of append-hive claims and plain-reference reads.

#[cfg(not(feature = "std"))]
extern crate std;

use core::cell::Cell;
use core::num::NonZeroU32;
use loom::sync::atomic::{AtomicUsize, Ordering};
use loom::sync::{Arc, Condvar, Mutex};
use loom::thread;
use std::panic::{AssertUnwindSafe, catch_unwind};

use super::ConcurrentAppendHive;
use crate::id_alloc::Id;

fn model(f: impl Fn() + Send + Sync + 'static) {
    let mut builder = loom::model::Builder::new();
    if builder.preemption_bound.is_none() {
        builder.preemption_bound = Some(2);
    }
    builder.max_branches = 10_000;
    builder.check(f);
}

fn first_id() -> Id {
    Id::new(0, NonZeroU32::MIN)
}

#[derive(Debug, PartialEq, Eq)]
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

    fn check(&self, tag: usize) {
        assert_eq!(self.tag, tag);
        assert_eq!(self.complement, !tag);
    }
}

#[test]
fn two_inserts_publish_distinct_readable_values() {
    model(|| {
        let hive = Arc::new(ConcurrentAppendHive::<Value, 2>::new());
        let writer = hive.clone();
        let a = thread::spawn(move || {
            let id = writer.insert(Value::new(1));
            writer.get(id).unwrap().check(1);
            id
        });
        let b = hive.insert(Value::new(2));
        let borrowed = hive.get(b).unwrap();
        thread::yield_now();
        borrowed.check(2);
        let a = a.join().unwrap();
        assert_ne!(a, b);
        hive.get(a).unwrap().check(1);
        assert_eq!(hive.len(), 2);
        assert_eq!(hive.iter().count(), 2);
    });
}

#[test]
fn polling_fresh_id_sees_none_or_a_fully_written_value() {
    model(|| {
        let hive = Arc::new(ConcurrentAppendHive::<Value, 2>::new());
        hive.prepare_fresh_page();
        let writer = hive.clone();
        let producer = thread::spawn(move || writer.insert(Value::new(7)));
        // Bounded polling keeps the branch budget small without an external
        // handshake publishing the value on behalf of the slot's state.
        for _ in 0..3 {
            if let Some(value) = hive.get(first_id()) {
                value.check(7);
                thread::yield_now();
                value.check(7);
                break;
            }
            thread::yield_now();
        }
        assert_eq!(producer.join().unwrap(), first_id());
        hive.get(first_id()).unwrap().check(7);
    });
}

#[test]
fn get_acquires_payload_written_before_publication() {
    model(|| {
        let hive = Arc::new(ConcurrentAppendHive::<u32, 2>::new());
        // Both the chunk and page are installed before the relaxed payload
        // write, so only slot-state publication can carry that write to get.
        hive.prepare_fresh_page();
        let payload = Arc::new(AtomicUsize::new(0));
        let writer = hive.clone();
        let written = payload.clone();
        let producer = thread::spawn(move || {
            written.store(42, Ordering::Relaxed);
            assert_eq!(writer.insert(7), first_id());
        });
        // Yielding lets the producer finish without spending a preemption;
        // without it loom never reaches a published read before the join.
        for _ in 0..2 {
            if let Some(value) = hive.get(first_id()) {
                assert_eq!(*value, 7);
                // No join or id-transfer handshake precedes this relaxed read.
                assert_eq!(payload.load(Ordering::Relaxed), 42);
                break;
            }
            thread::yield_now();
        }
        producer.join().unwrap();
        assert_eq!(hive.get(first_id()), Some(&7));
    });
}

#[test]
fn iter_acquires_payload_written_before_publication() {
    model(|| {
        let hive = Arc::new(ConcurrentAppendHive::<u32, 2>::new());
        // Preinstalled page: only slot-state publication can carry the payload.
        hive.prepare_fresh_page();
        let payload = Arc::new(AtomicUsize::new(0));
        let writer = hive.clone();
        let written = payload.clone();
        let producer = thread::spawn(move || {
            writer.insert_with(|_| {
                written.store(42, Ordering::Relaxed);
                7
            })
        });
        // Yield first (see above), so the iterator can see the reservation.
        thread::yield_now();
        for (id, value) in hive.iter() {
            assert_eq!((id, *value), (first_id(), 7));
            // No join or handshake precedes this relaxed read.
            assert_eq!(payload.load(Ordering::Relaxed), 42);
        }
        assert_eq!(producer.join().unwrap(), first_id());
    });
}

#[test]
fn panicking_factory_races_insert_without_reviving_its_spent_id() {
    model(|| {
        let hive = Arc::new(ConcurrentAppendHive::<Value, 2>::new());
        let failing = hive.clone();
        let factory = thread::spawn(move || {
            let attempted = Cell::new(None);
            let result = catch_unwind(AssertUnwindSafe(|| {
                failing.insert_with(|id| {
                    attempted.set(Some(id));
                    assert!(failing.get(id).is_none());
                    thread::yield_now();
                    panic!("factory failed");
                });
            }));
            assert!(result.is_err());
            attempted.get().unwrap()
        });
        let live = hive.insert(Value::new(9));
        let spent = factory.join().unwrap();
        assert_ne!(live, spent);
        assert!(hive.get(spent).is_none());
        hive.get(live).unwrap().check(9);
        assert_eq!(hive.len(), 1);
        let next = hive.insert(Value::new(11));
        assert_ne!(live, next);
        assert_ne!(spent, next);
        assert!(hive.get(spent).is_none());
        hive.get(live).unwrap().check(9);
        hive.get(next).unwrap().check(11);
        assert_eq!(hive.len(), 2);
    });
}

struct Gate {
    phase: Mutex<u8>,
    changed: Condvar,
}

#[test]
fn stale_stack_hint_cannot_claim_a_busy_factory_slot() {
    model(|| {
        let hive = Arc::new(ConcurrentAppendHive::<Value, 2>::new());
        let gate = Arc::new(Gate {
            phase: Mutex::new(0),
            changed: Condvar::new(),
        });
        let writer = hive.clone();
        let blocked = gate.clone();
        let producer = thread::spawn(move || {
            writer.insert_with(|id| {
                assert_eq!(id, first_id());
                let mut phase = blocked.phase.lock().unwrap();
                *phase = 1;
                blocked.changed.notify_all();
                while *phase != 2 {
                    phase = blocked.changed.wait(phase).unwrap();
                }
                Value::new(1)
            })
        });
        {
            let mut phase = gate.phase.lock().unwrap();
            while *phase != 1 {
                phase = gate.changed.wait(phase).unwrap();
            }
        }
        hive.inject_busy_hint(0);
        assert!(hive.get(first_id()).is_none());
        let other = hive.insert(Value::new(2));
        assert_ne!(other.index(), 0);
        assert_eq!(hive.len(), 1);
        {
            let mut phase = gate.phase.lock().unwrap();
            *phase = 2;
            gate.changed.notify_all();
        }
        let first = producer.join().unwrap();
        assert_eq!(first, first_id());
        hive.get(first).unwrap().check(1);
        hive.get(other).unwrap().check(2);
        assert_eq!(hive.len(), 2);
    });
}
