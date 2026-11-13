//! Bounded explorations of guard publication, removal, reuse and extraction.

use alloc::vec::Vec;
use loom::sync::atomic::{AtomicUsize, Ordering};
use loom::sync::{Arc, Condvar, Mutex};
use loom::thread;

use super::ConcurrentHive;

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
    drops: Arc<AtomicUsize>,
}

impl Drop for Value {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn get_remove_and_last_guard_drop() {
    model(|| {
        let hive = Arc::new(ConcurrentHive::<Value, 2>::new());
        let drops = Arc::new(AtomicUsize::new(0));
        let id = hive.insert(Value {
            tag: 7,
            drops: drops.clone(),
        });
        let reader = hive.clone();
        let observed = drops.clone();
        let a = thread::spawn(move || {
            if let Some(value) = reader.get(id) {
                assert_eq!(value.id(), id);
                assert_eq!(value.tag, 7);
                thread::yield_now();
                assert_eq!(observed.load(Ordering::SeqCst), 0);
                assert_eq!(value.tag, 7);
                drop(value);
            }
        });
        let remover = hive.clone();
        let b = thread::spawn(move || assert!(remover.remove(id)));
        a.join().unwrap();
        b.join().unwrap();
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert!(!hive.contains(id));
        assert!(hive.is_empty());
        drop(hive);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn two_removes_have_exactly_one_winner() {
    model(|| {
        let hive = Arc::new(ConcurrentHive::<Value, 2>::new());
        let drops = Arc::new(AtomicUsize::new(0));
        let id = hive.insert(Value {
            tag: 1,
            drops: drops.clone(),
        });
        let mut threads = Vec::new();
        for _ in 0..2 {
            let hive = hive.clone();
            threads.push(thread::spawn(move || hive.remove(id)));
        }
        let wins: usize = threads
            .into_iter()
            .map(|t| usize::from(t.join().unwrap()))
            .sum();
        assert_eq!(wins, 1);
        assert_eq!(hive.len(), 0);
        drop(hive);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn stale_get_races_removal_and_republication() {
    model(|| {
        let hive = Arc::new(ConcurrentHive::<Value, 2>::new());
        let drops = Arc::new(AtomicUsize::new(0));
        let old = hive.insert(Value {
            tag: 1,
            drops: drops.clone(),
        });
        let writer = hive.clone();
        let counted = drops.clone();
        let a = thread::spawn(move || {
            assert!(writer.remove(old));
            writer.insert(Value {
                tag: 2,
                drops: counted,
            })
        });
        let reader = hive.clone();
        let b = thread::spawn(move || {
            if let Some(value) = reader.get(old) {
                assert_eq!(value.tag, 1);
                thread::yield_now();
                assert_eq!(value.tag, 1);
            }
        });
        let new = a.join().unwrap();
        b.join().unwrap();
        assert_ne!(old, new);
        assert!(hive.get(old).is_none());
        assert_eq!(hive.get(new).unwrap().tag, 2);
        assert_eq!(hive.len(), 1);
        drop(hive);
        assert_eq!(drops.load(Ordering::SeqCst), 2);
    });
}

#[test]
fn stale_get_during_forced_same_slot_reuse() {
    model(|| {
        let hive = Arc::new(ConcurrentHive::<Value, 2>::new());
        let drops = Arc::new(AtomicUsize::new(0));
        let old = hive.insert(Value {
            tag: 1,
            drops: drops.clone(),
        });
        assert!(hive.remove(old));
        let writer = hive.clone();
        let counted = drops.clone();
        let a = thread::spawn(move || {
            let id = writer.insert(Value {
                tag: 2,
                drops: counted,
            });
            assert_eq!(id.index(), old.index());
            id
        });
        let reader = hive.clone();
        let b = thread::spawn(move || {
            assert!(reader.get(old).is_none());
            thread::yield_now();
            assert!(reader.get(old).is_none());
        });
        let new = a.join().unwrap();
        b.join().unwrap();
        assert_eq!(hive.get(new).unwrap().tag, 2);
        drop(hive);
        assert_eq!(drops.load(Ordering::SeqCst), 2);
    });
}

#[test]
fn take_waits_for_guard_without_dropping_its_value() {
    model(|| {
        let hive = Arc::new(ConcurrentHive::<Value, 2>::new());
        let drops = Arc::new(AtomicUsize::new(0));
        let id = hive.insert(Value {
            tag: 9,
            drops: drops.clone(),
        });
        let guard = hive.get(id).unwrap();
        let taker = hive.clone();
        let a = thread::spawn(move || taker.take(id).unwrap());
        thread::yield_now();
        assert_eq!(guard.tag, 9);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(guard);
        let value = a.join().unwrap();
        assert_eq!(value.tag, 9);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        assert_eq!(hive.len(), 0);
        drop(value);
        drop(hive);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn concurrent_inserts_publish_distinct_readable_values() {
    model(|| {
        let hive = Arc::new(ConcurrentHive::<Value, 2>::new());
        let drops = Arc::new(AtomicUsize::new(0));
        let mut threads = Vec::new();
        for _ in 0..2 {
            let hive = hive.clone();
            let drops = drops.clone();
            threads.push(thread::spawn(move || {
                hive.insert_with(|id| Value {
                    tag: id.index() as usize,
                    drops,
                })
            }));
        }
        let ids: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
        assert_ne!(ids[0], ids[1]);
        for id in ids {
            assert_eq!(hive.get(id).unwrap().tag, id.index() as usize);
        }
        assert_eq!(hive.len(), 2);
        drop(hive);
        assert_eq!(drops.load(Ordering::SeqCst), 2);
    });
}

struct Gate {
    phase: Mutex<u8>,
    changed: Condvar,
}

struct GatedValue {
    drops: Arc<AtomicUsize>,
    gate: Option<Arc<Gate>>,
}

impl Drop for GatedValue {
    fn drop(&mut self) {
        if let Some(gate) = &self.gate {
            let mut phase = gate.phase.lock().unwrap();
            *phase = 1;
            gate.changed.notify_all();
            while *phase != 2 {
                phase = gate.changed.wait(phase).unwrap();
            }
        }
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn stale_stack_hint_cannot_claim_a_mid_drop_cell() {
    model(|| {
        let hive = Arc::new(ConcurrentHive::<GatedValue, 2>::new());
        let drops = Arc::new(AtomicUsize::new(0));
        let gate = Arc::new(Gate {
            phase: Mutex::new(0),
            changed: Condvar::new(),
        });
        let old = hive.insert(GatedValue {
            drops: drops.clone(),
            gate: Some(gate.clone()),
        });
        let remover = hive.clone();
        let a = thread::spawn(move || assert!(remover.remove(old)));
        {
            let mut phase = gate.phase.lock().unwrap();
            while *phase != 1 {
                phase = gate.changed.wait(phase).unwrap();
            }
        }
        // Its destructor is active, with the value cell checked as a loom
        // mutable access. Simulate the stale index a wrapped stack can return.
        hive.inject_busy_hint(old.index());
        let new = hive.insert(GatedValue {
            drops: drops.clone(),
            gate: None,
        });
        assert_ne!(new.index(), old.index());
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        {
            let mut phase = gate.phase.lock().unwrap();
            *phase = 2;
            gate.changed.notify_all();
        }
        a.join().unwrap();
        let reused = hive.insert(GatedValue {
            drops: drops.clone(),
            gate: None,
        });
        assert_eq!(reused.index(), old.index());
        assert_ne!(reused, old);
        assert!(hive.get(old).is_none());
        assert_eq!(hive.len(), 2);
        drop(hive);
        assert_eq!(drops.load(Ordering::SeqCst), 3);
    });
}
