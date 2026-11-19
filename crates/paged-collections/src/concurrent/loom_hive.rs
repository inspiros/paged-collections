//! Bounded explorations of guard publication, removal, reuse and extraction.

#[cfg(not(feature = "std"))]
extern crate std;

use alloc::vec::Vec;
use core::num::NonZeroU32;
use loom::sync::atomic::{AtomicUsize, Ordering};
use loom::sync::{Arc, Condvar, Mutex};
use loom::thread;
use std::sync::atomic::{AtomicBool as StdAtomicBool, Ordering as StdOrdering};

use super::ConcurrentHive;
use crate::hive::{Hive, RawPage, RawSlot};
use crate::id_alloc::Id;

fn model(f: impl Fn() + Send + Sync + 'static) {
    let mut builder = loom::model::Builder::new();
    if builder.preemption_bound.is_none() {
        builder.preemption_bound = Some(2);
    }
    builder.max_branches = 10_000;
    builder.check(f);
}

// Count actual branches reached by the shared scan across loom schedules.
static SNAPSHOT_MARKED: StdAtomicBool = StdAtomicBool::new(false);
static SNAPSHOT_BUSY: StdAtomicBool = StdAtomicBool::new(false);
// Loom runs one thread at a time, so this plain flag is the writer's real
// phase whenever the scan observes BUSY.
static IN_FACTORY: StdAtomicBool = StdAtomicBool::new(false);
static CLAIM_BUSY: StdAtomicBool = StdAtomicBool::new(false);
static FINALIZE_BUSY: StdAtomicBool = StdAtomicBool::new(false);

pub(super) fn record_snapshot_marked() {
    SNAPSHOT_MARKED.store(true, StdOrdering::Relaxed);
}

pub(super) fn record_snapshot_busy(index: u32) {
    if index == 0 {
        SNAPSHOT_BUSY.store(true, StdOrdering::Relaxed);
        if IN_FACTORY.load(StdOrdering::SeqCst) {
            CLAIM_BUSY.store(true, StdOrdering::Relaxed);
        } else {
            FINALIZE_BUSY.store(true, StdOrdering::Relaxed);
        }
    }
}

#[test]
fn snapshot_races_removal_and_slot_reuse() {
    #[derive(Debug)]
    struct SnapshotValue {
        tag: u8,
        yield_on_drop: bool,
    }

    impl Clone for SnapshotValue {
        fn clone(&self) -> Self {
            Self {
                tag: self.tag,
                yield_on_drop: false,
            }
        }
    }

    impl Drop for SnapshotValue {
        fn drop(&mut self) {
            if self.yield_on_drop {
                thread::yield_now(); // let the snapshot see finalize-BUSY
            }
        }
    }

    static READ_REPLACEMENT: StdAtomicBool = StdAtomicBool::new(false);
    static REUSED: StdAtomicBool = StdAtomicBool::new(false);
    SNAPSHOT_MARKED.store(false, StdOrdering::Relaxed);
    CLAIM_BUSY.store(false, StdOrdering::Relaxed);
    FINALIZE_BUSY.store(false, StdOrdering::Relaxed);
    READ_REPLACEMENT.store(false, StdOrdering::Relaxed);
    REUSED.store(false, StdOrdering::Relaxed);
    model(|| {
        let plain = Hive::<_, 1>::try_from_raw_pages([RawPage {
            floor: 0,
            slots: alloc::vec![RawSlot::Occupied {
                version: 3,
                value: SnapshotValue {
                    tag: 11,
                    yield_on_drop: true
                },
            }],
        }])
        .unwrap();
        IN_FACTORY.store(false, StdOrdering::SeqCst);
        let hive = Arc::new(ConcurrentHive::from(plain));
        let stale = Id::new(0, NonZeroU32::MIN);
        let old = Id::new(0, NonZeroU32::new(2).unwrap());
        let writer = hive.clone();
        let mutation = thread::spawn(move || {
            let hive = &*writer;
            let guard = hive.get(old).unwrap();
            assert!(hive.remove(old));
            thread::yield_now(); // let the snapshot see MARKED
            drop(guard);
            hive.insert_with(|_| {
                IN_FACTORY.store(true, StdOrdering::SeqCst);
                thread::yield_now(); // let the snapshot see claim-BUSY
                IN_FACTORY.store(false, StdOrdering::SeqCst);
                SnapshotValue {
                    tag: 22,
                    yield_on_drop: false,
                }
            })
        });
        // Two yields: the first lets the writer reach MARKED, the second its
        // claim; preemptions cover the other phases.
        thread::yield_now();
        thread::yield_now();

        let mut snapshot = hive.snapshot();
        let page = snapshot.raw_pages().next().unwrap();
        match page.slots.as_slice() {
            [RawSlot::Occupied { version: 3, value }] => assert_eq!(value.tag, 11),
            [RawSlot::Occupied { version: 5, value }] => {
                assert_eq!(value.tag, 22);
                READ_REPLACEMENT.store(true, StdOrdering::Relaxed);
            }
            [RawSlot::Vacant { version: 4 }] => assert!(snapshot.get(old).is_none()),
            slots => panic!("invalid snapshot of slot zero: {slots:?}"),
        }
        assert!(snapshot.get(stale).is_none());
        let fork = snapshot.insert(SnapshotValue {
            tag: 33,
            yield_on_drop: false,
        });
        assert_ne!(fork, stale);
        assert!(snapshot.get(stale).is_none());

        let replacement = mutation.join().unwrap();
        if replacement.index() == old.index() {
            REUSED.store(true, StdOrdering::Relaxed);
            assert_eq!(replacement.generation().get(), 3);
        } else {
            // The scan can pin the old value beyond the remover's guard,
            // in which case the insert may need a fresh slot.
            assert_eq!(replacement.index(), 1);
        }
        assert_eq!(hive.get(replacement).unwrap().tag, 22);
        assert!(hive.get(old).is_none());
        assert!(hive.get(stale).is_none());
    });
    assert!(SNAPSHOT_MARKED.load(StdOrdering::Relaxed));
    assert!(CLAIM_BUSY.load(StdOrdering::Relaxed));
    assert!(FINALIZE_BUSY.load(StdOrdering::Relaxed));
    assert!(READ_REPLACEMENT.load(StdOrdering::Relaxed));
    assert!(REUSED.load(StdOrdering::Relaxed));
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

// A guard holder writes the payload immediately before releasing its guard.
// Checking it in the finalizer (or taker), before any join, tests the guard
// release as well as exactly-once destruction. The counter adds no barrier.
struct ReleaseCheckedValue {
    tag: usize,
    drops: Arc<AtomicUsize>,
    payload: Arc<AtomicUsize>,
}

impl Drop for ReleaseCheckedValue {
    fn drop(&mut self) {
        assert_eq!(self.payload.load(Ordering::Relaxed), self.tag);
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn take_and_remove_have_exactly_one_winner() {
    static TAKE_WON: StdAtomicBool = StdAtomicBool::new(false);
    static REMOVE_WON: StdAtomicBool = StdAtomicBool::new(false);
    TAKE_WON.store(false, StdOrdering::Relaxed);
    REMOVE_WON.store(false, StdOrdering::Relaxed);
    model(|| {
        let hive = Arc::new(ConcurrentHive::<ReleaseCheckedValue, 1>::new());
        let drops = Arc::new(AtomicUsize::new(0));
        let payload = Arc::new(AtomicUsize::new(0));
        let id = hive.insert(ReleaseCheckedValue {
            tag: 7,
            drops: drops.clone(),
            payload: payload.clone(),
        });
        let guard = hive.get(id).unwrap();
        let taker = hive.clone();
        let a = thread::spawn(move || {
            let Some(value) = taker.take(id) else {
                return false;
            };
            assert_eq!(value.tag, 7);
            // This read must acquire the guard's release, without a join.
            assert_eq!(value.payload.load(Ordering::Relaxed), 7);
            assert_eq!(value.drops.load(Ordering::Relaxed), 0);
            drop(value);
            true
        });
        let remover = hive.clone();
        let b = thread::spawn(move || remover.remove(id));
        thread::yield_now();
        let marked_while_pinned = !hive.contains(id);
        assert_eq!(guard.id(), id);
        assert_eq!(guard.tag, 7);
        assert_eq!(drops.load(Ordering::Relaxed), 0);
        // All cross-thread handshakes above precede this write.
        payload.store(7, Ordering::Relaxed);
        drop(guard);
        let taken = a.join().unwrap();
        let removed = b.join().unwrap();
        assert_eq!(usize::from(taken) + usize::from(removed), 1);
        if marked_while_pinned {
            if taken {
                TAKE_WON.store(true, StdOrdering::Relaxed);
            } else {
                REMOVE_WON.store(true, StdOrdering::Relaxed);
            }
        }
        assert!(hive.get(id).is_none());
        assert!(hive.is_empty());
        assert_eq!(drops.load(Ordering::Relaxed), 1);
        drop(hive);
        assert_eq!(drops.load(Ordering::Relaxed), 1);
    });
    assert!(TAKE_WON.load(StdOrdering::Relaxed));
    assert!(REMOVE_WON.load(StdOrdering::Relaxed));
}

#[test]
fn marked_slot_waits_for_both_guards() {
    static LAST_GUARD_REACHED: StdAtomicBool = StdAtomicBool::new(false);
    LAST_GUARD_REACHED.store(false, StdOrdering::Relaxed);
    model(|| {
        let hive = Arc::new(ConcurrentHive::<ReleaseCheckedValue, 1>::new());
        let drops = Arc::new(AtomicUsize::new(0));
        let payload = Arc::new(AtomicUsize::new(0));
        let id = hive.insert(ReleaseCheckedValue {
            tag: 7,
            drops: drops.clone(),
            payload: payload.clone(),
        });
        let first = hive.get(id).unwrap();
        let gate = Arc::new(Gate {
            phase: Mutex::new(0),
            changed: Condvar::new(),
        });
        let first_dropped = Arc::new(loom::sync::atomic::AtomicBool::new(false));
        let reader = hive.clone();
        let waiting = gate.clone();
        let released = first_dropped.clone();
        let a = thread::spawn(move || {
            let second = reader.get(id).unwrap();
            {
                let mut phase = waiting.phase.lock().unwrap();
                *phase = 1;
                waiting.changed.notify_all();
                while *phase != 2 {
                    phase = waiting.changed.wait(phase).unwrap();
                }
            }
            assert_eq!(second.id(), id);
            assert_eq!(second.tag, 7);
            assert_eq!(second.drops.load(Ordering::Relaxed), 0);
            if released.load(Ordering::Relaxed) {
                // The first guard is gone, but this guard still pins T.
                assert_eq!(second.drops.load(Ordering::Relaxed), 0);
                LAST_GUARD_REACHED.store(true, StdOrdering::Relaxed);
            }
            drop(second);
        });
        thread::yield_now();
        // Wait until guard two is held before marking the slot.
        {
            let mut phase = gate.phase.lock().unwrap();
            while *phase != 1 {
                phase = gate.changed.wait(phase).unwrap();
            }
            assert!(hive.remove(id)); // both guards are held at the mark
            assert!(hive.get(id).is_none());
            assert_eq!(drops.load(Ordering::Relaxed), 0);
            *phase = 2;
            gate.changed.notify_all();
        }
        assert_eq!(first.tag, 7);
        // The mutex handoff precedes this write, so only a guard release can
        // carry it to a finalizer on the other thread.
        payload.store(7, Ordering::Relaxed);
        drop(first);
        first_dropped.swap(true, Ordering::Relaxed);
        a.join().unwrap();
        assert!(hive.is_empty());
        assert!(hive.get(id).is_none());
        assert_eq!(drops.load(Ordering::Relaxed), 1);
        drop(hive);
        assert_eq!(drops.load(Ordering::Relaxed), 1);
    });
    assert!(LAST_GUARD_REACHED.load(StdOrdering::Relaxed));
}

#[test]
fn last_guard_finalization_races_slot_reuse() {
    static REUSED: StdAtomicBool = StdAtomicBool::new(false);
    static FRESH: StdAtomicBool = StdAtomicBool::new(false);
    REUSED.store(false, StdOrdering::Relaxed);
    FRESH.store(false, StdOrdering::Relaxed);
    model(|| {
        let hive = Arc::new(ConcurrentHive::<ReleaseCheckedValue, 1>::new());
        let old_drops = Arc::new(AtomicUsize::new(0));
        let new_drops = Arc::new(AtomicUsize::new(0));
        let old_payload = Arc::new(AtomicUsize::new(0));
        let old = hive.insert(ReleaseCheckedValue {
            tag: 1,
            drops: old_drops.clone(),
            payload: old_payload.clone(),
        });
        // The last guard is held, marked and dropped on its own thread, so
        // its finalization races the inserter in both orders.
        let finalizer = hive.clone();
        let b = thread::spawn(move || {
            let guard = finalizer.get(old).unwrap();
            assert!(finalizer.remove(old));
            assert!(finalizer.get(old).is_none());
            assert_eq!(guard.tag, 1);
            old_payload.store(1, Ordering::Relaxed);
            drop(guard); // last guard finalizes the marked slot
        });
        let writer = hive.clone();
        let counted = new_drops.clone();
        let finalized = old_drops.clone();
        let a = thread::spawn(move || {
            let new = writer.insert(ReleaseCheckedValue {
                tag: 2,
                drops: counted,
                payload: Arc::new(AtomicUsize::new(2)),
            });
            // The old id never resolves to the replacement.
            if let Some(value) = writer.get(old) {
                assert_eq!(value.tag, 1);
            }
            assert_eq!(writer.get(new).unwrap().tag, 2);
            if new.index() == old.index() {
                assert_ne!(new, old);
                assert_eq!(new.generation().get(), 2);
                // Reuse implies the finalizer's drop happened before the claim.
                assert_eq!(finalized.load(Ordering::Relaxed), 1);
                REUSED.store(true, StdOrdering::Relaxed);
            } else {
                FRESH.store(true, StdOrdering::Relaxed);
            }
            new
        });
        b.join().unwrap();
        let new = a.join().unwrap();
        assert!(hive.get(old).is_none());
        assert_eq!(hive.get(new).unwrap().tag, 2);
        assert_eq!(hive.len(), 1);
        assert_eq!(old_drops.load(Ordering::Relaxed), 1);
        assert_eq!(new_drops.load(Ordering::Relaxed), 0);
        drop(hive);
        assert_eq!(new_drops.load(Ordering::Relaxed), 1);
    });
    assert!(REUSED.load(StdOrdering::Relaxed));
    assert!(FRESH.load(StdOrdering::Relaxed));
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
fn guessed_id_lookup_races_publication_and_removal() {
    static MARKED_WHILE_PINNED: StdAtomicBool = StdAtomicBool::new(false);
    MARKED_WHILE_PINNED.store(false, StdOrdering::Relaxed);
    model(|| {
        let hive = Arc::new(ConcurrentHive::<ReleaseCheckedValue, 1>::new());
        let guessed = Id::new(0, NonZeroU32::MIN);
        let drops = Arc::new(AtomicUsize::new(0));
        let payload = Arc::new(AtomicUsize::new(0));
        let gate = Arc::new(Gate {
            phase: Mutex::new(0),
            changed: Condvar::new(),
        });
        let writer = hive.clone();
        let factory_gate = gate.clone();
        let counted = drops.clone();
        let producer = thread::spawn(move || {
            let id = writer.insert_with(|id| {
                assert_eq!(id, guessed);
                {
                    let mut phase = factory_gate.phase.lock().unwrap();
                    *phase = 1;
                    factory_gate.changed.notify_one();
                    while *phase != 2 {
                        phase = factory_gate.changed.wait(phase).unwrap();
                    }
                }
                // The gate releases before initialization. Successful get
                // must acquire the value through the slot's live state.
                payload.store(7, Ordering::Relaxed);
                ReleaseCheckedValue {
                    tag: 7,
                    drops: counted,
                    payload,
                }
            });
            thread::yield_now();
            assert!(writer.remove(id));
        });

        {
            let mut phase = gate.phase.lock().unwrap();
            while *phase != 1 {
                phase = gate.changed.wait(phase).unwrap();
            }
            assert!(!hive.contains(guessed));
            assert!(hive.get(guessed).is_none());
            *phase = 2;
            gate.changed.notify_one();
        }
        thread::yield_now();
        if let Some(guard) = hive.get(guessed) {
            assert_eq!(guard.id(), guessed);
            assert_eq!(guard.tag, 7);
            assert_eq!(guard.payload.load(Ordering::Relaxed), 7);
            thread::yield_now();
            if !hive.contains(guessed) {
                // Removal invalidated new lookups while this guard is live.
                assert!(hive.get(guessed).is_none());
                MARKED_WHILE_PINNED.store(true, StdOrdering::Relaxed);
            }
            assert_eq!(guard.tag, 7);
            assert_eq!(drops.load(Ordering::Relaxed), 0);
        }

        producer.join().unwrap();
        assert!(!hive.contains(guessed));
        assert!(hive.get(guessed).is_none());
        assert!(hive.is_empty());
        assert_eq!(drops.load(Ordering::Relaxed), 1);
    });
    assert!(MARKED_WHILE_PINNED.load(StdOrdering::Relaxed));
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

/// Interleavings of the recycled-page model where both inserts landed in the
/// recycled page, and where the second went to a fresh page. Both must occur.
static RECYCLED_BOTH: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static RECYCLED_FRESH: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[test]
fn inserts_reuse_a_freed_page_without_reviving_its_ids() {
    model(|| {
        let drops = Arc::new(AtomicUsize::new(0));
        let value = |tag| Value {
            tag,
            drops: drops.clone(),
        };
        let mut hive = ConcurrentHive::<Value, 2>::new();
        let ids: Vec<_> = (0..4).map(|tag| hive.insert(value(tag))).collect();
        assert!(hive.remove(ids[0]));
        assert!(hive.remove(ids[1]));
        hive.shrink_to_fit(); // frees page 0; page 1 stays full
        assert_eq!(drops.load(Ordering::SeqCst), 2);
        let hive = Arc::new(hive);
        let writer = hive.clone();
        let stale = ids[0];
        let made = value(10);
        let a = thread::spawn(move || {
            let id = writer.insert(made);
            assert_eq!(writer.get(id).unwrap().tag, 10);
            assert!(writer.get(stale).is_none());
            id
        });
        thread::yield_now();
        let b = hive.insert(value(20));
        assert_eq!(hive.get(b).unwrap().tag, 20);
        assert!(hive.get(ids[1]).is_none());
        let a = a.join().unwrap();
        assert_ne!(a, b);
        assert_eq!(hive.get(a).unwrap().tag, 10);
        for id in [a, b] {
            if id.index() < 2 {
                assert_eq!(
                    id.generation().get(),
                    2,
                    "recycled slots restart above their floor"
                );
            } else {
                assert_eq!(id.index(), 4);
            }
        }
        assert!(
            a.index() < 2 || b.index() < 2,
            "the freed page is reused first"
        );
        if a.index() < 2 && b.index() < 2 {
            RECYCLED_BOTH.store(true, std::sync::atomic::Ordering::Relaxed);
        } else {
            RECYCLED_FRESH.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        assert_eq!(hive.get(ids[2]).unwrap().tag, 2);
        assert_eq!(hive.len(), 4);
    });
    assert!(RECYCLED_BOTH.load(std::sync::atomic::Ordering::Relaxed));
    assert!(RECYCLED_FRESH.load(std::sync::atomic::Ordering::Relaxed));
}

#[test]
fn recycled_pages_remain_distinct_while_growing_into_the_next_chunk() {
    static RECYCLED_READ: StdAtomicBool = StdAtomicBool::new(false);
    static NEXT_CHUNK_READ: StdAtomicBool = StdAtomicBool::new(false);
    RECYCLED_READ.store(false, StdOrdering::Relaxed);
    NEXT_CHUNK_READ.store(false, StdOrdering::Relaxed);
    model(|| {
        // Loom's Spine BASE is 2: pages 0..2 belong to chunk 0, and page 2
        // starts chunk 1. PAGE=1 keeps two recycled pages plus growth tiny.
        let mut hive = ConcurrentHive::<Value, 1>::new();
        let drops = Arc::new(AtomicUsize::new(0));
        let stale: Vec<_> = (0..2)
            .map(|tag| {
                hive.insert(Value {
                    tag,
                    drops: drops.clone(),
                })
            })
            .collect();
        for &id in &stale {
            assert!(hive.remove(id));
        }
        hive.shrink_to_fit(); // both chunk-0 pages now await PageStack pops
        assert_eq!(hive.index_bound(), 2);
        assert_eq!(drops.load(Ordering::SeqCst), 2);
        let hive = Arc::new(hive);
        let payload = Arc::new(AtomicUsize::new(0));
        let writer = hive.clone();
        let written = payload.clone();
        let counted = drops.clone();
        let a = thread::spawn(move || {
            writer.insert_with(|_| {
                // The page and chunk exist before this write, so the reader
                // below must acquire the slot's publication, even in chunk 1.
                written.store(42, Ordering::Relaxed);
                Value {
                    tag: 10,
                    drops: counted,
                }
            })
        });
        thread::yield_now();
        let b = hive.insert(Value {
            tag: 20,
            drops: drops.clone(),
        });
        let c = hive.insert(Value {
            tag: 30,
            drops: drops.clone(),
        });
        // Three claims consume the two recycled pages and reserve page 2.
        // Read before joining so the producer's payload has no extra barrier.
        for _ in 0..2 {
            let mut saw_writer = false;
            for (id, value) in hive.iter() {
                if value.tag == 10 {
                    assert_eq!(payload.load(Ordering::Relaxed), 42);
                    if id.index() < 2 {
                        RECYCLED_READ.store(true, StdOrdering::Relaxed);
                    } else {
                        assert_eq!(id.index(), 2);
                        NEXT_CHUNK_READ.store(true, StdOrdering::Relaxed);
                    }
                    saw_writer = true;
                }
            }
            for &id in &stale {
                assert!(hive.get(id).is_none());
            }
            if saw_writer {
                break;
            }
            thread::yield_now();
        }
        let a = a.join().unwrap();
        let mut indices = [a.index(), b.index(), c.index()];
        indices.sort_unstable();
        assert_eq!(indices, [0, 1, 2]);
        for (id, tag) in [(a, 10), (b, 20), (c, 30)] {
            assert_eq!(hive.get(id).unwrap().tag, tag);
            assert_eq!(id.generation().get(), if id.index() < 2 { 2 } else { 1 });
        }
        for &id in &stale {
            assert!(hive.get(id).is_none());
        }
        assert_eq!(hive.index_bound(), 3);
        assert_eq!(hive.len(), 3);
        assert_eq!(drops.load(Ordering::SeqCst), 2);
        drop(hive);
        assert_eq!(drops.load(Ordering::SeqCst), 5);
    });
    assert!(RECYCLED_READ.load(StdOrdering::Relaxed));
    assert!(NEXT_CHUNK_READ.load(StdOrdering::Relaxed));
}
