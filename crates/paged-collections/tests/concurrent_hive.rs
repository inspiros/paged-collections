//! Guard ownership, counted destruction, exclusive recovery, conversions and a plain oracle.

#![cfg(all(feature = "concurrent", not(loom)))]
#![cfg(all(
    any(target_pointer_width = "32", target_pointer_width = "64"),
    target_has_atomic = "ptr",
    target_has_atomic = "64"
))]

use std::cell::Cell;
use std::collections::BTreeSet;
use std::num::NonZeroU32;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;

use paged_collections::concurrent::{ConcurrentHive, Ref};
use paged_collections::hive::{Hive, RawPage, RawSlot};
use paged_collections::id_alloc::{Id, MAX_GENERATION};
use proptest::prelude::*;

// These bodies are type-checked without running. Multiple applicable marker
// impls make `_` ambiguous if any of the forbidden auto traits appears.
const _: fn() = || {
    fn send<T: Send>() {}
    fn sync<T: Sync>() {}
    send::<ConcurrentHive<Cell<u8>, 2>>();
    send::<ConcurrentHive<u8, 2>>();
    sync::<ConcurrentHive<u8, 2>>();
    send::<Ref<'static, u8, 2>>();
    sync::<Ref<'static, u8, 2>>();

    trait AmbiguousSend<A> {
        fn check() {}
    }
    impl<T: ?Sized> AmbiguousSend<()> for T {}
    impl<T: ?Sized + Send> AmbiguousSend<u8> for T {}
    let _ = <ConcurrentHive<Rc<u8>, 2> as AmbiguousSend<_>>::check;
    let _ = <Ref<'static, Cell<u8>, 2> as AmbiguousSend<_>>::check;
    let _ = <Ref<'static, Rc<u8>, 2> as AmbiguousSend<_>>::check;
    let _ = <ConcurrentHive<std::sync::MutexGuard<'static, ()>, 2> as AmbiguousSend<_>>::check;
    let _ = <Ref<'static, std::sync::MutexGuard<'static, ()>, 2> as AmbiguousSend<_>>::check;

    trait AmbiguousSync<A> {
        fn check() {}
    }
    impl<T: ?Sized> AmbiguousSync<()> for T {}
    impl<T: ?Sized + Sync> AmbiguousSync<u8> for T {}
    let _ = <ConcurrentHive<Cell<u8>, 2> as AmbiguousSync<_>>::check;
    let _ = <ConcurrentHive<Rc<u8>, 2> as AmbiguousSync<_>>::check;
    let _ = <Ref<'static, Cell<u8>, 2> as AmbiguousSync<_>>::check;
    let _ = <Ref<'static, Rc<u8>, 2> as AmbiguousSync<_>>::check;
    let _ = <ConcurrentHive<std::sync::MutexGuard<'static, ()>, 2> as AmbiguousSync<_>>::check;
    let _ = <Ref<'static, std::sync::MutexGuard<'static, ()>, 2> as AmbiguousSync<_>>::check;

    trait AmbiguousClone<A> {
        fn check() {}
    }
    impl<T: ?Sized> AmbiguousClone<()> for T {}
    impl<T: Clone> AmbiguousClone<u8> for T {}
    let _ = <Ref<'static, u8, 2> as AmbiguousClone<_>>::check;
};

fn random(state: &mut u64) -> usize {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state as usize
}

struct Payload {
    id: Id,
    drops: Arc<AtomicUsize>,
}

impl Drop for Payload {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn borrowed_for_loops_yield_ids_and_values() {
    let mut hive = ConcurrentHive::<_, 2>::new();
    let first = hive.insert(1);
    let second = hive.insert(2);
    for (_, value) in &mut hive {
        *value *= 10;
    }
    let mut entries = Vec::new();
    for (id, value) in &hive {
        entries.push((id, *value));
    }
    assert_eq!(entries, [(first, 10), (second, 20)]);
}

#[test]
fn eight_threads_mix_inserts_guards_removes_takes_and_iteration() {
    const THREADS: usize = 8;
    let rounds = if cfg!(miri) { 12 } else { 10_000 };
    let hive = Arc::new(ConcurrentHive::<Payload, 2>::new());
    let made = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let recent = Arc::new((0..THREADS).map(|_| AtomicU64::new(0)).collect::<Vec<_>>());
    let mut workers = Vec::new();
    for worker in 0..THREADS {
        let hive = hive.clone();
        let made = made.clone();
        let drops = drops.clone();
        let recent = recent.clone();
        workers.push(thread::spawn(move || {
            let mut owned = Vec::new();
            let mut seed = worker as u64 + 1;
            for _ in 0..rounds {
                let choice = random(&mut seed);
                if owned.len() >= 16 {
                    let id = owned.swap_remove(choice % owned.len());
                    if choice & 1 == 0 {
                        hive.remove(id);
                    } else if let Some(value) = hive.take(id) {
                        assert_eq!(value.id, id);
                    }
                    continue;
                }
                let bits = recent[choice % THREADS].load(Ordering::Acquire);
                let candidate = Id::from_bits(bits);
                match choice % 6 {
                    0 | 1 => {
                        let id = hive.insert_with(|id| {
                            made.fetch_add(1, Ordering::SeqCst);
                            Payload {
                                id,
                                drops: drops.clone(),
                            }
                        });
                        recent[worker].store(id.to_bits(), Ordering::Release);
                        owned.push(id);
                    }
                    2 => {
                        if let Some(id) = candidate {
                            if let Some(value) = hive.get(id) {
                                assert_eq!(value.id(), id);
                                assert_eq!(value.id, id);
                                thread::yield_now();
                                assert_eq!(value.id, id);
                            }
                        }
                    }
                    3 => {
                        if let Some(id) = candidate {
                            hive.remove(id);
                        }
                    }
                    4 => {
                        if let Some(id) = candidate {
                            if let Some(value) = hive.take(id) {
                                assert_eq!(value.id, id);
                            }
                        }
                    }
                    _ => {
                        for (id, value) in hive.iter().take(8) {
                            assert_eq!(value.id(), id);
                            assert_eq!(value.id, id);
                        }
                    }
                }
            }
        }));
    }
    for worker in workers {
        worker.join().unwrap();
    }
    let survivors: BTreeSet<_> = hive
        .iter()
        .map(|(id, value)| {
            assert_eq!(value.id, id);
            id
        })
        .collect();
    assert_eq!(hive.len(), survivors.len());
    assert_eq!(
        made.load(Ordering::SeqCst) - drops.load(Ordering::SeqCst),
        survivors.len()
    );
    drop(hive);
    assert_eq!(made.load(Ordering::SeqCst), drops.load(Ordering::SeqCst));
}

fn id(index: u32, generation: u32) -> Id {
    Id::new(index, NonZeroU32::new(generation).unwrap())
}

#[test]
fn conversion_round_trip_preserves_removed_reused_and_shrunken_history() {
    let mut plain = Hive::<u32, 4>::with_capacity(32);
    let original: Vec<_> = (0..12).map(|value| plain.insert(value)).collect();
    let mut dead = Vec::new();
    for &old in &original[..8] {
        assert!(plain.remove(old).is_some());
        dead.push(old);
    }
    for value in 20..25 {
        let reused = plain.insert(value);
        if value % 2 == 0 {
            assert!(plain.remove(reused).is_some());
            dead.push(reused);
        }
    }
    plain.shrink_to_fit();
    assert!(
        plain
            .raw_pages()
            .any(|raw| raw.floor != 0 && raw.slots.is_empty())
    );
    plain.reserve(40);
    let mut live: Vec<_> = plain.iter().map(|(id, value)| (id, *value)).collect();
    let concurrent = ConcurrentHive::from(plain);
    assert_eq!(concurrent.len(), live.len());
    assert_eq!(
        concurrent
            .iter()
            .map(|(id, v)| (id, *v))
            .collect::<Vec<_>>(),
        live
    );
    for &old in &dead {
        assert!(concurrent.get(old).is_none());
    }
    // Check stale ids while filling reusable vacancies in shared storage.
    for value in 60..90 {
        let new = concurrent.insert(value);
        assert!(!dead.contains(&new));
        live.push((new, value));
        for &old in &dead {
            assert!(concurrent.get(old).is_none());
        }
    }
    // Also carry a removal performed in the concurrent hive back to Hive.
    let removed = concurrent.insert(99);
    assert!(concurrent.remove(removed));
    dead.push(removed);
    assert_eq!(concurrent.len(), live.len());
    let mut back = concurrent.into_hive();
    assert_eq!(back.len(), live.len());
    assert_eq!(back.iter().count(), live.len());
    for &old in &dead {
        assert!(back.get(old).is_none());
    }
    for (id, value) in live {
        assert_eq!(back.get(id), Some(&value));
    }
    for value in 100..160 {
        let new = back.insert(value);
        assert!(!dead.contains(&new));
        for &old in &dead {
            assert!(back.get(old).is_none());
        }
    }
}

fn floor_fixture() -> Hive<u32, 4> {
    Hive::try_from_raw_pages(vec![
        RawPage {
            floor: 0,
            slots: vec![
                RawSlot::Vacant { version: 0 },
                RawSlot::Vacant { version: 4 },
                RawSlot::Occupied {
                    version: 3,
                    value: 11,
                },
            ],
        },
        RawPage {
            floor: 8,
            slots: vec![],
        },
        RawPage {
            floor: 0,
            slots: vec![RawSlot::Occupied {
                version: 1,
                value: 22,
            }],
        },
        RawPage {
            floor: u32::MAX - 1,
            slots: vec![],
        },
        RawPage {
            floor: 0,
            slots: vec![],
        },
    ])
    .unwrap()
}

#[test]
fn conversion_keeps_retirement_and_materializes_nonzero_page_floors() {
    let mut dead = vec![
        id(0, 1),
        id(0, MAX_GENERATION),
        id(1, 1),
        id(1, 2),
        id(2, 1),
    ];
    for index in 4..8 {
        for generation in 1..=4 {
            dead.push(id(index, generation));
        }
    }
    for index in 12..16 {
        dead.push(id(index, 1));
        dead.push(id(index, MAX_GENERATION - 1));
    }
    // Exercise both immediate conversion back and inserts before conversion.
    for insert_first in [false, true] {
        let concurrent = ConcurrentHive::from(floor_fixture());
        assert_eq!(concurrent.index_bound(), 16);
        assert_eq!(concurrent.len(), 2);
        assert_eq!(concurrent.get(id(2, 2)).map(|v| *v), Some(11));
        assert_eq!(concurrent.get(id(8, 1)).map(|v| *v), Some(22));
        for &old in &dead {
            assert!(!concurrent.contains(old));
        }
        if insert_first {
            let mut issued = BTreeSet::new();
            for value in 100..120 {
                let new = concurrent.insert(value);
                assert!(issued.insert(new));
                assert!(!dead.contains(&new));
                assert_ne!(new.index(), 0);
                if (4..8).contains(&new.index()) {
                    assert_eq!(new.generation().get(), 5);
                }
                if (12..16).contains(&new.index()) {
                    assert_eq!(new.generation().get(), MAX_GENERATION);
                }
            }
            // Interior floor-zero holes spend never-issued generation 1 so
            // that a popped zero state remains unambiguously retired.
            for index in [3, 9, 10, 11] {
                assert!(issued.contains(&id(index, 2)));
            }
            assert!((4..8).all(|index| issued.contains(&id(index, 5))));
            assert!((12..16).all(|index| issued.contains(&id(index, MAX_GENERATION))));
        }
        let mut back = concurrent.into_hive();
        assert_eq!(back.len(), if insert_first { 22 } else { 2 });
        assert_eq!(back.get(id(2, 2)), Some(&11));
        assert_eq!(back.get(id(8, 1)), Some(&22));
        for value in 200..224 {
            let new = back.insert(value);
            assert!(!dead.contains(&new));
            assert_ne!(new.index(), 0);
            for &old in &dead {
                assert!(!back.contains(old));
            }
        }
        // Retire all final-generation slots, then preserve their zero states.
        back.clear();
        let concurrent = ConcurrentHive::from(back);
        for &old in &dead {
            assert!(!concurrent.contains(old));
        }
        for _ in 0..32 {
            let new = concurrent.insert(0);
            assert_ne!(new.index(), 0);
            assert!(!(12..16).contains(&new.index()));
        }
    }
}

#[test]
fn conversion_leaves_only_a_never_used_suffix_to_fresh_reservation() {
    let plain = Hive::<u32, 4>::with_capacity(32);
    let concurrent = ConcurrentHive::from(plain);
    assert_eq!(concurrent.index_bound(), 0);
    assert_eq!(concurrent.insert(1), id(0, 1));

    let mut plain = Hive::<u32, 4>::with_capacity(32);
    let first = plain.insert(1);
    let concurrent = ConcurrentHive::from(plain);
    assert_eq!(concurrent.index_bound(), 1);
    assert_eq!(concurrent.get(first).map(|v| *v), Some(1));
    assert_eq!(concurrent.insert(2), id(1, 1));
    assert_eq!(concurrent.into_hive().get(first), Some(&1));
}

#[test]
fn conversion_empty_hives_preserve_empty_state_and_stale_history() {
    let plain = ConcurrentHive::<u32, 4>::new().into_hive();
    assert!(plain.is_empty());
    assert_eq!(plain.len(), 0);
    let concurrent = ConcurrentHive::from(plain);
    assert!(concurrent.is_empty());
    assert_eq!(concurrent.len(), 0);
    assert_eq!(concurrent.index_bound(), 0);

    for shrink in [false, true] {
        let mut plain = Hive::<u32, 4>::new();
        let old = plain.insert(1);
        assert_eq!(plain.remove(old), Some(1));
        if shrink {
            plain.shrink_to_fit();
        }
        let concurrent = ConcurrentHive::from(plain);
        assert!(concurrent.is_empty());
        assert_eq!(concurrent.len(), 0);
        assert!(concurrent.get(old).is_none());
        let mut back = concurrent.into_hive();
        assert!(back.is_empty());
        assert_eq!(back.len(), 0);
        assert!(back.get(old).is_none());
        for value in 0..16 {
            assert_ne!(back.insert(value), old);
            assert!(back.get(old).is_none());
        }
    }
}

#[derive(Debug)]
struct Counted {
    value: i32,
    drops: Arc<AtomicUsize>,
}

impl Drop for Counted {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

fn counted(value: i32, drops: &Arc<AtomicUsize>) -> Counted {
    Counted {
        value,
        drops: drops.clone(),
    }
}

#[test]
fn conversion_finalizes_marked_values_and_releases_forgotten_guards() {
    for keep_live in [false, true] {
        let marked_drops = Arc::new(AtomicUsize::new(0));
        let live_drops = Arc::new(AtomicUsize::new(0));
        let mut plain = Hive::<Counted, 2>::new();
        let marked = plain.insert(counted(1, &marked_drops));
        let live = keep_live.then(|| plain.insert(counted(2, &live_drops)));
        let hive = ConcurrentHive::from(plain);
        assert_eq!(hive.len(), 1 + usize::from(keep_live));
        assert_eq!(marked_drops.load(Ordering::SeqCst), 0);
        assert_eq!(live_drops.load(Ordering::SeqCst), 0);
        std::mem::forget(hive.get(marked).unwrap());
        if let Some(live) = live {
            std::mem::forget(hive.get(live).unwrap());
        }
        assert!(hive.remove(marked));
        assert!(hive.get(marked).is_none());
        assert_eq!(hive.len(), usize::from(keep_live));
        assert_eq!(marked_drops.load(Ordering::SeqCst), 0);

        let plain = hive.into_hive();
        assert!(!plain.contains(marked));
        assert_eq!(plain.len(), usize::from(keep_live));
        assert_eq!(plain.iter().count(), plain.len());
        assert_eq!(marked_drops.load(Ordering::SeqCst), 1);
        assert_eq!(live_drops.load(Ordering::SeqCst), 0);
        if let Some(live) = live {
            assert_eq!(plain.get(live).unwrap().value, 2);
        }
        let hive = ConcurrentHive::from(plain);
        assert_eq!(hive.len(), usize::from(keep_live));
        assert!(!hive.contains(marked));
        if let Some(live) = live {
            assert_eq!(hive.get(live).unwrap().value, 2);
        }
        drop(hive);
        assert_eq!(marked_drops.load(Ordering::SeqCst), 1);
        assert_eq!(live_drops.load(Ordering::SeqCst), usize::from(keep_live));
    }
}

#[test]
fn removal_defers_destruction_and_reuse_until_all_guards_drop() {
    let hive = ConcurrentHive::<Counted, 2>::new();
    let drops = Arc::new(AtomicUsize::new(0));
    let old = hive.insert(counted(10, &drops));
    let a = hive.get(old).unwrap();
    let b = hive.get(old).unwrap();
    assert_eq!(a.id(), old);
    assert!(format!("{a:?}").contains("Counted"));
    assert!(hive.remove(old));
    assert!(!hive.remove(old));
    assert!(hive.take(old).is_none());
    assert!(hive.get(old).is_none());
    assert!(!hive.contains(old));
    assert_eq!(hive.len(), 0);
    assert_eq!(a.value, 10);
    drop(a);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    let other = hive.insert(counted(20, &drops));
    assert_ne!(other.index(), old.index());
    assert_eq!(b.value, 10);
    drop(b);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    let new = hive.insert(counted(30, &drops));
    assert_eq!(new.index(), old.index());
    assert_ne!(new, old);
    assert_eq!(hive.get(new).unwrap().value, 30);
    drop(hive);
    assert_eq!(drops.load(Ordering::SeqCst), 3);
}

#[test]
fn take_waits_on_another_threads_guard_and_transfers_ownership() {
    let hive = Arc::new(ConcurrentHive::<Counted, 2>::new());
    let drops = Arc::new(AtomicUsize::new(0));
    let id = hive.insert(counted(42, &drops));
    let guard = hive.get(id).unwrap();
    let (sent, received) = mpsc::channel();
    let taker = hive.clone();
    let worker = thread::spawn(move || sent.send(taker.take(id).unwrap()).unwrap());
    while hive.contains(id) {
        thread::yield_now();
    }
    assert!(hive.get(id).is_none());
    assert_eq!(guard.value, 42);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert!(matches!(
        received.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));
    drop(guard);
    let value = received.recv().unwrap();
    worker.join().unwrap();
    assert_eq!(hive.len(), 0);
    assert_eq!(value.value, 42);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(value);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    drop(hive);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn exclusive_methods_reset_forgotten_guards_and_finalize_marked_values() {
    for operation in 0..6 {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut hive = ConcurrentHive::<Counted, 2>::new();
        let marked = hive.insert(counted(1, &drops));
        let live = hive.insert(counted(2, &drops));
        std::mem::forget(hive.get(marked).unwrap());
        std::mem::forget(hive.get(live).unwrap());
        assert!(hive.remove(marked));
        assert_eq!(hive.len(), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        match operation {
            0 => {
                assert!(hive.get_mut(marked).is_none());
                assert_eq!(drops.load(Ordering::SeqCst), 1);
                hive.get_mut(live).unwrap().value = 3;
                assert_eq!(hive.get(live).unwrap().value, 3);
            }
            1 => {
                assert!(hive.take_mut(marked).is_none());
                let value = hive.take_mut(live).unwrap();
                assert_eq!(value.value, 2);
                assert!(hive.is_empty());
                drop(value);
            }
            2 => {
                let values: Vec<_> = hive.iter_mut().collect();
                assert_eq!(values.len(), 1);
                assert_eq!(values[0].0, live);
                values.into_iter().next().unwrap().1.value += 10;
                assert_eq!(hive.get(live).unwrap().value, 12);
            }
            3 => {
                hive.clear();
                assert!(hive.is_empty());
                assert_eq!(drops.load(Ordering::SeqCst), 2);
            }
            4 => {
                let mut drain = hive.drain();
                assert_eq!(drain.len(), 1);
                let (id, value) = drain.next_back().unwrap();
                assert_eq!(id, live);
                assert_eq!(value.value, 2);
                assert_eq!(drops.load(Ordering::SeqCst), 1);
                assert_eq!(drain.next().map(|(id, _)| id), None);
                drop(drain);
                assert!(hive.is_empty());
                drop(value);
            }
            _ => {
                hive.retain(|id, value| {
                    assert_eq!(id, live);
                    value.value += 1;
                    true
                });
                assert_eq!(hive.len(), 1);
                assert_eq!(hive.get(live).unwrap().value, 3);
            }
        }
        assert!(hive.get(marked).is_none());
        drop(hive);
        assert_eq!(drops.load(Ordering::SeqCst), 2);
    }
}

#[test]
fn destruction_recovers_both_live_and_marked_forgotten_guards() {
    let drops = Arc::new(AtomicUsize::new(0));
    let hive = ConcurrentHive::<Counted, 2>::new();
    let marked = hive.insert(counted(1, &drops));
    let live = hive.insert(counted(2, &drops));
    std::mem::forget(hive.get(marked).unwrap());
    std::mem::forget(hive.get(live).unwrap());
    assert!(hive.remove(marked));
    drop(hive);
    assert_eq!(drops.load(Ordering::SeqCst), 2);
}

#[test]
fn mutable_iteration_keeps_multiple_disjoint_values_alive() {
    let mut hive = ConcurrentHive::<i32, 2>::new();
    let ids: Vec<_> = (0..5).map(|v| hive.insert(v)).collect();
    assert!(hive.remove(ids[2]));
    let mut iter = hive.iter_mut();
    assert_eq!(iter.len(), 4);
    let (_, a) = iter.next().unwrap();
    let (_, b) = iter.next().unwrap();
    *a += 10;
    *b += 20;
    for (_, value) in iter {
        *value += 30;
    }
    assert_eq!((*a, *b), (10, 21));
    let before: Vec<_> = hive.iter().map(|(id, v)| (id, *v)).collect();
    let drained: Vec<_> = hive.drain().collect();
    assert_eq!(drained, before);
    assert!(hive.is_empty());
    for id in ids {
        assert!(!hive.contains(id));
    }
    let again = hive.insert(9);
    assert!(again.generation().get() > 1);
}

#[test]
fn factory_panics_return_fresh_and_reused_claims() {
    for reused in [false, true] {
        let hive = ConcurrentHive::<Id, 2>::new();
        if reused {
            let id = hive.insert_with(|id| id);
            assert!(hive.remove(id));
        }
        let attempted = Cell::new(None);
        let result = catch_unwind(AssertUnwindSafe(|| {
            hive.insert_with(|id| {
                attempted.set(Some(id));
                panic!("factory failed");
            });
        }));
        assert!(result.is_err());
        assert!(hive.is_empty());
        let attempted = attempted.get().unwrap();
        assert!(!hive.contains(attempted));
        let next = hive.try_insert_with(|id| id).unwrap();
        assert_eq!(next.index(), attempted.index());
        // The id shown to the panicking factory is spent: it never becomes
        // live, and the slot's next value gets the following generation.
        assert_eq!(next.generation().get(), attempted.generation().get() + 1);
        assert!(!hive.contains(attempted));
        assert!(hive.get(attempted).is_none());
        assert_eq!(*hive.get(next).unwrap(), next);
        assert_eq!(hive.len(), 1);
    }
}

struct Bomb {
    panic: bool,
    drops: Arc<AtomicUsize>,
}

impl Drop for Bomb {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
        assert!(!self.panic, "value destructor failed");
    }
}

#[test]
fn destructor_panics_release_busy_and_leave_len_exact() {
    for guarded in [false, true] {
        let hive = ConcurrentHive::<Bomb, 2>::new();
        let drops = Arc::new(AtomicUsize::new(0));
        let id = hive.insert(Bomb {
            panic: true,
            drops: drops.clone(),
        });
        let result = if guarded {
            let value = hive.get(id).unwrap();
            assert!(hive.remove(id));
            catch_unwind(AssertUnwindSafe(|| drop(value)))
        } else {
            catch_unwind(AssertUnwindSafe(|| {
                hive.remove(id);
            }))
        };
        assert!(result.is_err());
        assert!(hive.is_empty());
        assert!(!hive.contains(id));
        let next = hive.insert(Bomb {
            panic: false,
            drops: drops.clone(),
        });
        assert_eq!(next.index(), id.index());
        assert_ne!(next, id);
        assert_eq!(hive.len(), 1);
        drop(hive);
        assert_eq!(drops.load(Ordering::SeqCst), 2);
    }
}

#[test]
fn predicate_panic_preserves_prior_changes_and_live_values() {
    let mut hive = ConcurrentHive::<i32, 2>::new();
    let ids: Vec<_> = (0..4).map(|v| hive.insert(v)).collect();
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            hive.retain(|_, value| {
                assert_ne!(*value, 2, "predicate failed");
                *value == 1
            });
        }))
        .is_err()
    );
    assert_eq!(hive.len(), 3);
    assert!(!hive.contains(ids[0]));
    for id in ids.into_iter().skip(1) {
        assert!(hive.contains(id));
    }
}

#[test]
fn lookup_and_iteration_edges() {
    let mut hive = ConcurrentHive::<i32, 2>::default();
    assert_eq!(hive.index_bound(), 0);
    assert!(hive.is_empty());
    assert!(format!("{hive:?}").contains("ConcurrentHive"));
    let id = hive.try_insert(7).unwrap();
    let address = {
        let guard = hive.get(id).unwrap();
        &*guard as *const i32
    };
    for value in 0..5 {
        hive.insert(value);
    }
    assert_eq!(address, &*hive.get(id).unwrap() as *const i32);
    let mut iter = hive.iter();
    assert_eq!(iter.size_hint(), (0, Some(6)));
    assert_eq!(iter.by_ref().count(), 6);
    assert!(iter.next().is_none());
    assert!(iter.next().is_none());
    for invalid in [
        Id::new(u32::MAX, NonZeroU32::MIN),
        Id::new(0, NonZeroU32::new(MAX_GENERATION + 1).unwrap()),
    ] {
        assert!(!hive.contains(invalid));
        assert!(hive.get(invalid).is_none());
        assert!(!hive.remove(invalid));
        assert!(hive.take(invalid).is_none());
        assert!(hive.get_mut(invalid).is_none());
        assert!(hive.take_mut(invalid).is_none());
    }
    hive.clear();
    assert!(hive.is_empty());
    assert!(hive.iter_mut().next().is_none());
    assert!(hive.drain().next().is_none());
}

#[test]
#[should_panic(expected = "invalid page capacity")]
fn zero_page_is_rejected() {
    let _ = ConcurrentHive::<u8, 0>::new();
}

fn config() -> ProptestConfig {
    ProptestConfig {
        cases: if cfg!(miri) { 4 } else { 128 },
        ..ProptestConfig::default()
    }
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn single_thread_matches_plain_hive_semantics(
        actions in prop::collection::vec(
            (0u8..10, any::<usize>(), any::<i16>()),
            0..if cfg!(miri) { 20 } else { 200 },
        )
    ) {
        let mut plain = Hive::<i32, 2>::new();
        let mut concurrent = ConcurrentHive::<i32, 2>::new();
        // Free-list/page reuse orders differ. Pair issued handles and compare
        // every history through its corresponding handle, including stale ids.
        let mut issued = Vec::new();
        let mut unique = BTreeSet::new();
        for (action, choice, value) in actions {
            let selected = if issued.is_empty() {
                None
            } else {
                Some(issued[choice % issued.len()])
            };
            match action {
                0 | 1 => {
                    let a = plain.insert(i32::from(value));
                    let b = concurrent.insert_with(|_| i32::from(value));
                    prop_assert!(unique.insert(b));
                    issued.push((a, b));
                }
                2 => {
                    if let Some((a, b)) = selected {
                        prop_assert_eq!(concurrent.remove(b), plain.remove(a).is_some());
                    }
                }
                3 => {
                    if let Some((a, b)) = selected {
                        prop_assert_eq!(concurrent.take(b), plain.remove(a));
                    }
                }
                4 => {
                    if let Some((a, b)) = selected {
                        if let Some(value) = plain.get_mut(a) { *value += 1; }
                        if let Some(value) = concurrent.get_mut(b) { *value += 1; }
                    }
                }
                5 => {
                    plain.retain(|_, v| { *v += 1; *v % 3 != 0 });
                    concurrent.retain(|_, v| { *v += 1; *v % 3 != 0 });
                }
                6 => {
                    plain.clear();
                    concurrent.clear();
                }
                7 => {
                    let mut a: Vec<_> = plain.drain().map(|(_, v)| v).collect();
                    let mut b: Vec<_> = concurrent.drain().map(|(_, v)| v).collect();
                    a.sort_unstable();
                    b.sort_unstable();
                    prop_assert_eq!(a, b);
                }
                8 => {
                    for (_, v) in plain.iter_mut() { *v += 1; }
                    for (_, v) in concurrent.iter_mut() { *v += 1; }
                }
                _ => {
                    if let Some((a, b)) = selected {
                        prop_assert_eq!(concurrent.take_mut(b), plain.remove(a));
                    }
                }
            }
            prop_assert_eq!(concurrent.len(), plain.len());
            prop_assert_eq!(concurrent.is_empty(), plain.is_empty());
            prop_assert_eq!(concurrent.iter().count(), plain.len());
            for &(a, b) in &issued {
                prop_assert_eq!(concurrent.contains(b), plain.contains(a));
                prop_assert_eq!(concurrent.get(b).map(|v| *v), plain.get(a).copied());
            }
        }
    }
}

/// Random inserts, removals, shrinks and Hive round trips against an
/// oracle: live ids keep their values, no issued id is ever issued again,
/// and dead ids stay dead.
fn shrink_model(ops: Vec<(u8, u8)>) {
    let mut hive = ConcurrentHive::<u32, 4>::new();
    let mut live: std::collections::BTreeMap<Id, u32> = std::collections::BTreeMap::new();
    let mut issued: BTreeSet<Id> = BTreeSet::new();
    let mut next = 0;
    for (op, pick) in ops {
        match op % 8 {
            0..=3 => {
                let id = hive.insert(next);
                assert!(issued.insert(id), "{id:?} issued twice");
                live.insert(id, next);
                next += 1;
            }
            4 | 5 => {
                if let Some(&id) = live.keys().nth(usize::from(pick) % live.len().max(1)) {
                    assert!(hive.remove(id));
                    live.remove(&id);
                }
            }
            6 => hive.shrink_to_fit(),
            _ => hive = ConcurrentHive::from(hive.into_hive()),
        }
        assert_eq!(hive.len(), live.len());
        for id in &issued {
            assert_eq!(hive.get(*id).map(|v| *v), live.get(id).copied(), "{id:?}");
        }
        assert_eq!(hive.iter().count(), live.len());
    }
    hive.shrink_to_fit();
    let mut plain = hive.into_hive();
    for id in &issued {
        assert_eq!(plain.get(*id).copied(), live.get(id).copied());
    }
    for v in 0..20 {
        assert!(!issued.contains(&plain.insert(v)));
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: if cfg!(miri) { 4 } else { 256 }, ..ProptestConfig::default() })]
    #[test]
    fn shrink_to_fit_frees_reuses_and_never_revives(ops in prop::collection::vec((any::<u8>(), any::<u8>()), 0..120)) {
        shrink_model(ops);
    }
}

#[test]
fn shrink_to_fit_frees_empty_pages_below_the_bound_only() {
    let mut hive = ConcurrentHive::<u32, 4>::new();
    let ids: Vec<_> = (0..10).map(|v| hive.insert(v)).collect();
    for &id in &ids {
        assert!(hive.remove(id));
    }
    // Pages 0 and 1 are empty and below the bound (10); page 2 holds it.
    hive.shrink_to_fit();
    hive.shrink_to_fit(); // idempotent: freed pages stay queued once
    let again: Vec<_> = (0..10).map(|v| hive.insert(v + 100)).collect();
    let indices: BTreeSet<u32> = again.iter().map(|id| id.index()).collect();
    // Page 2's two free slots (8, 9) first, then freed pages 0 and 1.
    assert_eq!(indices, (0..10).collect());
    assert_eq!(again[0].index(), 8);
    assert_eq!(again[2].index(), 0);
    for (id, old) in again.iter().zip(&ids) {
        assert!(!ids.contains(id), "{id:?} revived ({old:?})");
    }
    assert_eq!(hive.index_bound(), 10);
}

#[test]
fn shrink_to_fit_finalizes_forgotten_guards_once() {
    let drops = Arc::new(AtomicUsize::new(0));
    let mut hive = ConcurrentHive::<Counted, 2>::new();
    let ids: Vec<_> = (0..4).map(|v| hive.insert(counted(v, &drops))).collect();
    std::mem::forget(hive.get(ids[0]).unwrap());
    assert!(hive.remove(ids[0])); // marked: the forgotten guard pins it
    assert!(hive.remove(ids[1]));
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    hive.shrink_to_fit(); // finalizes ids[0], then frees page 0
    assert_eq!(drops.load(Ordering::SeqCst), 2);
    let id = hive.insert(counted(9, &drops));
    assert_eq!(id.index(), 0);
    assert!(hive.get(ids[0]).is_none());
    drop(hive);
    assert_eq!(drops.load(Ordering::SeqCst), 5);
}

#[test]
fn contended_blocks_leave_no_retired_holes_after_conversion() {
    const THREADS: usize = 8;
    let rounds = if cfg!(miri) { 40 } else { 5_000 };
    let hive = ConcurrentHive::<usize, 64>::new();
    thread::scope(|scope| {
        for worker in 0..THREADS {
            let hive = &hive;
            scope.spawn(move || {
                for round in 0..rounds {
                    hive.insert(worker * rounds + round);
                }
            });
        }
    });
    let bound = hive.index_bound() as usize;
    assert!((THREADS * rounds..=THREADS * rounds + THREADS * 64).contains(&bound));
    let mut plain = hive.into_hive();
    assert_eq!(plain.len(), THREADS * rounds);
    let retired = plain
        .raw_pages()
        .flat_map(|page| page.slots)
        .filter(|slot| {
            matches!(
                slot,
                paged_collections::hive::RawSlot::Vacant { version: 0 }
            )
        })
        .count();
    assert_eq!(retired, 0, "never-issued indices must stay reusable");
    // The former holes are reused before the plain hive grows.
    for value in 0..bound - THREADS * rounds {
        let id = plain.insert(value);
        assert!((id.index() as usize) < bound);
    }
}
