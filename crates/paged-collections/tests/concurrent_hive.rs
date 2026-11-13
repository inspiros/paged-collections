//! Guard ownership, counted destruction, exclusive recovery and a plain oracle.

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
use paged_collections::hive::Hive;
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
