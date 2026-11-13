//! Overlapping insertion, generation conflicts, panic rollback and plain models.

#![cfg(all(feature = "concurrent", not(loom)))]
#![cfg(all(
    any(target_pointer_width = "32", target_pointer_width = "64"),
    target_has_atomic = "ptr",
    target_has_atomic = "64"
))]

use std::cell::Cell;
use std::num::NonZeroU32;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;

use paged_collections::concurrent::{ConcurrentSparseMap, ConcurrentSparseSet};
use paged_collections::id_alloc::Id;
use paged_collections::key::Key;
use paged_collections::sparse_map::PagedSparseMap;
use paged_collections::sparse_set::PagedSparseSet;
use proptest::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq)]
struct LocalKey(usize, std::marker::PhantomData<Rc<()>>);
impl Key for LocalKey {
    fn index(self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct SendKey(usize, std::marker::PhantomData<Cell<()>>);
impl Key for SendKey {
    fn index(self) -> usize {
        self.0
    }
}

const _: fn() = || {
    fn send<T: Send>() {}
    fn sync<T: Sync>() {}
    send::<ConcurrentSparseSet<SendKey, 2>>();
    send::<ConcurrentSparseMap<u32, Cell<u8>, 2>>();
    sync::<ConcurrentSparseSet<u32, 2>>();
    sync::<ConcurrentSparseMap<u32, u8, 2>>();
    trait AmbiguousSend<A> {
        fn check() {}
    }
    impl<T> AmbiguousSend<()> for T {}
    impl<T: Send> AmbiguousSend<u8> for T {}
    let _ = <ConcurrentSparseSet<LocalKey, 2> as AmbiguousSend<_>>::check;
    let _ = <ConcurrentSparseMap<u32, Rc<u8>, 2> as AmbiguousSend<_>>::check;
    trait AmbiguousSync<A> {
        fn check() {}
    }
    impl<T> AmbiguousSync<()> for T {}
    impl<T: Sync> AmbiguousSync<u8> for T {}
    let _ = <ConcurrentSparseSet<SendKey, 2> as AmbiguousSync<_>>::check;
    let _ = <ConcurrentSparseSet<LocalKey, 2> as AmbiguousSync<_>>::check;
    let _ = <ConcurrentSparseMap<SendKey, u8, 2> as AmbiguousSync<_>>::check;
    let _ = <ConcurrentSparseMap<u32, Cell<u8>, 2> as AmbiguousSync<_>>::check;
    let _ = <ConcurrentSparseMap<u32, std::sync::MutexGuard<'static, ()>, 2> as AmbiguousSync<_>>::check;
};

#[test]
fn overlapping_set_ranges_have_one_winner_per_key_and_full_pages() {
    let keys = if cfg!(miri) { 12 } else { 512 };
    let set = Arc::new(ConcurrentSparseSet::<usize, 7>::new());
    let wins = Arc::new((0..keys).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>());
    let mut workers = Vec::new();
    for worker in 0..8 {
        let set = set.clone();
        let wins = wins.clone();
        workers.push(thread::spawn(move || {
            for offset in 0..keys {
                let key = (offset + worker) % keys;
                if set.insert(key) {
                    // Relaxed counts winners; join synchronizes final reads.
                    wins[key].fetch_add(1, Ordering::Relaxed);
                }
                assert!(set.contains(key));
                assert!(set.position(key).is_some());
                for &stored in set.iter().take(8) {
                    assert!(stored < keys);
                }
            }
        }));
    }
    for worker in workers {
        worker.join().unwrap();
    }
    assert_eq!(set.len(), keys);
    for (key, wins) in wins.iter().enumerate() {
        // Relaxed: all increments happen before the completed joins.
        assert_eq!(wins.load(Ordering::Relaxed), 1);
        assert!(set.contains(key));
    }
    let pages: Vec<_> = set.pages().collect();
    assert!(pages.iter().all(|p| !p.is_empty() && p.len() <= 7));
    assert!(pages[..pages.len() - 1].iter().all(|p| p.len() == 7));
    let mut dense: Vec<_> = pages.into_iter().flatten().copied().collect();
    dense.sort_unstable();
    assert!(dense.into_iter().eq(0..keys));
}

struct Counted {
    key: usize,
    drops: Arc<AtomicUsize>,
}

impl Drop for Counted {
    fn drop(&mut self) {
        // SeqCst counts destruction across all writer and owner threads.
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn overlapping_map_ranges_count_winners_and_drop_losers_and_moved_values() {
    let keys = if cfg!(miri) { 8 } else { 256 };
    let map = Arc::new(ConcurrentSparseMap::<usize, Counted, 3>::new());
    let drops = Arc::new(AtomicUsize::new(0));
    let mut workers = Vec::new();
    for worker in 0..8 {
        let map = map.clone();
        let drops = drops.clone();
        workers.push(thread::spawn(move || {
            let mut wins = Vec::new();
            for offset in 0..keys {
                let key = (offset + worker) % keys;
                match map.try_insert(
                    key,
                    Counted {
                        key,
                        drops: drops.clone(),
                    },
                ) {
                    Ok(value) => {
                        assert_eq!(value.key, key);
                        wins.push(key);
                    }
                    Err((rejected, value)) => {
                        assert_eq!(rejected, value.key);
                    }
                }
                assert_eq!(map.get(&key).unwrap().key, key);
                assert!(map.contains_key(&key));
                for (&key, value) in map.iter().take(8) {
                    assert_eq!(key, value.key);
                }
            }
            wins
        }));
    }
    let mut wins: Vec<_> = workers
        .into_iter()
        .flat_map(|w| w.join().unwrap())
        .collect();
    wins.sort_unstable();
    assert!(wins.into_iter().eq(0..keys));
    assert_eq!(map.len(), keys);
    let mut map = Arc::try_unwrap(map).ok().unwrap();
    for key in (0..keys).step_by(2) {
        drop(map.remove(&key));
    }
    let mut map = ConcurrentSparseMap::from(map.into_inner());
    for (&key, value) in map.iter_mut() {
        assert_eq!(value.key, key);
    }
    map.clear();
    assert!(map.is_empty());
    assert!(map.get(&0).is_none());
    assert!(
        map.try_insert(
            0,
            Counted {
                key: 0,
                drops: drops.clone()
            }
        )
        .is_ok()
    );
    drop(map);
    // SeqCst observes every constructed value, including all failed inserts.
    assert_eq!(drops.load(Ordering::SeqCst), 8 * keys + 1);
}

#[test]
fn overlapping_factories_run_once_and_return_identical_addresses() {
    let keys = if cfg!(miri) { 4 } else { 64 };
    let map = Arc::new(ConcurrentSparseMap::<usize, usize, 2>::new());
    let calls = Arc::new((0..keys).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>());
    let mut workers = Vec::new();
    for _ in 0..8 {
        let map = map.clone();
        let calls = calls.clone();
        workers.push(thread::spawn(move || {
            (0..keys)
                .map(|key| {
                    let value = map
                        .get_or_insert_with(key, || {
                            // SeqCst detects duplicate factory execution per key.
                            calls[key].fetch_add(1, Ordering::SeqCst);
                            key + 17
                        })
                        .unwrap();
                    assert_eq!(*value, key + 17);
                    value as *const usize as usize
                })
                .collect::<Vec<_>>()
        }));
    }
    let addresses: Vec<_> = workers.into_iter().map(|w| w.join().unwrap()).collect();
    for row in &addresses {
        assert_eq!(row, &addresses[0]);
    }
    for calls in calls.iter() {
        // SeqCst observes all completed factories.
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
    assert_eq!(map.len(), keys);
}

fn id(index: u32, generation: u32) -> Id {
    Id::new(index, NonZeroU32::new(generation).unwrap())
}

#[test]
fn occupied_generations_are_refused_and_removal_reindexes_the_moved_entry() {
    let a = id(1, 1);
    let newer = id(1, 2);
    let b = id(8, 1);
    let c = id(3, 1);
    let mut set: ConcurrentSparseSet<Id, 2> = [a, b, c].into_iter().collect();
    assert!(!set.insert(a) && !set.insert(newer));
    assert!(!set.contains(newer) && !set.remove(newer));
    assert!(set.remove(a));
    assert_eq!(set.position(c), Some(0));
    assert_eq!(set.position(b), Some(1));
    assert!(set.insert(newer));
    let plain = set.into_inner();
    assert_eq!(plain.as_slice(), &[c, b, newer]);
    let mut set = ConcurrentSparseSet::from(plain);
    set.clear();
    assert!(set.iter().next().is_none());
    assert!(!set.contains(c));
    assert!(set.insert(a));

    let mut map = ConcurrentSparseMap::<Id, String, 2>::new();
    assert_eq!(
        map.try_insert(a, "old".into()).map(String::as_str),
        Ok("old")
    );
    assert_eq!(
        map.try_insert(newer, "new".into()),
        Err((newer, "new".into()))
    );
    assert_eq!(map.try_insert(a, "same".into()), Err((a, "same".into())));
    assert_eq!(
        map.get_or_insert_with(newer, || panic!("occupied generation")),
        None
    );
    assert_eq!(
        map.get_or_insert_with(a, || panic!("already present"))
            .map(String::as_str),
        Some("old")
    );
    assert_eq!(map.remove(&newer), None);
    assert!(map.try_insert(b, "b".into()).is_ok());
    assert!(map.try_insert(c, "c".into()).is_ok());
    assert_eq!(map.remove(&a), Some("old".into()));
    assert_eq!(map.iter().map(|(&k, _)| k).collect::<Vec<_>>(), [c, b]);
    assert_eq!(map.get(&c).map(String::as_str), Some("c"));
    map.get_mut(&b).unwrap().push('!');
    let map = ConcurrentSparseMap::from(map.into_inner());
    assert_eq!(map.get(&b).map(String::as_str), Some("b!"));
    assert!(format!("{map:?}").contains("b!"));
    assert!(format!("{set:?}").contains("Id"));
}

#[test]
fn factory_panic_restores_empty_and_a_waiter_can_insert() {
    let map = Arc::new(ConcurrentSparseMap::<u32, u32, 2>::new());
    let (started, ready) = mpsc::channel();
    let (release, wait) = mpsc::channel();
    let writer = map.clone();
    let a = thread::spawn(move || {
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                writer.get_or_insert_with(1, || {
                    started.send(()).unwrap();
                    wait.recv().unwrap();
                    panic!("factory failed");
                });
            }))
            .is_err()
        );
    });
    ready.recv().unwrap();
    assert!(map.get(&1).is_none());
    assert!(map.is_empty());
    let waiter = map.clone();
    let b = thread::spawn(move || *waiter.get_or_insert_with(1, || 42).unwrap());
    release.send(()).unwrap();
    a.join().unwrap();
    assert_eq!(b.join().unwrap(), 42);
    assert_eq!(map.get(&1), Some(&42));
    assert_eq!(map.len(), 1);

    let empty = ConcurrentSparseMap::<u32, u32, 2>::new();
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            empty.get_or_insert_with(7, || panic!("factory failed"));
        }))
        .is_err()
    );
    assert!(!empty.contains_key(&7));
    assert!(empty.is_empty());
    assert_eq!(empty.try_insert(7, 9), Ok(&9));
}

struct Bomb {
    panic: bool,
}
impl Drop for Bomb {
    fn drop(&mut self) {
        assert!(!self.panic, "drop bomb");
    }
}

#[test]
fn clear_panic_preserves_the_remaining_index_and_clears_removed_ready_bits() {
    let mut map = ConcurrentSparseMap::<u32, Bomb, 2>::new();
    assert!(map.try_insert(1, Bomb { panic: false }).is_ok());
    assert!(map.try_insert(2, Bomb { panic: true }).is_ok());
    assert!(catch_unwind(AssertUnwindSafe(|| map.clear())).is_err());
    assert_eq!(map.len(), 1);
    assert!(map.contains_key(&1));
    assert!(!map.contains_key(&2));
    assert!(map.try_insert(2, Bomb { panic: false }).is_ok());
    map.clear();
    assert!(map.is_empty());
}

#[test]
#[should_panic(expected = "key index exceeds sparse spine range")]
fn out_of_range_keys_panic_clearly() {
    let set = ConcurrentSparseSet::<usize, 1>::new();
    set.insert(usize::MAX);
}

#[test]
fn empty_defaults_and_mutable_iteration() {
    let mut set = ConcurrentSparseSet::<u32, 2>::default();
    assert!(set.is_empty());
    assert!(set.pages().next().is_none());
    assert!(!set.remove(0));
    let mut map = ConcurrentSparseMap::<u32, i32, 2>::default();
    assert!(map.iter_mut().next().is_none());
    assert_eq!(map.get_mut(&0), None);
    for key in 0..5 {
        assert!(map.try_insert(key, key as i32).is_ok());
    }
    {
        let mut iter = map.iter_mut();
        let (ka, a) = iter.next().unwrap();
        let (kb, b) = iter.next().unwrap();
        *a += 10;
        *b += 20;
        for (_, v) in iter {
            *v += 30;
        }
        assert_eq!((*ka, *a, *kb, *b), (0, 10, 1, 21));
    }
    assert_eq!(map.into_inner().value_slice(), &[10, 21, 32, 33, 34]);
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
    fn set_matches_plain_set_with_generations_and_conversions(
        actions in prop::collection::vec((0u8..6, 0u32..24, 1u32..4), 0..if cfg!(miri) { 20 } else { 200 })
    ) {
        let mut plain = PagedSparseSet::<Id, 3>::new();
        let mut set = ConcurrentSparseSet::<Id, 3>::new();
        for (action, index, generation) in actions {
            let key = id(index, generation);
            match action {
                0 | 1 => { prop_assert_eq!(set.insert(key), plain.insert(key)); }
                2 => { prop_assert_eq!(set.remove(key), plain.remove(key)); }
                3 => { set.clear(); plain.clear(); }
                4 => {
                    let moved = set.into_inner();
                    prop_assert_eq!(moved.as_slice(), plain.as_slice());
                    set = ConcurrentSparseSet::from(moved);
                }
                _ => { prop_assert_eq!(set.position(key), plain.position(key)); }
            }
            prop_assert_eq!(set.len(), plain.len());
            prop_assert_eq!(set.is_empty(), plain.is_empty());
            prop_assert!(set.iter().eq(plain.iter()));
            prop_assert!(set.pages().flatten().eq(plain.iter()));
            for index in 0..24 { for generation in 1..4 {
                let key = id(index, generation);
                prop_assert_eq!(set.contains(key), plain.contains(key));
                prop_assert_eq!(set.position(key), plain.position(key));
            } }
        }
        let moved = set.into_inner();
        prop_assert_eq!(moved.as_slice(), plain.as_slice());
    }

    #[test]
    fn map_matches_plain_insert_only_semantics_with_exclusive_edits(
        actions in prop::collection::vec((0u8..8, 0u32..24, 1u32..4, any::<i16>()), 0..if cfg!(miri) { 20 } else { 200 })
    ) {
        let mut plain = PagedSparseMap::<Id, i32, 3>::new();
        let mut map = ConcurrentSparseMap::<Id, i32, 3>::new();
        for (action, index, generation, value) in actions {
            let key = id(index, generation);
            let value = i32::from(value);
            match action {
                0 => {
                    let a = map.try_insert(key, value).copied();
                    let b = plain.try_insert(key, value).copied().map_err(|v| (key, v));
                    prop_assert_eq!(a, b);
                }
                1 => { prop_assert_eq!(map.remove(&key), plain.remove(key)); }
                2 => {
                    if let Some(v) = map.get_mut(&key) { *v = value; }
                    if let Some(v) = plain.get_mut(key) { *v = value; }
                }
                3 => {
                    for (_, v) in map.iter_mut() { *v += 1; }
                    for (_, v) in plain.iter_mut() { *v += 1; }
                }
                4 => { map.clear(); plain.clear(); }
                5 => {
                    let moved = map.into_inner();
                    prop_assert!(moved.iter().eq(plain.iter()));
                    map = ConcurrentSparseMap::from(moved);
                }
                6 => {
                    // Plain get_or_insert_with evicts generations; this API
                    // refuses them, so model that occupied-slot rule first.
                    let occupied = plain.as_set().occupant(index as usize).copied();
                    let expected = if occupied.is_some_and(|stored| stored != key) {
                        None
                    } else { Some(*plain.get_or_insert_with(key, || value)) };
                    prop_assert_eq!(map.get_or_insert_with(key, || value).copied(), expected);
                }
                _ => { prop_assert_eq!(map.get(&key), plain.get(key)); }
            }
            prop_assert_eq!(map.len(), plain.len());
            prop_assert_eq!(map.is_empty(), plain.is_empty());
            prop_assert!(map.iter().eq(plain.iter()));
            for index in 0..24 { for generation in 1..4 {
                let key = id(index, generation);
                prop_assert_eq!(map.contains_key(&key), plain.contains_key(key));
                prop_assert_eq!(map.get(&key), plain.get(key));
            } }
        }
        let moved = map.into_inner();
        prop_assert!(moved.iter().eq(plain.iter()));
    }
}
