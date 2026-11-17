//! Overlapping insertion, generation conflicts, panic rollback and plain models.

#![cfg(all(feature = "concurrent", not(loom)))]
#![cfg(all(
    any(target_pointer_width = "32", target_pointer_width = "64"),
    target_has_atomic = "ptr",
    target_has_atomic = "64"
))]

use std::cell::Cell;
use std::collections::BTreeMap;
use std::num::NonZeroU32;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Barrier};
use std::thread;

use paged_collections::concurrent::{ConcurrentSparseMap, ConcurrentSparseSet, SharedDirectory};
use paged_collections::id_alloc::Id;
use paged_collections::key::Key;
use paged_collections::page::{Flat, Radix};
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WideKey {
    index: usize,
    generation: u32,
}

impl Key for WideKey {
    fn index(self) -> usize {
        self.index
    }
}

fn wide_index() -> impl Strategy<Value = usize> {
    prop_oneof![
        0usize..24,
        (0usize..24).prop_map(|n| usize::MAX - n),
        any::<usize>()
    ]
}

fn set_model<D: SharedDirectory, K: Key + std::fmt::Debug>(
    actions: Vec<(u8, usize, u32)>,
    key_at: impl Fn(usize, u32) -> K,
) -> Result<(), TestCaseError> {
    let mut probes: Vec<_> = (0..24)
        .chain(actions.iter().map(|&(_, index, _)| index))
        .collect();
    probes.sort_unstable();
    probes.dedup();
    let mut plain = PagedSparseSet::<K, 3, D>::new();
    let mut set = ConcurrentSparseSet::<K, 3, D>::new();
    for (action, index, generation) in actions {
        let key = key_at(index, generation);
        match action {
            0 | 1 => {
                prop_assert_eq!(set.insert(key), plain.insert(key));
            }
            2 => {
                prop_assert_eq!(set.remove(key), plain.remove(key));
            }
            3 => {
                set.clear();
                plain.clear();
            }
            4 => {
                let moved = set.into_inner();
                prop_assert_eq!(moved.as_slice(), plain.as_slice());
                set = ConcurrentSparseSet::from(moved);
            }
            _ => {
                prop_assert_eq!(set.position(key), plain.position(key));
            }
        }
        prop_assert_eq!(set.len(), plain.len());
        prop_assert_eq!(set.is_empty(), plain.is_empty());
        prop_assert!(set.iter().eq(plain.iter()));
        prop_assert!(set.pages().flatten().eq(plain.iter()));
        for &index in &probes {
            for generation in 1..4 {
                let key = key_at(index, generation);
                prop_assert_eq!(set.contains(key), plain.contains(key));
                prop_assert_eq!(set.position(key), plain.position(key));
            }
        }
    }
    let moved = set.into_inner();
    prop_assert_eq!(moved.as_slice(), plain.as_slice());
    Ok(())
}

fn map_model<D: SharedDirectory, K: Key + std::fmt::Debug>(
    actions: Vec<(u8, usize, u32, i16)>,
    key_at: impl Fn(usize, u32) -> K,
) -> Result<(), TestCaseError> {
    let mut probes: Vec<_> = (0..24)
        .chain(actions.iter().map(|&(_, index, _, _)| index))
        .collect();
    probes.sort_unstable();
    probes.dedup();
    let mut plain = PagedSparseMap::<K, i32, 3, D>::new();
    let mut map = ConcurrentSparseMap::<K, i32, 3, D>::new();
    for (action, index, generation, value) in actions {
        let key = key_at(index, generation);
        let value = i32::from(value);
        match action {
            0 => {
                let a = map.try_insert(key, value).copied();
                let b = plain.try_insert(key, value).copied().map_err(|v| (key, v));
                prop_assert_eq!(a, b);
            }
            1 => {
                prop_assert_eq!(map.remove(&key), plain.remove(key));
            }
            2 => {
                if let Some(v) = map.get_mut(&key) {
                    *v = value;
                }
                if let Some(v) = plain.get_mut(key) {
                    *v = value;
                }
            }
            3 => {
                for (_, v) in map.iter_mut() {
                    *v += 1;
                }
                for (_, v) in plain.iter_mut() {
                    *v += 1;
                }
            }
            4 => {
                map.clear();
                plain.clear();
            }
            5 => {
                let moved = map.into_inner();
                prop_assert!(moved.iter().eq(plain.iter()));
                map = ConcurrentSparseMap::from(moved);
            }
            6 => {
                // Plain get_or_insert_with evicts generations; this API
                // refuses them, so model that occupied-slot rule first.
                let occupied = plain.as_set().occupant(index).copied();
                let expected = if occupied.is_some_and(|stored| stored != key) {
                    None
                } else {
                    Some(*plain.get_or_insert_with(key, || value))
                };
                prop_assert_eq!(map.get_or_insert_with(key, || value).copied(), expected);
            }
            _ => {
                prop_assert_eq!(map.get(&key), plain.get(key));
            }
        }
        prop_assert_eq!(map.len(), plain.len());
        prop_assert_eq!(map.is_empty(), plain.is_empty());
        prop_assert!(map.iter().eq(plain.iter()));
        for &index in &probes {
            for generation in 1..4 {
                let key = key_at(index, generation);
                prop_assert_eq!(map.contains_key(&key), plain.contains_key(key));
                prop_assert_eq!(map.get(&key), plain.get(key));
            }
        }
    }
    let moved = map.into_inner();
    prop_assert!(moved.iter().eq(plain.iter()));
    Ok(())
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn set_matches_plain_set_with_generations_and_conversions(
        actions in prop::collection::vec((0u8..6, 0u32..24, 1u32..4), 0..if cfg!(miri) { 20 } else { 200 })
    ) {
        set_model::<Flat, Id>(actions.into_iter().map(|(a, i, g)| (a, i as usize, g)).collect(), |i, g| id(i as u32, g))?;
    }

    #[test]
    fn map_matches_plain_insert_only_semantics_with_exclusive_edits(
        actions in prop::collection::vec((0u8..8, 0u32..24, 1u32..4, any::<i16>()), 0..if cfg!(miri) { 20 } else { 200 })
    ) {
        map_model::<Flat, Id>(actions.into_iter().map(|(a, i, g, v)| (a, i as usize, g, v)).collect(), |i, g| id(i as u32, g))?;
    }

    #[test]
    fn radix_set_matches_plain_set(
        actions in prop::collection::vec((0u8..6, wide_index(), 1u32..4), 0..if cfg!(miri) { 20 } else { 200 })
    ) {
        set_model::<Radix, WideKey>(actions, |index, generation| WideKey { index, generation })?;
    }

    #[test]
    fn radix_map_matches_plain_map(
        actions in prop::collection::vec((0u8..8, wide_index(), 1u32..4, any::<i16>()), 0..if cfg!(miri) { 20 } else { 200 })
    ) {
        map_model::<Radix, WideKey>(actions, |index, generation| WideKey { index, generation })?;
    }
}

#[test]
fn flat_set_rejects_wide_indices_while_radix_accepts_them() {
    let flat = ConcurrentSparseSet::<usize, 3>::new();
    assert!(catch_unwind(AssertUnwindSafe(|| flat.insert(usize::MAX))).is_err());
    assert!(catch_unwind(AssertUnwindSafe(|| flat.contains(usize::MAX))).is_err());
    let set = ConcurrentSparseSet::<usize, 3, Radix>::new();
    assert!(set.insert(usize::MAX));
    assert!(set.contains(usize::MAX));
    let plain: PagedSparseSet<usize, 3, Radix> = set.into_inner();
    assert!(ConcurrentSparseSet::from(plain).contains(usize::MAX));
}

#[test]
fn flat_map_rejects_wide_indices_while_radix_accepts_them() {
    let flat = ConcurrentSparseMap::<usize, u32, 3>::new();
    assert!(catch_unwind(AssertUnwindSafe(|| flat.try_insert(usize::MAX, 42))).is_err());
    assert!(catch_unwind(AssertUnwindSafe(|| flat.get(&usize::MAX))).is_err());
    let map = ConcurrentSparseMap::<usize, u32, 3, Radix>::new();
    assert_eq!(map.try_insert(usize::MAX, 42), Ok(&42));
    let plain: PagedSparseMap<usize, u32, 3, Radix> = map.into_inner();
    assert_eq!(ConcurrentSparseMap::from(plain).get(&usize::MAX), Some(&42));
}

#[test]
fn radix_scoped_insertions_span_the_index_space() {
    let set = ConcurrentSparseSet::<usize, 3, Radix>::new();
    let map = ConcurrentSparseMap::<usize, usize, 3, Radix>::new();
    let keys: Vec<_> = (0..usize::BITS)
        .map(|bit| 1usize << bit)
        .chain([0, usize::MAX - 1, usize::MAX])
        .collect();
    thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                for &key in &keys {
                    set.insert(key);
                    let _ = map.try_insert(key, key);
                    assert!(set.contains(key));
                    assert_eq!(map.get(&key), Some(&key));
                }
            });
        }
    });
    assert_eq!(set.len(), keys.len());
    assert_eq!(map.len(), keys.len());
    let set = ConcurrentSparseSet::from(set.into_inner());
    let map = ConcurrentSparseMap::from(map.into_inner());
    for key in keys {
        assert!(set.contains(key));
        assert_eq!(map.get(&key), Some(&key));
    }
}

#[test]
fn auto_traits_need_only_shared_directory() {
    fn check<D: SharedDirectory>() {
        fn send<T: Send>() {}
        fn sync<T: Sync>() {}
        send::<ConcurrentSparseSet<SendKey, 2, D>>();
        send::<ConcurrentSparseMap<u32, Cell<u8>, 2, D>>();
        sync::<ConcurrentSparseSet<u32, 2, D>>();
        sync::<ConcurrentSparseMap<u32, u8, 2, D>>();
        trait AmbiguousSend<A> {
            fn check() {}
        }
        impl<T> AmbiguousSend<()> for T {}
        impl<T: Send> AmbiguousSend<u8> for T {}
        let _ = <ConcurrentSparseSet<LocalKey, 2, D> as AmbiguousSend<_>>::check;
        let _ = <ConcurrentSparseMap<u32, Rc<u8>, 2, D> as AmbiguousSend<_>>::check;
        trait AmbiguousSync<A> {
            fn check() {}
        }
        impl<T> AmbiguousSync<()> for T {}
        impl<T: Sync> AmbiguousSync<u8> for T {}
        let _ = <ConcurrentSparseSet<SendKey, 2, D> as AmbiguousSync<_>>::check;
        let _ = <ConcurrentSparseMap<u32, Cell<u8>, 2, D> as AmbiguousSync<_>>::check;
    }
    check::<Flat>();
    check::<Radix>();
}

fn batched_oracle<const PAGE: usize, D: SharedDirectory>() {
    let set = ConcurrentSparseSet::<usize, PAGE, D>::new();
    let map = ConcurrentSparseMap::<usize, isize, PAGE, D>::new();
    let mut oracle = BTreeMap::new();
    assert_eq!(set.insert_many(core::iter::empty()), 0);
    assert_eq!(map.try_insert_many(core::iter::empty()), 0);
    assert!(set.is_empty() && map.is_empty());
    assert_eq!(set.iter().count(), 0);
    assert_eq!(map.iter().count(), 0);
    // Spread sparse keys over many pages; dense writes also cross pages and
    // bitmap words. In particular PAGE=63/65 clips a word at the page end.
    let key = |i: usize| i * 131;
    let initial: Vec<_> = (0..300).map(|i| (key(i), i as isize + 10_000)).collect();
    assert_eq!(set.insert_many(initial.iter().map(|&(k, _)| k)), 300);
    assert_eq!(map.try_insert_many(initial.iter().copied()), 300);
    oracle.extend(initial);
    // More than two full batches claim NOTHING before fresh keys appear.
    // A previous implementation stopped on the first such batch.
    let input: Vec<_> = (0..600)
        .map(|i| (key(i % 300), -1))
        .chain((300..1000).flat_map(|i| [(key(i), i as isize), (key(i), -(i as isize))]))
        .collect();
    let before = oracle.len();
    for &(key, value) in &input {
        oracle.entry(key).or_insert(value);
    }
    let expected = oracle.len() - before;
    assert_eq!(set.insert_many(input.iter().map(|&(k, _)| k)), expected);
    assert_eq!(map.try_insert_many(input.iter().copied()), expected);
    assert_eq!(set.len(), oracle.len());
    assert_eq!(map.len(), oracle.len());
    assert_eq!(set.iter().count(), oracle.len());
    assert_eq!(map.iter().count(), oracle.len());
    let mut keys: Vec<_> = set.iter().copied().collect();
    keys.sort_unstable();
    assert!(keys.into_iter().eq(oracle.keys().copied()));
    assert_eq!(
        map.iter()
            .map(|(&k, &v)| (k, v))
            .collect::<BTreeMap<_, _>>(),
        oracle
    );
    for (&key, value) in &oracle {
        assert!(set.contains(key));
        assert_eq!(map.get(&key), Some(value));
    }
    // Empty calls on populated collections reserve no positions, either.
    assert_eq!(set.insert_many([]), 0);
    assert_eq!(map.try_insert_many([]), 0);
    assert_eq!(set.len(), oracle.len());
    assert_eq!(map.len(), oracle.len());
    assert_eq!(set.pages().flatten().count(), oracle.len());
    let plain = map.into_inner();
    assert_eq!(plain.len(), oracle.len());
    assert_eq!(
        plain
            .iter()
            .map(|(&k, &v)| (k, v))
            .collect::<BTreeMap<_, _>>(),
        oracle
    );
}

#[test]
fn batched_set_and_map_match_oracle_at_page_and_word_boundaries() {
    batched_oracle::<3, Flat>();
    batched_oracle::<63, Flat>();
    batched_oracle::<64, Flat>();
    batched_oracle::<65, Flat>();
    batched_oracle::<128, Flat>();
    batched_oracle::<65, Radix>();
}

#[test]
fn batch_iterator_panic_keeps_completed_batches_and_rolls_back_partial_batch() {
    const COMPLETED: usize = 256;
    const PANIC_AT: usize = COMPLETED + 19;
    let set = ConcurrentSparseSet::<usize, 3>::new();
    let map = ConcurrentSparseMap::<usize, usize, 3>::new();
    let key = |i| i * 17;
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            set.insert_many((0..512).map(|i| {
                assert_ne!(i, PANIC_AT, "iterator failed");
                key(i)
            }));
        }))
        .is_err()
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            map.try_insert_many((0..512).map(|i| {
                assert_ne!(i, PANIC_AT, "iterator failed");
                (key(i), i)
            }));
        }))
        .is_err()
    );
    let mut oracle: BTreeMap<_, _> = (0..COMPLETED).map(|i| (key(i), i)).collect();
    assert_eq!(set.len(), COMPLETED);
    assert_eq!(map.len(), COMPLETED);
    assert!(set.iter().copied().eq(oracle.keys().copied()));
    assert_eq!(
        map.iter()
            .map(|(&k, &v)| (k, v))
            .collect::<BTreeMap<_, _>>(),
        oracle
    );
    for i in COMPLETED..PANIC_AT {
        assert!(!set.contains(key(i)));
        assert_eq!(set.position(key(i)), None);
        assert_eq!(map.get(&key(i)), None);
    }
    let retry: Vec<_> = (COMPLETED..300).map(|i| (key(i), i + 1000)).collect();
    assert_eq!(set.insert_many(retry.iter().map(|&(k, _)| k)), retry.len());
    assert_eq!(map.try_insert_many(retry.iter().copied()), retry.len());
    oracle.extend(retry);
    assert_eq!(set.len(), oracle.len());
    assert_eq!(map.len(), oracle.len());
    assert_eq!(set.iter().count(), oracle.len());
    assert_eq!(map.iter().count(), oracle.len());
    assert!(set.iter().copied().eq(oracle.keys().copied()));
    assert_eq!(
        map.iter()
            .map(|(&k, &v)| (k, v))
            .collect::<BTreeMap<_, _>>(),
        oracle
    );
    for (&key, value) in &oracle {
        assert!(set.contains(key));
        assert_eq!(map.get(&key), Some(value));
    }
}

#[test]
fn batch_sparse_allocation_range_panic_rolls_back_earlier_claims() {
    let set = ConcurrentSparseSet::<usize, 3>::new();
    let map = ConcurrentSparseMap::<usize, usize, 3>::new();
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            set.insert_many([0, 3, usize::MAX]);
        }))
        .is_err()
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            map.try_insert_many([(0, 10), (3, 30), (usize::MAX, 0)]);
        }))
        .is_err()
    );
    assert_eq!(set.len(), 0);
    assert_eq!(map.len(), 0);
    assert_eq!(set.iter().count(), 0);
    assert_eq!(map.iter().count(), 0);
    assert_eq!(set.insert_many([0, 3]), 2);
    assert_eq!(map.try_insert_many([(0, 10), (3, 30)]), 2);
    assert!(set.contains(0) && set.contains(3));
    assert_eq!(map.get(&0), Some(&10));
    assert_eq!(map.get(&3), Some(&30));
    assert_eq!(set.len(), 2);
    assert_eq!(map.len(), 2);
}

#[test]
fn four_scoped_batch_writers_match_overlapping_range_oracle() {
    let set = ConcurrentSparseSet::<usize, 65>::new();
    let map = ConcurrentSparseMap::<usize, usize, 65>::new();
    let gate = Barrier::new(4);
    let width = if cfg!(miri) { 260 } else { 1024 };
    let stride = width / 2;
    let oracle: BTreeMap<_, _> = (0..4)
        .flat_map(|worker| (worker * stride..worker * stride + width).map(|i| (i * 17, i)))
        .collect();
    let counts = thread::scope(|scope| {
        let mut workers = Vec::new();
        for worker in 0..4 {
            let (set, map, gate) = (&set, &map, &gate);
            workers.push(scope.spawn(move || {
                gate.wait();
                let entries: Vec<_> = (worker * stride..worker * stride + width)
                    .rev()
                    .flat_map(|i| [(i * 17, i), (i * 17, i)])
                    .collect();
                let sets = set.insert_many(entries.iter().map(|&(k, _)| k));
                let maps = map.try_insert_many(entries);
                (sets, maps)
            }));
        }
        workers
            .into_iter()
            .map(|w| w.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_eq!(counts.iter().map(|c| c.0).sum::<usize>(), oracle.len());
    assert_eq!(counts.iter().map(|c| c.1).sum::<usize>(), oracle.len());
    assert_eq!(set.len(), oracle.len());
    assert_eq!(map.len(), oracle.len());
    let mut keys: Vec<_> = set.iter().copied().collect();
    assert_eq!(keys.len(), oracle.len());
    keys.sort_unstable();
    assert!(keys.into_iter().eq(oracle.keys().copied()));
    assert_eq!(map.iter().count(), oracle.len());
    assert_eq!(
        map.iter()
            .map(|(&k, &v)| (k, v))
            .collect::<BTreeMap<_, _>>(),
        oracle
    );
    for (&key, value) in &oracle {
        assert!(set.contains(key));
        assert_eq!(map.get(&key), Some(value));
    }
}

#[test]
fn batch_keeps_first_duplicate_after_competing_factory_rolls_back() {
    // Regression for the first-wins contract: the first entry is deferred
    // while another writer owns the slot, which rolls back before entry two.
    let map = ConcurrentSparseMap::<u32, &'static str, 2>::new();
    thread::scope(|scope| {
        let (started, ready) = mpsc::channel();
        let (release, wait) = mpsc::channel();
        let writer = &map;
        let owner = scope.spawn(move || {
            assert!(
                catch_unwind(AssertUnwindSafe(|| {
                    writer.get_or_insert_with(1, || {
                        started.send(()).unwrap();
                        wait.recv().unwrap();
                        panic!("competing factory rolls back");
                    });
                }))
                .is_err()
            );
        });
        ready.recv().unwrap();
        let mut owner = Some(owner);
        let mut step = 0;
        let entries = core::iter::from_fn(move || {
            step += 1;
            match step {
                1 => Some((1, "first")),
                2 => {
                    // next() is called only after the first entry tried the
                    // still-reserved slot. Complete rollback before yielding
                    // the second entry; this iterator never inserts itself.
                    release.send(()).unwrap();
                    owner.take().unwrap().join().unwrap();
                    Some((1, "second"))
                }
                _ => None,
            }
        });
        assert_eq!(map.try_insert_many(entries), 1);
    });
    assert_eq!(map.len(), 1);
    assert_eq!(map.iter().count(), 1);
    assert_eq!(map.get(&1), Some(&"first"));
}
