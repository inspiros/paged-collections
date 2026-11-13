//! Sequential models, page contention, ownership and unwind safety.

#![cfg(all(feature = "concurrent", not(loom)))]
#![cfg(all(
    any(target_pointer_width = "32", target_pointer_width = "64"),
    target_has_atomic = "ptr",
    target_has_atomic = "64"
))]

use std::ops::Bound;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use paged_collections::concurrent::{ConcurrentDirectMap, Entry, SharedDirectory};
use paged_collections::direct_map::{Entry as PlainEntry, PagedDirectMap};
use paged_collections::key::Key;
use paged_collections::page::{Flat, Radix};
use proptest::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Gen {
    index: usize,
    generation: u8,
}

impl Key for Gen {
    fn index(self) -> usize {
        self.index
    }
}

fn contents<K: Key, V: Clone, const PAGE: usize, D: SharedDirectory>(
    map: &ConcurrentDirectMap<K, V, PAGE, D>,
) -> Vec<(K, V)> {
    let mut entries = Vec::new();
    map.for_each(|&key, value| entries.push((key, value.clone())));
    entries
}

fn sequential_model_for<D: SharedDirectory>(
    ops: Vec<(u8, usize, u8, i16)>,
) -> Result<(), TestCaseError> {
    let mut map = ConcurrentDirectMap::<Gen, i16, 4, D>::new();
    let mut plain = PagedDirectMap::<Gen, i16, 4, D>::new();
    for (op, index, generation, value) in ops {
        let key = Gen { index, generation };
        match op {
            0 => prop_assert_eq!(map.insert(key, value), plain.insert(key, value)),
            1 => prop_assert_eq!(map.replace(key, value), plain.replace(key, value)),
            2 => prop_assert_eq!(
                map.try_insert(key, value),
                plain.try_insert(key, value).map(|_| ())
            ),
            3 => prop_assert_eq!(map.remove(key), plain.remove(key)),
            4 => prop_assert_eq!(
                map.remove_occupant(key.index()),
                plain.remove_occupant(key.index())
            ),
            5 => {
                let mut actual = map.get_or_insert_with(key, || value);
                let expected = plain.get_or_insert_with(key, || value);
                *actual = actual.wrapping_add(1);
                *expected = expected.wrapping_add(1);
                prop_assert_eq!(*actual, *expected);
                prop_assert_eq!(*actual.key(), key);
            }
            6 => prop_assert_eq!(
                *map.entry(key).or_insert(value),
                *plain.entry(key).or_insert(value)
            ),
            7 => {
                let mut actual = map
                    .entry(key)
                    .and_modify(|v| *v = v.wrapping_add(1))
                    .or_insert_with_key(|k| k.index as i16);
                let expected = plain
                    .entry(key)
                    .and_modify(|v| *v = v.wrapping_add(1))
                    .or_insert_with_key(|k| k.index as i16);
                *actual = actual.wrapping_sub(1);
                *expected = expected.wrapping_sub(1);
                prop_assert_eq!(*actual, *expected);
            }
            8 => match (map.entry(key), plain.entry(key)) {
                (Entry::Occupied(mut a), PlainEntry::Occupied(mut b)) => {
                    prop_assert_eq!(a.key(), b.key());
                    prop_assert_eq!(a.get(), b.get());
                    *a.get_mut() = a.get().wrapping_add(1);
                    *b.get_mut() = b.get().wrapping_add(1);
                    prop_assert_eq!(a.insert(value), b.insert(value));
                    if value % 2 == 0 {
                        prop_assert_eq!(a.remove_entry(), b.remove_entry());
                    } else {
                        prop_assert_eq!(a.remove(), b.remove());
                    }
                }
                (Entry::Vacant(a), PlainEntry::Vacant(b)) => {
                    prop_assert_eq!(a.key(), b.key());
                    prop_assert_eq!(a.occupant(), b.occupant());
                    prop_assert_eq!(*a.insert(value), *b.insert(value));
                }
                _ => prop_assert!(false, "entry occupancy differs"),
            },
            9 => match (map.entry(key), plain.entry(key)) {
                (Entry::Occupied(a), PlainEntry::Occupied(b)) => {
                    prop_assert_eq!(*a.into_ref(), *b.into_mut());
                }
                (Entry::Vacant(a), PlainEntry::Vacant(b)) => {
                    prop_assert_eq!(a.occupant(), b.occupant());
                    prop_assert_eq!(a.into_key(), b.into_key());
                }
                _ => prop_assert!(false, "entry occupancy differs"),
            },
            10 => {
                let keep = |k: &Gen, v: &mut i16| {
                    *v = v.wrapping_add(value);
                    k.index % 3 != index % 3
                };
                map.retain(keep);
                plain.retain(keep);
            }
            11 => {
                if let Some(mut v) = map.get_mut(key) {
                    *v = v.wrapping_add(value);
                }
                if let Some(v) = plain.get_mut(key) {
                    *v = v.wrapping_add(value);
                }
            }
            12 => prop_assert_eq!(map.remove_entry(key), plain.remove_entry(key)),
            13 => prop_assert_eq!(*map.entry(key).or_default(), *plain.entry(key).or_default()),
            14 => prop_assert_eq!(
                *map.entry(key).or_insert_with(|| value),
                *plain.entry(key).or_insert_with(|| value)
            ),
            15 => {
                let actual = map.entry(key);
                let expected = plain.entry(key);
                prop_assert_eq!(actual.key(), expected.key());
            }
            16 => {
                for (_, v) in map.iter_mut() {
                    *v = v.wrapping_add(value);
                }
                for (_, v) in plain.iter_mut() {
                    *v = v.wrapping_add(value);
                }
            }
            17 => prop_assert_eq!(map.with(key, |v| *v), plain.get(key).copied()),
            18 => {
                map.clear();
                plain.clear();
            }
            _ => {
                map.reset();
                plain.reset();
            }
        }
        let expected: Vec<_> = plain.iter().map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(contents(&map), expected);
        prop_assert_eq!(map.len(), plain.len());
        prop_assert_eq!(map.is_empty(), plain.is_empty());
        prop_assert_eq!(map.contains_key(key), plain.contains_key(key));
        prop_assert_eq!(map.get_cloned(key), plain.get(key).copied());
        let mut actual_range = Vec::new();
        let bounds = (
            Bound::Excluded(key),
            Bound::Included(Gen {
                index: 20,
                generation: 0,
            }),
        );
        map.range_for_each(bounds, |&k, &v| actual_range.push((k, v)));
        let expected_range: Vec<_> = plain.range(bounds).map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(actual_range, expected_range);
    }
    let expected: Vec<_> = plain.into_iter().collect();
    let plain = map.into_inner();
    prop_assert_eq!(
        &plain.iter().map(|(&k, &v)| (k, v)).collect::<Vec<_>>(),
        &expected
    );
    let map = ConcurrentDirectMap::from(plain);
    prop_assert_eq!(contents(&map), expected);
    Ok(())
}

proptest! {
    #[test]
    fn sequential_model(ops in prop::collection::vec((0u8..20, 0u8..24, 0u8..3, any::<i16>()), 0..100)) {
        sequential_model_for::<Flat>(ops.into_iter().map(|(op, i, g, v)| (op, usize::from(i), g, v)).collect())?;
    }

    #[test]
    fn radix_sequential_model(ops in prop::collection::vec(
        (0u8..20, prop_oneof![0usize..24, (0usize..24).prop_map(|n| usize::MAX - n), any::<usize>()], 0u8..3, any::<i16>()),
        0..100,
    )) {
        sequential_model_for::<Radix>(ops)?;
    }
}

#[test]
fn disjoint_keys_contend_on_shared_pages() {
    let map = Arc::new(ConcurrentDirectMap::<usize, usize, 4>::new());
    let gate = Arc::new(Barrier::new(8));
    let mut workers = Vec::new();
    for worker in 0..8 {
        let map = map.clone();
        let gate = gate.clone();
        workers.push(thread::spawn(move || {
            gate.wait();
            for offset in 0..32 {
                let key = offset * 8 + worker;
                assert_eq!(map.insert(key, key), None);
                *map.get_mut(key).unwrap() += 1;
                assert_eq!(map.remove(key), Some(key + 1));
                assert!(map.try_insert(key, key + 2).is_ok());
            }
        }));
    }
    for worker in workers {
        worker.join().unwrap();
    }
    assert_eq!(map.len(), 256);
    assert_eq!(map.allocated_pages(), 64);
    assert_eq!(
        contents(&map),
        (0..256).map(|k| (k, k + 2)).collect::<Vec<_>>()
    );
}

#[test]
fn same_key_factory_runs_exactly_once() {
    let map = Arc::new(ConcurrentDirectMap::<u32, usize, 4>::new());
    let calls = Arc::new(AtomicUsize::new(0));
    let gate = Arc::new(Barrier::new(8));
    let mut workers = Vec::new();
    for _ in 0..8 {
        let map = map.clone();
        let calls = calls.clone();
        let gate = gate.clone();
        workers.push(thread::spawn(move || {
            gate.wait();
            assert_eq!(
                *map.get_or_insert_with(1, || {
                    calls.fetch_add(1, Ordering::Relaxed);
                    42
                }),
                42
            );
        }));
    }
    for worker in workers {
        worker.join().unwrap();
    }
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(map.len(), 1);
}

struct Counted {
    id: usize,
    drops: Arc<Vec<AtomicUsize>>,
}

impl Drop for Counted {
    fn drop(&mut self) {
        self.drops[self.id].fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn every_value_drops_once_across_eviction_rejection_reset_conversion_and_drop() {
    let drops = Arc::new((0..128).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>());
    let value = |id| Counted {
        id,
        drops: drops.clone(),
    };
    let mut map = ConcurrentDirectMap::<usize, Counted, 4>::new();
    for key in 0..32 {
        assert!(map.insert(key, value(key)).is_none());
        drop(map.replace(key, value(key + 32)));
        drop(map.try_insert(key, value(key + 64)));
        if key % 2 == 0 {
            drop(map.remove(key));
        }
    }
    map.reset();
    assert_eq!(map.len(), 0);
    assert_eq!(map.allocated_pages(), 0);
    assert!(drops[..96].iter().all(|n| n.load(Ordering::Relaxed) == 1));
    for key in 0..32 {
        assert!(map.insert(key, value(key + 96)).is_none());
    }
    let map = ConcurrentDirectMap::from(map.into_inner());
    drop(map);
    assert!(drops.iter().all(|n| n.load(Ordering::Relaxed) == 1));
}

fn assert_counts<V>(map: &ConcurrentDirectMap<Gen, V, 4>) {
    let mut count = 0;
    map.for_each(|_, _| count += 1);
    assert_eq!(map.len(), count);
}

#[test]
fn panicking_callbacks_preserve_entries_and_counts() {
    let map = ConcurrentDirectMap::<Gen, i32, 4>::new();
    let old = Gen {
        index: 1,
        generation: 0,
    };
    let new = Gen {
        index: 1,
        generation: 1,
    };
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            map.get_or_insert_with(old, || panic!("make empty"));
        }))
        .is_err()
    );
    assert_counts(&map);
    assert!(map.is_empty());
    map.insert(old, 7);
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            map.get_or_insert_with(new, || panic!("make other generation"));
        }))
        .is_err()
    );
    assert_eq!(map.get_cloned(old), Some(7));
    assert_counts(&map);
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            map.entry(old).and_modify(|v| {
                *v = 8;
                panic!("modify");
            });
        }))
        .is_err()
    );
    assert_eq!(map.get_cloned(old), Some(8));
    assert_counts(&map);
    map.insert(
        Gen {
            index: 2,
            generation: 0,
        },
        2,
    );
    map.insert(
        Gen {
            index: 7,
            generation: 0,
        },
        7,
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            map.retain(|key, value| {
                if key.index == 2 {
                    *value += 1;
                    panic!("keep");
                }
                false
            });
        }))
        .is_err()
    );
    assert_eq!(map.get_cloned(old), None);
    assert_eq!(map.len(), 2);
    assert_counts(&map);
}

struct Bomb {
    explode: bool,
}

impl Drop for Bomb {
    fn drop(&mut self) {
        assert!(!self.explode, "drop bomb");
    }
}

#[test]
fn panicking_destructors_leave_slots_and_counts_consistent() {
    let mut map = ConcurrentDirectMap::<Gen, Bomb, 4>::new();
    let key = Gen {
        index: 1,
        generation: 0,
    };
    let next = Gen {
        index: 1,
        generation: 1,
    };
    map.insert(key, Bomb { explode: true });
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            drop(map.get_or_insert_with(next, || Bomb { explode: false }));
        }))
        .is_err()
    );
    assert!(map.contains_key(next));
    assert_counts(&map);
    map.insert(key, Bomb { explode: true });
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            map.retain(|_, _| false);
        }))
        .is_err()
    );
    assert!(map.is_empty());
    assert_counts(&map);
    map.insert(key, Bomb { explode: true });
    map.insert(
        Gen {
            index: 2,
            generation: 0,
        },
        Bomb { explode: false },
    );
    assert!(catch_unwind(AssertUnwindSafe(|| map.clear())).is_err());
    assert_eq!(map.len(), 1);
    assert_counts(&map);
    map.clear();
    assert!(map.is_empty());
    assert_eq!(map.allocated_pages(), 1);
}

#[test]
fn ordered_range_bounds_and_generations() {
    let mut map = ConcurrentDirectMap::<usize, usize, 4>::new();
    map.extend([(9, 9), (0, 0), (4, 4), (7, 7)]);
    assert_eq!(format!("{map:?}"), "{0: 0, 4: 4, 7: 7, 9: 9}");
    let mut seen = Vec::new();
    map.range_for_each(4..=7, |&k, _| seen.push(k));
    assert_eq!(seen, [4, 7]);
    for bounds in [
        (Bound::Excluded(usize::MAX), Bound::Unbounded),
        (Bound::Unbounded, Bound::Excluded(0)),
        (Bound::Included(9), Bound::Included(4)),
        (Bound::Included(usize::MAX), Bound::Unbounded),
    ] {
        map.range_for_each(bounds, |_, _| panic!("empty range"));
    }
    map.clear();
    assert_eq!(map.allocated_pages(), 3);
    map.reset();
    assert_eq!(map.allocated_pages(), 0);
    let map = ConcurrentDirectMap::<Gen, i32, 4>::new();
    let stored = Gen {
        index: 3,
        generation: 2,
    };
    let bound = Gen {
        index: 3,
        generation: 0,
    };
    map.insert(stored, 3);
    map.range_for_each(bound..=bound, |&k, &v| assert_eq!((k, v), (stored, 3)));
}

#[derive(Clone, Copy, Debug, Eq)]
struct Tagged {
    index: u8,
    tag: u8,
}

impl PartialEq for Tagged {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl Key for Tagged {
    fn index(self) -> usize {
        usize::from(self.index)
    }
}

#[test]
fn equal_keys_preserve_stored_representation_except_replace() {
    let map = ConcurrentDirectMap::<Tagged, u32, 4>::new();
    let first = Tagged { index: 1, tag: 10 };
    let second = Tagged { index: 1, tag: 20 };
    map.insert(first, 1);
    assert_eq!(map.insert(second, 2), Some(1));
    assert_eq!(map.get(second).unwrap().key().tag, 10);
    match map.entry(second) {
        Entry::Occupied(mut entry) => {
            assert_eq!(entry.key().tag, 10);
            assert_eq!(entry.insert(3), 2);
            assert_eq!(entry.into_ref().key().tag, 10);
        }
        Entry::Vacant(_) => panic!("equal key is occupied"),
    }
    assert_eq!(map.replace(second, 4).unwrap().0.tag, 10);
    assert_eq!(map.get(first).unwrap().key().tag, 20);
    assert_eq!(map.len(), 1);
}

#[test]
fn holding_one_page_does_not_lock_another() {
    let map = ConcurrentDirectMap::<usize, usize, 4>::new();
    map.insert(0, 0);
    let mut first = map.get_mut(0).unwrap();
    map.insert(4, 4);
    let second = map.get(4).unwrap();
    *first = 1;
    assert_eq!(*second, 4);
    assert_eq!(*first, 1);
}

#[test]
fn flat_rejects_wide_indices_while_radix_accepts_them() {
    let flat = ConcurrentDirectMap::<usize, usize, 3>::new();
    assert_eq!(flat.get_cloned(usize::MAX), None);
    assert_eq!(flat.remove(usize::MAX), None);
    assert!(catch_unwind(AssertUnwindSafe(|| flat.insert(usize::MAX, 42))).is_err());
    let map = ConcurrentDirectMap::<usize, usize, 3, Radix>::new();
    assert_eq!(map.insert(usize::MAX, 42), None);
    assert_eq!(map.get_cloned(usize::MAX), Some(42));
    let mut seen = Vec::new();
    map.range_for_each(usize::MAX..=usize::MAX, |&k, &v| seen.push((k, v)));
    assert_eq!(seen, [(usize::MAX, 42)]);
    let plain: PagedDirectMap<usize, usize, 3, Radix> = map.into_inner();
    let mut map = ConcurrentDirectMap::from(plain);
    assert_eq!(map.remove(usize::MAX), Some(42));
    map.reset();
    assert_eq!(map.allocated_pages(), 0);
}

#[test]
fn radix_scoped_insertions_span_the_index_space() {
    let map = ConcurrentDirectMap::<usize, usize, 3, Radix>::new();
    let keys: Vec<_> = (0..usize::BITS)
        .map(|bit| 1usize << bit)
        .chain([0, usize::MAX - 1, usize::MAX])
        .collect();
    thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                for &key in &keys {
                    map.insert(key, key);
                    assert_eq!(map.get_cloned(key), Some(key));
                }
            });
        }
    });
    assert_eq!(map.len(), keys.len());
    let map = ConcurrentDirectMap::from(map.into_inner());
    for key in keys {
        assert_eq!(map.get_cloned(key), Some(key));
    }
}

#[test]
fn auto_traits_need_only_shared_directory() {
    fn check<D: SharedDirectory>() {
        fn send<T: Send>() {}
        fn sync<T: Sync>() {}
        send::<ConcurrentDirectMap<usize, std::cell::Cell<u8>, 2, D>>();
        sync::<ConcurrentDirectMap<usize, u8, 2, D>>();
        trait AmbiguousSend<A> {
            fn check() {}
        }
        impl<T> AmbiguousSend<()> for T {}
        impl<T: Send> AmbiguousSend<u8> for T {}
        let _ = <ConcurrentDirectMap<usize, std::rc::Rc<u8>, 2, D> as AmbiguousSend<_>>::check;
        trait AmbiguousSync<A> {
            fn check() {}
        }
        impl<T> AmbiguousSync<()> for T {}
        impl<T: Sync> AmbiguousSync<u8> for T {}
        let _ = <ConcurrentDirectMap<usize, std::cell::Cell<u8>, 2, D> as AmbiguousSync<_>>::check;
        let _ = <ConcurrentDirectMap<usize, std::sync::MutexGuard<'static, ()>, 2, D> as AmbiguousSync<_>>::check;
    }
    check::<Flat>();
    check::<Radix>();
}
