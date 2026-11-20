//! Sequential models, boundary cases and shared atomic updates.

#![cfg(all(feature = "concurrent", not(loom)))]
#![cfg(all(
    any(target_pointer_width = "32", target_pointer_width = "64"),
    target_has_atomic = "ptr",
    target_has_atomic = "64"
))]

use std::collections::BTreeMap;
use std::num::NonZeroU32;
use std::ops::Bound;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Barrier};
use std::thread;

use paged_collections::concurrent::{AtomicDirectMap, AtomicIdMap, SharedDirectory};
use paged_collections::direct_map::PagedDirectMap;
use paged_collections::id_alloc::Id;
use paged_collections::page::{Flat, Radix};
use proptest::prelude::*;

const MAX_VALUE: u64 = AtomicDirectMap::<4>::MAX_VALUE;

fn id(index: u32, generation: u32) -> Id {
    Id::new(index, NonZeroU32::new(generation).unwrap())
}

#[test]
fn borrowed_for_loops_copy_entries_in_index_order() {
    let direct: AtomicDirectMap<3, Radix> =
        [(usize::MAX, 13), (2, 0), (1, 9)].into_iter().collect();
    let mut entries = Vec::new();
    for entry in &direct {
        entries.push(entry);
    }
    assert_eq!(entries, [(1, 9), (2, 0), (usize::MAX, 13)]);

    let ids: AtomicIdMap<3> = [(id(8, 1), 13), (id(2, 3), 0)].into_iter().collect();
    let mut entries = Vec::new();
    for entry in &ids {
        entries.push(entry);
    }
    assert_eq!(entries, [(id(2, 3), 0), (id(8, 1), 13)]);
}

#[test]
fn atomic_direct_shrink_releases_empty_pages_and_keeps_live_words() {
    fn check<D: SharedDirectory>(last: usize) {
        let mut map = AtomicDirectMap::<3, D>::new();
        let mut model = BTreeMap::new();
        for (key, value) in [(18, 1), (0, 2), (8, 0), (7, 4), (last, MAX_VALUE)] {
            assert_eq!(map.insert(key, value), model.insert(key, value));
        }
        for key in [0, 7, 18] {
            assert_eq!(map.remove(key), model.remove(&key));
        }
        let before = map.allocated_pages();
        map.shrink_to_fit();
        assert_eq!(map.allocated_pages(), before - 2);
        assert_eq!(map.len(), model.len());
        assert!(
            map.iter()
                .eq(model.iter().map(|(&key, &value)| (key, value)))
        );

        let shared = &map;
        for key in [0, 18] {
            assert_eq!(shared.insert(key, 6), model.insert(key, 6));
        }
        assert_eq!(shared.allocated_pages(), before);
        assert_eq!(shared.len(), model.len());
        assert!(shared.iter().eq(model));
        map.clear();
        map.shrink_to_fit();
        assert_eq!(map.allocated_pages(), 0);
        assert_eq!(map.insert(last, 7), None);
    }
    check::<Flat>(121);
    check::<Radix>(usize::MAX);
}

#[test]
fn atomic_id_shrink_releases_empty_pages_and_preserves_generations() {
    let mut map = AtomicIdMap::<3>::new();
    let mut model = BTreeMap::new();
    for (key, value) in [
        (id(8, 3), 0),
        (id(7, 1), 4),
        (id(0, 3), u32::MAX),
        (id(18, 5), 6),
    ] {
        assert_eq!(map.insert(key, value), model.insert(key, value));
    }
    for key in [id(0, 3), id(7, 1), id(18, 5)] {
        assert_eq!(map.remove(key), model.remove(&key));
    }
    map.shrink_to_fit();
    assert_eq!(map.allocated_pages(), 1);
    assert_eq!(map.len(), model.len());
    assert!(
        map.iter()
            .eq(model.iter().map(|(&key, &value)| (key, value)))
    );
    assert_eq!(map.get(id(8, 2)), None);

    let shared = &map;
    for key in [id(0, 4), id(18, 6)] {
        assert_eq!(shared.insert(key, 0), model.insert(key, 0));
    }
    assert_eq!(shared.allocated_pages(), 3);
    assert_eq!(shared.len(), model.len());
    assert!(shared.iter().eq(model));
    assert_eq!(shared.get(id(18, 5)), None);
}

fn direct_value() -> impl Strategy<Value = u64> {
    prop_oneof![Just(0), Just(MAX_VALUE), 0u64..4, 0u64..u64::MAX]
}

fn direct_update(old: Option<u64>, action: u8, value: u64) -> Option<Option<u64>> {
    match action {
        0 => None,
        1 => Some(None),
        2 => Some(Some(value)),
        _ => old
            .unwrap_or(0)
            .checked_add(1)
            .filter(|&v| v <= MAX_VALUE)
            .map(Some),
    }
}

fn id_update(old: Option<u32>, action: u8, value: u32) -> Option<Option<u32>> {
    match action {
        0 => None,
        1 => Some(None),
        2 => Some(Some(value)),
        _ => Some(Some(old.unwrap_or(0).wrapping_add(1))),
    }
}

fn direct_model<D: SharedDirectory>(
    ops: Vec<(u8, usize, u64, u8, bool)>,
) -> Result<(), TestCaseError> {
    let map = AtomicDirectMap::<4, D>::new();
    let mut model = BTreeMap::new();
    for (op, key, value, action, store) in ops {
        let old = model.get(&key).copied();
        match op {
            0 => prop_assert_eq!(map.get(key), old),
            1 => prop_assert_eq!(map.insert(key, value), model.insert(key, value)),
            2 => {
                let expected = match old {
                    Some(current) => Err(current),
                    None => {
                        model.insert(key, value);
                        Ok(())
                    }
                };
                prop_assert_eq!(map.try_insert(key, value), expected);
            }
            3 => prop_assert_eq!(map.remove(key), model.remove(&key)),
            4 => {
                let current = match action {
                    0 => None,
                    1 => Some(value),
                    _ => old,
                };
                let new = store.then_some(value);
                let expected = if old == current {
                    match new {
                        Some(value) => {
                            model.insert(key, value);
                        }
                        None => {
                            model.remove(&key);
                        }
                    }
                    Ok(old)
                } else {
                    Err(old)
                };
                prop_assert_eq!(map.compare_exchange(key, current, new), expected);
            }
            _ => {
                let expected = match direct_update(old, action, value) {
                    None => Err(old),
                    Some(new) => {
                        match new {
                            Some(value) => {
                                model.insert(key, value);
                            }
                            None => {
                                model.remove(&key);
                            }
                        }
                        Ok(old)
                    }
                };
                prop_assert_eq!(
                    map.fetch_update(key, |v| direct_update(v, action, value)),
                    expected
                );
            }
        }
        let expected: Vec<_> = model.iter().map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(map.iter().collect::<Vec<_>>(), expected);
        prop_assert_eq!(map.len(), model.len());
        prop_assert_eq!(map.is_empty(), model.is_empty());
        prop_assert_eq!(map.contains_key(key), model.contains_key(&key));
        let expected: Vec<_> = model.range(5..=25).map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(map.range(5..=25).collect::<Vec<_>>(), expected);
    }
    let expected: Vec<_> = model.into_iter().collect();
    let plain: PagedDirectMap<usize, u64, 4, D> = map.into_inner();
    prop_assert!(
        plain
            .iter()
            .map(|(k, &v)| (k, v))
            .eq(expected.iter().copied())
    );
    let map = AtomicDirectMap::from(plain);
    prop_assert_eq!(map.iter().collect::<Vec<_>>(), expected);
    Ok(())
}

proptest! {
    #[test]
    fn direct_sequential_model(
        ops in prop::collection::vec(
            (0u8..6, 0usize..40, direct_value(), 0u8..4, any::<bool>()),
            0..200,
        ),
    ) {
        direct_model::<Flat>(ops)?;
    }

    #[test]
    fn radix_direct_sequential_model(
        ops in prop::collection::vec(
            (0u8..6, prop_oneof![0usize..40, (0usize..40).prop_map(|n| usize::MAX - n), any::<usize>()], direct_value(), 0u8..4, any::<bool>()),
            0..200,
        ),
    ) {
        direct_model::<Radix>(ops)?;
    }

    #[test]
    fn id_sequential_model(
        ops in prop::collection::vec(
            (0u8..9, 0u32..20, 1u32..=4, prop_oneof![Just(0), Just(u32::MAX), any::<u32>()], 0u8..4, any::<bool>()),
            0..200,
        ),
    ) {
        let map = AtomicIdMap::<4>::new();
        let mut model = PagedDirectMap::<Id, u32, 4>::new();
        for (op, index, generation, value, action, store) in ops {
            let key = id(index, generation);
            let old = model.get(key).copied();
            match op {
                0 => prop_assert_eq!(map.get(key), old),
                1 => prop_assert_eq!(map.insert(key, value), model.insert(key, value)),
                2 => {
                    let expected = match model.occupant(index as usize) {
                        Some((held, &current)) => Err((held, current)),
                        None => { model.insert(key, value); Ok(()) }
                    };
                    prop_assert_eq!(map.try_insert(key, value), expected);
                }
                3 => prop_assert_eq!(map.remove(key), model.remove(key)),
                4 => {
                    let current = match action {
                        0 => None,
                        1 => Some(value),
                        _ => old,
                    };
                    let new = store.then_some(value);
                    let expected = if old == current {
                        match new {
                            Some(value) => { model.insert(key, value); }
                            None => { model.remove(key); }
                        }
                        Ok(old)
                    } else {
                        Err(old)
                    };
                    prop_assert_eq!(map.compare_exchange(key, current, new), expected);
                }
                5 => {
                    let expected = match id_update(old, action, value) {
                        None => Err(old),
                        Some(new) => {
                            match new {
                                Some(value) => { model.insert(key, value); }
                                None => { model.remove(key); }
                            }
                            Ok(old)
                        }
                    };
                    prop_assert_eq!(map.fetch_update(key, |v| id_update(v, action, value)), expected);
                }
                6 => prop_assert_eq!(map.replace(key, value), model.replace(key, value)),
                7 => prop_assert_eq!(map.remove_occupant(index), model.remove_occupant(index as usize)),
                _ => prop_assert_eq!(map.occupant(index), model.occupant(index as usize).map(|(k, &v)| (k, v))),
            }
            let expected: Vec<_> = model.iter().map(|(k, &v)| (k, v)).collect();
            prop_assert_eq!(map.iter().collect::<Vec<_>>(), expected);
            prop_assert_eq!(map.len(), model.len());
            prop_assert_eq!(map.is_empty(), model.is_empty());
            prop_assert_eq!(map.contains_key(key), model.contains_key(key));
            let expected: Vec<_> = model.slot_range(5..=15).map(|(k, &v)| (k, v)).collect();
            prop_assert_eq!(map.range(5..=15).collect::<Vec<_>>(), expected);
        }
    }
}

#[test]
fn send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    fn check_iterator<T: Send + Sync>(_: T) {}
    fn directories<D: SharedDirectory>() {
        assert_send_sync::<AtomicDirectMap<3, D>>();
        check_iterator(AtomicDirectMap::<3, D>::new().iter());
    }
    directories::<Flat>();
    directories::<Radix>();
    assert_send_sync::<AtomicDirectMap>();
    assert_send_sync::<AtomicIdMap>();
    check_iterator(AtomicIdMap::<3>::new().iter());
}

#[test]
fn direct_rejects_unstorable_values() {
    let map = AtomicDirectMap::<4>::new();
    map.insert(1, 7);
    for operation in 0..4 {
        assert!(
            catch_unwind(AssertUnwindSafe(|| match operation {
                0 => {
                    map.insert(1, u64::MAX);
                }
                1 => {
                    let _ = map.try_insert(1, u64::MAX);
                }
                2 => {
                    let _ = map.compare_exchange(1, None, Some(u64::MAX));
                }
                _ => {
                    let _ = map.fetch_update(1, |_| Some(Some(u64::MAX)));
                }
            }))
            .is_err()
        );
        assert_eq!(map.get(1), Some(7));
        assert_eq!(map.len(), 1);
    }
    assert_eq!(map.compare_exchange(1, Some(u64::MAX), None), Err(Some(7)));
}

#[test]
fn absent_updates_do_not_install_pages_and_wide_lookups_are_absent() {
    let direct = AtomicDirectMap::<4>::new();
    for key in [0, 7, usize::MAX] {
        assert_eq!(direct.get(key), None);
        assert!(!direct.contains_key(key));
        assert_eq!(direct.remove(key), None);
        assert_eq!(direct.compare_exchange(key, None, None), Ok(None));
        assert_eq!(direct.compare_exchange(key, Some(1), None), Err(None));
        assert_eq!(direct.fetch_update(key, |_| Some(None)), Ok(None));
        assert_eq!(direct.fetch_update(key, |_| None), Err(None));
    }
    assert_eq!(direct.allocated_pages(), 0);
    let generational = AtomicIdMap::<1>::new();
    for index in [0, 7, u32::MAX] {
        let key = id(index, 1);
        assert_eq!(generational.get(key), None);
        assert!(!generational.contains_key(key));
        assert_eq!(generational.occupant(index), None);
        assert_eq!(generational.remove(key), None);
        assert_eq!(generational.remove_occupant(index), None);
        assert_eq!(generational.compare_exchange(key, None, None), Ok(None));
        assert_eq!(generational.fetch_update(key, |_| Some(None)), Ok(None));
    }
    assert_eq!(generational.allocated_pages(), 0);
}

#[test]
#[should_panic(expected = "key index exceeds direct spine range")]
fn out_of_range_insert_panics() {
    AtomicDirectMap::<4>::new().insert(usize::MAX, 0);
}

#[test]
fn conditional_generation_semantics_and_length() {
    let map = AtomicIdMap::<4>::new();
    let old = id(1, 1);
    let new = id(1, 2);
    map.insert(old, 0);
    assert_eq!(map.compare_exchange(new, Some(0), Some(10)), Err(None));
    assert_eq!(map.compare_exchange(new, None, None), Ok(None));
    assert_eq!(map.fetch_update(new, |_| Some(None)), Ok(None));
    assert_eq!(map.occupant(1), Some((old, 0)));
    assert_eq!(map.len(), 1);
    assert_eq!(map.compare_exchange(new, None, Some(u32::MAX)), Ok(None));
    assert_eq!(map.len(), 1);
    assert_eq!(map.remove(old), None);
    assert_eq!(map.fetch_update(old, |_| Some(None)), Ok(None));
    assert_eq!(map.occupant(1), Some((new, u32::MAX)));
    assert_eq!(map.fetch_update(old, |_| Some(Some(5))), Ok(None));
    assert_eq!(map.len(), 1);
    assert_eq!(map.compare_exchange(old, Some(5), None), Ok(Some(5)));
    assert_eq!(map.len(), 0);
    assert_eq!(map.fetch_update(new, |_| Some(Some(0))), Ok(None));
    assert_eq!(map.len(), 1);
    assert_eq!(map.remove_occupant(1), Some((new, 0)));
    assert_eq!(map.len(), 0);
    // Id::new accepts every nonzero u32 generation, not just allocator ids.
    let highest = id(1, u32::MAX);
    assert_eq!(map.insert(highest, u32::MAX), None);
    assert_eq!(map.iter().collect::<Vec<_>>(), [(highest, u32::MAX)]);
}

#[test]
fn callbacks_retry_after_reentrant_changes() {
    // This also covers page installation by a callback after an absent-page
    // observation, including a request that would otherwise be a no-op.
    let map = AtomicDirectMap::<4>::new();
    let mut calls = 0;
    assert_eq!(
        map.fetch_update(1, |old| {
            calls += 1;
            if calls == 1 {
                assert_eq!(old, None);
                map.insert(1, 7);
            } else {
                assert_eq!(old, Some(7));
            }
            Some(None)
        }),
        Ok(Some(7))
    );
    assert_eq!(calls, 2);
    assert_eq!(map.len(), 0);
    let map = AtomicIdMap::<4>::new();
    let key = id(1, 1);
    let other = id(1, 2);
    map.insert(key, 1);
    let mut calls = 0;
    assert_eq!(
        map.fetch_update(key, |old| {
            calls += 1;
            if calls == 1 {
                assert_eq!(old, Some(1));
                map.insert(other, 2);
            } else {
                assert_eq!(old, None);
            }
            Some(None)
        }),
        Ok(None)
    );
    assert_eq!(calls, 2);
    assert_eq!(map.occupant(1), Some((other, 2)));
    assert_eq!(map.len(), 1);
}

#[test]
fn traversal_bounds_clear_reset_and_collection_traits() {
    let mut direct: AtomicDirectMap<4> = [(9, MAX_VALUE), (0, 0), (4, 4)].into_iter().collect();
    direct.extend([(7, 7), (4, 5)]);
    assert_eq!(
        format!("{direct:?}"),
        format!("{{0: 0, 4: 5, 7: 7, 9: {MAX_VALUE}}}")
    );
    assert_eq!(direct.range(4..=7).collect::<Vec<_>>(), [(4, 5), (7, 7)]);
    assert_eq!(
        direct
            .range((Bound::Excluded(4), Bound::Included(9)))
            .collect::<Vec<_>>(),
        [(7, 7), (9, MAX_VALUE)]
    );
    for bounds in [
        (Bound::Excluded(usize::MAX), Bound::Unbounded),
        (Bound::Unbounded, Bound::Excluded(0)),
        (Bound::Included(9), Bound::Included(4)),
        (Bound::Included(usize::MAX), Bound::Unbounded),
    ] {
        assert_eq!(direct.range(bounds).next(), None);
    }
    let pages = direct.allocated_pages();
    assert_eq!(pages, 3);
    direct.clear();
    assert!(direct.is_empty());
    assert_eq!(direct.iter().next(), None);
    assert_eq!(direct.allocated_pages(), pages);
    direct.insert(4, 0);
    assert_eq!(direct.len(), 1);
    direct.reset();
    assert_eq!(direct.allocated_pages(), 0);
    assert!(direct.is_empty());
    direct.insert(4, MAX_VALUE);
    assert_eq!(direct.get(4), Some(MAX_VALUE));

    let mut generational: AtomicIdMap<4> = [(id(9, 1), 9), (id(0, 1), 0), (id(4, 1), 4)]
        .into_iter()
        .collect();
    generational.extend([(id(7, 1), 7), (id(4, 2), u32::MAX)]);
    assert_eq!(
        format!("{generational:?}"),
        format!(
            "{{Id(0v1): 0, Id(4v2): {}, Id(7v1): 7, Id(9v1): 9}}",
            u32::MAX
        )
    );
    assert_eq!(
        generational.range(4..=7).collect::<Vec<_>>(),
        [(id(4, 2), u32::MAX), (id(7, 1), 7)]
    );
    assert_eq!(
        generational
            .range((Bound::Excluded(4), Bound::Excluded(9)))
            .collect::<Vec<_>>(),
        [(id(7, 1), 7)]
    );
    for bounds in [
        (Bound::Excluded(u32::MAX), Bound::Unbounded),
        (Bound::Unbounded, Bound::Excluded(0)),
        (Bound::Included(9), Bound::Included(4)),
        (Bound::Included(u32::MAX), Bound::Unbounded),
    ] {
        assert_eq!(generational.range(bounds).next(), None);
    }
    generational.clear();
    assert!(generational.is_empty());
    assert_eq!(generational.iter().next(), None);
    assert_eq!(generational.allocated_pages(), 3);
    generational.insert(id(4, 3), 0);
    assert_eq!(generational.len(), 1);
    generational.reset();
    assert!(generational.is_empty());
    assert_eq!(generational.allocated_pages(), 0);
    generational.insert(id(4, 1), 4);
    assert_eq!(generational.get(id(4, 1)), Some(4));

    assert!(AtomicDirectMap::<4>::default().is_empty());
    assert!(AtomicIdMap::<4>::default().is_empty());
}

#[test]
fn invalid_page_capacities_panic() {
    assert!(catch_unwind(AtomicDirectMap::<0>::new).is_err());
    assert!(catch_unwind(AtomicIdMap::<0>::new).is_err());
    assert!(catch_unwind(AtomicDirectMap::<{ usize::MAX }>::new).is_err());
    assert!(catch_unwind(AtomicIdMap::<{ usize::MAX }>::new).is_err());
}

#[test]
fn eight_threads_increment_shared_counters() {
    let direct = Arc::new(AtomicDirectMap::<4>::new());
    let generational = Arc::new(AtomicIdMap::<4>::new());
    let start = Arc::new(Barrier::new(8));
    let workers: Vec<_> = (0..8)
        .map(|_| {
            let direct = direct.clone();
            let generational = generational.clone();
            let start = start.clone();
            thread::spawn(move || {
                start.wait();
                for step in 0..1000 {
                    let key = step % 4;
                    direct
                        .fetch_update(key, |old| Some(Some(old.unwrap_or(0) + 1)))
                        .unwrap();
                    generational
                        .fetch_update(id(key as u32, 1), |old| Some(Some(old.unwrap_or(0) + 1)))
                        .unwrap();
                }
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }
    for key in 0..4 {
        assert_eq!(direct.get(key), Some(2000));
        assert_eq!(generational.get(id(key as u32, 1)), Some(2000));
    }
    assert_eq!(direct.len(), 4);
    assert_eq!(generational.len(), 4);
}

#[test]
fn disjoint_inserts_and_removes_have_exact_length() {
    let direct = Arc::new(AtomicDirectMap::<4>::new());
    let generational = Arc::new(AtomicIdMap::<4>::new());
    let workers: Vec<_> = (0..8)
        .map(|worker| {
            let direct = direct.clone();
            let generational = generational.clone();
            thread::spawn(move || {
                for offset in 0..1000 {
                    let key = worker * 1000 + offset;
                    assert_eq!(direct.insert(key, key as u64), None);
                    assert_eq!(generational.insert(id(key as u32, 1), key as u32), None);
                    if offset % 2 == 0 {
                        assert_eq!(direct.remove(key), Some(key as u64));
                        assert_eq!(generational.remove(id(key as u32, 1)), Some(key as u32));
                    }
                }
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }
    assert_eq!(direct.len(), 4000);
    assert_eq!(generational.len(), 4000);
    assert_eq!(direct.iter().count(), direct.len());
    assert_eq!(generational.iter().count(), generational.len());
    assert!(
        direct
            .iter()
            .all(|(key, value)| key % 2 == 1 && value == key as u64)
    );
    assert!(
        generational
            .iter()
            .all(|(key, value)| key.index() % 2 == 1 && value == key.index())
    );
}

#[test]
fn pending_update_succeeds_across_a_full_word_aba() {
    // The CAS compares encoded words, so A -> empty -> A and
    // A -> other generation -> A during a pending update are invisible to
    // it: the update commits once, against the restored word, and every
    // intermediate transition keeps its own length delta.
    let map = AtomicDirectMap::<4>::new();
    map.insert(1, 5);
    let mut calls = 0;
    assert_eq!(
        map.fetch_update(1, |old| {
            calls += 1;
            assert_eq!(old, Some(5));
            if calls == 1 {
                assert_eq!(map.remove(1), Some(5));
                assert_eq!(map.len(), 0);
                assert_eq!(map.insert(1, 5), None);
            }
            Some(Some(6))
        }),
        Ok(Some(5))
    );
    assert_eq!(calls, 1);
    assert_eq!(map.get(1), Some(6));
    assert_eq!(map.len(), 1);

    let map = AtomicIdMap::<4>::new();
    let key = id(1, 1);
    let other = id(1, 2);
    map.insert(key, 5);
    let mut calls = 0;
    assert_eq!(
        map.fetch_update(key, |old| {
            calls += 1;
            assert_eq!(old, Some(5));
            if calls == 1 {
                assert_eq!(map.replace(other, 9), Some((key, 5)));
                assert_eq!(map.replace(key, 5), Some((other, 9)));
            }
            Some(Some(6))
        }),
        Ok(Some(5))
    );
    assert_eq!(calls, 1);
    assert_eq!(map.get(key), Some(6));
    assert_eq!(map.len(), 1);
}

#[test]
fn flat_rejects_wide_indices_while_radix_accepts_them() {
    let flat = AtomicDirectMap::<3>::new();
    assert_eq!(flat.get(usize::MAX), None);
    assert_eq!(flat.remove(usize::MAX), None);
    assert!(catch_unwind(AssertUnwindSafe(|| flat.insert(usize::MAX, 42))).is_err());
    let mut map = AtomicDirectMap::<3, Radix>::new();
    assert_eq!(map.insert(usize::MAX, 42), None);
    assert_eq!(map.get(usize::MAX), Some(42));
    assert_eq!(
        map.range(usize::MAX..=usize::MAX).collect::<Vec<_>>(),
        [(usize::MAX, 42)]
    );
    assert_eq!(
        map.range((Bound::Excluded(usize::MAX), Bound::Unbounded))
            .next(),
        None
    );
    assert_eq!(
        map.compare_exchange(usize::MAX, Some(42), Some(43)),
        Ok(Some(42))
    );
    let plain: PagedDirectMap<usize, u64, 3, Radix> = map.into_inner();
    map = AtomicDirectMap::from(plain);
    assert_eq!(map.remove(usize::MAX), Some(43));
    map.insert(usize::MAX - 1, 7);
    map.clear();
    assert!(map.is_empty());
    assert_eq!(map.iter().next(), None);
    map.reset();
    assert_eq!(map.allocated_pages(), 0);
}

#[test]
fn radix_scoped_insertions_span_the_index_space() {
    let map = AtomicDirectMap::<3, Radix>::new();
    let keys: Vec<_> = (0..usize::BITS)
        .map(|bit| 1usize << bit)
        .chain([0, usize::MAX - 1, usize::MAX])
        .collect();
    thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                for &key in &keys {
                    let value = (key as u64) >> 1;
                    map.insert(key, value);
                    assert_eq!(map.get(key), Some(value));
                }
            });
        }
    });
    assert_eq!(map.len(), keys.len());
    let map = AtomicDirectMap::from(map.into_inner());
    for key in keys {
        assert_eq!(map.get(key), Some((key as u64) >> 1));
    }
}
