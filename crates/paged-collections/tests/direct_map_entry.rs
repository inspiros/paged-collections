//! Generational Entry operations against a slot-indexed BTreeMap oracle.

use std::collections::{BTreeMap, BTreeSet};
use std::panic::{AssertUnwindSafe, catch_unwind};

use paged_collections::direct_map::{Entry, PagedDirectMap};
use paged_collections::key::Key;
use paged_collections::page::{Directory, Flat, Radix, Tree};
use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult};

fn config() -> ProptestConfig {
    if cfg!(miri) {
        ProptestConfig {
            cases: 4,
            failure_persistence: None,
            ..ProptestConfig::default()
        }
    } else {
        ProptestConfig {
            cases: 512,
            ..ProptestConfig::default()
        }
    }
}

fn max_len() -> usize {
    if cfg!(miri) { 24 } else { 200 }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Gen {
    index: u16,
    generation: u8,
}

impl Key for Gen {
    fn index(self) -> usize {
        usize::from(self.index)
    }
}

fn keys() -> BoxedStrategy<Gen> {
    let indices = prop_oneof![6 => 0_u16..12, 2 => 23_u16..27, 2 => 1022_u16..1027];
    (indices, 0_u8..4)
        .prop_map(|(index, generation)| Gen { index, generation })
        .boxed()
}

type Model = BTreeMap<usize, (Gen, u32)>;

#[derive(Clone, Copy, Debug)]
enum Op {
    Inspect,
    OccupiedMut,
    Remove,
    RemoveEntry,
    VacantInsert,
    VacantInsertEntry,
    OrInsert,
    OrInsertWith,
    OrInsertWithKey,
    OrDefault,
    AndModify,
    AndModifyOrInsert,
    InsertEntry,
}

const OPS: [Op; 13] = [
    Op::Inspect,
    Op::OccupiedMut,
    Op::Remove,
    Op::RemoveEntry,
    Op::VacantInsert,
    Op::VacantInsertEntry,
    Op::OrInsert,
    Op::OrInsertWith,
    Op::OrInsertWithKey,
    Op::OrDefault,
    Op::AndModify,
    Op::AndModifyOrInsert,
    Op::InsertEntry,
];

#[derive(Clone, Copy, Debug)]
struct Step {
    op: Op,
    key: Gen,
    value: u32,
    delta: u32,
}

fn steps() -> impl Strategy<Value = Step> {
    (
        prop::sample::select(OPS.to_vec()),
        keys(),
        any::<u32>(),
        any::<u32>(),
    )
        .prop_map(|(op, key, value, delta)| Step {
            op,
            key,
            value,
            delta,
        })
}

fn keyed_value(key: Gen, value: u32) -> u32 {
    value
        .wrapping_add(u32::from(key.index))
        .wrapping_add(u32::from(key.generation) << 16)
}

fn check_entry<const PAGE: usize, D: Directory>(
    map: &mut PagedDirectMap<Gen, u32, PAGE, D>,
    model: &Model,
    key: Gen,
) -> TestCaseResult {
    let pages = map.allocated_pages();
    let len = map.len();
    let occupant = model.get(&key.index()).copied();
    {
        let entry = map.entry(key);
        prop_assert_eq!(*entry.key(), key);
        match entry {
            Entry::Occupied(entry) => {
                prop_assert_eq!(occupant, Some((key, *entry.get())));
                prop_assert_eq!(*entry.key(), key);
            }
            Entry::Vacant(entry) => {
                prop_assert!(occupant.is_none_or(|(stored, _)| stored != key));
                prop_assert_eq!(*entry.key(), key);
                prop_assert_eq!(entry.occupant().map(|(&k, &v)| (k, v)), occupant);
                prop_assert_eq!(entry.into_key(), key);
            }
        }
    }
    prop_assert_eq!(map.allocated_pages(), pages, "inspection allocated a page");
    prop_assert_eq!(map.len(), len);
    Ok(())
}

fn check_state<const PAGE: usize, D: Directory>(
    map: &PagedDirectMap<Gen, u32, PAGE, D>,
    model: &Model,
    pages: &BTreeSet<usize>,
    key: Gen,
) -> TestCaseResult {
    prop_assert_eq!(map.len(), model.len());
    prop_assert_eq!(map.is_empty(), model.is_empty());
    prop_assert_eq!(map.allocated_pages(), pages.len());
    prop_assert_eq!(
        map.iter().map(|(&k, &v)| (k, v)).collect::<Vec<_>>(),
        model.values().copied().collect::<Vec<_>>()
    );
    prop_assert_eq!(
        map.occupant(key.index()).map(|(&k, &v)| (k, v)),
        model.get(&key.index()).copied()
    );
    for generation in 0..4 {
        let probe = Gen { generation, ..key };
        let expected = model
            .get(&probe.index())
            .filter(|(stored, _)| *stored == probe)
            .map(|(_, value)| *value);
        prop_assert_eq!(map.get(probe).copied(), expected);
        prop_assert_eq!(map.contains_key(probe), expected.is_some());
    }
    Ok(())
}

fn apply_step<const PAGE: usize, D: Directory>(
    map: &mut PagedDirectMap<Gen, u32, PAGE, D>,
    model: &mut Model,
    pages: &mut BTreeSet<usize>,
    step: Step,
) -> TestCaseResult {
    let Step {
        op,
        key,
        value,
        delta,
    } = step;
    let old = model.get(&key.index()).copied();
    let exact = old.filter(|(stored, _)| *stored == key).map(|(_, v)| v);
    check_entry(map, model, key)?;
    match op {
        Op::Inspect => {}
        Op::OccupiedMut => {
            if let Entry::Occupied(mut entry) = map.entry(key) {
                let previous = exact.unwrap();
                prop_assert_eq!(*entry.key(), key);
                prop_assert_eq!(*entry.get(), previous);
                *entry.get_mut() = previous.wrapping_add(delta);
                prop_assert_eq!(entry.insert(value), previous.wrapping_add(delta));
                prop_assert_eq!(*entry.key(), key);
                prop_assert_eq!(*entry.get(), value);
                *entry.into_mut() ^= delta;
                model.insert(key.index(), (key, value ^ delta));
            }
        }
        Op::Remove | Op::RemoveEntry => {
            if let Entry::Occupied(entry) = map.entry(key) {
                if matches!(op, Op::Remove) {
                    prop_assert_eq!(entry.remove(), exact.unwrap());
                } else {
                    prop_assert_eq!(entry.remove_entry(), (key, exact.unwrap()));
                }
                model.remove(&key.index());
            }
        }
        Op::VacantInsert | Op::VacantInsertEntry => {
            if let Entry::Vacant(entry) = map.entry(key) {
                prop_assert_eq!(entry.occupant().map(|(&k, &v)| (k, v)), old);
                if matches!(op, Op::VacantInsert) {
                    let stored = entry.insert(value);
                    prop_assert_eq!(*stored, value);
                    *stored ^= delta;
                } else {
                    let mut stored = entry.insert_entry(value);
                    prop_assert_eq!(*stored.key(), key);
                    prop_assert_eq!(*stored.get(), value);
                    *stored.get_mut() ^= delta;
                }
                model.insert(key.index(), (key, value ^ delta));
                pages.insert(key.index() / PAGE);
            }
        }
        Op::OrInsert | Op::OrInsertWith | Op::OrInsertWithKey | Op::OrDefault => {
            let default = match op {
                Op::OrDefault => 0,
                Op::OrInsertWithKey => keyed_value(key, value),
                _ => value,
            };
            let mut calls = 0;
            let mut seen_key = None;
            let stored = match op {
                Op::OrInsert => map.entry(key).or_insert(value),
                Op::OrInsertWith => map.entry(key).or_insert_with(|| {
                    calls += 1;
                    value
                }),
                Op::OrInsertWithKey => map.entry(key).or_insert_with_key(|&k| {
                    calls += 1;
                    seen_key = Some(k);
                    keyed_value(k, value)
                }),
                Op::OrDefault => map.entry(key).or_default(),
                _ => unreachable!(),
            };
            let expected = exact.unwrap_or(default);
            prop_assert_eq!(*stored, expected);
            *stored ^= delta;
            if matches!(op, Op::OrInsertWith | Op::OrInsertWithKey) {
                prop_assert_eq!(calls, usize::from(exact.is_none()));
            }
            if matches!(op, Op::OrInsertWithKey) {
                prop_assert_eq!(seen_key, exact.is_none().then_some(key));
            }
            model.insert(key.index(), (key, expected ^ delta));
            pages.insert(key.index() / PAGE);
        }
        Op::AndModify | Op::AndModifyOrInsert => {
            let mut calls = 0;
            let entry = map.entry(key).and_modify(|stored| {
                calls += 1;
                *stored = stored.wrapping_add(delta);
            });
            prop_assert_eq!(*entry.key(), key);
            prop_assert_eq!(calls, usize::from(exact.is_some()));
            if let Some(previous) = exact {
                model.insert(key.index(), (key, previous.wrapping_add(delta)));
            }
            if matches!(op, Op::AndModifyOrInsert) {
                let expected = exact.map_or(value, |v| v.wrapping_add(delta));
                prop_assert_eq!(*entry.or_insert(value), expected);
                model.insert(key.index(), (key, expected));
                pages.insert(key.index() / PAGE);
            } else {
                match entry {
                    Entry::Occupied(entry) => {
                        prop_assert_eq!(*entry.get(), exact.unwrap().wrapping_add(delta));
                    }
                    Entry::Vacant(entry) => {
                        prop_assert!(exact.is_none());
                        prop_assert_eq!(entry.occupant().map(|(&k, &v)| (k, v)), old);
                    }
                }
            }
        }
        Op::InsertEntry => {
            let entry = map.entry(key).insert_entry(value);
            prop_assert_eq!(*entry.key(), key);
            prop_assert_eq!(*entry.get(), value);
            *entry.into_mut() ^= delta;
            model.insert(key.index(), (key, value ^ delta));
            pages.insert(key.index() / PAGE);
        }
    }
    check_state(map, model, pages, key)?;
    // Inspect every generation after writes/removals, including the evicted key.
    for generation in 0..4 {
        check_entry(map, model, Gen { generation, ..key })?;
    }
    Ok(())
}

fn run_sequence<const PAGE: usize, D: Directory>(steps: &[Step]) -> TestCaseResult {
    let mut map = PagedDirectMap::<Gen, u32, PAGE, D>::new();
    let mut model = Model::new();
    let mut pages = BTreeSet::new();
    for (index, &step) in steps.iter().enumerate() {
        apply_step(&mut map, &mut model, &mut pages, step)
            .map_err(|error| TestCaseError::fail(format!("step {index}: {step:?}: {error}")))?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum Initial {
    Empty,
    Exact,
    OtherGeneration,
    AllocatedEmpty,
}

const INITIALS: [Initial; 4] = [
    Initial::Empty,
    Initial::Exact,
    Initial::OtherGeneration,
    Initial::AllocatedEmpty,
];

fn initial_state<const PAGE: usize, D: Directory>(
    initial: Initial,
    key: Gen,
    value: u32,
) -> (PagedDirectMap<Gen, u32, PAGE, D>, Model, BTreeSet<usize>) {
    let mut map = PagedDirectMap::<Gen, u32, PAGE, D>::new();
    let mut model = Model::new();
    let mut pages = BTreeSet::new();
    // Keep unrelated pages populated while leaving the target's page
    // unallocated, so inspection and non-inserting operations must not add it.
    for index in [0, PAGE - 1, PAGE, PAGE + 1, 3 * PAGE] {
        if index / PAGE == key.index() / PAGE {
            continue;
        }
        let neighbour = Gen {
            index: index as u16,
            generation: 0,
        };
        map.insert(neighbour, 17);
        model.insert(index, (neighbour, 17));
        pages.insert(index / PAGE);
    }
    match initial {
        Initial::Empty => {}
        Initial::Exact | Initial::OtherGeneration | Initial::AllocatedEmpty => {
            let stored = if matches!(initial, Initial::OtherGeneration) {
                Gen {
                    generation: (key.generation + 1) % 4,
                    ..key
                }
            } else {
                key
            };
            map.insert(stored, value);
            pages.insert(key.index() / PAGE);
            if matches!(initial, Initial::AllocatedEmpty) {
                assert_eq!(map.remove(key), Some(value));
            } else {
                model.insert(key.index(), (stored, value));
            }
        }
    }
    (map, model, pages)
}

fn every_entry_state<const PAGE: usize, D: Directory>(
    key: Gen,
    old: u32,
    value: u32,
    delta: u32,
) -> TestCaseResult {
    for initial in INITIALS {
        for op in OPS {
            let (mut map, mut model, mut pages) = initial_state::<PAGE, D>(initial, key, old);
            check_state(&map, &model, &pages, key)?;
            apply_step(
                &mut map,
                &mut model,
                &mut pages,
                Step {
                    op,
                    key,
                    value,
                    delta,
                },
            )
            .map_err(|error| TestCaseError::fail(format!("{initial:?}, {op:?}: {error}")))?;
        }
    }
    Ok(())
}

fn panicking_factories<const PAGE: usize, D: Directory>(key: Gen, value: u32) -> TestCaseResult {
    for initial in INITIALS {
        for with_key in [false, true] {
            let (mut map, model, pages) = initial_state::<PAGE, D>(initial, key, value);
            let mut calls = 0;
            let result = catch_unwind(AssertUnwindSafe(|| {
                if with_key {
                    map.entry(key).or_insert_with_key(|&k| {
                        calls += 1;
                        assert_eq!(k, key);
                        panic!("make");
                    });
                } else {
                    map.entry(key).or_insert_with(|| {
                        calls += 1;
                        panic!("make");
                    });
                }
            }));
            let vacant = !matches!(initial, Initial::Exact);
            prop_assert_eq!(result.is_err(), vacant);
            prop_assert_eq!(calls, usize::from(vacant));
            check_state(&map, &model, &pages, key)?;
            check_entry(&mut map, &model, key)?;
        }
    }
    Ok(())
}

macro_rules! suite {
    ($name:ident, $page:literal, $directory:ty) => {
        mod $name {
            use super::*;

            proptest! {
                #![proptest_config(config())]

                #[test]
                fn entry_model(ops in prop::collection::vec(steps(), 0..max_len())) {
                    run_sequence::<$page, $directory>(&ops)?;
                }

                #[test]
                fn all_operations_in_each_entry_state(
                    key in keys(), old in any::<u32>(), value in any::<u32>(), delta in any::<u32>(),
                ) {
                    every_entry_state::<$page, $directory>(key, old, value, delta)?;
                }

                #[test]
                fn factory_panics_preserve_entries_and_pages(key in keys(), value in any::<u32>()) {
                    panicking_factories::<$page, $directory>(key, value)?;
                }
            }
        }
    };
}

suite!(flat_1, 1, Flat);
suite!(flat_3, 3, Flat);
suite!(flat_8, 8, Flat);
suite!(tree_1, 1, Tree);
suite!(radix_1, 1, Radix);
suite!(tree_3, 3, Tree);
suite!(radix_3, 3, Radix);
suite!(tree_8, 8, Tree);
suite!(radix_8, 8, Radix);
