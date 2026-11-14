//! Generational Entry operations against a slot-indexed BTreeMap oracle.

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;
use std::ops::Bound::{Excluded, Included, Unbounded};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;

use paged_collections::direct_map::{Entry, PagedDirectMap};
use paged_collections::key::{DirectKey, Key};
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

impl DirectKey for Gen {
    type Tag = u8;

    fn tag(self) -> Self::Tag {
        self.generation
    }

    fn from_parts(index: usize, tag: Self::Tag) -> Self {
        Self {
            index: index as u16,
            generation: tag,
        }
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
                prop_assert_eq!(entry.occupant().map(|(k, &v)| (k, v)), occupant);
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
        map.iter().map(|(k, &v)| (k, v)).collect::<Vec<_>>(),
        model.values().copied().collect::<Vec<_>>()
    );
    prop_assert_eq!(
        map.occupant(key.index()).map(|(k, &v)| (k, v)),
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
    check_order(map, model, key, |value| *value)?;
    Ok(())
}

fn check_order<K: DirectKey + Debug, V, const PAGE: usize, D: Directory>(
    map: &PagedDirectMap<K, V, PAGE, D>,
    model: &BTreeMap<usize, (K, u32)>,
    probe: K,
    payload: impl Fn(&V) -> u32,
) -> TestCaseResult {
    prop_assert_eq!(map.len(), model.len());
    prop_assert_eq!(map.is_empty(), model.is_empty());
    prop_assert_eq!(
        map.iter().map(|(k, v)| (k, payload(v))).collect::<Vec<_>>(),
        model.values().copied().collect::<Vec<_>>()
    );
    prop_assert_eq!(
        map.iter()
            .rev()
            .map(|(k, v)| (k, payload(v)))
            .collect::<Vec<_>>(),
        model.values().rev().copied().collect::<Vec<_>>()
    );
    prop_assert_eq!(
        map.first_key_value().map(|(k, v)| (k, payload(v))),
        model.first_key_value().map(|(_, &entry)| entry)
    );
    prop_assert_eq!(
        map.last_key_value().map(|(k, v)| (k, payload(v))),
        model.last_key_value().map(|(_, &entry)| entry)
    );
    let index = probe.index();
    for index in [
        0,
        index.saturating_sub(1),
        index,
        index.saturating_add(1),
        usize::MAX,
    ] {
        prop_assert_eq!(
            map.next_occupant(index).map(|(k, v)| (k, payload(v))),
            model.range(index..).next().map(|(_, &entry)| entry)
        );
        prop_assert_eq!(
            map.prev_occupant(index).map(|(k, v)| (k, payload(v))),
            model.range(..=index).next_back().map(|(_, &entry)| entry)
        );
    }
    for bounds in [
        (Unbounded, Unbounded),
        (Included(probe), Included(probe)),
        (Included(probe), Unbounded),
        (Excluded(probe), Unbounded),
        (Unbounded, Included(probe)),
        (Unbounded, Excluded(probe)),
    ] {
        // Bounds use only the slot index, even for an absent generation.
        let expected: Vec<_> = model
            .iter()
            .filter(|&(&index, _)| {
                let after_start = match bounds.0 {
                    Included(k) => index >= k.index(),
                    Excluded(k) => index > k.index(),
                    Unbounded => true,
                };
                let before_end = match bounds.1 {
                    Included(k) => index <= k.index(),
                    Excluded(k) => index < k.index(),
                    Unbounded => true,
                };
                after_start && before_end
            })
            .map(|(_, &entry)| entry)
            .collect();
        prop_assert_eq!(
            map.range(bounds)
                .map(|(k, v)| (k, payload(v)))
                .collect::<Vec<_>>(),
            expected.clone()
        );
        prop_assert_eq!(
            map.range(bounds)
                .rev()
                .map(|(k, v)| (k, payload(v)))
                .collect::<Vec<_>>(),
            expected.into_iter().rev().collect::<Vec<_>>()
        );
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
                prop_assert_eq!(entry.occupant().map(|(k, &v)| (k, v)), old);
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
                        prop_assert_eq!(entry.occupant().map(|(k, &v)| (k, v)), old);
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

fn panicking_modify<const PAGE: usize, D: Directory>() -> TestCaseResult {
    let key = Gen {
        index: 9,
        generation: 1,
    };
    for initial in INITIALS {
        let (mut map, mut model, pages) = initial_state::<PAGE, D>(initial, key, 10);
        let mut calls = 0;
        let result = catch_unwind(AssertUnwindSafe(|| {
            map.entry(key).and_modify(|value| {
                calls += 1;
                *value = 42;
                panic!("and_modify");
            });
        }));
        let occupied = matches!(initial, Initial::Exact);
        prop_assert_eq!(result.is_err(), occupied);
        prop_assert_eq!(calls, usize::from(occupied));
        if occupied {
            // The callback's completed mutation survives unwinding.
            model.insert(key.index(), (key, 42));
        }
        check_state(&map, &model, &pages, key)?;
        check_entry(&mut map, &model, key)?;
        // Prove the borrow was released and the slot can be reused.
        prop_assert_eq!(map.entry(key).insert_entry(99).remove_entry(), (key, 99));
        model.remove(&key.index());
        let mut pages = pages;
        pages.insert(key.index() / PAGE);
        check_state(&map, &model, &pages, key)?;
    }
    Ok(())
}

struct PanicDrop {
    id: usize,
    drops: Rc<Cell<[usize; 5]>>,
    panic_on_drop: bool,
}

impl Drop for PanicDrop {
    fn drop(&mut self) {
        let mut counts = self.drops.get();
        counts[self.id] += 1;
        self.drops.set(counts);
        if std::mem::take(&mut self.panic_on_drop) {
            panic!("drop {}", self.id);
        }
    }
}

fn panicking_entry_drops<const PAGE: usize, D: Directory>() -> TestCaseResult {
    // Both vacant insertion methods share eviction code; occupied
    // Entry::insert_entry drops its old value through a different path.
    for mode in 0..3 {
        let drops = Rc::new(Cell::new([0; 5]));
        let make = |id, panic_on_drop| PanicDrop {
            id,
            drops: Rc::clone(&drops),
            panic_on_drop,
        };
        let old = Gen {
            index: 3,
            generation: 0,
        };
        let new = Gen {
            generation: u8::from(mode != 2),
            ..old
        };
        let mut map = PagedDirectMap::<Gen, PanicDrop, PAGE, D>::new();
        let mut model = Model::new();
        let mut pages = BTreeSet::new();
        for (index, id) in [(0, 0), (3, 1), (9, 3)] {
            let key = Gen {
                index,
                generation: 0,
            };
            map.entry(key).or_insert_with(|| make(id, id == 1));
            model.insert(key.index(), (key, id as u32));
            pages.insert(key.index() / PAGE);
        }
        let result = catch_unwind(AssertUnwindSafe(|| match mode {
            0 | 1 => {
                let Entry::Vacant(entry) = map.entry(new) else {
                    panic!("another generation must be vacant");
                };
                assert_eq!(entry.occupant().map(|(k, v)| (k, v.id)), Some((old, 1)));
                if mode == 0 {
                    entry.insert(make(2, false));
                } else {
                    entry.insert_entry(make(2, false));
                }
            }
            _ => {
                assert!(matches!(map.entry(new), Entry::Occupied(_)));
                map.entry(new).insert_entry(make(2, false));
            }
        }));
        prop_assert!(result.is_err());
        // This also ensures an assertion inside the catch was not mistaken
        // for the intended destructor panic.
        prop_assert_eq!(drops.get(), [0, 1, 0, 0, 0]);
        model.insert(new.index(), (new, 2));
        prop_assert_eq!(map.allocated_pages(), pages.len());
        prop_assert_eq!(map.occupant(3).map(|(k, v)| (k, v.id)), Some((new, 2)));
        if old != new {
            prop_assert!(map.get(old).is_none());
        }
        check_order(&map, &model, old, |value| value.id as u32)?;

        let Entry::Occupied(entry) = map.entry(new) else {
            return Err(TestCaseError::fail("replacement missing after drop panic"));
        };
        let (key, removed) = entry.remove_entry();
        prop_assert_eq!((key, removed.id), (new, 2));
        prop_assert_eq!(drops.get(), [0, 1, 0, 0, 0]);
        model.remove(&new.index());
        prop_assert!(map.occupant(3).is_none());
        prop_assert_eq!(map.allocated_pages(), pages.len());
        check_order(&map, &model, new, |value| value.id as u32)?;
        drop(removed);
        prop_assert_eq!(drops.get(), [0, 1, 1, 0, 0]);

        map.shrink_to_fit();
        pages = model.keys().map(|index| index / PAGE).collect();
        prop_assert_eq!(map.allocated_pages(), pages.len());
        let reused = Gen {
            generation: 2,
            ..old
        };
        let entry = map.entry(reused).insert_entry(make(4, false));
        prop_assert_eq!((*entry.key(), entry.get().id), (reused, 4));
        model.insert(reused.index(), (reused, 4));
        pages.insert(reused.index() / PAGE);
        prop_assert_eq!(map.allocated_pages(), pages.len());
        check_order(&map, &model, reused, |value| value.id as u32)?;
        let Entry::Occupied(entry) = map.entry(reused) else {
            return Err(TestCaseError::fail("reused entry missing"));
        };
        drop(entry.remove());
        model.remove(&reused.index());
        prop_assert!(map.occupant(3).is_none());
        prop_assert_eq!(map.allocated_pages(), pages.len());
        check_order(&map, &model, reused, |value| value.id as u32)?;
        drop(map);
        prop_assert_eq!(drops.get(), [1; 5]);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct Tagged {
    identity: Gen,
    tag: u8,
}

impl PartialEq for Tagged {
    fn eq(&self, other: &Self) -> bool {
        self.identity == other.identity
    }
}

impl Eq for Tagged {}

impl Key for Tagged {
    fn index(self) -> usize {
        self.identity.index()
    }
}

impl DirectKey for Tagged {
    type Tag = Self;

    fn tag(self) -> Self {
        self
    }

    fn from_parts(_: usize, key: Self) -> Self {
        key
    }
}

fn occupied_keeps_stored_key<const PAGE: usize, D: Directory>() {
    let stored = Tagged {
        identity: Gen {
            index: 9,
            generation: 1,
        },
        tag: 1,
    };
    let equal = Tagged { tag: 2, ..stored };
    let mut map = PagedDirectMap::<Tagged, u32, PAGE, D>::new();
    map.entry(stored).or_insert(10);
    let pages = map.allocated_pages();
    let entry = map.entry(equal);
    assert_eq!(entry.key().tag, 1);
    let Entry::Occupied(mut entry) = entry else {
        panic!("equal key must be occupied");
    };
    assert_eq!(entry.key().tag, 1);
    assert_eq!(entry.insert(20), 10);
    assert_eq!(entry.key().tag, 1);
    assert_eq!(*entry.into_mut(), 20);
    let mut calls = 0;
    assert_eq!(
        *map.entry(equal).or_insert_with_key(|_| {
            calls += 1;
            999
        }),
        20
    );
    assert_eq!(calls, 0);
    let entry = map.entry(equal).insert_entry(30);
    assert_eq!(entry.key().tag, 1);
    let (removed, value) = entry.remove_entry();
    assert_eq!(
        (removed.identity, removed.tag, value),
        (stored.identity, 1, 30)
    );
    assert_eq!(map.len(), 0);
    assert_eq!(map.allocated_pages(), pages);
    assert!(map.occupant(stored.index()).is_none());
    // Once removed, the new insertion really does store the supplied tag.
    let entry = map.entry(equal).insert_entry(40);
    assert_eq!((entry.key().tag, *entry.get()), (2, 40));
    assert_eq!(map.len(), 1);
    assert_eq!(map.allocated_pages(), pages);
    assert_eq!(map.first_key_value().unwrap().0.tag, 2);
}

macro_rules! suite {
    ($name:ident, $page:literal, $directory:ty) => {
        mod $name {
            use super::*;

            #[test]
            fn destructor_panics_commit_replacements_and_allow_reuse() {
                panicking_entry_drops::<$page, $directory>().unwrap();
            }

            #[test]
            fn occupied_entries_preserve_the_stored_key_tag() {
                occupied_keeps_stored_key::<$page, $directory>();
            }

            #[test]
            fn and_modify_panics_preserve_bookkeeping() {
                panicking_modify::<$page, $directory>().unwrap();
            }

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Wide {
    index: usize,
    generation: u8,
}

impl Key for Wide {
    fn index(self) -> usize {
        self.index
    }
}

impl DirectKey for Wide {
    type Tag = u8;

    fn tag(self) -> Self::Tag {
        self.generation
    }

    fn from_parts(index: usize, tag: Self::Tag) -> Self {
        Self {
            index,
            generation: tag,
        }
    }
}

fn check_wide_state<const PAGE: usize, D: Directory>(
    map: &PagedDirectMap<Wide, u32, PAGE, D>,
    model: &BTreeMap<usize, (Wide, u32)>,
    pages: &BTreeSet<usize>,
    key: Wide,
) -> TestCaseResult {
    prop_assert_eq!(map.allocated_pages(), pages.len());
    prop_assert_eq!(
        map.occupant(key.index).map(|(k, &v)| (k, v)),
        model.get(&key.index).copied()
    );
    for generation in 0..4 {
        let probe = Wide { generation, ..key };
        let expected = model
            .get(&key.index)
            .filter(|(stored, _)| *stored == probe)
            .map(|(_, value)| *value);
        prop_assert_eq!(map.get(probe).copied(), expected);
    }
    check_order(map, model, key, |value| *value)
}

fn wide_entries<const PAGE: usize, D: Directory>(
    random: usize,
    rotate: usize,
    reverse: bool,
) -> TestCaseResult {
    let mut indices = vec![
        0,
        1,
        63,
        64,
        1 << 12,
        1 << (usize::BITS - 1),
        usize::MAX - 1,
        usize::MAX,
        random,
    ];
    indices.sort_unstable();
    indices.dedup();
    let offset = rotate % indices.len();
    indices.rotate_left(offset);
    if reverse {
        indices.reverse();
    }
    let mut map = PagedDirectMap::<Wide, u32, PAGE, D>::new();
    let mut model = BTreeMap::new();
    let mut pages = BTreeSet::new();
    for &index in &indices {
        let key = Wide {
            index,
            generation: 0,
        };
        // Inspection at a wide, previously unallocated page must stay lazy.
        let before = map.allocated_pages();
        let Entry::Vacant(entry) = map.entry(key) else {
            return Err(TestCaseError::fail("fresh wide key was occupied"));
        };
        prop_assert!(entry.occupant().is_none());
        prop_assert_eq!(entry.into_key(), key);
        prop_assert_eq!(map.allocated_pages(), before);
        let value = index as u32;
        let entry = map.entry(key).insert_entry(value);
        prop_assert_eq!((*entry.key(), *entry.get()), (key, value));
        model.insert(index, (key, value));
        pages.insert(index / PAGE);
        check_wide_state(&map, &model, &pages, key)?;
    }
    for &index in &indices {
        let old = Wide {
            index,
            generation: 0,
        };
        let new = Wide {
            generation: 1,
            ..old
        };
        let value = (index as u32).wrapping_add(1);
        let Entry::Vacant(entry) = map.entry(new) else {
            return Err(TestCaseError::fail("new wide generation was occupied"));
        };
        prop_assert_eq!(entry.occupant().map(|(k, &v)| (k, v)), Some(model[&index]));
        prop_assert_eq!(*entry.insert(value), value);
        model.insert(index, (new, value));
        // Query with the evicted generation: bounds still use only its index.
        check_wide_state(&map, &model, &pages, old)?;
    }
    for &index in &indices {
        let key = Wide {
            index,
            generation: 1,
        };
        let Entry::Occupied(entry) = map.entry(key) else {
            return Err(TestCaseError::fail("wide replacement missing"));
        };
        prop_assert_eq!(Some(entry.remove_entry()), model.remove(&index));
        check_wide_state(&map, &model, &pages, key)?;
        // Force directory branch collapse where the emptied page is unique.
        map.shrink_to_fit();
        pages = model.keys().map(|index| index / PAGE).collect();
        check_wide_state(&map, &model, &pages, key)?;

        let reused = Wide {
            generation: 2,
            ..key
        };
        let entry = map.entry(reused).insert_entry(77);
        prop_assert_eq!((*entry.key(), *entry.get()), (reused, 77));
        model.insert(index, (reused, 77));
        pages.insert(index / PAGE);
        check_wide_state(&map, &model, &pages, reused)?;
        prop_assert_eq!(
            map.entry(reused).insert_entry(78).remove_entry(),
            (reused, 78)
        );
        model.remove(&index);
        check_wide_state(&map, &model, &pages, reused)?;
        map.shrink_to_fit();
        pages = model.keys().map(|index| index / PAGE).collect();
        check_wide_state(&map, &model, &pages, reused)?;
    }
    prop_assert!(map.is_empty());
    prop_assert_eq!(map.allocated_pages(), 0);
    let last = Wide {
        index: usize::MAX,
        generation: 3,
    };
    prop_assert_eq!(*map.entry(last).or_insert(88), 88);
    model.insert(last.index, (last, 88));
    pages.insert(last.index / PAGE);
    check_wide_state(&map, &model, &pages, last)
}

macro_rules! wide_suite {
    ($name:ident, $page:literal, $directory:ty) => {
        mod $name {
            use super::*;

            proptest! {
                #![proptest_config(config())]

                #[test]
                fn wide_entry_mutations_preserve_order_and_pages(
                    random in any::<usize>(), rotate in any::<usize>(), reverse in any::<bool>(),
                ) {
                    wide_entries::<$page, $directory>(random, rotate, reverse)?;
                }
            }
        }
    };
}

wide_suite!(wide_tree_1, 1, Tree);
wide_suite!(wide_tree_3, 3, Tree);
wide_suite!(wide_radix_1, 1, Radix);
wide_suite!(wide_radix_3, 3, Radix);
