//! Stable dense-order sorting, slice search oracles, and fallible hive insertion.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;

use paged_collections::key::Key;
use paged_collections::page::{Directory, Flat, Tree};
use paged_collections::sparse_map::PagedSparseMap;
use paged_collections::sparse_set::PagedSparseSet;
use proptest::prelude::*;
use proptest::test_runner::TestCaseResult;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Gen(u32);

impl Key for Gen {
    fn index(self) -> usize {
        (self.0 & 0xffff) as usize
    }
}

trait TestKey: Key + Ord + Debug {
    fn make(index: u16, generation: u8) -> Self;
}

impl TestKey for Gen {
    fn make(index: u16, generation: u8) -> Self {
        Self(u32::from(index) | (u32::from(generation) << 16))
    }
}

#[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
impl TestKey for u32 {
    fn make(index: u16, _: u8) -> Self {
        Self::from(index)
    }
}

fn keys<K: TestKey + 'static>() -> BoxedStrategy<K> {
    let indices = prop_oneof![
        6 => 1_u16..48,
        2 => 1022_u16..1028,
        2 => 2047_u16..2052,
    ];
    (indices, 0_u8..4)
        .prop_map(|(index, generation)| K::make(index, generation))
        .boxed()
}

fn stale_keys<K: TestKey>() -> Vec<K> {
    (0..64)
        .chain(1021..1029)
        .chain(2046..2053)
        .map(|index| K::make(index, 255))
        .collect()
}

fn probes<K: TestKey>(entries: &[(K, u32)], random: &[K]) -> Vec<K> {
    let mut probes = stale_keys();
    probes.extend_from_slice(random);
    for &(key, _) in entries {
        probes.push(key);
        for generation in [0, 1, 2, 3, 254, 255] {
            probes.push(K::make(key.index() as u16, generation));
        }
    }
    probes.extend([K::make(0, 0), K::make(u16::MAX, 255)]);
    probes
}

fn make_set<K: TestKey, const PAGE: usize, D: Directory>(
    entries: &[(K, u32)],
) -> PagedSparseSet<K, PAGE, D> {
    let mut set = PagedSparseSet::new();
    for key in stale_keys() {
        set.insert(key);
    }
    set.clear();
    for &(key, _) in entries {
        set.insert(key);
    }
    set
}

fn make_map<K: TestKey, const PAGE: usize, D: Directory>(
    entries: &[(K, u32)],
) -> PagedSparseMap<K, u32, PAGE, D> {
    let mut map = PagedSparseMap::new();
    for key in stale_keys() {
        map.insert(key, 0);
    }
    map.clear();
    for &(key, value) in entries {
        map.insert(key, value);
    }
    map
}

fn check_set<K: TestKey, const PAGE: usize, D: Directory>(
    set: &PagedSparseSet<K, PAGE, D>,
    expected: &[K],
    probes: &[K],
) -> TestCaseResult {
    prop_assert_eq!(set.as_slice(), expected);
    prop_assert_eq!(set.len(), expected.len());
    prop_assert_eq!(
        expected.iter().copied().collect::<BTreeSet<_>>().len(),
        expected.len()
    );
    for (pos, &key) in expected.iter().enumerate() {
        prop_assert_eq!(set.position(key), Some(pos));
        prop_assert!(set.contains(key));
        prop_assert_eq!(set.occupant(key.index()).copied(), Some(key));
    }
    for &key in probes {
        let pos = expected.iter().position(|&stored| stored == key);
        prop_assert_eq!(set.position(key), pos, "probe {:?}", key);
        prop_assert_eq!(set.contains(key), pos.is_some(), "probe {:?}", key);
        prop_assert_eq!(
            set.occupant(key.index()).copied(),
            expected
                .iter()
                .find(|stored| stored.index() == key.index())
                .copied()
        );
    }
    Ok(())
}

fn check_map<K: TestKey, const PAGE: usize, D: Directory>(
    map: &PagedSparseMap<K, u32, PAGE, D>,
    expected: &[(K, u32)],
    probes: &[K],
) -> TestCaseResult {
    let keys: Vec<_> = expected.iter().map(|&(k, _)| k).collect();
    let values: Vec<_> = expected.iter().map(|&(_, v)| v).collect();
    prop_assert_eq!(map.len(), expected.len());
    prop_assert_eq!(map.key_slice(), &keys);
    prop_assert_eq!(map.value_slice(), &values);
    prop_assert_eq!(
        map.iter().map(|(&k, &v)| (k, v)).collect::<Vec<_>>(),
        expected
    );
    check_set(map.as_set(), &keys, probes)?;
    for &key in probes.iter().chain(keys.iter()) {
        let entry = expected.iter().find(|&&(stored, _)| stored == key).copied();
        prop_assert_eq!(map.get(key).copied(), entry.map(|(_, v)| v));
        prop_assert_eq!(map.contains_key(key), entry.is_some());
        prop_assert_eq!(map.get_key_value(key).map(|(&k, &v)| (k, v)), entry);
    }
    Ok(())
}

fn entry_rank<K: Key>(key: K, value: u32) -> u32 {
    ((key.index() % 3) as u32 + value % 3) % 3
}

fn cached_rank<K: Key>(key: K, value: u32) -> u32 {
    ((key.index() % 5) as u32 + value % 2) % 3
}

fn check_searches<K: TestKey, const PAGE: usize, D: Directory>(
    map: &PagedSparseMap<K, u32, PAGE, D>,
    probes: &[K],
) -> TestCaseResult {
    let keys = map.key_slice();
    let pairs: BTreeMap<_, _> = map.iter().map(|(&k, &v)| (k, v)).collect();
    for probe in probes.iter().chain(keys.iter()) {
        prop_assert_eq!(map.binary_search_key(probe), keys.binary_search(probe));
        prop_assert_eq!(
            map.binary_search_by(|key, value| {
                assert_eq!(Some(value), pairs.get(key));
                key.cmp(probe)
            }),
            keys.binary_search_by(|key| key.cmp(probe))
        );
        prop_assert_eq!(
            map.partition_point(|key, value| {
                assert_eq!(Some(value), pairs.get(key));
                key < probe
            }),
            keys.partition_point(|key| key < probe)
        );
    }
    // Every possible true prefix, including always false and always true.
    let positions: BTreeMap<_, _> = keys.iter().enumerate().map(|(pos, &k)| (k, pos)).collect();
    for prefix in 0..=keys.len() {
        let expected = keys.partition_point(|key| positions[key] < prefix);
        prop_assert_eq!(expected, prefix);
        prop_assert_eq!(
            map.partition_point(|key, value| {
                assert_eq!(Some(value), pairs.get(key));
                positions[key] < prefix
            }),
            expected
        );
    }
    Ok(())
}

fn run_sorts<K: TestKey, const PAGE: usize, D: Directory>(
    entries: &[(K, u32)],
    random_probes: &[K],
) -> TestCaseResult {
    let probes = probes(entries, random_probes);
    let mut set = make_set::<K, PAGE, D>(entries);
    let mut expected_set = set.as_slice().to_vec();
    check_set(&set, &expected_set, &probes)?;

    expected_set.sort_by_key(|key| Reverse(key.index() % 3));
    set.sort_by(|a, b| (b.index() % 3).cmp(&(a.index() % 3)));
    check_set(&set, &expected_set, &probes)?;

    expected_set.sort_by_key(|key| key.index() % 3);
    set.sort_by_key(|key| key.index() % 3);
    check_set(&set, &expected_set, &probes)?;

    expected_set.sort();
    set.sort();
    check_set(&set, &expected_set, &probes)?;

    let mut map = make_map::<K, PAGE, D>(entries);
    let mut expected_map: Vec<_> = map.iter().map(|(&k, &v)| (k, v)).collect();
    check_map(&map, &expected_map, &probes)?;

    expected_map.sort_by_key(|&(key, _)| Reverse(key.index() % 3));
    map.sort_by(|(a, _), (b, _)| (b.index() % 3).cmp(&(a.index() % 3)));
    check_map(&map, &expected_map, &probes)?;

    expected_map.sort_by_key(|&(key, value)| entry_rank(key, value));
    map.sort_by_key(|&key, &value| entry_rank(key, value));
    check_map(&map, &expected_map, &probes)?;

    expected_map.sort_by_key(|&(key, value)| cached_rank(key, value));
    let mut calls = BTreeMap::<K, usize>::new();
    map.sort_by_cached_key(|&key, &value| {
        *calls.entry(key).or_default() += 1;
        cached_rank(key, value)
    });
    check_map(&map, &expected_map, &probes)?;
    // With fewer than two entries there is nothing to compare, and the
    // slice sort does not compute any key.
    if expected_map.len() < 2 {
        prop_assert!(calls.values().all(|&n| n == 1));
    } else {
        prop_assert_eq!(calls.len(), expected_map.len());
    }
    for &(key, _) in expected_map.iter().filter(|_| expected_map.len() >= 2) {
        prop_assert_eq!(calls.get(&key), Some(&1), "key {:?}", key);
    }

    expected_map.sort_by_key(|&(key, _)| key);
    map.sort_keys();
    check_map(&map, &expected_map, &probes)?;
    check_searches(&map, &probes)?;

    map.clear();
    check_map(&map, &[], &probes)?;
    check_searches(&map, &probes)
}

macro_rules! suite {
    ($name:ident, $key:ty, $page:literal, $directory:ty) => {
        mod $name {
            use super::*;

            proptest! {
                #![proptest_config(config())]
                #[test]
                fn stable_sorts_and_searches(
                    entries in prop::collection::vec((keys::<$key>(), any::<u32>()), 0..max_len()),
                    probes in prop::collection::vec(keys::<$key>(), 0..max_len()),
                ) {
                    run_sorts::<$key, $page, $directory>(&entries, &probes)?;
                }
            }

            #[cfg(feature = "std")]
            #[test]
            fn panicking_sort_callbacks() {
                panic_tests::check_panics::<$key, $page, $directory>();
            }
        }
    };
}

macro_rules! layouts {
    ($key:ty) => {
        suite!(flat_1, $key, 1, Flat);
        suite!(flat_2, $key, 2, Flat);
        suite!(flat_3, $key, 3, Flat);
        suite!(flat_4, $key, 4, Flat);
        suite!(flat_8, $key, 8, Flat);
        suite!(flat_1024, $key, 1024, Flat);
        suite!(tree_1, $key, 1, Tree);
        suite!(tree_2, $key, 2, Tree);
        suite!(tree_3, $key, 3, Tree);
        suite!(tree_4, $key, 4, Tree);
        suite!(tree_8, $key, 8, Tree);
        suite!(tree_1024, $key, 1024, Tree);
    };
}

mod generational {
    use super::*;
    layouts!(Gen);
}

#[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
mod integer {
    use super::*;
    layouts!(u32);
}

#[test]
fn cached_key_is_called_at_most_once_for_a_single_entry() {
    let key = Gen::make(7, 1);
    let mut map = PagedSparseMap::<Gen, u32, 3>::new();
    map.insert(key, 42);
    let mut calls = 0;
    map.sort_by_cached_key(|stored, value| {
        assert_eq!((*stored, *value), (key, 42));
        calls += 1;
        *value
    });
    // Nothing to compare: the slice sort need not compute the key at all.
    assert!(calls <= 1);
    assert_eq!(map.get(key), Some(&42));
    assert_eq!(map.position(key), Some(0));
}

#[cfg(feature = "std")]
mod panic_tests {
    use super::*;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    pub(super) fn check_panics<K: TestKey, const PAGE: usize, D: Directory>() {
        let entries: Vec<_> = (0..64_u16)
            .map(|i| {
                let index = (i * 37 + 13) % 64;
                (K::make(index, (i % 4) as u8), u32::from(63 - index))
            })
            .collect();
        let probes = probes(&entries, &[]);
        for panic_after in [0, 1, 10, 50] {
            for by_key in [false, true] {
                let mut set = make_set::<K, PAGE, D>(&entries);
                let original: BTreeSet<_> = set.iter().copied().collect();
                let mut calls = 0;
                let result = catch_unwind(AssertUnwindSafe(|| {
                    if by_key {
                        set.sort_by_key(|key| {
                            calls += 1;
                            assert!(calls <= panic_after, "set key panic");
                            key.index()
                        });
                    } else {
                        set.sort_by(|a, b| {
                            calls += 1;
                            assert!(calls <= panic_after, "set comparator panic");
                            a.index().cmp(&b.index())
                        });
                    }
                }));
                assert!(result.is_err(), "sort finished within {panic_after} calls");
                assert_eq!(calls, panic_after + 1);
                assert_eq!(set.iter().copied().collect::<BTreeSet<_>>(), original);
                check_set(&set, set.as_slice(), &probes).unwrap();
                set.sort();
                check_set(&set, &original.into_iter().collect::<Vec<_>>(), &probes).unwrap();
            }
            for mode in 0..3 {
                let mut map = make_map::<K, PAGE, D>(&entries);
                let original: Vec<_> = map.iter().map(|(&k, &v)| (k, v)).collect();
                let mut calls = 0;
                let result = catch_unwind(AssertUnwindSafe(|| match mode {
                    0 => map.sort_by(|(a, va), (b, vb)| {
                        calls += 1;
                        assert!(calls <= panic_after, "map comparator panic");
                        (va, a).cmp(&(vb, b))
                    }),
                    1 => map.sort_by_key(|key, value| {
                        calls += 1;
                        assert!(calls <= panic_after, "map key panic");
                        (*value, *key)
                    }),
                    _ => map.sort_by_cached_key(|key, value| {
                        calls += 1;
                        assert!(calls <= panic_after, "map cached key panic");
                        (*value, *key)
                    }),
                }));
                assert!(result.is_err(), "sort finished within {panic_after} calls");
                assert_eq!(calls, panic_after + 1);
                check_map(&map, &original, &probes).unwrap();
                map.sort_keys();
                let mut sorted = original;
                sorted.sort_by_key(|&(key, _)| key);
                check_map(&map, &sorted, &probes).unwrap();
            }
        }
    }
}

#[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
mod hive_tests {
    use super::*;
    use paged_collections::hive::Hive;
    use paged_collections::id_alloc::Id;

    #[derive(Clone, Copy, Debug)]
    enum Op {
        Insert(u32),
        InsertWith(u32),
        Remove(usize, bool),
    }

    fn operations() -> BoxedStrategy<Op> {
        prop_oneof![
            4 => any::<u32>().prop_map(Op::Insert),
            4 => any::<u32>().prop_map(Op::InsertWith),
            5 => (any::<usize>(), any::<bool>()).prop_map(|(pick, live)| Op::Remove(pick, live)),
        ]
        .boxed()
    }

    fn run<const PAGE: usize>(ops: &[Op]) -> TestCaseResult {
        let mut fallible = Hive::<(u64, u32), PAGE>::new();
        let mut twin = Hive::<(u64, u32), PAGE>::new();
        let mut model = BTreeMap::new();
        let mut history = Vec::<Id>::new();
        let prelude = [
            Op::Insert(0),
            Op::InsertWith(1),
            Op::Remove(0, true),
            Op::InsertWith(2),
        ];
        prop_assert!(!fallible.is_exhausted());
        prop_assert!(!twin.is_exhausted());
        for op in prelude.iter().chain(ops.iter()) {
            match *op {
                Op::Insert(value) => {
                    let id = twin.insert((0, value));
                    prop_assert_eq!(fallible.try_insert((0, value)), Ok(id));
                    prop_assert_eq!(model.insert(id, (0, value)), None);
                    history.push(id);
                }
                Op::InsertWith(value) => {
                    let mut calls = 0;
                    let mut passed_id = None;
                    let actual = fallible.try_insert_with(|id| {
                        calls += 1;
                        passed_id = Some(id);
                        (id.to_bits(), value)
                    });
                    let id = twin.insert_with(|id| (id.to_bits(), value));
                    prop_assert_eq!(actual, Some(id));
                    prop_assert_eq!(passed_id, Some(id));
                    prop_assert_eq!(calls, 1);
                    prop_assert_eq!(model.insert(id, (id.to_bits(), value)), None);
                    history.push(id);
                }
                Op::Remove(pick, live) => {
                    let candidates: Vec<_> = if live {
                        model.keys().copied().collect()
                    } else {
                        history.clone()
                    };
                    if !candidates.is_empty() {
                        let id = candidates[pick % candidates.len()];
                        let expected = model.remove(&id);
                        prop_assert_eq!(fallible.remove(id), expected);
                        prop_assert_eq!(twin.remove(id), expected);
                    }
                }
            }
            prop_assert!(!fallible.is_exhausted());
            prop_assert!(!twin.is_exhausted());
            prop_assert_eq!(fallible.len(), model.len());
            prop_assert_eq!(twin.len(), model.len());
            let expected: Vec<_> = model.iter().map(|(&id, &value)| (id, value)).collect();
            prop_assert_eq!(
                fallible
                    .iter()
                    .map(|(id, &value)| (id, value))
                    .collect::<Vec<_>>(),
                expected.clone()
            );
            prop_assert_eq!(
                twin.iter()
                    .map(|(id, &value)| (id, value))
                    .collect::<Vec<_>>(),
                expected
            );
            for &id in &history {
                prop_assert_eq!(fallible.get(id), model.get(&id));
                prop_assert_eq!(twin.get(id), model.get(&id));
                prop_assert_eq!(fallible.contains(id), model.contains_key(&id));
            }
        }
        for (id, value) in model {
            prop_assert_eq!(fallible.remove(id), Some(value));
            prop_assert_eq!(twin.remove(id), Some(value));
            prop_assert_eq!(fallible.remove(id), None);
            prop_assert!(!fallible.is_exhausted());
            prop_assert!(!twin.is_exhausted());
        }
        prop_assert!(fallible.is_empty());
        prop_assert!(twin.is_empty());
        Ok(())
    }

    proptest! {
        #![proptest_config(config())]
        #[test]
        fn fallible_insertions_match_infallible_twins(
            ops in prop::collection::vec(operations(), 0..max_len()),
        ) {
            run::<1>(&ops)?;
            run::<2>(&ops)?;
            run::<3>(&ops)?;
            run::<4>(&ops)?;
            run::<8>(&ops)?;
            run::<1024>(&ops)?;
        }
    }
}
