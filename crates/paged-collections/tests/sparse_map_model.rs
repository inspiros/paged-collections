//! Model-based tests for slot identity, parallel dense arrays, and map traits.
//!
//! The oracle owns a map from slot indices to entries and a separate dense
//! vector of keys. It models swap removal exactly, but accepts any valid dense
//! permutation after retain or sorting. Exhaustive small-domain probes and
//! historical wide keys keep checking stale slots and generation replacement.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;
use std::iter::FusedIterator;
use std::panic::{AssertUnwindSafe, catch_unwind};

use paged_collections::key::Key;
use paged_collections::page::{Directory, Flat, Radix, Tree};
use paged_collections::sparse_map::PagedSparseMap;
use paged_collections::sparse_set::Slot;
use proptest::prelude::*;
use proptest::test_runner::TestCaseResult;

/// Proptest settings: 512 cases normally, a handful under Miri, which is
/// orders of magnitude slower and cannot write failure-persistence files.
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

/// Upper bound for generated op sequences and entry lists.
fn max_len() -> usize {
    if cfg!(miri) { 24 } else { 200 }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct Gen {
    index: u16,
    generation: u8,
}

impl Key for Gen {
    fn index(self) -> usize {
        usize::from(self.index)
    }
}

trait TestKey: Key + Ord + Debug {
    fn generation(self) -> u8 {
        0
    }
}

impl TestKey for Gen {
    fn generation(self) -> u8 {
        self.generation
    }
}

#[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
impl TestKey for u32 {}

#[cfg(target_pointer_width = "64")]
impl TestKey for u64 {}

#[derive(Clone, Copy, Debug)]
struct Predicate {
    modulus: u8,
    residue: u8,
    generations: u8,
    invert: bool,
    delta: u32,
}

impl Predicate {
    fn apply<K: TestKey>(self, key: K, value: &mut u32) -> bool {
        *value = value.wrapping_add(self.delta);
        let matches = key.index() % usize::from(self.modulus)
            == usize::from(self.residue % self.modulus)
            && self.generations & (1_u8 << key.generation()) != 0;
        matches ^ self.invert
    }
}

#[derive(Clone, Debug)]
enum Op<K> {
    Insert(K, u32),
    Replace(K, u32),
    TryInsert(K, u32),
    GetOrInsertWith(K, u32),
    GetMut(K, u32),
    OccupantMut(usize, u32),
    Remove(K),
    RemoveEntry(K),
    RemoveOccupant(usize),
    Retain(Predicate),
    Clear,
    Reset,
    ShrinkToFit,
    SortUnstableKeys,
    SortUnstableBy,
    SortUnstableByKey(u8),
    Reserve(u8),
}

fn gen_keys() -> BoxedStrategy<Gen> {
    (0_u16..48, 0_u8..3)
        .prop_map(|(index, generation)| Gen { index, generation })
        .boxed()
}

fn gen_probes() -> Vec<Gen> {
    // Include never-inserted indices and a never-generated generation as well.
    (0_u16..52)
        .flat_map(|index| (0_u8..4).map(move |generation| Gen { index, generation }))
        .collect()
}

#[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
fn integer_keys() -> BoxedStrategy<u32> {
    (0_u32..48).boxed()
}

#[cfg(target_pointer_width = "64")]
fn wide_keys() -> BoxedStrategy<u64> {
    // Uniform full-range values, frequent collisions, and arithmetic edges.
    prop_oneof![
        4 => any::<u64>(),
        4 => 0_u64..48,
        1 => Just(u64::MAX),
        1 => Just(u64::MAX - 1),
        1 => Just(1_u64 << 63),
        1 => Just(1_u64 << 32),
    ]
    .boxed()
}

#[cfg(target_pointer_width = "64")]
fn wide_probes() -> Vec<u64> {
    vec![0, 1, 2, 3, 4, u64::MAX, u64::MAX - 1, 1 << 63, 1 << 32]
}

fn operations<K: TestKey + 'static>(keys: BoxedStrategy<K>) -> BoxedStrategy<Op<K>> {
    let indices = prop_oneof![
        4 => keys.clone().prop_map(|key| key.index()),
        1 => 0_usize..52,
        1 => Just(usize::MAX),
    ];
    let predicates = (1_u8..8, 0_u8..8, 0_u8..8, any::<bool>(), any::<u32>()).prop_map(
        |(modulus, residue, generations, invert, delta)| Predicate {
            modulus,
            residue,
            generations,
            invert,
            delta,
        },
    );
    let entries = (keys.clone(), any::<u32>());
    prop_oneof![
        12 => entries.clone().prop_map(|(k, v)| Op::Insert(k, v)),
        8 => entries.clone().prop_map(|(k, v)| Op::Replace(k, v)),
        6 => entries.clone().prop_map(|(k, v)| Op::TryInsert(k, v)),
        6 => entries.clone().prop_map(|(k, v)| Op::GetOrInsertWith(k, v)),
        5 => entries.prop_map(|(k, v)| Op::GetMut(k, v)),
        5 => (indices.clone(), any::<u32>()).prop_map(|(i, v)| Op::OccupantMut(i, v)),
        5 => keys.clone().prop_map(Op::Remove),
        5 => keys.prop_map(Op::RemoveEntry),
        5 => indices.prop_map(Op::RemoveOccupant),
        4 => predicates.prop_map(Op::Retain),
        1 => Just(Op::Clear),
        1 => Just(Op::Reset),
        2 => Just(Op::ShrinkToFit),
        2 => Just(Op::SortUnstableKeys),
        2 => Just(Op::SortUnstableBy),
        2 => (1_u8..8).prop_map(Op::SortUnstableByKey),
        1 => (0_u8..32).prop_map(Op::Reserve),
    ]
    .boxed()
}

struct Model<K> {
    slots: BTreeMap<usize, (K, u32)>,
    dense: Vec<K>,
    touched_pages: BTreeSet<usize>,
}

impl<K: TestKey> Model<K> {
    fn new() -> Self {
        Self {
            slots: BTreeMap::new(),
            dense: Vec::new(),
            touched_pages: BTreeSet::new(),
        }
    }

    fn replace<const PAGE: usize>(&mut self, key: K, value: u32) -> Option<(K, u32)> {
        let previous = self.slots.insert(key.index(), (key, value));
        if let Some((old, _)) = previous {
            let pos = self.dense.iter().position(|&k| k == old).unwrap();
            self.dense[pos] = key;
        } else {
            self.dense.push(key);
            self.touched_pages.insert(key.index() / PAGE);
        }
        previous
    }

    fn get(&self, key: K) -> Option<u32> {
        self.slots
            .get(&key.index())
            .filter(|(stored, _)| *stored == key)
            .map(|(_, value)| *value)
    }

    fn get_mut(&mut self, key: K) -> Option<&mut u32> {
        self.slots
            .get_mut(&key.index())
            .filter(|(stored, _)| *stored == key)
            .map(|(_, value)| value)
    }

    fn remove_occupant(&mut self, index: usize) -> Option<(K, u32)> {
        let entry = self.slots.remove(&index)?;
        let pos = self.dense.iter().position(|&k| k == entry.0).unwrap();
        self.dense.swap_remove(pos);
        Some(entry)
    }

    fn remove(&mut self, key: K) -> Option<(K, u32)> {
        self.get(key)?;
        self.remove_occupant(key.index())
    }

    fn clear(&mut self) {
        self.slots.clear();
        self.dense.clear();
    }

    fn entries(&self) -> Vec<(K, u32)> {
        self.dense
            .iter()
            .map(|key| self.slots[&key.index()])
            .collect()
    }

    fn adopt_permutation<const PAGE: usize, D: Directory>(
        &mut self,
        map: &PagedSparseMap<K, u32, PAGE, D>,
        context: &str,
    ) -> TestCaseResult {
        let actual: BTreeMap<_, _> = map.iter().map(|(&k, &v)| (k.index(), (k, v))).collect();
        prop_assert_eq!(map.len(), self.slots.len(), "{}", context);
        prop_assert_eq!(
            map.key_slice().len(),
            map.value_slice().len(),
            "{}",
            context
        );
        prop_assert_eq!(&actual, &self.slots, "{}", context);
        self.dense = map.key_slice().to_vec();
        Ok(())
    }
}

/// Exercise exact lengths and alternating front/back consumption, including
/// exhaustion from both ends. The bounds also check the iterator traits.
fn check_iterator<I, T>(mut iter: I, expected: &[T], context: &str) -> TestCaseResult
where
    I: ExactSizeIterator<Item = T> + DoubleEndedIterator + FusedIterator,
    T: PartialEq + Debug,
{
    let (mut front, mut back) = (0, expected.len());
    while front < back {
        let remaining = back - front;
        prop_assert_eq!(iter.len(), remaining, "{}", context);
        prop_assert_eq!(
            iter.size_hint(),
            (remaining, Some(remaining)),
            "{}",
            context
        );
        if remaining % 2 == 0 {
            let item = iter.next();
            prop_assert_eq!(item.as_ref(), Some(&expected[front]), "{}", context);
            front += 1;
        } else {
            back -= 1;
            let item = iter.next_back();
            prop_assert_eq!(item.as_ref(), Some(&expected[back]), "{}", context);
        }
    }
    prop_assert_eq!(iter.len(), 0, "{}", context);
    prop_assert_eq!(iter.size_hint(), (0, Some(0)), "{}", context);
    prop_assert_eq!(iter.next(), None, "{}", context);
    prop_assert_eq!(iter.next_back(), None, "{}", context);
    prop_assert_eq!(iter.next(), None, "{}", context);
    Ok(())
}

fn check_invariants<K: TestKey, const PAGE: usize, D: Directory>(
    map: &mut PagedSparseMap<K, u32, PAGE, D>,
    model: &Model<K>,
    probes: &BTreeSet<K>,
    indices: &BTreeSet<usize>,
    context: &str,
) -> TestCaseResult {
    prop_assert_eq!(map.len(), model.slots.len(), "{}", context);
    prop_assert_eq!(map.is_empty(), model.slots.is_empty(), "{}", context);
    prop_assert_eq!(
        map.key_slice().len(),
        map.value_slice().len(),
        "{}",
        context
    );
    prop_assert_eq!(map.key_slice(), model.dense.as_slice(), "{}", context);
    prop_assert_eq!(map.as_set().as_slice(), map.key_slice(), "{}", context);
    prop_assert_eq!(map.as_set().len(), map.len(), "{}", context);
    prop_assert!(map.capacity() >= map.len(), "{}", context);
    let expected = model.entries();
    let mut seen = BTreeSet::new();
    for (pos, (&key, &value)) in map.key_slice().iter().zip(map.value_slice()).enumerate() {
        prop_assert!(seen.insert(key.index()), "duplicate index: {}", context);
        prop_assert_eq!(
            model.slots.get(&key.index()),
            Some(&(key, value)),
            "{}",
            context
        );
        prop_assert_eq!(map.position(key), Some(pos), "{}", context);
        prop_assert_eq!(map.get(key), Some(&value), "{}", context);
        prop_assert_eq!(
            map.occupant(key.index()),
            Some((&key, &value)),
            "{}",
            context
        );
        prop_assert_eq!(map[key], value, "{}", context);
    }
    for &key in probes {
        let value = model.get(key);
        let entry = value.map(|v| (key, v));
        let position = model.dense.iter().position(|&stored| stored == key);
        prop_assert_eq!(
            map.contains_key(key),
            value.is_some(),
            "probe {:?}: {}",
            key,
            context
        );
        prop_assert_eq!(map.get(key).copied(), value, "probe {:?}: {}", key, context);
        prop_assert_eq!(map.position(key), position, "probe {:?}: {}", key, context);
        prop_assert_eq!(
            map.get_key_value(key).map(|(&k, &v)| (k, v)),
            entry,
            "probe {:?}: {}",
            key,
            context
        );
        prop_assert_eq!(map.as_set().contains(key), value.is_some(), "{}", context);
        prop_assert_eq!(map.as_set().position(key), position, "{}", context);
    }
    for &index in indices {
        prop_assert_eq!(
            map.occupant(index).map(|(&k, &v)| (k, v)),
            model.slots.get(&index).copied(),
            "{}",
            context
        );
        prop_assert_eq!(
            map.as_set().occupant(index).copied(),
            model.slots.get(&index).map(|&(key, _)| key),
            "{}",
            context
        );
    }
    prop_assert!(
        map.iter()
            .map(|(&k, &v)| (k, v))
            .eq(expected.iter().copied()),
        "{}",
        context
    );
    prop_assert!(
        map.iter()
            .rev()
            .map(|(&k, &v)| (k, v))
            .eq(expected.iter().rev().copied()),
        "{}",
        context
    );
    let keys = model.dense.as_slice();
    let values: Vec<_> = expected.iter().map(|&(_, value)| value).collect();
    check_iterator(map.iter().map(|(&k, &v)| (k, v)), &expected, context)?;
    check_iterator(map.iter_mut().map(|(&k, v)| (k, *v)), &expected, context)?;
    check_iterator(map.keys().copied(), keys, context)?;
    check_iterator(map.values().copied(), &values, context)?;
    check_iterator(map.values_mut().map(|v| *v), &values, context)?;
    check_iterator(
        (&*map).into_iter().map(|(&k, &v)| (k, v)),
        &expected,
        context,
    )?;
    check_iterator(
        (&mut *map).into_iter().map(|(&k, v)| (k, *v)),
        &expected,
        context,
    )?;
    prop_assert_eq!(map.value_slice_mut(), values.as_slice(), "{}", context);
    let live_pages: BTreeSet<_> = model.slots.keys().map(|index| index / PAGE).collect();
    prop_assert!(map.allocated_pages() >= live_pages.len(), "{}", context);
    prop_assert!(
        map.allocated_pages() <= model.touched_pages.len(),
        "{}",
        context
    );
    prop_assert_eq!(
        map.as_set().allocated_pages(),
        map.allocated_pages(),
        "{}",
        context
    );
    Ok(())
}

fn run_sequence<K: TestKey, const PAGE: usize, D: Directory>(
    ops: &[Op<K>],
    initial_probes: Vec<K>,
) -> TestCaseResult {
    let mut probes: BTreeSet<_> = initial_probes.into_iter().collect();
    let mut indices = BTreeSet::from([0, usize::MAX]);
    // Register all keys up front so removed and cleared keys remain probes
    // even after their old dense positions have been reused.
    for op in ops {
        match op {
            Op::Insert(k, _)
            | Op::Replace(k, _)
            | Op::TryInsert(k, _)
            | Op::GetOrInsertWith(k, _)
            | Op::GetMut(k, _)
            | Op::Remove(k)
            | Op::RemoveEntry(k) => {
                probes.insert(*k);
            }
            Op::OccupantMut(index, _) | Op::RemoveOccupant(index) => {
                indices.insert(*index);
            }
            _ => {}
        }
    }
    for &key in &probes {
        let index = key.index();
        indices.extend([index, index.saturating_sub(1), index.saturating_add(1)]);
    }

    let mut map = PagedSparseMap::<K, u32, PAGE, D>::new();
    let mut model = Model::new();
    check_invariants(&mut map, &model, &probes, &indices, "initial state")?;
    for (step, op) in ops.iter().enumerate() {
        let context = format!("step {step}: {op:?}");
        let pages_before = map.allocated_pages();
        match *op {
            Op::Insert(key, value) => {
                let expected = model.replace::<PAGE>(key, value).map(|(_, v)| v);
                prop_assert_eq!(map.insert(key, value), expected, "{}", context);
            }
            Op::Replace(key, value) => {
                prop_assert_eq!(
                    map.replace(key, value),
                    model.replace::<PAGE>(key, value),
                    "{}",
                    context
                );
            }
            Op::TryInsert(key, value) => {
                let expected = if model.slots.contains_key(&key.index()) {
                    Err(value)
                } else {
                    model.replace::<PAGE>(key, value);
                    Ok(value)
                };
                let actual = map.try_insert(key, value).map(|stored| {
                    let previous = *stored;
                    *stored ^= 1;
                    previous
                });
                prop_assert_eq!(actual, expected, "{}", context);
                if actual.is_ok() {
                    *model.get_mut(key).unwrap() ^= 1;
                }
                if expected.is_err() {
                    prop_assert_eq!(map.allocated_pages(), pages_before, "{}", context);
                }
            }
            Op::GetOrInsertWith(key, value) => {
                let existing = model.get(key);
                let expected = existing.unwrap_or(value);
                if existing.is_none() {
                    model.replace::<PAGE>(key, value);
                }
                let mut calls = 0;
                let actual = map.get_or_insert_with(key, || {
                    calls += 1;
                    value
                });
                prop_assert_eq!(*actual, expected, "{}", context);
                *actual ^= 1;
                *model.get_mut(key).unwrap() ^= 1;
                prop_assert_eq!(calls, usize::from(existing.is_none()), "{}", context);
            }
            Op::GetMut(key, delta) => {
                let expected = model.get_mut(key).map(|v| {
                    *v = v.wrapping_add(delta);
                    *v
                });
                let actual = map.get_mut(key).map(|v| {
                    *v = v.wrapping_add(delta);
                    *v
                });
                prop_assert_eq!(actual, expected, "{}", context);
            }
            Op::OccupantMut(index, delta) => {
                let expected = model.slots.get_mut(&index).map(|(k, v)| {
                    *v = v.wrapping_add(delta);
                    (*k, *v)
                });
                let actual = map.occupant_mut(index).map(|(&k, v)| {
                    *v = v.wrapping_add(delta);
                    (k, *v)
                });
                prop_assert_eq!(actual, expected, "{}", context);
            }
            Op::Remove(key) => {
                prop_assert_eq!(
                    map.remove(key),
                    model.remove(key).map(|(_, v)| v),
                    "{}",
                    context
                );
            }
            Op::RemoveEntry(key) => {
                prop_assert_eq!(map.remove_entry(key), model.remove(key), "{}", context);
            }
            Op::RemoveOccupant(index) => {
                prop_assert_eq!(
                    map.remove_occupant(index),
                    model.remove_occupant(index),
                    "{}",
                    context
                );
            }
            Op::Retain(predicate) => {
                let before: BTreeSet<_> = model.slots.keys().copied().collect();
                let mut calls = BTreeMap::<usize, usize>::new();
                map.retain(|&key, value| {
                    *calls.entry(key.index()).or_default() += 1;
                    predicate.apply(key, value)
                });
                prop_assert_eq!(
                    calls.keys().copied().collect::<BTreeSet<_>>(),
                    before,
                    "{}",
                    context
                );
                prop_assert!(calls.values().all(|&count| count == 1), "{}", context);
                model
                    .slots
                    .retain(|_, (key, value)| predicate.apply(*key, value));
                model.adopt_permutation(&map, &context)?;
            }
            Op::Clear => {
                map.clear();
                model.clear();
                prop_assert_eq!(map.allocated_pages(), pages_before, "{}", context);
            }
            Op::Reset => {
                map.reset();
                model.clear();
                model.touched_pages.clear();
                prop_assert_eq!(map.allocated_pages(), 0, "{}", context);
            }
            Op::ShrinkToFit => {
                map.shrink_to_fit();
                if model.slots.is_empty() {
                    model.touched_pages.clear();
                    prop_assert_eq!(map.allocated_pages(), 0, "{}", context);
                }
            }
            Op::SortUnstableKeys => {
                map.sort_unstable_keys();
                prop_assert!(
                    map.key_slice().windows(2).all(|w| w[0] <= w[1]),
                    "{}",
                    context
                );
                model.adopt_permutation(&map, &context)?;
            }
            Op::SortUnstableBy => {
                map.sort_unstable_by(|(_, a), (_, b)| b.cmp(a));
                prop_assert!(
                    map.value_slice().windows(2).all(|w| w[0] >= w[1]),
                    "{}",
                    context
                );
                model.adopt_permutation(&map, &context)?;
            }
            Op::SortUnstableByKey(modulus) => {
                let sort_key = |key: &K, value: &u32| {
                    (
                        value % u32::from(modulus),
                        key.index() % usize::from(modulus),
                    )
                };
                map.sort_unstable_by_key(sort_key);
                prop_assert!(
                    map.iter()
                        .map(|(k, v)| sort_key(k, v))
                        .collect::<Vec<_>>()
                        .windows(2)
                        .all(|w| w[0] <= w[1]),
                    "{}",
                    context
                );
                model.adopt_permutation(&map, &context)?;
            }
            Op::Reserve(additional) => {
                map.reserve(usize::from(additional));
                prop_assert!(
                    map.capacity() >= map.len() + usize::from(additional),
                    "{}",
                    context
                );
            }
        }
        check_invariants(&mut map, &model, &probes, &indices, &context)?;
    }
    Ok(())
}

macro_rules! sequence_test {
    ($name:ident, $key:ty, $page:literal, $directory:ty, $keys:expr, $probes:expr) => {
        proptest! {
            #![proptest_config(config())]
            #[test]
            fn $name(ops in prop::collection::vec(operations($keys), 0..max_len())) {
                run_sequence::<$key, $page, $directory>(&ops, $probes)?;
            }
        }
    };
}

sequence_test!(gen_flat_1, Gen, 1, Flat, gen_keys(), gen_probes());
sequence_test!(gen_flat_3, Gen, 3, Flat, gen_keys(), gen_probes());
sequence_test!(gen_flat_4, Gen, 4, Flat, gen_keys(), gen_probes());
sequence_test!(gen_flat_8, Gen, 8, Flat, gen_keys(), gen_probes());
sequence_test!(gen_tree_1, Gen, 1, Tree, gen_keys(), gen_probes());
sequence_test!(gen_radix_1, Gen, 1, Radix, gen_keys(), gen_probes());
sequence_test!(gen_tree_4, Gen, 4, Tree, gen_keys(), gen_probes());
sequence_test!(gen_radix_4, Gen, 4, Radix, gen_keys(), gen_probes());

#[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
mod integer_sequences {
    use super::*;

    sequence_test!(u32_flat_1, u32, 1, Flat, integer_keys(), (0..52).collect());
    sequence_test!(u32_flat_3, u32, 3, Flat, integer_keys(), (0..52).collect());
    sequence_test!(u32_flat_4, u32, 4, Flat, integer_keys(), (0..52).collect());
    sequence_test!(u32_flat_8, u32, 8, Flat, integer_keys(), (0..52).collect());
    sequence_test!(u32_tree_4, u32, 4, Tree, integer_keys(), (0..52).collect());
    sequence_test!(
        u32_radix_4,
        u32,
        4,
        Radix,
        integer_keys(),
        (0..52).collect()
    );
}

#[cfg(target_pointer_width = "64")]
sequence_test!(u64_tree_1, u64, 1, Tree, wide_keys(), wide_probes());
#[cfg(target_pointer_width = "64")]
sequence_test!(u64_radix_1, u64, 1, Radix, wide_keys(), wide_probes());
#[cfg(target_pointer_width = "64")]
sequence_test!(u64_tree_4, u64, 4, Tree, wide_keys(), wide_probes());
#[cfg(target_pointer_width = "64")]
sequence_test!(u64_radix_4, u64, 4, Radix, wide_keys(), wide_probes());

fn check_traits<K: TestKey, const PAGE: usize, D: Directory>(entries: &[(K, u32)]) -> TestCaseResult
where
    D::Store<Slot, PAGE>: Clone,
{
    let mut repeated = PagedSparseMap::<K, u32, PAGE, D>::new();
    let mut model = Model::new();
    for &(key, value) in entries {
        prop_assert_eq!(
            repeated.insert(key, value),
            model.replace::<PAGE>(key, value).map(|(_, v)| v)
        );
    }
    let probes = entries.iter().map(|&(key, _)| key).collect();
    let indices = entries.iter().map(|(key, _)| key.index()).collect();
    check_invariants(&mut repeated, &model, &probes, &indices, "repeated insert")?;

    let mut collected: PagedSparseMap<K, u32, PAGE, D> = entries.iter().copied().collect();
    check_invariants(&mut collected, &model, &probes, &indices, "collect")?;
    prop_assert_eq!(&collected, &repeated);

    let mut extended = PagedSparseMap::<K, u32, PAGE, D>::default();
    let mid = entries.len() / 2;
    extended.extend(entries[..mid].iter().copied());
    extended.extend(entries[mid..].iter().map(|(k, v)| (k, v)));
    check_invariants(
        &mut extended,
        &model,
        &probes,
        &indices,
        "extend owned and borrowed",
    )?;
    prop_assert_eq!(&extended, &repeated);

    let expected = model.entries();
    let mut cloned = repeated.clone();
    prop_assert_eq!(&cloned, &repeated);
    prop_assert_eq!(cloned.allocated_pages(), repeated.allocated_pages());
    check_invariants(&mut cloned, &model, &probes, &indices, "clone")?;
    if let Some(&(first, value)) = expected.first() {
        prop_assert_eq!(repeated[first], value);
        prop_assert_eq!(cloned.remove(first), Some(value));
        prop_assert_eq!(repeated.get(first), Some(&value));
        prop_assert_ne!(&cloned, &repeated);
        let mut changed = repeated.clone();
        *changed.get_mut(first).unwrap() ^= 1;
        prop_assert_ne!(&changed, &repeated);
        prop_assert_ne!(&repeated, &changed);
        let debug = format!("{repeated:?}");
        let key_debug = format!("{first:?}");
        let value_debug = format!("{value:?}");
        prop_assert!(debug.contains(&key_debug));
        prop_assert!(debug.contains(&value_debug));
    }
    let reversed: PagedSparseMap<K, u32, PAGE, D> = expected.iter().rev().copied().collect();
    prop_assert_eq!(&reversed, &repeated);
    prop_assert_eq!(&repeated, &reversed);
    prop_assert_eq!(
        repeated.clone().into_iter().collect::<Vec<_>>(),
        expected.as_slice()
    );
    prop_assert_eq!(
        repeated.clone().into_iter().rev().collect::<Vec<_>>(),
        expected.iter().rev().copied().collect::<Vec<_>>()
    );
    let expected_keys: Vec<_> = expected.iter().map(|&(key, _)| key).collect();
    let expected_values: Vec<_> = expected.iter().map(|&(_, value)| value).collect();
    prop_assert_eq!(
        repeated.clone().into_keys().collect::<Vec<_>>(),
        expected_keys.as_slice()
    );
    prop_assert_eq!(
        repeated.clone().into_values().collect::<Vec<_>>(),
        expected_values.as_slice()
    );
    check_iterator(repeated.clone().into_keys(), &expected_keys, "into_keys")?;
    check_iterator(
        repeated.clone().into_values(),
        &expected_values,
        "into_values",
    )?;

    // Clone after consuming from both ends, including singleton and empty
    // maps. The deterministic trait input below leaves several entries here.
    let mut borrowed = repeated.iter();
    let mut owned = repeated.clone().into_iter();
    prop_assert_eq!(
        borrowed.next().map(|(&key, &value)| (key, value)),
        expected.first().copied()
    );
    prop_assert_eq!(owned.next(), expected.first().copied());
    let front = usize::from(!expected.is_empty());
    let last = expected[front..].last().copied();
    prop_assert_eq!(borrowed.next_back().map(|(&k, &v)| (k, v)), last);
    prop_assert_eq!(owned.next_back(), last);
    let back = expected.len() - usize::from(last.is_some());
    let remaining = &expected[front..back];
    let borrowed_clone = borrowed.clone();
    let owned_clone = owned.clone();
    prop_assert_eq!(borrowed.len(), remaining.len());
    prop_assert_eq!(borrowed_clone.len(), remaining.len());
    prop_assert_eq!(owned.len(), remaining.len());
    prop_assert_eq!(owned_clone.len(), remaining.len());
    prop_assert_eq!(
        borrowed.map(|(&k, &v)| (k, v)).collect::<Vec<_>>(),
        remaining
    );
    prop_assert_eq!(
        borrowed_clone.map(|(&k, &v)| (k, v)).collect::<Vec<_>>(),
        remaining
    );
    prop_assert_eq!(owned.collect::<Vec<_>>(), remaining);
    prop_assert_eq!(owned_clone.collect::<Vec<_>>(), remaining);

    check_iterator(repeated.clone().into_iter(), &expected, "owned iterator")?;
    check_iterator(
        (&repeated).into_iter().map(|(&k, &v)| (k, v)),
        &expected,
        "borrowed",
    )?;
    check_iterator(
        (&mut repeated).into_iter().map(|(&k, v)| (k, *v)),
        &expected,
        "mutably borrowed",
    )?;
    // Exercise writes through both mutable entry iterators and the value
    // iterator, then restore the values before checking the oracle again.
    for (_, value) in repeated.iter_mut() {
        *value ^= 1;
    }
    prop_assert!(
        repeated
            .values()
            .copied()
            .eq(expected.iter().map(|&(_, v)| v ^ 1))
    );
    for (_, value) in &mut repeated {
        *value ^= 3;
    }
    prop_assert!(
        repeated
            .values()
            .copied()
            .eq(expected.iter().map(|&(_, v)| v ^ 2))
    );
    for value in repeated.values_mut() {
        *value ^= 2;
    }
    check_invariants(
        &mut repeated,
        &model,
        &probes,
        &indices,
        "original after mutations",
    )?;
    Ok(())
}

macro_rules! trait_test {
    ($name:ident, $key:ty, $page:literal, $directory:ty, $keys:expr) => {
        proptest! {
            #![proptest_config(config())]
            #[test]
            fn $name(entries in prop::collection::vec(($keys, any::<u32>()), 0..max_len())) {
                check_traits::<$key, $page, $directory>(&entries)?;
            }
        }
    };
}

trait_test!(traits_gen_flat_1, Gen, 1, Flat, gen_keys());
trait_test!(traits_gen_flat_3, Gen, 3, Flat, gen_keys());
trait_test!(traits_gen_flat_4, Gen, 4, Flat, gen_keys());
trait_test!(traits_gen_flat_8, Gen, 8, Flat, gen_keys());
trait_test!(traits_gen_tree_1, Gen, 1, Tree, gen_keys());
trait_test!(traits_gen_radix_1, Gen, 1, Radix, gen_keys());
trait_test!(traits_gen_tree_4, Gen, 4, Tree, gen_keys());
trait_test!(traits_gen_radix_4, Gen, 4, Radix, gen_keys());
#[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
trait_test!(traits_u32_flat_4, u32, 4, Flat, integer_keys());
#[cfg(target_pointer_width = "64")]
trait_test!(traits_u64_tree_4, u64, 4, Tree, wide_keys());
#[cfg(target_pointer_width = "64")]
trait_test!(traits_u64_radix_4, u64, 4, Radix, wide_keys());

fn generation_equality<const PAGE: usize, D: Directory>() {
    let old = Gen {
        index: 7,
        generation: 0,
    };
    let new = Gen {
        generation: 1,
        ..old
    };
    let other = Gen {
        index: 19,
        generation: 2,
    };
    let a: PagedSparseMap<Gen, u32, PAGE, D> = [(old, 10), (other, 20)].into_iter().collect();
    let b: PagedSparseMap<Gen, u32, PAGE, D> = [(other, 20), (old, 10)].into_iter().collect();
    let c: PagedSparseMap<Gen, u32, PAGE, D> = [(new, 10), (other, 20)].into_iter().collect();
    let d: PagedSparseMap<Gen, u32, PAGE, D> = [(old, 11), (other, 20)].into_iter().collect();
    assert_eq!(a, b);
    assert_eq!(b, a);
    assert_ne!(a, c);
    assert_ne!(c, a);
    assert_ne!(a, d);
    assert_ne!(d, a);
    assert!(catch_unwind(AssertUnwindSafe(|| a[new])).is_err());
    let absent = Gen {
        index: 51,
        generation: 0,
    };
    assert!(catch_unwind(AssertUnwindSafe(|| a[absent])).is_err());
}

#[test]
fn equality_and_indexing_distinguish_generations_in_every_directory() {
    generation_equality::<1, Flat>();
    generation_equality::<3, Flat>();
    generation_equality::<4, Flat>();
    generation_equality::<8, Flat>();
    generation_equality::<1, Tree>();
    generation_equality::<1, Radix>();
    generation_equality::<4, Tree>();
    generation_equality::<4, Radix>();
}

fn targeted_mutations<const PAGE: usize, D: Directory>() -> TestCaseResult
where
    D::Store<Slot, PAGE>: Clone,
{
    let a = Gen {
        index: 0,
        generation: 0,
    };
    let b = Gen {
        index: 1,
        generation: 1,
    };
    let c = Gen {
        index: 47,
        generation: 2,
    };
    let next_a = Gen { generation: 1, ..a };
    let next_b = Gen { generation: 2, ..b };
    let next_c = Gen { generation: 0, ..c };
    // No permutation-adopting ops here: the oracle requires every replacement
    // to keep its position and every removal to move the last key AND value.
    let ops = vec![
        Op::Insert(a, 10),
        Op::Insert(b, 20),
        Op::Insert(c, 30),
        Op::Insert(next_b, 40),
        Op::Replace(next_a, 50),
        Op::Replace(next_a, 60),
        Op::TryInsert(a, 70),
        Op::TryInsert(next_a, 80),
        Op::GetOrInsertWith(next_c, 90),
        Op::GetOrInsertWith(next_c, 100),
        Op::GetMut(c, 1),
        Op::GetMut(next_c, 1),
        Op::OccupantMut(c.index(), 2),
        Op::RemoveEntry(a),
        Op::Remove(next_a),
        Op::RemoveEntry(next_b),
        Op::RemoveOccupant(c.index()),
        Op::RemoveOccupant(c.index()),
        Op::Insert(a, 110),
        Op::Insert(b, 120),
        Op::Insert(c, 130),
        Op::Clear,
        Op::Insert(next_c, 140),
        Op::Insert(next_a, 150),
        Op::Insert(next_b, 160),
        Op::Remove(c),
        Op::RemoveEntry(a),
        Op::Clear,
        Op::ShrinkToFit,
        Op::Insert(b, 170),
        Op::Reset,
        Op::Insert(c, 180),
        Op::RemoveEntry(c),
        Op::ShrinkToFit,
    ];
    run_sequence::<Gen, PAGE, D>(&ops, gen_probes())?;
    // Explicit duplicate inputs make last-value and last-generation behavior
    // deterministic. Five occupied slots leave three entries after consuming
    // the front and back of the iterators before cloning them.
    let d = Gen {
        index: 2,
        generation: 0,
    };
    let e = Gen {
        index: 3,
        generation: 1,
    };
    check_traits::<Gen, PAGE, D>(&[
        (a, 1),
        (b, 2),
        (next_a, 3),
        (b, 4),
        (a, 5),
        (c, 6),
        (d, 7),
        (e, 8),
    ])
}

#[test]
fn dense_positions_swap_removal_and_stale_slots_in_every_directory() -> TestCaseResult {
    targeted_mutations::<1, Flat>()?;
    targeted_mutations::<3, Flat>()?;
    targeted_mutations::<4, Flat>()?;
    targeted_mutations::<8, Flat>()?;
    targeted_mutations::<1, Tree>()?;
    targeted_mutations::<1, Radix>()?;
    targeted_mutations::<4, Tree>()?;
    targeted_mutations::<4, Radix>()?;
    Ok(())
}

fn mutable_access<const PAGE: usize, D: Directory>() {
    let a = Gen {
        index: 1,
        generation: 0,
    };
    let b = Gen {
        index: 7,
        generation: 1,
    };
    let c = Gen {
        index: 47,
        generation: 2,
    };
    let next_a = Gen { generation: 1, ..a };
    let mut map = PagedSparseMap::<Gen, u32, PAGE, D>::new();
    *map.try_insert(a, 10).unwrap() = 11;
    assert_eq!(map.get(a), Some(&11));
    assert_eq!(map.value_slice(), &[11]);
    *map.get_or_insert_with(b, || 20) = 21;
    assert_eq!(map.get(b), Some(&21));
    assert_eq!(map.key_slice(), &[a, b]);
    assert_eq!(map.value_slice(), &[11, 21]);
    *map.get_or_insert_with(a, || panic!("equal key must not call make")) = 12;
    assert_eq!(map.get(a), Some(&12));
    assert_eq!(map.key_slice(), &[a, b]);
    assert_eq!(map.value_slice(), &[12, 21]);
    *map.get_or_insert_with(next_a, || 30) = 31;
    assert_eq!(map.get(a), None);
    assert_eq!(map.get(next_a), Some(&31));
    assert_eq!(map.key_slice(), &[next_a, b]);
    assert_eq!(map.value_slice(), &[31, 21]);
    *map.try_insert(c, 40).unwrap() = 41;
    assert_eq!(map.get(c), Some(&41));
    assert_eq!(map.value_slice(), &[31, 21, 41]);
    map.value_slice_mut().copy_from_slice(&[101, 202, 303]);
    assert_eq!(map.key_slice(), &[next_a, b, c]);
    assert_eq!(map.value_slice(), &[101, 202, 303]);
    for (key, value) in [(next_a, 101), (b, 202), (c, 303)] {
        assert_eq!(map.get(key), Some(&value));
    }
    assert_consistent(&map, &gen_probes());
}

#[test]
fn returned_mutable_values_and_value_slice_write_through() {
    mutable_access::<1, Flat>();
    mutable_access::<3, Flat>();
    mutable_access::<4, Flat>();
    mutable_access::<8, Flat>();
    mutable_access::<1, Tree>();
    mutable_access::<1, Radix>();
    mutable_access::<4, Tree>();
    mutable_access::<4, Radix>();
}

#[cfg(target_pointer_width = "64")]
#[test]
fn maximum_tree_index_with_single_slot_pages() -> TestCaseResult {
    let ops = vec![
        Op::Insert(0, 1),
        Op::Insert(u64::MAX, 2),
        Op::Insert(u64::MAX - 1, 3),
        Op::Insert(1 << 63, 4),
        Op::SortUnstableKeys,
        Op::Remove(u64::MAX),
        Op::Clear,
        Op::GetOrInsertWith(u64::MAX, 5),
        Op::Replace(u64::MAX, 6),
        Op::OccupantMut(usize::MAX, 7),
        Op::RemoveOccupant(usize::MAX),
        Op::ShrinkToFit,
    ];
    run_sequence::<u64, 1, Tree>(&ops, wide_probes())?;
    run_sequence::<u64, 1, Radix>(&ops, wide_probes())?;
    run_sequence::<u64, 4, Tree>(&ops, wide_probes())?;
    run_sequence::<u64, 4, Radix>(&ops, wide_probes())?;
    Ok(())
}

#[cfg(feature = "serde")]
mod serde_tests {
    use super::*;

    #[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
    fn round_trip<const PAGE: usize, D: Directory>(entries: &[(u32, u32)]) -> TestCaseResult {
        // JSON supports integer map keys, but cannot represent struct keys.
        let original: PagedSparseMap<u32, u32, PAGE, D> = entries.iter().copied().collect();
        let json = serde_json::to_string(&original).unwrap();
        let mut decoded: PagedSparseMap<u32, u32, PAGE, D> = serde_json::from_str(&json).unwrap();
        let mut model = Model::new();
        for &(key, value) in entries {
            model.replace::<PAGE>(key, value);
        }
        let probes = (0..52).collect();
        let indices = (0_usize..52).collect();
        prop_assert_eq!(&decoded, &original);
        check_invariants(&mut decoded, &model, &probes, &indices, "serde round trip")?;

        // Build the object directly: a JSON Value/BTreeMap would discard
        // duplicates before PagedSparseMap's visitor could see them.
        let fields: Vec<_> = entries
            .iter()
            .map(|(k, v)| format!("\"{k}\":{v}"))
            .collect();
        let input_json = format!("{{{}}}", fields.join(","));
        let mut from_input: PagedSparseMap<u32, u32, PAGE, D> =
            serde_json::from_str(&input_json).unwrap();
        check_invariants(
            &mut from_input,
            &model,
            &probes,
            &indices,
            "serde duplicates",
        )?;
        Ok(())
    }

    #[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
    macro_rules! serde_test {
        ($name:ident, $page:literal, $directory:ty) => {
            proptest! {
                #![proptest_config(config())]
                #[test]
                fn $name(entries in prop::collection::vec(
                    (integer_keys(), any::<u32>()), 0..max_len()
                )) {
                    round_trip::<$page, $directory>(&entries)?;
                }
            }
        };
    }

    #[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
    mod integer_round_trips {
        use super::*;

        serde_test!(round_trip_flat_1, 1, Flat);
        serde_test!(round_trip_flat_3, 3, Flat);
        serde_test!(round_trip_flat_4, 4, Flat);
        serde_test!(round_trip_flat_8, 8, Flat);
        serde_test!(round_trip_tree_1, 1, Tree);
        serde_test!(round_trip_radix_1, 1, Radix);
        serde_test!(round_trip_tree_4, 4, Tree);
        serde_test!(round_trip_radix_4, 4, Radix);

        #[test]
        fn duplicate_json_keys_keep_last_value() -> TestCaseResult {
            let entries = [(1, 10), (2, 20), (1, 30), (2, 40), (1, 50)];
            round_trip::<1, Flat>(&entries)?;
            round_trip::<3, Flat>(&entries)?;
            round_trip::<4, Flat>(&entries)?;
            round_trip::<8, Flat>(&entries)?;
            round_trip::<1, Tree>(&entries)?;
            round_trip::<1, Radix>(&entries)?;
            round_trip::<4, Tree>(&entries)?;
            round_trip::<4, Radix>(&entries)?;
            Ok(())
        }
    }

    fn deserialize_generations<const PAGE: usize, D: Directory>(
        entries: &[(Gen, u32)],
    ) -> TestCaseResult {
        // MapDeserializer supplies non-string struct keys directly to the
        // visitor. Individual JSON Values are deserializers for Gen/u32;
        // the outer map is never converted to a JSON object.
        let input = entries.iter().map(|(key, value)| {
            (
                serde_json::to_value(key).unwrap(),
                serde_json::to_value(value).unwrap(),
            )
        });
        let deserializer = serde::de::value::MapDeserializer::<_, serde_json::Error>::new(input);
        let mut decoded: PagedSparseMap<Gen, u32, PAGE, D> =
            serde::Deserialize::deserialize(deserializer).unwrap();
        let mut model = Model::new();
        for &(key, value) in entries {
            model.replace::<PAGE>(key, value);
        }
        let probes = gen_probes().into_iter().collect();
        let indices = (0_usize..52).collect();
        check_invariants(&mut decoded, &model, &probes, &indices, "serde generations")?;
        Ok(())
    }

    proptest! {
        #![proptest_config(config())]
        #[test]
        fn generated_generations_keep_last_entry(
            entries in prop::collection::vec((gen_keys(), any::<u32>()), 0..max_len())
        ) {
            deserialize_generations::<1, Flat>(&entries)?;
            deserialize_generations::<3, Flat>(&entries)?;
            deserialize_generations::<4, Tree>(&entries)?;
            deserialize_generations::<4, Radix>(&entries)?;
        }
    }

    #[test]
    fn collisions_keep_last_generation_and_value() -> TestCaseResult {
        let first = Gen {
            index: 1,
            generation: 0,
        };
        let last = Gen {
            generation: 1,
            ..first
        };
        let other = Gen {
            index: 2,
            generation: 2,
        };
        let entries = [(first, 10), (other, 20), (first, 30), (last, 40)];
        deserialize_generations::<1, Flat>(&entries)?;
        deserialize_generations::<3, Flat>(&entries)?;
        deserialize_generations::<4, Flat>(&entries)?;
        deserialize_generations::<8, Flat>(&entries)?;
        deserialize_generations::<1, Tree>(&entries)?;
        deserialize_generations::<1, Radix>(&entries)?;
        deserialize_generations::<4, Tree>(&entries)?;
        deserialize_generations::<4, Radix>(&entries)?;
        Ok(())
    }
}

/// Checks parallel arrays, pointer identity of lookups, and absence of stale
/// slots without requiring values to be Clone, Eq, or Debug.
fn assert_consistent<K: Key + Debug, V, const PAGE: usize, D: Directory>(
    map: &PagedSparseMap<K, V, PAGE, D>,
    probes: &[K],
) {
    assert_eq!(map.len(), map.key_slice().len());
    assert_eq!(map.key_slice().len(), map.value_slice().len());
    assert_eq!(map.as_set().as_slice(), map.key_slice());
    let mut seen = BTreeSet::new();
    for (pos, &key) in map.key_slice().iter().enumerate() {
        assert!(seen.insert(key.index()));
        assert_eq!(map.position(key), Some(pos));
        assert!(std::ptr::eq(map.get(key).unwrap(), &map.value_slice()[pos]));
        let (stored, value) = map.occupant(key.index()).unwrap();
        assert_eq!(*stored, key);
        assert!(std::ptr::eq(value, &map.value_slice()[pos]));
    }
    for &key in probes {
        let pos = map.key_slice().iter().position(|&stored| stored == key);
        assert_eq!(map.position(key), pos);
        assert_eq!(map.contains_key(key), pos.is_some());
        assert_eq!(map.get(key).is_some(), pos.is_some());
        assert_eq!(map.as_set().contains(key), pos.is_some());
        let occupant = map
            .key_slice()
            .iter()
            .position(|stored| stored.index() == key.index());
        assert_eq!(
            map.occupant(key.index()).map(|(&k, _)| k),
            occupant.map(|i| map.key_slice()[i])
        );
        assert_eq!(
            map.as_set().occupant(key.index()),
            occupant.map(|i| &map.key_slice()[i])
        );
    }
}

mod equality_panic_tests {
    use super::*;
    use std::cell::Cell;

    thread_local! {
        static PANIC_ON_EQ: Cell<bool> = const { Cell::new(false) };
    }

    #[derive(Clone, Copy, Debug)]
    struct PanicKey {
        identity: Gen,
        // Ignored by equality so equal keys can still be distinguished in
        // the stored slices when testing insert versus replace.
        tag: u8,
    }

    impl Key for PanicKey {
        fn index(self) -> usize {
            self.identity.index()
        }
    }

    impl PartialEq for PanicKey {
        fn eq(&self, other: &Self) -> bool {
            PANIC_ON_EQ.with(|flag| assert!(!flag.get(), "key equality panic"));
            self.identity == other.identity
        }
    }

    impl Eq for PanicKey {}

    struct EqPanicGuard;

    impl EqPanicGuard {
        fn arm() -> Self {
            PANIC_ON_EQ.with(|flag| flag.set(true));
            Self
        }
    }

    impl Drop for EqPanicGuard {
        fn drop(&mut self) {
            PANIC_ON_EQ.with(|flag| flag.set(false));
        }
    }

    fn occupied_slot<const PAGE: usize, D: Directory>() {
        let old = PanicKey {
            identity: Gen {
                index: 7,
                generation: 0,
            },
            tag: 1,
        };
        let other = PanicKey {
            identity: Gen {
                index: 47,
                generation: 2,
            },
            tag: 2,
        };
        for generation in [0, 1] {
            let candidate = PanicKey {
                identity: Gen {
                    generation,
                    ..old.identity
                },
                tag: 3,
            };
            let probes: Vec<_> = gen_probes()
                .into_iter()
                .map(|identity| PanicKey { identity, tag: 0 })
                .collect();
            for operation in 0..4 {
                let mut map: PagedSparseMap<PanicKey, u32, PAGE, D> =
                    [(other, 20), (old, 10)].into_iter().collect();
                let keys: Vec<_> = map.keys().map(|k| (k.identity, k.tag)).collect();
                let values = map.value_slice().to_vec();
                let positions: Vec<_> = map.keys().map(|&k| map.position(k)).collect();
                let pages = map.allocated_pages();
                let mut previous = None;
                let mut refused = None;
                let mut makes = 0;
                let result = catch_unwind(AssertUnwindSafe(|| {
                    let _guard = EqPanicGuard::arm();
                    match operation {
                        0 => {
                            map.insert(candidate, 90);
                        }
                        1 => {
                            previous = map
                                .replace(candidate, 90)
                                .map(|(k, v)| (k.identity, k.tag, v));
                        }
                        2 => {
                            refused = map.try_insert(candidate, 90).err();
                        }
                        _ => {
                            map.get_or_insert_with(candidate, || {
                                makes += 1;
                                90
                            });
                        }
                    }
                }));
                assert!(!PANIC_ON_EQ.with(Cell::get));
                assert_eq!(makes, 0);
                if operation == 1 {
                    // Replace never compares keys and must complete even
                    // with equality armed, storing the new key and value.
                    assert!(result.is_ok());
                    assert_eq!(previous, Some((old.identity, old.tag, 10)));
                    assert_eq!(map.key_slice()[1].identity, candidate.identity);
                    assert_eq!(map.key_slice()[1].tag, candidate.tag);
                    assert_eq!(map.value_slice(), &[20, 90]);
                    assert_eq!(map.position(candidate), Some(1));
                } else {
                    // Try-insert refuses without equality; insert and the
                    // lazy insertion compare before making any changes.
                    if operation == 2 {
                        assert!(result.is_ok());
                        assert_eq!(refused, Some(90));
                    } else {
                        assert!(result.is_err());
                    }
                    let actual: Vec<_> = map.keys().map(|k| (k.identity, k.tag)).collect();
                    assert_eq!(actual, keys);
                    assert_eq!(map.value_slice(), values);
                    assert_eq!(map.position(old), Some(1));
                }
                let actual_positions: Vec<_> = map.keys().map(|&k| map.position(k)).collect();
                assert_eq!(actual_positions, positions);
                assert_eq!(map.key_slice()[0].identity, other.identity);
                assert_eq!(map.key_slice()[0].tag, other.tag);
                assert_eq!(map.get(other), Some(&20));
                assert_eq!(map.allocated_pages(), pages);
                assert_consistent(&map, &probes);
            }
        }

        // With non-panicking equality, insert keeps the equal stored key
        // while replace writes the supplied key, including its ignored tag.
        let equal = PanicKey { tag: 9, ..old };
        let mut map: PagedSparseMap<PanicKey, u32, PAGE, D> =
            [(other, 20), (old, 10)].into_iter().collect();
        assert_eq!(map.insert(equal, 30), Some(10));
        assert_eq!(map.key_slice()[1].tag, old.tag);
        assert_eq!(map.position(equal), Some(1));
        assert_eq!(map.get(equal), Some(&30));
        let (previous, value) = map.replace(equal, 40).unwrap();
        assert_eq!(previous.tag, old.tag);
        assert_eq!(value, 30);
        assert_eq!(map.key_slice()[1].tag, equal.tag);
        assert_eq!(map.position(equal), Some(1));
        assert_eq!(map.get(equal), Some(&40));
        assert_consistent(&map, &[old, other, equal]);
    }

    #[test]
    fn panicking_key_equality_keeps_occupied_slot_operations_consistent() {
        occupied_slot::<1, Flat>();
        occupied_slot::<3, Flat>();
        occupied_slot::<4, Flat>();
        occupied_slot::<8, Flat>();
        occupied_slot::<1, Tree>();
        occupied_slot::<1, Radix>();
        occupied_slot::<4, Tree>();
        occupied_slot::<4, Radix>();
    }
}

fn panicking_sorts<const PAGE: usize, D: Directory>() {
    // Sorting these shuffled indices needs more than 50 comparisons even
    // under Miri. Panic after different amounts of work on the permutation.
    let entries: Vec<_> = (0..64_u16)
        .map(|i| {
            let index = (i * 37 + 13) % 64;
            (
                Gen {
                    index,
                    generation: 0,
                },
                u32::from(63 - index),
            )
        })
        .collect();
    for panic_after in [0, 1, 10, 50] {
        for by_key in [false, true] {
            let mut map: PagedSparseMap<Gen, u32, PAGE, D> = entries.iter().copied().collect();
            let keys = map.key_slice().to_vec();
            let values = map.value_slice().to_vec();
            let mut calls = 0;
            let result = catch_unwind(AssertUnwindSafe(|| {
                if by_key {
                    map.sort_unstable_by_key(|key, value| {
                        calls += 1;
                        assert!(calls <= panic_after, "sort key panic");
                        (*value, *key)
                    });
                } else {
                    map.sort_unstable_by(|(a, va), (b, vb)| {
                        calls += 1;
                        assert!(calls <= panic_after, "comparator panic");
                        (va, a).cmp(&(vb, b))
                    });
                }
            }));
            assert!(result.is_err(), "sort finished within {panic_after} calls");
            assert_eq!(map.key_slice(), keys);
            assert_eq!(map.value_slice(), values);
            assert_consistent(&map, &gen_probes());
            map.sort_unstable_keys();
            assert_consistent(&map, &gen_probes());
        }
    }
}

#[test]
fn panicking_comparator_and_sort_key_leave_map_unchanged() {
    panicking_sorts::<1, Flat>();
    panicking_sorts::<3, Flat>();
    panicking_sorts::<4, Tree>();
    panicking_sorts::<4, Radix>();
}

fn panicking_retain<const PAGE: usize, D: Directory>() -> TestCaseResult {
    let entries: Vec<_> = (0..12_u16)
        .map(|index| {
            (
                Gen {
                    index,
                    generation: 0,
                },
                u32::from(index) * 10,
            )
        })
        .collect();
    // First call, after removals, and the final call. Track completed effects
    // independently of visit order; the panicking call's mutation survives.
    for panic_after in [0, 1, 5, 11] {
        let mut map: PagedSparseMap<Gen, u32, PAGE, D> = entries.iter().copied().collect();
        let mut model = Model::new();
        for &(key, value) in &entries {
            model.replace::<PAGE>(key, value);
        }
        let mut visited = BTreeSet::new();
        let result = catch_unwind(AssertUnwindSafe(|| {
            map.retain(|&key, value| {
                assert!(visited.insert(key));
                *value += 1;
                *model.get_mut(key).unwrap() += 1;
                assert!(visited.len() <= panic_after, "predicate panic");
                let keep = key.index % 2 == 0;
                if !keep {
                    model.remove(key);
                }
                keep
            });
        }));
        assert!(result.is_err());
        assert_eq!(visited.len(), panic_after + 1);
        for &(key, value) in &entries {
            if !visited.contains(&key) {
                assert_eq!(map.get(key), Some(&value), "unvisited entry {key:?}");
            }
        }
        model.adopt_permutation(&map, "panicking retain")?;
        let probes = gen_probes().into_iter().collect();
        let indices = (0_usize..52).collect();
        check_invariants(&mut map, &model, &probes, &indices, "panicking retain")?;
        map.retain(|_, value| {
            *value += 1;
            true
        });
        assert_consistent(&map, &gen_probes());
    }
    Ok(())
}

#[test]
fn panicking_retain_keeps_unvisited_entries_and_mutations() -> TestCaseResult {
    panicking_retain::<1, Flat>()?;
    panicking_retain::<3, Flat>()?;
    panicking_retain::<4, Tree>()?;
    panicking_retain::<4, Radix>()?;
    Ok(())
}

fn panicking_make<const PAGE: usize, D: Directory>() {
    let old = Gen {
        index: 1,
        generation: 0,
    };
    let new = Gen {
        generation: 1,
        ..old
    };
    let absent = Gen {
        index: 47,
        generation: 2,
    };
    let mut map: PagedSparseMap<Gen, u32, PAGE, D> = [(old, 10)].into_iter().collect();
    let keys = map.key_slice().to_vec();
    let values = map.value_slice().to_vec();
    let pages = map.allocated_pages();
    assert_eq!(*map.get_or_insert_with(old, || panic!("must not run")), 10);
    for key in [new, absent] {
        let result = catch_unwind(AssertUnwindSafe(|| {
            map.get_or_insert_with(key, || panic!("make panic"));
        }));
        assert!(result.is_err());
        assert_eq!(map.key_slice(), keys);
        assert_eq!(map.value_slice(), values);
        assert_eq!(map.allocated_pages(), pages);
        assert_consistent(&map, &gen_probes());
    }
    assert_eq!(*map.get_or_insert_with(new, || 20), 20);
    assert_consistent(&map, &gen_probes());
}

#[test]
fn panicking_get_or_insert_with_make_leaves_map_unchanged() {
    panicking_make::<1, Flat>();
    panicking_make::<3, Flat>();
    panicking_make::<4, Tree>();
    panicking_make::<4, Radix>();
}

mod drop_tests {
    use super::*;
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;

    #[derive(Default)]
    struct DropTracker {
        counts: Rc<RefCell<Vec<usize>>>,
        panic_on_drop: Rc<Cell<Option<usize>>>,
    }

    impl DropTracker {
        fn make(&self) -> DropValue {
            let id = self.counts.borrow().len();
            self.counts.borrow_mut().push(0);
            DropValue {
                id,
                counts: Rc::clone(&self.counts),
                panic_on_drop: Rc::clone(&self.panic_on_drop),
            }
        }

        fn assert_all_dropped_once(&self) {
            assert!(self.counts.borrow().iter().all(|&count| count == 1));
        }
    }

    struct DropValue {
        id: usize,
        counts: Rc<RefCell<Vec<usize>>>,
        panic_on_drop: Rc<Cell<Option<usize>>>,
    }

    impl Drop for DropValue {
        fn drop(&mut self) {
            self.counts.borrow_mut()[self.id] += 1;
            if self.panic_on_drop.get() == Some(self.id) {
                // Disarm before unwinding so other values can drop normally.
                self.panic_on_drop.set(None);
                panic!("value {} drop panic", self.id);
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum During {
        Retain,
        Clear,
        Eviction,
    }

    fn panicking_drop<const PAGE: usize, D: Directory>() {
        for during in [During::Retain, During::Clear, During::Eviction] {
            let tracker = DropTracker::default();
            let mut map = PagedSparseMap::<Gen, DropValue, PAGE, D>::new();
            for index in 0..6 {
                map.insert(
                    Gen {
                        index,
                        generation: 0,
                    },
                    tracker.make(),
                );
            }
            let pages = map.allocated_pages();
            let old = Gen {
                index: 1,
                generation: 0,
            };
            let new = Gen {
                generation: 1,
                ..old
            };
            tracker.panic_on_drop.set(Some(1));
            let mut visited = BTreeSet::new();
            let result = catch_unwind(AssertUnwindSafe(|| match during {
                During::Retain => map.retain(|&key, _| {
                    assert!(visited.insert(key));
                    key != old
                }),
                During::Clear => map.clear(),
                During::Eviction => {
                    map.get_or_insert_with(new, || tracker.make());
                }
            }));
            assert!(result.is_err(), "{during:?}");
            assert_eq!(tracker.panic_on_drop.get(), None);
            assert_consistent(&map, &gen_probes());
            assert_eq!(map.allocated_pages(), pages);
            assert_eq!(tracker.counts.borrow()[1], 1);
            match during {
                During::Retain => {
                    assert_eq!(map.len(), 5);
                    assert!(visited.contains(&old));
                    assert!(!map.contains_key(old));
                    for (&key, value) in &map {
                        assert_eq!(value.id, key.index());
                        assert_eq!(tracker.counts.borrow()[value.id], 0);
                    }
                    // Every other entry must survive, including all entries
                    // the predicate has not visited when the drop panics.
                    for index in 0..6 {
                        let key = Gen {
                            index,
                            generation: 0,
                        };
                        if key != old {
                            assert_eq!(map.get(key).unwrap().id, usize::from(index));
                        }
                    }
                }
                During::Clear => {
                    assert!(map.is_empty());
                    tracker.assert_all_dropped_once();
                }
                During::Eviction => {
                    assert_eq!(map.len(), 6);
                    assert_eq!(map.position(new), Some(1));
                    assert_eq!(map.get(new).unwrap().id, 6);
                    assert!(!map.contains_key(old));
                    for (&key, value) in &map {
                        assert_eq!(value.id, if key == new { 6 } else { key.index() });
                        assert_eq!(tracker.counts.borrow()[value.id], 0);
                    }
                }
            }
            // Reuse the map after the caught panic and then destroy it. The
            // ledger covers every constructed value, including the evictee.
            let fresh = Gen {
                index: 47,
                generation: 2,
            };
            let id = tracker.counts.borrow().len();
            assert!(map.insert(fresh, tracker.make()).is_none());
            assert_eq!(map.get(fresh).unwrap().id, id);
            assert_consistent(&map, &gen_probes());
            drop(map);
            tracker.assert_all_dropped_once();
        }
    }

    #[test]
    fn panicking_drop_during_retain_clear_and_eviction_keeps_map_consistent() {
        panicking_drop::<1, Flat>();
        panicking_drop::<3, Flat>();
        panicking_drop::<4, Tree>();
        panicking_drop::<4, Radix>();
    }

    fn count_drops<const PAGE: usize, D: Directory>() {
        let tracker = DropTracker::default();
        let a = Gen {
            index: 1,
            generation: 0,
        };
        let b = Gen {
            index: 2,
            generation: 0,
        };
        let c = Gen {
            index: 3,
            generation: 0,
        };
        let next = Gen { generation: 1, ..a };
        let mut map = PagedSparseMap::<Gen, DropValue, PAGE, D>::new();
        assert!(map.insert(a, tracker.make()).is_none()); // id 0
        assert!(map.insert(b, tracker.make()).is_none()); // id 1
        let old = map.insert(next, tracker.make()).unwrap(); // id 2 replaces 0
        assert_eq!(old.id, 0);
        assert_eq!(tracker.counts.borrow()[0], 0);
        drop(old);
        let (key, old) = map.replace(a, tracker.make()).unwrap(); // id 3 replaces 2
        assert_eq!(key, next);
        assert_eq!(old.id, 2);
        drop(old);
        let removed = map.remove(b).unwrap();
        assert_eq!(removed.id, 1);
        drop(removed);
        // id 4 is refused.
        let refused = match map.try_insert(next, tracker.make()) {
            Ok(_) => panic!("occupied slot accepted"),
            Err(value) => value,
        };
        assert_eq!(refused.id, 4);
        drop(refused);
        assert_eq!(map.get_or_insert_with(a, || panic!("must not run")).id, 3);
        assert_eq!(map.get_or_insert_with(next, || tracker.make()).id, 5);
        assert_eq!(tracker.counts.borrow()[3], 1);
        let (key, value) = map.remove_entry(next).unwrap();
        assert_eq!(key, next);
        assert_eq!(value.id, 5);
        drop(value);
        assert!(map.insert(c, tracker.make()).is_none()); // id 6
        let (key, value) = map.remove_occupant(c.index()).unwrap();
        assert_eq!(key, c);
        assert_eq!(value.id, 6);
        drop(value);
        assert!(map.insert(a, tracker.make()).is_none()); // id 7
        assert!(map.insert(b, tracker.make()).is_none()); // id 8
        map.clear();
        tracker.assert_all_dropped_once();
        assert!(map.insert(a, tracker.make()).is_none()); // id 9
        map.reset();
        tracker.assert_all_dropped_once();
        assert!(map.insert(b, tracker.make()).is_none()); // id 10
        map.retain(|_, _| false);
        tracker.assert_all_dropped_once();
        assert!(map.insert(c, tracker.make()).is_none()); // id 11
        assert!(map.insert(a, tracker.make()).is_none()); // id 12
        assert_consistent(&map, &gen_probes());
        drop(map);
        tracker.assert_all_dropped_once();
    }

    #[test]
    fn every_value_drops_once_across_replacement_removal_clear_and_map_drop() {
        count_drops::<1, Flat>();
        count_drops::<3, Flat>();
        count_drops::<4, Tree>();
        count_drops::<4, Radix>();
    }
}
