//! Model-based tests for slot identity, dense order, lazy clearing, and traits.
//!
//! The oracle owns a map from slot indices to keys and a separate dense vector.
//! It models swap removal exactly, but accepts any valid dense permutation after
//! retain or sorting. Exhaustive small-domain probes and historical wide keys
//! keep checking stale slots after clear, removal, and generation replacement.
//! Set algebra uses a `BTreeSet` oracle for plain integer keys.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;

use paged_collections::key::Key;
use paged_collections::page::{Directory, Flat, Radix, Tree};
use paged_collections::sparse_set::{PagedSparseSet, Slot};
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

/// Upper bound for generated op sequences and key lists.
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
}

impl Predicate {
    fn keeps<K: TestKey>(self, key: K) -> bool {
        let matches = key.index() % usize::from(self.modulus)
            == usize::from(self.residue % self.modulus)
            && self.generations & (1_u8 << key.generation()) != 0;
        matches ^ self.invert
    }
}

#[derive(Clone, Debug)]
enum Op<K> {
    Insert(K),
    Replace(K),
    Remove(K),
    RemoveOccupant(usize),
    Clear,
    Reset,
    Retain(Predicate),
    SortUnstable,
    SortUnstableByKey(u8),
    SortUnstableBy(bool),
    ShrinkToFit,
    Reserve(u8),
    Extend(Vec<K>),
    Query(K),
    Occupant(usize),
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
    let predicates = (1_u8..8, 0_u8..8, 0_u8..8, any::<bool>()).prop_map(
        |(modulus, residue, generations, invert)| Predicate {
            modulus,
            residue,
            generations,
            invert,
        },
    );
    prop_oneof![
        12 => keys.clone().prop_map(Op::Insert),
        8 => keys.clone().prop_map(Op::Replace),
        7 => keys.clone().prop_map(Op::Remove),
        5 => indices.clone().prop_map(Op::RemoveOccupant),
        1 => Just(Op::Clear),
        1 => Just(Op::Reset),
        4 => predicates.prop_map(Op::Retain),
        2 => Just(Op::SortUnstable),
        2 => (1_u8..8).prop_map(Op::SortUnstableByKey),
        2 => any::<bool>().prop_map(Op::SortUnstableBy),
        2 => Just(Op::ShrinkToFit),
        1 => (0_u8..32).prop_map(Op::Reserve),
        2 => prop::collection::vec(keys.clone(), 0..6).prop_map(Op::Extend),
        5 => keys.prop_map(Op::Query),
        3 => indices.prop_map(Op::Occupant),
    ]
    .boxed()
}

struct Model<K> {
    slots: BTreeMap<usize, K>,
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

    fn insert<const PAGE: usize>(&mut self, key: K) -> bool {
        if self.slots.contains_key(&key.index()) {
            return false;
        }
        self.slots.insert(key.index(), key);
        self.dense.push(key);
        self.touched_pages.insert(key.index() / PAGE);
        true
    }

    fn replace<const PAGE: usize>(&mut self, key: K) -> Option<K> {
        if let Some(old) = self.slots.get_mut(&key.index()) {
            let previous = *old;
            *old = key;
            let pos = self.dense.iter().position(|&k| k == previous).unwrap();
            self.dense[pos] = key;
            Some(previous)
        } else {
            self.insert::<PAGE>(key);
            None
        }
    }

    fn remove_occupant(&mut self, index: usize) -> Option<K> {
        let key = self.slots.remove(&index)?;
        let pos = self.dense.iter().position(|&k| k == key).unwrap();
        self.dense.swap_remove(pos);
        Some(key)
    }

    fn remove(&mut self, key: K) -> bool {
        if self.slots.get(&key.index()) != Some(&key) {
            return false;
        }
        self.remove_occupant(key.index());
        true
    }

    fn clear(&mut self) {
        self.slots.clear();
        self.dense.clear();
    }

    fn adopt_permutation(&mut self, keys: &[K], context: &str) -> TestCaseResult {
        let actual: BTreeMap<_, _> = keys.iter().map(|&key| (key.index(), key)).collect();
        prop_assert_eq!(keys.len(), self.slots.len(), "{}", context);
        prop_assert_eq!(&actual, &self.slots, "{}", context);
        self.dense = keys.to_vec();
        Ok(())
    }
}

fn check_invariants<K: TestKey, const PAGE: usize, D: Directory>(
    set: &PagedSparseSet<K, PAGE, D>,
    model: &Model<K>,
    probes: &BTreeSet<K>,
    indices: &BTreeSet<usize>,
    context: &str,
) -> TestCaseResult {
    prop_assert_eq!(set.len(), model.slots.len(), "{}", context);
    prop_assert_eq!(set.is_empty(), model.slots.is_empty(), "{}", context);
    prop_assert_eq!(set.as_slice(), model.dense.as_slice(), "{}", context);
    prop_assert!(set.capacity() >= set.len(), "{}", context);
    let mut seen = BTreeSet::new();
    for (pos, &key) in set.as_slice().iter().enumerate() {
        prop_assert!(seen.insert(key.index()), "duplicate index: {}", context);
        prop_assert_eq!(model.slots.get(&key.index()), Some(&key), "{}", context);
        prop_assert_eq!(set.position(key), Some(pos), "{}", context);
        prop_assert_eq!(set.occupant(key.index()), Some(&key), "{}", context);
        prop_assert!(set.contains(key), "{}", context);
    }
    for &key in model.slots.values() {
        prop_assert!(
            set.contains(key),
            "missing model key {:?}: {}",
            key,
            context
        );
    }
    for &key in probes {
        let expected = model.slots.get(&key.index()) == Some(&key);
        prop_assert_eq!(set.contains(key), expected, "probe {:?}: {}", key, context);
        let position = model.dense.iter().position(|&stored| stored == key);
        prop_assert_eq!(set.position(key), position, "probe {:?}: {}", key, context);
    }
    for &index in indices {
        prop_assert_eq!(set.occupant(index), model.slots.get(&index), "{}", context);
    }
    prop_assert!(
        set.iter().copied().eq(model.dense.iter().copied()),
        "{}",
        context
    );
    prop_assert!(
        set.into_iter().copied().eq(model.dense.iter().copied()),
        "{}",
        context
    );
    let live_pages: BTreeSet<_> = model.slots.keys().map(|index| index / PAGE).collect();
    prop_assert!(set.allocated_pages() >= live_pages.len(), "{}", context);
    prop_assert!(
        set.allocated_pages() <= model.touched_pages.len(),
        "too many pages: {}",
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
    // Pre-register all past and future keys, so cleared keys remain probes for
    // the entire run, including after their old dense positions are reused.
    for op in ops {
        match op {
            Op::Insert(key) | Op::Replace(key) | Op::Remove(key) | Op::Query(key) => {
                probes.insert(*key);
            }
            Op::Extend(keys) => probes.extend(keys.iter().copied()),
            Op::RemoveOccupant(index) | Op::Occupant(index) => {
                indices.insert(*index);
            }
            _ => {}
        }
    }
    for &key in &probes {
        let index = key.index();
        indices.extend([index, index.saturating_sub(1), index.saturating_add(1)]);
    }

    let mut set = PagedSparseSet::<K, PAGE, D>::new();
    let mut model = Model::new();
    check_invariants(&set, &model, &probes, &indices, "initial state")?;
    for (step, op) in ops.iter().enumerate() {
        let context = format!("step {step}: {op:?}");
        let pages_before = set.allocated_pages();
        match op {
            Op::Insert(key) => {
                let expected = model.insert::<PAGE>(*key);
                prop_assert_eq!(set.insert(*key), expected, "{}", context);
                if !expected {
                    prop_assert_eq!(set.allocated_pages(), pages_before, "{}", context);
                }
            }
            Op::Replace(key) => {
                prop_assert_eq!(
                    set.replace(*key),
                    model.replace::<PAGE>(*key),
                    "{}",
                    context
                );
            }
            Op::Remove(key) => {
                prop_assert_eq!(set.remove(*key), model.remove(*key), "{}", context);
            }
            Op::RemoveOccupant(index) => {
                prop_assert_eq!(
                    set.remove_occupant(*index),
                    model.remove_occupant(*index),
                    "{}",
                    context
                );
            }
            Op::Clear => {
                set.clear();
                model.clear();
                prop_assert_eq!(set.allocated_pages(), pages_before, "{}", context);
            }
            Op::Reset => {
                set.reset();
                model.clear();
                model.touched_pages.clear();
                prop_assert_eq!(set.allocated_pages(), 0, "{}", context);
            }
            Op::Retain(predicate) => {
                let before: BTreeSet<_> = model.slots.keys().copied().collect();
                let mut calls = BTreeMap::<usize, usize>::new();
                set.retain(|&key| {
                    *calls.entry(key.index()).or_default() += 1;
                    predicate.keeps(key)
                });
                prop_assert_eq!(
                    calls.keys().copied().collect::<BTreeSet<_>>(),
                    before,
                    "{}",
                    context
                );
                prop_assert!(calls.values().all(|&count| count == 1), "{}", context);
                model.slots.retain(|_, key| predicate.keeps(*key));
                model.adopt_permutation(set.as_slice(), &context)?;
            }
            Op::SortUnstable => {
                set.sort_unstable();
                prop_assert!(
                    set.as_slice().windows(2).all(|w| w[0] <= w[1]),
                    "{}",
                    context
                );
                model.adopt_permutation(set.as_slice(), &context)?;
            }
            Op::SortUnstableByKey(modulus) => {
                let sort_key = |key: &K| (key.index() % usize::from(*modulus), key.generation());
                set.sort_unstable_by_key(sort_key);
                prop_assert!(
                    set.as_slice()
                        .windows(2)
                        .all(|w| sort_key(&w[0]) <= sort_key(&w[1])),
                    "{}",
                    context
                );
                model.adopt_permutation(set.as_slice(), &context)?;
            }
            Op::SortUnstableBy(descending) => {
                set.sort_unstable_by(|a, b| if *descending { b.cmp(a) } else { a.cmp(b) });
                prop_assert!(
                    set.as_slice().windows(2).all(|w| if *descending {
                        w[0] >= w[1]
                    } else {
                        w[0] <= w[1]
                    }),
                    "{}",
                    context
                );
                model.adopt_permutation(set.as_slice(), &context)?;
            }
            Op::ShrinkToFit => {
                set.shrink_to_fit();
                if model.slots.is_empty() {
                    model.touched_pages.clear();
                    prop_assert_eq!(set.allocated_pages(), 0, "{}", context);
                }
            }
            Op::Reserve(additional) => {
                set.reserve(usize::from(*additional));
                prop_assert!(
                    set.capacity() >= set.len() + usize::from(*additional),
                    "{}",
                    context
                );
            }
            Op::Extend(keys) => {
                set.extend(keys.iter().copied());
                for &key in keys {
                    model.insert::<PAGE>(key);
                }
            }
            Op::Query(key) => {
                prop_assert_eq!(
                    set.contains(*key),
                    model.slots.get(&key.index()) == Some(key),
                    "{}",
                    context
                );
                prop_assert_eq!(
                    set.position(*key),
                    model.dense.iter().position(|k| k == key),
                    "{}",
                    context
                );
            }
            Op::Occupant(index) => {
                prop_assert_eq!(set.occupant(*index), model.slots.get(index), "{}", context);
            }
        }
        check_invariants(&set, &model, &probes, &indices, &context)?;
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

    fn algebra_keys() -> BoxedStrategy<u32> {
        prop_oneof![
            16 => 0_u32..200,
            1 => any::<u32>(),
            1 => Just(u32::MAX),
            1 => Just(1_u32 << 31),
        ]
        .boxed()
    }

    fn check_algebra_keys(
        actual: impl Iterator<Item = u32>,
        expected: &BTreeSet<u32>,
        context: &str,
    ) -> TestCaseResult {
        let mut actual: Vec<_> = actual.collect();
        prop_assert_eq!(actual.len(), expected.len(), "{}", context);
        actual.sort_unstable();
        prop_assert_eq!(
            actual,
            expected.iter().copied().collect::<Vec<_>>(),
            "{}",
            context
        );
        Ok(())
    }

    proptest! {
        #![proptest_config(config())]
        #[test]
        fn set_algebra(
            left_keys in prop::collection::vec(algebra_keys(), 0..max_len()),
            right_keys in prop::collection::vec(algebra_keys(), 0..max_len()),
        ) {
            let left: PagedSparseSet<u32, 4, Tree> = left_keys.iter().copied().collect();
            let right: PagedSparseSet<u32, 64, Radix> = right_keys.iter().copied().collect();
            let model_left: BTreeSet<_> = left_keys.into_iter().collect();
            let model_right: BTreeSet<_> = right_keys.into_iter().collect();

            let union = &model_left | &model_right;
            let intersection = &model_left & &model_right;
            let difference = &model_left - &model_right;
            let symmetric = &model_left ^ &model_right;

            check_algebra_keys(left.union(&right).copied(), &union, "union")?;
            check_algebra_keys(left.intersection(&right).copied(), &intersection, "intersection")?;
            check_algebra_keys(left.difference(&right).copied(), &difference, "difference")?;
            check_algebra_keys(
                left.symmetric_difference(&right).copied(),
                &symmetric,
                "symmetric difference",
            )?;

            prop_assert_eq!(left.is_disjoint(&right), model_left.is_disjoint(&model_right));
            prop_assert_eq!(left.is_subset(&right), model_left.is_subset(&model_right));
            prop_assert_eq!(left.is_superset(&right), model_left.is_superset(&model_right));

            let check_set = |set: &PagedSparseSet<u32, 4, Tree>, expected: &BTreeSet<u32>, context| {
                check_algebra_keys(set.iter().copied(), expected, context)?;
                for (pos, &key) in set.iter().enumerate() {
                    prop_assert_eq!(set.position(key), Some(pos), "{}: key {}", context, key);
                }
                for &key in &union {
                    prop_assert_eq!(
                        set.contains(key),
                        expected.contains(&key),
                        "{}: key {}",
                        context,
                        key
                    );
                }
                Ok(())
            };

            check_set(&(&left | &right), &union, "|")?;
            check_set(&(&left & &right), &intersection, "&")?;
            check_set(&(&left - &right), &difference, "-")?;
            check_set(&(&left ^ &right), &symmetric, "^")?;

            let mut assigned = left.clone();
            assigned |= &right;
            check_set(&assigned, &union, "|=")?;
            let mut assigned = left.clone();
            assigned &= &right;
            check_set(&assigned, &intersection, "&=")?;
            let mut assigned = left.clone();
            assigned -= &right;
            check_set(&assigned, &difference, "-=")?;
            let mut assigned = left;
            assigned ^= &right;
            check_set(&assigned, &symmetric, "^=")?;
        }
    }
}

#[cfg(target_pointer_width = "64")]
sequence_test!(u64_tree_4, u64, 4, Tree, wide_keys(), wide_probes());
#[cfg(target_pointer_width = "64")]
sequence_test!(u64_radix_4, u64, 4, Radix, wide_keys(), wide_probes());

fn check_traits<K: TestKey, const PAGE: usize, D: Directory>(keys: &[K]) -> TestCaseResult
where
    D::Store<Slot, PAGE>: Clone,
{
    let mut repeated = PagedSparseSet::<K, PAGE, D>::new();
    let mut model = Model::new();
    for &key in keys {
        prop_assert_eq!(repeated.insert(key), model.insert::<PAGE>(key));
    }
    let probes = keys.iter().copied().collect();
    let indices = keys.iter().map(|key| key.index()).collect();
    check_invariants(&repeated, &model, &probes, &indices, "repeated insert")?;

    let collected: PagedSparseSet<K, PAGE, D> = keys.iter().copied().collect();
    check_invariants(&collected, &model, &probes, &indices, "collect")?;
    prop_assert_eq!(&collected, &repeated);

    let mut extended = PagedSparseSet::<K, PAGE, D>::default();
    let mid = keys.len() / 2;
    extended.extend(keys[..mid].iter().copied());
    extended.extend(keys[mid..].iter());
    check_invariants(
        &extended,
        &model,
        &probes,
        &indices,
        "extend owned and borrowed",
    )?;
    prop_assert_eq!(&extended, &repeated);

    let mut cloned = repeated.clone();
    prop_assert_eq!(&cloned, &repeated);
    prop_assert_eq!(cloned.allocated_pages(), repeated.allocated_pages());
    check_invariants(&cloned, &model, &probes, &indices, "clone")?;
    if let Some(&first) = repeated.as_slice().first() {
        prop_assert!(cloned.remove(first));
        prop_assert!(repeated.contains(first));
        prop_assert_ne!(&cloned, &repeated);
    }
    let reversed: PagedSparseSet<K, PAGE, D> = repeated.iter().rev().copied().collect();
    prop_assert_eq!(&reversed, &repeated);
    prop_assert_eq!(&repeated, &reversed);
    prop_assert_eq!(
        repeated.clone().into_iter().collect::<Vec<_>>(),
        repeated.as_slice()
    );
    check_invariants(
        &repeated,
        &model,
        &probes,
        &indices,
        "original after clone mutation",
    )?;
    Ok(())
}

macro_rules! trait_test {
    ($name:ident, $key:ty, $page:literal, $directory:ty, $keys:expr) => {
        proptest! {
            #![proptest_config(config())]
            #[test]
            fn $name(keys in prop::collection::vec($keys, 0..max_len())) {
                check_traits::<$key, $page, $directory>(&keys)?;
            }
        }
    };
}

trait_test!(traits_gen_flat_1, Gen, 1, Flat, gen_keys());
trait_test!(traits_gen_flat_3, Gen, 3, Flat, gen_keys());
trait_test!(traits_gen_flat_4, Gen, 4, Flat, gen_keys());
trait_test!(traits_gen_flat_8, Gen, 8, Flat, gen_keys());
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
    let a: PagedSparseSet<Gen, PAGE, D> = [old, other].into_iter().collect();
    let b: PagedSparseSet<Gen, PAGE, D> = [other, old].into_iter().collect();
    let c: PagedSparseSet<Gen, PAGE, D> = [new, other].into_iter().collect();
    assert_eq!(a, b);
    assert_ne!(a, c);
    assert_ne!(c, a);
}

#[test]
fn equality_distinguishes_generations_in_every_directory() {
    generation_equality::<1, Flat>();
    generation_equality::<3, Flat>();
    generation_equality::<4, Flat>();
    generation_equality::<8, Flat>();
    generation_equality::<4, Tree>();
    generation_equality::<4, Radix>();
}

#[test]
fn stale_slots_and_last_removal_in_every_directory() -> TestCaseResult {
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
    let ops = vec![
        Op::Insert(a),
        Op::Insert(b),
        Op::Insert(c),
        Op::Clear,
        Op::Insert(next_c),
        Op::Insert(next_a),
        Op::Insert(next_b),
        Op::Insert(a),
        Op::Remove(a),
        Op::Replace(a),
        Op::Remove(next_a),
        Op::Remove(next_b),
        Op::RemoveOccupant(c.index()),
        Op::Remove(a),
        Op::ShrinkToFit,
        Op::Insert(b),
        Op::Clear,
        Op::Insert(c),
        Op::Reset,
        Op::Insert(a),
        Op::RemoveOccupant(a.index()),
        Op::ShrinkToFit,
    ];
    run_sequence::<Gen, 1, Flat>(&ops, gen_probes())?;
    run_sequence::<Gen, 3, Flat>(&ops, gen_probes())?;
    run_sequence::<Gen, 4, Flat>(&ops, gen_probes())?;
    run_sequence::<Gen, 8, Flat>(&ops, gen_probes())?;
    run_sequence::<Gen, 4, Tree>(&ops, gen_probes())?;
    run_sequence::<Gen, 4, Radix>(&ops, gen_probes())?;
    Ok(())
}

#[cfg(target_pointer_width = "64")]
#[test]
fn maximum_tree_index_with_single_slot_pages() -> TestCaseResult {
    let ops = vec![
        Op::Insert(0),
        Op::Insert(u64::MAX),
        Op::Insert(u64::MAX - 1),
        Op::Insert(1 << 63),
        Op::SortUnstable,
        Op::Remove(u64::MAX),
        Op::Clear,
        Op::Insert(u64::MAX),
        Op::Replace(u64::MAX),
        Op::RemoveOccupant(usize::MAX),
        Op::ShrinkToFit,
    ];
    run_sequence::<u64, 1, Tree>(&ops, wide_probes())?;
    run_sequence::<u64, 1, Radix>(&ops, wide_probes())?;
    run_sequence::<u64, 4, Tree>(&ops, wide_probes())?;
    run_sequence::<u64, 4, Radix>(&ops, wide_probes())?;
    Ok(())
}

#[test]
fn flat_directory_overflow_leaves_existing_keys_usable() {
    let mut set = PagedSparseSet::<usize, 1, Flat>::new();
    assert!(set.insert(3));
    // PAGE=1 makes usize::MAX + 1 fail before any huge allocation is tried.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        set.insert(usize::MAX);
    }));
    assert!(result.is_err());
    assert_eq!(set.as_slice(), &[3]);
    assert_eq!(set.position(3), Some(0));
    assert_eq!(set.allocated_pages(), 1);
    assert!(!set.contains(usize::MAX));
    assert!(set.insert(4));
    assert_eq!(set.position(4), Some(1));
}

#[cfg(feature = "serde")]
mod serde_tests {
    use super::*;

    fn round_trip<const PAGE: usize, D: Directory>(keys: &[Gen]) -> TestCaseResult {
        let original: PagedSparseSet<Gen, PAGE, D> = keys.iter().copied().collect();
        let json = serde_json::to_string(&original).unwrap();
        let sequence: Vec<Gen> = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(sequence.as_slice(), original.as_slice());
        let decoded: PagedSparseSet<Gen, PAGE, D> = serde_json::from_str(&json).unwrap();
        let mut model = Model::new();
        for &key in original.as_slice() {
            model.insert::<PAGE>(key);
        }
        model.adopt_permutation(decoded.as_slice(), "serde round trip")?;
        let probes = gen_probes().into_iter().collect();
        let indices = (0_usize..52).collect();
        check_invariants(&decoded, &model, &probes, &indices, "serde round trip")?;

        // Deserialize the unfiltered input as well: collisions must keep the
        // first generation, just like repeated insert and FromIterator.
        let input_json = serde_json::to_string(keys).unwrap();
        let from_input: PagedSparseSet<Gen, PAGE, D> = serde_json::from_str(&input_json).unwrap();
        prop_assert_eq!(&from_input, &original);
        prop_assert_eq!(from_input.as_slice(), original.as_slice());
        check_invariants(&from_input, &model, &probes, &indices, "serde collisions")?;
        Ok(())
    }

    macro_rules! serde_test {
        ($name:ident, $page:literal, $directory:ty) => {
            proptest! {
                #![proptest_config(config())]
                #[test]
                fn $name(keys in prop::collection::vec(gen_keys(), 0..max_len())) {
                    round_trip::<$page, $directory>(&keys)?;
                }
            }
        };
    }

    serde_test!(round_trip_flat_1, 1, Flat);
    serde_test!(round_trip_flat_3, 3, Flat);
    serde_test!(round_trip_flat_4, 4, Flat);
    serde_test!(round_trip_flat_8, 8, Flat);
    serde_test!(round_trip_tree_4, 4, Tree);
    serde_test!(round_trip_radix_4, 4, Radix);

    fn first_per_slot<const PAGE: usize, D: Directory>() {
        let first = Gen {
            index: 1,
            generation: 0,
        };
        let other = Gen {
            index: 2,
            generation: 2,
        };
        let input = [
            first,
            Gen {
                generation: 1,
                ..first
            },
            other,
            first,
            Gen {
                generation: 0,
                ..other
            },
        ];
        let json = serde_json::to_string(&input).unwrap();
        let set: PagedSparseSet<Gen, PAGE, D> = serde_json::from_str(&json).unwrap();
        assert_eq!(set.as_slice(), &[first, other]);
        assert_eq!(set.occupant(1), Some(&first));
        assert_eq!(set.occupant(2), Some(&other));
        assert!(!set.contains(input[1]));
        assert!(!set.contains(input[4]));
    }

    #[test]
    fn collisions_keep_first_generation() {
        first_per_slot::<1, Flat>();
        first_per_slot::<3, Flat>();
        first_per_slot::<4, Flat>();
        first_per_slot::<8, Flat>();
        first_per_slot::<4, Tree>();
        first_per_slot::<4, Radix>();
    }
}

/// Checks that every dense key is found at its position and nothing else is.
#[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
fn assert_consistent<const PAGE: usize, D: Directory>(
    set: &PagedSparseSet<u32, PAGE, D>,
    universe: u32,
) {
    let mut seen = BTreeSet::new();
    for (pos, &key) in set.as_slice().iter().enumerate() {
        assert!(seen.insert(key));
        assert_eq!(set.position(key), Some(pos));
    }
    for key in 0..universe {
        assert_eq!(set.contains(key), seen.contains(&key));
    }
}

#[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
#[test]
fn panicking_sort_comparator_leaves_set_consistent() {
    // A shuffled input (fixed LCG permutation) so the sort moves keys before
    // the comparator panics; a sorted or reversed input is detected as one run
    // and would not move anything first.
    let keys: Vec<u32> = (0..1000_u32).map(|i| (i * 7919 + 13) % 1000).collect();
    for panic_after in [0_usize, 1, 50, 500, 2000, 5000] {
        let mut set: PagedSparseSet<u32, 4> = keys.iter().copied().collect();
        let mut calls = 0;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            set.sort_unstable_by(|a, b| {
                calls += 1;
                assert!(calls <= panic_after, "comparator panic");
                a.cmp(b)
            });
        }));
        assert!(
            result.is_err(),
            "sort finished within {panic_after} comparisons"
        );
        assert_eq!(set.len(), 1000);
        assert_consistent(&set, 1010);
        // A key cannot be inserted twice after the caught panic.
        for k in 0..1000 {
            assert!(!set.insert(k));
        }
        assert_eq!(set.len(), 1000);
    }
}

#[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
#[test]
fn panicking_retain_predicate_leaves_set_consistent() {
    let mut set: PagedSparseSet<u32, 4> = (0..64).collect();
    let mut calls = 0;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        set.retain(|&k| {
            calls += 1;
            assert!(calls < 30, "predicate panic");
            k % 2 == 0
        });
    }));
    assert!(result.is_err());
    assert_consistent(&set, 70);
    // Re-inserting succeeds exactly for the keys the partial retain removed.
    for k in 0..64 {
        let present = set.contains(k);
        assert_eq!(set.insert(k), !present);
    }
    assert_eq!(set.len(), 64);
    assert_consistent(&set, 70);
}
