//! An exact, slot-ordered BTreeMap oracle for the direct-address map.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;
use std::hash::{BuildHasher, Hash, RandomState};
use std::iter::FusedIterator;
use std::panic::{AssertUnwindSafe, catch_unwind};

use paged_collections::direct_map::PagedDirectMap;
use paged_collections::key::{DirectKey, Key};
use paged_collections::page::{Directory, Flat, Radix, Tree};
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(into = "String", try_from = "String"))]
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

// A string representation lets JSON exercise generational map keys too.
#[cfg(feature = "serde")]
impl From<Gen> for String {
    fn from(key: Gen) -> Self {
        format!("{}:{}", key.index, key.generation)
    }
}

#[cfg(feature = "serde")]
impl TryFrom<String> for Gen {
    type Error = String;

    fn try_from(text: String) -> Result<Self, Self::Error> {
        let (index, generation) = text.split_once(':').ok_or("missing generation")?;
        Ok(Self {
            index: index.parse().map_err(|_| "invalid index")?,
            generation: generation.parse().map_err(|_| "invalid generation")?,
        })
    }
}

trait TestKey: DirectKey + Ord + Debug + Hash {
    /// A key outside every generated input, on a separate page at all sizes.
    fn unused() -> Self;

    fn generation(self) -> u8 {
        0
    }
}

impl TestKey for Gen {
    fn unused() -> Self {
        Self {
            index: 4096,
            generation: 3,
        }
    }

    fn generation(self) -> u8 {
        self.generation
    }
}

#[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
impl TestKey for u32 {
    fn unused() -> Self {
        4096
    }
}

fn slot_indices() -> BoxedStrategy<u16> {
    prop_oneof![6 => 0_u16..48, 2 => 1022_u16..1028, 2 => 2047_u16..2052].boxed()
}

fn probe_indices() -> Vec<usize> {
    (0..52).chain(1021..1029).chain(2046..2053).collect()
}

fn gen_keys() -> BoxedStrategy<Gen> {
    (slot_indices(), 0_u8..3)
        .prop_map(|(index, generation)| Gen { index, generation })
        .boxed()
}

fn gen_probes() -> Vec<Gen> {
    probe_indices()
        .into_iter()
        .flat_map(|index| {
            (0..4).map(move |generation| Gen {
                index: index as u16,
                generation,
            })
        })
        .collect()
}

#[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
fn integer_keys() -> BoxedStrategy<u32> {
    slot_indices().prop_map(u32::from).boxed()
}

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
    Probe(K),
    Remove(K),
    RemoveEntry(K),
    RemoveOccupant(usize),
    Retain(Predicate),
    Clear,
    Reset,
    ShrinkToFit,
}

fn operations<K: TestKey + 'static>(keys: BoxedStrategy<K>) -> BoxedStrategy<Op<K>> {
    let indices = prop_oneof![
        5 => keys.clone().prop_map(|key| key.index()),
        1 => Just(usize::MAX),
    ];
    let entries = (keys.clone(), any::<u32>());
    let predicates = (1_u8..8, 0_u8..8, 0_u8..8, any::<bool>(), any::<u32>()).prop_map(
        |(modulus, residue, generations, invert, delta)| Predicate {
            modulus,
            residue,
            generations,
            invert,
            delta,
        },
    );
    prop_oneof![
        12 => entries.clone().prop_map(|(k, v)| Op::Insert(k, v)),
        8 => entries.clone().prop_map(|(k, v)| Op::Replace(k, v)),
        6 => entries.clone().prop_map(|(k, v)| Op::TryInsert(k, v)),
        6 => entries.clone().prop_map(|(k, v)| Op::GetOrInsertWith(k, v)),
        5 => entries.prop_map(|(k, v)| Op::GetMut(k, v)),
        5 => (indices.clone(), any::<u32>()).prop_map(|(i, v)| Op::OccupantMut(i, v)),
        3 => keys.clone().prop_map(Op::Probe),
        5 => keys.clone().prop_map(Op::Remove),
        5 => keys.prop_map(Op::RemoveEntry),
        5 => indices.prop_map(Op::RemoveOccupant),
        4 => predicates.prop_map(Op::Retain),
        1 => Just(Op::Clear),
        1 => Just(Op::Reset),
        2 => Just(Op::ShrinkToFit),
    ]
    .boxed()
}

struct Model<K> {
    slots: BTreeMap<usize, (K, u32)>,
    pages: BTreeSet<usize>,
}

impl<K: TestKey> Model<K> {
    fn new() -> Self {
        Self {
            slots: BTreeMap::new(),
            pages: BTreeSet::new(),
        }
    }

    fn insert<const PAGE: usize>(&mut self, key: K, value: u32) -> Option<u32> {
        self.pages.insert(key.index() / PAGE);
        if let Some((stored, old)) = self.slots.get_mut(&key.index()) {
            if *stored != key {
                *stored = key;
            }
            Some(std::mem::replace(old, value))
        } else {
            self.slots.insert(key.index(), (key, value));
            None
        }
    }

    fn get(&self, key: K) -> Option<(K, u32)> {
        self.slots
            .get(&key.index())
            .copied()
            .filter(|(stored, _)| *stored == key)
    }

    fn get_mut(&mut self, key: K) -> Option<&mut u32> {
        self.slots
            .get_mut(&key.index())
            .filter(|(stored, _)| *stored == key)
            .map(|(_, value)| value)
    }

    fn remove(&mut self, key: K) -> Option<(K, u32)> {
        self.get(key)?;
        self.slots.remove(&key.index())
    }

    fn entries(&self) -> Vec<(K, u32)> {
        self.slots.values().copied().collect()
    }
}

fn check_state<K: TestKey, const PAGE: usize, D: Directory>(
    map: &mut PagedDirectMap<K, u32, PAGE, D>,
    model: &Model<K>,
    probes: &BTreeSet<K>,
    indices: &BTreeSet<usize>,
    context: &str,
) -> TestCaseResult {
    prop_assert_eq!(map.len(), model.slots.len(), "{}", context);
    prop_assert_eq!(map.is_empty(), model.slots.is_empty(), "{}", context);
    prop_assert_eq!(map.allocated_pages(), model.pages.len(), "{}", context);
    let expected = model.entries();
    prop_assert_eq!(
        &map.iter().map(|(k, &v)| (k, v)).collect::<Vec<_>>(),
        &expected,
        "{}",
        context
    );
    prop_assert_eq!(
        map.iter().rev().map(|(k, &v)| (k, v)).collect::<Vec<_>>(),
        expected.iter().rev().copied().collect::<Vec<_>>(),
        "{}",
        context
    );
    for &key in probes {
        let entry = model.get(key);
        let value = entry.map(|(_, v)| v);
        prop_assert_eq!(map.get(key).copied(), value, "key {:?}: {}", key, context);
        prop_assert_eq!(
            map.get_mut(key).map(|v| *v),
            value,
            "key {:?}: {}",
            key,
            context
        );
        prop_assert_eq!(
            map.contains_key(key),
            entry.is_some(),
            "key {:?}: {}",
            key,
            context
        );
        prop_assert_eq!(
            map.get_key_value(key).map(|(k, &v)| (k, v)),
            entry,
            "key {:?}: {}",
            key,
            context
        );
        if let Some((_, value)) = entry {
            prop_assert_eq!(map[key], value, "{}", context);
        }
    }
    for &index in indices {
        let expected = model.slots.get(&index).copied();
        prop_assert_eq!(
            map.occupant(index).map(|(k, &v)| (k, v)),
            expected,
            "index {}: {}",
            index,
            context
        );
        prop_assert_eq!(
            map.occupant_mut(index).map(|(k, v)| (k, *v)),
            expected,
            "index {}: {}",
            index,
            context
        );
    }
    Ok(())
}

fn run_sequence<K: TestKey, const PAGE: usize, D: Directory>(
    ops: &[Op<K>],
    initial_probes: Vec<K>,
) -> TestCaseResult {
    let mut probes: BTreeSet<_> = initial_probes.into_iter().collect();
    let mut indices = BTreeSet::from([0, usize::MAX]);
    for op in ops {
        match op {
            Op::Insert(k, _)
            | Op::Replace(k, _)
            | Op::TryInsert(k, _)
            | Op::GetOrInsertWith(k, _)
            | Op::GetMut(k, _)
            | Op::Probe(k)
            | Op::Remove(k)
            | Op::RemoveEntry(k) => {
                probes.insert(*k);
            }
            Op::OccupantMut(i, _) | Op::RemoveOccupant(i) => {
                indices.insert(*i);
            }
            _ => {}
        }
    }
    for key in &probes {
        let i = key.index();
        indices.extend([i, i.saturating_sub(1), i.saturating_add(1)]);
    }
    let mut map = PagedDirectMap::<K, u32, PAGE, D>::new();
    let mut model = Model::new();
    check_state(&mut map, &model, &probes, &indices, "initial")?;
    for (step, op) in ops.iter().enumerate() {
        let context = format!("step {step}: {op:?}");
        match *op {
            Op::Insert(k, v) => {
                prop_assert_eq!(map.insert(k, v), model.insert::<PAGE>(k, v), "{}", context);
            }
            Op::Replace(k, v) => {
                model.pages.insert(k.index() / PAGE);
                prop_assert_eq!(
                    map.replace(k, v),
                    model.slots.insert(k.index(), (k, v)),
                    "{}",
                    context
                );
            }
            Op::TryInsert(k, v) => {
                model.pages.insert(k.index() / PAGE);
                let expected = if model.slots.contains_key(&k.index()) {
                    Err(v)
                } else {
                    model.insert::<PAGE>(k, v);
                    Ok(v)
                };
                let actual = map.try_insert(k, v).map(|value| {
                    let old = *value;
                    *value ^= 1;
                    old
                });
                prop_assert_eq!(&actual, &expected, "{}", context);
                if actual.is_ok() {
                    *model.get_mut(k).unwrap() ^= 1;
                }
            }
            Op::GetOrInsertWith(k, v) => {
                model.pages.insert(k.index() / PAGE);
                let old = model.get(k);
                if old.is_none() {
                    model.insert::<PAGE>(k, v);
                }
                let mut calls = 0;
                let value = map.get_or_insert_with(k, || {
                    calls += 1;
                    v
                });
                prop_assert_eq!(*value, old.map_or(v, |(_, v)| v), "{}", context);
                *value ^= 1;
                *model.get_mut(k).unwrap() ^= 1;
                prop_assert_eq!(calls, usize::from(old.is_none()), "{}", context);
            }
            Op::GetMut(k, delta) => {
                let actual = map.get_mut(k).map(|v| {
                    *v = v.wrapping_add(delta);
                    *v
                });
                let expected = model.get_mut(k).map(|v| {
                    *v = v.wrapping_add(delta);
                    *v
                });
                prop_assert_eq!(actual, expected, "{}", context);
            }
            Op::OccupantMut(i, delta) => {
                let actual = map.occupant_mut(i).map(|(k, v)| {
                    *v = v.wrapping_add(delta);
                    (k, *v)
                });
                let expected = model.slots.get_mut(&i).map(|(k, v)| {
                    *v = v.wrapping_add(delta);
                    (*k, *v)
                });
                prop_assert_eq!(actual, expected, "{}", context);
            }
            Op::Probe(_) => {} // All read APIs are checked below after every op.
            Op::Remove(k) => {
                prop_assert_eq!(
                    map.remove(k),
                    model.remove(k).map(|(_, v)| v),
                    "{}",
                    context
                );
            }
            Op::RemoveEntry(k) => {
                prop_assert_eq!(map.remove_entry(k), model.remove(k), "{}", context);
            }
            Op::RemoveOccupant(i) => {
                prop_assert_eq!(
                    map.remove_occupant(i),
                    model.slots.remove(&i),
                    "{}",
                    context
                );
            }
            Op::Retain(predicate) => {
                let expected_visits: Vec<_> = model.slots.keys().copied().collect();
                let mut visits = Vec::new();
                map.retain(|&k, v| {
                    visits.push(k.index());
                    predicate.apply(k, v)
                });
                prop_assert_eq!(visits, expected_visits, "{}", context);
                model.slots.retain(|_, (k, v)| predicate.apply(*k, v));
            }
            Op::Clear => {
                map.clear();
                model.slots.clear();
            }
            Op::Reset => {
                map.reset();
                model.slots.clear();
                model.pages.clear();
            }
            Op::ShrinkToFit => {
                map.shrink_to_fit();
                model.pages = model.slots.keys().map(|i| i / PAGE).collect();
            }
        }
        check_state(&mut map, &model, &probes, &indices, &context)?;
    }
    Ok(())
}

// Direction choices come exclusively from proptest; no incidental RNG affects
// either a failing case or its shrink. Adapters retain the iterator bounds.
fn check_iterator<I, T>(mut iter: I, expected: &[T], directions: &[bool]) -> TestCaseResult
where
    I: ExactSizeIterator<Item = T> + DoubleEndedIterator + FusedIterator,
    T: PartialEq + Debug,
{
    let (mut front, mut back) = (0, expected.len());
    for step in 0..expected.len() {
        let remaining = back - front;
        prop_assert_eq!(iter.len(), remaining);
        prop_assert_eq!(iter.size_hint(), (remaining, Some(remaining)));
        let from_back = directions
            .get(step % directions.len().max(1))
            .copied()
            .unwrap_or(false);
        let actual = if from_back {
            back -= 1;
            iter.next_back()
        } else {
            let item = iter.next();
            front += 1;
            item
        };
        let pos = if from_back { back } else { front - 1 };
        prop_assert_eq!(actual.as_ref(), Some(&expected[pos]));
        prop_assert_eq!(iter.len(), back - front);
    }
    for _ in 0..3 {
        prop_assert_eq!(iter.next(), None);
        prop_assert_eq!(iter.next_back(), None);
        prop_assert_eq!(iter.len(), 0);
        prop_assert_eq!(iter.size_hint(), (0, Some(0)));
    }
    Ok(())
}

fn check_cloned_iterator<I, T>(
    mut iter: I,
    expected: &[T],
    directions: &[bool],
    prefix: usize,
) -> TestCaseResult
where
    I: ExactSizeIterator<Item = T> + DoubleEndedIterator + FusedIterator + Clone,
    T: PartialEq + Debug,
{
    let (mut front, mut back) = (0, expected.len());
    for step in 0..prefix.min(expected.len()) {
        prop_assert_eq!(iter.len(), back - front);
        prop_assert_eq!(iter.size_hint(), (back - front, Some(back - front)));
        if directions
            .get(step % directions.len().max(1))
            .copied()
            .unwrap_or(false)
        {
            back -= 1;
            let item = iter.next_back();
            prop_assert_eq!(item.as_ref(), Some(&expected[back]));
        } else {
            let item = iter.next();
            prop_assert_eq!(item.as_ref(), Some(&expected[front]));
            front += 1;
        }
        prop_assert_eq!(iter.len(), back - front);
    }
    let clone = iter.clone();
    check_iterator(iter, &expected[front..back], directions)?;
    let opposite: Vec<_> = directions.iter().map(|b| !b).collect();
    check_iterator(clone, &expected[front..back], &opposite)
}

fn check_traits<K: TestKey, const PAGE: usize, D: Directory>(
    entries: &[(K, u32)],
    directions: &[bool],
    prefix: usize,
) -> TestCaseResult {
    let mut model = Model::new();
    for &(k, v) in entries {
        model.insert::<PAGE>(k, v);
    }
    let expected = model.entries();
    let probes = entries.iter().map(|&(k, _)| k).collect();
    let indices = entries.iter().map(|(k, _)| k.index()).collect();
    let mut map: PagedDirectMap<K, u32, PAGE, D> = entries.iter().copied().collect();
    check_state(&mut map, &model, &probes, &indices, "FromIterator")?;
    let mut extended = PagedDirectMap::<K, u32, PAGE, D>::default();
    let mid = entries.len() / 2;
    extended.extend(entries[..mid].iter().copied());
    extended.extend(entries[mid..].iter().map(|(k, v)| (k, v)));
    check_state(
        &mut extended,
        &model,
        &probes,
        &indices,
        "Extend owned/borrowed",
    )?;
    // Reversing canonical entries preserves their final values, even when
    // the original input contained collisions between generations.
    let reverse: PagedDirectMap<K, u32, PAGE, D> = expected.iter().rev().copied().collect();
    fn require_eq<T: Eq>(_: &T) {}
    require_eq(&map);
    prop_assert_eq!(&map, &reverse);
    prop_assert_eq!(&reverse, &extended);
    let hasher = RandomState::new();
    prop_assert_eq!(hasher.hash_one(&map), hasher.hash_one(&reverse));
    let unused = K::unused();
    prop_assert!(catch_unwind(AssertUnwindSafe(|| map[unused])).is_err());
    // Removing entries must not affect equality/hash through empty pages.
    prop_assert_eq!(extended.insert(unused, 123), None);
    prop_assert_eq!(extended.remove(unused), Some(123));
    extended.clear();
    extended.extend(expected.iter().copied());
    map.shrink_to_fit();
    prop_assert_eq!(&map, &extended);
    prop_assert_eq!(hasher.hash_one(&map), hasher.hash_one(&extended));
    if let Some(&(k, v)) = expected.first() {
        *extended.get_mut(k).unwrap() ^= 1;
        prop_assert_ne!(&map, &extended);
        *extended.get_mut(k).unwrap() = v;
        extended.remove(k);
        prop_assert_ne!(&map, &extended);
    }
    let keys: Vec<_> = expected.iter().map(|&(k, _)| k).collect();
    let values: Vec<_> = expected.iter().map(|&(_, v)| v).collect();
    check_iterator(map.iter().map(|(k, &v)| (k, v)), &expected, directions)?;
    check_iterator(
        (&map).into_iter().map(|(k, &v)| (k, v)),
        &expected,
        directions,
    )?;
    check_iterator(map.keys(), &keys, directions)?;
    check_iterator(map.values().copied(), &values, directions)?;
    check_cloned_iterator(
        map.iter().map(|(k, &v)| (k, v)),
        &expected,
        directions,
        prefix,
    )?;
    check_cloned_iterator(map.keys(), &keys, directions, prefix)?;
    check_cloned_iterator(map.values().copied(), &values, directions, prefix)?;
    // Mutation through each yielded reference must reach the original slot.
    check_iterator(
        map.iter_mut().map(|(k, v)| {
            let old = *v;
            *v = v.wrapping_add(17);
            (k, old)
        }),
        &expected,
        directions,
    )?;
    for (_, v) in model.slots.values_mut() {
        *v = v.wrapping_add(17);
    }
    check_state(&mut map, &model, &probes, &indices, "iter_mut writes")?;
    let values: Vec<_> = model.slots.values().map(|&(_, v)| v).collect();
    check_iterator(
        map.values_mut().map(|v| {
            let old = *v;
            *v ^= 23;
            old
        }),
        &values,
        directions,
    )?;
    for (_, v) in model.slots.values_mut() {
        *v ^= 23;
    }
    check_state(&mut map, &model, &probes, &indices, "values_mut writes")?;
    check_iterator(
        (&mut map).into_iter().map(|(k, v)| (k, *v)),
        &model.entries(),
        directions,
    )?;
    let owned = || {
        entries
            .iter()
            .copied()
            .collect::<PagedDirectMap<K, u32, PAGE, D>>()
    };
    check_iterator(owned().into_iter(), &expected, directions)?;
    check_iterator(owned().into_keys(), &keys, directions)?;
    let values: Vec<_> = expected.iter().map(|&(_, v)| v).collect();
    check_iterator(owned().into_values(), &values, directions)?;
    Ok(())
}

#[cfg(feature = "serde")]
fn check_serde<K, const PAGE: usize, D: Directory>(entries: &[(K, u32)]) -> TestCaseResult
where
    K: TestKey + serde::Serialize + serde::de::DeserializeOwned,
{
    // Singleton BTreeMaps produce legal JSON fields without depending on
    // DirectMap's serialization. Joining them preserves duplicate input keys.
    let object = |entries: &[(K, u32)]| {
        let fields: Vec<_> = entries
            .iter()
            .map(|&(k, v)| {
                let field = serde_json::to_string(&BTreeMap::from([(k, v)])).unwrap();
                field[1..field.len() - 1].to_owned()
            })
            .collect();
        format!("{{{}}}", fields.join(","))
    };
    let mut model = Model::new();
    for &(k, v) in entries {
        model.insert::<PAGE>(k, v);
    }
    let map: PagedDirectMap<K, u32, PAGE, D> = entries.iter().copied().collect();
    let json = serde_json::to_string(&map).unwrap();
    prop_assert_eq!(&json, &object(&model.entries()));
    let decoded: PagedDirectMap<K, u32, PAGE, D> = serde_json::from_str(&json).unwrap();
    prop_assert_eq!(&decoded, &map);
    let decoded: PagedDirectMap<K, u32, PAGE, D> = serde_json::from_str(&object(entries)).unwrap();
    prop_assert_eq!(&decoded, &map);
    Ok(())
}

macro_rules! suite {
    ($name:ident, $key:ty, $page:literal, $directory:ty, $keys:expr, $probes:expr) => {
        mod $name {
            use super::*;
            proptest! {
                #![proptest_config(config())]
                #[test]
                fn sequences(ops in prop::collection::vec(operations($keys), 0..max_len())) {
                    run_sequence::<$key, $page, $directory>(&ops, $probes)?;
                }

                #[test]
                fn iterators_and_traits(
                    entries in prop::collection::vec(($keys, any::<u32>()), 0..max_len()),
                    directions in prop::collection::vec(any::<bool>(), 1..max_len()),
                    prefix in 0..max_len(),
                ) {
                    check_traits::<$key, $page, $directory>(&entries, &directions, prefix)?;
                }

                #[cfg(feature = "serde")]
                #[test]
                fn json(entries in prop::collection::vec(($keys, any::<u32>()), 0..max_len())) {
                    check_serde::<$key, $page, $directory>(&entries)?;
                }
            }
        }
    };
}

suite!(gen_flat_1, Gen, 1, Flat, gen_keys(), gen_probes());
suite!(gen_flat_3, Gen, 3, Flat, gen_keys(), gen_probes());
suite!(gen_flat_4, Gen, 4, Flat, gen_keys(), gen_probes());
suite!(gen_flat_1024, Gen, 1024, Flat, gen_keys(), gen_probes());
suite!(gen_tree_1, Gen, 1, Tree, gen_keys(), gen_probes());
suite!(gen_radix_1, Gen, 1, Radix, gen_keys(), gen_probes());
suite!(gen_tree_3, Gen, 3, Tree, gen_keys(), gen_probes());
suite!(gen_radix_3, Gen, 3, Radix, gen_keys(), gen_probes());
suite!(gen_tree_4, Gen, 4, Tree, gen_keys(), gen_probes());
suite!(gen_radix_4, Gen, 4, Radix, gen_keys(), gen_probes());
suite!(gen_tree_1024, Gen, 1024, Tree, gen_keys(), gen_probes());
suite!(gen_radix_1024, Gen, 1024, Radix, gen_keys(), gen_probes());

#[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
mod integer_suites {
    use super::*;
    fn probes() -> Vec<u32> {
        probe_indices().into_iter().map(|i| i as u32).collect()
    }
    suite!(u32_flat_1, u32, 1, Flat, integer_keys(), probes());
    suite!(u32_flat_3, u32, 3, Flat, integer_keys(), probes());
    suite!(u32_flat_4, u32, 4, Flat, integer_keys(), probes());
    suite!(u32_flat_1024, u32, 1024, Flat, integer_keys(), probes());
    suite!(u32_tree_1, u32, 1, Tree, integer_keys(), probes());
    suite!(u32_radix_1, u32, 1, Radix, integer_keys(), probes());
    suite!(u32_tree_3, u32, 3, Tree, integer_keys(), probes());
    suite!(u32_radix_3, u32, 3, Radix, integer_keys(), probes());
    suite!(u32_tree_4, u32, 4, Tree, integer_keys(), probes());
    suite!(u32_radix_4, u32, 4, Radix, integer_keys(), probes());
    suite!(u32_tree_1024, u32, 1024, Tree, integer_keys(), probes());
    suite!(u32_radix_1024, u32, 1024, Radix, integer_keys(), probes());
}

macro_rules! all_layouts {
    ($test:ident) => {
        $test::<1, Flat>();
        $test::<3, Flat>();
        $test::<4, Flat>();
        $test::<1024, Flat>();
        $test::<1, Tree>();
        $test::<1, Radix>();
        $test::<3, Tree>();
        $test::<3, Radix>();
        $test::<4, Tree>();
        $test::<4, Radix>();
        $test::<1024, Tree>();
        $test::<1024, Radix>();
    };
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

fn generation_semantics<const PAGE: usize, D: Directory>() {
    let first = Tagged {
        identity: Gen {
            index: 1024,
            generation: 0,
        },
        tag: 1,
    };
    let equal = Tagged { tag: 2, ..first };
    let next = Tagged {
        identity: Gen {
            generation: 1,
            ..first.identity
        },
        tag: 3,
    };
    let mut map = PagedDirectMap::<Tagged, u32, PAGE, D>::new();
    assert_eq!(map.insert(first, 10), None);
    assert_eq!(map.insert(equal, 20), Some(10));
    assert_eq!(map.get_key_value(equal).unwrap().0.tag, first.tag);
    let (old, value) = map.replace(equal, 30).unwrap();
    assert_eq!((old.tag, value), (first.tag, 20));
    assert_eq!(map.get_key_value(first).unwrap().0.tag, equal.tag);
    assert_eq!(map.try_insert(first, 40), Err(40));
    assert_eq!(map.try_insert(next, 50), Err(50));
    assert_eq!(map.get(next), None);
    assert_eq!(map.occupant(first.index()).unwrap().0.tag, equal.tag);
    assert_eq!(
        *map.get_or_insert_with(first, || panic!("equal key must be present")),
        30
    );
    assert_eq!(map.occupant(first.index()).unwrap().0.tag, equal.tag);
    assert_eq!(map.insert(next, 60), Some(30));
    assert!(!map.contains_key(first));
    assert_eq!(map.get(next), Some(&60));
    let (old, value) = map.replace(first, 70).unwrap();
    assert_eq!(
        (old.identity, old.tag, value),
        (next.identity, next.tag, 60)
    );
    assert_eq!(map.remove(next), None);
    assert_eq!(map.remove_entry(next), None);
    let mut calls = 0;
    assert_eq!(
        *map.get_or_insert_with(next, || {
            calls += 1;
            80
        }),
        80
    );
    assert_eq!(calls, 1);
    assert_eq!(map.len(), 1);
    assert_eq!(map.get(first), None);
    assert_eq!(map.get_key_value(next).unwrap().0.tag, next.tag);
    let (old, value) = map.remove_occupant(first.index()).unwrap();
    assert_eq!((old.tag, value), (next.tag, 80));
    assert_eq!(
        *map.get_or_insert_with(first, || {
            calls += 1;
            90
        }),
        90
    );
    assert_eq!(calls, 2);
    assert_eq!(map.len(), 1);
    assert!(catch_unwind(AssertUnwindSafe(|| map[next])).is_err());
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            map[Tagged {
                identity: Gen {
                    index: 2051,
                    ..first.identity
                },
                ..first
            }]
        }))
        .is_err()
    );
}

#[test]
fn stored_keys_and_generations_follow_each_insert_contract() {
    all_layouts!(generation_semantics);
}

#[test]
#[cfg(target_pointer_width = "64")]
fn compact_slots_rebuild_id_generations_after_reuse() {
    use std::mem::size_of;

    use paged_collections::direct_map::Slot;
    use paged_collections::id_alloc::{Id, IdAllocator};

    assert_eq!(size_of::<Slot<u32, u64>>(), 16);
    assert_eq!(size_of::<Slot<u32, Box<u8>>>(), size_of::<usize>());
    assert_eq!(size_of::<Slot<Id, u64>>(), 16);

    let mut ids = IdAllocator::default();
    let keys: Vec<_> = (0..6).map(|_| ids.alloc()).collect();
    let mut map = PagedDirectMap::<Id, u64, 4>::new();
    for (value, &key) in keys.iter().enumerate() {
        assert_eq!(map.insert(key, value as u64), None);
    }
    assert_eq!(map.keys().collect::<Vec<_>>(), keys);

    let old = keys[5];
    assert!(ids.free(old));
    let new = ids.alloc();
    assert_eq!(new.index(), old.index());
    assert_eq!(new.generation().get(), old.generation().get() + 1);
    assert_eq!(map.insert(new, 99), Some(5));
    assert_eq!(map.get(old), None);
    assert_eq!(map.get(new), Some(&99));
    let mut expected: Vec<_> = keys
        .iter()
        .enumerate()
        .map(|(value, &key)| (key, value as u64))
        .collect();
    expected[5] = (new, 99);
    assert_eq!(
        map.iter()
            .map(|(key, &value)| (key, value))
            .collect::<Vec<_>>(),
        expected
    );
    let rebuilt: PagedDirectMap<Id, u64, 4> =
        map.iter().map(|(key, &value)| (key, value)).collect();
    assert_eq!(rebuilt, map);
    assert_eq!(rebuilt.last_key_value(), Some((new, &99)));
}

fn allocation_and_equality<const PAGE: usize, D: Directory>() {
    let key = |index| Gen {
        index,
        generation: 0,
    };
    let entries = [(key(1024), 4), (key(0), 1), (key(2050), 8)];
    let a: PagedDirectMap<Gen, u32, PAGE, D> = entries.into_iter().collect();
    let mut b: PagedDirectMap<Gen, u32, PAGE, D> = entries.into_iter().rev().collect();
    let live_pages: BTreeSet<_> = entries.iter().map(|(k, _)| k.index() / PAGE).collect();
    assert_eq!(a.allocated_pages(), live_pages.len());
    // This page is distinct even at PAGE=1024. Clear preserves it too.
    b.try_insert(key(4096), 999).unwrap();
    assert_eq!(b.remove(key(4096)), Some(999));
    assert_eq!(b.allocated_pages(), live_pages.len() + 1);
    assert_eq!(a, b);
    let hasher = RandomState::new();
    assert_eq!(hasher.hash_one(&a), hasher.hash_one(&b));
    b.clear();
    assert_eq!(b.allocated_pages(), live_pages.len() + 1);
    b.extend(entries);
    assert_eq!(a, b);
    assert_eq!(hasher.hash_one(&a), hasher.hash_one(&b));
    b.shrink_to_fit();
    assert_eq!(b.allocated_pages(), live_pages.len());
    assert_eq!(a, b);
    b.reset();
    assert_eq!(b.allocated_pages(), 0);
    assert_eq!(b.len(), 0);
    b.shrink_to_fit();
    assert_eq!(b.allocated_pages(), 0);
    b.extend(entries);
    let mut different: PagedDirectMap<Gen, u32, PAGE, D> = entries.into_iter().collect();
    different.replace(
        Gen {
            generation: 1,
            ..key(0)
        },
        1,
    );
    assert_ne!(a, different);
    // Force duplicate and generation collisions rather than hoping a random
    // input contains them. Later entries win for both collect and extend.
    let duplicate = [
        (key(3), 1),
        (key(3), 2),
        (
            Gen {
                generation: 1,
                ..key(3)
            },
            3,
        ),
    ];
    check_traits::<Gen, PAGE, D>(&duplicate, &[false, true], 1).unwrap();
    #[cfg(feature = "serde")]
    check_serde::<Gen, PAGE, D>(&duplicate).unwrap();
}

#[test]
fn empty_pages_do_not_affect_equality_or_hash_and_shrink_is_exact() {
    all_layouts!(allocation_and_equality);
}

fn cursor_edges<const PAGE: usize, D: Directory>() {
    let make = || {
        let mut map = PagedDirectMap::<Gen, u32, PAGE, D>::new();
        // Same-page endpoints, holes between pages, empty leading/trailing
        // pages, and two active cursor ends meeting within a single page.
        for index in [4096, 2049, 1025, 1024, 1023, 3, 2, 1, 0] {
            map.insert(
                Gen {
                    index,
                    generation: 0,
                },
                u32::from(index),
            );
        }
        for index in [0, 2, 1024, 4096] {
            map.remove(Gen {
                index,
                generation: 0,
            });
        }
        map
    };
    let expected: Vec<_> = [1, 3, 1023, 1025, 2049]
        .into_iter()
        .map(|index| {
            (
                Gen {
                    index,
                    generation: 0,
                },
                u32::from(index),
            )
        })
        .collect();
    let keys: Vec<_> = expected.iter().map(|&(k, _)| k).collect();
    let values: Vec<_> = expected.iter().map(|&(_, v)| v).collect();
    for directions in [
        &[false][..],
        &[true],
        &[true, false],
        &[false, true, true, false],
    ] {
        let mut map = make();
        check_iterator(map.iter().map(|(k, &v)| (k, v)), &expected, directions).unwrap();
        check_iterator(map.iter_mut().map(|(k, v)| (k, *v)), &expected, directions).unwrap();
        check_iterator(map.keys(), &keys, directions).unwrap();
        check_iterator(map.values().copied(), &values, directions).unwrap();
        check_iterator(map.values_mut().map(|v| *v), &values, directions).unwrap();
        for prefix in 0..=expected.len() {
            check_cloned_iterator(
                map.iter().map(|(k, &v)| (k, v)),
                &expected,
                directions,
                prefix,
            )
            .unwrap();
            check_cloned_iterator(map.keys(), &keys, directions, prefix).unwrap();
            check_cloned_iterator(map.values().copied(), &values, directions, prefix).unwrap();
        }
        check_iterator(make().into_iter(), &expected, directions).unwrap();
        check_iterator(make().into_keys(), &keys, directions).unwrap();
        check_iterator(make().into_values(), &values, directions).unwrap();
        map.clear();
        check_iterator(map.iter(), &[], directions).unwrap();
        check_iterator(map.iter_mut(), &[], directions).unwrap();
        check_iterator(map.keys(), &[], directions).unwrap();
        check_iterator(map.values(), &[], directions).unwrap();
        check_iterator(map.values_mut(), &[], directions).unwrap();
        check_iterator(map.into_iter(), &[], directions).unwrap();
    }
}

#[test]
fn cursors_cross_page_holes_and_meet_without_duplicates() {
    all_layouts!(cursor_edges);
}

fn panicking_make<const PAGE: usize, D: Directory>() {
    let old = Gen {
        index: 1024,
        generation: 0,
    };
    let entries = [
        (old, 10),
        (
            Gen {
                index: 0,
                generation: 2,
            },
            20,
        ),
    ];
    let mut map: PagedDirectMap<Gen, u32, PAGE, D> = entries.into_iter().collect();
    let before: Vec<_> = map.iter().map(|(k, &v)| (k, v)).collect();
    for key in [
        Gen {
            generation: 1,
            ..old
        },
        Gen { index: 3, ..old },
        Gen { index: 4096, ..old },
    ] {
        let mut calls = 0;
        let result = catch_unwind(AssertUnwindSafe(|| {
            map.get_or_insert_with(key, || {
                calls += 1;
                panic!("make panic")
            });
        }));
        assert!(result.is_err());
        assert_eq!(calls, 1);
        assert_eq!(map.len(), before.len());
        assert_eq!(map.iter().map(|(k, &v)| (k, v)).collect::<Vec<_>>(), before);
        assert_eq!(map.get(old), Some(&10));
        assert_eq!(map.get(key), None);
    }
    assert_eq!(
        *map.get_or_insert_with(old, || panic!("make must not run")),
        10
    );
    map.insert(Gen { index: 3, ..old }, 30);
    assert_eq!(map.len(), 3);
}

#[test]
fn a_panicking_make_preserves_entries_for_vacant_and_other_generation_slots() {
    all_layouts!(panicking_make);
}

mod panic_and_drop_tests {
    use super::*;
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;

    #[derive(Default)]
    struct Tracker {
        counts: RefCell<Vec<usize>>,
        panic_on: Cell<Option<usize>>,
    }

    impl Tracker {
        fn make(self: &Rc<Self>, payload: u32) -> DropValue {
            let id = self.counts.borrow().len();
            self.counts.borrow_mut().push(0);
            DropValue {
                id,
                payload,
                tracker: Rc::clone(self),
            }
        }

        fn assert_all_once(&self) {
            assert!(
                self.counts.borrow().iter().all(|&count| count == 1),
                "drop counts: {:?}",
                self.counts.borrow()
            );
        }
    }

    struct DropValue {
        id: usize,
        payload: u32,
        tracker: Rc<Tracker>,
    }

    impl Drop for DropValue {
        fn drop(&mut self) {
            self.tracker.counts.borrow_mut()[self.id] += 1;
            if self.tracker.panic_on.get() == Some(self.id) {
                // One panic per operation, including while an owning
                // iterator's remaining fields are dropped during unwind.
                self.tracker.panic_on.set(None);
                panic!("drop panic for value {}", self.id);
            }
        }
    }

    type Oracle = BTreeMap<usize, (Gen, (usize, u32))>;

    fn fixture<const PAGE: usize, D: Directory>(
        tracker: &Rc<Tracker>,
    ) -> (PagedDirectMap<Gen, DropValue, PAGE, D>, Oracle) {
        let mut map = PagedDirectMap::new();
        let mut oracle = BTreeMap::new();
        for index in [1025, 0, 2049, 3, 1023, 1024, 2048, 1] {
            let key = Gen {
                index,
                generation: 0,
            };
            let value = tracker.make(u32::from(index));
            oracle.insert(key.index(), (key, (value.id, value.payload)));
            assert!(map.insert(key, value).is_none());
        }
        (map, oracle)
    }

    // The expected entries come from the pre-operation oracle and the
    // contract, never from adopting the map's post-panic iteration.
    fn assert_remaining<const PAGE: usize, D: Directory>(
        map: &mut PagedDirectMap<Gen, DropValue, PAGE, D>,
        oracle: &Oracle,
        tracker: &Tracker,
    ) {
        let actual: Vec<_> = map.iter().map(|(k, v)| (k, (v.id, v.payload))).collect();
        let expected: Vec<_> = oracle.values().copied().collect();
        assert_eq!(map.len(), actual.len());
        assert_eq!(map.len(), oracle.len());
        assert_eq!(map.is_empty(), oracle.is_empty());
        assert_eq!(actual, expected);
        for index in probe_indices().into_iter().chain([4096]) {
            let occupant = oracle.get(&index).copied();
            assert_eq!(
                map.occupant(index).map(|(k, v)| (k, (v.id, v.payload))),
                occupant
            );
            assert_eq!(
                map.occupant_mut(index).map(|(k, v)| (k, (v.id, v.payload))),
                occupant
            );
            for generation in 0..4 {
                let key = Gen {
                    index: index as u16,
                    generation,
                };
                let expected = occupant.filter(|(k, _)| *k == key);
                assert_eq!(map.contains_key(key), expected.is_some());
                assert_eq!(
                    map.get(key).map(|v| (v.id, v.payload)),
                    expected.map(|(_, v)| v)
                );
                assert_eq!(
                    map.get_mut(key).map(|v| (v.id, v.payload)),
                    expected.map(|(_, v)| v)
                );
                assert_eq!(
                    map.get_key_value(key).map(|(k, v)| (k, (v.id, v.payload))),
                    expected
                );
            }
        }
        let live: BTreeSet<_> = oracle.values().map(|(_, (id, _))| *id).collect();
        for (id, &count) in tracker.counts.borrow().iter().enumerate() {
            assert_eq!(count, usize::from(!live.contains(&id)), "value {id}");
        }
    }

    fn reuse_then_drop<const PAGE: usize, D: Directory>(
        mut map: PagedDirectMap<Gen, DropValue, PAGE, D>,
        mut oracle: Oracle,
        tracker: &Rc<Tracker>,
    ) {
        let key = Gen {
            index: 4096,
            generation: 2,
        };
        let value = tracker.make(123);
        oracle.insert(key.index(), (key, (value.id, value.payload)));
        assert!(map.insert(key, value).is_none());
        assert_remaining(&mut map, &oracle, tracker);
        drop(map);
        tracker.assert_all_once();
    }

    fn panicking_predicate<const PAGE: usize, D: Directory>() {
        for panic_at in 0..8 {
            let tracker = Rc::new(Tracker::default());
            let (mut map, mut oracle) = fixture::<PAGE, D>(&tracker);
            let ordered: Vec<_> = oracle.keys().copied().collect();
            let mut visits = Vec::new();
            let result = catch_unwind(AssertUnwindSafe(|| {
                map.retain(|&key, value| {
                    let rank = visits.len();
                    visits.push(key.index());
                    value.payload = value.payload.wrapping_add(19);
                    assert!(rank != panic_at, "predicate panic");
                    rank % 2 == 0
                });
            }));
            assert!(result.is_err());
            assert_eq!(visits, ordered[..=panic_at]);
            for (rank, &index) in ordered.iter().enumerate().take(panic_at + 1) {
                oracle.get_mut(&index).unwrap().1.1 += 19;
                if rank != panic_at && rank % 2 != 0 {
                    oracle.remove(&index);
                }
            }
            assert_remaining(&mut map, &oracle, &tracker);
            reuse_then_drop(map, oracle, &tracker);
        }
    }

    #[test]
    fn retain_predicate_panics_keep_the_current_and_unvisited_entries() {
        all_layouts!(panicking_predicate);
    }

    fn panicking_clear_and_reset<const PAGE: usize, D: Directory>() {
        for reset in [false, true] {
            for panic_at in 0..8 {
                let tracker = Rc::new(Tracker::default());
                let (mut map, mut oracle) = fixture::<PAGE, D>(&tracker);
                let pages = map.allocated_pages();
                let ordered: Vec<_> = oracle.keys().copied().collect();
                let id = oracle[&ordered[panic_at]].1.0;
                tracker.panic_on.set(Some(id));
                let result = catch_unwind(AssertUnwindSafe(|| {
                    if reset {
                        map.reset();
                    } else {
                        map.clear();
                    }
                }));
                assert!(result.is_err());
                assert_eq!(tracker.panic_on.get(), None);
                if reset {
                    oracle.clear();
                    assert_eq!(map.allocated_pages(), 0);
                } else {
                    for index in &ordered[..=panic_at] {
                        oracle.remove(index);
                    }
                    assert_eq!(map.allocated_pages(), pages);
                }
                assert_remaining(&mut map, &oracle, &tracker);
                reuse_then_drop(map, oracle, &tracker);
            }
        }
    }

    #[test]
    fn panicking_drops_during_clear_and_reset_leave_exact_survivors() {
        all_layouts!(panicking_clear_and_reset);
    }

    fn panicking_retain_drop<const PAGE: usize, D: Directory>() {
        for panic_at in 0..5 {
            let tracker = Rc::new(Tracker::default());
            let (mut map, mut oracle) = fixture::<PAGE, D>(&tracker);
            let ordered: Vec<_> = oracle.keys().copied().collect();
            // Remove odd-indexed slots; keep and mutate even-indexed ones.
            let removals: Vec<_> = ordered.iter().copied().filter(|i| i % 2 != 0).collect();
            let panic_index = removals[panic_at];
            tracker.panic_on.set(Some(oracle[&panic_index].1.0));
            let mut visited = Vec::new();
            let result = catch_unwind(AssertUnwindSafe(|| {
                map.retain(|&key, value| {
                    visited.push(key.index());
                    value.payload += 7;
                    key.index() % 2 == 0
                });
            }));
            assert!(result.is_err());
            assert_eq!(tracker.panic_on.get(), None);
            let prefix: Vec<_> = ordered
                .into_iter()
                .take_while(|&i| i <= panic_index)
                .collect();
            assert_eq!(visited, prefix);
            for index in prefix {
                oracle.get_mut(&index).unwrap().1.1 += 7;
                if index % 2 != 0 {
                    oracle.remove(&index);
                }
            }
            assert_remaining(&mut map, &oracle, &tracker);
            reuse_then_drop(map, oracle, &tracker);
        }
    }

    #[test]
    fn retain_updates_length_before_a_removed_value_panics_in_drop() {
        all_layouts!(panicking_retain_drop);
    }

    fn panicking_eviction<const PAGE: usize, D: Directory>() {
        let tracker = Rc::new(Tracker::default());
        let (mut map, mut oracle) = fixture::<PAGE, D>(&tracker);
        let old = oracle[&1024].0;
        let next = Gen {
            generation: 1,
            ..old
        };
        tracker.panic_on.set(Some(oracle[&1024].1.0));
        let new_id = tracker.counts.borrow().len();
        let mut calls = 0;
        let result = catch_unwind(AssertUnwindSafe(|| {
            map.get_or_insert_with(next, || {
                calls += 1;
                tracker.make(99)
            });
        }));
        assert!(result.is_err());
        assert_eq!(calls, 1);
        assert_eq!(tracker.panic_on.get(), None);
        oracle.insert(1024, (next, (new_id, 99)));
        assert_remaining(&mut map, &oracle, &tracker);
        reuse_then_drop(map, oracle, &tracker);
    }

    #[test]
    fn lazy_insertion_commits_the_new_generation_before_dropping_the_evicted_value() {
        all_layouts!(panicking_eviction);
    }

    fn consuming_drop<const PAGE: usize, D: Directory>() {
        // Generic so the into_keys case can share this code for Copy keys.
        fn drop_owned<T>(value: T) {
            drop(value);
        }
        // Cover ordinary destruction as well as a panic at every remaining
        // value. Both cursor buffers and the page iterator own live values.
        for panic_at in 0..=8 {
            let tracker = Rc::new(Tracker::default());
            let (map, oracle) = fixture::<PAGE, D>(&tracker);
            let ordered: Vec<_> = oracle.values().copied().collect();
            if panic_at < ordered.len() {
                tracker.panic_on.set(Some(ordered[panic_at].1.0));
            }
            let result = catch_unwind(AssertUnwindSafe(|| drop(map)));
            assert_eq!(result.is_err(), panic_at < ordered.len());
            assert_eq!(tracker.panic_on.get(), None);
            tracker.assert_all_once();
        }
        for iterator_kind in 0..3 {
            for panic_at in 0..=6 {
                let tracker = Rc::new(Tracker::default());
                let (map, oracle) = fixture::<PAGE, D>(&tracker);
                let mut ordered: Vec<_> = oracle.values().copied().collect();
                let first = ordered.remove(0);
                let last = ordered.pop().unwrap();
                // into_keys drops values as keys are yielded; the other
                // owning iterators transfer values to their caller.
                macro_rules! consume {
                    ($iter:expr, $project:expr) => {{
                        let mut iter = $iter;
                        let front = iter.next().unwrap();
                        let back = iter.next_back().unwrap();
                        assert_eq!(($project)(&front), first.0.index());
                        assert_eq!(($project)(&back), last.0.index());
                        drop_owned(front);
                        drop_owned(back);
                        assert_eq!(iter.len(), ordered.len());
                        if panic_at < ordered.len() {
                            tracker.panic_on.set(Some(ordered[panic_at].1.0));
                        }
                        let result = catch_unwind(AssertUnwindSafe(|| drop(iter)));
                        assert_eq!(result.is_err(), panic_at < ordered.len());
                    }};
                }
                match iterator_kind {
                    0 => consume!(map.into_iter(), |entry: &(Gen, DropValue)| entry.0.index()),
                    1 => consume!(map.into_keys(), |key: &Gen| key.index()),
                    _ => consume!(map.into_values(), |value: &DropValue| value.payload
                        as usize),
                }
                assert_eq!(tracker.panic_on.get(), None);
                tracker.assert_all_once();
            }
        }
    }

    #[test]
    fn map_and_partially_consumed_owning_iterators_drop_every_value_once_even_on_panic() {
        all_layouts!(consuming_drop);
    }

    fn normal_drops<const PAGE: usize, D: Directory>() {
        for operation in 0..3 {
            let tracker = Rc::new(Tracker::default());
            let (mut map, mut oracle) = fixture::<PAGE, D>(&tracker);
            match operation {
                0 => {
                    map.clear();
                    oracle.clear();
                }
                1 => {
                    map.reset();
                    oracle.clear();
                }
                _ => {
                    map.retain(|&k, value| {
                        value.payload += 1;
                        k.index() % 2 == 0
                    });
                    oracle.retain(|&i, (_, (_, value))| {
                        *value += 1;
                        i % 2 == 0
                    });
                }
            }
            assert_remaining(&mut map, &oracle, &tracker);
            reuse_then_drop(map, oracle, &tracker);
        }
        // Returned and refused values stay owned by the caller until dropped.
        let tracker = Rc::new(Tracker::default());
        let mut map = PagedDirectMap::<Gen, DropValue, PAGE, D>::new();
        let old = Gen {
            index: 3,
            generation: 0,
        };
        let new = Gen {
            generation: 1,
            ..old
        };
        assert!(map.insert(old, tracker.make(0)).is_none());
        let evicted = map.insert(new, tracker.make(1)).unwrap();
        assert_eq!(evicted.id, 0);
        assert_eq!(tracker.counts.borrow()[0], 0);
        drop(evicted);
        let (key, evicted) = map.replace(old, tracker.make(2)).unwrap();
        assert_eq!(key, new);
        assert_eq!(evicted.id, 1);
        drop(evicted);
        let refused = match map.try_insert(new, tracker.make(3)) {
            Ok(_) => panic!("occupied slot accepted"),
            Err(value) => value,
        };
        assert_eq!(refused.id, 3);
        assert_eq!(tracker.counts.borrow()[3], 0);
        drop(refused);
        assert_eq!(map.get_or_insert_with(old, || panic!("present")).id, 2);
        for method in 0..3 {
            let value = match method {
                0 => map.remove(old).unwrap(),
                1 => map.remove_entry(old).unwrap().1,
                _ => map.remove_occupant(old.index()).unwrap().1,
            };
            assert_eq!(tracker.counts.borrow()[value.id], 0);
            drop(value);
            assert!(map.insert(old, tracker.make(10)).is_none());
        }
        drop(map);
        tracker.assert_all_once();
    }

    #[test]
    fn ordinary_clear_reset_retain_replacement_and_removal_drop_values_exactly_once() {
        all_layouts!(normal_drops);
    }
}
