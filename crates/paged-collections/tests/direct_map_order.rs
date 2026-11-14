//! Slot-ordered queries against a BTreeMap oracle, including invalid bounds.

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::iter::FusedIterator;
use std::ops::Bound::{self, Excluded, Included, Unbounded};

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Gen(u32);

impl Key for Gen {
    fn index(self) -> usize {
        (self.0 & 0xffff) as usize
    }
}

impl DirectKey for Gen {
    type Tag = u16;

    fn tag(self) -> Self::Tag {
        (self.0 >> 16) as u16
    }

    fn from_parts(index: usize, tag: Self::Tag) -> Self {
        Self(index as u32 | (u32::from(tag) << 16))
    }
}

trait TestKey: DirectKey + Debug {
    fn make(index: u16, generation: u8) -> Self;

    fn upper_bound() -> Self {
        Self::make(u16::MAX, u8::MAX)
    }
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

    fn upper_bound() -> Self {
        Self::MAX
    }
}

type Bounds<T> = (Bound<T>, Bound<T>);
type Model<K> = BTreeMap<usize, (K, u32)>;

fn indices() -> BoxedStrategy<u16> {
    prop_oneof![6 => 0_u16..48, 2 => 1022_u16..1028, 2 => 2047_u16..2052].boxed()
}

fn keys<K: TestKey + 'static>() -> BoxedStrategy<K> {
    (indices(), 0_u8..4)
        .prop_map(|(index, generation)| K::make(index, generation))
        .boxed()
}

fn bound<T: Clone + Debug + 'static>(values: BoxedStrategy<T>) -> BoxedStrategy<Bound<T>> {
    prop_oneof![
        1 => Just(Unbounded),
        3 => values.clone().prop_map(Included),
        3 => values.prop_map(Excluded),
    ]
    .boxed()
}

fn slot_bounds() -> BoxedStrategy<Bounds<usize>> {
    let values = prop_oneof![
        6 => indices().prop_map(usize::from),
        1 => any::<usize>(),
        1 => Just(0),
        1 => Just(usize::MAX),
    ]
    .boxed();
    (bound(values.clone()), bound(values)).boxed()
}

fn key_bounds<K: TestKey + 'static>() -> BoxedStrategy<Bounds<K>> {
    let values = prop_oneof![
        6 => (indices(), any::<u8>())
            .prop_map(|(index, generation)| K::make(index, generation)),
        1 => Just(K::upper_bound()),
        1 => Just(K::make(0, 255)),
    ]
    .boxed();
    (bound(values.clone()), bound(values)).boxed()
}

// Filtering independently handles all empty/inverted ranges without the
// panics BTreeMap::range specifies for some pairs of bounds.
fn contains(bounds: Bounds<usize>, index: usize) -> bool {
    let after_start = match bounds.0 {
        Included(start) => index >= start,
        Excluded(start) => index > start,
        Unbounded => true,
    };
    let before_end = match bounds.1 {
        Included(end) => index <= end,
        Excluded(end) => index < end,
        Unbounded => true,
    };
    after_start && before_end
}

fn selected<K: Copy>(model: &Model<K>, bounds: Bounds<usize>) -> Vec<(K, u32)> {
    model
        .iter()
        .filter(|(index, _)| contains(bounds, **index))
        .map(|(_, &entry)| entry)
        .collect()
}

fn index_bounds<K: Key>(bounds: Bounds<K>) -> Bounds<usize> {
    (bounds.0.map(Key::index), bounds.1.map(Key::index))
}

fn check_hint<I: Iterator>(iter: &I, remaining: usize) -> TestCaseResult {
    let (lower, upper) = iter.size_hint();
    prop_assert_eq!(lower, 0);
    prop_assert!(upper.is_some_and(|n| n >= remaining));
    Ok(())
}

fn check_range<I, T>(
    mut iter: I,
    expected: &[T],
    directions: &[bool],
    prefix: usize,
) -> TestCaseResult
where
    I: DoubleEndedIterator<Item = T> + FusedIterator + Clone,
    T: Copy + Debug + PartialEq,
{
    prop_assert_eq!(iter.clone().collect::<Vec<_>>(), expected);
    prop_assert_eq!(
        iter.clone().rev().collect::<Vec<_>>(),
        expected.iter().rev().copied().collect::<Vec<_>>()
    );
    let (mut front, mut back) = (0, expected.len());
    let clone_at = prefix % (expected.len() + 1);
    for step in 0..=expected.len() {
        check_hint(&iter, back - front)?;
        if step == clone_at {
            prop_assert_eq!(iter.clone().collect::<Vec<_>>(), &expected[front..back]);
            prop_assert_eq!(
                iter.clone().rev().collect::<Vec<_>>(),
                expected[front..back]
                    .iter()
                    .rev()
                    .copied()
                    .collect::<Vec<_>>()
            );
        }
        if front == back {
            break;
        }
        if directions[step % directions.len()] {
            back -= 1;
            prop_assert_eq!(iter.next_back(), Some(expected[back]));
        } else {
            prop_assert_eq!(iter.next(), Some(expected[front]));
            front += 1;
        }
    }
    for _ in 0..4 {
        prop_assert_eq!(iter.next(), None);
        check_hint(&iter, 0)?;
        prop_assert_eq!(iter.next_back(), None);
        check_hint(&iter, 0)?;
    }
    Ok(())
}

fn check_order<K: DirectKey + Debug, const PAGE: usize, D: Directory>(
    map: &PagedDirectMap<K, u32, PAGE, D>,
    model: &Model<K>,
    probes: &[usize],
) -> TestCaseResult {
    prop_assert_eq!(map.len(), model.len());
    prop_assert_eq!(
        map.first_key_value().map(|(k, &v)| (k, v)),
        model.first_key_value().map(|(_, &entry)| entry)
    );
    prop_assert_eq!(
        map.last_key_value().map(|(k, &v)| (k, v)),
        model.last_key_value().map(|(_, &entry)| entry)
    );
    for &index in probes {
        prop_assert_eq!(
            map.next_occupant(index).map(|(k, &v)| (k, v)),
            model.range(index..).next().map(|(_, &entry)| entry)
        );
        prop_assert_eq!(
            map.prev_occupant(index).map(|(k, &v)| (k, v)),
            model.range(..=index).next_back().map(|(_, &entry)| entry)
        );
    }
    Ok(())
}

fn check_queries<K: DirectKey + Debug, const PAGE: usize, D: Directory>(
    map: &PagedDirectMap<K, u32, PAGE, D>,
    model: &Model<K>,
    slots: Bounds<usize>,
    keys: Bounds<K>,
    directions: &[bool],
    prefix: usize,
) -> TestCaseResult {
    check_range(
        map.slot_range(slots).map(|(k, &v)| (k, v)),
        &selected(model, slots),
        directions,
        prefix,
    )?;
    let indices = index_bounds(keys);
    let expected = selected(model, indices);
    check_range(
        map.range(keys).map(|(k, &v)| (k, v)),
        &expected,
        directions,
        prefix,
    )?;
    prop_assert_eq!(
        map.slot_range(indices)
            .map(|(k, &v)| (k, v))
            .collect::<Vec<_>>(),
        expected
    );
    Ok(())
}

fn mutate_range<'a, K, I>(
    mut iter: I,
    expected: &[(K, u32)],
    directions: &[bool],
    delta: u32,
) -> TestCaseResult
where
    K: Key + Debug + 'a,
    I: DoubleEndedIterator<Item = (K, &'a mut u32)> + FusedIterator,
{
    let (mut front, mut back) = (0, expected.len());
    for step in 0..expected.len() {
        check_hint(&iter, back - front)?;
        let (entry, want) = if directions[step % directions.len()] {
            back -= 1;
            (iter.next_back(), expected[back])
        } else {
            let want = expected[front];
            front += 1;
            (iter.next(), want)
        };
        let (key, value) = entry.expect("model has an entry remaining");
        prop_assert_eq!((key, *value), want);
        *value = value.wrapping_add(delta);
    }
    for _ in 0..4 {
        check_hint(&iter, 0)?;
        prop_assert!(iter.next().is_none());
        prop_assert!(iter.next_back().is_none());
    }
    Ok(())
}

fn update_model<K: DirectKey + Debug, const PAGE: usize, D: Directory>(
    map: &PagedDirectMap<K, u32, PAGE, D>,
    model: &mut Model<K>,
    bounds: Bounds<usize>,
    delta: u32,
) -> TestCaseResult {
    for (&index, (_, value)) in model.iter_mut() {
        if contains(bounds, index) {
            *value = value.wrapping_add(delta);
        }
    }
    prop_assert_eq!(map.len(), model.len());
    prop_assert_eq!(
        map.iter().map(|(k, &v)| (k, v)).collect::<Vec<_>>(),
        model.values().copied().collect::<Vec<_>>()
    );
    Ok(())
}

fn check_mutations<K: DirectKey + Debug, const PAGE: usize, D: Directory>(
    map: &mut PagedDirectMap<K, u32, PAGE, D>,
    model: &mut Model<K>,
    slots: Bounds<usize>,
    keys: Bounds<K>,
    directions: &[bool],
    delta: u32,
) -> TestCaseResult {
    // Each iterator is confined to a block before the next map borrow.
    for choices in [&[false][..], &[true][..], directions] {
        {
            let expected = selected(model, slots);
            let iter = map.slot_range_mut(slots);
            mutate_range(iter, &expected, choices, delta)?;
        }
        update_model(map, model, slots, delta)?;
        {
            let expected = selected(model, index_bounds(keys));
            let iter = map.range_mut(keys);
            mutate_range(iter, &expected, choices, delta)?;
        }
        update_model(map, model, index_bounds(keys), delta)?;
    }
    Ok(())
}

fn pop_all<K: DirectKey + Debug, const PAGE: usize, D: Directory>(
    map: &mut PagedDirectMap<K, u32, PAGE, D>,
    model: &mut Model<K>,
    directions: &[bool],
) -> TestCaseResult {
    let mut step = 0;
    while !model.is_empty() {
        if directions[step % directions.len()] {
            prop_assert_eq!(map.pop_last(), model.pop_last().map(|(_, entry)| entry));
        } else {
            prop_assert_eq!(map.pop_first(), model.pop_first().map(|(_, entry)| entry));
        }
        check_order(map, model, &[0, usize::MAX])?;
        step += 1;
    }
    for _ in 0..4 {
        prop_assert_eq!(map.pop_first(), None);
        prop_assert_eq!(map.pop_last(), None);
    }
    check_order(map, model, &[0, usize::MAX])?;
    prop_assert!(map.is_empty());
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum Op<K> {
    Insert(K, u32),
    Remove(K),
    Clear,
    Shrink,
}

fn operations<K: TestKey + 'static>() -> BoxedStrategy<Op<K>> {
    prop_oneof![
        10 => (keys(), any::<u32>()).prop_map(|(k, v)| Op::Insert(k, v)),
        5 => keys().prop_map(Op::Remove),
        1 => Just(Op::Clear),
        2 => Just(Op::Shrink),
    ]
    .boxed()
}

fn run_sequence<K: TestKey, const PAGE: usize, D: Directory>(
    ops: &[Op<K>],
    queries: &[(Bounds<usize>, Bounds<K>)],
    directions: &[bool],
    prefix: usize,
    delta: u32,
) -> TestCaseResult {
    let mut map = PagedDirectMap::<K, u32, PAGE, D>::new();
    let mut model = Model::new();
    check_order(&map, &model, &[0, usize::MAX])?;
    // Force allocated-but-empty pages even when proptest generates no ops.
    let ghost = K::make(4096, 0);
    map.insert(ghost, 17);
    map.clear();
    prop_assert!(map.allocated_pages() > 0);
    for &(slots, keys) in queries {
        check_queries(&map, &model, slots, keys, directions, prefix)?;
        check_mutations(&mut map, &mut model, slots, keys, directions, delta)?;
    }
    pop_all(&mut map, &mut model, directions)?;
    // Keep entries on both sides of page boundaries and a removed page.
    for index in [0, PAGE - 1, PAGE, PAGE + 1, 3 * PAGE] {
        let key = K::make(index as u16, (index % 3) as u8);
        map.insert(key, index as u32);
        model.insert(index, (key, index as u32));
    }
    let removed = K::make((3 * PAGE) as u16, 0);
    map.remove(removed);
    model.remove(&(3 * PAGE));
    for &(slots, keys) in queries {
        check_queries(&map, &model, slots, keys, directions, prefix)?;
        check_mutations(&mut map, &mut model, slots, keys, directions, delta)?;
    }

    let probes = [0, PAGE - 1, PAGE, PAGE + 1, 2 * PAGE, usize::MAX];
    for (step, op) in ops.iter().enumerate() {
        match *op {
            Op::Insert(key, value) => {
                let old = model.insert(key.index(), (key, value)).map(|(_, v)| v);
                prop_assert_eq!(map.insert(key, value), old);
            }
            Op::Remove(key) => {
                let old = if model.get(&key.index()).is_some_and(|&(k, _)| k == key) {
                    model.remove(&key.index()).map(|(_, v)| v)
                } else {
                    None
                };
                prop_assert_eq!(map.remove(key), old);
            }
            Op::Clear => {
                map.clear();
                model.clear();
            }
            Op::Shrink => map.shrink_to_fit(),
        }
        let (slots, keys) = queries[step % queries.len()];
        check_queries(&map, &model, slots, keys, directions, prefix)?;
        check_order(&map, &model, &probes)?;
        for b in [slots.0, slots.1] {
            if let Included(index) | Excluded(index) = b {
                check_order(&map, &model, &[index])?;
            }
        }
    }
    for &(slots, keys) in queries {
        check_queries(&map, &model, slots, keys, directions, prefix)?;
        check_mutations(&mut map, &mut model, slots, keys, directions, delta)?;
    }
    pop_all(&mut map, &mut model, directions)?;
    map.shrink_to_fit();
    pop_all(&mut map, &mut model, directions)
}

macro_rules! suite {
    ($name:ident, $key:ty, $page:literal, $directory:ty) => {
        mod $name {
            use super::*;

            proptest! {
                #![proptest_config(config())]
                #[test]
                fn ordered_model(
                    ops in prop::collection::vec(operations::<$key>(), 0..max_len()),
                    queries in prop::collection::vec(
                        (slot_bounds(), key_bounds::<$key>()), 1..9,
                    ),
                    directions in prop::collection::vec(any::<bool>(), 1..max_len()),
                    prefix in 0..max_len(),
                    delta in 1_u32..=u32::MAX,
                ) {
                    let mut queries = queries;
                    // These cases are guaranteed, rather than left to random coverage.
                    for slots in [
                        (Unbounded, Unbounded),
                        (Excluded(usize::MAX), Unbounded),
                        (Included(usize::MAX), Included(usize::MAX)),
                        (Unbounded, Excluded(usize::MAX)),
                        (Unbounded, Included(usize::MAX)),
                        (Unbounded, Excluded(0)),
                        (Excluded(0), Included(0)),
                        (Included($page + 1), Included($page - 1)),
                        (Included($page - 1), Excluded($page + 1)),
                        (Excluded($page - 1), Included($page + 1)),
                        (Excluded($page), Excluded($page)),
                    ] {
                        queries.push((slots, (Unbounded, Unbounded)));
                    }
                    for generation in [0, 1, 127, 255] {
                        let key = <$key as TestKey>::make($page as u16, generation);
                        queries.push(((Included($page), Included($page)),
                            (Included(key), Included(key))));
                    }
                    let upper = <$key as TestKey>::upper_bound();
                    let zero = <$key as TestKey>::make(0, 255);
                    for bounds in [
                        (Included(upper), Unbounded),
                        (Excluded(upper), Unbounded),
                        (Unbounded, Included(upper)),
                        (Unbounded, Excluded(upper)),
                        (Unbounded, Excluded(zero)),
                        (Included(upper), Included(zero)),
                    ] {
                        queries.push(((Unbounded, Unbounded), bounds));
                    }
                    run_sequence::<$key, $page, $directory>(
                        &ops, &queries, &directions, prefix, delta,
                    )?;
                }
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
        suite!(radix_1, $key, 1, Radix);
        suite!(tree_2, $key, 2, Tree);
        suite!(radix_2, $key, 2, Radix);
        suite!(tree_3, $key, 3, Tree);
        suite!(radix_3, $key, 3, Radix);
        suite!(tree_4, $key, 4, Tree);
        suite!(radix_4, $key, 4, Radix);
        suite!(tree_8, $key, 8, Tree);
        suite!(radix_8, $key, 8, Radix);
        suite!(tree_1024, $key, 1024, Tree);
        suite!(radix_1024, $key, 1024, Radix);
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

fn high_indices<const PAGE: usize, D: Directory>(
    offsets: &[u16],
    bounds: &[Bounds<usize>],
    directions: &[bool],
) -> TestCaseResult {
    let mut map = PagedDirectMap::<usize, u32, PAGE, D>::new();
    let mut model = Model::new();
    for index in [0, usize::MAX - PAGE, usize::MAX - 1, usize::MAX]
        .into_iter()
        .chain(
            offsets
                .iter()
                .map(|&offset| usize::MAX - usize::from(offset)),
        )
    {
        map.insert(index, index as u32);
        model.insert(index, (index, index as u32));
    }
    for &range in bounds {
        check_queries(&map, &model, range, range, directions, 2)?;
        check_mutations(&mut map, &mut model, range, range, directions, 1)?;
    }
    check_order(
        &map,
        &model,
        &[0, usize::MAX - PAGE, usize::MAX - 1, usize::MAX],
    )?;
    pop_all(&mut map, &mut model, directions)
}

proptest! {
    #![proptest_config(config())]
    #[test]
    fn wide_ranges_near_usize_max(
        offsets in prop::collection::vec(0_u16..2052, 0..max_len()),
        bounds in prop::collection::vec(slot_bounds(), 1..9),
        directions in prop::collection::vec(any::<bool>(), 1..max_len()),
        a in 0_usize..2052,
        b in 0_usize..2052,
    ) {
        let mut ranges = bounds;
        for start in [Included(usize::MAX - a), Excluded(usize::MAX - a)] {
            for end in [Included(usize::MAX - b), Excluded(usize::MAX - b), Unbounded] {
                ranges.push((start, end));
            }
        }
        ranges.extend([
            (Unbounded, Unbounded),
            (Included(usize::MAX), Included(usize::MAX)),
            (Excluded(usize::MAX), Unbounded),
            (Unbounded, Excluded(usize::MAX)),
            (Included(usize::MAX - 1), Included(usize::MAX)),
        ]);
        high_indices::<1, Tree>(&offsets, &ranges, &directions)?;
        high_indices::<1, Radix>(&offsets, &ranges, &directions)?;
        high_indices::<2, Tree>(&offsets, &ranges, &directions)?;
        high_indices::<2, Radix>(&offsets, &ranges, &directions)?;
        high_indices::<3, Tree>(&offsets, &ranges, &directions)?;
        high_indices::<3, Radix>(&offsets, &ranges, &directions)?;
        high_indices::<4, Tree>(&offsets, &ranges, &directions)?;
        high_indices::<4, Radix>(&offsets, &ranges, &directions)?;
        high_indices::<8, Tree>(&offsets, &ranges, &directions)?;
        high_indices::<8, Radix>(&offsets, &ranges, &directions)?;
        high_indices::<1024, Tree>(&offsets, &ranges, &directions)?;
        high_indices::<1024, Radix>(&offsets, &ranges, &directions)?;
    }
}
