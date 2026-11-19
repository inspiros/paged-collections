//! Model-based tests for bits, slot order, page lifetime, and set algebra.

use std::collections::BTreeSet;
use std::iter::FusedIterator;
use std::ops::{Bound, RangeBounds};
use std::panic::{AssertUnwindSafe, catch_unwind};

use paged_collections::bit_set::{DEFAULT_BIT_PAGE, PagedBitSet};
use paged_collections::page::{Directory, Flat, Radix, Tree};
use proptest::prelude::*;
use proptest::test_runner::TestCaseResult;

type Bounds = (Bound<usize>, Bound<usize>);

#[derive(Clone, Debug)]
enum Op {
    Insert(usize),
    Remove(usize),
    Toggle(usize),
    InsertRange(Bounds),
    RemoveRange(Bounds),
    Query(usize, Bounds),
    Retain(usize),
    Extend(Vec<usize>),
    PopFirst,
    PopLast,
    Clear,
    Reset,
    Shrink,
}

fn keys<const PAGE: usize>() -> impl Strategy<Value = usize> {
    let page_bits = PAGE * 64;
    let boundaries = [0, 64, 128, page_bits, page_bits.saturating_mul(2)];
    (prop::sample::select(boundaries.to_vec()), 0..9_usize)
        .prop_map(|(boundary, offset)| boundary.saturating_sub(4).saturating_add(offset))
}

fn bound(index: usize, kind: u8) -> Bound<usize> {
    match kind {
        0 => Bound::Unbounded,
        1 => Bound::Excluded(index),
        _ => Bound::Included(index),
    }
}

fn operations<const PAGE: usize>() -> impl Strategy<Value = Op> {
    let bounds = (keys::<PAGE>(), keys::<PAGE>(), 0..4_u8, 0..4_u8)
        .prop_map(|(a, b, start, end)| (bound(a, start), bound(b, end)))
        .boxed();
    let inserts = (keys::<PAGE>(), 0..70_usize, 1..4_u8, 1..4_u8, any::<bool>()).prop_map(
        |(a, width, start, end, flipped)| {
            let b = a.saturating_add(width);
            let (a, b) = if flipped { (b, a) } else { (a, b) };
            Op::InsertRange((bound(a, start), bound(b, end)))
        },
    );
    prop_oneof![
        4 => keys::<PAGE>().prop_map(Op::Insert),
        2 => keys::<PAGE>().prop_map(Op::Remove),
        4 => keys::<PAGE>().prop_map(Op::Toggle),
        5 => inserts,
        4 => bounds.clone().prop_map(Op::RemoveRange),
        4 => (keys::<PAGE>(), bounds).prop_map(|(key, range)| Op::Query(key, range)),
        2 => (2..7_usize).prop_map(Op::Retain),
        4 => prop::collection::vec(keys::<PAGE>(), 0..10).prop_map(Op::Extend),
        1 => Just(Op::PopFirst),
        1 => Just(Op::PopLast),
        1 => Just(Op::Clear),
        1 => Just(Op::Reset),
        1 => Just(Op::Shrink),
    ]
}

fn page_numbers<const PAGE: usize>(keys: &BTreeSet<usize>) -> BTreeSet<usize> {
    keys.iter().map(|key| key / 64 / PAGE).collect()
}

fn elements(keys: &BTreeSet<usize>) -> Vec<usize> {
    keys.iter().copied().collect()
}

fn check_range<const PAGE: usize, D: Directory>(
    set: &PagedBitSet<usize, PAGE, D>,
    model: &BTreeSet<usize>,
    bounds: Bounds,
) -> TestCaseResult {
    let expected: Vec<_> = model
        .iter()
        .copied()
        .filter(|key| bounds.contains(key))
        .collect();
    let mut iter = set.range(bounds);
    prop_assert_eq!(set.count_range(bounds), expected.len());
    prop_assert!(iter.clone().eq(expected.iter().copied()));
    let mut keys = expected.into_iter();
    while let Some(first) = keys.next() {
        prop_assert_eq!(iter.next(), Some(first));
        prop_assert_eq!(iter.next_back(), keys.next_back());
    }
    prop_assert_eq!(iter.next(), None);
    prop_assert_eq!(iter.next_back(), None);
    Ok(())
}

fn check_exact(
    mut iter: impl DoubleEndedIterator<Item = usize> + ExactSizeIterator + FusedIterator,
    model: &BTreeSet<usize>,
    sides: &[bool],
) -> TestCaseResult {
    let mut keys = model.iter().copied();
    for &from_back in sides.iter().cycle().take(model.len() + 2) {
        prop_assert_eq!(iter.len(), keys.len());
        prop_assert_eq!(iter.size_hint(), (keys.len(), Some(keys.len())));
        if from_back {
            prop_assert_eq!(iter.next_back(), keys.next_back());
        } else {
            prop_assert_eq!(iter.next(), keys.next());
        }
    }
    prop_assert_eq!(iter.len(), 0);
    prop_assert_eq!(iter.next(), None);
    prop_assert_eq!(iter.next_back(), None);
    Ok(())
}

fn check_lazy(
    mut iter: impl FusedIterator<Item = usize> + Clone,
    expected: &BTreeSet<usize>,
) -> TestCaseResult {
    prop_assert!(iter.clone().eq(expected.iter().copied()));
    let mut keys = expected.iter().copied();
    prop_assert_eq!(iter.next(), keys.next());
    prop_assert!(iter.clone().eq(keys.clone()));
    prop_assert!(iter.by_ref().eq(keys));
    prop_assert_eq!(iter.next(), None);
    Ok(())
}

fn check_algebra<const PAGE: usize, D: Directory, D2: Directory>(
    left: &PagedBitSet<usize, PAGE, D>,
    a: &BTreeSet<usize>,
    left_pages: &BTreeSet<usize>,
    right_keys: Vec<usize>,
) -> TestCaseResult
where
    D::Store<u64, PAGE>: Clone,
{
    let mut right: PagedBitSet<usize, PAGE, D2> = right_keys.iter().copied().collect();
    let b: BTreeSet<_> = right_keys.into_iter().collect();
    // Try to leave an allocated zero page beyond the generated keys.
    let empty_key = (PAGE * 64).saturating_mul(3);
    if right.insert(empty_key) {
        right.remove(empty_key);
    }

    prop_assert_eq!(left.is_disjoint(&right), a.is_disjoint(&b));
    prop_assert_eq!(left.is_subset(&right), a.is_subset(&b));
    prop_assert_eq!(left.is_superset(&right), a.is_superset(&b));
    prop_assert_eq!(left == &right, a == &b);

    let union: BTreeSet<_> = a.union(&b).copied().collect();
    let intersection: BTreeSet<_> = a.intersection(&b).copied().collect();
    let difference: BTreeSet<_> = a.difference(&b).copied().collect();
    let symmetric_difference: BTreeSet<_> = a.symmetric_difference(&b).copied().collect();

    check_lazy(left.union(&right), &union)?;
    check_lazy(left.intersection(&right), &intersection)?;
    check_lazy(left.difference(&right), &difference)?;
    check_lazy(left.symmetric_difference(&right), &symmetric_difference)?;

    let grown_pages = left_pages.union(&page_numbers::<PAGE>(&b)).count();
    let mut or = left.clone();
    or |= &right;
    let mut and = left.clone();
    and &= &right;
    let mut sub = left.clone();
    sub -= &right;
    let mut xor = left.clone();
    xor ^= &right;
    let cases = [
        (left | &right, or, union, grown_pages),
        (left & &right, and, intersection, left_pages.len()),
        (left - &right, sub, difference, left_pages.len()),
        (left ^ &right, xor, symmetric_difference, grown_pages),
    ];
    for (fresh, assigned, expected, allocated) in cases {
        prop_assert!(fresh.iter().eq(expected.iter().copied()));
        prop_assert!(assigned.iter().eq(expected.iter().copied()));
        prop_assert_eq!(fresh.len(), expected.len());
        prop_assert_eq!(assigned.len(), expected.len());
        prop_assert_eq!(
            fresh.allocated_pages(),
            page_numbers::<PAGE>(&expected).len()
        );
        prop_assert_eq!(assigned.allocated_pages(), allocated);
    }
    Ok(())
}

fn check_model<const PAGE: usize, D: Directory, D2: Directory>(
    ops: Vec<Op>,
    right_keys: Vec<usize>,
    sides: &[bool],
) -> TestCaseResult
where
    D::Store<u64, PAGE>: Clone,
{
    let mut set = PagedBitSet::<usize, PAGE, D>::default();
    let mut model = BTreeSet::new();
    let mut allocated = BTreeSet::new();

    for op in ops {
        match op {
            Op::Insert(key) => prop_assert_eq!(set.insert(key), model.insert(key)),
            Op::Remove(key) => prop_assert_eq!(set.remove(key), model.remove(&key)),
            Op::Toggle(key) => {
                let present = !model.remove(&key);
                if present {
                    model.insert(key);
                }
                prop_assert_eq!(set.toggle(key), present);
            }
            Op::InsertRange(bounds) => {
                let before = model.len();
                // Generated insert ranges have two finite endpoints.
                let (Bound::Included(start) | Bound::Excluded(start)) = bounds.0 else {
                    unreachable!();
                };
                let (Bound::Included(end) | Bound::Excluded(end)) = bounds.1 else {
                    unreachable!();
                };
                for key in start..=end {
                    if bounds.contains(&key) {
                        model.insert(key);
                    }
                }
                prop_assert_eq!(set.insert_range(bounds), model.len() - before);
            }
            Op::RemoveRange(bounds) => {
                let before = model.len();
                model.retain(|key| !bounds.contains(key));
                prop_assert_eq!(set.remove_range(bounds), before - model.len());
            }
            Op::Query(key, bounds) => {
                prop_assert_eq!(set.contains(key), model.contains(&key));
                prop_assert_eq!(set.next_occupant(key), model.range(key..).next().copied());
                prop_assert_eq!(
                    set.prev_occupant(key),
                    model.range(..=key).next_back().copied()
                );
                check_range(&set, &model, bounds)?;
            }
            Op::Retain(modulus) => {
                let mut called = Vec::new();
                set.retain(|key| {
                    called.push(key);
                    key % modulus != 0
                });
                prop_assert_eq!(called, elements(&model));
                model.retain(|key| key % modulus != 0);
            }
            Op::Extend(keys) => {
                let at = keys.len() / 2;
                set.extend(keys[..at].iter());
                set.extend(keys[at..].iter().copied());
                model.extend(keys);
            }
            Op::PopFirst => prop_assert_eq!(set.pop_first(), model.pop_first()),
            Op::PopLast => prop_assert_eq!(set.pop_last(), model.pop_last()),
            Op::Clear => {
                set.clear();
                model.clear();
            }
            Op::Reset => {
                set.reset();
                model.clear();
                allocated.clear();
            }
            Op::Shrink => {
                set.shrink_to_fit();
                allocated = page_numbers::<PAGE>(&model);
            }
        }
        allocated.extend(page_numbers::<PAGE>(&model));
        prop_assert_eq!(set.allocated_pages(), allocated.len());
        prop_assert_eq!(set.len(), model.len());
        prop_assert_eq!(set.is_empty(), model.is_empty());
        prop_assert_eq!(set.first(), model.first().copied());
        prop_assert_eq!(set.last(), model.last().copied());
        prop_assert!(set.iter().eq(model.iter().copied()));
    }

    check_algebra::<PAGE, D, D2>(&set, &model, &allocated, right_keys)?;
    check_exact(IntoIterator::into_iter(&set), &model, sides)?;
    {
        // Generic over `D`, the iterator's borrow lasts to the end of its scope.
        let mut iter = set.iter();
        prop_assert_eq!(iter.next_back(), model.last().copied());
        if let Some(&last) = model.last() {
            prop_assert!(iter.clone().eq(model.range(..last).copied()));
        }
    }
    check_exact(set.into_iter(), &model, sides)
}

fn config() -> ProptestConfig {
    let defaults = ProptestConfig::default();
    if cfg!(miri) {
        ProptestConfig {
            cases: defaults.cases.min(8),
            failure_persistence: None,
            ..defaults
        }
    } else {
        ProptestConfig {
            cases: defaults.cases.min(64),
            ..defaults
        }
    }
}

macro_rules! model_test {
    ($name:ident, $page:literal, $left:ty, $right:ty) => {
        proptest! {
            #![proptest_config(config())]
            #[test]
            fn $name(
                ops in prop::collection::vec(operations::<$page>(), 0..if cfg!(miri) { 12 } else { 45 }),
                right in prop::collection::vec(keys::<$page>(), 0..30),
                sides in prop::collection::vec(any::<bool>(), 1..20),
            ) {
                check_model::<$page, $left, $right>(ops, right, &sides)?;
            }
        }
    };
}

model_test!(flat_1, 1, Flat, Radix);
model_test!(flat_2, 2, Flat, Radix);
model_test!(flat_512, 512, Flat, Radix);
model_test!(radix_1, 1, Radix, Tree);
model_test!(radix_2, 2, Radix, Tree);
model_test!(radix_512, 512, Radix, Tree);
model_test!(tree_1, 1, Tree, Flat);
model_test!(tree_2, 2, Tree, Flat);
model_test!(tree_512, 512, Tree, Flat);

#[test]
fn full_u8_domain() {
    let mut set = PagedBitSet::<u8, 1>::new();
    assert_eq!(DEFAULT_BIT_PAGE, 512);
    assert_eq!(PagedBitSet::<u8, 1>::PAGE_SIZE, 1);
    assert_eq!(PagedBitSet::<u8, 1>::BITS_PER_PAGE, 64);
    assert_eq!(set.insert_range(..=u8::MAX), 256);
    assert_eq!(set.allocated_pages(), 4);
    assert_eq!(set.len(), 256);
    assert!(set.iter().eq(u8::MIN..=u8::MAX));
    assert!(set.range(..).rev().eq((u8::MIN..=u8::MAX).rev()));
    assert_eq!(set.remove_range(63..=191), 129);
    assert_eq!(set.allocated_pages(), 4);
    set.shrink_to_fit();
    assert_eq!(set.allocated_pages(), 2);
    assert_eq!(set.count_range(64..192), 0);
    set.clear();
    assert_eq!(set.allocated_pages(), 2);
    set.shrink_to_fit();
    assert_eq!(set.allocated_pages(), 0);
}

fn check_max<const PAGE: usize, D: Directory>() {
    let mut set = PagedBitSet::<usize, PAGE, D>::new();
    let max = usize::MAX;
    assert!(set.insert(max));
    assert!(set.insert(0));
    assert_eq!(set.insert_range(max - 68..=max), 68);
    assert_eq!(set.insert_range(max - 5..), 0);
    assert_eq!(set.len(), 70);
    assert_eq!(set.count_range(..=max), 70);
    assert_eq!(set.next_occupant(max - 69), Some(max - 68));
    assert_eq!(set.next_occupant(max), Some(max));
    assert_eq!(set.prev_occupant(max), Some(max));
    assert_eq!(set.prev_occupant(max - 69), Some(0));

    let mut range = set.range(max - 68..=max);
    assert_eq!(range.next_back(), Some(max));
    assert_eq!(range.next(), Some(max - 68));
    assert!(range.eq(max - 67..max));
    assert!(set.iter().rev().eq((max - 68..=max).rev().chain([0])));
    let other: PagedBitSet<usize, PAGE, Tree> = [max - 1, max, 1].into_iter().collect();
    assert!(set.intersection(&other).eq([max - 1, max]));
    assert!((&set ^ &other).iter().rev().take(2).eq([max - 2, max - 3]));

    for empty in [
        (Bound::Excluded(max), Bound::Unbounded),
        (Bound::Unbounded, Bound::Excluded(0)),
        (Bound::Included(10), Bound::Included(9)),
        (Bound::Excluded(5), Bound::Included(5)),
    ] {
        assert_eq!(set.insert_range(empty), 0);
        assert_eq!(set.remove_range(empty), 0);
        assert_eq!(set.count_range(empty), 0);
        assert_eq!(set.range(empty).next_back(), None);
    }
    assert_eq!(set.remove_range(max - 66..max), 66);
    assert_eq!(set.pop_last(), Some(max));
    assert_eq!(set.pop_first(), Some(0));
    assert!(set.into_iter().eq([max - 68, max - 67]));
}

#[test]
fn maximum_slot_and_partly_addressable_page() {
    check_max::<1, Radix>();
    check_max::<2, Tree>();
    check_max::<3, Radix>();
    check_max::<512, Tree>();
}

#[test]
fn retain_panic_keeps_prior_removals_and_remaining_keys() {
    let original = [1, 2, 3, 63, 64, 127, 128, 129];
    let mut set: PagedBitSet<usize, 1, Radix> = original.into_iter().collect();
    let mut visited = Vec::new();
    let result = catch_unwind(AssertUnwindSafe(|| {
        set.retain(|key| {
            visited.push(key);
            assert_ne!(key, 3);
            key != 1
        });
    }));
    assert!(result.is_err());
    assert_eq!(visited, [1, 2, 3]);
    assert_eq!(set.len(), 7);
    assert_eq!(set.allocated_pages(), 3);
    assert!(set.iter().eq(original.into_iter().skip(1)));
    assert!(set.insert(1));
    assert_eq!(set.len(), original.len());
    assert_eq!(format!("{set:?}"), "{1, 2, 3, 63, 64, 127, 128, 129}");
}

#[cfg(feature = "serde")]
#[test]
fn serde_sequence_round_trip_and_sparse_set_interchange() {
    use paged_collections::sparse_set::PagedSparseSet;

    let mut bits: PagedBitSet<u16, 1, Tree> = [128, 65, 0].into_iter().collect();
    bits.insert(1024);
    bits.remove(1024);
    let payload = serde_json::to_string(&bits).unwrap();
    assert_eq!(payload, "[0,65,128]");
    let round_trip: PagedBitSet<u16, 1, Radix> = serde_json::from_str(&payload).unwrap();
    assert_eq!(round_trip, bits);
    assert_eq!(round_trip.allocated_pages(), 3);

    let sparse: PagedSparseSet<u16, 4> = serde_json::from_str(&payload).unwrap();
    assert_eq!(sparse.as_slice(), [0, 65, 128]);
    let sparse: PagedSparseSet<u16, 8, Tree> = [128, 0, 65].into_iter().collect();
    let payload = serde_json::to_string(&sparse).unwrap();
    let from_sparse: PagedBitSet<u16, 1> = serde_json::from_str(&payload).unwrap();
    assert_eq!(from_sparse, bits);
    let duplicates: PagedBitSet<u16, 1> = serde_json::from_str("[65,128,65,0,128]").unwrap();
    assert_eq!(duplicates, bits);
}
