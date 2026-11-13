//! Radix page operations and mixed-end iterators against a B-tree oracle.

use std::collections::{BTreeMap, VecDeque};
use std::ops::RangeInclusive;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};

use paged_collections::page::{PageStore, RadixPages};
use proptest::prelude::*;

type Store = RadixPages<u8, 4>;
type Model = BTreeMap<usize, [u8; 4]>;

fn config() -> ProptestConfig {
    ProptestConfig {
        cases: if cfg!(miri) { 4 } else { 512 },
        failure_persistence: None,
        ..ProptestConfig::default()
    }
}

fn max_len() -> usize {
    if cfg!(miri) { 16 } else { 200 }
}

fn numbers(bases: [usize; 3]) -> BoxedStrategy<usize> {
    let boundaries: Vec<_> = (6..usize::BITS)
        .step_by(6)
        .flat_map(|shift| {
            let boundary = 1_usize << shift;
            [boundary - 1, boundary, boundary + 1]
        })
        .chain([0, usize::MAX - 1, usize::MAX])
        .collect();
    prop_oneof![
        3 => 0_usize..300,
        3 => (0_usize..3, 0_usize..256).prop_map(move |(base, offset)| (bases[base] & !255) | offset),
        2 => prop::sample::select(boundaries),
        2 => any::<usize>(),
    ]
    .boxed()
}

#[derive(Clone, Debug)]
struct Op {
    kind: u8,
    number: usize,
    other: usize,
    marker: u8,
    directions: Vec<bool>,
}

fn operations() -> impl Strategy<Value = (Vec<usize>, Vec<Op>)> {
    any::<[usize; 3]>().prop_flat_map(|bases| {
        let number = numbers(bases);
        (
            prop::collection::vec(number.clone(), 0..max_len()),
            prop::collection::vec(
                (
                    0_u8..16,
                    number.clone(),
                    number,
                    any::<u8>(),
                    prop::collection::vec(any::<bool>(), 1..17),
                )
                    .prop_map(|(kind, number, other, marker, directions)| Op {
                        kind,
                        number,
                        other,
                        marker,
                        directions,
                    }),
                1..max_len(),
            ),
        )
    })
}

// Retain all yielded references together, including mutable ones. The oracle
// chooses the same end on every step; comparisons preserve the full sequence.
fn walk<I: DoubleEndedIterator>(mut iter: I, directions: &[bool]) -> Vec<I::Item> {
    let mut items = Vec::new();
    for &back in directions.iter().cycle() {
        let next = if back { iter.next_back() } else { iter.next() };
        match next {
            Some(item) => items.push(item),
            None => break,
        }
    }
    assert!(iter.next().is_none());
    assert!(iter.next_back().is_none());
    items
}

fn expected(
    model: &Model,
    range: RangeInclusive<usize>,
    directions: &[bool],
) -> Vec<(usize, [u8; 4])> {
    let entries: VecDeque<_> = if range.is_empty() {
        VecDeque::new()
    } else {
        model.range(range).map(|(&n, &page)| (n, page)).collect()
    };
    walk(entries.into_iter(), directions)
}

fn shared(store: &Store, model: &Model, range: RangeInclusive<usize>, directions: &[bool]) {
    let mut iter = store.pages_in(range.clone());
    let mut oracle: VecDeque<_> = expected(model, range, &[false]).into();
    for &back in directions {
        let actual = if back { iter.next_back() } else { iter.next() };
        let wanted = if back {
            oracle.pop_back()
        } else {
            oracle.pop_front()
        };
        assert_eq!(actual.map(|(n, p)| (n, *p)), wanted);
        // Clone a partially consumed iterator, with both sides expanded.
        let copy = iter.clone();
        let remaining = walk(copy, directions);
        let wanted = walk(oracle.clone().into_iter(), directions);
        assert_eq!(
            remaining
                .into_iter()
                .map(|(n, p)| (n, *p))
                .collect::<Vec<_>>(),
            wanted
        );
    }
    assert_eq!(
        iter.map(|(n, p)| (n, *p)).collect::<Vec<_>>(),
        Vec::from(oracle)
    );
}

fn mutable(
    store: &mut Store,
    model: &mut Model,
    range: RangeInclusive<usize>,
    directions: &[bool],
    marker: u8,
    all: bool,
) {
    let wanted = expected(model, range.clone(), directions);
    let iter = if all {
        store.pages_mut()
    } else {
        store.pages_in_mut(range)
    };
    let pages = walk(iter, directions);
    assert_eq!(pages.len(), wanted.len());
    for ((n, page), (expected_n, expected_page)) in pages.into_iter().zip(wanted) {
        assert_eq!((n, *page), (expected_n, expected_page));
        *page = [marker; 4];
        model.insert(n, [marker; 4]);
    }
}

fn check(store: &Store, model: &Model) {
    assert_eq!(store.allocated(), model.len());
    assert_eq!(
        store.pages().map(|(n, p)| (n, *p)).collect::<Vec<_>>(),
        model.iter().map(|(&n, &p)| (n, p)).collect::<Vec<_>>()
    );
    for (&n, p) in model {
        assert_eq!(store.get(n), Some(p));
    }
}

fn run(initial: Vec<usize>, ops: Vec<Op>) {
    let mut store = Store::default();
    let mut model = Model::new();
    for n in initial {
        store.get_or_alloc(n);
        model.entry(n).or_default();
        check(&store, &model);
    }
    for op in ops {
        // Half the operations target a live page when possible.
        let n = if op.marker & 1 == 0 && !model.is_empty() {
            *model.keys().nth(op.number % model.len()).unwrap()
        } else {
            op.number
        };
        let range = if op.marker & 2 == 0 {
            n..=op.other
        } else {
            op.other..=n
        };
        match op.kind {
            0 => assert_eq!(store.get(n), model.get(&n)),
            1 | 2 => {
                if let Some(page) = store.get_mut(n) {
                    *page = [op.marker; 4];
                    *model.get_mut(&n).unwrap() = [op.marker; 4];
                } else {
                    assert!(!model.contains_key(&n));
                }
            }
            3..=6 => {
                assert_eq!(store.get_or_alloc(n), model.entry(n).or_default());
                // Write through get_mut, independently of allocation.
                *store.get_mut(n).unwrap() = [op.marker; 4];
                model.insert(n, [op.marker; 4]);
            }
            7 => assert_eq!(store.remove(n).map(|p| *p), model.remove(&n)),
            8 => {
                store.clear();
                model.clear();
            }
            9 => store.shrink_to_fit(),
            10 => {
                let mut clone = store.clone();
                check(&clone, &model);
                clone.get_or_alloc(n)[0] ^= 1;
                check(&store, &model);
            }
            11 => {
                let actual = walk(store.pages(), &op.directions);
                let wanted = expected(&model, 0..=usize::MAX, &op.directions);
                assert_eq!(
                    actual.into_iter().map(|(n, p)| (n, *p)).collect::<Vec<_>>(),
                    wanted
                );
            }
            12 => shared(&store, &model, range, &op.directions),
            13 => mutable(
                &mut store,
                &mut model,
                range,
                &op.directions,
                op.marker,
                false,
            ),
            14 => mutable(
                &mut store,
                &mut model,
                0..=usize::MAX,
                &op.directions,
                op.marker,
                true,
            ),
            15 => {
                let actual = walk(store.clone().into_pages(), &op.directions);
                let wanted = expected(&model, 0..=usize::MAX, &op.directions);
                assert_eq!(
                    actual.into_iter().map(|(n, p)| (n, *p)).collect::<Vec<_>>(),
                    wanted
                );
            }
            _ => unreachable!(),
        }
        check(&store, &model);
    }
}

proptest! {
    #![proptest_config(config())]
    #[test]
    fn model((initial, ops) in operations()) {
        run(initial, ops);
    }
}

#[test]
fn boundaries_and_every_direction_pattern() {
    let mut keys: Vec<_> = (0..usize::BITS)
        .step_by(6)
        .flat_map(|shift| {
            let n = 1_usize << shift;
            [n - 1, n, n + 1]
        })
        .chain([0, usize::MAX - 1, usize::MAX])
        .collect();
    if cfg!(miri) {
        keys = vec![0, 1, 63, 64, 4095, 4096, usize::MAX - 1, usize::MAX];
    }
    let mut store = Store::default();
    let mut model = Model::new();
    for &n in &keys {
        store.get_or_alloc(n).fill(n as u8);
        model.insert(n, [n as u8; 4]);
    }
    for pattern in 0_u8..if cfg!(miri) { 2 } else { 16 } {
        let directions: Vec<_> = (0..4).map(|bit| pattern & (1 << bit) != 0).collect();
        for &lo in &keys {
            for hi in [lo.saturating_sub(1), lo, lo.saturating_add(1), usize::MAX] {
                shared(&store, &model, lo..=hi, &directions);
                mutable(&mut store, &mut model, lo..=hi, &directions, pattern, false);
            }
        }
        check(&store, &model);
    }
    for n in keys {
        assert_eq!(store.remove(n).map(|p| *p), model.remove(&n));
        check(&store, &model);
    }
}

#[test]
fn deepest_paths_on_both_ends() {
    // One leaf per level on the lowest path and on the highest path, so a
    // walker started from both ends holds a slice iterator for every node of
    // both paths at once (the most `LEVELS` must hold).
    let depth = usize::BITS.div_ceil(6);
    let low: Vec<usize> = core::iter::once(0)
        .chain((0..depth - 1).map(|k| 1 << (6 * k)))
        .collect();
    let keys: Vec<usize> = low
        .iter()
        .copied()
        .chain(low.iter().map(|&c| usize::MAX - c))
        .collect();
    let mut store = Store::default();
    let mut model = Model::new();
    for &n in &keys {
        store.get_or_alloc(n).fill(n as u8);
        model.insert(n, [n as u8; 4]);
    }
    check(&store, &model);
    let mut iter = store.pages();
    let mut oracle: VecDeque<_> = model.iter().map(|(&n, &p)| (n, p)).collect();
    assert_eq!(iter.next().map(|(n, p)| (n, *p)), oracle.pop_front());
    assert_eq!(iter.next_back().map(|(n, p)| (n, *p)), oracle.pop_back());
    for pattern in [[false, true], [true, true], [false, false], [true, false]] {
        let rest = walk(iter.clone(), &pattern);
        let wanted = walk(oracle.clone().into_iter(), &pattern);
        assert_eq!(
            rest.into_iter().map(|(n, p)| (n, *p)).collect::<Vec<_>>(),
            wanted
        );
    }
    for pattern in [[false, true], [true, false]] {
        mutable(&mut store, &mut model, 0..=usize::MAX, &pattern, 7, true);
    }
    check(&store, &model);
}

#[test]
fn exhausted_ranges() {
    let mut store = Store::default();
    store.get_or_alloc(usize::MAX);
    let mut range = usize::MAX..=usize::MAX;
    range.next();
    assert!(store.pages_in(range.clone()).next().is_none());
    assert!(store.pages_in_mut(range).next_back().is_none());
}

#[test]
fn panicking_default_preserves_directory() {
    static PANIC: AtomicBool = AtomicBool::new(false);
    #[derive(Debug)]
    struct Slot;
    impl Default for Slot {
        fn default() -> Self {
            assert!(!PANIC.load(Ordering::Relaxed), "default panic");
            Self
        }
    }
    let mut store = RadixPages::<Slot, 4>::default();
    for n in [0, 64, usize::MAX] {
        store.get_or_alloc(n);
    }
    PANIC.store(true, Ordering::Relaxed);
    // Existing pages must not call Default at all.
    store.get_or_alloc(64);
    for n in [1, 4096, usize::MAX - 1] {
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                store.get_or_alloc(n);
            }))
            .is_err()
        );
        assert_eq!(store.allocated(), 3);
        assert_eq!(
            store.pages().map(|(n, _)| n).collect::<Vec<_>>(),
            [0, 64, usize::MAX]
        );
        assert!(store.get(n).is_none());
    }
    PANIC.store(false, Ordering::Relaxed);
}

#[test]
fn panicking_drop_leaves_clear_empty() {
    static PANIC: AtomicBool = AtomicBool::new(true);
    #[derive(Default)]
    struct Slot;
    impl Drop for Slot {
        fn drop(&mut self) {
            assert!(!PANIC.swap(false, Ordering::Relaxed), "drop panic");
        }
    }
    let mut store = RadixPages::<Slot, 2>::default();
    for n in [0, 64, usize::MAX] {
        store.get_or_alloc(n);
    }
    assert!(catch_unwind(AssertUnwindSafe(|| store.clear())).is_err());
    assert_eq!(store.allocated(), 0);
    assert!(store.pages().next().is_none());
    assert!(store.get(0).is_none());
    store.get_or_alloc(usize::MAX);
    assert_eq!(store.allocated(), 1);
}
