//! Model tests for heap ordering, keyed mutation, generations and panic safety.

#![cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]

use std::cell::Cell;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;
use std::num::NonZeroU32;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;

use paged_collections::id_alloc::Id;
use paged_collections::key::Key;
use paged_collections::page::{Directory, Flat, Radix};
use paged_collections::priority_queue::PagedPriorityQueue;
use paged_collections::sparse_set::Slot;
use proptest::prelude::*;
use proptest::test_runner::TestCaseResult;

fn config() -> ProptestConfig {
    ProptestConfig {
        cases: if cfg!(miri) { 4 } else { 512 },
        failure_persistence: if cfg!(miri) {
            None
        } else {
            ProptestConfig::default().failure_persistence
        },
        ..ProptestConfig::default()
    }
}

fn max_len() -> usize {
    if cfg!(miri) { 24 } else { 200 }
}

#[derive(Clone, Debug)]
enum Op<K> {
    Push(K, i32),
    TryPush(K, i32),
    Increase(K, i32),
    Decrease(K, i32),
    Change(K, i32),
    ChangeWith(K, i32),
    Remove(K),
    Pop,
    Clear,
    Reserve(u8),
    Shrink,
    Drain,
    Extend(Vec<(K, i32)>),
}

fn operations<K: Key + Debug + 'static>(keys: BoxedStrategy<K>) -> BoxedStrategy<Op<K>> {
    // Frequent equal priorities and duplicate keys exercise strict updates
    // and unspecified tie order, alongside all i32 boundary values.
    let priorities = prop_oneof![4 => -8_i32..9, 1 => any::<i32>()];
    let entries = (keys.clone(), priorities);
    prop_oneof![
        12 => entries.clone().prop_map(|(k, p)| Op::Push(k, p)),
        8 => entries.clone().prop_map(|(k, p)| Op::TryPush(k, p)),
        6 => entries.clone().prop_map(|(k, p)| Op::Increase(k, p)),
        6 => entries.clone().prop_map(|(k, p)| Op::Decrease(k, p)),
        6 => entries.clone().prop_map(|(k, p)| Op::Change(k, p)),
        6 => entries.clone().prop_map(|(k, p)| Op::ChangeWith(k, p)),
        6 => keys.prop_map(Op::Remove),
        6 => Just(Op::Pop),
        1 => Just(Op::Clear),
        1 => (0_u8..24).prop_map(Op::Reserve),
        1 => Just(Op::Shrink),
        1 => Just(Op::Drain),
        2 => prop::collection::vec(entries, 0..8).prop_map(Op::Extend),
    ]
    .boxed()
}

fn model_push<K: Key + Ord>(model: &mut BTreeMap<K, i32>, key: K, priority: i32) -> Option<i32> {
    if let Some(old) = model.keys().find(|k| k.index() == key.index()).copied() {
        let previous = model.remove(&old).unwrap();
        model.insert(key, priority);
        if old == key { Some(previous) } else { None }
    } else {
        model.insert(key, priority);
        None
    }
}

fn check<K: Key + Ord + Debug, const PAGE: usize, D: Directory>(
    queue: &PagedPriorityQueue<K, i32, PAGE, D>,
    model: &BTreeMap<K, i32>,
    probes: &[K],
) -> TestCaseResult
where
    D::Store<Slot, PAGE>: Clone,
{
    prop_assert_eq!(queue.len(), model.len());
    prop_assert_eq!(queue.is_empty(), model.is_empty());
    prop_assert_eq!(queue.key_slice().len(), queue.priority_slice().len());
    prop_assert!(queue.capacity() >= queue.len());
    let actual: BTreeMap<_, _> = queue.iter().map(|(&k, &p)| (k, p)).collect();
    prop_assert_eq!(&actual, model);
    for key in probes {
        prop_assert_eq!(queue.contains_key(key), model.contains_key(key));
        prop_assert_eq!(queue.priority(key), model.get(key));
        prop_assert_eq!(queue.position(key).is_some(), model.contains_key(key));
    }
    for (pos, key) in queue.key_slice().iter().enumerate() {
        prop_assert_eq!(queue.position(key), Some(pos));
        prop_assert_eq!(queue.priority(key), Some(&queue.priority_slice()[pos]));
        if pos > 0 {
            prop_assert!(queue.priority_slice()[(pos - 1) / 4] >= queue.priority_slice()[pos]);
        }
    }
    prop_assert_eq!(queue.keys().copied().collect::<Vec<_>>(), queue.key_slice());
    prop_assert_eq!(
        queue.priorities().copied().collect::<Vec<_>>(),
        queue.priority_slice()
    );
    prop_assert_eq!(queue.iter().len(), model.len());
    prop_assert_eq!(queue.iter().rev().count(), model.len());
    match queue.peek() {
        Some((key, priority)) => {
            prop_assert_eq!(model.get(key), Some(priority));
            prop_assert_eq!(Some(priority), model.values().max());
        }
        None => prop_assert!(model.is_empty()),
    }

    let mut expected_priorities: Vec<_> = model.values().copied().collect();
    expected_priorities.sort_unstable();
    let sorted = queue.clone().into_sorted_vec();
    prop_assert_eq!(
        &sorted.iter().map(|e| e.1).collect::<Vec<_>>(),
        &expected_priorities
    );
    // Full pair multisets catch missing/duplicated keys within priority ties.
    prop_assert_eq!(
        sorted.iter().copied().collect::<BTreeMap<_, _>>(),
        model.clone()
    );
    prop_assert_eq!(sorted.len(), model.len());
    let mut descending = queue.clone().into_sorted_iter();
    prop_assert_eq!(descending.len(), model.len());
    expected_priorities.reverse();
    let entries: Vec<_> = descending.by_ref().collect();
    prop_assert_eq!(
        entries.iter().map(|e| e.1).collect::<Vec<_>>(),
        expected_priorities
    );
    prop_assert_eq!(
        entries.into_iter().collect::<BTreeMap<_, _>>(),
        model.clone()
    );
    prop_assert_eq!(descending.len(), 0);
    prop_assert_eq!(descending.next(), None);
    prop_assert_eq!(descending.next(), None);
    prop_assert_eq!(
        queue.clone().into_iter().collect::<BTreeMap<_, _>>(),
        model.clone()
    );
    Ok(())
}

fn run<K: Key + Ord + Debug, const PAGE: usize, D: Directory>(
    ops: &[Op<K>],
    probes: &[K],
) -> TestCaseResult
where
    D::Store<Slot, PAGE>: Clone,
{
    let mut queue = PagedPriorityQueue::<K, i32, PAGE, D>::with_capacity(3);
    let mut model = BTreeMap::new();
    let mut pages = BTreeSet::new();
    check(&queue, &model, probes)?;
    for op in ops {
        match op {
            &Op::Push(key, priority) => {
                prop_assert_eq!(
                    queue.push(key, priority),
                    model_push(&mut model, key, priority)
                );
                pages.insert(key.index() / PAGE);
            }
            &Op::TryPush(key, priority) => {
                let occupied_by_other = model.keys().any(|k| k.index() == key.index() && *k != key);
                let expected = if occupied_by_other {
                    Err((key, priority))
                } else {
                    pages.insert(key.index() / PAGE);
                    Ok(model_push(&mut model, key, priority))
                };
                prop_assert_eq!(queue.try_push(key, priority), expected);
            }
            &Op::Increase(key, priority) | &Op::Decrease(key, priority) => {
                let increase = matches!(op, Op::Increase(..));
                let changed = model.get(&key).is_none_or(|old| {
                    if increase {
                        priority > *old
                    } else {
                        priority < *old
                    }
                });
                let actual = if increase {
                    queue.push_increase(key, priority)
                } else {
                    queue.push_decrease(key, priority)
                };
                prop_assert_eq!(actual, changed);
                if changed {
                    model_push(&mut model, key, priority);
                    pages.insert(key.index() / PAGE);
                }
            }
            &Op::Change(key, priority) => {
                let expected = model.get_mut(&key).map(|p| std::mem::replace(p, priority));
                prop_assert_eq!(queue.change_priority(&key, priority), expected);
            }
            &Op::ChangeWith(key, delta) => {
                let mut calls = 0;
                let changed = queue.change_priority_with(&key, |p| {
                    calls += 1;
                    *p = p.wrapping_add(delta);
                });
                prop_assert_eq!(changed, model.contains_key(&key));
                prop_assert_eq!(calls, usize::from(changed));
                if let Some(p) = model.get_mut(&key) {
                    *p = p.wrapping_add(delta);
                }
            }
            Op::Remove(key) => {
                prop_assert_eq!(queue.remove(key), model.remove(key).map(|p| (*key, p)));
            }
            Op::Pop => {
                let greatest = model.values().max().copied();
                match queue.pop() {
                    Some((key, p)) => {
                        prop_assert_eq!(Some(p), greatest);
                        prop_assert_eq!(model.remove(&key), Some(p));
                    }
                    None => prop_assert!(model.is_empty()),
                }
            }
            Op::Clear => {
                let capacity = queue.capacity();
                queue.clear();
                model.clear();
                prop_assert_eq!(queue.capacity(), capacity);
            }
            &Op::Reserve(additional) => {
                queue.reserve(usize::from(additional));
                prop_assert!(queue.capacity() >= queue.len() + usize::from(additional));
            }
            Op::Shrink => {
                queue.shrink_to_fit();
                if model.is_empty() {
                    pages.clear();
                }
            }
            Op::Drain => {
                let capacity = queue.capacity();
                let mut drain = queue.drain();
                prop_assert_eq!(drain.len(), model.len());
                let drained: BTreeMap<_, _> = drain.by_ref().rev().collect();
                prop_assert_eq!(&drained, &model);
                prop_assert_eq!(drain.next(), None);
                drop(drain);
                model.clear();
                prop_assert_eq!(queue.capacity(), capacity);
            }
            Op::Extend(entries) => {
                queue.extend(entries.iter().copied());
                for &(key, p) in entries {
                    model_push(&mut model, key, p);
                    pages.insert(key.index() / PAGE);
                }
            }
        }
        prop_assert_eq!(queue.allocated_pages(), pages.len());
        check(&queue, &model, probes)?;
    }
    let rebuilt: PagedPriorityQueue<K, i32, PAGE, D> =
        model.iter().map(|(&k, &p)| (k, p)).collect();
    check(&rebuilt, &model, probes)
}

fn id(index: u32, generation: u32) -> Id {
    Id::new(index, NonZeroU32::new(generation).unwrap())
}

fn id_keys() -> BoxedStrategy<Id> {
    (0_u32..32, 1_u32..4).prop_map(|(i, g)| id(i, g)).boxed()
}

fn id_probes() -> Vec<Id> {
    (0..36)
        .flat_map(|i| (1..5).map(move |g| id(i, g)))
        .collect()
}

macro_rules! sequence_test {
    ($name:ident, $key:ty, $page:literal, $directory:ty, $keys:expr, $probes:expr) => {
        proptest! {
            #![proptest_config(config())]
            #[test]
            fn $name(ops in prop::collection::vec(operations($keys), 0..max_len())) {
                run::<$key, $page, $directory>(&ops, &$probes)?;
            }
        }
    };
}

sequence_test!(
    u32_flat_1,
    u32,
    1,
    Flat,
    (0_u32..32).boxed(),
    (0..36).collect::<Vec<_>>()
);
sequence_test!(
    u32_flat_4,
    u32,
    4,
    Flat,
    (0_u32..32).boxed(),
    (0..36).collect::<Vec<_>>()
);
sequence_test!(
    u32_radix_1,
    u32,
    1,
    Radix,
    (0_u32..32).boxed(),
    (0..36).collect::<Vec<_>>()
);
sequence_test!(
    u32_radix_4,
    u32,
    4,
    Radix,
    (0_u32..32).boxed(),
    (0..36).collect::<Vec<_>>()
);
sequence_test!(id_flat_1, Id, 1, Flat, id_keys(), id_probes());
sequence_test!(id_flat_4, Id, 4, Flat, id_keys(), id_probes());
sequence_test!(id_radix_1, Id, 1, Radix, id_keys(), id_probes());
sequence_test!(id_radix_4, Id, 4, Radix, id_keys(), id_probes());

#[test]
fn generations_evict_in_place_or_refuse_and_strict_updates_insert_absent_generations() {
    let mut queue: PagedPriorityQueue<Id, i32, 4> = PagedPriorityQueue::new();
    let old = id(3, 1);
    let new = id(3, 2);
    queue.push(old, 10);
    assert_eq!(queue.try_push(new, 100), Err((new, 100)));
    assert_eq!(queue.priority(&old), Some(&10));
    assert_eq!(queue.priority(&new), None);
    assert_eq!(queue.change_priority(&new, 30), None);
    assert!(!queue.change_priority_with(&new, |_| panic!("absent generation")));
    assert_eq!(queue.remove(&new), None);
    assert_eq!(queue.push(new, -1), None);
    assert_eq!(queue.position(&new), Some(0));
    assert!(!queue.contains_key(&old));
    assert_eq!(queue.try_push(new, 2), Ok(Some(-1)));
    assert!(queue.push_increase(old, -100));
    assert_eq!(queue.priority(&old), Some(&-100));
    assert!(!queue.contains_key(&new));
    assert!(queue.push_decrease(new, 100));
    assert_eq!(queue.pop(), Some((new, 100)));
    assert!(queue.is_empty());
}

#[derive(Clone, Debug)]
struct PanicPriority {
    value: i32,
    remaining: Rc<Cell<Option<usize>>>,
}

impl PartialEq for PanicPriority {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}
impl Eq for PanicPriority {}
impl PartialOrd for PanicPriority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for PanicPriority {
    fn cmp(&self, other: &Self) -> Ordering {
        if let Some(remaining) = self.remaining.get() {
            assert!(remaining > 0, "priority comparison panic");
            self.remaining.set(Some(remaining - 1));
        }
        self.value.cmp(&other.value)
    }
}

fn membership<const PAGE: usize, D: Directory>(
    queue: &PagedPriorityQueue<u32, PanicPriority, PAGE, D>,
    expected: &BTreeMap<u32, i32>,
) {
    assert_eq!(queue.len(), expected.len());
    assert_eq!(queue.key_slice().len(), queue.priority_slice().len());
    assert_eq!(
        queue
            .iter()
            .map(|(&k, p)| (k, p.value))
            .collect::<BTreeMap<_, _>>(),
        *expected
    );
    for key in 0..84 {
        assert_eq!(
            queue.priority(&key).map(|p| p.value),
            expected.get(&key).copied()
        );
        assert_eq!(queue.contains_key(&key), expected.contains_key(&key));
    }
    for (pos, key) in queue.key_slice().iter().enumerate() {
        assert_eq!(queue.position(key), Some(pos));
    }
}

fn panicking_comparisons<const PAGE: usize, D: Directory>() {
    // Budgets cover failure before a swap and after multiple completed swaps.
    for operation in 0..8 {
        let mut caught = 0;
        for budget in [0, 1, 2, 4, 8, 16] {
            let remaining = Rc::new(Cell::new(None));
            let priority = |value| PanicPriority {
                value,
                remaining: Rc::clone(&remaining),
            };
            let mut queue: PagedPriorityQueue<u32, PanicPriority, PAGE, D> =
                (0..80).map(|k| (k, priority(k as i32))).collect();
            let mut expected: BTreeMap<_, _> = (0..80).map(|k| (k, k as i32)).collect();
            let target = match operation {
                0 => {
                    expected.insert(80, 1000);
                    80
                }
                1 | 3 | 5 => {
                    expected.insert(0, 1000);
                    0
                }
                2 | 4 => {
                    expected.insert(79, -1000);
                    79
                }
                6 => {
                    expected.remove(&79);
                    79
                }
                7 => {
                    let key = queue.key_slice()[1];
                    expected.remove(&key);
                    key
                }
                _ => unreachable!(),
            };
            remaining.set(Some(budget));
            let result = catch_unwind(AssertUnwindSafe(|| match operation {
                0 | 1 => {
                    queue.push(target, priority(1000));
                }
                2 => {
                    queue.push(target, priority(-1000));
                }
                3 => {
                    queue.change_priority(&target, priority(1000));
                }
                4 => {
                    queue.change_priority(&target, priority(-1000));
                }
                5 => {
                    queue.change_priority_with(&target, |p| p.value = 1000);
                }
                6 => {
                    queue.pop();
                }
                7 => {
                    queue.remove(&target);
                }
                _ => unreachable!(),
            }));
            caught += usize::from(result.is_err());
            remaining.set(None);
            membership(&queue, &expected);
        }
        assert!(
            caught > 0,
            "operation {operation} never compared priorities"
        );
    }
}

#[test]
fn comparison_panics_preserve_remaining_membership_and_pairing() {
    panicking_comparisons::<1, Flat>();
    panicking_comparisons::<4, Flat>();
    panicking_comparisons::<1, Radix>();
    panicking_comparisons::<4, Radix>();
}

#[test]
fn closure_panic_preserves_its_mutation_and_all_other_entries() {
    let mut queue: PagedPriorityQueue<u32, i32, 4> = (0..40).map(|k| (k, k as i32)).collect();
    let result = catch_unwind(AssertUnwindSafe(|| {
        queue.change_priority_with(&3, |p| {
            *p = 1000;
            panic!("closure panic");
        });
    }));
    assert!(result.is_err());
    assert_eq!(queue.len(), 40);
    for key in 0..40 {
        assert_eq!(
            queue.priority(&key),
            Some(&if key == 3 { 1000 } else { key as i32 })
        );
        let pos = queue.position(&key).unwrap();
        assert_eq!(queue.key_slice()[pos], key);
    }
}

#[cfg(feature = "serde")]
#[test]
fn serde_sequence_round_trip_and_later_duplicates_win() {
    let queue: PagedPriorityQueue<u32, i32, 4, Radix> =
        [(1, 5), (2, 20), (3, 20), (4, -1)].into_iter().collect();
    let json = serde_json::to_string(&queue).unwrap();
    let serialized: Vec<(u32, i32)> = serde_json::from_str(&json).unwrap();
    assert_eq!(
        serialized,
        queue.iter().map(|(&k, &p)| (k, p)).collect::<Vec<_>>()
    );
    let loaded: PagedPriorityQueue<u32, i32, 4, Radix> = serde_json::from_str(&json).unwrap();
    let model = queue.into_iter().collect();
    check(&loaded, &model, &(0..6).collect::<Vec<_>>()).unwrap();
    let loaded: PagedPriorityQueue<u32, i32, 1> =
        serde_json::from_str("[[1,10],[2,40],[1,-1]]").unwrap();
    assert_eq!(loaded.into_sorted_vec(), [(1, -1), (2, 40)]);
    let empty: PagedPriorityQueue<u32, i32> = serde_json::from_str("[]").unwrap();
    assert!(empty.is_empty());
}
