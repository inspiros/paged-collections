//! Model, contention, unwind safety and wire compatibility for the sharded queue.

#![cfg(all(feature = "concurrent", not(loom)))]
#![cfg(all(
    any(target_pointer_width = "32", target_pointer_width = "64"),
    target_has_atomic = "ptr",
    target_has_atomic = "64"
))]

use std::cell::Cell;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU32;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;
use std::sync::Barrier;
use std::thread;

use paged_collections::concurrent::ConcurrentPriorityQueue;
use paged_collections::id_alloc::Id;
use paged_collections::page::Tree;
use proptest::prelude::*;
use proptest::test_runner::TestCaseResult;

#[derive(Clone, Debug)]
enum Op {
    Push(usize, i32),
    TryPush(usize, i32),
    Replace(usize, i32),
    Increase(usize, i32),
    Decrease(usize, i32),
    Change(usize, i32),
    ChangeWith(usize, i32),
    Remove(usize),
    Pop,
    PopExact,
}

const KEYS: usize = 24;

fn operations() -> impl Strategy<Value = Vec<Op>> {
    let keys = 0..KEYS;
    let values = prop_oneof![4 => -8_i32..9, 1 => any::<i32>()];
    let entries = (keys.clone(), values);
    prop::collection::vec(
        prop_oneof![
            8 => entries.clone().prop_map(|(k, p)| Op::Push(k, p)),
            4 => entries.clone().prop_map(|(k, p)| Op::TryPush(k, p)),
            4 => entries.clone().prop_map(|(k, p)| Op::Replace(k, p)),
            4 => entries.clone().prop_map(|(k, p)| Op::Increase(k, p)),
            4 => entries.clone().prop_map(|(k, p)| Op::Decrease(k, p)),
            4 => entries.clone().prop_map(|(k, p)| Op::Change(k, p)),
            4 => entries.prop_map(|(k, p)| Op::ChangeWith(k, p)),
            4 => keys.prop_map(Op::Remove),
            4 => Just(Op::Pop),
            4 => Just(Op::PopExact),
        ],
        0..if cfg!(miri) { 24 } else { 150 },
    )
}

fn check_model<const PAGE: usize>(shards: usize, ops: &[Op]) -> TestCaseResult {
    let queue = ConcurrentPriorityQueue::<usize, i32, PAGE>::with_shards(shards);
    let mut model = BTreeMap::new();
    let mut pages = BTreeSet::new();
    prop_assert_eq!(queue.shards(), shards);

    for op in ops {
        match *op {
            Op::Push(key, priority) => {
                prop_assert_eq!(queue.push(key, priority), model.insert(key, priority));
            }
            Op::TryPush(key, priority) => {
                prop_assert_eq!(
                    queue.try_push(key, priority),
                    Ok(model.insert(key, priority))
                );
            }
            Op::Replace(key, priority) => {
                let previous = model.insert(key, priority).map(|old| (key, old));
                prop_assert_eq!(queue.replace(key, priority), previous);
            }
            Op::Increase(key, priority) => {
                let changed = model.get(&key).is_none_or(|&old| priority > old);
                prop_assert_eq!(queue.push_increase(key, priority), changed);
                if changed {
                    model.insert(key, priority);
                }
            }
            Op::Decrease(key, priority) => {
                let changed = model.get(&key).is_none_or(|&old| priority < old);
                prop_assert_eq!(queue.push_decrease(key, priority), changed);
                if changed {
                    model.insert(key, priority);
                }
            }
            Op::Change(key, priority) => {
                let previous = model.get_mut(&key).map(|p| std::mem::replace(p, priority));
                prop_assert_eq!(queue.change_priority(&key, priority), previous);
            }
            Op::ChangeWith(key, delta) => {
                let mut calls = 0;
                let found = queue.change_priority_with(&key, |p| {
                    calls += 1;
                    *p = p.wrapping_add(delta);
                });
                let expected = model.get_mut(&key).map(|p| *p = p.wrapping_add(delta));
                prop_assert_eq!(found, expected.is_some());
                prop_assert_eq!(calls, usize::from(found));
            }
            Op::Remove(key) => {
                prop_assert_eq!(queue.remove(&key), model.remove_entry(&key));
            }
            Op::Pop | Op::PopExact => {
                let exact = matches!(op, Op::PopExact);
                let entry = if exact {
                    queue.pop_exact()
                } else {
                    queue.pop()
                };
                if let Some((key, priority)) = entry {
                    if exact || shards == 1 {
                        prop_assert_eq!(Some(&priority), model.values().max());
                    }
                    prop_assert_eq!(model.remove(&key), Some(priority));
                } else {
                    prop_assert!(model.is_empty());
                }
            }
        }

        pages.extend(model.keys().map(|key| key / PAGE));
        prop_assert_eq!(queue.allocated_pages(), pages.len());
        prop_assert_eq!(queue.len(), model.len());
        prop_assert_eq!(queue.is_empty(), model.is_empty());
        if let Some((key, priority)) = queue.peek_exact() {
            prop_assert_eq!(model.get(&key), Some(&priority));
            prop_assert_eq!(Some(&priority), model.values().max());
        } else {
            prop_assert!(model.is_empty());
        }
        for key in 0..KEYS {
            let expected = model.get(&key).copied();
            prop_assert_eq!(queue.contains_key(&key), expected.is_some());
            prop_assert_eq!(queue.priority(&key), expected);
            let mut calls = 0;
            let actual = queue.with_priority(&key, |p| {
                calls += 1;
                p.wrapping_neg()
            });
            prop_assert_eq!(actual, expected.map(i32::wrapping_neg));
            prop_assert_eq!(calls, usize::from(expected.is_some()));
        }
    }
    prop_assert_eq!(
        queue.snapshot().into_iter().collect::<BTreeMap<_, _>>(),
        model
    );
    Ok(())
}

fn config() -> ProptestConfig {
    let defaults = ProptestConfig::default();
    ProptestConfig {
        cases: if cfg!(miri) { 4 } else { 128 },
        failure_persistence: if cfg!(miri) {
            None
        } else {
            defaults.failure_persistence
        },
        ..defaults
    }
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn matches_btree_model(ops in operations()) {
        for shards in [1, 3] {
            check_model::<1>(shards, &ops)?;
            check_model::<4>(shards, &ops)?;
        }
    }
}

#[test]
fn generations_share_a_slot() {
    let id = |generation| Id::new(17, NonZeroU32::new(generation).unwrap());
    let [first, second, third] = [1, 2, 3].map(id);
    let queue = ConcurrentPriorityQueue::<Id, i32, 4>::with_shards(3);

    assert_eq!(queue.try_push(first, 10), Ok(None));
    assert_eq!(queue.try_push(second, 20), Err((second, 20)));
    assert_eq!(queue.try_push(first, 11), Ok(Some(10)));
    assert_eq!(queue.push(second, 20), None);
    assert_eq!(queue.remove(&first), None);
    assert_eq!(queue.change_priority(&first, 30), None);
    assert!(!queue.contains_key(&first));
    assert_eq!(queue.priority(&second), Some(20));
    assert_eq!(queue.replace(third, 4), Some((second, 20)));
    assert_eq!(queue.replace(third, 5), Some((third, 4)));
    assert!(queue.push_increase(first, 3));
    assert!(queue.push_decrease(second, 10));
    assert_eq!(queue.len(), 1);
    assert_eq!(queue.allocated_pages(), 1);
    assert_eq!(queue.pop(), Some((second, 10)));
    assert_eq!(queue.pop(), None);
}

#[test]
fn exclusive_operations_and_plain_conversions() {
    type Queue = ConcurrentPriorityQueue<usize, i32, 4, Tree>;
    let mut queue = Queue::with_shards_and_capacity(3, 7);
    assert_eq!(queue.allocated_pages(), 0);
    queue.extend([(1, 10), (2, 15), (usize::MAX, 20), (1, 30)]);
    let expected = [(2, 15), (usize::MAX, 20), (1, 30)];
    assert_eq!(queue.allocated_pages(), 2);

    let cloned = queue.clone();
    assert_eq!(cloned.shards(), 3);
    assert_eq!(cloned.into_sorted_vec(), expected);
    let plain = queue.snapshot();
    assert_eq!(plain.clone().into_sorted_vec(), expected);
    let merged = Queue::from(plain).into_queue();
    assert_eq!(merged.into_sorted_vec(), expected);

    let drained = queue.drain().collect::<BTreeMap<_, _>>();
    assert_eq!(drained, BTreeMap::from(expected));
    assert!(queue.is_empty());
    assert_eq!(queue.allocated_pages(), 2);

    queue.extend([(usize::MAX, 3)]);
    assert_eq!(format!("{queue:?}"), format!("{{{}: 3}}", usize::MAX));
    queue.clear();
    assert_eq!(queue.pop_exact(), None);
    assert!(queue.is_empty());
    assert_eq!(queue.allocated_pages(), 2);
    queue.shrink_to_fit();
    assert_eq!(queue.allocated_pages(), 0);
    assert!(Queue::default().is_empty());
}

#[test]
#[should_panic(expected = "requires at least one shard")]
fn rejects_zero_shards() {
    let _ = ConcurrentPriorityQueue::<usize, i32>::with_shards(0);
}

#[test]
fn concurrent_updates_and_pops_preserve_the_entry_multiset() {
    let workers = if cfg!(miri) { 2 } else { 4 };
    let per_worker = if cfg!(miri) { 12 } else { 512 };
    let total = workers * per_worker;
    let queue = ConcurrentPriorityQueue::<usize, usize, 4>::with_shards(3);
    let start = Barrier::new(workers);

    let (changed, removed, popped) = thread::scope(|scope| {
        let mut handles = Vec::new();
        for worker in 0..workers {
            let (queue, start) = (&queue, &start);
            handles.push(scope.spawn(move || {
                let mut changed = Vec::new();
                let mut removed = Vec::new();
                let mut popped = Vec::new();
                start.wait();
                for offset in 0..per_worker {
                    let key = worker * per_worker + offset;
                    assert_eq!(queue.push(key, key), None);
                    if let Some(old) = queue.change_priority(&key, key + total) {
                        assert_eq!(old, key);
                        changed.push(key);
                    }
                    if offset % 3 == 0 {
                        removed.extend(queue.remove(&key));
                    }
                    if offset % 2 == 0 {
                        popped.extend(queue.pop());
                    }
                }
                (changed, removed, popped)
            }));
        }

        let mut changed = BTreeSet::new();
        let mut removed = Vec::new();
        let mut popped = Vec::new();
        for handle in handles {
            let (updates, removals, pops) = handle.join().unwrap();
            changed.extend(updates);
            removed.extend(removals);
            popped.extend(pops);
        }
        (changed, removed, popped)
    });

    let mut remaining: BTreeMap<_, _> = (0..total)
        .map(|key| (key, key + usize::from(changed.contains(&key)) * total))
        .collect();
    for entry @ (key, _) in removed.into_iter().chain(popped) {
        assert_eq!(remaining.remove_entry(&key), Some(entry));
    }
    assert_eq!(queue.len(), remaining.len());

    // Producers have joined. Any entry still in the model now prevents None.
    while !remaining.is_empty() {
        let entry @ (key, _) = queue.pop().expect("pop missed a remaining entry");
        assert_eq!(remaining.remove_entry(&key), Some(entry));
    }
    assert_eq!(queue.pop(), None);
    assert!(queue.is_empty());
}

#[derive(Clone, Debug)]
struct Priority(i32, Rc<Cell<bool>>);

impl PartialEq for Priority {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for Priority {}

impl PartialOrd for Priority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Priority {
    fn cmp(&self, other: &Self) -> Ordering {
        assert!(!self.1.get(), "priority comparison panicked");
        self.0.cmp(&other.0)
    }
}

#[test]
fn comparison_and_callback_panics_preserve_remaining_membership() {
    let panic_on_cmp = Rc::new(Cell::new(false));
    let priority = |value| Priority(value, Rc::clone(&panic_on_cmp));
    let queue = ConcurrentPriorityQueue::<usize, Priority, 4>::with_shards(3);
    queue.push(0, priority(10));
    queue.push(1, priority(1));
    queue.push(4, priority(20));

    panic_on_cmp.set(true);
    assert!(catch_unwind(AssertUnwindSafe(|| queue.push(2, priority(30)))).is_err());
    assert_eq!(queue.len(), 4);
    assert_eq!(queue.priority(&2), Some(priority(30)));

    // The two sampled shards are both nonempty. Comparing their tops
    // panics before mutation and must release both write locks.
    assert!(catch_unwind(AssertUnwindSafe(|| queue.pop())).is_err());
    for (key, value) in [(0, 10), (1, 1), (2, 30), (4, 20)] {
        assert_eq!(queue.priority(&key), Some(priority(value)));
    }

    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            queue.change_priority_with(&4, |p| {
                p.0 = 25;
                panic!("change panicked");
            });
        }))
        .is_err()
    );

    // As in the plain queue, remove has committed before the heap repair.
    assert!(catch_unwind(AssertUnwindSafe(|| queue.remove(&0))).is_err());
    assert!(!queue.contains_key(&0));
    assert_eq!(queue.len(), 3);

    panic_on_cmp.set(false);
    assert_eq!(
        queue.into_sorted_vec(),
        [(1, priority(1)), (4, priority(25)), (2, priority(30))]
    );
}

#[cfg(feature = "serde")]
#[test]
fn serde_round_trips_with_the_plain_queue() {
    use paged_collections::priority_queue::PagedPriorityQueue;

    #[derive(Debug, Eq, Ord, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize)]
    struct Value(u32);

    type Plain = PagedPriorityQueue<u32, Value, 1>;
    type Concurrent = ConcurrentPriorityQueue<u32, Value, 1>;

    // A non-Clone value verifies serialization works directly from the shards.
    let plain: Plain = [(0, Value(10)), (1, Value(20)), (3, Value(30))]
        .into_iter()
        .collect();
    let concurrent: Concurrent =
        serde_json::from_str(&serde_json::to_string(&plain).unwrap()).unwrap();
    assert_eq!(concurrent.len(), 3);
    assert_eq!(concurrent.with_priority(&1, |value| value.0), Some(20));

    let concurrent = Concurrent::with_shards(3);
    for (key, priority) in [(0, Value(10)), (1, Value(20)), (3, Value(30))] {
        assert_eq!(concurrent.push(key, priority), None);
    }
    let decoded: Plain =
        serde_json::from_str(&serde_json::to_string(&concurrent).unwrap()).unwrap();
    assert_eq!(decoded, plain);
}
