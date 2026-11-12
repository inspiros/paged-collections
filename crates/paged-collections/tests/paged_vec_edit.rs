//! Vec oracles for editing, bounded iteration, stable addresses and ownership.

use std::cell::RefCell;
use std::iter::FusedIterator;
use std::ops::Bound;
use std::rc::Rc;

use paged_collections::paged_vec::PagedVec;
use proptest::prelude::*;

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

fn addresses<T, const PAGE: usize>(v: &PagedVec<T, PAGE>) -> Vec<*const T> {
    v.iter().map(std::ptr::from_ref).collect()
}

// Check the relationship independently of is_empty's implementation.
#[allow(clippy::len_zero)]
fn layout<T, const PAGE: usize>(v: &PagedVec<T, PAGE>) {
    assert_eq!(v.is_empty(), v.len() == 0);
    assert!(v.capacity() >= v.len());
    assert_eq!(v.capacity(), v.allocated_pages() * PAGE);
    assert_eq!(v.pages().len(), v.len().div_ceil(PAGE));
    for (i, page) in v.pages().enumerate() {
        assert_eq!(page.len(), PAGE.min(v.len() - i * PAGE));
    }
    assert_eq!(v.pages().flatten().count(), v.len());
}

fn check<const PAGE: usize>(v: &PagedVec<i32, PAGE>, model: &[i32]) {
    layout(v);
    assert_eq!(v.len(), model.len());
    assert!(v.iter().eq(model.iter()));
    assert!(v.iter().rev().eq(model.iter().rev()));
}

fn mixed<I, F>(mut iter: I, model: &[i32], mut read: F, salt: usize)
where
    I: DoubleEndedIterator + ExactSizeIterator + FusedIterator,
    F: FnMut(I::Item) -> i32,
{
    let (mut front, mut back) = (0, model.len());
    while front < back {
        assert_eq!(iter.len(), back - front);
        assert_eq!(iter.size_hint(), (back - front, Some(back - front)));
        let (item, expected) = if (front + model.len() - back + salt) % 5 < 2 {
            let item = iter.next();
            let expected = model[front];
            front += 1;
            (item, expected)
        } else {
            back -= 1;
            (iter.next_back(), model[back])
        };
        assert_eq!(read(item.expect("iterator ended early")), expected);
    }
    for _ in 0..3 {
        assert_eq!(iter.len(), 0);
        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert!(iter.next().is_none());
        assert!(iter.next_back().is_none());
    }
}

// Exercise every bound form while representing exactly start..end.
fn bounds(start: usize, end: usize, len: usize, salt: usize) -> (Bound<usize>, Bound<usize>) {
    let start = if start == 0 && salt % 2 == 0 {
        Bound::Unbounded
    } else if start > 0 && salt % 3 == 0 {
        Bound::Excluded(start - 1)
    } else {
        Bound::Included(start)
    };
    let end = if end == len && salt % 2 == 0 {
        Bound::Unbounded
    } else if end > 0 && salt % 3 == 0 {
        Bound::Included(end - 1)
    } else {
        Bound::Excluded(end)
    };
    (start, end)
}

#[derive(Clone, Debug)]
enum Op {
    Dedup,
    DedupBy,
    DedupByKey,
    Split(usize),
    Append(Vec<i32>),
    Drain(usize, usize),
    Range(usize, usize),
    RangeMut(usize, usize),
    Extend(Vec<i32>),
    Reserve(usize),
}

fn op() -> impl Strategy<Value = Op> {
    let index = 0..max_len() + 3;
    prop_oneof![
        Just(Op::Dedup),
        Just(Op::DedupBy),
        Just(Op::DedupByKey),
        index.clone().prop_map(Op::Split),
        prop::collection::vec(-8i32..8, 0..8).prop_map(Op::Append),
        (index.clone(), index.clone()).prop_map(|(a, b)| Op::Drain(a, b)),
        (index.clone(), index.clone()).prop_map(|(a, b)| Op::Range(a, b)),
        (index.clone(), index).prop_map(|(a, b)| Op::RangeMut(a, b)),
        prop::collection::vec(-8i32..8, 0..8).prop_map(Op::Extend),
        (0..12usize).prop_map(Op::Reserve),
    ]
}

// Mutation makes argument order and the identity of the earlier kept element
// observable, even when the predicate is not an equivalence relation.
fn same_bucket(later: &mut i32, earlier: &mut i32) -> bool {
    let same = later.rem_euclid(4) == earlier.rem_euclid(4);
    *later = later.wrapping_add(1);
    *earlier = earlier.wrapping_sub(1);
    same
}

fn key(value: &mut i32) -> i32 {
    *value = value.wrapping_add(1);
    value.rem_euclid(5)
}

fn sequence<const PAGE: usize>(initial: &[i32], ops: &[Op]) {
    let mut v: PagedVec<i32, PAGE> = initial.iter().copied().collect();
    let mut model = initial.to_vec();
    for (step, op) in ops.iter().enumerate() {
        match op {
            Op::Dedup => {
                v.dedup();
                model.dedup();
            }
            Op::DedupBy => {
                let (mut actual_calls, mut model_calls) = (Vec::new(), Vec::new());
                v.dedup_by(|a, b| {
                    actual_calls.push((*a, *b));
                    same_bucket(a, b)
                });
                model.dedup_by(|a, b| {
                    model_calls.push((*a, *b));
                    same_bucket(a, b)
                });
                assert_eq!(actual_calls, model_calls);
            }
            Op::DedupByKey => {
                let (mut actual_calls, mut model_calls) = (Vec::new(), Vec::new());
                v.dedup_by_key(|x| {
                    actual_calls.push(*x);
                    key(x)
                });
                model.dedup_by_key(|x| {
                    model_calls.push(*x);
                    key(x)
                });
                assert_eq!(actual_calls, model_calls);
            }
            Op::Split(raw) => {
                let at = raw % (model.len() + 1);
                let mut suffix = v.split_off(at);
                let mut expected = model.split_off(at);
                check(&v, &model);
                check(&suffix, &expected);
                mixed(suffix.iter(), &expected, |x| *x, step);
                assert!(suffix.iter().rev().eq(expected.iter().rev()));
                if step % 2 == 0 {
                    v.append(&mut suffix);
                    model.append(&mut expected);
                    check(&suffix, &expected);
                } else {
                    mixed(suffix.into_iter(), &expected, |x| x, step + 1);
                }
            }
            Op::Append(values) => {
                let mut other: PagedVec<i32, PAGE> = values.iter().copied().collect();
                other.reserve(PAGE + 1);
                let capacity = other.capacity();
                let mut expected = values.clone();
                v.append(&mut other);
                model.append(&mut expected);
                check(&other, &expected);
                assert_eq!(other.capacity(), capacity);
                other.push(19);
                check(&other, &[19]);
            }
            Op::Drain(a, b) | Op::Range(a, b) | Op::RangeMut(a, b) => {
                let a = a % (model.len() + 1);
                let b = b % (model.len() + 1);
                let (start, end) = (a.min(b), a.max(b));
                let range = bounds(start, end, model.len(), step);
                match op {
                    Op::Drain(_, _) => {
                        let mut reverse = v.clone();
                        let expected: Vec<_> = model.drain(range).collect();
                        let iter = v.drain(range);
                        assert_eq!(format!("{iter:?}"), format!("{expected:?}"));
                        mixed(iter, &expected, |x| x, step);
                        assert!(
                            reverse
                                .drain(range)
                                .rev()
                                .eq(expected.iter().copied().rev())
                        );
                        check(&reverse, &model);
                    }
                    Op::Range(_, _) => {
                        let expected = &model[start..end];
                        mixed(v.range(range), expected, |x| *x, step);
                        assert!(v.range(range).rev().eq(expected.iter().rev()));
                        assert_eq!(format!("{:?}", v.range(range)), format!("{expected:?}"));
                    }
                    Op::RangeMut(_, _) => {
                        let expected = &mut model[start..end];
                        mixed(
                            v.range_mut(range),
                            expected,
                            |x| {
                                let old = *x;
                                *x = x.wrapping_add(7);
                                old
                            },
                            step,
                        );
                        for x in expected.iter_mut() {
                            *x = x.wrapping_add(7);
                        }
                        assert!(
                            v.range_mut(range)
                                .rev()
                                .map(|x| *x)
                                .eq(expected.iter().copied().rev())
                        );
                    }
                    _ => unreachable!(),
                }
            }
            Op::Extend(values) => {
                v.extend_from_slice(values);
                model.extend_from_slice(values);
            }
            Op::Reserve(n) => v.reserve(*n),
        }
        check(&v, &model);
    }
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn editing_sequences_match_vec(
        initial in prop::collection::vec(-8i32..8, 0..max_len()),
        ops in prop::collection::vec(op(), 0..max_len()),
    ) {
        sequence::<1>(&initial, &ops);
        sequence::<2>(&initial, &ops);
        sequence::<3>(&initial, &ops);
        sequence::<4>(&initial, &ops);
        sequence::<7>(&initial, &ops);
    }
}

fn boundaries<const PAGE: usize>() {
    let n = 3 * PAGE + 2;
    let model: Vec<_> = (0..n).map(|i| i as i32).collect();
    for at in 0..=n {
        let mut v: PagedVec<i32, PAGE> = model.iter().copied().collect();
        v.reserve(2 * PAGE);
        let before = addresses(&v);
        let suffix = v.split_off(at);
        check(&v, &model[..at]);
        check(&suffix, &model[at..]);
        assert_eq!(addresses(&v), before[..at]);
        if at % PAGE == 0 {
            assert_eq!(addresses(&suffix), before[at..]);
        }
        // Transferred and freshly allocated pages must both remain stable
        // when a partial final page is filled and the directory grows.
        let mut suffix = suffix;
        let returned_addresses = addresses(&suffix);
        suffix.extend_from_slice(&vec![99; 2 * PAGE + 1]);
        assert_eq!(
            &addresses(&suffix)[..returned_addresses.len()],
            returned_addresses
        );
        layout(&suffix);
        for end in at..=n {
            let mut v: PagedVec<i32, PAGE> = model.iter().copied().collect();
            v.reserve(PAGE + 1);
            let before = addresses(&v);
            mixed(v.range(at..end), &model[at..end], |x| *x, at + end);
            mixed(v.range_mut(at..end), &model[at..end], |x| *x, at + end + 1);
            let mut iter = v.drain(at..end);
            let mut expected = model[at..end].iter();
            assert_eq!(iter.next(), expected.next().copied());
            assert_eq!(iter.next_back(), expected.next_back().copied());
            assert_eq!(format!("{iter:?}"), format!("{:?}", expected.as_slice()));
            mixed(iter, expected.as_slice(), |x| x, at);
            let remaining: Vec<_> = model[..at].iter().chain(&model[end..]).copied().collect();
            check(&v, &remaining);
            assert_eq!(&addresses(&v)[..at], &before[..at]);
            if at == end {
                assert_eq!(addresses(&v), before);
            }
            if at % PAGE == 0 && end % PAGE == 0 {
                assert_eq!(&addresses(&v)[at..], &before[end..]);
            }
            v.push(100);
            layout(&v);
        }
    }
    for len in 0..=n {
        let mut v: PagedVec<i32, PAGE> = model[..len].iter().copied().collect();
        let before = addresses(&v);
        let mut other: PagedVec<i32, PAGE> = model.iter().copied().collect();
        other.reserve(PAGE + 1);
        let capacity = other.capacity();
        v.append(&mut other);
        assert_eq!(&addresses(&v)[..len], before);
        assert!(other.is_empty());
        assert_eq!(other.capacity(), capacity);
        let expected: Vec<_> = model[..len].iter().chain(&model).copied().collect();
        check(&v, &expected);
        layout(&other);
    }
    let mut v: PagedVec<i32, PAGE> = [10, 20, 30, 30, 40, 40, 50].into();
    let before = addresses(&v);
    v.dedup();
    check(&v, &[10, 20, 30, 40, 50]);
    assert_eq!(&addresses(&v)[..3], &before[..3]);
    let before = addresses(&v);
    v.dedup();
    assert_eq!(addresses(&v), before);
}

#[test]
fn all_boundaries_and_stable_prefixes() {
    boundaries::<1>();
    boundaries::<2>();
    boundaries::<3>();
    boundaries::<4>();
    boundaries::<7>();
}

#[test]
fn slice_extension_clones_non_copy_values_and_keeps_addresses() {
    let mut v: PagedVec<String, 2> = [String::from("first")].into();
    let before = addresses(&v);
    let values = [String::from("second"), String::from("third")];
    v.extend_from_slice(&values);
    assert!(std::ptr::eq(&v[0], before[0]));
    assert!(
        v.iter()
            .map(String::as_str)
            .eq(["first", "second", "third"])
    );
    assert_eq!(values, ["second", "third"]);
}

// Deliberately leaking owned allocations is incompatible with Miri's leak check.
#[cfg(not(miri))]
#[test]
fn leaked_drain_leaves_eager_result_ready_for_reuse() {
    let mut v: PagedVec<i32, 3> = (0..10).collect();
    std::mem::forget(v.drain(2..7));
    check(&v, &[0, 1, 7, 8, 9]);
    v.push(10);
    v.insert(2, 11);
    check(&v, &[0, 1, 11, 7, 8, 9, 10]);
}

#[derive(Debug)]
struct Item {
    id: usize,
    value: i32,
    drops: Rc<RefCell<Vec<usize>>>,
}

impl Drop for Item {
    fn drop(&mut self) {
        let mut drops = self.drops.borrow_mut();
        drops[self.id] += 1;
        assert_eq!(drops[self.id], 1, "element dropped twice");
    }
}

fn items<const PAGE: usize>(n: usize, drops: &Rc<RefCell<Vec<usize>>>) -> PagedVec<Item, PAGE> {
    (0..n)
        .map(|id| Item {
            id,
            value: (id / 2) as i32,
            drops: Rc::clone(drops),
        })
        .collect()
}

fn ownership<const PAGE: usize>() {
    let n = 4 * PAGE + 3;
    for at in 0..=n {
        let drops = Rc::new(RefCell::new(vec![0; n]));
        let mut v = items::<PAGE>(n, &drops);
        let mut suffix = v.split_off(at);
        assert!(drops.borrow().iter().all(|&count| count == 0));
        assert!(v.iter().map(|x| x.id).eq(0..at));
        assert!(suffix.iter().map(|x| x.id).eq(at..n));
        v.append(&mut suffix);
        assert!(v.iter().map(|x| x.id).eq(0..n));
        assert!(suffix.is_empty());
        drop(suffix);
        assert!(drops.borrow().iter().all(|&count| count == 0));
        drop(v);
        assert!(drops.borrow().iter().all(|&count| count == 1));

        let drops = Rc::new(RefCell::new(vec![0; n]));
        let mut v = items::<PAGE>(n, &drops);
        drop(v.split_off(at));
        for (id, &count) in drops.borrow().iter().enumerate() {
            assert_eq!(count, usize::from(id >= at));
        }
        drop(v);
        assert!(drops.borrow().iter().all(|&count| count == 1));
    }
    for start in 0..=n {
        for end in start..=n {
            let drops = Rc::new(RefCell::new(vec![0; n]));
            let mut v = items::<PAGE>(n, &drops);
            let mut iter = v.drain(start..end);
            for step in 0..(end - start) / 2 {
                if step % 2 == 0 {
                    drop(iter.next());
                } else {
                    drop(iter.next_back());
                }
            }
            drop(iter);
            assert!(v.iter().map(|x| x.id).eq((0..start).chain(end..n)));
            layout(&v);
            for (id, &count) in drops.borrow().iter().enumerate() {
                assert_eq!(count, usize::from((start..end).contains(&id)));
            }
            drop(v);
            assert!(drops.borrow().iter().all(|&count| count == 1));
        }
    }
    let drops = Rc::new(RefCell::new(vec![0; n]));
    let mut v = items::<PAGE>(n, &drops);
    v.dedup_by(|a, b| a.value == b.value);
    assert!(v.iter().map(|x| x.id).eq((0..n).step_by(2)));
    for (id, &count) in drops.borrow().iter().enumerate() {
        assert_eq!(count, id % 2);
    }
    drop(v);
    assert!(drops.borrow().iter().all(|&count| count == 1));
}

#[test]
fn split_append_dedup_and_partial_drain_drop_every_element_once() {
    ownership::<1>();
    ownership::<2>();
    ownership::<3>();
    ownership::<4>();
    ownership::<7>();
}

#[test]
fn zero_sized_edits_and_ranges() {
    let mut v: PagedVec<(), 3> = [(); 10].into();
    assert_eq!(v.range(2..=6).len(), 5);
    assert_eq!(v.range_mut(3..8).rev().count(), 5);
    assert_eq!(v.drain(2..5).rev().count(), 3);
    let mut suffix = v.split_off(4);
    assert_eq!((v.len(), suffix.len()), (4, 3));
    v.append(&mut suffix);
    v.extend_from_slice(&[(); 2]);
    assert_eq!(v.len(), 9);
    v.dedup();
    assert_eq!(v.len(), 1);
    layout(&v);
    assert!(suffix.is_empty());
}

#[cfg(feature = "std")]
mod panics {
    use super::*;
    use std::cell::Cell;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    fn invalid_ranges<const PAGE: usize>() {
        let mut v: PagedVec<i32, PAGE> = (0..8).collect();
        let before = addresses(&v);
        for range in [
            (Bound::Included(5), Bound::Excluded(3)),
            (Bound::Included(0), Bound::Excluded(9)),
            (Bound::Included(9), Bound::Unbounded),
            (Bound::Included(usize::MAX), Bound::Unbounded),
            (Bound::Unbounded, Bound::Included(usize::MAX)),
            (Bound::Excluded(usize::MAX), Bound::Unbounded),
            (Bound::Included(0), Bound::Excluded(usize::MAX)),
        ] {
            assert!(
                catch_unwind(AssertUnwindSafe(|| {
                    v.range(range);
                }))
                .is_err()
            );
            assert!(
                catch_unwind(AssertUnwindSafe(|| {
                    v.range_mut(range);
                }))
                .is_err()
            );
            assert!(
                catch_unwind(AssertUnwindSafe(|| {
                    v.drain(range);
                }))
                .is_err()
            );
            check(&v, &[0, 1, 2, 3, 4, 5, 6, 7]);
            assert_eq!(addresses(&v), before);
        }
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                v.split_off(9);
            }))
            .is_err()
        );
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                v.range(..=usize::MAX);
            }))
            .is_err()
        );
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                v.range_mut(..=usize::MAX);
            }))
            .is_err()
        );
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                v.drain(..=usize::MAX);
            }))
            .is_err()
        );
        check(&v, &[0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(addresses(&v), before);
    }

    #[test]
    fn invalid_and_overflowing_bounds_panic_before_mutation() {
        invalid_ranges::<1>();
        invalid_ranges::<2>();
        invalid_ranges::<3>();
        invalid_ranges::<4>();
        invalid_ranges::<7>();
    }

    fn dedup_panics<const PAGE: usize>() {
        let n = 3 * PAGE + 4;
        for panic_at in 1..n {
            let drops = Rc::new(RefCell::new(vec![0; n]));
            let mut v = items::<PAGE>(n, &drops);
            let before = addresses(&v);
            let mut calls = 0;
            let result = catch_unwind(AssertUnwindSafe(|| {
                v.dedup_by(|later, earlier| {
                    calls += 1;
                    // Mutations made before unwinding must survive, too.
                    later.value += 100;
                    earlier.value += 1000;
                    assert_ne!(later.id, panic_at, "predicate bomb");
                    later.id / 2 == earlier.id / 2
                });
            }));
            assert!(result.is_err());
            assert_eq!(calls, panic_at);
            let expected_ids: Vec<_> = (0..n).filter(|&id| id >= panic_at || id % 2 == 0).collect();
            assert!(v.iter().map(|x| x.id).eq(expected_ids.iter().copied()));
            let mut expected_values: Vec<_> = (0..n).map(|id| (id / 2) as i32).collect();
            let mut kept = 0;
            for later in 1..=panic_at {
                expected_values[later] += 100;
                expected_values[kept] += 1000;
                if later % 2 == 0 {
                    kept = later;
                }
            }
            for item in &v {
                assert_eq!(item.value, expected_values[item.id]);
            }
            assert!(std::ptr::eq(&v[0], before[0]));
            layout(&v);
            for (id, &count) in drops.borrow().iter().enumerate() {
                assert_eq!(count, usize::from(id < panic_at && id % 2 == 1));
            }
            // Exercise the compacted layout after unwinding.
            v.reserve(PAGE + 1);
            let mut tail = v.split_off(v.len() / 2);
            v.append(&mut tail);
            drop(v.drain(..));
            assert!(v.is_empty());
            assert!(drops.borrow().iter().all(|&count| count == 1));
        }
    }

    #[test]
    fn panicking_dedup_keeps_mutations_and_the_exact_unvisited_tail() {
        dedup_panics::<1>();
        dedup_panics::<2>();
        dedup_panics::<3>();
        dedup_panics::<4>();
        dedup_panics::<7>();
    }

    fn key_panics<const PAGE: usize>() {
        let drops = Rc::new(RefCell::new(vec![0; 7]));
        let mut v = items::<PAGE>(7, &drops);
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                v.dedup_by_key(|item| {
                    item.value += 10;
                    assert_ne!(item.id, 3, "key bomb");
                    item.id / 2
                });
            }))
            .is_err()
        );
        assert!(v.iter().map(|item| (item.id, item.value)).eq([
            (0, 20),
            (2, 11),
            (3, 11),
            (4, 2),
            (5, 2),
            (6, 3),
        ]));
        layout(&v);
        assert_eq!(*drops.borrow(), [0, 1, 0, 0, 0, 0, 0]);
        drop(v);
        assert!(drops.borrow().iter().all(|&count| count == 1));
    }

    #[test]
    fn panicking_key_keeps_its_mutations_and_unvisited_elements() {
        key_panics::<1>();
        key_panics::<2>();
        key_panics::<3>();
        key_panics::<4>();
        key_panics::<7>();
    }

    struct CloneBomb {
        value: i32,
        remaining: Rc<Cell<usize>>,
    }

    impl Clone for CloneBomb {
        fn clone(&self) -> Self {
            let remaining = self.remaining.get();
            assert!(remaining > 0, "clone bomb");
            self.remaining.set(remaining - 1);
            Self {
                value: self.value,
                remaining: Rc::clone(&self.remaining),
            }
        }
    }

    fn clone_panics<const PAGE: usize>() {
        for limit in 0..7 {
            let remaining = Rc::new(Cell::new(limit));
            let source: Vec<_> = (0..7)
                .map(|value| CloneBomb {
                    value,
                    remaining: Rc::clone(&remaining),
                })
                .collect();
            let mut v: PagedVec<CloneBomb, PAGE> = [CloneBomb {
                value: 99,
                remaining: Rc::clone(&remaining),
            }]
            .into();
            let before = addresses(&v);
            assert!(
                catch_unwind(AssertUnwindSafe(|| {
                    v.extend_from_slice(&source);
                }))
                .is_err()
            );
            assert_eq!(v.len(), 1 + limit);
            assert!(
                v.iter()
                    .map(|item| item.value)
                    .eq(std::iter::once(99).chain((0..limit).map(|i| i as i32)))
            );
            assert!(source.iter().map(|item| item.value).eq(0..7));
            assert!(std::ptr::eq(&v[0], before[0]));
            layout(&v);
        }
    }

    #[test]
    fn panicking_slice_clone_keeps_completed_clones() {
        clone_panics::<1>();
        clone_panics::<2>();
        clone_panics::<3>();
        clone_panics::<4>();
        clone_panics::<7>();
    }
}
