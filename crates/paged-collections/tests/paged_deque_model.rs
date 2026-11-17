//! VecDeque oracles for all deque operations, iterators, searches and traits.

use std::collections::{VecDeque, hash_map::DefaultHasher};
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::iter::FusedIterator;
use std::ops::Bound;

use paged_collections::paged_deque::PagedDeque;
use proptest::prelude::*;

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

fn limit() -> usize {
    if cfg!(miri) { 12 } else { 64 }
}

fn mixed<I, F, V>(mut iter: I, expected: &[V], mut read: F, salt: usize)
where
    I: DoubleEndedIterator + ExactSizeIterator + FusedIterator,
    F: FnMut(I::Item) -> V,
    V: Debug + PartialEq,
{
    let (mut front, mut back) = (0, expected.len());
    while front < back {
        assert_eq!(iter.len(), back - front);
        assert_eq!(iter.size_hint(), (back - front, Some(back - front)));
        if (front + expected.len() - back + salt) % 5 < 2 {
            assert_eq!(read(iter.next().unwrap()), expected[front]);
            front += 1;
        } else {
            back -= 1;
            assert_eq!(read(iter.next_back().unwrap()), expected[back]);
        }
    }
    for _ in 0..3 {
        assert!(iter.next().is_none());
        assert!(iter.next_back().is_none());
        assert_eq!(iter.len(), 0);
        assert_eq!(iter.size_hint(), (0, Some(0)));
    }
}

fn check<const PAGE: usize>(
    deque: &mut PagedDeque<i32, PAGE>,
    model: &mut VecDeque<i32>,
    salt: usize,
) {
    assert_eq!(PagedDeque::<i32, PAGE>::PAGE, PAGE);
    assert_eq!(deque.len(), model.len());
    assert_eq!(deque.is_empty(), model.is_empty());
    assert!(deque.capacity() >= deque.len());
    let head = deque.allocated_pages() * PAGE - deque.capacity();
    assert!(head < PAGE);
    let pages: Vec<_> = deque.pages().map(<[i32]>::to_vec).collect();
    let used = if deque.is_empty() {
        0
    } else {
        (head + deque.len()).div_ceil(PAGE)
    };
    assert_eq!(pages.len(), used);
    assert_eq!(deque.pages().len(), used);
    for (i, page) in pages.iter().enumerate() {
        let start = if i == 0 { head } else { 0 };
        let end = PAGE.min(head + deque.len() - i * PAGE);
        assert_eq!(page.len(), end - start);
        assert!(!page.is_empty() && page.len() <= PAGE);
    }
    assert!(pages.iter().flatten().eq(model.iter()));
    assert!(deque.iter().eq(model.iter()));
    assert!(deque.iter().rev().eq(model.iter().rev()));
    assert_eq!(deque.front(), model.front());
    assert_eq!(deque.back(), model.back());
    assert_eq!(deque.get(deque.len()), None);
    assert_eq!(deque.get(usize::MAX), None);
    assert_eq!(deque.get_mut(usize::MAX), None);
    for (i, value) in model.iter().enumerate() {
        assert_eq!(deque.get(i), Some(value));
        assert_eq!(deque[i], *value);
    }
    let values: Vec<_> = model.iter().copied().collect();
    mixed(deque.iter(), &values, |x| *x, salt);
    let mut iter = deque.iter();
    iter.next();
    iter.next_back();
    let tail: Vec<_> = iter.clone().copied().collect();
    mixed(iter, &tail, |x| *x, salt + 1);
    mixed(deque.clone().into_iter(), &values, |x| x, salt + 2);
    mixed(deque.pages(), &pages, <[i32]>::to_vec, salt + 3);
    mixed(deque.pages().clone(), &pages, <[i32]>::to_vec, salt + 4);
    mixed(
        deque.iter_mut(),
        &values,
        |x| {
            let old = *x;
            *x = x.wrapping_add(1);
            old
        },
        salt + 5,
    );
    for x in model.iter_mut() {
        *x = x.wrapping_add(1);
    }
    let pages: Vec<_> = deque.pages().map(<[i32]>::to_vec).collect();
    mixed(
        deque.pages_mut(),
        &pages,
        |page| {
            let old = page.to_vec();
            for x in page {
                *x = x.wrapping_sub(1);
            }
            old
        },
        salt + 6,
    );
    for x in model.iter_mut() {
        *x = x.wrapping_sub(1);
    }
    assert!(deque.iter().eq(model.iter()));
}

// Each operation is small and independent of page size; out-of-range removal
// indices deliberately include len, len+1, and the far end of the input range.
fn run<const PAGE: usize>(initial: &[i32], ops: &[(u8, usize, i32)]) {
    let mut deque = PagedDeque::<i32, PAGE>::with_capacity(initial.len());
    deque.extend_from_slice(initial);
    let mut model: VecDeque<_> = initial.iter().copied().collect();
    check(&mut deque, &mut model, 0);
    for (step, &(code, raw, value)) in ops.iter().enumerate() {
        let index = raw % (model.len() + 2);
        let n = raw % (limit() + 1);
        match code {
            0 => {
                deque.push_back(value);
                model.push_back(value);
            }
            1 => {
                deque.push_front(value);
                model.push_front(value);
            }
            2 => assert_eq!(deque.pop_back(), model.pop_back()),
            3 => assert_eq!(deque.pop_front(), model.pop_front()),
            4 => {
                let i = raw % (model.len() + 1);
                deque.insert(i, value);
                model.insert(i, value);
            }
            5 => assert_eq!(deque.remove(index), model.remove(index)),
            6 => assert_eq!(deque.swap_remove_back(index), model.swap_remove_back(index)),
            7 => assert_eq!(
                deque.swap_remove_front(index),
                model.swap_remove_front(index)
            ),
            8 => {
                deque.truncate(n);
                model.truncate(n);
            }
            9 => {
                deque.truncate_front(n);
                while model.len() > n {
                    model.pop_front();
                }
            }
            10 => {
                deque.clear();
                model.clear();
            }
            11 => {
                deque.resize(n, value);
                model.resize(n, value);
            }
            12 => {
                let mut next = value;
                deque.resize_with(n, || {
                    next = next.wrapping_add(1);
                    next
                });
                let mut next = value;
                model.resize_with(n, || {
                    next = next.wrapping_add(1);
                    next
                });
            }
            13 => {
                deque.retain(|x| x & value != 0);
                model.retain(|x| x & value != 0);
            }
            14 => {
                let keep = |x: &mut i32| {
                    *x = x.wrapping_add(7);
                    *x & value != 0
                };
                deque.retain_mut(keep);
                model.retain_mut(keep);
            }
            15 => {
                deque.reserve(n);
                assert!(deque.capacity() >= deque.len() + n);
            }
            16 => {
                deque.try_reserve(n).unwrap();
                assert!(deque.capacity() >= deque.len() + n);
            }
            17 => {
                deque.shrink_to_fit();
                assert_eq!(deque.allocated_pages(), deque.pages().len());
            }
            18 => {
                let mut other: PagedDeque<_, PAGE> =
                    (0..n % 6).map(|i| value.wrapping_add(i as i32)).collect();
                let mut expected: VecDeque<_> = other.iter().copied().collect();
                other.reserve(PAGE + 1);
                let pages = other.allocated_pages();
                deque.append(&mut other);
                model.append(&mut expected);
                assert!(other.is_empty());
                assert_eq!(other.allocated_pages(), pages);
                other.push_front(value);
                assert_eq!(other.pop_back(), Some(value));
            }
            19..=21 => {
                let values = [value, value.wrapping_add(1), value.wrapping_sub(1)];
                match code {
                    19 => deque.extend(values),
                    20 => deque.extend(values.iter()),
                    _ => deque.extend_from_slice(&values),
                }
                model.extend(values);
            }
            22 => {
                deque = deque.clone();
            }
            23 => {
                deque = deque.into_iter().collect();
            }
            24 => {
                deque = PagedDeque::from(model.iter().copied().collect::<Vec<_>>());
            }
            25 => {
                deque = PagedDeque::from([value, value.wrapping_add(1)]);
                model = VecDeque::from([value, value.wrapping_add(1)]);
            }
            26 => {
                assert_eq!(deque.get_mut(index).map(|x| *x), model.get(index).copied());
                if let Some(x) = deque.get_mut(index) {
                    *x = value;
                    model[index] = value;
                    deque[index] = value.wrapping_add(1);
                    model[index] = value.wrapping_add(1);
                }
            }
            27 => {
                if let Some(x) = deque.front_mut() {
                    *x = value;
                    *model.front_mut().unwrap() = value;
                }
                if let Some(x) = deque.back_mut() {
                    *x = value.wrapping_add(1);
                    *model.back_mut().unwrap() = value.wrapping_add(1);
                }
            }
            28 if !model.is_empty() => {
                let i = raw % model.len();
                deque.swap(0, i);
                model.swap(0, i);
            }
            29 => assert_eq!(deque.contains(&value), model.contains(&value)),
            30..=32 => {
                let a = raw % (model.len() + 1);
                let b = (value.unsigned_abs() as usize) % (model.len() + 1);
                let (start, end) = (a.min(b), a.max(b));
                let expected: Vec<_> = model.range(start..end).copied().collect();
                mixed(deque.range(start..end), &expected, |x| *x, step);
                let bounds = (Bound::Excluded(start), Bound::Included(end));
                if start < end && end < model.len() {
                    assert!(deque.range(bounds).eq(model.range(bounds)));
                }
                for x in deque.range_mut(start..end) {
                    *x = x.wrapping_add(2);
                }
                for x in model.range_mut(start..end) {
                    *x = x.wrapping_add(2);
                }
                if start < end {
                    let values: Vec<_> = model.range(start..end).copied().collect();
                    mixed(deque.range_mut(start..=end - 1), &values, |x| *x, step + 1);
                }
            }
            33 => {
                for x in &mut deque {
                    *x = x.wrapping_mul(3);
                }
                for x in &mut model {
                    *x = x.wrapping_mul(3);
                }
                assert!((&deque).into_iter().eq(model.iter()));
            }
            34 => {
                let mut iter = std::mem::take(&mut deque).into_iter();
                let mut expected = std::mem::take(&mut model);
                for i in 0..n {
                    assert_eq!(iter.len(), expected.len());
                    if i % 2 == 0 {
                        assert_eq!(iter.next(), expected.pop_front());
                    } else {
                        assert_eq!(iter.next_back(), expected.pop_back());
                    }
                }
                // The unconsumed tail is dropped here.
                drop(iter);
            }
            _ => {}
        }
        check(&mut deque, &mut model, step);
    }
}

fn hash(value: &impl Hash) -> u64 {
    let mut state = DefaultHasher::new();
    value.hash(&mut state);
    state.finish()
}

fn comparisons<const PAGE: usize>(a: &[i32], b: &[i32]) {
    let left: PagedDeque<i32, PAGE> = a.iter().copied().collect();
    let right: PagedDeque<i32, PAGE> = b.iter().copied().collect();
    let model_left: VecDeque<_> = a.iter().copied().collect();
    let model_right: VecDeque<_> = b.iter().copied().collect();
    assert_eq!(left == right, model_left == model_right);
    assert_eq!(left.cmp(&right), model_left.cmp(&model_right));
    assert_eq!(
        left.partial_cmp(&right),
        model_left.partial_cmp(&model_right)
    );
    let different: PagedDeque<i32, 3> = b.iter().copied().collect();
    assert_eq!(
        left.partial_cmp(&different),
        model_left.partial_cmp(&model_right)
    );
    assert_eq!(
        different.partial_cmp(&left),
        model_right.partial_cmp(&model_left)
    );
    assert_eq!(format!("{left:?}"), format!("{model_left:?}"));
    let mut shifted: PagedDeque<_, 5> = PagedDeque::new();
    shifted.push_back(0);
    shifted.extend(a.iter());
    shifted.pop_front();
    assert_eq!(left, shifted);
    assert_eq!(hash(&left), hash(&shifted));
    // Verify the promised hash scheme, independently of page grouping.
    let mut expected = DefaultHasher::new();
    expected.write_usize(a.len());
    for x in a {
        x.hash(&mut expected);
    }
    assert_eq!(hash(&left), expected.finish());
}

fn search<const PAGE: usize>(values: &[i32], target: i32, head: usize) {
    let mut sorted = values.to_vec();
    sorted.sort();
    let mut deque = PagedDeque::<i32, PAGE>::new();
    deque.extend(std::iter::repeat_n(0, head));
    deque.extend(sorted.iter());
    for _ in 0..head {
        deque.pop_front();
    }
    deque.reserve(PAGE);
    let model: VecDeque<_> = sorted.iter().copied().collect();
    let expected = model.binary_search(&target);
    for result in [
        deque.binary_search(&target),
        deque.binary_search_by(|x| x.cmp(&target)),
        deque.binary_search_by_key(&target, |x| *x),
    ] {
        match (result, expected) {
            (Ok(i), Ok(_)) => assert_eq!(deque[i], target),
            (Err(i), Err(j)) => assert_eq!(i, j),
            pair => panic!("search mismatch: {pair:?}"),
        }
    }
    assert_eq!(
        deque.partition_point(|x| *x < target),
        sorted.partition_point(|x| *x < target)
    );
    assert_eq!(
        deque.partition_point(|x| *x <= target),
        sorted.partition_point(|x| *x <= target)
    );
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn operations_match_vecdeque(
        initial in prop::collection::vec(any::<i32>(), 0..12),
        ops in prop::collection::vec((0u8..35, 0usize..80, any::<i32>()), 0..limit()),
    ) {
        run::<1>(&initial, &ops);
        run::<3>(&initial, &ops);
        run::<8>(&initial, &ops);
    }

    #[test]
    fn comparison_traits_match_vecdeque(
        a in prop::collection::vec(any::<i32>(), 0..limit()),
        b in prop::collection::vec(any::<i32>(), 0..limit()),
    ) {
        comparisons::<1>(&a, &b);
        comparisons::<3>(&a, &b);
        comparisons::<8>(&a, &b);
    }

    #[test]
    fn sorted_searches_match_vecdeque(
        values in prop::collection::vec(-20i32..20, 0..limit()),
        target in -25i32..25,
        head in 0usize..16,
    ) {
        search::<1>(&values, target, head);
        search::<3>(&values, target, head);
        search::<8>(&values, target, head);
    }

    #[test]
    fn zero_sized_operations_match_vecdeque(ops in prop::collection::vec((0u8..10, 0usize..40), 0..limit())) {
        fn run_zst<const PAGE: usize>(ops: &[(u8, usize)]) {
            let mut deque = PagedDeque::<(), PAGE>::new();
            let mut model = VecDeque::new();
            for &(code, n) in ops {
                let index = n % (model.len() + 1);
                match code {
                    0 => { deque.push_front(()); model.push_front(()); }
                    1 => { deque.push_back(()); model.push_back(()); }
                    2 => assert_eq!(deque.pop_front(), model.pop_front()),
                    3 => assert_eq!(deque.pop_back(), model.pop_back()),
                    4 => { deque.insert(index, ()); model.insert(index, ()); }
                    5 => assert_eq!(deque.remove(index), model.remove(index)),
                    6 => { deque.resize(n, ()); model.resize(n, ()); }
                    7 => { deque.truncate_front(n); while model.len() > n { model.pop_front(); } }
                    8 => deque.reserve(n),
                    _ => deque.shrink_to_fit(),
                }
                assert_eq!(deque.len(), model.len());
                assert!(deque.iter().eq(model.iter()));
                assert!(deque.iter().rev().eq(model.iter().rev()));
                assert_eq!(deque.pages().flatten().count(), model.len());
                assert_eq!(deque.pages_mut().flatten().count(), model.len());
                assert_eq!(deque.clone().into_iter().count(), model.len());
            }
        }
        run_zst::<1>(&ops);
        run_zst::<3>(&ops);
        run_zst::<8>(&ops);
    }
}

#[test]
fn partial_order_handles_incomparable_elements() {
    for a in [vec![], vec![f64::NAN], vec![1.0, f64::NAN]] {
        for b in [vec![], vec![1.0], vec![1.0, f64::NAN]] {
            let left: PagedDeque<_, 3> = a.clone().into();
            let right: PagedDeque<_, 3> = b.clone().into();
            assert_eq!(
                left.partial_cmp(&right),
                VecDeque::from(a.clone()).partial_cmp(&VecDeque::from(b))
            );
        }
    }
}

#[cfg(feature = "serde")]
proptest! {
    #![proptest_config(config())]

    #[test]
    fn serde_matches_vecdeque(values in prop::collection::vec(any::<i32>(), 0..limit())) {
        fn round_trip<const PAGE: usize>(values: &[i32]) {
            let mut deque = PagedDeque::<_, PAGE>::new();
            deque.push_back(0);
            deque.extend_from_slice(values);
            deque.pop_front();
            let model: VecDeque<_> = values.iter().copied().collect();
            let json = serde_json::to_string(&deque).unwrap();
            assert_eq!(json, serde_json::to_string(&model).unwrap());
            let mut back: PagedDeque<i32, PAGE> = serde_json::from_str(&json).unwrap();
            assert_eq!(back, deque);
            back.push_front(42);
            back.push_back(99);
            assert_eq!(back.pop_front(), Some(42));
            assert_eq!(back.pop_back(), Some(99));
            assert_eq!(back, deque);
        }
        round_trip::<1>(&values);
        round_trip::<3>(&values);
        round_trip::<8>(&values);
    }
}
