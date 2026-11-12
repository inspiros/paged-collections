//! Sorting, searching and reversing `PagedVec`, against `Vec` as the oracle.

use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind};

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
            cases: 256,
            ..ProptestConfig::default()
        }
    }
}

fn max_len() -> usize {
    if cfg!(miri) { 24 } else { 300 }
}

/// Addresses of the element slots, which must not change when the order does.
fn slots<T, const PAGE: usize>(v: &PagedVec<T, PAGE>) -> Vec<*const T> {
    v.iter().map(|x| x as *const T).collect()
}

fn check_pages<T, const PAGE: usize>(v: &PagedVec<T, PAGE>) {
    let pages: Vec<&[T]> = v.pages().collect();
    assert_eq!(pages.len(), v.len().div_ceil(PAGE));
    if let Some((_, full)) = pages.split_last() {
        assert!(full.iter().all(|p| p.len() == PAGE));
    }
    assert_eq!(v.allocated_pages() * PAGE, v.capacity());
}

/// Elements are `(key, original position)`, so stability is observable.
fn sorts<const PAGE: usize>(data: &[(u8, u16)]) {
    let model: Vec<(u8, u16)> = data.to_vec();
    let base: PagedVec<(u8, u16), PAGE> = model.iter().copied().collect();

    // Each case runs one operation on a copy of the PagedVec and the same
    // slice operation on a copy of the model, then compares.
    macro_rules! case {
        ($name:literal, |$v:ident| $paged:expr, |$s:ident| $slice:expr) => {{
            let mut $v = base.clone();
            let before = slots(&$v);
            $paged;
            let mut $s = model.clone();
            $slice;
            assert!($v.iter().eq($s.iter()), "{} PAGE={PAGE}", $name);
            assert_eq!(slots(&$v), before, "{} moved a page", $name);
            check_pages(&$v);
        }};
    }

    case!("sort", |v| v.sort(), |s| s.sort());
    case!("sort_by", |v| v.sort_by(|a, b| b.cmp(a)), |s| s
        .sort_by(|a, b| b.cmp(a)));
    // Stable sorts by key alone must keep the original positions in order.
    case!("sort_by_key", |v| v.sort_by_key(|x| x.0), |s| s
        .sort_by_key(|x| x.0));
    case!(
        "sort_by_cached_key",
        |v| v.sort_by_cached_key(|x| u32::from(x.0) * 3),
        |s| s.sort_by_cached_key(|x| u32::from(x.0) * 3)
    );
    case!("sort_unstable", |v| v.sort_unstable(), |s| s
        .sort_unstable());
    case!(
        "sort_unstable_by",
        |v| v.sort_unstable_by(|a, b| b.cmp(a)),
        |s| s.sort_unstable_by(|a, b| b.cmp(a))
    );
    case!("reverse", |v| v.reverse(), |s| s.reverse());

    // Unstable by key: only the keys are determined.
    let mut v = base.clone();
    v.sort_unstable_by_key(|x| x.0);
    let mut keys: Vec<u8> = model.iter().map(|x| x.0).collect();
    keys.sort_unstable();
    assert!(v.iter().map(|x| x.0).eq(keys.iter().copied()));
    let mut got: Vec<_> = v.iter().copied().collect();
    let mut want = model.clone();
    got.sort();
    want.sort();
    assert_eq!(got, want);

    // A vector stays usable after sorting: pushes land in the same pages.
    let mut v = base.clone();
    v.sort();
    let before = slots(&v);
    v.extend(model.iter().copied());
    assert_eq!(&slots(&v)[..before.len()], &before[..]);
    check_pages(&v);
}

fn searches<const PAGE: usize>(data: &[u8], probes: &[u8]) {
    let mut model: Vec<u8> = data.to_vec();
    model.sort();
    let v: PagedVec<u8, PAGE> = model.iter().copied().collect();
    for &x in probes.iter().chain(&[0, u8::MAX]) {
        assert_eq!(v.contains(&x), model.contains(&x));
        let pp = v.partition_point(|&y| y < x);
        assert_eq!(pp, model.partition_point(|&y| y < x), "PAGE={PAGE}");
        let le = v.partition_point(|&y| y <= x);
        assert_eq!(le, model.partition_point(|&y| y <= x));
        // Any matching index is allowed, so check the result's meaning.
        match v.binary_search(&x) {
            Ok(i) => assert_eq!(v[i], x),
            Err(i) => {
                assert!(!model.contains(&x));
                assert_eq!(i, pp);
            }
        }
        match v.binary_search_by_key(&(u16::from(x) * 2), |&y| u16::from(y) * 2) {
            Ok(i) => assert_eq!(v[i], x),
            Err(i) => assert_eq!(i, pp),
        }
        let by = v.binary_search_by(|y| y.cmp(&x));
        assert_eq!(by.is_ok(), model.binary_search(&x).is_ok());
    }
}

/// A panicking comparator leaves the same elements, in some order, in
/// correctly shaped pages, and drops nothing.
fn panicking_compare<const PAGE: usize>(data: &[u16], panic_after: u16) {
    let model: Vec<u16> = data.to_vec();
    let mut v: PagedVec<u16, PAGE> = model.iter().copied().collect();
    let before = slots(&v);
    let calls = Cell::new(0u16);
    let result = catch_unwind(AssertUnwindSafe(|| {
        v.sort_by(|a, b| {
            calls.set(calls.get() + 1);
            assert!(calls.get() <= panic_after, "comparator bomb");
            a.cmp(b)
        });
    }));
    let mut got: Vec<u16> = v.iter().copied().collect();
    let mut want = model.clone();
    if result.is_ok() {
        assert!(got.is_sorted());
    }
    got.sort();
    want.sort();
    assert_eq!(got, want);
    assert_eq!(slots(&v), before);
    check_pages(&v);
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn sorts_match_slice_sorts(data in prop::collection::vec((0u8..8, any::<u16>()), 0..max_len())) {
        // Make the second field the original position, for stability checks.
        let data: Vec<(u8, u16)> = data.iter().enumerate().map(|(i, x)| (x.0, i as u16)).collect();
        sorts::<1>(&data);
        sorts::<2>(&data);
        sorts::<3>(&data);
        sorts::<7>(&data);
        sorts::<1024>(&data);
    }

    #[test]
    fn searches_match_slice_searches(
        data in prop::collection::vec(prop_oneof![0u8..16, any::<u8>()], 0..max_len()),
        probes in prop::collection::vec(any::<u8>(), 0..16),
    ) {
        searches::<1>(&data, &probes);
        searches::<2>(&data, &probes);
        searches::<5>(&data, &probes);
        searches::<1024>(&data, &probes);
    }

    #[test]
    fn panicking_comparator_keeps_elements(
        data in prop::collection::vec(any::<u16>(), 0..max_len()),
        panic_after in 0u16..400,
    ) {
        panicking_compare::<1>(&data, panic_after);
        panicking_compare::<3>(&data, panic_after);
        panicking_compare::<16>(&data, panic_after);
    }
}

#[test]
fn with_contiguous_mut_sees_all_elements_in_order() {
    let mut v: PagedVec<u32, 4> = (0..10).collect();
    let seen = v.with_contiguous_mut(|s| {
        let seen = s.to_vec();
        s.rotate_right(3);
        seen
    });
    assert_eq!(seen, (0..10).collect::<Vec<_>>());
    assert!(v.iter().copied().eq([7, 8, 9, 0, 1, 2, 3, 4, 5, 6]));
    check_pages(&v);

    let mut empty: PagedVec<u32, 4> = PagedVec::new();
    assert_eq!(empty.with_contiguous_mut(|s| s.len()), 0);
    empty.sort();
    assert_eq!(empty.binary_search(&3), Err(0));
    assert_eq!(empty.partition_point(|_| true), 0);
}

#[test]
fn sorting_drops_nothing_and_handles_zsts() {
    use std::rc::Rc;
    let token = Rc::new(());
    let mut v: PagedVec<(u8, Rc<()>), 3> =
        (0..20).map(|i| ((i * 7) % 5, Rc::clone(&token))).collect();
    v.sort_by_key(|x| x.0);
    assert_eq!(Rc::strong_count(&token), 21);
    assert!(v.iter().map(|x| x.0).is_sorted());
    drop(v);
    assert_eq!(Rc::strong_count(&token), 1);

    let mut z: PagedVec<(), 4> = PagedVec::new();
    z.resize(9, ());
    z.sort();
    z.reverse();
    assert_eq!(z.len(), 9);
    assert_eq!(z.binary_search(&()).map(|i| i < 9), Ok(true));
}
