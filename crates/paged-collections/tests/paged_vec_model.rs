//! Vec oracles for values, page layout, stable addresses, and unwind ownership.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeSet, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};
use std::iter::FusedIterator;
use std::panic::{AssertUnwindSafe, catch_unwind};
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

#[derive(Clone, Debug)]
enum Op {
    Push(i32),
    PushMut(i32),
    Pop,
    Access(usize, i32),
    Ends(i32),
    Swap(usize, usize),
    Insert(usize, i32),
    InsertEnd(i32),
    InsertBoundary(usize, i32),
    Remove(usize),
    SwapRemove(usize),
    Truncate(usize),
    Clear,
    Resize(usize, i32),
    ResizeWith(usize, i32),
    Retain(i32),
    RetainMut(i32),
    Reserve(usize),
    TryReserve(usize),
    Shrink,
    Extend(Vec<i32>),
    ExtendRef(Vec<i32>),
    Clone,
    Collect,
    FromVec,
    FromArray([i32; 4]),
}

fn op() -> impl Strategy<Value = Op> {
    let index = 0..max_len() + 3;
    prop_oneof![
        11 => prop_oneof![
            3 => any::<i32>().prop_map(Op::Push),
            2 => any::<i32>().prop_map(Op::PushMut),
            2 => Just(Op::Pop),
            2 => (index.clone(), any::<i32>()).prop_map(|(i, x)| Op::Access(i, x)),
            1 => any::<i32>().prop_map(Op::Ends),
            1 => (index.clone(), index.clone()).prop_map(|(a, b)| Op::Swap(a, b)),
        ].boxed(),
        10 => prop_oneof![
            2 => (index.clone(), any::<i32>()).prop_map(|(i, x)| Op::Insert(i, x)),
            1 => any::<i32>().prop_map(Op::InsertEnd),
            1 => (index.clone(), any::<i32>()).prop_map(|(i, x)| Op::InsertBoundary(i, x)),
            2 => index.clone().prop_map(Op::Remove),
            2 => index.clone().prop_map(Op::SwapRemove),
            1 => index.clone().prop_map(Op::Truncate),
            1 => Just(Op::Clear),
        ].boxed(),
        9 => prop_oneof![
            2 => (index.clone(), any::<i32>()).prop_map(|(n, x)| Op::Resize(n, x)),
            2 => (index.clone(), any::<i32>()).prop_map(|(n, x)| Op::ResizeWith(n, x)),
            1 => any::<i32>().prop_map(Op::Retain),
            1 => any::<i32>().prop_map(Op::RetainMut),
            1 => (0..12usize).prop_map(Op::Reserve),
            1 => (0..12usize).prop_map(Op::TryReserve),
            1 => Just(Op::Shrink),
        ].boxed(),
        8 => prop_oneof![
            2 => prop::collection::vec(any::<i32>(), 0..8).prop_map(Op::Extend),
            2 => prop::collection::vec(any::<i32>(), 0..8).prop_map(Op::ExtendRef),
            1 => Just(Op::Clone),
            1 => Just(Op::Collect),
            1 => Just(Op::FromVec),
            1 => any::<[i32; 4]>().prop_map(Op::FromArray),
        ].boxed(),
    ]
}

fn addresses<T, const PAGE: usize>(v: &PagedVec<T, PAGE>) -> Vec<*const T> {
    v.iter().map(std::ptr::from_ref).collect()
}

fn assert_stable<T, const PAGE: usize>(v: &PagedVec<T, PAGE>, before: &[*const T]) {
    for (value, &address) in v.iter().zip(before) {
        assert!(std::ptr::eq(value, address), "PAGE={PAGE}");
    }
}

// `is_empty` is checked against `len` on purpose.
#[allow(clippy::len_zero)]
fn layout<T, const PAGE: usize>(v: &PagedVec<T, PAGE>) {
    assert_eq!(v.is_empty(), v.len() == 0);
    assert!(v.capacity() >= v.len());
    assert_eq!(v.allocated_pages() * PAGE, v.capacity());
    let count = v.len().div_ceil(PAGE);
    assert_eq!(v.pages().len(), count);
    for (i, page) in v.pages().enumerate() {
        assert_eq!(page.len(), PAGE.min(v.len() - i * PAGE));
    }
    assert_eq!(v.pages().flatten().count(), v.len());
    assert_eq!(v.iter().count(), v.len());
}

// Choose runs from both ends, including meeting inside a partially read page.
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

fn check<const PAGE: usize>(v: &mut PagedVec<i32, PAGE>, model: &mut [i32], salt: usize) {
    layout(v);
    assert_eq!(v.len(), model.len());
    assert!(v.iter().eq(model.iter()));
    assert!(v.iter().rev().eq(model.iter().rev()));
    assert!(v.pages().flatten().eq(model.iter()));
    assert_eq!(v.first(), model.first());
    assert_eq!(v.last(), model.last());
    mixed(v.iter(), model, |x| *x, salt);
    mixed(v.clone().into_iter(), model, |x| x, salt + 1);
    mixed(
        v.iter_mut(),
        model,
        |x| {
            let old = *x;
            *x = x.wrapping_add(1);
            old
        },
        salt + 2,
    );
    for x in model.iter_mut() {
        *x = x.wrapping_add(1);
    }
    assert!(v.iter().eq(model.iter()));
    assert_eq!(v.pages_mut().len(), model.len().div_ceil(PAGE));
    for page in v.pages_mut() {
        for x in page {
            *x = x.wrapping_sub(3);
        }
    }
    for x in model.iter_mut() {
        *x = x.wrapping_sub(3);
    }
    assert!(v.iter().eq(model.iter()));
}

fn push_probe<const PAGE: usize>(v: &mut PagedVec<i32, PAGE>) {
    let before = addresses(v);
    v.push(123);
    assert_stable(v, &before);
    assert_eq!(v.pop(), Some(123));
}

fn run<const PAGE: usize>(initial: &[i32], ops: &[Op]) {
    let mut v = PagedVec::<i32, PAGE>::with_capacity(initial.len() + 3);
    v.extend(initial);
    let mut model = initial.to_vec();
    check(&mut v, &mut model, 0);
    for (step, op) in ops.iter().enumerate() {
        let stable = matches!(
            op,
            Op::Push(_)
                | Op::PushMut(_)
                | Op::Pop
                | Op::Truncate(_)
                | Op::Clear
                | Op::Resize(..)
                | Op::ResizeWith(..)
                | Op::Reserve(_)
                | Op::TryReserve(_)
                | Op::Shrink
                | Op::Extend(_)
                | Op::ExtendRef(_)
        );
        let before = stable.then(|| addresses(&v));
        match op {
            Op::Push(x) | Op::PushMut(x) if model.len() < max_len() => {
                if matches!(op, Op::PushMut(_)) {
                    *v.push_mut(*x) = x.wrapping_add(7);
                    model.push(x.wrapping_add(7));
                } else {
                    v.push(*x);
                    model.push(*x);
                }
            }
            Op::Pop => assert_eq!(v.pop(), model.pop()),
            Op::Access(raw, x) => {
                let i = raw % (model.len() + 3);
                assert_eq!(v.get(i), model.get(i));
                assert_eq!(v.get_mut(i).map(|x| *x), model.get(i).copied());
                if i < model.len() {
                    assert_eq!(v[i], model[i]);
                    *v.get_mut(i).unwrap() = *x;
                    model[i] = *x;
                    v[i] = x.wrapping_add(1);
                    model[i] = x.wrapping_add(1);
                } else {
                    assert!(catch_unwind(|| v[i]).is_err());
                    assert!(catch_unwind(AssertUnwindSafe(|| v[i] = *x)).is_err());
                }
            }
            Op::Ends(x) => {
                assert_eq!(v.first_mut().map(|x| *x), model.first().copied());
                if let Some(first) = v.first_mut() {
                    *first = *x;
                    model[0] = *x;
                }
                assert_eq!(v.last_mut().map(|x| *x), model.last().copied());
                if let Some(last) = v.last_mut() {
                    *last = x.wrapping_sub(1);
                    *model.last_mut().unwrap() = x.wrapping_sub(1);
                }
            }
            Op::Swap(a, b) if !model.is_empty() => {
                let (a, b) = (a % model.len(), b % model.len());
                v.swap(a, b);
                model.swap(a, b);
            }
            Op::Insert(raw, x) | Op::InsertBoundary(raw, x) if model.len() < max_len() => {
                let i = if matches!(op, Op::InsertBoundary(..)) {
                    (raw % (model.len() / PAGE + 1)) * PAGE
                } else {
                    raw % (model.len() + 1)
                };
                v.insert(i, *x);
                model.insert(i, *x);
            }
            Op::InsertEnd(x) if model.len() < max_len() => {
                v.insert(model.len(), *x);
                model.push(*x);
            }
            Op::Remove(raw) if !model.is_empty() => {
                let i = raw % model.len();
                assert_eq!(v.remove(i), model.remove(i));
            }
            Op::SwapRemove(raw) if !model.is_empty() => {
                let i = raw % model.len();
                assert_eq!(v.swap_remove(i), model.swap_remove(i));
            }
            Op::Truncate(n) => {
                v.truncate(*n);
                model.truncate(*n);
            }
            Op::Clear => {
                v.clear();
                model.clear();
            }
            Op::Resize(n, x) => {
                let n = n % (max_len() + 1);
                v.resize(n, *x);
                model.resize(n, *x);
            }
            Op::ResizeWith(n, x) => {
                let n = n % (max_len() + 1);
                let mut next = *x;
                v.resize_with(n, || {
                    next = next.wrapping_add(1);
                    next
                });
                let mut next = *x;
                model.resize_with(n, || {
                    next = next.wrapping_add(1);
                    next
                });
            }
            Op::Retain(mask) => {
                v.retain(|x| x & mask != 0);
                model.retain(|x| x & mask != 0);
            }
            Op::RetainMut(mask) => {
                let keep = |x: &mut i32| {
                    *x = x.wrapping_add(3);
                    *x & mask != 0
                };
                v.retain_mut(keep);
                model.retain_mut(keep);
            }
            Op::Reserve(n) => {
                v.reserve(*n);
                assert!(v.capacity() >= v.len() + n);
            }
            Op::TryReserve(n) => {
                v.try_reserve(*n).unwrap();
                assert!(v.capacity() >= v.len() + n);
            }
            Op::Shrink => {
                v.shrink_to_fit();
                assert_eq!(v.allocated_pages(), v.len().div_ceil(PAGE));
            }
            Op::Extend(xs) | Op::ExtendRef(xs) => {
                let xs = &xs[..xs.len().min(max_len().saturating_sub(model.len()))];
                if matches!(op, Op::ExtendRef(_)) {
                    v.extend(xs);
                } else {
                    v.extend(xs.iter().copied());
                }
                model.extend_from_slice(xs);
            }
            Op::Clone => {
                let mut copy = v.clone();
                assert_eq!(copy, v);
                push_probe(&mut copy);
                v = copy;
            }
            Op::Collect => {
                v = model.iter().copied().collect();
                push_probe(&mut v);
            }
            Op::FromVec => {
                v = PagedVec::from(model.clone());
                push_probe(&mut v);
            }
            Op::FromArray(xs) => {
                v = PagedVec::from(*xs);
                model = xs.to_vec();
                push_probe(&mut v);
            }
            _ => {}
        }
        if let Some(before) = before {
            assert_stable(&v, &before);
        }
        check(&mut v, &mut model, step);
    }
}

fn hash(v: &impl Hash) -> u64 {
    let mut state = DefaultHasher::new();
    v.hash(&mut state);
    state.finish()
}

fn comparisons<const PAGE: usize>(a: &[i32], b: &[i32]) {
    let left: PagedVec<i32, PAGE> = a.iter().copied().collect();
    let right: PagedVec<i32, PAGE> = b.iter().copied().collect();
    let other: PagedVec<i32, 7> = a.iter().copied().collect();
    let different: PagedVec<i32, 3> = b.iter().copied().collect();
    assert_eq!(left == different, a == b);
    assert_eq!(left, other);
    assert_eq!(other, left);
    assert_eq!(left.cmp(&right), a.cmp(b));
    assert_eq!(left.partial_cmp(&right), a.partial_cmp(b));
    assert_eq!(hash(&left), hash(&other));
    assert_eq!(hash(&left), hash(&left.clone()));
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn sequences_match_vec(
        initial in prop::collection::vec(any::<i32>(), 0..16),
        ops in prop::collection::vec(op(), 0..max_len()),
    ) {
        run::<1>(&initial, &ops);
        run::<2>(&initial, &ops);
        run::<3>(&initial, &ops);
        run::<7>(&initial, &ops);
        run::<1024>(&initial, &ops);
    }

    #[test]
    fn ordering_equality_and_hash(
        a in prop::collection::vec(any::<i32>(), 0..max_len()),
        b in prop::collection::vec(any::<i32>(), 0..max_len()),
    ) {
        comparisons::<1>(&a, &b);
        comparisons::<2>(&a, &b);
        comparisons::<3>(&a, &b);
        comparisons::<1024>(&a, &b);
    }
}

// In particular, fill and cross the default page: short random sequences
// otherwise exercise only its first page. Push through every spare-page slot.
fn boundaries<const PAGE: usize>() {
    let n = 2 * PAGE + 1;
    let model: Vec<i32> = (0..n).map(|i| i as i32).collect();
    let collected: PagedVec<i32, PAGE> = model.iter().copied().collect();
    for mut v in [collected.clone(), collected, PagedVec::from(model.clone())] {
        let mut expected = model.clone();
        v.reserve(2 * PAGE);
        let before = addresses(&v);
        // Push one at a time to check stability after each push.
        #[allow(clippy::same_item_push)]
        for _ in 0..2 * PAGE {
            v.push(42);
            expected.push(42);
            assert_stable(&v, &before);
        }
        v.truncate(n);
        expected.truncate(n);
        v.shrink_to_fit();
        assert_stable(&v, &before);
        assert_eq!(v.allocated_pages(), n.div_ceil(PAGE));
        // The last used page must retain capacity after shrinking.
        #[allow(clippy::same_item_push)]
        for _ in 0..PAGE {
            v.push(123);
            expected.push(123);
            assert_stable(&v, &before);
        }
        check(&mut v, &mut expected, 0);
        v.truncate(n);
        expected.truncate(n);
        for i in [PAGE, 0, 2 * PAGE, v.len()] {
            v.insert(i, -1);
            expected.insert(i, -1);
            check(&mut v, &mut expected, i);
            assert_eq!(v.remove(i), expected.remove(i));
            check(&mut v, &mut expected, i + 1);
        }
    }
}

#[test]
fn full_and_partial_page_boundaries() {
    boundaries::<1>();
    boundaries::<2>();
    boundaries::<3>();
    boundaries::<7>();
    if !cfg!(miri) {
        boundaries::<1024>();
    }
    let default: PagedVec<i32> = PagedVec::new();
    assert_eq!(PagedVec::<i32>::PAGE, 1024);
    assert!(default.is_empty());
}

fn invalid_indices<const PAGE: usize>() {
    let mut v = PagedVec::<i32, PAGE>::from([10, 20, 30]);
    v.reserve(9);
    for empty in [false, true] {
        if empty {
            v.clear();
        }
        let expected: Vec<_> = v.iter().copied().collect();
        for i in [v.len(), v.len() + PAGE, usize::MAX] {
            assert_eq!(v.get(i), None);
            assert_eq!(v.get_mut(i), None);
            assert!(catch_unwind(|| v[i]).is_err());
            assert!(catch_unwind(AssertUnwindSafe(|| v[i] = 99)).is_err());
            layout(&v);
            assert!(v.iter().eq(expected.iter()));
        }
    }
    v.push(42);
    assert_eq!(v[0], 42);
}

#[test]
fn indexing_panics_beyond_len_even_with_spare_pages() {
    invalid_indices::<1>();
    invalid_indices::<2>();
    invalid_indices::<3>();
    invalid_indices::<7>();
    invalid_indices::<1024>();
}

#[derive(Default)]
struct Ledger {
    drops: RefCell<Vec<usize>>,
    panic_drop: Cell<Option<usize>>,
    clones_until_panic: Cell<Option<usize>>,
}

impl Ledger {
    fn make(self: &Rc<Self>, value: i32) -> Item {
        let mut drops = self.drops.borrow_mut();
        let id = drops.len();
        drops.push(0);
        Item {
            id,
            value,
            ledger: Rc::clone(self),
        }
    }

    fn check_live(&self, live: impl IntoIterator<Item = usize>) {
        let ids: Vec<_> = live.into_iter().collect();
        let set: BTreeSet<_> = ids.iter().copied().collect();
        assert_eq!(ids.len(), set.len(), "duplicate live identity");
        for (id, &drops) in self.drops.borrow().iter().enumerate() {
            assert_eq!(drops, usize::from(!set.contains(&id)), "id={id}");
        }
    }
}

struct Item {
    id: usize,
    value: i32,
    ledger: Rc<Ledger>,
}

impl Clone for Item {
    fn clone(&self) -> Self {
        if let Some(remaining) = self.ledger.clones_until_panic.get() {
            if remaining == 0 {
                self.ledger.clones_until_panic.set(None);
                panic!("clone bomb");
            }
            self.ledger.clones_until_panic.set(Some(remaining - 1));
        }
        self.ledger.make(self.value)
    }
}

impl Drop for Item {
    fn drop(&mut self) {
        self.ledger.drops.borrow_mut()[self.id] += 1;
        if self.ledger.panic_drop.get() == Some(self.id) {
            self.ledger.panic_drop.set(None);
            panic!("drop bomb");
        }
    }
}

fn item_state<const PAGE: usize>(v: &PagedVec<Item, PAGE>) -> Vec<(usize, i32)> {
    v.iter().map(|x| (x.id, x.value)).collect()
}

// A separate identity oracle catches leaks/duplicate drops that value equality
// cannot see. Clone allocates new identities, including resize's fill value.
fn drop_sequence<const PAGE: usize>(ops: &[(u8, usize, i32)]) {
    let ledger = Rc::new(Ledger::default());
    let mut v = PagedVec::<Item, PAGE>::new();
    let mut model = Vec::new();
    for &(code, raw, value) in ops {
        let n = raw % 25;
        match code {
            0 | 1 => {
                let item = ledger.make(value);
                model.push((
                    item.id,
                    if code == 1 {
                        value.wrapping_add(1)
                    } else {
                        value
                    },
                ));
                if code == 1 {
                    v.push_mut(item).value = value.wrapping_add(1);
                } else {
                    v.push(item);
                }
            }
            2 => {
                let popped = v.pop();
                assert_eq!(popped.as_ref().map(|x| (x.id, x.value)), model.pop());
                drop(popped);
            }
            3 => {
                let i = raw % (model.len() + 1);
                let item = ledger.make(value);
                model.insert(i, (item.id, value));
                v.insert(i, item);
            }
            4 | 5 if !model.is_empty() => {
                let i = raw % model.len();
                let (item, expected) = if code == 4 {
                    (v.remove(i), model.remove(i))
                } else {
                    (v.swap_remove(i), model.swap_remove(i))
                };
                assert_eq!((item.id, item.value), expected);
                drop(item);
            }
            6 => {
                v.truncate(n);
                model.truncate(n);
            }
            7 => {
                v.clear();
                model.clear();
            }
            8 | 9 => {
                let old_len = model.len();
                if code == 8 {
                    v.resize_with(n, || ledger.make(value));
                } else {
                    v.resize(n, ledger.make(value));
                }
                assert_eq!(v.len(), n);
                model.truncate(n);
                // New identities may be clones or the moved fill value;
                // their values and the surviving prefix have independent oracles.
                model.extend(v.iter().skip(old_len).map(|x| (x.id, value)));
            }
            10 => {
                v.retain(|x| x.value & value != 0);
                model.retain(|&(_, x)| x & value != 0);
            }
            11 => {
                v.retain_mut(|x| {
                    x.value = x.value.wrapping_add(1);
                    x.value & value != 0
                });
                model.retain_mut(|(_, x)| {
                    *x = x.wrapping_add(1);
                    *x & value != 0
                });
            }
            12 => {
                let mut copy = v.clone();
                assert!(copy.iter().map(|x| x.value).eq(model.iter().map(|x| x.1)));
                let before = addresses(&copy);
                copy.push(ledger.make(value));
                assert_stable(&copy, &before);
                drop(copy);
            }
            13 => v.reserve(n),
            14 => v.try_reserve(n).unwrap(),
            15 => {
                v.shrink_to_fit();
                assert_eq!(v.allocated_pages(), v.len().div_ceil(PAGE));
            }
            16 => {
                let items: Vec<_> = (0..n.min(5)).map(|_| ledger.make(value)).collect();
                model.extend(items.iter().map(|x| (x.id, x.value)));
                v.extend(items);
            }
            17 => {
                // Leave a partially consumed front AND back page to drop.
                let mut iter = std::mem::take(&mut v).into_iter();
                let mut expected = model.into_iter();
                for i in 0..n {
                    assert_eq!(iter.len(), expected.len());
                    let (item, pair) = if i % 2 == 0 {
                        (iter.next(), expected.next())
                    } else {
                        (iter.next_back(), expected.next_back())
                    };
                    assert_eq!(item.as_ref().map(|x| (x.id, x.value)), pair);
                    drop(item);
                }
                drop(iter);
                model = Vec::new();
            }
            18 if !model.is_empty() => {
                let i = raw % model.len();
                v.swap(0, i);
                model.swap(0, i);
                v.get_mut(i).unwrap().value = value;
                model[i].1 = value;
            }
            19 => {
                v = v.into_iter().collect();
            }
            20 => {
                v = PagedVec::from(v.into_iter().collect::<Vec<_>>());
            }
            21 => {
                let items = [ledger.make(value), ledger.make(value.wrapping_add(1))];
                model = items.iter().map(|x| (x.id, x.value)).collect();
                v = PagedVec::from(items);
            }
            _ => {}
        }
        layout(&v);
        assert_eq!(item_state(&v), model);
        ledger.check_live(model.iter().map(|x| x.0));
    }
    drop(v);
    ledger.check_live([]);
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn every_created_identity_is_dropped_once(
        ops in prop::collection::vec((0u8..22, 0..max_len(), any::<i32>()), 0..max_len()),
    ) {
        drop_sequence::<1>(&ops);
        drop_sequence::<3>(&ops);
        drop_sequence::<7>(&ops);
    }
}

fn retain_panics<const PAGE: usize>() {
    let n = 2 * PAGE + 3;
    for mutable in [false, true] {
        for k in 0..n {
            let ledger = Rc::new(Ledger::default());
            let mut v: PagedVec<Item, PAGE> = (0..n).map(|i| ledger.make(i as i32)).collect();
            let mut calls = 0;
            let result = catch_unwind(AssertUnwindSafe(|| {
                if mutable {
                    v.retain_mut(|x| {
                        let call = calls;
                        calls += 1;
                        x.value += 100;
                        assert_ne!(call, k, "predicate bomb");
                        x.id % 3 == 0
                    });
                } else {
                    v.retain(|x| {
                        let call = calls;
                        calls += 1;
                        assert_ne!(call, k, "predicate bomb");
                        x.id % 3 == 0
                    });
                }
            }));
            assert!(result.is_err());
            assert_eq!(calls, k + 1);
            let expected: Vec<_> = (0..n)
                .filter(|&i| i >= k || i % 3 == 0)
                .map(|i| (i, i as i32 + if mutable && i <= k { 100 } else { 0 }))
                .collect();
            layout(&v);
            assert_eq!(item_state(&v), expected);
            ledger.check_live(expected.iter().map(|x| x.0));
            v.push(ledger.make(-1));
            v.retain(|_| true);
            drop(v);
            ledger.check_live([]);
        }
    }
}

#[test]
fn retain_and_retain_mut_preserve_exact_unwind_tail() {
    retain_panics::<1>();
    retain_panics::<2>();
    retain_panics::<3>();
    retain_panics::<7>();
}

fn drop_panics<const PAGE: usize>() {
    let n = 3 * PAGE + 2;
    for target in [0, 1, PAGE, PAGE + 1, n - 1] {
        for bomb in target..n {
            let ledger = Rc::new(Ledger::default());
            let mut v: PagedVec<Item, PAGE> = (0..n).map(|i| ledger.make(i as i32)).collect();
            ledger.panic_drop.set(Some(bomb));
            let before = addresses(&v);
            let result = catch_unwind(AssertUnwindSafe(|| {
                if target == 0 {
                    v.clear();
                } else {
                    v.truncate(target);
                }
            }));
            assert!(result.is_err());
            // Elements drop one at a time from the back: everything after
            // the bomb is gone, the bomb itself is dropped, and everything
            // before it stays.
            let kept = bomb;
            let expected: Vec<_> = (0..kept).map(|i| (i, i as i32)).collect();
            assert_eq!(item_state(&v), expected);
            layout(&v);
            assert_stable(&v, &before);
            ledger.check_live(0..kept);
            v.push(ledger.make(123));
            v.insert(0, ledger.make(456));
            drop(v.remove(0));
            v.reserve(PAGE + 1);
            layout(&v);
            ledger.check_live(v.iter().map(|x| x.id));
            v.clear();
            v.shrink_to_fit();
            assert_eq!(v.allocated_pages(), 0);
            drop(v);
            ledger.check_live([]);
        }
    }
}

#[test]
fn truncate_and_clear_survive_a_panicking_destructor() {
    drop_panics::<1>();
    drop_panics::<2>();
    drop_panics::<3>();
    drop_panics::<7>();
}

fn clone_panics<const PAGE: usize>() {
    let n = 2 * PAGE + 3;
    for k in [0, 1, PAGE - 1, PAGE, PAGE + 1, n - 1] {
        let ledger = Rc::new(Ledger::default());
        let v: PagedVec<Item, PAGE> = (0..n).map(|i| ledger.make(i as i32)).collect();
        let original = item_state(&v);
        let before = addresses(&v);
        ledger.clones_until_panic.set(Some(k));
        assert!(catch_unwind(AssertUnwindSafe(|| v.clone())).is_err());
        assert_eq!(ledger.drops.borrow().len(), n + k);
        assert_eq!(item_state(&v), original);
        layout(&v);
        assert_stable(&v, &before);
        ledger.check_live(0..n);
        drop(v.clone());
        ledger.check_live(0..n);
        drop(v);
        ledger.check_live([]);
    }
}

#[test]
fn clone_unwind_drops_completed_and_partial_pages() {
    clone_panics::<1>();
    clone_panics::<2>();
    clone_panics::<3>();
    clone_panics::<7>();
}

fn partial_into_iter_drop<const PAGE: usize>() {
    let n = 2 * PAGE + 3;
    for consumed in 0..=n + 1 {
        let ledger = Rc::new(Ledger::default());
        let v: PagedVec<Item, PAGE> = (0..n).map(|i| ledger.make(i as i32)).collect();
        let mut iter = v.into_iter();
        let mut expected = (0..n).collect::<Vec<_>>().into_iter();
        for step in 0..consumed {
            assert_eq!(iter.len(), expected.len());
            let (item, id) = if step % 3 == 0 {
                (iter.next(), expected.next())
            } else {
                (iter.next_back(), expected.next_back())
            };
            assert_eq!(item.as_ref().map(|x| x.id), id);
            drop(item);
            ledger.check_live(expected.as_slice().iter().copied());
        }
        drop(iter);
        ledger.check_live([]);
    }
}

#[test]
fn dropping_an_owning_iterator_drops_both_partial_pages_and_middle() {
    partial_into_iter_drop::<1>();
    partial_into_iter_drop::<2>();
    partial_into_iter_drop::<3>();
    partial_into_iter_drop::<7>();
}

#[test]
fn partial_order_matches_vec_with_incomparable_values() {
    for a in [vec![], vec![1.0], vec![f64::NAN], vec![1.0, f64::NAN]] {
        for b in [vec![], vec![1.0], vec![f64::NAN], vec![1.0, 2.0]] {
            let left: PagedVec<f64, 2> = a.iter().copied().collect();
            let right: PagedVec<f64, 2> = b.iter().copied().collect();
            assert_eq!(left.partial_cmp(&right), a.partial_cmp(&b));
        }
    }
}

fn resize_panics<const PAGE: usize>() {
    for k in 0..2 * PAGE + 2 {
        let ledger = Rc::new(Ledger::default());
        let mut v: PagedVec<Item, PAGE> = (0..PAGE + 1).map(|i| ledger.make(i as i32)).collect();
        let before = addresses(&v);
        let mut calls = 0;
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                v.resize_with(4 * PAGE + 5, || {
                    let call = calls;
                    calls += 1;
                    assert_ne!(call, k, "factory bomb");
                    ledger.make(100 + call as i32)
                });
            }))
            .is_err()
        );
        let expected: Vec<_> = (0..PAGE + 1)
            .map(|i| (i, i as i32))
            .chain((0..k).map(|i| (PAGE + 1 + i, 100 + i as i32)))
            .collect();
        assert_eq!(calls, k + 1);
        assert_eq!(item_state(&v), expected);
        assert_stable(&v, &before);
        layout(&v);
        ledger.check_live(expected.iter().map(|x| x.0));
        v.push(ledger.make(-1));
        drop(v);
        ledger.check_live([]);
    }
}

#[test]
fn resize_with_keeps_each_value_created_before_panic() {
    resize_panics::<1>();
    resize_panics::<2>();
    resize_panics::<3>();
    resize_panics::<7>();
}

fn zst_sequence<const PAGE: usize>(ops: &[(u8, usize)]) {
    let mut v = PagedVec::<(), PAGE>::new();
    let mut model = Vec::new();
    for &(code, raw) in ops {
        let n = raw % max_len();
        match code {
            0 => {
                v.push(());
                model.push(());
            }
            1 => {
                *v.push_mut(()) = ();
                model.push(());
            }
            2 => assert_eq!(v.pop(), model.pop()),
            3 => {
                v.resize(n, ());
                model.resize(n, ());
            }
            4 => {
                v.resize_with(n, || ());
                model.resize_with(n, || ());
            }
            5 => {
                let i = raw % (model.len() + 1);
                v.insert(i, ());
                model.insert(i, ());
            }
            6 | 7 if !model.is_empty() => {
                let i = raw % model.len();
                if code == 6 {
                    v.remove(i);
                    model.remove(i);
                } else {
                    v.swap_remove(i);
                    model.swap_remove(i);
                }
            }
            8 => {
                v.truncate(n);
                model.truncate(n);
            }
            9 => {
                v.clear();
                model.clear();
            }
            10 => {
                v.reserve(n);
                v.try_reserve(n).unwrap();
            }
            11 => {
                v.shrink_to_fit();
                assert_eq!(v.allocated_pages(), v.len().div_ceil(PAGE));
            }
            12 => {
                v.extend([(); 3]);
                model.extend([(); 3]);
                let values = [(); 2];
                v.extend(values.iter());
                model.extend(values);
            }
            13 => {
                let mut calls = 0;
                v.retain(|_| {
                    calls += 1;
                    calls % 2 == 0
                });
                let mut calls = 0;
                model.retain(|_| {
                    calls += 1;
                    calls % 2 == 0
                });
            }
            14 => {
                let mut calls = 0;
                v.retain_mut(|_| {
                    calls += 1;
                    calls % 3 == 0
                });
                let mut calls = 0;
                model.retain_mut(|_| {
                    calls += 1;
                    calls % 3 == 0
                });
            }
            _ => {}
        }
        layout(&v);
        assert_eq!(v.len(), model.len());
        assert!(v.iter().eq(model.iter()));
        assert!(v.iter().rev().eq(model.iter().rev()));
        assert_eq!(v.iter_mut().count(), model.len());
        assert_eq!(v.clone().into_iter().count(), model.len());
        assert_eq!(v.get(v.len()), None);
        assert_eq!(v.first(), model.first());
        assert_eq!(v.last(), model.last());
    }
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn zero_sized_sequences(ops in prop::collection::vec((0u8..15, 0..max_len()), 0..max_len())) {
        zst_sequence::<1>(&ops);
        zst_sequence::<4>(&ops);
    }
}

#[cfg(feature = "serde")]
fn serde_round_trip<const PAGE: usize>(model: &[i32]) {
    let v: PagedVec<i32, PAGE> = model.iter().copied().collect();
    let json = serde_json::to_string(&v).unwrap();
    assert_eq!(json, serde_json::to_string(model).unwrap());
    let mut back: PagedVec<i32, PAGE> = serde_json::from_str(&json).unwrap();
    assert_eq!(back, v);
    layout(&back);
    let before = addresses(&back);
    // Fill the partial last page and cross into the next one.
    for _ in 0..PAGE + 1 {
        back.push(123);
        assert_stable(&back, &before);
    }
    assert!(back.iter().take(model.len()).eq(model.iter()));
    layout(&back);
}

#[cfg(feature = "serde")]
proptest! {
    #![proptest_config(config())]

    #[test]
    fn serde_is_a_vec_sequence(model in prop::collection::vec(any::<i32>(), 0..max_len())) {
        serde_round_trip::<1>(&model);
        serde_round_trip::<2>(&model);
        serde_round_trip::<3>(&model);
        serde_round_trip::<7>(&model);
        // Random short inputs still test the default page's partial capacity.
        // Only one push is needed here; the full crossing is tested below.
        let json = serde_json::to_string(&model).unwrap();
        let mut back: PagedVec<i32> = serde_json::from_str(&json).unwrap();
        assert!(back.iter().eq(model.iter()));
        assert_eq!(serde_json::to_string(&back).unwrap(), json);
        push_probe(&mut back);
    }
}

#[cfg(feature = "serde")]
#[test]
fn serde_default_page_crossing() {
    if !cfg!(miri) {
        serde_round_trip::<1024>(&vec![42; 1025]);
    }
}
