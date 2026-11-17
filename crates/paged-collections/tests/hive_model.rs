//! Slot/version and page-stack oracles, stable addresses, and unwind ownership.

#![cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]

use std::cell::{Cell, RefCell};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;
use std::iter::FusedIterator;
use std::num::NonZeroU32;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;

use paged_collections::hive::{Hive, Ids, IntoIter, Iter, IterMut, Values, ValuesMut};
use paged_collections::id_alloc::{Id, MAX_GENERATION, MAX_INDICES};
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
    // Every step checks six iterator families in three directions, on four hives.
    if cfg!(miri) { 12 } else { 64 }
}

fn id(index: u32, generation: u32) -> Id {
    Id::new(index, NonZeroU32::new(generation).unwrap())
}

#[derive(Clone, Debug, Default)]
struct Page {
    versions: Vec<u32>,
    free: Vec<usize>,
    floor: u32,
    allocated: bool,
}

impl Page {
    fn has_room<const PAGE: usize>(&self) -> bool {
        !self.free.is_empty() || self.versions.len() < PAGE
    }

    fn room<const PAGE: usize>(&self) -> usize {
        self.free.len() + PAGE - self.versions.len()
    }
}

#[derive(Clone, Debug, Default)]
struct Model {
    pages: Vec<Page>,
    open: Vec<usize>,
    live: BTreeMap<u32, (Id, i32)>,
    issued: BTreeSet<Id>,
    dead: BTreeSet<Id>,
}

impl Model {
    fn pairs(&self) -> Vec<(Id, i32)> {
        self.live.values().copied().collect()
    }

    fn get(&self, candidate: Id) -> Option<i32> {
        self.live
            .get(&candidate.index())
            .filter(|(current, _)| *current == candidate)
            .map(|(_, value)| *value)
    }

    fn sort_by(&mut self, compare: impl FnMut(&i32, &i32) -> Ordering) {
        let mut values: Vec<_> = self.live.values().map(|&(_, value)| value).collect();
        values.sort_by(compare);
        // Ids name slots: keep each id and all allocation history unchanged,
        // assigning only the stably sorted values back in slot-index order.
        for ((_, value), sorted) in self.live.values_mut().zip(values) {
            *value = sorted;
        }
    }

    fn insert<const PAGE: usize>(&mut self, value: i32) -> Id {
        let p = match self.open.last() {
            Some(&p) => p,
            None => {
                let p = self.pages.len();
                self.pages.push(Page::default());
                self.open.push(p);
                p
            }
        };
        let page = &mut self.pages[p];
        page.allocated = true;
        let offset = if let Some(offset) = page.free.pop() {
            assert_eq!(page.versions[offset] % 2, 0);
            assert_ne!(page.versions[offset], 0);
            page.versions[offset] += 1;
            offset
        } else {
            let offset = page.versions.len();
            page.versions.push(page.floor + 1);
            offset
        };
        let candidate = id((p * PAGE + offset) as u32, page.versions[offset] / 2 + 1);
        if !page.has_room::<PAGE>() {
            assert_eq!(self.open.pop(), Some(p));
        }
        assert!(
            self.issued.insert(candidate),
            "id issued twice: {candidate:?}"
        );
        assert!(!self.dead.contains(&candidate));
        assert!(
            self.live
                .insert(candidate.index(), (candidate, value))
                .is_none()
        );
        candidate
    }

    fn remove<const PAGE: usize>(&mut self, candidate: Id) -> Option<i32> {
        let value = self.get(candidate)?;
        self.live.remove(&candidate.index());
        let index = candidate.index() as usize;
        let (p, offset) = (index / PAGE, index % PAGE);
        let page = &mut self.pages[p];
        let had_room = page.has_room::<PAGE>();
        page.versions[offset] = page.versions[offset].checked_add(1).unwrap_or(0);
        if page.versions[offset] != 0 {
            page.free.push(offset);
        }
        // A page already open keeps its position, even after another removal.
        if !had_room && page.has_room::<PAGE>() {
            self.open.push(p);
        }
        assert!(self.dead.insert(candidate));
        Some(value)
    }

    fn reserve<const PAGE: usize>(&mut self, additional: usize) {
        let mut room: usize = self
            .pages
            .iter()
            .filter(|page| page.allocated)
            .map(Page::room::<PAGE>)
            .sum();
        for page in &mut self.pages {
            if room >= additional {
                return;
            }
            if !page.allocated {
                page.allocated = true;
                room += PAGE;
            }
        }
        let first = self.pages.len();
        while room < additional {
            self.pages.push(Page {
                allocated: true,
                ..Page::default()
            });
            room += PAGE;
        }
        // Existing open pages are used before the new pages, lowest new first.
        self.open.splice(0..0, (first..self.pages.len()).rev());
    }

    fn rebuild_open<const PAGE: usize>(&mut self) {
        self.open.clear();
        for allocated in [false, true] {
            for (p, page) in self.pages.iter().enumerate().rev() {
                if page.allocated == allocated && page.has_room::<PAGE>() {
                    self.open.push(p);
                }
            }
        }
    }

    fn shrink<const PAGE: usize>(&mut self) {
        for page in &mut self.pages {
            if page.allocated && page.versions.iter().all(|&v| v != 0 && v % 2 == 0) {
                let floor = page.versions.iter().copied().fold(page.floor, u32::max);
                *page = Page {
                    floor,
                    ..Page::default()
                };
            }
        }
        while self
            .pages
            .last()
            .is_some_and(|p| !p.allocated && p.floor == 0)
        {
            self.pages.pop();
        }
        self.rebuild_open::<PAGE>();
    }

    #[cfg(feature = "serde")]
    fn after_deserialization<const PAGE: usize>(&self) -> Self {
        let mut model = self.clone();
        // Wire format omits capacity and free-list order. Rebuild lowest first.
        for page in &mut model.pages {
            page.allocated = !page.versions.is_empty();
            page.free = page
                .versions
                .iter()
                .enumerate()
                .rev()
                .filter_map(|(offset, &v)| (v != 0 && v % 2 == 0).then_some(offset))
                .collect();
        }
        model.rebuild_open::<PAGE>();
        model
    }
}

#[derive(Clone, Copy, Debug)]
enum Target {
    Live(u16),
    Stale(u16),
    Never(u32),
}

impl Target {
    fn resolve(self, model: &Model) -> Id {
        match self {
            Self::Live(raw) if !model.live.is_empty() => {
                model.pairs()[usize::from(raw) % model.live.len()].0
            }
            Self::Stale(raw) if !model.dead.is_empty() => *model
                .dead
                .iter()
                .nth(usize::from(raw) % model.dead.len())
                .unwrap(),
            // Generations above MAX_GENERATION can never have been issued.
            Self::Never(index) => id(index, MAX_GENERATION + 1),
            _ => id(MAX_INDICES, 1),
        }
    }
}

fn target() -> impl Strategy<Value = Target> {
    prop_oneof![
        3 => any::<u16>().prop_map(Target::Live),
        2 => any::<u16>().prop_map(Target::Stale),
        1 => any::<u32>().prop_map(Target::Never),
    ]
}

#[derive(Clone, Copy, Debug)]
struct Rule {
    salt: u32,
    divisor: u8,
    delta: i32,
}

impl Rule {
    fn keep(self, candidate: Id, value: i32) -> bool {
        (candidate.index() ^ candidate.generation().get() ^ value as u32 ^ self.salt)
            % u32::from(self.divisor)
            != 0
    }
}

fn rule() -> impl Strategy<Value = Rule> {
    (any::<u32>(), 1u8..7, any::<i32>()).prop_map(|(salt, divisor, delta)| Rule {
        salt,
        divisor,
        delta,
    })
}

#[derive(Clone, Debug)]
enum Op {
    Insert(i32),
    InsertWith(i32),
    Remove(Target),
    Access(Target, i32),
    Occupant(u32),
    Retain(Rule),
    Clear,
    Reserve(u8),
    TryReserve(u8),
    Overflow,
    Shrink,
    Clone,
    Extend(Vec<i32>),
    FromIter(Vec<i32>),
    Mutate(i32),
    Reuse(u8),
    Sort,
    SortBy(u8),
    SortByKey(u8),
    SortedIds(u8),
}

fn operation() -> impl Strategy<Value = Op> {
    prop_oneof![
        17 => prop_oneof![
            5 => any::<i32>().prop_map(Op::Insert),
            3 => any::<i32>().prop_map(Op::InsertWith),
            5 => target().prop_map(Op::Remove),
            3 => (target(), any::<i32>()).prop_map(|(t, x)| Op::Access(t, x)),
            1 => prop_oneof![0u32..64, any::<u32>()].prop_map(Op::Occupant),
        ].boxed(),
        6 => prop_oneof![
            2 => rule().prop_map(Op::Retain),
            1 => Just(Op::Clear),
            1 => (0u8..9).prop_map(Op::Reserve),
            1 => (0u8..9).prop_map(Op::TryReserve),
            1 => Just(Op::Overflow),
        ].boxed(),
        9 => prop_oneof![
            2 => Just(Op::Shrink),
            1 => Just(Op::Clone),
            2 => prop::collection::vec(any::<i32>(), 0..5).prop_map(Op::Extend),
            1 => prop::collection::vec(any::<i32>(), 0..9).prop_map(Op::FromIter),
            1 => any::<i32>().prop_map(Op::Mutate),
            2 => (0u8..9).prop_map(Op::Reuse),
        ].boxed(),
        6 => prop_oneof![
            Just(Op::Sort),
            (1u8..5).prop_map(Op::SortBy),
            (1u8..5).prop_map(Op::SortByKey),
            (1u8..5).prop_map(Op::SortedIds),
        ].boxed(),
    ]
}

fn addresses<T, const PAGE: usize>(hive: &Hive<T, PAGE>) -> Vec<(Id, *const T)> {
    hive.iter()
        .map(|(id, v)| (id, std::ptr::from_ref(v)))
        .collect()
}

fn assert_stable<T, const PAGE: usize>(hive: &Hive<T, PAGE>, before: &[(Id, *const T)]) {
    for &(candidate, address) in before {
        if let Some(value) = hive.get(candidate) {
            assert!(
                std::ptr::eq(value, address),
                "moved {candidate:?}, PAGE={PAGE}"
            );
        }
    }
}

fn check_iter<I, R, F>(mut iter: I, expected: &[R], mode: usize, mut read: F)
where
    I: DoubleEndedIterator + ExactSizeIterator + FusedIterator,
    R: Copy + Debug + PartialEq,
    F: FnMut(I::Item) -> R,
{
    let (mut front, mut back) = (0, expected.len());
    while front < back {
        assert_eq!(iter.len(), back - front);
        assert_eq!(iter.size_hint(), (back - front, Some(back - front)));
        let from_back = match mode {
            0 => false,
            1 => true,
            _ => (front + expected.len() - back) % 5 < 2,
        };
        let (item, value) = if from_back {
            back -= 1;
            (iter.next_back(), expected[back])
        } else {
            let value = expected[front];
            front += 1;
            (iter.next(), value)
        };
        assert_eq!(read(item.expect("iterator ended early")), value);
    }
    for _ in 0..3 {
        assert_eq!(iter.len(), 0);
        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert!(iter.next().is_none());
        assert!(iter.next_back().is_none());
    }
}

fn check<const PAGE: usize>(hive: &mut Hive<i32, PAGE>, model: &Model) {
    assert_eq!(hive.len(), model.live.len());
    assert_eq!(hive.is_empty(), model.live.is_empty());
    assert!(hive.len() <= hive.capacity());
    let allocated = model.pages.iter().filter(|p| p.allocated).count();
    assert_eq!(hive.allocated_pages(), allocated);
    assert_eq!(hive.capacity(), allocated * PAGE);
    assert_eq!(hive.index_bound(), model.pages.len() * PAGE);
    let expected = model.pairs();
    for &(candidate, value) in &expected {
        assert!((candidate.index() as usize) < hive.index_bound());
        assert!(hive.contains(candidate));
        assert_eq!(hive.get(candidate), Some(&value));
        assert_eq!(hive.get_mut(candidate).copied(), Some(value));
        assert_eq!(hive[candidate], value);
    }
    for index in 0..hive.index_bound() {
        assert_eq!(
            hive.occupant(index as u32),
            model.live.get(&(index as u32)).map(|&(id, _)| id)
        );
    }
    assert_eq!(hive.occupant(hive.index_bound() as u32), None);
    assert_eq!(hive.occupant(MAX_INDICES), None);
    for &stale in &model.dead {
        assert!(!hive.contains(stale), "stale id revived: {stale:?}");
        assert_eq!(hive.get(stale), None);
        assert_eq!(hive.get_mut(stale), None);
    }
    let ids: Vec<_> = expected.iter().map(|&(id, _)| id).collect();
    let values: Vec<_> = expected.iter().map(|&(_, v)| v).collect();
    for mode in 0..3 {
        let iter: Iter<'_, i32> = hive.iter();
        check_iter(iter, &expected, mode, |(id, v)| (id, *v));
        let iter: IterMut<'_, i32> = hive.iter_mut();
        check_iter(iter, &expected, mode, |(id, v)| (id, *v));
        let iter: Ids<'_, i32> = hive.ids();
        check_iter(iter, &ids, mode, |id| id);
        let iter: Values<'_, i32> = hive.values();
        check_iter(iter, &values, mode, |v| *v);
        let iter: ValuesMut<'_, i32> = hive.values_mut();
        check_iter(iter, &values, mode, |v| *v);
        let iter: IntoIter<i32> = hive.clone().into_iter();
        check_iter(iter, &expected, mode, |pair| pair);
    }
    check_iter((&*hive).into_iter(), &expected, 2, |(id, v)| (id, *v));
    check_iter(hive.into_iter(), &expected, 2, |(id, v)| (id, *v));
}

struct Machine<const PAGE: usize> {
    hive: Hive<i32, PAGE>,
    model: Model,
}

impl<const PAGE: usize> Machine<PAGE> {
    fn new(capacity: usize) -> Self {
        let mut model = Model::default();
        model.reserve::<PAGE>(capacity);
        let mut machine = Self {
            hive: Hive::with_capacity(capacity),
            model,
        };
        assert_eq!(machine.hive, Hive::<i32, PAGE>::new());
        assert_eq!(machine.hive, Hive::<i32, PAGE>::default());
        check(&mut machine.hive, &machine.model);
        machine
    }

    fn insert(&mut self, value: i32, with: bool) {
        let expected = self.model.insert::<PAGE>(value);
        let actual = if with {
            let mut calls = 0;
            let actual = self.hive.insert_with(|candidate| {
                calls += 1;
                assert_eq!(candidate, expected);
                value
            });
            assert_eq!(calls, 1);
            actual
        } else {
            self.hive.insert(value)
        };
        assert_eq!(actual, expected);
    }

    fn step(&mut self, op: &Op) {
        let before = addresses(&self.hive);
        match *op {
            Op::Insert(value) => self.insert(value, false),
            Op::InsertWith(value) => self.insert(value, true),
            Op::Remove(target) => {
                let candidate = target.resolve(&self.model);
                assert_eq!(
                    self.hive.remove(candidate),
                    self.model.remove::<PAGE>(candidate)
                );
            }
            Op::Access(target, value) => {
                let candidate = target.resolve(&self.model);
                let expected = self.model.get(candidate);
                assert_eq!(self.hive.get(candidate).copied(), expected);
                assert_eq!(self.hive.get_mut(candidate).copied(), expected);
                assert_eq!(self.hive.contains(candidate), expected.is_some());
                if let Some(old) = expected {
                    assert_eq!(self.hive[candidate], old);
                    *self.hive.get_mut(candidate).unwrap() = value;
                    assert_eq!(self.hive[candidate], value);
                    self.hive[candidate] = value.wrapping_add(1);
                    self.model.live.get_mut(&candidate.index()).unwrap().1 = value.wrapping_add(1);
                } else {
                    assert!(catch_unwind(|| self.hive[candidate]).is_err());
                    assert!(
                        catch_unwind(AssertUnwindSafe(|| self.hive[candidate] = value)).is_err()
                    );
                }
            }
            Op::Occupant(index) => {
                assert_eq!(
                    self.hive.occupant(index),
                    self.model.live.get(&index).map(|x| x.0)
                );
            }
            Op::Retain(rule) => {
                let expected = self.model.pairs();
                let mut visits = Vec::new();
                self.hive.retain(|candidate, value| {
                    visits.push((candidate, *value));
                    *value = value.wrapping_add(rule.delta);
                    rule.keep(candidate, *value)
                });
                assert_eq!(visits, expected);
                for (candidate, value) in expected {
                    let value = value.wrapping_add(rule.delta);
                    if rule.keep(candidate, value) {
                        self.model.live.get_mut(&candidate.index()).unwrap().1 = value;
                    } else {
                        assert!(self.model.remove::<PAGE>(candidate).is_some());
                    }
                }
            }
            Op::Clear => {
                self.hive.clear();
                for (candidate, _) in self.model.pairs() {
                    assert!(self.model.remove::<PAGE>(candidate).is_some());
                }
            }
            Op::Reserve(n) | Op::TryReserve(n) => {
                if matches!(op, Op::Reserve(_)) {
                    self.hive.reserve(usize::from(n));
                } else {
                    self.hive.try_reserve(usize::from(n)).unwrap();
                }
                self.model.reserve::<PAGE>(usize::from(n));
            }
            Op::Overflow => {
                let pairs = self.model.pairs();
                let layout = (
                    self.hive.capacity(),
                    self.hive.allocated_pages(),
                    self.hive.index_bound(),
                );
                assert!(self.hive.try_reserve(usize::MAX).is_err());
                assert_eq!(
                    self.hive.iter().map(|(id, v)| (id, *v)).collect::<Vec<_>>(),
                    pairs
                );
                assert_eq!(
                    (
                        self.hive.capacity(),
                        self.hive.allocated_pages(),
                        self.hive.index_bound()
                    ),
                    layout
                );
                // Capacity and the future allocation history both stay unchanged.
                let mut probe = self.hive.clone();
                let mut oracle = self.model.clone();
                assert_eq!(probe.insert(17), oracle.insert::<PAGE>(17));
            }
            Op::Shrink => {
                self.hive.shrink_to_fit();
                self.model.shrink::<PAGE>();
                let live_pages: BTreeSet<_> =
                    self.model.live.keys().map(|&i| i as usize / PAGE).collect();
                assert_eq!(self.hive.allocated_pages(), live_pages.len());
            }
            Op::Clone => {
                let mut cloned = self.hive.clone();
                let mut oracle = self.model.clone();
                assert_eq!(cloned, self.hive);
                check(&mut cloned, &oracle);
                assert_eq!(cloned.insert(99), oracle.insert::<PAGE>(99));
                check(&mut cloned, &oracle);
                assert_ne!(cloned, self.hive);
            }
            Op::Extend(ref values) => {
                self.hive.extend(values.iter().copied());
                for &value in values {
                    self.model.insert::<PAGE>(value);
                }
            }
            Op::FromIter(ref values) => {
                // A fresh history is checked separately, never restored over this one.
                let mut collected: Hive<i32, PAGE> = values.iter().copied().collect();
                let mut oracle = Model::default();
                for &value in values {
                    oracle.insert::<PAGE>(value);
                }
                check(&mut collected, &oracle);
                assert_eq!(collected == self.hive, oracle.pairs() == self.model.pairs());
            }
            Op::Mutate(delta) => {
                for (candidate, value) in &mut self.hive {
                    *value = value
                        .wrapping_add(delta)
                        .wrapping_add(candidate.index() as i32);
                }
                for value in self.hive.values_mut().rev() {
                    *value = value.wrapping_sub(1);
                }
                for (candidate, value) in self.model.live.values_mut() {
                    *value = value
                        .wrapping_add(delta)
                        .wrapping_add(candidate.index() as i32)
                        .wrapping_sub(1);
                }
            }
            Op::Sort => {
                self.hive.sort();
                self.model.sort_by(i32::cmp);
            }
            Op::SortBy(divisor) => {
                // Small key spaces make unequal values compare equal, so
                // stability is observable even though the payload is i32.
                let key = |value: &i32| value.rem_euclid(i32::from(divisor));
                self.hive.sort_by(|a, b| key(b).cmp(&key(a)));
                self.model.sort_by(|a, b| key(b).cmp(&key(a)));
            }
            Op::SortByKey(divisor) => {
                let key = |value: &i32| value.rem_euclid(i32::from(divisor));
                self.hive.sort_by_key(key);
                self.model.sort_by(|a, b| key(a).cmp(&key(b)));
            }
            Op::SortedIds(divisor) => {
                let mut pairs = self.model.pairs();
                pairs.sort_by_key(|&(_, value)| value);
                let expected: Vec<_> = pairs.iter().map(|&(id, _)| id).collect();
                assert_eq!(self.hive.sorted_ids(), expected);

                let key = |value: &i32| value.rem_euclid(i32::from(divisor));
                // Start again in slot order: ties retain that order, not the
                // value order used by the sorted_ids oracle above.
                let mut pairs = self.model.pairs();
                pairs.sort_by_key(|&(_, value)| key(&value));
                let expected: Vec<_> = pairs.iter().map(|&(id, _)| id).collect();
                assert_eq!(self.hive.sorted_ids_by_key(key), expected);
                assert_eq!(
                    self.hive.sorted_ids_by(|a, b| key(a).cmp(&key(b))),
                    expected
                );
                // The unchanged model is checked below: no value may move.
            }
            Op::Reuse(raw) => {
                let removed: Vec<_> = self
                    .model
                    .pairs()
                    .into_iter()
                    .take(usize::from(raw))
                    .collect();
                let bound = self.hive.index_bound();
                for &(candidate, value) in &removed {
                    assert_eq!(self.hive.remove(candidate), Some(value));
                    self.model.remove::<PAGE>(candidate).unwrap();
                    check(&mut self.hive, &self.model);
                }
                for _ in removed {
                    self.insert(41, true);
                    assert_eq!(self.hive.index_bound(), bound);
                    check(&mut self.hive, &self.model);
                }
            }
        }
        // This compares addresses by id (slot), not by value. Sorting moves
        // values between those addresses while preserving the occupied slots.
        assert_stable(&self.hive, &before);
        check(&mut self.hive, &self.model);
    }

    fn run(&mut self, operations: &[Op]) {
        for op in operations {
            self.step(op);
        }
    }
}

fn reuse_order<const PAGE: usize>() {
    let mut machine = Machine::<PAGE>::new(0);
    for i in 0..3 * PAGE {
        machine.insert(i as i32, false);
    }
    let original = machine.model.pairs();
    // Open page 0, then page 1; a second free slot in page 0 does not move
    // that already-open page above page 1. Offsets within a page are LIFO.
    let mut removed = vec![original[0].0, original[PAGE].0];
    if PAGE > 1 {
        removed.push(original[PAGE - 1].0);
    }
    for &candidate in &removed {
        machine.step(&Op::Remove(Target::Live(
            machine
                .model
                .pairs()
                .iter()
                .position(|x| x.0 == candidate)
                .unwrap() as u16,
        )));
    }
    let bound = machine.hive.index_bound();
    let mut expected = vec![id(PAGE as u32, 2)];
    if PAGE > 1 {
        expected.push(id((PAGE - 1) as u32, 2));
    }
    expected.push(id(0, 2));
    for candidate in expected {
        let actual = machine.hive.insert(90);
        assert_eq!(actual, candidate);
        assert_eq!(actual, machine.model.insert::<PAGE>(90));
        assert_eq!(machine.hive.index_bound(), bound);
        check(&mut machine.hive, &machine.model);
    }
    // Reservations put fresh pages below an existing page with a free slot.
    let victim = machine.model.pairs()[0].0;
    machine.hive.remove(victim).unwrap();
    machine.model.remove::<PAGE>(victim).unwrap();
    machine.step(&Op::Reserve((2 * PAGE + 1) as u8));
    let allocated = machine.hive.allocated_pages();
    assert_eq!(machine.hive.insert(91), id(victim.index(), 3));
    assert_eq!(machine.model.insert::<PAGE>(91), id(victim.index(), 3));
    for _ in 0..2 * PAGE {
        machine.insert(92, false);
        assert_eq!(machine.hive.allocated_pages(), allocated);
    }
    check(&mut machine.hive, &machine.model);
}

#[test]
fn page_preference_and_per_page_lifo_reuse() {
    reuse_order::<1>();
    reuse_order::<2>();
    reuse_order::<3>();
    reuse_order::<4>();
}

fn shrink_floor<const PAGE: usize>() {
    let mut machine = Machine::<PAGE>::new(0);
    for _ in 0..2 * PAGE {
        machine.insert(10, false);
    }
    let original = machine.model.pairs();
    // Raise just one slot; releasing the whole page must raise every
    // recreated offset to this page's maximum.
    let mut hot = original[PAGE - 1].0;
    for _ in 0..3 {
        machine.hive.remove(hot).unwrap();
        machine.model.remove::<PAGE>(hot).unwrap();
        hot = machine.hive.insert(11);
        assert_eq!(hot, machine.model.insert::<PAGE>(11));
    }
    for (candidate, _) in machine.model.pairs().into_iter().take(PAGE) {
        machine.hive.remove(candidate).unwrap();
        machine.model.remove::<PAGE>(candidate).unwrap();
    }
    let stable = addresses(&machine.hive);
    machine.step(&Op::Shrink);
    assert_eq!(machine.hive.allocated_pages(), 1);
    for offset in 0..PAGE {
        let candidate = machine.hive.insert(12);
        assert_eq!(candidate, id(offset as u32, 5));
        assert_eq!(candidate, machine.model.insert::<PAGE>(12));
        assert_stable(&machine.hive, &stable);
    }
    machine.step(&Op::Clear);
    machine.step(&Op::Shrink);
    assert_eq!(machine.hive.allocated_pages(), 0);
    let bound = machine.hive.index_bound();
    for _ in 0..2 * PAGE {
        machine.insert(13, true);
        assert_eq!(machine.hive.index_bound(), bound);
    }
    check(&mut machine.hive, &machine.model);
}

#[test]
fn shrink_remembers_highest_generation_and_reuses_empty_pages() {
    shrink_floor::<1>();
    shrink_floor::<2>();
    shrink_floor::<3>();
    shrink_floor::<4>();
}

#[test]
fn equality_uses_live_ids_and_values_across_page_sizes() {
    let mut a: Hive<i32, 1> = [7, 7, 7, 7].into_iter().collect();
    let mut b: Hive<i32, 3> = [7, 7, 7, 7].into_iter().collect();
    let default: Hive<i32> = [7, 7, 7, 7].into_iter().collect();
    assert_eq!(a, b);
    assert_eq!(a, default);
    let old = id(1, 1);
    assert_eq!(a.remove(old), Some(7));
    a.insert(7);
    assert_ne!(a, b); // Same indices and values, different live generation.
    assert_eq!(b.remove(old), Some(7));
    b.insert(7);
    assert_eq!(a, b);
    b[id(1, 2)] = 8;
    assert_ne!(a, b);
    b[id(1, 2)] = 7;
    b.reserve(20);
    b.shrink_to_fit();
    assert_eq!(a, b);
    assert_eq!(a, a.clone());
}

fn insert_panic<const PAGE: usize>(operations: &[Op]) {
    let mut machine = sequence::<PAGE>(&[], operations);
    let pairs = machine.model.pairs();
    let before = addresses(&machine.hive);
    let mut future = machine.model.clone();
    let expected = future.insert::<PAGE>(123);
    let received = Cell::new(None);
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            machine.hive.insert_with(|candidate| {
                received.set(Some(candidate));
                panic!("insert factory bomb");
            });
        }))
        .is_err()
    );
    assert_eq!(received.get(), Some(expected));
    assert_eq!(machine.hive.len(), pairs.len());
    assert_eq!(
        machine
            .hive
            .iter()
            .map(|(id, v)| (id, *v))
            .collect::<Vec<_>>(),
        pairs
    );
    assert_stable(&machine.hive, &before);
    // A factory panic may allocate capacity, but cannot consume an id.
    machine.insert(123, true);
    assert_stable(&machine.hive, &before);
    check(&mut machine.hive, &machine.model);
    machine.step(&Op::Clear);
    machine.step(&Op::Shrink);
    machine.insert(124, false);
    check(&mut machine.hive, &machine.model);
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn insert_with_panic_preserves_values_and_next_id(
        operations in prop::collection::vec(operation(), 0..max_len() / 3),
    ) {
        insert_panic::<1>(&operations);
        insert_panic::<2>(&operations);
        insert_panic::<3>(&operations);
        insert_panic::<4>(&operations);
    }
}

#[test]
fn insert_with_panic_on_fresh_full_vacant_and_released_pages() {
    for operations in [
        vec![],
        vec![Op::Extend(vec![1, 2, 3, 4])],
        vec![Op::Insert(1), Op::Remove(Target::Live(0))],
        vec![Op::Insert(1), Op::Clear, Op::Shrink],
    ] {
        insert_panic::<1>(&operations);
        insert_panic::<2>(&operations);
        insert_panic::<3>(&operations);
        insert_panic::<4>(&operations);
    }
}

fn invalid_ids<const PAGE: usize>() {
    let mut machine = sequence::<PAGE>(&[10, 20, 30], &[]);
    let original = machine.model.pairs()[0].0;
    machine.step(&Op::Remove(Target::Live(0)));
    machine.step(&Op::Insert(40));
    let before = addresses(&machine.hive);
    for candidate in [
        original,
        id(1, 2), // Valid generation, but never issued at this live index.
        id(MAX_INDICES - 1, 1),
        id(MAX_INDICES, 1),
        id(0, MAX_GENERATION),
        id(0, MAX_GENERATION + 1),
        id(0, u32::MAX),
    ] {
        assert!(!machine.hive.contains(candidate));
        assert_eq!(machine.hive.get(candidate), None);
        assert_eq!(machine.hive.get_mut(candidate), None);
        assert_eq!(machine.hive.remove(candidate), None);
        assert!(catch_unwind(|| machine.hive[candidate]).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| machine.hive[candidate] = 99)).is_err());
        check(&mut machine.hive, &machine.model);
        assert_stable(&machine.hive, &before);
    }
}

#[test]
fn stale_and_never_issued_ids_reject_access_remove_and_indexing() {
    invalid_ids::<1>();
    invalid_ids::<2>();
    invalid_ids::<3>();
    invalid_ids::<4>();
    if !cfg!(miri) {
        invalid_ids::<{ Hive::<i32>::PAGE }>();
    }
}

#[derive(Default)]
struct Ledger {
    drops: RefCell<Vec<usize>>,
    panic_drop: Cell<Option<usize>>,
}

impl Ledger {
    fn make(self: &Rc<Self>, value: i32) -> Item {
        let mut drops = self.drops.borrow_mut();
        let serial = drops.len();
        drops.push(0);
        Item {
            serial,
            value,
            ledger: Rc::clone(self),
        }
    }

    fn check_live(&self, live: impl IntoIterator<Item = usize>) {
        let live: Vec<_> = live.into_iter().collect();
        let unique: BTreeSet<_> = live.iter().copied().collect();
        assert_eq!(live.len(), unique.len(), "duplicate live ownership");
        for (serial, &drops) in self.drops.borrow().iter().enumerate() {
            assert_eq!(
                drops,
                usize::from(!unique.contains(&serial)),
                "serial={serial}"
            );
        }
    }
}

struct Item {
    serial: usize,
    value: i32,
    ledger: Rc<Ledger>,
}

impl Drop for Item {
    fn drop(&mut self) {
        self.ledger.drops.borrow_mut()[self.serial] += 1;
        if self.ledger.panic_drop.get() == Some(self.serial) {
            self.ledger.panic_drop.set(None);
            panic!("drop bomb");
        }
    }
}

fn insert_item<const PAGE: usize>(
    hive: &mut Hive<Item, PAGE>,
    model: &mut Model,
    ledger: &Rc<Ledger>,
    serials: &mut BTreeMap<Id, usize>,
    value: i32,
    with: bool,
) -> Id {
    let expected = model.insert::<PAGE>(value);
    let item = ledger.make(value);
    let serial = item.serial;
    let actual = if with {
        hive.insert_with(|candidate| {
            assert_eq!(candidate, expected);
            item
        })
    } else {
        hive.insert(item)
    };
    assert_eq!(actual, expected);
    assert!(serials.insert(actual, serial).is_none());
    actual
}

fn check_items<const PAGE: usize>(
    hive: &mut Hive<Item, PAGE>,
    model: &Model,
    ledger: &Ledger,
    serials: &BTreeMap<Id, usize>,
) {
    let expected: Vec<_> = model
        .pairs()
        .into_iter()
        .map(|(id, v)| (id, serials[&id], v))
        .collect();
    assert_eq!(hive.len(), expected.len());
    assert_eq!(hive.is_empty(), expected.is_empty());
    assert!(hive.len() <= hive.capacity());
    for mode in 0..3 {
        check_iter(hive.iter(), &expected, mode, |(id, v)| {
            (id, v.serial, v.value)
        });
        check_iter(hive.iter_mut(), &expected, mode, |(id, v)| {
            (id, v.serial, v.value)
        });
    }
    for &(candidate, serial, value) in &expected {
        assert!(hive.contains(candidate));
        assert_eq!(hive.occupant(candidate.index()), Some(candidate));
        assert_eq!(
            (hive[candidate].serial, hive[candidate].value),
            (serial, value)
        );
    }
    for &candidate in &model.dead {
        assert!(!hive.contains(candidate));
    }
    ledger.check_live(expected.iter().map(|x| x.1));
}

fn drop_sequence<const PAGE: usize>(operations: &[(u8, u16, i32)]) {
    let ledger = Rc::new(Ledger::default());
    let mut hive = Hive::<Item, PAGE>::new();
    let mut model = Model::default();
    let mut serials = BTreeMap::new();
    for &(code, raw, value) in operations {
        let before = addresses(&hive);
        match code {
            0 | 1 => {
                insert_item(
                    &mut hive,
                    &mut model,
                    &ledger,
                    &mut serials,
                    value,
                    code == 1,
                );
            }
            2 if !model.live.is_empty() => {
                let (candidate, expected) = model.pairs()[usize::from(raw) % model.live.len()];
                let removed = hive.remove(candidate).unwrap();
                assert_eq!(
                    (removed.serial, removed.value),
                    (serials[&candidate], expected)
                );
                assert_eq!(model.remove::<PAGE>(candidate), Some(expected));
                drop(removed);
            }
            3 => {
                let rule = Rule {
                    salt: u32::from(raw),
                    divisor: 3,
                    delta: value,
                };
                hive.retain(|candidate, item| {
                    item.value = item.value.wrapping_add(rule.delta);
                    rule.keep(candidate, item.value)
                });
                for (candidate, old) in model.pairs() {
                    let new = old.wrapping_add(value);
                    if rule.keep(candidate, new) {
                        model.live.get_mut(&candidate.index()).unwrap().1 = new;
                    } else {
                        model.remove::<PAGE>(candidate).unwrap();
                    }
                }
            }
            4 => {
                hive.clear();
                for (candidate, _) in model.pairs() {
                    model.remove::<PAGE>(candidate).unwrap();
                }
            }
            5 => {
                hive.shrink_to_fit();
                model.shrink::<PAGE>();
            }
            6 => {
                let n = usize::from(raw % 7);
                hive.reserve(n);
                model.reserve::<PAGE>(n);
            }
            _ => {}
        }
        assert_stable(&hive, &before);
        check_items(&mut hive, &model, &ledger, &serials);
    }
    // Dropping the owning iterator must drop unconsumed elements in both
    // partially visited end pages and every untouched page between them.
    let expected: Vec<_> = model
        .pairs()
        .into_iter()
        .map(|(id, v)| (id, serials[&id], v))
        .collect();
    let mut iter = hive.into_iter();
    let (mut front, mut back) = (0, expected.len());
    let steps = operations
        .last()
        .map_or(0, |x| usize::from(x.1) % (expected.len() + 2));
    for step in 0..steps {
        assert_eq!(iter.len(), back - front);
        let (actual, wanted) = if front == back {
            (iter.next(), None)
        } else if step % 3 == 0 {
            back -= 1;
            (iter.next_back(), Some(expected[back]))
        } else {
            let wanted = expected[front];
            front += 1;
            (iter.next(), Some(wanted))
        };
        assert_eq!(
            actual.as_ref().map(|(id, v)| (*id, v.serial, v.value)),
            wanted
        );
        drop(actual);
        ledger.check_live(expected[front..back].iter().map(|x| x.1));
    }
    drop(iter);
    ledger.check_live([]);
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn every_value_has_one_owner_and_one_drop(
        operations in prop::collection::vec((0u8..7, any::<u16>(), any::<i32>()), 0..max_len()),
    ) {
        drop_sequence::<1>(&operations);
        drop_sequence::<2>(&operations);
        drop_sequence::<3>(&operations);
        drop_sequence::<4>(&operations);
    }
}

fn same_pairs<const A: usize, const B: usize>(left: &Machine<A>, right: &Machine<B>) {
    assert_eq!(
        left.hive == right.hive,
        left.model.pairs() == right.model.pairs()
    );
    assert_eq!(
        right.hive == left.hive,
        right.model.pairs() == left.model.pairs()
    );
}

fn sequence<const PAGE: usize>(initial: &[i32], operations: &[Op]) -> Machine<PAGE> {
    let mut machine = Machine::new(initial.len() + 2);
    machine.step(&Op::Extend(initial.to_vec()));
    machine.run(operations);
    machine
}

fn sorts_with_holes<const PAGE: usize>() {
    let mut machine = sequence::<PAGE>(
        &[5, 2, 9, 1, 6, 3, 10, 4],
        &[Op::Remove(Target::Live(1)), Op::Remove(Target::Live(3))],
    );
    let ids: Vec<_> = machine.hive.ids().collect();
    machine.step(&Op::SortBy(3));
    assert_eq!(
        machine.hive.values().copied().collect::<Vec<_>>(),
        [5, 1, 10, 4, 9, 3]
    );
    machine.step(&Op::SortedIds(3));
    machine.step(&Op::SortByKey(3));
    assert_eq!(
        machine.hive.values().copied().collect::<Vec<_>>(),
        [9, 3, 1, 10, 4, 5]
    );
    machine.step(&Op::Sort);
    assert_eq!(machine.hive.ids().collect::<Vec<_>>(), ids);
    // Follow sorting with reuse and another sort, checking that neither
    // generations nor the free-list history were reordered with the values.
    machine.step(&Op::Reuse(2));
    machine.step(&Op::SortBy(1)); // Every comparison is equal.
    machine.step(&Op::SortedIds(1));
}

#[test]
fn stable_sorts_match_model_with_holes_and_reuse() {
    sorts_with_holes::<1>();
    sorts_with_holes::<2>();
    sorts_with_holes::<3>();
    sorts_with_holes::<4>();
}

fn reserve_probe<const PAGE: usize>(operations: &[Op], n: usize, fallible: bool) {
    let mut machine = sequence::<PAGE>(&[3, 4, 5], operations);
    let before = addresses(&machine.hive);
    if fallible {
        machine.hive.try_reserve(n).unwrap();
    } else {
        machine.hive.reserve(n);
    }
    machine.model.reserve::<PAGE>(n);
    let allocated = machine.hive.allocated_pages();
    check(&mut machine.hive, &machine.model);
    for i in 0..n {
        machine.insert(i as i32, i % 2 == 0);
        assert_eq!(machine.hive.allocated_pages(), allocated);
        assert_stable(&machine.hive, &before);
        check(&mut machine.hive, &machine.model);
    }
}

// Probe every old slot, including free slots on pages with no live elements.
// Equality alone does not compare free lists or vacant-slot generations.
fn compare_future_ids<const PAGE: usize>(machine: &mut Machine<PAGE>, twin: &mut Hive<i32, PAGE>) {
    check(&mut machine.hive, &machine.model);
    check(twin, &machine.model);
    let stable = addresses(&machine.hive);
    let probes = machine.hive.index_bound() + 1;
    for i in 0..probes {
        let value = i as i32;
        let expected = twin.insert(value);
        assert_eq!(machine.hive.insert(value), expected);
        assert_eq!(machine.model.insert::<PAGE>(value), expected);
    }
    assert_eq!(machine.hive, *twin);
    assert_stable(&machine.hive, &stable);
    check(&mut machine.hive, &machine.model);
    check(twin, &machine.model);
}

fn drain_prefix<const PAGE: usize>(initial: &[i32], operations: &[Op], raw: u16, leak: bool) {
    let mut machine = sequence::<PAGE>(initial, operations);
    let mut twin = machine.hive.clone();
    let expected = machine.model.pairs();
    let stable = addresses(&machine.hive);
    let k = usize::from(raw) % (expected.len() + 1);
    let mut drain = machine.hive.drain();
    let mut yielded = Vec::new();
    for (i, &pair) in expected.iter().take(k).enumerate() {
        let left = expected.len() - i;
        assert_eq!(drain.len(), left);
        assert_eq!(drain.size_hint(), (left, Some(left)));
        let actual = drain.next().unwrap();
        assert_eq!(actual, pair);
        yielded.push(actual);
    }
    let left = expected.len() - k;
    assert_eq!(drain.len(), left);
    assert_eq!(drain.size_hint(), (left, Some(left)));
    if left == 0 {
        assert_eq!(drain.next(), None);
        assert_eq!(drain.next(), None);
    }
    assert_eq!(yielded, expected[..k]);
    if leak {
        std::mem::forget(drain);
        for &(candidate, value) in &yielded {
            assert_eq!(twin.remove(candidate), Some(value));
            assert_eq!(machine.model.remove::<PAGE>(candidate), Some(value));
        }
        assert_eq!(machine.model.pairs(), expected[k..]);
    } else {
        drop(drain);
        twin.clear();
        for (candidate, value) in expected {
            assert_eq!(machine.model.remove::<PAGE>(candidate), Some(value));
        }
        assert!(machine.hive.is_empty());
    }
    assert_stable(&machine.hive, &stable);
    compare_future_ids(&mut machine, &mut twin);
}

fn drain_drops<const PAGE: usize>(values: &[i32], raw: u16) {
    let ledger = Rc::new(Ledger::default());
    let mut hive = Hive::<Item, PAGE>::new();
    let mut model = Model::default();
    let mut serials = BTreeMap::new();
    for &value in values {
        insert_item(&mut hive, &mut model, &ledger, &mut serials, value, false);
    }
    // Include holes and values already dropped before the drain starts.
    for (candidate, _) in model.pairs().into_iter().skip(1).step_by(3) {
        drop(hive.remove(candidate));
        model.remove::<PAGE>(candidate).unwrap();
    }
    let expected = model.pairs();
    let k = usize::from(raw) % (expected.len() + 1);
    let mut drain = hive.drain();
    let mut yielded = Vec::new();
    for &(candidate, value) in expected.iter().take(k) {
        let (actual, item) = drain.next().unwrap();
        assert_eq!(
            (actual, item.serial, item.value),
            (candidate, serials[&candidate], value)
        );
        yielded.push(item);
    }
    ledger.check_live(expected.iter().map(|&(candidate, _)| serials[&candidate]));
    drop(drain);
    assert!(hive.is_empty());
    ledger.check_live(yielded.iter().map(|item| item.serial));
    drop(yielded);
    for (candidate, _) in expected {
        model.remove::<PAGE>(candidate).unwrap();
    }
    check_items(&mut hive, &model, &ledger, &serials);
    drop(hive);
    ledger.check_live([]);
}

fn extract_prefix<const PAGE: usize>(initial: &[i32], operations: &[Op], rule: Rule, raw: u16) {
    let mut machine = sequence::<PAGE>(initial, operations);
    let mut twin = machine.hive.clone();
    let before = machine.model.pairs();
    let stable = addresses(&machine.hive);
    // One extra next() can exhaust the scan even when no value matches.
    let k = usize::from(raw) % (before.len() + 2);
    let mut expected = Vec::new();
    let mut expected_visits = Vec::new();
    for &(candidate, old) in &before {
        if expected.len() == k {
            break;
        }
        expected_visits.push((candidate, old));
        let value = old.wrapping_add(rule.delta);
        twin[candidate] = value;
        machine.model.live.get_mut(&candidate.index()).unwrap().1 = value;
        if !rule.keep(candidate, value) {
            expected.push((candidate, value));
            assert_eq!(twin.remove(candidate), Some(value));
            assert_eq!(machine.model.remove::<PAGE>(candidate), Some(value));
        }
    }
    let mut visits = Vec::new();
    let mut yielded = Vec::new();
    {
        let mut extract = machine.hive.extract_if(|candidate, value| {
            visits.push((candidate, *value));
            *value = value.wrapping_add(rule.delta);
            !rule.keep(candidate, *value)
        });
        for _ in 0..k {
            match extract.next() {
                Some(pair) => yielded.push(pair),
                None => {
                    assert_eq!(extract.next(), None);
                    break;
                }
            }
        }
    }
    assert_eq!(yielded, expected);
    assert_eq!(visits, expected_visits);
    assert_eq!(visits, before[..visits.len()]);
    assert_stable(&machine.hive, &stable);
    compare_future_ids(&mut machine, &mut twin);
}

fn extract_panic<const PAGE: usize>(rule: Rule, raw: u16) {
    let ledger = Rc::new(Ledger::default());
    let mut hive = Hive::<Item, PAGE>::new();
    let mut model = Model::default();
    let mut serials = BTreeMap::new();
    for i in 0..2 * PAGE + 5 {
        insert_item(
            &mut hive,
            &mut model,
            &ledger,
            &mut serials,
            i as i32,
            false,
        );
    }
    for index in [1, PAGE + 1] {
        let candidate = id(index as u32, 1);
        drop(hive.remove(candidate));
        model.remove::<PAGE>(candidate).unwrap();
    }
    let before = model.pairs();
    let stable = addresses(&hive);
    let stop = usize::from(raw) % before.len();
    let mut visits = Vec::new();
    let mut yielded = Vec::new();
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let extract = hive.extract_if(|candidate, item| {
                let turn = visits.len();
                visits.push((candidate, item.value));
                item.value = item.value.wrapping_add(rule.delta);
                assert_ne!(turn, stop, "extract predicate bomb");
                turn == 0 || !rule.keep(candidate, item.value)
            });
            for (candidate, item) in extract {
                yielded.push((candidate, item.value));
                drop(item);
            }
        }))
        .is_err()
    );
    assert_eq!(visits, before[..=stop]);
    let mut expected = Vec::new();
    for (turn, &(candidate, old)) in before.iter().enumerate().take(stop + 1) {
        let value = old.wrapping_add(rule.delta);
        // The panicking call mutates its value but never returns a decision.
        if turn == stop || (turn != 0 && rule.keep(candidate, value)) {
            model.live.get_mut(&candidate.index()).unwrap().1 = value;
        } else {
            expected.push((candidate, value));
            model.remove::<PAGE>(candidate).unwrap();
        }
    }
    assert_eq!(yielded, expected);
    assert_stable(&hive, &stable);
    check_items(&mut hive, &model, &ledger, &serials);
    // Exercise the free lists left behind by completed extractions.
    for i in 0..=before.len() {
        insert_item(&mut hive, &mut model, &ledger, &mut serials, i as i32, true);
    }
    assert_stable(&hive, &stable);
    check_items(&mut hive, &model, &ledger, &serials);
    drop(hive);
    ledger.check_live([]);
}

fn disjoint_batch<const PAGE: usize, const N: usize>(
    machine: &mut Machine<PAGE>,
    ids: [Id; N],
    marker: i32,
) {
    let stable = addresses(&machine.hive);
    let expected = ids.map(|candidate| machine.model.get(candidate));
    let got = machine.hive.get_disjoint_mut(ids);
    for (i, value) in got.into_iter().enumerate() {
        assert_eq!(value.as_deref().copied(), expected[i]);
        if let Some(value) = value {
            let new = marker.wrapping_add(i as i32);
            *value = new;
            machine.model.live.get_mut(&ids[i].index()).unwrap().1 = new;
        }
    }
    assert_stable(&machine.hive, &stable);
    check(&mut machine.hive, &machine.model);
}

fn disjoint_model<const PAGE: usize>(operations: &[Op], targets: &[Target], marker: i32) {
    let mut machine = sequence::<PAGE>(&[10, 20, 30, 40], operations);
    // Guarantee both live and stale choices even if the sequence cleared it.
    machine.insert(-1, false);
    let stale = machine.model.pairs().last().unwrap().0;
    assert_eq!(
        machine.hive.remove(stale),
        machine.model.remove::<PAGE>(stale)
    );
    for value in [10, 20, 30, 40] {
        machine.insert(value, false);
    }
    let mut twin = machine.hive.clone();
    let mut selected = BTreeSet::new();
    let ids: [Id; 4] = std::array::from_fn(|i| {
        let candidate = targets[i].resolve(&machine.model);
        if machine.model.get(candidate).is_some() && !selected.insert(candidate) {
            Target::Never(candidate.index()).resolve(&machine.model)
        } else {
            candidate
        }
    });
    disjoint_batch(&mut machine, [], marker);
    disjoint_batch(&mut machine, [ids[0]], marker);
    disjoint_batch(&mut machine, [ids[0], ids[1]], marker);
    disjoint_batch(&mut machine, [ids[0], ids[1], ids[2]], marker);
    disjoint_batch(&mut machine, ids, marker);
    // Force input order to differ from slot order and mix all three kinds.
    let live = machine.model.pairs();
    let first = live[0].0;
    let last = live.last().unwrap().0;
    disjoint_batch(
        &mut machine,
        [last, stale, first, id(MAX_INDICES, 1)],
        marker,
    );
    disjoint_batch(&mut machine, [stale, stale], marker);
    for (candidate, value) in machine.model.pairs() {
        twin[candidate] = value;
    }
    compare_future_ids(&mut machine, &mut twin);
}

fn disjoint_duplicates<const PAGE: usize>(operations: &[Op]) {
    let mut machine = sequence::<PAGE>(&[], operations);
    for value in [10, 20, 30] {
        machine.insert(value, false);
    }
    let pairs = machine.model.pairs();
    let (a, b, c) = (pairs[0].0, pairs[1].0, pairs[2].0);
    let stable = addresses(&machine.hive);
    let mut twin = machine.hive.clone();
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _ = machine.hive.get_disjoint_mut([a, a]);
        }))
        .is_err()
    );
    check(&mut machine.hive, &machine.model);
    assert_stable(&machine.hive, &stable);
    // For PAGE=1 these distinct live ids necessarily occupy different pages.
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _ = machine.hive.get_disjoint_mut([c, a, b, c]);
        }))
        .is_err()
    );
    assert_stable(&machine.hive, &stable);
    compare_future_ids(&mut machine, &mut twin);
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn drain_prefix_drop_matches_clear_and_future_ids(
        initial in prop::collection::vec(any::<i32>(), 0..9),
        operations in prop::collection::vec(operation(), 0..max_len() / 3),
        raw in any::<u16>(),
    ) {
        drain_prefix::<1>(&initial, &operations, raw, false);
        drain_prefix::<2>(&initial, &operations, raw, false);
        drain_prefix::<4>(&initial, &operations, raw, false);
    }

    #[test]
    fn leaked_drain_keeps_unyielded_values_and_matches_remove(
        initial in prop::collection::vec(any::<i32>(), 0..9),
        operations in prop::collection::vec(operation(), 0..max_len() / 3),
        raw in any::<u16>(),
    ) {
        drain_prefix::<1>(&initial, &operations, raw, true);
        drain_prefix::<2>(&initial, &operations, raw, true);
        drain_prefix::<4>(&initial, &operations, raw, true);
    }

    #[test]
    fn drain_caller_and_iterator_drop_each_value_once(
        values in prop::collection::vec(any::<i32>(), 0..max_len() / 3),
        raw in any::<u16>(),
    ) {
        drain_drops::<1>(&values, raw);
        drain_drops::<2>(&values, raw);
        drain_drops::<4>(&values, raw);
    }

    #[test]
    fn extract_if_prefix_matches_visited_mutations_and_removes(
        initial in prop::collection::vec(any::<i32>(), 0..9),
        operations in prop::collection::vec(operation(), 0..max_len() / 3),
        rule in rule(),
        raw in any::<u16>(),
    ) {
        extract_prefix::<1>(&initial, &operations, rule, raw);
        extract_prefix::<2>(&initial, &operations, rule, raw);
        extract_prefix::<4>(&initial, &operations, rule, raw);
    }

    #[test]
    fn extract_if_predicate_panic_preserves_state_and_ownership(rule in rule(), raw in any::<u16>()) {
        extract_panic::<1>(rule, raw);
        extract_panic::<2>(rule, raw);
        extract_panic::<4>(rule, raw);
    }

    #[test]
    fn get_disjoint_mut_matches_model_in_input_order(
        operations in prop::collection::vec(operation(), 0..max_len() / 3),
        targets in prop::collection::vec(target(), 4),
        marker in any::<i32>(),
    ) {
        disjoint_model::<1>(&operations, &targets, marker);
        disjoint_model::<2>(&operations, &targets, marker);
        disjoint_model::<4>(&operations, &targets, marker);
    }

    #[test]
    fn get_disjoint_mut_repeated_live_ids_panic_without_changes(
        operations in prop::collection::vec(operation(), 0..max_len() / 3),
    ) {
        disjoint_duplicates::<1>(&operations);
        disjoint_duplicates::<2>(&operations);
        disjoint_duplicates::<4>(&operations);
    }
}

fn retain_panic<const PAGE: usize>(rule: Rule, raw: u16) {
    let ledger = Rc::new(Ledger::default());
    let mut hive = Hive::<Item, PAGE>::new();
    let mut model = Model::default();
    let mut serials = BTreeMap::new();
    for i in 0..2 * PAGE + 5 {
        insert_item(&mut hive, &mut model, &ledger, &mut serials, i as i32, true);
    }
    // Holes exercise scanning across page boundaries before the panic.
    for index in [1, PAGE + 1] {
        let candidate = id(index as u32, 1);
        drop(hive.remove(candidate));
        model.remove::<PAGE>(candidate).unwrap();
    }
    let before = model.pairs();
    let stable = addresses(&hive);
    let stop = 1 + usize::from(raw) % (before.len() - 1);
    let mut visits = Vec::new();
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            hive.retain(|candidate, item| {
                let turn = visits.len();
                visits.push((candidate, item.value));
                item.value = item.value.wrapping_add(rule.delta);
                assert_ne!(turn, stop, "retain predicate bomb");
                turn != 0 && rule.keep(candidate, item.value)
            });
        }))
        .is_err()
    );
    assert_eq!(visits, before[..=stop]);
    for (turn, &(candidate, value)) in before.iter().enumerate().take(stop + 1) {
        let value = value.wrapping_add(rule.delta);
        // Mutation before the panic is observable, but no removal decision
        // was returned for that entry. Later entries were never visited.
        if turn == stop || (turn != 0 && rule.keep(candidate, value)) {
            model.live.get_mut(&candidate.index()).unwrap().1 = value;
        } else {
            model.remove::<PAGE>(candidate).unwrap();
        }
    }
    assert_stable(&hive, &stable);
    check_items(&mut hive, &model, &ledger, &serials);
    insert_item(&mut hive, &mut model, &ledger, &mut serials, 99, false);
    hive.shrink_to_fit();
    model.shrink::<PAGE>();
    assert_stable(&hive, &stable);
    check_items(&mut hive, &model, &ledger, &serials);
    drop(hive);
    ledger.check_live([]);
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn retain_predicate_panic_commits_only_completed_decisions(rule in rule(), raw in any::<u16>()) {
        retain_panic::<1>(rule, raw);
        retain_panic::<2>(rule, raw);
        retain_panic::<3>(rule, raw);
        retain_panic::<4>(rule, raw);
    }
}

fn drop_panics<const PAGE: usize>() {
    let n = 2 * PAGE + 3;
    for action in 0..3 {
        for stop in [0, PAGE - 1, PAGE, n - 1] {
            let ledger = Rc::new(Ledger::default());
            let mut hive = Hive::<Item, PAGE>::new();
            let mut model = Model::default();
            let mut serials = BTreeMap::new();
            for i in 0..n {
                insert_item(
                    &mut hive,
                    &mut model,
                    &ledger,
                    &mut serials,
                    i as i32,
                    false,
                );
            }
            let before = model.pairs();
            let stable = addresses(&hive);
            ledger.panic_drop.set(Some(serials[&before[stop].0]));
            let mut visits = Vec::new();
            assert!(
                catch_unwind(AssertUnwindSafe(|| {
                    match action {
                        0 => drop(hive.remove(before[stop].0)),
                        1 => hive.retain(|candidate, item| {
                            let turn = visits.len();
                            visits.push(candidate);
                            item.value = item.value.wrapping_add(11);
                            turn % 2 == 1 && turn != stop
                        }),
                        _ => hive.clear(),
                    }
                }))
                .is_err()
            );
            assert_eq!(ledger.panic_drop.get(), None);
            match action {
                0 => {
                    model.remove::<PAGE>(before[stop].0).unwrap();
                }
                1 => {
                    assert_eq!(
                        visits,
                        before[..=stop].iter().map(|x| x.0).collect::<Vec<_>>()
                    );
                    for (turn, &(candidate, value)) in before.iter().enumerate().take(stop + 1) {
                        if turn % 2 == 1 && turn != stop {
                            model.live.get_mut(&candidate.index()).unwrap().1 =
                                value.wrapping_add(11);
                        } else {
                            model.remove::<PAGE>(candidate).unwrap();
                        }
                    }
                }
                _ => {
                    for &(candidate, _) in &before[..=stop] {
                        model.remove::<PAGE>(candidate).unwrap();
                    }
                }
            }
            assert_stable(&hive, &stable);
            check_items(&mut hive, &model, &ledger, &serials);
            // Continue using the damaged history, including its free lists.
            insert_item(&mut hive, &mut model, &ledger, &mut serials, 100, true);
            hive.shrink_to_fit();
            model.shrink::<PAGE>();
            check_items(&mut hive, &model, &ledger, &serials);
            hive.clear();
            assert!(hive.is_empty());
            assert_eq!(hive.iter().len(), 0);
            drop(hive);
            ledger.check_live([]);
        }
    }
}

#[test]
fn panicking_drops_leave_remove_retain_and_clear_consistent() {
    drop_panics::<1>();
    drop_panics::<2>();
    drop_panics::<3>();
    drop_panics::<4>();
}

fn owning_iterator_drop<const PAGE: usize>() {
    let n = 3 * PAGE + 2;
    for consumed in 0..=n + 1 {
        let ledger = Rc::new(Ledger::default());
        let hive: Hive<Item, PAGE> = (0..n).map(|i| ledger.make(i as i32)).collect();
        let mut iter = hive.into_iter();
        let (mut front, mut back) = (0, n);
        for step in 0..consumed {
            assert_eq!(iter.len(), back - front);
            assert_eq!(iter.size_hint(), (back - front, Some(back - front)));
            let (actual, expected) = if front == back {
                (iter.next_back(), None)
            } else if step % 3 == 0 {
                back -= 1;
                (iter.next_back(), Some(back))
            } else {
                let expected = front;
                front += 1;
                (iter.next(), Some(expected))
            };
            assert_eq!(actual.as_ref().map(|(_, item)| item.serial), expected);
            drop(actual);
            ledger.check_live(front..back);
        }
        drop(iter);
        ledger.check_live([]);
    }
    // Also exercise the hive's own destructor without an owning iterator.
    let ledger = Rc::new(Ledger::default());
    let hive: Hive<Item, PAGE> = (0..n).map(|i| ledger.make(i as i32)).collect();
    drop(hive);
    ledger.check_live([]);
}

#[test]
fn partially_consumed_into_iter_and_hive_drop_every_value_once() {
    owning_iterator_drop::<1>();
    owning_iterator_drop::<2>();
    owning_iterator_drop::<3>();
    owning_iterator_drop::<4>();
}

#[test]
fn default_page_crossing_preserves_addresses() {
    if cfg!(miri) {
        return;
    }
    let mut hive: Hive<i32> = Hive::new();
    let mut model = Model::default();
    let page = Hive::<i32>::PAGE;
    let first = hive.insert(0);
    assert_eq!(first, model.insert::<{ Hive::<i32>::PAGE }>(0));
    let stable = addresses(&hive);
    hive.reserve(page + 2);
    model.reserve::<{ Hive::<i32>::PAGE }>(page + 2);
    let allocated = hive.allocated_pages();
    for i in 1..=page {
        assert_eq!(
            hive.insert(i as i32),
            model.insert::<{ Hive::<i32>::PAGE }>(i as i32)
        );
        assert_eq!(hive.allocated_pages(), allocated);
        assert_stable(&hive, &stable);
    }
    check(&mut hive, &model);
    for (candidate, _) in model.pairs().into_iter().take(page) {
        hive.remove(candidate).unwrap();
        model.remove::<{ Hive::<i32>::PAGE }>(candidate).unwrap();
    }
    let stable = addresses(&hive);
    hive.shrink_to_fit();
    model.shrink::<{ Hive::<i32>::PAGE }>();
    assert_eq!(hive.allocated_pages(), 1);
    assert_stable(&hive, &stable);
    check(&mut hive, &model);
}

#[cfg(feature = "serde")]
mod serde_tests {
    use super::*;
    use serde_json::json;

    pub(super) fn round_trip<const PAGE: usize>(machine: &Machine<PAGE>) {
        let json = serde_json::to_string(&machine.hive).unwrap();
        let mut back: Hive<i32, PAGE> = serde_json::from_str(&json).unwrap();
        let mut model = machine.model.after_deserialization::<PAGE>();
        assert_eq!(back, machine.hive);
        check(&mut back, &model);
        // Compare future ids against the wire-format policy, which need not
        // preserve the original's LIFO order or empty-page allocations.
        let probes = (model.pages.len() * PAGE).min(if cfg!(miri) { 4 } else { 12 }) + 3;
        for i in 0..probes {
            let before = addresses(&back);
            assert_eq!(back.insert(i as i32), model.insert::<PAGE>(i as i32));
            assert_stable(&back, &before);
            check(&mut back, &model);
        }
        back.clear();
        for (candidate, _) in model.pairs() {
            model.remove::<PAGE>(candidate).unwrap();
        }
        back.shrink_to_fit();
        model.shrink::<PAGE>();
        assert_eq!(back.insert(90), model.insert::<PAGE>(90));
        check(&mut back, &model);
    }

    fn rebuilt_order<const PAGE: usize>() {
        let mut machine = Machine::<PAGE>::new(0);
        for i in 0..2 * PAGE {
            machine.insert(i as i32, false);
        }
        let before = machine.model.pairs();
        for &(candidate, _) in &before {
            machine.hive.remove(candidate).unwrap();
            machine.model.remove::<PAGE>(candidate).unwrap();
        }
        let json = serde_json::to_string(&machine.hive).unwrap();
        let mut back: Hive<i32, PAGE> = serde_json::from_str(&json).unwrap();
        for index in 0..2 * PAGE {
            assert_eq!(back.insert(5), id(index as u32, 2));
            assert!(before.iter().all(|&(stale, _)| !back.contains(stale)));
        }
        round_trip(&machine);
        machine.step(&Op::Shrink);
        round_trip(&machine); // Unallocated pages with non-zero floors survive.
    }

    #[test]
    fn serde_rebuilds_lowest_page_and_offset_first_and_preserves_floors() {
        rebuilt_order::<1>();
        rebuilt_order::<2>();
        rebuilt_order::<3>();
        rebuilt_order::<4>();
        let default = sequence::<{ Hive::<i32>::PAGE }>(
            &[4, 5, 6],
            &[Op::Remove(Target::Live(1)), Op::Shrink],
        );
        round_trip(&default);
    }

    fn malformed<const PAGE: usize>() {
        let mut bad = vec![
            json!([{"floor": 1, "slots": []}]),
            json!([{"floor": u32::MAX, "slots": []}]),
            json!([{"floor": 0, "slots": [{"Vacant": 1}]}]),
            json!([{"floor": 0, "slots": [{"Vacant": u32::MAX}]}]),
            json!([{"floor": 0, "slots": [{"Occupied": [0, 3]}]}]),
            json!([{"floor": 0, "slots": [{"Occupied": [2, 3]}]}]),
        ];
        let slots = vec![json!({"Vacant": 2}); PAGE + 1];
        bad.push(json!([{"floor": 0, "slots": slots}]));
        for input in bad {
            assert!(
                serde_json::from_value::<Hive<i32, PAGE>>(input.clone()).is_err(),
                "{input}"
            );
        }
    }

    #[test]
    fn serde_rejects_odd_floors_parity_mismatches_and_oversized_pages() {
        malformed::<1>();
        malformed::<2>();
        malformed::<3>();
        malformed::<4>();
    }

    fn retirement<const PAGE: usize>() {
        let slots: Vec<_> = (0..PAGE).map(|offset| {
            json!({"Occupied": [if offset == 0 { u32::MAX } else { 1 }, offset as i32]})
        }).collect();
        let mut hive: Hive<i32, PAGE> = serde_json::from_value(json!([
            {"floor": 0, "slots": slots}
        ]))
        .unwrap();
        let last = id(0, MAX_GENERATION);
        assert!(hive.contains(last));
        assert_eq!(hive.remove(last), Some(0));
        for offset in 1..PAGE {
            assert_eq!(hive.remove(id(offset as u32, 1)), Some(offset as i32));
        }
        assert!(hive.is_empty());
        hive.shrink_to_fit();
        assert_eq!(
            hive.allocated_pages(),
            1,
            "retired slot pins its empty page"
        );
        // Both retirement and the allocated page survive another snapshot.
        let json = serde_json::to_string(&hive).unwrap();
        let mut hive: Hive<i32, PAGE> = serde_json::from_str(&json).unwrap();
        hive.shrink_to_fit();
        assert_eq!(hive.allocated_pages(), 1);
        let mut issued = BTreeSet::new();
        for _ in 0..PAGE + 3 {
            let candidate = hive.insert(8);
            assert_ne!(candidate.index(), 0);
            assert!(issued.insert(candidate));
            assert!(!hive.contains(last));
            assert_eq!(hive.occupant(0), None);
        }
        hive.clear();
        hive.shrink_to_fit();
        assert_eq!(hive.allocated_pages(), 1);
        assert_ne!(hive.insert(9).index(), 0);
        assert!(!hive.contains(last));
    }

    #[test]
    fn last_generation_retires_slot_and_pins_page_without_brute_force() {
        retirement::<1>();
        retirement::<2>();
        retirement::<3>();
        retirement::<4>();
    }
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn operation_sequences_match_model(
        initial in prop::collection::vec(any::<i32>(), 0..9),
        operations in prop::collection::vec(operation(), 0..max_len()),
    ) {
        let a = sequence::<1>(&initial, &operations);
        let b = sequence::<2>(&initial, &operations);
        let c = sequence::<3>(&initial, &operations);
        let d = sequence::<4>(&initial, &operations);
        same_pairs(&a, &b);
        same_pairs(&a, &c);
        same_pairs(&b, &d);
        // Cheap default-page coverage, including the default type parameter.
        let short = &operations[..operations.len().min(8)];
        let default: Hive<i32> = Hive::new();
        assert_eq!(default, Hive::<i32, 1>::new());
        let large = sequence::<{ Hive::<i32>::PAGE }>(&initial, short);
        let small = sequence::<1>(&initial, short);
        same_pairs(&large, &small);
        #[cfg(feature = "serde")]
        {
            serde_tests::round_trip(&a);
            serde_tests::round_trip(&b);
            serde_tests::round_trip(&c);
            serde_tests::round_trip(&d);
        }
    }

    #[test]
    fn reserve_then_insert_allocates_no_pages(
        operations in prop::collection::vec(operation(), 0..max_len() / 3),
        n in 0usize..10,
        fallible in any::<bool>(),
    ) {
        reserve_probe::<1>(&operations, n, fallible);
        reserve_probe::<2>(&operations, n, fallible);
        reserve_probe::<3>(&operations, n, fallible);
        reserve_probe::<4>(&operations, n, fallible);
    }
}
