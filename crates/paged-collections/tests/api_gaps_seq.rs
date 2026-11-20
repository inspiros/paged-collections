//! Models and panic-path checks for the sequential collection APIs.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashSet};
use std::num::NonZeroU32;
use std::ops::Bound::{Excluded, Included, Unbounded};
use std::panic::{AssertUnwindSafe, catch_unwind};

use paged_collections::id_alloc::Id;
use paged_collections::key::Key;
use paged_collections::page::Tree;
use paged_collections::paged_vec::PagedVec;
use paged_collections::priority_queue::PagedPriorityQueue;
use paged_collections::sparse_map::{Entry, PagedSparseMap};
use paged_collections::sparse_set::PagedSparseSet;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig {
        cases: if cfg!(miri) { 4 } else { 128 },
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    #[test]
    fn sparse_entries_and_set_queries_match_models(
        steps in prop::collection::vec((0u8..10, 0u32..12, 1u32..4, -20i32..20), 0..64)
    ) {
        let mut map = PagedSparseMap::<Id, i32, 3, Tree>::new();
        let mut set = PagedSparseSet::<Id, 3, Tree>::new();
        let mut entries = BTreeMap::<u32, (Id, i32)>::new();
        let mut keys = HashSet::<Id>::new();

        for (op, index, generation, value) in steps {
            let key = Id::new(index, NonZeroU32::new(generation).unwrap());
            let exact = entries.get(&index).copied().filter(|(stored, _)| *stored == key);
            let old_pos = map.occupant(index as usize).and_then(|(&stored, _)| map.position(stored));
            let pages = map.allocated_pages();
            let entry = map.entry(key);
            prop_assert_eq!(entry.key(), &key);
            prop_assert_eq!(matches!(&entry, Entry::Occupied(_)), exact.is_some());

            match op {
                0 => {
                    match entry {
                        Entry::Occupied(entry) => prop_assert_eq!(*entry.get(), exact.unwrap().1),
                        Entry::Vacant(entry) => prop_assert_eq!(entry.into_key(), key),
                    }
                    prop_assert_eq!(map.allocated_pages(), pages);
                }
                1 => {
                    let expected = exact.map_or(value, |(_, old)| old);
                    prop_assert_eq!(*entry.or_insert(value), expected);
                    entries.insert(index, (key, expected));
                }
                2 => {
                    let expected = exact.map_or(value, |(_, old)| old);
                    prop_assert_eq!(*entry.or_insert_with(|| {
                        assert!(exact.is_none());
                        value
                    }), expected);
                    entries.insert(index, (key, expected));
                }
                3 => {
                    let expected = exact.map_or(generation as i32, |(_, old)| old);
                    prop_assert_eq!(*entry.or_insert_with_key(|given| {
                        assert!(exact.is_none());
                        assert_eq!(*given, key);
                        given.generation().get() as i32
                    }), expected);
                    entries.insert(index, (key, expected));
                }
                4 => {
                    let expected = exact.map_or(0, |(_, old)| old);
                    prop_assert_eq!(*entry.or_default(), expected);
                    entries.insert(index, (key, expected));
                }
                5 => {
                    let expected = exact.map_or(value, |(_, old)| old.wrapping_add(1));
                    let stored = entry.and_modify(|old| *old = old.wrapping_add(1)).or_insert(value);
                    prop_assert_eq!(*stored, expected);
                    entries.insert(index, (key, expected));
                }
                6 => {
                    let occupied = entry.insert_entry(value);
                    prop_assert_eq!(occupied.key(), &key);
                    prop_assert_eq!(*occupied.get(), value);
                    entries.insert(index, (key, value));
                }
                7 => match entry {
                    Entry::Occupied(mut entry) => {
                        *entry.get_mut() = exact.unwrap().1.wrapping_add(1);
                        prop_assert_eq!(entry.insert(value), exact.unwrap().1.wrapping_add(1));
                        *entry.into_mut() = value + 1;
                        entries.insert(index, (key, value + 1));
                    }
                    Entry::Vacant(entry) => {
                        prop_assert_eq!(*entry.insert(value), value);
                        entries.insert(index, (key, value));
                    }
                },
                8 => {
                    if let Entry::Occupied(entry) = entry {
                        prop_assert_eq!(entry.remove(), exact.unwrap().1);
                        entries.remove(&index);
                    }
                }
                _ => {
                    if let Entry::Occupied(entry) = entry {
                        prop_assert_eq!(Some(entry.remove_entry()), exact);
                        entries.remove(&index);
                    }
                }
            }
            if let (Some(pos), Some(&(stored, _))) = (old_pos, entries.get(&index)) {
                prop_assert_eq!(map.position(stored), Some(pos));
            }
            let actual: BTreeMap<_, _> = map.iter().map(|(&key, &v)| (key.index(), (key, v))).collect();
            prop_assert_eq!(map.len(), entries.len());
            prop_assert_eq!(&actual, &entries);

            let present = keys.get(&key).copied();
            prop_assert_eq!(set.get(key).copied(), present);
            if op % 2 == 0 {
                let vacant = keys.iter().all(|stored| stored.index() != index);
                prop_assert_eq!(set.insert(key), vacant);
                if vacant {
                    keys.insert(key);
                }
            } else {
                prop_assert_eq!(set.take(key), keys.take(&key));
            }
            let actual: HashSet<_> = set.iter().copied().collect();
            prop_assert_eq!(set.len(), keys.len());
            prop_assert_eq!(&actual, &keys);
            for &stored in &keys {
                prop_assert_eq!(set.get(stored), Some(&stored));
            }
        }
    }

    #[test]
    fn extract_if_matches_vec(
        initial in prop::collection::vec(-8i32..8, 0..32),
        bounds in (any::<usize>(), any::<usize>()),
        limit in 0usize..32
    ) {
        let a = bounds.0 % (initial.len() + 1);
        let b = bounds.1 % (initial.len() + 1);
        let range = a.min(b)..a.max(b);
        let mut model = initial.clone();
        let mut vec: PagedVec<_, 3> = initial.into_iter().collect();
        vec.reserve(10);
        let pages = vec.allocated_pages();
        let mut expected = Vec::new();
        let mut expected_visits = Vec::new();
        let mut pos = range.start;
        while pos < range.end - expected.len() && expected.len() < limit {
            expected_visits.push(model[pos]);
            if extract_even(&mut model[pos]) {
                expected.push(model.remove(pos));
            } else {
                pos += 1;
            }
        }

        let mut visits = Vec::new();
        let removed: Vec<_> = vec.extract_if(range, |value| {
            visits.push(*value);
            extract_even(value)
        }).take(limit).collect();
        prop_assert_eq!(removed, expected);
        prop_assert_eq!(visits, expected_visits);
        prop_assert_eq!(vec.allocated_pages(), pages);
        prop_assert!(vec.iter().eq(&model));
        prop_assert!(vec.iter().rev().eq(model.iter().rev()));
        vec.push(42);
        model.push(42);
        prop_assert!(vec.into_iter().eq(model));
    }
}

fn extract_even(value: &mut i32) -> bool {
    *value += 1;
    *value % 2 == 0
}

#[test]
fn sparse_entries_and_take_use_the_stored_key() {
    #[derive(Clone, Copy)]
    struct TaggedKey(usize, &'static str);

    impl PartialEq for TaggedKey {
        fn eq(&self, other: &Self) -> bool {
            self.0 == other.0
        }
    }

    impl Eq for TaggedKey {}

    impl Key for TaggedKey {
        fn index(self) -> usize {
            self.0
        }
    }

    let stored = TaggedKey(2, "stored");
    let probe = TaggedKey(2, "probe");
    let mut map: PagedSparseMap<_, _> = PagedSparseMap::new();
    map.insert(stored, 3);
    let entry = map.entry(probe);
    assert_eq!(entry.key().1, "stored");
    let entry = entry.insert_entry(4);
    assert_eq!(entry.remove_entry().0.1, "stored");

    let mut set: PagedSparseSet<_> = PagedSparseSet::new();
    set.insert(stored);
    assert_eq!(set.get(probe).unwrap().1, "stored");
    assert_eq!(set.take(probe).unwrap().1, "stored");
    assert!(set.take(probe).is_none());
}

#[test]
fn queue_try_reserve_grows_dense_arrays_and_preserves_state_on_overflow() {
    let mut queue: PagedPriorityQueue<Id, i64, 2> = PagedPriorityQueue::with_capacity(2);
    let ids = [0, 3, 9].map(|index| Id::new(index, NonZeroU32::MIN));
    queue.push(ids[0], 30);
    queue.push(ids[1], 10);
    let pages = queue.allocated_pages();
    queue.try_reserve(12).unwrap();
    assert!(queue.capacity() >= queue.len() + 12);
    assert_eq!(queue.allocated_pages(), pages);

    let capacity = queue.capacity();
    let keys = queue.key_slice().as_ptr();
    let priorities = queue.priority_slice().as_ptr();
    for additional in [usize::MAX, isize::MAX as usize] {
        assert!(queue.try_reserve(additional).is_err());
        assert_eq!(queue.capacity(), capacity);
        assert_eq!(queue.allocated_pages(), pages);
        assert_eq!(queue.key_slice().as_ptr(), keys);
        assert_eq!(queue.priority_slice().as_ptr(), priorities);
        assert_eq!(queue.key_slice(), &ids[..2]);
        assert_eq!(queue.priority_slice(), &[30, 10]);
    }
    queue.push(ids[2], 20);
    let model = BTreeMap::from([(10, ids[1]), (20, ids[2]), (30, ids[0])]);
    assert!(
        queue
            .into_sorted_iter()
            .eq(model.into_iter().rev().map(|(p, k)| (k, p)))
    );

    let mut zst: PagedPriorityQueue<usize, ()> = PagedPriorityQueue::new();
    zst.try_reserve(8).unwrap();
    assert!(zst.capacity() >= 8);
}

#[test]
fn extract_if_panic_keeps_current_and_unvisited_elements_and_drops_each_once() {
    struct Tracked<'a> {
        index: usize,
        visited: bool,
        drops: &'a RefCell<[usize; 8]>,
    }

    impl Drop for Tracked<'_> {
        fn drop(&mut self) {
            self.drops.borrow_mut()[self.index] += 1;
        }
    }

    let drops = RefCell::new([0; 8]);
    let mut vec: PagedVec<_, 2> = (0..8)
        .map(|index| Tracked {
            index,
            visited: false,
            drops: &drops,
        })
        .collect();
    let result = catch_unwind(AssertUnwindSafe(|| {
        vec.extract_if(1..7, |item| {
            item.visited = true;
            assert_ne!(item.index, 4, "filter panicked");
            item.index % 2 == 1
        })
        .for_each(drop);
    }));
    assert!(result.is_err());
    assert!(vec.iter().map(|item| (item.index, item.visited)).eq([
        (0, false),
        (2, true),
        (4, true),
        (5, false),
        (6, false),
        (7, false)
    ]));
    assert_eq!(*drops.borrow(), [0, 1, 0, 1, 0, 0, 0, 0]);
    drop(vec);
    assert_eq!(*drops.borrow(), [1; 8]);
}

#[test]
fn extract_if_bounds_and_early_drop() {
    let mut vec: PagedVec<_, 2> = (0..8).collect();
    for bounds in [
        (Unbounded, Included(usize::MAX)),
        (Excluded(usize::MAX), Unbounded),
        (Included(5), Excluded(3)),
        (Unbounded, Excluded(9)),
    ] {
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                drop(vec.extract_if(bounds, |_| false));
            }))
            .is_err()
        );
        assert!(vec.iter().copied().eq(0..8));
    }

    let mut iter = vec.extract_if((Excluded(0), Included(5)), |item| {
        assert!(*item <= 2, "early drop must not call the filter");
        *item += 10;
        *item == 12
    });
    assert_eq!(iter.size_hint(), (0, Some(5)));
    assert_eq!(iter.next(), Some(12));
    assert_eq!(iter.size_hint(), (0, Some(3)));
    drop(iter);
    assert!(vec.into_iter().eq([0, 11, 3, 4, 5, 6, 7]));

    let mut zst: PagedVec<(), 2> = [(); 5].into_iter().collect();
    assert_eq!(zst.extract_if(1..4, |_| true).count(), 3);
    assert_eq!(zst.len(), 2);
}
