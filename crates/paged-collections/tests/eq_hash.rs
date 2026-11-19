//! Equality and hashing across collection layouts and mutation histories.

use std::collections::{HashMap, HashSet, hash_map::DefaultHasher};
use std::fmt::Debug;
use std::hash::{Hash, Hasher};

use paged_collections::bit_set::PagedBitSet;
use paged_collections::direct_map::PagedDirectMap;
use paged_collections::page::{Directory, Radix, Tree};
use paged_collections::priority_queue::PagedPriorityQueue;
use paged_collections::sparse_map::PagedSparseMap;
use paged_collections::sparse_set::PagedSparseSet;
use proptest::prelude::*;

fn hash(value: &impl Hash) -> u64 {
    let mut state = DefaultHasher::new();
    value.hash(&mut state);
    state.finish()
}

fn assert_eq_and_hash<L, R>(left: &L, right: &R)
where
    L: Debug + Eq + Hash + PartialEq<R>,
    R: Debug + Eq + Hash + PartialEq<L>,
{
    assert_eq!(left, right);
    assert_eq!(right, left);
    assert_eq!(hash(left), hash(right));
}

#[test]
fn bit_set_ignores_directory_and_empty_pages() {
    let flat: PagedBitSet<usize, 2> = [127, 0, 128, 65].into_iter().collect();
    let tree: PagedBitSet<usize, 2, Tree> = [128, 65, 127, 0].into_iter().collect();
    let mut radix: PagedBitSet<usize, 2, Radix> =
        [0, usize::MAX, 127, 65, 128].into_iter().collect();
    assert!(radix.remove(usize::MAX));
    assert_eq!(flat.allocated_pages(), 2);
    assert_eq!(radix.allocated_pages(), 3);
    assert_eq_and_hash(&flat, &tree);
    assert_eq_and_hash(&tree, &radix);
    assert_eq_and_hash(&flat, &radix);

    assert!(radix.remove(65));
    assert!(radix.insert(64));
    assert_eq!(flat.len(), radix.len());
    assert_ne!(flat, radix);
}

#[test]
fn sparse_set_ignores_dense_order_and_layout() {
    let left: PagedSparseSet<u16, 4> = [1, 9, 21].into_iter().collect();
    let mut right: PagedSparseSet<u16, 16, Radix> = [9, 200, 21, 1].into_iter().collect();
    let tree: PagedSparseSet<u16, 8, Tree> = [1, 9, 21].into_iter().collect();
    assert_ne!(left, right);
    assert!(right.remove(200));
    assert_ne!(left.as_slice(), right.as_slice());
    assert_eq_and_hash(&left, &right);
    assert_eq_and_hash(&tree, &right);

    assert!(right.remove(9));
    assert!(right.insert(10));
    assert_ne!(left, right);
}

#[test]
fn sparse_map_ignores_dense_order_and_layout() {
    let entries = [(1, 10), (9, 20), (21, 30)];
    let left: PagedSparseMap<u16, i32, 4> = entries.into_iter().collect();
    let mut right: PagedSparseMap<u16, i32, 16, Tree> =
        [(9, 20), (200, 0), (21, 30), (1, 10)].into_iter().collect();
    let radix: PagedSparseMap<u16, i32, 8, Radix> = entries.into_iter().collect();
    assert_ne!(left, right);
    assert_eq!(right.remove(200), Some(0));
    assert_ne!(left.key_slice(), right.key_slice());
    assert_eq_and_hash(&left, &right);
    assert_eq_and_hash(&radix, &right);

    assert_eq!(right.insert(9, 99), Some(20));
    assert_ne!(left, right);
}

#[test]
fn direct_map_compares_entries_in_slot_order_across_layouts() {
    let entries = [(1, 10), (9, 20), (21, 30)];
    let left: PagedDirectMap<u16, i32, 4> = entries.into_iter().collect();
    let mut right: PagedDirectMap<u16, i32, 16, Radix> =
        [(21, 30), (200, 0), (9, 20), (1, 10)].into_iter().collect();
    let tree: PagedDirectMap<u16, i32, 8, Tree> = entries.into_iter().collect();
    assert_ne!(left, right);
    assert_eq!(right.remove(200), Some(0));
    assert!(left.iter().eq(right.iter()));
    assert_eq_and_hash(&left, &right);
    assert_eq_and_hash(&tree, &right);

    assert_eq!(right.insert(9, 99), Some(20));
    assert_ne!(left, right);
}

#[test]
fn priority_queue_ignores_heap_order_and_layout() {
    let entries = [(1, 10), (9, 5), (21, 3)];
    let left: PagedPriorityQueue<u16, i32, 4> = entries.into_iter().collect();
    let mut right: PagedPriorityQueue<u16, i32, 16, Tree> =
        [(21, 3), (9, 5), (200, 0), (1, 10)].into_iter().collect();
    let radix: PagedPriorityQueue<u16, i32, 8, Radix> = entries.into_iter().collect();
    assert_ne!(left, right);
    assert_eq!(right.remove(&200), Some((200, 0)));
    assert_ne!(left.key_slice(), right.key_slice());
    assert_eq_and_hash(&left, &right);
    assert_eq_and_hash(&radix, &right);

    assert_eq!(right.change_priority(&9, 6), Some(5));
    assert_ne!(left, right);
}

#[test]
fn maps_and_queues_compare_different_value_types() {
    let sparse: PagedSparseMap<u16, String, 4> = [(9, String::from("value"))].into_iter().collect();
    let sparse_str: PagedSparseMap<u16, &str, 16, Radix> = [(9, "value")].into_iter().collect();
    assert_eq!(sparse, sparse_str);

    let direct: PagedDirectMap<u16, String, 4> = [(9, String::from("value"))].into_iter().collect();
    let direct_str: PagedDirectMap<u16, &str, 16, Tree> = [(9, "value")].into_iter().collect();
    assert_eq!(direct, direct_str);

    let queue: PagedPriorityQueue<u16, String, 4> =
        [(9, String::from("priority"))].into_iter().collect();
    let queue_str: PagedPriorityQueue<u16, &str, 16, Radix> =
        [(9, "priority")].into_iter().collect();
    assert_eq!(queue, queue_str);
}

#[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
#[test]
fn equality_distinguishes_key_generations() {
    use std::num::NonZeroU32;

    use paged_collections::id_alloc::Id;

    let old = Id::new(9, NonZeroU32::new(1).unwrap());
    let new = Id::new(9, NonZeroU32::new(2).unwrap());

    let left: PagedSparseSet<_, 4> = [old].into_iter().collect();
    let right: PagedSparseSet<_, 16, Radix> = [new].into_iter().collect();
    assert_ne!(left, right);

    let left: PagedSparseMap<_, _, 4> = [(old, 20)].into_iter().collect();
    let right: PagedSparseMap<_, _, 16, Tree> = [(new, 20)].into_iter().collect();
    assert_ne!(left, right);

    let left: PagedDirectMap<_, _, 4> = [(old, 20)].into_iter().collect();
    let right: PagedDirectMap<_, _, 16, Radix> = [(new, 20)].into_iter().collect();
    assert_ne!(left, right);

    let left: PagedPriorityQueue<_, _, 4> = [(old, 20)].into_iter().collect();
    let right: PagedPriorityQueue<_, _, 16, Tree> = [(new, 20)].into_iter().collect();
    assert_ne!(left, right);
}

#[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
#[test]
fn hive_compares_and_hashes_live_ids_and_values() {
    use paged_collections::hive::Hive;

    let left: Hive<i32, 2> = [11, 22, 33].into_iter().collect();
    let wide: Hive<i32, 8> = [11, 22, 33].into_iter().collect();
    assert_eq_and_hash(&left, &wide);

    let mut right = left.clone();
    assert_eq_and_hash(&left, &right);
    let extra = right.insert(44);
    assert_ne!(left, right);
    assert_eq!(right.remove(extra), Some(44));
    assert_eq_and_hash(&left, &right);

    let first = right.ids().next().unwrap();
    right[first] = 99;
    assert_ne!(left, right);
    assert_eq!(right.remove(first), Some(99));
    let reused = right.insert(11);
    assert_eq!(first.index(), reused.index());
    assert_ne!(first, reused);
    assert_ne!(left, right);
}

type Entry = (u16, i32);

fn shuffled_entries() -> impl Strategy<Value = (Vec<Entry>, Vec<Entry>)> {
    prop::collection::btree_map(0..512_u16, any::<i32>(), 0..32).prop_flat_map(|entries| {
        let entries: Vec<_> = entries.into_iter().collect();
        (
            Just(entries.clone()).prop_shuffle(),
            Just(entries).prop_shuffle(),
        )
    })
}

fn sparse_with_removals<const PAGE: usize, D: Directory>(
    entries: &[Entry],
    temporary: u16,
) -> (
    PagedSparseSet<u16, PAGE, D>,
    PagedSparseMap<u16, i32, PAGE, D>,
) {
    let mut set = PagedSparseSet::new();
    let mut map = PagedSparseMap::new();
    for &(key, value) in entries {
        set.insert(temporary);
        map.insert(temporary, value);
        set.insert(key);
        map.insert(key, value);
        set.remove(temporary);
        map.remove(temporary);
    }
    (set, map)
}

proptest! {
    #[test]
    fn sparse_equality_and_hash_match_shuffled_models((left, right) in shuffled_entries()) {
        let (mut left_set, mut left_map) = sparse_with_removals::<4, Radix>(&left, 512);
        let (right_set, right_map) = sparse_with_removals::<16, Tree>(&right, 513);
        let mut model_left_set: HashSet<_> = left.iter().map(|&(key, _)| key).collect();
        let model_right_set: HashSet<_> = right.iter().map(|&(key, _)| key).collect();
        let mut model_left_map: HashMap<_, _> = left.into_iter().collect();
        let model_right_map: HashMap<_, _> = right.into_iter().collect();

        let actual_set: HashSet<_> = left_set.iter().copied().collect();
        let actual_map: HashMap<_, _> = left_map.iter().map(|(&key, &value)| (key, value)).collect();
        prop_assert_eq!(&actual_set, &model_left_set);
        prop_assert_eq!(&actual_map, &model_left_map);
        prop_assert_eq!(&left_set, &right_set);
        prop_assert_eq!(&left_map, &right_map);
        prop_assert_eq!(hash(&left_set), hash(&right_set));
        prop_assert_eq!(hash(&left_map), hash(&right_map));
        prop_assert_eq!(left_set == right_set, model_left_set == model_right_set);
        prop_assert_eq!(left_map == right_map, model_left_map == model_right_map);

        left_set.insert(512);
        model_left_set.insert(512);
        left_map.insert(512, 0);
        model_left_map.insert(512, 0);
        prop_assert_eq!(left_set == right_set, model_left_set == model_right_set);
        prop_assert_eq!(left_map == right_map, model_left_map == model_right_map);
    }
}
