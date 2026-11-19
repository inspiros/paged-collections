//! Parallel iteration against the sequential order of each collection.

#![cfg(feature = "rayon")]

use std::sync::LazyLock;

use paged_collections::bit_set::PagedBitSet;
use paged_collections::direct_map::{PagedDirectMap, Slot};
use paged_collections::hive::Hive;
use paged_collections::key::{DirectKey, Key};
use paged_collections::page::{Directory, Flat, Radix, Tree};
use paged_collections::paged_deque::PagedDeque;
use paged_collections::paged_vec::PagedVec;
use paged_collections::priority_queue::PagedPriorityQueue;
use paged_collections::sparse_map::PagedSparseMap;
use paged_collections::sparse_set::PagedSparseSet;
use proptest::prelude::*;
use proptest::test_runner::TestCaseResult;
use rayon::iter::plumbing::{Producer, ProducerCallback};
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

static POOL: LazyLock<ThreadPool> =
    LazyLock::new(|| ThreadPoolBuilder::new().num_threads(4).build().unwrap());

struct SplitAt(usize);

impl<T> ProducerCallback<T> for SplitAt {
    type Output = (Vec<T>, Vec<T>);

    fn callback<P: Producer<Item = T>>(self, producer: P) -> Self::Output {
        let (left, right) = producer.split_at(self.0);
        (left.into_iter().collect(), right.into_iter().collect())
    }
}

fn bump(value: &mut i32) -> i32 {
    let before = *value;
    *value += 1;
    before
}

fn check_paged_vec<const PAGE: usize>() {
    for len in 0..=13 {
        let mut v = PagedVec::<i32, PAGE>::with_capacity(19);
        v.extend(0..len);
        let expected: Vec<_> = v.iter().copied().collect();
        for at in 0..=v.len() {
            let (left, right) = v.par_iter().copied().with_producer(SplitAt(at));
            assert_eq!(left, expected[..at]);
            assert_eq!(right, expected[at..]);

            let (left, right) = v.par_iter_mut().with_producer(SplitAt(at));
            assert!(left.into_iter().eq(expected[..at].iter()));
            assert!(right.into_iter().eq(expected[at..].iter()));
        }

        let iter = v.par_iter();
        assert_eq!(
            iter.clone().collect::<Vec<_>>(),
            v.iter().collect::<Vec<_>>()
        );
        assert_eq!(
            iter.with_max_len(1)
                .copied()
                .enumerate()
                .collect::<Vec<_>>(),
            expected.iter().copied().enumerate().collect::<Vec<_>>(),
        );
        assert_eq!(
            v.par_iter().rev().copied().collect::<Vec<_>>(),
            v.iter().rev().copied().collect::<Vec<_>>(),
        );
        let other: Vec<_> = (30..30 + len).collect();
        assert_eq!(
            v.par_iter().copied().zip(&other).collect::<Vec<_>>(),
            v.iter().copied().zip(&other).collect::<Vec<_>>(),
        );
        v.par_iter_mut().with_max_len(1).for_each(|x| *x += 1);
        assert!(v.iter().copied().eq(expected.into_iter().map(|x| x + 1)));
    }
}

#[test]
fn paged_vec_splits_at_every_element_boundary() {
    POOL.install(|| {
        check_paged_vec::<1>();
        check_paged_vec::<3>();
        check_paged_vec::<4>();
    });
}

fn check_paged_vec_split_sizes<const PAGE: usize>(
    len: i32,
    min: usize,
    max: usize,
) -> TestCaseResult {
    let mut v: PagedVec<i32, PAGE> = (0..len).collect();
    let actual: Vec<_> = v
        .par_iter()
        .with_min_len(min)
        .with_max_len(max)
        .copied()
        .collect();
    prop_assert_eq!(actual, v.iter().copied().collect::<Vec<_>>());

    let expected: Vec<_> = v.iter().rev().copied().collect();
    let actual: Vec<_> = v
        .par_iter_mut()
        .with_min_len(min)
        .with_max_len(max)
        .rev()
        .map(bump)
        .collect();
    prop_assert_eq!(actual, expected);
    prop_assert!(v.iter().copied().eq(1..=len));
    Ok(())
}

proptest! {
    #[test]
    fn paged_vec_split_sizes(
        len in 0i32..=257,
        page in prop_oneof![Just(1), Just(2), Just(3), Just(7)],
        min in 1usize..=17,
        extra in 0usize..=32,
    ) {
        POOL.install(|| {
            let max = min + extra;
            match page {
                1 => check_paged_vec_split_sizes::<1>(len, min, max),
                2 => check_paged_vec_split_sizes::<2>(len, min, max),
                3 => check_paged_vec_split_sizes::<3>(len, min, max),
                7 => check_paged_vec_split_sizes::<7>(len, min, max),
                _ => unreachable!(),
            }
        })?;
    }
}

#[test]
fn paged_vec_parallel_collect_and_extend() {
    POOL.install(|| {
        let mut v: PagedVec<i32, 3> = (0..13).into_par_iter().rev().collect();
        assert!(v.iter().copied().eq((0..13).rev()));
        v.par_extend((13..37).into_par_iter().filter(|x| x % 2 == 0));
        assert!(
            v.iter()
                .copied()
                .eq((0..13).rev().chain((13..37).filter(|x| x % 2 == 0)))
        );
        let empty: PagedVec<i32, 1> = rayon::iter::empty().collect();
        assert!(empty.is_empty());
    });
}

fn check_paged_deque<const PAGE: usize>() {
    for populated in [false, true] {
        let mut deque = PagedDeque::<i32, PAGE>::with_capacity(23);
        if populated {
            deque.extend(0..12);
            deque.push_front(-1);
            assert_eq!(deque.pop_back(), Some(11));
        }
        let expected: Vec<_> = deque.iter().copied().collect();
        assert_eq!(deque.par_iter().copied().collect::<Vec<_>>(), expected);
        assert_eq!(deque.par_iter_mut().map(bump).collect::<Vec<_>>(), expected);
        assert!(
            deque
                .iter()
                .copied()
                .eq(expected.into_iter().map(|v| v + 1))
        );
    }
}

#[test]
fn paged_deque_parallel_iteration() {
    POOL.install(|| {
        check_paged_deque::<1>();
        check_paged_deque::<3>();
    });
}

fn check_hive<const PAGE: usize>() {
    for populated in [false, true] {
        let mut hive = Hive::<i32, PAGE>::new();
        if populated {
            let ids: Vec<_> = (0..(5 * PAGE) as i32).map(|v| hive.insert(v)).collect();
            assert_eq!(hive.remove(ids[0]), Some(0));
            let replacement = hive.insert(99);
            assert_eq!(replacement.index(), ids[0].index());
            assert_ne!(replacement, ids[0]);

            assert!(hive.remove(ids[PAGE + PAGE / 2]).is_some());
            for &id in &ids[2 * PAGE..3 * PAGE] {
                assert!(hive.remove(id).is_some());
            }
            hive.shrink_to_fit();
        }
        let expected: Vec<_> = hive.iter().map(|(id, &value)| (id, value)).collect();
        assert_eq!(
            hive.par_iter()
                .map(|(id, &value)| (id, value))
                .collect::<Vec<_>>(),
            expected,
        );
        assert_eq!(
            hive.par_iter_mut()
                .map(|(id, value)| (id, bump(value)))
                .collect::<Vec<_>>(),
            expected,
        );
        assert!(
            hive.iter()
                .map(|(id, &v)| (id, v))
                .eq(expected.into_iter().map(|(id, v)| (id, v + 1)))
        );
    }
}

#[test]
fn hive_parallel_iteration() {
    POOL.install(|| {
        check_hive::<1>();
        check_hive::<3>();
    });
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Gen(u32);

impl Key for Gen {
    fn index(self) -> usize {
        (self.0 & 0xffff) as usize
    }
}

impl DirectKey for Gen {
    type Tag = u16;

    fn tag(self) -> Self::Tag {
        (self.0 >> 16) as u16
    }

    fn from_parts(index: usize, tag: Self::Tag) -> Self {
        Self(index as u32 | (u32::from(tag) << 16))
    }
}

fn check_directory<const PAGE: usize, D: Directory>()
where
    D::Store<Slot<Gen, i32>, PAGE>: Sync,
{
    for populated in [false, true] {
        let mut set = PagedSparseSet::<Gen, PAGE, D>::new();
        let mut sparse = PagedSparseMap::<Gen, i32, PAGE, D>::new();
        let mut queue = PagedPriorityQueue::<Gen, i32, PAGE, D>::new();
        let mut direct = PagedDirectMap::<Gen, i32, PAGE, D>::new();
        if populated {
            for (index, generation, value) in
                [(385, 2, 5), (1, 8, 6), (0, 7, 9), (128, 5, 3), (2, 11, 21)]
            {
                let key = Gen::from_parts(index, generation);
                set.insert(key);
                sparse.insert(key, value);
                queue.push(key, value);
                direct.insert(key, value);
            }
            let old = Gen::from_parts(0, 7);
            let replacement = Gen::from_parts(0, 9);
            assert_eq!(set.replace(replacement), Some(old));
            assert_eq!(sparse.replace(replacement, 32), Some((old, 9)));
            assert_eq!(queue.replace(replacement, 32), Some((old, 9)));
            assert_eq!(direct.replace(replacement, 32), Some((old, 9)));
            for (key, value) in [(Gen::from_parts(1, 8), 6), (Gen::from_parts(128, 5), 3)] {
                assert!(set.remove(key));
                assert_eq!(sparse.remove(key), Some(value));
                assert_eq!(queue.remove(&key), Some((key, value)));
                assert_eq!(direct.remove(key), Some(value));
            }
        }

        assert_eq!(
            set.par_iter().copied().enumerate().collect::<Vec<_>>(),
            set.iter().copied().enumerate().collect::<Vec<_>>(),
        );
        assert_eq!(
            queue.par_iter().enumerate().collect::<Vec<_>>(),
            queue.iter().enumerate().collect::<Vec<_>>(),
        );
        check_sparse_map(&mut sparse);
        check_direct_map(&mut direct);
    }
}

fn check_sparse_map<const PAGE: usize, D: Directory>(map: &mut PagedSparseMap<Gen, i32, PAGE, D>) {
    let expected: Vec<_> = map.iter().map(|(&key, &value)| (key, value)).collect();
    assert_eq!(
        map.par_iter()
            .map(|(&key, &value)| (key, value))
            .collect::<Vec<_>>(),
        expected,
    );
    assert_eq!(
        map.par_keys().copied().collect::<Vec<_>>(),
        map.keys().copied().collect::<Vec<_>>(),
    );
    assert_eq!(
        map.par_values().copied().collect::<Vec<_>>(),
        map.values().copied().collect::<Vec<_>>(),
    );
    assert_eq!(map.par_iter_mut().len(), expected.len());
    assert_eq!(
        map.par_iter_mut()
            .map(|(&key, value)| (key, bump(value)))
            .collect::<Vec<_>>(),
        expected,
    );
    assert_eq!(
        map.par_values_mut().map(bump).collect::<Vec<_>>(),
        expected
            .iter()
            .map(|&(_, value)| value + 1)
            .collect::<Vec<_>>(),
    );
    assert!(
        map.iter()
            .map(|(&k, &v)| (k, v))
            .eq(expected.into_iter().map(|(k, v)| (k, v + 2)))
    );
}

fn check_direct_map<const PAGE: usize, D: Directory>(map: &mut PagedDirectMap<Gen, i32, PAGE, D>)
where
    D::Store<Slot<Gen, i32>, PAGE>: Sync,
{
    let expected: Vec<_> = map.iter().map(|(key, &value)| (key, value)).collect();
    assert_eq!(
        map.par_iter()
            .map(|(key, &value)| (key, value))
            .collect::<Vec<_>>(),
        expected,
    );
    assert_eq!(
        map.par_iter_mut()
            .map(|(key, value)| (key, bump(value)))
            .collect::<Vec<_>>(),
        expected,
    );
    assert!(
        map.iter()
            .map(|(k, &v)| (k, v))
            .eq(expected.into_iter().map(|(k, v)| (k, v + 1)))
    );
}

#[test]
fn directory_backed_parallel_iteration() {
    POOL.install(|| {
        check_directory::<1, Flat>();
        check_directory::<1, Radix>();
        check_directory::<1, Tree>();
        check_directory::<3, Flat>();
        check_directory::<3, Radix>();
        check_directory::<3, Tree>();
    });
}

fn check_bit_set<const PAGE: usize, D: Directory>() {
    let mut set = PagedBitSet::<usize, PAGE, D>::new();
    let page_bits = PAGE * 64;
    for page in (0..8).rev() {
        let start = page * page_bits;
        set.insert_range(start..=start + 6);
        set.insert(start + page_bits - 1);
    }
    set.remove_range(2 * page_bits..3 * page_bits);
    assert_eq!(set.allocated_pages(), 8);
    assert_eq!(
        (&set).into_par_iter().collect::<Vec<_>>(),
        set.iter().collect::<Vec<_>>(),
    );
    assert_eq!((&set).into_par_iter().count(), set.len());
    set.clear();
    assert!((&set).into_par_iter().collect::<Vec<_>>().is_empty());
}

#[test]
fn bit_set_parallel_iteration() {
    POOL.install(|| {
        check_bit_set::<1, Flat>();
        check_bit_set::<3, Radix>();
        check_bit_set::<8, Tree>();
    });
}
