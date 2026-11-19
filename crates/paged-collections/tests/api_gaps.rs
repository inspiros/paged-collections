//! drain, extract_if, get_disjoint_mut and ordered queries on the sparse set
//! and map, the direct map and the bit set, checked against BTree models.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::panic::{AssertUnwindSafe, catch_unwind};

use paged_collections::bit_set::PagedBitSet;
use paged_collections::direct_map::PagedDirectMap;
use paged_collections::key::Key;
use paged_collections::page::{Directory, Flat, Radix, Tree};
use paged_collections::sparse_map::PagedSparseMap;
use paged_collections::sparse_set::PagedSparseSet;
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

#[derive(Clone, Debug)]
enum Op {
    Insert(u16),
    Remove(u16),
    Clear,
}

fn ops() -> impl Strategy<Value = Vec<Op>> {
    let n = if cfg!(miri) { 24 } else { 120 };
    prop::collection::vec(
        prop_oneof![
            6 => (0u16..80).prop_map(Op::Insert),
            3 => (0u16..80).prop_map(Op::Remove),
            1 => Just(Op::Clear),
        ],
        0..n,
    )
}

fn build<const P: usize, D: Directory>(
    ops: &[Op],
) -> (PagedSparseMap<u16, u32, P, D>, BTreeMap<u16, u32>) {
    let mut map = PagedSparseMap::new();
    let mut model = BTreeMap::new();
    for (step, op) in ops.iter().enumerate() {
        match *op {
            Op::Insert(k) => {
                map.insert(k, step as u32);
                model.insert(k, step as u32);
            }
            Op::Remove(k) => {
                assert_eq!(map.remove(k), model.remove(&k));
            }
            // Leaves stale slots for the ordered walks to skip.
            Op::Clear => {
                map.clear();
                model.clear();
            }
        }
    }
    (map, model)
}

fn check_ordered<const P: usize, D: Directory>(ops: &[Op], lo: u16, hi: u16, pattern: &[bool]) {
    let (map, model) = build::<P, D>(ops);
    let set = map.as_set();

    let all: Vec<(u16, u32)> = model.iter().map(|(&k, &v)| (k, v)).collect();
    let got: Vec<(u16, u32)> = map.range(..).map(|(&k, &v)| (k, v)).collect();
    assert_eq!(got, all);
    let back: Vec<(u16, u32)> = map.range(..).rev().map(|(&k, &v)| (k, v)).collect();
    assert_eq!(back, all.iter().rev().copied().collect::<Vec<_>>());

    let range = lo..hi;
    // BTreeMap::range panics on an inverted range; ours yields nothing.
    let want: Vec<u16> = if lo <= hi {
        model.range(range.clone()).map(|(&k, _)| k).collect()
    } else {
        Vec::new()
    };
    assert_eq!(set.range(range.clone()).copied().collect::<Vec<_>>(), want);
    // Alternate ends to exercise the front and back pages meeting.
    // Every call must match the model, and stay exhausted afterwards.
    let mut it = map.range(range.clone());
    let mut deque: VecDeque<u16> = want.iter().copied().collect();
    for &from_front in pattern.iter().cycle().take(want.len() + 4) {
        let (got, expected) = if from_front {
            (it.next(), deque.pop_front())
        } else {
            (it.next_back(), deque.pop_back())
        };
        assert_eq!(got.map(|(&k, _)| k), expected);
    }
    let inclusive: Vec<u16> = if lo <= hi {
        model.range(lo..=hi).map(|(&k, _)| k).collect()
    } else {
        Vec::new()
    };
    assert_eq!(set.range(lo..=hi).copied().collect::<Vec<_>>(), inclusive);

    assert_eq!(set.first(), model.keys().next());
    assert_eq!(set.last(), model.keys().next_back());
    assert_eq!(map.first_key_value(), model.iter().next());
    assert_eq!(map.last_key_value(), model.iter().next_back());
    let i = usize::from(lo);
    assert_eq!(
        set.next_occupant(i),
        model.range(lo..).next().map(|(k, _)| k)
    );
    assert_eq!(
        set.prev_occupant(i),
        model.range(..=lo).next_back().map(|(k, _)| k)
    );
    assert_eq!(map.next_occupant(i), model.range(lo..).next());
    assert_eq!(map.prev_occupant(i), model.range(..=lo).next_back());

    let (mut popped, mut model2) = (build::<P, D>(ops).0, model.clone());
    while let Some(entry) = if pattern.first() == Some(&true) {
        popped.pop_first()
    } else {
        popped.pop_last()
    } {
        let want = if pattern.first() == Some(&true) {
            model2.pop_first()
        } else {
            model2.pop_last()
        };
        assert_eq!(Some(entry), want);
        for (&k, &v) in &model2 {
            assert_eq!(popped.get(k), Some(&v));
        }
    }
    assert!(model2.is_empty());
}

fn check_removals<const P: usize, D: Directory>(ops: &[Op], modulus: u32, stop: usize) {
    let (map, model) = build::<P, D>(ops);

    // extract_if: removed exactly the matches, rest intact.
    let mut m = build::<P, D>(ops).0;
    let mut out: Vec<(u16, u32)> = m.extract_if(|_, v| *v % modulus == 0).collect();
    out.sort();
    let want: Vec<(u16, u32)> = model
        .iter()
        .filter(|&(_, v)| v % modulus == 0)
        .map(|(&k, &v)| (k, v))
        .collect();
    assert_eq!(out, want);
    assert_eq!(m.len(), model.len() - want.len());
    for (&k, &v) in &model {
        assert_eq!(m.get(k).is_some(), v % modulus != 0);
    }

    // extract_if dropped early: removed ones are gone, the rest stay.
    let mut m = build::<P, D>(ops).0;
    let taken: Vec<(u16, u32)> = m.extract_if(|_, _| true).take(stop).collect();
    assert_eq!(m.len(), model.len() - taken.len());
    for (k, _) in &taken {
        assert!(m.get(*k).is_none());
    }

    // The set's extract_if.
    let mut s: PagedSparseSet<u16, P, D> = map.keys().copied().collect();
    let mut keys: Vec<u16> = s.extract_if(|k| u32::from(*k) % modulus == 0).collect();
    keys.sort();
    let want: Vec<u16> = model
        .keys()
        .copied()
        .filter(|k| u32::from(*k) % modulus == 0)
        .collect();
    assert_eq!(keys, want);
    assert_eq!(s.len(), model.len() - want.len());

    // drain: dense order, everything gone, pages kept.
    let mut m = build::<P, D>(ops).0;
    let dense: Vec<(u16, u32)> = m.iter().map(|(&k, &v)| (k, v)).collect();
    let pages = m.allocated_pages();
    assert_eq!(m.drain().collect::<Vec<_>>(), dense);
    assert!(m.is_empty());
    assert_eq!(m.allocated_pages(), pages);
    for k in model.keys() {
        assert!(m.get(*k).is_none());
    }
    // Dropped early, or leaked: empty either way.
    let mut m = build::<P, D>(ops).0;
    drop(m.drain().take(stop));
    assert!(m.is_empty() && m.first_key_value().is_none());
    let mut s: PagedSparseSet<u16, P, D> = map.keys().copied().collect();
    std::mem::forget(s.drain());
    assert!(s.is_empty() && s.first().is_none());
    s.insert(3);
    assert_eq!(s.as_slice(), &[3]);
}

fn check_disjoint<const P: usize, D: Directory>(ops: &[Op], keys: [u16; 4]) {
    let (mut map, model) = build::<P, D>(ops);
    let present: Vec<u16> = keys
        .iter()
        .copied()
        .filter(|k| model.contains_key(k))
        .collect();
    let unique = present.iter().collect::<BTreeSet<_>>().len() == present.len();
    let bumped = |k: u16, v: u32| {
        if present.contains(&k) {
            u32::from(k) + 1_000_000
        } else {
            v
        }
    };
    if !unique {
        // Only the call itself is inside `catch_unwind`, so no other panic
        // can stand in for the duplicate check.
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                map.get_disjoint_mut(keys);
            }))
            .is_err()
        );
    } else {
        let refs = map.get_disjoint_mut(keys);
        for (r, k) in refs.into_iter().zip(keys) {
            assert_eq!(r.is_some(), model.contains_key(&k));
            if let Some(v) = r {
                *v = bumped(k, 0);
            }
        }
        for (&k, &v) in &model {
            assert_eq!(map.get(k), Some(&bumped(k, v)));
        }
    }

    // The direct map, with the same contents.
    let mut direct: PagedDirectMap<u16, u32, P, D> = model.iter().map(|(&k, &v)| (k, v)).collect();
    if !unique {
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                direct.get_disjoint_mut(keys);
            }))
            .is_err()
        );
    } else {
        let refs = direct.get_disjoint_mut(keys);
        for (r, k) in refs.into_iter().zip(keys) {
            assert_eq!(r.is_some(), model.contains_key(&k));
            if let Some(v) = r {
                *v = bumped(k, 0);
            }
        }
        for (&k, &v) in &model {
            assert_eq!(direct.get(k), Some(&bumped(k, v)));
        }
    }
}

fn check_direct_and_bits<const P: usize, D: Directory>(ops: &[Op], modulus: u32, stop: usize) {
    let (_, model) = build::<P, D>(ops);
    let entries: Vec<(u16, u32)> = model.iter().map(|(&k, &v)| (k, v)).collect();

    let mut m: PagedDirectMap<u16, u32, P, D> = entries.iter().copied().collect();
    let out: Vec<(u16, u32)> = m.extract_if(|_, v| *v % modulus == 0).collect();
    let want: Vec<(u16, u32)> = entries
        .iter()
        .copied()
        .filter(|(_, v)| v % modulus == 0)
        .collect();
    assert_eq!(out, want);
    assert_eq!(m.len(), entries.len() - want.len());
    assert!(m.iter().all(|(_, v)| v % modulus != 0));

    let mut m: PagedDirectMap<u16, u32, P, D> = entries.iter().copied().collect();
    let pages = m.allocated_pages();
    let mut it = m.drain();
    assert_eq!(it.len(), entries.len());
    let first: Vec<(u16, u32)> = it.by_ref().take(stop).collect();
    assert_eq!(first, entries[..stop.min(entries.len())]);
    drop(it);
    assert!(m.is_empty() && m.iter().next().is_none());
    assert_eq!(m.allocated_pages(), pages);

    // Leaked: what was not yielded stays, consistently.
    let mut m: PagedDirectMap<u16, u32, P, D> = entries.iter().copied().collect();
    let mut it = m.drain();
    let gone: Vec<(u16, u32)> = it.by_ref().take(stop).collect();
    std::mem::forget(it);
    assert_eq!(m.len(), entries.len() - gone.len());
    assert_eq!(
        m.iter().map(|(k, &v)| (k, v)).collect::<Vec<_>>(),
        entries[gone.len()..]
    );

    let mut b: PagedBitSet<u16, 1, D> = model.keys().copied().collect();
    let out: Vec<u16> = b.extract_if(|k| u32::from(k) % modulus == 0).collect();
    let want: Vec<u16> = model
        .keys()
        .copied()
        .filter(|k| u32::from(*k) % modulus == 0)
        .collect();
    assert_eq!(out, want);
    assert_eq!(b.len(), model.len() - want.len());
    let kept: Vec<u16> = b.iter().collect();
    assert_eq!(b.drain().collect::<Vec<_>>(), kept);
    assert!(b.is_empty() && b.first().is_none());

    let all: Vec<u16> = model.keys().copied().collect();
    let mut b: PagedBitSet<u16, 1, D> = all.iter().copied().collect();
    let mut it = b.drain();
    assert_eq!(it.len(), all.len());
    assert_eq!(
        it.by_ref().take(stop).collect::<Vec<_>>(),
        all[..stop.min(all.len())]
    );
    drop(it);
    assert!(b.is_empty());
    let mut b: PagedBitSet<u16, 1, D> = all.iter().copied().collect();
    let mut it = b.drain();
    let gone = it.by_ref().take(stop).count();
    std::mem::forget(it);
    assert_eq!(b.iter().collect::<Vec<_>>(), all[gone..]);
    assert_eq!(b.len(), all.len() - gone);
}

macro_rules! for_layouts {
    ($check:ident($($arg:expr),*)) => {
        $check::<4, Flat>($($arg),*);
        $check::<4, Radix>($($arg),*);
        $check::<4, Tree>($($arg),*);
        $check::<1, Flat>($($arg),*);
        $check::<64, Flat>($($arg),*);
    };
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn ordered_queries_match_btree(
        ops in ops(),
        lo in 0u16..90,
        hi in 0u16..90,
        pattern in prop::collection::vec(any::<bool>(), 1..5),
    ) {
        for_layouts!(check_ordered(&ops, lo, hi, &pattern));
    }

    #[test]
    fn drain_and_extract_if_match_btree(ops in ops(), modulus in 1u32..5, stop in 0usize..10) {
        for_layouts!(check_removals(&ops, modulus, stop));
        for_layouts!(check_direct_and_bits(&ops, modulus, stop));
    }

    #[test]
    fn get_disjoint_mut_matches_btree(ops in ops(), keys in prop::array::uniform4(0u16..80)) {
        for_layouts!(check_disjoint(&ops, keys));
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Gen(u32);

impl Key for Gen {
    fn index(self) -> usize {
        (self.0 & 0xffff) as usize
    }
}

#[test]
fn generational_keys_bound_ranges_by_index() {
    let mut set: PagedSparseSet<Gen, 4> = PagedSparseSet::new();
    let old = Gen(7);
    let new = Gen(7 | 1 << 16);
    set.insert(Gen(3));
    set.insert(old);
    set.insert(Gen(12));
    assert_eq!(set.replace(new), Some(old));
    // The bound's generation does not matter, only its index.
    assert_eq!(set.range(old..=old).copied().collect::<Vec<_>>(), [new]);
    assert_eq!(
        set.range(Gen(4)..).copied().collect::<Vec<_>>(),
        [new, Gen(12)]
    );
    assert_eq!(set.next_occupant(4), Some(&new));

    let mut map: PagedSparseMap<Gen, u8, 4> = [(Gen(3), 1), (new, 2)].into_iter().collect();
    // A stale generation finds nothing, the stored one its value.
    assert_eq!(map.get_disjoint_mut([old, new]), [None, Some(&mut 2)]);
}

#[test]
fn extract_if_panic_leaves_a_consistent_map() {
    let mut map: PagedSparseMap<u16, u16, 4> = (0..20).map(|k| (k, k)).collect();
    let result = catch_unwind(AssertUnwindSafe(|| {
        map.extract_if(|&k, _| {
            assert!(k != 9, "boom");
            k % 3 == 0
        })
        .for_each(drop);
    }));
    assert!(result.is_err());
    for (pos, (&k, &v)) in map.iter().enumerate() {
        assert_eq!(k, v);
        assert_eq!(map.position(k), Some(pos));
    }
    assert!(map.get(9).is_some());

    let mut direct: PagedDirectMap<u16, u16, 4> = (0..20).map(|k| (k, k)).collect();
    let result = catch_unwind(AssertUnwindSafe(|| {
        direct
            .extract_if(|&k, _| {
                assert!(k != 9, "boom");
                k % 3 == 0
            })
            .for_each(drop);
    }));
    assert!(result.is_err());
    // 0, 3, 6 were removed before the panic; 9 and later stay.
    assert_eq!(direct.len(), 17);
    assert_eq!(direct.iter().count(), 17);
    assert!(direct.get(9).is_some() && direct.get(12).is_some() && direct.get(6).is_none());
}

#[test]
fn ranges_skip_extreme_and_inverted_bounds() {
    let set: PagedSparseSet<usize, 4, Radix> =
        [0, 5, usize::MAX - 1, usize::MAX].into_iter().collect();
    assert_eq!(
        set.range(usize::MAX..).copied().collect::<Vec<_>>(),
        [usize::MAX]
    );
    assert_eq!(set.range(..1).copied().collect::<Vec<_>>(), [0]);
    assert_eq!(set.last(), Some(&usize::MAX));
    assert_eq!(set.prev_occupant(usize::MAX - 2), Some(&5));
    #[allow(clippy::reversed_empty_ranges)]
    let inverted = set.range(9..3).count();
    assert_eq!(inverted, 0);
    assert_eq!(set.range(5..5).count(), 0);
    assert_eq!(
        set.range((
            std::ops::Bound::Excluded(usize::MAX),
            std::ops::Bound::Unbounded
        ))
        .count(),
        0
    );
}

#[test]
fn sparse_extract_if_tests_each_entry_once_across_a_panic() {
    let mut map: PagedSparseMap<u16, u16, 4> = (0..12).map(|k| (k, k)).collect();
    let mut seen = Vec::new();
    let mut it = map.extract_if(|&k, _| {
        seen.push(k);
        assert!(
            k != 7 || seen.iter().filter(|&&s| s == 7).count() > 1,
            "boom"
        );
        k % 2 == 0
    });
    let mut out = Vec::new();
    loop {
        match catch_unwind(AssertUnwindSafe(|| it.next())) {
            Ok(Some(entry)) => out.push(entry.0),
            Ok(None) => break,
            Err(_) => {}
        }
    }
    let mut tested = seen.clone();
    tested.sort();
    tested.dedup();
    assert_eq!(
        tested.len(),
        seen.len(),
        "an entry was tested twice: {seen:?}"
    );
    assert_eq!(tested, (0..12).collect::<Vec<_>>());
    out.sort();
    assert_eq!(out, [0, 2, 4, 6, 8, 10]);
    assert_eq!(map.len(), 6);
    assert!(map.get(7).is_some());

    let mut set: PagedSparseSet<u16, 4> = (0..12).collect();
    let mut seen = Vec::new();
    let mut it = set.extract_if(|&k| {
        seen.push(k);
        assert!(
            k != 7 || seen.iter().filter(|&&s| s == 7).count() > 1,
            "boom"
        );
        k % 2 == 0
    });
    while !matches!(catch_unwind(AssertUnwindSafe(|| it.next())), Ok(None)) {}

    seen.sort();
    assert_eq!(seen, (0..12).collect::<Vec<_>>());
    assert_eq!(set.len(), 6);
}

#[test]
fn direct_drain_keeps_unreached_entries_when_a_drop_panics() {
    struct Bomb(bool);
    impl Drop for Bomb {
        fn drop(&mut self) {
            if self.0 && !std::thread::panicking() {
                panic!("boom");
            }
        }
    }
    let mut map: PagedDirectMap<u32, Bomb, 4> = PagedDirectMap::new();
    map.insert(1, Bomb(true));
    map.insert(9, Bomb(false));
    map.insert(2000, Bomb(false));
    assert!(catch_unwind(AssertUnwindSafe(|| drop(map.drain()))).is_err());
    assert_eq!(map.len(), 2);
    assert_eq!(map.keys().collect::<Vec<_>>(), [9, 2000]);
    map.drain().for_each(drop);
    assert!(map.is_empty());

    let mut sparse: PagedSparseMap<u32, Bomb, 4> = PagedSparseMap::new();
    sparse.insert(1, Bomb(true));
    sparse.insert(9, Bomb(false));
    assert!(catch_unwind(AssertUnwindSafe(|| drop(sparse.drain()))).is_err());
    assert!(sparse.is_empty() && sparse.key_slice().is_empty() && sparse.value_slice().is_empty());
}

#[test]
fn sparse_drain_from_both_ends_then_leaked() {
    let mut map: PagedSparseMap<u32, u32, 4> = (0..10).map(|k| (k * 7, k)).collect();
    let mut it = map.drain();
    assert_eq!(it.len(), 10);
    assert_eq!(it.next(), Some((0, 0)));
    assert_eq!(it.next_back(), Some((63, 9)));
    assert_eq!(it.len(), 8);
    std::mem::forget(it);
    assert!(map.is_empty() && map.value_slice().is_empty());
    assert!(map.first_key_value().is_none() && map.get(14).is_none());
    map.insert(14, 1);
    assert_eq!(map.iter().collect::<Vec<_>>(), [(&14, &1)]);
}

#[test]
fn bit_drain_spans_pages_up_to_the_last_slot() {
    let keys = [0usize, 63, 64, 70_000, usize::MAX - 1, usize::MAX];
    let mut set: PagedBitSet<usize, 1, Radix> = keys.into_iter().collect();
    let mut it = set.drain();
    assert_eq!(it.by_ref().take(3).collect::<Vec<_>>(), keys[..3]);
    std::mem::forget(it);
    assert_eq!(set.iter().collect::<Vec<_>>(), keys[3..]);
    assert_eq!(set.drain().collect::<Vec<_>>(), keys[3..]);
    assert!(set.is_empty());
}
