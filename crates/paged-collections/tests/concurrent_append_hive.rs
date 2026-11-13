//! Shared publication, exclusive mutation, conversions and a plain-hive oracle.

#![cfg(all(feature = "concurrent", not(loom)))]
#![cfg(all(
    any(target_pointer_width = "32", target_pointer_width = "64"),
    target_has_atomic = "ptr",
    target_has_atomic = "64"
))]

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU32;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;

use paged_collections::concurrent::{AppendDrain, AppendIter, AppendIterMut, ConcurrentAppendHive};
use paged_collections::hive::{Hive, RawPage, RawSlot};
use paged_collections::id_alloc::{Id, MAX_GENERATION};
use proptest::prelude::*;

// Type-checked without running. Multiple applicable marker implementations
// make `_` ambiguous if a forbidden auto trait appears.
const _: fn() = || {
    fn send<T: Send>() {}
    fn sync<T: Sync>() {}
    send::<ConcurrentAppendHive<Cell<u8>, 4>>();
    send::<ConcurrentAppendHive<u8, 4>>();
    sync::<ConcurrentAppendHive<u8, 4>>();
    send::<AppendIter<'static, u8, 4>>();
    sync::<AppendIter<'static, u8, 4>>();
    send::<AppendIterMut<'static, u8, 4>>();
    sync::<AppendIterMut<'static, u8, 4>>();
    send::<AppendDrain<'static, u8, 4>>();
    sync::<AppendDrain<'static, u8, 4>>();

    trait AmbiguousSend<A> {
        fn check() {}
    }
    impl<T: ?Sized> AmbiguousSend<()> for T {}
    impl<T: ?Sized + Send> AmbiguousSend<u8> for T {}
    let _ = <ConcurrentAppendHive<Rc<()>, 4> as AmbiguousSend<_>>::check;
    let _ =
        <ConcurrentAppendHive<std::sync::MutexGuard<'static, ()>, 4> as AmbiguousSend<_>>::check;

    trait AmbiguousSync<A> {
        fn check() {}
    }
    impl<T: ?Sized> AmbiguousSync<()> for T {}
    impl<T: ?Sized + Sync> AmbiguousSync<u8> for T {}
    let _ = <ConcurrentAppendHive<Cell<u8>, 4> as AmbiguousSync<_>>::check;
    let _ = <ConcurrentAppendHive<Rc<()>, 4> as AmbiguousSync<_>>::check;
    let _ =
        <ConcurrentAppendHive<std::sync::MutexGuard<'static, ()>, 4> as AmbiguousSync<_>>::check;
};

fn id(index: u32, generation: u32) -> Id {
    Id::new(index, NonZeroU32::new(generation).unwrap())
}

#[test]
fn eight_inserters_publish_to_a_concurrent_reader() {
    const THREADS: usize = 8;
    let rounds = if cfg!(miri) { 12 } else { 1000 };
    let hive = ConcurrentAppendHive::<u32, 4>::new();
    let mut received = BTreeMap::new();
    let (sent, messages) = mpsc::channel();
    thread::scope(|scope| {
        for worker in 0..THREADS {
            let sent = sent.clone();
            let hive = &hive;
            scope.spawn(move || {
                for offset in 0..rounds {
                    let value = (worker * rounds + offset) as u32;
                    let id = hive.insert(value);
                    assert_eq!(hive.get(id), Some(&value));
                    sent.send((id, value)).unwrap();
                }
            });
        }
        drop(sent);
        // Retain ordinary references while other threads continue inserting.
        let mut borrowed = Vec::new();
        for (id, value) in messages {
            let reference = hive.get(id).expect("received id is published");
            assert_eq!(*reference, value);
            borrowed.push((reference, value));
            assert!(received.insert(id, value).is_none());
        }
        for (reference, value) in borrowed {
            assert_eq!(*reference, value);
        }
    });
    assert_eq!(hive.len(), THREADS * rounds);
    assert_eq!(hive.index_bound() as usize, THREADS * rounds);
    let actual: BTreeMap<_, _> = hive.iter().map(|(id, value)| (id, *value)).collect();
    assert_eq!(actual, received);
    let mut contents: Vec<_> = actual.into_values().collect();
    contents.sort_unstable();
    assert_eq!(contents, (0..(THREADS * rounds) as u32).collect::<Vec<_>>());
}

#[test]
fn conversion_round_trip_preserves_removed_reused_and_shrunken_history() {
    let mut plain = Hive::<u32, 4>::with_capacity(32);
    let original: Vec<_> = (0..12).map(|value| plain.insert(value)).collect();
    let mut dead = Vec::new();
    for &old in &original[..8] {
        assert!(plain.remove(old).is_some());
        dead.push(old);
    }
    for value in 20..25 {
        let reused = plain.insert(value);
        if value % 2 == 0 {
            assert!(plain.remove(reused).is_some());
            dead.push(reused);
        }
    }
    plain.shrink_to_fit();
    plain.reserve(40);
    let live: Vec<_> = plain.iter().map(|(id, value)| (id, *value)).collect();
    let append = ConcurrentAppendHive::from(plain);
    assert_eq!(append.len(), live.len());
    assert_eq!(
        append.iter().map(|(id, v)| (id, *v)).collect::<Vec<_>>(),
        live
    );
    for &old in &dead {
        assert!(append.get(old).is_none());
    }
    let mut back = append.into_hive();
    for (id, value) in live {
        assert_eq!(back.get(id), Some(&value));
    }
    for value in 100..160 {
        let new = back.insert(value);
        assert!(!dead.contains(&new));
        for &old in &dead {
            assert!(back.get(old).is_none());
        }
    }
}

fn floor_fixture() -> Hive<u32, 4> {
    Hive::try_from_raw_pages(vec![
        RawPage {
            floor: 0,
            slots: vec![
                RawSlot::Vacant { version: 0 },
                RawSlot::Vacant { version: 4 },
                RawSlot::Occupied {
                    version: 3,
                    value: 11,
                },
            ],
        },
        RawPage {
            floor: 8,
            slots: vec![],
        },
        RawPage {
            floor: 0,
            slots: vec![RawSlot::Occupied {
                version: 1,
                value: 22,
            }],
        },
        RawPage {
            floor: u32::MAX - 1,
            slots: vec![],
        },
        RawPage {
            floor: 0,
            slots: vec![],
        },
    ])
    .unwrap()
}

#[test]
fn conversion_keeps_retirement_and_materializes_nonzero_page_floors() {
    let mut dead = vec![
        id(0, 1),
        id(0, MAX_GENERATION),
        id(1, 1),
        id(1, 2),
        id(2, 1),
    ];
    for index in 4..8 {
        for generation in 1..=4 {
            dead.push(id(index, generation));
        }
    }
    for index in 12..16 {
        dead.push(id(index, 1));
        dead.push(id(index, MAX_GENERATION - 1));
    }
    // Exercise both immediate conversion back and inserts before conversion.
    for insert_first in [false, true] {
        let append = ConcurrentAppendHive::from(floor_fixture());
        assert_eq!(append.index_bound(), 16);
        assert_eq!(append.get(id(2, 2)), Some(&11));
        assert_eq!(append.get(id(8, 1)), Some(&22));
        for &old in &dead {
            assert!(!append.contains(old));
        }
        if insert_first {
            let mut issued = BTreeSet::new();
            for value in 100..120 {
                let new = append.insert(value);
                assert!(issued.insert(new));
                assert!(!dead.contains(&new));
                assert_ne!(new.index(), 0);
                if (4..8).contains(&new.index()) {
                    assert_eq!(new.generation().get(), 5);
                }
                if (12..16).contains(&new.index()) {
                    assert_eq!(new.generation().get(), MAX_GENERATION);
                }
            }
            // Interior floor-zero holes spend never-issued generation 1 so
            // that a popped zero state remains unambiguously retired.
            for index in [3, 9, 10, 11] {
                assert!(issued.contains(&id(index, 2)));
            }
            assert!((4..8).all(|index| issued.contains(&id(index, 5))));
            assert!((12..16).all(|index| issued.contains(&id(index, MAX_GENERATION))));
        }
        let mut back = append.into_hive();
        assert_eq!(back.get(id(2, 2)), Some(&11));
        assert_eq!(back.get(id(8, 1)), Some(&22));
        for value in 200..224 {
            let new = back.insert(value);
            assert!(!dead.contains(&new));
            assert_ne!(new.index(), 0);
            for &old in &dead {
                assert!(!back.contains(old));
            }
        }
        // Retire all final-generation slots, then preserve their zero states.
        back.clear();
        let append = ConcurrentAppendHive::from(back);
        for &old in &dead {
            assert!(!append.contains(old));
        }
        for _ in 0..32 {
            let new = append.insert(0);
            assert_ne!(new.index(), 0);
            assert!(!(12..16).contains(&new.index()));
        }
    }
}

#[test]
fn conversion_leaves_only_a_never_used_suffix_to_fresh_reservation() {
    let plain = Hive::<u32, 4>::with_capacity(32);
    let append = ConcurrentAppendHive::from(plain);
    assert_eq!(append.index_bound(), 0);
    assert_eq!(append.insert(1), id(0, 1));

    let mut plain = Hive::<u32, 4>::with_capacity(32);
    let first = plain.insert(1);
    let append = ConcurrentAppendHive::from(plain);
    assert_eq!(append.index_bound(), 1);
    assert_eq!(append.get(first), Some(&1));
    assert_eq!(append.insert(2), id(1, 1));
    assert_eq!(append.into_hive().get(first), Some(&1));
}

#[derive(Debug)]
struct Counted {
    tag: usize,
    drops: Arc<Vec<AtomicUsize>>,
}

impl Drop for Counted {
    fn drop(&mut self) {
        self.drops[self.tag].fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn each_value_drops_once_across_exclusive_operations_and_conversions() {
    let drops = Arc::new((0..10).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>());
    let value = |tag| Counted {
        tag,
        drops: drops.clone(),
    };
    let mut hive = ConcurrentAppendHive::<Counted, 4>::new();
    let ids: Vec<_> = (0..4).map(|tag| hive.insert(value(tag))).collect();
    let removed = hive.remove(ids[0]).unwrap();
    assert_eq!(drops[0].load(Ordering::SeqCst), 0);
    assert!(hive.remove(ids[0]).is_none());
    drop(removed);
    let mut drain = hive.drain();
    assert_eq!(drain.len(), 3);
    let (_, last) = drain.next_back().unwrap();
    assert_eq!(last.tag, 3);
    drop(drain);
    assert!(hive.is_empty());
    drop(last);
    for tag in 4..6 {
        hive.insert(value(tag));
    }
    hive.clear();
    for tag in 6..9 {
        hive.insert(value(tag));
    }
    hive.retain(|_, value| value.tag % 2 == 1);
    let last = hive.insert(value(9));
    assert_eq!(hive.len(), 2);
    let plain = hive.into_hive();
    assert_eq!(plain.get(last).unwrap().tag, 9);
    assert_eq!(drops[7].load(Ordering::SeqCst), 0);
    assert_eq!(drops[9].load(Ordering::SeqCst), 0);
    let hive = ConcurrentAppendHive::from(plain);
    assert_eq!(hive.get(last).unwrap().tag, 9);
    drop(hive);
    for count in drops.iter() {
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn factory_panics_spend_fresh_reused_and_final_generation_ids() {
    for reused in [false, true] {
        let mut hive = ConcurrentAppendHive::<Id, 4>::new();
        if reused {
            let old = hive.insert_with(|id| id);
            assert_eq!(hive.remove(old), Some(old));
        }
        let attempted = Cell::new(None);
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                hive.insert_with(|id| {
                    attempted.set(Some(id));
                    assert!(hive.get(id).is_none());
                    assert!(!hive.contains(id));
                    assert_eq!(hive.iter().count(), 0);
                    panic!("factory failed");
                });
            }))
            .is_err()
        );
        let spent = attempted.get().unwrap();
        assert!(hive.is_empty());
        let next = hive.try_insert_with(|id| id).unwrap();
        assert_eq!(next.index(), spent.index());
        assert_eq!(next.generation().get(), spent.generation().get() + 1);
        assert!(hive.get(spent).is_none());
        assert_eq!(hive.get(next), Some(&next));
        assert_eq!(hive.len(), 1);
    }

    let plain = Hive::<u8, 1>::try_from_raw_pages([RawPage {
        floor: u32::MAX - 1,
        slots: vec![],
    }])
    .unwrap();
    let hive = ConcurrentAppendHive::from(plain);
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            hive.insert_with(|last| {
                assert_eq!(last, id(0, MAX_GENERATION));
                panic!("last factory failed");
            });
        }))
        .is_err()
    );
    assert!(hive.is_empty());
    assert_eq!(hive.insert(1), id(1, 1));
    assert!(hive.get(id(0, MAX_GENERATION)).is_none());
}

struct Bomb {
    panic: bool,
    tag: usize,
    drops: Arc<Vec<AtomicUsize>>,
}

impl Drop for Bomb {
    fn drop(&mut self) {
        self.drops[self.tag].fetch_add(1, Ordering::SeqCst);
        assert!(!self.panic, "value destructor failed");
    }
}

#[test]
fn retain_and_drop_remain_consistent_when_user_code_panics() {
    let mut hive = ConcurrentAppendHive::<u32, 4>::new();
    let ids: Vec<_> = (0..4).map(|value| hive.insert(value)).collect();
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            hive.retain(|_, value| {
                assert_ne!(*value, 2, "predicate failed");
                *value == 1
            });
        }))
        .is_err()
    );
    assert_eq!(hive.len(), 3);
    assert!(!hive.contains(ids[0]));
    assert_eq!(hive.iter().count(), 3);

    for drop_hive in [false, true] {
        let drops = Arc::new((0..3).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>());
        let mut hive = ConcurrentAppendHive::<Bomb, 4>::new();
        let ids: Vec<_> = (0..3)
            .map(|tag| {
                hive.insert(Bomb {
                    panic: tag == 0,
                    tag,
                    drops: drops.clone(),
                })
            })
            .collect();
        if drop_hive {
            assert!(catch_unwind(AssertUnwindSafe(|| drop(hive))).is_err());
        } else {
            assert!(catch_unwind(AssertUnwindSafe(|| hive.retain(|_, _| false))).is_err());
            assert_eq!(hive.len(), 2);
            assert!(!hive.contains(ids[0]));
            assert_eq!(hive.iter().count(), 2);
            // Reuse is possible even after a destructor unwound.
            let new = hive.insert(Bomb {
                panic: false,
                tag: 0,
                drops: drops.clone(),
            });
            assert_eq!(new.index(), ids[0].index());
            assert_ne!(new, ids[0]);
            hive.clear();
            assert!(hive.is_empty());
            assert_eq!(drops[0].load(Ordering::SeqCst), 2);
        }
        for count in drops.iter().skip(usize::from(!drop_hive)) {
            assert_eq!(count.load(Ordering::SeqCst), 1);
        }
    }
}

#[test]
fn iteration_mutation_and_lookup_edges() {
    let mut hive = ConcurrentAppendHive::<u32, 4>::default();
    assert!(hive.is_empty());
    assert_eq!(hive.index_bound(), 0);
    assert!(format!("{hive:?}").contains("ConcurrentAppendHive"));
    let first = hive.try_insert(1).unwrap();
    let borrowed = &hive[first];
    let address = borrowed as *const u32;
    let mut snapshot = hive.iter();
    for value in 2..10 {
        hive.insert(value);
    }
    assert_eq!(*borrowed, 1);
    assert!(std::ptr::eq(address, &hive[first]));
    assert_eq!(snapshot.size_hint(), (0, Some(1)));
    assert_eq!(snapshot.next(), Some((first, &1)));
    assert!(snapshot.next().is_none());
    assert!(snapshot.next().is_none());

    let ids: Vec<_> = hive.iter().map(|(id, _)| id).collect();
    assert_eq!(hive.remove(ids[2]), Some(3));
    let mut iter = hive.iter_mut();
    assert_eq!(iter.len(), 8);
    let (_, a) = iter.next().unwrap();
    let (_, b) = iter.next().unwrap();
    *a += 10;
    *b += 20;
    for (_, value) in iter {
        *value += 30;
    }
    assert_eq!((*a, *b), (11, 22));
    let before: Vec<_> = hive.iter().map(|(id, value)| (id, *value)).collect();
    assert_eq!(hive.drain().collect::<Vec<_>>(), before);
    assert!(hive.is_empty());
    for old in ids {
        assert!(hive.get(old).is_none());
    }
    for invalid in [id(u32::MAX, 1), id(0, MAX_GENERATION + 1), id(1000, 1)] {
        assert!(!hive.contains(invalid));
        assert!(hive.get(invalid).is_none());
        assert!(hive.get_mut(invalid).is_none());
        assert!(hive.remove(invalid).is_none());
    }
    assert!(catch_unwind(AssertUnwindSafe(|| hive[first])).is_err());
    assert!(hive.iter_mut().next().is_none());
    assert!(hive.drain().next().is_none());
    hive.clear();
}

#[test]
fn eager_drain_leaves_hive_empty_even_when_forgotten() {
    let mut hive = ConcurrentAppendHive::<u32, 4>::new();
    let first = hive.insert(7);
    std::mem::forget(hive.drain());
    assert!(hive.is_empty());
    assert!(hive.get(first).is_none());
    let next = hive.insert(8);
    assert_eq!(next.index(), first.index());
    assert_ne!(next, first);
}

#[test]
#[should_panic(expected = "invalid page capacity")]
fn zero_page_is_rejected() {
    let _ = ConcurrentAppendHive::<u8, 0>::new();
}

fn config() -> ProptestConfig {
    ProptestConfig {
        cases: if cfg!(miri) { 4 } else { 128 },
        ..ProptestConfig::default()
    }
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn sequential_operations_match_plain_hive(
        actions in prop::collection::vec(
            (0u8..12, any::<usize>(), any::<u32>()),
            0..if cfg!(miri) { 20 } else { 200 },
        )
    ) {
        let mut plain = Hive::<u32, 4>::new();
        let mut append = ConcurrentAppendHive::<u32, 4>::new();
        // Allocation orders may differ. Keep the append -> plain handle
        // mapping forever so every removed handle is also checked as stale.
        let mut issued = Vec::new();
        let mut unique = BTreeSet::new();
        for (action, choice, value) in actions {
            let selected = if issued.is_empty() {
                None
            } else {
                Some(issued[choice % issued.len()])
            };
            match action {
                0 | 1 => {
                    let a = append.insert(value);
                    let b = plain.insert(value);
                    prop_assert!(unique.insert(a));
                    issued.push((a, b));
                }
                2 => {
                    if let Some((a, b)) = selected {
                        prop_assert_eq!(append.get(a).copied(), plain.get(b).copied());
                    }
                }
                3 => {
                    if let Some((a, b)) = selected {
                        prop_assert_eq!(append.contains(a), plain.contains(b));
                    }
                }
                4 => {
                    if let Some((a, b)) = selected {
                        prop_assert_eq!(append.remove(a), plain.remove(b));
                    }
                }
                5 => {
                    append.retain(|_, v| { *v = v.wrapping_add(1); *v % 3 != 0 });
                    plain.retain(|_, v| { *v = v.wrapping_add(1); *v % 3 != 0 });
                }
                6 => {
                    append.clear();
                    plain.clear();
                }
                7 => {
                    if let Some((a, b)) = selected {
                        if let Some(v) = append.get_mut(a) { *v = value; }
                        if let Some(v) = plain.get_mut(b) { *v = value; }
                    }
                }
                8 => {
                    let mut a: Vec<_> = append.drain().map(|(_, v)| v).collect();
                    let mut b: Vec<_> = plain.drain().map(|(_, v)| v).collect();
                    a.sort_unstable();
                    b.sort_unstable();
                    prop_assert_eq!(a, b);
                }
                9 => {
                    for (_, v) in append.iter_mut() { *v = v.wrapping_add(1); }
                    for (_, v) in plain.iter_mut() { *v = v.wrapping_add(1); }
                }
                _ => {
                    // Both conversion directions must keep this mapping and
                    // the stale histories even between arbitrary operations.
                    append = ConcurrentAppendHive::from(append.into_hive());
                }
            }
            prop_assert_eq!(append.len(), plain.len());
            prop_assert_eq!(append.is_empty(), plain.is_empty());
            prop_assert_eq!(append.iter().count(), plain.len());
            for &(a, b) in &issued {
                prop_assert_eq!(append.contains(a), plain.contains(b));
                prop_assert_eq!(append.get(a).copied(), plain.get(b).copied());
            }
        }
    }
}
