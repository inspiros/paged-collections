//! `Hive::id_of`, `sorted_ids*` and `sort*` against a slot-order oracle.

use paged_collections::hive::{Hive, RawPage, RawSlot};
use paged_collections::id_alloc::MAX_GENERATION;
use proptest::prelude::*;

/// Builds a hive with holes: inserts every value, then removes the ones
/// whose flag is set (so live slots are not contiguous).
fn build(values: &[(u8, bool)]) -> Hive<(u8, usize), 4> {
    let mut hive = Hive::new();
    let ids: Vec<_> = values
        .iter()
        .enumerate()
        .map(|(i, &(v, _))| hive.insert((v, i)))
        .collect();
    for (id, &(_, gone)) in ids.iter().zip(values) {
        if gone {
            hive.remove(*id);
        }
    }
    hive
}

proptest! {
    #[test]
    fn sort_is_stable_and_keeps_slots(values in prop::collection::vec((0_u8..8, any::<bool>()), 0..60)) {
        let mut hive = build(&values);
        let ids: Vec<_> = hive.ids().collect();
        let mut expected: Vec<_> = hive.values().copied().collect();
        expected.sort_by_key(|&(v, _)| v); // stable: ties keep slot order
        // sorted_ids names the same order without moving anything.
        let by_id: Vec<_> = hive.sorted_ids_by_key(|&(v, _)| v).iter().map(|&id| hive[id]).collect();
        prop_assert_eq!(&by_id, &expected);
        hive.sort_by_key(|&(v, _)| v);
        prop_assert_eq!(hive.ids().collect::<Vec<_>>(), ids.clone());
        prop_assert_eq!(hive.values().copied().collect::<Vec<_>>(), expected.clone());
        for (id, value) in ids.iter().zip(&expected) {
            prop_assert_eq!(&hive[*id], value);
        }
        prop_assert_eq!(hive.len(), expected.len());
    }

    #[test]
    fn unstable_and_cached_sorts_match_std(values in prop::collection::vec((0_u8..8, any::<bool>()), 0..60)) {
        let base = build(&values);
        let ids: Vec<_> = base.ids().collect();
        let mut stable: Vec<_> = base.values().copied().collect();
        stable.sort_by_key(|&(v, _)| v);
        let mut whole = stable.clone();
        whole.sort();

        let mut cached = base.clone();
        let mut calls = 0;
        cached.sort_by_cached_key(|&(v, _)| { calls += 1; v });
        prop_assert_eq!(calls, stable.len());
        prop_assert_eq!(cached.values().copied().collect::<Vec<_>>(), stable.clone());

        // Unstable: only the key sequence is fixed; the multiset is kept.
        let mut by_key = base.clone();
        by_key.sort_unstable_by_key(|&(v, _)| v);
        let keys = |h: &Hive<(u8, usize), 4>| h.values().map(|&(v, _)| v).collect::<Vec<_>>();
        prop_assert_eq!(keys(&by_key), stable.iter().map(|&(v, _)| v).collect::<Vec<_>>());
        let mut sorted_back: Vec<_> = by_key.values().copied().collect();
        sorted_back.sort();
        prop_assert_eq!(&sorted_back, &whole);

        // Elements are distinct (slot number inside), so sort_unstable is exact.
        let mut full = base.clone();
        full.sort_unstable();
        prop_assert_eq!(full.values().copied().collect::<Vec<_>>(), whole.clone());
        let mut reversed = base.clone();
        reversed.sort_unstable_by(|a, b| b.cmp(a));
        prop_assert_eq!(reversed.values().copied().collect::<Vec<_>>(), whole.iter().rev().copied().collect::<Vec<_>>());
        for hive in [&cached, &by_key, &full, &reversed] {
            prop_assert_eq!(hive.ids().collect::<Vec<_>>(), ids.clone());
        }
    }

    #[test]
    fn id_of_inverts_get(values in prop::collection::vec((any::<u8>(), any::<bool>()), 0..60)) {
        let hive = build(&values);
        for (id, value) in hive.iter() {
            prop_assert_eq!(hive.id_of(value), Some(id));
        }
        let other = build(&values);
        for value in other.values() {
            prop_assert_eq!(hive.id_of(value), None);
        }
        prop_assert_eq!(hive.id_of(&(0, 0)), None);
    }
}

#[test]
fn id_of_rejects_copies_and_handles_zero_sized_elements() {
    let mut hive: Hive<(u64, u64), 4> = Hive::new();
    let a = hive.insert((1, 2));
    let b = hive.insert((3, 4));
    assert_eq!(hive.id_of(&hive[a]), Some(a));
    let copy = hive[b];
    assert_eq!(hive.id_of(&copy), None);
    hive.remove(b);
    assert_eq!(hive.id_of(&copy), None);
    assert_eq!(hive.iter().count(), 1);

    let mut units: Hive<(), 2> = Hive::new();
    let ids: Vec<_> = (0..5).map(|_| units.insert(())).collect();
    for &id in &ids {
        assert_eq!(units.id_of(&units[id]), Some(id));
    }
}

fn check_id_of<T: Copy, const PAGE: usize>(hive: &Hive<T, PAGE>, other: &Hive<T, PAGE>) {
    for (id, value) in hive.iter() {
        let found = hive.id_of(value).expect("live address must have an id");
        assert_eq!(found, id);
        assert!(std::ptr::eq(hive.get(found).unwrap(), value));
        let copy = *value;
        assert_eq!(hive.id_of(&copy), None);
    }
    for value in other.values() {
        assert_eq!(hive.id_of(value), None);
    }
}

fn with_empty_pages() -> Hive<(u8, usize), 4> {
    let values: Vec<_> = (0..16).map(|i| (i, i / 4 == 0 || i / 4 == 2)).collect();
    let mut hive = build(&values);
    hive.reserve(12); // Empty interior pages and spare trailing pages.
    hive
}

#[test]
fn id_of_after_shrink_and_page_reallocation() {
    let mut hive = with_empty_pages();
    let other = hive.clone();
    let before: Vec<_> = hive
        .iter()
        .map(|(id, value)| (id, std::ptr::from_ref(value)))
        .collect();
    assert!(hive.allocated_pages() > 2);
    hive.shrink_to_fit();
    assert_eq!(hive.allocated_pages(), 2);
    for (id, address) in before {
        assert!(std::ptr::eq(hive.get(id).unwrap(), address));
    }
    check_id_of(&hive, &other);
    // Recreate the freed pages above their saved generation floors.
    let reused = hive.insert((42, 42));
    assert_eq!(reused.index(), 0);
    assert_eq!(reused.generation().get(), 2);
    hive.extend((0..8).map(|i| (42, i)));
    check_id_of(&hive, &other);
}

#[test]
fn id_of_after_last_generation_retires_a_slot() {
    let mut hive: Hive<(u8, usize), 4> = Hive::try_from_raw_pages([
        RawPage {
            floor: 0,
            slots: vec![
                RawSlot::Occupied {
                    version: u32::MAX,
                    value: (9, 0),
                },
                RawSlot::Occupied {
                    version: 1,
                    value: (7, 1),
                },
            ],
        },
        RawPage {
            floor: 0,
            slots: vec![RawSlot::Occupied {
                version: 1,
                value: (5, 4),
            }],
        },
    ])
    .unwrap();
    let other = hive.clone();
    let last = hive.occupant(0).unwrap();
    assert_eq!(last.generation().get(), MAX_GENERATION);
    let removed = hive.remove(last).unwrap();
    assert_eq!(hive.id_of(&removed), None);
    assert_eq!(hive.get(last), None);
    check_id_of(&hive, &other); // Live elements beside and beyond the retired slot.

    hive.remove(hive.occupant(1).unwrap());
    hive.shrink_to_fit();
    assert_eq!(hive.allocated_pages(), 2); // The retired slot pins empty page 0.
    check_id_of(&hive, &other);
    for i in 0..8 {
        let fresh = hive.insert((3, i));
        assert_ne!(fresh.index(), 0);
        assert_eq!(hive.occupant(0), None);
        check_id_of(&hive, &other);
    }
    assert_eq!(hive.id_of(&removed), None);
}

#[test]
fn id_of_after_clone_and_from_iter() {
    let mut original = with_empty_pages();
    original.shrink_to_fit();
    let reused = original.insert((42, 42));
    assert_eq!(reused.generation().get(), 2);
    let cloned = original.clone();
    let collected: Hive<_, 4> = original.values().copied().collect();
    assert_eq!(cloned, original); // Clone preserves ids and holes.
    assert_eq!(
        collected.values().copied().collect::<Vec<_>>(),
        original.values().copied().collect::<Vec<_>>()
    ); // FromIterator starts a fresh allocation history.
    check_id_of(&cloned, &original);
    check_id_of(&collected, &original);
    check_id_of(&original, &cloned);
    check_id_of(&original, &collected);
}

#[cfg(feature = "serde")]
#[test]
fn id_of_after_deserialization() {
    let mut original = with_empty_pages();
    original.shrink_to_fit();
    original.insert((42, 42));
    let wire = serde_json::to_string(&original).unwrap();
    let rebuilt: Hive<(u8, usize), 4> = serde_json::from_str(&wire).unwrap();
    assert_eq!(rebuilt, original);
    check_id_of(&rebuilt, &original);
    check_id_of(&original, &rebuilt);
}

#[test]
fn sort_with_panicking_comparator_moves_nothing() {
    let mut hive: Hive<u32, 4> = [5, 3, 9, 1].into_iter().collect();
    let before: Vec<_> = hive.iter().map(|(id, &v)| (id, v)).collect();
    let mut calls = 0;
    let result = std::panic::catch_unwind(core::panic::AssertUnwindSafe(|| {
        hive.sort_by(|a, b| {
            calls += 1;
            assert!(calls < 3, "comparator panics");
            a.cmp(b)
        })
    }));
    assert!(result.is_err());
    let after: Vec<_> = hive.iter().map(|(id, &v)| (id, v)).collect();
    assert_eq!(after, before);
    hive.sort();
    assert_eq!(hive.values().copied().collect::<Vec<_>>(), [1, 3, 5, 9]);
    assert_eq!(
        hive.sorted_ids()
            .iter()
            .map(|&id| hive[id])
            .collect::<Vec<_>>(),
        [1, 3, 5, 9]
    );
}
