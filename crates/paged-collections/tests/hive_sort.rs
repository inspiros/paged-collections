//! `Hive::id_of`, `sorted_ids*` and `sort*` against a slot-order oracle.

use paged_collections::hive::Hive;
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
