//! Raw-page round trips preserve live entries and the history of stale ids.

#![cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU32;

use paged_collections::hive::{Hive, RawPage, RawPagesError, RawSlot};
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
    if cfg!(miri) { 12 } else { 64 }
}

fn snapshot<const PAGE: usize>(hive: &Hive<i32, PAGE>) -> Vec<RawPage<i32>> {
    let mut pages = hive.raw_pages();
    let count = hive.index_bound().div_ceil(PAGE);
    assert_eq!(pages.len(), count);
    assert_eq!(pages.size_hint(), (count, Some(count)));
    let mut raw = Vec::new();
    while let Some(page) = pages.next() {
        raw.push(RawPage {
            floor: page.floor,
            slots: page
                .slots
                .into_iter()
                .map(|slot| match slot {
                    RawSlot::Occupied { version, value } => RawSlot::Occupied {
                        version,
                        value: *value,
                    },
                    RawSlot::Vacant { version } => RawSlot::Vacant { version },
                })
                .collect(),
        });
        assert_eq!(pages.len(), count - raw.len());
    }
    assert_eq!(raw.len(), count);
    raw
}

#[derive(Clone, Debug)]
enum Op {
    Insert(i32),
    Remove(u16),
    Shrink,
    Clear,
    Retain(u8),
    Reserve(u8),
}

fn operation() -> impl Strategy<Value = Op> {
    prop_oneof![
        5 => any::<i32>().prop_map(Op::Insert),
        3 => any::<u16>().prop_map(Op::Remove),
        2 => Just(Op::Shrink),
        1 => Just(Op::Clear),
        2 => any::<u8>().prop_map(Op::Retain),
        1 => (0u8..12).prop_map(Op::Reserve),
    ]
}

fn assert_state<const PAGE: usize>(
    hive: &Hive<i32, PAGE>,
    live: &BTreeMap<Id, i32>,
    dead: &BTreeSet<Id>,
) {
    let actual: BTreeMap<_, _> = hive.iter().map(|(id, &value)| (id, value)).collect();
    assert_eq!(&actual, live);
    assert_eq!(hive.len(), live.len());
    for (&id, value) in live {
        assert_eq!(hive.get(id), Some(value));
        assert_eq!(hive.occupant(id.index()), Some(id));
    }
    for &id in dead {
        assert!(!hive.contains(id));
        assert_eq!(hive.get(id), None);
    }
}

fn round_trip<const PAGE: usize>(initial: &[i32], operations: &[Op]) {
    let mut hive = Hive::<i32, PAGE>::new();
    let mut live = BTreeMap::new();
    let mut dead = BTreeSet::new();
    for &value in initial {
        live.insert(hive.insert(value), value);
    }
    for op in operations {
        match *op {
            Op::Insert(value) => {
                let id = hive.insert(value);
                assert!(!dead.contains(&id));
                assert_eq!(live.insert(id, value), None);
            }
            Op::Remove(raw) => {
                if !live.is_empty() {
                    let id = *live.keys().nth(usize::from(raw) % live.len()).unwrap();
                    assert_eq!(hive.remove(id), live.remove(&id));
                    dead.insert(id);
                }
            }
            Op::Shrink => hive.shrink_to_fit(),
            Op::Clear => {
                dead.extend(live.keys().copied());
                live.clear();
                hive.clear();
            }
            Op::Retain(mask) => {
                let keep = |id: Id, value: i32| (id.index() ^ value as u32) & u32::from(mask) == 0;
                hive.retain(|id, value| keep(id, *value));
                live.retain(|&id, value| {
                    if keep(id, *value) {
                        true
                    } else {
                        dead.insert(id);
                        false
                    }
                });
            }
            Op::Reserve(n) => hive.reserve(usize::from(n)),
        }
        assert_state(&hive, &live, &dead);
    }
    let raw = snapshot(&hive);
    let mut rebuilt = Hive::<i32, PAGE>::try_from_raw_pages(raw.clone()).unwrap();
    assert_eq!(rebuilt, hive);
    assert_eq!(rebuilt.len(), hive.len());
    assert_eq!(rebuilt.index_bound(), hive.index_bound());
    assert_eq!(snapshot(&rebuilt), raw);
    assert_state(&rebuilt, &live, &dead);

    // Fill every existing reusable slot and cross into new pages. Reuse
    // order may change across the round trip; no removed id may revive.
    let probes = rebuilt.index_bound() + 2 * PAGE + 3;
    for value in 0..probes {
        let id = rebuilt.insert(value as i32);
        assert!(!dead.contains(&id));
        assert_eq!(live.insert(id, value as i32), None);
        assert_state(&rebuilt, &live, &dead);
    }
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn operation_histories_round_trip(
        initial in prop::collection::vec(any::<i32>(), 0..9),
        operations in prop::collection::vec(operation(), 0..max_len()),
    ) {
        round_trip::<1>(&initial, &operations);
        round_trip::<2>(&initial, &operations);
        round_trip::<3>(&initial, &operations);
        round_trip::<4>(&initial, &operations);
    }
}

#[test]
fn rejects_each_invalid_raw_state() {
    use RawPagesError::*;

    let empty = || RawPage::<i32> {
        floor: 0,
        slots: Vec::new(),
    };
    assert_eq!(
        Hive::<i32, { MAX_INDICES as usize }>::try_from_raw_pages([empty(), empty()]),
        Err(TooManyPages)
    );
    assert_eq!(
        Hive::<i32, 2>::try_from_raw_pages([RawPage {
            floor: 0,
            slots: vec![RawSlot::Vacant { version: 2 }; 3],
        }]),
        Err(PageTooLong)
    );
    // The final page's limit can be less than PAGE. Empty preceding
    // pages avoid allocating enormous slot buffers for this boundary.
    assert_eq!(
        Hive::<i32, { MAX_INDICES as usize - 1 }>::try_from_raw_pages([
            empty(),
            RawPage {
                floor: 0,
                slots: vec![RawSlot::Vacant { version: 2 }; 2],
            },
        ]),
        Err(PageTooLong)
    );
    for floor in [1, u32::MAX] {
        assert_eq!(
            Hive::<i32, 2>::try_from_raw_pages([RawPage { floor, ..empty() }]),
            Err(OddFloor)
        );
    }
    for slot in [
        RawSlot::Occupied {
            version: 0,
            value: 5,
        },
        RawSlot::Occupied {
            version: 2,
            value: 5,
        },
        RawSlot::Vacant { version: 1 },
        RawSlot::Vacant { version: u32::MAX },
    ] {
        assert_eq!(
            Hive::<i32, 2>::try_from_raw_pages([RawPage {
                floor: 0,
                slots: vec![slot],
            }]),
            Err(VersionParity)
        );
    }
    for (error, message) in [
        (TooManyPages, "more pages than slot indices"),
        (PageTooLong, "page holds too many slots"),
        (OddFloor, "odd page floor"),
        (VersionParity, "slot version parity mismatch"),
    ] {
        let error: &dyn std::error::Error = &error;
        assert_eq!(error.to_string(), message);
    }
}

#[test]
fn retired_slots_and_unallocated_floors_survive_repeated_round_trips() {
    let raw = vec![
        RawPage {
            floor: 0,
            slots: vec![
                RawSlot::Vacant { version: 0 },
                RawSlot::Occupied {
                    version: u32::MAX,
                    value: 7,
                },
            ],
        },
        RawPage {
            floor: u32::MAX - 1,
            slots: Vec::new(),
        },
        RawPage {
            floor: 0,
            slots: Vec::new(),
        },
    ];
    let mut hive = Hive::<i32, 2>::try_from_raw_pages(raw.clone()).unwrap();
    assert_eq!(snapshot(&hive), raw);
    let last = |index| Id::new(index, NonZeroU32::new(MAX_GENERATION).unwrap());
    let mut dead = BTreeSet::from([last(0), last(1)]);
    assert_eq!(hive.remove(last(1)), Some(7));
    for _ in 0..3 {
        hive.shrink_to_fit();
        hive = Hive::try_from_raw_pages(snapshot(&hive)).unwrap();
        assert_state(&hive, &BTreeMap::new(), &dead);
        assert_eq!(hive.allocated_pages(), 1);
    }
    let a = hive.insert(10);
    let b = hive.insert(11);
    assert_eq!((a.index(), b.index()), (2, 3));
    assert_eq!(a.generation().get(), MAX_GENERATION);
    assert_eq!(b.generation().get(), MAX_GENERATION);
    dead.extend([a, b]);
    hive.clear();
    hive.shrink_to_fit();
    hive = Hive::try_from_raw_pages(snapshot(&hive)).unwrap();
    assert_eq!(hive.allocated_pages(), 2);
    assert_state(&hive, &BTreeMap::new(), &dead);
    for value in 0..20 {
        assert!(hive.insert(value).index() >= 4);
        for &id in &dead {
            assert!(!hive.contains(id));
        }
    }
}
