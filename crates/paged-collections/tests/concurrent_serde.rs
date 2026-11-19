//! Wire compatibility, snapshot history and shared serialization.

#![cfg(all(feature = "concurrent", feature = "serde", not(loom)))]
#![cfg(all(
    any(target_pointer_width = "32", target_pointer_width = "64"),
    target_has_atomic = "ptr",
    target_has_atomic = "64"
))]

use std::collections::BTreeMap;
use std::num::NonZeroU32;
use std::thread;

use paged_collections::concurrent::{
    AtomicDirectMap, AtomicIdMap, ConcurrentAppendHive, ConcurrentDirectMap, ConcurrentHive,
    ConcurrentIdAllocator, ConcurrentPagedVec, ConcurrentSparseMap, ConcurrentSparseSet,
};
use paged_collections::direct_map::PagedDirectMap;
use paged_collections::hive::{Hive, RawPage, RawSlot};
use paged_collections::id_alloc::{Id, IdAllocator, MAX_GENERATION};
use paged_collections::page::Radix;
use paged_collections::paged_vec::PagedVec;
use paged_collections::sparse_map::PagedSparseMap;
use paged_collections::sparse_set::PagedSparseSet;
use serde::{Serialize, de::DeserializeOwned};

// Successful serialization of this value type proves the containers do
// not require Clone for serde.
#[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
struct Value(u32);

fn json(value: &impl Serialize) -> String {
    serde_json::to_string(value).unwrap()
}

fn from_json<T: DeserializeOwned>(value: &impl Serialize) -> T {
    serde_json::from_str(&json(value)).unwrap()
}

fn cross_round_trip<P, C>(plain: &P, concurrent: &C) -> (P, C)
where
    P: Serialize + DeserializeOwned,
    C: Serialize + DeserializeOwned,
{
    let plain_json = json(plain);
    let concurrent_json = json(concurrent);
    assert_eq!(plain_json, concurrent_json);
    (
        serde_json::from_str(&concurrent_json).unwrap(),
        serde_json::from_str(&plain_json).unwrap(),
    )
}

fn id(index: u32, generation: u32) -> Id {
    Id::new(index, NonZeroU32::new(generation).unwrap())
}

fn assert_json_error<T: DeserializeOwned>(input: &str, expected: &str) {
    let err = serde_json::from_str::<T>(input)
        .err()
        .expect("hostile input should return a serde error");
    assert!(err.to_string().contains(expected), "{err}");
}

#[test]
fn flat_deserialization_rejects_out_of_range_keys() {
    let index = usize::MAX;
    let map = format!(r#"{{"{index}":7}}"#);
    assert_json_error::<ConcurrentSparseSet<usize, 1>>(&format!("[{index}]"), "sparse spine range");
    assert_json_error::<ConcurrentSparseMap<usize, u32, 1>>(&map, "sparse spine range");
    assert_json_error::<ConcurrentDirectMap<usize, u32, 1>>(&map, "direct spine range");
    assert_json_error::<AtomicDirectMap<1>>(&map, "direct spine range");

    // The same index fits the Radix directory.
    let radix: AtomicDirectMap<1, Radix> = serde_json::from_str(&map).unwrap();
    assert_eq!(radix.get(index), Some(7));
}

#[test]
fn atomic_direct_deserialization_rejects_reserved_value() {
    assert_json_error::<AtomicDirectMap<1>>(
        r#"{"0":18446744073709551615}"#,
        "u64::MAX cannot be stored",
    );
}

// On 64-bit targets every u32 id index fits the Flat spine even with PAGE=1.
#[cfg(target_pointer_width = "32")]
#[test]
fn atomic_id_deserialization_rejects_out_of_range_ids() {
    let bits = id(u32::MAX, 1).to_bits();
    let input = format!(r#"{{"{bits}":7}}"#);
    assert_json_error::<AtomicIdMap<1>>(&input, "id index exceeds direct spine range");
}

#[test]
fn vector_and_sparse_prefixes_keep_dense_order() {
    let plain: PagedVec<Value, 2> = [4, 1, 7].map(Value).into_iter().collect();
    let vec: ConcurrentPagedVec<Value, 2> = [4, 1, 7].map(Value).into_iter().collect();
    let (plain, vec) = cross_round_trip(&plain, &vec);
    assert_eq!(plain.get(2), Some(&Value(7)));
    assert_eq!(vec.get(2), Some(&Value(7)));

    let old = id(3, 1);
    let live = id(3, 2);
    let other = id(1, 1);
    let plain: PagedSparseSet<Id, 2, Radix> = [live, other].into_iter().collect();
    let set: ConcurrentSparseSet<Id, 2, Radix> = [live, other].into_iter().collect();
    let (plain, set) = cross_round_trip(&plain, &set);
    assert_eq!(plain.iter().copied().collect::<Vec<_>>(), [live, other]);
    assert!(set.contains(live));
    assert!(!set.contains(old));
    assert_eq!(json(&set.snapshot()), json(&set));
    assert_eq!(json(&set.clone()), json(&set));

    let plain: PagedSparseMap<Id, Value, 2, Radix> =
        [(live, Value(3)), (other, Value(1))].into_iter().collect();
    let map = ConcurrentSparseMap::<Id, Value, 2, Radix>::new();
    assert_eq!(map.try_insert(live, Value(3)), Ok(&Value(3)));
    assert_eq!(map.try_insert(other, Value(1)), Ok(&Value(1)));
    let (plain, map) = cross_round_trip(&plain, &map);
    assert_eq!(plain.get(live), Some(&Value(3)));
    assert_eq!(map.get(&live), Some(&Value(3)));
    assert_eq!(map.get(&old), None);
}

#[test]
fn sparse_deserialization_preserves_duplicate_slot_semantics() {
    let first = id(3, 1);
    let other = id(1, 1);
    let later = id(3, 2);
    let [a, b, c] = [first, other, later].map(Id::to_bits);
    let set: ConcurrentSparseSet<Id, 2> = serde_json::from_str(&format!("[{a},{b},{c}]")).unwrap();
    assert_eq!(set.iter().copied().collect::<Vec<_>>(), [first, other]);

    let map: ConcurrentSparseMap<Id, u32, 2> =
        serde_json::from_str(&format!(r#"{{"{a}":10,"{b}":20,"{c}":30}}"#)).unwrap();
    assert_eq!(
        map.iter()
            .map(|(&key, &value)| (key, value))
            .collect::<Vec<_>>(),
        [(later, 30), (other, 20)]
    );
    assert!(!map.contains_key(&first));
}

#[test]
fn direct_maps_keep_slot_order_and_id_tags() {
    let old = id(5, 1);
    let live = id(5, 2);
    let other = id(1, 1);
    let plain: PagedDirectMap<Id, Value, 3> =
        [(live, Value(5)), (other, Value(1))].into_iter().collect();
    let map: ConcurrentDirectMap<Id, Value, 3> =
        [(old, Value(50)), (live, Value(5)), (other, Value(1))]
            .into_iter()
            .collect();
    let (plain, map) = cross_round_trip(&plain, &map);
    assert_eq!(plain.get(live), Some(&Value(5)));
    assert_eq!(map.get(live).as_deref(), Some(&Value(5)));
    assert!(map.get(old).is_none());

    let entries = [(17, u64::MAX - 1), (1, 0)];
    let plain: PagedDirectMap<usize, u64, 3, Radix> = entries.into_iter().collect();
    let map: AtomicDirectMap<3, Radix> = entries.into_iter().collect();
    let (plain, map) = cross_round_trip(&plain, &map);
    assert_eq!(plain.get(17), Some(&(u64::MAX - 1)));
    assert_eq!(map.get(1), Some(0));
    assert_eq!(json(&map.snapshot()), json(&plain));
    assert_eq!(json(&map.clone()), json(&map));

    // AtomicIdMap has no plain counterpart. Its map format is portable
    // through a BTreeMap, whose Id order also follows slot indices.
    let plain = BTreeMap::from([(live, u32::MAX), (other, 0)]);
    let map: AtomicIdMap<3> = [(old, 10), (live, u32::MAX), (other, 0)]
        .into_iter()
        .collect();
    let (plain, map) = cross_round_trip(&plain, &map);
    assert_eq!(plain.get(&live), Some(&u32::MAX));
    assert_eq!(map.get(live), Some(u32::MAX));
    assert_eq!(map.get(old), None);
    assert_eq!(json(&map.clone()), json(&map));

    // Deserialization follows input order even for different ids at the
    // same index.
    let input = format!(r#"{{"{}":10,"{}":20}}"#, live.to_bits(), old.to_bits());
    let map: AtomicIdMap<3> = serde_json::from_str(&input).unwrap();
    assert_eq!(map.get(old), Some(20));
    assert_eq!(map.get(live), None);
}

#[test]
fn direct_serialization_stops_at_the_first_error() {
    use paged_collections::key::{DirectKey, Key};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static RECONSTRUCTED: AtomicUsize = AtomicUsize::new(0);

    #[derive(Clone, Copy, Eq, PartialEq, serde::Serialize)]
    struct CountedKey(usize);

    impl Key for CountedKey {
        fn index(self) -> usize {
            self.0
        }
    }

    impl DirectKey for CountedKey {
        type Tag = ();

        fn tag(self) {}

        fn from_parts(index: usize, (): ()) -> Self {
            RECONSTRUCTED.fetch_add(1, Ordering::Relaxed);
            Self(index)
        }
    }

    struct FailingValue;

    impl Serialize for FailingValue {
        fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
            Err(serde::ser::Error::custom("first value failed"))
        }
    }

    let map = ConcurrentDirectMap::<CountedKey, FailingValue, 1>::new();
    map.insert(CountedKey(0), FailingValue);
    map.insert(CountedKey(1), FailingValue);
    let err = serde_json::to_string(&map).unwrap_err();
    assert!(err.to_string().contains("first value failed"), "{err}");
    assert_eq!(RECONSTRUCTED.load(Ordering::Relaxed), 1);
}

#[test]
fn value_snapshots_and_clones_match_quiescent_serialization() {
    let vec: ConcurrentPagedVec<u32, 2> = [4, 1, 7].into_iter().collect();
    assert_eq!(json(&vec.snapshot()), json(&vec));
    assert_eq!(json(&vec.clone()), json(&vec));

    let sparse = ConcurrentSparseMap::<usize, u32, 2>::new();
    assert_eq!(sparse.try_insert(4, 1), Ok(&1));
    assert_eq!(sparse.try_insert(1, 7), Ok(&7));
    assert_eq!(json(&sparse.snapshot()), json(&sparse));
    assert_eq!(json(&sparse.clone()), json(&sparse));

    let direct: ConcurrentDirectMap<usize, u32, 2> = [(4, 1), (1, 7)].into_iter().collect();
    assert_eq!(json(&direct.snapshot()), json(&direct));
    assert_eq!(json(&direct.clone()), json(&direct));

    let append = ConcurrentAppendHive::<u32, 2>::new();
    append.insert(4);
    append.insert(1);
    assert_eq!(json(&append.snapshot()), json(&append));
    assert_eq!(json(&append.clone()), json(&append));
}

#[test]
fn both_hives_keep_live_ids_and_retired_slots() {
    fn fixture() -> Hive<Value, 4> {
        Hive::try_from_raw_pages([RawPage {
            floor: 0,
            slots: vec![
                RawSlot::Occupied {
                    version: u32::MAX,
                    value: Value(1),
                },
                RawSlot::Occupied {
                    version: 3,
                    value: Value(20),
                },
            ],
        }])
        .unwrap()
    }

    let last = id(0, MAX_GENERATION);
    let stale = id(1, 1);
    let live = id(1, 2);
    let mut plain = fixture();
    assert_eq!(plain.remove(last), Some(Value(1)));

    let hive: ConcurrentHive<Value, 4> = from_json(&fixture());
    assert!(hive.remove(last));
    let (serialized, hive) = cross_round_trip(&plain, &hive);
    assert_eq!(serialized.get(live), Some(&Value(20)));
    assert_eq!(hive.get(live).as_deref(), Some(&Value(20)));
    assert!(hive.get(last).is_none() && hive.get(stale).is_none());
    assert_ne!(hive.insert(Value(30)).index(), last.index());

    let mut hive: ConcurrentAppendHive<Value, 4> = from_json(&fixture());
    assert_eq!(hive.remove(last), Some(Value(1)));
    let (serialized, hive) = cross_round_trip(&plain, &hive);
    assert_eq!(serialized.get(live), Some(&Value(20)));
    assert_eq!(hive.get(live), Some(&Value(20)));
    assert!(hive.get(last).is_none() && hive.get(stale).is_none());
    assert_ne!(hive.insert(Value(30)).index(), last.index());
}

#[test]
fn allocator_derives_the_free_list_from_emitted_versions() {
    let allocated = IdAllocator::try_from_raw_parts(vec![1, 3, 1, u32::MAX, 3], vec![]).unwrap();
    let ids: ConcurrentIdAllocator<2> = from_json(&allocated);
    assert!(ids.free(id(0, 1)));
    assert!(ids.free(id(2, 1)));
    assert!(ids.free(id(3, MAX_GENERATION)));

    let expected = IdAllocator::try_from_raw_parts(vec![2, 3, 2, 0, 3], vec![2, 0]).unwrap();
    let (mut plain, decoded) = cross_round_trip(&expected, &ids);
    for index in [1, 4] {
        assert!(plain.is_alive(id(index, 2)));
        assert!(decoded.is_live(id(index, 2)));
        assert!(!decoded.is_live(id(index, 1)));
    }
    assert_eq!(plain.retired_len(), 1);
    assert_eq!(json(&ids.snapshot()), json(&expected));
    assert_eq!(json(&ids.clone()), json(&ids));
    assert_eq!(plain.alloc(), id(0, 2));
    assert_eq!(decoded.allocate(), id(0, 2));
    assert_eq!(decoded.allocate(), id(2, 2));
    assert!(!decoded.is_live(id(3, MAX_GENERATION)));
    assert_eq!(ids.allocate(), id(2, 2));
}

#[test]
fn marked_slot_keeps_its_next_even_version_while_a_guard_is_live() {
    let hive = ConcurrentHive::<u32, 2>::new();
    let removed = hive.insert(10);
    let live = hive.insert(20);
    let guard = hive.get(removed).unwrap();
    assert!(hive.remove(removed));
    let snapshot = hive.snapshot();
    let decoded: Hive<u32, 2> = from_json(&hive);
    let mut cloned = hive.clone().into_hive();
    assert_eq!(json(&decoded), json(&snapshot));
    assert_eq!(json(&cloned), json(&snapshot));
    assert_eq!(
        snapshot.raw_pages().next().unwrap().slots[0],
        RawSlot::Vacant { version: 2 }
    );
    assert_eq!(snapshot.get(live), Some(&20));
    assert_eq!(snapshot.get(removed), None);
    assert_eq!(cloned.insert(30), id(0, 2));
    assert_eq!(*guard, 10);
}

#[test]
fn take_pending_is_already_dead_in_the_snapshot() {
    let hive = ConcurrentHive::<u32, 1>::new();
    let removed = hive.insert(10);
    thread::scope(|scope| {
        let guard = hive.get(removed).unwrap();
        let take = scope.spawn(|| hive.take(removed));
        while hive.contains(removed) {
            thread::yield_now();
        }
        let snapshot = hive.snapshot();
        let decoded: Hive<u32, 1> = from_json(&hive);
        drop(guard);
        assert_eq!(take.join().unwrap(), Some(10));
        assert_eq!(json(&decoded), json(&snapshot));
        assert_eq!(
            snapshot.raw_pages().next().unwrap().slots,
            [RawSlot::Vacant { version: 2 }]
        );
    });
}

#[test]
fn fresh_and_reused_busy_slots_serialize_as_vacant() {
    let hive = ConcurrentHive::<u32, 1>::new();
    let append = ConcurrentAppendHive::<u32, 1>::new();
    let old = hive.insert_with(|_| {
        let decoded: Hive<u32, 1> = from_json(&hive);
        assert_eq!(
            decoded.raw_pages().next().unwrap().slots,
            [RawSlot::Vacant { version: 0 }]
        );
        10
    });
    assert!(hive.remove(old));
    let new = hive.insert_with(|_| {
        let snapshot = hive.snapshot();
        assert_eq!(
            snapshot.raw_pages().next().unwrap().slots,
            [RawSlot::Vacant { version: 2 }]
        );
        20
    });
    assert_eq!(new, id(0, 2));
    append.insert_with(|_| {
        let decoded: Hive<u32, 1> = from_json(&append);
        assert_eq!(
            decoded.raw_pages().next().unwrap().slots,
            [RawSlot::Vacant { version: 0 }]
        );
        10
    });
}

#[test]
fn freed_pages_keep_their_generation_floor_in_both_hives() {
    // Vacancies in page zero have different versions. After freeing the
    // page, all of its slots must restart above the larger one.
    let plain = Hive::<u32, 2>::try_from_raw_pages([
        RawPage {
            floor: 0,
            slots: vec![
                RawSlot::Vacant { version: 4 },
                RawSlot::Vacant { version: 2 },
            ],
        },
        RawPage {
            floor: 0,
            slots: vec![RawSlot::Occupied {
                version: 1,
                value: 20,
            }],
        },
    ])
    .unwrap();
    let mut hive: ConcurrentHive<u32, 2> = from_json(&plain);
    let mut append: ConcurrentAppendHive<u32, 2> = from_json(&plain);
    hive.shrink_to_fit();
    append.shrink_to_fit();

    let expected = hive.snapshot();
    assert_eq!(
        expected.raw_pages().next().unwrap(),
        RawPage {
            floor: 4,
            slots: vec![]
        }
    );
    assert_eq!(json(&hive), json(&expected));
    assert_eq!(json(&append), json(&expected));
    assert_eq!(json(&append.snapshot()), json(&expected));

    let verify = |mut decoded: Hive<u32, 2>| {
        assert_eq!(decoded.get(id(2, 1)), Some(&20));
        assert_eq!(decoded.get(id(0, 2)), None);
        assert_eq!(decoded.get(id(1, 1)), None);
        // The plain hive fills an allocated page before recycling page zero.
        assert_eq!(decoded.insert(30), id(3, 1));
        assert_eq!(decoded.insert(40), id(0, 3));
        assert_eq!(decoded.insert(50), id(1, 3));
    };
    verify(from_json(&hive));
    verify(from_json(&append));
    // Converting to concurrent installs the pending page at its floor.
    for mut cloned in [hive.clone().into_hive(), append.clone().into_hive()] {
        assert_eq!(cloned.get(id(2, 1)), Some(&20));
        assert_eq!(cloned.get(id(0, 2)), None);
        assert_eq!(cloned.get(id(1, 1)), None);
        assert_eq!(cloned.insert(30), id(0, 3));
        assert_eq!(cloned.insert(40), id(1, 3));
    }
}

#[cfg(not(miri))]
#[test]
fn hive_serialization_under_shared_insert_and_remove() {
    use std::sync::{
        Barrier,
        atomic::{AtomicU64, Ordering},
    };

    const WRITERS: usize = 4;
    let hive = ConcurrentHive::<u64, 8>::new();
    let anchor = hive.insert_with(Id::to_bits);
    let stale = hive.insert_with(Id::to_bits);
    assert!(hive.remove(stale));
    let current: [AtomicU64; WRITERS] = std::array::from_fn(|_| AtomicU64::new(0));
    let dead: [AtomicU64; WRITERS] = std::array::from_fn(|_| AtomicU64::new(0));
    let start = Barrier::new(WRITERS + 1);
    thread::scope(|scope| {
        for (current, dead) in current.iter().zip(&dead) {
            let (hive, start) = (&hive, &start);
            scope.spawn(move || {
                let mut live = hive.insert_with(Id::to_bits);
                current.store(live.to_bits(), Ordering::Release);
                start.wait();
                for i in 0..4_000 {
                    assert!(hive.remove(live));
                    dead.store(live.to_bits(), Ordering::Release);
                    live = hive.insert_with(Id::to_bits);
                    current.store(live.to_bits(), Ordering::Release);
                    if i % 8 == 0 {
                        thread::yield_now();
                    }
                }
            });
        }
        start.wait();
        for _ in 0..300 {
            let stable_candidates = current
                .each_ref()
                .map(|word| Id::from_bits(word.load(Ordering::Acquire)).unwrap());
            let already_dead = dead
                .each_ref()
                .map(|word| Id::from_bits(word.load(Ordering::Acquire)));
            let encoded = json(&hive);
            let still_live = stable_candidates.map(|id| hive.contains(id));
            let decoded: Hive<u64, 8> = serde_json::from_str(&encoded).unwrap();
            assert_eq!(decoded.get(anchor), Some(&anchor.to_bits()));
            for (id, live) in stable_candidates.into_iter().zip(still_live) {
                if live {
                    assert_eq!(decoded.get(id), Some(&id.to_bits()));
                }
            }
            for id in already_dead.into_iter().flatten().chain([stale]) {
                assert!(decoded.get(id).is_none());
            }
            for (id, payload) in decoded.iter() {
                assert_eq!(*payload, id.to_bits());
            }
            thread::yield_now();
        }
    });
}
