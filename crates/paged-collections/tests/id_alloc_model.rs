//! A per-slot state oracle, independent of the allocator's version encoding.

#![cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]

use std::collections::{BTreeSet, VecDeque};
use std::iter::FusedIterator;
use std::num::NonZeroU32;
use std::panic::{AssertUnwindSafe, catch_unwind};

use paged_collections::id_alloc::{Id, IdAllocator, MAX_GENERATION};
use paged_collections::sparse_map::PagedSparseMap;
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
    if cfg!(miri) { 24 } else { 200 }
}

fn id(index: u32, generation: u32) -> Id {
    Id::new(index, NonZeroU32::new(generation).unwrap())
}

#[derive(Clone, Copy, Debug)]
enum Slot {
    Live(u32),
    Vacant(u32),
    Retired,
}

#[derive(Clone, Debug, Default)]
struct Model {
    slots: Vec<Slot>,
    free: Vec<u32>,
    len: usize,
    peak_live: usize,
    issued: Vec<Id>,
    ever_issued: BTreeSet<Id>,
    dead: BTreeSet<Id>,
}

impl Model {
    fn occupant(&self, index: u32) -> Option<Id> {
        match self.slots.get(index as usize) {
            Some(&Slot::Live(generation)) => Some(id(index, generation)),
            _ => None,
        }
    }

    fn is_alive(&self, candidate: Id) -> bool {
        self.occupant(candidate.index()) == Some(candidate)
    }

    fn live_ids(&self) -> Vec<Id> {
        (0..self.slots.len())
            .filter_map(|index| self.occupant(index as u32))
            .collect()
    }

    fn retired_len(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| matches!(slot, Slot::Retired))
            .count()
    }

    fn alloc(&mut self) -> Id {
        // A new index is legal only after the entire LIFO stack is drained.
        let next = if let Some(index) = self.free.pop() {
            let Slot::Vacant(generation) = self.slots[index as usize] else {
                panic!("oracle free stack contains a non-vacant slot");
            };
            self.slots[index as usize] = Slot::Live(generation);
            id(index, generation)
        } else {
            let index = self.slots.len() as u32;
            self.slots.push(Slot::Live(1));
            id(index, 1)
        };
        self.len += 1;
        self.peak_live = self.peak_live.max(self.len);
        assert!(self.ever_issued.insert(next), "id issued twice: {next:?}");
        assert!(!self.dead.contains(&next), "stale id revived: {next:?}");
        self.issued.push(next);
        next
    }

    fn free(&mut self, candidate: Id) -> bool {
        if !self.is_alive(candidate) {
            return false;
        }
        let index = candidate.index();
        self.slots[index as usize] = if candidate.generation().get() == MAX_GENERATION {
            Slot::Retired
        } else {
            self.free.push(index);
            Slot::Vacant(candidate.generation().get() + 1)
        };
        self.len -= 1;
        self.dead.insert(candidate);
        true
    }

    fn retain(&mut self, rule: Rule) {
        for candidate in self.live_ids() {
            if !rule.keep(candidate) {
                assert!(self.free(candidate));
            }
        }
    }

    #[cfg(feature = "serde")]
    fn from_slots(slots: Vec<Slot>, free: Vec<u32>) -> Self {
        let mut model = Self {
            slots,
            free,
            ..Self::default()
        };
        model.issued = model.live_ids();
        model.len = model.issued.len();
        model.ever_issued.extend(model.issued.iter().copied());
        // The deserialized snapshot need not contain its historical peak.
        // Every non-retired slot could have been live simultaneously.
        model.peak_live = model.slots.len() - model.retired_len();
        model
    }
}

fn mix(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[derive(Clone, Copy, Debug)]
struct Rule {
    seed: u64,
    divisor: u8,
}

impl Rule {
    fn keep(self, candidate: Id) -> bool {
        mix(candidate.to_bits() ^ self.seed) % u64::from(self.divisor) != 0
    }
}

fn rule() -> impl Strategy<Value = Rule> {
    (any::<u64>(), 1u8..8).prop_map(|(seed, divisor)| Rule { seed, divisor })
}

fn arbitrary_id() -> impl Strategy<Value = Id> {
    (
        prop_oneof![3 => 0u32..64, 1 => Just(u32::MAX), 1 => any::<u32>()],
        prop_oneof![
            3 => 1u32..16,
            1 => Just(MAX_GENERATION),
            1 => Just(MAX_GENERATION + 1),
            1 => Just(u32::MAX),
            1 => (MAX_GENERATION + 1)..=u32::MAX,
            1 => 1u32..=u32::MAX,
        ],
    )
        .prop_map(|(index, generation)| id(index, generation))
}

#[derive(Clone, Debug)]
enum Target {
    Issued(u16),
    Forged(Id),
}

impl Target {
    fn resolve(&self, model: &Model) -> Id {
        match *self {
            Self::Issued(selector) if !model.issued.is_empty() => {
                model.issued[usize::from(selector) % model.issued.len()]
            }
            Self::Issued(_) => id(0, 1),
            Self::Forged(candidate) => candidate,
        }
    }
}

fn target() -> impl Strategy<Value = Target> {
    prop_oneof![
        3 => any::<u16>().prop_map(Target::Issued),
        1 => arbitrary_id().prop_map(Target::Forged),
    ]
}

#[derive(Clone, Debug)]
enum Op {
    Alloc,
    TryAlloc,
    Free(Target),
    IsAlive(Target),
    Occupant(u32),
    Retain(Rule),
    Clear,
    Reserve(u8),
    TryReserve(u8),
    Shrink,
    CloneEq,
}

fn operation() -> impl Strategy<Value = Op> {
    prop_oneof![
        6 => Just(Op::Alloc),
        4 => Just(Op::TryAlloc),
        6 => target().prop_map(Op::Free),
        2 => target().prop_map(Op::IsAlive),
        2 => prop_oneof![0u32..64, any::<u32>()].prop_map(Op::Occupant),
        2 => rule().prop_map(Op::Retain),
        1 => Just(Op::Clear),
        1 => (0u8..16).prop_map(Op::Reserve),
        1 => (0u8..16).prop_map(Op::TryReserve),
        1 => Just(Op::Shrink),
        1 => Just(Op::CloneEq),
    ]
}

fn check_iter(
    mut iter: impl DoubleEndedIterator<Item = Id> + ExactSizeIterator + FusedIterator,
    expected: &[Id],
    mode: u8,
) {
    let mut remaining: VecDeque<_> = expected.iter().copied().collect();
    let mut turn = 0u64;
    while !remaining.is_empty() {
        assert_eq!(iter.len(), remaining.len());
        assert_eq!(iter.size_hint(), (remaining.len(), Some(remaining.len())));
        let back = match mode {
            0 => false,
            1 => true,
            _ => mix(turn) & 1 != 0,
        };
        if back {
            assert_eq!(iter.next_back(), remaining.pop_back());
        } else {
            assert_eq!(iter.next(), remaining.pop_front());
        }
        turn += 1;
    }
    for _ in 0..3 {
        assert_eq!(iter.len(), 0);
        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }
}

fn check(actual: &IdAllocator, model: &Model) {
    assert_eq!(actual.len(), model.len);
    assert_eq!(actual.is_empty(), model.len == 0);
    assert_eq!(actual.index_bound(), model.slots.len());
    assert_eq!(actual.free_len(), model.free.len());
    assert_eq!(actual.retired_len(), model.retired_len());
    assert!(actual.index_bound() <= model.peak_live + model.retired_len());
    for index in 0..model.slots.len() {
        assert_eq!(actual.occupant(index as u32), model.occupant(index as u32));
        for generation in [MAX_GENERATION + 1, u32::MAX] {
            assert!(!actual.is_alive(id(index as u32, generation)));
        }
    }
    assert_eq!(actual.occupant(model.slots.len() as u32), None);
    assert_eq!(actual.occupant(u32::MAX), None);
    for &candidate in &model.issued {
        assert_eq!(actual.is_alive(candidate), model.is_alive(candidate));
    }
    for &stale in &model.dead {
        assert!(!actual.is_alive(stale), "freed id became live: {stale:?}");
    }
    let live = model.live_ids();
    check_iter(actual.iter(), &live, 0);
    check_iter(actual.iter(), &live, 1);
    check_iter(IntoIterator::into_iter(actual), &live, 2);
    let _ = format!("{actual:?}");
}

fn check_future(actual: &IdAllocator, model: &Model) {
    let mut actual = actual.clone();
    let mut model = model.clone();
    // Drain every reusable index and then require fresh indices too.
    for _ in 0..model.free.len() + 3 {
        assert_eq!(actual.alloc(), model.alloc());
        check(&actual, &model);
    }
}

struct Machine {
    actual: IdAllocator,
    model: Model,
}

impl Machine {
    fn new(capacity: usize) -> Self {
        let machine = Self {
            actual: IdAllocator::with_capacity(capacity),
            model: Model::default(),
        };
        assert_eq!(machine.actual, IdAllocator::new());
        assert_eq!(machine.actual, IdAllocator::default());
        check(&machine.actual, &machine.model);
        machine
    }

    fn step(&mut self, operation: &Op) {
        match *operation {
            Op::Alloc => assert_eq!(self.actual.alloc(), self.model.alloc()),
            Op::TryAlloc => assert_eq!(self.actual.try_alloc(), Some(self.model.alloc())),
            Op::Free(ref target) => {
                let candidate = target.resolve(&self.model);
                assert_eq!(self.actual.free(candidate), self.model.free(candidate));
            }
            Op::IsAlive(ref target) => {
                let candidate = target.resolve(&self.model);
                assert_eq!(
                    self.actual.is_alive(candidate),
                    self.model.is_alive(candidate)
                );
            }
            Op::Occupant(index) => {
                assert_eq!(self.actual.occupant(index), self.model.occupant(index));
            }
            Op::Retain(rule) => {
                let expected_visits = self.model.live_ids();
                let mut visits = Vec::new();
                self.actual.retain(|candidate| {
                    visits.push(candidate);
                    rule.keep(candidate)
                });
                assert_eq!(visits, expected_visits);
                self.model.retain(rule);
            }
            Op::Clear => {
                self.actual.clear();
                for candidate in self.model.live_ids() {
                    assert!(self.model.free(candidate));
                }
            }
            Op::Reserve(additional) => self.actual.reserve(usize::from(additional)),
            Op::TryReserve(additional) => {
                self.actual.try_reserve(usize::from(additional)).unwrap();
            }
            Op::Shrink => self.actual.shrink_to_fit(),
            Op::CloneEq => {
                let mut cloned = self.actual.clone();
                assert_eq!(cloned, self.actual);
                cloned.reserve(7);
                cloned.shrink_to_fit();
                assert_eq!(cloned, self.actual);
                check_future(&cloned, &self.model);
                cloned.alloc();
                assert_ne!(cloned, self.actual);
            }
        }
        check(&self.actual, &self.model);
    }

    fn run(&mut self, operations: &[Op]) {
        for operation in operations {
            self.step(operation);
        }
    }
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn operation_sequences_match_model(
        operations in prop::collection::vec(operation(), 0..max_len()),
        capacity in 0usize..32,
    ) {
        let mut machine = Machine::new(capacity);
        machine.run(&operations);
        check_future(&machine.actual, &machine.model);
        #[cfg(feature = "serde")]
        serde_tests::check_round_trip(&machine.actual, &machine.model);
    }

    #[test]
    fn retain_panic_commits_only_completed_visits(
        operations in prop::collection::vec(operation(), 0..max_len() / 2),
        extra_live in 1usize..9,
        panic_at in any::<u16>(),
        rule in rule(),
    ) {
        let mut machine = Machine::new(0);
        machine.run(&operations);
        for _ in 0..extra_live {
            machine.step(&Op::Alloc);
        }
        let before = machine.model.live_ids();
        let stop = usize::from(panic_at) % before.len();
        let mut visits = Vec::new();
        let result = catch_unwind(AssertUnwindSafe(|| {
            machine.actual.retain(|candidate| {
                visits.push(candidate);
                assert!(visits.len() != stop + 1, "injected retain predicate panic");
                rule.keep(candidate)
            });
        }));
        assert!(result.is_err());
        assert_eq!(visits, before[..=stop]);
        // The panicking call has made no decision; that id remains live.
        for &candidate in &before[..stop] {
            if !rule.keep(candidate) {
                assert!(machine.model.free(candidate));
            }
        }
        check(&machine.actual, &machine.model);
        check_future(&machine.actual, &machine.model);
        // Continue mutating the original, not just a snapshot of it.
        machine.run(&operations);
    }

    #[test]
    fn id_bits_and_index_then_generation_order(
        a in arbitrary_id(),
        b in arbitrary_id(),
        same_index in any::<bool>(),
        bits in any::<u64>(),
    ) {
        let b = if same_index { id(a.index(), b.generation().get()) } else { b };
        for candidate in [a, b] {
            assert_eq!(Id::from_bits(candidate.to_bits()), Some(candidate));
            assert_eq!(candidate.to_bits(), (u64::from(candidate.generation().get()) << 32)
                | u64::from(candidate.index()));
            assert_eq!(Id::from_bits(u64::from(candidate.index())), None);
            #[cfg(feature = "serde")]
            {
                let json = serde_json::to_string(&candidate).unwrap();
                assert_eq!(json, candidate.to_bits().to_string());
                assert_eq!(serde_json::from_str::<Id>(&json).unwrap(), candidate);
                assert!(serde_json::from_str::<Id>(&candidate.index().to_string()).is_err());
            }
        }
        match Id::from_bits(bits) {
            Some(candidate) => {
                assert_ne!(bits >> 32, 0);
                assert_eq!(candidate.index(), bits as u32);
                assert_eq!(candidate.generation().get(), (bits >> 32) as u32);
                assert_eq!(candidate.to_bits(), bits);
            }
            None => assert_eq!(bits >> 32, 0),
        }
        assert_eq!(a.cmp(&b),
            (a.index(), a.generation()).cmp(&(b.index(), b.generation())));
    }

    #[test]
    fn sparse_map_evicts_stale_generations(
        prefix in 0usize..24,
        values in prop::collection::vec(any::<u32>(), 2..max_len()),
    ) {
        let mut ids = IdAllocator::new();
        for _ in 0..prefix {
            ids.alloc();
        }
        let mut map: PagedSparseMap<Id, u32> = PagedSparseMap::new();
        let mut current = ids.alloc();
        assert_eq!(map.insert(current, values[0]), None);
        let mut stale = Vec::new();
        for pair in values.windows(2) {
            let old = current;
            assert!(ids.free(old));
            current = ids.alloc();
            assert_eq!(current.index(), old.index());
            assert_eq!(current.generation().get(), old.generation().get() + 1);
            assert!(!ids.is_alive(old));
            assert_eq!(map.get(current), None);
            assert_eq!(map.get(old), Some(&pair[0]));
            assert_eq!(map.insert(current, pair[1]), Some(pair[0]));
            stale.push(old);
            for &old in &stale {
                assert_eq!(map.get(old), None);
            }
            assert_eq!(map.get(current), Some(&pair[1]));
            assert_eq!(map.len(), 1);
            assert_eq!(ids.index_bound(), prefix + 1);
        }
    }
}

#[test]
fn equality_includes_free_stack_order() {
    let mut a = IdAllocator::new();
    let issued: Vec<_> = (0..3).map(|_| a.alloc()).collect();
    let mut b = a.clone();
    assert!(a.free(issued[0]));
    assert!(a.free(issued[1]));
    assert!(b.free(issued[1]));
    assert!(b.free(issued[0]));
    assert_eq!(a.iter().collect::<Vec<_>>(), b.iter().collect::<Vec<_>>());
    assert_ne!(a, b);
    assert_eq!(a.alloc(), id(1, 2));
    assert_eq!(b.alloc(), id(0, 2));
}

#[cfg(feature = "serde")]
mod serde_tests {
    use super::*;

    pub(super) fn check_round_trip(actual: &IdAllocator, model: &Model) {
        let json = serde_json::to_string(actual).unwrap();
        let mut back: IdAllocator = serde_json::from_str(&json).unwrap();
        assert_eq!(&back, actual);
        check(&back, model);
        let mut original = actual.clone();
        let mut model = model.clone();
        for _ in 0..model.free.len() + 3 {
            let expected = model.alloc();
            assert_eq!(original.alloc(), expected);
            assert_eq!(back.alloc(), expected);
            assert_eq!(back, original);
            check(&back, &model);
        }
    }

    proptest! {
        #![proptest_config(config())]

        #[test]
        fn deserialized_boundary_states_follow_model(
            versions in prop::collection::vec(
                prop_oneof![
                    Just(0u32), Just(1u32), Just(2u32),
                    Just(u32::MAX - 2), Just(u32::MAX - 1), Just(u32::MAX),
                    any::<u32>(),
                ],
                0..if cfg!(miri) { 4 } else { 16 },
            ),
            priorities in prop::collection::vec(any::<u32>(), 16),
            operations in prop::collection::vec(operation(), 0..max_len()),
        ) {
            // Decode only at the serde boundary; model transitions use generations.
            let slots: Vec<_> = versions.iter().map(|&version| {
                if version == 0 {
                    Slot::Retired
                } else if version % 2 == 1 {
                    Slot::Live(version / 2 + 1)
                } else {
                    Slot::Vacant(version / 2 + 1)
                }
            }).collect();
            let mut free: Vec<_> = slots.iter().enumerate().filter_map(|(index, slot)| {
                matches!(slot, Slot::Vacant(_)).then_some(index as u32)
            }).collect();
            free.sort_by_key(|&index| (priorities[index as usize], index));
            let json = serde_json::json!({ "versions": versions, "free": free });
            let actual: IdAllocator = serde_json::from_value(json).unwrap();
            let model = Model::from_slots(slots, free);
            check(&actual, &model);
            check_round_trip(&actual, &model);
            let mut machine = Machine { actual, model };
            machine.run(&operations);
            check_round_trip(&machine.actual, &machine.model);
        }

        #[test]
        fn invalid_free_lists_are_rejected(generation in 2u32..=MAX_GENERATION, index in 0usize..16) {
            let mut versions = vec![1u32; index + 3];
            versions[index] = 2 * (generation - 1);
            versions[index + 1] = 0;
            let valid = serde_json::json!({ "versions": versions, "free": [index] });
            assert!(serde_json::from_value::<IdAllocator>(valid).is_ok());
            for free in [
                vec![index, index + 2], // live
                vec![index, index + 1], // retired
                vec![index, versions.len()], // out of range
                vec![index, index], // duplicate
                vec![], // missing vacant slot
            ] {
                let bad = serde_json::json!({ "versions": versions, "free": free });
                assert!(serde_json::from_value::<IdAllocator>(bad.clone()).is_err(), "{bad}");
            }
        }
    }

    #[test]
    fn last_generation_is_retired_and_never_reissued() {
        let actual = serde_json::from_str::<IdAllocator>(
            r#"{"versions":[4294967295,4294967294,5,0],"free":[1]}"#,
        )
        .unwrap();
        let model = Model::from_slots(
            vec![
                Slot::Live(MAX_GENERATION),
                Slot::Vacant(MAX_GENERATION),
                Slot::Live(3),
                Slot::Retired,
            ],
            vec![1],
        );
        let mut machine = Machine { actual, model };
        check(&machine.actual, &machine.model);
        let last = id(0, MAX_GENERATION);
        assert!(machine.actual.is_alive(last));
        machine.step(&Op::Free(Target::Forged(last)));
        assert_eq!(machine.actual.retired_len(), 2);
        assert_eq!(machine.actual.occupant(0), None);
        assert!(!machine.actual.is_alive(last));
        machine.step(&Op::TryAlloc); // vacant index 1 reaches its last generation
        assert_eq!(machine.actual.occupant(1), Some(id(1, MAX_GENERATION)));
        machine.step(&Op::Free(Target::Forged(id(1, MAX_GENERATION))));
        assert_eq!(machine.actual.retired_len(), 3);
        assert_eq!(machine.actual.occupant(1), None);
        assert_eq!(machine.actual.free_len(), 0);
        for generation in 1..=16 {
            machine.step(&Op::Alloc);
            assert_eq!(machine.actual.occupant(4), Some(id(4, generation)));
            machine.step(&Op::Free(Target::Forged(id(4, generation))));
            for index in [0, 1, 3] {
                assert_eq!(machine.actual.occupant(index), None);
            }
            assert_eq!(machine.actual.retired_len(), 3);
        }
        machine.step(&Op::Clear);
        machine.step(&Op::Shrink);
        check_round_trip(&machine.actual, &machine.model);
    }
}
