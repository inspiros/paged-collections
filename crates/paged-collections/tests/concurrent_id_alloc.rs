//! Threaded ownership checks and comparison with the plain allocator.

#![cfg(all(feature = "concurrent", not(loom)))]
#![cfg(all(
    any(target_pointer_width = "32", target_pointer_width = "64"),
    target_has_atomic = "64"
))]

use std::collections::BTreeSet;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use paged_collections::concurrent::ConcurrentIdAllocator;
use paged_collections::id_alloc::{Id, IdAllocator, MAX_GENERATION};
use proptest::prelude::*;

fn random(state: &mut u64) -> usize {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state as usize
}

#[test]
fn eight_threads_keep_each_index_exclusively_owned() {
    const THREADS: usize = 8;
    let rounds = if cfg!(miri) { 20 } else { 20_000 };
    let max_held = if cfg!(miri) { 4 } else { 32 };
    let ids = Arc::new(ConcurrentIdAllocator::<2>::new());
    // At most one fresh reservation per thread per round. Contending threads
    // reserve fresh indices in blocks of 64 (`index_bound`), so each fresh
    // reservation advances the highest index by at most one block.
    const BLOCK: usize = 64;
    let owned = Arc::new(
        (0..THREADS * rounds * BLOCK)
            .map(|_| AtomicBool::new(false))
            .collect::<Vec<_>>(),
    );
    let mut threads = Vec::new();
    for worker in 0..THREADS {
        let ids = ids.clone();
        let owned = owned.clone();
        threads.push(thread::spawn(move || {
            let mut held = Vec::new();
            let mut state = worker as u64 + 1;
            for _ in 0..rounds {
                let choice = random(&mut state);
                if held.is_empty() || (held.len() < max_held && choice & 1 == 0) {
                    let id = ids
                        .try_allocate()
                        .expect("small test cannot exhaust indices");
                    assert!(!owned[id.index() as usize].swap(true, Ordering::SeqCst));
                    assert!(ids.is_live(id));
                    held.push(id);
                } else {
                    let position = choice % held.len();
                    let id = held.swap_remove(position);
                    assert!(owned[id.index() as usize].swap(false, Ordering::SeqCst));
                    assert!(ids.free(id));
                    assert!(!ids.is_live(id));
                    assert!(!ids.free(id));
                }
            }
            held
        }));
    }
    let held: Vec<_> = threads
        .into_iter()
        .flat_map(|t| t.join().unwrap())
        .collect();
    assert_eq!(ids.len(), held.len());
    assert_eq!(
        owned
            .iter()
            .filter(|flag| flag.load(Ordering::SeqCst))
            .count(),
        held.len()
    );
    for id in held {
        assert!(ids.is_live(id));
        assert!(ids.free(id));
    }
    assert!(ids.is_empty());
}

fn config() -> ProptestConfig {
    ProptestConfig {
        cases: if cfg!(miri) { 4 } else { 128 },
        ..ProptestConfig::default()
    }
}

fn history_len() -> usize {
    if cfg!(miri) { 24 } else { 200 }
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn single_thread_matches_plain_history(
        actions in prop::collection::vec((0u8..5, any::<usize>()), 0..history_len())
    ) {
        let mut plain = IdAllocator::new();
        let mut concurrent = ConcurrentIdAllocator::<2>::new();
        let mut issued = Vec::new();
        let mut dead = BTreeSet::new();
        let mut all = BTreeSet::new();
        for (action, choice) in actions {
            match action {
                0 | 1 => {
                    let expected = plain.alloc();
                    let actual = concurrent.allocate();
                    prop_assert_eq!(actual, expected);
                    prop_assert!(all.insert(actual));
                    issued.push(actual);
                }
                2 if !issued.is_empty() => {
                    let id = issued[choice % issued.len()];
                    let expected = plain.free(id);
                    prop_assert_eq!(concurrent.free(id), expected);
                    if expected {
                        dead.insert(id);
                    }
                }
                3 => {
                    plain.clear();
                    concurrent.clear();
                    dead.extend(issued.iter().copied());
                }
                4 => {
                    let restored = concurrent.into_inner();
                    prop_assert_eq!(&restored, &plain);
                    concurrent = ConcurrentIdAllocator::from(restored);
                }
                _ => {}
            }
            prop_assert_eq!(concurrent.len(), plain.len());
            prop_assert_eq!(concurrent.is_empty(), plain.is_empty());
            prop_assert_eq!(concurrent.index_bound() as usize, plain.index_bound());
            for &id in &issued {
                prop_assert_eq!(concurrent.is_live(id), plain.is_alive(id));
            }
            for &id in &dead {
                prop_assert!(!concurrent.is_live(id));
                prop_assert!(!concurrent.free(id));
            }
        }
        prop_assert_eq!(concurrent.into_inner(), plain);
    }
}

#[test]
fn conversion_preserves_last_generation_retirement_and_free_order() {
    let initial =
        IdAllocator::try_from_raw_parts(vec![u32::MAX, u32::MAX - 1, 0, 2, 1], vec![3, 1]).unwrap();
    let concurrent = ConcurrentIdAllocator::<2>::from(initial.clone());
    assert_eq!(concurrent.into_inner(), initial);
    let mut plain = initial.clone();
    let mut concurrent = ConcurrentIdAllocator::<2>::from(initial);
    let last = Id::new(0, NonZeroU32::new(MAX_GENERATION).unwrap());
    assert!(concurrent.is_live(last));
    assert_eq!(concurrent.free(last), plain.free(last));
    let reused_last = concurrent.allocate();
    assert_eq!(reused_last, plain.alloc());
    assert_eq!(reused_last.index(), 1);
    assert_eq!(reused_last.generation().get(), MAX_GENERATION);
    assert_eq!(concurrent.free(reused_last), plain.free(reused_last));
    let reused = concurrent.allocate();
    assert_eq!(reused, plain.alloc());
    assert_eq!(reused.index(), 3);
    let too_new = Id::new(3, NonZeroU32::new(MAX_GENERATION + 1).unwrap());
    let absent = Id::new(u32::MAX, NonZeroU32::MIN);
    for id in [last, reused_last, too_new, absent] {
        assert!(!concurrent.is_live(id));
        assert!(!concurrent.free(id));
    }
    concurrent.clear();
    plain.clear();
    assert_eq!(concurrent.into_inner(), plain);
}

#[test]
fn empty_default_and_debug() {
    let ids = ConcurrentIdAllocator::<2>::default();
    assert_eq!(ids.index_bound(), 0);
    assert!(ids.is_empty());
    assert!(format!("{ids:?}").contains("ConcurrentIdAllocator"));
    assert_eq!(ids.into_inner(), IdAllocator::new());
}

#[test]
#[should_panic(expected = "invalid page capacity")]
fn zero_page_is_rejected() {
    let _ = ConcurrentIdAllocator::<0>::new();
}

#[test]
fn contended_blocks_leave_no_retired_holes_after_conversion() {
    const THREADS: usize = 8;
    let rounds = if cfg!(miri) { 40 } else { 5_000 };
    let ids = ConcurrentIdAllocator::<64>::new();
    let issued: Vec<Id> = thread::scope(|scope| {
        let workers: Vec<_> = (0..THREADS)
            .map(|_| scope.spawn(|| (0..rounds).map(|_| ids.allocate()).collect::<Vec<_>>()))
            .collect();
        workers
            .into_iter()
            .flat_map(|w| w.join().unwrap())
            .collect()
    });
    let bound = ids.index_bound() as usize;
    // Threads that contend reserve blocks of 64; at most one open per thread.
    assert!((THREADS * rounds..=THREADS * rounds + THREADS * 64).contains(&bound));
    let mut plain = ids.into_inner();
    assert_eq!(plain.len(), THREADS * rounds);
    assert_eq!(plain.index_bound(), bound);
    // Never-issued block indices became free, not retired.
    assert_eq!(plain.retired_len(), 0);
    let holes = plain.free_len();
    assert_eq!(holes, bound - THREADS * rounds);
    assert!(issued.iter().all(|&id| plain.is_alive(id)));
    // Reusing the former holes issues ids no thread ever saw.
    let seen: BTreeSet<Id> = issued.into_iter().collect();
    for _ in 0..holes {
        let id = plain.alloc();
        assert!(!seen.contains(&id));
        assert!((id.index() as usize) < bound);
    }
}
