//! Shared append publication, stable slices, exclusive editing and a plain oracle.

#![cfg(all(feature = "concurrent", not(loom)))]
#![cfg(all(
    any(target_pointer_width = "32", target_pointer_width = "64"),
    target_has_atomic = "ptr",
    target_has_atomic = "64"
))]

use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use paged_collections::concurrent::ConcurrentPagedVec;
use paged_collections::paged_vec::PagedVec;
use proptest::prelude::*;

const _: fn() = || {
    fn send<T: Send>() {}
    fn sync<T: Sync>() {}
    send::<ConcurrentPagedVec<Cell<u8>, 2>>();
    sync::<ConcurrentPagedVec<u8, 2>>();
    trait AmbiguousSend<A> {
        fn check() {}
    }
    impl<T> AmbiguousSend<()> for T {}
    impl<T: Send> AmbiguousSend<u8> for T {}
    let _ = <ConcurrentPagedVec<Rc<u8>, 2> as AmbiguousSend<_>>::check;
    trait AmbiguousSync<A> {
        fn check() {}
    }
    impl<T> AmbiguousSync<()> for T {}
    impl<T: Sync> AmbiguousSync<u8> for T {}
    let _ = <ConcurrentPagedVec<Cell<u8>, 2> as AmbiguousSync<_>>::check;
    let _ = <ConcurrentPagedVec<Rc<u8>, 2> as AmbiguousSync<_>>::check;
    let _ = <ConcurrentPagedVec<std::sync::MutexGuard<'static, ()>, 2> as AmbiguousSync<_>>::check;
};

#[test]
fn eight_writers_publish_every_tag_once_and_pages_have_no_holes() {
    const THREADS: usize = 8;
    const PAGE: usize = 7;
    let rounds = if cfg!(miri) { 8 } else { 2000 };
    let vector = Arc::new(ConcurrentPagedVec::<(usize, usize, usize), PAGE>::new());
    let mut workers = Vec::new();
    for worker in 0..THREADS {
        let vector = vector.clone();
        workers.push(thread::spawn(move || {
            let mut positions = Vec::new();
            for sequence in 0..rounds {
                let value = (worker, sequence, worker ^ sequence ^ 0x55);
                let pos = vector.push(value);
                assert_eq!(vector.get(pos), Some(&value));
                positions.push(pos);
                if sequence % 127 == 0 {
                    let pages: Vec<_> = vector.pages().collect();
                    for (p, page) in pages.iter().enumerate() {
                        assert!(!page.is_empty() && page.len() <= PAGE);
                        if p + 1 < pages.len() {
                            assert_eq!(page.len(), PAGE);
                        }
                        for &(w, s, checksum) in *page {
                            assert!(w < THREADS && s < rounds);
                            assert_eq!(checksum, w ^ s ^ 0x55);
                        }
                    }
                }
            }
            positions
        }));
    }
    let mut positions: Vec<_> = workers
        .into_iter()
        .flat_map(|w| w.join().unwrap())
        .collect();
    positions.sort_unstable();
    assert!(positions.into_iter().eq(0..THREADS * rounds));
    assert_eq!(vector.len(), THREADS * rounds);
    assert_eq!(vector.len(), vector.reserved_len());
    let mut tags: Vec<_> = vector
        .pages()
        .flatten()
        .map(|&(w, s, _)| w * rounds + s)
        .collect();
    tags.sort_unstable();
    assert!(tags.into_iter().eq(0..THREADS * rounds));
}

#[test]
fn snapshots_and_existing_addresses_survive_appends() {
    let vector = ConcurrentPagedVec::<Cell<u32>, 4>::new();
    vector.push_all((0..3).map(Cell::new));
    let first_page = vector.pages().next().unwrap();
    let address = first_page.as_ptr();
    let snapshot = vector.pages();
    let iter = vector.iter();
    vector.push_all((3..12).map(Cell::new));
    assert_eq!(snapshot.flatten().count(), 3);
    assert_eq!(iter.count(), 3);
    assert_eq!(first_page.len(), 3);
    assert!(std::ptr::eq(address, vector.get(0).unwrap()));
    // SAFETY: shared appends never move, overwrite or reborrow this written
    // slot mutably; its page remains allocated and vector is still borrowed.
    unsafe { &*address }.set(99);
    assert_eq!(vector[0].get(), 99);
}

#[test]
fn exclusive_views_can_keep_disjoint_references_and_reuse_cleared_bits() {
    let mut vector: ConcurrentPagedVec<i32, 2> = (0..7).collect();
    {
        let mut pages = vector.pages_mut();
        let a = pages.next().unwrap();
        let b = pages.next().unwrap();
        a[0] = 10;
        b[0] = 20;
        for page in pages {
            for value in page {
                *value += 30;
            }
        }
        assert_eq!((a[0], b[0]), (10, 20));
    }
    {
        let mut iter = vector.iter_mut();
        let a = iter.next().unwrap();
        let b = iter.next().unwrap();
        *a += 1;
        *b += 2;
        for value in iter {
            *value += 3;
        }
        assert_eq!((*a, *b), (11, 3));
    }
    vector.truncate(3);
    assert_eq!(vector.pop(), Some(23));
    assert_eq!(vector.get(2), None);
    vector.clear();
    assert_eq!(vector.reserved_len(), 0);
    assert!(vector.pages().next().is_none());
    assert!(vector.iter_mut().next().is_none());
    for pos in 0..10 {
        assert!(vector.get(pos).is_none());
    }
    assert_eq!(vector.push(8), 0);
    assert_eq!(vector.get(1), None);
    vector.extend([9, 10]);
    vector[1] = 90;
    assert_eq!(format!("{vector:?}"), "[8, 90, 10]");
    assert_eq!(
        vector.into_inner().into_iter().collect::<Vec<_>>(),
        [8, 90, 10]
    );
}

struct Counted {
    drops: Arc<AtomicUsize>,
}

impl Drop for Counted {
    fn drop(&mut self) {
        // SeqCst makes the final count independent of the dropping thread.
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn moving_conversions_pop_truncate_clear_and_drop_destroy_every_value_once() {
    let drops = Arc::new(AtomicUsize::new(0));
    let rounds = if cfg!(miri) { 2 } else { 64 };
    let made = 8 * rounds;
    let vector = Arc::new(ConcurrentPagedVec::<Counted, 3>::new());
    let mut workers = Vec::new();
    for _ in 0..8 {
        let vector = vector.clone();
        let drops = drops.clone();
        workers.push(thread::spawn(move || {
            vector.push_all((0..rounds).map(|_| Counted {
                drops: drops.clone(),
            }));
        }));
    }
    for worker in workers {
        worker.join().unwrap();
    }
    let mut vector = Arc::try_unwrap(vector).ok().unwrap();
    drop(vector.pop());
    vector.truncate(made / 2);
    let mut vector = ConcurrentPagedVec::from(vector.into_inner());
    vector.clear();
    vector.push(Counted {
        drops: drops.clone(),
    });
    drop(vector);
    // SeqCst observes all completed destructor increments.
    assert_eq!(drops.load(Ordering::SeqCst), made + 1);
}

struct Bomb {
    panic: bool,
    drops: Arc<AtomicUsize>,
}

impl Drop for Bomb {
    fn drop(&mut self) {
        // SeqCst counts destruction even on the unwinding path.
        self.drops.fetch_add(1, Ordering::SeqCst);
        assert!(!self.panic, "drop bomb");
    }
}

#[test]
fn panicking_truncate_leaves_a_valid_reusable_prefix() {
    let drops = Arc::new(AtomicUsize::new(0));
    let mut vector = ConcurrentPagedVec::<Bomb, 2>::new();
    vector.push(Bomb {
        panic: false,
        drops: drops.clone(),
    });
    vector.push(Bomb {
        panic: true,
        drops: drops.clone(),
    });
    assert!(catch_unwind(AssertUnwindSafe(|| vector.clear())).is_err());
    assert_eq!(vector.len(), 1);
    assert_eq!(vector.reserved_len(), 1);
    assert!(vector.get(1).is_none());
    assert_eq!(
        vector.push(Bomb {
            panic: false,
            drops: drops.clone()
        }),
        1
    );
    drop(vector);
    // SeqCst observes all three destructor calls.
    assert_eq!(drops.load(Ordering::SeqCst), 3);
}

#[test]
fn destruction_continues_across_pages_after_one_drop_panics() {
    let drops = Arc::new(AtomicUsize::new(0));
    let vector = ConcurrentPagedVec::<Bomb, 2>::new();
    for pos in 0..5 {
        vector.push(Bomb {
            panic: pos == 3,
            drops: drops.clone(),
        });
    }
    assert!(catch_unwind(AssertUnwindSafe(|| drop(vector))).is_err());
    // SeqCst observes cleanup of all pages, including those before the bomb.
    assert_eq!(drops.load(Ordering::SeqCst), 5);
}

#[test]
fn zero_sized_values_and_invalid_indices() {
    let mut vector = ConcurrentPagedVec::<(), 3>::default();
    vector.push_all([(); 10]);
    assert_eq!(
        vector.pages().map(<[()]>::len).collect::<Vec<_>>(),
        [3, 3, 3, 1]
    );
    assert_eq!(vector.get(usize::MAX), None);
    assert_eq!(vector.get_mut(usize::MAX), None);
    assert_eq!(vector.iter_mut().count(), 10);
    assert_eq!(vector.pop(), Some(()));
    vector.clear();
    assert!(vector.is_empty());
    assert!(catch_unwind(AssertUnwindSafe(|| vector[0])).is_err());
}

#[test]
fn readiness_spans_bitmap_words_and_is_cleared_before_reuse() {
    let mut vector = ConcurrentPagedVec::<usize, 65>::new();
    vector.push_all(0..130);
    assert_eq!(
        vector.pages().map(<[usize]>::len).collect::<Vec<_>>(),
        [65, 65]
    );
    vector.truncate(63);
    for pos in 63..130 {
        assert_eq!(vector.get(pos), None);
    }
    vector.push_all(1000..1067);
    assert_eq!(vector.len(), 130);
    assert!(vector.iter().copied().eq((0..63).chain(1000..1067)));
    vector.clear();
    assert_eq!(vector.push(9), 0);
    assert_eq!(vector.get(64), None);
    assert_eq!(vector.get(65), None);
    assert_eq!(vector.get(129), None);
}

#[test]
#[should_panic(expected = "invalid page capacity")]
fn zero_page_is_rejected() {
    let _ = ConcurrentPagedVec::<u8, 0>::new();
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
    fn exclusive_operations_match_paged_vec(
        actions in prop::collection::vec((0u8..12, any::<usize>(), any::<i16>()), 0..if cfg!(miri) { 20 } else { 200 })
    ) {
        let mut plain = PagedVec::<i32, 3>::new();
        let mut vector = ConcurrentPagedVec::<i32, 3>::new();
        for (action, choice, value) in actions {
            let value = i32::from(value);
            match action {
                0 | 1 => {
                    prop_assert_eq!(vector.push(value), plain.len());
                    plain.push(value);
                }
                2 => { prop_assert_eq!(vector.pop(), plain.pop()); }
                3 => {
                    let len = choice % (plain.len() + 3);
                    vector.truncate(len); plain.truncate(len);
                }
                4 => { vector.clear(); plain.clear(); }
                5 => {
                    let pos = choice % (plain.len() + 1);
                    if let Some(v) = vector.get_mut(pos) { *v = value; }
                    if let Some(v) = plain.get_mut(pos) { *v = value; }
                }
                6 => {
                    for v in vector.iter_mut() { *v += 1; }
                    for v in plain.iter_mut() { *v += 1; }
                }
                7 => {
                    for page in vector.pages_mut() { for v in page { *v -= 1; } }
                    for page in plain.pages_mut() { for v in page { *v -= 1; } }
                }
                8 => {
                    let moved = vector.into_inner();
                    prop_assert_eq!(&moved, &plain);
                    vector = ConcurrentPagedVec::from(moved);
                }
                9 => { vector.push_all([value, value + 1]); plain.extend([value, value + 1]); }
                10 if !plain.is_empty() => {
                    let pos = choice % plain.len();
                    vector[pos] = value; plain[pos] = value;
                }
                _ => { vector.extend([value]); plain.extend([value]); }
            }
            prop_assert_eq!(vector.len(), plain.len());
            prop_assert_eq!(vector.reserved_len(), plain.len());
            prop_assert_eq!(vector.is_empty(), plain.is_empty());
            prop_assert!(vector.iter().eq(plain.iter()));
            prop_assert!(vector.iter().rev().eq(plain.iter().rev()));
            prop_assert!(vector.pages().eq(plain.pages()));
            prop_assert_eq!(vector.get(plain.len()), None);
        }
        prop_assert_eq!(vector.into_inner(), plain);
    }
}
