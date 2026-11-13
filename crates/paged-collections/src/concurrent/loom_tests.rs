//! Small, bounded schedule explorations of each shared protocol.

use alloc::boxed::Box;
use alloc::vec::Vec;
use loom::sync::Arc;
use loom::sync::atomic::AtomicBool;
use loom::thread;

use super::ConcurrentIdAllocator;
use super::atomic::{AtomicU32, AtomicUsize, Ordering};
use super::free_stack::FreeStack;
use super::spine::Spine;

fn model(f: impl Fn() + Send + Sync + 'static) {
    let mut builder = loom::model::Builder::new();
    // Bound preemptions, not the number of threads, for normal test runs.
    // LOOM_MAX_PREEMPTIONS can raise the bound for deeper verification.
    if builder.preemption_bound.is_none() {
        builder.preemption_bound = Some(2);
    }
    builder.max_branches = 10_000;
    builder.check(f);
}

struct CountedPage {
    made: Arc<AtomicUsize>,
    dropped: Arc<AtomicUsize>,
}

impl Drop for CountedPage {
    fn drop(&mut self) {
        self.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn spine_same_page_has_one_survivor() {
    model(|| {
        let spine = Arc::new(Spine::<CountedPage, 2>::new());
        let made = Arc::new(AtomicUsize::new(0));
        let dropped = Arc::new(AtomicUsize::new(0));
        let mut threads = Vec::new();
        for _ in 0..2 {
            let spine = spine.clone();
            let made = made.clone();
            let dropped = dropped.clone();
            threads.push(thread::spawn(move || {
                let page = spine.get_or_insert_with(2, || {
                    made.fetch_add(1, Ordering::SeqCst);
                    Box::new(CountedPage { made, dropped })
                });
                assert!(page.made.load(Ordering::SeqCst) > 0);
                page as *const CountedPage as usize
            }));
        }
        let addresses: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
        assert_eq!(addresses[0], addresses[1]);
        assert_eq!(
            made.load(Ordering::SeqCst) - dropped.load(Ordering::SeqCst),
            1
        );
        let mut spine = Arc::try_unwrap(spine).ok().unwrap();
        assert_eq!(spine.iter_mut().map(|(i, _)| i).collect::<Vec<_>>(), [2]);
        assert!(spine.get(6).is_none());
        spine.clear();
        assert!(spine.get(2).is_none());
        drop(spine);
        assert_eq!(made.load(Ordering::SeqCst), dropped.load(Ordering::SeqCst));
    });
}

#[test]
fn spine_exclusive_iteration_crosses_chunk_boundaries() {
    model(|| {
        let mut spine = Spine::<u32, 2>::new();
        for page in [5, 0, 2, 1] {
            assert_eq!(
                *spine.get_or_insert_with(page, || Box::new(page as u32)),
                page as u32
            );
        }
        assert!(spine.get(6).is_none());
        assert!(spine.get(usize::MAX).is_none());
        for (_, value) in spine.iter_mut() {
            *value += 10;
        }
        assert_eq!(
            spine
                .iter_mut()
                .map(|(i, value)| (i, *value))
                .collect::<Vec<_>>(),
            [(0, 10), (1, 11), (2, 12), (5, 15)]
        );
        spine.clear();
        assert_eq!(spine.iter_mut().count(), 0);
        assert_eq!(*spine.get_or_insert_with(0, || Box::new(7)), 7);
    });
}

#[test]
#[should_panic(expected = "Spine: page exceeds directory capacity")]
fn spine_rejects_pages_outside_capacity() {
    model(|| {
        let spine = Spine::<u32, 2>::new();
        spine.get_or_insert_with(6, || panic!("out-of-range factory must not run"));
    });
}

#[test]
fn fresh_reservations_do_not_cross_the_limit() {
    model(|| {
        let stack = Arc::new(FreeStack::new());
        let contender = stack.clone();
        let a = thread::spawn(move || contender.fresh(1));
        let contender = stack.clone();
        let b = thread::spawn(move || contender.fresh(1));
        let results = [a.join().unwrap(), b.join().unwrap()];
        assert_eq!(results.iter().filter(|&&i| i == Some(0)).count(), 1);
        assert_eq!(results.iter().filter(|i| i.is_none()).count(), 1);
        assert_eq!(stack.index_bound(), 1);
        assert_eq!(stack.fresh(1), None);
    });
}

#[test]
fn stack_aba_preserves_exclusive_index_ownership() {
    model(|| {
        let stack = Arc::new(FreeStack::new());
        let links = Arc::new([AtomicU32::new(0), AtomicU32::new(0)]);
        let owned = Arc::new([AtomicBool::new(false), AtomicBool::new(false)]);
        stack.push(0, &links[0]);
        stack.push(1, &links[1]);
        let mut threads = Vec::new();
        for _ in 0..3 {
            let stack = stack.clone();
            let links = links.clone();
            let owned = owned.clone();
            threads.push(thread::spawn(move || {
                if let Some(i) = stack.pop(|i| &links[i as usize]) {
                    let i = i as usize;
                    assert!(!owned[i].swap(true, Ordering::SeqCst));
                    thread::yield_now();
                    assert!(owned[i].swap(false, Ordering::SeqCst));
                    stack.push(i as u32, &links[i]);
                }
            }));
        }
        for thread in threads {
            thread.join().unwrap();
        }
        let first = stack.pop(|i| &links[i as usize]).unwrap();
        let second = stack.pop(|i| &links[i as usize]).unwrap();
        assert_ne!(first, second);
        assert!(stack.pop(|i| &links[i as usize]).is_none());
        assert_eq!(stack.fresh(2), Some(0));
        assert_eq!(stack.fresh(2), Some(1));
        assert_eq!(stack.fresh(2), None);
        assert_eq!(stack.index_bound(), 2);
    });
}

#[test]
fn allocator_threads_own_distinct_live_ids() {
    model(|| {
        let ids = Arc::new(ConcurrentIdAllocator::<2>::new());
        // Seed one reusable slot, so the model also exercises slot reuse.
        assert!(ids.free(ids.allocate()));
        let owned = Arc::new([AtomicBool::new(false), AtomicBool::new(false)]);
        let mut threads = Vec::new();
        for _ in 0..2 {
            let ids = ids.clone();
            let owned = owned.clone();
            threads.push(thread::spawn(move || {
                let id = ids.allocate();
                let flag = &owned[id.index() as usize];
                assert!(!flag.swap(true, Ordering::SeqCst));
                assert!(ids.is_live(id));
                thread::yield_now();
                assert!(flag.swap(false, Ordering::SeqCst));
                assert!(ids.free(id));
                assert!(!ids.is_live(id));
                id
            }));
        }
        let a = threads.remove(0).join().unwrap();
        let b = threads.remove(0).join().unwrap();
        assert_ne!(a, b);
        assert!(ids.is_empty());
    });
}

#[test]
fn exactly_one_double_free_wins() {
    model(|| {
        let ids = Arc::new(ConcurrentIdAllocator::<2>::new());
        let id = ids.allocate();
        let mut threads = Vec::new();
        for _ in 0..2 {
            let ids = ids.clone();
            threads.push(thread::spawn(move || ids.free(id)));
        }
        let wins = threads
            .into_iter()
            .map(|thread| usize::from(thread.join().unwrap()))
            .sum::<usize>();
        assert_eq!(wins, 1);
        assert!(!ids.is_live(id));
        assert!(ids.is_empty());
        let next = ids.allocate();
        assert_eq!(next.index(), id.index());
        assert!(ids.is_live(next));
        assert!(!ids.is_live(id));
    });
}

#[test]
fn stale_id_stays_dead_during_same_index_allocation() {
    model(|| {
        let ids = Arc::new(ConcurrentIdAllocator::<2>::new());
        let stale = ids.allocate();
        assert!(ids.free(stale));
        let allocator = ids.clone();
        let writer = thread::spawn(move || {
            let fresh = allocator.allocate();
            assert_eq!(fresh.index(), stale.index());
            assert_ne!(fresh, stale);
            assert!(allocator.is_live(fresh));
            assert!(allocator.free(fresh));
        });
        let reader = ids.clone();
        let observer = thread::spawn(move || {
            assert!(!reader.is_live(stale));
            thread::yield_now();
            assert!(!reader.is_live(stale));
        });
        writer.join().unwrap();
        observer.join().unwrap();
        assert!(!ids.is_live(stale));
        assert_eq!(ids.len(), 0);
    });
}
