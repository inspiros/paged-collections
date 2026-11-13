//! Bounded explorations of page locks, factories and value lifetime.

use loom::sync::Arc;
use loom::sync::atomic::{AtomicUsize, Ordering};
use loom::thread;

use super::ConcurrentDirectMap;

fn model(f: impl Fn() + Send + Sync + 'static) {
    let mut builder = loom::model::Builder::new();
    if builder.preemption_bound.is_none() {
        builder.preemption_bound = Some(2);
    }
    builder.max_branches = 10_000;
    builder.check(f);
}

#[test]
fn two_writers_one_page_and_reader() {
    // Two threads in all: three loom threads spinning on one page lock
    // exceed the branch limit without exercising more of the protocol.
    model(|| {
        let map = Arc::new(ConcurrentDirectMap::<u32, u32, 2>::new());
        let first = map.clone();
        let a = thread::spawn(move || assert_eq!(first.insert(0, 10), None));
        assert_eq!(map.insert(1, 20), None);
        if let Some(value) = map.get(0) {
            assert_eq!(*value, 10);
            assert_eq!(*value.key(), 0);
        }
        a.join().unwrap();
        assert_eq!(map.len(), 2);
        assert_eq!(map.allocated_pages(), 1);
        assert_eq!(map.get_cloned(0), Some(10));
        assert_eq!(map.get_cloned(1), Some(20));
    });
}

#[test]
fn two_readers_then_writer_after_both_release() {
    // Reader count goes 1 -> 2 -> 1 -> 0 in either order; the write must
    // wait for the last reader, whichever it is.
    model(|| {
        let map = Arc::new(ConcurrentDirectMap::<u32, u32, 2>::new());
        map.insert(0, 10);
        let held = map.get(0).unwrap();
        let reader = map.clone();
        let r = thread::spawn(move || {
            // The reader may start after the write; either way the value
            // must not change while its guard is held.
            let guard = reader.get(0).unwrap();
            let first = *guard;
            assert!(first == 10 || first == 30);
            thread::yield_now();
            assert_eq!(*guard, first);
        });
        assert_eq!(*held, 10);
        drop(held);
        *map.get_mut(0).unwrap() = 30;
        r.join().unwrap();
        assert_eq!(map.get_cloned(0), Some(30));
    });
}

#[test]
fn traversal_races_installs_in_old_and_new_chunks() {
    // Loom's spine has BASE = 2: page 1 joins chunk 0, page 2 needs chunk 1.
    model(|| {
        let map = Arc::new(ConcurrentDirectMap::<u32, u32, 1>::new());
        map.insert(0, 100);
        let writer = map.clone();
        let w = thread::spawn(move || {
            writer.insert(1, 101);
            writer.insert(2, 102);
        });
        // Yields cost no preemption and let loom run the writer before and
        // during the traversal; without them every traversal saw only key 0.
        thread::yield_now();
        let mut seen = alloc::vec::Vec::new();
        map.for_each(|&key, &value| {
            assert_eq!(value, 100 + key);
            seen.push(key);
            thread::yield_now();
        });
        assert_eq!(seen.first(), Some(&0));
        assert!(seen.windows(2).all(|pair| pair[0] < pair[1]));
        w.join().unwrap();
        let mut all = alloc::vec::Vec::new();
        map.for_each(|&key, _| all.push(key));
        assert_eq!(all, [0, 1, 2]);
        assert_eq!(map.allocated_pages(), 3);
    });
}

struct Value {
    tag: usize,
    drops: Arc<AtomicUsize>,
}

impl Drop for Value {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn insert_races_remove_with_exact_drops() {
    model(|| {
        let map = Arc::new(ConcurrentDirectMap::<u32, Value, 2>::new());
        let drops = Arc::new(AtomicUsize::new(0));
        let writer = map.clone();
        let counted = drops.clone();
        let a = thread::spawn(move || {
            assert!(
                writer
                    .insert(
                        0,
                        Value {
                            tag: 7,
                            drops: counted
                        }
                    )
                    .is_none()
            );
        });
        let remover = map.clone();
        let b = thread::spawn(move || {
            let removed = remover.remove(0);
            let won = removed.is_some();
            if let Some(value) = removed {
                assert_eq!(value.tag, 7);
            }
            won
        });
        a.join().unwrap();
        let removed = b.join().unwrap();
        assert_eq!(map.len(), usize::from(!removed));
        assert_eq!(map.contains_key(0), !removed);
        assert_eq!(drops.load(Ordering::SeqCst), usize::from(removed));
        if let Some(value) = map.get(0) {
            assert_eq!(value.tag, 7);
        }
        drop(map);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn same_key_factories_run_once() {
    model(|| {
        let map = Arc::new(ConcurrentDirectMap::<u32, usize, 2>::new());
        let calls = Arc::new(AtomicUsize::new(0));
        let first = map.clone();
        let first_calls = calls.clone();
        let a = thread::spawn(move || {
            let value = first.get_or_insert_with(1, || {
                first_calls.fetch_add(1, Ordering::SeqCst);
                42
            });
            assert_eq!(*value, 42);
        });
        let second = map.clone();
        let second_calls = calls.clone();
        let b = thread::spawn(move || {
            let value = second.get_or_insert_with(1, || {
                second_calls.fetch_add(1, Ordering::SeqCst);
                42
            });
            assert_eq!(*value, 42);
        });
        a.join().unwrap();
        b.join().unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(map.len(), 1);
        assert_eq!(map.get_cloned(1), Some(42));
    });
}

#[test]
fn reader_guard_excludes_page_writer() {
    model(|| {
        let map = Arc::new(ConcurrentDirectMap::<u32, u32, 2>::new());
        map.insert(0, 10);
        let guard = map.get(0).unwrap();
        let writer = map.clone();
        let a = thread::spawn(move || {
            // A neighbour shares the same write lock as the guarded key.
            writer.insert(1, 20);
            *writer.get_mut(0).unwrap() = 30;
        });
        assert_eq!(*guard, 10);
        thread::yield_now();
        assert_eq!(*guard, 10);
        drop(guard);
        a.join().unwrap();
        assert_eq!(map.get_cloned(0), Some(30));
        assert_eq!(map.get_cloned(1), Some(20));
        assert_eq!(map.len(), 2);
    });
}

#[test]
fn radix_split_races_install_and_reader() {
    // With loom DIGIT = 2, pages 0 and 16 seed a root at shift 4.
    // Pages 1 and 2 both reach leaf 0: one splits it while the other may
    // still be preparing its install. Failed CAS must descend the new branch.
    // The main-thread yield lets loom schedule those splits before/during reads.
    model(|| {
        let map = Arc::new(ConcurrentDirectMap::<usize, usize, 1, crate::page::Radix>::new());
        map.insert(0, 100);
        map.insert(16, 116);
        let first = map.clone();
        let a = thread::spawn(move || assert_eq!(first.insert(1, 101), None));
        let second = map.clone();
        let b = thread::spawn(move || assert_eq!(second.insert(2, 102), None));
        thread::yield_now();
        assert_eq!(map.get_cloned(0), Some(100));
        for key in [1, 2] {
            if let Some(value) = map.get_cloned(key) {
                assert_eq!(value, 100 + key);
                assert_eq!(map.get_cloned(key), Some(value));
            }
        }
        a.join().unwrap();
        b.join().unwrap();
        for key in [0, 1, 2, 16] {
            assert_eq!(map.get_cloned(key), Some(100 + key));
        }
        assert_eq!(map.len(), 4);
        assert_eq!(map.allocated_pages(), 4);
    });
}
