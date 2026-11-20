//! FIFO modeling, ownership and threaded recycling for the concurrent queue.

#![cfg(all(feature = "concurrent", not(loom)))]
#![cfg(all(
    any(target_pointer_width = "32", target_pointer_width = "64"),
    target_has_atomic = "ptr",
    target_has_atomic = "64"
))]

use std::cell::Cell;
use std::collections::VecDeque;
use std::sync::Barrier;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;

use paged_collections::concurrent::ConcurrentQueue;
use paged_collections::paged_deque::PagedDeque;
use proptest::prelude::*;
use proptest::test_runner::TestCaseResult;

#[derive(Clone, Debug)]
enum Op {
    Push(i32),
    Pop,
    Clear,
    Shrink,
    Drain,
    Edit(i32),
    Convert,
}

fn operations() -> impl Strategy<Value = Vec<Op>> {
    prop::collection::vec(
        prop_oneof![
            10 => any::<i32>().prop_map(Op::Push),
            4 => Just(Op::Pop),
            1 => Just(Op::Clear),
            2 => Just(Op::Shrink),
            1 => Just(Op::Drain),
            2 => any::<i32>().prop_map(Op::Edit),
            2 => Just(Op::Convert),
        ],
        0..if cfg!(miri) { 30 } else { 160 },
    )
}

fn check_model<const PAGE: usize>(ops: &[Op]) -> TestCaseResult {
    let mut queue = ConcurrentQueue::<i32, PAGE>::new();
    let mut model = VecDeque::new();
    for op in ops {
        match *op {
            Op::Push(value) => {
                queue.push(value);
                model.push_back(value);
            }
            Op::Pop => prop_assert_eq!(queue.pop(), model.pop_front()),
            Op::Clear => {
                let pages = queue.allocated_pages();
                queue.clear();
                model.clear();
                prop_assert_eq!(queue.allocated_pages(), pages);
            }
            Op::Shrink => {
                let pages = queue.allocated_pages();
                queue.shrink_to_fit();
                prop_assert!(queue.allocated_pages() <= pages);
                if model.is_empty() {
                    prop_assert_eq!(queue.allocated_pages(), 0);
                }
            }
            Op::Drain => {
                let pages = queue.allocated_pages();
                prop_assert!(queue.drain().eq(model.drain(..)));
                prop_assert_eq!(queue.allocated_pages(), pages);
            }
            Op::Edit(delta) => {
                for (index, value) in queue.iter_mut().enumerate() {
                    *value = value.wrapping_add(delta).wrapping_add(index as i32);
                }
                for (index, value) in model.iter_mut().enumerate() {
                    *value = value.wrapping_add(delta).wrapping_add(index as i32);
                }
            }
            Op::Convert => {
                let plain = PagedDeque::from(queue);
                prop_assert!(plain.iter().eq(model.iter()));
                queue = ConcurrentQueue::from(plain);
            }
        }
        prop_assert_eq!(queue.len(), model.len());
        prop_assert_eq!(queue.is_empty(), model.is_empty());
        prop_assert!(
            queue
                .iter_mut()
                .map(|value| *value)
                .eq(model.iter().copied())
        );
    }
    prop_assert!(queue.into_iter().eq(model));
    Ok(())
}

fn config() -> ProptestConfig {
    let defaults = ProptestConfig::default();
    ProptestConfig {
        cases: if cfg!(miri) { 4 } else { 128 },
        failure_persistence: if cfg!(miri) {
            None
        } else {
            defaults.failure_persistence
        },
        ..defaults
    }
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn matches_vec_deque(ops in operations()) {
        check_model::<1>(&ops)?;
        check_model::<2>(&ops)?;
        check_model::<4>(&ops)?;
    }
}

#[test]
fn capacity_and_shrink_preserve_the_active_chain() {
    let mut queue = ConcurrentQueue::<_, 2>::with_capacity(15);
    assert_eq!(queue.allocated_pages(), 8);
    queue.extend(0..5);
    assert_eq!(queue.pop(), Some(0));
    assert_eq!(queue.pop(), Some(1));
    queue.shrink_to_fit();
    assert_eq!(queue.allocated_pages(), 2);
    assert_eq!(format!("{queue:?}"), "ConcurrentQueue { len: 3 }");

    {
        let mut values = queue.iter_mut();
        let first = values.next().unwrap();
        let second = values.next().unwrap();
        assert_eq!(values.next().map(|value| *value), Some(4));
        assert_eq!(values.next(), None);
        *first += 10;
        *second += 10;
    }

    queue.extend(5..10);
    assert!(queue.drain().eq([12, 13, 4, 5, 6, 7, 8, 9]));
    queue.shrink_to_fit();
    assert_eq!(queue.allocated_pages(), 0);
    queue.push(10);
    assert_eq!(queue.pop(), Some(10));
}

#[test]
fn values_are_dropped_exactly_once() {
    struct Value<'a>(&'a Cell<usize>);

    impl Drop for Value<'_> {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }

    fn check<const PAGE: usize>() {
        let counts = [const { Cell::new(0) }; 20];
        let mut queue = ConcurrentQueue::<_, PAGE>::new();
        queue.extend(counts[..5].iter().map(Value));
        drop(queue.pop().unwrap());
        queue.clear();
        assert!(counts[..5].iter().all(|count| count.get() == 1));

        queue.extend(counts[5..10].iter().map(Value));
        let mut drain = queue.drain();
        drop(drain.next().unwrap());
        assert_eq!(counts[5].get(), 1);
        drop(drain);
        assert!(counts[5..10].iter().all(|count| count.get() == 1));

        queue.extend(counts[10..15].iter().map(Value));
        drop(queue);
        let mut iter = counts[15..]
            .iter()
            .map(Value)
            .collect::<ConcurrentQueue<_, PAGE>>()
            .into_iter();
        drop(iter.next().unwrap());
        drop(iter);
        assert!(counts.iter().all(|count| count.get() == 1));
    }

    check::<1>();
    check::<2>();
    check::<4>();
}

#[test]
fn a_panicking_drop_still_drops_the_rest() {
    struct Value<'a>(&'a Cell<usize>, bool);

    impl Drop for Value<'_> {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
            if self.1 {
                panic!("value drop");
            }
        }
    }

    fn check<const PAGE: usize>() {
        let count = Cell::new(0);
        let queue = ConcurrentQueue::<_, PAGE>::new();
        queue.push(Value(&count, true));
        for _ in 0..4 {
            queue.push(Value(&count, false));
        }
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(queue)));
        assert!(unwound.is_err());
        assert_eq!(count.get(), 5);

        // The owning iterator drops through the same path.
        let count = Cell::new(0);
        let queue = ConcurrentQueue::<_, PAGE>::new();
        queue.push(Value(&count, false));
        queue.push(Value(&count, true));
        queue.push(Value(&count, false));
        let mut iter = queue.into_iter();
        drop(iter.next().unwrap());
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(iter)));
        assert!(unwound.is_err());
        assert_eq!(count.get(), 3);
    }

    check::<1>();
    check::<2>();
    check::<4>();
}

#[test]
fn repeated_shrinks_keep_only_needed_pages() {
    let mut queue = ConcurrentQueue::<_, 1>::new();
    queue.push(0);
    for value in 1..if cfg!(miri) { 16 } else { 2_000 } {
        queue.push(value);
        assert_eq!(queue.pop(), Some(value - 1));
        queue.shrink_to_fit();
        assert_eq!(queue.len(), 1);
        assert_eq!(queue.allocated_pages(), 2);
    }
    assert!(queue.drain().eq([if cfg!(miri) { 15 } else { 1_999 }]));
}

#[test]
fn steady_state_reuses_pages() {
    fn check<const PAGE: usize>() {
        let queue = ConcurrentQueue::<_, PAGE>::new();
        for value in 0..if cfg!(miri) { 24 } else { 40_000 } {
            queue.push(value);
            assert_eq!(queue.pop(), Some(value));
        }
        assert!(queue.is_empty());
        assert_eq!(queue.allocated_pages(), 2);
    }

    check::<1>();
    check::<2>();
    check::<4>();
}

#[test]
fn multiple_producers_and_consumers() {
    fn check<const PAGE: usize>() {
        let producers = if cfg!(miri) { 2 } else { 4 };
        let consumers = if cfg!(miri) { 2 } else { 4 };
        let per_producer = if cfg!(miri) { 12 } else { 2_000 };
        let queue = ConcurrentQueue::<_, PAGE>::new();
        let start = Barrier::new(producers + consumers);
        let done = AtomicBool::new(false);
        // completed[p] is one past the highest sequence of producer p that a
        // finished pop returned. A pop that starts after another pop finished
        // must return a later value of that producer, whichever consumer ran
        // it: this catches FIFO reversals across consumers.
        let completed: Vec<_> = (0..producers).map(|_| AtomicUsize::new(0)).collect();

        let observed = thread::scope(|scope| {
            let writers: Vec<_> = (0..producers)
                .map(|id| {
                    let (queue, start) = (&queue, &start);
                    scope.spawn(move || {
                        start.wait();
                        for sequence in 0..per_producer {
                            queue.push((id, sequence));
                        }
                    })
                })
                .collect();
            let readers: Vec<_> = (0..consumers)
                .map(|_| {
                    scope.spawn(|| {
                        start.wait();
                        let mut values = Vec::new();
                        let mut floor = vec![0; producers];
                        loop {
                            let finished = done.load(Ordering::Acquire);
                            for (low, done) in floor.iter_mut().zip(&completed) {
                                *low = done.load(Ordering::Acquire);
                            }
                            if let Some(value) = queue.pop() {
                                let (id, sequence) = value;
                                assert!(
                                    sequence >= floor[id],
                                    "pop returned {value:?} after a finished pop of \
                                     producer {id} sequence {}",
                                    floor[id] - 1
                                );
                                completed[id].fetch_max(sequence + 1, Ordering::AcqRel);
                                values.push(value);
                            } else if finished {
                                return values;
                            } else {
                                thread::yield_now();
                            }
                        }
                    })
                })
                .collect();

            for writer in writers {
                writer.join().unwrap();
            }
            done.store(true, Ordering::Release);
            readers
                .into_iter()
                .map(|reader| reader.join().unwrap())
                .collect::<Vec<_>>()
        });

        let mut seen = vec![vec![false; per_producer]; producers];
        for values in observed {
            let mut last = vec![None; producers];
            for (id, sequence) in values {
                assert!(last[id].is_none_or(|previous| previous < sequence));
                last[id] = Some(sequence);
                assert!(!std::mem::replace(&mut seen[id][sequence], true));
            }
        }
        assert!(seen.into_iter().flatten().all(|found| found));
        assert_eq!(queue.pop(), None);
    }

    check::<1>();
    check::<4>();
}

#[test]
fn store_buffering_cannot_hide_a_finished_push() {
    // Litmus test for the SeqCst tail CAS in `push` and the SeqCst fence
    // before the tail load in `pop`. Loom weakens SeqCst read-modify-writes,
    // so this ordering is only checked on real hardware. On x86 the locked
    // CAS is a full barrier anyway, so this mainly guards weaker targets.
    let rounds = if cfg!(miri) { 20 } else { 20_000 };
    let mut queue = ConcurrentQueue::<u8, 4>::new();
    let flag = AtomicBool::new(false);
    let start = Barrier::new(2);
    for _ in 0..rounds {
        let (saw_flag, popped) = thread::scope(|scope| {
            let pusher = scope.spawn(|| {
                start.wait();
                queue.push(1);
                flag.load(Ordering::SeqCst)
            });
            start.wait();
            flag.store(true, Ordering::SeqCst);
            let popped = queue.pop();
            let saw_flag = pusher.join().unwrap();
            (saw_flag, popped)
        });
        assert!(
            saw_flag || popped.is_some(),
            "the pusher missed the flag and the pop missed the push"
        );
        queue.clear();
        flag.store(false, Ordering::Relaxed);
    }
}

#[test]
fn zero_sized_values_and_send_without_sync() {
    #[repr(align(64))]
    struct Zst;

    let mut queue = ConcurrentQueue::<Zst, 2>::default();
    queue.extend([Zst, Zst, Zst, Zst, Zst]);
    assert_eq!(queue.iter_mut().count(), 5);
    assert_eq!(queue.drain().count(), 5);
    assert!(queue.is_empty());

    let queue = ConcurrentQueue::<Cell<u32>, 1>::new();
    thread::scope(|scope| {
        scope.spawn(|| queue.push(Cell::new(42))).join().unwrap();
        assert_eq!(queue.pop().unwrap().into_inner(), 42);
    });
}

#[cfg(feature = "serde")]
#[test]
fn serde_round_trips_through_paged_deque() {
    #[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
    struct Value(u32);

    let plain: PagedDeque<_, 2> = [Value(1), Value(2), Value(3)].into_iter().collect();
    let encoded = serde_json::to_string(&plain).unwrap();
    let queue: ConcurrentQueue<Value, 2> = serde_json::from_str(&encoded).unwrap();
    assert_eq!(queue.len(), 3);
    assert_eq!(
        serde_json::to_string(&PagedDeque::from(queue)).unwrap(),
        encoded
    );
}
