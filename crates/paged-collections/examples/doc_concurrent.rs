//! Documentation example (concurrent); its output is checked by docs/site/verify_examples.py.
use paged_collections::concurrent::{ConcurrentAppendHive, ConcurrentSparseSet};

// ANCHOR: queue
fn fifo_example() {
    use paged_collections::concurrent::ConcurrentQueue;

    let queue = ConcurrentQueue::<_, 2>::new();
    queue.push("first");
    queue.push("second");
    assert_eq!(queue.pop(), Some("first"));
    assert_eq!(queue.pop(), Some("second"));
}
// ANCHOR_END: queue

// ANCHOR: priority_queue
fn priority_example() {
    use paged_collections::concurrent::ConcurrentPriorityQueue;

    let queue = ConcurrentPriorityQueue::<u32, i32, 4>::with_shards(3);
    queue.push(1, 10);
    queue.push(5, 20);
    assert_eq!(queue.change_priority(&1, 30), Some(10));
    assert_eq!(queue.pop_exact(), Some((1, 30)));
    assert_eq!(queue.pop(), Some((5, 20)));
}
// ANCHOR_END: priority_queue

fn main() {
    fifo_example();
    priority_example();
    let hive: ConcurrentAppendHive<usize> = ConcurrentAppendHive::new();
    let keys: ConcurrentSparseSet<usize> = ConcurrentSparseSet::new();
    std::thread::scope(|scope| {
        for worker in 0..4 {
            let (hive, keys) = (&hive, &keys);
            scope.spawn(move || {
                for value in worker * 4..(worker + 1) * 4 {
                    hive.insert(value);
                    keys.insert(value);
                }
            });
        }
    });
    println!("hive count = {}", hive.len());
    println!("set count = {}", keys.len());
    let mut values: Vec<_> = hive.into_hive().values().copied().collect();
    values.sort_unstable();
    let mut contents: Vec<_> = keys.iter().copied().collect();
    contents.sort_unstable();
    println!("hive values = {values:?}");
    println!("set keys = {contents:?}");
}
