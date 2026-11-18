//! Documentation example (concurrent); its output is checked by docs/site/verify_examples.py.
use paged_collections::concurrent::{ConcurrentAppendHive, ConcurrentSparseSet};

fn main() {
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
