//! Memory after churn: build, remove 90% of the entries at random, then
//! `shrink_to_fit`. Prints CSV: live bytes after each step.

#[path = "../counting/mod.rs"]
mod counting;

use std::collections::BTreeMap;
use std::sync::atomic::Ordering;

use counting::CURRENT;
use paged_collections::{hive::Hive, sparse_map::PagedSparseMap};
use paged_collections_bench::{
    Direct, DirectRadix, FxMap, Map, PATTERNS, SIZES, build_map, keys, shuffle,
};

/// A container that can give memory back.
trait Shrink {
    fn shrink(&mut self);
}
impl Shrink for PagedSparseMap<u32, u64> {
    fn shrink(&mut self) {
        self.shrink_to_fit();
    }
}
impl Shrink for Direct {
    fn shrink(&mut self) {
        self.shrink_to_fit();
    }
}
impl Shrink for DirectRadix {
    fn shrink(&mut self) {
        self.shrink_to_fit();
    }
}
impl Shrink for FxMap {
    fn shrink(&mut self) {
        self.shrink_to_fit();
    }
}
/// `BTreeMap` frees nodes as it removes; it has nothing to shrink.
impl Shrink for BTreeMap<u32, u64> {
    fn shrink(&mut self) {}
}

fn live(base: usize) -> usize {
    CURRENT.load(Ordering::Relaxed) - base
}

fn churn<M: Map + Shrink>(name: &str, pattern: &str, input: &[u32], removed: &[u32]) {
    let base = CURRENT.load(Ordering::Relaxed);
    let mut m = build_map::<M>(input);
    let built = live(base);
    for &k in removed {
        m.delete(k);
    }
    let after_remove = live(base);
    m.shrink();
    let after_shrink = live(base);
    drop(m);
    println!(
        "map,{name},{pattern},{},{built},{after_remove},{after_shrink}",
        input.len()
    );
}

fn main() {
    println!("group,container,pattern,n,built_bytes,after_remove_bytes,after_shrink_bytes");
    for n in SIZES {
        for pattern in PATTERNS {
            if paged_collections_bench::skipped(pattern, n) {
                continue;
            }
            let input = keys(pattern, n, false);
            let mut removed = input.clone();
            shuffle(&mut removed);
            removed.truncate(n * 9 / 10);
            churn::<PagedSparseMap<u32, u64>>("PagedSparseMap", pattern, &input, &removed);
            churn::<Direct>("PagedDirectMapFlat", pattern, &input, &removed);
            churn::<DirectRadix>("PagedDirectMapRadix", pattern, &input, &removed);
            churn::<FxMap>("HashMap", pattern, &input, &removed);
            churn::<BTreeMap<u32, u64>>("BTreeMap", pattern, &input, &removed);
        }
        // Hive: ids are sequential; remove 90% at random.
        let base = CURRENT.load(Ordering::Relaxed);
        let mut hive = Hive::<u64>::new();
        let mut ids: Vec<_> = (0..n).map(|i| hive.insert(i as u64)).collect();
        let built = live(base);
        shuffle(&mut ids);
        for &id in &ids[..n * 9 / 10] {
            let _ = hive.remove(id);
        }
        let after_remove = live(base);
        hive.shrink_to_fit();
        let after_shrink = live(base);
        drop(hive);
        drop(ids);
        println!("ids,Hive,sequential,{n},{built},{after_remove},{after_shrink}");
    }
}
