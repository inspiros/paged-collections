use paged_collections::{
    hive::Hive, paged_deque::PagedDeque, paged_vec::PagedVec, priority_queue::PagedPriorityQueue,
    sparse_map::PagedSparseMap, sparse_set::PagedSparseSet,
};
use paged_collections_bench::{
    DenseMap, DenseSet, Direct, DirectRadix, FxMap, FxSet, IdMap, IdStore, PATTERNS, SIZES,
    build_map, build_set, dense_len, keys,
};
#[path = "../counting/mod.rs"]
mod counting;

use counting::{CURRENT, PEAK};
use std::{
    collections::{BTreeMap, BTreeSet, BinaryHeap, VecDeque},
    hint::black_box,
    sync::atomic::Ordering,
};
fn measure<T>(group: &str, name: &str, pattern: &str, n: usize, build: impl FnOnce() -> T) {
    let baseline = CURRENT.load(Ordering::Relaxed);
    PEAK.store(baseline, Ordering::Relaxed);
    let value = black_box(build());
    let after = CURRENT.load(Ordering::Relaxed) - baseline;
    let peak = PEAK.load(Ordering::Relaxed) - baseline;
    drop(value);
    println!("{group},{name},{pattern},{n},{peak},{after}");
}
fn ids<S: IdStore>(n: usize) -> S {
    let mut s = S::default();
    for i in 0..n {
        s.add(i as u64);
    }
    s
}
fn main() {
    println!("group,container,pattern,n,peak_bytes,bytes_after_build");
    for n in SIZES {
        for pattern in PATTERNS {
            if paged_collections_bench::skipped(pattern, n) {
                continue;
            }
            let input = keys(pattern, n, false);
            macro_rules! m {
                ($group:literal,$t:ty,$name:literal) => {
                    measure($group, $name, pattern, n, || build_map::<$t>(&input));
                };
            }
            macro_rules! s {
                ($t:ty,$name:literal) => {
                    measure("sparse", $name, pattern, n, || build_set::<$t>(&input));
                };
            }
            m!("sparse",PagedSparseMap<u32,u64>,"PagedSparseMap");
            m!("sparse", FxMap, "HashMap");
            m!("sparse",BTreeMap<u32,u64>,"BTreeMap");
            s!(PagedSparseSet<u32>, "PagedSparseSet");
            s!(FxSet, "HashSet");
            s!(BTreeSet<u32>, "BTreeSet");
            m!("direct", Direct, "PagedDirectMapFlat");
            m!("direct", DirectRadix, "PagedDirectMapRadix");
            m!("direct", FxMap, "HashMap");
            m!("direct",BTreeMap<u32,u64>,"BTreeMap");
            if dense_len(&input).is_some() {
                m!("sparse", DenseMap, "VecOption");
                s!(DenseSet, "VecBool");
                m!("direct", DenseMap, "VecOption");
            }
        }
        macro_rules! id {
            ($t:ty,$name:literal) => {
                measure("ids", $name, "sequential", n, || ids::<$t>(n));
            };
        }
        id!(Hive<u64>, "Hive");
        id!(IdMap, "IdAllocatorIdMap");
        id!(slotmap::SlotMap<slotmap::DefaultKey,u64>,"SlotMap");
        id!(slotmap::DenseSlotMap<slotmap::DefaultKey,u64>,"DenseSlotMap");
        id!(slab::Slab<u64>, "Slab");
        id!(thunderdome::Arena<u64>, "Arena");
        measure("seq", "PagedVec", "sequential", n, || {
            let mut v = PagedVec::<u64>::new();
            for i in 0..n {
                v.push(i as u64);
            }
            v
        });
        measure("seq", "Vec", "sequential", n, || {
            let mut v = Vec::new();
            for i in 0..n {
                v.push(i as u64);
            }
            v
        });
        measure("seq", "PagedDeque", "sequential", n, || {
            let mut v = PagedDeque::<u64>::new();
            for i in 0..n {
                v.push_back(i as u64);
            }
            v
        });
        measure("seq", "VecDeque", "sequential", n, || {
            let mut v = VecDeque::new();
            for i in 0..n {
                v.push_back(i as u64);
            }
            v
        });
        measure("pq", "PagedPriorityQueue", "sequential", n, || {
            let mut q = PagedPriorityQueue::<u32, u64>::new();
            for i in 0..n {
                let _ = q.push(i as u32, i as u64);
            }
            q
        });
        measure("pq", "PriorityQueue", "sequential", n, || {
            let mut q = priority_queue::PriorityQueue::new();
            for i in 0..n {
                let _ = q.push(i as u32, i as u64);
            }
            q
        });
        measure("pq", "BinaryHeap", "sequential", n, || {
            let mut q = BinaryHeap::new();
            for i in 0..n {
                q.push((i as u64, i as u32));
            }
            q
        });
    }
}
