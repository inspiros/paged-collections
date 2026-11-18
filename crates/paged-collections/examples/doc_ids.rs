//! Documentation example (ids); its output is checked by docs/site/verify_examples.py.
use paged_collections::id_alloc::{Id, IdAllocator};
use paged_collections::sparse_map::PagedSparseMap;

fn main() {
    let mut ids = IdAllocator::new();
    let (a, b) = (ids.alloc(), ids.alloc());
    let mut scores: PagedSparseMap<Id, i32> = PagedSparseMap::new();
    scores.insert(a, 10);
    scores.insert(b, 20);
    println!("freed = {}", ids.free(a));
    let c = ids.alloc();
    println!("reused slot = {:?}", (c.index(), c.generation().get()));
    // The side table is independent of the allocator: writing the new generation evicts the stale entry.
    scores.insert(c, 30);
    println!("stale alive = {}", ids.is_alive(a));
    println!("stale side = {:?}", scores.get(a));
    println!("scores = {:?}", scores.values().collect::<Vec<_>>());
    println!("live ids = {}", ids.len());
}
