//! Documentation example (overview); its output is checked by docs/site/verify_examples.py.
use paged_collections::direct_map::PagedDirectMap;
use paged_collections::hive::Hive;
use paged_collections::id_alloc::Id;

fn main() {
    let mut hive: Hive<&str> = Hive::new();
    let a = hive.insert("a");
    let b = hive.insert("b");
    let mut score: PagedDirectMap<Id, u32> = PagedDirectMap::new();
    score.insert(a, 10);
    score.insert(b, 20);
    hive.remove(a);
    let c = hive.insert("c");
    println!("reused slot = {:?}", (c.index(), c.generation().get()));
    println!("new side entry = {:?}", score.get(c));
    // Writing the new generation evicts the stale entry from the side table.
    score.insert(c, 30);
    println!("stale hive = {:?}", hive.get(a));
    println!("stale side = {:?}", score.get(a));
    let values: Vec<_> = score.range(a..=b).map(|(_, &v)| v).collect();
    println!("range = {values:?}");
}
