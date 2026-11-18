//! Documentation example (hive); its output is checked by docs/site/verify_examples.py.
use paged_collections::hive::Hive;

fn main() {
    let mut hive: Hive<i32> = Hive::new();
    let a = hive.insert(30);
    let b = hive.insert(10);
    println!("removed = {}", hive.remove(a).unwrap());
    let c = hive.insert(20);
    println!("reused slot = {:?}", (c.index(), c.generation().get()));
    println!("stale = {:?}", hive.get(a));
    println!("id_of index = {}", hive.id_of(&hive[b]).unwrap().index());
    let ordered = hive.sorted_ids();
    let indices: Vec<_> = ordered.iter().map(|id| id.index()).collect();
    println!("sorted indices = {indices:?}");
    let sorted_values: Vec<_> = ordered.iter().map(|&id| hive[id]).collect();
    println!("sorted values = {sorted_values:?}");
    println!(
        "slots before sort = {:?}",
        hive.values().collect::<Vec<_>>()
    );
    // Sorting moves values between slots; ids continue to name slots.
    hive.sort();
    println!("slots after sort = {:?}", hive.values().collect::<Vec<_>>());
    println!("reused id now names = {}", hive[c]);
}
