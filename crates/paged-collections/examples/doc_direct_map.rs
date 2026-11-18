//! Documentation example (direct map); its output is checked by docs/site/verify_examples.py.
use paged_collections::direct_map::PagedDirectMap;

fn main() {
    let mut values: PagedDirectMap<usize, i32> = PagedDirectMap::new();
    for (key, value) in [(20, 200), (3, 30), (10, 100)] {
        values.insert(key, value);
    }
    println!("get 3 = {}", values.get(3).unwrap());
    println!("default = {}", values.entry(7).or_insert(70));
    println!("existing = {}", values.entry(7).or_insert(999));
    println!("ordered = {:?}", values.iter().collect::<Vec<_>>());
    println!("range = {:?}", values.range(3..20).collect::<Vec<_>>());
    println!("removed = {}", values.remove(10).unwrap());
    println!("keys = {:?}", values.keys().collect::<Vec<_>>());
}
