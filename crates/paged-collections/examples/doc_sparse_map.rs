//! Documentation example (sparse map); its output is checked by docs/site/verify_examples.py.
use paged_collections::sparse_map::PagedSparseMap;

fn main() {
    let mut values: PagedSparseMap<usize, f64> = PagedSparseMap::new();
    values.insert(7, 1.5);
    values.insert(2, 2.5);
    values.insert(9, 3.5);
    println!("get 7 = {}", values.get(7).unwrap());
    println!("removed = {}", values.remove(2).unwrap());
    println!("keys = {:?}", values.keys().collect::<Vec<_>>());
    println!("values = {:?}", values.values().collect::<Vec<_>>());
    let mut objects: PagedSparseMap<usize, (i32, i32)> = PagedSparseMap::new();
    objects.insert(7, (4, 5));
    println!("object = {:?}", objects.get(7).unwrap());
}
