//! Documentation example (sparse set); its output is checked by docs/site/verify_examples.py.
use paged_collections::sparse_set::PagedSparseSet;

fn main() {
    let mut keys: PagedSparseSet<usize> = PagedSparseSet::new();
    keys.extend([3, 7, 11]);
    keys.insert(19);
    println!("inserted = {:?}", keys.as_slice());
    keys.remove(7);
    println!("contains 7 = {}", keys.contains(7));
    println!("dense = {:?}", keys.as_slice());
    keys.clear();
    println!("cleared len = {}", keys.len());
}
