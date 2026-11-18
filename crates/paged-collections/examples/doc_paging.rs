//! Documentation example (paging); its output is checked by docs/site/verify_examples.py.
use paged_collections::direct_map::PagedDirectMap;
use paged_collections::page::Radix;

fn main() {
    const PAGE: usize = 4096;
    let keys = [7, 5000, 4_000_000_000usize];
    let mut values: PagedDirectMap<usize, usize, PAGE, Radix> = PagedDirectMap::new();
    for key in keys {
        values.insert(key, key + 1);
    }
    println!("page size = {PAGE}");
    println!("keys = {:?}", values.keys().collect::<Vec<_>>());
    println!("len = {}", values.len());
    println!("allocated pages = {}", values.allocated_pages());
}
