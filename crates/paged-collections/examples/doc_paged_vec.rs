//! Documentation example (paged vec); its output is checked by docs/site/verify_examples.py.
use paged_collections::paged_vec::PagedVec;

fn main() {
    let mut values: PagedVec<i32> = PagedVec::new();
    values.extend([1, 2, 3]);
    let first: *const i32 = &values[0];
    for value in 4..=10_000 {
        values.push(value);
    }
    println!("len = {}", values.len());
    println!("first = {}", values[0]);
    println!("last = {}", values[values.len() - 1]);
    println!("same address = {}", std::ptr::eq(first, &values[0]));
    println!("popped = {}", values.pop().unwrap());
    values.insert(1, 99);
    println!("removed = {}", values.remove(1));
    println!("prefix = {:?}", values.iter().take(3).collect::<Vec<_>>());
}
