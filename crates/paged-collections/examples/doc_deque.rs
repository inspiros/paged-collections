//! Documentation example (deque); its output is checked by docs/site/verify_examples.py.
use paged_collections::paged_deque::PagedDeque;

fn main() {
    let mut values: PagedDeque<i32> = PagedDeque::new();
    values.extend([1, 2, 3]);
    values.push_front(0);
    values.push_back(4);
    println!("values = {:?}", values.iter().collect::<Vec<_>>());
    println!("index 2 = {}", values[2]);
    println!("front = {}", values.pop_front().unwrap());
    println!("back = {}", values.pop_back().unwrap());
    println!("remaining = {:?}", values.iter().collect::<Vec<_>>());
}
