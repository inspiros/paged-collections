//! Documentation example (priority queue); its output is checked by docs/site/verify_examples.py.
use paged_collections::priority_queue::PagedPriorityQueue;

fn main() {
    let mut queue: PagedPriorityQueue<usize, i32> = PagedPriorityQueue::new();
    for (key, priority) in [(7, 10), (2, 30), (9, 20), (4, 15)] {
        queue.push(key, priority);
    }
    println!("old priority = {}", queue.change_priority(&7, 40).unwrap());
    println!("removed = {}", queue.remove(&9).unwrap().1);
    let mut order = Vec::new();
    while let Some(item) = queue.pop() {
        order.push(item);
    }
    println!("pop order = {order:?}");
}
