use criterion::{BatchSize, Criterion};
use paged_collections::priority_queue::PagedPriorityQueue;
use paged_collections_bench::{SIZES, shuffle};
use std::{collections::BinaryHeap, hint::black_box, time::Duration};
trait Queue: Default {
    fn push(&mut self, k: u32, p: u64);
    fn change(&mut self, k: u32, p: u64);
    fn pop(&mut self) -> bool;
}
impl Queue for PagedPriorityQueue<u32, u64> {
    fn push(&mut self, k: u32, p: u64) {
        let _ = self.push(k, p);
    }
    fn change(&mut self, k: u32, p: u64) {
        let _ = self.change_priority(&k, p);
    }
    fn pop(&mut self) -> bool {
        self.pop().is_some()
    }
}
impl Queue for priority_queue::PriorityQueue<u32, u64> {
    fn push(&mut self, k: u32, p: u64) {
        let _ = self.push(k, p);
    }
    fn change(&mut self, k: u32, p: u64) {
        let _ = self.change_priority(&k, p);
    }
    fn pop(&mut self) -> bool {
        self.pop().is_some()
    }
}
impl Queue for BinaryHeap<(u64, u32)> {
    fn push(&mut self, k: u32, p: u64) {
        self.push((p, k));
    }
    fn change(&mut self, _k: u32, _p: u64) {}
    fn pop(&mut self) -> bool {
        self.pop().is_some()
    }
}
fn build<Q: Queue>(n: usize) -> Q {
    let mut q = Q::default();
    for i in 0..n {
        q.push(i as u32, (i as u64).wrapping_mul(0x9e3779b97f4a7c15));
    }
    q
}
fn run<Q: Queue>(c: &mut Criterion, name: &str, n: usize, indices: &[u32], indexed: bool) {
    let mut g = c.benchmark_group(format!("pq/{n}/{name}"));
    if n == 10_000_000 {
        g.sample_size(10).measurement_time(Duration::from_secs(3));
    }
    g.bench_function("push", |b| b.iter(|| black_box(build::<Q>(n))));
    if indexed {
        g.bench_function("change_priority", |b| {
            b.iter_batched(
                || build::<Q>(n),
                |mut q| {
                    for &k in &indices[..n / 2] {
                        q.change(black_box(k), black_box(k as u64));
                    }
                    let _ = black_box(q);
                },
                BatchSize::PerIteration,
            )
        });
    }
    g.bench_function("pop_all", |b| {
        b.iter_batched(
            || build::<Q>(n),
            |mut q| {
                while black_box(q.pop()) {}
            },
            BatchSize::PerIteration,
        )
    });
    g.finish();
}
fn bench(c: &mut Criterion) {
    for n in SIZES {
        let mut indices: Vec<_> = (0..n as u32).collect();
        shuffle(&mut indices);
        run::<PagedPriorityQueue<u32, u64>>(c, "PagedPriorityQueue", n, &indices, true);
        run::<priority_queue::PriorityQueue<u32, u64>>(c, "PriorityQueue", n, &indices, true);
        run::<BinaryHeap<(u64, u32)>>(c, "BinaryHeap", n, &indices, false);
    }
}
criterion::criterion_group!(benches, bench);
criterion::criterion_main!(benches);
