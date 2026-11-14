use criterion::{BatchSize, Criterion};
use paged_collections::{paged_deque::PagedDeque, paged_vec::PagedVec};
use paged_collections_bench::{SIZES, shuffle};
use std::{collections::VecDeque, hint::black_box, time::Duration};
trait Seq: Default {
    fn push(&mut self, v: u64);
    fn get(&self, i: usize) -> Option<&u64>;
    fn sum(&self) -> u64;
    fn visit(&self);
}
impl Seq for Vec<u64> {
    fn push(&mut self, v: u64) {
        self.push(v);
    }
    fn get(&self, i: usize) -> Option<&u64> {
        self.as_slice().get(i)
    }
    fn sum(&self) -> u64 {
        self.iter().copied().sum()
    }
    fn visit(&self) {
        for x in self {
            let _ = black_box(x);
        }
    }
}
impl Seq for PagedVec<u64> {
    fn push(&mut self, v: u64) {
        self.push(v);
    }
    fn get(&self, i: usize) -> Option<&u64> {
        self.get(i)
    }
    fn sum(&self) -> u64 {
        self.iter().copied().sum()
    }
    fn visit(&self) {
        for x in self.iter() {
            let _ = black_box(x);
        }
    }
}
fn build<S: Seq>(n: usize) -> S {
    let mut s = S::default();
    for i in 0..n {
        s.push(i as u64);
    }
    s
}
fn seq<S: Seq>(c: &mut Criterion, name: &str, n: usize, indices: &[usize]) {
    let mut g = c.benchmark_group(format!("seq/{n}/{name}"));
    if n == 10_000_000 {
        g.sample_size(10).measurement_time(Duration::from_secs(3));
    }
    g.bench_function("push", |b| b.iter(|| black_box(build::<S>(n))));
    let s = build::<S>(n);
    g.bench_function("random_get", |b| {
        b.iter(|| {
            for &i in indices {
                let _ = black_box(s.get(black_box(i)));
            }
        })
    });
    g.bench_function("iterate", |b| b.iter(|| s.visit()));
    g.bench_function("sum", |b| b.iter(|| black_box(s.sum())));
    g.finish();
}
trait Deque: Default {
    fn back(&mut self, v: u64);
    fn front(&mut self, v: u64);
    fn pop(&mut self) -> Option<u64>;
    fn get(&self, i: usize) -> Option<&u64>;
}
impl Deque for VecDeque<u64> {
    fn back(&mut self, v: u64) {
        self.push_back(v);
    }
    fn front(&mut self, v: u64) {
        self.push_front(v);
    }
    fn pop(&mut self) -> Option<u64> {
        self.pop_front()
    }
    fn get(&self, i: usize) -> Option<&u64> {
        self.get(i)
    }
}
impl Deque for PagedDeque<u64> {
    fn back(&mut self, v: u64) {
        self.push_back(v);
    }
    fn front(&mut self, v: u64) {
        self.push_front(v);
    }
    fn pop(&mut self) -> Option<u64> {
        self.pop_front()
    }
    fn get(&self, i: usize) -> Option<&u64> {
        self.get(i)
    }
}
fn build_deque<D: Deque>(n: usize) -> D {
    let mut d = D::default();
    for i in 0..n {
        d.back(i as u64);
    }
    d
}
fn deque<D: Deque>(c: &mut Criterion, name: &str, n: usize, indices: &[usize]) {
    let mut g = c.benchmark_group(format!("seq/{n}/{name}"));
    if n == 10_000_000 {
        g.sample_size(10).measurement_time(Duration::from_secs(3));
    }
    g.bench_function("queue_churn", |b| {
        b.iter_batched(
            || build_deque::<D>(n),
            |mut d| {
                for i in 0..n {
                    let _ = black_box(d.pop());
                    d.back(black_box(i as u64));
                }
                let _ = black_box(d);
            },
            BatchSize::PerIteration,
        )
    });
    g.bench_function("push_front", |b| {
        b.iter(|| {
            let mut d = D::default();
            for i in 0..n {
                d.front(black_box(i as u64));
            }
            black_box(d)
        })
    });
    let d = build_deque::<D>(n);
    g.bench_function("random_index", |b| {
        b.iter(|| {
            for &i in indices {
                let _ = black_box(d.get(black_box(i)));
            }
        })
    });
    g.finish();
}
fn bench(c: &mut Criterion) {
    for n in SIZES {
        let mut indices: Vec<_> = (0..n).collect();
        shuffle(&mut indices);
        seq::<PagedVec<u64>>(c, "PagedVec", n, &indices);
        seq::<Vec<u64>>(c, "Vec", n, &indices);
        deque::<PagedDeque<u64>>(c, "PagedDeque", n, &indices);
        deque::<VecDeque<u64>>(c, "VecDeque", n, &indices);
    }
}
criterion::criterion_group!(benches, bench);
criterion::criterion_main!(benches);
