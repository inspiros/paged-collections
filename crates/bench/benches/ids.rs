use criterion::{BatchSize, Criterion};
use paged_collections_bench::{IdMap, IdStore, SIZES, build_ids, shuffle};
use std::{hint::black_box, time::Duration};
fn run<S: IdStore>(c: &mut Criterion, name: &str, n: usize) {
    let mut g = c.benchmark_group(format!("ids/{n}/{name}"));
    if n == 10_000_000 {
        g.sample_size(10).measurement_time(Duration::from_secs(3));
    }
    g.bench_function("insert", |b| b.iter(|| black_box(build_ids::<S>(n))));
    let (s, mut ids) = build_ids::<S>(n);
    shuffle(&mut ids);
    g.bench_function("get", |b| {
        b.iter(|| {
            for &id in &ids {
                let _ = black_box(s.fetch(black_box(id)));
            }
        })
    });
    g.bench_function("iterate", |b| b.iter(|| black_box(s.visit())));
    drop(s);
    g.bench_function("remove_half_reinsert", |b| {
        b.iter_batched(
            || {
                let (s, mut ids) = build_ids::<S>(n);
                shuffle(&mut ids);
                (s, ids)
            },
            |(mut s, ids)| {
                for &id in &ids[..n / 2] {
                    s.erase(id);
                }
                for i in 0..n / 2 {
                    let _ = black_box(s.add(i as u64));
                }
                let _ = black_box(s);
            },
            BatchSize::PerIteration,
        )
    });
    g.finish();
}
fn bench(c: &mut Criterion) {
    for n in SIZES {
        run::<paged_collections::hive::Hive<u64>>(c, "Hive", n);
        run::<IdMap>(c, "IdAllocatorIdMap", n);
        run::<slotmap::SlotMap<slotmap::DefaultKey, u64>>(c, "SlotMap", n);
        run::<slotmap::DenseSlotMap<slotmap::DefaultKey, u64>>(c, "DenseSlotMap", n);
        run::<slab::Slab<u64>>(c, "Slab", n);
        run::<thunderdome::Arena<u64>>(c, "Arena", n);
    }
}
criterion::criterion_group!(benches, bench);
criterion::criterion_main!(benches);
