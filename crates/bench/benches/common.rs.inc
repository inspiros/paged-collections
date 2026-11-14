use criterion::{BatchSize, Criterion};
#[allow(unused_imports)]
use paged_collections_bench::{
    Map, PATTERNS, SIZES, Set, build_map, build_set, dense_len, keys, shuffle,
};
use std::{hint::black_box, time::Duration};

macro_rules! timed {
    ($c:expr,$group:expr,$op:expr,$pattern:expr,$n:expr,$name:expr,$f:expr) => {{
        let mut group = $c.benchmark_group(format!("{}/{}/{}/{}", $group, $op, $pattern, $n));
        if $n == 10_000_000 {
            group
                .sample_size(10)
                .measurement_time(Duration::from_secs(3));
        }
        group.bench_function($name, $f);
        group.finish();
    }};
}
fn maps<M: Map>(
    c: &mut Criterion,
    group_name: &str,
    name: &str,
    pattern: &str,
    input: &[u32],
    shuffled: &[u32],
    misses: &[u32],
) {
    let n = input.len();
    timed!(
        c,
        group_name,
        "insert",
        pattern,
        n,
        name,
        |b: &mut criterion::Bencher<'_>| b.iter(|| black_box(build_map::<M>(black_box(input))))
    );
    let m = build_map::<M>(input);
    timed!(
        c,
        group_name,
        "lookup_hit",
        pattern,
        n,
        name,
        |b: &mut criterion::Bencher<'_>| b.iter(|| for &k in shuffled {
            let _ = black_box(m.lookup(black_box(k)));
        })
    );
    timed!(
        c,
        group_name,
        "lookup_miss",
        pattern,
        n,
        name,
        |b: &mut criterion::Bencher<'_>| b.iter(|| for &k in misses {
            let _ = black_box(m.lookup(black_box(k)));
        })
    );
    timed!(
        c,
        group_name,
        "iterate",
        pattern,
        n,
        name,
        |b: &mut criterion::Bencher<'_>| b.iter(|| black_box(m.visit()))
    );
    drop(m);
    timed!(
        c,
        group_name,
        "remove",
        pattern,
        n,
        name,
        |b: &mut criterion::Bencher<'_>| b.iter_batched(
            || build_map::<M>(input),
            |mut m| {
                for &k in shuffled {
                    m.delete(black_box(k));
                }
                let _ = black_box(m);
            },
            BatchSize::PerIteration
        )
    );
}
#[allow(dead_code)]
fn sets<S: Set>(
    c: &mut Criterion,
    name: &str,
    pattern: &str,
    input: &[u32],
    shuffled: &[u32],
    misses: &[u32],
) {
    let n = input.len();
    timed!(
        c,
        "sparse",
        "insert",
        pattern,
        n,
        name,
        |b: &mut criterion::Bencher<'_>| b.iter(|| black_box(build_set::<S>(black_box(input))))
    );
    let s = build_set::<S>(input);
    timed!(
        c,
        "sparse",
        "lookup_hit",
        pattern,
        n,
        name,
        |b: &mut criterion::Bencher<'_>| b.iter(|| for &k in shuffled {
            let _ = black_box(s.lookup(black_box(k)));
        })
    );
    timed!(
        c,
        "sparse",
        "lookup_miss",
        pattern,
        n,
        name,
        |b: &mut criterion::Bencher<'_>| b.iter(|| for &k in misses {
            let _ = black_box(s.lookup(black_box(k)));
        })
    );
    timed!(
        c,
        "sparse",
        "iterate",
        pattern,
        n,
        name,
        |b: &mut criterion::Bencher<'_>| b.iter(|| black_box(s.visit()))
    );
    drop(s);
    timed!(
        c,
        "sparse",
        "remove",
        pattern,
        n,
        name,
        |b: &mut criterion::Bencher<'_>| b.iter_batched(
            || build_set::<S>(input),
            |mut s| {
                for &k in shuffled {
                    s.delete(black_box(k));
                }
                let _ = black_box(s);
            },
            BatchSize::PerIteration
        )
    );
}
