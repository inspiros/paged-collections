include!("common.rs.inc");
use paged_collections::{direct_map::PagedDirectMap, page::Directory};
use paged_collections_bench::{DenseMap, Direct, DirectRadix, FxMap};
use std::collections::BTreeMap;
fn ordered<D: Directory>(
    c: &mut Criterion,
    name: &str,
    pattern: &str,
    input: &[u32],
    queries: &[u32],
) {
    let map = build_map::<PagedDirectMap<u32, u64, 1024, D>>(input);
    let mut g = c.benchmark_group(format!("direct/{pattern}/{}/{name}", input.len()));
    if input.len() == 10_000_000 {
        g.sample_size(10).measurement_time(Duration::from_secs(3));
    }
    g.bench_function("range_1000", |b| {
        b.iter(|| {
            for &k in queries {
                for x in map.range(k..=k.saturating_add(999)) {
                    let _ = black_box(x);
                }
            }
        })
    });
    g.bench_function("next", |b| {
        b.iter(|| {
            for &k in queries {
                let _ = black_box(map.next_occupant(k as usize));
            }
        })
    });
    g.finish();
}
fn bench(c: &mut Criterion) {
    for n in SIZES {
        for pattern in PATTERNS {
            if paged_collections_bench::skipped(pattern, n) {
                continue;
            }
            let input = keys(pattern, n, false);
            let misses = keys(pattern, n, true);
            let mut shuffled = input.clone();
            shuffle(&mut shuffled);
            macro_rules! m {
                ($t:ty,$name:literal) => {
                    maps::<$t>(c, "direct", $name, pattern, &input, &shuffled, &misses)
                };
            }
            m!(Direct, "PagedDirectMapFlat");
            m!(DirectRadix, "PagedDirectMapRadix");
            m!(FxMap, "HashMap");
            m!(BTreeMap<u32,u64>,"BTreeMap");
            if dense_len(&input).is_some() {
                m!(DenseMap, "VecOption");
            }
            let queries = &shuffled[..shuffled.len().min(1000)];
            ordered::<paged_collections::page::Flat>(
                c,
                "PagedDirectMapFlat",
                pattern,
                &input,
                queries,
            );
            ordered::<paged_collections::page::Radix>(
                c,
                "PagedDirectMapRadix",
                pattern,
                &input,
                queries,
            );
            let tree = build_map::<BTreeMap<u32, u64>>(&input);
            let mut g = c.benchmark_group(format!("direct/{pattern}/{n}/BTreeMap"));
            if n == 10_000_000 {
                g.sample_size(10).measurement_time(Duration::from_secs(3));
            }
            g.bench_function("range_1000", |b| {
                b.iter(|| {
                    for &k in queries {
                        for x in tree.range(k..=k.saturating_add(999)) {
                            let _ = black_box(x);
                        }
                    }
                })
            });
            g.bench_function("next", |b| {
                b.iter(|| {
                    for &k in queries {
                        let _ = black_box(tree.range(k..).next());
                    }
                })
            });
            g.finish();
        }
    }
}
criterion::criterion_group!(benches, bench);
criterion::criterion_main!(benches);
