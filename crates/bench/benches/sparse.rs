include!("common.rs.inc");
use paged_collections::{sparse_map::PagedSparseMap, sparse_set::PagedSparseSet};
use paged_collections_bench::{DenseMap, DenseSet, FxMap, FxSet};
use std::collections::{BTreeMap, BTreeSet};
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
                    maps::<$t>(c, "sparse", $name, pattern, &input, &shuffled, &misses)
                };
            }
            macro_rules! s {
                ($t:ty,$name:literal) => {
                    sets::<$t>(c, $name, pattern, &input, &shuffled, &misses)
                };
            }
            m!(PagedSparseMap<u32,u64>,"PagedSparseMap");
            m!(FxMap, "HashMap");
            m!(BTreeMap<u32,u64>,"BTreeMap");
            s!(PagedSparseSet<u32>, "PagedSparseSet");
            s!(FxSet, "HashSet");
            s!(BTreeSet<u32>, "BTreeSet");
            if dense_len(&input).is_some() {
                m!(DenseMap, "VecOption");
                s!(DenseSet, "VecBool");
            }
        }
    }
}
criterion::criterion_group!(benches, bench);
criterion::criterion_main!(benches);
