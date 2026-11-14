use std::{
    hint::black_box,
    io::{self, Write},
    mem::size_of,
    sync::atomic::Ordering,
    time::Instant,
};

use paged_collections::{
    direct_map::{PagedDirectMap, Slot as DirectSlot},
    sparse_map::PagedSparseMap,
    sparse_set::{PagedSparseSet, Slot as SparseSlot},
};
use paged_collections_bench::{
    DenseMap, DenseSet, FxMap, FxSet, Map, PATTERNS, SIZES, Set, build_map, build_set, dense_len,
    keys, shuffle, skipped,
};

#[path = "../counting/mod.rs"]
mod counting;

use counting::{CURRENT, PEAK};

const LIMIT: u128 = 8 * 1024 * 1024 * 1024;

struct Cell<'a> {
    name: &'static str,
    page: usize,
    pattern: &'a str,
    input: &'a [u32],
    hits: &'a [u32],
    misses: &'a [u32],
}

impl Cell<'_> {
    fn row(&self, op: &str, ns: Option<u128>, memory: Option<(usize, usize)>) -> io::Result<()> {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        write!(
            out,
            "{},{},{},{},{op},",
            self.name,
            self.page,
            self.pattern,
            self.input.len()
        )?;
        if let Some(ns) = ns {
            write!(out, "{ns},{:.3},", ns as f64 / self.input.len() as f64)?;
        } else {
            write!(out, ",,")?;
        }
        if let Some((peak, after)) = memory {
            write!(out, "{peak},{after}")?;
        } else {
            write!(out, ",")?;
        }
        writeln!(out)?;
        out.flush()
    }
}

fn median<T: Ord + Copy>(samples: &mut [T]) -> T {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn timed(reps: usize, mut op: impl FnMut()) -> u128 {
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = Instant::now();
        op();
        samples.push(start.elapsed().as_nanos());
    }
    median(&mut samples)
}

fn run<T>(
    cell: &Cell<'_>,
    build: impl Fn(&[u32]) -> T,
    lookup: impl Fn(&T, u32),
    visit: impl Fn(&T),
) -> io::Result<()> {
    let reps = if cell.input.len() <= 100_000 { 7 } else { 3 };
    let mut times = Vec::with_capacity(reps);
    let mut peaks = Vec::with_capacity(reps);
    let mut afters = Vec::with_capacity(reps);
    for _ in 0..reps {
        let baseline = CURRENT.load(Ordering::Relaxed);
        PEAK.store(baseline, Ordering::Relaxed);
        let start = Instant::now();
        let value = black_box(build(black_box(cell.input)));
        let elapsed = start.elapsed().as_nanos();
        let after = CURRENT.load(Ordering::Relaxed) - baseline;
        let peak = PEAK.load(Ordering::Relaxed) - baseline;
        drop(value);
        times.push(elapsed);
        peaks.push(peak);
        afters.push(after);
    }
    cell.row(
        "insert",
        Some(median(&mut times)),
        Some((median(&mut peaks), median(&mut afters))),
    )?;

    let value = build(black_box(cell.input));
    for (op, keys) in [("lookup_hit", cell.hits), ("lookup_miss", cell.misses)] {
        let ns = timed(reps, || {
            for &key in black_box(keys) {
                lookup(black_box(&value), black_box(key));
            }
        });
        cell.row(op, Some(ns), None)?;
    }
    let ns = timed(reps, || visit(black_box(&value)));
    cell.row("iterate", Some(ns), None)
}

fn run_cell<M: Map>(cell: &Cell<'_>) -> io::Result<()> {
    run(
        cell,
        build_map::<M>,
        |map, key| {
            black_box(map.lookup(key));
        },
        |map| {
            black_box(map.visit());
        },
    )
}

fn run_set<S: Set>(cell: &Cell<'_>) -> io::Result<()> {
    run(
        cell,
        build_set::<S>,
        |set, key| {
            black_box(set.lookup(key));
        },
        |set| {
            black_box(set.visit());
        },
    )
}

fn main() -> io::Result<()> {
    let filter = std::env::args().nth(1).unwrap_or_default();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    writeln!(
        out,
        "container,page,pattern,n,op,median_ns,ns_per_elem,peak_bytes,bytes_after_build"
    )?;
    out.flush()?;
    drop(out);

    for n in SIZES {
        for pattern in PATTERNS {
            if skipped(pattern, n) {
                continue;
            }
            let input = keys(pattern, n, false);
            let mut hits = input.clone();
            shuffle(&mut hits);
            let misses = keys(pattern, n, true);
            let cell = |name, page| Cell {
                name,
                page,
                pattern,
                input: &input,
                hits: &hits,
                misses: &misses,
            };
            macro_rules! baseline {
                ($name:literal, $run:ident, $ty:ty) => {
                    if $name.contains(filter.as_str()) {
                        $run::<$ty>(&cell($name, 0))?;
                    }
                };
            }
            baseline!("FxMap", run_cell, FxMap);
            baseline!("FxSet", run_set, FxSet);
            if dense_len(&input).is_some() {
                baseline!("DenseMap", run_cell, DenseMap);
                baseline!("DenseSet", run_set, DenseSet);
            }
            macro_rules! page {
                ($page:literal) => {{
                    let mut pages: Vec<u32> = input.iter().map(|&key| key / $page).collect();
                    pages.sort_unstable();
                    pages.dedup();
                    let slots = pages.len() as u128 * $page as u128;
                    drop(pages);
                    macro_rules! paged {
                        ($name:literal, $run:ident, $ty:ty, $slot:ty) => {
                            if $name.contains(filter.as_str()) {
                                let cell = cell($name, $page);
                                if slots * size_of::<$slot>() as u128 > LIMIT {
                                    cell.row("skipped", None, None)?;
                                } else {
                                    $run::<$ty>(&cell)?;
                                }
                            }
                        };
                    }
                    paged!("PagedDirectMapFlat", run_cell, PagedDirectMap<u32, u64, $page>, DirectSlot<u32, u64>);
                    paged!("PagedSparseMap", run_cell, PagedSparseMap<u32, u64, $page>, SparseSlot);
                    paged!("PagedSparseSet", run_set, PagedSparseSet<u32, $page>, SparseSlot);
                }};
            }
            page!(64);
            page!(256);
            page!(1024);
            page!(4096);
            page!(16384);
            page!(131072);
            page!(1048576);
            page!(16777216);
        }
    }
    Ok(())
}
