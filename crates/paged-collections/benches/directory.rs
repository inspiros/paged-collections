//! Directory insertion and hit/miss lookup with one-byte pages.

use std::hint::black_box;
use std::time::Instant;

use paged_collections::page::{FlatPages, PageStore, RadixPages, TreePages};

fn random(state: &mut u64) -> usize {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state as usize
}

fn bench<S: PageStore<u8, 1>>(pattern: &str, directory: &str, keys: &[usize], misses: &[usize]) {
    let mut store = S::default();
    let start = Instant::now();
    for &number in keys {
        black_box(store.get_or_alloc(black_box(number)));
    }
    let insert = start.elapsed().as_nanos() as f64 / keys.len() as f64;
    assert!(keys.iter().all(|&n| store.get(n).is_some()));
    assert!(misses.iter().all(|&n| store.get(n).is_none()));

    // Look up in a shuffled order, so cache-friendly insertion order does
    // not flatter the hit numbers.
    let mut order = keys.to_vec();
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    for i in (1..order.len()).rev() {
        order.swap(i, random(&mut state) % (i + 1));
    }
    const ROUNDS: usize = 4;
    let start = Instant::now();
    for _ in 0..ROUNDS {
        for &number in &order {
            black_box(store.get(black_box(number)));
        }
    }
    let hit = start.elapsed().as_nanos() as f64 / (keys.len() * ROUNDS) as f64;
    let start = Instant::now();
    for _ in 0..ROUNDS {
        for &number in misses {
            black_box(store.get(black_box(number)));
        }
    }
    let miss = start.elapsed().as_nanos() as f64 / (misses.len() * ROUNDS) as f64;
    println!(
        "{pattern:10} {directory:5} insert {insert:9.2} ns/op  hit {hit:9.2} ns/op  miss {miss:9.2} ns/op"
    );
}

fn wide(pattern: &str, keys: &[usize], state: &mut u64) {
    // Construct actual misses outside the timing loop, identically for both
    // directories, including on targets where a random usize is only 32 bits.
    let mut occupied = keys.to_vec();
    occupied.sort_unstable();
    let misses: Vec<_> = (0..keys.len())
        .map(|_| {
            let mut n = random(state);
            while occupied.binary_search(&n).is_ok() {
                n = n.wrapping_add(1);
            }
            n
        })
        .collect();
    bench::<TreePages<u8, 1>>(pattern, "Tree", keys, &misses);
    bench::<RadixPages<u8, 1>>(pattern, "Radix", keys, &misses);
}

fn main() {
    println!(
        "Directory benchmarks, PAGE = 1, deterministic inputs, shuffled hits, {}-bit usize",
        usize::BITS
    );
    let dense: Vec<_> = (0..1 << 20).collect();
    let misses: Vec<_> = (1 << 20..2 << 20).collect();
    bench::<FlatPages<u8, 1>>("dense", "Flat", &dense, &misses);
    bench::<TreePages<u8, 1>>("dense", "Tree", &dense, &misses);
    bench::<RadixPages<u8, 1>>("dense", "Radix", &dense, &misses);

    let mut state = 0x4d59_5df4_d0f3_3173_u64;
    let mut clustered = Vec::with_capacity(8 << 14);
    for cluster in 0..8 {
        // Distinct top chunks keep the random, aligned regions disjoint.
        let top_shift = usize::BITS - 3;
        let base =
            (random(&mut state) & (usize::MAX >> 3) & !((1 << 14) - 1)) | (cluster << top_shift);
        clustered.extend((0..1 << 14).map(|offset| base | offset));
    }
    wide("8 clusters", &clustered, &mut state);
    // Holes: the misses share every populated prefix (odd offsets between
    // even-offset keys), so rejection happens at the deepest level.
    let mut holes = Vec::with_capacity(8 << 14);
    let mut hole_misses = Vec::with_capacity(8 << 14);
    for cluster in 0..8 {
        let top_shift = usize::BITS - 3;
        let base =
            (random(&mut state) & (usize::MAX >> 3) & !((1 << 15) - 1)) | (cluster << top_shift);
        holes.extend((0..1 << 14).map(|i| base | (2 * i)));
        hole_misses.extend((0..1 << 14).map(|i| base | (2 * i + 1)));
    }
    bench::<TreePages<u8, 1>>("holes", "Tree", &holes, &hole_misses);
    bench::<RadixPages<u8, 1>>("holes", "Radix", &holes, &hole_misses);
    let uniform: Vec<_> = (0..1 << 18).map(|_| random(&mut state)).collect();
    wide("uniform", &uniform, &mut state);
}
