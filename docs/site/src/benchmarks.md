# Benchmarks

These numbers are a reference from one machine. Ratios move with the CPU,
the allocator and the key distribution; measure your own workload with the
commands below before choosing a container on speed.

## Setup

| Item | Value |
|---|---|
| Machine | 4 × Intel Xeon E7-8860 v4 at 2.20 GHz (72 cores, 144 threads), 503 GB RAM |
| OS | Linux 5.15, glibc 2.35 (system allocator) |
| Rust | rustc 1.100.0, `--release`, no `target-cpu` flags |
| Source | commit `b814a0e` |
| Pinning | each Criterion group on its own physical core (`taskset`), groups in parallel on separate cores; scaling runs alone on 8 physical cores of one NUMA node |
| Load | other users idle (load average 0.0 at start) |

Comparisons:

| Name in tables | Type |
|---|---|
| `HashMap`, `HashSet` | `hashbrown` 0.17 with `FxHasher` (`rustc-hash` 2) |
| `BTreeMap`, `BTreeSet`, `Vec`, `VecDeque`, `BinaryHeap` | `std` |
| `VecOption`, `VecBool` | `Vec<Option<u64>>` / `Vec<bool>` indexed by key (dense keys only) |
| `PriorityQueue` | `priority-queue` 2 |
| `SlotMap`, `DenseSlotMap` | `slotmap` 1 |
| `Slab` | `slab` 0.4 |
| `Arena` | `thunderdome` 0.6 |
| `IdAllocatorIdMap` | `IdAllocator` + `PagedSparseMap<Id, u64>` |

## Method

- Times are Criterion medians (default settings, one run), divided by the
  number of keys or elements `n`, so they read as nanoseconds per element.
  Differences under about 10% are noise.
- **insert** builds the container from `n` keys in pattern order, including
  allocation. **hit** looks up all keys in shuffled order; **miss** looks up
  `n` absent keys from the same pattern. **iterate** visits every entry.
  **remove** removes all keys in shuffled order.
- Key patterns (`u32` keys):
  - **dense**: `0..n`.
  - **clustered**: runs of 64 consecutive keys, run starts spread
    pseudo-randomly over the `u32` range.
  - **uniform**: pseudo-random `u32` keys (a xorshift permutation).
- Uniform keys stop at 10⁵: at 10⁷ the paged containers would need tens of
  GiB, one page per key.
- Memory counts bytes requested from the global allocator by the container,
  not RSS. Bold marks the smallest value in a row.

## Reproduce

```sh
git clone https://github.com/inspiros/paged-collections
cd paged-collections
cargo bench -p paged-collections-bench --bench sparse     # also: direct, ids, seq, pq
cargo run --release -p paged-collections-bench --bin memory
cargo run --release -p paged-collections-bench --bin churn
cargo run --release -p paged-collections-bench --bin page_sweep
cargo run --release -p paged-collections-bench --bin concurrent_scaling -- 2000000 7
python benchmarks/python/concurrent_scaling.py --threads 1 2 4 8 --repeat 7
```

The largest cases need tens of GiB of memory; pass a Criterion filter
(for example `-- /1000/`) on smaller machines. Source:
[`crates/bench`](https://github.com/inspiros/paged-collections/tree/main/crates/bench),
[`benchmarks/python`](https://github.com/inspiros/paged-collections/tree/main/benchmarks/python).

## Keyed Collections

Findings:

- **Misses** with the `Flat` directory cost 1 to 4 ns on every pattern: an
  unallocated page answers without touching a slot. Hash misses cost 4 to
  13 ns, B-tree misses 24 to 80 ns.
- **Iteration** over the sparse set and map reads only the dense array and
  is 4 to 13 times faster than `HashMap`. The direct map scans whole pages,
  so it is slow when pages are sparsely filled (clustered and uniform keys).
- **Dense keys**: inserts are 4 to 19 times faster than `HashMap`. Direct-map
  hits are 2 to 2.6 times faster; sparse-map hits range from 1.2 times faster
  (10⁵) to 1.1 times slower (10⁷). A plain `Vec` indexed by key is faster
  still, but needs memory up to the largest key.
- **Uniform keys** are the worst case: each insert allocates and zeroes a
  page (3 to 11 µs per key). Use a hash map.
- **Radix** matches Flat on dense keys and is 3 to 4 times slower on
  clustered hits; use it only for keys beyond the `u32` range.

### Sparse Map (`u32 -> u64`)

**dense**, ns per key

| n, op | PagedSparseMap | HashMap | BTreeMap | VecOption |
|---|---:|---:|---:|---:|
| 10³ insert | 6.44 | 24.1 | 75.2 | **4.7** |
| 10³ hit | 2.8 | 5.26 | 45.6 | **1.15** |
| 10³ miss | 1.24 | 4.13 | 24.4 | **0.779** |
| 10³ iterate | **0.155** | 2.02 | 2.76 | 0.578 |
| 10³ remove | 8.02 | 10.4 | 93.5 | **1.05** |
| 10⁵ insert | **6.05** | 33.9 | 99.8 | 6.42 |
| 10⁵ hit | 8.94 | 11 | 132 | **2.57** |
| 10⁵ miss | 1.18 | 8 | 42.8 | **0.803** |
| 10⁵ iterate | **0.265** | 1.84 | 3.05 | 0.699 |
| 10⁵ remove | 23.2 | **16.8** | 178 | 19.8 |
| 10⁷ insert | **14** | 130 | 145 | 14.2 |
| 10⁷ hit | 47.9 | 42.3 | 450 | **14.5** |
| 10⁷ miss | 1.18 | 8.42 | 66.5 | **0.826** |
| 10⁷ iterate | **0.907** | 3.86 | 8.87 | 2.27 |
| 10⁷ remove | 194 | 141 | 543 | **107** |

**clustered**, ns per key

| n, op | PagedSparseMap | HashMap | BTreeMap | VecOption |
|---|---:|---:|---:|---:|
| 10³ insert | 41.1 | **25.8** | 75.6 | 57.4 |
| 10³ hit | 2.94 | 5.23 | 49.6 | **1.15** |
| 10³ miss | 1.22 | 4.15 | 28.6 | **0.762** |
| 10³ iterate | **0.155** | 1.82 | 2.75 | 18,291 |
| 10³ remove | 24.4 | **10.4** | 93.5 | 20.4 |
| 10⁵ insert | 212 | **32.6** | 116 | – |
| 10⁵ hit | 15.6 | **11.3** | 134 | – |
| 10⁵ miss | **1.53** | 7.73 | 35 | – |
| 10⁵ iterate | **0.263** | 1.83 | 3.23 | – |
| 10⁵ remove | 109 | **22** | 179 | – |
| 10⁷ insert | **56.8** | 140 | 147 | – |
| 10⁷ hit | 60.3 | **41.5** | 467 | – |
| 10⁷ miss | **2.89** | 13.2 | 51.5 | – |
| 10⁷ iterate | **0.894** | 3.9 | 9.91 | – |
| 10⁷ remove | 307 | **129** | 573 | – |

**uniform**, ns per key

| n, op | PagedSparseMap | HashMap | BTreeMap | VecOption |
|---|---:|---:|---:|---:|
| 10³ insert | 825 | **26.8** | 83.6 | – |
| 10³ hit | 5.4 | **5.24** | 46.9 | – |
| 10³ miss | **1.22** | 5.02 | 24.7 | – |
| 10³ iterate | **0.155** | 1.8 | 3.06 | – |
| 10³ remove | 286 | **10.4** | 93.8 | – |
| 10⁵ insert | 3,345 | **33.1** | 113 | – |
| 10⁵ hit | 24.9 | **11.4** | 133 | – |
| 10⁵ miss | **4** | 10.7 | 78.8 | – |
| 10⁵ iterate | **0.265** | 1.84 | 3.68 | – |
| 10⁵ remove | 813 | **22.6** | 172 | – |

### Sparse Set (`u32`)

**dense**, ns per key

| n, op | PagedSparseSet | HashSet | BTreeSet | VecBool |
|---|---:|---:|---:|---:|
| 10³ insert | 6.2 | 21.9 | 63.9 | **3.47** |
| 10³ hit | 2.69 | 4.82 | 48.7 | **0.967** |
| 10³ miss | 1.01 | 4.47 | 24.7 | **0.878** |
| 10³ iterate | **0.207** | 1.68 | 2.78 | 0.571 |
| 10³ remove | 4.66 | 11 | 85.3 | **0.906** |
| 10⁵ insert | 5.69 | 28.5 | 89.3 | **3.32** |
| 10⁵ hit | 6.7 | 9.82 | 128 | **1.24** |
| 10⁵ miss | 0.995 | 7.83 | 42.7 | **0.872** |
| 10⁵ iterate | **0.211** | 1.57 | 2.93 | 0.558 |
| 10⁵ remove | 11.7 | 16.4 | 157 | **5.78** |
| 10⁷ insert | 9.49 | 57 | 120 | **3.55** |
| 10⁷ hit | 31.1 | 36.1 | 366 | **3.88** |
| 10⁷ miss | 1.01 | 7.93 | 66.5 | **0.887** |
| 10⁷ iterate | **0.341** | 1.93 | 4.91 | 0.565 |
| 10⁷ remove | 128 | 113 | 433 | **21.5** |

**clustered**, ns per key

| n, op | PagedSparseSet | HashSet | BTreeSet | VecBool |
|---|---:|---:|---:|---:|
| 10³ insert | 41.5 | **23.8** | 64.1 | 1,557 |
| 10³ hit | 2.74 | 4.83 | 48.6 | **0.952** |
| 10³ miss | 0.999 | 4.48 | 24.3 | **0.878** |
| 10³ iterate | **0.207** | 1.54 | 2.78 | 17,803 |
| 10³ remove | 20.7 | 11 | 78.9 | **1.43** |
| 10⁵ insert | 207 | **27** | 91.3 | – |
| 10⁵ hit | 11.8 | **10.1** | 130 | – |
| 10⁵ miss | **1.26** | 7.67 | 34.8 | – |
| 10⁵ iterate | **0.21** | 1.78 | 3.02 | – |
| 10⁵ remove | 99.8 | **21.9** | 157 | – |
| 10⁷ insert | **46.5** | 75 | 148 | – |
| 10⁷ hit | 42.1 | **38.1** | 394 | – |
| 10⁷ miss | **2.63** | 12.2 | 55 | – |
| 10⁷ iterate | **0.343** | 2.3 | 9.31 | – |
| 10⁷ remove | 238 | **109** | 437 | – |

**uniform**, ns per key

| n, op | PagedSparseSet | HashSet | BTreeSet | VecBool |
|---|---:|---:|---:|---:|
| 10³ insert | 837 | **24.4** | 69.9 | – |
| 10³ hit | **4.49** | 4.81 | 48.9 | – |
| 10³ miss | **0.999** | 5.23 | 24.7 | – |
| 10³ iterate | **0.207** | 1.54 | 3.11 | – |
| 10³ remove | 270 | **11.1** | 79.1 | – |
| 10⁵ insert | 3,300 | **27.7** | 103 | – |
| 10⁵ hit | 22.9 | **10.1** | 128 | – |
| 10⁵ miss | **3.74** | 10.5 | 76.2 | – |
| 10⁵ iterate | **0.21** | 1.77 | 3.63 | – |
| 10⁵ remove | 797 | **22.4** | 151 | – |

### Direct Map (`u32 -> u64`)

**dense**, ns per key

| n, op | PagedDirectMapFlat | PagedDirectMapRadix | HashMap | BTreeMap | VecOption |
|---|---:|---:|---:|---:|---:|
| 10³ insert | 4.69 | 7.3 | 24.7 | 80.1 | **4.62** |
| 10³ hit | 2.37 | 2.38 | 5 | 45.1 | **1.15** |
| 10³ miss | 1.21 | 1.2 | 4.11 | 24.4 | **0.764** |
| 10³ iterate | 2.25 | 2.55 | 1.72 | 2.76 | **0.564** |
| 10³ remove | 2.31 | 2.6 | 10.4 | 93.9 | **1.06** |
| 10⁵ insert | **5.13** | 7.66 | 33.9 | 111 | 6.41 |
| 10⁵ hit | 4.25 | 4.26 | 10.9 | 131 | **2.56** |
| 10⁵ miss | 1.19 | 1.18 | 8 | 42.8 | **0.801** |
| 10⁵ iterate | 2.25 | 2.54 | 1.61 | 3.08 | **0.692** |
| 10⁵ remove | **4.1** | 4.53 | 17.1 | 178 | 19.7 |
| 10⁷ insert | **7** | 16.3 | 131 | 147 | 14.2 |
| 10⁷ hit | 19.2 | 19.3 | 42.1 | 451 | **14.6** |
| 10⁷ miss | 1.2 | 1.2 | 8.01 | 66.3 | **0.828** |
| 10⁷ iterate | 2.56 | 2.82 | 3.83 | 9.54 | **2.26** |
| 10⁷ remove | **29.3** | 30.4 | 140 | 539 | 106 |

**clustered**, ns per key

| n, op | PagedDirectMapFlat | PagedDirectMapRadix | HashMap | BTreeMap | VecOption |
|---|---:|---:|---:|---:|---:|
| 10³ insert | 50.8 | 38.5 | **26.4** | 81.3 | 56.4 |
| 10³ hit | 2.49 | 17 | 4.91 | 45.1 | **1.15** |
| 10³ miss | 1.2 | 2.62 | 4.12 | 24.4 | **0.763** |
| 10³ iterate | 60.2 | 39.9 | **1.55** | 2.74 | 18,305 |
| 10³ remove | 19.1 | 18.2 | **10.3** | 94 | 20.8 |
| 10⁵ insert | 249 | 99.2 | **32.6** | 115 | – |
| 10⁵ hit | **10.5** | 37.7 | 11 | 133 | – |
| 10⁵ miss | **1.53** | 13 | 7.49 | 35.1 | – |
| 10⁵ iterate | 75.1 | 39.9 | **1.79** | 3.25 | – |
| 10⁵ remove | 84.1 | 41.5 | **22.1** | 179 | – |
| 10⁷ insert | 161 | 200 | **128** | 142 | – |
| 10⁷ hit | **26.7** | 108 | 41.5 | 468 | – |
| 10⁷ miss | **2.85** | 16.5 | 12.7 | 49.3 | – |
| 10⁷ iterate | 33.2 | 47.4 | **3.91** | 10.3 | – |
| 10⁷ remove | 177 | 196 | **125** | 569 | – |

**uniform**, ns per key

| n, op | PagedDirectMapFlat | PagedDirectMapRadix | HashMap | BTreeMap | VecOption |
|---|---:|---:|---:|---:|---:|
| 10³ insert | 8,687 | 6,401 | **27** | 85.3 | – |
| 10³ hit | **4.61** | 21 | 5.16 | 47 | – |
| 10³ miss | **1.19** | 2.62 | 5 | 24.7 | – |
| 10³ iterate | 1,208 | 2,489 | **1.56** | 3.04 | – |
| 10³ remove | 1,065 | 807 | **10.6** | 94.4 | – |
| 10⁵ insert | 10,974 | 11,033 | **33** | 113 | – |
| 10⁵ hit | 23.3 | 69.1 | **11.1** | 132 | – |
| 10⁵ miss | **3.95** | 26.1 | 10.4 | 78 | – |
| 10⁵ iterate | 2,093 | 3,001 | **1.79** | 3.67 | – |
| 10⁵ remove | 1,484 | 1,062 | **22.6** | 173 | – |

### Ordered Queries

Dense keys, 1,000 queries per measurement; time per query. `next` finds the
first entry at or after a random key; `range_1000` visits the 1,000 entries
of a key range.

| n, query | PagedDirectMapFlat | PagedDirectMapRadix | BTreeMap |
|---|---:|---:|---:|
| 10³ next | **18.7** | 39.5 | 52.2 |
| 10⁵ next | **20.2** | 42.4 | 129 |
| 10⁷ next | **21.9** | 44.9 | 270 |
| 10³ range_1000 | 3,967 | 2,970 | **1,324** |
| 10⁵ range_1000 | 7,930 | 5,799 | **2,989** |
| 10⁷ range_1000 | 8,012 | 5,863 | **3,252** |

`next_occupant` is 6 to 12 times faster than `BTreeMap::range(k..).next()`.
Scanning a range is 2 to 2.7 times slower than a B-tree range, because the
direct map checks every slot for occupancy.

## Memory

Bytes requested from the allocator after building, divided by the number of
keys.

| pattern, n | PagedSparseMap | HashMap | BTreeMap | PagedSparseSet | HashSet | BTreeSet | VecOption | VecBool |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense 10³ | 16.4 | 34.8 | 26.4 | 8.22 | 10.3 | 11.7 | 16 | **1** |
| dense 10⁵ | 19.8 | 22.3 | 26.3 | 9.27 | 6.55 | 11.6 | 16 | **1** |
| dense 10⁷ | 24.1 | 28.5 | 26.3 | 10.7 | 8.39 | 11.6 | 16 | **1** |
| clustered 10³ | 343 | 34.8 | 26.4 | 335 | **10.3** | 11.7 | 506,344 | 31,647 |
| clustered 10⁵ | 419 | 22.3 | 25.8 | 409 | **6.55** | 11.3 | – | – |
| clustered 10⁷ | 87.5 | 28.5 | 25.8 | 74.1 | **8.39** | 11.3 | – | – |
| uniform 10³ | 6,279 | 34.8 | 24.3 | 6,271 | **10.3** | 10.7 | – | – |
| uniform 10⁵ | 4,459 | 22.3 | 19.7 | 4,449 | **6.55** | 8.53 | – | – |

| pattern, n | PagedDirectMapFlat | PagedDirectMapRadix | HashMap | BTreeMap | VecOption |
|---|---:|---:|---:|---:|---:|
| dense 10³ | 16.4 | 16.4 | 34.8 | 26.4 | **16** |
| dense 10⁵ | 16.1 | 16.1 | 22.3 | 26.3 | **16** |
| dense 10⁷ | 16 | 16 | 28.5 | 26.3 | **16** |
| clustered 10³ | 527 | 263 | 34.8 | **26.4** | 506,344 |
| clustered 10⁵ | 596 | 257 | **22.3** | 25.8 | – |
| clustered 10⁷ | 259 | 260 | 28.5 | **25.8** | – |
| uniform 10³ | 18,555 | 16,435 | 34.8 | **24.3** | – |
| uniform 10⁵ | 16,731 | 16,639 | 22.3 | **19.7** | – |

A sparse page costs 4 bytes per slot and a direct-map page 16 bytes per slot
(`u32 -> u64`), whether or not the slot is used. Clustered runs of 64 keys
fill 6% of a 1024-slot page; uniform keys fill one slot per page.

### Removal and Shrinking

90% of the keys removed at random, then `shrink_to_fit`, in MB:

| Container | Pattern | n | Built | After removal | After shrink |
|---|---|---:|---:|---:|---:|
| PagedSparseMap | uniform | 10⁵ | 445.9 | 445.9 | 443.3 |
| PagedDirectMapFlat | uniform | 10⁵ | 1,673.1 | 1,673.1 | 197.4 |
| PagedDirectMapRadix | uniform | 10⁵ | 1,663.9 | 1,663.9 | 170.1 |
| HashMap | uniform | 10⁵ | 2.2 | 2.2 | 0.3 |
| BTreeMap | uniform | 10⁵ | 2.0 | 0.2 | 0.2 |
| PagedSparseMap | dense | 10⁷ | 241.5 | 241.5 | 52.1 |
| PagedDirectMapFlat | dense | 10⁷ | 160.1 | 160.1 | 160.1 |
| PagedDirectMapRadix | dense | 10⁷ | 160.1 | 160.1 | 160.1 |
| HashMap | dense | 10⁷ | 285.2 | 285.2 | 35.7 |
| BTreeMap | dense | 10⁷ | 262.9 | 22.7 | 22.7 |
| PagedSparseMap | clustered | 10⁷ | 875.3 | 875.3 | 685.6 |
| PagedDirectMapFlat | clustered | 10⁷ | 2,594.0 | 2,594.0 | 2,590.6 |
| PagedDirectMapRadix | clustered | 10⁷ | 2,603.6 | 2,603.6 | 2,592.8 |
| HashMap | clustered | 10⁷ | 285.2 | 285.2 | 35.7 |
| BTreeMap | clustered | 10⁷ | 257.7 | 22.7 | 22.7 |
| Hive | sequential | 10⁷ | 240.7 | 240.8 | 240.5 |

Random removal leaves nearly every page partly occupied, so `shrink_to_fit`
frees few pages. The sparse map shrinks its dense arrays; the direct map
frees only empty pages.

## Page Size

`page_sweep`, n = 10⁵, ns per key and bytes per key.

| Pattern | Measure | PAGE=64 | PAGE=256 | PAGE=1024 | PAGE=4096 | PAGE=16384 |
|---|---|---:|---:|---:|---:|---:|
| dense | insert ns/key | 11.3 | 11.1 | 10.6 | 10.4 | 10.5 |
| dense | hit ns/key | 11.8 | 11.6 | 11.4 | 11.4 | 11.4 |
| dense | iterate ns/key | 0.266 | 0.267 | 0.266 | 0.266 | 0.267 |
| dense | bytes/key | 19.9 | 19.8 | 19.8 | 19.8 | 20.3 |
| clustered | insert ns/key | 2,632 | 669 | 187 | 40 | 602 |
| clustered | hit ns/key | 17.5 | 18.1 | 19 | 19.2 | 18.1 |
| clustered | iterate ns/key | 0.266 | 0.266 | 0.266 | 0.267 | 0.267 |
| clustered | bytes/key | 5,452 | 1,390 | 419 | 357 | 1,061 |
| uniform | insert ns/key | 3,643 | 1,278 | 2,536 | 9,926 | 26,248 |
| uniform | hit ns/key | 27.5 | 28.9 | 28 | 25.6 | 24 |
| uniform | iterate ns/key | 0.267 | 0.267 | 0.266 | 0.267 | 0.267 |
| uniform | bytes/key | 5,810 | 2,425 | 4,459 | 16,488 | 42,988 |

| Pattern | Measure | PAGE=64 | PAGE=256 | PAGE=1024 | PAGE=4096 | PAGE=16384 |
|---|---|---:|---:|---:|---:|---:|
| dense | insert ns/key | 6.04 | 5.19 | 5.02 | 5.02 | 5.47 |
| dense | hit ns/key | 5.85 | 5.72 | 5.67 | 5.67 | 5.67 |
| dense | iterate ns/key | 2.24 | 2.24 | 2.25 | 2.25 | 2.25 |
| dense | bytes/key | 16.2 | 16.1 | 16.1 | 16.4 | 18.4 |
| clustered | insert ns/key | 2,615 | 663 | 210 | 592 | 2,384 |
| clustered | hit ns/key | 10.1 | 11.9 | 12.3 | 12.4 | 10.4 |
| clustered | iterate ns/key | 1,002 | 262 | 87.8 | 136 | 481 |
| clustered | bytes/key | 5,448 | 1,422 | 596 | 1,109 | 4,119 |
| uniform | insert ns/key | 3,797 | 3,251 | 9,695 | 37,356 | – |
| uniform | hit ns/key | 26.5 | 27.1 | 23.7 | 21.8 | – |
| uniform | iterate ns/key | 1,504 | 899 | 2,106 | 7,753 | – |
| uniform | bytes/key | 6,562 | 5,481 | 16,731 | 65,624 | – |

Small pages waste less memory per partly filled page, but with `Flat` the
directory holds one pointer for each page up to the highest key: for keys
spread over the `u32` range, PAGE = 64 makes a 512 MiB directory, which
dominates the memory column. 1024 is the default because it is close to the
best across patterns. On dense keys the page size barely matters.

## Priority Queue

ns per element

| n, op | PagedPriorityQueue | PriorityQueue | BinaryHeap |
|---|---:|---:|---:|
| 10³ push | 18.1 | 64 | **7.74** |
| 10⁵ push | 17.8 | 111 | **9.72** |
| 10⁷ push | 28.9 | 390 | **20.7** |
| 10³ change_priority | **7.39** | 28.7 | – |
| 10⁵ change_priority | **18.4** | 85.4 | – |
| 10⁷ change_priority | **145** | 551 | – |
| 10³ pop_all | 71.1 | 109 | **48.3** |
| 10⁵ pop_all | 161 | 319 | **155** |
| 10⁷ pop_all | 632 | 2,329 | **564** |

At 10⁵ and 10⁷, push and pop take 1.0 to 1.8 times as long as with
`BinaryHeap`, which has no index. Against the indexed `priority-queue` crate,
push is 6 to 13 times faster and
`change_priority` (n/2 changes, time divided by n) 3.8 to 4.6 times faster.

## Paged Vector and Deque

ns per element

| n, op | PagedVec | Vec |
|---|---:|---:|
| 10³ push | 2.85 | **2.29** |
| 10⁵ push | 2.95 | **1.68** |
| 10⁷ push | 7.96 | **5.09** |
| 10³ random_get | 2.23 | **1.21** |
| 10⁵ random_get | 2.23 | **1.21** |
| 10⁷ random_get | 3.22 | **1.4** |
| 10³ iterate | 0.394 | **0.381** |
| 10⁵ iterate | 0.394 | **0.372** |
| 10⁷ iterate | 0.395 | **0.372** |
| 10³ sum | 0.487 | **0.155** |
| 10⁵ sum | 0.512 | **0.266** |
| 10⁷ sum | 1.05 | **0.9** |

ns per element

| n, op | PagedDeque | VecDeque |
|---|---:|---:|
| 10³ push_front | 4.96 | **3.84** |
| 10⁵ push_front | 5.62 | **3.34** |
| 10⁷ push_front | 9.78 | **9.73** |
| 10³ queue_churn | 8.8 | **4.98** |
| 10⁵ queue_churn | 8.27 | **4.94** |
| 10⁷ queue_churn | **9.04** | 9.17 |
| 10³ random_index | 2.52 | **1.74** |
| 10⁵ random_index | 2.51 | **1.73** |
| 10⁷ random_index | 2.68 | **1.87** |

Iteration matches `Vec`. Indexing costs one extra indirection (1 to 2 ns).
At 10⁷ elements the deque is level with `VecDeque`, whose growth copies
every element; at small sizes `VecDeque` is faster.

## Generational Ids

`insert` builds n entries; `get` looks up all ids in random order;
`remove_half_reinsert` removes half and inserts as many again.

ns per element

| n, op | Hive | IdAllocatorIdMap | SlotMap | DenseSlotMap | Slab | Arena |
|---|---:|---:|---:|---:|---:|---:|
| 10³ insert | 9.3 | 19.3 | 7.73 | 12.9 | 7.76 | **6.37** |
| 10⁵ insert | 15.6 | 18.1 | 8.27 | 12.3 | 7.97 | **7.93** |
| 10⁷ insert | 24.3 | 35.9 | 20.1 | 29.5 | **19.7** | 19.9 |
| 10³ get | 4.19 | 4.84 | 2.92 | 3.37 | **1.7** | 2.61 |
| 10⁵ get | 6.94 | 11.5 | 5.2 | 7.19 | **3.33** | 4.08 |
| 10⁷ get | 31 | 59 | 22.4 | 36.9 | **16.9** | 18.3 |
| 10³ iterate | 0.884 | **0.156** | 1.4 | 0.46 | 0.885 | 0.882 |
| 10⁵ iterate | 0.985 | **0.264** | 1.4 | 0.501 | 0.884 | 0.931 |
| 10⁷ iterate | 2.34 | **0.895** | 2.36 | 1.05 | 2.32 | 2.36 |
| 10³ remove_half_reinsert | 8.6 | 15.8 | **3.71** | 6.5 | 3.73 | 4.92 |
| 10⁵ remove_half_reinsert | **12.1** | 34 | 13.1 | 29.5 | 13.2 | 14.5 |
| 10⁷ remove_half_reinsert | **41.9** | 293 | 67.1 | 179 | 67.7 | 72.2 |

`Hive` and `IdAllocator` + `PagedSparseMap` keep elements at stable
addresses and pay for it on raw lookup speed against `slotmap`, `slab` and
`thunderdome`, which store values in one growable vector.

## Concurrent Scaling

### Rust

`concurrent_scaling 2000000 7`: a random permutation of 2·10⁶ keys split
into T chunks, one thread per chunk; median of 7 runs, throughput in
millions of operations per second.

| Workload | T=1 (M/s) | T=2 | T=4 | T=8 | speedup T=4 |
|---|---:|---:|---:|---:|---:|
| sparse try_insert (fresh) | 9.4 | 4.4 | 3.5 | 3.3 | 0.38x |
| sparse try_insert_many (fresh) | 19.1 | 37.3 | 72.7 | 119.2 | 3.81x |
| sparse get | 31.4 | 61.0 | 115.8 | 229.2 | 3.68x |
| direct insert (fresh) | 18.8 | 27.8 | 46.6 | 85.4 | 2.48x |
| direct insert (overwrite) | 23.6 | 35.8 | 61.5 | 109.0 | 2.60x |
| direct get | 15.8 | 23.1 | 39.9 | 72.8 | 2.52x |
| atomic direct insert (fresh) | 25.0 | 47.1 | 91.3 | 178.6 | 3.66x |
| hive insert (fresh) | 19.7 | 40.2 | 78.2 | 143.6 | 3.97x |
| hive remove | 6.6 | 12.8 | 24.7 | 48.4 | 3.74x |
| hive churn (64 live per thread) | 12.6 | 21.7 | 32.1 | 52.8 | 2.54x |
| append hive insert (fresh) | 22.9 | 40.6 | 77.2 | 141.2 | 3.37x |
| id allocate (fresh) | 28.6 | 56.4 | 114.7 | 216.5 | 4.01x |
| id free | 10.1 | 19.3 | 37.3 | 71.2 | 3.71x |
| id churn (64 live per thread) | 12.6 | 21.8 | 30.6 | 52.1 | 2.43x |
| shared counter fetch_add only | 141.1 | 49.1 | 50.2 | 51.4 | 0.36x |

Single-key `try_insert` into a fresh sparse map does not scale: every insert
reserves a dense-array position from one shared counter, so threads contend
on it. `try_insert_many` takes one reservation per batch and scales. The last row is a
single shared `fetch_add` counter, as a reference for cache-line contention.

### Python

`benchmarks/python/concurrent_scaling.py --threads 1 2 4 8 --repeat 7`,
2·10⁶ keys, typed `float64` maps, batch methods (which release the GIL).
Throughput in millions of keys per second, including NumPy input and output
copies.

Free-threaded CPython 3.14.10 (GIL disabled):

| Class | Operation | Threads | Median (s) | Throughput (M keys/s) | Speedup |
|---|---|---:|---:|---:|---:|
| ConcurrentDirectMapF64 | get_many | 1 | 0.192651 | 10.38 | 1.00x |
| ConcurrentDirectMapF64 | get_many | 2 | 0.111697 | 17.91 | 1.72x |
| ConcurrentDirectMapF64 | get_many | 4 | 0.061031 | 32.77 | 3.16x |
| ConcurrentDirectMapF64 | get_many | 8 | 0.039717 | 50.36 | 4.85x |
| ConcurrentDirectMapF64 | set_many | 1 | 0.117383 | 17.04 | 1.00x |
| ConcurrentDirectMapF64 | set_many | 2 | 0.081900 | 24.42 | 1.43x |
| ConcurrentDirectMapF64 | set_many | 4 | 0.048361 | 41.36 | 2.43x |
| ConcurrentDirectMapF64 | set_many | 8 | 0.027625 | 72.40 | 4.25x |
| ConcurrentSparseMapF64 | get_many | 1 | 0.105083 | 19.03 | 1.00x |
| ConcurrentSparseMapF64 | get_many | 2 | 0.055440 | 36.08 | 1.90x |
| ConcurrentSparseMapF64 | get_many | 4 | 0.037057 | 53.97 | 2.84x |
| ConcurrentSparseMapF64 | get_many | 8 | 0.025559 | 78.25 | 4.11x |
| ConcurrentSparseMapF64 | insert_many | 1 | 0.138173 | 14.47 | 1.00x |
| ConcurrentSparseMapF64 | insert_many | 2 | 0.062856 | 31.82 | 2.20x |
| ConcurrentSparseMapF64 | insert_many | 4 | 0.034273 | 58.36 | 4.03x |
| ConcurrentSparseMapF64 | insert_many | 8 | 0.021569 | 92.73 | 6.41x |

CPython 3.12.16 (GIL enabled):

| Class | Operation | Threads | Median (s) | Throughput (M keys/s) | Speedup |
|---|---|---:|---:|---:|---:|
| ConcurrentDirectMapF64 | get_many | 1 | 0.174405 | 11.47 | 1.00x |
| ConcurrentDirectMapF64 | get_many | 2 | 0.109154 | 18.32 | 1.60x |
| ConcurrentDirectMapF64 | get_many | 4 | 0.066318 | 30.16 | 2.63x |
| ConcurrentDirectMapF64 | get_many | 8 | 0.048947 | 40.86 | 3.56x |
| ConcurrentDirectMapF64 | set_many | 1 | 0.141601 | 14.12 | 1.00x |
| ConcurrentDirectMapF64 | set_many | 2 | 0.092276 | 21.67 | 1.53x |
| ConcurrentDirectMapF64 | set_many | 4 | 0.056619 | 35.32 | 2.50x |
| ConcurrentDirectMapF64 | set_many | 8 | 0.039759 | 50.30 | 3.56x |
| ConcurrentSparseMapF64 | get_many | 1 | 0.092247 | 21.68 | 1.00x |
| ConcurrentSparseMapF64 | get_many | 2 | 0.050604 | 39.52 | 1.82x |
| ConcurrentSparseMapF64 | get_many | 4 | 0.046505 | 43.01 | 1.98x |
| ConcurrentSparseMapF64 | get_many | 8 | 0.028807 | 69.43 | 3.20x |
| ConcurrentSparseMapF64 | insert_many | 1 | 0.142124 | 14.07 | 1.00x |
| ConcurrentSparseMapF64 | insert_many | 2 | 0.066021 | 30.29 | 2.15x |
| ConcurrentSparseMapF64 | insert_many | 4 | 0.040706 | 49.13 | 3.49x |
| ConcurrentSparseMapF64 | insert_many | 8 | 0.036124 | 55.37 | 3.93x |

With the GIL, batch calls still overlap because they release it while in
Rust; the free-threaded build also runs the Python-side dispatch in parallel.
