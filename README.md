# paged-collections

[![CI](https://img.shields.io/github/actions/workflow/status/inspiros/paged-collections/ci.yml?label=CI)](https://github.com/inspiros/paged-collections/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/paged-collections)](https://crates.io/crates/paged-collections)
[![crates.io downloads](https://img.shields.io/crates/d/paged-collections)](https://crates.io/crates/paged-collections)
[![PyPI](https://img.shields.io/pypi/v/paged-collections)](https://pypi.org/project/paged-collections/)
[![PyPI downloads](https://static.pepy.tech/badge/paged-collections)](https://pepy.tech/project/paged-collections)
[![Documentation](https://img.shields.io/badge/docs-website-475d7a?logo=mdbook&logoColor=white)](https://inspiros.github.io/paged-collections/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)

Paged sparse sets, maps, a priority queue and stable-address containers for
Rust (`no_std` + `alloc`), with Python bindings built on PyO3 and NumPy.

<p align="center">
  <img src="https://inspiros.github.io/paged-collections/assets/paging-overview.svg" alt="A paged index allocates pages on demand. Paged storage grows by adding pages without moving existing elements.">
</p>

The keyed collections use a paged index. The first write to a page
allocates it. The default `Flat` directory stores a pointer slot for every
page number up to the highest one used. `Radix` and `Tree` can skip gaps
between page numbers, which saves directory space when keys are far apart.

With paged storage, existing elements keep their addresses as the container
grows.

## Install

```sh
cargo add paged-collections             # features: std (default), serde, rayon, concurrent
pip install paged-collections           # Python 3.10+, NumPy 1.24+
```

Wheels cover CPython 3.10+ (`abi3`) and free-threaded 3.14t and 3.15t+.
See [Installation](https://inspiros.github.io/paged-collections/installation.html).

## Example

Rust:

```rust
use paged_collections::direct_map::PagedDirectMap;
use paged_collections::hive::Hive;
use paged_collections::id_alloc::Id;

let mut hive: Hive<&str> = Hive::new();
let a = hive.insert("a");
let b = hive.insert("b");

let mut score: PagedDirectMap<Id, u32> = PagedDirectMap::new();
score.insert(a, 10);
score.insert(b, 20);

// Removing frees the slot; its next occupant gets a new generation, so the
// stale id misses in the hive, and the new id misses the side-table entry
// left by the old one. Side tables are not cleaned up automatically.
hive.remove(a);
let c = hive.insert("c");
assert_eq!(c.index(), a.index());
assert_eq!(hive.get(a), None);
assert_eq!(score.get(c), None);
assert_eq!(score.remove(a), Some(10));

// A direct map iterates in slot order and answers range queries.
let in_range: Vec<u32> = score.range(a..=b).map(|(_, &s)| s).collect();
assert_eq!(in_range, [20]);
```

Python:

```python
import numpy as np
from paged_collections import sparse_map

m = sparse_map("float64")
m.set_many(np.arange(0, 3_000_000, 1000), 1.0)  # batch insert
m[7] = 2.5
m.get_many([7, 1000])                            # array([2.5, 1. ])

values = m.values_view()   # zero-copy, writeable
values *= 2                # updates the map in place
del values                 # resizing raises BufferError while a view is alive
```

## Containers

| Rust | Python | Paging | Purpose |
|---|---|---|---|
| `PagedBitSet<K>` | `BitSet` | index | One bit per slot in each allocated page, with key order and range operations |
| `PagedSparseSet<K>` | `SparseSet` | index | Integer set: O(1) insert, remove, contains, clear; dense iteration |
| `PagedSparseMap<K, V>` | `SparseMap`, `SparseMapF64`, ... | index | Sparse set with a dense value array |
| `PagedDirectMap<K, V>` | `DirectMap`, `DirectMapF64`, ... | index | Values in place; key order and range queries |
| `PagedPriorityQueue<K, P>` | `PriorityQueue`, `PriorityQueueF64` | index | Indexed heap: priority changes and removal by key |
| `PagedVec<T>` | `PagedList`, `PagedListF64`, ... | storage | Growable vector with stable addresses |
| `PagedDeque<T>` | `PagedDeque` | storage | Double-ended queue with stable addresses |
| `Hive<T>` | `Hive` | storage | Unordered slots with generational ids |
| `IdAllocator` | `IdAllocator`, `IdMap` | none | Generational ids for use as keys |

The `rayon` feature enables parallel iteration through `rayon::prelude`.

### Concurrent containers

The `concurrent` feature (`no_std` compatible) adds thread-safe containers.
They grow through shared references (`&self`), and pages stay at fixed
addresses until exclusive access (`&mut self`), so lookups need no epochs or
hazard pointers. The Python classes can be shared between threads, including
on free-threaded builds; their batch methods (`*_many`) release the GIL.

| Rust | Python | Shared operations (`&self`) | Lookups |
|---|---|---|---|
| `ConcurrentSparseSet<K>` | `ConcurrentSparseSet` | insert | `&K` |
| `ConcurrentSparseMap<K, V>` | `ConcurrentSparseMapF64`, ... | insert | `&V` |
| `ConcurrentDirectMap<K, V>` | `ConcurrentDirectMapF64`, ... | insert, replace, remove, entry (per-page `RwLock`) | closure, clone or entry guard |
| `AtomicDirectMap` (`usize -> u64`), `AtomicIdMap` (`Id -> u32`) | — | lock-free insert, update, remove | copied value |
| `ConcurrentQueue<T>` | `ConcurrentQueue` | push, pop (FIFO) | owned value from `pop` |
| `ConcurrentPriorityQueue<K, P>` | `ConcurrentPriorityQueue`, `ConcurrentPriorityQueueF64` | push, change priority, remove, relaxed `pop`, exact `pop_exact` | cloned priority |
| `ConcurrentPagedVec<T>` | — | push | `&T` |
| `ConcurrentHive<T>` | — | insert, remove, take | `Ref` guard |
| `ConcurrentAppendHive<T>` | — | insert | `&T` |
| `ConcurrentIdAllocator` | — | alloc, free | |

Removal from the sparse set, sparse map and append-only containers, as well
as `clear`, `shrink_to_fit` and conversion to the sequential types, needs
`&mut self`; in Python, such a call raises `RuntimeError` if it conflicts
with another thread. See
[Concurrent containers](https://inspiros.github.io/paged-collections/concurrent.html).

## Documentation

- [Paging model](https://inspiros.github.io/paged-collections/paging.html):
  page sizes, `Flat`/`Radix`/`Tree` directories, memory behaviour.
- [Choosing a container](https://inspiros.github.io/paged-collections/choosing.html):
  when each container fits, and when a standard one is better.
- [Benchmarks](https://inspiros.github.io/paged-collections/benchmarks.html):
  reference measurements and the commands to reproduce them.
- API: [docs.rs](https://docs.rs/paged-collections) (Rust),
  [`_core.pyi`](python/paged_collections/_core.pyi) (Python).

## Layout

```text
paged-collections/
├── crates/
│   ├── paged-collections/         # Rust library (no_std + alloc)
│   ├── paged-collections-python/  # PyO3 bindings for the _core extension
│   └── bench/                     # Rust benchmarks outside the default build
├── python/
│   └── paged_collections/         # Python sources and type stubs
├── tests/                         # Python tests
├── benchmarks/
│   └── python/                    # Python benchmarks
└── docs/
    └── site/                      # mdBook documentation
```

Development commands are in [CONTRIBUTING.md](https://github.com/inspiros/paged-collections/blob/master/CONTRIBUTING.md).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/inspiros/paged-collections/blob/master/LICENSE-APACHE))
- MIT license ([LICENSE-MIT](https://github.com/inspiros/paged-collections/blob/master/LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in this work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
