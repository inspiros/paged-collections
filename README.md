# paged-collections

[![CI](https://img.shields.io/github/actions/workflow/status/inspiros/paged-collections/ci.yml?label=CI)](https://github.com/inspiros/paged-collections/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/paged-collections)](https://crates.io/crates/paged-collections)
[![crates.io downloads](https://img.shields.io/crates/d/paged-collections)](https://crates.io/crates/paged-collections)
[![PyPI](https://img.shields.io/pypi/v/paged-collections)](https://pypi.org/project/paged-collections/)
[![PyPI monthly downloads](https://img.shields.io/pypi/dm/paged-collections)](https://pypi.org/project/paged-collections/)
[![Documentation](https://img.shields.io/badge/docs-website-475d7a?logo=mdbook&logoColor=white)](https://inspiros.github.io/paged-collections/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)

Paged sparse sets, maps, a priority queue and stable-address containers for
Rust (`no_std` + `alloc`), with Python bindings built on PyO3 and NumPy.

<p align="center">
  <img src="https://inspiros.github.io/paged-collections/assets/paging-overview.svg" alt="A paged index allocates pages on demand. Paged storage grows by adding pages without moving existing elements.">
</p>

- **Paged index**: a key-indexed table split into pages allocated on first
  use, so index pages follow the occupied key ranges. The default `Flat`
  directory still holds one pointer per page up to the largest key; for keys
  spread over a wide range, the `Radix` and `Tree` directories avoid that.
- **Paged storage**: elements keep their addresses as the container grows.

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
| `ConcurrentPriorityQueue<K, P>` | `ConcurrentPriorityQueue`, `ConcurrentPriorityQueueF64` | index | Sharded indexed max-priority queue; relaxed `pop`, exact `pop_exact` |
| `PagedVec<T>` | `PagedList`, `PagedListF64`, ... | storage | Growable vector with stable addresses |
| `PagedDeque<T>` | `PagedDeque` | storage | Double-ended queue with stable addresses |
| `ConcurrentQueue<T>` | `ConcurrentQueue` | storage | Shared FIFO queue; waits at page boundaries |
| `Hive<T>` | `Hive` | storage | Unordered slots with generational ids |
| `IdAllocator` | `IdAllocator`, `IdMap` | none | Generational ids for use as keys |

The `rayon` feature enables parallel iteration through `rayon::prelude`.

The `concurrent` feature adds thread-safe variants: `ConcurrentHive`,
`ConcurrentAppendHive`, `ConcurrentPagedVec`, `ConcurrentSparseSet`/`Map`,
`ConcurrentDirectMap`, `AtomicDirectMap`, `AtomicIdMap` and
`ConcurrentIdAllocator`, `ConcurrentQueue` and `ConcurrentPriorityQueue`.
Python binds the concurrent sparse set, primitive-valued concurrent maps,
`ConcurrentQueue` for Python objects, and the numeric priority queues
`ConcurrentPriorityQueue` and `ConcurrentPriorityQueueF64`.

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
