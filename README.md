# paged-collections

<!-- Uncomment after the first release.
[![crates.io](https://img.shields.io/crates/v/paged-collections)](https://crates.io/crates/paged-collections)
[![docs.rs](https://img.shields.io/docsrs/paged-collections)](https://docs.rs/paged-collections)
[![PyPI](https://img.shields.io/pypi/v/paged-collections)](https://pypi.org/project/paged-collections/)
-->
[![CI](https://img.shields.io/github/actions/workflow/status/inspiros/paged-collections/ci.yml?label=CI)](https://github.com/inspiros/paged-collections/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-website-475d7a?logo=mdbook&logoColor=white)](https://inspiros.github.io/paged-collections/)
[![License](https://img.shields.io/github/license/inspiros/paged-collections)](LICENSE)

Paged sparse sets, maps, a priority queue and stable-address containers for
Rust (`no_std` + `alloc`), with Python bindings built on PyO3 and NumPy.

- **Paged index**: a key-indexed table split into pages allocated on first
  use, so memory follows the occupied key ranges, not the largest key.
- **Paged storage**: elements in pages that never move, so references stay
  valid while the container grows.

> Status: in development, not yet published.

## Install

```sh
cargo add paged-collections                         # features: std (default), serde, concurrent
pip install paged-collections                       # Python 3.10+, NumPy 1.24+
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
// stale id finds nothing, in the hive or in the side table.
hive.remove(a);
let c = hive.insert("c");
assert_eq!(c.index(), a.index());
assert_eq!(hive.get(a), None);
assert_eq!(score.get(c), None);

// A direct map iterates in slot order and answers range queries.
let in_range: Vec<u32> = score.range(a..=b).map(|(_, &s)| s).collect();
assert_eq!(in_range, [10, 20]);
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
| `PagedSparseSet<K>` | `SparseSet` | index | Integer set: O(1) insert, remove, contains, clear; dense iteration |
| `PagedSparseMap<K, V>` | `SparseMap`, `SparseMapF64`, ... | index | Sparse set with a dense value array |
| `PagedDirectMap<K, V>` | `DirectMap`, `DirectMapF64`, ... | index | Values in place; key order and range queries |
| `PagedPriorityQueue<K, P>` | `PriorityQueue`, `PriorityQueueF64` | index | Indexed heap: priority changes and removal by key |
| `PagedVec<T>` | `PagedList`, `PagedListF64`, ... | storage | Growable vector with stable addresses |
| `PagedDeque<T>` | `PagedDeque` | storage | Double-ended queue with stable addresses |
| `Hive<T>` | `Hive` | storage | Unordered slots with generational ids |
| `IdAllocator` | `IdAllocator`, `IdMap` | none | Generational ids for use as keys |

The `concurrent` feature adds thread-safe variants: `ConcurrentHive`,
`ConcurrentAppendHive`, `ConcurrentPagedVec`, `ConcurrentSparseSet`/`Map`,
`ConcurrentDirectMap`, `AtomicDirectMap`, `AtomicIdMap` and
`ConcurrentIdAllocator`. Python binds the primitive-valued concurrent sets and
maps.

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
crates/paged-collections/         Rust library (no_std + alloc)
crates/paged-collections-python/  PyO3 bindings (the `_core` extension)
crates/bench/                     Rust benchmarks (not a default member)
python/paged_collections/         Python sources and type stubs
tests/                            Python tests
benchmarks/python/                Python benchmarks
docs/site/                        mdBook documentation
```

Development commands are in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).
