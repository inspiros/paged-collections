# paged-collections

Paged sparse sets, maps and stable-address containers for Rust, with Python
bindings built on PyO3.

> Status: in development. Every collection below is implemented in Rust;
> Python bindings cover the sparse sets and maps, `DirectMap`, `IdAllocator`,
> `IdMap`, `Hive`, `PagedList`, `PagedDeque` and `PagedPriorityQueue`.
> Concurrent types have no Python bindings. Nothing is published.

## Why paging

Paging serves two separate purposes:

- **Paged index.** A table indexed by key is split into fixed-size pages
  allocated on first use, so memory follows the occupied key ranges, not the
  largest key. A plain sparse set with key `4_000_000_000` needs a 16 GB
  sparse array; a paged one needs one page.
- **Paged storage.** Elements live in fixed-size pages that never move, so
  references stay valid while the collection grows and growth never copies
  the whole buffer.

## Collections

| Type | Paging | Purpose |
|---|---|---|
| `PagedSparseSet<K>` | index (storage optional) | Integer set with O(1) insert, remove, contains, clear, and dense iteration |
| `PagedSparseMap<K, V>` | index (storage optional) | Sparse set with a dense value array |
| `PagedPriorityQueue<K, P>` | index | Indexed max-heap with priority changes and removal by key |
| `PagedDirectMap<K, V>` | index | Direct-address map: values stored in place, iterated in slot-index (for integers, key) order |
| `PagedVec<T>` | storage | Growable vector with stable addresses |
| `PagedDeque<T>` | storage | Double-ended queue with stable addresses through pushes and pops at either end |
| `Hive<T>` | storage | Unordered container with stable addresses and generational handles |
| `IdAllocator` | none | Generational id allocator for use as keys |

The indexed collections use `page::Flat` by default, for dense keys below a
few billion. Choose `page::Radix` for wide keys spread over the `usize` range;
its compressed radix directory uses memory proportional to allocated pages.
Choose `page::Tree` for a standard B-tree, or when benchmarks show it wins
for your key pattern. The directory is the final type parameter, for example
`PagedDirectMap<usize, u32, 1024, page::Radix>`.

## When to use

Numbers come from the bench crate `crates/bench` (see
`crates/bench/README.md`), one run per case; treat the ratios as indicative.

| Container | Use it when | Pick something else when |
|---|---|---|
| `PagedSparseSet`, `PagedSparseMap`, `PagedDirectMap` | Keys come in runs or ranges: iteration is 2-10x faster than `HashMap`/`BTreeMap`, misses 2-3x faster. Dense `u32`-like keys with `page::Flat` work well too | Keys are spread uniformly over a wide range (use a `HashMap`). Dense hits are 1.1-2.5x slower than `HashSet`, and much slower than a plain `Vec<bool>` |
| `PagedPriorityQueue` | You need `change_priority` or removal by key: push is 3-17x faster than the `priority-queue` crate | You never change priorities: `BinaryHeap` is 1.3-2x faster |
| `PagedDeque` | Large queues (parity with `VecDeque` at 10 million elements) or stable addresses | Small queues: `VecDeque` is 1.4-1.7x faster |
| `PagedVec` | You need stable addresses: iteration matches `Vec` | Random access dominates: `Vec` is 1.8-2.5x faster |
| `Hive`, `IdMap` | Generational handles with stable addresses: `IdMap` is within 1.2-4x of `SlotMap` and iterates faster at small and medium sizes | Raw lookup speed matters more than stable addresses: `SlotMap` is faster |

- **Spread-out keys are the worst case.** Each key in a new page allocates
  and zeroes a whole page, so uniformly spread keys made inserts 36-480x
  slower and used far more memory.
- **Clustered keys still cost memory.** Short runs (64 keys) fill about 6%
  of a 1024-slot page, up to roughly 19x the memory of a `HashMap`. Long runs
  and dense ranges do not have this problem.
- **Pages are freed only when whole key ranges go.** Removing 90% of the
  keys at random frees nothing, because nearly every page stays partly full.
  There is no automatic page freeing, which keeps addresses stable.
- **Radix** is for page numbers spread over 64 bits; use `page::Flat` when
  keys are dense and below a few billion.

## Rust

A `Hive` owns elements at stable addresses and hands out generational ids;
the ids are keys for the paged maps, which makes side tables:

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

`Hive::id_of` turns a reference to an element back into its id, by address,
in time proportional to the number of pages. To visit elements in order,
`sorted_ids` (and `_by`, `_by_key`) returns the live ids sorted by element
and moves nothing. `sort` (and `_by`, `_by_key`) instead reorders the
elements across the occupied slots, so iteration yields them sorted. Ids
name slots, so after `sort` each id names whatever element moved into its
slot.

```rust
use paged_collections::hive::Hive;

let mut hive: Hive<u32> = Hive::new();
let (x, y) = (hive.insert(30), hive.insert(10));
assert_eq!(hive.id_of(&hive[y]), Some(y));
assert_eq!(hive.sorted_ids(), [y, x]); // ids follow their elements
hive.sort();
assert_eq!(hive[x], 10);               // ids now name sorted slots
```

## Concurrent

The `concurrent` feature adds separate types for shared (`&self`) mutation
across threads. They use only `core` atomics and `alloc`, so they also work
without `std`. One rule holds throughout: *grow shared, shrink exclusive*.
Shared access may install pages and fill slots, but memory is freed only
under `&mut self` or on drop, so no epochs or hazard pointers are needed.

| Type | Shared operations | Notes |
|---|---|---|
| `ConcurrentHive<T>` | insert, remove, lookup | Lookups return a `Ref` guard that pins the value |
| `ConcurrentAppendHive<T>` | insert, lookup | Lookups return plain `&T`; removal needs `&mut self` |
| `ConcurrentPagedVec<T>` | push, lookup | `len` and iteration cover the hole-free prefix |
| `ConcurrentSparseSet<K>`, `ConcurrentSparseMap<K, V>` | insert-only, lookup | Plain `&V` references; removal needs `&mut self` |
| `ConcurrentDirectMap<K, V>` | insert, replace, remove, entry | One reader-writer lock per page |
| `AtomicDirectMap`, `AtomicIdMap` | lock-free one-word updates | `usize -> u64` (below `u64::MAX`), `Id -> u32` |
| `ConcurrentIdAllocator` | allocate, free | Generational ids, as `IdAllocator` |

Each type converts to and from its plain counterpart under exclusive
ownership (`into_inner`, or `into_hive` for the hives, and `From`), keeping
every id and generation. Under `&mut self`, `shrink_to_fit` on both hives
frees pages that are wholly empty. Later shared inserts reuse those pages
with generations above any id they ever held, so a stale id stays dead. The sparse types, `ConcurrentDirectMap` and
`AtomicDirectMap` take a final `SharedDirectory` parameter: `page::Flat` by
default, or `page::Radix` for keys anywhere in the `usize` range.
`page::Tree` has no concurrent form.

```rust
# #[cfg(feature = "concurrent")] {
use paged_collections::concurrent::{AtomicDirectMap, ConcurrentSparseMap};
use paged_collections::page::Radix;

let names: ConcurrentSparseMap<usize, String, 1024, Radix> = ConcurrentSparseMap::new();
let hits: AtomicDirectMap = AtomicDirectMap::new();
std::thread::scope(|s| {
    for t in 0..4usize {
        let (names, hits) = (&names, &hits);
        s.spawn(move || {
            // Wide keys are fine with Radix.
            let _ = names.try_insert(usize::MAX - t, format!("worker {t}"));
            for _ in 0..100 {
                let _ = hits.fetch_update(7, |n| Some(Some(n.unwrap_or(0) + 1)));
            }
        });
    }
});
assert_eq!(names.len(), 4);
assert_eq!(names.get(&(usize::MAX - 2)).map(String::as_str), Some("worker 2"));
assert_eq!(hits.get(7), Some(400));

// Exclusive ownership converts to the plain map, keeping the directory.
let plain = names.into_inner();
assert_eq!(plain.len(), 4);
# }
```

## Python

Keys are integers in `0 <= key < 2**32`. `SparseSet` holds keys,
`SparseMap` maps keys to any objects, and `SparseMapF64`, `SparseMapI64` and
the other typed maps (or `sparse_map(dtype)`) store numbers unboxed:

```python
import numpy as np
from paged_collections import sparse_map

m = sparse_map("float64")
m.set_many(np.arange(0, 3_000_000, 1000), 1.0)  # batch insert
m[7] = 2.5
m.get_many([7, 1000])                            # array([2.5, 1. ])

values = m.values_view()   # zero-copy, writeable
values *= 2                # updates the map in place
# While a view is alive, resizing the map raises BufferError.
del values
```

Views share memory with the collection, like a `memoryview` of a
`bytearray`. NumPy releases the GIL inside many operations, so do not use a
view in one thread while another thread reads or writes the collection or
the view; synchronize such access yourself. A NumPy array stored inside the
`SparseMap` it views is never garbage-collected (NumPy arrays are invisible
to the collector).

The classes can be subclassed. As with `dict`, the built-in methods do not
call overridden methods (`set_many` and `update` do not go through your
`__setitem__`), and `copy()` returns the base class. `pickle`,
`copy.copy` and `copy.deepcopy` keep the subclass and its attributes, and
never call the subclass's `__init__`.

`DirectMap` and its typed variants (or `direct_map(dtype)`) store values in
pages at their key's slot and always visit entries in ascending key order.
They support nearest-entry and half-open range queries. Array results are
copies, and structural changes are always allowed.

```python
from paged_collections import direct_map

ordered = direct_map("float64", [2048, 7, 1024], [3.0, 1.0, 2.0])
ordered.next_occupant(8)          # (1024, 2.0)
ordered.items_range(7, 2048)      # [(7, 1.0), (1024, 2.0)]
ordered.popitem()                 # (2048, 3.0)
```

`IdAllocator` issues generational ids as Python ints packed as
`generation << 32 | index`. Freed indices are reused last freed first with
the next generation, so a stale id is never alive again; pickles and copies
keep the whole allocation history. Store whole `uint64` ids in `IdMap` (or
`id_map(dtype)` for typed values): inserting a new generation replaces the
entry at that index, and stale ids no longer find it.

```python
from paged_collections import IdAllocator, IdMap, id_generation, id_index

ids = IdAllocator()
a, b = ids.alloc(), ids.alloc()
names = IdMap([a], ["old"])
ids.free(a)                       # True
c = ids.alloc()                   # reuses index 0
names[c] = "new"                  # replaces a's entry
(id_index(c), id_generation(c))   # (0, 2)
a in ids                          # False
a in names                        # False
names[c]                          # "new"
```

`Hive` stores any objects in reusable slots and hands out ids of the same
form: `insert` returns an id, removing a value makes its id stale for good,
and a freed slot is reused with the next generation. Iteration is in slot
order; pickles and copies keep every id and the generation history. An
`IdMap` keyed by hive ids makes a side table.

```python
from paged_collections import Hive

hive = Hive()
a = hive.insert("a")
ids = hive.insert_many(["b", "c"])   # array of uint64 ids
del hive[a]
c = hive.insert("d")                 # reuses a's slot, next generation
a in hive                            # False
hive[c]                              # "d"
```

`PagedList` is a list stored in pages of elements, so appending never
moves existing elements; `paged_list(dtype)` picks a typed list (float64,
int64 and the other map dtypes). Slices are copies. A typed list's
`page_view(i)` is a zero-copy NumPy view of one page that stays valid while
the list grows; while any view is alive, operations that remove, move or
reorder elements raise `BufferError`.

```python
from paged_collections import paged_list

values = paged_list("float64", [1.0, 2.0, 3.0])
page = values.page_view(0)
page[0] = 4.0                     # updates values[0]
values.extend(range(5000))        # allowed: existing pages never move
values.pop()                      # BufferError while `page` is alive
```

Every paged class takes a keyword-only `page_size`, one of
`paged_collections.PAGE_SIZES` (by default 256, 512, 1024, 2048 and 4096),
with `DEFAULT_PAGE_SIZE` (1024) when omitted. Instances keep their size
through `copy`, pickling and `repr`. Pickles from earlier versions load at
1024.

```python
from paged_collections import SparseMap, PAGE_SIZES

small = SparseMap(page_size=256)   # less memory per sparse cluster
small.page_size                    # 256
SparseMap(page_size=300)           # ValueError: not compiled in
```

Rust needs page sizes at compile time, so every size in the list is compiled
into the extension, and each one adds code and build time. To build a wheel
with other sizes, set these variables when building from source:

```bash
PAGED_COLLECTIONS_PAGE_SIZES=64,1024,16384 PAGED_COLLECTIONS_DEFAULT_PAGE_SIZE=1024 maturin build --release
```

Sizes are integers in `1..=1048576` without duplicates, and the default must
be one of them. A pickle made with a size the reading build lacks raises
`ValueError` when loaded.

## Layout

```text
pyproject.toml                    Python package (maturin), at the repository root
python/paged_collections/         Python sources and type stubs
tests/                            Python tests
crates/paged-collections/         Rust library (no_std + alloc)
crates/paged-collections-python/  PyO3 bindings (the `_core` extension)
crates/bench/                     Benchmarks (not a default member)
```

## Development

Rust:

```bash
cargo test
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo build -p paged-collections --no-default-features
```

Python (from the repository root):

```bash
uv sync
uv run pytest
```

`cargo build --workspace` fails to link the Python crate unless the Python
development library is installed; build it through maturin (`uv sync` or
`maturin develop`) instead. `cargo check` and `cargo clippy` work either way.

## License

MIT. See [LICENSE](LICENSE).
