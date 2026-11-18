# Concurrent Containers

Rust feature `concurrent` (see [Installation](installation.md#rust)). The
module uses `core` atomics and `alloc`; it does not need `std`.

## Model: Grow Shared, Shrink Exclusive

Shared access (`&self`) may install pages and recycle slots. Allocated pages
and directory chunks stay at stable addresses until exclusive access (`&mut
self`) or destruction, so memory reclamation needs no epochs or hazard
pointers. Removal of pages, conversion to plain types and `shrink_to_fit`
need `&mut self`.

## Example

```rust
{{#include ../../../crates/paged-collections/examples/doc_concurrent.rs:2:}}
```

```text
{{#include ../examples/concurrent.rust.txt}}
```

## Containers

| Type | Shared (`&self`) | Exclusive (`&mut self`) | Lookups return |
|---|---|---|---|
| `ConcurrentHive<T>` | insert, remove, take | iteration with `&mut`, drain, `shrink_to_fit`, `into_hive` | `Ref` guard (two atomic writes per lookup) |
| `ConcurrentAppendHive<T>` | insert | remove, mutable access, `shrink_to_fit`, `into_hive` | `&T` (one Acquire load) |
| `ConcurrentPagedVec<T>` | push | removal, editing, conversion | `&T` |
| `ConcurrentSparseSet<K>`, `ConcurrentSparseMap<K, V>` | insert | removal, clear, conversion | `&K`, `&V` |
| `ConcurrentDirectMap<K, V>` | insert, replace, remove, entry (per-page `RwLock`) | `reset`, `into_inner` | closures (`with`), clones (`get_cloned`), entry guards |
| `AtomicDirectMap` (`usize -> u64`) | lock-free insert, update, remove | | copied `u64` |
| `AtomicIdMap` (`Id -> u32`) | lock-free insert, update, remove | | copied `u32` |
| `ConcurrentIdAllocator` | alloc, free | | |

Notes per type:

- **ConcurrentHive**: a lookup guard pins its value until dropped; hot values
  contend even between readers. Prefer the plain `Hive` when shared mutation
  is not needed.
- **ConcurrentAppendHive**: iteration may miss insertions in flight.
  `From<Hive>` and `into_hive` keep live ids and stale-id history, in
  O(index bound).
- **ConcurrentPagedVec**: pushes do not wait for earlier writers. Point
  lookups see completed pushes; `len`, iteration and page slices use a
  hole-free prefix that can lag while an earlier push is in flight.
- **Sparse set and map**: entries cannot be removed or changed while shared.
  Inserts of the same slot wait for its writer; other slots proceed
  independently. An occupied slot refuses other generations. Each
  `try_insert` reserves a dense position from one shared counter; under
  contention use `try_insert_many`, which reserves once per batch.
- **ConcurrentDirectMap**: neighbouring keys share a page lock and contend.
  Release guards and entries before locking the same page again. Ordered
  traversals are consistent per page only.
- **AtomicDirectMap, AtomicIdMap**: prefer them over `ConcurrentDirectMap`
  when values fit: `u64` values below `u64::MAX`, or any `u32` value keyed by
  `Id`. Traversals copy values in index order.

## Directories

The sparse types, `ConcurrentDirectMap` and `AtomicDirectMap` take a
`SharedDirectory` parameter: `Flat` (default; O(1) page lookup, finite index
range) or `Radix` (any `usize` index, memory proportional to installed pages,
O(depth) lookup). `AtomicIdMap` is Flat-only; `Tree` has no concurrent form.

## Id Allocation Under Contention

With `std`, freed slots go to one of 32 per-thread shards (`no_std` builds
use one shared stack), each with its own free stack
and block of fresh indices; a thread steals from other shards only when its
own is empty. A single-threaded caller takes an inline fast path. Measured
scaling is in [Benchmarks](benchmarks.md#concurrent-scaling).

## Element Invariance

Shared insertion requires every element type to be invariant: shortening a
stored reference's lifetime would let a shorter-lived reference be inserted
while the owner still expects `'static`. The containers are invariant in
their element types; compile-fail tests check this for each type.

## Without `std`

Under `no_std`, the per-page locks of `ConcurrentDirectMap` spin. Lock guards
are `!Send` there.

API: [`concurrent`](https://docs.rs/paged-collections/latest/paged_collections/concurrent/index.html).
