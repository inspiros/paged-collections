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
| `ConcurrentPagedVec<T>` | push | removal, editing, `shrink_to_fit`, conversion | `&T` |
| `ConcurrentQueue<T>` (Python `ConcurrentQueue`) | push, pop | clear, drain, `shrink_to_fit` | owned value from `pop` |
| `ConcurrentPriorityQueue<K, P>` (Python `ConcurrentPriorityQueue`, `ConcurrentPriorityQueueF64`) | push, change priority, remove, pop | clear, drain, `shrink_to_fit` | cloned priority or closure |
| `ConcurrentSparseSet<K>`, `ConcurrentSparseMap<K, V>` | insert | removal, clear, drain, `shrink_to_fit`, conversion | `&K`, `&V` |
| `ConcurrentDirectMap<K, V>` | insert, replace, remove, entry (per-page `RwLock`) | `drain`, `shrink_to_fit`, `reset`, `into_inner` | closures (`with`), clones (`get_cloned`), entry guards |
| `AtomicDirectMap` (`usize -> u64`) | lock-free insert, update, remove | `shrink_to_fit` | copied `u64` |
| `AtomicIdMap` (`Id -> u32`) | lock-free insert, update, remove | `shrink_to_fit` | copied `u32` |
| `ConcurrentIdAllocator` | allocate, free, occupant, iter | clear | copied `Id` |

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
  `shrink_to_fit` frees pages wholly beyond the vector's length.
- **Sparse set and map**: entries cannot be removed or changed while shared.
  `shrink_to_fit` frees index pages with no entry and dense pages beyond the
  last entry; stored entries keep their addresses.
  Inserts of the same slot wait for its writer; other slots proceed
  independently. An occupied slot refuses other generations. Each
  `try_insert` reserves a dense position from one shared counter; under
  contention use `try_insert_many`, which reserves once per batch.
  `extend` and `collect` keep the first occupant of each slot, including
  existing occupants when extending. `drain` yields owned items in dense
  order and keeps allocated pages for reuse.
- **ConcurrentDirectMap**: neighbouring keys share a page lock and contend.
  Release guards and entries before locking the same page again. Ordered
  shared traversals are consistent per page only. `drain` yields owned
  entries in key-index order and keeps allocated pages for reuse.
- **AtomicDirectMap, AtomicIdMap**: prefer them over `ConcurrentDirectMap`
  when values fit: `u64` values below `u64::MAX`, or any `u32` value keyed by
  `Id`. Traversals copy values in index order.

The direct map and both atomic maps release empty pages with `shrink_to_fit`.
Flat directory chunks stay allocated. Radix collapses branches with only one
remaining child. Shared insertion can reinstall released pages.

Use `for item in &collection` on containers with `iter` and `for item in
&mut collection` on containers with `iter_mut`. These loops use the same
item types, order and consistency as the respective iterator methods.

## ConcurrentQueue

`ConcurrentQueue` is an unbounded multi-producer, multi-consumer FIFO queue,
adapted from Crossbeam's `SegQueue`. It recycles pages through a free stack and
waits at page boundaries. A pop can also wait for a push that has claimed a
slot to publish its value. Shared peeking, iteration, cloning, snapshots and
serialization are unavailable; an owned queue can be converted to a
`PagedDeque` for serialization. The Python `ConcurrentQueue` holds Python
objects and drains into a list. See [Python Threads](python-threads.md#concurrentqueue).

```rust
{{#include ../../../crates/paged-collections/examples/doc_concurrent.rs:queue}}
```

## ConcurrentPriorityQueue

`ConcurrentPriorityQueue` is an indexed max-priority queue split into shards
by key page. Its `pop` is relaxed: it chooses the greatest of up to two sampled
shard tops, with a fallback scan, and does not guarantee the global maximum.
Use `pop_exact` to select the global maximum. Per-key operations lock one
shard; `pop_exact` locks all shards. Callbacks and user trait implementations
run under shard locks and must not call back into the same queue. The Python
classes `ConcurrentPriorityQueue` and `ConcurrentPriorityQueueF64` store
numeric priorities and also support min queues. See [Python Threads](python-threads.md#concurrent-priority-queues).

```rust
{{#include ../../../crates/paged-collections/examples/doc_concurrent.rs:priority_queue}}
```

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

`occupant(index)` reads the live generation at one index. Iterating over
`&allocator` scans live ids in index order up to the bound read at the start.
The scan takes no snapshot. Allocations and frees during iteration may or may
not be reflected, and a yielded id can already have been freed.

## Element Invariance

Shared insertion requires every element type to be invariant: shortening a
stored reference's lifetime would let a shorter-lived reference be inserted
while the owner still expects `'static`. The containers are invariant in
their element types; compile-fail tests check this for each type.

## Without `std`

Under `no_std`, the per-page locks of `ConcurrentDirectMap` spin. Lock guards
are `!Send` there.

API: [`concurrent`](https://docs.rs/paged-collections/latest/paged_collections/concurrent/index.html).
