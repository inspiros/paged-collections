# Python Threads

## Concurrent Classes

`ConcurrentSparseSet`, the typed `ConcurrentSparseMap*` classes (`F64`, `F32`,
`I64`, `I32`, `U64`, `U32`, `Bool`) and the typed `ConcurrentDirectMap*`
classes support shared access from threads.

| | Sparse set and maps | Direct maps |
|---|---|---|
| Shared inserts | set: `insert`, `insert_many`; maps: `try_insert`, `get_or_insert`, `insert_many` | `m[k] = v`, `insert`, `try_insert`, `get_or_insert`, `setdefault`, `set_many` |
| Shared removal | no | `del m[k]`, `remove`, `remove_many` |
| Exclusive (conflicts raise) | `remove`, `remove_many`, `clear`, `shrink_to_fit` | `clear`, `shrink_to_fit` |

On a direct map, assignment, `insert` and `set_many` replace a stored value.
`get_or_insert(key, value)` and `setdefault(key, value)` atomically insert if
the key is absent and return the stored value. `shrink_to_fit()` (on every
concurrent set and map) frees pages with no live entries. Concurrent sparse map insertions keep existing values.

- Default page size and `Flat` directory only; the constructors take no
  `page_size=`.
- Equality and hashing are by identity, unlike the other classes.
- Batch methods (`*_many`) release the GIL; single-key calls do not.
- Results are copies (arrays, lists); there are no guards or views.
- A conflicting exclusive call raises `RuntimeError("Already borrowed")`.
- Iteration visits a snapshot of keys with the same order and consistency as
  `keys()`. The maps' `contains_many(keys)` returns a boolean array, treating
  invalid keys as absent just as `key in m` does.
- Direct-map snapshots are in ascending key order. `items_range(start=None,
  stop=None)`, `keys_range` and `values_range` cover the half-open interval
  `[start, stop)`. Bounds are optional and clamped to the key domain. Like
  `items()`, each call is consistent within a page and weakly consistent
  across pages.
- `len()` is exact only when no write is in flight.
- `ConcurrentSparseSet(keys)` and `ConcurrentSparseMapF64(keys, values)`
  accept NumPy arrays or sequences, validated before any write. Later duplicate
  keys win, even in the insert-only sparse maps.
- `pickle`, `copy.copy`, `copy.deepcopy` and `copy()` take one snapshot,
  with the same consistency as `items()`: entries inserted or removed during
  the call may or may not be included, but keys and values always pair up.

## Example

```python
{{#include ../examples/python_threads.py}}
```

```text
{{#include ../examples/python_threads.python.txt}}
```

## ConcurrentQueue

`ConcurrentQueue(iterable=None)` holds Python objects in FIFO order. Threads
can share `push`, `push_many`, `pop` and `pop_many`. `pop` raises `IndexError`
when empty, and `pop_many(n)` returns up to `n` items, stopping at the first
empty pop. Length and truthiness are approximate during concurrent use and
exact when no operation is in flight.

`clear()`, `drain()` and `shrink_to_fit()` require exclusive access. If another
call holds a borrow, they raise `RuntimeError("Already borrowed")`.
`allocated_pages` counts all allocated pages, including those pooled for reuse.
`shrink_to_fit()` frees pages the current values do not need and preserves FIFO
order. The queue has no iteration, peek, copy or pickle support. Use `drain()`
to move the items into a list in FIFO order before copying or pickling. Python's
garbage collector cannot collect reference cycles through a queue.

## Concurrent Priority Queues

`ConcurrentPriorityQueue` stores int64 priorities and
`ConcurrentPriorityQueueF64` stores float64 priorities, with NaN rejected.
Both take integer keys in `0 <= key < 2**32`. Pass `min=True` to prefer the
lowest priority and `shards=` to choose a shard count from 1 to 65536. The initial
items can be a mapping or iterable of pairs. Later duplicate keys win.
`concurrent_priority_queue(dtype, items, min, shards=...)` selects the class
by dtype.

`push_increase(key, priority)` and `push_decrease(key, priority)` insert an
absent key or update its priority only when the new numeric value is strictly
greater or strictly smaller, respectively, in either queue direction. The
methods return whether the queue changed.

`pop()` uses relaxed ordering. With multiple shards, the entry it returns
can rank below the global best. `pop_exact()` removes the global best and
`peek_exact()` reads it. Both exact operations lock all shards. Ties have
unspecified order. All three methods raise `IndexError` when they find no entry.

`items()`, `sorted()`, `repr()`, copying and pickling each read one snapshot.
It is consistent within each shard and weakly consistent across shards.
`sorted()` lists the best priorities first. The repr includes the class,
`min`, `shards` and up to 20 entries. Copies and pickles preserve `min` and
`shards`. Length and truthiness are exact when no write is in flight.

`clear()`, `drain()` and `shrink_to_fit()` require exclusive access and raise
`RuntimeError("Already borrowed")` if another call holds a borrow. `drain()`
returns all `(key, priority)` pairs in arbitrary order and keeps allocated
pages. `shrink_to_fit()` shrinks shard storage and frees the index pages of
empty shards.

## Free-threaded CPython

Wheels exist for free-threaded 3.14t and 3.15t+ (see
[Installation](installation.md#python)). Importing the package keeps the GIL
disabled, and the concurrent classes run in parallel.

The other classes are safe to share but are not designed for it: a call that
conflicts with another thread's call on the same object raises
`RuntimeError("Already borrowed")`, or `BufferError` while views exist. Guard
shared use of those classes with a lock.

Zero-copy views (`view()`, `keys_view()`, `values_view()`) are not covered by
those checks: NumPy reads and writes their memory directly and may release
the GIL while doing so. Writing through a view while another thread reads or
writes the collection, or reading a view while another thread writes the
collection, is a data race, with or without the GIL. Hold the same lock for
the views as for the collection, or work on copies (`keys()`, `values()`).

Free-threaded 3.13t is not supported.

## Scaling

`benchmarks/python/concurrent_scaling.py` measures batch throughput over
thread counts; results are in [Benchmarks](benchmarks.md#concurrent-scaling).
