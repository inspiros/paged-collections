# Python Threads

## Concurrent Classes

`ConcurrentSparseSet`, the typed `ConcurrentSparseMap*` classes (`F64`, `F32`,
`I64`, `I32`, `U64`, `U32`, `Bool`) and the typed `ConcurrentDirectMap*`
classes support shared access from threads.

| | Sparse set and maps | Direct maps |
|---|---|---|
| Shared inserts | set: `insert`, `insert_many`; maps: `try_insert`, `get_or_insert`, `insert_many` | `m[k] = v`, `insert`, `try_insert`, `set_many` |
| Shared removal | no | `del m[k]`, `remove`, `remove_many` |
| Exclusive (conflicts raise) | `remove`, `remove_many`, `clear` | `clear` |
| Existing values on insert | kept (`try_insert` semantics) | replaced |

- Default page size and `Flat` directory only; the constructors take no
  `page_size=`.
- Equality and hashing are by identity, unlike the other classes.
- Batch methods (`*_many`) release the GIL; single-key calls do not.
- Results are copies (arrays, lists); there are no guards or views.
- A conflicting exclusive call raises `RuntimeError("Already borrowed")`.
- Direct-map snapshots (`keys`, `values`, `items`) are in key order.
- `len()` is exact only when no write is in flight.
- Constructors take initial contents like the plain classes:
  `ConcurrentSparseSet(keys)` and `ConcurrentSparseMapF64(keys, values)`,
  from NumPy arrays or sequences, validated before any write. Later duplicate
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
