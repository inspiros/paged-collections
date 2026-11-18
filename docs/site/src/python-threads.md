# Python Threads

## Concurrent Classes

`ConcurrentSparseSet`, the typed `ConcurrentSparseMap*` classes (`F64`, `F32`,
`I64`, `I32`, `U64`, `U32`, `Bool`) and the typed `ConcurrentDirectMap*`
classes support shared access from threads.

| | Sparse set and maps | Direct maps |
|---|---|---|
| Shared inserts | `insert`, `insert_many`, `try_insert`, `get_or_insert` | `m[k] = v`, `insert`, `set_many` |
| Shared removal | no | `del m[k]`, `remove` |
| Exclusive (conflicts raise) | `remove`, `remove_many`, `clear` | `clear` |
| Existing values on insert | kept (`try_insert` semantics) | replaced |

- Default page size and `Flat` directory only.
- Batch methods (`*_many`) release the GIL; single-key calls do not.
- Results are copies (arrays, lists); there are no guards or views.
- A conflicting exclusive call raises `RuntimeError("Already borrowed")`.
- Direct-map snapshots (`keys`, `values`, `items`) are in key order.
- `len()` is exact only when no write is in flight.
- Not picklable.

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

Free-threaded 3.13t is not supported.

## Scaling

`benchmarks/python/concurrent_scaling.py` measures batch throughput over
thread counts; results are in [Benchmarks](benchmarks.md#concurrent-scaling).
