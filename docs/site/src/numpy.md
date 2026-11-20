# NumPy Interop

## Keys

Integer-keyed classes (`SparseSet`, `SparseMap*`, `DirectMap*`, priority
queues) take keys in `0 <= key < 2**32` (`KEY_LIMIT`). `IdMap*` takes packed
uint64 ids. Keys may be Python ints or objects with `__index__`.

- Batch lookups (`contains_many`, `discard_many`, `get_many`) treat
  out-of-range integers as absent, including arbitrarily large ints inside
  Python sequences or (for integer-keyed classes) object arrays. `IdMap*`
  batches reject object-dtype arrays.
- Batch insertion (`add_many`, `set_many`) rejects out-of-range keys.
- The concurrent maps' `contains_many` accepts an iterable or a
  one-dimensional array and reports invalid elements, including floats, as
  absent. Other batch lookup methods reject float key arrays.

## Batch Methods

| Method | Input | Result |
|---|---|---|
| `add_many(keys)` (set) | array-like of keys | number added |
| `set_many(keys, values)` | keys, and values: typed maps take an array-like or a scalar to broadcast; object maps take an iterable | number of new keys |
| `get_many(keys, default)` | keys | typed maps: array of the value dtype; object maps: list. Without a default, an absent key raises `KeyError` |
| `contains_many(keys)` | keys | bool array |
| `discard_many(keys)` | keys | number removed |
| `keys()`, `values()` | | arrays (copies); object maps return values as a list |

Batch methods run in one call into Rust. Native numeric arrays can avoid
creating a Python object per element.

## Typed Values

`SparseMapF64`, `DirectMapF64`, `PagedListF64`, `IdMapF64` and the other typed
classes store values unboxed in the dtype named by the suffix: `F64`, `F32`,
`I64`, `I32`, `U64`, `U32`, `Bool`. `sparse_map(dtype)`, `direct_map(dtype)`,
`id_map(dtype)`, `paged_list(dtype)`, `concurrent_sparse_map(dtype)`,
`concurrent_direct_map(dtype)` and `priority_queue(dtype)` create an instance
of the class for a dtype.

A `dtype` argument takes anything `numpy.dtype` accepts: `"float64"`,
`np.float64`, `np.dtype("f8")` and `float` all select `F64`. Python's `int`
means NumPy's default integer, which is int32 on Windows with NumPy 1.x, so
name the width (`"int64"`) for portable code.

Typed containers store values, not objects, so NaN != NaN as in NumPy: a
typed container holding NaN does not compare equal even to itself.

## Zero-Copy Views

| Method | Array | Writeable |
|---|---|---|
| `SparseSet.view()` | dense keys, uint32 | no |
| `keys_view()` (sparse and id maps) | dense keys | no |
| `values_view()` (typed sparse and id maps) | dense values | yes, except `Bool` |
| `page_view(i)`, `page_views()` (typed lists) | one page of elements each | yes, except `Bool` |

Writes through a writeable view update the container. While any view is
alive, operations that would move or resize the underlying buffer raise
`BufferError`. Delete the view (or let it go out of scope) to release it.
A view's `base` is an `ExportGuard` that keeps the container alive. NumPy
arrays are not tracked by Python's garbage collector, so storing a view inside
the container it views (`m[k] = m.keys_view()`) makes a cycle that is never
freed.

Views bypass the borrow checks described in [Python Threads](python-threads.md):
synchronise access to a view with access to its container when threads share
them.

Direct maps have no views: their values live in pages, not one array.

## Example

```python
{{#include ../examples/numpy.py}}
```

```text
{{#include ../examples/numpy.python.txt}}
```
