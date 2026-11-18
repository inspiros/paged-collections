# Sparse Map

`PagedSparseMap<K, V, PAGE = 1024, D = Flat>` is a [sparse set](sparse-set.md)
of keys plus a dense value array kept parallel to the dense keys: the value at
position `i` belongs to the key at position `i`. Keys and values can be
iterated or borrowed as slices without touching the index.

See [where sparse and direct maps store their values](choosing.md#sparse-map-or-direct-map).

Python: `SparseMap` holds arbitrary objects. `SparseMapF64`, `SparseMapF32`,
`SparseMapI64`, `SparseMapI32`, `SparseMapU64`, `SparseMapU32` and
`SparseMapBool` store values unboxed; `sparse_map(dtype)` returns an instance
of the class for a NumPy dtype.

## Example

<div class="lang-tabs">

```python
{{#include ../examples/sparse_map.py}}
```

```rust
{{#include ../../../crates/paged-collections/examples/doc_sparse_map.rs:2:}}
```

</div>

Python output:

```text
{{#include ../examples/sparse_map.python.txt}}
```

The Rust example prints the same lines.

## Complexity

| Operation | Cost |
|---|---|
| get, remove | O(1) with `Flat` (see [directories](paging.md#directories)) |
| insert | as get, amortized over dense-array growth; the first key in a page allocates the page |
| clear | O(1) for the index, plus dropping the values |
| iteration, `key_slice`, `value_slice` | dense arrays only |
| sort | O(n log n) comparisons, O(n) swaps, one index write per entry |
| `binary_search_key`, `binary_search_by`, `partition_point` | O(log n) on entries kept sorted by the caller |

Removal swaps the last entry into the removed position, as in the set.

## Generational Keys

With plain integer keys the map behaves like `HashMap`. With generational
keys the methods differ when the slot is held by a *different* key:

| Method | Slot held by another generation |
|---|---|
| `insert`, `get_or_insert_with` | evicts it |
| `replace` | evicts it and returns the evicted entry |
| `try_insert` | refuses |

`insert` evicts so that `get(key)` returns the new value afterwards, as with
`HashMap`.

## Rust and Python

| Rust | Python |
|---|---|
| `insert(k, v)` | `m[k] = v`, `set_many(keys, values)`, `update(mapping)` |
| `get(k)`, `get_mut(k)` | `m[k]`, `get(k, default)`, `get_many(keys, default)` |
| `get_or_insert_with` | `setdefault(k, default)` |
| `remove(k)` | `del m[k]`, `pop(k[, default])`, `popitem()`, `discard_many(keys)` |
| `contains_key(k)` | `k in m`, `contains_many(keys)` |
| `key_slice()` | `keys()` (copy), `keys_view()` (zero-copy, read-only) |
| `value_slice()`, `value_slice_mut()` | typed maps: `values()` (copy), `values_view()` (zero-copy; writeable except for `SparseMapBool`); `SparseMap.values()` returns a list |
| `sort_keys`, `sort_by*` | `sort_keys()` |
| `allocated_pages()` | `allocated_pages` |

Batch methods are described in [NumPy Interop](numpy.md).

API: [`PagedSparseMap`](https://docs.rs/paged-collections/latest/paged_collections/sparse_map/struct.PagedSparseMap.html).
