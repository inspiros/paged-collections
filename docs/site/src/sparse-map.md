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
| `entry` | returns `Vacant`. Inserting through it evicts the other generation in place |
| `replace` | evicts it and returns the evicted entry |
| `try_insert` | refuses |

`insert` evicts so that `get(key)` returns the new value afterwards, as with
`HashMap`.

In Rust, `entry(key)` gives mutable access to an exact key or lets you insert
it after a single slot lookup. For example, use
`*counts.entry(key).or_default() += 1` to update a counter.

## Rust and Python

| Rust | Python |
|---|---|
| `insert(k, v)` | `m[k] = v`, `set_many(keys, values)`, `update(mapping)` |
| `get(k)`, `get_mut(k)` | `m[k]`, `get(k, default)`, `get_many(keys, default)` |
| `get_or_insert_with` | `setdefault(k, default)` |
| `entry(k)` | |
| `remove(k)` | `del m[k]`, `pop(k[, default])`, `popitem()`, `discard_many(keys)` |
| `first_key_value()`, `last_key_value()` | `first_item()`, `last_item()` |
| `pop_first()`, `pop_last()` | `pop_first()`, `pop_last()` |
| `next_occupant`, `prev_occupant` | `next_occupant(index)`, `prev_occupant(index)` |
| `range(a..b)` | `items_range(a, b)`, `keys_range(a, b)`, `values_range(a, b)` |
| `contains_key(k)` | `k in m`, `contains_many(keys)` |
| `key_slice()` | `keys()` (copy), `keys_view()` (zero-copy, read-only) |
| `value_slice()`, `value_slice_mut()` | typed maps: `values()` (copy), `values_view()` (zero-copy; writeable except for `SparseMapBool`); `SparseMap.values()` returns a list |
| `sort_keys`, `sort_by*` | `sort_keys()` |
| `allocated_pages()` | `allocated_pages` |

Python `SparseMap`, `IdMap` and their typed variants accept parallel keys and
values, one mapping, or one iterable of `(key, value)` pairs. They validate
all input before writing and keep the last value for duplicate keys.

Ordered queries follow the sparse index. `first_item()` and `last_item()`
return the `(key, value)` pairs with the smallest and largest slot indices.
`pop_first()` and `pop_last()` remove and return those pairs by swap removal.
All four raise `KeyError` on an empty map, and ordered pops raise `BufferError`
while a NumPy view is alive.

`next_occupant(index)` and `prev_occupant(index)` return the nearest pair at or
after, or at or before, the given slot index. They return `None` when no entry
matches. Both accept integers outside the slot domain. A negative index is
before the first slot, and an index of `2**32` or higher is past the last slot.
For `IdMap` and its typed variants, pass a slot index and receive a packed ID
as the key in the returned pair.

`items_range(start=None, stop=None)`, `keys_range` and `values_range` return
copies in ascending slot order for the half-open interval `[start, stop)`.
`None` leaves a bound open and integer bounds are clamped to the slot domain.
For `IdMap`, the bounds are slot indices and the returned keys are packed IDs.

Batch methods are described in [NumPy Interop](numpy.md).

API: [`PagedSparseMap`](https://docs.rs/paged-collections/latest/paged_collections/sparse_map/struct.PagedSparseMap.html).
