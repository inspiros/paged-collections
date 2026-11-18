# Direct Map

`PagedDirectMap<K, V, PAGE = 1024, D = Flat>` stores each entry in the slot
its key names, in pages allocated on first use. There is no dense array: a
lookup is one page lookup and one slot read, entries never move, and
iteration visits entries in ascending slot-index order (key order for
integers).

Python: `DirectMap` (objects) and `DirectMapF64`, `DirectMapF32`,
`DirectMapI64`, `DirectMapI32`, `DirectMapU64`, `DirectMapU32`,
`DirectMapBool`; `direct_map(dtype)` creates an instance of the class for a dtype.

## Example

<div class="lang-tabs">

```python
{{#include ../examples/direct_map.py}}
```

```rust
{{#include ../../../crates/paged-collections/examples/doc_direct_map.rs:2:}}
```

</div>

Python output:

```text
{{#include ../examples/direct_map.python.txt}}
```

The Rust example prints the same lines, using the entry API in place of `setdefault`.

## Complexity

| Operation | Cost |
|---|---|
| get, insert, remove | O(1) with `Flat`; the first key in a page allocates the page |
| iteration, clear, retain | O(allocated pages × `PAGE`), plus with `Flat` O(directory length) to skip unallocated pages |
| `range`, `next_occupant`, `prev_occupant`, `first_key_value`, `last_key_value`, `pop_first`, `pop_last` | a seek to the first page (O(1) with `Flat`, O(depth) with `Radix` and `Tree`), then a scan of the allocated slots up to the entry; with `Flat`, unallocated directory entries in between are also skipped one by one |
| len | O(1) |

Iteration and `clear` scan every allocated slot, occupied or not, and every
allocated slot takes the space of an entry. Keys that fill their pages
sparsely make both expensive; see [Choosing a Container](choosing.md#sparse-map-or-direct-map).

## What a Slot Stores

A slot holds the value and the key's `DirectKey::Tag`, the part of the key its
slot index does not give. Integer keys have the tag `()`, so a slot of a
`u32 -> u64` map is 16 bytes and a slot of a `u32 -> Box<T>` map is 8 bytes
(niche optimization). Iterators rebuild each key from slot index and tag and
yield keys by value: `(K, &V)`.

## Ordered Queries

Entries are ordered by slot index, so the map answers ordered-map queries
without a tree:

- `range(a..b)`, `range_mut`: entries between two keys.
- `first_key_value`, `last_key_value`, `pop_first`, `pop_last`: the ends.
- `next_occupant(i)`, `prev_occupant(i)`: the nearest entry at or after, or at
  or before, slot index `i`.

A generational key bounds a range by its index only: `range(id..=id)` yields
whichever generation holds the slot.

## Entry API

`entry(k)` returns `Occupied` only when the slot holds `k` itself. A vacant
entry's insert evicts another generation held in the slot. The usual
combinators exist: `or_insert`, `or_insert_with`, `or_insert_with_key`,
`or_default`, `and_modify`, `insert_entry`.

## Generational Keys

As in the [sparse map](sparse-map.md#generational-keys): `insert` and
`get_or_insert_with` evict another generation, `replace` evicts and returns
it, `try_insert` refuses. Keys must implement `DirectKey` (the `Key` slot
index plus the tag); `u8`, `u16`, `u32`, `u64`, `usize` and `Id` implement it.

## Memory

Pages are not freed when they empty. `shrink_to_fit` frees every page without
entries and shrinks the directory; `reset` frees all pages.

## Rust and Python

| Rust | Python |
|---|---|
| `insert(k, v)` | `m[k] = v`, `set_many(keys, values)`, `update(mapping)` |
| `get(k)` | `m[k]`, `get(k, default)`, `get_many(keys, default)` |
| `get_or_insert_with`, `entry` | `setdefault(k, default)` |
| `remove(k)` | `del m[k]`, `pop(k[, default])`, `popitem()`, `discard_many(keys)` |
| `keys()`, `values()`, `iter()` | `keys()` (uint32 array), `values()`, `items()` |
| `range(a..b)` | `keys_range(a, b)`, `values_range(a, b)`, `items_range(a, b)` (stop exclusive) |
| `first_key_value`, `last_key_value` | `first_item()`, `last_item()` |
| `pop_first`, `pop_last` | `pop_first()`, `pop_last()` |
| `next_occupant`, `prev_occupant` | `next_occupant(i)`, `prev_occupant(i)` |
| `shrink_to_fit`, `reset` | `shrink_to_fit()` |

Python direct maps have no zero-copy views: values are in pages, not in one
array.

API: [`PagedDirectMap`](https://docs.rs/paged-collections/latest/paged_collections/direct_map/struct.PagedDirectMap.html).
