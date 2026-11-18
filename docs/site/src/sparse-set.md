# Sparse Set

`PagedSparseSet<K, PAGE = 1024, D = Flat>` (Python: `SparseSet`) stores keys in
a contiguous *dense* array and finds them through a paged *sparse* index from
slot index to dense position.

## Example

<div class="lang-tabs">

```python
{{#include ../examples/sparse_set.py}}
```

```rust
{{#include ../../../crates/paged-collections/examples/doc_sparse_set.rs:2:}}
```

</div>

Python output:

```text
{{#include ../examples/sparse_set.python.txt}}
```

The Rust example prints the same lines without the set algebra.

## Complexity

| Operation | Cost |
|---|---|
| insert, remove, contains | O(1) with `Flat`; O(depth) with `Radix`; O(log pages) with `Tree` |
| clear | O(1): pages are kept, stale slots are ignored |
| iteration, `as_slice` | dense array only, O(len) |
| sort | O(n log n) comparisons, then one index write per key |

Lookups follow Briggs and Torczon: a slot is trusted only if it points inside
the dense array at a key with the same slot index, so stale slots left by
`clear` are harmless and clearing does not touch the pages.

## Order

Iteration follows the dense array: insertion order until a removal. `remove`
swaps the last key into the removed position (O(1)), so removal changes the
order. `sort` and its variants reorder the dense array; sorted order then
lasts until the next removal or insertion out of order.

## Generational Keys

The set holds at most one key per slot index. With generational keys,
`insert` refuses a key whose slot is held by another generation and returns
`false`; `replace` evicts the other generation; `occupant(index)` returns the
key holding a slot. See [Ids and Side Tables](ids.md).

## Rust and Python

| Rust | Python | Notes |
|---|---|---|
| `insert(k) -> bool` | `add(k)`, `add_many(keys) -> int` | |
| `remove(k) -> bool` | `discard(k)`, `remove(k)` (raises `KeyError`), `discard_many(keys)` | swap removal |
| `contains(k)` | `k in s`, `contains_many(keys) -> ndarray[bool]` | |
| `as_slice()` | `to_array()` (copy), `view()` (zero-copy, read-only) | |
| `iter()` | `iter(s)` | dense order |
| `sort*` | `sort()` | |
| `position(k)` | | dense position |
| `clear()` / `reset()` | `clear()` | `reset` frees pages |
| `allocated_pages()` | `allocated_pages` | |
| | `union`, `intersection`, `difference`, `symmetric_difference`, the `*_update` forms, `\| & - ^` and in-place operators, `issubset`, `issuperset`, `isdisjoint`, `< <= > >=` | Python set protocol |

Python keys are integers in `0 <= key < 2**32`. Batch lookups
(`contains_many`, `discard_many`) treat out-of-range integers as absent;
`add` and `add_many` raise for them. Set algebra:

- Operators take `SparseSet` operands; methods accept any iterable of keys.
- Results use the receiver's page size, keep the receiver's surviving keys in
  order, then append new keys in operand iteration order.
- `difference_update` and `-=` use swap removal; other in-place operations
  keep the receiver's order.
- In-place operations raise `BufferError` while a `view()` is alive.

API: [`PagedSparseSet`](https://docs.rs/paged-collections/latest/paged_collections/sparse_set/struct.PagedSparseSet.html).
