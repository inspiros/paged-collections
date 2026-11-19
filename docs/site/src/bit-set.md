# Bit Set

`PagedBitSet<K, PAGE = 512, D = Flat>` (Python `BitSet`) stores membership
as one bit per slot in each allocated page. Pick it for integer keys that
cluster in dense ranges, especially for ordered traversal, range updates
or set algebra. A [sparse set](sparse-set.md) visits its dense key array
in O(len), which can be faster when the occupied pages hold few keys. It
also supports generational keys and contiguous access to the keys.

## Pages and Costs

`PAGE` in Rust and `page_size` in Python count **64-bit words**. The default
of 512 words uses 4 KiB per allocated page and covers 32,768 keys. Pages are
allocated as keys are inserted. Clearing the set or removing keys keeps the
pages for reuse. In Rust, `shrink_to_fit()` frees empty pages and `reset()`
frees all pages.

The Python binding uses the `Flat` directory and accepts the sizes in
`PAGE_SIZES`. Each directory slot holds one page pointer up to the
highest allocated page. At the default size, the full Python key domain
needs at most 131,072 directory slots (1 MiB of pointers on a 64-bit
platform). Rust can use [a different directory](paging.md#directories)
for keys spread over a wider domain.

| Operation | Cost with `Flat` |
|---|---|
| Membership, insert, remove | O(1) per key, plus allocation on first touch |
| Length | O(1) |
| Iteration, nearest-key queries | Scan directory slots and words in the visited span, then visit set bits |
| Range add | Visit words in the range and allocate newly touched pages |
| Range discard, count | Scan directory slots and words in allocated pages that overlap the range |
| Clear | Zero the words in all allocated pages |
| Same-size set algebra | Scan relevant pages word by word |

## Rust

Integer key types implement `DirectKey<Tag = ()>`. Ranges take Rust
`RangeBounds<K>` and refer to the keys' slot indices. Use explicit upper
bounds for narrow integer key types when adding a range. Bit sets with
the same `PAGE` support algebra across directory layouts.

```rust
use paged_collections::bit_set::PagedBitSet;

let mut bits: PagedBitSet<u32> = PagedBitSet::new();
assert_eq!(bits.insert_range(62..67), 5);
assert_eq!(bits.remove_range(64..66), 2);
assert_eq!(bits.range(63..).collect::<Vec<_>>(), [63, 66]);
assert_eq!(bits.next_occupant(64), Some(66));
assert_eq!(bits.pop_first(), Some(62));
```

## Python

```python
{{#include ../examples/bit_set.py}}
```

```text
{{#include ../examples/bit_set.python.txt}}
```

- Keys are integers in `0 <= key < 2**32`. Membership and discard treat
  invalid keys as absent. `add` raises for invalid keys and `remove`
  raises `KeyError` for absent keys.
- `BitSet(keys=None, *, page_size=DEFAULT_BIT_PAGE_SIZE)` accepts an
  initial one-dimensional integer array or sequence.
- `first()` and `last()` return the smallest and largest keys. `pop()`
  removes and returns the smallest key. All three raise `KeyError` on an
  empty set. `next_occupant(index)` and `prev_occupant(index)` return the
  nearest key at or beyond the given key in the respective direction,
  or `None` if there is no match. The index must be a valid key.
- `add_range` and `discard_range` return the number of keys changed.
  `count_range` counts keys and `range` iterates over them. Both endpoints
  of `[start, stop)` must be integers in `0 <= endpoint <= 2**32`.
  With `start >= stop` the interval is empty.
- Iteration is ascending, or descending with `reversed(bits)`. Iterators
  raise `RuntimeError` if membership changes during iteration.
- `union`, `intersection`, `difference`, `symmetric_difference`, their
  in-place methods (`update` for union) and the predicates accept iterable
  operands.
  Operators `| & - ^`, their in-place forms and comparisons take
  `BitSet` operands. Two bit sets with the same page size use word-wise
  algebra. With different sizes, algebra visits keys individually.
  New algebra results are base `BitSet` instances using the receiver's
  page size. Pickling and copying preserve the page size.
- `add_many` and `discard_many` return the number of keys changed.
  `contains_many` returns a boolean NumPy array. `to_numpy()` returns
  a new ascending `uint32` array.
- `from_mask(mask, offset=0)` reads a one-dimensional boolean array or
  sequence. A true value at position `i` sets `offset + i`. The whole
  mask must fit in the key domain. `to_mask(start, stop)` returns a new
  boolean array with one element per key in the interval, using
  O(max(0, stop − start)) memory.

API [`PagedBitSet`](https://docs.rs/paged-collections/latest/paged_collections/bit_set/struct.PagedBitSet.html).
