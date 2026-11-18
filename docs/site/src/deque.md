# Deque

`PagedDeque<T, PAGE = 1024>` is a double-ended queue stored in fixed-size
pages. An element keeps its address for as long as it stays in the deque,
through pushes and pops at either end, `reserve` and `shrink_to_fit`. Spare
pages are kept at the back and reused at either end.

Python: `PagedDeque` follows `collections.deque` (CPython 3.12 semantics),
including `maxlen`, `rotate` and the comparison operators.

## Example

<div class="lang-tabs">

```python
{{#include ../examples/deque.py}}
```

```rust
{{#include ../../../crates/paged-collections/examples/doc_deque.rs:2:}}
```

</div>

Python output:

```text
{{#include ../examples/deque.python.txt}}
```

The Rust example prints the same lines without the rotation.

## Complexity

| Operation | Cost |
|---|---|
| pops, access, indexing, swapping | O(1) |
| pushes | amortized O(1), occasional page allocation and directory growth |
| `insert`, `remove` | O(min(index, len − index)) moves |
| `truncate`, `clear` | O(removed) drops; pages are kept |
| `binary_search`, `partition_point` | O(log len) |
| `reserve`, `shrink_to_fit` | O(allocated or freed pages) |

Front operations change indices without moving the remaining elements.
Insertion, removal, swapping and retention can move elements.

Elements are not contiguous: `pages()` returns the initialized part of each
used page, and the first and last slices can be partial.

## Rust and Python

| Rust | Python |
|---|---|
| `push_back`, `push_front` | `append`, `appendleft` |
| `pop_back`, `pop_front` | `pop`, `popleft` |
| `extend` | `extend`, `extendleft` |
| `front`, `back`, indexing | `d[0]`, `d[-1]`, `d[i]` |
| `insert`, `remove` | `insert(i, v)`, `del d[i]`; `remove(value)` searches by value |
| | `rotate(n)`, `reverse()`, `count`, `index`, `maxlen` |
| `reserve`, `shrink_to_fit` | same names |

API: [`PagedDeque`](https://docs.rs/paged-collections/latest/paged_collections/paged_deque/struct.PagedDeque.html).
