# Paged Vector

`PagedVec<T, PAGE = 1024>` is a growable vector stored in fixed-size pages. A
page is allocated once and never reallocated, so pushing never moves or copies
existing elements.

See [how paged storage grows](paging.md#paged-storage).

Python: `PagedList` (objects) and `PagedListF64`, `PagedListF32`,
`PagedListI64`, `PagedListI32`, `PagedListU64`, `PagedListU32`,
`PagedListBool`; `paged_list(dtype)` creates an instance of the class for a dtype. They implement the
`list` protocol.

## Example

<div class="lang-tabs">

```python
{{#include ../examples/paged_vec.py}}
```

```rust
{{#include ../../../crates/paged-collections/examples/doc_paged_vec.rs:2:}}
```

</div>

Python output:

```text
{{#include ../examples/paged_vec.python.txt}}
```

The Rust example also prints `same address = true`: element 0 kept its address through 10,000 pushes.

## Address Stability

- An element keeps its address while it stays at the same index, through any
  number of `push`, `extend` and `reserve` calls.
- Operations that change which element sits at an index (`insert`, `remove`,
  `swap`, `swap_remove`, `retain`, `dedup`, `drain`, `extract_if`) move
  elements, as in a `Vec`.
- `split_off` at a page boundary transfers whole pages without moving their
  elements.

## Complexity

| Operation | Cost |
|---|---|
| `get`, indexing, `pop` | O(1) |
| `push` | amortized O(1): one page allocation every `PAGE` pushes, rare directory growth |
| `insert`, `remove` | O(len − index) |
| `truncate`, `clear` | O(removed) drops; pages are kept |
| `extract_if` | O(visited), plus shifting the tail after a removal. No allocation. |
| `split_off` | O(transferred pages) at a page boundary, else O(len − at) |
| `binary_search`, `partition_point` | O(log n), no allocation |
| `sort` and variants | the slice sort, plus moving all elements out to a temporary buffer and back |

The worst-case push costs one page allocation plus a copy of the page
directory, which is `PAGE` times shorter than the elements. A `Vec` instead
copies all elements on growth.

In Rust, `extract_if(range, filter)` visits that range in order and yields
the elements for which the filter returns `true`. The filter can mutate each
element. Dropping the iterator early keeps the unvisited elements in order.
If the filter panics, the current element is kept with its mutations.

## No Contiguous Slice

There is no `as_slice` or `Deref<Target = [T]>`. Use `pages()` and
`pages_mut()` for per-page slices, or `with_contiguous_mut` to run a slice
algorithm on all elements through a temporary buffer. Typed Python lists
expose pages as NumPy views (`page_view(i)`, `page_views()`) and copy
everything with `to_numpy()`.

## Rust and Python

| Rust | Python |
|---|---|
| `push`, `extend` | `append`, `extend`, `+=` |
| `pop`, `insert`, `remove` | `pop([i])`, `insert(i, v)`, `del l[i]`; `remove(value)` searches by value |
| indexing, `range` | `l[i]`, `l[a:b]` (slices copy) |
| `extract_if(range, filter)` | |
| `sort*` | `sort(key=, reverse=)` (typed lists: `reverse=` only) |
| `pages()` | typed: `page_view(i)`, `page_views()` |
| | typed: `get_many(indices)`, `set_many(indices, values)`, `to_numpy()`, `np.asarray(l)` |
| `reserve`, `shrink_to_fit`, `capacity` | same names; `capacity`, `allocated_pages` are properties |

API: [`PagedVec`](https://docs.rs/paged-collections/latest/paged_collections/paged_vec/struct.PagedVec.html).
