# paged-collections

Paged sparse sets, maps and stable-address containers for Rust, with Python
bindings built on PyO3.

> Status: in development. `PagedSparseSet` and `PagedSparseMap` are
> implemented in Rust and wrapped for Python; `PagedDirectMap`, `IdAllocator` and
> `PagedVec` are in Rust only; the other collections are not yet. Nothing is published.

## Why paging

Paging serves two separate purposes:

- **Paged index.** A table indexed by key is split into fixed-size pages
  allocated on first use, so memory follows the occupied key ranges, not the
  largest key. A plain sparse set with key `4_000_000_000` needs a 16 GB
  sparse array; a paged one needs one page.
- **Paged storage.** Elements live in fixed-size pages that never move, so
  references stay valid while the collection grows and growth never copies
  the whole buffer.

## Planned collections

| Type | Paging | Purpose |
|---|---|---|
| `PagedSparseSet<K>` | index (storage optional) | Integer set with O(1) insert, remove, contains, clear, and dense iteration |
| `PagedSparseMap<K, V>` | index (storage optional) | Sparse set with a dense value array |
| `PagedDirectMap<K, V>` | index | Direct-address map: values stored in place, iterated in slot-index (for integers, key) order |
| `PagedVec<T>` | storage | Growable vector with stable addresses |
| `Hive<T>` | storage | Unordered container with stable addresses and generational handles |
| `IdAllocator` | none | Generational id allocator for use as keys |

## Python

Keys are integers in `0 <= key < 2**32`. `SparseSet` holds keys,
`SparseMap` maps keys to any objects, and `SparseMapF64`, `SparseMapI64` and
the other typed maps (or `sparse_map(dtype)`) store numbers unboxed:

```python
import numpy as np
from paged_collections import sparse_map

m = sparse_map("float64")
m.set_many(np.arange(0, 3_000_000, 1000), 1.0)  # batch insert
m[7] = 2.5
m.get_many([7, 1000])                            # array([2.5, 1. ])

values = m.values_view()   # zero-copy, writeable
values *= 2                # updates the map in place
# While a view is alive, resizing the map raises BufferError.
del values
```

Views share memory with the collection, like a `memoryview` of a
`bytearray`. NumPy releases the GIL inside many operations, so do not use a
view in one thread while another thread reads or writes the collection or
the view; synchronize such access yourself. A NumPy array stored inside the
`SparseMap` it views is never garbage-collected (NumPy arrays are invisible
to the collector).

The classes can be subclassed. As with `dict`, the built-in methods do not
call overridden methods (`set_many` and `update` do not go through your
`__setitem__`), and `copy()` returns the base class. `pickle`,
`copy.copy` and `copy.deepcopy` keep the subclass and its attributes, and
never call the subclass's `__init__`.

## Layout

```
crates/paged-collections      Rust library (no_std + alloc)
crates/paged-collections-py   PyO3 bindings, built with maturin
```

## Development

Rust:

```bash
cargo test
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo build -p paged-collections --no-default-features
```

Python (from `crates/paged-collections-py`):

```bash
uv sync
uv run pytest
```

`cargo build --workspace` fails to link the Python crate unless the Python
development library is installed; build it through maturin (`uv sync` or
`maturin develop`) instead. `cargo check` and `cargo clippy` work either way.

## License

MIT OR Apache-2.0 (to be confirmed before the first release).
