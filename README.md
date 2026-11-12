# paged-collections

Paged sparse sets, maps and stable-address containers for Rust, with Python
bindings built on PyO3.

> Status: in development. `PagedSparseSet` and `PagedSparseMap` are
> implemented in Rust; the other collections and the Python bindings are not
> yet. Nothing is published.

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
| `PagedVecMap<K, V>` | index | Direct-indexed map, values stored in place |
| `PagedVec<T>` | storage | Growable vector with stable addresses |
| `Hive<T>` | storage | Unordered container with stable addresses and generational handles |
| `IdAllocator` | none | Generational id allocator for use as keys |

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
