# Installation

## Rust

```sh
cargo add paged-collections
```

Minimum supported Rust version: 1.85 (edition 2024).

| Feature | Default | Effect |
|---|---|---|
| `std` | yes | `std`-only conveniences. Without it the crate is `no_std` and needs only `alloc`. |
| `serde` | no | `Serialize` / `Deserialize` for the collections. |
| `concurrent` | no | The [`concurrent`](concurrent.md) module. Needs pointer-sized and 64-bit atomics on a 32- or 64-bit target; works without `std` (locks then spin). |

```toml
[dependencies]
paged-collections = { version = "0.1", default-features = false, features = ["concurrent"] }
```

`Hive`, `IdAllocator` and the concurrent module exist only on 32- and 64-bit
targets, because slot indices are `u32` and must convert to `usize` without
loss.

## Python

```sh
pip install paged-collections
```

Requires Python 3.10 or newer and NumPy 1.24 or newer. Wheels:

| Interpreter | Wheel tag |
|---|---|
| CPython 3.10+ with the GIL | `abi3` |
| Free-threaded CPython 3.14t | version-specific |
| Free-threaded CPython 3.15t and newer | `abi3t` |

Free-threaded 3.13t is not supported. See [Python Threads](python-threads.md).

### Page Sizes

Python classes take a `page_size=` keyword. The values a build accepts are
fixed at compile time, because each page size is a separate monomorphized
Rust type:

```python
import paged_collections as pc

pc.PAGE_SIZES         # (256, 512, 1024, 2048, 4096)
pc.DEFAULT_PAGE_SIZE  # 1024
m = pc.SparseMap(page_size=4096)
```

`BitSet` uses the same compiled sizes, measured in 64-bit words, and
defaults to `DEFAULT_BIT_PAGE_SIZE` (512, or `DEFAULT_PAGE_SIZE` when a
custom build leaves 512 out). To build from source with other sizes, set the variables before building:

```sh
PAGED_COLLECTIONS_PAGE_SIZES=64,1024,16384 \
PAGED_COLLECTIONS_DEFAULT_PAGE_SIZE=1024 \
pip install --no-binary paged-collections paged-collections
```

Each extra size adds code for every class, so the extension grows with the
list.

## From Source

```sh
git clone https://github.com/inspiros/paged-collections
cd paged-collections
cargo test                # Rust
uv sync && uv run pytest  # Python (builds the extension with maturin)
```

`cargo build --workspace` fails to link the Python crate unless the Python
development library is installed; build it through maturin (`uv sync` or
`maturin develop`). `cargo check` and `cargo clippy` work either way.
