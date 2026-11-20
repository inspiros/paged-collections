# Testing and Soundness

## Where `unsafe` Is Used

| Module | Why |
|---|---|
| `paged_deque` | uninitialized slots in pages; elements are moved in and out manually |
| `concurrent` | atomics publish slots and pages to other threads; guards hand out references to shared slots |
| Python bindings | NumPy views over container memory (guarded by export counts) |

The other single-threaded containers (sparse set and map, direct map,
priority queue, hive, id allocator, page directories) use no `unsafe` code.

## Test Layers

| Layer | What it checks | Where |
|---|---|---|
| Unit and doctests | API contracts, documented examples (including the README) | `crates/paged-collections/src`, `cargo test` |
| Model tests (proptest) | random operation sequences against a standard-library oracle (`Vec`, `VecDeque`, `HashMap`, `BTreeMap`, `BinaryHeap`) | `crates/paged-collections/tests/*_model.rs` |
| Panic safety | invariants after panicking `Drop`, `Ord` and user closures | unit tests per container |
| Miri | undefined behaviour in the Rust tests, including concurrent tests | Before release, outside CI |
| loom | exhaustive interleavings (preemption bound 2) of the concurrent protocols: publication, free-stack sharding, page installation, guards | `src/concurrent/loom_*.rs` |
| ThreadSanitizer | data races in the concurrent stress tests | manual |
| Compile-fail doctests | lifetime invariance of every concurrent element type | `concurrent` module docs |
| Python tests | CPython protocol parity (`list`, `deque`, `set`, `dict`), batch methods, views, pickling, free threading | `tests/` |

## Running Them

```sh
cargo test --all-features
cargo +nightly miri nextest run -p paged-collections --all-features -E 'not binary(rayon)'
MIRIFLAGS="-Zmiri-tree-borrows -Zmiri-ignore-leaks" cargo +nightly miri nextest run -p paged-collections --all-features --test rayon
MIRIFLAGS="-Zmiri-tree-borrows -Zmiri-ignore-leaks" cargo +nightly miri test -p paged-collections --all-features --doc
RUSTFLAGS="--cfg loom" cargo test -p paged-collections --features concurrent --lib --release loom
cargo build -p paged-collections --no-default-features          # no_std
uv sync && uv run pytest
```

Miri over the full suite is slow. `cargo miri nextest run` runs tests in
parallel. The rayon tests and doctests run under Tree Borrows because
rayon's `crossbeam-epoch` dependency fails the default Stacked Borrows check. The loom suite takes several minutes in release mode.
