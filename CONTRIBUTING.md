# Contributing

## Rust

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --all-features
cargo build -p paged-collections --no-default-features      # no_std
PROPTEST_CASES=8 cargo +nightly miri nextest run -p paged-collections --all-features -j <cores> -E 'not binary(rayon)'
MIRIFLAGS="-Zmiri-tree-borrows -Zmiri-ignore-leaks" cargo +nightly miri nextest run -p paged-collections --all-features --test rayon
MIRIFLAGS="-Zmiri-tree-borrows -Zmiri-ignore-leaks" cargo +nightly miri test -p paged-collections --all-features --doc
RUSTFLAGS="--cfg loom" cargo test -p paged-collections --features concurrent --lib --release loom
```

CI does not run Miri or loom. The full Miri suite takes hours even with
`nextest` on many cores (`cargo miri test` runs tests serially). Run both
before a release. The rayon tests and doctests (which include rayon
examples) run under Tree Borrows: rayon's `crossbeam-epoch` dependency fails
the default Stacked Borrows check in its own code. Tests too slow for Miri
are marked `#[cfg_attr(miri, ignore)]`. CI checks the MSRV (Rust 1.85).

`cargo build --workspace` fails to link the Python crate unless the Python
development library is installed. Build it through maturin instead.
`cargo check` and `cargo clippy` work either way.

## Python

```sh
uv sync
uv run pytest
```

## Documentation

```sh
mdbook build docs/site
python docs/site/verify_examples.py            # run examples, compare captured output
python docs/site/verify_examples.py --update   # rewrite the captures
```

The Python examples need the package installed (`uv sync`).

## Benchmarks

See [crates/bench/README.md](crates/bench/README.md) and
[benchmarks/python](benchmarks/python).

## Releases

1. Run Miri and loom (see [Rust](#rust)) and wait for both to pass.
2. Bump `version` in the root `Cargo.toml` (the crate and the Python package share it) and commit.
3. Push a `vX.Y.Z` tag. This runs CI, builds and tests the wheels, and deploys the book.
4. Run the **Publish** workflow by hand with the tag as the ref. It checks the tag against the version, reruns CI and the wheel builds, checks the wheel and sdist names, then publishes the crate to crates.io and the wheels and sdist to PyPI, in that order. If the PyPI step fails after the crate is published, use **Re-run failed jobs** on that run; a new run would stop at crates.io.
