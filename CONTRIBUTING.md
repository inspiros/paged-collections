# Contributing

## Rust

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --all-features
cargo build -p paged-collections --no-default-features      # no_std
cargo +nightly miri test -p paged-collections --all-features
RUSTFLAGS="--cfg loom" cargo test -p paged-collections --features concurrent --lib --release loom
```

`cargo build --workspace` fails to link the Python crate unless the Python
development library is installed; build it through maturin instead.
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

1. Bump `version` in the root `Cargo.toml` (the crate and the Python package share it) and commit.
2. Push a `vX.Y.Z` tag. This runs CI, builds and tests the wheels, and deploys the book.
3. Run the **Publish** workflow by hand with the tag as the ref. It checks the tag against the version, reruns CI and the wheel builds, then publishes the crate to crates.io and the wheels and sdist to PyPI, in that order.
