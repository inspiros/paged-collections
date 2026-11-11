//! Python extension module `paged_collections._core`.
//!
//! Classes are concrete (monomorphized) wrappers over the Rust collections,
//! because Rust generics cannot be exposed to Python directly.

use pyo3::prelude::*;

/// Native core of the `paged_collections` Python package.
#[pymodule]
mod _core {
    use super::*;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add("__version__", env!("CARGO_PKG_VERSION"))
    }
}
