//! Key iterators that detect changes to their collection, as `dict` does.

/// Defines `$iter`, an iterator over the keys of `$owner` in dense order.
///
/// `$owner` must have a `mods: u64` field that changes on every structural
/// change, and a `dense_keys(&self) -> &[Key]` method. The iterator raises
/// `RuntimeError` on the next step after such a change.
macro_rules! key_iterator {
    ($iter:ident, $owner:ident, $owner_name:literal) => {
        #[doc = concat!("Iterator over the keys of a ``", $owner_name, "``.")]
        #[pyclass(module = "paged_collections._core")]
        pub(crate) struct $iter {
            owner: Py<$owner>,
            pos: usize,
            mods: u64,
            /// Set once exhausted: later changes to the owner no longer matter.
            done: bool,
        }

        impl $iter {
            pub(crate) fn new(owner: &Bound<'_, $owner>) -> PyResult<Self> {
                let mods = owner.try_borrow()?.mods;
                Ok(Self {
                    owner: owner.clone().unbind(),
                    pos: 0,
                    mods,
                    done: false,
                })
            }
        }

        #[pymethods]
        impl $iter {
            fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
                slf
            }

            fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<$crate::convert::Key>> {
                if self.done {
                    return Ok(None);
                }
                let owner = self.owner.try_borrow(py)?;
                if owner.mods != self.mods {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(concat!(
                        $owner_name,
                        " changed during iteration"
                    )));
                }
                let key = owner.dense_keys().get(self.pos).copied();
                self.pos += 1;
                self.done = key.is_none();
                Ok(key)
            }

            fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
                visit.call(&self.owner)
            }

            fn __length_hint__(&self, py: Python<'_>) -> usize {
                if self.done {
                    return 0;
                }
                self.owner
                    .try_borrow(py)
                    .map_or(0, |owner| owner.dense_keys().len().saturating_sub(self.pos))
            }
        }
    };
}

pub(crate) use key_iterator;
