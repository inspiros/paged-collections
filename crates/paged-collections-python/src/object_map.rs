//! `SparseMap` and `IdMap`: `PagedSparseMap<u32 | Id, Py<PyAny>>`, for
//! arbitrary values.

use crate::storage::MapInner;
use numpy::PyArray1;
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::{PyTraverseError, PyVisit};

use crate::convert::{Key, KeyBatch, PyKey};
use crate::export::{Exports, read_only_view};
use crate::iter::key_iterator;
use crate::repr_items;
use crate::typed_map::optional_default;
use paged_collections::id_alloc::Id;

/// Defines `$name`, a `SparseMap`-like class over keys of type `$K`.
macro_rules! object_map {
    (
        $name:ident, $iter:ident, $pyname:literal,
        $K:ty, $keydoc:literal, $keydtype:literal
    ) => {
        #[doc = concat!(
            "A map from ", $keydoc, " to arbitrary Python objects,\n",
            "with O(1) insert, delete and lookup.\n\n",
            "Keys and values live in two parallel contiguous arrays (in insertion\n",
            "order until a removal moves the last entry into the gap); the key index\n",
            "is paged, so memory follows the key ranges in use. For numeric values,\n",
            "the typed maps store values unboxed and expose them as NumPy arrays.\n\n",
            "``keys_view()`` returns a zero-copy, read-only array of the keys; while\n",
            "any view is alive, methods that would resize or reorder the map raise\n",
            "``BufferError``. Do not store a view of a map in that map: NumPy arrays\n",
            "are invisible to the garbage collector, so the cycle is never freed."
        )]
        #[pyclass(module = "paged_collections._core", name = $pyname, subclass)]
        pub(crate) struct $name {
            map: MapInner<$K, Py<PyAny>>,
            exports: Exports,
            /// Changes on every structural change, so iterators can detect them.
            mods: u64,
        }

        impl $name {
            fn empty() -> Self {
                Self {
                    map: MapInner::default(),
                    exports: Exports::default(),
                    mods: 0,
                }
            }

            fn changed(&mut self) {
                self.mods = self.mods.wrapping_add(1);
            }

            pub(crate) fn dense_keys(&self) -> &[$K] {
                self.map.key_slice()
            }

            fn set_batch(
                slf: &Bound<'_, Self>,
                keys: &Bound<'_, PyAny>,
                values: &Bound<'_, PyAny>,
            ) -> PyResult<usize> {
                let keys = KeyBatch::<$K>::new(keys)?.to_insert_keys()?;
                let values: Vec<Py<PyAny>> = values
                    .try_iter()?
                    .map(|value| value.map(Bound::unbind))
                    .collect::<PyResult<_>>()?;
                if values.len() != keys.len() {
                    return Err(PyValueError::new_err(format!(
                        "got {} values for {} keys",
                        values.len(),
                        keys.len()
                    )));
                }
                let py = slf.py();
                // Keep input values alive too: duplicate keys may replace an
                // earlier input value while the batch is being written.
                let writes = values.iter().map(|v| v.clone_ref(py)).collect();
                let mut removed = Vec::with_capacity(keys.len());
                let new = {
                    let mut borrow = slf.try_borrow_mut()?;
                    let this = &mut *borrow;
                    if keys.iter().any(|&k| !this.map.contains_key(k)) {
                        this.exports.check($pyname)?;
                    }
                    let new = paged!(MapInner, &mut this.map, map => {
                        for &key in &keys {
                            // A different generation of this slot can be evicted.
                            let index = <$K as paged_collections::key::Key>::index(key);
                            if let Some((_, value)) = map.occupant_mut(index) {
                                let retained = value.clone_ref(py);
                                removed.push(std::mem::replace(value, retained));
                            }
                        }
                        // Only retained references can be decref'd by this helper;
                        // the original objects stay alive until after the borrow.
                        crate::typed_map::write_batch(map, &this.exports, $pyname, keys, writes)
                    })?;
                    if new > 0 {
                        this.changed();
                    }
                    new
                };
                drop(removed);
                drop(values);
                Ok(new)
            }

            // Keep the allocated pages, but return ownership of all values so
            // callers can release the map borrow before running finalizers.
            fn remove_all(&mut self) -> Vec<Py<PyAny>> {
                let keys = self.map.key_slice().to_vec();
                let removed = keys.into_iter().filter_map(|k| self.map.remove(k)).collect();
                self.map.clear();
                self.changed();
                removed
            }

            fn remove_present(&mut self, key: $K) -> PyResult<Py<PyAny>> {
                self.exports.check($pyname)?;
                let value = self.map.remove(key);
                self.changed();
                Ok(value.expect("caller checked the key is present"))
            }
        }

        #[pymethods]
        impl $name {
            // `__new__` accepts and ignores any arguments, so subclasses can define
            // their own `__init__`; `__init__` fills the collection, as for `dict`.
            #[new]
            #[pyo3(signature = (*_args, **_kwargs), text_signature = "(keys=None, values=None, *, page_size=DEFAULT_PAGE_SIZE)")]
            fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
                Self::empty()
            }

            #[pyo3(signature = (keys = None, values = None, *, page_size = crate::DEFAULT_PAGE_SIZE))]
            fn __init__(
                slf: &Bound<'_, Self>,
                keys: Option<&Bound<'_, PyAny>>,
                values: Option<&Bound<'_, PyAny>>,
                #[pyo3(from_py_with = crate::extract_page_size)] page_size: usize,
            ) -> PyResult<()> {
                {
                    let mut this = slf.try_borrow_mut()?;
                    if this.map.page_size() != page_size {
                        this.exports.check($pyname)?;
                        this.map.set_page_size(page_size)?;
                        this.changed();
                    }
                }
                match (keys, values) {
                    (None, None) => Ok(()),
                    (Some(keys), Some(values)) => Self::set_batch(slf, keys, values).map(drop),
                    _ => Err(PyTypeError::new_err(
                        "give both keys and values, or neither",
                    )),
                }
            }

            fn __len__(&self) -> usize {
                self.map.len()
            }

            fn __contains__(&self, key: &Bound<'_, PyAny>) -> bool {
                <$K as PyKey>::lookup(key).is_some_and(|k| self.map.contains_key(k))
            }

            fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                <$K as PyKey>::lookup(key)
                    .and_then(|k| self.map.get(k))
                    .map(|value| value.clone_ref(py))
                    .ok_or_else(|| PyKeyError::new_err(key.clone().unbind()))
            }

            fn __setitem__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>, value: Py<PyAny>) -> PyResult<()> {
                let key = <$K as PyKey>::insert(key)?;
                let old = {
                    let mut this = slf.try_borrow_mut()?;
                    if this.map.contains_key(key) {
                        this.map.insert(key, value)
                    } else {
                        this.exports.check($pyname)?;
                        let old = this.map.insert(key, value);
                        this.changed();
                        old
                    }
                };
                drop(old);
                Ok(())
            }

            fn __delitem__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<()> {
                let lookup = <$K as PyKey>::lookup(key);
                let removed = {
                    let mut this = slf.try_borrow_mut()?;
                    match lookup.filter(|&k| this.map.contains_key(k)) {
                        Some(k) => this.remove_present(k)?,
                        None => return Err(PyKeyError::new_err(key.clone().unbind())),
                    }
                };
                drop(removed);
                Ok(())
            }

            fn __iter__(slf: &Bound<'_, Self>) -> PyResult<$iter> {
                $iter::new(slf)
            }

            /// Equal when both hold the same keys with equal values (compared with
            /// ``is`` first, then ``==``, as ``dict`` does).
            fn __eq__(slf: &Bound<'_, Self>, other: &Bound<'_, Self>) -> PyResult<bool> {
                let py = slf.py();
                let pairs = {
                    let this = slf.try_borrow()?;
                    let other = other.try_borrow()?;
                    if this.map.len() != other.map.len() {
                        return Ok(false);
                    }
                    let mut pairs = Vec::with_capacity(this.map.len());
                    for (&key, value) in this.map.iter() {
                        let Some(other_value) = other.map.get(key) else {
                            return Ok(false);
                        };
                        pairs.push((value.clone_ref(py), other_value.clone_ref(py)));
                    }
                    pairs
                };
                for (value, other_value) in pairs {
                    if !value.is(&other_value) && !value.bind(py).eq(other_value.bind(py))? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }

            fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
                let py = slf.py();
                let name = slf.get_type().name()?.to_string();
                crate::repr_guarded(slf.as_any(), &name, || {
                    let (items, page_size) = {
                        let this = slf.try_borrow()?;
                        (this.items(py), this.map.page_size())
                    };
                    let items = repr_items(items.into_iter().map(|(k, v)| (k, v.into_bound(py))), py)?;
                    Ok(format!("{name}({{{items}}}{})", crate::page_size_repr(page_size)))
                })
            }

            /// Pickles as an empty map plus its items, so `pickle` and
            /// `copy.deepcopy` create the map before its values and handle maps that
            /// contain themselves. Items are restored with `__setitem__`, as for
            /// `dict`.
            fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
                let py = slf.py();
                let items = PyList::new(py, slf.try_borrow()?.items(py))?;
                crate::reduce(
                    slf.as_any(),
                    &py.get_type::<Self>(),
                    Vec::new(),
                    Some(items.try_iter()?),
                )
            }

            /// Return a shallow copy: the values are shared, not copied. The copy
            /// has no views.
            fn copy(&self, py: Python<'_>) -> Self {
                let mut map = self.map.same_size_empty();
                for (&key, value) in self.map.iter() {
                    map.insert(key, value.clone_ref(py));
                }
                Self {
                    map,
                    exports: Exports::default(),
                    mods: 0,
                }
            }

            /// Return the value for ``key``, or ``default`` if it is absent.
            #[pyo3(signature = (key, default = None))]
            fn get(
                slf: &Bound<'_, Self>,
                py: Python<'_>,
                key: &Bound<'_, PyAny>,
                default: Option<Py<PyAny>>,
            ) -> PyResult<Py<PyAny>> {
                let lookup = <$K as PyKey>::lookup(key);
                let this = slf.try_borrow()?;
                Ok(match lookup.and_then(|k| this.map.get(k)) {
                    Some(value) => value.clone_ref(py),
                    None => default.unwrap_or_else(|| py.None()),
                })
            }

            /// Remove ``key`` and return its value. If it is absent, return
            /// ``default`` if given, else raise ``KeyError``.
            #[pyo3(signature = (key, *default))]
            fn pop<'py>(
                slf: &Bound<'py, Self>,
                py: Python<'py>,
                key: &Bound<'py, PyAny>,
                default: &Bound<'py, PyTuple>,
            ) -> PyResult<Bound<'py, PyAny>> {
                let default = optional_default(default, "pop")?;
                let lookup = <$K as PyKey>::lookup(key);
                let mut this = slf.try_borrow_mut()?;
                match lookup.filter(|&k| this.map.contains_key(k)) {
                    Some(k) => Ok(this.remove_present(k)?.into_bound(py)),
                    None => default.ok_or_else(|| PyKeyError::new_err(key.clone().unbind())),
                }
            }

            /// Remove and return the last ``(key, value)`` pair in dense order;
            /// raise ``KeyError`` if the map is empty.
            fn popitem(slf: &Bound<'_, Self>) -> PyResult<(<$K as PyKey>::Bits, Py<PyAny>)> {
                let mut this = slf.try_borrow_mut()?;
                let &key = this.map.key_slice().last().ok_or_else(|| {
                    PyKeyError::new_err(concat!("popitem(): ", $pyname, " is empty"))
                })?;
                Ok((key.to_bits(), this.remove_present(key)?))
            }

            /// Return the value for ``key``, first inserting ``default`` if ``key``
            /// is absent.
            #[pyo3(signature = (key, default = None))]
            fn setdefault(
                slf: &Bound<'_, Self>,
                py: Python<'_>,
                key: &Bound<'_, PyAny>,
                default: Option<Py<PyAny>>,
            ) -> PyResult<Py<PyAny>> {
                let key = <$K as PyKey>::insert(key)?;
                let (value, old) = {
                    let mut this = slf.try_borrow_mut()?;
                    if let Some(value) = this.map.get(key) {
                        return Ok(value.clone_ref(py));
                    }
                    this.exports.check($pyname)?;
                    let default = default.unwrap_or_else(|| py.None());
                    let old = this.map.insert(key, default.clone_ref(py));
                    this.changed();
                    (default, old)
                };
                drop(old);
                Ok(value)
            }

            /// Insert the pairs of ``other``: a mapping, or an iterable of
            /// ``(key, value)`` pairs. Like ``dict.update``, a failure part way leaves
            /// the earlier pairs inserted; use ``set_many`` for an all-or-nothing
            /// batch.
            fn update(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<()> {
                if other.is(slf) {
                    return Ok(());
                }
                let pairs = if other.hasattr("keys")? {
                    other.call_method0("items")?
                } else {
                    other.clone()
                };
                for pair in pairs.try_iter()? {
                    let (key, value): (Bound<'_, PyAny>, Py<PyAny>) = pair?.extract()?;
                    Self::__setitem__(slf, &key, value)?;
                }
                Ok(())
            }

            /// Remove every entry. Pages stay allocated for reuse.
            fn clear(slf: &Bound<'_, Self>) -> PyResult<()> {
                let removed = {
                    let mut this = slf.try_borrow_mut()?;
                    if this.map.is_empty() {
                        return Ok(());
                    }
                    this.exports.check($pyname)?;
                    this.remove_all()
                };
                drop(removed);
                Ok(())
            }

            #[doc = concat!("Return the keys as a new ``", $keydtype, "`` array, in dense order.")]
            fn keys<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<<$K as PyKey>::Bits>> {
                PyArray1::from_slice(py, <$K as PyKey>::bits_slice(self.map.key_slice()))
            }

            /// Return the values as a new list, in dense order.
            fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
                PyList::new(py, self.map.values().map(|value| value.bind(py)))
            }

            /// Return a list of ``(key, value)`` pairs in dense order.
            fn items(&self, py: Python<'_>) -> Vec<(<$K as PyKey>::Bits, Py<PyAny>)> {
                self.map
                    .iter()
                    .map(|(&k, v)| (k.to_bits(), v.clone_ref(py)))
                    .collect()
            }

            #[doc = concat!(
                "Return a zero-copy, read-only ``", $keydtype, "`` array of the keys in dense ",
                "order. See the class docs for what views prevent."
            )]
            fn keys_view<'py>(
                slf: &Bound<'py, Self>,
            ) -> PyResult<Bound<'py, PyArray1<<$K as PyKey>::Bits>>> {
                let this = slf.try_borrow()?;
                let keys = <$K as PyKey>::bits_slice(this.map.key_slice());
                // SAFETY: the keys are the map's dense key `Vec` (seen as their
                // bits), and every method that could move them checks `exports`
                // first.
                unsafe { read_only_view(slf.as_any(), &this.exports, keys) }
            }

            /// Set the value of each key in ``keys`` to the matching item of
            /// ``values`` (an iterable of the same length) and return how many keys
            /// were new. For a key listed twice, the last value wins.
            ///
            /// The batch is validated first: a bad key or a length mismatch raises
            /// and leaves the map unchanged.
            fn set_many(
                slf: &Bound<'_, Self>,
                keys: &Bound<'_, PyAny>,
                values: &Bound<'_, PyAny>,
            ) -> PyResult<usize> {
                Self::set_batch(slf, keys, values)
            }

            /// Return a list of the values of ``keys``. A missing key takes
            /// ``default`` if given, else raises ``KeyError``.
            #[pyo3(signature = (keys, *default))]
            fn get_many<'py>(
                slf: &Bound<'py, Self>,
                py: Python<'py>,
                keys: &Bound<'py, PyAny>,
                default: &Bound<'py, PyTuple>,
            ) -> PyResult<Bound<'py, PyList>> {
                let default = optional_default(default, "get_many")?;
                let batch = KeyBatch::<$K>::lookup(keys)?;
                let mut out = Vec::with_capacity(batch.len());
                let mut missing = None;
                {
                    let this = slf.try_borrow()?;
                    batch.for_each(|k| match (k.and_then(|k| this.map.get(k)), &default) {
                        (Some(value), _) => out.push(value.bind(py).clone()),
                        (None, Some(value)) => out.push(value.clone()),
                        (None, None) => {
                            missing.get_or_insert(out.len());
                            out.push(py.None().into_bound(py));
                        }
                    });
                }
                match missing {
                    Some(pos) => Err(PyKeyError::new_err(batch.item(pos)?.unbind())),
                    None => PyList::new(py, out),
                }
            }

            /// Return a boolean array: whether each key in ``keys`` is present.
            fn contains_many<'py>(
                slf: &Bound<'py, Self>,
                py: Python<'py>,
                keys: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyArray1<bool>>> {
                let batch = KeyBatch::<$K>::lookup(keys)?;
                let mut out = Vec::with_capacity(batch.len());
                {
                    let this = slf.try_borrow()?;
                    batch.for_each(|k| out.push(k.is_some_and(|k| this.map.contains_key(k))));
                }
                Ok(PyArray1::from_vec(py, out))
            }

            /// Remove every key in ``keys`` that is present and return how many were
            /// removed. Absent and out-of-range keys are ignored.
            fn discard_many(slf: &Bound<'_, Self>, keys: &Bound<'_, PyAny>) -> PyResult<usize> {
                let batch = KeyBatch::<$K>::lookup(keys)?;
                let mut keys = Vec::with_capacity(batch.len());
                batch.for_each(|k| keys.extend(k));
                let removed: Vec<Py<PyAny>> = {
                    let mut this = slf.try_borrow_mut()?;
                    let present: Vec<_> = keys.into_iter().filter(|&k| this.map.contains_key(k)).collect();
                    if present.is_empty() {
                        return Ok(0);
                    }
                    this.exports.check($pyname)?;
                    let removed = present.into_iter().filter_map(|k| this.map.remove(k)).collect();
                    this.changed();
                    removed
                };
                // Finalizers run only after the batch is complete and the borrow is released.
                let count = removed.len();
                drop(removed);
                Ok(count)
            }

            /// Sort the entries by key, ascending (ids: by index, then generation).
            fn sort_keys(&mut self) -> PyResult<()> {
                self.exports.check($pyname)?;
                self.map.sort_unstable_keys();
                self.changed();
                Ok(())
            }

            /// Reserve room for at least ``additional`` more entries.
            fn reserve(&mut self, additional: usize) -> PyResult<()> {
                self.exports.check($pyname)?;
                self.map
                    .try_reserve(additional)
                    .map_err(crate::reserve_error)
            }

            /// Release unused memory; frees every page if the map is empty.
            fn shrink_to_fit(&mut self) -> PyResult<()> {
                self.exports.check($pyname)?;
                self.map.shrink_to_fit();
                Ok(())
            }

            /// Number of allocated pages of the key index.
            #[getter]
            fn allocated_pages(&self) -> usize {
                self.map.allocated_pages()
            }

            fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
                for value in self.map.values() {
                    visit.call(value)?;
                }
                Ok(())
            }

            fn __clear__(slf: &Bound<'_, Self>) -> PyResult<()> {
                let removed = {
                    let mut this = slf.try_borrow_mut()?;
                    // A live keys view must stay put, so replace only the values.
                    if this.exports.active() {
                        this.map.values_mut().map(|value| std::mem::replace(value, slf.py().None())).collect()
                    } else {
                        this.remove_all()
                    }
                };
                drop(removed);
                Ok(())
            }

            /// Slots per page for this instance.
            #[getter]
            fn page_size(&self) -> usize { self.map.page_size() }

            /// Values are arbitrary Python objects.
            #[classattr]
            fn dtype() -> &'static str {
                "object"
            }
        }

        key_iterator!($iter, $name, $pyname, $K);
    };
}

object_map!(
    SparseMap,
    SparseMapIterator,
    "SparseMap",
    Key,
    "integers in ``0 <= key < 2**32``",
    "uint32"
);
object_map!(
    IdMap,
    IdMapIterator,
    "IdMap",
    Id,
    "generational ids (see ``IdAllocator``)",
    "uint64"
);
