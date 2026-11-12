"""Paged lists, hives, sets and maps, and generational id allocation, implemented in Rust.

``SparseSet``, ``SparseMap`` and ``DirectMap`` use integer keys in
``0 <= key < 2**32``. ``SparseSet`` holds keys only;
``SparseMap`` maps keys to arbitrary objects; ``SparseMapF64`` and the other
typed maps store numeric values unboxed and expose them as NumPy arrays.
``DirectMap`` and its typed variants keep entries in ascending key order and
provide nearest-entry and range queries. ``IdMap`` and its typed variants
map generational ids to values. ``sparse_map(dtype)``, ``direct_map(dtype)``
and ``id_map(dtype)`` pick the map class for a dtype.

``PagedList`` stores arbitrary objects in pages; typed variants store numbers
unboxed and expose individual pages as NumPy views. ``paged_list(dtype)``
picks the list class for a dtype.

``Hive`` stores objects in reusable slots and issues generational ids.
Removing a value makes its id stale; insertions reuse slots with new generations.
Iteration visits ids in slot order, and ``IdMap`` can hold related values.

``IdAllocator`` issues integer ids packed as ``generation << 32 | index``.
Freed indices are reused last freed first with a new generation. ``id_index``,
``id_generation``, ``make_id`` and vectorized ``id_indices`` convert their parts;
``make_id`` accepts generations through 2**32-1, while the allocator issues at
most ``MAX_GENERATION`` (2**31). Use ``IdMap`` to key a map by these ids.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from paged_collections._core import (
    KEY_LIMIT,
    MAX_GENERATION,
    MAX_INDICES,
    DirectMap,
    DirectMapBool,
    DirectMapF32,
    DirectMapF64,
    DirectMapI32,
    DirectMapI64,
    DirectMapU32,
    DirectMapU64,
    Hive,
    IdAllocator,
    IdMap,
    IdMapBool,
    IdMapF32,
    IdMapF64,
    IdMapI32,
    IdMapI64,
    IdMapU32,
    IdMapU64,
    PagedList,
    PagedListBool,
    PagedListF32,
    PagedListF64,
    PagedListI32,
    PagedListI64,
    PagedListU32,
    PagedListU64,
    SparseMap,
    SparseMapBool,
    SparseMapF32,
    SparseMapF64,
    SparseMapI32,
    SparseMapI64,
    SparseMapU32,
    SparseMapU64,
    SparseSet,
    __version__,
    id_generation,
    id_index,
    id_indices,
    make_id,
)

_TYPED_MAPS = {
    np.dtype(cls.dtype): cls
    for cls in (
        SparseMapF64,
        SparseMapF32,
        SparseMapI64,
        SparseMapI32,
        SparseMapU64,
        SparseMapU32,
        SparseMapBool,
    )
}

_TYPED_DIRECT_MAPS = {
    np.dtype(cls.dtype): cls
    for cls in (
        DirectMapF64,
        DirectMapF32,
        DirectMapI64,
        DirectMapI32,
        DirectMapU64,
        DirectMapU32,
        DirectMapBool,
    )
}

_TYPED_ID_MAPS = {
    np.dtype(cls.dtype): cls
    for cls in (
        IdMapF64,
        IdMapF32,
        IdMapI64,
        IdMapI32,
        IdMapU64,
        IdMapU32,
        IdMapBool,
    )
}


_TYPED_LISTS = {
    np.dtype(cls.dtype): cls
    for cls in (
        PagedListF64,
        PagedListF32,
        PagedListI64,
        PagedListI32,
        PagedListU64,
        PagedListU32,
        PagedListBool,
    )
}


def _map_class(dtype: Any, object_class: Any, typed_maps: dict, kind: str) -> Any:
    """Resolve the value dtype shared by the map families."""
    if dtype is None or dtype is object:
        return object_class
    resolved = np.dtype(dtype)
    if resolved == np.dtype(object):
        return object_class
    if resolved in typed_maps:
        return typed_maps[resolved]
    supported = ", ".join(str(d) for d in typed_maps)
    raise TypeError(f"no {kind} map for dtype {resolved}; supported: object, {supported}")


def paged_list(dtype: Any = None, values: Any = None) -> Any:
    """Create a paged list of objects or values of the given NumPy dtype.

    Supported typed dtypes are float64, float32, int64, int32, uint64,
    uint32 and bool. Slices are copies; typed page views share storage.
    """
    if dtype is None or dtype is object:
        cls = PagedList
    else:
        resolved = np.dtype(dtype)
        if resolved == np.dtype(object):
            cls = PagedList
        elif resolved in _TYPED_LISTS:
            cls = _TYPED_LISTS[resolved]
        else:
            supported = ", ".join(str(d) for d in _TYPED_LISTS)
            raise TypeError(f"no paged list for dtype {resolved}; supported: object, {supported}")
    return cls() if values is None else cls(values)


def sparse_map(dtype: Any = None, keys: Any = None, values: Any = None) -> Any:
    """Create a map whose values have ``dtype``.

    ``dtype=None`` or ``object`` gives a ``SparseMap`` of arbitrary objects.
    Otherwise ``dtype`` is anything ``numpy.dtype`` accepts and must be one of
    float64, float32, int64, int32, uint64, uint32 or bool. ``keys`` and
    ``values`` are passed to the constructor.
    """
    cls = _map_class(dtype, SparseMap, _TYPED_MAPS, "sparse")
    if keys is None and values is None:
        return cls()
    return cls(keys, values)


def id_map(dtype: Any = None, keys: Any = None, values: Any = None) -> Any:
    """Create a generational-id map whose values have ``dtype``.

    ``dtype=None`` or ``object`` gives an ``IdMap`` of arbitrary objects.
    Otherwise ``dtype`` is anything ``numpy.dtype`` accepts and must be one of
    float64, float32, int64, int32, uint64, uint32 or bool. ``keys`` and
    ``values`` are passed to the constructor.
    """
    cls = _map_class(dtype, IdMap, _TYPED_ID_MAPS, "id")
    if keys is None and values is None:
        return cls()
    return cls(keys, values)


def direct_map(dtype: Any = None, keys: Any = None, values: Any = None) -> Any:
    """Create an ordered direct map whose values have ``dtype``.

    ``dtype=None`` or ``object`` gives a ``DirectMap`` of arbitrary objects.
    Otherwise ``dtype`` is anything ``numpy.dtype`` accepts and must be one of
    float64, float32, int64, int32, uint64, uint32 or bool. ``keys`` and
    ``values`` are passed to the constructor. Entries are always in ascending
    key order; array results are copies.
    """
    cls = _map_class(dtype, DirectMap, _TYPED_DIRECT_MAPS, "direct")
    if keys is None and values is None:
        return cls()
    return cls(keys, values)


__all__ = [
    "KEY_LIMIT",
    "MAX_GENERATION",
    "MAX_INDICES",
    "DirectMap",
    "DirectMapBool",
    "DirectMapF32",
    "DirectMapF64",
    "DirectMapI32",
    "DirectMapI64",
    "DirectMapU32",
    "DirectMapU64",
    "Hive",
    "IdAllocator",
    "IdMap",
    "IdMapBool",
    "IdMapF32",
    "IdMapF64",
    "IdMapI32",
    "IdMapI64",
    "IdMapU32",
    "IdMapU64",
    "PagedList",
    "PagedListBool",
    "PagedListF32",
    "PagedListF64",
    "PagedListI32",
    "PagedListI64",
    "PagedListU32",
    "PagedListU64",
    "SparseMap",
    "SparseMapBool",
    "SparseMapF32",
    "SparseMapF64",
    "SparseMapI32",
    "SparseMapI64",
    "SparseMapU32",
    "SparseMapU64",
    "SparseSet",
    "__version__",
    "id_generation",
    "id_index",
    "id_indices",
    "id_map",
    "make_id",
    "paged_list",
    "sparse_map",
    "direct_map",
]
