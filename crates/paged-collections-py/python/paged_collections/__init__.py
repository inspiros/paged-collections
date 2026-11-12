"""Paged sets and maps, and generational id allocation, implemented in Rust.

Keys are integers in ``0 <= key < 2**32``. ``SparseSet`` holds keys only;
``SparseMap`` maps keys to arbitrary objects; ``SparseMapF64`` and the other
typed maps store numeric values unboxed and expose them as NumPy arrays.
``DirectMap`` and its typed variants keep entries in ascending key order and
provide nearest-entry and range queries. ``sparse_map(dtype)`` and
``direct_map(dtype)`` pick the map class for a dtype.

``IdAllocator`` issues integer ids packed as ``generation << 32 | index``.
Freed indices are reused last freed first with a new generation. ``id_index``,
``id_generation``, ``make_id`` and vectorized ``id_indices`` convert their parts;
``make_id`` accepts generations through 2**32-1, while the allocator issues at
most ``MAX_GENERATION`` (2**31). Ids are distinct from the maps' uint32 keys.
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
    IdAllocator,
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


def _map_class(dtype: Any, object_class: Any, typed_maps: dict, kind: str) -> Any:
    """Resolve the value dtype shared by the two map families."""
    if dtype is None or dtype is object:
        return object_class
    resolved = np.dtype(dtype)
    if resolved == np.dtype(object):
        return object_class
    if resolved in typed_maps:
        return typed_maps[resolved]
    supported = ", ".join(str(d) for d in typed_maps)
    raise TypeError(f"no {kind} map for dtype {resolved}; supported: object, {supported}")


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
    "IdAllocator",
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
    "make_id",
    "sparse_map",
    "direct_map",
]
