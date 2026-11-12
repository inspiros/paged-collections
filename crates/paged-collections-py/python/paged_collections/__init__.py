"""Paged sparse sets and maps with integer keys, implemented in Rust.

Keys are integers in ``0 <= key < 2**32``. ``SparseSet`` holds keys only;
``SparseMap`` maps keys to arbitrary objects; ``SparseMapF64`` and the other
typed maps store numeric values unboxed and expose them as NumPy arrays.
``sparse_map(dtype)`` picks the map class for a dtype.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from paged_collections._core import (
    KEY_LIMIT,
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


def sparse_map(dtype: Any = None, keys: Any = None, values: Any = None) -> Any:
    """Create a map whose values have ``dtype``.

    ``dtype=None`` or ``object`` gives a ``SparseMap`` of arbitrary objects.
    Otherwise ``dtype`` is anything ``numpy.dtype`` accepts and must be one of
    float64, float32, int64, int32, uint64, uint32 or bool. ``keys`` and
    ``values`` are passed to the constructor.
    """
    if dtype is None or dtype is object:
        cls: Any = SparseMap
    else:
        resolved = np.dtype(dtype)
        if resolved == np.dtype(object):
            cls = SparseMap
        elif resolved in _TYPED_MAPS:
            cls = _TYPED_MAPS[resolved]
        else:
            supported = ", ".join(str(d) for d in _TYPED_MAPS)
            raise TypeError(f"no sparse map for dtype {resolved}; supported: object, {supported}")
    if keys is None and values is None:
        return cls()
    return cls(keys, values)


__all__ = [
    "KEY_LIMIT",
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
    "sparse_map",
]
