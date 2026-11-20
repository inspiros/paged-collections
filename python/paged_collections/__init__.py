"""Paged lists, queues, hives, sets and maps, and generational id allocation in Rust.

``BitSet``, ``SparseSet``, ``SparseMap`` and ``DirectMap`` use integer keys in
``0 <= key < 2**32``. ``BitSet`` stores one bit per slot in each allocated
page and iterates in ascending order. Its page size counts 64-bit words.
``SparseSet`` holds keys only;
``SparseMap`` maps keys to arbitrary objects; ``SparseMapF64`` and the other
typed maps store numeric values unboxed and expose them as NumPy arrays.
``DirectMap`` and its typed variants keep entries in ascending key order and
provide nearest-entry and range queries. ``IdMap`` and its typed variants
map generational ids to values. ``sparse_map(dtype)``, ``direct_map(dtype)``,
``id_map(dtype)``, ``concurrent_sparse_map(dtype)`` and
``concurrent_direct_map(dtype)`` pick the map class for a dtype.

``PagedList`` stores arbitrary objects in pages; typed variants store numbers
unboxed and expose individual pages as NumPy views. ``paged_list(dtype)``
picks the list class for a dtype. ``PagedDeque`` is a paged
``collections.deque`` of objects with an optional ``maxlen``.
``ConcurrentQueue`` is a shared FIFO of arbitrary objects.

``PriorityQueue`` (int64 priorities) and ``PriorityQueueF64`` (float64) map
integer keys to priorities; ``pop`` returns the greatest, or the least with
``min=True``. Pushing a present key changes its priority.
``priority_queue(dtype)`` picks the queue class for a dtype.
``ConcurrentPriorityQueue`` (int64) and ``ConcurrentPriorityQueueF64``
(float64) add shared updates, relaxed ``pop`` and global ``pop_exact``.
``concurrent_priority_queue(dtype)`` picks the concurrent class for a dtype.

Every ``dtype`` argument takes anything ``numpy.dtype`` accepts: a name such
as ``"float64"``, a scalar type such as ``np.float64``, a ``np.dtype``, or a
Python type. ``int`` means NumPy's default integer, which is int32 on Windows
with NumPy 1.x and int64 elsewhere; name the width to be portable.

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

from typing import Any, Literal, SupportsIndex, overload

import numpy as np
import numpy.typing as npt

from paged_collections._core import (
    DEFAULT_BIT_PAGE_SIZE,
    DEFAULT_PAGE_SIZE,
    PAGE_SIZES,
    KEY_LIMIT,
    MAX_GENERATION,
    MAX_INDICES,
    BitSet,
    ConcurrentDirectMapBool,
    ConcurrentDirectMapF32,
    ConcurrentDirectMapF64,
    ConcurrentDirectMapI32,
    ConcurrentDirectMapI64,
    ConcurrentDirectMapU32,
    ConcurrentDirectMapU64,
    ConcurrentPriorityQueue,
    ConcurrentPriorityQueueF64,
    ConcurrentQueue,
    ConcurrentSparseMapBool,
    ConcurrentSparseMapF32,
    ConcurrentSparseMapF64,
    ConcurrentSparseMapI32,
    ConcurrentSparseMapI64,
    ConcurrentSparseMapU32,
    ConcurrentSparseMapU64,
    ConcurrentSparseSet,
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
    PagedDeque,
    PagedList,
    PagedListBool,
    PagedListF32,
    PagedListF64,
    PagedListI32,
    PagedListI64,
    PagedListU32,
    PagedListU64,
    PriorityQueue,
    PriorityQueueF64,
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


_TYPED_CONCURRENT_SPARSE_MAPS = {
    np.dtype(cls.dtype): cls
    for cls in (
        ConcurrentSparseMapF64,
        ConcurrentSparseMapF32,
        ConcurrentSparseMapI64,
        ConcurrentSparseMapI32,
        ConcurrentSparseMapU64,
        ConcurrentSparseMapU32,
        ConcurrentSparseMapBool,
    )
}

_TYPED_CONCURRENT_DIRECT_MAPS = {
    np.dtype(cls.dtype): cls
    for cls in (
        ConcurrentDirectMapF64,
        ConcurrentDirectMapF32,
        ConcurrentDirectMapI64,
        ConcurrentDirectMapI32,
        ConcurrentDirectMapU64,
        ConcurrentDirectMapU32,
        ConcurrentDirectMapBool,
    )
}

_TYPED_PRIORITY_QUEUES = {np.dtype(cls.dtype): cls for cls in (PriorityQueue, PriorityQueueF64)}
_TYPED_CONCURRENT_PRIORITY_QUEUES = {
    np.dtype(cls.dtype): cls for cls in (ConcurrentPriorityQueue, ConcurrentPriorityQueueF64)
}


def _typed_class(dtype: npt.DTypeLike, typed: dict, kind: str) -> Any:
    """Resolve a dtype for a family without an object-valued class."""
    resolved = np.dtype(dtype)
    if resolved in typed:
        return typed[resolved]
    supported = ", ".join(str(d) for d in typed)
    raise TypeError(f"no {kind} for dtype {resolved}; supported: {supported}")


def _map_class(dtype: npt.DTypeLike, object_class: Any, typed_maps: dict, kind: str) -> Any:
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


def paged_list(dtype: npt.DTypeLike = None, values: Any = None, *, page_size: int = DEFAULT_PAGE_SIZE) -> Any:
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
    return cls(page_size=page_size) if values is None else cls(values, page_size=page_size)


def sparse_map(dtype: npt.DTypeLike = None, keys: Any = None, values: Any = None, *, page_size: int = DEFAULT_PAGE_SIZE) -> Any:
    """Create a map whose values have ``dtype``.

    ``dtype=None`` or ``object`` gives a ``SparseMap`` of arbitrary objects.
    Otherwise ``dtype`` is anything ``numpy.dtype`` accepts and must be one of
    float64, float32, int64, int32, uint64, uint32 or bool. ``keys`` and
    ``values`` are passed to the constructor.
    """
    cls = _map_class(dtype, SparseMap, _TYPED_MAPS, "sparse")
    if keys is None and values is None:
        return cls(page_size=page_size)
    return cls(keys, values, page_size=page_size)


def id_map(dtype: npt.DTypeLike = None, keys: Any = None, values: Any = None, *, page_size: int = DEFAULT_PAGE_SIZE) -> Any:
    """Create a generational-id map whose values have ``dtype``.

    ``dtype=None`` or ``object`` gives an ``IdMap`` of arbitrary objects.
    Otherwise ``dtype`` is anything ``numpy.dtype`` accepts and must be one of
    float64, float32, int64, int32, uint64, uint32 or bool. ``keys`` and
    ``values`` are passed to the constructor.
    """
    cls = _map_class(dtype, IdMap, _TYPED_ID_MAPS, "id")
    if keys is None and values is None:
        return cls(page_size=page_size)
    return cls(keys, values, page_size=page_size)


def direct_map(dtype: npt.DTypeLike = None, keys: Any = None, values: Any = None, *, page_size: int = DEFAULT_PAGE_SIZE) -> Any:
    """Create an ordered direct map whose values have ``dtype``.

    ``dtype=None`` or ``object`` gives a ``DirectMap`` of arbitrary objects.
    Otherwise ``dtype`` is anything ``numpy.dtype`` accepts and must be one of
    float64, float32, int64, int32, uint64, uint32 or bool. ``keys`` and
    ``values`` are passed to the constructor. Entries are always in ascending
    key order; array results are copies.
    """
    cls = _map_class(dtype, DirectMap, _TYPED_DIRECT_MAPS, "direct")
    if keys is None and values is None:
        return cls(page_size=page_size)
    return cls(keys, values, page_size=page_size)



def concurrent_sparse_map(dtype: npt.DTypeLike) -> Any:
    """Create an empty concurrent sparse map whose values have ``dtype``.

    ``dtype`` must be one of float64, float32, int64, int32, uint64, uint32
    or bool; there is no object-valued concurrent map.
    """
    return _typed_class(dtype, _TYPED_CONCURRENT_SPARSE_MAPS, "concurrent sparse map")()


def concurrent_direct_map(dtype: npt.DTypeLike) -> Any:
    """Create an empty concurrent direct map whose values have ``dtype``.

    ``dtype`` must be one of float64, float32, int64, int32, uint64, uint32
    or bool; there is no object-valued concurrent map.
    """
    return _typed_class(dtype, _TYPED_CONCURRENT_DIRECT_MAPS, "concurrent direct map")()


def priority_queue(
    dtype: npt.DTypeLike = "int64",
    items: Any = None,
    min: bool = False,
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> Any:
    """Create a priority queue whose priorities have ``dtype``.

    ``dtype`` is int64 (``PriorityQueue``) or float64 (``PriorityQueueF64``).
    ``items`` and ``min`` are passed to the constructor.
    """
    cls = _typed_class(dtype, _TYPED_PRIORITY_QUEUES, "priority queue")
    return cls(items, min, page_size=page_size)


@overload
def concurrent_priority_queue(
    dtype: Literal["int64"] | None = None,
    items: Any = None,
    min: bool = False,
    *,
    shards: SupportsIndex | None = None,
) -> ConcurrentPriorityQueue: ...


@overload
def concurrent_priority_queue(
    dtype: Literal["float64"],
    items: Any = None,
    min: bool = False,
    *,
    shards: SupportsIndex | None = None,
) -> ConcurrentPriorityQueueF64: ...


@overload
def concurrent_priority_queue(
    dtype: npt.DTypeLike,
    items: Any = None,
    min: bool = False,
    *,
    shards: SupportsIndex | None = None,
) -> ConcurrentPriorityQueue | ConcurrentPriorityQueueF64: ...


def concurrent_priority_queue(
    dtype: npt.DTypeLike = None,
    items: Any = None,
    min: bool = False,
    *,
    shards: SupportsIndex | None = None,
) -> ConcurrentPriorityQueue | ConcurrentPriorityQueueF64:
    """Create a shared priority queue whose priorities have ``dtype``.

    ``None`` or int64 gives a ``ConcurrentPriorityQueue``; float64 gives a
    ``ConcurrentPriorityQueueF64``. The constructor takes ``items``, ``min``
    and the optional shard count.
    """
    if dtype is None:
        dtype = "int64"
    cls = _typed_class(dtype, _TYPED_CONCURRENT_PRIORITY_QUEUES, "concurrent priority queue")
    return cls(items, min, shards=shards)


__all__ = [
    "DEFAULT_BIT_PAGE_SIZE",
    "DEFAULT_PAGE_SIZE",
    "PAGE_SIZES",
    "KEY_LIMIT",
    "MAX_GENERATION",
    "MAX_INDICES",
    "BitSet",
    "ConcurrentDirectMapBool",
    "ConcurrentDirectMapF32",
    "ConcurrentDirectMapF64",
    "ConcurrentDirectMapI32",
    "ConcurrentDirectMapI64",
    "ConcurrentDirectMapU32",
    "ConcurrentDirectMapU64",
    "ConcurrentPriorityQueue",
    "ConcurrentPriorityQueueF64",
    "ConcurrentQueue",
    "ConcurrentSparseMapBool",
    "ConcurrentSparseMapF32",
    "ConcurrentSparseMapF64",
    "ConcurrentSparseMapI32",
    "ConcurrentSparseMapI64",
    "ConcurrentSparseMapU32",
    "ConcurrentSparseMapU64",
    "ConcurrentSparseSet",
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
    "PagedDeque",
    "PagedList",
    "PagedListBool",
    "PagedListF32",
    "PagedListF64",
    "PagedListI32",
    "PagedListI64",
    "PagedListU32",
    "PagedListU64",
    "PriorityQueue",
    "PriorityQueueF64",
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
    "concurrent_direct_map",
    "concurrent_priority_queue",
    "concurrent_sparse_map",
    "id_generation",
    "id_index",
    "id_indices",
    "id_map",
    "make_id",
    "paged_list",
    "priority_queue",
    "sparse_map",
    "direct_map",
]
