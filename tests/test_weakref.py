import weakref

import pytest

import paged_collections as pc


_DTYPES = ("F64", "F32", "I64", "I32", "U64", "U32", "Bool")
_CONTAINERS = (
    "SparseSet",
    "ConcurrentSparseSet",
    "Hive",
    "IdAllocator",
    "PagedDeque",
    "PagedList",
    "PriorityQueue",
    "PriorityQueueF64",
    "SparseMap",
    "DirectMap",
    "IdMap",
    *(f"{family}{dtype}" for family in (
        "PagedList", "SparseMap", "DirectMap", "IdMap",
        "ConcurrentSparseMap", "ConcurrentDirectMap",
    ) for dtype in _DTYPES),
)


@pytest.mark.parametrize("name", _CONTAINERS)
def test_container_weakref(name):
    obj = getattr(pc, name)()
    ref = weakref.ref(obj)
    assert ref() is obj
    del obj
    assert ref() is None
