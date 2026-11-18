"""Dtype factories accept every form ``numpy.dtype`` does."""

import numpy as np
import pytest

import paged_collections as pc

FACTORIES = [
    (pc.sparse_map, "SparseMap"),
    (pc.direct_map, "DirectMap"),
    (pc.id_map, "IdMap"),
    (pc.paged_list, "PagedList"),
    (pc.concurrent_sparse_map, "ConcurrentSparseMap"),
    (pc.concurrent_direct_map, "ConcurrentDirectMap"),
]

SUFFIXES = {
    "float64": "F64",
    "float32": "F32",
    "int64": "I64",
    "int32": "I32",
    "uint64": "U64",
    "uint32": "U32",
    "bool": "Bool",
}


def forms(name):
    """The same dtype written as a string, scalar type, dtype and byte-order code."""
    dtype = np.dtype(name)
    return [name, dtype.type, dtype, dtype.str]


@pytest.mark.parametrize(("factory", "prefix"), FACTORIES)
@pytest.mark.parametrize("name", list(SUFFIXES))
def test_every_dtype_form_picks_the_same_class(factory, prefix, name):
    expected = getattr(pc, prefix + SUFFIXES[name])
    for form in forms(name):
        obj = factory(form)
        assert type(obj) is expected, form
        assert np.dtype(obj.dtype) == np.dtype(name)


@pytest.mark.parametrize(("factory", "prefix"), FACTORIES)
def test_python_types_follow_numpy(factory, prefix):
    assert type(factory(float)) is getattr(pc, prefix + "F64")
    assert type(factory(bool)) is getattr(pc, prefix + "Bool")
    # ``int`` is NumPy's default integer, whose width depends on the platform.
    default_int = SUFFIXES[np.dtype(int).name]
    assert type(factory(int)) is getattr(pc, prefix + default_int)


@pytest.mark.parametrize(("factory", "prefix"), FACTORIES)
def test_unsupported_dtype_lists_the_supported_ones(factory, prefix):
    with pytest.raises(TypeError, match="supported: .*float64"):
        factory(np.int8)


@pytest.mark.parametrize("factory", [pc.concurrent_sparse_map, pc.concurrent_direct_map])
def test_concurrent_maps_have_no_object_variant(factory):
    for dtype in (object, "O"):
        with pytest.raises(TypeError, match="concurrent"):
            factory(dtype)


@pytest.mark.parametrize(
    ("name", "cls"), [("int64", pc.PriorityQueue), ("float64", pc.PriorityQueueF64)]
)
def test_priority_queue_dtype_forms(name, cls):
    for form in forms(name):
        assert type(pc.priority_queue(form)) is cls, form
    assert type(pc.priority_queue()) is pc.PriorityQueue
    assert type(pc.priority_queue(float)) is pc.PriorityQueueF64


def test_priority_queue_passes_items_min_and_page_size():
    q = pc.priority_queue(np.float64, {1: 2.5, 2: 0.5}, True, page_size=pc.PAGE_SIZES[0])
    assert q.is_min
    assert q.page_size == pc.PAGE_SIZES[0]
    assert q.pop() == (2, 0.5)
    with pytest.raises(TypeError, match="priority queue"):
        pc.priority_queue(np.int32)
