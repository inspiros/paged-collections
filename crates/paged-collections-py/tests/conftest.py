"""Class and array cases shared by the binding contract tests."""

import copy
import pickle

import numpy as np
import pytest

import paged_collections as pc


TYPED_MAP_CLASSES = (
    pc.SparseMapF64,
    pc.SparseMapF32,
    pc.SparseMapI64,
    pc.SparseMapI32,
    pc.SparseMapU64,
    pc.SparseMapU32,
    pc.SparseMapBool,
)
MAP_CLASSES = (pc.SparseMap, *TYPED_MAP_CLASSES)


@pytest.fixture(params=MAP_CLASSES, ids=lambda cls: cls.__name__)
def map_cls(request):
    return request.param


@pytest.fixture(params=TYPED_MAP_CLASSES, ids=lambda cls: cls.__name__)
def typed_map_cls(request):
    return request.param


@pytest.fixture(params=(pc.SparseSet, *MAP_CLASSES), ids=lambda cls: cls.__name__)
def container_cls(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param((pc.SparseSet, "view"), id="SparseSet-view"),
        *[
            pytest.param((cls, method), id=f"{cls.__name__}-{method}")
            for cls in MAP_CLASSES
            for method in (("keys_view",) if cls is pc.SparseMap else
                           ("keys_view", "values_view"))
        ],
    ]
)
def export_case(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param((code, order, layout), id=f"{order}{code}-{layout}")
        for code in ("i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8")
        for order in ("<", ">")
        for layout in ("contiguous", "strided", "reversed")
    ]
)
def key_batch(request):
    """The same keys in every integer width, endian order and memory layout."""
    code, order, layout = request.param
    dtype = np.dtype(order + code)
    if layout == "strided":
        return np.array([7, 99, 1, 99, 9, 99, 7, 99], dtype=dtype)[::2]
    if layout == "reversed":
        return np.array([7, 9, 1, 7], dtype=dtype)[::-1]
    return np.array([7, 1, 9, 7], dtype=dtype)


@pytest.fixture(params=["unaligned", "big_endian", "unaligned_big_endian"])
def unusual_u32_keys(request):
    dtype = ">u4" if "big_endian" in request.param else np.uint32
    if "unaligned" in request.param:
        keys = np.frombuffer(bytearray(4 * 4 + 1), dtype=dtype, offset=1, count=4)
        assert not keys.flags.aligned
    else:
        keys = np.empty(4, dtype=dtype)
    keys[:] = [7, 1, 9, 7]
    return keys


@pytest.fixture
def hostile_array():
    class HostileArray(np.ndarray):
        def astype(self, *args, **kwargs):
            return np.zeros(1, dtype=np.int64)

    # int64 forces key conversion instead of the native uint32 fast path.
    return np.array([7, 1, 9, 7], dtype=np.int64).view(HostileArray)


@pytest.fixture(
    params=[*range(2, pickle.HIGHEST_PROTOCOL + 1), "deepcopy"],
    ids=lambda method: f"pickle-{method}" if isinstance(method, int) else method,
)
def object_roundtrip(request):
    if request.param == "deepcopy":
        return copy.deepcopy
    return lambda obj: pickle.loads(pickle.dumps(obj, protocol=request.param))


def pytest_collection_modifyitems(config, items):
    # Most suites pin page boundaries at 1024 (the stock default).
    # test_page_size.py covers every compiled size on any build.
    if pc.DEFAULT_PAGE_SIZE == 1024:
        return
    skip = pytest.mark.skip(reason="assumes DEFAULT_PAGE_SIZE == 1024 (custom build)")
    for item in items:
        if item.path.name != "test_page_size.py":
            item.add_marker(skip)
