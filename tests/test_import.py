import paged_collections
import pytest


def test_version_is_exposed() -> None:
    assert isinstance(paged_collections.__version__, str)
    assert paged_collections.__version__.count(".") == 2


@pytest.mark.parametrize("name", [
    "SparseSet", "SparseMap", "SparseMapF64", "DirectMap", "DirectMapF64",
    "IdMap", "IdMapF64", "PagedDeque", "PriorityQueue", "PriorityQueueF64",
])
def test_native_repr_and_reduce_hooks(name):
    cls = getattr(paged_collections, name)
    obj = cls()
    assert "__repr__" in cls.__dict__ and "__reduce__" in cls.__dict__
    assert repr(obj).startswith(f"{name}(")
    assert isinstance(obj.__reduce__(), tuple)
