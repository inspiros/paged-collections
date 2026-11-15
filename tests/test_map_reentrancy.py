"""Python callbacks must run without borrowing their owning map."""

import sys

import numpy as np
import pytest

import paged_collections as pc


OBJECT_MAPS = (pc.SparseMap, pc.IdMap, pc.DirectMap)
DIRECT_MAPS = (
    pc.DirectMap, pc.DirectMapF64, pc.DirectMapF32, pc.DirectMapI64,
    pc.DirectMapI32, pc.DirectMapU64, pc.DirectMapU32, pc.DirectMapBool,
)


@pytest.fixture(params=OBJECT_MAPS, ids=lambda cls: cls.__name__)
def object_map_cls(request):
    return request.param


@pytest.fixture
def unraisable(monkeypatch):
    errors = []
    monkeypatch.setattr(sys, "unraisablehook", errors.append)
    yield errors
    assert errors == []


def _key(cls, index, generation=1):
    return generation << 32 | index if cls is pc.IdMap else index


@pytest.mark.parametrize("operation", [
    "delete", "overwrite", "clear", "discard_many", "set_many", "update",
    "init", "pop", "popitem",
])
def test_removed_value_can_write_map(object_map_cls, operation, unraisable):
    m = object_map_cls()
    key, marker = (_key(object_map_cls, i) for i in (1, 9))
    events = []

    class Finalizer:
        def __del__(self):
            events.append(m.get(key))
            m[marker] = "finalized"

    m[key] = Finalizer()
    if operation == "delete":
        del m[key]
    elif operation == "overwrite":
        m[key] = "replacement"
    elif operation == "clear":
        m.clear()
    elif operation == "discard_many":
        assert m.discard_many([key, key]) == 1
    elif operation == "set_many":
        assert m.set_many([key], ["replacement"]) == 0
    elif operation == "update":
        m.update([(key, "replacement")])
    elif operation == "init":
        m.__init__([key], ["replacement"])
    elif operation == "pop":
        m.pop(key)
    else:
        m.popitem()
    expected = "replacement" if operation in {"overwrite", "set_many", "update", "init"} else None
    assert events == [expected]
    assert m[marker] == "finalized"
    assert unraisable == []


@pytest.mark.parametrize("operation", ["clear", "discard_many", "set_many"])
def test_finalizers_see_complete_batch(object_map_cls, operation, unraisable):
    m = object_map_cls()
    first, second, marker = (_key(object_map_cls, i) for i in (1, 2, 9))
    events = []

    class Finalizer:
        def __del__(self):
            events.append((m.get(first), m.get(second)))
            m[marker] = len(events)

    m[first] = Finalizer()
    m[second] = Finalizer()
    if operation == "clear":
        m.clear()
    elif operation == "discard_many":
        assert m.discard_many([first, second, first]) == 2
    else:
        assert m.set_many([first, second], ["first", "second"]) == 0
    expected = ("first", "second") if operation == "set_many" else (None, None)
    assert events == [expected, expected]
    assert m[marker] == 2
    assert unraisable == []


def test_duplicate_batch_value_finalizes_after_write(object_map_cls, unraisable):
    m = object_map_cls()
    key, marker = (_key(object_map_cls, i) for i in (1, 9))
    events = []

    class Finalizer:
        def __del__(self):
            events.append(m[key])
            m[marker] = "finalized"

    # No caller-owned list keeps the first input value alive.
    def values():
        yield Finalizer()
        yield "last"

    assert m.set_many([key, key], values()) == 1
    assert events == ["last"]
    assert m[marker] == "finalized"
    assert unraisable == []


@pytest.mark.parametrize("operation", ["overwrite", "set_many", "setdefault", "update"])
def test_id_generation_eviction_releases_borrow(operation, unraisable):
    m = pc.IdMap()
    old, new, marker = (1 << 32 | 1), (2 << 32 | 1), (1 << 32 | 9)
    events = []

    class Finalizer:
        def __del__(self):
            events.append((old in m, m[new]))
            m[marker] = "finalized"

    m[old] = Finalizer()
    if operation == "overwrite":
        m[new] = "replacement"
    elif operation == "set_many":
        assert m.set_many([new], ["replacement"]) == 1
    elif operation == "setdefault":
        assert m.setdefault(new, "replacement") == "replacement"
    else:
        m.update({new: "replacement"})
    assert events == [(False, "replacement")]
    assert m[marker] == "finalized"
    assert unraisable == []


@pytest.mark.parametrize("operation", ["get", "setdefault"])
def test_unused_default_can_finalize(object_map_cls, operation, unraisable):
    key, marker = (_key(object_map_cls, i) for i in (1, 9))
    m = object_map_cls([key], ["existing"])
    events = []

    class Finalizer:
        def __del__(self):
            m[marker] = "finalized"
            events.append(True)

    assert getattr(m, operation)(key, Finalizer()) == "existing"
    assert events == [True]
    assert m[marker] == "finalized"
    assert unraisable == []


@pytest.mark.parametrize("method", ["pop_first", "pop_last"])
def test_direct_pop_finalizer_can_write(method, unraisable):
    m = pc.DirectMap()
    events = []

    class Finalizer:
        def __del__(self):
            m[9] = "finalized"
            events.append(True)

    m[1] = Finalizer()
    getattr(m, method)()
    assert events == [True]
    assert m.items() == [(9, "finalized")]
    assert unraisable == []


def test_equality_can_mutate_both_maps(object_map_cls):
    key, marker = (_key(object_map_cls, i) for i in (1, 9))
    left, right = object_map_cls(), object_map_cls()
    events = []

    class Compared:
        def __eq__(self, other):
            left.clear()
            right.clear()
            left[marker] = "left"
            right[marker] = "right"
            events.append(True)
            return True

    left[key] = Compared()
    right[key] = object()
    assert left == right  # The values are compared from a snapshot.
    assert events == [True]
    assert left.items() == [(marker, "left")]
    assert right.items() == [(marker, "right")]


def test_equality_keeps_identity_shortcut(object_map_cls):
    class NeverCompare:
        def __eq__(self, other):
            raise AssertionError("identity must win")

    key = _key(object_map_cls, 1)
    value = NeverCompare()
    m = object_map_cls([key], [value])
    assert m == m
    assert m == m.copy()
    nan = float("nan")
    assert object_map_cls([key], [nan]) == object_map_cls([key], [nan])


def test_repr_can_read_and_mutate_map(object_map_cls):
    key, marker = (_key(object_map_cls, i) for i in (1, 9))
    m = object_map_cls()

    class Represented:
        def __repr__(self):
            assert len(m) == 1
            assert key in m
            m.clear()
            m[marker] = "from repr"
            return "Represented()"

    m[key] = Represented()
    assert repr(m) == f"{object_map_cls.__name__}({{{key}: Represented()}})"
    assert m.items() == [(marker, "from repr")]
    m.clear()
    m[key] = m
    name = object_map_cls.__name__
    assert repr(m) == f"{name}({{{key}: {name}(...)}})"


@pytest.mark.parametrize("method", ["set_many", "__init__"])
def test_values_generator_can_read_map(object_map_cls, method):
    key = _key(object_map_cls, 1)
    m = object_map_cls()

    def values():
        yield len(m)

    result = getattr(m, method)([key], values())
    assert result == (1 if method == "set_many" else None)
    assert m[key] == 0


@pytest.mark.parametrize("cls", (pc.SparseMap, pc.IdMap, *DIRECT_MAPS), ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("method", ["set_many", "__init__", "discard_many", "get_many", "contains_many"])
def test_key_array_conversion_can_mutate_map(cls, method):
    key, marker = (_key(cls, i) for i in (1, 9))
    value = True if cls.dtype == "bool" else 7
    m = cls([key], [value])
    calls = []

    class Keys:
        def __array__(self, dtype=None, copy=None):
            m[marker] = value
            calls.append(True)
            return np.array([key], dtype=np.uint64 if cls is pc.IdMap else np.uint32)

    args = (Keys(), [value]) if method in {"set_many", "__init__"} else (Keys(),)
    getattr(m, method)(*args)
    assert calls
    assert m[marker] == value


@pytest.mark.parametrize("cls", DIRECT_MAPS[1:], ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("method", ["set_many", "__init__"])
def test_typed_direct_values_conversion_can_read_map(cls, method):
    value = True if cls.dtype == "bool" else 7
    m = cls()
    calls = []

    class Values:
        def __array__(self, dtype=None, copy=None):
            calls.append(len(m))
            return np.array([value], dtype=np.dtype(cls.dtype))

    getattr(m, method)([1], Values())
    assert calls == [0]
    assert m[1] == value


@pytest.mark.parametrize("failure", ["length", "iteration"])
def test_failed_batch_finalizers_release_borrow(object_map_cls, failure, unraisable):
    m = object_map_cls()
    key, marker = (_key(object_map_cls, i) for i in (1, 9))
    events = []

    class Finalizer:
        def __del__(self):
            m[marker] = "finalized"
            events.append(True)

    def values():
        yield Finalizer()
        if failure == "iteration":
            raise ArithmeticError("input failure")
        yield "extra"

    error = ArithmeticError if failure == "iteration" else ValueError
    with pytest.raises(error):
        m.set_many([key], values())
    assert key not in m
    assert m.items() == [(marker, "finalized")]
    assert events == [True]
    assert unraisable == []


@pytest.mark.parametrize("cls", [pc.SparseMap, pc.IdMap], ids=lambda cls: cls.__name__)
def test_exported_keys_allow_reentrant_overwrite(cls, unraisable):
    key, marker = (_key(cls, i) for i in (1, 9))
    m = cls([key, marker], [None, None])
    events = []

    class Finalizer:
        def __del__(self):
            m[marker] = "finalized"
            events.append(True)

    m[key] = Finalizer()
    view = m.keys_view()
    assert m.set_many([key], ["replacement"]) == 0
    assert m[marker] == "finalized"
    assert events == [True]
    np.testing.assert_array_equal(view, [key, marker])
    assert unraisable == []


@pytest.mark.parametrize("cls", [pc.SparseMap, pc.IdMap], ids=lambda cls: cls.__name__)
def test_export_rejection_releases_input_values(cls, unraisable):
    key, new_key = (_key(cls, i) for i in (1, 2))
    m = cls([key], ["original"])
    events = []

    class Finalizer:
        def __del__(self):
            events.append(m[key])
            m[key] = "finalized"

    def values():
        yield Finalizer()
        yield "new"

    view = m.keys_view()
    with pytest.raises(BufferError):
        m.set_many([key, new_key], values())
    assert events == ["original"]
    assert m.items() == [(key, "finalized")]
    np.testing.assert_array_equal(view, [key])
    assert unraisable == []
