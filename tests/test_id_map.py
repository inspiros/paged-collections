"""Generational-key contracts shared by object and typed IdMaps."""

from collections.abc import MutableMapping
import copy
import operator
import pickle
from types import MappingProxyType

import numpy as np
import pytest

import paged_collections as pc
from paged_collections import _core


TYPED_ID_MAP_CLASSES = (
    pc.IdMapF64, pc.IdMapF32, pc.IdMapI64, pc.IdMapI32,
    pc.IdMapU64, pc.IdMapU32, pc.IdMapBool,
)
ID_MAP_CLASSES = (pc.IdMap, *TYPED_ID_MAP_CLASSES)


@pytest.fixture(params=ID_MAP_CLASSES, ids=lambda cls: cls.__name__)
def id_map_cls(request):
    return request.param


@pytest.fixture(params=TYPED_ID_MAP_CLASSES, ids=lambda cls: cls.__name__)
def typed_id_map_cls(request):
    return request.param


def _id(index, generation=1):
    return generation << 32 | index


def _keys(*ids):
    # Explicit dtype preserves high-bit ids that NumPy might infer as float64
    # when a Python sequence mixes signed-range and unsigned-range integers.
    return np.array(ids, dtype=np.uint64)


def _values(cls, numbers):
    if cls is pc.IdMapBool:
        return [bool(n % 2) for n in numbers]
    return list(numbers)


def _assert_values(mapping, actual, expected):
    if isinstance(mapping, pc.IdMap):
        assert isinstance(actual, list)
        assert actual == expected
    else:
        assert isinstance(actual, np.ndarray)
        assert actual.dtype == np.dtype(mapping.dtype)
        np.testing.assert_array_equal(actual, expected)


def _write(mapping, operation, keys, values):
    if operation == "setitem":
        mapping[keys[0]] = values[0]
    elif operation == "setdefault":
        return mapping.setdefault(keys[0], values[0])
    elif operation == "set_many":
        return mapping.set_many(keys, values)
    elif operation == "update":
        return mapping.update(list(zip(keys, values)))
    elif operation == "init":
        return mapping.__init__(keys, values)
    else:
        raise AssertionError(operation)


def test_dict_operations_and_dense_removal(id_map_cls):
    m = id_map_cls()
    a, b, c, d = _values(id_map_cls, [1, 2, 3, 4])
    k1, k2, k3, k4 = (_id(i) for i in (9, 2, 7, 4))
    assert not m and len(m) == 0 and list(m) == []
    m[k1] = a
    m[np.uint64(k2)] = b
    m[np.int64(k1)] = c
    assert len(m) == 2
    assert m[k1] == m.get(k1) == c
    assert m[np.uint64(k2)] == b
    assert m.setdefault(k1, a) == c
    assert m.setdefault(k3, a) == a
    m.update({k4: d})
    m.update([(k3, b)])
    m.update(m)
    assert m.items() == [(k1, c), (k2, b), (k3, b), (k4, d)]
    del m[k2]
    assert list(m) == [k1, k4, k3]
    assert m.pop(k4) == d
    assert m.popitem() == (k3, b)
    assert m.pop(k1) == c
    with pytest.raises(KeyError):
        m.popitem()
    assert m.clear() is None
    m[k2] = a
    assert m.clear() is None
    assert not m


@pytest.mark.parametrize("kwargs", [{"keys": [_id(1)]}, {"values": [True]}])
def test_constructor_requires_both_arrays(id_map_cls, kwargs):
    with pytest.raises(TypeError):
        id_map_cls(**kwargs)


def test_constructor_mapping_and_pairs_resolve_generations(id_map_cls):
    a, b = _values(id_map_cls, [1, 2])
    old, current, other = _id(7), _id(7, 0xFFFFFFFF), _id(2)
    expected = {current: b, other: a}
    assert dict(id_map_cls(MappingProxyType(expected))) == expected
    m = id_map_cls(iter([(old, a), (other, a), (current, b)]))
    assert isinstance(m, MutableMapping)
    assert dict(m) == expected and old not in m
    with pytest.raises(ValueError, match="not a valid id"):
        m.__init__([(current, a), (2, b)])
    assert dict(m) == expected


def test_range_snapshots_use_slot_indices(id_map_cls):
    a, b, c = _values(id_map_cls, [1, 2, 3])
    high_generation, last, middle = _id(2, 0xFFFFFFFF), _id(4096), _id(7, 8)
    m = id_map_cls([last, middle, high_generation], [a, b, c])
    assert m.items_range() == [(high_generation, c), (middle, b), (last, a)]
    assert m.items_range(start=np.int64(2), stop=7) == [(high_generation, c)]
    assert m.items_range(7, 7) == m.items_range(start=2**100) == []
    keys, values = m.keys_range(7, 4096), m.values_range(7, 4096)
    m.clear()
    assert keys.dtype == np.dtype(np.uint64)
    assert keys.tolist() == [middle]
    _assert_values(m, values, [b])
    for method in (m.keys_range, m.values_range, m.items_range):
        with pytest.raises(TypeError):
            method(stop="7")


@pytest.mark.parametrize("generation", [1, 2, 4, 0xFFFFFFFF])
def test_older_and_newer_ids_are_absent_on_every_lookup_path(id_map_cls, generation):
    current, stale = _id(7, 3), _id(7, generation)
    value, default = _values(id_map_cls, [1, 2])
    m = id_map_cls([current], [value])
    sentinel = object()
    assert stale not in m
    assert m.get(stale) is None
    assert m.get(stale, sentinel) is sentinel
    assert m.pop(stale, sentinel) is sentinel
    assert m.pop(stale, None) is None
    for op in (lambda: m[stale], lambda: m.pop(stale),
               lambda: m.__delitem__(stale)):
        with pytest.raises(KeyError) as exc:
            op()
        assert exc.value.args == (stale,)
    query = _keys(current, stale, current)
    with pytest.raises(KeyError) as exc:
        m.get_many(query)
    assert exc.value.args == (stale,)
    assert type(exc.value.args[0]) is int
    _assert_values(m, m.get_many(query, default), [value, default, value])
    np.testing.assert_array_equal(m.contains_many(query), [True, False, True])
    assert m.discard_many(_keys(stale, stale)) == 0
    assert m.items() == [(current, value)]


@pytest.mark.parametrize(
    "key, error",
    [(None, TypeError), ("id", TypeError), (1.0, TypeError), ([], TypeError),
     (-1, OverflowError), (2**64, OverflowError), (2**100, OverflowError),
     (0, ValueError), (7, ValueError), (2**32 - 1, ValueError)],
)
def test_invalid_scalar_lookup_and_insertion(id_map_cls, key, error):
    value = _values(id_map_cls, [1])[0]
    m = id_map_cls([_id(7)], [value])
    sentinel = object()
    assert key not in m
    assert m.get(key) is None
    assert m.get(key, sentinel) is sentinel
    assert m.pop(key, sentinel) is sentinel
    for op in (lambda: m[key], lambda: m.pop(key), lambda: m.__delitem__(key)):
        with pytest.raises(KeyError):
            op()
    for operation in ("setitem", "setdefault", "update"):
        with pytest.raises(error) as exc:
            _write(m, operation, [key], [value])
        if error is ValueError:
            assert f"{key} is not a valid id" in str(exc.value)
        assert m.items() == [(_id(7), value)]


def test_invalid_integer_batch_lookups_are_absent(id_map_cls):
    current = _id(7)
    value, default = _values(id_map_cls, [1, 2])
    m = id_map_cls([current], [value])
    for query in (np.array([current, -1, 7, 0], dtype=np.int64),
                  _keys(current, 0, 7, 2**32 - 1)):
        np.testing.assert_array_equal(m.contains_many(query), [True, False, False, False])
        _assert_values(m, m.get_many(query, default), [value, default, default, default])
        with pytest.raises(KeyError) as exc:
            m.get_many(query)
        assert exc.value.args == (int(query[1]),)
        assert type(exc.value.args[0]) is int
        assert m.discard_many(query[1:]) == 0
    assert m.items() == [(current, value)]


@pytest.mark.parametrize("operation", ["setitem", "setdefault", "set_many", "update", "init"])
@pytest.mark.parametrize("generation", [1, 4])
def test_eviction_keeps_length_and_dense_position(id_map_cls, operation, generation):
    # Insertion can evict a newer generation too; no allocator is consulted.
    left, old, right = _id(9), _id(2, 3), _id(7)
    replacement = _id(2, generation)
    a, b, c, d = _values(id_map_cls, [1, 2, 3, 4])
    m = id_map_cls([left, old, right], [a, b, c])
    iterator = iter(m)
    assert next(iterator) == left
    result = _write(m, operation, [replacement], [d])
    if operation == "set_many":
        assert result == 1
    elif operation == "setdefault":
        assert result == d
    assert len(m) == 3
    np.testing.assert_array_equal(m.keys(), [left, replacement, right])
    assert m.items() == [(left, a), (replacement, d), (right, c)]
    assert old not in m and m[replacement] == d
    with pytest.raises(RuntimeError, match="changed during iteration"):
        next(iterator)


def test_batch_counts_each_generation_replacement(id_map_cls):
    old, new, other = _id(7), _id(7, 2), _id(2, 0xFFFFFFFF)
    values = _values(id_map_cls, [1, 2, 3, 4, 5, 6])
    m = id_map_cls([old], values[:1])
    keys = _keys(old, new, new, old, new, other)
    assert m.set_many(keys, values) == 4  # three evictions and one vacant index
    assert len(m) == 2
    assert m.items() == [(new, values[4]), (other, values[5])]
    assert m.set_many(_keys(new, new, other), values[:3]) == 0
    restored = id_map_cls(keys, values)
    assert restored.items() == [(new, values[4]), (other, values[5])]


@pytest.mark.parametrize("operation", ["setitem", "setdefault", "set_many", "update", "init"])
@pytest.mark.parametrize("view_method", ["keys_view", "values_view"])
def test_live_views_block_eviction(id_map_cls, operation, view_method):
    if id_map_cls is pc.IdMap and view_method == "values_view":
        pytest.skip("object maps expose only keys_view")
    first, old, replacement = _id(2), _id(7), _id(7, 2)
    a, b = _values(id_map_cls, [1, 2])
    m = id_map_cls([first, old], [a, b])
    view = getattr(m, view_method)()
    before = m.items()
    iterator = iter(m)
    # Batch writes must check for eviction before overwriting an exact id.
    keys = [first, replacement] if operation in ("set_many", "init") else [replacement]
    values = [b, a] if len(keys) == 2 else [a]
    with pytest.raises(BufferError):
        _write(m, operation, keys, values)
    assert m.items() == before
    assert list(iterator) == [first, old]
    assert len(view) == 2
    del view
    _write(m, operation, keys, values)
    assert len(m) == 2 and old not in m and replacement in m


@pytest.mark.parametrize("view_method", ["keys_view", "values_view"])
def test_exact_value_writes_work_with_views_and_keep_iterator_valid(typed_id_map_cls, view_method):
    cls = typed_id_map_cls
    key = _id(7, 0xFFFFFFFF)
    a, b = _values(cls, [1, 2])
    m = cls(_keys(key), [a])
    view = getattr(m, view_method)()
    iterator = iter(m)
    m[key] = b
    assert m[key] == b
    if view_method == "values_view":
        assert view[0] == b
        assert view.flags.writeable == (cls is not pc.IdMapBool)
        if cls is not pc.IdMapBool:
            view[0] = a
            assert m[key] == a
    assert m.set_many(_keys(key, key), [b, a]) == 0
    assert m.setdefault(key, b) == a
    m.update({key: b})
    assert m[key] == b
    assert list(iterator) == [key]
    if view_method == "values_view":
        assert view[0] == b


def test_object_exact_writes_under_keys_view():
    key = _id(7)
    m = pc.IdMap([key], [[1]])
    view = m.keys_view()
    iterator = iter(m)
    value = [2]
    m[key] = value
    assert m[key] is value
    assert m.set_many([key], [None]) == 0
    assert m[key] is None
    assert list(iterator) == [key]
    assert int(view[0]) == key


def test_both_views_keep_eviction_blocked_until_last_release(typed_id_map_cls):
    cls = typed_id_map_cls
    old, new = _id(7), _id(7, 2)
    value = _values(cls, [1])[0]
    m = cls([old], [value])
    keys, values = m.keys_view(), m.values_view()
    with pytest.raises(BufferError):
        m[new] = value
    del keys
    with pytest.raises(BufferError):
        m[new] = value
    assert int(values[0]) == int(value)
    del values
    m[new] = value
    assert m.items() == [(new, value)]


@pytest.mark.parametrize(
    "keys, error, message",
    [([_id(7), _id(2, 2), 0], ValueError, "key 0 at position 2 is not a valid id"),
     (_keys(_id(7), _id(2, 2), 9), ValueError,
      "key 9 at position 2 is not a valid id"),
     (np.array([_id(7), _id(2, 2), -1], dtype=np.int64), ValueError,
      "key -1 at position 2 is not a valid id"),
     ([_id(7), _id(2, 2), 2**64], ValueError,
      f"key {2**64} at position 2 is not a valid id"),
     ([_id(7), _id(2, 2), -1], ValueError, "key -1 at position 2 is not a valid id"),
     ([_id(7), _id(2, 2), 1.5], TypeError, None),
     (np.array([True, False, True]), TypeError, None),
     (np.array([_id(7), _id(2, 2), 0], dtype=object), TypeError, None),
     ([[_id(7), _id(2, 2), _id(9)]], ValueError, None)],
)
def test_bad_batches_are_atomic(id_map_cls, keys, error, message):
    values = _values(id_map_cls, [1, 2, 3])
    m = id_map_cls([_id(7), _id(2)], values[:2])
    before = m.items()
    pages = m.allocated_pages
    for operation in ("set_many", "init"):
        iterator = iter(m)
        with pytest.raises(error) as exc:
            _write(m, operation, keys, values)
        if message is not None:
            assert str(exc.value) == message
        assert m.items() == before
        assert m.allocated_pages == pages
        assert list(iterator) == [key for key, _ in before]
    with pytest.raises(error):
        id_map_cls(keys, values)


def test_length_mismatch_cannot_partially_write_or_evict(id_map_cls):
    values = _values(id_map_cls, [1, 2, 3])
    m = id_map_cls([_id(7), _id(2)], values[:2])
    before = m.items()
    for bad_values in (values[:1], values):
        with pytest.raises(ValueError):
            m.set_many([_id(7), _id(2, 2)], bad_values)
        assert m.items() == before
        with pytest.raises(ValueError):
            id_map_cls([_id(7), _id(2, 2)], bad_values)


@pytest.mark.parametrize("operation", ["setitem", "setdefault", "set_many", "update", "init"])
def test_typed_bad_values_do_not_evict(typed_id_map_cls, operation):
    cls = typed_id_map_cls
    value = _values(cls, [1])[0]
    m = cls([_id(7)], [value])
    with pytest.raises((TypeError, ValueError)):
        _write(m, operation, [_id(7, 2)], [object()])
    assert m.items() == [(_id(7), value)]


def test_update_is_sequential_like_dict(id_map_cls):
    a, b = _values(id_map_cls, [1, 2])
    m = id_map_cls([_id(7)], [a])
    with pytest.raises(ValueError, match="0 is not a valid id"):
        m.update([(_id(7, 2), b), (0, a)])
    assert m.items() == [(_id(7, 2), b)]


@pytest.mark.parametrize("layout", ["native", "big_endian", "unaligned", "unaligned_big_endian", "strided", "reversed"])
def test_uint64_batch_layouts_and_high_bits(id_map_cls, layout):
    wanted = [_id(7, 0xFFFFFFFF), _id(1, 0x80000000), _id(9), _id(7, 0xFFFFFFFF)]
    dtype = np.dtype(">u8") if "big_endian" in layout else np.dtype(np.uint64)
    if "unaligned" in layout:
        keys = np.frombuffer(bytearray(8 * 4 + 1), dtype=dtype, offset=1, count=4)
        assert not keys.flags.aligned
        keys[:] = wanted
    elif layout == "strided":
        keys = np.empty(8, dtype=dtype)[::2]
        keys[:] = wanted
    elif layout == "reversed":
        keys = np.array(wanted[::-1], dtype=dtype)[::-1]
    else:
        keys = np.array(wanted, dtype=dtype)
    values = _values(id_map_cls, [1, 2, 3, 4])
    m = id_map_cls(keys, values)
    assert m.items() == [(wanted[0], values[3]), (wanted[1], values[1]), (wanted[2], values[2])]
    assert m.set_many(keys, values) == 0
    np.testing.assert_array_equal(m.contains_many(keys), [True] * 4)
    _assert_values(m, m.get_many(keys), [values[3], values[1], values[2], values[3]])
    assert m.discard_many(keys) == 3
    assert not m
    assert m.set_many(keys, values) == 3


def test_empty_batches(id_map_cls):
    m = id_map_cls([], [])
    assert not m
    assert m.set_many([], []) == 0
    assert m.discard_many([]) == 0
    result = m.contains_many([])
    assert result.shape == (0,) and result.dtype == np.dtype(bool)
    _assert_values(m, m.get_many([]), [])


@pytest.mark.parametrize("method", ["get_many", "contains_many", "discard_many"])
@pytest.mark.parametrize("keys, error", [([1.5], TypeError), ([True], TypeError),
                                         (np.array([_id(1)], dtype=object), TypeError),
                                         ([[_id(1)]], ValueError)])
def test_lookup_batches_validate_kind_and_shape(id_map_cls, method, keys, error):
    value = _values(id_map_cls, [1])[0]
    m = id_map_cls([_id(1)], [value])
    with pytest.raises(error):
        getattr(m, method)(keys)
    assert m.items() == [(_id(1), value)]


def test_keys_view_is_uint64_readonly_and_zero_copy(id_map_cls):
    keys = _keys(_id(7, 0xFFFFFFFF), _id(1, 2), _id(9))
    values = _values(id_map_cls, [1, 2, 3])
    m = id_map_cls(keys, values)
    view, other_view = m.keys_view(), m.keys_view()
    detached = m.keys()
    assert detached.dtype == view.dtype == np.dtype(np.uint64)
    assert not view.flags.writeable
    assert np.shares_memory(view, other_view)
    np.testing.assert_array_equal(view, detached)
    with pytest.raises(ValueError):
        view[0] = 0
    with pytest.raises(ValueError):
        view.setflags(write=True)
    detached[0] = 0
    assert int(view[0]) == int(keys[0])
    assert m.items() == list(zip(map(int, keys), values))
    del view, other_view
    m[_id(7, 2)] = values[1]
    assert _id(7, 0xFFFFFFFF) not in m


def test_values_and_items_are_detached(id_map_cls):
    keys = [_id(7), _id(2, 3)]
    a, b = _values(id_map_cls, [1, 2])
    m = id_map_cls(keys, [a, b])
    values, items = m.values(), m.items()
    _assert_values(m, values, [a, b])
    values[0] = b
    items[0] = (_id(9), b)
    assert m.items() == list(zip(keys, [a, b]))
    m[keys[1]] = a
    assert values[1] == b
    assert items[1] == (keys[1], b)


def test_typed_batch_scalar_broadcast(typed_id_map_cls):
    cls = typed_id_map_cls
    a, b = _values(cls, [1, 2])
    m = cls([_id(7)], [a])
    assert m.set_many([_id(7, 2), _id(2, 3)], b) == 2
    assert m.items() == [(_id(7, 2), b), (_id(2, 3), b)]


def test_sort_keys_uses_index_then_generation(id_map_cls):
    keys = [_id(9), _id(1, 0xFFFFFFFF), _id(7, 2), _id(2, 3)]
    values = _values(id_map_cls, [1, 2, 3, 4])
    m = id_map_cls(_keys(*keys), values)
    expected = sorted(zip(keys, values), key=lambda pair: (pair[0] & 0xFFFFFFFF, pair[0] >> 32))
    assert [key for key, _ in expected] != sorted(keys)
    iterator = iter(m)
    assert m.sort_keys() is None
    assert m.items() == expected
    with pytest.raises(RuntimeError):
        next(iterator)
    assert m.popitem() == expected[-1]


def test_iterator_class_and_exhaustion(id_map_cls):
    value = _values(id_map_cls, [1])[0]
    key = _id(7, 0xFFFFFFFF)
    m = id_map_cls(_keys(key), [value])
    iterator = iter(m)
    assert type(iterator) is getattr(_core, f"{id_map_cls.__name__}Iterator")
    assert iter(iterator) is iterator
    assert operator.length_hint(iterator) == 1
    result = next(iterator)
    assert type(result) is int and result == key
    with pytest.raises(StopIteration):
        next(iterator)
    m[_id(7, 2)] = value
    assert operator.length_hint(iterator) == 0
    with pytest.raises(StopIteration):
        next(iterator)


def test_equality_repr_and_unhashability(id_map_cls):
    a, b = _values(id_map_cls, [1, 2])
    keys = [_id(7), _id(2, 3)]
    m = id_map_cls(keys, [a, b])
    assert m == id_map_cls(keys[::-1], [b, a])
    assert m != id_map_cls(keys, [b, a])
    assert m != id_map_cls([_id(7, 2), keys[1]], [a, b])
    assert m != id_map_cls(keys[:1], [a])
    shown = ", ".join(f"{key}: {m[key]!r}" for key in keys)
    assert repr(m) == f"{id_map_cls.__name__}({{{shown}}})"
    assert repr(id_map_cls()) == f"{id_map_cls.__name__}({{}})"
    with pytest.raises(TypeError):
        hash(m)


_COPIERS = {
    "method": lambda obj: obj.copy(),
    "copy": copy.copy,
    "deepcopy": copy.deepcopy,
    **{f"pickle-{p}": (lambda obj, p=p: pickle.loads(pickle.dumps(obj, protocol=p)))
       for p in range(2, pickle.HIGHEST_PROTOCOL + 1)},
}


@pytest.mark.parametrize("copier", _COPIERS.values(), ids=_COPIERS.keys())
def test_copy_and_pickle_preserve_ids_order_and_independence(id_map_cls, copier):
    old, replacement = _id(7), _id(7, 0xFFFFFFFF)
    keys = _keys(_id(9), old, _id(2, 3))
    values = _values(id_map_cls, [1, 2, 3])
    m = id_map_cls(keys, values)
    m[replacement] = values[0]
    restored = copier(m)
    assert type(restored) is id_map_cls
    assert restored is not m and restored == m
    assert restored.items() == m.items()
    assert old not in restored
    restored[old] = values[1]
    del restored[_id(9)]
    assert old not in m and replacement in m and _id(9) in m


def _labelled(base):
    def __init__(self, label, keys=None, values=None):
        base.__init__(self, keys, values)
        self.label = label

    cls = type(f"Labelled{base.__name__}", (base,), {"__init__": __init__})
    cls.__module__ = __name__
    globals()[cls.__name__] = cls
    return cls


# Module-level names allow pickle to recover each Python subclass.
_LABELLED = {base: _labelled(base) for base in ID_MAP_CLASSES}


@pytest.mark.parametrize("copier", _COPIERS.values(), ids=_COPIERS.keys())
def test_subclass_roundtrip_and_method_copy(id_map_cls, copier):
    cls = _LABELLED[id_map_cls]
    values = _values(id_map_cls, [1, 2])
    original = cls("label", _keys(_id(7, 0xFFFFFFFF), _id(2, 3)), values)
    restored = copier(original)
    assert restored is not original
    assert restored.items() == original.items()
    if copier is _COPIERS["method"]:
        assert type(restored) is id_map_cls
    else:
        assert type(restored) is cls
        assert restored.label == "label"
    assert repr(original).startswith(f"{cls.__name__}(")


def test_object_value_ownership_and_recursive_roundtrip(object_roundtrip):
    old, new = _id(7), _id(7, 2)
    nested = [1]
    m = pc.IdMap([old], [nested])
    assert m[old] is nested
    assert m.copy()[old] is nested
    assert copy.copy(m)[old] is nested
    deep = copy.deepcopy(m)
    assert deep[old] == nested and deep[old] is not nested
    m[new] = m
    restored = object_roundtrip(m)
    assert old not in restored
    assert restored[new] is restored
    assert repr(m) == f"IdMap({{{new}: IdMap(...)}})"


def test_dtype_factory_and_package_exports(id_map_cls):
    expected = {
        pc.IdMap: "object", pc.IdMapF64: "float64", pc.IdMapF32: "float32",
        pc.IdMapI64: "int64", pc.IdMapI32: "int32", pc.IdMapU64: "uint64",
        pc.IdMapU32: "uint32", pc.IdMapBool: "bool",
    }[id_map_cls]
    assert id_map_cls.dtype == expected
    assert id_map_cls.__name__ in pc.__all__
    assert getattr(pc, id_map_cls.__name__) is getattr(_core, id_map_cls.__name__)
    assert "id_map" in pc.__all__
    specs = [expected, np.dtype(expected), np.dtype(expected).type]
    if id_map_cls is pc.IdMap:
        specs += [None, object]
    keys = _keys(_id(7, 0xFFFFFFFF), _id(2, 3))
    values = _values(id_map_cls, [1, 2])
    for spec in specs:
        assert type(pc.id_map(spec)) is id_map_cls
        m = pc.id_map(spec, keys=keys, values=values)
        assert type(m) is id_map_cls
        assert m.items() == list(zip(map(int, keys), values))


@pytest.mark.parametrize("dtype", ["int8", "uint16", "float16", "complex128", "U4", "datetime64[ns]"])
def test_factory_rejects_unsupported_dtypes(dtype):
    with pytest.raises(TypeError, match="no id map for dtype"):
        pc.id_map(dtype)


@pytest.mark.parametrize("kwargs", [{"keys": [_id(7)]}, {"values": [1]}])
def test_factory_requires_both_keys_and_values(kwargs):
    with pytest.raises(TypeError):
        pc.id_map(**kwargs)


def test_allocator_reuse_replaces_the_old_generation(id_map_cls):
    allocator = pc.IdAllocator()
    old, other = allocator.alloc(), allocator.alloc()
    a, b, c = _values(id_map_cls, [1, 2, 3])
    m = id_map_cls([old, other], [a, b])
    assert allocator.free(old)
    new = allocator.alloc()
    assert pc.id_index(new) == pc.id_index(old)
    assert pc.id_generation(new) == pc.id_generation(old) + 1
    assert old not in allocator and new in allocator
    # Allocators and maps have independent storage: insert the replacement id.
    assert old in m and new not in m
    m[new] = c
    assert len(m) == 2 and old not in m
    assert list(m) == [new, other]
    assert m.get(old) is None and m[new] == c
    assert m.discard_many([old]) == 0
    np.testing.assert_array_equal(m.contains_many(allocator.ids()), [True, True])
    assert m.discard_many(_keys(old, new, new)) == 1
    assert m.items() == [(other, b)]


def test_plain_lists_mixing_high_generations_are_read_exactly(id_map_cls):
    # NumPy infers float64 for this list; ids must not go through floats.
    high, low = _id(7, 0x80000000), _id(9, 1)
    values = _values(id_map_cls, [1, 2])
    m = id_map_cls([high, low], values)
    assert m.keys().tolist() == [high, low]
    assert m.contains_many([high, low, _id(7, 1)]).tolist() == [True, True, False]
    assert m.set_many([low, high], values[::-1]) == 0
    assert m.discard_many([np.uint64(high), low]) == 2
    with pytest.raises(TypeError):
        m.contains_many([high, 1.5])
    with pytest.raises(ValueError, match="at position 1 is not a valid id"):
        m.set_many([high, 2**64 + 1], values)
    assert len(m) == 0


def test_batch_replacing_then_restoring_a_generation(id_map_cls):
    old, new = _id(3, 1), _id(3, 2)
    a, b, c = _values(id_map_cls, [1, 2, 3])
    m = id_map_cls([old], [a])
    it = iter(m)
    # Both keys are new when written: the slot ends holding `old` again.
    assert m.set_many([new, old], [b, c]) == 2
    assert m.keys().tolist() == [old] and m[old] == c and new not in m
    with pytest.raises(RuntimeError):
        next(it)
