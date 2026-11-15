"""Shared map contracts, numeric conversions, object ownership and model tests."""

import copy
import gc
import pickle
import random
import sys

import numpy as np
import pytest

import paged_collections as pc


def _values(cls, numbers):
    # Use distinguishable, exactly representable values in every numeric class.
    if cls is pc.SparseMapBool:
        return [bool(n % 2) for n in numbers]
    return list(numbers)


def _assert_values(mapping, actual, expected):
    if type(mapping) is pc.SparseMap:
        assert isinstance(actual, list)
        assert actual == expected
    else:
        assert isinstance(actual, np.ndarray)
        assert actual.dtype == np.dtype(mapping.dtype)
        np.testing.assert_array_equal(actual, expected)


def test_constructor_and_scalar_operations(map_cls):
    m = map_cls()
    a, b, c = _values(map_cls, [1, 2, 3])
    assert not m
    assert len(m) == 0
    assert list(m) == []
    m[9] = a
    m[np.int64(2)] = b
    m[9] = c
    assert len(m) == 2
    assert m[9] == c
    assert m[np.uint32(2)] == b
    assert m.get(9) == c
    assert m.get(100) is None
    sentinel = object()
    assert m.get(100, sentinel) is sentinel
    assert m.setdefault(9, b) == c
    assert m.setdefault(7, a) == a
    assert list(m) == [9, 2, 7]
    assert m.pop(2) == b
    assert m.pop(100, sentinel) is sentinel
    assert m.pop(100, None) is None
    del m[9]
    assert m.popitem() == (7, a)
    with pytest.raises(KeyError):
        m.popitem()
    with pytest.raises(KeyError):
        m.pop(7)
    with pytest.raises(KeyError):
        del m[7]
    assert m.clear() is None


@pytest.mark.parametrize("kwargs", [{"keys": [1]}, {"values": [1]}])
def test_constructor_requires_both_keys_and_values(map_cls, kwargs):
    with pytest.raises(TypeError):
        map_cls(**kwargs)


@pytest.mark.parametrize(
    "key, error",
    [(-1, OverflowError), (2**32, OverflowError), (2**100, OverflowError),
     (1.0, TypeError), ("1", TypeError), (None, TypeError), ([], TypeError)],
)
def test_invalid_scalar_keys_are_absent_but_cannot_be_inserted(map_cls, key, error):
    value = _values(map_cls, [1])[0]
    m = map_cls([1], [value])
    sentinel = object()
    assert key not in m
    assert m.get(key) is None
    assert m.get(key, sentinel) is sentinel
    assert m.pop(key, sentinel) is sentinel
    for operation in (lambda: m[key], lambda: m.__delitem__(key), lambda: m.pop(key)):
        with pytest.raises(KeyError):
            operation()
    with pytest.raises(error):
        m[key] = value
    with pytest.raises(error):
        m.setdefault(key, value)
    assert m.items() == [(1, value)]


def test_key_boundaries(map_cls):
    values = _values(map_cls, [1, 2])
    m = map_cls([0, 2**32 - 1], values)
    assert m[0] == values[0]
    assert m[np.uint64(2**32 - 1)] == values[1]
    assert m.discard_many([0, 2**32 - 1, 2**32]) == 2
    assert not m


def test_integer_batch_formats(map_cls, key_batch):
    values = _values(map_cls, [1, 2, 3, 4])
    m = map_cls(key_batch, values)
    assert list(m) == [7, 1, 9]
    assert m.items() == [(7, values[3]), (1, values[1]), (9, values[2])]
    _assert_values(m, m.get_many(key_batch),
                   [values[3], values[1], values[2], values[3]])
    result = m.contains_many(key_batch)
    assert result.dtype == np.dtype(bool)
    np.testing.assert_array_equal(result, [True] * 4)
    assert m.set_many(key_batch, values) == 0
    assert m.discard_many(key_batch) == 3
    assert not m
    assert m.set_many(key_batch, values) == 3


def test_unaligned_and_big_endian_keys(typed_map_cls, unusual_u32_keys):
    keys = unusual_u32_keys
    aligned = np.array(keys, dtype=np.uint32, copy=True)
    assert aligned.flags.aligned
    values = _values(typed_map_cls, [1, 2, 3, 4])
    m, reference = typed_map_cls([2], values[:1]), typed_map_cls([2], values[:1])
    assert m.set_many(keys, values) == reference.set_many(aligned, values) == 3
    assert m.items() == reference.items()
    assert list(m) == [2, 7, 1, 9]
    np.testing.assert_array_equal(m.contains_many(keys), reference.contains_many(aligned))
    np.testing.assert_array_equal(m.get_many(keys), reference.get_many(aligned))
    assert m.discard_many(keys) == reference.discard_many(aligned) == 3
    assert m.items() == reference.items() == [(2, values[0])]
    np.testing.assert_array_equal(m.contains_many(keys), [False] * 4)


def test_array_subclass_astype_is_ignored(typed_map_cls, hostile_array):
    plain_values = np.array(_values(typed_map_cls, [1, 2, 3, 4]),
                            dtype=typed_map_cls.dtype)
    values = plain_values.view(type(hostile_array))
    m, reference = typed_map_cls(), typed_map_cls()
    assert m.set_many(hostile_array, values) == 3
    assert reference.set_many(np.asarray(hostile_array), plain_values) == 3
    assert m.items() == reference.items()
    np.testing.assert_array_equal(m.get_many(hostile_array),
                                  reference.get_many(np.asarray(hostile_array)))
    assert m.discard_many(hostile_array) == 3
    assert not m


@pytest.mark.parametrize("additional", [sys.maxsize, 2**62])
def test_huge_reserve_raises_memoryerror_without_changing_map(map_cls, additional):
    values = _values(map_cls, [1, 2, 3])
    m = map_cls([9, 2, 7], values)
    before = m.items()
    pages = m.allocated_pages
    with pytest.raises(MemoryError):
        m.reserve(additional)
    assert m.items() == before
    assert m.allocated_pages == pages
    m[11] = values[1]
    del m[2]
    assert m.items() == [(9, values[0]), (11, values[1]), (7, values[2])]


def test_batch_duplicates_and_new_key_counts(map_cls):
    values = _values(map_cls, [1, 2, 3, 4, 5])
    m = map_cls([9], [values[0]])
    assert m.set_many([9, 2, 9, 2, 7], values) == 2
    assert m.items() == [(9, values[2]), (2, values[3]), (7, values[4])]
    assert m.discard_many([2, 2, -1, 2**32, 100]) == 1
    assert m.items() == [(9, values[2]), (7, values[4])]


def test_empty_batches(map_cls):
    m = map_cls([], [])
    assert not m
    assert m.set_many([], []) == 0
    assert m.discard_many([]) == 0
    result = m.contains_many([])
    assert result.shape == (0,) and result.dtype == np.dtype(bool)
    _assert_values(m, m.get_many([]), [])


@pytest.mark.parametrize(
    "keys, error",
    [([1, 3, -1], OverflowError),
     (np.array([1, 3, 2**32], dtype=np.uint64), OverflowError),
     ([1.0, 3.0, 4.0], TypeError),
     (np.array([True, False, True]), TypeError),
     (np.array([1, 3, 4], dtype=object), TypeError),
     ([[1, 3, 4]], ValueError),
     (np.empty((0, 2), dtype=np.int32), ValueError)],
)
def test_batch_key_validation_is_atomic(map_cls, keys, error):
    values = _values(map_cls, [1, 2, 3])
    m = map_cls([1, 8], values[:2])
    before = m.items()
    pages = m.allocated_pages
    with pytest.raises(error):
        m.set_many(keys, values)
    assert m.items() == before
    assert m.allocated_pages == pages
    with pytest.raises(error):
        map_cls(keys, values)


@pytest.mark.parametrize("method", ["get_many", "contains_many", "discard_many"])
@pytest.mark.parametrize(
    "keys, error",
    [(np.array([1.0]), TypeError), (np.array([True]), TypeError),
     (np.array([1], dtype=object), TypeError), ([[1]], ValueError)],
)
def test_batch_lookup_rejects_wrong_kind_or_shape(map_cls, method, keys, error):
    m = map_cls([1], _values(map_cls, [1]))
    before = m.items()
    with pytest.raises(error):
        getattr(m, method)(keys)
    assert m.items() == before


def test_get_many_defaults_and_plain_integer_keyerror(map_cls):
    a, default = _values(map_cls, [1, 2])
    m = map_cls([7], [a])
    for keys, missing in [
        (np.array([7, 99, 98], dtype=np.uint32), 99),
        (np.array([7, -1], dtype=np.int64), -1),
        (np.array([7, 2**32], dtype=">u8"), 2**32),
        (np.array([2**64 - 1], dtype=np.uint64), 2**64 - 1),
    ]:
        with pytest.raises(KeyError) as exc:
            m.get_many(keys)
        assert exc.value.args == (missing,)
        assert type(exc.value.args[0]) is int
    keys = [7, -1, 2**32, 99, 7]
    _assert_values(m, m.get_many(keys, default), [a, default, default, default, a])
    present = m.contains_many(keys)
    assert present.dtype == np.dtype(bool)
    np.testing.assert_array_equal(present, [True, False, False, False, True])
    if map_cls is not pc.SparseMap:
        _assert_values(m, m.get_many(keys, default=default),
                       [a, default, default, default, a])


def test_length_mismatch_is_atomic(map_cls):
    m = map_cls([1], _values(map_cls, [1]))
    before = m.items()
    for values in (_values(map_cls, [2]), _values(map_cls, [2, 3, 4])):
        with pytest.raises(ValueError):
            m.set_many([1, 8], values)
        assert m.items() == before
        with pytest.raises(ValueError):
            map_cls([1, 8], values)


def test_dense_order_and_detached_outputs(map_cls):
    a, b, c, d = _values(map_cls, [1, 2, 3, 4])
    m = map_cls([9, 2, 7, 4], [a, b, c, d])
    del m[2]
    assert list(m) == [9, 4, 7]
    assert m.items() == [(9, a), (4, d), (7, c)]
    m[2] = b
    assert m.popitem() == (2, b)
    keys = m.keys()
    values = m.values()
    items = m.items()
    assert keys.dtype == np.dtype(np.uint32)
    np.testing.assert_array_equal(keys, [9, 4, 7])
    _assert_values(m, values, [a, d, c])
    keys[0] = 100
    values[0] = b
    items[0] = (100, b)
    assert m.items() == [(9, a), (4, d), (7, c)]
    assert m.sort_keys() is None
    assert m.items() == [(4, d), (7, c), (9, a)]
    m[1] = b  # Copies of the arrays do not pin the map.
    assert keys.tolist() == [100, 4, 7]


def test_update_mapping_pairs_self_and_partial_failure(map_cls):
    a, b, c = _values(map_cls, [1, 2, 3])
    m = map_cls()
    assert m.update({9: a, 2: b}) is None
    m.update(iter([(9, c), (7, a)]))
    assert m.items() == [(9, c), (2, b), (7, a)]
    m.update(m)
    assert m.items() == [(9, c), (2, b), (7, a)]
    with pytest.raises(OverflowError):
        m.update([(8, b), (-1, a), (6, c)])
    assert m.items() == [(9, c), (2, b), (7, a), (8, b)]


def test_pages_clear_reuse_and_shrink(map_cls):
    m = map_cls()
    assert m.allocated_pages == 0
    m.set_many([0, 1023, 1024, 2048], _values(map_cls, [1, 2, 3, 4]))
    assert m.allocated_pages == 3
    del m[2048]
    assert m.allocated_pages == 3
    m.clear()
    assert not m and m.allocated_pages == 3
    m[1024] = _values(map_cls, [1])[0]
    assert m.allocated_pages == 3
    m.clear()
    m.shrink_to_fit()
    assert m.allocated_pages == 0
    m.reserve(20)
    m[3] = _values(map_cls, [1])[0]
    assert list(m) == [3]


@pytest.mark.parametrize("method", ["copy", "shallow", "deep", "pickle"])
def test_copy_and_pickle_roundtrip(map_cls, method):
    values = _values(map_cls, [1, 2, 3])
    m = map_cls([8, 1, 4], values)
    if method == "copy":
        clone = m.copy()
    elif method == "shallow":
        clone = copy.copy(m)
    elif method == "deep":
        clone = copy.deepcopy(m)
    else:
        clone = pickle.loads(pickle.dumps(m))
    assert type(clone) is map_cls
    assert clone is not m
    assert clone == m
    assert clone.items() == m.items()
    del clone[1]
    clone[8] = _values(map_cls, [2])[0]
    assert m.items() == list(zip([8, 1, 4], values))


def test_equality_hash_and_repr(map_cls):
    a, b = _values(map_cls, [1, 2])
    assert map_cls([8, 1], [a, b]) == map_cls([1, 8], [b, a])
    assert map_cls([8], [a]) != map_cls([8], [b])
    assert map_cls([8], [a]) != map_cls([1], [a])
    assert map_cls([8], [a]) != map_cls([8, 1], [a, b])
    with pytest.raises(TypeError):
        hash(map_cls())
    assert repr(map_cls()) == f"{map_cls.__name__}({{}})"
    m = map_cls(list(range(21)), _values(map_cls, range(21)))
    shown = ", ".join(f"{key}: {value!r}" for key, value in m.items()[:20])
    assert repr(m) == f"{map_cls.__name__}({{{shown}, ...}})"
    del m[20]
    assert "..." not in repr(m)


@pytest.mark.parametrize("change", ["insert", "remove", "clear", "sort", "batch_set", "batch_remove"])
def test_iterator_detects_structural_change(map_cls, change):
    values = _values(map_cls, [1, 2, 3])
    m = map_cls([9, 2, 7], values)
    it = iter(m)
    assert iter(it) is it
    assert next(it) == 9
    if change == "insert":
        m[5] = values[0]
    elif change == "remove":
        del m[2]
    elif change == "clear":
        m.clear()
    elif change == "sort":
        m.sort_keys()
    elif change == "batch_set":
        m.set_many([2, 5], values[:2])
    else:
        m.discard_many([2])
    with pytest.raises(RuntimeError):
        next(it)


def test_iterator_survives_value_updates_and_stays_exhausted(map_cls):
    a, b = _values(map_cls, [1, 2])
    m = map_cls([9, 2, 7], [a, b, a])
    it = iter(m)
    assert next(it) == 9
    m[2] = a
    m.set_many([9, 7], [b, b])
    m.update({2: b})
    assert m.setdefault(2, a) == b
    assert m.discard_many([100]) == 0
    assert list(it) == [2, 7]
    m[5] = a
    for _ in range(2):
        with pytest.raises(StopIteration):
            next(it)


def test_iterator_detects_replacement_even_when_length_is_unchanged(map_cls):
    a, b = _values(map_cls, [1, 2])
    m = map_cls([9, 2, 7], [a, b, a])
    it = iter(m)
    assert next(it) == 9
    del m[2]
    m[5] = b
    assert len(m) == 3
    with pytest.raises(RuntimeError):
        next(it)


def test_empty_iterator_stays_exhausted(map_cls):
    m = map_cls()
    it = iter(m)
    with pytest.raises(StopIteration):
        next(it)
    m[1] = _values(map_cls, [1])[0]
    with pytest.raises(StopIteration):
        next(it)


def test_typed_scalar_broadcast(typed_map_cls):
    a, b = _values(typed_map_cls, [1, 2])
    m = typed_map_cls([9, 2], a)
    _assert_values(m, m.values(), [a, a])
    assert m.set_many([9, 7, 7], np.array(b, dtype=typed_map_cls.dtype)) == 1
    _assert_values(m, m.get_many([9, 2, 7]), [b, a, b])
    assert m.set_many([], b) == 0


@pytest.mark.parametrize("dtype", [np.bool_, np.int8, np.uint64, np.float32, np.float64])
@pytest.mark.parametrize("cls", [pc.SparseMapF32, pc.SparseMapF64])
def test_float_maps_accept_numeric_arrays(cls, dtype):
    numbers = [1.25, -2.5, 3.75] if np.dtype(dtype).kind == "f" else [1, 0, 1]
    source = np.array(numbers, dtype=dtype)
    m = cls([1, 2, 3], source)
    _assert_values(m, m.values(), source.astype(cls.dtype))
    source[0] = 0
    assert m[1] == numbers[0]


@pytest.mark.parametrize("dtype", [np.bool_, np.int8, np.int64, np.uint8, np.uint64])
@pytest.mark.parametrize("cls", [pc.SparseMapI32, pc.SparseMapI64, pc.SparseMapU32, pc.SparseMapU64])
def test_integer_maps_accept_integer_and_bool_arrays(cls, dtype):
    source = np.array([1, 0, 1], dtype=dtype)
    m = cls([1, 2, 3], source)
    _assert_values(m, m.values(), [1, 0, 1])


def test_typed_value_byte_order_and_strides(typed_map_cls):
    if typed_map_cls is pc.SparseMapBool:
        dtypes = ["bool"]
    else:
        dtypes = ["<i2", ">i2", "<u8", ">u8"]
        if np.dtype(typed_map_cls.dtype).kind == "f":
            dtypes += ["<f8", ">f8"]
    for dtype in dtypes:
        numbers = ([True, False, False, True, True, False] if dtype == "bool"
                   else [1, 99, 2, 99, 3, 99])
        source = np.array(numbers, dtype=dtype)[::2]
        m = typed_map_cls([1, 2, 3], source)
        _assert_values(m, m.values(), source.astype(typed_map_cls.dtype))
        assert m.set_many([1, 2, 3], source[::-1]) == 0
        _assert_values(m, m.values(), source[::-1].astype(typed_map_cls.dtype))


@pytest.mark.parametrize("bad_kind", ["object", "complex128", "U2"])
def test_typed_value_kind_validation_is_atomic(typed_map_cls, bad_kind):
    bad = np.array([1, 2], dtype=bad_kind)
    m = typed_map_cls([1], _values(typed_map_cls, [1]))
    before = m.items()
    with pytest.raises(TypeError):
        m.set_many([1, 8], bad)
    assert m.items() == before
    with pytest.raises(TypeError):
        typed_map_cls([1, 8], bad)


@pytest.mark.parametrize("cls", [pc.SparseMapI32, pc.SparseMapI64, pc.SparseMapU32, pc.SparseMapU64])
def test_integer_maps_reject_float_arrays_and_broadcasts(cls):
    m = cls([1], [7])
    for values in ([1.0, 2.0], np.array([1.0, 2.0], dtype=np.float32), 2.0):
        with pytest.raises(TypeError):
            m.set_many([1, 8], values)
        assert m.items() == [(1, 7)]


@pytest.mark.parametrize("dtype", [np.int8, np.uint64, np.float32, np.float64])
def test_bool_map_rejects_non_bool_arrays_and_broadcasts(dtype):
    m = pc.SparseMapBool([1], [True])
    for values in (np.array([0, 1], dtype=dtype), np.array(0, dtype=dtype)):
        with pytest.raises(TypeError):
            m.set_many([1, 8], values)
        assert m.items() == [(1, True)]


def test_bool_map_normalizes_noncanonical_bytes():
    values = np.zeros(3, dtype=np.bool_)
    values.view(np.uint8)[:] = [0, 2, 255]
    m = pc.SparseMapBool()
    assert m.set_many([0, 1, 2], values) == 3
    assert m[0] is False
    assert m[1] is True
    assert m[2] is True
    np.testing.assert_array_equal(m.values(), [False, True, True])


@pytest.mark.parametrize("cls", [pc.SparseMapF32, pc.SparseMapF64])
def test_float_nonfinite_values(cls):
    values = np.array([np.nan, np.inf, -np.inf], dtype=cls.dtype)
    m = cls([1, 2, 3], values)
    _assert_values(m, m.get_many([1, 2, 3]), values)


@pytest.mark.parametrize(
    "cls, bad_values",
    [(pc.SparseMapI32, np.array([0, 2**31], dtype=np.int64)),
     (pc.SparseMapI32, np.array([0, -(2**31) - 1], dtype=np.int64)),
     (pc.SparseMapI64, np.array([0, 2**63], dtype=np.uint64)),
     (pc.SparseMapU32, np.array([0, -1], dtype=np.int64)),
     (pc.SparseMapU32, np.array([0, 2**32], dtype=np.uint64)),
     (pc.SparseMapU64, np.array([0, -1], dtype=np.int64))],
)
def test_integer_value_overflow_is_atomic(cls, bad_values):
    m = cls([1], [7])
    with pytest.raises(OverflowError):
        m.set_many([1, 8], bad_values)
    assert m.items() == [(1, 7)]
    with pytest.raises(OverflowError):
        cls([1, 8], bad_values)
    with pytest.raises(OverflowError):
        m.set_many([1, 8], int(bad_values[-1]))
    assert m.items() == [(1, 7)]


@pytest.mark.parametrize("cls", [pc.SparseMapI32, pc.SparseMapI64, pc.SparseMapU32, pc.SparseMapU64])
def test_integer_value_boundaries(cls):
    dtype = np.dtype(cls.dtype)
    limits = np.iinfo(dtype)
    values = np.array([limits.min, limits.max], dtype=dtype)
    m = cls([1, 2], values)
    _assert_values(m, m.values(), values)
    assert m[1] == int(limits.min)
    assert m[2] == int(limits.max)


def test_typed_two_dimensional_values_are_rejected(typed_map_cls):
    m = typed_map_cls([1], _values(typed_map_cls, [1]))
    before = m.items()
    with pytest.raises(ValueError):
        m.set_many([1, 2], np.zeros((1, 2), dtype=typed_map_cls.dtype))
    assert m.items() == before


def test_object_values_identity_defaults_and_copy_depth():
    nested = {"items": [1, 2]}
    marker = object()
    m = pc.SparseMap([9, 2, 7], [nested, None, marker])
    assert m[9] is nested and m[2] is None and m[7] is marker
    assert m.values()[0] is nested
    assert m.items()[0][1] is nested
    assert m.get_many([9, 99], None) == [nested, None]
    assert m.get_many([99], marker)[0] is marker
    assert m.setdefault(8) is None
    assert m.pop(99, None) is None
    for clone in (m.copy(), copy.copy(m)):
        assert clone[9] is nested
    deep = copy.deepcopy(m)
    assert deep[9] == nested and deep[9] is not nested
    deep[9]["items"].append(3)
    assert nested == {"items": [1, 2]}
    restored = pickle.loads(pickle.dumps(pc.SparseMap([9, 2], [nested, None])))
    assert restored[9] == nested and restored[2] is None
    assert restored[9] is not nested


def test_object_batch_iterable_failure_is_atomic():
    marker = object()
    m = pc.SparseMap([1], [marker])

    def broken():
        yield "replacement"
        raise RuntimeError("broken values")

    with pytest.raises(RuntimeError, match="broken values"):
        m.set_many([1, 2], broken())
    assert m.items() == [(1, marker)]
    assert m.set_many([1, 2], iter([None, marker])) == 1
    assert m[1] is None and m[2] is marker


def test_object_equality_uses_identity_before_equality():
    nan = float("nan")
    assert pc.SparseMap([1], [nan]) == pc.SparseMap([1], [nan])
    assert pc.SparseMap([1], [[1, 2]]) == pc.SparseMap([1], [[1, 2]])
    assert pc.SparseMap([1], [[1, 2]]) != pc.SparseMap([1], [[2, 1]])


def test_object_self_cycle_roundtrip(object_roundtrip):
    m = pc.SparseMap()
    m[1] = m
    restored = object_roundtrip(m)
    assert type(restored) is pc.SparseMap
    assert list(restored) == [1]
    assert restored[1] is restored


def test_object_two_map_cycle_roundtrip(object_roundtrip):
    first, second = pc.SparseMap(), pc.SparseMap()
    first[1], second[2] = second, first
    restored = object_roundtrip(first)
    assert restored is not first
    assert type(restored[1]) is pc.SparseMap
    assert restored[1] is not second
    assert list(restored) == [1]
    assert list(restored[1]) == [2]
    assert restored[1][2] is restored


def test_object_self_cycle_repr():
    m = pc.SparseMap()
    m[1] = m
    assert repr(m) == "SparseMap({1: SparseMap(...)})"


def test_object_two_map_cycle_repr():
    first, second = pc.SparseMap(), pc.SparseMap()
    first[1], second[2] = second, first
    assert repr(first) == "SparseMap({1: SparseMap({2: SparseMap(...)})})"


def test_plain_object_map_roundtrip_preserves_order_and_values(object_roundtrip):
    shared = ["value"]
    m = pc.SparseMap([9, 2, 7, 4], [shared, "removed", None, shared])
    del m[2]
    restored = object_roundtrip(m)
    assert type(restored) is pc.SparseMap
    assert restored is not m
    assert restored.items() == [(9, ["value"]), (4, ["value"]), (7, None)]
    assert restored[9] is restored[4]
    assert restored[9] is not shared


# Subclasses are module-level so pickle can find them by name.
_SUBCLASS_BASES = (pc.SparseSet, pc.SparseMap, pc.SparseMapF64, pc.SparseMapI32, pc.SparseMapBool)


def _labelled(base):
    def __init__(self, label, *args):
        base.__init__(self, *args)
        self.label = label

    cls = type(f"Labelled{base.__name__}", (base,), {"__init__": __init__})
    cls.__module__ = __name__
    globals()[cls.__name__] = cls
    return cls


def _slotted(base):
    cls = type(f"Slotted{base.__name__}", (base,), {"__slots__": ("tag",)})
    cls.__module__ = __name__
    globals()[cls.__name__] = cls
    return cls


_LABELLED = {base: _labelled(base) for base in _SUBCLASS_BASES}
_SLOTTED = {base: _slotted(base) for base in _SUBCLASS_BASES}


def _contents(base):
    if base is pc.SparseSet:
        return ([9, 2, 5],)
    if base is pc.SparseMapBool:
        return ([9, 2, 5], [True, False, True])
    if base is pc.SparseMap:
        return ([9, 2, 5], [[1], None, "x"])
    return ([9, 2, 5], [3, 1, 4])


def _snapshot(obj):
    return list(obj) if isinstance(obj, pc.SparseSet) else obj.items()


_COPIERS = {
    **{f"pickle-{p}": (lambda o, p=p: pickle.loads(pickle.dumps(o, protocol=p)))
       for p in range(2, pickle.HIGHEST_PROTOCOL + 1)},
    "copy": copy.copy,
    "deepcopy": copy.deepcopy,
}


@pytest.mark.parametrize("copier", _COPIERS.values(), ids=_COPIERS.keys())
@pytest.mark.parametrize("base", _SUBCLASS_BASES, ids=lambda b: b.__name__)
def test_subclass_with_own_init_roundtrips(base, copier):
    cls = _LABELLED[base]
    obj = cls("tag", *_contents(base))
    restored = copier(obj)
    assert type(restored) is cls
    assert restored.label == "tag"
    assert _snapshot(restored) == _snapshot(obj)
    assert isinstance(restored, base)


@pytest.mark.skipif(sys.version_info < (3, 11), reason="object.__getstate__ covers slots from 3.11")
@pytest.mark.parametrize("copier", _COPIERS.values(), ids=_COPIERS.keys())
@pytest.mark.parametrize("base", _SUBCLASS_BASES, ids=lambda b: b.__name__)
def test_subclass_slots_roundtrip(base, copier):
    cls = _SLOTTED[base]
    obj = cls(*_contents(base))
    obj.tag = 7
    restored = copier(obj)
    assert type(restored) is cls
    assert restored.tag == 7
    assert _snapshot(restored) == _snapshot(obj)


@pytest.mark.parametrize("base", _SUBCLASS_BASES, ids=lambda b: b.__name__)
def test_subclass_repr_and_copy_method(base):
    obj = _LABELLED[base]("tag", *_contents(base))
    assert repr(obj).startswith(f"Labelled{base.__name__}(")
    plain = obj.copy()
    assert type(plain) is base
    assert _snapshot(plain) == _snapshot(obj)


def test_base_pickle_has_no_extra_state():
    m = pc.SparseMapF64([1], [2.0])
    assert m.__reduce__()[2] is None
    assert pc.SparseMap([1], [2]).__reduce__()[2] is None


def test_batch_methods_bypass_overridden_setitem():
    calls = []

    class Logged(pc.SparseMapF64):
        def __setitem__(self, key, value):
            calls.append(key)
            super().__setitem__(key, value)

    m = Logged()
    m[1] = 1.0
    m.set_many([2, 3], 2.0)
    m.update({4: 4.0})
    assert calls == [1]
    assert m.items() == [(1, 1.0), (2, 2.0), (3, 2.0), (4, 4.0)]


def test_object_subclass_self_cycle_pickles():
    cls = _LABELLED[pc.SparseMap]
    m = cls("loop")
    m[1] = m
    restored = pickle.loads(pickle.dumps(m))
    assert type(restored) is cls
    assert restored.label == "loop"
    assert restored[1] is restored


def test_rebuild_refuses_unrelated_class():
    with pytest.raises(TypeError):
        pc._core._rebuild(pc.SparseSet, pc.SparseMap)


@pytest.mark.parametrize(
    "cycle",
    [
        "self",
        pytest.param(
            "keys_view",
            marks=pytest.mark.xfail(
                strict=True,
                reason="NumPy arrays are not GC-tracked, so a cycle through a view leaks",
            ),
        ),
        "iterator",
    ],
)
def test_object_reference_cycles_are_collected(cycle):
    finalized = []

    class Finalizer:
        def __del__(self):
            finalized.append(True)

    m = pc.SparseMap([1, 2], [None, Finalizer()])
    if cycle == "self":
        m[1] = m
    elif cycle == "keys_view":
        m[1] = m.keys_view()
    else:
        m[1] = iter(m)
    del m
    gc.collect()
    assert finalized == [True]


def test_object_optional_default_argument_count():
    m = pc.SparseMap()
    with pytest.raises(TypeError):
        m.get_many([1], None, None)
    with pytest.raises(TypeError):
        m.pop(1, None, None)


def test_dtype_attribute_and_factory(map_cls):
    expected = {
        "SparseMap": "object", "SparseMapF64": "float64", "SparseMapF32": "float32",
        "SparseMapI64": "int64", "SparseMapI32": "int32", "SparseMapU64": "uint64",
        "SparseMapU32": "uint32", "SparseMapBool": "bool",
    }[map_cls.__name__]
    assert map_cls.dtype == expected
    specs = [expected, np.dtype(expected), np.dtype(expected).type]
    if map_cls is pc.SparseMap:
        specs += [None, object]
    for spec in specs:
        assert type(pc.sparse_map(spec)) is map_cls
        values = _values(map_cls, [1, 2])
        m = pc.sparse_map(spec, [9, 2], values)
        assert type(m) is map_cls
        assert m.items() == list(zip([9, 2], values))
        assert m.dtype == expected


@pytest.mark.parametrize("dtype", ["int8", "uint16", "float16", "complex128", "U4", "datetime64[ns]"])
def test_factory_rejects_unsupported_dtypes(dtype):
    with pytest.raises(TypeError):
        pc.sparse_map(dtype)


def test_seeded_map_model(map_cls):
    rng = random.Random(0xD1C7)
    m = map_cls()
    model = {}
    dense = []
    sentinel = object()

    def insert(key, value):
        if key not in model:
            dense.append(key)
        model[key] = value

    def remove(key):
        if key in model:
            del model[key]
            pos = dense.index(key)
            dense[pos] = dense[-1]
            dense.pop()

    for step in range(3000):
        op = rng.randrange(13)
        key = rng.randrange(96) + rng.choice([0, 1024, 4096])
        value = _values(map_cls, [rng.randrange(100)])[0]
        batch = [key, rng.randrange(96), key, rng.randrange(96) + 1024]
        if op == 0:
            m[key] = value
            insert(key, value)
        elif op == 1:
            if key in model:
                del m[key]
                remove(key)
            else:
                with pytest.raises(KeyError):
                    del m[key]
        elif op == 2:
            expected = model.get(key, sentinel)
            assert m.pop(key, sentinel) == expected
            remove(key)
        elif op == 3:
            values = _values(map_cls, [rng.randrange(100) for _ in batch])
            expected = len(set(batch) - model.keys())
            assert m.set_many(np.array(batch, dtype=np.uint32), values) == expected
            for k, v in zip(batch, values):
                insert(k, v)
        elif op == 4:
            batch += [-1, 2**32]
            expected = len(model.keys() & set(batch))
            assert m.discard_many(batch) == expected
            for item in batch:
                remove(item)
        elif op == 5:
            np.testing.assert_array_equal(m.contains_many(batch), [k in model for k in batch])
            _assert_values(m, m.get_many(batch, value), [model.get(k, value) for k in batch])
        elif op == 6:
            if dense:
                last = dense[-1]
                assert m.popitem() == (last, model[last])
                remove(last)
            else:
                with pytest.raises(KeyError):
                    m.popitem()
        elif op == 7:
            m.sort_keys()
            dense.sort()
        elif op == 8:
            assert m.setdefault(key, value) == model.get(key, value)
            if key not in model:
                insert(key, value)
        elif op == 9:
            pairs = [(batch[0], value), (batch[1], value)]
            m.update(dict(pairs) if step % 2 else iter(pairs))
            for k, v in pairs:
                insert(k, v)
        elif op == 10 and rng.randrange(8) == 0:
            m.clear()
            model.clear()
            dense.clear()
        elif op == 11:
            assert (key in m) == (key in model)
            assert m.get(key, sentinel) == model.get(key, sentinel)
            if key in model:
                assert m[key] == model[key]
            else:
                with pytest.raises(KeyError):
                    m[key]
        else:
            m.reserve(4)
        assert len(m) == len(model), step
        assert list(m) == dense, step
        assert m.items() == [(k, model[k]) for k in dense], step
        if step % 101 == 0:
            np.testing.assert_array_equal(m.keys(), dense)
            _assert_values(m, m.values(), [model[k] for k in dense])
