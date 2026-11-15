"""Ordered direct-map contracts, conversions, ownership and model tests."""

import copy
import gc
import operator
import pickle
import random
import sys
import weakref

import numpy as np
import pytest

import paged_collections as pc


TYPED_CLASSES = (
    pc.DirectMapF64, pc.DirectMapF32, pc.DirectMapI64, pc.DirectMapI32,
    pc.DirectMapU64, pc.DirectMapU32, pc.DirectMapBool,
)
CLASSES = (pc.DirectMap, *TYPED_CLASSES)
LIMIT = 2**32


@pytest.fixture(params=CLASSES, ids=lambda cls: cls.__name__)
def direct_cls(request):
    return request.param


@pytest.fixture(params=TYPED_CLASSES, ids=lambda cls: cls.__name__)
def typed_direct_cls(request):
    return request.param


def _values(cls, numbers):
    if cls.dtype == "bool":
        return [bool(n % 2) for n in numbers]
    if cls.dtype.startswith("float"):
        return [float(n) for n in numbers]
    return list(numbers)


def _assert_values(m, actual, expected):
    if isinstance(m, pc.DirectMap):
        assert isinstance(actual, list)
        assert actual == expected
    else:
        assert isinstance(actual, np.ndarray)
        assert actual.dtype == np.dtype(m.dtype)
        np.testing.assert_array_equal(actual, expected)


def test_dict_methods(direct_cls):
    a, b, c = _values(direct_cls, [1, 2, 3])
    m = direct_cls()
    assert not m and len(m) == 0
    assert m.get(4) is None
    sentinel = object()
    assert m.get(4, sentinel) is sentinel
    assert m.pop(4, sentinel) is sentinel
    with pytest.raises(KeyError):
        m.pop(4)
    m[9] = a
    m[2] = b
    assert m[9] == a and 9 in m
    assert m.setdefault(2, c) == b
    assert m.setdefault(7, c) == c
    assert m.pop(9) == a
    del m[2]
    assert m.items() == [(7, c)]
    assert m.update({9: a, 2: b}) is None
    m.update(iter([(7, b), (1, c)]))
    m.update(m)
    assert m.items() == [(1, c), (2, b), (7, b), (9, a)]
    with pytest.raises(OverflowError):
        m.update([(8, a), (-1, b), (6, c)])
    assert m[8] == a and 6 not in m
    with pytest.raises(TypeError):
        m.pop(1, None, None)
    assert direct_cls([7, 2], [a, b]) == direct_cls([2, 7], [b, a])
    assert direct_cls([7], [a]) != direct_cls([2], [a])
    assert direct_cls([7], [a]) != direct_cls([2, 7], [a, a])
    with pytest.raises(TypeError):
        hash(m)
    for kwargs in ({"keys": [1]}, {"values": [a]}):
        with pytest.raises(TypeError, match="give both keys and values"):
            direct_cls(**kwargs)


@pytest.mark.parametrize(
    "key, error",
    [(-1, OverflowError), (LIMIT, OverflowError), (2**100, OverflowError),
     (1.0, TypeError), ("1", TypeError), (None, TypeError), ([], TypeError)],
)
def test_scalar_key_validation(direct_cls, key, error):
    a = _values(direct_cls, [1])[0]
    m = direct_cls([1], [a])
    sentinel = object()
    assert key not in m
    assert m.get(key) is None
    assert m.pop(key, sentinel) is sentinel
    for operation in (lambda: m[key], lambda: m.__delitem__(key), lambda: m.pop(key)):
        with pytest.raises(KeyError):
            operation()
    with pytest.raises(error):
        m[key] = a
    with pytest.raises(error):
        m.setdefault(key, a)
    assert m.items() == [(1, a)]


def test_key_limits_and_cursor(direct_cls):
    a, b = _values(direct_cls, [1, 2])
    m = direct_cls([LIMIT - 1, 0], [b, a])
    assert m[0] == a and m[np.uint64(LIMIT - 1)] == b
    it = iter(m)
    assert operator.length_hint(it) == 2
    assert next(it) == 0
    assert operator.length_hint(it) == 1
    assert next(it) == LIMIT - 1
    assert operator.length_hint(it) == 0
    with pytest.raises(StopIteration):
        next(it)
    assert m.items_range() == [(0, a), (LIMIT - 1, b)]
    assert m.items_range(LIMIT - 1, LIMIT) == [(LIMIT - 1, b)]
    assert m.next_occupant(LIMIT - 1) == (LIMIT - 1, b)
    assert m.prev_occupant(LIMIT) == (LIMIT - 1, b)
    assert m.discard_many([0, LIMIT - 1, LIMIT]) == 2


def test_order_and_detached_results(direct_cls):
    a, b, c, d = _values(direct_cls, [1, 2, 3, 4])
    m = direct_cls([9, 2, 7, 4], [a, b, c, d])
    del m[2]
    m[1] = b
    expected = [(1, b), (4, d), (7, c), (9, a)]
    assert list(m) == [k for k, _ in expected]
    assert m.items() == expected
    keys, values, items = m.keys(), m.values(), m.items()
    assert keys.dtype == np.dtype(np.uint32)
    np.testing.assert_array_equal(keys, [1, 4, 7, 9])
    _assert_values(m, values, [b, d, c, a])
    keys[0], values[0], items[0] = 100, a, (100, a)
    assert m.items() == expected
    range_keys, range_values = m.keys_range(4, 9), m.values_range(4, 9)
    assert range_keys.dtype == np.dtype(np.uint32)
    np.testing.assert_array_equal(range_keys, [4, 7])
    _assert_values(m, range_values, [d, c])
    range_keys[0], range_values[0] = 200, a
    m[3] = c
    assert m[4] == d and 3 in m
    for name in ("reserve", "sort_keys", "keys_view", "values_view"):
        assert not hasattr(m, name)
    name = direct_cls.__name__
    assert repr(direct_cls([9, 1], [a, b])) == f"{name}({{1: {b!r}, 9: {a!r}}})"
    many = direct_cls(list(reversed(range(21))), _values(direct_cls, reversed(range(21))))
    shown = ", ".join(f"{k}: {v!r}" for k, v in many.items()[:20])
    assert repr(many) == f"{name}({{{shown}, ...}})"


@pytest.mark.parametrize(
    "change", ["insert", "delete", "clear", "pop", "popitem", "pop_first", "pop_last",
               "set_many", "discard_many", "setdefault", "update", "replace_key"],
)
def test_iterator_invalidation(direct_cls, change):
    a, b = _values(direct_cls, [1, 2])
    m = direct_cls([9, 2, 7], [a, b, a])
    it = iter(m)
    assert iter(it) is it
    assert next(it) == 2
    if change == "insert":
        m[5] = b
    elif change == "delete":
        del m[7]
    elif change == "clear":
        m.clear()
    elif change == "pop":
        m.pop(7)
    elif change in ("popitem", "pop_first", "pop_last"):
        getattr(m, change)()
    elif change == "set_many":
        m.set_many([2, 5], [a, b])
    elif change == "discard_many":
        m.discard_many([7, 7])
    elif change == "setdefault":
        m.setdefault(5, b)
    elif change == "update":
        m.update({5: b})
    else:
        del m[7]
        m[5] = b
        assert len(m) == 3
    with pytest.raises(RuntimeError, match=f"^{direct_cls.__name__} changed during iteration$"):
        next(it)


def test_iterator_overwrites_and_exhaustion(direct_cls):
    a, b = _values(direct_cls, [1, 2])
    m = direct_cls([9, 2, 7], [a, b, a])
    it = iter(m)
    assert next(it) == 2
    m[7] = b
    m.set_many([2, 9], [a, b])
    m.update({7: a})
    m.setdefault(2, b)
    m.discard_many([-1, LIMIT])
    m.shrink_to_fit()
    assert operator.length_hint(it) == 2
    assert list(it) == [7, 9]
    m.clear()
    m[4] = a
    assert list(it) == [] and operator.length_hint(it) == 0
    empty = iter(direct_cls())
    assert list(empty) == []
    m = direct_cls()
    empty = iter(m)
    with pytest.raises(StopIteration):
        next(empty)
    m[0] = a
    assert list(empty) == []
    # Yielding the last key alone has not yet observed exhaustion.
    it = iter(m)
    assert next(it) == 0
    m[1] = b
    with pytest.raises(RuntimeError):
        next(it)


def test_empty_queries(direct_cls):
    m = direct_cls()
    for method in ("first_item", "last_item", "pop_first", "pop_last", "popitem"):
        with pytest.raises(KeyError) as error:
            getattr(m, method)()
        assert error.value.args == (f"{method}(): {direct_cls.__name__} is empty",)
    for index in (-2**200, -1, 0, 1024, LIMIT, 2**200):
        assert m.next_occupant(index) is None
        assert m.prev_occupant(index) is None
    assert m.items_range() == []
    assert m.keys_range().dtype == np.dtype(np.uint32)
    _assert_values(m, m.values_range(), [])


def test_ordered_queries_across_pages(direct_cls):
    keys = [0, 1023, 1024, 2047, 2048, 70_000]
    values = _values(direct_cls, range(len(keys)))
    oracle = dict(zip(keys, values))
    m = direct_cls(keys[::-1], values[::-1])
    assert m.first_item() == (keys[0], values[0])
    assert m.last_item() == (keys[-1], values[-1])
    for index in [-2**200, -1, 0, 1, 1022, 1023, 1024, 1025, 2047, 2048,
                  2049, 69_999, 70_000, 70_001, LIMIT - 1, LIMIT, 2**200]:
        next_key = next((k for k in keys if k >= index), None)
        prev_key = next((k for k in reversed(keys) if k <= index), None)
        assert m.next_occupant(index) == (None if next_key is None else (next_key, oracle[next_key]))
        assert m.prev_occupant(index) == (None if prev_key is None else (prev_key, oracle[prev_key]))
    for method in ("popitem", "pop_last"):
        clone = m.copy()
        assert [getattr(clone, method)() for _ in keys] == list(zip(keys, values))[::-1]
    assert [m.pop_first() for _ in keys] == list(zip(keys, values))


@pytest.mark.parametrize(
    "start, stop",
    [(None, None), (None, 1024), (1024, None), (1023, 2048), (1024, 1025),
     (1024, 1024), (2048, 1024), (-1, 1024), (-2**200, 2**200),
     (None, -1), (-2**200, -1), (LIMIT, None), (2**200, 2**201),
     (LIMIT - 1, LIMIT), (0, 1), (70_000, 70_001)],
)
def test_ranges(direct_cls, start, stop):
    keys = [0, 1023, 1024, 2047, 2048, 70_000, LIMIT - 1]
    values = _values(direct_cls, range(len(keys)))
    m = direct_cls(keys[::-1], values[::-1])
    expected = [(k, v) for k, v in zip(keys, values)
                if (start is None or k >= start) and (stop is None or k < stop)]
    assert m.items_range(start, stop) == expected
    result = m.keys_range(start=start, stop=stop)
    assert result.dtype == np.dtype(np.uint32)
    np.testing.assert_array_equal(result, [k for k, _ in expected])
    _assert_values(m, m.values_range(start, stop), [v for _, v in expected])


@pytest.mark.parametrize("bad", [1.0, "1", [], object()])
def test_query_types(direct_cls, bad):
    m = direct_cls()
    for method in ("next_occupant", "prev_occupant"):
        with pytest.raises(TypeError):
            getattr(m, method)(bad)
    for method in ("items_range", "keys_range", "values_range"):
        for bounds in ((bad, None), (None, bad), (LIMIT, bad)):
            with pytest.raises(TypeError):
                getattr(m, method)(*bounds)
    for method in ("next_occupant", "prev_occupant"):
        with pytest.raises(TypeError):
            getattr(m, method)(None)


def test_batches(direct_cls):
    a, b, c, d = _values(direct_cls, [1, 2, 3, 4])
    m = direct_cls([9, 1], [a, b])
    assert m.set_many([9, 4, 9, 2], [a, b, c, d]) == 2
    assert m.items() == [(1, b), (2, d), (4, b), (9, c)]
    assert m.set_many([4, 9, 4], [a, b, c]) == 0
    before = m.items()
    for keys in ([4, -1, 8], [4, LIMIT, 8]):
        with pytest.raises(OverflowError):
            m.set_many(keys, [a, b, c])
        assert m.items() == before
    with pytest.raises(ValueError, match="got 1 values for 2 keys"):
        m.set_many([4, 8], [a])
    assert m.items() == before
    assert m.set_many([], []) == 0
    _assert_values(m, m.get_many([9, 1, 9]), [b, b, b])
    _assert_values(m, m.get_many([9, -1, LIMIT, 1], d), [b, d, d, b])
    with pytest.raises(KeyError) as error:
        m.get_many([9, LIMIT, 1])
    assert error.value.args == (LIMIT,)
    assert type(error.value.args[0]) is int
    present = m.contains_many([9, -1, LIMIT, 1, 8])
    assert present.dtype == np.dtype(bool)
    np.testing.assert_array_equal(present, [True, False, False, True, False])
    assert m.discard_many([4, -1, LIMIT, 4, 8]) == 1
    assert m.discard_many([]) == 0


def test_key_array_formats(direct_cls, key_batch):
    values = _values(direct_cls, [1, 2, 3, 4])
    m = direct_cls(key_batch, values)
    assert m.items() == [(1, values[1]), (7, values[3]), (9, values[2])]
    _assert_values(m, m.get_many(key_batch), [values[3], values[1], values[2], values[3]])


def test_pages_clear_and_shrink(direct_cls):
    m = direct_cls()
    assert m.allocated_pages == 0
    m.set_many([1023, 1024, 2047, 2048, 70_000], _values(direct_cls, range(5)))
    assert m.allocated_pages == 4
    del m[1024]
    del m[2047]
    assert m.allocated_pages == 4
    m.shrink_to_fit()
    assert m.allocated_pages == 3
    assert list(m) == [1023, 2048, 70_000]
    m.clear()
    assert not m and m.allocated_pages == 3
    m[2048] = _values(direct_cls, [1])[0]
    assert m.allocated_pages == 3
    m.clear()
    m.shrink_to_fit()
    assert m.allocated_pages == 0


def test_typed_conversion_and_broadcast(typed_direct_cls):
    a, b = _values(typed_direct_cls, [1, 2])
    m = typed_direct_cls([9, 2], a)
    _assert_values(m, m.values(), [a, a])
    assert m.set_many([2, 7], b) == 1
    assert m.items() == [(2, b), (7, b), (9, a)]
    before = m.items()
    bad = [1, 0] if m.dtype == "bool" else ["x", "y"]
    with pytest.raises(TypeError):
        m.set_many([9, 5], bad)
    assert m.items() == before
    with pytest.raises(ValueError):
        m.set_many([9, 5], [[a, b]])
    assert m.items() == before
    if m.dtype.startswith(("int", "uint")):
        with pytest.raises(TypeError):
            m.set_many([9, 5], [1.0, 2.0])
        with pytest.raises(TypeError):
            m[9] = 1.5
        assert m.items() == before
    if m.dtype.startswith("uint") or m.dtype == "int32":
        bad = -1 if m.dtype.startswith("uint") else 2**31
        with pytest.raises(OverflowError):
            m.set_many([9, 5], [a, bad])
        assert m.items() == before


def test_object_batch_failure_and_defaults():
    marker = object()
    m = pc.DirectMap([1], [marker])

    def broken():
        yield "replacement"
        raise RuntimeError("broken values")

    with pytest.raises(RuntimeError, match="broken values"):
        m.set_many([1, 2], broken())
    assert m[1] is marker and len(m) == 1
    assert m.setdefault(2) is None
    assert m.get_many([1, 3], None) == [marker, None]
    assert m.get_many([3], marker)[0] is marker
    with pytest.raises(TypeError):
        m.get_many([1], None, None)


@pytest.mark.parametrize("method", ["copy", "shallow", "deep", "pickle"])
def test_copy_roundtrip(direct_cls, method):
    values = _values(direct_cls, [1, 2, 3])
    m = direct_cls([9, 2, 7], values)
    clone = {"copy": lambda x: x.copy(), "shallow": copy.copy,
             "deep": copy.deepcopy, "pickle": lambda x: pickle.loads(pickle.dumps(x))}[method](m)
    assert type(clone) is direct_cls
    assert clone is not m and clone == m
    assert clone.items() == [(2, values[1]), (7, values[2]), (9, values[0])]
    del clone[2]
    assert len(m) == 3


# Module-level subclasses let pickle resolve them without calling their init.
def _subclasses(base):
    def __init__(self, label, *args):
        base.__init__(self, *args)
        self.label = label

    labelled = type(f"Labelled{base.__name__}", (base,), {"__init__": __init__})
    slotted = type(f"Slotted{base.__name__}", (base,), {"__slots__": ("tag",)})
    for cls in (labelled, slotted):
        cls.__module__ = __name__
        globals()[cls.__name__] = cls
    return labelled, slotted


SUBCLASSES = {base: _subclasses(base) for base in CLASSES}


def test_subclass_roundtrip(direct_cls, object_roundtrip):
    labelled, _ = SUBCLASSES[direct_cls]
    values = _values(direct_cls, [1, 2])
    m = labelled("tag", [9, 2], values)
    m.extra = ["state"]
    restored = object_roundtrip(m)
    assert type(restored) is labelled
    assert restored.label == "tag" and restored.extra == ["state"]
    assert restored.items() == m.items()
    assert repr(m).startswith(f"{labelled.__name__}(")
    assert type(m.copy()) is direct_cls
    assert direct_cls([9, 2], values) == m


@pytest.mark.skipif(sys.version_info < (3, 11), reason="object.__getstate__ covers slots from 3.11")
def test_subclass_slots(direct_cls, object_roundtrip):
    _, slotted = SUBCLASSES[direct_cls]
    m = slotted([9, 2], _values(direct_cls, [1, 2]))
    m.tag = [7]
    restored = object_roundtrip(m)
    assert type(restored) is slotted and restored.tag == [7]
    assert restored.items() == m.items()


def test_object_self_cycle_roundtrip(object_roundtrip):
    for cls in (pc.DirectMap, SUBCLASSES[pc.DirectMap][0]):
        m = cls() if cls is pc.DirectMap else cls("loop")
        m[0] = m
        restored = object_roundtrip(m)
        assert type(restored) is cls and restored is not m
        assert list(restored) == [0] and restored[0] is restored
        if cls is not pc.DirectMap:
            assert restored.label == "loop"
        assert repr(m) == f"{cls.__name__}({{0: {cls.__name__}(...)}})"


def test_object_copy_and_equality():
    value = ["shared"]
    m = pc.DirectMap([7], [value])
    assert m.copy()[7] is value
    assert copy.copy(m)[7] is value
    assert copy.deepcopy(m)[7] == value and copy.deepcopy(m)[7] is not value
    nan = float("nan")
    assert pc.DirectMap([1], [nan]) == pc.DirectMap([1], [nan])
    assert pc.DirectMap([1], [[1, 2]]) == pc.DirectMap([1], [[1, 2]])
    assert pc.DirectMap([1], [[1, 2]]) != pc.DirectMap([1], [[2, 1]])
    visits = []

    class Compared:
        def __init__(self, key):
            self.key = key

        def __eq__(self, other):
            visits.append(self.key)
            return self.key == other.key

    assert pc.DirectMap([9, 1], [Compared(9), Compared(1)]) == pc.DirectMap(
        [9, 1], [Compared(9), Compared(1)])
    assert visits == [1, 9]


@pytest.mark.parametrize("cycle", ["self", "iterator"])
def test_gc_collects_object_cycles(cycle):
    m = pc.DirectMap()
    ref = weakref.ref(m)
    m[0] = m if cycle == "self" else iter(m)
    del m
    gc.collect()
    assert ref() is None


def test_factory(direct_cls):
    dtype = np.dtype(direct_cls.dtype)
    specs = [str(dtype), dtype, dtype.type]
    if direct_cls is pc.DirectMap:
        specs += [None, object]
    for spec in specs:
        assert type(pc.direct_map(spec)) is direct_cls
        values = _values(direct_cls, [1, 2])
        m = pc.direct_map(spec, [9, 2], values)
        assert type(m) is direct_cls
        assert m.items() == [(2, values[1]), (9, values[0])]
    assert direct_cls.__name__ in pc.__all__


@pytest.mark.parametrize("dtype", ["int8", "uint16", "float16", "complex128", "U4"])
def test_factory_error(dtype):
    with pytest.raises(TypeError) as error:
        pc.direct_map(dtype)
    assert str(error.value) == (
        f"no direct map for dtype {np.dtype(dtype)}; supported: "
        "object, float64, float32, int64, int32, uint64, uint32, bool"
    )


def test_seeded_dict_oracle(direct_cls):
    rng = random.Random(0xD1EC7)
    m, oracle = direct_cls(), {}
    sentinel = object()
    for step in range(1500):
        op = rng.randrange(14)
        key = rng.choice([0, 1024, 2048, 70_000]) + rng.randrange(32)
        value = _values(direct_cls, [rng.randrange(100)])[0]
        batch = [key, rng.randrange(32), key, rng.randrange(32) + 1024]
        if op == 0:
            m[key] = value
            oracle[key] = value
        elif op == 1:
            if key in oracle:
                del m[key]
                del oracle[key]
            else:
                with pytest.raises(KeyError):
                    del m[key]
        elif op == 2:
            assert m.pop(key, sentinel) == oracle.pop(key, sentinel)
        elif op == 3:
            values = _values(direct_cls, [rng.randrange(100) for _ in batch])
            assert m.set_many(batch, values) == len(set(batch) - oracle.keys())
            oracle.update(zip(batch, values))
        elif op == 4:
            batch += [-1, LIMIT]
            assert m.discard_many(batch) == len(set(batch) & oracle.keys())
            for k in batch:
                oracle.pop(k, None)
        elif op == 5:
            np.testing.assert_array_equal(m.contains_many(batch), [k in oracle for k in batch])
            _assert_values(m, m.get_many(batch, value), [oracle.get(k, value) for k in batch])
        elif op in (6, 7, 8):
            method = ("popitem", "pop_first", "pop_last")[op - 6]
            if oracle:
                k = min(oracle) if op == 7 else max(oracle)
                assert getattr(m, method)() == (k, oracle.pop(k))
            else:
                with pytest.raises(KeyError):
                    getattr(m, method)()
        elif op == 9:
            assert m.setdefault(key, value) == oracle.setdefault(key, value)
        elif op == 10:
            pairs = [(batch[0], value), (batch[1], value)]
            m.update(dict(pairs) if step % 2 else iter(pairs))
            oracle.update(pairs)
        elif op == 11 and rng.randrange(8) == 0:
            m.clear()
            oracle.clear()
        elif op == 12:
            assert (key in m) == (key in oracle)
            assert m.get(key, sentinel) == oracle.get(key, sentinel)
        else:
            m.shrink_to_fit()
        expected = sorted(oracle.items())
        assert len(m) == len(oracle), step
        assert list(m) == sorted(oracle), step
        assert m.items() == expected, step
        if step % 31 == 0:
            np.testing.assert_array_equal(m.keys(), sorted(oracle))
            _assert_values(m, m.values(), [v for _, v in expected])
            start, stop = sorted([key, rng.randrange(71_000)])
            assert m.items_range(start, stop) == [(k, v) for k, v in expected if start <= k < stop]
            next_item = next(((k, v) for k, v in expected if k >= key), None)
            prev_item = next(((k, v) for k, v in reversed(expected) if k <= key), None)
            assert m.next_occupant(key) == next_item
            assert m.prev_occupant(key) == prev_item


def test_unusual_key_arrays(direct_cls, unusual_u32_keys):
    values = _values(direct_cls, [1, 2, 3, 4])
    m = direct_cls(unusual_u32_keys, values)
    assert m.items() == [(1, values[1]), (7, values[3]), (9, values[2])]


def test_hostile_array_conversion(direct_cls, hostile_array):
    values = _values(direct_cls, [1, 2, 3, 4])
    m = direct_cls(hostile_array, values)
    assert m.items() == [(1, values[1]), (7, values[3]), (9, values[2])]


def test_typed_strided_and_endian_values(typed_direct_cls):
    dtype = np.dtype(typed_direct_cls.dtype).newbyteorder(">")
    values = np.array(_values(typed_direct_cls, [1, 99, 2, 99, 3, 99]), dtype=dtype)[::2]
    m = typed_direct_cls([9, 2, 7], values)
    _assert_values(m, m.values(), [values[1], values[2], values[0]])


def test_bool_noncanonical_bytes():
    values = np.array([0, 2, 255], dtype=np.uint8).view(np.bool_)
    m = pc.DirectMapBool([9, 2, 7], values)
    assert m.items() == [(2, True), (7, True), (9, False)]


def test_methods_bypass_subclass_setitem():
    calls = []

    class Logged(pc.DirectMapF64):
        def __setitem__(self, key, value):
            calls.append(key)
            super().__setitem__(key, value)

    m = Logged()
    m[1] = 1.0
    m.set_many([2, 3], 2.0)
    m.update({4: 4.0})
    assert calls == [1]
    assert m.items() == [(1, 1.0), (2, 2.0), (3, 2.0), (4, 4.0)]


def test_query_index_ignores_int_subclass_comparisons():
    from paged_collections import DirectMap

    class I(int):
        def __lt__(self, other):
            return False

    m = DirectMap()
    m[0] = "a"
    m[2] = "b"
    assert m.prev_occupant(I(-1)) is None
    assert m.items_range(I(-1)) == [(0, "a"), (2, "b")]
