"""Copied concurrent APIs, conversion contracts and shared-call threading."""

from collections.abc import Mapping, MutableMapping
from concurrent.futures import ThreadPoolExecutor
import copy
import pickle
import random
import threading

import numpy as np
import pytest

import paged_collections as pc


SUFFIXES = ("F64", "F32", "I64", "I32", "U64", "U32", "Bool")
MAPS = tuple(
    getattr(pc, f"Concurrent{kind}Map{suffix}")
    for kind in ("Sparse", "Direct") for suffix in SUFFIXES
)
DIRECT_MAPS = tuple(cls for cls in MAPS if cls.__name__.startswith("ConcurrentDirect"))
CLASSES = (pc.ConcurrentSparseSet, *MAPS)


@pytest.fixture(params=MAPS, ids=lambda cls: cls.__name__)
def map_cls(request):
    return request.param


def _sparse(cls):
    return cls.__name__.startswith("ConcurrentSparse")


def _values(cls, numbers):
    dtype = np.dtype(cls.dtype)
    if dtype == np.dtype(bool):
        return [bool(n % 2) for n in numbers]
    return [dtype.type(n).item() for n in numbers]


def _write(m, keys, values):
    method = m.insert_many if _sparse(type(m)) else m.set_many
    return method(keys, values)


def _assert_map(m, oracle):
    items = m.items()
    assert len(m) == len(oracle)
    assert len(items) == len(oracle)
    assert dict(items) == oracle
    if not _sparse(type(m)):
        assert items == sorted(oracle.items())
    keys, values = m.keys(), m.values()
    assert keys.dtype == np.dtype(np.uint32)
    assert values.dtype == np.dtype(m.dtype)
    assert keys.tolist() == [k for k, _ in items]
    np.testing.assert_array_equal(values, [v for _, v in items])


def test_map_key_snapshot_and_batch_membership(map_cls):
    m = map_cls([4096, 1], _values(map_cls, [1, 2]))
    assert isinstance(m, Mapping)
    assert not isinstance(m, MutableMapping)
    keys = m.keys().tolist()
    iterator = iter(m)
    probes = [1, 4096, 2, -1, 2**32, 2**100, 1.0, "1", None]
    expected = [key in m for key in probes]
    for batch in (probes, np.array(probes, dtype=object), iter(probes)):
        result = m.contains_many(batch)
        assert result.dtype == np.dtype(bool)
        assert result.tolist() == expected
    assert m.contains_many(np.array([1, -1, 4096], dtype=np.int64)).tolist() == [True, False, True]
    assert m.contains_many(np.array([1.0])).tolist() == [False]
    assert m.contains_many([]).tolist() == []
    with pytest.raises(ValueError, match="one-dimensional"):
        m.contains_many(np.array([[1]]))
    m.clear()
    assert list(iterator) == keys
    assert list(m) == []


@pytest.mark.parametrize("cls", DIRECT_MAPS, ids=lambda cls: cls.__name__)
def test_direct_range_snapshots(cls):
    a, b, c = _values(cls, [1, 2, 3])
    m = cls([4096, 0, 9], [a, b, c])
    assert m.items_range() == [(0, b), (9, c), (4096, a)]
    assert m.items_range(stop=9) == [(0, b)]
    assert m.items_range(start=4096, stop=2**100) == [(4096, a)]
    assert m.items_range(10, 9) == m.items_range(stop=-1) == []
    keys = m.keys_range(start=np.int64(9), stop=4096)
    values = m.values_range(9, 4096)
    assert keys.dtype == np.dtype(np.uint32)
    assert values.dtype == np.dtype(cls.dtype)
    m.clear()
    assert keys.tolist() == [9] and values.tolist() == [c]
    for method in (m.keys_range, m.values_range, m.items_range):
        with pytest.raises(TypeError):
            method(stop=1.5)


@pytest.mark.parametrize("cls", DIRECT_MAPS, ids=lambda cls: cls.__name__)
def test_direct_get_or_insert_and_setdefault(cls):
    a, b = _values(cls, [1, 2])
    m = cls()
    assert m.get_or_insert(1, a) == a
    assert m.get_or_insert(1, b) == a
    assert m.setdefault(2, value=b) == b
    assert m.setdefault(2, a) == b
    for method in (m.get_or_insert, m.setdefault):
        with pytest.raises(OverflowError):
            method(-1, a)
        with pytest.raises(TypeError):
            method("1", a)
    assert m.items() == [(1, a), (2, b)]

    barrier = threading.Barrier(2)

    def insert(method, value):
        barrier.wait(timeout=30)
        return method(3, value)

    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(insert, m.get_or_insert, a)
        second = pool.submit(insert, m.setdefault, b)
        assert first.result(timeout=30) == second.result(timeout=30) == m[3]


def test_direct_shrink_to_fit_requires_exclusive_access():
    m = pc.ConcurrentDirectMapI64([1, 4096], [10, 20])
    del m[4096]

    def keys():
        with pytest.raises(RuntimeError, match="^Already borrowed$"):
            m.shrink_to_fit()
        yield 1

    assert m.contains_many(keys()).tolist() == [True]
    m.shrink_to_fit()
    assert m.items() == [(1, 10)]
    assert m.get_or_insert(4096, 30) == 30
    assert m.items() == [(1, 10), (4096, 30)]


def test_set_semantics_and_snapshots():
    s, oracle = pc.ConcurrentSparseSet(), set()
    rng = random.Random(101)
    assert s.position(0) is None
    assert s.keys().dtype == np.dtype(np.uint32)
    for _ in range(300):
        key = rng.randrange(64) + rng.choice([0, 1024, 4096])
        batch = [key, key, rng.randrange(64)]
        op = rng.randrange(6)
        if op == 0:
            assert s.insert(key) == (key not in oracle)
            oracle.add(key)
        elif op == 1:
            assert s.insert_many(batch) == len(set(batch) - oracle)
            oracle.update(batch)
        elif op == 2:
            assert s.remove(key) == (key in oracle)
            oracle.discard(key)
        elif op == 3:
            assert s.remove_many(batch) == len(set(batch) & oracle)
            oracle.difference_update(batch)
        elif op == 4:
            result = s.contains_many(batch + [-1, 2**100])
            assert result.dtype == np.dtype(bool)
            np.testing.assert_array_equal(result, [k in oracle for k in batch] + [False, False])
        else:
            assert (key in s) == (key in oracle)
        assert len(s) == len(oracle)
        keys = s.keys()
        assert set(keys.tolist()) == oracle
        assert list(s) == keys.tolist()
        assert all(s.position(int(k)) == i for i, k in enumerate(keys))
    snapshot, iterator = s.keys(), iter(s)
    before = snapshot.tolist()
    s.clear()
    s.insert(999)
    assert snapshot.tolist() == before
    assert list(iterator) == before
    assert s.remove_many([-1, 2**100, 999, 999]) == 1
    assert not s
    assert repr(s) == "ConcurrentSparseSet([])"
    for bad in ([1, -1], [1, 2**32]):
        with pytest.raises(OverflowError):
            s.insert_many(bad)
        assert not s
    for bad in ([[1, 2]], [1.5]):
        with pytest.raises((ValueError, TypeError)):
            s.insert_many(bad)
        assert not s


def test_scalar_map_contract(map_cls):
    m = map_cls()
    a, b, c = _values(map_cls, [1, 2, 3])
    marker = object()
    assert not m
    assert m.get(9) is None
    assert m.get(9, marker) is marker
    with pytest.raises(KeyError) as error:
        m[9]
    assert error.value.args == (9,)
    assert m.try_insert(np.uint32(9), a)
    assert not m.try_insert(9, b)
    assert m[9] == a and 9 in m
    if _sparse(map_cls):
        assert m.get_or_insert(9, b) == a
        assert m.get_or_insert(2, b) == b
        assert m.remove(9) == a
        assert m.remove(9) is None
        assert m.try_insert(9, c)
    else:
        assert m.insert(9, b) == a
        assert m.insert(2, b) is None
        m[9] = c
        assert m.remove(9) == c
        assert m.remove(9) is None
        assert m.remove(9, marker) is marker
        m[9] = c
        del m[9]
        with pytest.raises(KeyError):
            del m[9]
        m[9] = c
    _assert_map(m, {2: b, 9: c})
    assert repr(m).startswith(f"{map_cls.__name__}({{")
    for bad in [-1, 2**32, 2**100, 1.5, "1", None]:
        assert bad not in m
        assert m.get(bad, marker) is marker
        with pytest.raises(KeyError):
            m[bad]
        with pytest.raises((OverflowError, TypeError)):
            m.try_insert(bad, a)
    m.clear()
    _assert_map(m, {})


def test_batches_and_copies(map_cls):
    m = map_cls()
    a, b, c, default = _values(map_cls, [1, 2, 3, 0])
    result = _write(m, [9, 2, 9], [a, b, c])
    oracle = {9: a if _sparse(map_cls) else c, 2: b}
    assert result == (2 if _sparse(map_cls) else None)
    _assert_map(m, oracle)
    result = _write(m, np.array([9, 4], dtype=np.uint32), [b, c])
    assert result == (1 if _sparse(map_cls) else None)
    if not _sparse(map_cls):
        oracle[9] = b
    oracle[4] = c
    _assert_map(m, oracle)
    result = m.get_many([9, 4, 9, -1, 2**100], default)
    assert result.dtype == np.dtype(map_cls.dtype)
    np.testing.assert_array_equal(result, [oracle[9], c, oracle[9], default, default])
    # No output array aliases the container.
    keys, values, items = m.keys(), m.values(), m.items()
    keys[:] = 0
    values[:] = default
    items.clear()
    result[:] = default
    _assert_map(m, oracle)
    assert m.remove_many([9, 9, 2**100, -1, 4]) == 2
    assert m.remove_many([]) == 0
    _assert_map(m, {2: b})
    assert _write(m, [], []) == (0 if _sparse(map_cls) else None)
    assert m.get_many([], default).dtype == np.dtype(map_cls.dtype)
    m.clear()
    _assert_map(m, {})


def test_batch_validation_precedes_writes(map_cls):
    m = map_cls()
    a, b = _values(map_cls, [1, 2])
    _write(m, [1], [a])
    for keys, values, error in [
        ([1, 2], [b], ValueError),
        ([1], [b, a], ValueError),
        ([1, -1], [b, a], OverflowError),
        ([1, 2**32], [b, a], OverflowError),
        ([[1, 2]], [b, a], ValueError),
        ([1, 2], [[b, a]], ValueError),
        ([1, 2], ["x", "y"], TypeError),
    ]:
        with pytest.raises(error):
            _write(m, keys, values)
        _assert_map(m, {1: a})
    bad_keys = [1, 2**100, object()]
    for method, args in ((m.get_many, (bad_keys, a)), (m.remove_many, (bad_keys,))):
        with pytest.raises(TypeError):
            method(*args)
        _assert_map(m, {1: a})
    # value_batch's scalar broadcast is shared with the plain typed maps.
    _write(m, [4, 5], b)
    _assert_map(m, {1: a, 4: b, 5: b})


def test_seeded_dict_oracle(map_cls):
    m, oracle = map_cls(), {}
    rng = random.Random(0xC011)
    sparse = _sparse(map_cls)
    for step in range(200):
        key = rng.randrange(24) + rng.choice([0, 1024, 4096])
        value, default = _values(map_cls, [rng.randrange(100), 0])
        batch = [key, key, rng.randrange(24)]
        op = rng.randrange(8)
        if op == 0:
            assert m.try_insert(key, value) == (key not in oracle)
            oracle.setdefault(key, value)
        elif op == 1:
            if sparse:
                assert m.get_or_insert(key, value) == oracle.setdefault(key, value)
            else:
                assert m.insert(key, value) == oracle.get(key)
                oracle[key] = value
        elif op == 2:
            assert m.remove(key) == oracle.pop(key, None)
        elif op == 3:
            values = _values(map_cls, [1, 2, 3])
            added = len(set(batch) - oracle.keys())
            assert _write(m, batch, values) == (added if sparse else None)
            for k, v in zip(batch, values):
                if sparse:
                    oracle.setdefault(k, v)
                else:
                    oracle[k] = v
        elif op == 4:
            assert m.remove_many(batch) == len(set(batch) & oracle.keys())
            for k in batch:
                oracle.pop(k, None)
        elif op == 5:
            actual = m.get_many(batch, default)
            assert actual.dtype == np.dtype(map_cls.dtype)
            np.testing.assert_array_equal(actual, [oracle.get(k, default) for k in batch])
        elif op == 6 and step % 17 == 0:
            m.clear()
            oracle.clear()
        else:
            assert (key in m) == (key in oracle)
            assert m.get(key) == oracle.get(key)
        _assert_map(m, oracle)


@pytest.mark.parametrize("cls", CLASSES, ids=lambda cls: cls.__name__)
def test_exports_and_no_page_size(cls):
    assert cls.__name__ in pc.__all__
    value = cls()
    with pytest.raises(TypeError):
        cls(page_size=1024)
    with pytest.raises(TypeError):
        memoryview(value)
    assert not hasattr(value, "keys_view")
    assert not hasattr(value, "values_view")


def test_set_constructor_pickle_and_copy():
    keys = np.array([7, 3, 7, 4096, 3, 0], dtype=np.uint32)
    for source in (keys, keys.tolist(), keys.astype(np.int64), range(0)):
        s = pc.ConcurrentSparseSet(source)
        expected = list(dict.fromkeys(int(k) for k in source))
        assert s.keys().tolist() == expected
    s = pc.ConcurrentSparseSet(keys)
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        restored = pickle.loads(pickle.dumps(s, protocol=protocol))
        assert type(restored) is pc.ConcurrentSparseSet
        assert restored.keys().tolist() == [7, 3, 4096, 0]
    for clone in (s.copy(), copy.copy(s), copy.deepcopy(s)):
        assert clone is not s
        assert clone.keys().tolist() == [7, 3, 4096, 0]
        clone.insert(5)
        clone.remove(7)
    assert s.keys().tolist() == [7, 3, 4096, 0]
    for bad in ([1, -1], [2**32], [1.5], [[1, 2]]):
        with pytest.raises((OverflowError, ValueError, TypeError)):
            pc.ConcurrentSparseSet(bad)


def test_map_constructor_pickle_and_copy(map_cls):
    a, b, c = _values(map_cls, [1, 2, 3])
    # Later duplicates win in the constructor, as for the plain maps.
    keys = np.array([9, 2, 9, 5000], dtype=np.uint32)
    values = np.array([a, b, c, a], dtype=map_cls.dtype)
    oracle = {9: c, 2: b, 5000: a}
    for k, v in ((keys, values), (keys.tolist(), values.tolist()), (keys.astype(np.int64), values)):
        _assert_map(map_cls(k, v), oracle)
    if _sparse(map_cls):
        assert map_cls(keys, values).keys().tolist() == [9, 2, 5000]
    _assert_map(map_cls([], []), {})
    _assert_map(map_cls([1, 2], b), {1: b, 2: b})
    for args, error in [
        (([1],), TypeError),
        ((None, [a]), TypeError),
        (([1, 2], [a]), ValueError),
        (([1, -1], [a, b]), OverflowError),
        (([2**32], [a]), OverflowError),
        (([1, 2], ["x", "y"]), TypeError),
    ]:
        with pytest.raises(error):
            map_cls(*args)
    m = map_cls(keys, values)
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        restored = pickle.loads(pickle.dumps(m, protocol=protocol))
        assert type(restored) is map_cls
        _assert_map(restored, oracle)
        assert restored.items() == m.items()
    for clone in (m.copy(), copy.copy(m), copy.deepcopy(m)):
        assert clone is not m
        assert clone.items() == m.items()
        clone.clear()
    _assert_map(m, oracle)


def test_pickle_during_shared_writes(map_cls):
    m = map_cls()
    (v,) = _values(map_cls, [1])
    stop = threading.Event()

    def writer():
        key = 0
        while not stop.is_set():
            _write(m, np.arange(key, key + 64, dtype=np.uint32), v)
            key = (key + 64) % 200_000

    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(writer) for _ in range(2)]
        try:
            for _ in range(20):
                restored = pickle.loads(pickle.dumps(m))
                keys, values = restored.keys(), restored.values()
                assert len(keys) == len(values) == len(restored)
                assert len(set(keys.tolist())) == len(keys)
                np.testing.assert_array_equal(values, np.full(len(values), v, dtype=map_cls.dtype))
        finally:
            stop.set()
            for future in futures:
                future.result()


def test_threaded_maps(map_cls):
    m = map_cls()
    barrier = threading.Barrier(4)
    width = 4096
    # Existing overlap plus a new range contested by all four writers.
    shared = np.arange(3 * width, 5 * width, dtype=np.uint32)
    default = _values(map_cls, [0])[0]

    def values_for(keys):
        return np.asarray(_values(map_cls, (int(k) % 101 for k in keys)), dtype=map_cls.dtype)

    def worker(thread):
        own = np.arange(thread * width, (thread + 1) * width, dtype=np.uint32)
        barrier.wait(timeout=30)
        added = _write(m, own, values_for(own))
        barrier.wait(timeout=30)
        overlap_added = _write(m, shared, values_for(shared))
        actual = m.get_many(own, default)
        np.testing.assert_array_equal(actual, values_for(own))
        assert actual.dtype == np.dtype(map_cls.dtype)
        if _sparse(map_cls):
            assert added == width
        return overlap_added

    with ThreadPoolExecutor(max_workers=4) as pool:
        added = list(pool.map(worker, range(4)))
    if _sparse(map_cls):
        assert sum(added) == width
    oracle = dict(zip(range(5 * width), _values(map_cls, (k % 101 for k in range(5 * width)))))
    _assert_map(m, oracle)


def test_threaded_set():
    s = pc.ConcurrentSparseSet()
    barrier = threading.Barrier(4)
    width = 4096

    def worker(thread):
        own = np.arange(thread * width, (thread + 1) * width, dtype=np.uint32)
        barrier.wait(timeout=30)
        assert s.insert_many(own) == width
        barrier.wait(timeout=30)
        added = s.insert_many(np.arange(3 * width, 5 * width, dtype=np.uint32))
        np.testing.assert_array_equal(s.contains_many(own), np.ones(width, dtype=bool))
        return added

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(worker, range(4))) == width
    assert len(s) == 5 * width
    assert set(s) == set(range(5 * width))


def test_clear_during_get_many(map_cls):
    m = map_cls()
    value, default = _values(map_cls, [7, 0])
    _write(m, np.arange(4096, dtype=np.uint32), value)
    keys = np.tile(np.arange(4096, dtype=np.uint32), 256)
    entered = threading.Event()

    class SignallingKeys:
        def __array__(self, dtype=None, copy=None):
            # Called from get_many after PyO3 acquired the shared borrow.
            # No sleep or assumption about how long the detached loop takes.
            entered.set()
            return np.asarray(keys, dtype=dtype)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(m.get_many, SignallingKeys(), default)
        assert entered.wait(timeout=30)
        try:
            m.clear()
        except RuntimeError as error:
            assert "borrow" in str(error).lower()
            cleared = False
        else:
            # The shared call may already have finished: both schedules work.
            cleared = True
        actual = future.result(timeout=30)
    assert actual.dtype == np.dtype(map_cls.dtype)
    np.testing.assert_array_equal(actual, np.full(keys.size, value, dtype=map_cls.dtype))
    assert len(m) == (0 if cleared else 4096)
    m.clear()
    _write(m, [1], [value])
    _assert_map(m, {1: value})
