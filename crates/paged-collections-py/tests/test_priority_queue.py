"""PriorityQueue and PriorityQueueF64 against a dict model."""

import copy
from collections import UserDict
from collections.abc import Mapping
from contextlib import contextmanager
import gc
import math
import pickle
import random
import subprocess
import sys
import textwrap
from types import MappingProxyType

import pytest

import paged_collections as pc

CLASSES = (pc.PriorityQueue, pc.PriorityQueueF64)


class QueueSub(pc.PriorityQueue):
    pass


class QueueSubF64(pc.PriorityQueueF64):
    pass


SUBCLASSES = {pc.PriorityQueue: QueueSub, pc.PriorityQueueF64: QueueSubF64}


@contextmanager
def no_unraisable():
    errors = []
    previous = sys.unraisablehook
    sys.unraisablehook = errors.append
    try:
        yield
        gc.collect()
        assert errors == []
    finally:
        sys.unraisablehook = previous


@pytest.fixture(params=CLASSES, ids=lambda c: c.__name__)
def cls(request):
    return request.param


@pytest.mark.parametrize("is_min", [False, True])
def test_random_ops_match_model(cls, is_min):
    rng = random.Random(int(is_min))
    q, model = cls(min=is_min), {}
    best = min if is_min else max
    for _ in range(5000):
        op = rng.randrange(6)
        k = rng.choice([rng.randrange(64), rng.randrange(2**32)])
        p = rng.randrange(-100, 100)
        if op < 3:
            assert q.push(k, p) == model.get(k)
            model[k] = p
        elif op == 3 and model:
            key, prio = q.pop()
            assert prio == best(model.values())
            assert model.pop(key) == prio
        elif op == 4:
            if k in model:
                assert q.remove(k) == model.pop(k)
            else:
                with pytest.raises(KeyError):
                    q.remove(k)
        else:
            assert q.get(k) == model.get(k)
            assert (k in q) == (k in model)
        assert len(q) == len(model)
    assert sorted(q.items()) == sorted(model.items())
    s = q.sorted()
    assert [p for _, p in s] == sorted(model.values(), reverse=not is_min)
    assert len(q) == len(model)  # sorted() leaves the queue alone


def test_basic(cls):
    q = cls({5: 1, 7: 3}, min=False)
    assert q.peek() == (7, 3)
    q[5] = 10
    assert q[5] == 10 and q.peek() == (5, 10)
    del q[5]
    assert 5 not in q and len(q) == 1
    with pytest.raises(KeyError):
        q[5]
    assert q.get(5, "d") == "d"
    assert "x" not in q and -1 not in q and 2**32 not in q
    with pytest.raises(OverflowError):
        q.push(2**32, 1)
    with pytest.raises((OverflowError, TypeError)):
        q.push(-1, 1)
    q.clear()
    with pytest.raises(IndexError):
        q.pop()
    with pytest.raises(IndexError):
        q.peek()
    assert not q
    assert cls.page_size == 1024


def test_constructor_accepts_mapping_inputs(cls):
    class Priorities(Mapping):
        def __iter__(self):
            return iter([5, 7])

        def __len__(self):
            return 2

        def __getitem__(self, key):
            return {5: 1, 7: 3}[key]

    for items in (Priorities(), UserDict({5: 1, 7: 3}), MappingProxyType({5: 1, 7: 3})):
        q = cls(items, min=True)
        assert dict(q.items()) == dict(items)
        assert q.is_min and q.peek() == (5, 1)


@pytest.mark.parametrize("dict_subclass", [False, True])
def test_constructor_uses_keys_and_getitem_like_dict(cls, dict_subclass):
    calls = []

    class Priorities(dict if dict_subclass else object):
        def keys(self):
            calls.append("keys")
            return iter([5, 7])

        def __getitem__(self, key):
            calls.append(("getitem", key))
            return {5: 1, 7: 3}[key]

        def __iter__(self):
            raise AssertionError("mapping must use keys(), not __iter__")

        def items(self):
            raise AssertionError("mapping need not provide items()")

    items = Priorities()
    expected = dict(items)
    assert calls == ["keys", ("getitem", 5), ("getitem", 7)]
    calls.clear()
    q = cls(items)
    assert dict(q.items()) == expected
    assert calls == ["keys", ("getitem", 5), ("getitem", 7)]


@pytest.mark.parametrize("keys", [None, 0])
def test_constructor_keys_attribute_selects_mapping(cls, keys):
    class Pairs:
        def __iter__(self):
            return iter([(1, 1)])

    items = Pairs()
    items.keys = keys
    with pytest.raises(TypeError):
        cls(items)


@pytest.mark.parametrize("failure", ["keys", "getitem", "priority"])
def test_bad_mapping_init_preserves_state(cls, failure):
    calls = []

    class Priorities:
        def keys(self):
            calls.append("keys")
            if failure == "keys":
                raise LookupError("keys failed")
            return [2, 3]

        def __getitem__(self, key):
            calls.append(key)
            if key == 3:
                if failure == "getitem":
                    raise LookupError("getitem failed")
                return object()
            return key

    q = cls({1: 1}, min=True)
    error = TypeError if failure == "priority" else LookupError
    with pytest.raises(error):
        q.__init__(Priorities(), min=False)
    assert calls == (["keys"] if failure == "keys" else ["keys", 2, 3])
    assert q.items() == [(1, 1)] and q.is_min


def test_mapping_callbacks_are_unborrowed(cls):
    q = cls({99: 1})
    calls = []

    class Priorities:
        def keys(self):
            calls.append("keys")
            q.clear()
            return [1, 2]

        def __getitem__(self, key):
            calls.append(key)
            q.push(99, key)
            return key

    q.__init__(Priorities(), min=True)
    assert calls == ["keys", 1, 2]
    assert dict(q.items()) == {1: 1, 2: 2} and q.is_min


def test_dtypes():
    q = pc.PriorityQueue()
    with pytest.raises(TypeError):
        q.push(1, 1.5)
    assert pc.PriorityQueue.dtype == "int64"
    f = pc.PriorityQueueF64(min=True)
    f.push(1, 2)
    f.push(2, -math.inf)
    assert f.peek() == (2, -math.inf)
    with pytest.raises(ValueError):
        f.push(3, math.nan)
    assert 3 not in f
    assert pc.PriorityQueueF64.dtype == "float64"


def test_bad_init_preserves_state(cls):
    with pytest.raises(OverflowError):
        cls([(1, 1), (2**32, 2)])
    q = cls([(1, 1)], min=True)
    with pytest.raises(OverflowError):
        q.__init__([(2, 2), (2**40, 1)], min=False)
    assert q.items() == [(1, 1)] and q.is_min
    with pytest.raises(TypeError):
        q.__init__([(2, 2), (3, object())], min=False)
    assert q.items() == [(1, 1)] and q.is_min

    def invalid_iterator():
        yield 2, 2
        raise LookupError("bad input")

    with pytest.raises(LookupError):
        q.__init__(invalid_iterator(), min=False)
    assert q.items() == [(1, 1)] and q.is_min


def test_eq_copy_pickle(cls):
    q = cls([(1, 5), (2, 7), (3, 6)], min=True)
    assert q.is_min
    for r in (q.copy(), copy.copy(q), copy.deepcopy(q), pickle.loads(pickle.dumps(q))):
        assert r == q and r.is_min and r.pop() == (1, 5)
    assert q != cls([(1, 5), (2, 7), (3, 6)])
    assert sorted(q) == [1, 2, 3]
    assert sorted(q.keys()) == [1, 2, 3]
    assert repr(cls([(4, 2)], min=True)).endswith("({4: 2" + (".0" if cls is pc.PriorityQueueF64 else "") + "}, min=True)")

    sub = SUBCLASSES[cls]
    s = sub([(9, 1)])
    s.tag = 3
    for r in (s.copy(), copy.copy(s), copy.deepcopy(s), pickle.loads(pickle.dumps(s))):
        assert type(r) is sub and r.tag == 3 and r[9] == 1


def test_capacity(cls):
    q = cls()
    q.reserve(100)
    assert q.capacity >= 100
    q.push(3_000_000, 1)
    assert q.allocated_pages >= 1
    q.shrink_to_fit()


@pytest.mark.parametrize("additional", [sys.maxsize, 2**32, 2**100, -1])
def test_impossible_reserve_is_python_error(cls, additional):
    q = cls([(1, 7)], min=True)
    # On 32-bit, sys.maxsize entries fit u32 but exceed Vec's byte limit.
    with pytest.raises((OverflowError, MemoryError)):
        q.reserve(additional)
    assert q.items() == [(1, 7)] and q.is_min
    q.reserve(0)
    q.reserve(10)
    assert q.capacity >= 11


def test_reserve_length_plus_additional_overflow(cls):
    q = cls([(1, 1)])
    # The requested additional count alone fits u32; the final length does not.
    with pytest.raises(OverflowError):
        q.reserve(2**32 - 1)
    assert q.items() == [(1, 1)]


@pytest.mark.skipif(sys.platform != "linux", reason="uses Linux address-space accounting")
def test_failed_reserve_probe_is_memory_error(cls):
    # Bound memory in a child, so a regression to an aborting core allocation
    # cannot terminate pytest. Import numpy/the extension before imposing it.
    script = textwrap.dedent(f"""
        import os
        import resource
        import paged_collections as pc

        q = pc.{cls.__name__}([(1, 7)], min=True)
        with open('/proc/self/statm') as f:
            virtual = int(f.read().split()[0]) * os.sysconf('SC_PAGE_SIZE')
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        limit = virtual + 64 * 1024 * 1024
        if hard != resource.RLIM_INFINITY:
            limit = min(limit, hard)
        resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
        try:
            q.reserve(20_000_000)
        except MemoryError:
            pass
        else:
            raise AssertionError('reservation should fail within the memory limit')
        assert q.items() == [(1, 7)] and q.is_min
    """)
    result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_key_and_priority_conversions_are_unborrowed(cls):
    q = cls([(7, 1)])
    effects = []

    class Key:
        def __index__(self):
            effects.append("key")
            q.push(8, 2)
            return 7

    class Priority:
        def __index__(self):
            effects.append("priority")
            assert len(q) >= 1
            q.clear()
            return 9

        def __float__(self):
            return float(self.__index__())

    assert q.push(Key(), Priority()) is None
    assert q[7] == 9
    assert Key() in q
    assert q[Key()] == 9
    assert q.get(Key()) == 9
    assert q.remove(Key()) == 9
    q[Key()] = Priority()
    assert q[7] == 9
    del q[Key()]
    assert 7 not in q

    class Additional:
        def __index__(self):
            effects.append("reserve")
            q.clear()
            return 3

    q.reserve(Additional())
    assert len(q) == 0 and q.capacity >= 3
    assert effects.count("priority") == 2
    assert effects.count("key") == 7
    assert effects[-1] == "reserve"


def test_constructor_conversions_and_destructors_are_unborrowed(cls):
    q = cls([(99, 1)])
    effects = []

    class Priority:
        def __init__(self, value):
            self.value = value

        def __index__(self):
            effects.append(("convert", self.value))
            q.push(99, self.value)
            return self.value

        def __float__(self):
            return float(self.__index__())

        def __del__(self):
            q.clear()
            effects.append(("drop", self.value))

    with no_unraisable():
        q.__init__(((key, Priority(key)) for key in [1, 2]), min=True)
        assert dict(q.items()) == {1: 1, 2: 2}
        assert q.is_min
        assert sorted(effects) == [("convert", 1), ("convert", 2), ("drop", 1), ("drop", 2)]

        class Bad(Priority):
            def __index__(self):
                raise ValueError("invalid priority")

        with pytest.raises(ValueError):
            q.__init__(((key, Bad(key)) for key in [3, 4]))
        gc.collect()
        # The failing priority's destructor is allowed to clear the old queue.
        assert len(q) == 0
        assert ("drop", 3) in effects


@pytest.mark.parametrize("is_min", [False, True])
def test_constructor_duplicate_updates(cls, is_min):
    q = cls([(1, 1), (1, 5), (2, 2), (1, 3)], min=is_min)
    assert len(q) == 2 and dict(q.items()) == {1: 3, 2: 2}
    assert q.peek() == ((2, 2) if is_min else (1, 3))


@pytest.mark.parametrize("is_min", [False, True])
def test_f64_nan_updates_init_and_signed_zero(is_min):
    q = pc.PriorityQueueF64([(1, -0.0), (2, 0.0), (3, math.inf), (4, -math.inf)], min=is_min)
    assert math.copysign(1, q[1]) == -1
    assert math.copysign(1, q[2]) == 1
    assert [p for _, p in q.sorted()] == sorted([-0.0, 0.0, math.inf, -math.inf], reverse=not is_min)
    before = dict(q.items())
    with pytest.raises(ValueError):
        q.push(1, math.nan)
    assert dict(q.items()) == before
    with pytest.raises(ValueError):
        q.__init__([(5, 1), (6, math.nan)], min=not is_min)
    assert dict(q.items()) == before and q.is_min == is_min
    assert q == pc.PriorityQueueF64([(1, 0.0), (2, -0.0), (3, math.inf), (4, -math.inf)], min=is_min)


@pytest.mark.skipif(
    sys.version_info >= (3, 12),
    reason="Later interpreters may defer allocation-triggered GC until returning to Python",
)
@pytest.mark.parametrize("method", ["keys", "items", "sorted", "__iter__", "__reduce__"])
def test_gc_finalizers_during_python_allocation(cls, method):
    # These methods allocate GC-tracked Python objects. repr only creates
    # strings, so it cannot reliably trigger synchronous cyclic GC.
    q = cls([(1, 1)])
    calls = 0
    call = getattr(q, method)

    class Cycle:
        def __del__(self):
            nonlocal calls
            q.push(2, 2)
            calls += 1

    thresholds = gc.get_threshold()
    enabled = gc.isenabled()
    with no_unraisable():
        try:
            gc.collect()
            gc.disable()
            cycle = Cycle()
            cycle.self = cycle
            del cycle
            gc.set_threshold(1, 1, 1)
            gc.enable()
            result = call()
            # No cleanup collection (or allocating assertion) before this:
            # the finalizer must have run inside the native call itself.
            assert calls == 1
            assert result is not None
            assert dict(q.items()) == {1: 1, 2: 2}
        finally:
            gc.set_threshold(*thresholds)
            if enabled:
                gc.enable()
            else:
                gc.disable()


def test_gc_finalizers_during_constructor_conversion(cls):
    q = cls({99: 1})
    effects = []

    class Cycle:
        def __del__(self):
            assert effects == ["convert"]
            q.push(2, 2)
            effects.append("del")

    class Priority:
        def __index__(self):
            effects.append("convert")
            gc.collect()
            assert dict(q.items()) == {99: 1, 2: 2}
            effects.append("collected")
            return 3

        def __float__(self):
            return float(self.__index__())

    enabled = gc.isenabled()
    with no_unraisable():
        try:
            gc.collect()
            gc.disable()
            cycle = Cycle()
            cycle.self = cycle
            del cycle
            q.__init__([(1, Priority())], min=True)
            assert effects == ["convert", "del", "collected"]
            assert q.items() == [(1, 3)] and q.is_min
        finally:
            if enabled:
                gc.enable()
            else:
                gc.disable()


def test_gc_finalizers_during_reduce_callback(cls):
    effects = []

    class Queue(cls):
        def __getstate__(self):
            effects.append("getstate")
            gc.collect()
            effects.append("collected")
            return None

    q = Queue({1: 1})

    class Cycle:
        def __del__(self):
            assert effects == ["getstate"]
            q.push(2, 2)
            effects.append("del")

    enabled = gc.isenabled()
    with no_unraisable():
        try:
            gc.collect()
            gc.disable()
            cycle = Cycle()
            cycle.self = cycle
            del cycle
            reduced = q.__reduce__()
            assert effects == ["getstate", "del", "collected"]
            assert dict(q.items()) == {1: 1, 2: 2}
            # The constructor arguments were snapshotted before __getstate__.
            assert reduced[1][2] == [(1, 1)]
        finally:
            if enabled:
                gc.enable()
            else:
                gc.disable()


def test_snapshot_iteration_survives_mutation(cls):
    q = cls([(1, 1), (2, 2)])
    it = iter(q)
    q.clear()
    assert sorted(it) == [1, 2]


def test_copy_restores_slot_state_without_subclass_init(cls):
    calls = []

    class Slots(cls):
        __slots__ = ("tag",)

        def __init__(self, tag):
            calls.append(tag)
            super().__init__([(1, 7)], min=True)
            self.tag = tag

        def __getstate__(self):
            self.reserve(0)
            assert len(self) == 1
            return None, {"tag": self.tag}

    value = []
    s = Slots(value)
    for result in (s.copy(), copy.copy(s)):
        assert type(result) is Slots and result.tag is value
        assert result.items() == [(1, 7)] and result.is_min
    assert calls == [value]
    assert "__copy__" not in cls.__dict__
