"""PagedList's list protocol, reentrancy and stable page-view contract."""

import copy
import gc
import operator
import pickle
import random
import weakref

import numpy as np
import pytest

import paged_collections as pc


TYPED = (
    pc.PagedListF64, pc.PagedListF32, pc.PagedListI64, pc.PagedListI32,
    pc.PagedListU64, pc.PagedListU32, pc.PagedListBool,
)
CLASSES = (pc.PagedList, *TYPED)


@pytest.fixture(params=CLASSES, ids=lambda cls: cls.__name__)
def list_cls(request):
    return request.param


@pytest.fixture(params=TYPED, ids=lambda cls: cls.__name__)
def typed_cls(request):
    return request.param


def data(cls, values):
    return [bool(v % 2) for v in values] if cls is pc.PagedListBool else list(values)


def scalar(cls, value):
    return data(cls, [value])[0]


@pytest.mark.parametrize("length", [0, 1, 1023, 1024, 1025, 2047, 2048, 2049])
def test_random_list_equivalence(list_cls, length):
    rng = random.Random(4103 + length)
    expected = data(list_cls, (i % 19 for i in range(length)))
    actual = list_cls(expected)
    for _ in range(150):
        op = rng.randrange(13)
        i = rng.randrange(-len(expected) - 8, len(expected) + 9)
        value = scalar(list_cls, rng.randrange(19))
        if op == 0:
            actual.append(value)
            expected.append(value)
        elif op == 1:
            actual.insert(i, value)
            expected.insert(i, value)
        elif op == 2:
            if expected and -len(expected) <= i < len(expected):
                assert actual.pop(i) == expected.pop(i)
            else:
                before = list(actual)
                with pytest.raises(IndexError):
                    actual.pop(i)
                assert list(actual) == before
        elif op in (3, 4, 5):
            j = rng.randrange(-len(expected) - 8, len(expected) + 9)
            step = rng.choice([None, 1, -1, 2, -2, 3, -7])
            sl = slice(i, j, step)
            if op == 3:
                assert list(actual[sl]) == expected[sl]
                assert type(actual[sl]) is list_cls
            elif op == 4:
                vals = data(list_cls, (rng.randrange(19) for _ in range(
                    rng.randrange(8) if step in (None, 1) else len(expected[sl]))))
                actual[sl] = iter(vals)
                expected[sl] = vals
            else:
                del actual[sl]
                del expected[sl]
        elif op == 6:
            vals = data(list_cls, [rng.randrange(19) for _ in range(rng.randrange(8))])
            actual.extend(iter(vals))
            expected.extend(vals)
        elif op == 7 and expected:
            i %= len(expected)
            actual[i] = value
            expected[i] = value
        elif op == 8:
            actual.reverse()
            expected.reverse()
        elif op == 9:
            assert actual.count(value) == expected.count(value)
            assert (value in actual) == (value in expected)
            if value in expected:
                assert actual.index(value) == expected.index(value)
                actual.remove(value)
                expected.remove(value)
            else:
                with pytest.raises(ValueError):
                    actual.remove(value)
        elif op == 10:
            reverse = bool(rng.randrange(2))
            actual.sort(reverse=reverse)
            expected.sort(reverse=reverse)
        elif op == 11:
            vals = data(list_cls, [1, 2])
            actual += iter(vals)
            expected += vals
        else:
            assert list(reversed(actual)) == expected[::-1]
        assert list(actual) == expected
        assert len(actual) == len(expected)


@pytest.mark.parametrize("length", [1023, 1024, 1025, 2047, 2048, 2049])
def test_boundary_insert_pop_and_slices(list_cls, length):
    a = list_cls(data(list_cls, range(length)))
    b = list(a)
    for i in [-2049, -1024, -1, 0, 1, 1023, 1024, 2048, 10**6]:
        v = scalar(list_cls, 7)
        a.insert(i, v)
        b.insert(i, v)
        assert list(a) == b
    for i in [0, 1023, 1024, -1, -1024]:
        assert a.pop(i) == b.pop(i)
        assert list(a) == b
    for sl in [slice(None, None, -1), slice(2048, 1020, -3), slice(-2049, -1, 2),
               slice(1, 0), slice(None, None, -10**100), slice(None, None, 10**100)]:
        assert list(a[sl]) == b[sl]
        vals = data(list_cls, [2] * len(b[sl]))
        a[sl] = vals
        b[sl] = vals
        assert list(a) == b


def test_protocol_and_self_assignment(list_cls):
    a = list_cls(data(list_cls, [1, 2, 3, 4]))
    assert a.dtype == ("object" if list_cls is pc.PagedList else np.dtype(list_cls.dtype).name)
    assert a.page_size == 1024
    assert a.capacity >= len(a)
    assert a.allocated_pages == 1
    assert list(a.copy()) == list(a)
    with pytest.raises(TypeError):
        hash(a)
    assert a.__eq__([]) is NotImplemented
    assert a.__ne__([]) is NotImplemented
    assert a != []
    assert a == list_cls(list(a))
    assert not a != list_cls(list(a))
    expected = list(a)
    a[:] = a
    assert list(a) == expected
    a[1:] = reversed(a)
    expected[1:] = reversed(expected)
    assert list(a) == expected
    a += a
    expected += expected
    assert list(a) == expected
    assert list(a + a) == expected + expected
    assert list(a * 2) == expected * 2
    assert list(2 * a) == expected * 2
    assert list(a * -1) == []
    identity = a
    a *= 2
    assert a is identity
    assert list(a) == expected * 2
    a *= 0
    assert not a
    a.reserve(2049)
    assert a.capacity >= 2049
    assert a.allocated_pages >= 3
    a.shrink_to_fit()
    assert a.allocated_pages == 0
    with pytest.raises((MemoryError, OverflowError)):
        a.reserve(2**64 - 1)
    with pytest.raises(IndexError, match=f"pop from empty {list_cls.__name__}"):
        a.pop()
    with pytest.raises(TypeError):
        _ = a + []


def test_indices_and_slice_errors(list_cls):
    class Index:
        def __index__(self):
            return -1

    a = list_cls(data(list_cls, [1, 2, 3, 4]))
    assert a[Index()] == a[-1]
    a[Index()] = scalar(list_cls, 8)
    assert a[-1] == scalar(list_cls, 8)
    for i in [10**100, -10**100]:
        with pytest.raises(IndexError, match=f"{list_cls.__name__} index out of range"):
            _ = a[i]
        with pytest.raises(IndexError):
            a[i] = scalar(list_cls, 1)
        with pytest.raises(IndexError):
            del a[i]
    with pytest.raises(TypeError):
        a.pop(None)
    with pytest.raises(TypeError):
        a.index(a[0], None)
    with pytest.raises(TypeError):
        _ = a[1.0]
    with pytest.raises(ValueError, match="slice step cannot be zero"):
        _ = a[::0]
    before = list(a)
    with pytest.raises(ValueError, match="attempt to assign sequence of size 1 to extended slice of size 2"):
        a[::2] = data(list_cls, [1])
    assert list(a) == before
    a.__init__(data(list_cls, [3, 4]))
    assert list(a) == data(list_cls, [3, 4])


def test_live_iterators(list_cls):
    a = list_cls(data(list_cls, [1, 2]))
    it = iter(a)
    assert operator.length_hint(it) == 2
    assert next(it) == a[0]
    a.append(scalar(list_cls, 3))
    assert list(it) == list(a)[1:]
    a.append(scalar(list_cls, 4))
    assert list(it) == []
    reverse = reversed(a)
    assert next(reverse) == a[-1]
    a.pop()
    assert list(reverse) == list(a)[::-1]
    it = iter(a)
    a.clear()
    assert list(it) == []
    a.extend(data(list_cls, [1, 2, 3]))
    it = reversed(a)
    a.clear()
    assert list(it) == []


def test_search_bounds_and_conversion(list_cls):
    a = list_cls(data(list_cls, [1, 2, 1, 3]))
    b = list(a)
    for start, stop in [(-10, 100), (-2, 4), (1, 3), (3, 1), (-10**100, 10**100)]:
        for v in data(list_cls, [0, 1, 2, 3]):
            try:
                expected = b.index(v, start, stop)
            except ValueError:
                with pytest.raises(ValueError):
                    a.index(v, start, stop)
            else:
                assert a.index(v, start, stop) == expected
    if list_cls is not pc.PagedList:
        assert object() not in a
        assert a.count(object()) == 0
        with pytest.raises(ValueError):
            a.index(object())
        with pytest.raises(ValueError):
            a.remove(object())


@pytest.mark.parametrize("probe", [False, True, 0, 1, 1.0, 3, 3.0, 3.5, 2**70,
                                   np.int64(3), np.float64(3.0)])
def test_typed_search_matches_python_equality(typed_cls, probe):
    a = typed_cls(data(typed_cls, [0, 1, 3, 1, 3]))
    expected = list(a)
    assert (probe in a) == (probe in expected)
    assert a.count(probe) == expected.count(probe)
    for start, stop in [(0, 100), (1, 4), (-3, -1), (4, 1)]:
        if probe in expected[start:stop]:
            assert a.index(probe, start, stop) == expected.index(probe, start, stop)
        else:
            with pytest.raises(ValueError):
                a.index(probe, start, stop)
    if probe in expected:
        assert a.remove(probe) is None
        expected.remove(probe)
    else:
        with pytest.raises(ValueError):
            a.remove(probe)
    assert list(a) == expected


@pytest.mark.parametrize("cls, stored, probe", [
    (pc.PagedListF64, float(2**53), 2**53 + 1),
    (pc.PagedListF32, 1.0, 1.0 + 2**-30),
    (pc.PagedListI64, 3, 3.5),
    (pc.PagedListI64, 3, 2**70),
    (pc.PagedListU64, 2**64 - 1, float(2**64)),
])
def test_typed_search_does_not_round_probe(cls, stored, probe):
    a = cls([stored])
    assert probe not in a and a.count(probe) == 0
    with pytest.raises(ValueError):
        a.index(probe)
    with pytest.raises(ValueError):
        a.remove(probe)
    assert list(a) == [stored]


@pytest.mark.parametrize("method", ["__contains__", "index", "count", "remove"])
def test_typed_search_custom_equality_and_reentrancy(typed_cls, method):
    a = typed_cls(data(typed_cls, [0]))
    target = scalar(typed_cls, 1)

    class Probe:
        appended = False

        def __eq__(self, other):
            if not self.appended:
                self.appended = True
                a.append(target)
            return other == target

    result = getattr(a, method)(Probe())
    assert result == {"__contains__": True, "index": 1, "count": 1, "remove": None}[method]
    assert list(a) == (data(typed_cls, [0]) if method == "remove" else
                       data(typed_cls, [0, 1]))


def test_typed_search_uses_equality_instead_of_conversion(typed_cls):
    a = typed_cls(data(typed_cls, [1]))

    class Probe:
        def __index__(self):
            raise AssertionError("search must not convert the probe")

        __float__ = __index__

        def __eq__(self, other):
            raise RuntimeError("probe equality")

    for method in (a.__contains__, a.index, a.count, a.remove):
        with pytest.raises(RuntimeError, match="probe equality"):
            method(Probe())

    class UnequalInt(int):
        def __eq__(self, other):
            return False

    probe = UnequalInt(1)
    assert (probe in a) == (probe in list(a))
    assert a.count(probe) == list(a).count(probe)


def test_typed_numpy_batches(typed_cls, key_batch):
    a = typed_cls(data(typed_cls, range(12)))
    indices = key_batch.astype(key_batch.dtype, copy=True)
    expected = np.asarray(list(a), dtype=typed_cls.dtype)
    np.testing.assert_array_equal(a.get_many(indices), expected[indices])
    np.testing.assert_array_equal(a.get_many([-1, -12, 0, -1]), expected[[-1, -12, 0, -1]])
    vals = data(typed_cls, [1, 2, 3, 4])
    a.set_many([-1, 0, 0, -1], vals)
    expected[-1], expected[0] = vals[-1], vals[-2]
    np.testing.assert_array_equal(a.to_numpy(), expected)
    a.set_many([0, -1], scalar(typed_cls, 9))
    assert a[0] == a[-1] == scalar(typed_cls, 9)
    assert a.tolist() == list(a)
    assert a.get_many([]).size == 0
    out = a.to_numpy()
    assert out.dtype == np.dtype(typed_cls.dtype)
    out[0] = scalar(typed_cls, 0)
    assert a[0] == scalar(typed_cls, 9)
    for bad in [[0, 12], [-13], [0, 10**100], np.array([2**64 - 1], dtype=np.uint64)]:
        before = list(a)
        with pytest.raises(IndexError):
            a.set_many(bad, scalar(typed_cls, 1))
        with pytest.raises(IndexError):
            a.get_many(bad)
        assert list(a) == before
    for bad in [[1.0], np.array([1.0]), np.array([], dtype=float), [True], ["1"]]:
        with pytest.raises(TypeError):
            a.get_many(bad)
    with pytest.raises(ValueError):
        a.set_many([0, 1], data(typed_cls, [1]))
    with pytest.raises(ValueError):
        a.get_many([[0]])


def test_typed_conversion_atomic(typed_cls):
    a = typed_cls(data(typed_cls, [1, 0, 1]))
    a.extend(x for x in data(typed_cls, [0, 1]))
    assert list(a) == data(typed_cls, [1, 0, 1, 0, 1])
    bad = [1] if typed_cls is pc.PagedListBool else ["bad"]
    if typed_cls not in (pc.PagedListF32, pc.PagedListF64, pc.PagedListBool):
        bad = [1.5]
    for op in [lambda: a.extend(bad), lambda: a.__iadd__(bad),
               lambda: a.__setitem__(slice(None), bad), lambda: a.__init__(bad),
               lambda: a.set_many([0], bad)]:
        before = list(a)
        with pytest.raises((TypeError, ValueError)):
            op()
        assert list(a) == before
    with pytest.raises(TypeError):
        a.extend(42)
    with pytest.raises(ValueError):
        a.extend(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        a.extend(np.array(3))
    if typed_cls in (pc.PagedListU32, pc.PagedListU64):
        with pytest.raises(OverflowError):
            a.extend([-1])
    if typed_cls is pc.PagedListI32:
        with pytest.raises(OverflowError):
            a.extend([2**31])


def test_view_allowed_operations(typed_cls):
    a = typed_cls(data(typed_cls, range(1023)))
    view = a.page_view(0)
    assert len(view) == 1023
    assert not view.flags.owndata
    assert view.flags.writeable is (typed_cls is not pc.PagedListBool)
    a[0] = scalar(typed_cls, 7)
    assert view[0] == a[0]
    a[1:3] = data(typed_cls, [8, 9])
    a[5:1:-2] = data(typed_cls, [4, 6])
    a.set_many([0, 0, -1], data(typed_cls, [1, 2, 3]))
    np.testing.assert_array_equal(view, np.asarray(list(a), dtype=typed_cls.dtype))
    if typed_cls is not pc.PagedListBool:
        view[1] = scalar(typed_cls, 11)
        assert a[1] == scalar(typed_cls, 11)
    else:
        with pytest.raises(ValueError):
            view.flags.writeable = True
    before = view.copy()
    a.append(scalar(typed_cls, 1))  # fills the current page
    a.extend(data(typed_cls, range(2049)))  # allocates multiple pages
    a += data(typed_cls, [1, 0])
    a.insert(len(a), scalar(typed_cls, 1))
    a.insert(2**62, scalar(typed_cls, 0))
    a.reserve(4096)
    a *= 1
    a *= 2
    np.testing.assert_array_equal(view, before)
    assert len(view) == 1023  # a fixed prefix, not the growing page
    assert a == a.copy()
    assert list(pickle.loads(pickle.dumps(a))) == list(a)
    assert list(a[:3]) == list(a)[:3]
    assert list(reversed(a)) == list(a)[::-1]
    assert a.count(a[0]) >= 1
    assert a.index(a[0]) == 0
    assert a[0] in a
    np.testing.assert_array_equal(a.get_many([0]), [a[0]])
    assert len(a.to_numpy()) == len(a)
    views = a.page_views()
    assert sum(map(len, views)) == len(a)
    assert all(v.base is view.base for v in views)
    np.testing.assert_array_equal(a.page_view(-1), views[-1])
    with pytest.raises(IndexError):
        a.page_view(len(views))


@pytest.mark.parametrize("operation", [
    lambda a: a.pop(), lambda a: a.pop(0), lambda a: a.remove(a[0]),
    lambda a: a.__delitem__(0), lambda a: a.__delitem__(slice(None, None, 2)),
    lambda a: a.clear(), lambda a: a.insert(0, a[0]),
    lambda a: a.__setitem__(slice(0, 1), []),
    lambda a: a.__setitem__(slice(0, 0), [a[0]]),
    lambda a: a.sort(), lambda a: a.sort(reverse=True), lambda a: a.reverse(),
    lambda a: a.shrink_to_fit(), lambda a: a.__imul__(0), lambda a: a.__imul__(-2),
    lambda a: a.__init__([a[0]]),
])
def test_view_blocks_moves_atomically(typed_cls, operation):
    a = typed_cls(data(typed_cls, range(1025)))
    view = a.page_view(0)
    before = list(a)
    with pytest.raises(BufferError, match=typed_cls.__name__):
        operation(a)
    assert list(a) == before
    np.testing.assert_array_equal(view, np.asarray(before[:1024], dtype=typed_cls.dtype))
    del view
    gc.collect()
    operation(a)


def test_view_retains_owner_and_derived_views(typed_cls):
    a = typed_cls(data(typed_cls, range(2049)))
    first, second, third = a.page_views()
    assert [len(v) for v in (first, second, third)] == [1024, 1024, 1]
    derived = first[::2]
    del first, second, third
    gc.collect()
    with pytest.raises(BufferError):
        a.clear()
    expected = derived.copy()
    del a
    gc.collect()
    np.testing.assert_array_equal(derived, expected)
    empty = typed_cls()
    assert empty.page_views() == []
    with pytest.raises(IndexError):
        empty.page_view(0)


@pytest.mark.parametrize("cls", [pc.PagedListF64, pc.PagedListF32])
def test_float_sort_nan_stability(cls):
    vals = [np.nan, 0.0, -0.0, 3.0, np.nan, -2.0, -0.0, 0.0]
    a = cls(vals)
    a.sort()
    arr = a.to_numpy()
    np.testing.assert_array_equal(arr, [-2, 0, -0., -0., 0, 3, np.nan, np.nan])
    assert np.signbit(arr[1:5]).tolist() == [False, True, True, False]
    nan = float("nan")
    assert nan not in a
    assert a.count(nan) == 0
    assert a != a.copy()  # typed comparisons use numeric ==, including NaN
    assert a != a
    assert "NaN != NaN as in NumPy" in cls.__doc__
    with pytest.raises(ValueError):
        a.index(nan)
    a = cls(vals)
    a.sort(reverse=True)
    arr = a.to_numpy()
    assert np.isnan(arr[:2]).all()
    assert np.signbit(arr[3:7]).tolist() == [False, True, True, False]
    with pytest.raises(TypeError):
        a.sort(key=lambda x: x)


def test_object_sort_stable_and_transactional():
    a = pc.PagedList([(1, "a"), (0, "b"), (1, "c"), (0, "d")])
    a.sort(key=lambda x: x[0], reverse=True)
    assert list(a) == [(1, "a"), (1, "c"), (0, "b"), (0, "d")]
    before = list(a)

    def fail(value):
        raise LookupError("key failed")

    with pytest.raises(LookupError, match="key failed"):
        a.sort(key=fail)
    assert list(a) == before
    calls = []

    def mutate(value):
        calls.append(value)
        if len(calls) == 1:
            a.append((9, "new"))
        return value[0]

    with pytest.raises(ValueError, match="PagedList modified during sort"):
        a.sort(key=mutate)
    assert calls == before
    assert list(a) == before + [(9, "new")]
    before = list(a)

    def overwrite(value):
        a[0] = (7, "written")
        return value[0]

    with pytest.raises(ValueError, match="modified during sort"):
        a.sort(key=overwrite)
    assert list(a) == [(7, "written"), *before[1:]]


def test_sort_comparison_mutation_and_exception():
    class Compared:
        def __init__(self, n):
            self.n = n

        def __lt__(self, other):
            a.append("mutation")
            return self.n < other.n

    a = pc.PagedList([Compared(2), Compared(1)])
    before = list(a)
    with pytest.raises(ValueError, match="modified during sort"):
        a.sort()
    assert list(a)[:2] == before
    assert a[-1] == "mutation"

    class Fail:
        def __lt__(self, other):
            raise ArithmeticError("comparison")

    a = pc.PagedList([Fail(), Fail()])
    before = list(a)
    with pytest.raises(ArithmeticError, match="comparison"):
        a.sort()
    assert list(a) == before


def test_object_comparisons_can_mutate():
    class AppendOnce:
        def __eq__(self, other):
            if len(a) == 1:
                a.append("found")
            return False

    a = pc.PagedList([AppendOnce()])
    assert "found" in a
    a = pc.PagedList([AppendOnce()])
    assert a.index("found") == 1
    a = pc.PagedList([AppendOnce()])
    assert a.count("found") == 1
    a = pc.PagedList([AppendOnce()])
    a.remove("found")
    assert len(a) == 1

    class Clear:
        def __eq__(self, other):
            a.clear()
            return True

    a = pc.PagedList([Clear()])
    a.remove(object())
    assert not a
    a = pc.PagedList([Clear()])
    b = pc.PagedList([object()])
    assert not (a == b)  # equality rechecks lengths after the callback
    assert not a and len(b) == 1

    class GrowBoth:
        def __eq__(self, other):
            a.append("tail")
            b.append("tail")
            return True

    a = pc.PagedList([GrowBoth()])
    b = pc.PagedList([object()])
    assert a == b
    assert len(a) == len(b) == 2


def test_object_identity_before_equality():
    class Never:
        def __eq__(self, other):
            raise AssertionError("identity should win")

    obj = Never()
    a = pc.PagedList([obj])
    assert obj in a
    assert a.count(obj) == 1
    assert a == a.copy()
    nan = float("nan")
    a = pc.PagedList([nan])
    assert nan in a


@pytest.mark.parametrize("operation", [
    lambda a: a.clear(), lambda a: a.__delitem__(0),
    lambda a: a.__delitem__(slice(None)), lambda a: a.__setitem__(0, "replacement"),
    lambda a: a.__setitem__(slice(None), ["replacement"]),
    lambda a: a.__imul__(0), lambda a: a.remove(a[0]), lambda a: a.pop(),
])
def test_removed_finalizer_runs_without_borrow(operation):
    events = []
    a = pc.PagedList()

    class Finalizer:
        def __del__(self):
            a.append("finalized")
            events.append(list(a))

    a.append(Finalizer())
    operation(a)
    assert events
    assert a[-1] == "finalized"


def test_reentrant_index_repr_and_iterable():
    a = pc.PagedList(["old"])

    class Index:
        def __index__(self):
            a.append("new")
            return -1

    assert a[Index()] == "new"
    a[slice(Index(), None)] = ["slice"]
    assert a[-1] == "slice"

    class Repr:
        def __repr__(self):
            a.append("from repr")
            return "Repr()"

    a[:] = [Repr()]
    assert repr(a) == "PagedList([Repr()])"
    assert a[-1] == "from repr"

    def iterable():
        a.append("during iteration")
        yield "yielded"

    a.extend(iterable())
    assert list(a)[-2:] == ["during iteration", "yielded"]


class ListSubclass(pc.PagedList):
    def __init__(self, name):
        self.name = name
        super().__init__([name])


class TypedListSubclass(pc.PagedListI32):
    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name
        super().__init__([1, 2])


def test_pickle_and_copy(list_cls, object_roundtrip):
    a = list_cls(data(list_cls, range(2049)))
    b = object_roundtrip(a)
    assert type(b) is list_cls
    assert list(b) == list(a)
    assert list(copy.copy(a)) == list(a)
    assert list(copy.deepcopy(a)) == list(a)


@pytest.mark.parametrize("cls", [ListSubclass, TypedListSubclass])
def test_subclass_pickle_copy_and_base_results(cls, object_roundtrip):
    a = cls("subclass")
    b = object_roundtrip(a)
    assert type(b) is cls
    assert b.name == a.name
    assert list(b) == list(a)
    base = pc.PagedList if cls is ListSubclass else pc.PagedListI32
    for result in [a.copy(), a[:], a + a, a * 2, 2 * a]:
        assert type(result) is base
    b = copy.copy(a)
    assert type(b) is cls
    assert b.name == "subclass"


def test_self_containing_pickle_deepcopy_and_shallow(object_roundtrip):
    a = pc.PagedList()
    a.append(a)
    b = object_roundtrip(a)
    assert b[0] is b
    assert a.copy()[0] is a
    assert copy.copy(a)[0] is a
    assert repr(a) == "PagedList([PagedList(...)])"
    a = ListSubclass("self")
    a.append(a)
    a.link = a
    b = object_roundtrip(a)
    assert b[-1] is b
    assert b.link is b


def test_gc_list_and_iterator_cycles():
    a = pc.PagedList()
    ref = weakref.ref(a)
    a.append(iter(a))
    del a
    gc.collect()
    assert ref() is None
    a = pc.PagedList()
    ref = weakref.ref(a)
    a.append(a)
    del a
    gc.collect()
    assert ref() is None


def test_repr_truncation(list_cls):
    a = list_cls(data(list_cls, range(25)))
    assert repr(a) == f"{list_cls.__name__}({repr(list(a)[:20])[:-1]}, ...])"
    assert repr(list_cls()) == f"{list_cls.__name__}([])"


def test_factory():
    assert type(pc.paged_list()) is pc.PagedList
    assert type(pc.paged_list(object, [1])) is pc.PagedList
    assert type(pc.paged_list("object", [1])) is pc.PagedList
    for cls in TYPED:
        a = pc.paged_list(np.dtype(cls.dtype), data(cls, [0, 1]))
        assert type(a) is cls
        assert list(a) == data(cls, [0, 1])
    with pytest.raises(TypeError, match="no paged list for dtype"):
        pc.paged_list("complex128")


def test_object_extend_keeps_partial_progress():
    a = pc.PagedList(["original"])

    def fail():
        yield "first"
        raise LookupError("iterator failed")

    with pytest.raises(LookupError, match="iterator failed"):
        a.extend(fail())
    assert list(a) == ["original", "first"]


@pytest.mark.parametrize("cls", [pc.PagedListI64, pc.PagedListI32, pc.PagedListU64, pc.PagedListU32])
def test_scalar_float_into_int_refused(cls):
    a = cls([1, 2])
    for op in [lambda: a.append(1.5), lambda: a.insert(0, 1.5),
               lambda: a.__setitem__(0, 1.5)]:
        with pytest.raises(TypeError):
            op()
        assert list(a) == [1, 2]


def test_huge_ints_match_list(list_cls):
    a = list_cls(data(list_cls, [1, 2, 3]))
    b = list(a)
    for i in [10**100, -10**100]:
        with pytest.raises(OverflowError):
            b.insert(i, b[0])
        with pytest.raises(OverflowError):
            a.insert(i, a[0])
        with pytest.raises(OverflowError):
            _ = b * i
        with pytest.raises(OverflowError):
            _ = a * i
    assert list(a) == b


def test_large_slice_edits_are_linear(list_cls):
    # Quadratic slice deletion or replacement would take minutes here.
    n = 300_000
    a = list_cls(data(list_cls, [i % 2 for i in range(n)]))
    b = list(a)
    del a[::2]
    del b[::2]
    assert list(a) == b
    a[10:10] = data(list_cls, [1] * 50_000)
    b[10:10] = list(a[10:50_010])
    assert list(a) == b
    del a[5:]
    del b[5:]
    assert list(a) == b


def test_numpy_conversion(typed_cls):
    a = typed_cls(data(typed_cls, range(3000)))
    arr = np.asarray(a)
    assert arr.dtype == np.dtype(typed_cls.dtype)
    np.testing.assert_array_equal(arr, a.to_numpy())
    np.testing.assert_array_equal(np.asarray(a, dtype=np.float64), a.to_numpy().astype(np.float64))
    if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
        with pytest.raises(ValueError):
            np.asarray(a, copy=False)
    big = np.arange(2000, dtype=np.int64)
    np.testing.assert_array_equal(a.get_many(big), a.to_numpy()[big])
    np.testing.assert_array_equal(a.get_many(big.astype(np.uint16)), a.to_numpy()[big])


class _Index:
    """An index whose ``__index__`` runs a callback first."""

    def __init__(self, value, callback):
        self.value, self.callback = value, callback

    def __index__(self):
        self.callback()
        return self.value


def test_review_reentrant_reserve_and_slice_order(list_cls):
    a = list_cls(data(list_cls, [1, 2, 3]))
    a.reserve(_Index(10, lambda: a.append(a[0])))
    assert len(a) == 4
    calls = []
    with pytest.raises(ValueError, match="slice step cannot be zero"):
        _ = a[_Index(0, lambda: calls.append(1))::0]
    assert calls == [] and len(a) == 4


def test_review_sort_reverse_int_and_reinit(list_cls):
    a = list_cls(data(list_cls, [1, 0, 1, 0]))
    b = list(a)
    a.sort(reverse=1)
    b.sort(reverse=1)
    assert list(a) == b
    with pytest.raises(TypeError):
        a.sort(reverse="yes")
    a.__init__()
    assert len(a) == 0
    assert len(a * 0) == 0 and len(a * -5) == 0


def test_review_append_into_viewed_partial_page(typed_cls):
    a = typed_cls(data(typed_cls, [1, 0, 1]))
    view = a.page_view(0)
    before = view.copy()
    for _ in range(3000):
        a.append(scalar(typed_cls, 1))
    np.testing.assert_array_equal(view, before)
    assert list(a[:3]) == list(before)


def test_review_array_dtype_property_can_mutate(typed_cls):
    a = typed_cls(data(typed_cls, [1, 0]))

    class Proxy:
        @property
        def dtype(self):
            a.append(scalar(typed_cls, 1))
            return np.dtype("float64")

    arr = np.asarray(a, dtype=np.float64)
    assert arr.dtype == np.float64
    try:
        a.__array__(dtype=Proxy())
    except TypeError:
        pass  # NumPy may refuse the proxy; it must not be a borrow error
    except RuntimeError as err:  # pragma: no cover - the bug being guarded
        pytest.fail(f"borrow held during astype: {err}")


@pytest.mark.parametrize("op", [operator.lt, operator.le, operator.gt, operator.ge])
@pytest.mark.parametrize("left,right", [([], []), ([], [1]), ([1], []), ([1], [1]),
                                        ([1], [1, 2]), ([1, 2], [1]),
                                        ([1, 3], [1, 2]), ([1, 2], [2])])
def test_ordering_matches_list(list_cls, op, left, right):
    left, right = data(list_cls, left), data(list_cls, right)
    assert op(list_cls(left), list_cls(right)) == op(left, right)


@pytest.mark.parametrize("op,name", [(operator.lt, "__lt__"), (operator.le, "__le__"),
                                     (operator.gt, "__gt__"), (operator.ge, "__ge__")])
def test_ordering_foreign_types_and_reflection(list_cls, op, name):
    value = list_cls(data(list_cls, [1]))
    others = (cls(data(cls, [1])) for cls in CLASSES if cls is not list_cls)
    for other in ([1], (1,), object(), *others):
        assert getattr(value, name)(other) is NotImplemented
        with pytest.raises(TypeError):
            op(value, other)
        with pytest.raises(TypeError):
            op(other, value)

    class Reflected:
        def __lt__(self, other):
            return "reflected"
        __le__ = __gt__ = __ge__ = __lt__

    assert op(value, Reflected()) == "reflected"


@pytest.mark.parametrize("op", [operator.lt, operator.le, operator.gt, operator.ge])
def test_ordering_subclasses_bypass_iter(list_cls, op):
    class SubList(list_cls):
        def __iter__(self):
            raise AssertionError("list ordering does not use __iter__")

    class SubBuiltin(list):
        def __iter__(self):
            raise AssertionError("list ordering does not use __iter__")

    for left, right in (([1], [1, 2]), ([2], [1]), ([1], [1])):
        left, right = data(list_cls, left), data(list_cls, right)
        assert op(SubList(left), list_cls(right)) == op(SubBuiltin(left), list(right))
        assert op(list_cls(left), SubList(right)) == op(list(left), SubBuiltin(right))


@pytest.mark.parametrize("op", [operator.lt, operator.le, operator.gt, operator.ge])
def test_object_ordering_callback_mutation_matches_list(op):
    def exercise(cls):
        left, right = cls(), cls()
        events = []

        class Value:
            def __eq__(self, other):
                events.append("eq")
                left.clear()
                right.clear()
                return False

            def __lt__(self, other):
                events.append("lt")
                left.append(8)
                return True

            def __le__(self, other):
                events.append("le")
                left.append(8)
                return True

            def __gt__(self, other):
                events.append("gt")
                left.append(8)
                return False

            def __ge__(self, other):
                events.append("ge")
                left.append(8)
                return False

        left.append(Value())
        right.append(Value())
        return op(left, right), events, list(left), list(right)

    assert exercise(pc.PagedList) == exercise(list)


def test_object_ordering_returns_element_result_and_propagates_exceptions():
    marker = object()

    class Value:
        def __lt__(self, other):
            return marker

        def __le__(self, other):
            raise LookupError("comparison failed")

    a, b = Value(), Value()
    assert (pc.PagedList([a]) < pc.PagedList([b])) is ([a] < [b]) is marker
    with pytest.raises(LookupError, match="comparison failed"):
        pc.PagedList([a]) <= pc.PagedList([b])


@pytest.mark.parametrize("cls", [pc.PagedListF64, pc.PagedListF32])
@pytest.mark.parametrize("op", [operator.lt, operator.le, operator.gt, operator.ge])
def test_typed_ordering_nan(cls, op):
    # Separate NaN objects avoid list's object-identity equality shortcut.
    assert op(cls([float("nan")]), cls([float("nan")])) == op([float("nan")], [float("nan")])
