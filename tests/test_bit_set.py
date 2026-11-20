"""BitSet's ordered, set-algebra and NumPy contracts."""

from collections.abc import MutableSet
import copy
import gc
import operator
import pickle
import random
import weakref

import numpy as np
import pytest

from paged_collections import DEFAULT_BIT_PAGE_SIZE, KEY_LIMIT, PAGE_SIZES, BitSet, SparseSet
from paged_collections import _core

MAX_KEY = KEY_LIMIT - 1


class TaggedBitSet(BitSet):
    def __init__(self, label, keys=(), *, page_size=DEFAULT_BIT_PAGE_SIZE):
        super().__init__(keys, page_size=page_size)
        self.label = label


def test_default_and_scalar_operations():
    assert DEFAULT_BIT_PAGE_SIZE == _core.DEFAULT_BIT_PAGE_SIZE == 512
    s = BitSet()
    assert isinstance(s, MutableSet)
    assert s.page_size == DEFAULT_BIT_PAGE_SIZE
    assert repr(s) == "BitSet([])"
    for query in (s.first, s.last, s.pop):
        with pytest.raises(KeyError):
            query()
    assert s.next_occupant(0) is None
    assert s.prev_occupant(MAX_KEY) is None
    assert s.add(np.uint32(MAX_KEY)) is None
    s.add(0)
    s.add(17)
    s.add(17)
    assert list(s) == [0, 17, MAX_KEY]
    assert list(reversed(s)) == [MAX_KEY, 17, 0]
    assert s.first() == s.next_occupant(0) == 0
    assert s.last() == s.prev_occupant(MAX_KEY) == MAX_KEY
    assert s.next_occupant(1) == s.prev_occupant(18) == 17
    assert s.next_occupant(17) == s.prev_occupant(17) == 17
    assert s.pop() == 0
    assert s.prev_occupant(0) is None
    assert s.remove(MAX_KEY) is None
    assert s.next_occupant(MAX_KEY) is None
    assert s.discard(19) is None
    assert s.clear() is None
    assert not s
    assert not hasattr(s, "view")
    with pytest.raises(TypeError):
        hash(s)
    with pytest.raises(ValueError, match="page_size must be one of"):
        BitSet(page_size=0)


@pytest.mark.parametrize("key", [-1, KEY_LIMIT, 2**100, 1.0, "1", [], None])
def test_scalar_validation_matches_sparse_set(key):
    with pytest.raises((OverflowError, TypeError)) as expected:
        SparseSet().add(key)
    s = BitSet([1])
    for operation in (s.add, s.toggle, s.next_occupant, s.prev_occupant):
        with pytest.raises(type(expected.value)) as error:
            operation(key)
        assert str(error.value) == str(expected.value)
    assert key not in s
    assert s.discard(key) is None
    with pytest.raises(KeyError):
        s.remove(key)
    assert list(s) == [1]


def test_toggle_and_ordered_pops_invalidate_iterators():
    s = BitSet([4, 2, MAX_KEY])
    iterator = iter(s)
    assert s.toggle(np.uint32(4)) is False
    with pytest.raises(RuntimeError, match="BitSet changed during iteration"):
        next(iterator)
    assert s.toggle(4) is True
    iterator = iter(s)
    assert s.pop_last() == MAX_KEY
    with pytest.raises(RuntimeError, match="BitSet changed during iteration"):
        next(iterator)
    assert s.pop_first() == 2
    assert s.pop() == 4
    for pop in (s.pop, s.pop_first, s.pop_last):
        with pytest.raises(KeyError, match="pop from an empty BitSet"):
            pop()


@pytest.mark.parametrize("size", [
    pytest.param(1, marks=pytest.mark.skipif(1 not in PAGE_SIZES, reason="requires page size 1")),
    DEFAULT_BIT_PAGE_SIZE,
])
def test_range_boundaries(size):
    page = size * 64
    s = BitSet([62, 63, page + 2], page_size=size)
    model = {62, 63, page + 2}
    for start, stop in ((63, 67), (page - 1, page + 3), (page, page), (page, 0)):
        keys = set(range(start, stop))
        assert s.add_range(start, stop) == len(keys - model)
        model.update(keys)
        assert s.count_range(start, stop) == len(model & keys)
        assert list(s.range(start, stop)) == sorted(model & keys)
    assert list(s) == sorted(model)
    for start, stop in ((page - 1, page + 1), (63, 66), (4, 4), (5, 2)):
        keys = set(range(start, stop))
        assert s.discard_range(start, stop) == len(model & keys)
        model.difference_update(keys)
        assert list(s) == sorted(model)


def test_range_at_u32_limit():
    s = BitSet([0])
    assert s.add_range(np.uint64(MAX_KEY - 1), KEY_LIMIT) == 2
    assert s.count_range(0, KEY_LIMIT) == 3
    assert list(s.range(MAX_KEY, KEY_LIMIT)) == [MAX_KEY]
    assert s.add_range(KEY_LIMIT, KEY_LIMIT) == 0
    assert s.count_range(KEY_LIMIT, 0) == 0
    assert list(s.range(KEY_LIMIT, 0)) == []
    assert s.discard_range(MAX_KEY, KEY_LIMIT) == 1
    assert list(s) == [0, MAX_KEY - 1]


@pytest.mark.parametrize("bound", [-1, KEY_LIMIT + 1, 2**100, 1.5, "1", None])
def test_invalid_range_bounds(bound):
    with pytest.raises((OverflowError, TypeError)) as expected:
        SparseSet().add(bound)
    s = BitSet([1, 2])
    for operation in (s.add_range, s.discard_range, s.count_range, s.range, s.to_mask):
        for args in ((bound, 0), (0, bound)):
            with pytest.raises(type(expected.value)) as error:
                operation(*args)
            assert str(error.value) == str(expected.value)
    assert list(s) == [1, 2]


@pytest.mark.parametrize("factory", [iter, reversed, lambda s: s.range(0, 10)])
def test_iteration_detects_changes_and_ignores_noops(factory):
    s = BitSet([3, 4, 5])
    it = factory(s)
    assert iter(it) is it
    next(it)
    s.add(4)
    assert s.add_range(3, 6) == 0
    assert s.discard_many([9, 10]) == 0
    s &= s.copy()
    next(it)
    s ^= BitSet([4, 6])  # Same length, different membership.
    with pytest.raises(RuntimeError, match="BitSet changed during iteration"):
        next(it)
    it = factory(s)
    list(it)
    s.add(7)
    with pytest.raises(StopIteration):
        next(it)


def check_array_batch(keys):
    s = BitSet()
    assert s.add_many(keys) == 3
    assert s.add_many(keys) == 0
    found = s.contains_many(keys)
    assert found.dtype == np.dtype(np.bool_)
    np.testing.assert_array_equal(found, [True] * 4)
    array = s.to_numpy()
    assert array.dtype == np.dtype(np.uint32)
    np.testing.assert_array_equal(array, [1, 7, 9])
    array[0] = 99
    assert list(s) == [1, 7, 9]
    assert s.discard_many(keys) == 3
    assert s.discard_many(keys) == 0


def test_array_dtypes_and_counts(key_batch):
    check_array_batch(key_batch)


def test_unaligned_and_big_endian_keys(unusual_u32_keys):
    check_array_batch(unusual_u32_keys)


def test_batch_errors_and_lookup_edges():
    s = BitSet([1, MAX_KEY])
    with pytest.raises(OverflowError, match=r"key -1 at position 1 is out of range for uint32"):
        s.add_many([2, -1, 4])
    for operation in (s.add_many, s.discard_many, s.contains_many):
        with pytest.raises(ValueError, match="one-dimensional"):
            operation(np.array([[1]]))
        with pytest.raises(TypeError, match="keys must be integers"):
            operation(np.array([1.0]))
    assert list(s) == [1, MAX_KEY]
    np.testing.assert_array_equal(
        s.contains_many([1, MAX_KEY, -1, KEY_LIMIT, 2**100]), [True, True, False, False, False]
    )
    assert s.discard_many([1, 1, -1, KEY_LIMIT]) == 1
    assert list(s) == [MAX_KEY]


@pytest.mark.parametrize("size", PAGE_SIZES)
def test_mask_roundtrip(size):
    mask = np.array([True, False, False, True, True], dtype=bool)[::-1]
    # Use the uint32 edge at the usual page size and cross a page boundary
    # at other sizes, including builds with very small pages.
    offset = KEY_LIMIT - len(mask) if size == 512 else size * 64 - 2
    stop = offset + len(mask)
    s = BitSet.from_mask(mask, np.uint64(offset), page_size=size)
    assert s.page_size == size
    assert list(s) == [offset, offset + 1, stop - 1]
    result = s.to_mask(offset, stop)
    assert result.dtype == np.dtype(bool)
    np.testing.assert_array_equal(result, mask)
    result[:] = False
    assert len(s) == 3
    assert s.to_mask(stop, 0).shape == (0,)
    empty = BitSet.from_mask([], page_size=size)
    assert not empty
    assert empty.page_size == size


def test_mask_errors_and_special_cases():
    assert list(BitSet.from_mask([False, True])) == [1]
    # NumPy permits noncanonical bool bytes, which are still true.
    assert list(BitSet.from_mask(np.array([0, 255], dtype=np.uint8).view(bool))) == [1]
    for bad_offset in (-1, KEY_LIMIT, 2**100):
        with pytest.raises(OverflowError):
            BitSet.from_mask([True], bad_offset)
    with pytest.raises(TypeError):
        BitSet.from_mask([True], None)
    with pytest.raises(OverflowError):
        BitSet.from_mask([True, False], MAX_KEY)
    with pytest.raises(TypeError, match="bool"):
        BitSet.from_mask(np.array([1], dtype=np.uint8))
    with pytest.raises(ValueError, match="one-dimensional"):
        BitSet.from_mask([[True, False]])
    with pytest.raises(TypeError):
        BitSet.from_mask([True], 0, DEFAULT_BIT_PAGE_SIZE)
    with pytest.raises(ValueError) as expected:
        BitSet(page_size=0)
    with pytest.raises(ValueError) as error:
        BitSet.from_mask([True], page_size=0)
    assert error.value.args == expected.value.args


def test_pickle_copy_and_subclass():
    size = PAGE_SIZES[-1]
    s = TaggedBitSet(["tag"], [17, 0, MAX_KEY], page_size=size)
    assert not hasattr(s, "__copy__")
    assert not hasattr(s, "__deepcopy__")
    assert weakref.ref(s)() is s
    # Bind first: pytest's assertion rewriting would keep a temporary alive.
    dead = weakref.ref(BitSet())
    gc.collect()
    assert dead() is None
    for rebuilt in (copy.copy(s), copy.deepcopy(s), pickle.loads(pickle.dumps(s))):
        assert type(rebuilt) is TaggedBitSet
        assert rebuilt is not s
        assert rebuilt.label == s.label
        assert rebuilt.page_size == size
        assert rebuilt == s
        rebuilt.remove(17)
        assert 17 in s
    assert copy.copy(s).label is s.label
    assert copy.deepcopy(s).label is not s.label
    base = s.copy()
    assert type(base) is BitSet
    assert base.page_size == size
    assert pickle.loads(pickle.dumps(base)) == base
    assert BitSet.from_mask([True]).page_size == DEFAULT_BIT_PAGE_SIZE
    from_mask = TaggedBitSet.from_mask([True], page_size=size)
    assert type(from_mask) is TaggedBitSet
    assert from_mask.page_size == size
    assert repr(s).startswith("TaggedBitSet([0, 17, 4294967295]")


@pytest.mark.parametrize("sizes", [
    (DEFAULT_BIT_PAGE_SIZE, DEFAULT_BIT_PAGE_SIZE),
    (PAGE_SIZES[0], PAGE_SIZES[-1]),
])
@pytest.mark.parametrize("left,right", [
    ([], []), ([1], []), ([], [1]), ([1], [1]), ([1], [1, 2]),
    ([1, 2], [1]), ([1, 65, 33200], [65, 66, 33201]),
])
def test_algebra_operators_against_builtin(sizes, left, right):
    a = TaggedBitSet("tag", left, page_size=sizes[0])
    b = BitSet(right, page_size=sizes[1])
    x, y = set(left), set(right)
    for op in (operator.or_, operator.and_, operator.sub, operator.xor):
        result = op(a, b)
        assert type(result) is BitSet
        assert result.page_size == a.page_size
        assert list(result) == sorted(op(x, y))
    for op in (operator.ior, operator.iand, operator.isub, operator.ixor):
        receiver = a.copy()
        assert op(receiver, b) is receiver
        assert receiver.page_size == a.page_size
        assert list(receiver) == sorted(op(x.copy(), y))
        alias = a.copy()
        assert op(alias, alias) is alias
        assert list(alias) == sorted(op(x.copy(), x.copy()))
    for op in (operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge):
        assert op(a, b) == op(x, y)
    for name in ("issubset", "issuperset", "isdisjoint"):
        assert getattr(a, name)(b) == getattr(x, name)(y)
    assert list(a) == sorted(x)
    assert list(b) == sorted(y)


@pytest.mark.parametrize("op,update", [
    ("union", "update"),
    ("intersection", "intersection_update"),
    ("difference", "difference_update"),
    ("symmetric_difference", "symmetric_difference_update"),
])
def test_algebra_iterable_methods(op, update):
    values = [9, 4, 1, 7]
    others = [[4, 8, 8], [1, 8, 9]] if op != "symmetric_difference" else [[4, 8, 8]]
    s = BitSet(values, page_size=PAGE_SIZES[0])
    reference = set(values)
    result = getattr(s, op)(*(iter(o) for o in others))
    assert result.page_size == s.page_size
    assert list(result) == sorted(getattr(reference, op)(*others))
    assert list(s) == sorted(values)
    assert getattr(s, update)(*(iter(o) for o in others)) is None
    assert getattr(reference, update)(*others) is None
    assert list(s) == sorted(reference)


def test_algebra_foreign_objects_and_reentrancy():
    s = BitSet([1, 2, 3])
    assert s != {1, 2, 3}
    for op in (operator.or_, operator.and_, operator.sub, operator.xor, operator.le):
        with pytest.raises(TypeError):
            op(s, {1})
    assert s.issuperset(iter([1.0, 3]))
    assert s.issubset(iter([1.0, 2, 3, -1, "foreign"]))
    assert not s.isdisjoint(iter([3.0]))
    assert list(s.intersection([1.0])) == [1]
    assert list(s.difference([1.0])) == [2, 3]

    def keys():
        s.add(4)
        yield 2

    with pytest.raises(RuntimeError, match="BitSet changed during set operation"):
        s.intersection_update(keys())
    assert list(s) == [1, 2, 3, 4]

    def broken():
        yield 8
        raise LookupError("iterator failed")

    with pytest.raises(LookupError, match="iterator failed"):
        s.update(broken())
    assert list(s) == [1, 2, 3, 4, 8]


def test_seeded_set_model():
    rng = random.Random(0xB175E7)
    s = BitSet()
    model = set()
    for _ in range(400):
        start = rng.choice([0, 64, DEFAULT_BIT_PAGE_SIZE * 64]) + rng.randrange(64)
        stop = start + rng.randrange(24) - 3
        stop = max(0, stop)
        data = set(range(start, stop))
        operation = rng.randrange(8)
        if operation == 0:
            s.add(start)
            model.add(start)
        elif operation == 1:
            s.discard(start)
            model.discard(start)
        elif operation == 2:
            assert s.add_range(start, stop) == len(data - model)
            model.update(data)
        elif operation == 3:
            assert s.discard_range(start, stop) == len(data & model)
            model.difference_update(data)
        elif operation == 4:
            assert s.count_range(start, stop) == len(data & model)
            assert list(s.range(start, stop)) == sorted(data & model)
        elif operation == 5 and model:
            assert s.pop() == min(model)
            model.remove(min(model))
        elif operation == 6:
            s.symmetric_difference_update(iter(data))
            model.symmetric_difference_update(data)
        else:
            assert (start in s) == (start in model)
        assert len(s) == len(model)
        assert list(s) == sorted(model)
        assert list(reversed(s)) == sorted(model, reverse=True)
