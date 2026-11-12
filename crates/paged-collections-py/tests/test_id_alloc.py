"""Generational id allocation, transactional batches and reconstruction."""

import copy
import gc
import operator
import pickle
import sys
import weakref

import numpy as np
import pytest

import paged_collections as pc
from paged_collections import (
    MAX_GENERATION,
    MAX_INDICES,
    IdAllocator,
    id_generation,
    id_index,
    id_indices,
    make_id,
)


def test_empty_scalar_operations_and_exports():
    assert MAX_GENERATION == 2**31
    assert MAX_INDICES == 2**32 - 1
    assert IdAllocator.__module__ == "paged_collections._core"
    for name in ("IdAllocator", "MAX_GENERATION", "MAX_INDICES", "id_index",
                 "id_generation", "make_id", "id_indices"):
        assert name in pc.__all__
    assert "IdAllocatorIterator" not in pc.__all__
    assert not hasattr(pc, "IdAllocatorIterator")
    a = IdAllocator()
    assert not a and len(a) == 0
    assert (a.index_bound, a.free_len, a.retired_len) == (0, 0, 0)
    assert list(a) == [] and a.occupant(0) is None
    assert repr(a) == "IdAllocator(len=0, index_bound=0)"
    first, second = a.alloc(), a.alloc()
    assert type(first) is int and type(second) is int
    assert (first, second) == (make_id(0, 1), make_id(1, 1))
    assert a and len(a) == 2
    assert first in a and a.is_alive(first)
    assert a.occupant(0) == first and a.occupant(1) == second
    assert a.free(first) is True
    assert a.free(first) is False
    assert first not in a and not a.is_alive(first)
    assert a.occupant(0) is None
    assert (a.index_bound, a.free_len, a.retired_len) == (2, 1, 0)
    assert repr(a) == "IdAllocator(len=1, index_bound=2)"
    for name in ("index_bound", "free_len", "retired_len"):
        with pytest.raises(AttributeError):
            setattr(a, name, 7)
    assert IdAllocator.__hash__ is None
    with pytest.raises(TypeError):
        hash(a)


def test_lifo_reuse_generations_and_stale_ids():
    a = IdAllocator()
    ids = a.alloc_many(5).tolist()
    assert a.free(ids[1]) and a.free(ids[3])
    assert a.alloc() == make_id(3, 2)
    assert a.alloc() == make_id(1, 2)
    assert a.alloc() == make_id(5, 1)
    for stale in (ids[1], ids[3], make_id(1, 1), make_id(1, 3)):
        assert stale not in a and not a.free(stale)
    assert list(a) == [make_id(i, 2 if i in (1, 3) else 1) for i in range(6)]
    assert a.free_many([make_id(1, 2), make_id(3, 2)]) == 2
    np.testing.assert_array_equal(a.alloc_many(3), [make_id(3, 3), make_id(1, 3), make_id(6, 1)])


@pytest.mark.parametrize("id", [-2**200, -1, 0, 1, 2**32 - 1, 2**64, 2**200])
def test_invalid_bits_are_not_alive(id):
    a = IdAllocator()
    a.alloc()
    assert not a.is_alive(id) and id not in a
    assert not a.free(id)
    assert a.free_many([id]) == 0
    np.testing.assert_array_equal(a.is_alive_many([id]), [False])
    for helper in (id_index, id_generation, id_indices):
        with pytest.raises(ValueError, match="invalid id bits"):
            helper([id] if helper is id_indices else id)
    assert len(a) == 1


class IndexOnly:
    def __index__(self):
        return make_id(0, 1)


@pytest.mark.parametrize("bad", [1.0, "1", None, [], object(), IndexOnly(), np.uint64(2**32)])
def test_scalar_type_checks(bad):
    a = IdAllocator()
    a.alloc()
    assert bad not in a
    for method in (a.free, a.is_alive, a.occupant, id_index, id_generation):
        with pytest.raises(TypeError):
            method(bad)
    for args in ((bad, 1), (0, bad)):
        with pytest.raises(TypeError):
            make_id(*args)
    assert len(a) == 1


def test_int_subclass_overrides_are_ignored():
    class I(int):
        def __lt__(self, other):
            raise AssertionError("subclass comparison called")

        def __index__(self):
            raise AssertionError("subclass index called")

        def __int__(self):
            raise AssertionError("subclass conversion called")

    a = IdAllocator()
    id = a.alloc()
    assert a.is_alive(I(id)) and I(id) in a
    assert a.occupant(I(0)) == id
    assert a.occupant(I(-1)) is None
    assert not a.is_alive(I(-1)) and not a.free(I(2**200))
    assert id_index(I(id)) == 0 and id_generation(I(id)) == 1
    assert make_id(I(0), I(1)) == id
    np.testing.assert_array_equal(id_indices([I(id)]), [0])
    assert a.free_many([I(-1), I(id), I(2**200)]) == 1
    assert a.alloc_many(I(0)).size == 0
    with pytest.raises(ValueError):
        a.alloc_many(I(-1))


@pytest.mark.parametrize("index", [-2**200, -1, 2**32 - 1, 2**32, 2**200])
def test_occupant_outside_allocated_indices(index):
    a = IdAllocator()
    a.alloc()
    assert a.occupant(index) is None


@pytest.mark.parametrize("index", [0, 1, 2**32 - 1])
@pytest.mark.parametrize("generation", [1, MAX_GENERATION, MAX_GENERATION + 1, 2**32 - 1])
def test_id_parts(index, generation):
    id = make_id(index, generation)
    assert id == generation << 32 | index
    assert id_index(id) == index and id_generation(id) == generation
    result = id_indices(np.array([id, id], dtype=np.uint64))
    assert result.dtype == np.dtype(np.uint32) and result.shape == (2,)
    np.testing.assert_array_equal(result, [index, index])
    assert not IdAllocator().is_alive(id)


@pytest.mark.parametrize("index, generation", [(-1, 1), (2**32, 1), (2**200, 1),
                                             (0, -1), (0, 0), (0, 2**32), (0, 2**200)])
def test_make_id_rejects_bad_parts(index, generation):
    with pytest.raises(ValueError):
        make_id(index, generation)


def test_alloc_many_validation_and_exhaustion_precheck():
    a = IdAllocator()
    assert a.alloc_many(0).dtype == np.dtype(np.uint64)
    assert a.index_bound == 0
    a.alloc_many(4)
    a.free(make_id(1, 1))
    before = a.copy()
    for n in (-1, -2**200):
        with pytest.raises(ValueError):
            a.alloc_many(n)
        assert a == before
    # Exceeds the exact available count, without constructing billions of slots.
    available = a.free_len + (MAX_INDICES - a.index_bound)
    for n in (available + 1, MAX_INDICES + 1, 2**200):
        with pytest.raises(OverflowError, match="^IdAllocator is exhausted$"):
            a.alloc_many(n)
        assert a == before
    for bad in (1.0, "1", None, IndexOnly()):
        with pytest.raises(TypeError):
            a.alloc_many(bad)
        assert a == before
    empty = a.alloc_many(0)
    assert empty.shape == (0,) and a == before
    assert a.alloc_many(2).tolist() == [make_id(1, 2), make_id(4, 1)]


def test_alloc_many_precheck_accounts_for_retired_indices():
    a = IdAllocator(([0, 1, 2, 0], [2]))
    before = a.copy()
    available = a.free_len + (MAX_INDICES - a.index_bound)
    with pytest.raises(OverflowError, match="^IdAllocator is exhausted$"):
        a.alloc_many(available + 1)
    assert a == before


def test_batches_iterables_duplicates_and_arbitrary_ints():
    a = IdAllocator()
    ids = a.alloc_many(3).tolist()
    batch = [ids[1], -2**200, ids[1], 2**64, 0, ids[0], 2**200]
    alive = a.is_alive_many(iter(batch))
    assert alive.dtype == np.dtype(bool)
    np.testing.assert_array_equal(alive, [True, False, True, False, False, True, False])
    assert a.free_many(x for x in batch) == 2
    assert list(a) == [ids[2]]
    assert a.free_many(batch) == 0
    assert a.free_many([]) == 0
    assert a.is_alive_many([]).shape == (0,)
    assert id_indices(iter([ids[0], ids[2]])).tolist() == [0, 2]
    assert id_indices([]).dtype == np.dtype(np.uint32)


@pytest.mark.parametrize("dtype", ["<u8", ">u8", "<i8", ">i8"])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "reversed", "unaligned"])
def test_numpy_id_batches(dtype, layout):
    a = IdAllocator()
    first, second = a.alloc_many(2).tolist()
    expected = [second, first, second, 0]
    if layout == "strided":
        batch = np.array([x for id in expected for x in (id, 99)], dtype=dtype)[::2]
    elif layout == "reversed":
        batch = np.array(expected[::-1], dtype=dtype)[::-1]
    elif layout == "unaligned":
        batch = np.frombuffer(bytearray(33), dtype=dtype, offset=1, count=4)
        batch[:] = expected
        assert not batch.flags.aligned
    else:
        batch = np.array(expected, dtype=dtype)
    np.testing.assert_array_equal(a.is_alive_many(batch), [True, True, True, False])
    assert id_indices(batch[:3]).tolist() == [1, 0, 1]
    assert a.free_many(batch) == 2
    assert not a


def test_numpy_signed_negative_unsigned_max_and_subclass():
    a = IdAllocator()
    id = a.alloc()
    np.testing.assert_array_equal(a.is_alive_many(np.array([-1, id], dtype=np.int64)), [False, True])
    # All nonzero u32 generations are valid id bits, even above MAX_GENERATION.
    last = np.array([2**64 - 1], dtype=np.uint64)
    assert id_indices(last).tolist() == [2**32 - 1]
    assert a.free_many(last) == 0

    class Hostile(np.ndarray):
        def astype(self, *args, **kwargs):
            raise AssertionError("array subclass conversion called")

    batch = np.array([id, id], dtype=np.int64).view(Hostile)
    assert a.is_alive_many(batch).tolist() == [True, True]
    assert a.free_many(batch) == 1


@pytest.mark.parametrize("method", ["free_many", "is_alive_many"])
@pytest.mark.parametrize("bad, error", [([1.0], TypeError), ([None], TypeError),
                                      (np.array([1.0]), TypeError),
                                      (np.array([True]), TypeError),
                                      (np.array([1], dtype=object), TypeError),
                                      (np.array([[1]], dtype=np.int64), ValueError),
                                      (np.empty((0, 2), dtype=np.uint64), ValueError),
                                      (np.array(1, dtype=np.int64), ValueError)])
def test_batch_validation(method, bad, error):
    a = IdAllocator()
    a.alloc_many(2)
    before = a.copy()
    with pytest.raises(error):
        getattr(a, method)(bad)
    assert a == before
    with pytest.raises(error):
        id_indices(bad)


def test_failed_batch_is_atomic():
    a = IdAllocator()
    ids = a.alloc_many(3).tolist()
    before = a.copy()
    it = iter(a)
    for bad in ([ids[0], ids[1], "bad"], iter([ids[0], object()])):
        with pytest.raises(TypeError):
            a.free_many(bad)
        assert a == before

    def broken():
        yield ids[0]
        raise RuntimeError("broken iterable")

    with pytest.raises(RuntimeError, match="broken iterable"):
        a.free_many(broken())
    assert a == before
    assert list(it) == ids


def test_ids_arrays_are_detached_and_in_index_order():
    a = IdAllocator()
    allocated = a.alloc_many(4)
    assert allocated.dtype == np.dtype(np.uint64)
    allocated[0] = 0
    a.free(make_id(1, 1))
    a.alloc()
    ids = a.ids()
    assert ids.dtype == np.dtype(np.uint64)
    assert ids.tolist() == list(a) == [make_id(0, 1), make_id(1, 2), make_id(2, 1), make_id(3, 1)]
    ids[:] = 0
    a.clear()
    assert allocated.tolist()[1:] == [make_id(i, 1) for i in range(1, 4)]
    assert a.ids().shape == (0,)


def test_clear_reserve_shrink_and_reinit():
    a = IdAllocator()
    assert a.reserve(10) is None
    ids = a.alloc_many(5).tolist()
    it = iter(a)
    assert a.reserve(10) is None and a.shrink_to_fit() is None
    assert list(it) == ids
    assert a.clear() is None
    assert (len(a), a.index_bound, a.free_len, a.retired_len) == (0, 5, 5, 0)
    a.shrink_to_fit()
    assert a.alloc() == make_id(4, 2)
    assert not any(a.is_alive(id) for id in ids)
    assert a.__init__() is None
    assert a == IdAllocator()
    assert (a.index_bound, a.free_len, a.retired_len) == (0, 0, 0)
    assert a.alloc() == ids[0]  # Reinitializing starts a new allocation history.
    obj = IdAllocator.__new__(IdAllocator, "ignored", arbitrary=True)
    assert not obj and obj.index_bound == 0


@pytest.mark.parametrize("additional", [sys.maxsize, sys.maxsize // 4])
def test_reserve_failure_preserves_state(additional):
    a = IdAllocator()
    a.alloc_many(3)
    before = a.copy()
    with pytest.raises(MemoryError):
        a.reserve(additional)
    assert a == before


@pytest.mark.parametrize("change", ["alloc", "alloc_many", "free", "free_many",
                                   "retain", "clear", "init", "raw_state", "same_size"])
def test_iterator_invalidation(change):
    a = IdAllocator()
    ids = a.alloc_many(3).tolist()
    it = iter(a)
    assert type(it).__name__ == "IdAllocatorIterator" and iter(it) is it
    assert operator.length_hint(it) == 3 and next(it) == ids[0]
    assert operator.length_hint(it) == 2
    if change == "alloc":
        a.alloc()
    elif change == "alloc_many":
        a.alloc_many(2)
    elif change == "free":
        a.free(ids[1])
    elif change == "free_many":
        a.free_many([ids[1], ids[1]])
    elif change == "retain":
        a.retain(lambda id: id != ids[1])
    elif change == "clear":
        a.clear()
    elif change == "init":
        a.__init__()
    elif change == "raw_state":
        a.__init__(([1, 1, 1], []))
    else:
        a.free(ids[1])
        a.alloc()
        assert len(a) == 3
    with pytest.raises(RuntimeError, match="^IdAllocator changed during iteration$"):
        next(it)


def test_iterator_noops_and_sticky_exhaustion():
    a = IdAllocator()
    ids = a.alloc_many(2).tolist()
    it = iter(a)
    assert not a.free(-1) and a.free_many([-1, 0]) == 0
    a.retain(lambda id: True)
    a.alloc_many(0)
    a.reserve(20)
    a.shrink_to_fit()
    assert list(it) == ids
    assert operator.length_hint(it) == 0
    a.clear()
    a.alloc()
    assert list(it) == []
    it = iter(a)
    next(it)  # Yielding the last id has not observed exhaustion yet.
    a.clear()
    with pytest.raises(RuntimeError):
        next(it)
    empty = iter(a)
    with pytest.raises(StopIteration):
        next(empty)
    a.alloc()
    assert list(empty) == []
    a.clear()
    empty = iter(a)
    a.clear()  # clear and __init__ always invalidate, including when empty.
    with pytest.raises(RuntimeError):
        next(empty)


def test_retain_order_truthiness_and_atomic_failure():
    a = IdAllocator()
    ids = a.alloc_many(5).tolist()
    a.free(ids[1])
    a.alloc()
    before = a.copy()
    visited = []

    def broken(id):
        visited.append(id)
        assert a.is_alive(id) and a.occupant(id_index(id)) == id
        if id_index(id) == 3:
            raise LookupError("predicate failed")
        return False

    with pytest.raises(LookupError, match="predicate failed"):
        a.retain(broken)
    assert a == before
    assert visited == list(a)[:4]

    class BadTruth:
        def __bool__(self):
            raise ValueError("truth failed")

    with pytest.raises(ValueError, match="truth failed"):
        a.retain(lambda id: BadTruth())
    assert a == before
    visited.clear()

    def keep(id):
        visited.append(id)
        return [1] if id_index(id) % 2 == 0 else []

    assert a.retain(keep) is None
    assert visited == list(before)
    assert list(a) == [ids[0], ids[2], ids[4]]
    assert a.alloc() == make_id(3, 2)  # Frees were committed in index order.
    assert a.alloc() == make_id(1, 3)
    assert IdAllocator().retain(lambda id: pytest.fail("empty predicate called")) is None


@pytest.mark.parametrize("mutation", ["alloc", "replace", "clear", "init", "truth"])
def test_retain_detects_reentrant_mutation(mutation):
    a = IdAllocator()
    ids = a.alloc_many(3).tolist()
    after_mutation = []

    def mutate():
        if mutation == "replace":
            a.free(ids[1])
            a.alloc()
        elif mutation == "clear":
            a.clear()
        elif mutation == "init":
            a.__init__()
        else:
            a.alloc()
        after_mutation.append(a.copy())

    class MutatingTruth:
        def __bool__(self):
            mutate()
            return False

    def predicate(id):
        if id == ids[1]:
            if mutation == "truth":
                return MutatingTruth()
            mutate()
        return False

    with pytest.raises(RuntimeError, match="^IdAllocator changed during retain$"):
        a.retain(predicate)
    # Callback effects remain; retain applies none of its own pending frees.
    assert a == after_mutation[0]


def test_retain_allows_nonstructural_callback_operations():
    a = IdAllocator()
    a.alloc_many(3)
    before = a.copy()

    def predicate(id):
        a.reserve(10)
        a.shrink_to_fit()
        assert not a.free(0) and a.free_many([]) == 0
        return a.is_alive(id)

    a.retain(predicate)
    assert a == before


def test_retirement_raw_parts_and_roundtrip(object_roundtrip):
    a = IdAllocator(_state=(np.array([2**32 - 1, 2, 0], dtype=np.uint32),
                            np.array([1], dtype=np.uint32)))
    last = make_id(0, MAX_GENERATION)
    assert list(a) == [last]
    assert (a.index_bound, a.free_len, a.retired_len) == (3, 1, 1)
    a = object_roundtrip(a)
    assert a.free(last)
    assert (a.index_bound, a.free_len, a.retired_len) == (3, 1, 2)
    assert a.occupant(0) is None and not a.is_alive(last)
    assert a.alloc() == make_id(1, 2)
    assert a.alloc() == make_id(3, 1)
    assert a.clear() is None
    a.shrink_to_fit()
    assert a.retired_len == 2 and a.index_bound == 4
    assert a == object_roundtrip(a)


def test_pickle_rebuild_uses_detached_raw_parts():
    from paged_collections._core import _rebuild

    a = IdAllocator(([1, 4, 0, 2], [3, 1]))
    rebuild, args, state = a.__reduce__()
    assert rebuild is _rebuild and state is None
    cls, base, (versions, free) = args
    assert cls is IdAllocator and base is IdAllocator
    assert versions.dtype == free.dtype == np.dtype(np.uint32)
    assert versions.tolist() == [1, 4, 0, 2] and free.tolist() == [3, 1]
    restored = rebuild(*args)
    assert restored == a
    assert restored.alloc_many(2).tolist() == [make_id(1, 3), make_id(3, 2)]
    versions[:] = 0
    free[:] = 0
    assert a == IdAllocator(([1, 4, 0, 2], [3, 1]))
    with pytest.raises(ValueError, match="free list misses a vacant index"):
        _rebuild(IdAllocator, IdAllocator, ([2], []))


@pytest.mark.parametrize("state, message", [
    (([2], []), "free list misses a vacant index"),
    (([2], [0, 0]), "free list holds an index twice"),
    (([1], [0]), "free list holds an index that is not vacant"),
    (([0], [0]), "free list holds an index that is not vacant"),
    (([2], [1]), "free list holds an index that is not vacant"),
    (([2], [2**32 - 1]), "free list holds an index that is not vacant"),
    (([], [0]), "free list holds an index that is not vacant"),
    (([-1], []), None), (([2**32], []), None), (([2], [-1]), None),
    (([],), None), (([], [], []), None), ([], None), (1, None),
])
def test_invalid_raw_state_is_valueerror_and_atomic(state, message):
    a = IdAllocator()
    id = a.alloc()
    it = iter(a)
    with pytest.raises(ValueError, match=message):
        a.__init__(state)
    assert list(it) == [id] and a.is_alive(id)
    with pytest.raises(ValueError, match=message):
        IdAllocator(_state=state)


def test_equality_compares_history_and_free_order():
    a, b = IdAllocator(), IdAllocator()
    assert a == b and a.__eq__([]) is NotImplemented
    assert a != [] and a != object()
    ids = a.alloc_many(3).tolist()
    b.alloc_many(3)
    assert a == b
    a.free_many(ids[:2])
    b.free_many(ids[1::-1])
    assert list(a) == list(b) and a != b
    assert a.alloc() != b.alloc()
    a.clear()
    assert a != IdAllocator()  # Both empty; allocation histories differ.
    assert IdAllocator(([1], [])) != IdAllocator(([3], []))


@pytest.mark.parametrize("method", ["copy", "shallow", "deep", "pickle"])
def test_copy_roundtrip(method):
    a = IdAllocator()
    ids = a.alloc_many(5).tolist()
    a.free_many([ids[3], ids[0], ids[2]])
    clone = {"copy": lambda x: x.copy(), "shallow": copy.copy,
             "deep": copy.deepcopy, "pickle": lambda x: pickle.loads(pickle.dumps(x))}[method](a)
    assert type(clone) is IdAllocator and clone is not a and clone == a
    assert list(clone) == list(a)
    assert clone.alloc_many(4).tolist() == a.alloc_many(4).tolist()
    clone.clear()
    assert len(a) == 6


class LabelledAllocator(IdAllocator):
    def __init__(self, label):
        super().__init__()
        self.label = label


class SlottedAllocator(IdAllocator):
    __slots__ = ("tag",)


def test_subclass_copy_and_pickle(object_roundtrip):
    a = LabelledAllocator("tag")
    a.extra = ["state"]
    ids = a.alloc_many(3).tolist()
    a.free(ids[1])
    restored = object_roundtrip(a)
    assert type(restored) is LabelledAllocator and restored == a
    assert restored.label == "tag" and restored.extra == ["state"]
    assert repr(a) == "LabelledAllocator(len=2, index_bound=3)"
    assert type(a.copy()) is IdAllocator and a.copy() == a
    shallow = copy.copy(a)
    assert type(shallow) is LabelledAllocator and shallow == a
    assert shallow.extra is a.extra
    deep = copy.deepcopy(a)
    assert type(deep) is LabelledAllocator and deep == a
    assert deep.extra == a.extra and deep.extra is not a.extra


@pytest.mark.skipif(sys.version_info < (3, 11), reason="object.__getstate__ covers slots from 3.11")
def test_subclass_slots(object_roundtrip):
    a = SlottedAllocator()
    a.alloc_many(2)
    a.tag = [7]
    restored = object_roundtrip(a)
    assert type(restored) is SlottedAllocator and restored == a and restored.tag == [7]


def test_subclass_cycles_and_deepcopy_memo(object_roundtrip):
    a = LabelledAllocator("loop")
    a.alloc()
    a.self = a
    a.extra = [a]
    restored = object_roundtrip(a)
    assert restored is not a and restored == a
    assert restored.self is restored and restored.extra[0] is restored
    shallow = copy.copy(a)
    assert shallow.self is a and shallow.extra is a.extra
    memo = {}
    deep = a.__deepcopy__(memo)
    assert memo[id(a)] is deep and deep.self is deep


def test_weakrefs_and_iterator_cycles_are_collectable():
    a = IdAllocator()
    ref = weakref.ref(a)
    it = iter(a)
    del a
    gc.collect()
    assert ref() is not None  # Iterator keeps its owner alive.
    del it
    gc.collect()
    assert ref() is None
    a = LabelledAllocator("cycle")
    a.iterator = iter(a)
    ref = weakref.ref(a)
    del a
    gc.collect()
    assert ref() is None


def test_batches_may_read_or_be_the_allocator():
    a = IdAllocator()
    ids = a.alloc_many(3).tolist()
    assert a.is_alive_many(a).tolist() == [True, True, True]
    assert a.is_alive_many(id for id in ids if a.is_alive(id)).tolist() == [True] * 3
    assert a.free_many(id for id in ids[:1] if a.is_alive(id)) == 1
    assert a.free_many(a) == 2
    assert len(a) == 0


def test_free_many_generator_mutating_the_allocator_changes_nothing_else():
    a = IdAllocator()
    ids = a.alloc_many(2).tolist()

    def batch():
        a.free(ids[0])
        yield ids[0]
        yield ids[1]

    assert a.free_many(batch()) == 1
    assert len(a) == 0
