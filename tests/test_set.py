"""SparseSet's scalar, batch, ordering and iterator contracts."""

import copy
import pickle
import random
import sys

import numpy as np
import pytest

from paged_collections import KEY_LIMIT, SparseSet


def test_empty_and_scalar_operations():
    assert KEY_LIMIT == 2**32
    s = SparseSet()
    assert len(s) == 0
    assert not s
    assert list(s) == []
    assert s.add(4) is None
    s.add(np.int64(2))
    s.add(np.uint32(4))
    assert len(s) == 2
    assert 4 in s and np.int16(2) in s
    assert list(s) == [4, 2]
    assert s.discard(123) is None
    assert s.remove(4) is None
    assert list(s) == [2]
    assert s.pop() == 2
    with pytest.raises(KeyError):
        s.pop()
    with pytest.raises(KeyError):
        s.remove(2)
    assert s.clear() is None


@pytest.mark.parametrize("key", [-1, 2**32, 2**63, 2**100])
def test_out_of_range_scalar_keys(key):
    s = SparseSet([1])
    assert key not in s
    s.discard(key)
    with pytest.raises(KeyError):
        s.remove(key)
    with pytest.raises(OverflowError):
        s.add(key)
    assert list(s) == [1]


@pytest.mark.parametrize("key", [1.0, "1", None, [], object()])
def test_non_integer_scalar_keys(key):
    s = SparseSet([1])
    assert key not in s
    s.discard(key)
    with pytest.raises(KeyError):
        s.remove(key)
    with pytest.raises(TypeError):
        s.add(key)
    assert list(s) == [1]


def test_key_boundaries():
    # At most two pages plus the sparse directory, even at the u32 limit.
    s = SparseSet([0, 2**32 - 1])
    assert list(s) == [0, 2**32 - 1]
    np.testing.assert_array_equal(s.contains_many([0, 2**32 - 1, 2**32]),
                                  [True, True, False])
    assert s.discard_many([2**32 - 1, -1, 2**32]) == 1
    assert list(s) == [0]


def test_integer_batch_formats(key_batch):
    s = SparseSet(key_batch)
    assert list(s) == [7, 1, 9]
    result = s.contains_many(key_batch)
    assert result.dtype == np.dtype(bool)
    np.testing.assert_array_equal(result, [True] * 4)
    assert s.add_many(key_batch) == 0
    assert s.discard_many(key_batch) == 3
    assert not s
    assert s.add_many(key_batch) == 3
    assert list(s) == [7, 1, 9]


def test_unaligned_and_big_endian_keys(unusual_u32_keys):
    keys = unusual_u32_keys
    aligned = np.array(keys, dtype=np.uint32, copy=True)
    assert aligned.flags.aligned
    s, reference = SparseSet([2]), SparseSet([2])
    assert s.add_many(keys) == reference.add_many(aligned) == 3
    assert list(s) == list(reference) == [2, 7, 1, 9]
    np.testing.assert_array_equal(s.contains_many(keys), reference.contains_many(aligned))
    assert s.discard_many(keys) == reference.discard_many(aligned) == 3
    assert list(s) == list(reference) == [2]
    np.testing.assert_array_equal(s.contains_many(keys), [False] * 4)


def test_key_array_subclass_astype_is_ignored(hostile_array):
    s = SparseSet()
    assert s.add_many(hostile_array) == 3
    assert list(s) == [7, 1, 9]
    np.testing.assert_array_equal(s.contains_many(hostile_array), [True] * 4)
    assert s.discard_many(hostile_array) == 3
    assert not s


@pytest.mark.parametrize("additional", [sys.maxsize, 2**62])
def test_huge_reserve_raises_memoryerror_without_changing_set(additional):
    s = SparseSet([9, 2, 7])
    pages = s.allocated_pages
    with pytest.raises(MemoryError):
        s.reserve(additional)
    assert list(s) == [9, 2, 7]
    assert s.allocated_pages == pages
    s.add(11)
    s.remove(2)
    assert list(s) == [9, 11, 7]


def test_sequence_and_empty_batches():
    s = SparseSet((5, 2, 5))
    assert list(s) == [5, 2]
    assert s.add_many([]) == 0
    assert s.discard_many([]) == 0
    result = s.contains_many([])
    assert result.shape == (0,)
    assert result.dtype == np.dtype(bool)
    assert list(SparseSet([])) == []


@pytest.mark.parametrize(
    "keys, error",
    [
        ([2, -1], OverflowError),
        (np.array([2, 2**32], dtype=np.uint64), OverflowError),
        ([2.0, 1.0], TypeError),
        (np.array([True, False]), TypeError),
        (np.array([2, 1], dtype=object), TypeError),
        ([[2, 1]], ValueError),
        (np.empty((0, 2), dtype=np.int32), ValueError),
        (np.array(2, dtype=np.int32), ValueError),
    ],
)
def test_batch_insert_validation_is_atomic(keys, error):
    s = SparseSet([8, 1])
    before = s.to_array()
    pages = s.allocated_pages
    with pytest.raises(error):
        s.add_many(keys)
    np.testing.assert_array_equal(s.to_array(), before)
    assert s.allocated_pages == pages
    with pytest.raises(error):
        SparseSet(keys)


@pytest.mark.parametrize("method", ["contains_many", "discard_many"])
@pytest.mark.parametrize(
    "keys, error",
    [
        (np.array([1.0]), TypeError),
        (np.array([True]), TypeError),
        (np.array([1.0], dtype=object), TypeError),
        (np.array([[1]], dtype=np.int32), ValueError),
    ],
)
def test_batch_lookup_rejects_wrong_shape_or_kind(method, keys, error):
    s = SparseSet([1, 2])
    with pytest.raises(error):
        getattr(s, method)(keys)
    assert list(s) == [1, 2]


def test_batch_lookup_ignores_out_of_range_and_counts_duplicates_once():
    s = SparseSet([1, 2, 3])
    keys = np.array([1, -1, 1, 2**32, 9, 3], dtype=np.int64)
    np.testing.assert_array_equal(s.contains_many(keys),
                                  [True, False, True, False, False, True])
    assert s.discard_many(keys) == 2
    assert list(s) == [2]
    assert s.discard_many(np.array([2**64 - 1], dtype=np.uint64)) == 0


@pytest.mark.parametrize("huge", [2**63, 2**64, 2**200, -2**200])
@pytest.mark.parametrize("form", ["list", "tuple", "object", "object-strided"])
def test_lookup_batches_with_arbitrary_size_ints(huge, form):
    keys = [1, huge, 1, 3]
    if form == "tuple":
        keys = tuple(keys)
    elif form == "object":
        keys = np.array(keys, dtype=object)
    elif form == "object-strided":
        keys = np.array([x for key in keys for x in (key, None)], dtype=object)[::2]
    s = SparseSet([1, 2])
    np.testing.assert_array_equal(s.contains_many(keys), [True, False, True, False])
    with pytest.raises((OverflowError, TypeError)):
        s.add_many(keys)
    assert sorted(s) == [1, 2]
    assert s.discard_many(keys) == 1
    assert list(s) == [2]


@pytest.mark.parametrize("bad", [1.5, None, "1", object()])
@pytest.mark.parametrize("array", [False, True])
def test_lookup_batch_fallback_rejects_nonintegers_atomically(bad, array):
    keys = [1, 2**200, bad]
    if array:
        keys = np.array(keys, dtype=object)
    s = SparseSet([1, 2])
    for method in (s.contains_many, s.discard_many):
        with pytest.raises(TypeError):
            method(keys)
        assert list(s) == [1, 2]


def test_dense_order_swap_removal_sort_and_detached_array():
    s = SparseSet([9, 2, 7, 4])
    s.remove(2)
    assert list(s) == [9, 4, 7]
    s.add(2)
    assert list(s) == [9, 4, 7, 2]
    assert s.pop() == 2
    array = s.to_array()
    assert array.dtype == np.dtype(np.uint32)
    assert array.shape == (3,)
    array[0] = 100
    assert list(s) == [9, 4, 7]
    assert s.sort() is None
    assert list(s) == [4, 7, 9]
    s.add(1)  # A detached array does not pin the set.
    assert array.tolist() == [100, 4, 7]


def test_pages_clear_reuse_and_shrink():
    s = SparseSet()
    assert s.allocated_pages == 0
    s.add_many([0, 1023, 1024, 2048])
    assert s.allocated_pages == 3
    s.remove(2048)
    assert s.allocated_pages == 3
    s.clear()
    assert not s
    assert s.allocated_pages == 3
    s.add(1024)
    assert s.allocated_pages == 3
    s.clear()
    s.shrink_to_fit()
    assert s.allocated_pages == 0
    s.reserve(20)
    s.add(3)
    assert list(s) == [3]


@pytest.mark.parametrize("method", ["copy", "shallow", "deep", "pickle"])
def test_copy_and_pickle_roundtrip(method):
    s = SparseSet([8, 1, 4])
    if method == "copy":
        clone = s.copy()
    elif method == "shallow":
        clone = copy.copy(s)
    elif method == "deep":
        clone = copy.deepcopy(s)
    else:
        clone = pickle.loads(pickle.dumps(s))
    assert type(clone) is SparseSet
    assert clone is not s
    assert clone == s
    assert list(clone) == [8, 1, 4]
    clone.remove(1)
    assert list(s) == [8, 1, 4]


def test_equality_hash_and_repr():
    assert SparseSet([8, 1, 4]) == SparseSet([4, 8, 1])
    assert SparseSet([1]) != SparseSet([2])
    assert SparseSet([1]) != SparseSet([1, 2])
    with pytest.raises(TypeError):
        hash(SparseSet())
    assert repr(SparseSet()) == "SparseSet([])"
    assert repr(SparseSet(range(20))) == f"SparseSet({list(range(20))!r})"
    assert repr(SparseSet(range(21))) == (
        "SparseSet([" + ", ".join(map(str, range(20))) + ", ...])"
    )


@pytest.mark.parametrize("change", ["insert", "remove", "clear", "sort", "batch_add", "batch_remove"])
def test_iterator_detects_structural_change(change):
    s = SparseSet([9, 2, 7])
    it = iter(s)
    assert iter(it) is it
    assert next(it) == 9
    if change == "insert":
        s.add(5)
    elif change == "remove":
        s.remove(2)
    elif change == "clear":
        s.clear()
    elif change == "sort":
        s.sort()
    elif change == "batch_add":
        s.add_many([2, 5])
    else:
        s.discard_many([2])
    with pytest.raises(RuntimeError):
        next(it)


def test_iterator_noops_and_exhaustion():
    s = SparseSet([9, 2])
    it = iter(s)
    assert next(it) == 9
    s.add(2)
    s.discard(100)
    assert s.add_many([9, 2]) == 0
    assert s.discard_many([100]) == 0
    assert next(it) == 2
    with pytest.raises(StopIteration):
        next(it)
    s.add(7)
    for _ in range(2):
        with pytest.raises(StopIteration):
            next(it)


def test_iterator_detects_replacement_even_when_length_is_unchanged():
    s = SparseSet([9, 2, 7])
    it = iter(s)
    assert next(it) == 9
    s.remove(2)
    s.add(5)
    assert len(s) == 3
    with pytest.raises(RuntimeError):
        next(it)


def test_empty_iterator_stays_exhausted():
    s = SparseSet()
    it = iter(s)
    with pytest.raises(StopIteration):
        next(it)
    s.add(1)
    with pytest.raises(StopIteration):
        next(it)


def test_seeded_set_model():
    rng = random.Random(0x5E7)
    s = SparseSet()
    model = set()
    # Also model dense order to check swap removal and pop, not just membership.
    dense = []

    def insert(key):
        if key not in model:
            model.add(key)
            dense.append(key)

    def remove(key):
        if key in model:
            model.remove(key)
            pos = dense.index(key)
            dense[pos] = dense[-1]
            dense.pop()

    for step in range(3000):
        op = rng.randrange(11)
        key = rng.randrange(96) + rng.choice([0, 1024, 4096])
        batch = [key, rng.randrange(96), key, rng.randrange(96) + 1024]
        if op == 0:
            s.add(key)
            insert(key)
        elif op == 1:
            s.discard(key)
            remove(key)
        elif op == 2:
            if key in model:
                s.remove(key)
                remove(key)
            else:
                with pytest.raises(KeyError):
                    s.remove(key)
        elif op == 3:
            expected = len(set(batch) - model)
            assert s.add_many(np.array(batch, dtype=np.int64)) == expected
            for item in batch:
                insert(item)
        elif op == 4:
            batch += [-1, 2**32]
            expected = len(model.intersection(batch))
            assert s.discard_many(batch) == expected
            for item in batch:
                remove(item)
        elif op == 5:
            np.testing.assert_array_equal(s.contains_many(batch), [k in model for k in batch])
        elif op == 6:
            if dense:
                expected = dense[-1]
                assert s.pop() == expected
                remove(expected)
            else:
                with pytest.raises(KeyError):
                    s.pop()
        elif op == 7:
            s.sort()
            dense.sort()
        elif op == 8 and rng.randrange(8) == 0:
            s.clear()
            model.clear()
            dense.clear()
        elif op == 9:
            s.reserve(4)
        else:
            assert (key in s) == (key in model)
        assert len(s) == len(model), step
        assert list(s) == dense, step
        assert set(s) == model, step


@pytest.mark.parametrize("method", ["union", "intersection", "difference"])
@pytest.mark.parametrize("others", [[], [[]], [[1, 4, 4, 6]], [[1, 4, 6], [4, 8]],
                                     [[], [1, 2, 3]]])
def test_algebra_methods_match_set(method, others):
    from paged_collections import PAGE_SIZES

    source = [9, 4, 1, 7]
    actual = SparseSet(source, page_size=PAGE_SIZES[0])
    expected = getattr(set(source), method)(*others)
    result = getattr(actual, method)(*(iter(other) for other in others))
    assert type(result) is SparseSet
    assert set(result) == expected
    assert result.page_size == actual.page_size
    assert list(actual) == source
    candidates = list(dict.fromkeys(source + [k for other in others for k in other]))
    assert list(result) == [k for k in candidates if k in expected]
    assert result is not actual


@pytest.mark.parametrize("method", ["union", "intersection", "difference", "symmetric_difference"])
@pytest.mark.parametrize("form", [list, tuple, iter, set, SparseSet])
def test_algebra_any_key_iterable(method, form):
    source, other = [9, 4, 1, 7], [1, 6, 6, 4]
    actual = SparseSet(source)
    result = getattr(actual, method)(form(other))
    assert set(result) == getattr(set(source), method)(other)


@pytest.mark.parametrize("method,builtin", [
    ("update", "update"), ("intersection_update", "intersection_update"),
    ("difference_update", "difference_update"),
    ("symmetric_difference_update", "symmetric_difference_update"),
])
def test_inplace_methods_match_set_and_order(method, builtin):
    source, other = [9, 4, 1, 7], [1, 6, 6, 4]
    actual, expected = SparseSet(source), set(source)
    assert getattr(actual, method)(iter(other)) is getattr(expected, builtin)(other) is None
    candidates = list(dict.fromkeys(source + other))
    assert list(actual) == [k for k in candidates if k in expected]
    assert set(actual) == expected
    if method != "symmetric_difference_update":
        actual, expected = SparseSet(source), set(source)
        assert getattr(actual, method)() is getattr(expected, builtin)() is None
        assert list(actual) == source
        getattr(actual, method)([1, 6], iter([6, 9]))
        getattr(expected, builtin)([1, 6], [6, 9])
        assert set(actual) == expected


@pytest.mark.parametrize("name", ["issubset", "issuperset", "isdisjoint"])
@pytest.mark.parametrize("other", [[], [1], [9, 4, 1, 7], [1, 6, 4], [1, 2**100, -1],
                                    [1.0, 4.0], ["foreign", None]])
def test_set_predicates_any_iterable(name, other):
    source = [9, 4, 1, 7]
    assert getattr(SparseSet(source), name)(iter(other)) == getattr(set(source), name)(iter(other))


@pytest.mark.parametrize("left,right", [([], []), ([], [1]), ([1], []), ([1], [1]),
                                        ([1], [1, 2]), ([1, 2], [1]), ([1, 3], [2])])
def test_set_operators_match_builtin(left, right):
    import operator

    for op in (operator.or_, operator.and_, operator.sub, operator.xor):
        actual = op(SparseSet(left), SparseSet(right))
        assert set(actual) == op(set(left), set(right))
        assert type(actual) is SparseSet
    for op in (operator.lt, operator.le, operator.gt, operator.ge, operator.eq, operator.ne):
        assert op(SparseSet(left), SparseSet(right)) == op(set(left), set(right))
    for op in (operator.ior, operator.iand, operator.isub, operator.ixor):
        actual, expected = SparseSet(left), set(left)
        result = op(actual, SparseSet(right))
        assert result is actual
        assert set(actual) == op(expected, set(right))


@pytest.mark.parametrize("name", ["__or__", "__and__", "__sub__", "__xor__",
                                  "__ior__", "__iand__", "__isub__", "__ixor__",
                                  "__lt__", "__le__", "__gt__", "__ge__"])
def test_operators_reject_foreign_types_and_allow_reflection(name):
    import operator

    actual = SparseSet([1, 2])
    op = {"__or__": operator.or_, "__and__": operator.and_, "__sub__": operator.sub,
          "__xor__": operator.xor, "__ior__": operator.ior, "__iand__": operator.iand,
          "__isub__": operator.isub, "__ixor__": operator.ixor, "__lt__": operator.lt,
          "__le__": operator.le, "__gt__": operator.gt, "__ge__": operator.ge}[name]
    for other in ([1], iter([1]), {1}, frozenset([1]), object()):
        assert getattr(actual, name)(other) is NotImplemented
        with pytest.raises(TypeError):
            op(actual, other)
    assert actual != {1, 2}
    assert list(actual) == [1, 2]


@pytest.mark.parametrize("name", ["update", "intersection_update", "difference_update",
                                  "symmetric_difference_update", "__ior__", "__iand__",
                                  "__isub__", "__ixor__"])
def test_inplace_algebra_respects_exports(name):
    actual = SparseSet([9, 4, 1])
    view = actual.view()
    other = SparseSet([4, 7])
    with pytest.raises(BufferError):
        getattr(actual, name)(other)
    assert view.tolist() == list(actual) == [9, 4, 1]
    # Even no-op in-place algebra checks exports; copied results remain allowed.
    with pytest.raises(BufferError):
        getattr(actual, name)(actual)
    assert set(actual.union(other)) == {9, 4, 1, 7}
    del view
    getattr(actual, name)(other)


def test_algebra_subclasses_bypass_iter_and_return_base():
    class SubSet(SparseSet):
        def __iter__(self):
            raise AssertionError("native set algebra must bypass subclass __iter__")

    class SubBuiltin(set):
        def __iter__(self):
            raise AssertionError("native set algebra must bypass subclass __iter__")

    import operator
    for op in (operator.or_, operator.and_, operator.sub, operator.xor):
        result = op(SubSet([9, 4, 1]), SubSet([4, 7]))
        assert type(result) is SparseSet
        assert set(result) == op(SubBuiltin([9, 4, 1]), SubBuiltin([4, 7]))
    for method in ("union", "intersection", "difference", "symmetric_difference"):
        result = getattr(SubSet([9, 4, 1]), method)(SubSet([4, 7]))
        assert type(result) is SparseSet
        assert set(result) == getattr(SubBuiltin([9, 4, 1]), method)(SubBuiltin([4, 7]))
    assert SubSet([1]) < SparseSet([1, 2])
    assert SparseSet([1, 2]) > SubSet([1])


@pytest.mark.parametrize("name", ["update", "intersection_update", "difference_update",
                                  "symmetric_difference_update"])
def test_algebra_callbacks_run_unborrowed_and_do_not_overwrite_mutations(name):
    actual = SparseSet([1, 2, 3])

    def keys():
        assert list(actual) == [1, 2, 3]
        actual.add(8)
        yield 2

    if name in ("update", "difference_update"):
        getattr(actual, name)(keys())
        assert set(actual) == ({1, 2, 3, 8} if name == "update" else {1, 3, 8})
    else:
        with pytest.raises(RuntimeError, match="SparseSet changed during"):
            getattr(actual, name)(keys())
        assert list(actual) == [1, 2, 3, 8]


@pytest.mark.parametrize("method", ["union", "intersection", "difference", "symmetric_difference"])
def test_algebra_propagates_iterator_failure(method):
    actual = SparseSet([1, 2, 3])

    def broken():
        yield 2
        raise LookupError("iterator failed")

    with pytest.raises(LookupError, match="iterator failed"):
        getattr(actual, method)(broken())
    assert list(actual) == [1, 2, 3]


@pytest.mark.parametrize("name", ["update", "difference_update", "intersection_update",
                                  "symmetric_difference_update"])
def test_update_iterator_error_progress_matches_builtin(name):
    def broken():
        yield 2
        yield 6
        raise LookupError("iterator failed")

    actual, expected = SparseSet([1, 2, 3]), {1, 2, 3}
    for value in (actual, expected):
        with pytest.raises(LookupError, match="iterator failed"):
            getattr(value, name)(broken())
    assert set(actual) == expected


def test_streaming_updates_expose_progress_to_callbacks():
    for name in ("update", "difference_update"):
        def exercise(cls):
            value = cls([1, 2, 3])
            observed = []

            def keys():
                yield 2
                observed.append(set(value))
                yield 6
                observed.append(set(value))

            getattr(value, name)(keys())
            return set(value), observed

        assert exercise(SparseSet) == exercise(set)


def test_difference_update_swap_order_and_float_membership():
    actual = SparseSet([9, 4, 1, 7])
    actual.difference_update([4.0])
    assert list(actual) == [9, 7, 1]
    expected = {9, 4, 1, 7}
    expected.difference_update([4.0])
    assert set(actual) == expected
    with pytest.raises(TypeError):
        actual.difference_update([[]])
