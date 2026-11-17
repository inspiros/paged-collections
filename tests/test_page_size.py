"""Per-instance storage selection, including reconstruction and exports."""

import copy
import pickle
import re

import numpy as np
import pytest

import paged_collections as pc
from paged_collections import _core

SUFFIXES = ("", "F64", "F32", "I64", "I32", "U64", "U32", "Bool")
MAPS = tuple(
    getattr(pc, family + suffix)
    for family in ("SparseMap", "IdMap", "DirectMap")
    for suffix in SUFFIXES
)
LISTS = tuple(getattr(pc, "PagedList" + suffix) for suffix in SUFFIXES)
QUEUES = (pc.PriorityQueue, pc.PriorityQueueF64)
CLASSES = (pc.SparseSet, *MAPS, *LISTS, pc.Hive, pc.PagedDeque, *QUEUES)
SIZES = pc.PAGE_SIZES
# DirectMap exposes copies, not buffer/view exports.
EXPORT_CASES = (pytest.param(pc.SparseSet, "view", id="SparseSet-view"),) + tuple(
    pytest.param(base, method, id=f"{base.__name__}-{method}")
    for base in MAPS
    if not base.__name__.startswith("DirectMap")
    for method in (
        ("keys_view",) if base in (pc.SparseMap, pc.IdMap)
        else ("keys_view", "values_view")
    )
)


def _child(base):
    def init(self, marker, *, page_size=pc.DEFAULT_PAGE_SIZE):
        base.__init__(self, page_size=page_size)
        self.marker = marker

    cls = type(
        "PageSizeChild" + base.__name__,
        (base,),
        {"__module__": __name__, "__init__": init},
    )
    globals()[cls.__name__] = cls
    return cls


CHILDREN = {base: _child(base) for base in CLASSES}


def _values(base, n):
    if base.__name__.endswith("Bool"):
        return [i % 3 == 0 for i in range(n)]
    return [i % 97 for i in range(n)]


def _filled(base, size, n, subclass=False):
    values = _values(base, n)
    keys = [3 * i for i in range(n)]
    if base.__name__.startswith("IdMap"):
        keys = [pc.make_id(k, i % 7 + 1) for i, k in enumerate(keys)]
    if base is pc.SparseSet:
        args, oracle = (keys,), set(keys)
    elif base in MAPS or base in QUEUES:
        args = (keys, values) if base in MAPS else (list(zip(keys, values)),)
        oracle = dict(zip(keys, values))
    elif base is pc.Hive:
        args, oracle = (values,), None
    else:
        args, oracle = (values,), values.copy()
    if subclass:
        obj = CHILDREN[base]({"size": size}, page_size=size)
        base.__init__(obj, *args, page_size=size)
    else:
        obj = base(*args, page_size=size)
    if base is pc.Hive:
        oracle = dict(zip(map(int, obj.ids()), values))
    return obj, oracle


def _snapshot(base, obj):
    if base is pc.SparseSet:
        return set(obj)
    if base in MAPS or base in QUEUES or base is pc.Hive:
        return dict(obj.items())
    return list(obj)


def _remove_some(base, obj, oracle, size):
    positions = sorted({0, size - 1, size, 2 * size + 1, len(oracle) - 1})
    if isinstance(oracle, (dict, set)):
        keys = list(oracle)
        for i in positions:
            key = keys[i]
            if base is pc.SparseSet:
                obj.discard(key)
                oracle.remove(key)
            else:
                del obj[key]
                del oracle[key]
    else:
        for i in reversed(positions):
            del obj[i]
            del oracle[i]


STOCK = SIZES == (256, 512, 1024, 2048, 4096) and pc.DEFAULT_PAGE_SIZE == 1024


@pytest.mark.skipif(not STOCK, reason="custom PAGED_COLLECTIONS_PAGE_SIZES build")
def test_stock_build_sizes():
    assert SIZES == (256, 512, 1024, 2048, 4096)
    assert pc.DEFAULT_PAGE_SIZE == 1024


def test_compiled_defaults():
    assert list(SIZES) == sorted(set(SIZES)) and pc.DEFAULT_PAGE_SIZE in SIZES
    assert _core.PAGE_SIZES == SIZES
    assert _core.DEFAULT_PAGE_SIZE == pc.DEFAULT_PAGE_SIZE
    assert "PAGE_SIZES" in pc.__all__ and "DEFAULT_PAGE_SIZE" in pc.__all__
    assert len(CLASSES) == 37
    for base in CLASSES:
        assert base().page_size == pc.DEFAULT_PAGE_SIZE


@pytest.mark.parametrize("base", CLASSES, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("size", SIZES)
def test_pages_contents_and_reconstruction(base, size):
    n = 3 * size + 5
    obj, oracle = _filled(base, size, n)
    assert obj.page_size == size
    assert obj.allocated_pages >= 3
    assert _snapshot(base, obj) == oracle
    with pytest.raises(AttributeError):
        obj.page_size = size
    _remove_some(base, obj, oracle, size)
    assert _snapshot(base, obj) == oracle

    default, default_oracle = _filled(base, pc.DEFAULT_PAGE_SIZE, n)
    # Remove the same positions even when the number of entries was based on
    # the non-default size. Hive ids are independent of page boundaries.
    _remove_some(base, default, default_oracle, size)
    assert default_oracle == oracle
    assert obj == default and default == obj
    for clone in (obj.copy(), copy.copy(obj), copy.deepcopy(obj), pickle.loads(pickle.dumps(obj))):
        assert clone is not obj
        assert clone.page_size == size
        assert _snapshot(base, clone) == oracle
        assert clone == obj

    child, child_oracle = _filled(base, size, n, subclass=True)
    _remove_some(base, child, child_oracle, size)
    method_copy = child.copy()
    assert method_copy.page_size == size
    assert _snapshot(base, method_copy) == child_oracle
    for clone in (copy.copy(child), copy.deepcopy(child), pickle.loads(pickle.dumps(child))):
        assert type(clone) is CHILDREN[base]
        assert clone.marker == {"size": size}
        assert clone.page_size == size
        assert _snapshot(base, clone) == child_oracle


@pytest.mark.parametrize("base", CLASSES, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("invalid", (0, -1, 300, 2**100))
def test_unsupported_size(base, invalid):
    # Works for a custom build that happens to support 300 as well.
    if invalid in SIZES:
        pytest.skip("this size was compiled in")
    expected = (
        f"page_size must be one of ({', '.join(map(str, SIZES))}), got {invalid}; "
        "rebuild with PAGED_COLLECTIONS_PAGE_SIZES to add sizes"
    )
    with pytest.raises(ValueError, match=re.escape(expected)):
        base(page_size=invalid)


@pytest.mark.parametrize("base", CLASSES, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("invalid", (None, "1024", 1024.0, object()))
def test_non_integer_size(base, invalid):
    with pytest.raises(TypeError):
        base(page_size=invalid)


@pytest.mark.parametrize("base", (*LISTS, pc.PagedDeque), ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("size", SIZES)
def test_sequence_results_take_left_size(base, size):
    other_size = next((s for s in SIZES if s != size), size)
    left, left_values = _filled(base, size, 3 * size + 5)
    right, right_values = _filled(base, other_size, 7)
    joined = left + right
    assert joined.page_size == size
    assert list(joined) == left_values + right_values
    for n in (0, 2):
        for repeated in (left * n, n * left):
            assert repeated.page_size == size
            assert list(repeated) == left_values * n
    if base in LISTS:
        for key in (slice(None, None, 2), slice(None, None, -1), slice(0, 0)):
            result = left[key]
            assert result.page_size == size
            assert list(result) == left_values[key]


@pytest.mark.parametrize("base", CLASSES, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("size", SIZES)
def test_repr_size(base, size):
    obj, _ = _filled(base, size, 3)
    text = repr(obj)
    assert (f"page_size={size}" in text) == (size != pc.DEFAULT_PAGE_SIZE)
    # These reprs were already accepted by their constructors.
    if base is pc.SparseSet or base in LISTS or base is pc.PagedDeque or base in QUEUES:
        restored = eval(text, vars(pc))
        assert restored.page_size == size
        assert restored == obj


@pytest.mark.parametrize("base", LISTS[1:], ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("size", SIZES)
def test_list_page_views(base, size):
    obj, oracle = _filled(base, size, 3 * size + 5)
    views = obj.page_views()
    assert [len(v) for v in views] == [size] * (len(oracle) // size) + ([len(oracle) % size] if len(oracle) % size else [])
    np.testing.assert_array_equal(np.concatenate(views), oracle)
    first = obj.page_view(0)
    assert np.shares_memory(first, views[0])
    if base is pc.PagedListBool:
        assert not first.flags.writeable
    else:
        first[0] = 42
        assert obj[0] == 42
    obj.append(_values(base, 1)[0])
    np.testing.assert_array_equal(first, views[0])
    with pytest.raises(BufferError):
        obj.pop()
    with pytest.raises(BufferError):
        base.__init__(obj, page_size=next((s for s in SIZES if s != size), size))


@pytest.mark.parametrize("base", (pc.SparseSet, *MAPS[:16]), ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("size", SIZES)
def test_dense_views(base, size):
    obj, oracle = _filled(base, size, 3 * size + 5)
    keys = obj.view() if base is pc.SparseSet else obj.keys_view()
    assert not keys.flags.writeable
    assert set(map(int, keys)) == set(oracle)
    if base in MAPS and base not in (pc.SparseMap, pc.IdMap):
        values = obj.values_view()
        np.testing.assert_array_equal(values, [oracle[int(k)] for k in keys])
        if base.__name__.endswith("Bool"):
            assert not values.flags.writeable
        else:
            values[0] = 42
            assert obj[int(keys[0])] == 42
    with pytest.raises(BufferError):
        obj.clear()


@pytest.mark.parametrize("base", (pc.SparseMap, pc.IdMap, pc.DirectMap, pc.PagedList, pc.PagedDeque, pc.Hive))
@pytest.mark.parametrize("size", SIZES)
def test_recursive_pickle_and_deepcopy(base, size):
    obj = base(page_size=size)
    if base in (pc.PagedList, pc.PagedDeque):
        obj.append(obj)
        item = lambda result: result[0]
    elif base is pc.Hive:
        key = obj.insert(obj)
        item = lambda result: result[key]
    else:
        key = pc.make_id(0, 1) if base is pc.IdMap else 0
        obj[key] = obj
        item = lambda result: result[key]
    for result in (copy.deepcopy(obj), pickle.loads(pickle.dumps(obj))):
        assert result.page_size == size
        assert item(result) is result


@pytest.mark.parametrize("size", SIZES)
def test_hive_history(size):
    hive = pc.Hive(range(3 * size + 5), page_size=size)
    dead = list(map(int, hive.ids()))[::2]
    hive.discard_many(dead)
    hive.shrink_to_fit()
    expected = dict(hive.items())
    for clone in (hive.copy(), copy.deepcopy(hive), pickle.loads(pickle.dumps(hive))):
        assert clone.page_size == size
        assert dict(clone.items()) == expected
        fresh = clone.insert_many(range(len(dead)))
        assert set(map(int, fresh)).isdisjoint(dead)


@pytest.mark.parametrize("factory", (pc.sparse_map, pc.id_map, pc.direct_map, pc.paged_list))
@pytest.mark.parametrize("size", SIZES)
def test_factories(factory, size):
    assert factory(page_size=size).page_size == size
    assert factory(np.int64, page_size=size).page_size == size


@pytest.mark.skipif(1024 not in SIZES, reason="build lacks page size 1024")
def test_old_rebuild_arguments_load_at_1024():
    # Pickles from before page_size carry no size; they were all 1024,
    # whatever this build's default is.
    obj = _core._rebuild(pc.PagedList, pc.PagedList, [1, 2, 3])
    assert list(obj) == [1, 2, 3]
    assert obj.page_size == 1024
    obj = _core._rebuild(pc.SparseMapI64, pc.SparseMapI64, [1, 2], [3, 4])
    assert dict(obj.items()) == {1: 3, 2: 4}
    assert obj.page_size == 1024


@pytest.mark.skipif(1024 not in SIZES, reason="build lacks page size 1024")
def test_old_hive_pickle_keeps_floors_of_1024_slot_pages():
    # An old Hive pickle: one freed page of 1024 slots with floor 2, then a
    # page whose slot 0 (index 1024) is vacant at version 4.
    floors = np.array([2, 0], dtype=np.uint32)
    page_lens = np.array([0, 1], dtype=np.uint32)
    versions = np.array([4], dtype=np.uint32)
    hive = _core._rebuild(pc.Hive, pc.Hive, None, (floors, page_lens, versions))
    assert hive.page_size == 1024
    hive.insert_many(range(2200))
    # Every index of the freed page restarts above floor 2: generation 2.
    # Read at 256-slot pages, indices 256.. would restart at generation 1.
    assert {pc.id_generation(hive.occupant(i)) for i in range(1024)} == {2}
    assert pc.id_generation(hive.occupant(1024)) == 3


@pytest.mark.parametrize("base", CLASSES, ids=lambda cls: cls.__name__)
def test_page_size_is_keyword_only(base):
    if base is pc.SparseSet or base in LISTS:
        args = (None,)
    elif base is pc.PagedDeque:
        args = ((), None)
    elif base in QUEUES:
        args = (None, False)
    else:
        args = (None, None)
    with pytest.raises(TypeError):
        base(*args, pc.DEFAULT_PAGE_SIZE)


@pytest.mark.parametrize("base", (pc.SparseSet, *MAPS), ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("with_data", (False, True), ids=("preserve", "merge"))
@pytest.mark.skipif(len(SIZES) < 2, reason="build has only one page size")
def test_reinitialization_retiles_without_losing_contents(base, size, with_data):
    source = next(s for s in SIZES if s != size)
    n = 3 * source + 5
    obj, oracle = _filled(base, source, n)
    _remove_some(base, obj, oracle, source)
    args = ()
    if with_data:
        old_key, new_key = next(iter(oracle)), 3 * n
        if base is pc.SparseSet:
            args = ([old_key, new_key],)
            oracle.update(args[0])
        else:
            if base.__name__.startswith("IdMap"):
                # Re-init merges entries, evicting an older generation at
                # the same slot just like set_many, rather than clearing.
                replacement = pc.make_id(pc.id_index(old_key), pc.id_generation(old_key) + 1)
                del oracle[old_key]
                old_key = replacement
                new_key = pc.make_id(new_key, 1)
            keys, values = [old_key, new_key], _values(base, 2)
            args = (keys, values)
            oracle.update(zip(keys, values))
    base.__init__(obj, *args, page_size=size)
    assert obj.page_size == size
    assert _snapshot(base, obj) == oracle


@pytest.mark.parametrize("base, method", EXPORT_CASES)
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("export_kind", ("view", "buffer"))
@pytest.mark.skipif(len(SIZES) < 2, reason="build has only one page size")
def test_reinitialization_with_different_page_size_is_blocked_by_export(
    base, method, size, export_kind,
):
    source = next(s for s in SIZES if s != size)
    obj, oracle = _filled(base, source, 7)
    view = getattr(obj, method)()
    array_before = view.copy()
    export = memoryview(view) if export_kind == "buffer" else view
    del view  # A derived buffer must keep the export alive on its own.
    pages = obj.allocated_pages
    before = list(obj)
    iterator = iter(obj)
    # No input data: failure must come from retiling itself, even when
    # there is no insertion or generation eviction to trigger a guard.
    with pytest.raises(BufferError):
        base.__init__(obj, page_size=size)
    assert obj.page_size == source
    assert obj.allocated_pages == pages
    assert _snapshot(base, obj) == oracle
    assert list(iterator) == before
    np.testing.assert_array_equal(export, array_before)
    del export
    base.__init__(obj, page_size=size)
    assert obj.page_size == size
    assert _snapshot(base, obj) == oracle


@pytest.mark.parametrize("size", SIZES)
def test_hive_reinitialization_keeps_live_and_stale_ids(size):
    source = next((s for s in SIZES if s != size), size)
    hive = pc.Hive(range(3 * source + 5), page_size=source)
    dead = list(map(int, hive.ids()))[:source]
    hive.discard_many(dead)
    hive.shrink_to_fit()  # Leaves an unallocated page with generation history.
    expected = dict(hive.items())
    pc.Hive.__init__(hive, page_size=size)
    assert hive.page_size == size
    assert dict(hive.items()) == expected
    assert all(key not in hive for key in dead)
    fresh = set(map(int, hive.insert_many(range(4 * max(source, size) + 5))))
    assert fresh.isdisjoint(dead)


@pytest.mark.parametrize("base", QUEUES, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("size", SIZES)
def test_min_queue_reconstruction(base, size):
    queue = base({1: 2, 3: -4}, min=True, page_size=size)
    for clone in (queue.copy(), copy.copy(queue), copy.deepcopy(queue), pickle.loads(pickle.dumps(queue))):
        assert clone.page_size == size
        assert clone.is_min
        assert clone.sorted() == [(3, -4), (1, 2)]
    restored = eval(repr(queue), vars(pc))
    assert restored.page_size == size
    assert restored == queue


@pytest.mark.parametrize("size", SIZES)
def test_bounded_deque_reconstruction(size):
    deque = pc.PagedDeque(range(10), maxlen=4, page_size=size)
    for clone in (deque.copy(), copy.copy(deque), copy.deepcopy(deque), pickle.loads(pickle.dumps(deque))):
        assert clone.page_size == size
        assert clone.maxlen == 4
        assert list(clone) == [6, 7, 8, 9]
    restored = eval(repr(deque), vars(pc))
    assert restored.page_size == size
    assert restored.maxlen == 4
    assert restored == deque
