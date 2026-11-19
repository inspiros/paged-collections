"""NumPy export ownership, write protection and structural mutation guards."""

import copy
import gc
import pickle

import numpy as np
import pytest

import paged_collections as pc


def _new(cls):
    if cls is pc.SparseSet:
        return cls([9, 2, 7])
    return cls([9, 2, 7], [True, False, True])


def _snapshot(container):
    if isinstance(container, pc.SparseSet):
        return list(container)
    return container.items()


def _insert(container, key):
    if isinstance(container, pc.SparseSet):
        container.add(key)
    else:
        container[key] = True


def _structural_change(container, operation):
    is_set = isinstance(container, pc.SparseSet)
    if operation == "insert":
        _insert(container, 11)
    elif operation == "remove":
        if is_set:
            container.remove(2)
        else:
            del container[2]
    elif operation == "discard_many":
        container.discard_many([2, 2, 100])
    elif operation == "batch_insert":
        if is_set:
            container.add_many([9, 11])
        else:
            # The existing value must not be changed before the new key fails.
            container.set_many([9, 11], [False, True])
    elif operation == "pop":
        if is_set:
            container.pop()
        else:
            container.pop(2)
    elif operation == "sort":
        if is_set:
            container.sort()
        else:
            container.sort_keys()
    elif operation == "reserve":
        container.reserve(100)
    else:
        getattr(container, operation)()


@pytest.mark.parametrize(
    "operation",
    ["insert", "remove", "clear", "pop", "pop_first", "pop_last", "sort", "reserve", "shrink_to_fit",
     "batch_insert", "discard_many"],
)
def test_live_export_blocks_structural_changes_atomically(export_case, operation):
    cls, method = export_case
    container = _new(cls)
    view = getattr(container, method)()
    before = _snapshot(container)
    array_before = view.copy()
    pages = container.allocated_pages
    with pytest.raises(BufferError):
        _structural_change(container, operation)
    assert _snapshot(container) == before
    np.testing.assert_array_equal(view, array_before)
    assert container.allocated_pages == pages
    del view
    _structural_change(container, operation)
    # The formerly blocked call must work, including after sort/reserve/shrink.
    _insert(container, 12)
    assert 12 in container


def test_set_discard_is_blocked_only_for_present_keys():
    s = pc.SparseSet([9, 2, 7])
    view = s.view()
    with pytest.raises(BufferError):
        s.discard(2)
    assert list(s) == [9, 2, 7]
    s.add(9)
    s.discard(100)
    s.discard(-1)
    assert s.add_many([9, 2, 9]) == 0
    assert s.add_many([]) == 0
    assert s.discard_many([-1, 2**32, 100]) == 0
    with pytest.raises(KeyError):
        s.remove(100)
    del view
    s.discard(2)
    assert list(s) == [9, 7]


def test_keys_are_read_only(container_cls):
    container = _new(container_cls)
    view = container.view() if container_cls is pc.SparseSet else container.keys_view()
    assert isinstance(view, np.ndarray)
    assert view.dtype == np.dtype(np.uint32)
    assert view.shape == (3,)
    assert not view.flags.writeable
    np.testing.assert_array_equal(view, [9, 2, 7])
    for array in (view, view[::2], view.view(np.uint8), np.asarray(view)):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
        with pytest.raises(ValueError):
            array[0] = 0
    assert list(container) == [9, 2, 7]


def test_numeric_values_views_share_writes_in_both_directions(typed_map_cls):
    m = _new(typed_map_cls)
    view = m.values_view()
    second = m.values_view()
    keys = m.keys_view()
    assert view.dtype == np.dtype(m.dtype)
    assert view.shape == (3,)
    np.testing.assert_array_equal(view, [True, False, True])
    assert np.shares_memory(view, second)
    if typed_map_cls is pc.SparseMapBool:
        assert not view.flags.writeable
        with pytest.raises(ValueError):
            view.setflags(write=True)
        with pytest.raises(ValueError):
            view[0] = False
        bytes_view = view.view(np.uint8)
        assert not bytes_view.flags.writeable
        with pytest.raises(ValueError):
            bytes_view.setflags(write=True)
    else:
        assert view.flags.writeable
        view[0] = 4
        assert m[9] == 4
        assert second[0] == 4
        # Writing a derived slice must also reach the map.
        view[1:][0] = 5
        assert m[2] == 5
    m[2] = True
    assert view[1] == 1
    assert m.set_many([9, 7], [False, False]) == 0
    np.testing.assert_array_equal(view, [False, True, False])
    m.update({7: True})
    np.testing.assert_array_equal(view, [False, True, True])
    np.testing.assert_array_equal(keys, [9, 2, 7])


def test_existing_key_writes_and_absent_removals_with_export(export_case):
    cls, method = export_case
    m = _new(cls)
    view = getattr(m, method)()
    if cls is pc.SparseSet:
        m.add(9)
        assert m.add_many([9, 2]) == 0
        assert m.discard_many([100, -1, 2**32]) == 0
        assert list(m) == [9, 2, 7]
        del view
        m.add(11)
        assert 11 in m
        return
    m[9] = False
    assert m.set_many([2, 7, 2], [True, False, False]) == 0
    m.update([(9, True), (7, True)])
    assert m.setdefault(2, True) == 0
    assert m.items() == [(9, True), (2, False), (7, True)]
    assert m.discard_many([100, -1, 2**32]) == 0
    assert m.pop(100, None) is None
    assert m.set_many([], []) == 0
    with pytest.raises(KeyError):
        del m[100]
    with pytest.raises(KeyError):
        m.pop(100)
    with pytest.raises(BufferError):
        m.popitem()
    with pytest.raises(BufferError):
        m.setdefault(11, True)
    with pytest.raises(BufferError):
        m.update({11: True})
    assert list(m) == [9, 2, 7]
    del view
    assert m.popitem() == (7, True)


@pytest.mark.parametrize("derived_kind", ["slice", "bytes", "asarray"])
def test_derived_arrays_keep_export_until_last_reference_is_gone(export_case, derived_kind):
    cls, method = export_case
    container = _new(cls)
    original = getattr(container, method)()
    if derived_kind == "slice":
        derived = original[::2]
    elif derived_kind == "bytes":
        derived = original.view(np.uint8)
    else:
        derived = np.asarray(original)
    expected = derived.copy()
    del original
    gc.collect()
    with pytest.raises(BufferError):
        _insert(container, 11)
    np.testing.assert_array_equal(derived, expected)
    del derived
    gc.collect()
    _insert(container, 11)
    assert list(container) == [9, 2, 7, 11]


def test_independent_exports_are_counted_separately(export_case):
    cls, method = export_case
    container = _new(cls)
    first = getattr(container, method)()
    second = getattr(container, method)()
    del first
    with pytest.raises(BufferError):
        container.clear()
    del second
    container.clear()
    assert not container


def test_live_views_share_one_base(export_case):
    cls, method = export_case
    container = _new(cls)
    first = getattr(container, method)()
    second = getattr(container, method)()
    assert first.base is second.base
    del first
    gc.collect()
    with pytest.raises(BufferError):
        _insert(container, 11)
    del second
    gc.collect()
    _insert(container, 11)
    assert list(container) == [9, 2, 7, 11]


def test_typed_keys_and_values_views_share_one_base(typed_map_cls):
    m = _new(typed_map_cls)
    keys = m.keys_view()
    values = m.values_view()
    assert keys.base is values.base
    del keys
    gc.collect()
    with pytest.raises(BufferError):
        m[11] = True
    np.testing.assert_array_equal(values, [True, False, True])
    del values
    gc.collect()
    m[11] = True
    assert list(m) == [9, 2, 7, 11]


def test_object_cycle_with_live_keys_view_survives_gc():
    m = pc.SparseMap()
    m[1] = m
    m[2] = [m]
    view = m.keys_view()
    del m
    gc.collect()
    # The base keeps the owner alive; collection of the cycle is not required.
    np.testing.assert_array_equal(view, [1, 2])
    assert view.base is not None
    assert not view.flags.writeable


@pytest.mark.parametrize("operation", ["lookup_during_iteration", "write"])
def test_reentrant_index_mutation_keeps_container_consistent(container_cls, operation):
    container = _new(container_cls)
    calls = []

    class ReentrantKey:
        def __index__(self):
            calls.append(True)
            container.clear()
            return 11

    try:
        if operation == "lookup_during_iteration":
            iterator = iter(container)
            assert next(iterator) == 9
            ReentrantKey() in container
            list(iterator)
        else:
            _insert(container, ReentrantKey())
    except (BufferError, RuntimeError):
        pass
    assert calls
    keys = list(iter(container))
    assert len(container) == len(keys)
    assert len(keys) == len(set(keys))
    assert all(key in container for key in keys)
    _insert(container, 12)
    assert 12 in container


def test_export_keeps_owner_alive(export_case):
    cls, method = export_case
    container = _new(cls)
    view = getattr(container, method)()
    expected = view.copy()
    del container
    gc.collect()
    # Reading after dropping the sole Python owner checks the export guard.
    np.testing.assert_array_equal(view, expected)
    if method == "values_view" and cls is not pc.SparseMapBool:
        view[0] = 4
        assert view[0] == 4


def test_copies_and_pickle_do_not_inherit_exports(export_case):
    cls, method = export_case
    container = _new(cls)
    view = getattr(container, method)()
    clones = [container.copy(), copy.copy(container), copy.deepcopy(container),
              pickle.loads(pickle.dumps(container))]
    for clone in clones:
        assert clone == container
        assert _snapshot(clone) == _snapshot(container)
        _insert(clone, 11)
        clone.clear()
        assert not clone
    assert list(container) == [9, 2, 7]
    with pytest.raises(BufferError):
        _insert(container, 11)
    detached = view.copy()
    del view
    gc.collect()
    _insert(container, 11)
    assert detached.shape == (3,)
    assert len(container) == 4


def test_empty_export_still_pins_storage(export_case):
    cls, method = export_case
    container = cls()
    view = getattr(container, method)()
    assert view.shape == (0,)
    with pytest.raises(BufferError):
        _insert(container, 1)
    with pytest.raises(BufferError):
        container.reserve(1)
    with pytest.raises(BufferError):
        container.shrink_to_fit()
    container.clear()  # Already empty: no structural change.
    del view
    _insert(container, 1)
    assert 1 in container
