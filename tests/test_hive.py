"""Hive history, Python object lifetime, and reconstruction contracts."""

import copy
import gc
import operator
import pickle
import random
import weakref

import numpy as np
import pytest

import paged_collections as pc
from paged_collections import Hive, IdMap, id_generation, id_index, make_id


def ordered(model):
    return sorted(model.items(), key=lambda pair: id_index(pair[0]))


def check_model(hive, model, dead):
    pairs = ordered(model)
    ids = [key for key, _ in pairs]
    assert len(hive) == len(model)
    assert hive.items() == pairs
    assert list(hive) == ids
    assert hive.ids().dtype == np.dtype("uint64")
    assert hive.ids().tolist() == ids
    assert hive.values() == [value for _, value in pairs]
    assert not (set(ids) & dead)
    assert hive.contains_many(ids).tolist() == [True] * len(ids)
    assert hive.get_many(ids) == [value for _, value in pairs]
    assert not hive.contains_many(list(dead)).any()
    for key, value in pairs:
        index, generation = id_index(key), id_generation(key)
        assert key == make_id(index, generation)
        assert 0 <= index < hive.index_bound
        assert generation >= 1
        assert hive.occupant(index) == key
        assert key in hive and hive[key] == value
    for key in list(dead)[-10:]:
        assert key not in hive
        assert hive.get(key, "absent") == "absent"
    assert hive.capacity == hive.allocated_pages * hive.page_size
    assert hive.capacity >= len(hive)
    assert hive.is_exhausted is False


@pytest.mark.parametrize("seed", range(8))
def test_random_history_model(seed):
    rng = random.Random(seed)
    hive, model, dead = Hive(), {}, set()
    generations = {}

    def record(ids, values):
        for key, value in zip(ids, values):
            key = int(key)
            index, generation = id_index(key), id_generation(key)
            assert key not in dead and key not in model
            assert generation > generations.get(index, 0)
            generations[index] = generation
            model[key] = value

    for step in range(350):
        op = rng.randrange(13)
        key = rng.choice(list(model)) if model and rng.random() < 0.8 else make_id(5000, 99)
        if op == 0:
            record([hive.insert(step)], [step])
        elif op == 1:
            values = [step * 10 + i for i in range(rng.randrange(12))]
            ids = hive.insert_many(iter(values))
            assert ids.dtype == np.dtype("uint64")
            record(ids, values)
        elif op in (2, 3, 4):
            if key in model:
                value = model.pop(key)
                dead.add(key)
                if op == 2:
                    del hive[key]
                elif op == 3:
                    assert hive.pop(key) == value
                else:
                    assert hive.discard(key)
            elif op == 4:
                assert not hive.discard(key)
            else:
                with pytest.raises(KeyError):
                    if op == 2:
                        del hive[key]
                    else:
                        hive.pop(key)
        elif op == 5:
            keys = rng.choices(list(model), k=8) if model else [key]
            victims = set(keys) & model.keys()
            assert hive.discard_many(keys) == len(victims)
            for victim in victims:
                del model[victim]
            dead.update(victims)
        elif op == 6:
            if key in model:
                hive[key] = -step
                model[key] = -step
            else:
                with pytest.raises(KeyError):
                    hive[key] = -step
        elif op == 7:
            divisor = rng.randrange(2, 6)
            victims = {k for k, v in model.items() if v % divisor == 0}
            hive.retain(lambda k, v: v % divisor)
            for victim in victims:
                del model[victim]
            dead.update(victims)
        elif op == 8:
            if rng.random() < 0.3:
                dead.update(model)
                model.clear()
                hive.clear()
        elif op == 9:
            hive.shrink_to_fit()
        elif op == 10:
            hive.reserve(rng.randrange(2000))
        elif op == 11:
            hive = pickle.loads(pickle.dumps(hive, protocol=step % 6))
        else:
            hive = hive.copy() if step % 2 else copy.copy(hive)
        check_model(hive, model, dead)

    # Exercise the restored floors/free lists for many more allocations.
    hive = pickle.loads(pickle.dumps(hive))
    values = list(range(2200))
    record(hive.insert_many(values), values)
    check_model(hive, model, dead)


def test_basics_reuse_batches_and_side_map():
    assert "Hive" in pc.__all__ and not hasattr(pc, "HiveIterator")
    assert Hive.dtype == "object" and Hive().page_size == 1024
    hive = Hive(["a", "b", "c"])
    first, second, third = hive.ids().tolist()
    assert repr(hive) == f"Hive({{{first}: 'a', {second}: 'b', {third}: 'c'}})"
    assert hive.discard(second) is True
    replacement = hive.insert("d")
    assert id_index(replacement) == id_index(second)
    assert id_generation(replacement) == id_generation(second) + 1
    assert hive.items() == [(first, "a"), (replacement, "d"), (third, "c")]
    assert hive.get_many([second, replacement], None) == [None, "d"]
    assert hive.contains_many([second, replacement]).dtype == np.dtype("bool")
    with pytest.raises(KeyError):
        hive.get_many([first, second])
    with pytest.raises(TypeError):
        hive.pop(first, None, None)
    with pytest.raises(TypeError):
        hive.get_many([first], None, None)
    assert hive.pop(second, "default") == "default"
    assert hive.get(second) is None
    assert hive.discard_many([first, first, second]) == 1
    assert hive.get_many([]) == []
    assert hive.discard_many([]) == 0
    ids = hive.ids()
    ids[:] = 0
    assert list(hive) == [replacement, third]
    side = IdMap([replacement, third], ["extra", "data"])
    assert side[replacement] == "extra"
    del hive[replacement]
    newer = hive.insert("e")
    assert newer not in side
    side[newer] = "new"
    assert side[newer] == "new" and replacement not in side
    assert Hive.__hash__ is None
    with pytest.raises(TypeError):
        hash(hive)
    assert hive.__eq__({}) is NotImplemented
    assert hive != {}


@pytest.mark.parametrize("bad", [-2**200, -1, 0, 1, 2**32 - 1, 2**64, 2**200,
                                 None, "id", 1.0, [], object(), np.float64(2**32)])
def test_garbage_scalar_ids(bad):
    hive = Hive(["live"])
    assert bad not in hive
    assert hive.get(bad, "missing") == "missing"
    assert hive.pop(bad, "missing") == "missing"
    assert hive.discard(bad) is False
    for operation in (lambda: hive[bad], lambda: operator.delitem(hive, bad),
                      lambda: operator.setitem(hive, bad, "bad"), lambda: hive.pop(bad)):
        with pytest.raises(KeyError):
            operation()
    assert hive.values() == ["live"]


def test_numpy_ids_and_invalid_batches():
    hive = Hive(["a", "b"])
    ids = hive.ids()
    for key in (np.uint64(ids[0]), np.int64(ids[0])):
        assert key in hive and hive.get(key) == "a"
    assert hive.contains_many(np.array([ids[0], 0], dtype=">u8")).tolist() == [True, False]
    assert hive.get_many(ids[::-1]) == ["b", "a"]
    assert hive.contains_many([-1, 2**100, int(ids[0])]).tolist() == [False, False, True]
    for bad in ([[int(ids[0])]], np.array([1.5]), ["bad"]):
        before = hive.items()
        with pytest.raises((TypeError, ValueError)):
            hive.discard_many(bad)
        assert hive.items() == before


def test_insert_many_materializes_before_mutation():
    hive = Hive([0, 1, 2])
    original = hive.items()
    inserted = hive.insert_many(hive.values())
    assert hive.get_many(inserted) == [0, 1, 2]
    assert hive.get_many([k for k, _ in original]) == [0, 1, 2]
    before = hive.items()

    def broken():
        yield "not inserted"
        raise ValueError("iteration failed")

    with pytest.raises(ValueError, match="iteration failed"):
        hive.insert_many(broken())
    assert hive.items() == before
    seen = []

    def mutating():
        seen.append(hive.insert("callback"))
        yield "batch"

    ids = hive.insert_many(mutating())
    assert hive[seen[0]] == "callback" and hive[ids[0]] == "batch"
    assert hive.insert_many([]).dtype == np.dtype("uint64")


def test_pages_floors_and_restored_history():
    hive = Hive(range(2100))
    original = hive.ids().tolist()
    hive.discard_many(original[:2048])
    hive.shrink_to_fit()
    assert hive.allocated_pages == 1
    assert hive.index_bound == 3072
    for clone in (hive.copy(), copy.copy(hive), copy.deepcopy(hive),
                  pickle.loads(pickle.dumps(hive))):
        clone.insert_many(range(3000))
        assert not clone.contains_many(original[:2048]).any()
        assert clone.get_many(original[2048:]) == list(range(2048, 2100))
        clone.clear()
        clone.shrink_to_fit()
        clone = pickle.loads(pickle.dumps(clone))
        clone.insert_many(range(3500))
        assert not clone.contains_many(original).any()
    hive.clear()
    hive.shrink_to_fit()
    assert hive.allocated_pages == 0 and hive.capacity == 0
    assert hive.index_bound == 3072
    empty = Hive()
    empty.reserve(3000)
    assert empty.allocated_pages >= 3
    empty.shrink_to_fit()
    assert empty.index_bound == 0


@pytest.mark.parametrize("mutation", ["insert", "insert_many", "del", "pop", "discard",
                                     "discard_many", "retain", "clear"])
def test_iterator_invalidation(mutation):
    hive = Hive([1, 2, 3])
    first = next(iter(hive))
    iterator = iter(hive)
    assert operator.length_hint(iterator) == 3
    assert next(iterator) == first
    if mutation == "insert":
        hive.insert(4)
    elif mutation == "insert_many":
        hive.insert_many([4])
    elif mutation == "del":
        del hive[first]
    elif mutation == "pop":
        hive.pop(first)
    elif mutation == "discard":
        hive.discard(first)
    elif mutation == "discard_many":
        hive.discard_many([first])
    elif mutation == "retain":
        hive.retain(lambda k, v: k != first)
    else:
        hive.clear()
    with pytest.raises(RuntimeError, match="^Hive changed during iteration$"):
        next(iterator)


def test_iterator_nonstructural_changes_and_exhaustion():
    hive = Hive([1, 2, 3])
    keys = list(hive)
    iterator = iter(hive)
    hive[keys[0]] = 99
    hive.reserve(5000)
    hive.shrink_to_fit()
    hive.discard(0)
    hive.discard_many([])
    hive.insert_many([])
    hive.retain(lambda k, v: True)
    assert list(iterator) == keys
    assert operator.length_hint(iterator) == 0
    hive.clear()
    assert next(iterator, "done") == "done"
    iterator = iter(hive)
    hive.clear()
    with pytest.raises(RuntimeError, match="Hive changed during iteration"):
        next(iterator)


def test_equality_reentrancy_and_identity():
    hive, other = Hive(), Hive()

    class Mutating:
        def __eq__(self, rhs):
            hive.clear()
            other.insert("new")
            return True

    key = hive.insert(Mutating())
    assert other.insert(object()) == key
    assert hive == other  # Comparison uses detached snapshots.
    assert not hive and len(other) == 2
    value = float("nan")
    hive = Hive([value])
    assert hive == hive.copy()  # Identity before equality.
    assert hive != Hive([float("nan")])
    assert Hive([1]) != Hive([2])
    assert Hive([1]) != Hive()
    other = Hive([1])
    other.discard(next(iter(other)))
    other.insert(1)
    assert Hive([1]) != other


def test_lookup_and_reserve_index_reentrancy():
    hive = Hive(["old"])
    key = next(iter(hive))

    class Query:
        def __index__(self):
            hive.insert("callback")
            return key

    # Match IdMap's PyKey conversion, accepting integer-like objects.
    assert hive.get(Query()) == "old"
    assert len(hive) == 2

    class Additional:
        def __index__(self):
            hive.insert("reserve callback")
            return 2000

    hive.reserve(Additional())
    assert len(hive) == 3


@pytest.mark.parametrize("operation", ["del", "pop", "discard", "discard_many", "clear",
                                      "replace", "retain", "reinit"])
def test_removed_value_destructors_can_touch_hive(operation):
    hive = Hive()
    owner = weakref.ref(hive)
    events = []

    class Touch:
        def __del__(self):
            try:
                current = owner()
                events.append((len(current), current.insert("from destructor")))
            except Exception as error:
                events.append(error)

    key = hive.insert(Touch())
    if operation == "del":
        del hive[key]
    elif operation == "pop":
        hive.pop(key)
    elif operation == "discard":
        hive.discard(key)
    elif operation == "discard_many":
        hive.discard_many([key, key])
    elif operation == "clear":
        hive.clear()
    elif operation == "replace":
        hive[key] = "replacement"
    elif operation == "retain":
        hive.retain(lambda k, v: False)
    else:
        hive.__init__(_state=([], [], []))
    assert len(events) == 1 and isinstance(events[0], tuple)
    assert hive[events[0][1]] == "from destructor"
    assert events[0][0] == (1 if operation == "replace" else 0)


def test_retain_mutation_uses_snapshot_and_live_ids():
    hive = Hive(["a", "b", "c"])
    a, b, c = list(hive)
    visited, added = [], []

    def predicate(key, value):
        visited.append((key, value))
        if key == a:
            del hive[b]
            added.append(hive.insert("new"))
            hive[c] = "changed"
        return False

    hive.retain(predicate)
    assert visited == [(a, "a"), (b, "b"), (c, "c")]
    assert hive.items() == [(added[0], "new")]
    assert b not in hive


def test_retain_exception_preserves_callback_mutations_only():
    hive = Hive([1, 2, 3])
    original = list(hive)
    visited = []

    def predicate(key, value):
        visited.append(key)
        if len(visited) == 2:
            hive.insert(4)
            raise ValueError("predicate failed")
        return False

    with pytest.raises(ValueError, match="predicate failed"):
        hive.retain(predicate)
    assert hive.get_many(original) == [1, 2, 3]
    assert hive.values() == [1, 2, 3, 4]


def test_repr_truncation_recursion_and_reentrancy():
    hive = Hive(range(25))
    text = repr(hive)
    assert text.startswith("Hive({") and text.endswith(", ...})")
    assert text.count(": ") == 20
    hive.clear()
    hive.insert(hive)
    assert "Hive(...)" in repr(hive)

    class Repr:
        def __repr__(self):
            hive.clear()
            return "callback"

    hive.clear()
    hive.insert(Repr())
    assert "callback" in repr(hive)
    assert not hive


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_self_containing_pickle_and_deepcopy(protocol):
    hive = Hive()
    dead = hive.insert("dead")
    hive.discard(dead)
    key = hive.insert(hive)
    restored = pickle.loads(pickle.dumps(hive, protocol=protocol))
    assert restored[key] is restored and dead not in restored
    deep = copy.deepcopy(hive)
    assert deep[key] is deep and dead not in deep
    shallow = copy.copy(hive)
    assert shallow[key] is hive
    base_copy = hive.copy()
    assert type(base_copy) is Hive and base_copy[key] is hive


class LabelledHive(Hive):
    def __init__(self, label):
        self.label = label
        super().__init__()


class SlottedHive(Hive):
    __slots__ = ("label",)

    def __init__(self, label):
        self.label = label
        super().__init__()


@pytest.mark.parametrize("cls", [LabelledHive, SlottedHive])
def test_subclass_reconstruction_and_memo(cls):
    hive = cls(["label"])
    hive.label.append(hive)
    key = hive.insert(hive.label)
    dead = hive.insert("remove")
    hive.discard(dead)
    for restored in (pickle.loads(pickle.dumps(hive)), copy.deepcopy(hive)):
        assert type(restored) is cls
        assert restored.label[0] == "label"
        assert restored.label[1] is restored
        assert restored[key] is restored.label
        assert dead not in restored
    shallow = copy.copy(hive)
    assert type(shallow) is cls and shallow.label is hive.label
    assert shallow[key] is hive.label
    assert type(hive.copy()) is Hive


def test_copy_shares_values_deepcopy_preserves_aliases():
    value = []
    hive = Hive([value, value])
    shallow = hive.copy()
    assert shallow.values()[0] is value and shallow.values()[1] is value
    deep = copy.deepcopy(hive)
    assert deep.values()[0] is deep.values()[1]
    assert deep.values()[0] is not value


def test_gc_and_iterator_owner_traversal():
    hive = Hive()
    hive.insert(hive)
    ref = weakref.ref(hive)
    del hive
    gc.collect()
    assert ref() is None
    hive = Hive()
    iterator = iter(hive)
    hive.insert(iterator)
    ref = weakref.ref(hive)
    del hive, iterator
    gc.collect()
    assert ref() is None


@pytest.mark.parametrize("state", [
    (), ([], []), ([], [], [], []), [[], [], []], 42,
    ([0], [], []), ([], [0], []), ([0], [1], []), ([0], [0], [1]),
    ([1], [0], []), ([0], [1025], [0] * 1025),
    ([-1], [0], []), ([2**32], [0], []), ([0], [-1], []),
    ([0], [1], [-1]), ([0], [1], [2**32]), ([0.5], [0], []),
    ([[0]], [0], []), (object(), [], []),
])
def test_bad_state_rejected(state):
    with pytest.raises(ValueError):
        Hive(_state=state)
    hive = Hive(["keep"])
    before = hive.items()
    with pytest.raises(ValueError):
        hive.__init__(_state=state)
    assert hive.items() == before


def test_state_encoding_placeholders_retired_slots_and_floors():
    hive = Hive(_state=(np.array([0, 8], dtype=np.uint32),
                        np.array([3, 0], dtype=np.uint32),
                        np.array([0, 2, 5], dtype=np.uint32)))
    live = make_id(2, 3)
    assert hive.items() == [(live, None)]
    hive[live] = "restored"
    key = hive.insert("reuse")
    assert key == make_id(1, 2)
    assert hive.occupant(0) is None  # Retired, never reusable.
    reduced = hive.__reduce__()
    state = reduced[1][3]
    assert len(state) == 3
    assert all(array.dtype == np.dtype("uint32") for array in state)
    assert state[0].tolist() == [0, 8]
    assert state[1].tolist() == [3, 0]
    assert state[2].tolist() == [0, 3, 5]
    hive = pickle.loads(pickle.dumps(hive))
    hive.insert_many(range(2200))
    assert hive.occupant(0) is None
    assert hive.get(live) == "restored"
    floor_id = hive.occupant(1024)
    assert id_generation(floor_id) == 5


@pytest.mark.skip(reason="Exhaustion requires all 2**32-1 slots occupied or retired")
def test_exhaustion():
    pass


def test_review_bad_state_rejected_before_allocation():
    # Many one-slot pages, then an odd floor: must be ValueError, quickly.
    n = 4000
    floors = np.zeros(n, dtype=np.uint32)
    floors[-1] = 1
    lens = np.ones(n, dtype=np.uint32)
    versions = np.ones(n, dtype=np.uint32)
    with pytest.raises(ValueError, match="odd page floor"):
        Hive(None, (floors, lens, versions))


def test_review_many_single_inserts_are_fast():
    import time

    hive = Hive()
    start = time.perf_counter()
    ids = [hive.insert(i) for i in range(200_000)]
    assert time.perf_counter() - start < 5
    assert len(hive) == 200_000 and hive[ids[-1]] == 199_999


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("use_key", [False, True])
def test_sort_and_sorted_ids_match_builtin(reverse, use_key):
    values = [(2, "a"), (1, "a"), (2, "b"), (1, "b"), (3, "a")]
    hive = Hive(values)
    dead = list(hive)[2]
    del hive[dead]  # Sorting must leave the hole and ids unchanged.
    before = hive.items()
    key = (lambda value: value[0]) if use_key else None
    expected = sorted(hive.values(), key=key, reverse=reverse)
    expected_ids = sorted(list(hive), key=lambda id: key(hive[id]) if key else hive[id],
                          reverse=reverse)
    assert hive.sorted_ids(key=key, reverse=reverse) == expected_ids
    assert hive.items() == before
    ids = list(hive)
    assert hive.sort(key=key, reverse=reverse) is None
    assert list(hive) == ids
    assert hive.values() == expected
    assert dead not in hive and hive.occupant(id_index(dead)) is None
    assert hive.get_many(ids) == expected  # Ids name slots, not objects.


@pytest.mark.parametrize("method", ["sort", "sorted_ids"])
@pytest.mark.parametrize("reverse", [False, True])
def test_sort_stability_only_uses_less_than(method, reverse):
    class Key:
        def __init__(self, value):
            self.value = value

        def __eq__(self, other):
            raise AssertionError("sort must not use equality")

        def __lt__(self, other):
            return self.value < other.value

    values = [(i % 3, i) for i in range(30)]
    hive = Hive(values)
    expected = sorted(values, key=lambda v: Key(v[0]), reverse=reverse)
    calls = []

    def key(value):
        calls.append(value)
        assert len(hive) == len(values)  # No PyRef is held during callbacks.
        return Key(value[0])

    result = getattr(hive, method)(key=key, reverse=reverse)
    assert calls == values
    assert (hive.get_many(result) if method == "sorted_ids" else hive.values()) == expected


@pytest.mark.parametrize("method", ["sort", "sorted_ids"])
def test_sorted_ids_distinguishes_repeated_object_slots(method):
    value = object()
    hive = Hive([value, value, value])
    ids = list(hive)
    keys = iter([2, 0, 1])
    result = getattr(hive, method)(key=lambda _: next(keys))
    if method == "sorted_ids":
        assert result == [ids[1], ids[2], ids[0]]
    assert all(v is value for v in hive.values())


@pytest.mark.parametrize("method", ["sort", "sorted_ids"])
@pytest.mark.parametrize("failure", ["key", "compare"])
def test_sort_exceptions_leave_state_unchanged(method, failure):
    hive = Hive([3, 1, 2])
    before = hive.items()

    class Key:
        def __lt__(self, other):
            assert hive.items() == before
            raise LookupError("comparison failed")

    def key(value):
        assert hive.items() == before
        if failure == "key" and value == 1:
            raise LookupError("key failed")
        return Key()

    with pytest.raises(LookupError, match="failed"):
        getattr(hive, method)(key=key)
    assert hive.items() == before


@pytest.mark.parametrize("method", ["sort", "sorted_ids"])
@pytest.mark.parametrize("mutation", ["replace", "insert", "remove", "reinit", "restore"])
def test_sort_rejects_callback_mutation(method, mutation):
    hive = Hive([3, 1, 2])
    ids = list(hive)
    calls = 0

    def key(value):
        nonlocal calls
        calls += 1
        if calls == 1:
            if mutation == "replace":
                hive[ids[0]] = 9
            elif mutation == "insert":
                hive.insert(4)
            elif mutation == "remove":
                del hive[ids[1]]
            elif mutation == "reinit":
                hive.__init__([8], _state=([], [], []))
            else:
                hive[ids[0]] = 9
                hive[ids[0]] = 3  # Detect mutations even if state was restored.
        return value

    with pytest.raises(RuntimeError, match="Hive changed during"):
        getattr(hive, method)(key=key)
    assert hive.values() == {
        "replace": [9, 1, 2], "insert": [3, 1, 2, 4], "remove": [3, 2],
        "reinit": [8], "restore": [3, 1, 2],
    }[mutation]


@pytest.mark.parametrize("method", ["sort", "sorted_ids"])
def test_sort_comparison_and_key_destructor_mutation(method):
    for during in ("compare", "del"):
        hive = Hive([3, 1, 2])
        first = next(iter(hive))

        class Key:
            def __init__(self, value):
                self.value = value

            def __lt__(self, other):
                if during == "compare":
                    hive[first] = 9
                return self.value < other.value

            def __del__(self):
                if during == "del":
                    hive[first] = 9

        with pytest.raises(RuntimeError, match="Hive changed during"):
            getattr(hive, method)(key=Key)
        assert hive.values() == [9, 1, 2]


def test_sort_subclass_and_keyword_signature():
    class SubHive(Hive):
        def __iter__(self):
            raise AssertionError("sort uses slots")

    hive = SubHive([3, 1, 2])
    ids = Hive.sorted_ids(hive)
    assert hive.get_many(ids) == [1, 2, 3]
    hive.sort()
    assert hive.values() == [1, 2, 3]
    for method in (hive.sort, hive.sorted_ids):
        with pytest.raises(TypeError):
            method(None)
        with pytest.raises(TypeError):
            method(reverse="yes")


@pytest.mark.parametrize("method", ["sort", "sorted_ids"])
def test_hive_sort_reverse_index_protocol(method):
    class Reverse:
        def __index__(self):
            return 1

    hive = Hive([2, 1, 3])
    result = getattr(hive, method)(reverse=Reverse())
    expected = sorted([2, 1, 3], reverse=Reverse())
    assert (hive.values() if method == "sort" else hive.get_many(result)) == expected
    with pytest.raises(OverflowError):
        getattr(hive, method)(reverse=2**100)
