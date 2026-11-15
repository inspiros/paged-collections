"""PagedDeque against collections.deque, plus reentrancy and pickling."""

import collections
from contextlib import contextmanager
import copy
import gc
import operator
import pickle
import random
import sys
import weakref

import pytest

import paged_collections as pc


class DequeSub(pc.PagedDeque):
    def __init__(self, tag):
        super().__init__([tag])
        self.tag = tag


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


def check(d, ref):
    assert list(d) == list(ref)
    assert len(d) == len(ref)
    assert list(reversed(d)) == list(reversed(ref))


@pytest.mark.parametrize("maxlen", [None, 0, 1, 5, 3000])
def test_random_ops_match_deque(maxlen):
    rng = random.Random(maxlen or 7)
    d, ref = pc.PagedDeque(maxlen=maxlen), collections.deque(maxlen=maxlen)
    for step in range(6000):
        op = rng.randrange(12)
        v = rng.randrange(50)
        if op < 3:
            d.append(v); ref.append(v)
        elif op < 5:
            d.appendleft(v); ref.appendleft(v)
        elif op == 5 and ref:
            assert d.pop() == ref.pop()
        elif op == 6 and ref:
            assert d.popleft() == ref.popleft()
        elif op == 7:
            n = rng.randrange(-2000, 2000)
            d.rotate(n); ref.rotate(n)
        elif op == 8 and ref:
            i = rng.randrange(-len(ref), len(ref))
            assert d[i] == ref[i]
            d[i] = v; ref[i] = v
        elif op == 9 and ref:
            i = rng.randrange(-len(ref), len(ref))
            del d[i]; del ref[i]
        elif op == 10:
            i = rng.randrange(-len(ref) - 3, len(ref) + 3)
            if maxlen is not None and len(ref) == maxlen:
                with pytest.raises(IndexError):
                    d.insert(i, v)
                with pytest.raises(IndexError):
                    ref.insert(i, v)
            else:
                d.insert(i, v); ref.insert(i, v)
        elif op == 11:
            vals = [rng.randrange(50) for _ in range(rng.randrange(5))]
            if rng.random() < 0.5:
                d.extend(vals); ref.extend(vals)
            else:
                d.extendleft(vals); ref.extendleft(vals)
        if step % 500 == 0:
            check(d, ref)
    check(d, ref)
    assert d.count(3) == ref.count(3)
    if 3 in ref:
        assert d.index(3) == ref.index(3)
        d.remove(3); ref.remove(3)
    d.reverse(); ref.reverse()
    check(d, ref)


def test_basic_api():
    d = pc.PagedDeque("abc", 4)
    assert d.maxlen == 4
    assert repr(d) == "PagedDeque(['a', 'b', 'c'], maxlen=4)"
    assert repr(pc.PagedDeque([1])) == "PagedDeque([1])"
    d.append("d"); d.append("e")
    assert list(d) == ["b", "c", "d", "e"]
    assert d.index("d", 1, 3) == 2
    with pytest.raises(ValueError):
        d.index("b", 1)
    with pytest.raises(ValueError):
        d.remove("z")
    with pytest.raises(IndexError):
        d[4]
    with pytest.raises(IndexError):
        pc.PagedDeque().pop()
    with pytest.raises(IndexError):
        pc.PagedDeque().popleft()
    with pytest.raises(ValueError):
        pc.PagedDeque(maxlen=-1)
    assert not pc.PagedDeque() and d
    assert pc.PagedDeque([1, 2]) == pc.PagedDeque([1, 2])
    assert pc.PagedDeque([1, 2]) < pc.PagedDeque([1, 3])
    assert pc.PagedDeque([1]) != [1]
    d += [1]
    assert d[-1] == 1
    d.extend(d)
    assert len(d) == 4
    c = d.copy()
    assert c == d and c.maxlen == d.maxlen and c is not d
    assert pc.PagedDeque().page_size == 1024
    assert pc.PagedDeque[int] is not None


@pytest.mark.parametrize("method", ["extend", "extendleft", "iadd"])
@pytest.mark.parametrize("maxlen", [None, 0, 1, 3, 8])
def test_self_extension_honors_subclass_iterator(method, maxlen):
    def outcome(base):
        calls = []

        class IterSub(base):
            def __iter__(self):
                calls.append(list(base.__iter__(self)))
                self.append(3)  # The callback must run without a borrow.
                return iter([4, 5])

        d = IterSub([1, 2], maxlen=maxlen)
        alias = d
        if method == "iadd":
            d += d
        else:
            getattr(d, method)(d)
        assert d is alias
        assert len(calls) == 1
        # Inspect storage through the base iterator, not the override.
        return list(base.__iter__(d)), calls

    assert outcome(pc.PagedDeque) == outcome(collections.deque)


@pytest.mark.parametrize("method", ["extend", "extendleft", "iadd"])
@pytest.mark.parametrize("maxlen", [None, 0, 3])
def test_self_extension_snapshots_before_mutating(method, maxlen):
    def outcome(base):
        calls = []

        class IterSub(base):
            def __iter__(self):
                calls.append(True)
                yield 4
                raise LookupError("iterator failed")

        d = IterSub([1, 2], maxlen=maxlen)
        with pytest.raises(LookupError, match="iterator failed"):
            if method == "iadd":
                d += d
            else:
                getattr(d, method)(d)
        assert calls == [True]
        return list(base.__iter__(d))

    assert outcome(pc.PagedDeque) == outcome(collections.deque)


@pytest.mark.parametrize("method", ["extend", "extendleft", "iadd"])
@pytest.mark.parametrize("maxlen", [None, 0, 1, 3, 8])
def test_exact_self_extension_matches_deque(method, maxlen):
    def outcome(cls):
        d = cls([1, 2], maxlen=maxlen)
        if method == "iadd":
            d += d
        else:
            getattr(d, method)(d)
        return list(d)

    assert outcome(pc.PagedDeque) == outcome(collections.deque)


@pytest.mark.parametrize("maxlen", [None, 30, 40])
def test_repr_shows_all_elements_like_deque(maxlen):
    d = pc.PagedDeque(range(30), maxlen=maxlen)
    ref = collections.deque(range(30), maxlen=maxlen)
    assert repr(d).replace("PagedDeque", "deque") == repr(ref)


@pytest.mark.parametrize("maxlen", [None, 40])
def test_recursive_repr_matches_deque(maxlen):
    d = pc.PagedDeque(range(25), maxlen=maxlen)
    ref = collections.deque(range(25), maxlen=maxlen)
    d.append(d)
    ref.append(ref)
    assert repr(d).replace("PagedDeque", "deque") == repr(ref)


def test_pages_and_capacity():
    d = pc.PagedDeque(range(5000))
    assert d.allocated_pages >= 5
    assert d.capacity >= 5000
    d.clear()
    d.shrink_to_fit()
    assert len(d) == 0
    d.reserve(10)
    assert d.capacity >= 10


def test_mutation_during_iteration():
    d = pc.PagedDeque([1, 2, 3])
    it = iter(d)
    next(it)
    d.append(4)
    with pytest.raises(RuntimeError):
        next(it)
    with pytest.raises(RuntimeError):
        next(it)


def test_reentrant_eq():
    d = pc.PagedDeque()

    class Evil:
        def __eq__(self, other):
            d.clear()
            return False

    d.extend([Evil(), Evil()])
    with no_unraisable(), pytest.raises(RuntimeError):
        d.count(1)
    assert len(d) == 0

    class Popper:
        def __eq__(self, other):
            d.append(0)
            return True

    d.extend([Popper()])
    with no_unraisable(), pytest.raises(IndexError):
        d.remove(1)


def test_del_reentry():
    d = pc.PagedDeque(maxlen=1)

    class Reenter:
        def __del__(self):
            d.append(None)

    with no_unraisable():
        d.append(Reenter())
        d.append(1)
        assert list(d) == [None]
        d.clear()
        d.append(Reenter())
        del d[0]
        assert list(d) == [None]


def test_pickle_copy_and_subclass():
    d = pc.PagedDeque([1, "x", None], maxlen=7)
    for p in (pickle.loads(pickle.dumps(d)), copy.copy(d), copy.deepcopy(d)):
        assert p == d and p.maxlen == 7

    s = DequeSub("t")
    for r in (s.copy(), copy.copy(s), copy.deepcopy(s), pickle.loads(pickle.dumps(s))):
        assert type(r) is DequeSub and r.tag == "t" and list(r) == ["t"]

    self_ref = pc.PagedDeque()
    self_ref.append(self_ref)
    assert repr(self_ref) == "PagedDeque([[...]])"
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        r = pickle.loads(pickle.dumps(self_ref, protocol))
        assert r[0] is r
    r = copy.deepcopy(self_ref)
    assert r[0] is r
    assert self_ref.copy()[0] is self_ref


def test_cycles_collected():
    d = pc.PagedDeque()
    d.append(d)
    w = weakref.ref(d)
    del d
    gc.collect()
    assert w() is None


def test_reinit_installs_maxlen_before_destructors():
    d = pc.PagedDeque(maxlen=3)
    effects = []

    class Refill:
        def __del__(self):
            effects.append(d.maxlen)
            d.extend([1, 2, 3])

    with no_unraisable():
        d.append(Refill())
        d.__init__([], maxlen=1)
        assert effects == [1]
        assert list(d) == [3]
        d.append(4)
        assert list(d) == [4] and d.maxlen == 1
        d.append(Refill())
        d.__init__([], maxlen=0)
        assert effects == [1, 0]
        assert list(d) == []


@pytest.mark.parametrize("action", ["clear", "replace", "delete", "evict_left", "evict_right"])
def test_all_removed_objects_drop_without_borrow(action):
    d = pc.PagedDeque(maxlen=1)
    effects = []

    class Reenter:
        def __del__(self):
            effects.append(len(d))
            d.append("from destructor")

    with no_unraisable():
        d.append(Reenter())
        if action == "clear":
            d.clear()
        elif action == "replace":
            d[0] = "replacement"
        elif action == "delete":
            del d[0]
        elif action == "evict_left":
            d.append("replacement")
        else:
            d.appendleft("replacement")
        assert len(effects) == 1
        assert list(d) == ["from destructor"]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("values,maxlen,action", [
    ([], None, lambda d: d.clear()),
    ([1], None, lambda d: d.rotate(0)),
    ([1], None, lambda d: d.rotate(1)),
    ([1, 2, 3], None, lambda d: d.__setitem__(0, 9)),
    ([1, 2, 3], None, lambda d: d.reverse()),
    ([1, 2, 3], None, lambda d: d.rotate(0)),
    ([1, 2, 3], None, lambda d: d.rotate(3)),
    ([1, 2, 3], None, lambda d: d.clear()),
    ([1, 2, 3], None, lambda d: d.__delitem__(1)),
    ([1, 2, 3], None, lambda d: d.insert(1, 9)),
    ([], 0, lambda d: d.append(9)),
    ([], 0, lambda d: d.extend([9])),
    ([], 0, lambda d: d.extendleft([9])),
    ([1], 1, lambda d: d.__imul__(2)),
    ([1, 2], 2, lambda d: d.__imul__(2)),
    ([1, 2], None, lambda d: d.__imul__(1)),
    ([1, 2], None, lambda d: d.__imul__(0)),
])
def test_iterator_state_matches_cpython(values, maxlen, action, reverse):
    def outcomes(cls):
        d = cls(values, maxlen=maxlen)
        it = reversed(d) if reverse else iter(d)
        action(d)
        result = []
        for _ in range(len(values) + 3):
            try:
                result.append(("value", next(it)))
            except (StopIteration, RuntimeError) as exc:
                result.append(("error", type(exc).__name__))
        return result

    assert outcomes(pc.PagedDeque) == outcomes(collections.deque)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("exhausted", [False, True])
def test_mutation_after_last_yield(reverse, exhausted):
    def outcomes(cls):
        d = cls([1])
        it = reversed(d) if reverse else iter(d)
        assert next(it) == 1
        if exhausted:
            assert next(it, None) is None
        d.append(2)
        errors = []
        for _ in range(2):
            try:
                next(it)
            except (StopIteration, RuntimeError) as exc:
                errors.append(type(exc).__name__)
        return errors

    assert outcomes(pc.PagedDeque) == outcomes(collections.deque)


@pytest.mark.parametrize("method", ["contains", "index", "count", "remove"])
@pytest.mark.parametrize("equal", [False, True])
def test_search_mutation_policy(method, equal):
    def outcome(cls):
        d = cls()

        class Mutate:
            def __eq__(self, other):
                d.clear()
                return equal

        d.append(Mutate())
        with no_unraisable():
            try:
                result = 0 in d if method == "contains" else getattr(d, method)(0)
                return "value", result, list(d)
            except (ValueError, IndexError, RuntimeError) as exc:
                return "error", type(exc).__name__, list(d)

    assert outcome(pc.PagedDeque) == outcome(collections.deque)


@pytest.mark.parametrize("op", [operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge])
def test_richcompare_detects_live_mutation(op):
    left = pc.PagedDeque()

    class Mutate:
        def __eq__(self, other):
            left.clear()
            return True

    left.extend([Mutate(), 1])
    with no_unraisable(), pytest.raises(RuntimeError):
        op(left, pc.PagedDeque([0, 1]))
    assert list(left) == []


def test_richcompare_honors_subclass_iterators_and_bool_results():
    class IterSub(pc.PagedDeque):
        def __iter__(self):
            self.append("callback")
            return iter([4, 5])

    a = IterSub([1, 2])
    assert a == pc.PagedDeque([4, 5])
    assert len(a) == 3

    class Truth:
        def __bool__(self):
            a.clear()
            return True

    class Ordered:
        def __eq__(self, other):
            return False

        def __lt__(self, other):
            return Truth()

    assert (pc.PagedDeque([Ordered()]) < pc.PagedDeque([0])) is True
    assert len(a) == 0


@pytest.mark.parametrize("op", [operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge])
def test_self_comparison_with_subclass_iterator_matches_deque(op):
    # CPython answers d == d and d != d by identity without iterating, but
    # the ordering operators iterate, through a subclass's __iter__.
    def outcome(base):
        calls = []

        class Element:
            def __eq__(self, other):
                calls.append("eq")
                return False

            def __lt__(self, other):
                calls.append("lt")
                return True

            __le__ = __gt__ = __ge__ = __lt__

        class IterSub(base):
            def __iter__(self):
                calls.append("iter")
                # Each iterator yields a distinct, non-reflexive element.
                return iter([Element()])

        d = IterSub([0])
        return op(d, d), calls

    expected = outcome(collections.deque)
    assert expected[1] == ([] if op in (operator.eq, operator.ne) else ["iter", "iter", "eq", "lt"])
    assert outcome(pc.PagedDeque) == expected


@pytest.mark.parametrize("op", [operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge])
@pytest.mark.parametrize("shared", [False, True])
def test_richcompare_nan_element_identity_matches_deque(op, shared):
    a = float("nan")
    b = a if shared else float("nan")
    d, other = pc.PagedDeque([a]), pc.PagedDeque([b])
    ref, ref_other = collections.deque([a]), collections.deque([b])
    assert op(d, d) == op(ref, ref)
    assert op(d, other) == op(ref, ref_other)


@pytest.mark.parametrize("n", [10**100, -(10**100)])
def test_strict_integer_overflow_and_saturated_search(n):
    d = pc.PagedDeque([1, 2, 3])
    with pytest.raises(OverflowError):
        pc.PagedDeque(maxlen=n)
    with pytest.raises(OverflowError):
        d.rotate(n)
    with pytest.raises(OverflowError):
        d.insert(n, 9)
    with pytest.raises(IndexError):
        d[n]
    with pytest.raises(IndexError):
        d[n] = 9
    with pytest.raises(IndexError):
        del d[n]
    assert list(d) == [1, 2, 3]
    assert d.index(2, -(10**100), 10**100) == 1
    with pytest.raises(ValueError):
        d.index(2, 10**100)


def test_explicit_none_and_constructor_arguments():
    d = pc.PagedDeque([1, 2])
    for cls in (pc.PagedDeque, collections.deque):
        with pytest.raises(TypeError):
            cls(None)
        with pytest.raises(TypeError):
            cls([], iterable=[])
        with pytest.raises(TypeError):
            cls([], 1, maxlen=1)
        with pytest.raises(TypeError):
            cls(unknown=1)
    with pytest.raises(TypeError):
        d.rotate(None)
    with pytest.raises(TypeError):
        d.index(1, None)
    with pytest.raises(TypeError):
        d.index(1, 0, None)
    assert list(d) == [1, 2]


def test_index_and_iterator_callbacks_are_unborrowed():
    d = pc.PagedDeque([1, 2])
    effects = []

    class Index:
        def __index__(self):
            effects.append(len(d))
            d.append(3)
            return 0

    assert d[Index()] == 1
    d[Index()] = 9
    d.insert(Index(), 8)
    d.rotate(Index())
    d.reserve(Index())
    assert len(effects) == 5

    def values():
        d.clear()
        yield 4
        assert list(d) == [4]
        d.append(5)
        yield 6
        raise LookupError("iterator failed")

    with pytest.raises(LookupError):
        d.extend(values())
    assert list(d) == [4, 5, 6]


@pytest.mark.parametrize("values", [[], [1], [1, 2, 3]])
@pytest.mark.parametrize("maxlen", [None, 0, 1, 2, 5])
@pytest.mark.parametrize("n", [-2, 0, 1, 2, 4])
def test_arithmetic_matches_deque(values, maxlen, n):
    d = pc.PagedDeque(values, maxlen)
    ref = collections.deque(values, maxlen)
    assert list(d * n) == list(ref * n)
    assert list(n * d) == list(n * ref)
    assert (d * n).maxlen == maxlen
    joined = d + pc.PagedDeque([7, 8])
    assert list(joined) == list(ref + collections.deque([7, 8]))
    assert joined.maxlen == maxlen
    alias = d
    d *= n
    ref *= n
    assert d is alias
    check(d, ref)
    with pytest.raises(TypeError):
        d + [1]


def test_repeat_limits_and_subclasses():
    s = DequeSub("tag")
    for result in (s.copy(), s + pc.PagedDeque([2]), s * 2, 2 * s):
        assert type(result) is DequeSub and result.tag == "tag"
    for operation in (lambda d: d * 10**100, lambda d: d.__imul__(10**100)):
        with pytest.raises(OverflowError):
            operation(pc.PagedDeque([1]))
    for maxlen in (None, 2):
        with pytest.raises(MemoryError):
            pc.PagedDeque([1, 2], maxlen) * sys.maxsize
    assert list(pc.PagedDeque([1], 1) * sys.maxsize) == [1]


def test_copy_restores_slot_state_without_subclass_init():
    calls = []

    class Slots(pc.PagedDeque):
        __slots__ = ("tag",)

        def __init__(self, tag):
            calls.append(tag)
            super().__init__([tag], maxlen=4)
            self.tag = tag

        def __getstate__(self):
            # State retrieval also must run without a borrow.
            self.reserve(0)
            assert len(self) == 1
            return None, {"tag": self.tag}

    value = []
    s = Slots(value)
    for result in (s.copy(), copy.copy(s)):
        assert type(result) is Slots
        assert result.tag is value and result[0] is value
        assert result.maxlen == 4
    assert calls == [value]
    assert "__copy__" not in pc.PagedDeque.__dict__


def test_maxlen_requires_an_actual_integer():
    class Index:
        def __index__(self):
            raise AssertionError("deque maxlen must not call __index__")

    with pytest.raises(TypeError):
        pc.PagedDeque(maxlen=Index())


def test_gc_finalizers_during_comparison():
    # Collect explicitly inside __eq__: a collection after == returns does
    # not exercise comparison reentry, especially on Python 3.12+.
    d = pc.PagedDeque()
    other = pc.PagedDeque([1])
    effects = []

    class Cycle:
        def __del__(self):
            assert effects == ["eq"]
            d.append(2)
            effects.append("del")

    class Equal:
        def __eq__(self, other):
            effects.append("eq")
            gc.collect()
            effects.append("collected")
            return True

    d.append(Equal())
    enabled = gc.isenabled()
    with no_unraisable():
        try:
            gc.collect()
            gc.disable()
            cycle = Cycle()
            cycle.self = cycle
            del cycle
            with pytest.raises(RuntimeError, match="mutated during iteration"):
                d == other
            assert effects == ["eq", "del", "collected"]
            assert len(d) == 2 and d[1] == 2
        finally:
            if enabled:
                gc.enable()
            else:
                gc.disable()
