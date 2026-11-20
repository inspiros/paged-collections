"""Shared FIFO and indexed priority queues."""

from concurrent.futures import ThreadPoolExecutor
import copy
import math
import pickle
import random
import threading
from types import MappingProxyType
import weakref

import numpy as np
import pytest

import paged_collections as pc


@pytest.fixture(params=[pc.ConcurrentPriorityQueue, pc.ConcurrentPriorityQueueF64])
def priority_queue_cls(request):
    return request.param


def test_fifo_order_and_errors():
    marker = object()
    queue = pc.ConcurrentQueue(iter([marker, None]))
    ref = weakref.ref(queue)
    assert ref() is queue
    assert len(queue) == 2 and queue
    assert repr(queue) == "ConcurrentQueue(len=2)"
    assert queue.pop() is marker
    assert queue.pop() is None
    assert len(queue) == 0 and not queue
    with pytest.raises(IndexError, match="^pop from an empty ConcurrentQueue$"):
        queue.pop()
    with pytest.raises(TypeError):
        iter(queue)
    with pytest.raises(TypeError):
        pc.ConcurrentQueue(1)
    with pytest.raises(TypeError):
        queue.push_many(1)


def test_fifo_batches_clear_and_drain():
    queue = pc.ConcurrentQueue()
    assert queue.push_many([]) == 0
    queue.push("first")
    assert queue.push_many(str(i) for i in range(6)) == 6
    for n in (-1, -(2**100)):
        with pytest.raises(ValueError):
            queue.pop_many(n)
    with pytest.raises(TypeError):
        queue.pop_many(1.5)
    assert queue.pop_many(0) == []
    assert queue.pop_many(3) == ["first", "0", "1"]
    assert queue.pop_many(2**100) == ["2", "3", "4", "5"]
    assert queue.pop_many(3) == []

    queue.push_many(range(4096))
    queue.clear()
    assert queue.drain() == []
    queue.push_many(range(4096))
    assert queue.drain() == list(range(4096))
    assert not queue
    queue.push("reused")
    assert queue.pop() == "reused"


def test_fifo_push_many_keeps_items_yielded_before_error():
    queue = pc.ConcurrentQueue()

    def broken():
        yield 1
        raise LookupError("iteration failed")

    with pytest.raises(LookupError, match="iteration failed"):
        queue.push_many(broken())
    assert queue.drain() == [1]


def test_fifo_clear_drain_and_shrink_require_exclusive_access():
    queue = pc.ConcurrentQueue()
    entered = threading.Event()
    resume = threading.Event()

    def items():
        entered.set()
        assert resume.wait(timeout=30)
        yield 1

    with ThreadPoolExecutor(1) as pool:
        writer = pool.submit(queue.push_many, items())
        try:
            assert entered.wait(timeout=30)
            for call in (queue.clear, queue.drain, queue.shrink_to_fit):
                with pytest.raises(RuntimeError, match="^Already borrowed$"):
                    call()
        finally:
            resume.set()
        assert writer.result(timeout=30) == 1
    assert queue.drain() == [1]


def test_fifo_shrink_preserves_order_and_reclaims_pages():
    queue = pc.ConcurrentQueue(range(4096))
    assert queue.allocated_pages > 0
    assert queue.pop_many(4093) == list(range(4093))
    queue.shrink_to_fit()
    assert queue.allocated_pages > 0
    assert queue.drain() == [4093, 4094, 4095]
    queue.shrink_to_fit()
    assert queue.allocated_pages == 0
    with pytest.raises(IndexError):
        queue.pop()


@pytest.mark.parametrize("operation", [copy.copy, copy.deepcopy, pickle.dumps])
def test_fifo_copy_and_pickle_are_unsupported(operation):
    queue = pc.ConcurrentQueue([1])
    with pytest.raises(TypeError, match=r"drain\(\) into a list first"):
        operation(queue)
    assert queue.pop() == 1


@pytest.mark.parametrize("operation", ["pop", "clear", "drain", "del"])
def test_fifo_releases_values(operation):
    class Value:
        pass

    queue = pc.ConcurrentQueue()
    value = Value()
    ref = weakref.ref(value)
    queue.push(value)
    del value
    assert ref() is not None

    if operation == "pop":
        popped = queue.pop()
        assert popped is ref()
        del popped
    elif operation == "clear":
        queue.clear()
    elif operation == "drain":
        drained = queue.drain()
        assert len(drained) == 1 and drained[0] is ref()
        del drained
    else:
        del queue
    assert ref() is None


def test_fifo_multi_producer_multi_consumer():
    producers = consumers = 4
    width = 4096
    queue = pc.ConcurrentQueue()
    start = threading.Barrier(producers + consumers)
    producers_done = threading.Event()

    def produce(producer):
        start.wait(timeout=30)
        for offset in range(width):
            queue.push((producer, offset))

    def consume():
        start.wait(timeout=30)
        received = []
        while True:
            done = producers_done.is_set()
            try:
                received.append(queue.pop())
            except IndexError:
                if done:
                    return received
                producers_done.wait(0.001)

    with ThreadPoolExecutor(producers + consumers) as pool:
        readers = [pool.submit(consume) for _ in range(consumers)]
        writers = [pool.submit(produce, i) for i in range(producers)]
        try:
            for writer in writers:
                writer.result(timeout=30)
        finally:
            producers_done.set()
        delivered = [reader.result(timeout=30) for reader in readers]

    for received in delivered:
        last = [-1] * producers
        for producer, offset in received:
            assert offset > last[producer]
            last[producer] = offset
    assert sorted(item for received in delivered for item in received) == [
        (producer, offset) for producer in range(producers) for offset in range(width)
    ]
    assert not queue


@pytest.mark.parametrize("is_min", [False, True])
def test_priority_queue_exact_order(priority_queue_cls, is_min):
    items = [(1, 8), (8192, -2), (16384, 5), (32768, 3)]
    queue = priority_queue_cls(items, min=is_min, shards=5)
    expected = sorted(items, key=lambda item: item[1], reverse=not is_min)
    assert queue.is_min is is_min
    assert len(queue) == len(items) and queue
    assert queue.allocated_pages == len(items)
    assert dict(queue.items()) == dict(items)
    assert queue.sorted() == expected
    for entry in expected:
        assert queue.peek_exact() == entry
        assert queue.pop_exact() == entry
    assert len(queue) == 0 and not queue
    for call in (queue.pop, queue.pop_exact, queue.peek_exact):
        with pytest.raises(IndexError):
            call()


@pytest.mark.parametrize("is_min", [False, True])
def test_pop_exact_is_globally_ordered_across_many_shards(priority_queue_cls, is_min):
    # Many nonempty shards, so a relaxed pop in place of pop_exact would
    # break the order with near certainty.
    rng = random.Random(7)
    priorities = rng.sample(range(-1000, 1000), 64)
    items = [(index * 8192, priority) for index, priority in enumerate(priorities)]
    queue = priority_queue_cls(items, min=is_min, shards=8)
    expected = sorted(items, key=lambda item: item[1], reverse=not is_min)
    assert [queue.pop_exact() for _ in expected] == expected


def test_priority_queue_rejects_huge_shard_counts(priority_queue_cls):
    assert priority_queue_cls(shards=65536).shards == 65536
    for shards in (65537, 2**63, 2**100):
        with pytest.raises(OverflowError):
            priority_queue_cls(shards=shards)


def test_priority_queue_is_not_a_sequence(priority_queue_cls):
    queue = priority_queue_cls([(0, 7)])
    with pytest.raises(TypeError):
        iter(queue)


def test_clear_lets_finalizers_push_back():
    queue = pc.ConcurrentQueue()
    pushed = []

    class PushOnDel:
        def __del__(self):
            try:
                queue.push("from finalizer")
                pushed.append(True)
            except RuntimeError:
                pushed.append(False)

    queue.push(PushOnDel())
    queue.clear()
    assert pushed == [True]
    assert queue.drain() == ["from finalizer"]


def test_priority_queue_relaxed_pop_drains(priority_queue_cls):
    remaining = {i * 1024: -i for i in range(128)}
    queue = priority_queue_cls(remaining, shards=13)
    for _ in range(len(remaining)):
        key, priority = queue.pop()
        assert remaining.pop(key) == priority
    assert not remaining and queue.items() == []
    with pytest.raises(IndexError):
        queue.pop()


def test_priority_queue_mapping_operations_and_key_range(priority_queue_cls):
    queue = priority_queue_cls(shards=2)
    marker = object()
    assert queue.get(10) is None
    assert queue.get(10, marker) is marker
    assert queue.push(10, 1) is None
    assert queue.push(10, 2) == 1
    assert queue[10] == queue.get(10, marker) == 2
    assert 10 in queue
    queue[11] = 3
    assert queue.remove(11) == 3
    del queue[10]
    assert 10 not in queue

    for key in (-1, 2**32, 2**100, "10"):
        error = TypeError if isinstance(key, str) else OverflowError
        with pytest.raises(error):
            queue.push(key, 1)
        with pytest.raises(error):
            priority_queue_cls([(1, 1), (key, 2)])
        assert key not in queue
        assert queue.get(key, marker) is marker
        with pytest.raises(KeyError):
            queue[key]
        with pytest.raises(KeyError):
            queue.remove(key)
        with pytest.raises(KeyError):
            del queue[key]

    queue.push(2**32 - 1, 9)
    assert queue[2**32 - 1] == 9
    queue.clear()
    assert queue.items() == []


def test_priority_queue_f64_rejects_nan():
    queue = pc.ConcurrentPriorityQueueF64([(1, -math.inf), (2, math.inf)])
    with pytest.raises(ValueError, match="priority must not be NaN"):
        queue.push(1, math.nan)
    with pytest.raises(ValueError, match="priority must not be NaN"):
        queue[3] = math.nan
    for push in (queue.push_increase, queue.push_decrease):
        with pytest.raises(ValueError, match="priority must not be NaN"):
            push(1, math.nan)
    with pytest.raises(ValueError, match="priority must not be NaN"):
        pc.ConcurrentPriorityQueueF64([(1, 1.0), (2, math.nan)])
    assert queue.sorted() == [(2, math.inf), (1, -math.inf)]


@pytest.mark.parametrize("is_min", [False, True])
def test_conditional_push_uses_numeric_priority(priority_queue_cls, is_min):
    queue = priority_queue_cls(min=is_min, shards=4)
    assert queue.push_increase(1, 2)
    assert queue.push_decrease(2, 4)
    assert not queue.push_increase(1, 2)
    assert not queue.push_increase(1, 1)
    assert queue.push_increase(1, 5)
    assert not queue.push_decrease(2, 4)
    assert not queue.push_decrease(2, 5)
    assert queue.push_decrease(2, 3)
    for push in (queue.push_increase, queue.push_decrease):
        with pytest.raises(OverflowError):
            push(2**32, 1)
        with pytest.raises(TypeError):
            push("1", 1)
    expected = [(2, 3), (1, 5)]
    if not is_min:
        expected.reverse()
    assert [queue.pop_exact(), queue.pop_exact()] == expected


def test_priority_drain_and_shrink_require_exclusive_access(priority_queue_cls):
    queue = priority_queue_cls({4096: 10}, min=True, shards=2)

    class Key:
        def __index__(self):
            for call in (queue.drain, queue.shrink_to_fit):
                with pytest.raises(RuntimeError, match="^Already borrowed$"):
                    call()
            return 1

    queue.push(Key(), -3)
    queue.shrink_to_fit()
    pages = queue.allocated_pages
    assert pages > 0
    assert dict(queue.drain()) == {4096: 10, 1: -3}
    assert not queue
    assert queue.allocated_pages == pages
    assert queue.drain() == []
    with pytest.raises(IndexError):
        queue.pop_exact()
    queue.shrink_to_fit()
    assert queue.allocated_pages == 0
    queue.push(2, 4)
    assert queue.pop_exact() == (2, 4)


@pytest.mark.parametrize("is_min", [False, True])
def test_priority_repr_includes_config_and_truncates(priority_queue_cls, is_min):
    queue = priority_queue_cls({7: 2}, min=is_min, shards=3)
    priority = repr(float(2) if priority_queue_cls is pc.ConcurrentPriorityQueueF64 else 2)
    name = priority_queue_cls.__name__
    assert repr(queue) == f"{name}({{7: {priority}}}, min={is_min}, shards=3)"
    assert "__repr__" in priority_queue_cls.__dict__
    for key in range(25):
        queue.push(key, key)
    display = repr(queue)
    assert display.endswith(f"...}}, min={is_min}, shards=3)")
    assert display.count(": ") == 20


def test_priority_queue_constructor_inputs(priority_queue_cls):
    class KeysAndGetItem:
        def __init__(self):
            self.pairs = iter([(1, 2), (2, 5), (1, 7)])

        def keys(self):
            return iter([1, 2, 1])

        def __getitem__(self, key):
            expected, value = next(self.pairs)
            assert key == expected
            return value

        def __iter__(self):
            pytest.fail("mapping inputs use keys()")

    class DictSubclass(KeysAndGetItem, dict):
        def __init__(self):
            dict.__init__(self, {99: 99})
            super().__init__()

    expected = {1: 7, 2: 5}
    for items in (
        expected,
        MappingProxyType(expected),
        KeysAndGetItem(),
        DictSubclass(),
        iter([(1, 2), (2, 5), (1, 7)]),
    ):
        assert dict(priority_queue_cls(items).items()) == expected
    with pytest.raises((TypeError, ValueError)):
        priority_queue_cls([(1, 2), (3,)])


def test_priority_queue_shards_and_factory(priority_queue_cls):
    dtype = priority_queue_cls.dtype
    assert dtype == ("int64" if priority_queue_cls is pc.ConcurrentPriorityQueue else "float64")
    assert priority_queue_cls().shards == priority_queue_cls(shards=None).shards >= 1
    for shards in (1, 3, np.int64(6)):
        assert priority_queue_cls(shards=shards).shards == shards
    for shards in (0, -1, -(2**100)):
        with pytest.raises(ValueError, match="shards must be at least 1"):
            priority_queue_cls(shards=shards)
    with pytest.raises(TypeError):
        priority_queue_cls(shards=1.5)
    with pytest.raises(TypeError):
        priority_queue_cls(None, False, 1)
    queue = pc.concurrent_priority_queue(np.dtype(dtype), [(1, 2)], min=True, shards=3)
    assert type(queue) is priority_queue_cls
    assert queue.is_min and queue.shards == 3 and queue.items() == [(1, 2)]
    assert weakref.ref(queue)() is queue
    with pytest.raises(TypeError):
        hash(queue)


def test_factory_default_and_exports():
    assert type(pc.concurrent_priority_queue()) is pc.ConcurrentPriorityQueue
    assert type(pc.concurrent_priority_queue("float64")) is pc.ConcurrentPriorityQueueF64
    with pytest.raises(TypeError, match="concurrent priority queue"):
        pc.concurrent_priority_queue("float32")
    assert {
        "ConcurrentQueue", "ConcurrentPriorityQueue", "ConcurrentPriorityQueueF64",
        "concurrent_priority_queue",
    } <= set(pc.__all__)


@pytest.mark.parametrize("is_min", [False, True])
def test_priority_queue_copies_and_pickles(priority_queue_cls, is_min):
    items = {1: 10, 16384: 5, 32768: 9}
    queue = priority_queue_cls(items, min=is_min, shards=7)
    for restored in (
        queue.copy(),
        copy.copy(queue),
        copy.deepcopy(queue),
        pickle.loads(pickle.dumps(queue, protocol=0)),
        pickle.loads(pickle.dumps(queue, protocol=pickle.HIGHEST_PROTOCOL)),
    ):
        assert type(restored) is priority_queue_cls and restored is not queue
        assert restored.is_min is is_min and restored.shards == 7
        assert restored.sorted() == queue.sorted()
        restored.clear()
    assert dict(queue.items()) == items


def test_threaded_priority_queue_matches_accounting():
    threads, width = 8, 256
    queue = pc.ConcurrentPriorityQueue(shards=11)
    start = threading.Barrier(threads)
    inputs = [
        [(worker * 4096 + offset, worker * width + offset) for offset in range(width)]
        for worker in range(threads)
    ]

    def worker(entries):
        removed, popped, missed_removes = [], [], []
        start.wait(timeout=30)
        for offset, (key, priority) in enumerate(entries):
            assert queue.push(key, priority) is None
            if offset % 3 == 0:
                try:
                    removed.append((key, queue.remove(key)))
                except KeyError:
                    missed_removes.append(key)
            elif offset % 3 == 1:
                try:
                    popped.append(queue.pop())
                except IndexError:
                    pass
        return removed, popped, missed_removes

    with ThreadPoolExecutor(threads) as pool:
        results = list(pool.map(worker, inputs))

    expected = dict(entry for entries in inputs for entry in entries)
    popped_keys = {key for _, popped, _ in results for key, _ in popped}
    assert popped_keys
    for removed, popped, missed_removes in results:
        for key, priority in removed + popped:
            assert expected.pop(key) == priority
        assert set(missed_removes) <= popped_keys
    items = queue.items()
    assert len(items) == len(queue) == len(expected)
    assert dict(items) == expected
