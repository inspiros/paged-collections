"""Regressions for review findings 8, 10 and 11."""

import pickle
import queue
import sys
import threading
import time

import numpy as np
import pytest

import paged_collections as pc


class NewArgsList(pc.PagedList):
    def __new__(cls, token, *, marker):
        obj = super().__new__(cls)
        obj.new_args = (token, marker)
        return obj

    def __init__(self, token, *, marker):
        super().__init__([token, marker])

    def __getnewargs_ex__(self):
        return ("restored token",), {"marker": "restored marker"}

    def __getstate__(self):
        # Do not let instance state hide whether __new__ received the arguments.
        return None


class EmptyStateList(pc.PagedList):
    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        self.received_state = state


@pytest.mark.parametrize("protocol", range(2, pickle.HIGHEST_PROTOCOL + 1))
def test_pickle_subclass_passes_getnewargs_ex_to_required_new(protocol):
    # Finding 11: subclass reconstruction must honor __getnewargs_ex__ arguments.
    original = NewArgsList("original token", marker="original marker")
    restored = pickle.loads(pickle.dumps(original, protocol=protocol))
    assert type(restored) is NewArgsList
    assert restored.new_args == ("restored token", "restored marker")
    assert list(restored) == ["original token", "original marker"]


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_pickle_subclass_calls_setstate_for_empty_dict(protocol):
    # Finding 11: an empty __getstate__ dictionary must still reach __setstate__.
    original = EmptyStateList([1, 2, 3])
    restored = pickle.loads(pickle.dumps(original, protocol=protocol))
    assert type(restored) is EmptyStateList
    assert list(restored) == [1, 2, 3]
    assert restored.received_state == {}


@pytest.mark.parametrize("large", [2**63, 2**63 + 1])
def test_paged_list_u64_accepts_mixed_numpy_integer_scalars_exactly(large):
    # Finding 10: NumPy's mixed signed/unsigned promotion must not reject integer sequences.
    floats = np.array([0, large], dtype=np.float64)
    with pytest.raises(TypeError):
        pc.PagedListU64(floats)

    values = pc.PagedListU64([np.int64(0), np.uint64(large)])
    assert list(values) == [0, large]
    assert values.to_numpy().dtype == np.dtype(np.uint64)
    np.testing.assert_array_equal(
        values.to_numpy(), np.array([0, large], dtype=np.uint64)
    )


@pytest.mark.skipif(
    getattr(sys, "_is_gil_enabled", lambda: True)(),
    reason="the post-search borrow gap requires a GIL-disabled interpreter",
)
def test_deque_remove_does_not_use_index_stale_after_concurrent_popleft():
    # Finding 8: remove must not delete a bystander after a concurrent popleft shifts its match.
    requests = queue.Queue()
    responses = queue.Queue()
    needle = object()

    class Match:
        def __init__(self, ready):
            self.ready = ready

        def __eq__(self, other):
            if other is needle:
                # Let popleft race the final mutation check and removal borrow.
                self.ready.set()
                return True
            return False

    def pop_worker():
        try:
            while True:
                request = requests.get()
                if request is None:
                    return
                deque, ready, armed, delay = request
                deadline = time.monotonic() + 5
                armed.set()
                spins = 0
                # Stay runnable on another core: waking a sleeping waiter
                # generally takes longer than the entire vulnerable gap.
                while not ready.is_set():
                    spins += 1
                    if spins % 64 == 0 and time.monotonic() >= deadline:
                        raise TimeoutError("remove did not reach its comparison")
                for _ in range(delay):
                    pass
                while True:
                    try:
                        popped = deque.popleft()
                        break
                    except RuntimeError as exc:
                        if "borrow" not in str(exc) or time.monotonic() >= deadline:
                            raise
                responses.put((popped, None))
        except BaseException as exc:
            responses.put((None, exc))

    worker = threading.Thread(target=pop_worker, daemon=True)
    worker.start()
    try:
        # An equality callback cannot pause *after* find's mutation check, so
        # repeat this bounded race on free-threaded builds rather than claim
        # a deterministic callback-only reproduction.
        for attempt in range(20_000):
            ready = threading.Event()
            armed = threading.Event()
            head, tail = object(), object()
            match = Match(ready)
            deque = pc.PagedDeque([head, match, tail])
            requests.put((deque, ready, armed, attempt % 8))
            assert armed.wait(timeout=6), "popleft worker did not start before the deadline"
            removed = False
            try:
                deque.remove(needle)
                removed = True
            except (IndexError, ValueError, RuntimeError):
                pass
            finally:
                # Also release the pop worker if a borrow failed before __eq__.
                ready.set()

            popped, error = responses.get(timeout=6)
            assert error is None, repr(error)
            assert popped is head, f"popleft lost its original head on attempt {attempt}"
            remaining = list(deque)
            expected = [tail] if removed else [match, tail]
            assert len(remaining) == len(expected) and all(
                actual is wanted for actual, wanted in zip(remaining, expected)
            ), f"remove deleted the wrong item on attempt {attempt}: {remaining!r}"
    finally:
        requests.put(None)
        worker.join(timeout=6)
        assert not worker.is_alive(), "popleft worker did not stop before the deadline"
