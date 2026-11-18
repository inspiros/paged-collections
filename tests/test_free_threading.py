"""Shared use of the non-concurrent classes from many threads.

On free-threaded builds these calls really run in parallel; on GIL builds a
short switch interval interleaves them. Conflicting calls on one object may
raise ``RuntimeError`` (a held borrow) or ``BufferError`` (a live view), and
must never crash or lose an update that a call reported as done.
"""

from concurrent.futures import ThreadPoolExecutor
import sys
import sysconfig
import threading

import numpy as np
import pytest

import paged_collections as pc


FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
THREADS = 8


@pytest.fixture(autouse=True)
def _short_switch_interval():
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    yield
    sys.setswitchinterval(old)


def _retry(call, *args):
    """``call(*args)``, repeated while another thread holds the object."""
    while True:
        try:
            return call(*args)
        except RuntimeError as err:
            if "borrowed" not in str(err):
                raise
        except BufferError:
            pass


def _run(worker, n=THREADS):
    barrier = threading.Barrier(n)

    def start(i):
        barrier.wait()
        return worker(i)

    with ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(start, range(n)))


@pytest.mark.skipif(not FREE_THREADED, reason="GIL build")
def test_import_keeps_the_gil_disabled() -> None:
    assert not sys._is_gil_enabled()


def test_shared_sparse_set_keeps_every_reported_add() -> None:
    s = pc.SparseSet()
    per = 2000

    def worker(i):
        base = i * per
        for k in range(base, base + per):
            _retry(s.add, k)
            if k % 3 == 0:
                _retry(s.discard, k)
            if k % 97 == 0:
                _retry(len, s)
                _retry(s.__contains__, k)

    _run(worker)
    expected = {k for k in range(THREADS * per) if k % 3}
    assert set(s) == expected
    assert len(s) == len(expected)


def test_shared_typed_map_views_and_inserts() -> None:
    m = pc.SparseMapF64()
    per = 500
    readers = THREADS // 2

    def worker(i):
        if i < readers:
            # Views block resizing; a reader sees a consistent length.
            for _ in range(200):
                v = _retry(m.values_view)
                assert v.shape[0] <= THREADS * per
                float(np.sum(v))
                del v
        else:
            base = (i - readers) * per
            for k in range(base, base + per):
                _retry(m.__setitem__, k, float(k))

    _run(worker)
    assert len(m) == (THREADS - readers) * per
    keys = np.sort(m.keys_view())
    np.testing.assert_array_equal(keys, np.arange(len(m), dtype=np.uint32))
    assert all(m[k] == float(k) for k in range(len(m)))


def test_shared_paged_list_appends_and_pops() -> None:
    lst = pc.PagedListI64()
    per = 2000

    def worker(i):
        popped = []
        for j in range(per):
            _retry(lst.append, i * per + j)
            if j % 4 == 0:
                popped.append(_retry(lst.pop))
        return popped

    popped = [v for part in _run(worker) for v in part]
    assert sorted(list(lst) + popped) == list(range(THREADS * per))


def test_iterating_while_another_thread_mutates() -> None:
    m = pc.SparseMap()
    for k in range(1000):
        m[k] = k
    stop = threading.Event()

    def mutate():
        k = 1000
        while not stop.is_set():
            _retry(m.__setitem__, k, k)
            _retry(m.pop, k)
            k += 1

    t = threading.Thread(target=mutate)
    t.start()
    try:
        for _ in range(200):
            try:
                for k, v in m.items():
                    assert k == v
            except RuntimeError:
                pass  # changed size during iteration, or a held borrow
    finally:
        stop.set()
        t.join()
    assert len(m) == 1000
