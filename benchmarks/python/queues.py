"""Queue and priority-queue throughput against the standard library.

Run against an installed release build. Prints one table row per case:
median seconds over ``--repeat`` runs and millions of items per second.

Cases
-----
fifo/1t      push N then pop N on one thread: deque, SimpleQueue, ConcurrentQueue
             (item at a time, and push_many/pop_many in batches).
fifo/mpmc    T producers and T consumers move N items in total. SimpleQueue
             uses blocking get(); deque and ConcurrentQueue retry on empty.
pq/1t        push N distinct keys with random priorities, then pop all:
             heapq on (priority, key) tuples, PriorityQueueF64,
             ConcurrentPriorityQueueF64 (pop and pop_exact).
pq/update    push N keys, re-prioritise N/2 random keys, pop all. heapq uses
             lazy invalidation (dict of current priority, skip stale entries);
             the paged queues update in place.
pq/mtT       T threads each push their own N/T keys, then all pop until empty:
             heapq under a Lock vs ConcurrentPriorityQueueF64.pop.
"""

import argparse
from collections import deque
import heapq
from queue import SimpleQueue
import random
from statistics import median
import sys
import threading
from time import perf_counter

from paged_collections import (
    ConcurrentPriorityQueueF64,
    ConcurrentQueue,
    PriorityQueueF64,
)

BATCH = 1024


def timed(setup, body, repeat):
    times = []
    for _ in range(repeat):
        state = setup()
        start = perf_counter()
        body(state)
        times.append(perf_counter() - start)
        del state
    return median(times)


def run_threads(targets):
    """Start threads, release them together, return wall time of the work."""
    gate = threading.Barrier(len(targets) + 1)

    def wrap(fn):
        def run():
            gate.wait()
            fn()

        return run

    threads = [threading.Thread(target=wrap(fn)) for fn in targets]
    for t in threads:
        t.start()
    gate.wait()
    start = perf_counter()
    for t in threads:
        t.join()
    return perf_counter() - start


def row(case, impl, n, seconds, extra=""):
    print(f"{case:<11} {impl:<34} {seconds:9.4f} s {n / seconds / 1e6:8.2f} M/s {extra}", flush=True)


# ---------------------------------------------------------------- FIFO

def fifo_single(n, repeat):
    items = list(range(n))

    def deque_body(q):
        for x in items:
            q.append(x)
        for _ in items:
            q.popleft()

    def simple_body(q):
        for x in items:
            q.put(x)
        for _ in items:
            q.get_nowait()

    def cq_body(q):
        for x in items:
            q.push(x)
        for _ in items:
            q.pop()

    def cq_batch_body(q):
        for i in range(0, n, BATCH):
            q.push_many(items[i : i + BATCH])
        got = 0
        while got < n:
            got += len(q.pop_many(BATCH))

    for name, make, body in [
        ("collections.deque", deque, deque_body),
        ("queue.SimpleQueue", SimpleQueue, simple_body),
        ("ConcurrentQueue", ConcurrentQueue, cq_body),
        (f"ConcurrentQueue batch={BATCH}", ConcurrentQueue, cq_batch_body),
    ]:
        row("fifo/1t", name, n, timed(make, body, repeat))


def fifo_mpmc(n, threads, repeat):
    per = n // threads
    total = per * threads

    def bench(make, produce, consume):
        times = []
        for _ in range(repeat):
            q = make()
            counts = [0] * threads
            targets = [lambda: produce(q, per) for _ in range(threads)]
            targets += [(lambda i: lambda: consume(q, per, counts, i))(i) for i in range(threads)]
            times.append(run_threads(targets))
            assert sum(counts) == total, (sum(counts), total)
        return median(times)

    def deque_produce(q, k):
        for x in range(k):
            q.append(x)

    def deque_consume(q, k, counts, i):
        got = 0
        pop = q.popleft
        while got < k:
            try:
                pop()
                got += 1
            except IndexError:
                pass
        counts[i] = got

    def simple_produce(q, k):
        for x in range(k):
            q.put(x)

    def simple_consume(q, k, counts, i):
        get = q.get
        for _ in range(k):
            get()
        counts[i] = k

    def cq_produce(q, k):
        for x in range(k):
            q.push(x)

    def cq_consume(q, k, counts, i):
        got = 0
        pop = q.pop
        while got < k:
            try:
                pop()
                got += 1
            except IndexError:
                pass
        counts[i] = got

    def cq_batch_produce(q, k):
        chunk = list(range(BATCH))
        done = 0
        while done < k:
            m = min(BATCH, k - done)
            q.push_many(chunk[:m])
            done += m

    def cq_batch_consume(q, k, counts, i):
        got = 0
        while got < k:
            got += len(q.pop_many(min(BATCH, k - got)))
        counts[i] = got

    tag = f"fifo/{threads}+{threads}"
    for name, make, prod, cons in [
        ("collections.deque (spin)", deque, deque_produce, deque_consume),
        ("queue.SimpleQueue (blocking get)", SimpleQueue, simple_produce, simple_consume),
        ("ConcurrentQueue (spin)", ConcurrentQueue, cq_produce, cq_consume),
        (f"ConcurrentQueue batch={BATCH}", ConcurrentQueue, cq_batch_produce, cq_batch_consume),
    ]:
        row(tag, name, total, bench(make, prod, cons))


# ---------------------------------------------------------------- priority queues

def pq_data(n, seed=0):
    rng = random.Random(seed)
    keys = list(range(n))
    rng.shuffle(keys)
    prios = [rng.random() for _ in range(n)]
    return keys, prios


def pq_single(n, repeat):
    keys, prios = pq_data(n)

    def heap_body(_):
        h = []
        push = heapq.heappush
        for k, p in zip(keys, prios):
            push(h, (p, k))
        pop = heapq.heappop
        while h:
            pop(h)

    def make_pq(cls):
        return lambda: cls(min=True)

    def pq_body(q):
        push = q.push
        for k, p in zip(keys, prios):
            push(k, p)
        pop = q.pop
        for _ in range(n):
            pop()

    def cpq_exact_body(q):
        push = q.push
        for k, p in zip(keys, prios):
            push(k, p)
        pop = q.pop_exact
        for _ in range(n):
            pop()

    for name, make, body in [
        ("heapq (prio, key) tuples", lambda: None, heap_body),
        ("PriorityQueueF64", make_pq(PriorityQueueF64), pq_body),
        ("ConcurrentPriorityQueueF64.pop", make_pq(ConcurrentPriorityQueueF64), pq_body),
        ("ConcurrentPriorityQueueF64.pop_exact", make_pq(ConcurrentPriorityQueueF64), cpq_exact_body),
    ]:
        row("pq/1t", name, n, timed(make, body, repeat))


def pq_update(n, repeat):
    keys, prios = pq_data(n)
    rng = random.Random(1)
    upd_keys = [rng.randrange(n) for _ in range(n // 2)]
    upd_prios = [rng.random() for _ in range(n // 2)]
    ops = n + n // 2

    def heap_body(_):
        h, cur = [], {}
        push, pop = heapq.heappush, heapq.heappop
        for k, p in zip(keys, prios):
            cur[k] = p
            push(h, (p, k))
        for k, p in zip(upd_keys, upd_prios):
            cur[k] = p
            push(h, (p, k))
        popped = 0
        while h:
            p, k = pop(h)
            if cur.get(k) == p:
                del cur[k]
                popped += 1
        assert popped == n

    def pq_body(q):
        push = q.push
        for k, p in zip(keys, prios):
            push(k, p)
        for k, p in zip(upd_keys, upd_prios):
            push(k, p)
        pop = q.pop
        for _ in range(n):
            pop()
        assert not q

    for name, make, body in [
        ("heapq + lazy invalidation", lambda: None, heap_body),
        ("PriorityQueueF64", lambda: PriorityQueueF64(min=True), pq_body),
        ("ConcurrentPriorityQueueF64.pop", lambda: ConcurrentPriorityQueueF64(min=True), pq_body),
    ]:
        row("pq/update", name, ops, timed(make, body, repeat))


def pq_threads(n, threads, repeat):
    keys, prios = pq_data(n)
    per = n // threads
    parts = [(keys[i * per : (i + 1) * per], prios[i * per : (i + 1) * per]) for i in range(threads)]
    total = per * threads

    def bench(make, worker):
        times = []
        for _ in range(repeat):
            state = make()
            counts = [0] * threads
            mid = threading.Barrier(threads)
            targets = [(lambda i: lambda: worker(state, parts[i], mid, counts, i))(i) for i in range(threads)]
            times.append(run_threads(targets))
            assert sum(counts) == total, (sum(counts), total)
        return median(times)

    def heap_worker(state, part, mid, counts, i):
        h, lock = state
        push, pop = heapq.heappush, heapq.heappop
        for k, p in zip(*part):
            with lock:
                push(h, (p, k))
        mid.wait()
        got = 0
        while True:
            with lock:
                if not h:
                    break
                pop(h)
            got += 1
        counts[i] = got

    def cpq_worker(q, part, mid, counts, i):
        push, pop = q.push, q.pop
        for k, p in zip(*part):
            push(k, p)
        mid.wait()
        got = 0
        while True:
            try:
                pop()
            except IndexError:
                break
            got += 1
        counts[i] = got

    tag = f"pq/mt{threads}"
    row(tag, "heapq + threading.Lock", 2 * total, bench(lambda: ([], threading.Lock()), heap_worker))
    row(tag, "ConcurrentPriorityQueueF64.pop", 2 * total,
        bench(lambda: ConcurrentPriorityQueueF64(min=True), cpq_worker))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=1_000_000)
    parser.add_argument("--threads", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    gil = sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else True
    print(f"# Python {sys.version.split()[0]}  GIL={'on' if gil else 'off'}  n={args.n}  repeat={args.repeat}")
    print("# M/s = items (pq/update: operations; pq/mtT: pushes + pops) per second")
    fifo_single(args.n, args.repeat)
    for t in args.threads:
        fifo_mpmc(args.n, t, args.repeat)
    pq_single(args.n, args.repeat)
    pq_update(args.n, args.repeat)
    for t in args.threads:
        pq_threads(args.n, t, args.repeat)


if __name__ == "__main__":
    main()
