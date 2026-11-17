"""Batch scaling on stock CPython; run against an installed release build."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from statistics import median
import threading
from time import perf_counter

import numpy as np

from paged_collections import ConcurrentDirectMapF64, ConcurrentSparseMapF64


def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def measure(cls, operation, keys, values, threads, repeat):
    chunks = list(zip(np.array_split(keys, threads), np.array_split(values, threads)))
    timings = []
    with ThreadPoolExecutor(max_workers=threads) as pool:
        # Start all workers outside the timed region. Reuse them for every run.
        ready = threading.Barrier(threads + 1)
        started = [pool.submit(ready.wait) for _ in range(threads)]
        ready.wait()
        for future in started:
            future.result()
        for _ in range(repeat):
            m = cls()
            write = m.set_many if cls is ConcurrentDirectMapF64 else m.insert_many
            if operation == "get_many":
                write(keys, values)

            def run(chunk):
                batch_keys, batch_values = chunk
                if operation == "get_many":
                    return m.get_many(batch_keys, 0.0)
                return write(batch_keys, batch_values)

            start = perf_counter()
            results = list(pool.map(run, chunks))
            timings.append(perf_counter() - start)
            # Retain outputs until the timer stops; insertion always starts empty.
            del results
    return median(timings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=positive_int, default=2_000_000)
    parser.add_argument("--threads", type=positive_int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--repeat", type=positive_int, default=5)
    args = parser.parse_args()
    if args.n > 2**32:
        parser.error("--n must be at most 2**32 (uint32 key domain)")
    if any(t > args.n for t in args.threads):
        parser.error("--threads must not exceed --n")
    # A random permutation gives random accesses and N genuine insertions.
    # A bounded key range avoids mostly empty pages and huge Flat directories.
    keys = np.random.default_rng(12345).permutation(np.arange(args.n, dtype=np.uint32))
    values = keys.astype(np.float64)
    thread_counts = sorted(set([1, *args.threads]))
    print(f"N={args.n:,}; median of {args.repeat} runs; includes input/output copies and dispatch.\n")
    print("| Class | Operation | Threads | Median (s) | Throughput (M keys/s) | Speedup |")
    print("|---|---|---:|---:|---:|---:|")
    for cls in (ConcurrentDirectMapF64, ConcurrentSparseMapF64):
        write_name = "set_many" if cls is ConcurrentDirectMapF64 else "insert_many"
        for operation in ("get_many", write_name):
            baseline = None
            for threads in thread_counts:
                seconds = measure(cls, operation, keys, values, threads, args.repeat)
                if baseline is None:
                    baseline = seconds
                print(f"| {cls.__name__} | {operation} | {threads} | {seconds:.6f} | "
                      f"{args.n / seconds / 1e6:.2f} | {baseline / seconds:.2f}x |", flush=True)


if __name__ == "__main__":
    main()
