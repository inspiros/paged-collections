from concurrent.futures import ThreadPoolExecutor

import numpy as np
from paged_collections import ConcurrentSparseMapF64

values = ConcurrentSparseMapF64()


def insert_chunk(worker):
    keys = np.arange(worker * 1000, (worker + 1) * 1000, dtype=np.uint32)
    return values.insert_many(keys, keys.astype(np.float64))


with ThreadPoolExecutor(max_workers=4) as pool:
    inserted = list(pool.map(insert_chunk, range(4)))
print("inserted =", sum(inserted))
print("len =", len(values))
print("checksum =", float(values.values().sum()))
