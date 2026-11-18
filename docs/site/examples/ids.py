from paged_collections import IdAllocator, IdMap, id_generation, id_index

ids = IdAllocator()
a, b = ids.alloc(), ids.alloc()
scores = IdMap([a, b], [10, 20])
print("freed =", ids.free(a))
c = ids.alloc()
print("reused slot =", (id_index(c), id_generation(c)))
# The side table is independent of the allocator; replace its old generation.
scores[c] = 30
print("stale alive =", ids.is_alive(a))
print("stale side =", scores.get(a))
print("scores =", scores.values())
print("live ids =", len(ids))
