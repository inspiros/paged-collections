# Rust and Python Names

| Rust | Python | Python variants |
|---|---|---|
| `sparse_set::PagedSparseSet<u32>` | `SparseSet` | |
| `sparse_map::PagedSparseMap<u32, V>` | `SparseMap` | `SparseMapF64`, `F32`, `I64`, `I32`, `U64`, `U32`, `Bool`; `sparse_map(dtype)` |
| `sparse_map::PagedSparseMap<Id, V>` | `IdMap` | `IdMapF64`, ...; `id_map(dtype)` |
| `direct_map::PagedDirectMap<u32, V>` | `DirectMap` | `DirectMapF64`, ...; `direct_map(dtype)` |
| `priority_queue::PagedPriorityQueue<u32, P>` | `PriorityQueue` (int64), `PriorityQueueF64` | `min=True` |
| `paged_vec::PagedVec<T>` | `PagedList` | `PagedListF64`, ...; `paged_list(dtype)` |
| `paged_deque::PagedDeque<T>` | `PagedDeque` | |
| `hive::Hive<T>` | `Hive` | |
| `id_alloc::IdAllocator`, `Id` | `IdAllocator`, packed `int` ids | `id_index`, `id_generation`, `make_id`, `id_indices` |
| `concurrent::ConcurrentSparseSet<u32>` | `ConcurrentSparseSet` | |
| `concurrent::ConcurrentSparseMap<u32, V>` | `ConcurrentSparseMapF64`, ... | typed only |
| `concurrent::ConcurrentDirectMap<u32, V>` | `ConcurrentDirectMapF64`, ... | typed only |
| `concurrent::ConcurrentHive`, `ConcurrentAppendHive`, `ConcurrentPagedVec`, `AtomicDirectMap`, `AtomicIdMap`, `ConcurrentIdAllocator` | | Rust only |
| `PAGE` const generic | `page_size=` keyword | `PAGE_SIZES`, `DEFAULT_PAGE_SIZE` |
| `page::Flat`, `Radix`, `Tree` | always `Flat` | |

Module constants: `KEY_LIMIT` (2**32), `MAX_GENERATION` (2**31),
`MAX_INDICES`.

Reference documentation:

- Rust: [docs.rs/paged-collections](https://docs.rs/paged-collections)
- Python: [`_core.pyi`](https://github.com/inspiros/paged-collections/blob/main/python/paged_collections/_core.pyi)
  (type stubs; also used by IDEs)
