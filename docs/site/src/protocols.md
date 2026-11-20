# Protocols

## Pickle and Copy

Plain containers, concurrent maps and sets, and concurrent priority queues
support `pickle`, `copy.copy` and `copy.deepcopy`. Use `ConcurrentQueue.drain()`
to move FIFO contents into a list for copying or pickling.

- The page size is part of the pickled state. Pickles made before
  `page_size=` existed load with page size 1024.
- `Hive` and `IdAllocator` pickles keep generations and free slots, so ids
  stay valid across a round trip. A `Hive` rebuilds its free lists on load,
  so the slot the next insert takes can differ from the original. Restoring an older pickle revives ids
  removed since it was taken (see [Hive](hive.md#limits-and-stale-ids)).
- Concurrent maps, sets and priority queues pickle a snapshot of their contents (see
  [Python threads](python-threads.md#concurrent-classes)).

Loading a pickle whose page size the running build does not include raises
`ValueError` (see [page sizes](installation.md#page-sizes)).

## Weak References

Every container class supports `weakref.ref`.

## Subclassing

All container classes except the concurrent ones can be
subclassed. Pickle and copy keep the subclass, its `__dict__` and its
`__slots__` values (on 3.10 too, which lacks `object.__getstate__`).

## Equality and Hashing

Plain containers compare by contents with `==`. Concurrent containers compare
by identity. Concurrent maps, sets and FIFO queues hash by identity.
Plain containers and concurrent priority queues are unhashable.
Lists and deques also support `<`, `<=`, `>`, `>=` against the same class, as
`list` and `deque` do.

## Collection ABCs

The package registers these classes with `collections.abc` for `isinstance`
and `issubclass` checks. Virtual registration keeps each class's native methods
and return types.

| ABC | Classes |
|---|---|
| `MutableMapping` | `SparseMap`, `DirectMap`, `IdMap` and their typed variants |
| `Mapping` | Typed concurrent sparse and direct maps |
| `MutableSet` | `SparseSet` and `BitSet` |
| `MutableSequence` | `PagedList` and its typed variants |

## Example

```python
{{#include ../examples/protocols.py}}
```

```text
{{#include ../examples/protocols.python.txt}}
```
