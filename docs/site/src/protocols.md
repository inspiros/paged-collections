# Protocols

## Pickle and Copy

All non-concurrent classes support `pickle`, `copy.copy` and `copy.deepcopy`.

- The page size is part of the pickled state. Pickles made before
  `page_size=` existed load with page size 1024.
- `Hive` and `IdAllocator` pickles keep generations and free lists, so ids
  stay valid across a round trip. Restoring an older pickle revives ids
  removed since it was taken (see [Hive](hive.md#limits-and-stale-ids)).
- Concurrent classes raise on pickling.

Loading a pickle whose page size the running build does not include raises
`ValueError` (see [page sizes](installation.md#page-sizes)).

## Weak References

Every container class supports `weakref.ref`.

## Subclassing

All container classes except the concurrent ones can be
subclassed. Pickle and copy keep the subclass, its `__dict__` and its
`__slots__` values (on 3.10 too, which lacks `object.__getstate__`).

## Equality and Hashing

Containers compare by contents with `==`. Mutable containers are unhashable.
Lists and deques also support `<`, `<=`, `>`, `>=` against the same class, as
`list` and `deque` do.

## Example

```python
{{#include ../examples/protocols.py}}
```

```text
{{#include ../examples/protocols.python.txt}}
```
