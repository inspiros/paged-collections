import pickle
import weakref

from paged_collections import SparseMap


class TaggedMap(SparseMap):
    __slots__ = ("tag",)


values = SparseMap([7, 2], [10, 20], page_size=256)
restored = pickle.loads(pickle.dumps(values))
print("page size =", restored.page_size)
print("items =", restored.items())
reference = weakref.ref(values)
print("weakref alive =", reference() is values)
del values
print("weakref cleared =", reference() is None)
tagged = TaggedMap([9], [30], page_size=256)
tagged.tag = "scores"
restored_tagged = pickle.loads(pickle.dumps(tagged))
print("subclass =", type(restored_tagged).__name__)
print("slot =", restored_tagged.tag)
print("subclass page size =", restored_tagged.page_size)
print("subclass items =", restored_tagged.items())
