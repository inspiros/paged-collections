from paged_collections import SparseMap, sparse_map

values = sparse_map("float64")
values[7] = 1.5
values[2] = 2.5
values[9] = 3.5
print("get 7 =", values.get(7))
print("removed =", values.pop(2))
print("keys =", list(values))
print("values =", values.values().tolist())
objects = SparseMap()
objects[7] = (4, 5)
print("object =", objects[7])
