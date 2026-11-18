import numpy as np
from paged_collections import sparse_map

values = sparse_map("float64")
keys = np.array([7, 2, 9], dtype=np.uint32)
print("inserted =", values.set_many(keys, np.array([1.5, 2.5, 3.5])))
print("lookup =", values.get_many([9, 99, 7], default=-1.0).tolist())
view = values.values_view()
view *= 2
print("write through =", values.get_many(keys).tolist())
try:
    values[100] = 4.5
except BufferError as error:
    print("resize with view =", type(error).__name__)
del view
values[100] = 4.5
print("len after release =", len(values))
