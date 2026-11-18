from paged_collections import DirectMap

values = DirectMap([20, 3, 10], [200, 30, 100])
print("get 3 =", values.get(3))
print("default =", values.setdefault(7, 70))
print("existing =", values.setdefault(7, 999))
print("ordered =", values.items())
print("range =", values.items_range(3, 20))
print("removed =", values.pop(10))
print("keys =", list(values))
