from paged_collections import PagedList

values = PagedList([1, 2, 3])
values.extend(range(4, 10_001))
print("len =", len(values))
print("first =", values[0])
print("last =", values[-1])
print("popped =", values.pop())
values.insert(1, 99)
print("removed =", values.pop(1))
print("prefix =", list(values[:3]))
