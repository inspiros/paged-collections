from paged_collections import PagedDeque

values = PagedDeque([1, 2, 3])
values.appendleft(0)
values.append(4)
print("values =", list(values))
print("index 2 =", values[2])
print("front =", values.popleft())
print("back =", values.pop())
print("remaining =", list(values))
values.rotate(1)
print("rotated =", list(values))
