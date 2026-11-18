from paged_collections import DirectMap

page_size = 4096
keys = [7, 5000, 4_000_000_000]
values = DirectMap(page_size=page_size)
for key in keys:
    values[key] = key + 1
print("page size =", values.page_size)
print("keys =", list(values))
print("len =", len(values))
print("allocated pages =", values.allocated_pages)
