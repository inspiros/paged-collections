from paged_collections import SparseSet

keys = SparseSet([3, 7, 11])
keys.add(19)
print("inserted =", list(keys))
keys.remove(7)
print("contains 7 =", 7 in keys)
print("dense =", list(keys))
other = SparseSet([11, 23])
print("union =", sorted(keys | other))
print("intersection =", sorted(keys & other))
print("difference =", sorted(keys - other))
keys.clear()
print("cleared len =", len(keys))
