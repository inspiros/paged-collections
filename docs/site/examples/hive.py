from paged_collections import Hive, id_generation, id_index

hive = Hive()
a = hive.insert(30)
b = hive.insert(10)
print("removed =", hive.pop(a))
c = hive.insert(20)
print("reused slot =", (id_index(c), id_generation(c)))
print("stale =", hive.get(a))
ordered = hive.sorted_ids()
print("sorted indices =", [id_index(key) for key in ordered])
print("sorted values =", [hive[key] for key in ordered])
print("slots before sort =", hive.values())
# Sorting moves values between slots; ids continue to name slots.
hive.sort()
print("slots after sort =", hive.values())
print("reused id now names =", hive[c])
