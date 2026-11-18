from paged_collections import Hive, IdMap, id_generation, id_index

hive = Hive()
a = hive.insert("a")
b = hive.insert("b")
score = IdMap([a, b], [10, 20])
hive.pop(a)
c = hive.insert("c")
print("reused slot =", (id_index(c), id_generation(c)))
print("new side entry =", score.get(c))
# Writing the new generation evicts the stale entry from the side table.
score[c] = 30
print("stale hive =", hive.get(a))
print("stale side =", score.get(a))
print("side values =", score.values())
