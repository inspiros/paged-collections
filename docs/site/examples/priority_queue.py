from paged_collections import PriorityQueue, PriorityQueueF64

queue = PriorityQueue()
for key, priority in [(7, 10), (2, 30), (9, 20), (4, 15)]:
    queue.push(key, priority)
# Pushing an existing key changes its priority and returns the old priority.
print("old priority =", queue.push(7, 40))
print("removed =", queue.remove(9))
print("pop order =", [queue.pop() for _ in range(len(queue))])

floats = PriorityQueueF64(min=True)
floats.push(7, 1.5)
floats.push(2, 0.5)
floats.push(9, 2.5)
floats.push(7, 0.25)
floats.remove(9)
print("float min order =", [floats.pop() for _ in range(len(floats))])
