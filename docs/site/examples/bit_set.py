from paged_collections import BitSet

bits = BitSet([62, 65])
print(bits.add_range(63, 67))
print(list(bits.range(63, 67)))
print(bits.next_occupant(64))
print(bits.to_mask(61, 68).tolist())
print(BitSet.from_mask([True, False, True], offset=62).to_numpy().tolist())
