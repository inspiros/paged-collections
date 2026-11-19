use core::hash::{Hash, Hasher};

/// Hashes items consistently with order-insensitive equality.
///
/// Each item is hashed with fixed keys. Against crafted inputs, the outer
/// hasher can distinguish inputs only through the combined count, sum and XOR.
pub(crate) fn hash_unordered<T: Hash, H: Hasher>(items: impl Iterator<Item = T>, state: &mut H) {
    let mut count = 0;
    let mut sum = 0_u64;
    let mut xor = 0;
    for item in items {
        // Core has no other stable keyed hasher, and RandomState requires std.
        #[allow(deprecated)]
        let mut item_state = core::hash::SipHasher::new_with_keys(0, 0);
        item.hash(&mut item_state);
        let hash = item_state.finish();
        count += 1;
        sum = sum.wrapping_add(hash);
        xor ^= hash;
    }
    state.write_usize(count);
    state.write_u64(sum);
    state.write_u64(xor);
}
