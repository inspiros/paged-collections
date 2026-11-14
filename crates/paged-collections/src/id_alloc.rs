//! [`IdAllocator`]: hands out generational [`Id`]s and recycles freed ones.
//!
//! An [`Id`] is a 32-bit slot index plus a generation. Freeing an id makes
//! its index available again, and the next id at that index gets the next
//! generation, so a stale copy of the freed id never matches a live one.
//! [`Id`] implements [`Key`], so ids work directly as keys of the other
//! collections, where all generations of an index share one slot.
//!
//! | Operation | Cost |
//! |---|---|
//! | [`alloc`](IdAllocator::alloc), [`free`](IdAllocator::free) | amortized O(1) |
//! | [`is_alive`](IdAllocator::is_alive), [`occupant`](IdAllocator::occupant) | O(1) |
//! | [`iter`](IdAllocator::iter), [`retain`](IdAllocator::retain), [`clear`](IdAllocator::clear) | O([`index_bound`](IdAllocator::index_bound)) |
//!
//! Freed indices are reused before new ones are issued, last freed first.
//! [`index_bound`](IdAllocator::index_bound) is therefore at most the peak
//! number of live ids plus the number of retired indices, which keeps the
//! paged indices of the collections compact.
//!
//! Each index issues at most [`MAX_GENERATION`] ids. When an id with the
//! last generation is freed, its index is *retired*: it is never issued
//! again, so generations never wrap.
//!
//! # Scope of the stale-id guarantee
//!
//! A freed id is never alive again *within one allocation history*. Ids
//! carry no allocator identity: two allocators issue equal ids, and each
//! accepts the other's. Restoring an earlier state (assigning an older
//! clone, or deserializing an older snapshot) rolls generations back, so ids
//! freed after that state become alive again.
//!
//! The module exists only on 32- and 64-bit targets, where a `u32` index
//! converts to `usize` without loss.
//!
//! ```
//! use paged_collections::id_alloc::{Id, IdAllocator};
//! use paged_collections::sparse_map::PagedSparseMap;
//!
//! let mut ids = IdAllocator::new();
//! let mut names: PagedSparseMap<Id, &str> = PagedSparseMap::new();
//!
//! let a = ids.alloc();
//! names.insert(a, "a");
//! assert!(ids.free(a));
//!
//! // The index is reused with a new generation; the stale id is not found.
//! let b = ids.alloc();
//! assert_eq!(b.index(), a.index());
//! assert_ne!(b, a);
//! assert!(!ids.is_alive(a));
//! assert_eq!(names.get(b), None);
//! ```

use alloc::collections::TryReserveError;
use alloc::vec::Vec;
use core::fmt;
use core::iter::{Enumerate, FusedIterator};
use core::num::NonZeroU32;

use crate::key::{DirectKey, Key};

/// The largest generation an allocator issues, 2³¹. An index whose id of
/// this generation is freed is retired. [`Id::new`] accepts larger
/// generations, but no allocator considers such an id alive.
pub const MAX_GENERATION: u32 = 1 << 31;

/// The number of indices an allocator can issue: indices are
/// `0..MAX_INDICES`, so [`IdAllocator::index_bound`] fits in a `u32`.
pub const MAX_INDICES: u32 = u32::MAX;

/// A generational id: a slot index and a generation.
///
/// Ids order by index, then generation. [`to_bits`](Id::to_bits) packs one
/// into a `u64` (generation in the high half), for storage or for passing
/// across an FFI boundary.
///
/// # Layout
///
/// An `Id` has size 8 and alignment 8 (at least the alignment of `u64` on
/// every target), and its bytes are exactly those of
/// [`to_bits`](Id::to_bits) in native byte order. FFI
/// code may therefore view a `[Id]` as a `[u64]` of bits without copying.
/// (The reverse is not sound: a `u64` with a zero high half is not an `Id`.)
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C, align(8))]
pub struct Id {
    // Field order follows the byte order, so the high half of the native
    // `u64` holds the generation.
    #[cfg(target_endian = "little")]
    index: u32,
    generation: NonZeroU32,
    #[cfg(target_endian = "big")]
    index: u32,
}

const _: () = assert!(core::mem::size_of::<Id>() == 8 && core::mem::align_of::<Id>() == 8);

impl Ord for Id {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        (self.index, self.generation).cmp(&(other.index, other.generation))
    }
}

impl PartialOrd for Id {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Id {
    /// An id from its parts. An allocator treats it as alive only if it
    /// currently has a live id with this index and generation (see the
    /// [module docs](crate::id_alloc) on the scope of that check).
    #[inline]
    pub const fn new(index: u32, generation: NonZeroU32) -> Self {
        Self { index, generation }
    }

    /// The slot index.
    #[inline]
    pub const fn index(self) -> u32 {
        self.index
    }

    /// The generation, starting at 1 for the first id at an index.
    #[inline]
    pub const fn generation(self) -> NonZeroU32 {
        self.generation
    }

    /// Packs the id as `generation << 32 | index`. Never zero.
    #[inline]
    pub const fn to_bits(self) -> u64 {
        ((self.generation.get() as u64) << 32) | self.index as u64
    }

    /// Unpacks [`to_bits`](Id::to_bits); `None` if the generation half is
    /// zero.
    #[inline]
    pub const fn from_bits(bits: u64) -> Option<Self> {
        match NonZeroU32::new((bits >> 32) as u32) {
            Some(generation) => Some(Self {
                index: bits as u32,
                generation,
            }),
            None => None,
        }
    }
}

impl fmt::Debug for Id {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Id({}v{})", self.index, self.generation)
    }
}

impl Key for Id {
    #[inline]
    fn index(self) -> usize {
        self.index as usize
    }
}

/// A direct map stores only the generation; its `NonZeroU32` niche keeps
/// `Option<(NonZeroU32, V)>` as small as `(u32, V)`.
impl DirectKey for Id {
    type Tag = NonZeroU32;

    #[inline]
    fn tag(self) -> NonZeroU32 {
        self.generation
    }

    #[inline]
    fn from_parts(index: usize, generation: NonZeroU32) -> Self {
        // `index` came from an `Id`, so it fits in `u32`.
        Self::new(index as u32, generation)
    }
}

// Each index has a *version* that encodes its state:
//
// - 0: retired, never issued again;
// - odd `v`: live, holding generation `v / 2 + 1`;
// - even `v > 0`: vacant and in the free list; the next generation is
//   `v / 2 + 1`.
//
// New indices start at version 1 (generation 1). Freeing adds 1, reusing
// adds 1, so generation `g` is live at version `2g - 1`. Freeing the live
// version `u32::MAX` (generation 2^31) would overflow, so the index is
// retired instead.

/// The generation that a live (odd) version holds.
#[inline]
pub(crate) fn generation_of(version: u32) -> NonZeroU32 {
    debug_assert!(version % 2 == 1);
    NonZeroU32::MIN.saturating_add(version >> 1)
}

/// The version at which `generation` is live, or `None` above
/// [`MAX_GENERATION`].
#[inline]
pub(crate) fn live_version(generation: NonZeroU32) -> Option<u32> {
    (generation.get() - 1).checked_mul(2).map(|v| v | 1)
}

/// Hands out generational [`Id`]s and recycles freed ones.
///
/// Invariants:
///
/// - `versions.len() <= MAX_INDICES`;
/// - `free` holds every vacant (even, non-zero version) index exactly once,
///   and nothing else;
/// - `len` is the number of live (odd version) indices.
///
/// Equality compares the full state, including the order of the free list,
/// so equal allocators issue the same ids from then on.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct IdAllocator {
    versions: Vec<u32>,
    free: Vec<u32>,
    len: usize,
}

impl IdAllocator {
    /// An empty allocator. Does not allocate.
    #[inline]
    pub const fn new() -> Self {
        Self {
            versions: Vec::new(),
            free: Vec::new(),
            len: 0,
        }
    }

    /// An empty allocator with room for `capacity` indices.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            versions: Vec::with_capacity(capacity),
            free: Vec::new(),
            len: 0,
        }
    }

    /// Number of live ids.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether no id is live.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// One past the largest index ever issued: every live, vacant or retired
    /// index is below it.
    #[inline]
    pub fn index_bound(&self) -> usize {
        self.versions.len()
    }

    /// Number of freed indices waiting to be reused.
    #[inline]
    pub fn free_len(&self) -> usize {
        self.free.len()
    }

    /// Number of retired indices, which are never issued again.
    #[inline]
    pub fn retired_len(&self) -> usize {
        self.versions.len() - self.len - self.free.len()
    }

    /// Reserves room so that the next `additional` allocations do not
    /// reallocate.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity overflows `isize::MAX` bytes.
    pub fn reserve(&mut self, additional: usize) {
        self.versions
            .reserve(additional.saturating_sub(self.free.len()));
    }

    /// Like [`reserve`](IdAllocator::reserve), but returns an error instead
    /// of panicking or aborting.
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.versions
            .try_reserve(additional.saturating_sub(self.free.len()))
    }

    /// Issues an id: the most recently freed index with its next
    /// generation, or else a new index with generation 1.
    ///
    /// # Panics
    ///
    /// Panics if all [`MAX_INDICES`] indices are live or retired. See
    /// [`try_alloc`](IdAllocator::try_alloc).
    pub fn alloc(&mut self) -> Id {
        match self.try_alloc() {
            Some(id) => id,
            None => panic!("IdAllocator: all {MAX_INDICES} indices are live or retired"),
        }
    }

    /// Like [`alloc`](IdAllocator::alloc), but `None` if all indices are
    /// live or retired.
    pub fn try_alloc(&mut self) -> Option<Id> {
        if let Some(index) = self.free.pop() {
            let version = &mut self.versions[index as usize];
            debug_assert!(*version != 0 && *version % 2 == 0);
            // A vacant version is even, so at most `u32::MAX - 1`.
            *version += 1;
            self.len += 1;
            return Some(Id {
                index,
                generation: generation_of(*version),
            });
        }
        let index = u32::try_from(self.versions.len())
            .ok()
            .filter(|&i| i < MAX_INDICES)?;
        // Push first: if it panics, nothing has changed.
        self.versions.push(1);
        self.len += 1;
        Some(Id {
            index,
            generation: NonZeroU32::MIN,
        })
    }

    /// Whether `id` is live: issued by this allocator and not freed since.
    #[inline]
    pub fn is_alive(&self, id: Id) -> bool {
        live_version(id.generation)
            .is_some_and(|v| self.versions.get(id.index as usize) == Some(&v))
    }

    /// The live id at `index`, if any.
    #[inline]
    pub fn occupant(&self, index: u32) -> Option<Id> {
        let version = *self.versions.get(index as usize)?;
        (version % 2 == 1).then(|| Id {
            index,
            generation: generation_of(version),
        })
    }

    /// Frees `id`. Returns `false`, and changes nothing, if `id` is not
    /// live (already freed, never issued, or another generation).
    pub fn free(&mut self, id: Id) -> bool {
        if !self.is_alive(id) {
            return false;
        }
        self.free_live(id.index);
        true
    }

    /// Frees the live index `index`, retiring it after its last generation.
    fn free_live(&mut self, index: u32) {
        let version = self.versions[index as usize];
        debug_assert!(version % 2 == 1);
        match version.checked_add(1) {
            Some(vacant) => {
                // Push first: if it panics, nothing has changed.
                self.free.push(index);
                self.versions[index as usize] = vacant;
            }
            None => self.versions[index as usize] = 0,
        }
        self.len -= 1;
    }

    /// Frees every live id for which `keep` returns `false`, visiting ids in
    /// index order. If `keep` panics, the ids already visited stay freed or
    /// kept and the allocator stays consistent.
    pub fn retain(&mut self, mut keep: impl FnMut(Id) -> bool) {
        for i in 0..self.versions.len() {
            let version = self.versions[i];
            if version % 2 == 1 {
                // `i < versions.len() <= MAX_INDICES`, so it fits.
                let index = i as u32;
                if !keep(Id {
                    index,
                    generation: generation_of(version),
                }) {
                    self.free_live(index);
                }
            }
        }
    }

    /// Frees every live id. Indices keep their generations, so ids issued
    /// before stay dead.
    pub fn clear(&mut self) {
        self.retain(|_| false);
    }

    /// Shrinks the buffers to fit. Indices are never forgotten, because
    /// reissuing one from generation 1 could revive stale ids.
    pub fn shrink_to_fit(&mut self) {
        self.versions.shrink_to_fit();
        self.free.shrink_to_fit();
    }

    /// The raw state: the per-index versions and the free list, in the
    /// order indices are reused (last entry first). Together with
    /// [`try_from_raw_parts`](IdAllocator::try_from_raw_parts) this
    /// round-trips an allocator exactly, for formats other than serde.
    ///
    /// A version is `0` for a retired index, odd (`2g - 1`) for a live index
    /// of generation `g`, and even and non-zero (`2g`) for a vacant index
    /// whose last generation was `g`.
    #[inline]
    pub fn raw_parts(&self) -> (&[u32], &[u32]) {
        (&self.versions, &self.free)
    }

    /// Rebuilds an allocator from [`raw_parts`](IdAllocator::raw_parts),
    /// validating them: at most [`MAX_INDICES`] versions, and a free list
    /// holding every vacant index exactly once and nothing else. Memory is
    /// linear in the input size.
    pub fn try_from_raw_parts(versions: Vec<u32>, free: Vec<u32>) -> Result<Self, RawPartsError> {
        if versions.len() > MAX_INDICES as usize {
            return Err(RawPartsError::TooManyIndices);
        }
        let mut listed = alloc::vec![false; versions.len()];
        for &index in &free {
            let i = index as usize;
            match versions.get(i) {
                Some(&v) if v != 0 && v % 2 == 0 => {}
                _ => return Err(RawPartsError::NotVacant),
            }
            if core::mem::replace(&mut listed[i], true) {
                return Err(RawPartsError::ListedTwice);
            }
        }
        let vacant = versions.iter().filter(|&&v| v != 0 && v % 2 == 0).count();
        if vacant != free.len() {
            return Err(RawPartsError::MissingVacant);
        }
        let len = versions.iter().filter(|&&v| v % 2 == 1).count();
        Ok(IdAllocator {
            versions,
            free,
            len,
        })
    }

    /// Iterates over the live ids in index order.
    pub fn iter(&self) -> Iter<'_> {
        Iter {
            inner: self.versions.iter().enumerate(),
            remaining: self.len,
        }
    }
}

impl fmt::Debug for IdAllocator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl<'a> IntoIterator for &'a IdAllocator {
    type Item = Id;
    type IntoIter = Iter<'a>;

    fn into_iter(self) -> Iter<'a> {
        self.iter()
    }
}

/// Why [`IdAllocator::try_from_raw_parts`] rejected its input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RawPartsError {
    /// More than [`MAX_INDICES`] versions.
    TooManyIndices,
    /// The free list holds an index that is not vacant (or out of range).
    NotVacant,
    /// The free list holds an index twice.
    ListedTwice,
    /// A vacant index is missing from the free list.
    MissingVacant,
}

impl fmt::Display for RawPartsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            RawPartsError::TooManyIndices => "more than u32::MAX indices",
            RawPartsError::NotVacant => "free list holds an index that is not vacant",
            RawPartsError::ListedTwice => "free list holds an index twice",
            RawPartsError::MissingVacant => "free list misses a vacant index",
        })
    }
}

impl core::error::Error for RawPartsError {}

/// Iterator over the live ids of an [`IdAllocator`], in index order.
#[derive(Clone)]
pub struct Iter<'a> {
    inner: Enumerate<core::slice::Iter<'a, u32>>,
    remaining: usize,
}

impl Iter<'_> {
    #[inline]
    fn live(i: usize, version: u32) -> Option<Id> {
        // `i < MAX_INDICES`, so it fits.
        (version % 2 == 1).then(|| Id {
            index: i as u32,
            generation: generation_of(version),
        })
    }
}

impl Iterator for Iter<'_> {
    type Item = Id;

    fn next(&mut self) -> Option<Id> {
        if self.remaining == 0 {
            return None;
        }
        let id = self.inner.by_ref().find_map(|(i, &v)| Self::live(i, v))?;
        self.remaining -= 1;
        Some(id)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl DoubleEndedIterator for Iter<'_> {
    fn next_back(&mut self) -> Option<Id> {
        if self.remaining == 0 {
            return None;
        }
        let id = self
            .inner
            .by_ref()
            .rev()
            .find_map(|(i, &v)| Self::live(i, v))?;
        self.remaining -= 1;
        Some(id)
    }
}

impl ExactSizeIterator for Iter<'_> {}
impl FusedIterator for Iter<'_> {}

impl fmt::Debug for Iter<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

#[cfg(feature = "serde")]
mod serde_impl {
    use super::*;
    use serde::de::{Deserialize, Deserializer, Error};
    use serde::ser::{Serialize, Serializer};

    /// Serializes as its bits ([`Id::to_bits`]).
    impl Serialize for Id {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            serializer.serialize_u64(self.to_bits())
        }
    }

    /// Deserializes from its bits; rejects a zero generation.
    impl<'de> Deserialize<'de> for Id {
        fn deserialize<De: Deserializer<'de>>(deserializer: De) -> Result<Self, De::Error> {
            let bits = u64::deserialize(deserializer)?;
            Id::from_bits(bits).ok_or_else(|| De::Error::custom("id with generation 0"))
        }
    }

    #[derive(serde::Serialize)]
    struct ReprRef<'a> {
        versions: &'a [u32],
        free: &'a [u32],
    }

    #[derive(serde::Deserialize)]
    struct Repr {
        versions: Vec<u32>,
        free: Vec<u32>,
    }

    /// Serializes the per-index versions and the free list, so a round trip
    /// gives an equal allocator that issues the same ids.
    impl Serialize for IdAllocator {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            ReprRef {
                versions: &self.versions,
                free: &self.free,
            }
            .serialize(serializer)
        }
    }

    /// Deserializes and validates the state: rejects more than
    /// [`MAX_INDICES`] indices, and a free list that is not exactly the
    /// vacant indices, each once. Memory is linear in the input size.
    impl<'de> Deserialize<'de> for IdAllocator {
        fn deserialize<De: Deserializer<'de>>(deserializer: De) -> Result<Self, De::Error> {
            let Repr { versions, free } = Repr::deserialize(deserializer)?;
            IdAllocator::try_from_raw_parts(versions, free).map_err(De::Error::custom)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g(n: u32) -> NonZeroU32 {
        NonZeroU32::new(n).unwrap()
    }

    #[test]
    fn reuses_last_freed_index_with_next_generation() {
        let mut ids = IdAllocator::new();
        let a = ids.alloc();
        let b = ids.alloc();
        let c = ids.alloc();
        assert_eq!((a.index(), b.index(), c.index()), (0, 1, 2));
        assert!(ids.free(a));
        assert!(ids.free(c));
        assert!(!ids.free(c));
        assert_eq!(ids.len(), 1);
        assert_eq!(ids.free_len(), 2);
        assert_eq!(ids.alloc(), Id::new(2, g(2)));
        assert_eq!(ids.alloc(), Id::new(0, g(2)));
        assert_eq!(ids.alloc(), Id::new(3, g(1)));
        assert!(!ids.is_alive(a) && !ids.is_alive(c));
        assert!(ids.is_alive(b));
        assert_eq!(ids.occupant(0), Some(Id::new(0, g(2))));
        assert_eq!(ids.occupant(9), None);
    }

    #[test]
    fn last_generation_retires_the_index() {
        let mut ids = IdAllocator::new();
        let a = ids.alloc();
        // Jump index 0 to its last generation.
        ids.versions[0] = u32::MAX;
        let last = Id::new(0, g(MAX_GENERATION));
        assert!(!ids.is_alive(a));
        assert!(ids.is_alive(last));
        assert!(ids.free(last));
        assert_eq!(ids.retired_len(), 1);
        assert_eq!(ids.free_len(), 0);
        assert_eq!(ids.occupant(0), None);
        assert_eq!(ids.alloc().index(), 1);
        // Generations above the maximum are never alive.
        assert!(!ids.is_alive(Id::new(1, g(MAX_GENERATION + 1))));
    }

    #[test]
    fn retain_clear_and_iter() {
        let mut ids = IdAllocator::new();
        let all: Vec<Id> = (0..6).map(|_| ids.alloc()).collect();
        ids.retain(|id| id.index() % 2 == 0);
        assert_eq!(ids.iter().collect::<Vec<_>>(), [all[0], all[2], all[4]]);
        assert_eq!(
            ids.iter().rev().collect::<Vec<_>>(),
            [all[4], all[2], all[0]]
        );
        let mut it = ids.iter();
        assert_eq!(it.len(), 3);
        assert_eq!(
            (it.next(), it.next_back(), it.len()),
            (Some(all[0]), Some(all[4]), 1)
        );
        ids.clear();
        assert!(ids.is_empty());
        assert_eq!(ids.iter().next(), None);
        assert!(all.iter().all(|&id| !ids.is_alive(id)));
        assert_eq!(ids.free_len(), 6);
    }

    #[test]
    fn bits_round_trip() {
        let id = Id::new(7, g(3));
        assert_eq!(id.to_bits(), (3 << 32) | 7);
        assert_eq!(Id::from_bits(id.to_bits()), Some(id));
        assert_eq!(Id::from_bits(7), None);
        assert!(Id::new(1, g(9)) < Id::new(2, g(1)));
        assert_eq!(alloc::format!("{id:?}"), "Id(7v3)");
    }

    #[test]
    fn ids_share_a_slot_across_generations() {
        use crate::sparse_set::PagedSparseSet;
        let mut ids = IdAllocator::new();
        let mut set = PagedSparseSet::<Id>::new();
        let a = ids.alloc();
        assert!(set.insert(a));
        ids.free(a);
        let b = ids.alloc();
        assert!(!set.contains(b));
        assert!(!set.insert(b));
        assert_eq!(set.replace(b), Some(a));
        assert!(set.contains(b));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_round_trip_and_validation() {
        let mut ids = IdAllocator::new();
        let v: Vec<Id> = (0..4).map(|_| ids.alloc()).collect();
        ids.free(v[1]);
        ids.free(v[3]);
        let json = serde_json::to_string(&ids).unwrap();
        assert_eq!(json, r#"{"versions":[1,2,1,2],"free":[1,3]}"#);
        let back: IdAllocator = serde_json::from_str(&json).unwrap();
        assert_eq!(back, ids);
        assert_eq!(back.len(), 2);

        for bad in [
            r#"{"versions":[1,2],"free":[]}"#,
            r#"{"versions":[1,2],"free":[0]}"#,
            r#"{"versions":[1,2],"free":[1,1]}"#,
            r#"{"versions":[0],"free":[0]}"#,
            r#"{"versions":[2],"free":[5]}"#,
        ] {
            assert!(serde_json::from_str::<IdAllocator>(bad).is_err(), "{bad}");
        }

        assert_eq!(
            serde_json::to_string(&v[1]).unwrap(),
            alloc::format!("{}", (1u64 << 32) | 1)
        );
        assert!(serde_json::from_str::<Id>("5").is_err());
    }

    #[test]
    fn raw_parts_round_trip_and_validation() {
        let mut ids = IdAllocator::new();
        let a = ids.alloc();
        let b = ids.alloc();
        ids.alloc();
        ids.free(b);
        ids.free(a);
        let (versions, free) = ids.raw_parts();
        assert_eq!((versions, free), (&[2, 2, 1][..], &[1, 0][..]));
        let back = IdAllocator::try_from_raw_parts(versions.to_vec(), free.to_vec()).unwrap();
        assert_eq!(back, ids);
        assert_eq!(back.len(), 1);
        let bad = |v: &[u32], f: &[u32]| IdAllocator::try_from_raw_parts(v.to_vec(), f.to_vec());
        assert_eq!(bad(&[1], &[0]), Err(RawPartsError::NotVacant));
        assert_eq!(bad(&[0], &[0]), Err(RawPartsError::NotVacant));
        assert_eq!(bad(&[2], &[1]), Err(RawPartsError::NotVacant));
        assert_eq!(bad(&[2], &[0, 0]), Err(RawPartsError::ListedTwice));
        assert_eq!(bad(&[2, 4], &[1]), Err(RawPartsError::MissingVacant));
        assert_eq!(
            bad(&[0, 3], &[]).map(|a| (a.len(), a.retired_len())),
            Ok((1, 1))
        );
    }

    #[test]
    fn ids_order_by_index_then_generation() {
        assert!(Id::new(0, g(2)) < Id::new(1, g(1)));
        assert!(Id::new(1, g(1)) < Id::new(1, g(2)));
        assert_eq!(
            Id::new(3, g(4)).cmp(&Id::new(3, g(4))),
            core::cmp::Ordering::Equal
        );
    }
}
