//! [`Hive`]: an unordered container with stable addresses, reusable slots
//! and generational handles.
//!
//! Elements live in slots, in pages of `PAGE` slots. Inserting returns an
//! [`Id`]: the slot index plus a generation. Removing an element frees its
//! slot for reuse; the next element there gets the next generation, so a
//! stale id of a removed element is never mistaken for a live one.
//!
//! - **Stable addresses.** An element does not move while it is in the hive:
//!   inserts, removals of other elements, [`reserve`](Hive::reserve) and
//!   [`shrink_to_fit`](Hive::shrink_to_fit) leave its address unchanged.
//!   Only [`sort`](Hive::sort) and its variants move elements between
//!   slots; ids name slots, so after a sort an id names whatever element
//!   moved into its slot.
//!   Pages are allocated once with room for `PAGE` slots and never
//!   reallocated.
//! - **Slot reuse.** Inserts fill one page with room at a time: a page that
//!   was full and gains a free slot is used next, and within a page the
//!   last freed slot is reused first. A page that has room is used before a
//!   new page is created, so memory follows the peak number of elements, not
//!   the number of inserts. Which page with room goes first is otherwise
//!   unspecified.
//! - **Memory return.** [`shrink_to_fit`](Hive::shrink_to_fit) frees every
//!   empty page. The page keeps a small record of its highest generation,
//!   so ids issued before stay dead when the page is used again.
//!
//! [`Id`] implements [`Key`](crate::key::Key), so ids index the other
//! collections directly: a [`PagedSparseMap<Id, V>`](crate::sparse_map) or
//! [`PagedDirectMap<Id, V>`](crate::direct_map) makes a side table for hive
//! elements. As there, all generations of a slot share one key slot.
//!
//! Comparable in purpose to C++26 `std::hive`. Iteration is in slot order
//! (page by page), not insertion order.
//!
//! | Operation | Cost |
//! |---|---|
//! | [`insert`](Hive::insert), [`remove`](Hive::remove) | amortized O(1): a page allocation every `PAGE` new slots, rare directory growth |
//! | [`get`](Hive::get), [`contains`](Hive::contains), [`occupant`](Hive::occupant) | O(1) |
//! | [`iter`](Hive::iter) and friends | O(len) on full pages; skips empty pages, scans the free slots of the others |
//! | [`retain`](Hive::retain), [`clear`](Hive::clear), [`drain`](Hive::drain), [`extract_if`](Hive::extract_if) | O(pages + len) plus the scan, as for iteration |
//! | [`reserve`](Hive::reserve), [`shrink_to_fit`](Hive::shrink_to_fit) | O(pages) bookkeeping; `shrink_to_fit` also scans the slots of empty pages |
//!
//! # Limits and stale ids
//!
//! Slot indices are `u32`, below [`MAX_INDICES`]. Each slot issues at most
//! [`MAX_GENERATION`](crate::id_alloc::MAX_GENERATION) ids; after the last
//! one is removed the slot is *retired* and never reused, so generations
//! never wrap. A page with a retired slot is never freed.
//!
//! As with [`IdAllocator`](crate::id_alloc::IdAllocator), the stale-id
//! guarantee holds within one history of a hive. Ids carry no hive
//! identity, so another hive (or a clone) accepts them, and restoring an
//! older state (an older clone, or an older serialized snapshot) revives the
//! ids removed since.
//!
//! The module exists only on 32- and 64-bit targets, like
//! [`id_alloc`](crate::id_alloc).
//!
//! ```
//! use paged_collections::hive::Hive;
//!
//! let mut hive: Hive<&str, 4> = Hive::new();
//! let a = hive.insert("a");
//! let b = hive.insert("b");
//! let addr: *const &str = &hive[b];
//!
//! assert_eq!(hive.remove(a), Some("a"));
//! assert_eq!(hive.get(a), None);
//!
//! // The slot is reused with a new generation; the stale id stays dead.
//! let c = hive.insert("c");
//! assert_eq!(c.index(), a.index());
//! assert_ne!(c, a);
//! assert_eq!(hive.get(a), None);
//!
//! // Other elements never moved.
//! hive.extend(["d", "e", "f", "g"]);
//! assert!(core::ptr::eq(addr, &hive[b]));
//! assert_eq!(hive.values().copied().collect::<Vec<_>>(), ["c", "b", "d", "e", "f", "g"]);
//! ```
//!
//! With the `rayon` feature, borrowed parallel iterators are unindexed and
//! visit occupied slots page by page. Collecting into a `Vec` preserves slot
//! order. The order of `for_each` is unspecified.
//!
//! ```rust
//! # #[cfg(feature = "rayon")] {
//! use paged_collections::hive::Hive;
//! use rayon::prelude::*;
//!
//! let mut hive: Hive<_, 2> = Hive::new();
//! let a = hive.insert(10);
//! let b = hive.insert(20);
//! hive.par_iter_mut().for_each(|(_, value)| *value += 1);
//! assert_eq!(hive.par_iter().map(|(id, &v)| (id, v)).collect::<Vec<_>>(), [(a, 11), (b, 21)]);
//! # }
//! ```

use alloc::collections::TryReserveError;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::iter::{Enumerate, FusedIterator};
use core::ops::{Index, IndexMut};

use crate::id_alloc::{Id, MAX_INDICES, generation_of, live_version};
use crate::page::{DEFAULT_PAGE, split};
use crate::paged_vec::capacity_overflow;

/// End of a page's free list.
const NONE: u32 = u32::MAX;

/// One slot of a [`RawPage`], with versions as in
/// [`IdAllocator::raw_parts`](crate::id_alloc::IdAllocator::raw_parts).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RawSlot<T> {
    /// An occupied slot: an odd version (`2g - 1`) and its element.
    Occupied {
        /// The odd version, for generation `g = version / 2 + 1`.
        version: u32,
        /// The element in the slot.
        value: T,
    },
    /// A vacant slot: an even non-zero version (`2g`), or `0` if retired.
    Vacant {
        /// The even version, or `0` if the slot can never be reused.
        version: u32,
    },
}

/// A page's raw state: its even generation floor and its slots in offset order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawPage<T> {
    /// The even generation floor; new slots start at `floor + 1`.
    pub floor: u32,
    /// The slots in offset order, including vacant and retired slots.
    pub slots: Vec<RawSlot<T>>,
}

/// Why [`Hive::try_from_raw_pages`] rejected its input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RawPagesError {
    /// More pages than the slot indices below [`MAX_INDICES`] can cover.
    TooManyPages,
    /// A page exceeds `PAGE`, or the last page exceeds its usable slot indices.
    PageTooLong,
    /// A page's generation floor is odd.
    OddFloor,
    /// An occupied slot has an even version, or a vacant slot an odd version.
    VersionParity,
}

impl fmt::Display for RawPagesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            RawPagesError::TooManyPages => "more pages than slot indices",
            RawPagesError::PageTooLong => "page holds too many slots",
            RawPagesError::OddFloor => "odd page floor",
            RawPagesError::VersionParity => "slot version parity mismatch",
        })
    }
}

impl core::error::Error for RawPagesError {}

/// One slot. Versions follow [`IdAllocator`](crate::id_alloc::IdAllocator):
/// odd while occupied (generation `version / 2 + 1`), even and non-zero
/// while vacant and in the page's free list, 0 once retired.
enum Slot<T> {
    Occupied {
        version: u32,
        value: T,
    },
    /// `next` is the next free offset in the page, or [`NONE`]; unused when
    /// retired.
    Vacant {
        version: u32,
        next: u32,
    },
}

impl<T> Slot<T> {
    #[inline]
    fn version(&self) -> u32 {
        match *self {
            Slot::Occupied { version, .. } | Slot::Vacant { version, .. } => version,
        }
    }
}

/// One page of slots.
///
/// Invariants, for page number `p` and `limit = limit(p)`:
///
/// - `slots.capacity()` is 0 (the page is *unallocated*: `slots` is empty)
///   or at least `PAGE` (*allocated*); `slots.len() <= limit <= PAGE`, so an
///   allocated page never reallocates;
/// - `free_head` starts a list, linked through `Vacant::next`, holding each
///   vacant non-retired slot of the page exactly once;
/// - `len` counts the occupied slots and `retired` the retired ones;
/// - `floor` is even and at least the version every slot at an offset
///   `>= slots.len()` ever had; a slot created there starts at `floor + 1`.
struct Page<T> {
    slots: Vec<Slot<T>>,
    free_head: u32,
    len: u32,
    retired: u32,
    floor: u32,
}

impl<T> Page<T> {
    #[inline]
    const fn unallocated(floor: u32) -> Self {
        Self {
            slots: Vec::new(),
            free_head: NONE,
            len: 0,
            retired: 0,
            floor,
        }
    }

    #[inline]
    fn allocated(&self) -> bool {
        self.slots.capacity() != 0
    }

    /// Whether an insert can use this page.
    #[inline]
    fn has_room(&self, limit: usize) -> bool {
        self.free_head != NONE || self.slots.len() < limit
    }

    /// Number of inserts the page takes, once allocated.
    #[inline]
    fn room(&self, limit: usize) -> usize {
        limit - self.len as usize - self.retired as usize
    }
}

/// An unordered container with stable addresses and generational [`Id`]s.
///
/// `PAGE` is the number of slots per page; it must be non-zero and fit in a
/// `u32`. See the [module docs](crate::hive).
///
/// Invariants, besides those of each page:
///
/// - page `p` exists only if `p * PAGE < MAX_INDICES`;
/// - `open` holds each page that has room (`Page::has_room`) exactly once,
///   and nothing else; inserts use its last entry;
/// - `len` is the total number of occupied slots.
pub struct Hive<T, const PAGE: usize = DEFAULT_PAGE> {
    pages: Vec<Page<T>>,
    open: Vec<usize>,
    len: usize,
}

impl<T, const PAGE: usize> Hive<T, PAGE> {
    /// Number of slots per page.
    pub const PAGE: usize = PAGE;

    /// An empty hive. Does not allocate.
    #[inline]
    pub const fn new() -> Self {
        const {
            assert!(PAGE > 0, "page size must be non-zero");
            assert!(PAGE <= u32::MAX as usize, "page size must fit in a u32");
        };
        Self {
            pages: Vec::new(),
            open: Vec::new(),
            len: 0,
        }
    }

    /// An empty hive with pages allocated for `capacity` elements.
    pub fn with_capacity(capacity: usize) -> Self {
        let mut hive = Self::new();
        hive.reserve(capacity);
        hive
    }

    /// Number of usable slots in page `p`: `PAGE`, except for a last page
    /// cut short by [`MAX_INDICES`].
    #[inline]
    fn limit(p: usize) -> usize {
        // Pages exist only for `p * PAGE < MAX_INDICES`.
        (MAX_INDICES as usize - p * PAGE).min(PAGE)
    }

    /// Number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether there are no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of usable slots in allocated pages, occupied or not (a last
    /// page cut short by [`MAX_INDICES`] counts only its usable slots).
    /// Takes O(pages) time: it scans the page directory.
    pub fn capacity(&self) -> usize {
        self.pages
            .iter()
            .enumerate()
            .filter(|(_, page)| page.allocated())
            .map(|(p, _)| Self::limit(p))
            .sum()
    }

    /// Number of allocated pages. Takes O(pages) time: it scans the page
    /// directory.
    pub fn allocated_pages(&self) -> usize {
        self.pages.iter().filter(|page| page.allocated()).count()
    }

    /// One past the largest slot index the hive can have used: every id it
    /// issued has a smaller index. Use it to size a side table.
    #[inline]
    pub fn index_bound(&self) -> usize {
        self.pages
            .len()
            .saturating_mul(PAGE)
            .min(MAX_INDICES as usize)
    }

    /// The raw state: every page in order, including unallocated pages,
    /// with elements borrowed. Together with
    /// [`try_from_raw_pages`](Hive::try_from_raw_pages) this round-trips a
    /// hive for formats other than serde.
    ///
    /// A round trip keeps every id, generation, value, floor and retired
    /// slot. It need not keep allocations or free-list and page reuse
    /// order, so the next insert may use a different slot.
    pub fn raw_pages(&self) -> impl ExactSizeIterator<Item = RawPage<&T>> + '_ {
        self.pages.iter().map(|page| RawPage {
            floor: page.floor,
            slots: page
                .slots
                .iter()
                .map(|slot| match slot {
                    Slot::Occupied { version, value } => RawSlot::Occupied {
                        version: *version,
                        value,
                    },
                    Slot::Vacant { version, .. } => RawSlot::Vacant { version: *version },
                })
                .collect(),
        })
    }

    /// Rebuilds a hive from [`raw_pages`](Hive::raw_pages), validating
    /// them: pages must fit below [`MAX_INDICES`], each page must fit its
    /// usable slot limit (at most `PAGE`), floors must be even, occupied
    /// versions odd, and vacant versions even (including `0` for retired).
    ///
    /// Free lists are rebuilt lowest offset first. Allocated pages with
    /// room are reused before unallocated pages, lowest page first within
    /// each group. Every non-empty page is allocated with room for `PAGE`
    /// slots; empty pages remain unallocated. Memory can therefore reach
    /// `PAGE` slots per page read.
    pub fn try_from_raw_pages(
        pages: impl IntoIterator<Item = RawPage<T>>,
    ) -> Result<Self, RawPagesError> {
        let mut hive = Self::new();
        let pages: Vec<_> = pages.into_iter().collect();
        if pages.len() > (MAX_INDICES as usize).div_ceil(PAGE) {
            return Err(RawPagesError::TooManyPages);
        }
        // Validate everything before allocating pages, so that bad input is
        // rejected without allocating `PAGE` slots per page read.
        for (p, raw) in pages.iter().enumerate() {
            if raw.slots.len() > Self::limit(p) {
                return Err(RawPagesError::PageTooLong);
            }
            if raw.floor % 2 == 1 {
                return Err(RawPagesError::OddFloor);
            }
            let parity_ok = raw.slots.iter().all(|slot| match slot {
                RawSlot::Occupied { version, .. } => version % 2 == 1,
                RawSlot::Vacant { version } => version % 2 == 0,
            });
            if !parity_ok {
                return Err(RawPagesError::VersionParity);
            }
        }
        hive.pages.reserve_exact(pages.len());
        for raw in pages {
            let mut page = Page::unallocated(raw.floor);
            if !raw.slots.is_empty() {
                page.slots.reserve_exact(PAGE);
            }
            for slot in raw.slots {
                page.slots.push(match slot {
                    RawSlot::Occupied { version, value } if version % 2 == 1 => {
                        page.len += 1;
                        Slot::Occupied { version, value }
                    }
                    RawSlot::Vacant { version } if version % 2 == 0 => {
                        if version == 0 {
                            page.retired += 1;
                        }
                        Slot::Vacant {
                            version,
                            next: NONE,
                        }
                    }
                    _ => unreachable!("slot parity was validated"),
                });
            }
            for (offset, slot) in page.slots.iter_mut().enumerate().rev() {
                if let Slot::Vacant { version, next } = slot {
                    if *version != 0 {
                        *next = page.free_head;
                        // `offset < PAGE <= u32::MAX`.
                        page.free_head = offset as u32;
                    }
                }
            }
            hive.len += page.len as usize;
            hive.pages.push(page);
        }
        hive.rebuild_open();
        Ok(hive)
    }

    /// Appends an unallocated page to the directory, with room reserved in
    /// `open` to list it. `None` if no more slot indices exist.
    fn push_page(&mut self) -> Option<usize> {
        let p = self.pages.len();
        if p.checked_mul(PAGE)
            .is_none_or(|base| base >= MAX_INDICES as usize)
        {
            return None;
        }
        self.open.reserve(1);
        self.pages.push(Page::unallocated(0));
        Some(p)
    }

    /// Allocates pages so that the next `additional` inserts allocate
    /// nothing. Allocates the free pages first, then appends new ones,
    /// which inserts use after the pages that already have room.
    ///
    /// # Panics
    ///
    /// Panics if `additional` more elements exceed the slot indices left, or
    /// an allocation size overflows `isize::MAX` bytes.
    pub fn reserve(&mut self, additional: usize) {
        if let Err(e) = self.reserve_with(additional, || Ok(Vec::with_capacity(PAGE))) {
            panic!("Hive::reserve: {e}");
        }
    }

    /// Like [`reserve`](Hive::reserve), but returns an error instead of
    /// panicking or aborting. On error, pages allocated so far are kept and
    /// usable.
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.reserve_with(additional, || {
            let mut slots = Vec::new();
            slots.try_reserve_exact(PAGE)?;
            Ok(slots)
        })
    }

    fn reserve_with(
        &mut self,
        additional: usize,
        mut alloc_page: impl FnMut() -> Result<Vec<Slot<T>>, TryReserveError>,
    ) -> Result<(), TryReserveError> {
        let mut room: usize = (0..self.pages.len())
            .filter(|&p| self.pages[p].allocated())
            .map(|p| self.pages[p].room(Self::limit(p)))
            .sum();
        // Fail before allocating if the slot indices cannot cover it.
        let free_pages: usize = (0..self.pages.len())
            .filter(|&p| !self.pages[p].allocated())
            .map(Self::limit)
            .sum();
        let new_slots =
            (MAX_INDICES as usize).saturating_sub(self.pages.len().saturating_mul(PAGE));
        if additional > room.saturating_add(free_pages).saturating_add(new_slots) {
            return Err(capacity_overflow());
        }
        // Free pages are already listed in `open`.
        for p in 0..self.pages.len() {
            if room >= additional {
                return Ok(());
            }
            if !self.pages[p].allocated() {
                self.pages[p].slots = alloc_page()?;
                room += Self::limit(p);
            }
        }
        if room >= additional {
            return Ok(());
        }
        // `room < additional <= room + free_pages + new_slots` and every free
        // page is allocated now, so the new slots suffice. Only the last
        // possible page is shorter than `PAGE`, so this many pages do.
        let count = (additional - room).div_ceil(PAGE);
        // Reserve all metadata first, so the pushes below cannot allocate.
        self.pages.try_reserve(count)?;
        self.open.try_reserve(count)?;
        let mut result = Ok(());
        let mut pushed = 0;
        while pushed < count {
            // Allocate before touching the directory: if this fails or
            // panics, every page pushed so far is allocated and listed.
            let slots = match alloc_page() {
                Ok(slots) => slots,
                Err(e) => {
                    result = Err(e);
                    break;
                }
            };
            let p = self.pages.len();
            debug_assert!(p * PAGE < MAX_INDICES as usize);
            self.pages.push(Page {
                slots,
                ..Page::unallocated(0)
            });
            self.open.push(p);
            pushed += 1;
        }
        // Move the new pages below the listed ones, lowest on top, so inserts
        // fill the pages that already had room first, then the new pages in
        // order.
        let n = self.open.len();
        self.open[n - pushed..].reverse();
        self.open.rotate_right(pushed);
        result
    }

    /// Frees every empty page, except pages with a retired slot. Elements
    /// never move. Ids issued before stay dead: a freed page remembers its
    /// highest generation and starts there when used again.
    pub fn shrink_to_fit(&mut self) {
        for page in &mut self.pages {
            if page.allocated() && page.len == 0 && page.retired == 0 {
                // All slots are vacant, so all versions are even.
                let floor = page
                    .slots
                    .iter()
                    .map(Slot::version)
                    .fold(page.floor, u32::max);
                *page = Page::unallocated(floor);
            }
        }
        // Trailing pages that never issued an id can go entirely.
        while self
            .pages
            .last()
            .is_some_and(|page| !page.allocated() && page.floor == 0)
        {
            self.pages.pop();
        }
        self.pages.shrink_to_fit();
        self.rebuild_open();
        self.open.shrink_to_fit();
    }

    /// Lists every page with room in `open`: unallocated pages at the
    /// bottom, then allocated ones, each group lowest page on top.
    fn rebuild_open(&mut self) {
        self.open.clear();
        for allocated in [false, true] {
            for p in (0..self.pages.len()).rev() {
                let page = &self.pages[p];
                if page.allocated() == allocated && page.has_room(Self::limit(p)) {
                    self.open.push(p);
                }
            }
        }
    }

    /// Inserts `value` and returns its id.
    ///
    /// # Panics
    ///
    /// Panics if every slot index is occupied or retired. See
    /// [`try_insert`](Hive::try_insert).
    #[inline]
    pub fn insert(&mut self, value: T) -> Id {
        self.insert_with(|_| value)
    }

    /// Like [`insert`](Hive::insert), but returns `value` back if every slot
    /// index is occupied or retired. Still panics or aborts if a page cannot
    /// be allocated; [`try_reserve`](Hive::try_reserve) first to avoid that.
    #[inline]
    pub fn try_insert(&mut self, value: T) -> Result<Id, T> {
        match self.next_page() {
            Some(p) => Ok(self.insert_in(p, |_| value)),
            None => Err(value),
        }
    }

    /// Inserts the value `f` makes from its id, and returns the id. If `f`
    /// panics, the hive is unchanged (apart from capacity).
    ///
    /// # Panics
    ///
    /// Panics if every slot index is occupied or retired. See
    /// [`try_insert_with`](Hive::try_insert_with).
    pub fn insert_with(&mut self, f: impl FnOnce(Id) -> T) -> Id {
        match self.next_page() {
            Some(p) => self.insert_in(p, f),
            None => panic!("Hive: all {MAX_INDICES} slot indices are occupied or retired"),
        }
    }

    /// Like [`insert_with`](Hive::insert_with), but returns `None`, without
    /// calling `f`, if every slot index is occupied or retired.
    pub fn try_insert_with(&mut self, f: impl FnOnce(Id) -> T) -> Option<Id> {
        let p = self.next_page()?;
        Some(self.insert_in(p, f))
    }

    /// Whether no insert is possible: every slot index is occupied or
    /// retired. Only a hive holding [`MAX_INDICES`] slots can be exhausted.
    pub fn is_exhausted(&self) -> bool {
        self.open.is_empty()
            && self
                .pages
                .len()
                .checked_mul(PAGE)
                .is_none_or(|slots| slots >= MAX_INDICES as usize)
    }

    /// The page the next insert uses: the top of `open`, or a new page,
    /// which it lists there. `None` if no slot index is left.
    fn next_page(&mut self) -> Option<usize> {
        if let Some(&p) = self.open.last() {
            return Some(p);
        }
        let p = self.push_page()?;
        // `push_page` reserved room for this entry.
        self.open.push(p);
        Some(p)
    }

    /// Inserts into page `p`, which must be the top of `open`.
    fn insert_in(&mut self, p: usize, f: impl FnOnce(Id) -> T) -> Id {
        let limit = Self::limit(p);
        let page = &mut self.pages[p];
        if !page.allocated() {
            page.slots = Vec::with_capacity(PAGE);
        }
        let (offset, version) = match page.free_head {
            NONE => (page.slots.len(), page.floor + 1),
            head => (head as usize, page.slots[head as usize].version() + 1),
        };
        debug_assert!(offset < limit && version % 2 == 1);
        // `p * PAGE + offset < MAX_INDICES`, so it fits.
        let id = Id::new((p * PAGE + offset) as u32, generation_of(version));
        // User code: nothing has changed yet if it panics.
        let value = f(id);
        let occupied = Slot::Occupied { version, value };
        if offset == page.slots.len() {
            debug_assert!(page.slots.len() < page.slots.capacity());
            page.slots.push(occupied);
        } else {
            match core::mem::replace(&mut page.slots[offset], occupied) {
                Slot::Vacant { next, .. } => page.free_head = next,
                Slot::Occupied { .. } => unreachable!(),
            }
        }
        page.len += 1;
        self.len += 1;
        if !page.has_room(limit) {
            debug_assert_eq!(self.open.last(), Some(&p));
            self.open.pop();
        }
        id
    }

    /// The `(page, offset)` of the slot `id` occupies, if it is live.
    #[inline]
    fn locate(&self, id: Id) -> Option<(usize, usize)> {
        let version = live_version(id.generation())?;
        let (p, offset) = split::<PAGE>(id.index() as usize);
        match self.pages.get(p)?.slots.get(offset)? {
            Slot::Occupied { version: v, .. } if *v == version => Some((p, offset)),
            _ => None,
        }
    }

    /// Whether `id` is live: returned by an insert and not removed since.
    #[inline]
    pub fn contains(&self, id: Id) -> bool {
        self.locate(id).is_some()
    }

    /// The element of `id`, if it is live.
    #[inline]
    pub fn get(&self, id: Id) -> Option<&T> {
        let (p, offset) = self.locate(id)?;
        match &self.pages[p].slots[offset] {
            Slot::Occupied { value, .. } => Some(value),
            Slot::Vacant { .. } => None,
        }
    }

    /// The element of `id`, mutably, if it is live.
    #[inline]
    pub fn get_mut(&mut self, id: Id) -> Option<&mut T> {
        let (p, offset) = self.locate(id)?;
        match &mut self.pages[p].slots[offset] {
            Slot::Occupied { value, .. } => Some(value),
            Slot::Vacant { .. } => None,
        }
    }

    /// The elements of several ids at once, mutably: `None` for each id
    /// that is not live. O(N log N) for N ids.
    ///
    /// ```
    /// use paged_collections::hive::Hive;
    ///
    /// let mut hive: Hive<u32> = Hive::new();
    /// let (a, b) = (hive.insert(1), hive.insert(2));
    /// if let [Some(x), Some(y)] = hive.get_disjoint_mut([a, b]) {
    ///     core::mem::swap(x, y);
    /// }
    /// assert_eq!((hive[a], hive[b]), (2, 1));
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if a live id is given twice.
    pub fn get_disjoint_mut<const N: usize>(&mut self, ids: [Id; N]) -> [Option<&mut T>; N] {
        // The live ids as (page, offset, position in `ids`), in slot order.
        let mut order = [(0, 0, 0); N];
        let mut n = 0;
        for (i, &id) in ids.iter().enumerate() {
            if let Some((p, offset)) = self.locate(id) {
                order[n] = (p, offset, i);
                n += 1;
            }
        }
        let order = &mut order[..n];
        order.sort_unstable();
        for pair in order.windows(2) {
            assert!(
                (pair[0].0, pair[0].1) != (pair[1].0, pair[1].1),
                "Hive::get_disjoint_mut: {:?} given twice",
                ids[pair[1].2]
            );
        }
        // Walk the slots in order, splitting each one off the front of what
        // is left: the borrows are disjoint by construction.
        let mut out = [const { None }; N];
        let mut pages: &mut [Page<T>] = &mut self.pages;
        let mut pages_base = 0;
        let mut k = 0;
        while k < n {
            let p = order[k].0;
            let rest = core::mem::take(&mut pages);
            let Some((page, rest)) = rest[p - pages_base..].split_first_mut() else {
                unreachable!("located ids name existing pages")
            };
            pages = rest;
            pages_base = p + 1;
            let mut slots: &mut [Slot<T>] = &mut page.slots;
            let mut slots_base = 0;
            while k < n && order[k].0 == p {
                let (_, offset, i) = order[k];
                let rest = core::mem::take(&mut slots);
                let Some((slot, rest)) = rest[offset - slots_base..].split_first_mut() else {
                    unreachable!("located ids name existing slots")
                };
                slots = rest;
                slots_base = offset + 1;
                if let Slot::Occupied { value, .. } = slot {
                    out[i] = Some(value);
                }
                k += 1;
            }
        }
        out
    }

    /// The live id at slot `index`, if the slot is occupied.
    #[inline]
    pub fn occupant(&self, index: u32) -> Option<Id> {
        let (p, offset) = split::<PAGE>(index as usize);
        match self.pages.get(p)?.slots.get(offset)? {
            Slot::Occupied { version, .. } => Some(Id::new(index, generation_of(*version))),
            Slot::Vacant { .. } => None,
        }
    }

    /// The id of the element at `value`'s address, if `value` is an element
    /// of this hive: the inverse of [`get`](Self::get).
    ///
    /// Takes O(pages) time: it compares the address with each allocated
    /// page's slot range, then checks that the slot found is occupied and
    /// holds exactly that element. A reference to anything else, including
    /// a field inside an element or an element of another hive, gives
    /// `None`. Uses address comparison only; no pointer is dereferenced.
    ///
    /// ```
    /// use paged_collections::hive::Hive;
    ///
    /// let mut hive: Hive<String> = Hive::new();
    /// let id = hive.insert("x".to_owned());
    /// let element: &String = hive.values().next().unwrap();
    /// assert_eq!(hive.id_of(element), Some(id));
    /// assert_eq!(hive.id_of(&"x".to_owned()), None);
    /// ```
    pub fn id_of(&self, value: &T) -> Option<Id> {
        let address = core::ptr::from_ref(value).addr();
        let size = core::mem::size_of::<Slot<T>>();
        for (p, page) in self.pages.iter().enumerate() {
            let start = page.slots.as_ptr().addr();
            let Some(offset) = address.checked_sub(start).map(|bytes| bytes / size) else {
                continue;
            };
            let Some(slot) = page.slots.get(offset) else {
                continue;
            };
            return match slot {
                Slot::Occupied { version, value: v } if core::ptr::eq(v, value) => {
                    Some(Id::new((p * PAGE + offset) as u32, generation_of(*version)))
                }
                _ => None,
            };
        }
        None
    }

    /// The live ids, ordered by their elements with `compare` (a stable
    /// sort: equal elements keep slot order). Moves no element; see
    /// [`sort_by`](Self::sort_by) to reorder the elements themselves.
    pub fn sorted_ids_by(&self, mut compare: impl FnMut(&T, &T) -> Ordering) -> Vec<Id> {
        let mut pairs: Vec<(Id, &T)> = self.iter().collect();
        pairs.sort_by(|a, b| compare(a.1, b.1));
        pairs.into_iter().map(|(id, _)| id).collect()
    }

    /// The live ids, ordered by their elements' keys (a stable sort).
    pub fn sorted_ids_by_key<K: Ord>(&self, mut key: impl FnMut(&T) -> K) -> Vec<Id> {
        self.sorted_ids_by(|a, b| key(a).cmp(&key(b)))
    }

    /// The live ids, ordered by their elements (a stable sort).
    ///
    /// ```
    /// use paged_collections::hive::Hive;
    ///
    /// let mut hive: Hive<u32> = Hive::new();
    /// let (a, b, c) = (hive.insert(3), hive.insert(1), hive.insert(2));
    /// assert_eq!(hive.sorted_ids(), [b, c, a]);
    /// ```
    pub fn sorted_ids(&self) -> Vec<Id>
    where
        T: Ord,
    {
        self.sorted_ids_by(T::cmp)
    }

    /// Sorts the elements in place with `compare`, so iteration (slot
    /// order) yields them sorted. A stable sort; O(n log n) comparisons,
    /// O(n) swaps and O(n) extra memory.
    ///
    /// **Ids name slots, not elements:** after sorting, each live id names
    /// whatever element moved into its slot. No id becomes stale and no
    /// generation changes, so the hive cannot detect ids held across a sort.
    /// Use [`sorted_ids_by`](Self::sorted_ids_by) to keep ids attached to
    /// their elements. Elements move, so their addresses change too (the
    /// set of occupied slots does not). If `compare` panics, no element has
    /// moved yet.
    ///
    /// ```
    /// use paged_collections::hive::Hive;
    ///
    /// let mut hive: Hive<u32> = [3, 1, 2].into_iter().collect();
    /// let first = hive.ids().next().unwrap();
    /// hive.sort();
    /// assert_eq!(hive.values().copied().collect::<Vec<_>>(), [1, 2, 3]);
    /// assert_eq!(hive[first], 1); // the id now names the smallest element
    /// ```
    pub fn sort_by(&mut self, mut compare: impl FnMut(&T, &T) -> Ordering) {
        self.permute_by(|values, order| order.sort_by(|&a, &b| compare(values[a], values[b])));
    }

    /// Sorts the elements in place with `compare`, without preserving the
    /// order of equal elements. Otherwise as [`sort_by`](Self::sort_by)
    /// (same id and address semantics; O(n) extra memory).
    pub fn sort_unstable_by(&mut self, mut compare: impl FnMut(&T, &T) -> Ordering) {
        self.permute_by(|values, order| {
            order.sort_unstable_by(|&a, &b| compare(values[a], values[b]));
        });
    }

    /// Sorts the elements in place by key, without preserving the order of
    /// equal elements; see [`sort_unstable_by`](Self::sort_unstable_by).
    pub fn sort_unstable_by_key<K: Ord>(&mut self, mut key: impl FnMut(&T) -> K) {
        self.sort_unstable_by(|a, b| key(a).cmp(&key(b)));
    }

    /// Sorts the elements in place, without preserving the order of equal
    /// elements; see [`sort_unstable_by`](Self::sort_unstable_by).
    pub fn sort_unstable(&mut self)
    where
        T: Ord,
    {
        self.sort_unstable_by(T::cmp);
    }

    /// Sorts the elements in place by key, calling `key` once per element
    /// (a stable sort); see [`sort_by`](Self::sort_by).
    pub fn sort_by_cached_key<K: Ord>(&mut self, mut key: impl FnMut(&T) -> K) {
        self.permute_by(|values, order| {
            let keys: Vec<K> = values.iter().map(|v| key(v)).collect();
            order.sort_by(|&a, &b| keys[a].cmp(&keys[b]));
        });
    }

    /// Computes a permutation of the live elements with `arrange` (given
    /// the elements in slot order and the identity permutation to reorder),
    /// then moves the elements to match. If `arrange` panics, nothing moves.
    fn permute_by(&mut self, arrange: impl FnOnce(&[&T], &mut [usize])) {
        let mut values: Vec<&mut T> = self.values_mut().collect();
        // `order[k]` is the position of the element that belongs at `k`.
        let mut order: Vec<usize> = (0..values.len()).collect();
        {
            let shared: Vec<&T> = values.iter().map(|v| &**v).collect();
            arrange(&shared, &mut order);
        }
        // Apply the permutation one cycle at a time: the element for
        // position `k` sits at `order[k]`; follow the cycle with swaps.
        for start in 0..order.len() {
            let mut k = start;
            while order[k] != start {
                let from = order[k];
                let (low, high) = (k.min(from), k.max(from));
                let (left, right) = values.split_at_mut(high);
                core::mem::swap(&mut *left[low], &mut *right[0]);
                order[k] = k;
                k = from;
            }
            order[k] = k;
        }
    }

    /// Sorts the elements in place by key; see [`sort_by`](Self::sort_by).
    pub fn sort_by_key<K: Ord>(&mut self, mut key: impl FnMut(&T) -> K) {
        self.sort_by(|a, b| key(a).cmp(&key(b)));
    }

    /// Sorts the elements in place; see [`sort_by`](Self::sort_by).
    pub fn sort(&mut self)
    where
        T: Ord,
    {
        self.sort_by(T::cmp);
    }

    /// Removes and returns the element of `id`; `None`, and nothing changes,
    /// if `id` is not live.
    pub fn remove(&mut self, id: Id) -> Option<T> {
        let (p, offset) = self.locate(id)?;
        Some(self.vacate(p, offset))
    }

    /// Empties the occupied slot `(p, offset)`: it joins the page's free
    /// list with the next version, or retires after the last generation.
    fn vacate(&mut self, p: usize, offset: usize) -> T {
        // Make the `open.push` below infallible before changing anything.
        self.open.reserve(1);
        let limit = Self::limit(p);
        let page = &mut self.pages[p];
        let had_room = page.has_room(limit);
        let version = page.slots[offset].version();
        debug_assert!(version % 2 == 1);
        let vacant = match version.checked_add(1) {
            Some(version) => Slot::Vacant {
                version,
                next: page.free_head,
            },
            None => Slot::Vacant {
                version: 0,
                next: NONE,
            },
        };
        let retires = vacant.version() == 0;
        let Slot::Occupied { value, .. } = core::mem::replace(&mut page.slots[offset], vacant)
        else {
            unreachable!()
        };
        if retires {
            page.retired += 1;
        } else {
            // `offset < PAGE <= u32::MAX`, and `NONE` is never an offset.
            page.free_head = offset as u32;
        }
        page.len -= 1;
        self.len -= 1;
        if !had_room && page.has_room(limit) {
            self.open.push(p);
        }
        value
    }

    /// Removes every element for which `keep` returns `false`, visiting
    /// them in slot order. If `keep` or a dropped element panics, the
    /// elements already visited stay removed or kept and the hive stays
    /// consistent.
    pub fn retain(&mut self, mut keep: impl FnMut(Id, &mut T) -> bool) {
        for p in 0..self.pages.len() {
            let mut left = self.pages[p].len;
            let mut offset = 0;
            while left > 0 {
                let kept = match &mut self.pages[p].slots[offset] {
                    Slot::Occupied { version, value } => {
                        left -= 1;
                        // `p * PAGE + offset < MAX_INDICES`, so it fits.
                        let id = Id::new((p * PAGE + offset) as u32, generation_of(*version));
                        keep(id, value)
                    }
                    Slot::Vacant { .. } => true,
                };
                if !kept {
                    drop(self.vacate(p, offset));
                }
                offset += 1;
            }
        }
    }

    /// Removes every element. Pages are kept, and slots keep their
    /// generations, so ids issued before stay dead.
    pub fn clear(&mut self) {
        self.retain(|_, _| false);
    }

    /// Removes every element and returns them with their ids, in slot
    /// order. Slots are freed as [`clear`](Hive::clear) frees them.
    ///
    /// Elements are removed as the iterator yields them; dropping it removes
    /// the rest. If it is leaked instead (`mem::forget`), the elements not
    /// yet yielded stay in the hive.
    pub fn drain(&mut self) -> Drain<'_, T, PAGE> {
        Drain(Visit::new(self))
    }

    /// Removes the elements for which `pred` returns `true` and returns them
    /// with their ids, visiting the elements in slot order.
    ///
    /// Elements are tested and removed as the iterator advances: those it
    /// has not reached when it is dropped stay in the hive. Use
    /// [`retain`](Hive::retain) if the removed elements are not needed.
    pub fn extract_if<F: FnMut(Id, &mut T) -> bool>(
        &mut self,
        pred: F,
    ) -> ExtractIf<'_, T, F, PAGE> {
        ExtractIf {
            visit: Visit::new(self),
            pred,
        }
    }

    /// Iterates over the ids and elements, in slot order.
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            pages: self.pages.iter().enumerate(),
            front: PageSlots::default(),
            back: PageSlots::default(),
            page_size: PAGE,
            remaining: self.len,
        }
    }

    /// Iterates over the ids and mutable elements, in slot order.
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        IterMut {
            pages: self.pages.iter_mut().enumerate(),
            front: PageSlots::default(),
            back: PageSlots::default(),
            page_size: PAGE,
            remaining: self.len,
        }
    }

    /// Iterates over the ids, in slot order.
    pub fn ids(&self) -> Ids<'_, T> {
        Ids(self.iter())
    }

    /// Iterates over the elements, in slot order.
    pub fn values(&self) -> Values<'_, T> {
        Values(self.iter())
    }

    /// Iterates over the elements mutably, in slot order.
    pub fn values_mut(&mut self) -> ValuesMut<'_, T> {
        ValuesMut(self.iter_mut())
    }
}

impl<T, const PAGE: usize> Default for Hive<T, PAGE> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Clones page by page, with the same ids, free lists and free pages. Each
/// allocated page of the clone gets the full capacity `PAGE`, so the clone
/// keeps the stable-address guarantee. The clone accepts the original's ids
/// (see the [module docs](crate::hive) on stale ids).
impl<T: Clone, const PAGE: usize> Clone for Hive<T, PAGE> {
    fn clone(&self) -> Self {
        let mut pages = Vec::with_capacity(self.pages.len());
        for page in &self.pages {
            let slots = if page.allocated() {
                let mut slots = Vec::with_capacity(PAGE);
                slots.extend(page.slots.iter().map(|slot| match slot {
                    Slot::Occupied { version, value } => Slot::Occupied {
                        version: *version,
                        value: value.clone(),
                    },
                    Slot::Vacant { version, next } => Slot::Vacant {
                        version: *version,
                        next: *next,
                    },
                }));
                slots
            } else {
                Vec::new()
            };
            pages.push(Page {
                slots,
                free_head: page.free_head,
                len: page.len,
                retired: page.retired,
                floor: page.floor,
            });
        }
        Self {
            pages,
            open: self.open.clone(),
            len: self.len,
        }
    }
}

impl<T: fmt::Debug, const PAGE: usize> fmt::Debug for Hive<T, PAGE> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

/// Equal if both hold the same ids with equal elements, regardless of page
/// size. Free slots, generations of vacant slots and capacity are not
/// compared, so equal hives may issue different ids from then on.
impl<T: PartialEq<U>, U, const PAGE: usize, const PAGE2: usize> PartialEq<Hive<U, PAGE2>>
    for Hive<T, PAGE>
{
    fn eq(&self, other: &Hive<U, PAGE2>) -> bool {
        self.len == other.len
            && self
                .iter()
                .zip(other.iter())
                .all(|((a, x), (b, y))| a == b && x == y)
    }
}

/// Hive [`PartialEq`] is an equivalence relation when elements implement [`Eq`].
impl<T: Eq, const PAGE: usize> Eq for Hive<T, PAGE> {}

/// Hashes the length, then the id-element pairs in slot order, consistent
/// with [`PartialEq`].
impl<T: Hash, const PAGE: usize> Hash for Hive<T, PAGE> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.len);
        for entry in self {
            entry.hash(state);
        }
    }
}

impl<T, const PAGE: usize> Index<Id> for Hive<T, PAGE> {
    type Output = T;

    /// # Panics
    ///
    /// Panics if `id` is not live.
    #[inline]
    fn index(&self, id: Id) -> &T {
        match self.get(id) {
            Some(value) => value,
            None => panic!("Hive: {id:?} is not live"),
        }
    }
}

impl<T, const PAGE: usize> IndexMut<Id> for Hive<T, PAGE> {
    /// # Panics
    ///
    /// Panics if `id` is not live.
    #[inline]
    fn index_mut(&mut self, id: Id) -> &mut T {
        match self.get_mut(id) {
            Some(value) => value,
            None => panic!("Hive: {id:?} is not live"),
        }
    }
}

/// Inserts each element; the ids are discarded.
impl<T, const PAGE: usize> Extend<T> for Hive<T, PAGE> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for value in iter {
            self.insert(value);
        }
    }
}

impl<T, const PAGE: usize> FromIterator<T> for Hive<T, PAGE> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut hive = Self::new();
        hive.extend(iter);
        hive
    }
}

impl<'a, T, const PAGE: usize> IntoIterator for &'a Hive<T, PAGE> {
    type Item = (Id, &'a T);
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

impl<'a, T, const PAGE: usize> IntoIterator for &'a mut Hive<T, PAGE> {
    type Item = (Id, &'a mut T);
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> IterMut<'a, T> {
        self.iter_mut()
    }
}

impl<T, const PAGE: usize> IntoIterator for Hive<T, PAGE> {
    type Item = (Id, T);
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> IntoIter<T> {
        IntoIter {
            pages: self.pages.into_iter().enumerate(),
            front: PageSlots::default(),
            back: PageSlots::default(),
            page_size: PAGE,
            remaining: self.len,
        }
    }
}

/// The occupied slots of one page: `left` of them remain in `slots`, whose
/// offsets start at slot index `base`.
#[derive(Clone)]
struct PageSlots<I> {
    base: usize,
    slots: Enumerate<I>,
    left: u32,
}

impl<I: Default + Iterator> Default for PageSlots<I> {
    fn default() -> Self {
        Self {
            base: 0,
            slots: I::default().enumerate(),
            left: 0,
        }
    }
}

impl<I: DoubleEndedIterator + ExactSizeIterator> PageSlots<I> {
    #[inline]
    fn new(base: usize, slots: I, left: u32) -> Self {
        Self {
            base,
            slots: slots.enumerate(),
            left,
        }
    }

    #[inline]
    fn next<R>(&mut self, take: fn(usize, I::Item) -> Option<R>) -> Option<R> {
        if self.left == 0 {
            return None;
        }
        let base = self.base;
        let item = self
            .slots
            .find_map(|(offset, slot)| take(base + offset, slot));
        debug_assert!(item.is_some());
        self.left -= 1;
        item
    }

    #[inline]
    fn next_back<R>(&mut self, take: fn(usize, I::Item) -> Option<R>) -> Option<R> {
        if self.left == 0 {
            return None;
        }
        let base = self.base;
        let item = self
            .slots
            .by_ref()
            .rev()
            .find_map(|(offset, slot)| take(base + offset, slot));
        debug_assert!(item.is_some());
        self.left -= 1;
        item
    }
}

/// The id of the occupied slot at `index` with `version`.
#[inline]
fn id_at(index: usize, version: u32) -> Id {
    // Slot indices are below `MAX_INDICES`, so they fit.
    Id::new(index as u32, generation_of(version))
}

fn take_ref<T>(index: usize, slot: &Slot<T>) -> Option<(Id, &T)> {
    match slot {
        Slot::Occupied { version, value } => Some((id_at(index, *version), value)),
        Slot::Vacant { .. } => None,
    }
}

fn take_mut<T>(index: usize, slot: &mut Slot<T>) -> Option<(Id, &mut T)> {
    match slot {
        Slot::Occupied { version, value } => Some((id_at(index, *version), value)),
        Slot::Vacant { .. } => None,
    }
}

fn take_owned<T>(index: usize, slot: Slot<T>) -> Option<(Id, T)> {
    match slot {
        Slot::Occupied { version, value } => Some((id_at(index, version), value)),
        Slot::Vacant { .. } => None,
    }
}

/// Defines an iterator over the occupied slots, page by page. `$open(base,
/// page)` makes the [`PageSlots`] of one page and `$take` extracts an item
/// from an occupied slot; `front` and `back` hold the partly consumed pages
/// at either end.
macro_rules! hive_iter {
    (
        $(#[$m:meta])* $name:ident<$($lt:lifetime,)? $t:ident>, $item:ty,
        $pages:ty, $slots:ty, $open:expr, $take:expr
    ) => {
        $(#[$m])*
        pub struct $name<$($lt,)? $t> {
            pages: Enumerate<$pages>,
            front: PageSlots<$slots>,
            back: PageSlots<$slots>,
            page_size: usize,
            remaining: usize,
        }

        impl<$($lt,)? $t> Iterator for $name<$($lt,)? $t> {
            type Item = $item;

            fn next(&mut self) -> Option<$item> {
                if self.remaining == 0 {
                    return None;
                }
                loop {
                    if let Some(x) = self.front.next($take) {
                        self.remaining -= 1;
                        return Some(x);
                    }
                    match self.pages.next() {
                        Some((p, page)) => self.front = $open(p * self.page_size, page),
                        None => {
                            let x = self.back.next($take)?;
                            self.remaining -= 1;
                            return Some(x);
                        }
                    }
                }
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                (self.remaining, Some(self.remaining))
            }
        }

        impl<$($lt,)? $t> DoubleEndedIterator for $name<$($lt,)? $t> {
            fn next_back(&mut self) -> Option<$item> {
                if self.remaining == 0 {
                    return None;
                }
                loop {
                    if let Some(x) = self.back.next_back($take) {
                        self.remaining -= 1;
                        return Some(x);
                    }
                    match self.pages.next_back() {
                        Some((p, page)) => self.back = $open(p * self.page_size, page),
                        None => {
                            let x = self.front.next_back($take)?;
                            self.remaining -= 1;
                            return Some(x);
                        }
                    }
                }
            }
        }

        impl<$($lt,)? $t> ExactSizeIterator for $name<$($lt,)? $t> {}
        impl<$($lt,)? $t> FusedIterator for $name<$($lt,)? $t> {}
    };
}

hive_iter!(
    /// Iterator over the ids and elements of a [`Hive`], in slot order.
    Iter<'a, T>, (Id, &'a T), core::slice::Iter<'a, Page<T>>, core::slice::Iter<'a, Slot<T>>,
    |base, page: &'a Page<T>| PageSlots::new(base, page.slots.iter(), page.len),
    take_ref
);

hive_iter!(
    /// Iterator over the ids and mutable elements of a [`Hive`], in slot
    /// order.
    IterMut<'a, T>, (Id, &'a mut T), core::slice::IterMut<'a, Page<T>>,
    core::slice::IterMut<'a, Slot<T>>,
    |base, page: &'a mut Page<T>| PageSlots::new(base, page.slots.iter_mut(), page.len),
    take_mut
);

hive_iter!(
    /// Owning iterator over the ids and elements of a [`Hive`], in slot
    /// order.
    IntoIter<T>, (Id, T), alloc::vec::IntoIter<Page<T>>, alloc::vec::IntoIter<Slot<T>>,
    |base, page: Page<T>| PageSlots::new(base, page.slots.into_iter(), page.len),
    take_owned
);

impl<T> Clone for Iter<'_, T> {
    fn clone(&self) -> Self {
        Self {
            pages: self.pages.clone(),
            front: self.front.clone(),
            back: self.back.clone(),
            page_size: self.page_size,
            remaining: self.remaining,
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for Iter<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

/// A walk over the occupied slots in slot order that may remove the slot it
/// is at. Removing keeps the walk valid: a vacated slot stays in place.
struct Visit<'a, T, const PAGE: usize> {
    hive: &'a mut Hive<T, PAGE>,
    /// The page being walked, the next offset in it, and the number of its
    /// occupied slots not yet visited.
    page: usize,
    offset: usize,
    left: u32,
    /// The next page to walk.
    next_page: usize,
}

impl<'a, T, const PAGE: usize> Visit<'a, T, PAGE> {
    fn new(hive: &'a mut Hive<T, PAGE>) -> Self {
        Self {
            hive,
            page: 0,
            offset: 0,
            left: 0,
            next_page: 0,
        }
    }

    /// Visits the next occupied slot and removes its element if `remove`
    /// says so; returns the first element removed. `None` once every slot
    /// has been visited. If `remove` panics, its element stays and the walk
    /// goes on after it.
    fn next(&mut self, mut remove: impl FnMut(Id, &mut T) -> bool) -> Option<(Id, T)> {
        loop {
            while self.left == 0 {
                let page = self.hive.pages.get(self.next_page)?;
                self.page = self.next_page;
                self.next_page += 1;
                self.offset = 0;
                self.left = page.len;
            }
            let (p, offset) = (self.page, self.offset);
            self.offset += 1;
            if let Slot::Occupied { version, value } = &mut self.hive.pages[p].slots[offset] {
                self.left -= 1;
                let id = id_at(p * PAGE + offset, *version);
                if remove(id, value) {
                    return Some((id, self.hive.vacate(p, offset)));
                }
            }
        }
    }
}

/// Draining iterator over the ids and elements of a [`Hive`], from
/// [`drain`](Hive::drain).
pub struct Drain<'a, T, const PAGE: usize = DEFAULT_PAGE>(Visit<'a, T, PAGE>);

impl<T, const PAGE: usize> Iterator for Drain<'_, T, PAGE> {
    type Item = (Id, T);

    #[inline]
    fn next(&mut self) -> Option<(Id, T)> {
        self.0.next(|_, _| true)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // Every element still in the hive is yet to be yielded.
        (self.0.hive.len, Some(self.0.hive.len))
    }
}

impl<T, const PAGE: usize> ExactSizeIterator for Drain<'_, T, PAGE> {}
impl<T, const PAGE: usize> FusedIterator for Drain<'_, T, PAGE> {}

impl<T, const PAGE: usize> Drop for Drain<'_, T, PAGE> {
    fn drop(&mut self) {
        self.0.hive.clear();
    }
}

impl<T, const PAGE: usize> fmt::Debug for Drain<'_, T, PAGE> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Drain")
            .field("remaining", &self.0.hive.len)
            .finish_non_exhaustive()
    }
}

/// Iterator that removes and yields the elements of a [`Hive`] matching a
/// predicate, from [`extract_if`](Hive::extract_if).
pub struct ExtractIf<'a, T, F, const PAGE: usize = DEFAULT_PAGE> {
    visit: Visit<'a, T, PAGE>,
    pred: F,
}

impl<T, F: FnMut(Id, &mut T) -> bool, const PAGE: usize> Iterator for ExtractIf<'_, T, F, PAGE> {
    type Item = (Id, T);

    #[inline]
    fn next(&mut self) -> Option<(Id, T)> {
        self.visit.next(&mut self.pred)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.visit.hive.len))
    }
}

impl<T, F: FnMut(Id, &mut T) -> bool, const PAGE: usize> FusedIterator
    for ExtractIf<'_, T, F, PAGE>
{
}

impl<T, F, const PAGE: usize> fmt::Debug for ExtractIf<'_, T, F, PAGE> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExtractIf").finish_non_exhaustive()
    }
}

/// Defines an iterator that maps each item of `$inner` with `$f`.
macro_rules! mapped_iter {
    ($(#[$m:meta])* $name:ident<$lt:lifetime, $t:ident>, $inner:ident, $item:ty, $f:expr) => {
        $(#[$m])*
        pub struct $name<$lt, $t>($inner<$lt, $t>);

        impl<$lt, $t> Iterator for $name<$lt, $t> {
            type Item = $item;

            #[inline]
            fn next(&mut self) -> Option<$item> {
                self.0.next().map($f)
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                self.0.size_hint()
            }
        }

        impl<$lt, $t> DoubleEndedIterator for $name<$lt, $t> {
            #[inline]
            fn next_back(&mut self) -> Option<$item> {
                self.0.next_back().map($f)
            }
        }

        impl<$t> ExactSizeIterator for $name<'_, $t> {}
        impl<$t> FusedIterator for $name<'_, $t> {}
    };
}

mapped_iter!(
    /// Iterator over the ids of a [`Hive`], in slot order.
    Ids<'a, T>, Iter, Id, |(id, _)| id
);

mapped_iter!(
    /// Iterator over the elements of a [`Hive`], in slot order.
    Values<'a, T>, Iter, &'a T, |(_, value)| value
);

mapped_iter!(
    /// Iterator over the mutable elements of a [`Hive`], in slot order.
    ValuesMut<'a, T>, IterMut, &'a mut T, |(_, value)| value
);

impl<T> Clone for Ids<'_, T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T> Clone for Values<'_, T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

#[cfg(feature = "rayon")]
pub use rayon_impl::{ParIter, ParIterMut};

#[cfg(feature = "rayon")]
mod rayon_impl {
    use super::*;
    use core::iter::from_fn;
    use rayon::iter::plumbing::UnindexedConsumer;
    use rayon::prelude::*;

    /// Unindexed parallel iterator over the ids and elements of a [`Hive`].
    pub struct ParIter<'a, T, const PAGE: usize = DEFAULT_PAGE>(&'a [Page<T>]);

    /// Unindexed parallel iterator over the ids and mutable elements of a [`Hive`].
    pub struct ParIterMut<'a, T, const PAGE: usize = DEFAULT_PAGE>(&'a mut [Page<T>]);

    impl<'a, T: Sync, const PAGE: usize> IntoParallelIterator for &'a Hive<T, PAGE> {
        type Item = (Id, &'a T);
        type Iter = ParIter<'a, T, PAGE>;

        fn into_par_iter(self) -> Self::Iter {
            ParIter(&self.pages)
        }
    }

    impl<'a, T: Send, const PAGE: usize> IntoParallelIterator for &'a mut Hive<T, PAGE> {
        type Item = (Id, &'a mut T);
        type Iter = ParIterMut<'a, T, PAGE>;

        fn into_par_iter(self) -> Self::Iter {
            ParIterMut(&mut self.pages)
        }
    }

    impl<'a, T: Sync, const PAGE: usize> ParallelIterator for ParIter<'a, T, PAGE> {
        type Item = (Id, &'a T);

        fn drive_unindexed<C: UnindexedConsumer<Self::Item>>(self, consumer: C) -> C::Result {
            self.0
                .par_iter()
                .enumerate()
                .flat_map_iter(|(number, page)| {
                    let mut slots = PageSlots::new(number * PAGE, page.slots.iter(), page.len);
                    from_fn(move || slots.next(take_ref))
                })
                .drive_unindexed(consumer)
        }
    }

    impl<'a, T: Send, const PAGE: usize> ParallelIterator for ParIterMut<'a, T, PAGE> {
        type Item = (Id, &'a mut T);

        fn drive_unindexed<C: UnindexedConsumer<Self::Item>>(self, consumer: C) -> C::Result {
            self.0
                .par_iter_mut()
                .enumerate()
                .flat_map_iter(|(number, page)| {
                    let mut slots = PageSlots::new(number * PAGE, page.slots.iter_mut(), page.len);
                    from_fn(move || slots.next(take_mut))
                })
                .drive_unindexed(consumer)
        }
    }
}

#[cfg(feature = "serde")]
pub(crate) mod serde_impl {
    use super::*;
    use serde::de::{Deserialize, Deserializer, Error};
    use serde::ser::{Serialize, Serializer};

    /// A slot as serialized: its version, and its element if occupied.
    #[derive(serde::Serialize, serde::Deserialize)]
    enum SlotRepr<V> {
        Occupied(u32, V),
        Vacant(u32),
    }

    struct SlotsRef<'a, T>(&'a [Slot<T>]);

    impl<T: Serialize> Serialize for SlotsRef<'_, T> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            serializer.collect_seq(self.0.iter().map(|slot| match slot {
                Slot::Occupied { version, value } => SlotRepr::Occupied(*version, value),
                Slot::Vacant { version, .. } => SlotRepr::Vacant(*version),
            }))
        }
    }

    #[derive(serde::Serialize)]
    struct PageRef<Slots: Serialize> {
        floor: u32,
        slots: Slots,
    }

    #[cfg(all(
        feature = "concurrent",
        target_has_atomic = "ptr",
        target_has_atomic = "64",
        any(target_pointer_width = "32", target_pointer_width = "64")
    ))]
    pub(crate) mod raw_pages {
        use super::*;
        use core::ops::{Deref, Range};
        use serde::ser::SerializeSeq;

        struct SlotsRef<'a, F> {
            indices: Range<u32>,
            scan: &'a F,
        }

        impl<V: Deref, F: Fn(u32) -> RawSlot<V>> Serialize for SlotsRef<'_, F>
        where
            V::Target: Serialize,
        {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                let mut seq = serializer.serialize_seq(Some(self.indices.len()))?;
                // Each occupied value stays borrowed or guarded only for
                // its own serialize call.
                for index in self.indices.clone() {
                    match (self.scan)(index) {
                        RawSlot::Occupied { version, value } => {
                            seq.serialize_element(&SlotRepr::Occupied(version, &*value))?;
                        }
                        RawSlot::Vacant { version } => {
                            seq.serialize_element(&SlotRepr::<&V::Target>::Vacant(version))?;
                        }
                    }
                }
                seq.end()
            }
        }

        /// Streams raw pages and their guarded or borrowed slot values.
        pub(crate) fn serialize<S: Serializer, V: Deref>(
            pages: impl Iterator<Item = (u32, Range<u32>)>,
            scan: impl Fn(u32) -> RawSlot<V>,
            serializer: S,
        ) -> Result<S::Ok, S::Error>
        where
            V::Target: Serialize,
        {
            serializer.collect_seq(pages.map(|(floor, indices)| PageRef {
                floor,
                slots: SlotsRef {
                    indices,
                    scan: &scan,
                },
            }))
        }
    }

    #[derive(serde::Deserialize)]
    struct PageRepr<T> {
        floor: u32,
        slots: Vec<SlotRepr<T>>,
    }

    /// Serializes the pages in order, each as its generation floor and its
    /// slots (version, and element if occupied). Free lists and capacity
    /// are not stored; a round trip keeps every id and generation.
    impl<T: Serialize, const PAGE: usize> Serialize for Hive<T, PAGE> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            serializer.collect_seq(self.pages.iter().map(|page| PageRef {
                floor: page.floor,
                slots: SlotsRef(&page.slots),
            }))
        }
    }

    /// Deserializes and validates the pages: rejects pages beyond the slot
    /// indices, a page with more than `PAGE` slots, an odd floor, and
    /// versions whose parity does not match the slot state. Free lists are
    /// rebuilt, lowest offset first. Every page that holds a slot is
    /// allocated with room for `PAGE` slots, so memory can reach `PAGE`
    /// slots per page read.
    impl<'de, T: Deserialize<'de>, const PAGE: usize> Deserialize<'de> for Hive<T, PAGE> {
        fn deserialize<De: Deserializer<'de>>(deserializer: De) -> Result<Self, De::Error> {
            let reprs = Vec::<PageRepr<T>>::deserialize(deserializer)?;
            Hive::try_from_raw_pages(reprs.into_iter().map(|repr| {
                RawPage {
                    floor: repr.floor,
                    slots: repr
                        .slots
                        .into_iter()
                        .map(|slot| match slot {
                            SlotRepr::Occupied(version, value) => {
                                RawSlot::Occupied { version, value }
                            }
                            SlotRepr::Vacant(version) => RawSlot::Vacant { version },
                        })
                        .collect(),
                }
            }))
            .map_err(De::Error::custom)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::num::NonZeroU32;

    fn g(n: u32) -> NonZeroU32 {
        NonZeroU32::new(n).unwrap()
    }

    /// Checks every invariant documented on `Hive` and `Page`.
    fn check<T, const PAGE: usize>(hive: &Hive<T, PAGE>) {
        let mut len = 0;
        for (p, page) in hive.pages.iter().enumerate() {
            assert!(p * PAGE < MAX_INDICES as usize);
            let limit = Hive::<T, PAGE>::limit(p);
            let cap = page.slots.capacity();
            assert!(cap == 0 || cap >= PAGE);
            assert!(page.slots.len() <= limit);
            assert!(page.floor % 2 == 0);
            let mut free = alloc::vec![false; page.slots.len()];
            let mut at = page.free_head;
            while at != NONE {
                assert!(!core::mem::replace(&mut free[at as usize], true));
                match page.slots[at as usize] {
                    Slot::Vacant { version, next } => {
                        assert!(version != 0 && version % 2 == 0);
                        at = next;
                    }
                    Slot::Occupied { .. } => panic!("occupied slot in free list"),
                }
            }
            let (mut occupied, mut retired) = (0, 0);
            for (o, slot) in page.slots.iter().enumerate() {
                match *slot {
                    Slot::Occupied { version, .. } => {
                        assert!(version % 2 == 1);
                        occupied += 1;
                    }
                    Slot::Vacant { version: 0, .. } => retired += 1,
                    Slot::Vacant { version, .. } => {
                        assert!(version % 2 == 0);
                        assert!(free[o], "vacant slot missing from free list");
                    }
                }
            }
            assert_eq!(page.len, occupied);
            assert_eq!(page.retired, retired);
            len += occupied as usize;
            let listed = hive.open.iter().filter(|&&q| q == p).count();
            assert_eq!(listed, usize::from(page.has_room(limit)));
        }
        assert!(hive.open.iter().all(|&p| p < hive.pages.len()));
        assert_eq!(hive.len, len);
    }

    #[test]
    fn raw_rebuild_preserves_page_and_hive_invariants() {
        let mut original = Hive::<u32, 4>::with_capacity(20);
        let ids: Vec<_> = (0..9).map(|value| original.insert(value)).collect();
        for &id in &ids[..5] {
            original.remove(id);
        }
        original.shrink_to_fit();
        check(&original);
        let snapshot = |hive: &Hive<u32, 4>| {
            hive.raw_pages()
                .map(|raw| RawPage {
                    floor: raw.floor,
                    slots: raw
                        .slots
                        .into_iter()
                        .map(|slot| match slot {
                            RawSlot::Occupied { version, value } => RawSlot::Occupied {
                                version,
                                value: *value,
                            },
                            RawSlot::Vacant { version } => RawSlot::Vacant { version },
                        })
                        .collect(),
                })
                .collect::<Vec<_>>()
        };
        let rebuilt = Hive::try_from_raw_pages(snapshot(&original)).unwrap();
        check(&rebuilt);
        assert_eq!(original, rebuilt);
        assert_eq!(snapshot(&original), snapshot(&rebuilt));
        check(&Hive::<u32, 4>::try_from_raw_pages([]).unwrap());

        // Retired slots pin their page; empty pages keep their floors and
        // directory positions, including never-used trailing pages.
        let raw = alloc::vec![
            RawPage {
                floor: 2,
                slots: alloc::vec![
                    RawSlot::Vacant { version: 0 },
                    RawSlot::Vacant { version: 4 },
                    RawSlot::Occupied {
                        version: u32::MAX,
                        value: 7,
                    },
                    RawSlot::Vacant { version: 2 },
                ],
            },
            RawPage {
                floor: 8,
                slots: Vec::new(),
            },
            RawPage {
                floor: 0,
                slots: alloc::vec![RawSlot::Occupied {
                    version: 1,
                    value: 9,
                }],
            },
            RawPage {
                floor: u32::MAX - 1,
                slots: Vec::new(),
            },
            RawPage {
                floor: 0,
                slots: Vec::new(),
            },
        ];
        let mut rebuilt = Hive::<u32, 4>::try_from_raw_pages(raw.clone()).unwrap();
        check(&rebuilt);
        assert_eq!(snapshot(&rebuilt), raw);
        assert_eq!(rebuilt.allocated_pages(), 2);
        assert_eq!(rebuilt.index_bound(), 20);
        let last = Id::new(2, g(crate::id_alloc::MAX_GENERATION));
        assert_eq!(rebuilt.remove(last), Some(7));
        check(&rebuilt);
        for expected in [1, 3, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15, 16] {
            assert_eq!(rebuilt.insert(10).index(), expected);
            assert!(!rebuilt.contains(last));
            check(&rebuilt);
        }
        rebuilt.clear();
        check(&rebuilt);
        rebuilt.shrink_to_fit();
        check(&rebuilt);
        let rebuilt = Hive::<u32, 4>::try_from_raw_pages(snapshot(&rebuilt)).unwrap();
        check(&rebuilt);
    }

    #[test]
    fn insert_remove_reuse() {
        let mut hive: Hive<u32, 4> = Hive::new();
        let ids: Vec<Id> = (0..6).map(|i| hive.insert(i)).collect();
        check(&hive);
        assert_eq!(
            ids.iter().map(|id| id.index()).collect::<Vec<_>>(),
            [0, 1, 2, 3, 4, 5]
        );
        assert_eq!(hive.remove(ids[1]), Some(1));
        assert_eq!(hive.remove(ids[1]), None);
        assert_eq!(hive.remove(ids[2]), Some(2));
        check(&hive);
        // Page 0 gained room last, so it is used first, last freed first.
        assert_eq!(hive.insert(10), Id::new(2, g(2)));
        assert_eq!(hive.insert(11), Id::new(1, g(2)));
        assert_eq!(hive.insert(12), Id::new(6, g(1)));
        check(&hive);
        assert_eq!(hive.get(ids[1]), None);
        assert_eq!(hive[Id::new(1, g(2))], 11);
        assert_eq!(hive.occupant(2), Some(Id::new(2, g(2))));
        assert_eq!(hive.occupant(7), None);
        assert_eq!(hive.occupant(99), None);
        assert_eq!(hive.len(), 7);
        assert_eq!(
            hive.values().copied().collect::<Vec<_>>(),
            [0, 11, 10, 3, 4, 5, 12]
        );
        assert_eq!(
            hive.values().rev().copied().collect::<Vec<_>>(),
            [12, 5, 4, 3, 10, 11, 0]
        );
    }

    #[test]
    fn insert_with_sees_its_id_and_panics_cleanly() {
        let mut hive: Hive<Id, 2> = Hive::new();
        let a = hive.insert_with(|id| id);
        assert_eq!(hive[a], a);
        #[cfg(feature = "std")]
        {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                hive.insert_with(|_| panic!("boom"))
            }));
            assert!(r.is_err());
            check(&hive);
            assert_eq!(hive.len(), 1);
            assert_eq!(hive.insert_with(|id| id).index(), 1);
        }
    }

    #[test]
    fn try_insert_reports_exhaustion() {
        let mut hive: Hive<u8, 2> = Hive::new();
        assert!(!hive.is_exhausted());
        assert_eq!(hive.try_insert(1), Ok(Id::new(0, g(1))));
        assert_eq!(
            hive.try_insert_with(|id| id.index() as u8 + 10),
            Some(Id::new(1, g(1)))
        );
        assert_eq!(hive[Id::new(1, g(1))], 11);
        check(&hive);
        // Pretend the directory is full: no page has room, and no page
        // number is left.
        let mut full: Hive<u8, { 1 << 31 }> = Hive::new();
        full.pages.push(Page::unallocated(0));
        full.pages.push(Page::unallocated(0));
        assert!(full.is_exhausted());
        assert_eq!(full.try_insert(7), Err(7));
        assert_eq!(full.try_insert_with(|_| unreachable!()), None);
        assert_eq!(full.len(), 0);
    }

    #[test]
    fn drain_and_extract_if_remove_what_they_yield() {
        let mut hive: Hive<u32, 3> = (0..8).collect();
        let ids: Vec<Id> = hive.ids().collect();
        hive.remove(ids[4]);
        let odd: Vec<(Id, u32)> = hive
            .extract_if(|_, v| {
                *v += 10;
                *v % 2 == 1
            })
            .collect();
        check(&hive);
        assert_eq!(
            odd,
            [(ids[1], 11), (ids[3], 13), (ids[5], 15), (ids[7], 17)]
        );
        assert_eq!(hive.values().copied().collect::<Vec<_>>(), [10, 12, 16]);
        // Dropped early: the elements not reached are kept, untested.
        assert_eq!(hive.extract_if(|_, _| true).next(), Some((ids[0], 10)));
        check(&hive);
        assert_eq!(hive.values().copied().collect::<Vec<_>>(), [12, 16]);

        hive.extend([20, 21]);
        check(&hive);
        let mut drain = hive.drain();
        assert_eq!(drain.len(), 4);
        let first = drain.next().unwrap();
        assert_eq!(drain.len(), 3);
        drop(drain);
        check(&hive);
        assert!(hive.is_empty());
        assert!(!hive.contains(first.0));
        // Slots are reused with new generations after a drain.
        let again = hive.insert(1);
        assert!(again.generation().get() > 1);
        assert_eq!(hive.drain().collect::<Vec<_>>(), [(again, 1)]);
        assert_eq!(hive.drain().next(), None);
        check(&hive);
    }

    #[test]
    fn get_disjoint_mut_borrows_each_live_id() {
        let mut hive: Hive<u32, 2> = (0..5).collect();
        let ids: Vec<Id> = hive.ids().collect();
        hive.remove(ids[2]);
        // Out of slot order, two in one page, one stale, one unknown.
        let unknown = Id::new(99, g(1));
        let got = hive.get_disjoint_mut([ids[4], ids[1], ids[2], ids[0], unknown]);
        let values: Vec<Option<u32>> = got.iter().map(|v| v.as_deref().copied()).collect();
        assert_eq!(values, [Some(4), Some(1), None, Some(0), None]);
        let [Some(e), Some(b), None, Some(a), None] = got else {
            panic!()
        };
        *e += 100;
        *b += 10;
        core::mem::swap(a, b);
        assert_eq!(hive.values().copied().collect::<Vec<_>>(), [11, 0, 3, 104]);
        assert!(hive.get_disjoint_mut([]).is_empty());
        // The same stale id twice is fine: neither is live.
        assert!(
            hive.get_disjoint_mut([ids[2], ids[2]])
                .iter()
                .all(Option::is_none)
        );
        check(&hive);
    }

    #[cfg(feature = "std")]
    #[test]
    #[should_panic(expected = "given twice")]
    fn get_disjoint_mut_rejects_a_repeated_live_id() {
        let mut hive: Hive<u32, 2> = (0..3).collect();
        let id = hive.ids().nth(1).unwrap();
        let _ = hive.get_disjoint_mut([id, id]);
    }

    #[test]
    fn retire_after_last_generation() {
        let mut hive: Hive<u8, 2> = Hive::new();
        let a = hive.insert(1);
        let b = hive.insert(2);
        // Jump slot 0 to its last generation.
        let Slot::Occupied { version, .. } = &mut hive.pages[0].slots[0] else {
            unreachable!()
        };
        *version = u32::MAX;
        let last = Id::new(0, g(crate::id_alloc::MAX_GENERATION));
        assert!(!hive.contains(a));
        assert_eq!(hive.remove(last), Some(1));
        check(&hive);
        assert_eq!(hive.pages[0].retired, 1);
        // Page 0 has no room: the retired slot is never reused.
        assert_eq!(hive.insert(3).index(), 2);
        hive.remove(b);
        hive.shrink_to_fit();
        check(&hive);
        // A page with a retired slot is never freed.
        assert!(hive.pages[0].allocated());
        assert_eq!(hive.insert(4), Id::new(1, g(2)));
    }

    #[test]
    fn shrink_frees_empty_pages_and_keeps_generations() {
        let mut hive: Hive<u32, 2> = Hive::new();
        let ids: Vec<Id> = (0..6).map(|i| hive.insert(i)).collect();
        let addr: *const u32 = &hive[ids[4]];
        for &id in &ids[..4] {
            hive.remove(id);
        }
        // Reuse slot 3 a few times to raise its generation.
        for _ in 0..3 {
            let id = hive.insert(9);
            hive.remove(id);
        }
        hive.shrink_to_fit();
        check(&hive);
        assert_eq!(hive.allocated_pages(), 1);
        assert_eq!(hive.capacity(), 2);
        assert!(core::ptr::eq(addr, &hive[ids[4]]));
        assert_eq!(hive.pages[0].floor, 2);
        assert_eq!(hive.pages[1].floor, 8);
        // Freed pages are used again, lowest first, above every old generation.
        let a = hive.insert(20);
        assert_eq!(a, Id::new(0, g(2)));
        assert!(
            ids.iter()
                .all(|&id| id == ids[4] || id == ids[5] || !hive.contains(id))
        );
        hive.clear();
        hive.shrink_to_fit();
        check(&hive);
        assert_eq!(hive.allocated_pages(), 0);
        assert_eq!(hive.pages.len(), 3);
        assert!(ids.iter().all(|&id| !hive.contains(id)));
        // Trailing pages that never issued ids are dropped.
        let mut fresh: Hive<u32, 2> = Hive::with_capacity(10);
        assert_eq!(fresh.allocated_pages(), 5);
        fresh.insert(1);
        fresh.shrink_to_fit();
        check(&fresh);
        assert_eq!(fresh.pages.len(), 1);
    }

    #[test]
    fn reserve_fills_open_pages_first() {
        let mut hive: Hive<u32, 4> = Hive::new();
        let ids: Vec<Id> = (0..4).map(|i| hive.insert(i)).collect();
        hive.remove(ids[2]);
        hive.reserve(9);
        check(&hive);
        assert_eq!(hive.allocated_pages(), 3);
        let next: Vec<u32> = (0..9).map(|i| hive.insert(i).index()).collect();
        assert_eq!(next, [2, 4, 5, 6, 7, 8, 9, 10, 11]);
        assert_eq!(hive.allocated_pages(), 3);
        check(&hive);
        assert!(hive.try_reserve(usize::MAX).is_err());
        check(&hive);
    }

    #[test]
    fn retain_iter_and_eq() {
        let mut hive: Hive<u32, 3> = (0..10).collect();
        hive.retain(|id, v| {
            *v += 100;
            id.index() % 3 != 1
        });
        check(&hive);
        assert_eq!(
            hive.ids().map(Id::index).collect::<Vec<_>>(),
            [0, 2, 3, 5, 6, 8, 9]
        );
        let mut it = hive.iter();
        assert_eq!(it.len(), 7);
        assert_eq!(it.next().map(|(_, v)| *v), Some(100));
        assert_eq!(it.next_back().map(|(_, v)| *v), Some(109));
        assert_eq!(it.len(), 5);
        for v in hive.values_mut() {
            *v -= 100;
        }
        let clone = hive.clone();
        check(&clone);
        assert_eq!(clone, hive);
        let other: Hive<u32, 16> = hive.iter().map(|(_, v)| *v).collect();
        assert_ne!(other, hive);
        let owned: Vec<(Id, u32)> = hive.into_iter().rev().collect();
        assert_eq!(owned.len(), 7);
        assert_eq!(owned[0].1, 9);
        assert_eq!(
            alloc::format!("{:?}", clone.iter().next()),
            "Some((Id(0v1), 0))"
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_round_trip_and_validation() {
        let mut hive: Hive<u8, 2> = Hive::new();
        let ids: Vec<Id> = (0..3).map(|i| hive.insert(i)).collect();
        hive.remove(ids[0]);
        let json = serde_json::to_string(&hive).unwrap();
        assert_eq!(
            json,
            r#"[{"floor":0,"slots":[{"Vacant":2},{"Occupied":[1,1]}]},{"floor":0,"slots":[{"Occupied":[1,2]}]}]"#
        );
        let mut back: Hive<u8, 2> = serde_json::from_str(&json).unwrap();
        check(&back);
        assert_eq!(back, hive);
        assert_eq!(back.insert(7), Id::new(0, g(2)));

        for bad in [
            r#"[{"floor":1,"slots":[]}]"#,
            r#"[{"floor":0,"slots":[{"Vacant":1}]}]"#,
            r#"[{"floor":0,"slots":[{"Occupied":[2,0]}]}]"#,
            r#"[{"floor":0,"slots":[{"Vacant":2},{"Vacant":2},{"Vacant":2}]}]"#,
        ] {
            assert!(serde_json::from_str::<Hive<u8, 2>>(bad).is_err(), "{bad}");
        }
    }
}
