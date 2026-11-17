//! [`PagedDeque`]: a double-ended queue stored in fixed-size pages.
//!
//! An element keeps its address for as long as it stays in the deque,
//! through pushes and pops at either end, reservations, and shrinking.
//! Front operations change indices without moving the surviving elements.
//! Operations that reorder elements, such as insertion, removal, swapping,
//! and retention, can move them. Truncation drops only the removed elements.
//!
//! Elements are not contiguous across pages. Use [`pages`](PagedDeque::pages)
//! for the initialized part of each used page; the first and last slices
//! can be partial. Spare pages are kept at the back and reused at either end.
//!
//! | Operation | Cost |
//! |---|---|
//! | pops, access, indexing, swapping | O(1) |
//! | pushes | amortized O(1), occasional page allocation and directory growth |
//! | [`insert`](PagedDeque::insert), [`remove`](PagedDeque::remove) | O(min(index, len − index)) moves, plus allocation for insertion |
//! | truncation, clearing | O(removed) drops; pages are kept |
//! | retention | O(len), no allocation |
//! | append, extension | O(added), plus allocation and any cloning |
//! | iteration, ranges | O(1) setup, O(1) amortized per element |
//! | binary search, partition point | O(log len), no allocation |
//! | reserve, shrink | O(allocated or freed pages), plus directory resizing |
//!
//! ```
//! use paged_collections::paged_deque::PagedDeque;
//!
//! let mut queue: PagedDeque<u32, 4> = [10, 20].into();
//! let first: *const u32 = &queue[0];
//! queue.push_front(5);
//! queue.extend(30..100);
//! queue.reserve(100);
//! assert!(core::ptr::eq(first, &queue[1]));
//! assert_eq!(queue.pop_front(), Some(5));
//! assert_eq!(queue.front(), Some(&10));
//! ```

use alloc::boxed::Box;
use alloc::collections::{TryReserveError, VecDeque, vec_deque};
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::iter::FusedIterator;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ops::{Bound, Index, IndexMut, RangeBounds};
use core::ptr::{self, NonNull};

use crate::page::DEFAULT_PAGE;
use crate::paged_vec::capacity_overflow;

/// A double-ended queue whose surviving elements stay at stable addresses
/// through growth and pops at either end.
///
/// `PAGE` is the non-zero number of slots per page. See the
/// [module documentation](crate::paged_deque) for costs and address guarantees.
// Invariants:
// 1. PAGE > 0 and head < PAGE.
// 2. Exactly head..head + len is initialized and owned; head + len cannot
//    overflow usize. All other slots are uninitialized (contain no owned T).
// 3. pages.len() * PAGE >= head + len; the product cannot overflow usize.
//    Fully unused spare pages occur only at the back, except that an empty
//    deque may have an unused first page containing its head offset.
// 4. Every entry uniquely owns a live allocation from try_new_page, with
//    exactly PAGE MaybeUninit<T> slots (a dangling Box pointer for ZSTs).
// Raw pointers preserve page provenance when the directory moves. Never
// create a page-wide reference to access a single slot or a partial range.
pub struct PagedDeque<T, const PAGE: usize = DEFAULT_PAGE> {
    pages: VecDeque<NonNull<T>>,
    head: usize,
    len: usize,
    _owns: PhantomData<T>,
}

// SAFETY: invariant 4 gives exclusive ownership of all allocations; moving
// the deque between threads transfers its owned T values, just like Vec<T>.
unsafe impl<T: Send, const PAGE: usize> Send for PagedDeque<T, PAGE> {}
// SAFETY: invariants 2 and 4 permit shared access only to initialized T;
// all mutation requires exclusive access to the deque.
unsafe impl<T: Sync, const PAGE: usize> Sync for PagedDeque<T, PAGE> {}

impl<T, const PAGE: usize> PagedDeque<T, PAGE> {
    /// Number of slots per page.
    pub const PAGE: usize = PAGE;

    /// An empty deque. Does not allocate.
    pub const fn new() -> Self {
        const { assert!(PAGE > 0, "page size must be non-zero") };
        Self {
            pages: VecDeque::new(),
            head: 0,
            len: 0,
            _owns: PhantomData,
        }
    }

    /// An empty deque with pages allocated for at least `capacity` elements.
    /// Panics on capacity overflow, as [`reserve`](Self::reserve) does.
    pub fn with_capacity(capacity: usize) -> Self {
        let mut deque = Self::new();
        deque.reserve(capacity);
        deque
    }

    fn try_new_page() -> Result<NonNull<T>, TryReserveError> {
        let mut page: Vec<MaybeUninit<T>> = Vec::new();
        page.try_reserve_exact(PAGE)?;
        // SAFETY: establishing invariant 4: capacity is at least PAGE and
        // MaybeUninit<T> requires no initialization.
        unsafe { page.set_len(PAGE) };
        let raw = Box::into_raw(page.into_boxed_slice());
        Ok(NonNull::new(raw.cast::<T>()).expect("Box pointers are non-null"))
    }

    /// Releases a page allocation without dropping any slots.
    ///
    /// # Safety
    /// `page` must uniquely own an allocation from try_new_page (invariant
    /// 4), have no initialized owned slots, and never be used again.
    unsafe fn free_page(page: NonNull<T>) {
        let raw = ptr::slice_from_raw_parts_mut(page.as_ptr().cast::<MaybeUninit<T>>(), PAGE);
        // SAFETY: invariant 4 and the caller's contract give the exact Box
        // allocation and layout, with no owned values or further users.
        let allocation = unsafe { Box::from_raw(raw) };
        drop(allocation);
    }

    // Produces only a raw pointer, never a reference to neighboring slots.
    // Invariant 4 supplies the allocation; callers choose an in-page range.
    fn page_slice(page: NonNull<T>, start: usize, end: usize) -> *mut [T] {
        debug_assert!(start <= end && end <= PAGE);
        // SAFETY: invariant 4 gives PAGE slots; start <= PAGE, including
        // the allowed one-past pointer for an empty range.
        let first = unsafe { page.as_ptr().add(start) };
        ptr::slice_from_raw_parts_mut(first, end - start)
    }

    // Absolute slot within the directory, initialized or the next free slot.
    fn slot(&self, absolute: usize) -> *mut T {
        let page = self.pages[absolute / PAGE];
        // SAFETY: invariants 1 and 4 give a live page and offset < PAGE.
        unsafe { page.as_ptr().add(absolute % PAGE) }
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether there are no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Length reachable by back pushes without allocating another page.
    /// Spare pages can also serve front pushes.
    pub fn capacity(&self) -> usize {
        self.pages.len() * PAGE - self.head
    }

    /// Number of allocated pages, including empty spares.
    pub fn allocated_pages(&self) -> usize {
        self.pages.len()
    }

    /// Allocates pages now so that `capacity() >= len() + additional`.
    /// Spare pages are reusable at either end; front pushes may also use
    /// the unused slots before the head.
    ///
    /// # Panics
    ///
    /// Panics on length or slot-count overflow, or allocation failure.
    pub fn reserve(&mut self, additional: usize) {
        self.try_reserve(additional)
            .unwrap_or_else(|error| panic!("{error}"));
    }

    /// Fallible [`reserve`](Self::reserve). On error, any pages allocated
    /// so far are kept and all existing elements stay unchanged.
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        let total = self
            .head
            .checked_add(self.len)
            .and_then(|n| n.checked_add(additional));
        let total = total.ok_or_else(capacity_overflow)?;
        let needed = total.div_ceil(PAGE);
        needed.checked_mul(PAGE).ok_or_else(capacity_overflow)?;
        if needed > self.pages.len() {
            self.pages.try_reserve(needed - self.pages.len())?;
            while self.pages.len() < needed {
                self.pages.push_back(Self::try_new_page()?);
            }
        }
        Ok(())
    }

    /// Frees spare pages and shrinks the directory without moving elements.
    /// An empty deque releases all pages and resets its head offset.
    pub fn shrink_to_fit(&mut self) {
        let used = if self.is_empty() {
            0
        } else {
            (self.head + self.len).div_ceil(PAGE)
        };
        if self.is_empty() {
            self.head = 0;
        }
        while self.pages.len() > used {
            let page = self.pages.pop_back().expect("spare page exists");
            // SAFETY: invariants 2–4: pages past used are uninitialized,
            // uniquely allocated pages, removed from the directory forever.
            unsafe { Self::free_page(page) };
        }
        self.pages.shrink_to_fit();
    }

    /// Appends an element. Writes only its new slot, preserving raw pointers
    /// into all existing elements, including those in the same page.
    /// Panics on capacity overflow or allocation failure.
    pub fn push_back(&mut self, value: T) {
        self.reserve(1);
        let slot = self.slot(self.head + self.len);
        // SAFETY: invariants 2–4 and reserve give an uninitialized allocated
        // slot immediately after the initialized interval; access only it.
        unsafe { slot.write(value) };
        self.len += 1;
    }

    /// Prepends an element without moving existing elements or creating
    /// references to their slots. Panics on capacity overflow or allocation
    /// failure, leaving the existing elements unchanged.
    pub fn push_front(&mut self, value: T) {
        let new_len = self.len.checked_add(1).expect("PagedDeque length overflow");
        if self.head == 0 {
            if self.len == 0 && !self.pages.is_empty() {
                self.head = PAGE;
            } else {
                let used = self.len.div_ceil(PAGE);
                let page = if self.pages.len() > used {
                    // No directory growth: removing a spare makes room.
                    self.pages.pop_back().expect("spare page exists")
                } else {
                    let count = self.pages.len().checked_add(1).expect("capacity overflow");
                    count.checked_mul(PAGE).expect("capacity overflow");
                    // Reserve the directory before allocating the raw page:
                    // neither a growth panic nor a failed page can leak it.
                    self.pages.reserve(1);
                    Self::try_new_page().unwrap_or_else(|error| panic!("{error}"))
                };
                self.pages.push_front(page);
                self.head = PAGE;
            }
        }
        self.head -= 1;
        let slot = self.slot(self.head);
        // SAFETY: invariants 2–4: the slot just before the old initialized
        // interval is allocated and uninitialized; no other slot is touched.
        unsafe { slot.write(value) };
        self.len = new_len;
    }

    /// Removes and returns the last element. Pages are kept for reuse.
    pub fn pop_back(&mut self) -> Option<T> {
        self.len = self.len.checked_sub(1)?;
        let slot = self.slot(self.head + self.len);
        // SAFETY: invariants 2 and 4: this was the last initialized slot;
        // it has left the owned interval and is read exactly once.
        Some(unsafe { slot.read() })
    }

    /// Removes and returns the first element. An emptied front page rotates
    /// to the back as a spare without allocating or moving any elements.
    pub fn pop_front(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }
        let slot = self.slot(self.head);
        self.head += 1;
        self.len -= 1;
        if self.head == PAGE {
            self.head = 0;
            let page = self.pages.pop_front().expect("front page exists");
            self.pages.push_back(page);
        }
        // SAFETY: invariants 2 and 4: the former first initialized slot is
        // no longer owned by the deque and is read exactly once.
        Some(unsafe { slot.read() })
    }

    /// The element at `index`, or `None` if out of bounds.
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            return None;
        }
        // SAFETY: invariants 2 and 4 give one initialized live slot; the
        // shared borrow of self prevents mutation for the reference's life.
        Some(unsafe { &*self.slot(self.head + index) })
    }

    /// The element at `index`, mutably, or `None` if out of bounds.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.len {
            return None;
        }
        // SAFETY: invariants 2 and 4 give one initialized live slot;
        // exclusive self access grants exclusive access to exactly this T.
        Some(unsafe { &mut *self.slot(self.head + index) })
    }

    /// The first element.
    pub fn front(&self) -> Option<&T> {
        self.get(0)
    }

    /// The first element, mutably.
    pub fn front_mut(&mut self) -> Option<&mut T> {
        self.get_mut(0)
    }

    /// The last element.
    pub fn back(&self) -> Option<&T> {
        self.get(self.len.checked_sub(1)?)
    }

    /// The last element, mutably.
    pub fn back_mut(&mut self) -> Option<&mut T> {
        self.get_mut(self.len.checked_sub(1)?)
    }

    /// Swaps two elements. Panics if either index is out of bounds.
    pub fn swap(&mut self, a: usize, b: usize) {
        assert!(
            a < self.len && b < self.len,
            "swap indices (are {a} and {b}) should be < len (is {})",
            self.len
        );
        let a = self.slot(self.head + a);
        let b = self.slot(self.head + b);
        // SAFETY: invariants 2 and 4 give initialized slots; exclusive self
        // access permits writes, and ptr::swap allows overlapping pointers.
        unsafe { ptr::swap(a, b) };
    }

    /// Whether an element equals `value`. Costs O(len).
    pub fn contains(&self, value: &T) -> bool
    where
        T: PartialEq,
    {
        self.pages().any(|page| page.contains(value))
    }

    /// Inserts at `index`, shifting the shorter side by one slot.
    /// Costs O(min(index, len − index)), plus allocation if needed.
    /// Panics if `index > len` or allocation fails.
    pub fn insert(&mut self, index: usize, value: T) {
        assert!(
            index <= self.len,
            "insertion index (is {index}) should be <= len (is {})",
            self.len
        );
        if index < self.len - index {
            self.push_front(value);
            for i in 0..index {
                self.swap(i, i + 1);
            }
        } else {
            self.push_back(value);
            for i in (index..self.len - 1).rev() {
                self.swap(i, i + 1);
            }
        }
    }

    /// Removes at `index`, shifting the shorter side by one slot.
    /// Returns `None` if out of bounds. Costs O(min(index, len − index)).
    pub fn remove(&mut self, index: usize) -> Option<T> {
        if index >= self.len {
            return None;
        }
        if index < self.len - index - 1 {
            for i in (0..index).rev() {
                self.swap(i, i + 1);
            }
            self.pop_front()
        } else {
            for i in index..self.len - 1 {
                self.swap(i, i + 1);
            }
            self.pop_back()
        }
    }

    /// Removes at `index`, replacing it with the last element. O(1).
    /// Returns `None` if out of bounds.
    pub fn swap_remove_back(&mut self, index: usize) -> Option<T> {
        if index >= self.len {
            return None;
        }
        self.swap(index, self.len - 1);
        self.pop_back()
    }

    /// Removes at `index`, replacing it with the first element. O(1).
    /// Returns `None` if out of bounds.
    pub fn swap_remove_front(&mut self, index: usize) -> Option<T> {
        if index >= self.len {
            return None;
        }
        self.swap(index, 0);
        self.pop_front()
    }

    /// Drops elements from the back until at most `len` remain. Pages are
    /// kept. A panicking destructor leaves its element removed and all
    /// elements before it still in the deque.
    pub fn truncate(&mut self, len: usize) {
        while self.len > len {
            drop(self.pop_back());
        }
    }

    /// Keeps the last `len` elements, dropping from the front. Pages are
    /// kept. A panicking destructor leaves its element removed and the
    /// unvisited suffix still in the deque.
    pub fn truncate_front(&mut self, len: usize) {
        while self.len > len {
            drop(self.pop_front());
        }
    }

    /// Drops all elements, keeping pages. See [`truncate`](Self::truncate)
    /// for destructor panic behavior.
    pub fn clear(&mut self) {
        self.truncate(0);
    }

    /// Resizes by appending `f()` results or truncating from the back.
    /// If `f` panics, all elements appended so far stay in the deque.
    pub fn resize_with(&mut self, new_len: usize, mut f: impl FnMut() -> T) {
        if new_len <= self.len {
            self.truncate(new_len);
        } else {
            self.reserve(new_len - self.len);
            while self.len < new_len {
                self.push_back(f());
            }
        }
    }

    /// Resizes by appending clones of `value` or truncating from the back.
    /// Completed clones stay in the deque if cloning panics.
    pub fn resize(&mut self, new_len: usize, value: T)
    where
        T: Clone,
    {
        self.resize_with(new_len, || value.clone());
    }

    /// Keeps matching elements in order, without allocating. Rejected
    /// elements are dropped at the end. If the predicate panics, kept and
    /// unvisited elements remain in order. If a rejected element's drop
    /// panics, not-yet-dropped rejected elements remain at the back.
    pub fn retain(&mut self, mut keep: impl FnMut(&T) -> bool) {
        self.retain_mut(|value| keep(value));
    }

    /// Like [`retain`](Self::retain), with mutable access to each element.
    pub fn retain_mut(&mut self, mut keep: impl FnMut(&mut T) -> bool) {
        // All slots stay initialized. Kept values occupy ..write,
        // rejected values write..read, and the unvisited tail read..len.
        struct Guard<'a, T, const PAGE: usize> {
            deque: &'a mut PagedDeque<T, PAGE>,
            read: usize,
            write: usize,
        }
        impl<T, const PAGE: usize> Drop for Guard<'_, T, PAGE> {
            fn drop(&mut self) {
                let mut write = self.write;
                for read in self.read..self.deque.len {
                    self.deque.swap(write, read);
                    write += 1;
                }
                self.deque.truncate(write);
            }
        }
        let mut guard = Guard {
            deque: self,
            read: 0,
            write: 0,
        };
        while guard.read < guard.deque.len {
            if keep(
                guard
                    .deque
                    .get_mut(guard.read)
                    .expect("visited element exists"),
            ) {
                guard.deque.swap(guard.write, guard.read);
                guard.write += 1;
            }
            guard.read += 1;
        }
    }

    /// Moves all of `other` to the back in order. Costs O(other.len()),
    /// plus allocation. `other` keeps its pages. All allocation precedes
    /// the moves; existing elements in `self` keep their addresses.
    pub fn append(&mut self, other: &mut Self) {
        self.reserve(other.len);
        while let Some(value) = other.pop_front() {
            self.push_back(value);
        }
    }

    /// Appends clones of a slice, preserving existing element addresses.
    /// Completed clones stay in the deque if cloning panics.
    pub fn extend_from_slice(&mut self, other: &[T])
    where
        T: Clone,
    {
        self.reserve(other.len());
        for value in other {
            self.push_back(value.clone());
        }
    }

    /// Binary search in a sorted deque, like [`slice::binary_search`].
    /// Returns a matching index or the insertion index on failure.
    pub fn binary_search(&self, value: &T) -> Result<usize, usize>
    where
        T: Ord,
    {
        self.binary_search_by(|item| item.cmp(value))
    }

    /// Binary search with the element's ordering relative to the target.
    /// Results are unspecified, but safe, if the deque is not sorted
    /// consistently with the comparator.
    pub fn binary_search_by<'a>(
        &'a self,
        mut compare: impl FnMut(&'a T) -> Ordering,
    ) -> Result<usize, usize> {
        let (mut lo, mut hi) = (0, self.len);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match compare(&self[mid]) {
                Ordering::Less => lo = mid + 1,
                Ordering::Equal => return Ok(mid),
                Ordering::Greater => hi = mid,
            }
        }
        Err(lo)
    }

    /// Binary search by a key, like [`slice::binary_search_by_key`].
    pub fn binary_search_by_key<'a, K: Ord>(
        &'a self,
        key: &K,
        mut f: impl FnMut(&'a T) -> K,
    ) -> Result<usize, usize> {
        self.binary_search_by(|value| f(value).cmp(key))
    }

    /// First index where `pred` is false, given a true prefix followed by
    /// a false suffix, like [`slice::partition_point`]. No allocation.
    pub fn partition_point(&self, mut pred: impl FnMut(&T) -> bool) -> usize {
        let (mut lo, mut hi) = (0, self.len);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if pred(&self[mid]) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// Iterates over elements in order, from either end.
    pub fn iter(&self) -> Iter<'_, T, PAGE> {
        self.range(..)
    }

    /// Iterates over elements mutably, from either end.
    pub fn iter_mut(&mut self) -> IterMut<'_, T, PAGE> {
        self.range_mut(..)
    }

    /// Iterates over a range, with O(1) setup and no allocation.
    ///
    /// # Panics
    ///
    /// Panics if the start exceeds the end, the end exceeds `len`, or an
    /// excluded start or included end overflows `usize`.
    ///
    /// ```
    /// use paged_collections::paged_deque::PagedDeque;
    /// let deque: PagedDeque<_, 2> = [1, 2, 3, 4].into();
    /// assert!(deque.range(1..=2).copied().eq([2, 3]));
    /// ```
    pub fn range<R: RangeBounds<usize>>(&self, range: R) -> Iter<'_, T, PAGE> {
        let (start, end) = range_indices(range, self.len);
        Iter {
            pages: Pages {
                cursor: self.cursor(start, end),
                _borrow: PhantomData,
            },
            front: [].iter(),
            back: [].iter(),
            remaining: end - start,
        }
    }

    /// Mutable [`range`](Self::range), with the same bounds and panics.
    /// References cover only the requested slots, even in partial pages.
    pub fn range_mut<R: RangeBounds<usize>>(&mut self, range: R) -> IterMut<'_, T, PAGE> {
        let (start, end) = range_indices(range, self.len);
        IterMut {
            pages: PagesMut {
                cursor: self.cursor(start, end),
                _borrow: PhantomData,
            },
            front: [].iter_mut(),
            back: [].iter_mut(),
            remaining: end - start,
        }
    }

    /// Initialized parts of used pages, in order, without empty slices.
    /// The first and last slices can be partial; all others have PAGE items.
    pub fn pages(&self) -> Pages<'_, T, PAGE> {
        Pages {
            cursor: self.cursor(0, self.len),
            _borrow: PhantomData,
        }
    }

    /// Initialized parts of used pages as disjoint mutable slices.
    pub fn pages_mut(&mut self) -> PagesMut<'_, T, PAGE> {
        PagesMut {
            cursor: self.cursor(0, self.len),
            _borrow: PhantomData,
        }
    }

    fn cursor(&self, start: usize, end: usize) -> PageCursor<'_, T, PAGE> {
        if start == end {
            return PageCursor {
                pages: self.pages.range(0..0),
                front: 0,
                back: 0,
            };
        }
        let start = self.head + start;
        let end = self.head + end;
        PageCursor {
            pages: self.pages.range(start / PAGE..=(end - 1) / PAGE),
            front: start % PAGE,
            back: (end - 1) % PAGE + 1,
        }
    }

    // Drops a page at a time. Unlink and update the initialized interval
    // before invoking user destructors; slice drop glue finishes that page
    // on unwind, and the allocation guard then frees it.
    fn drop_pages(&mut self) {
        struct Allocation<T, const PAGE: usize>(NonNull<T>);
        impl<T, const PAGE: usize> Drop for Allocation<T, PAGE> {
            fn drop(&mut self) {
                // SAFETY: invariant 4 supplies the unique allocation;
                // slice drop glue finished every owned slot before this
                // guard runs, on both normal return and unwinding.
                unsafe { PagedDeque::<T, PAGE>::free_page(self.0) };
            }
        }
        while let Some(page) = self.pages.pop_front() {
            let allocation = Allocation::<T, PAGE>(page);
            let count = self.len.min(PAGE - self.head);
            let slice = Self::page_slice(page, self.head, self.head + count);
            self.len -= count;
            self.head = 0;
            // SAFETY: invariants 2 and 4 identify exactly this initialized
            // sub-slice. It has left the deque and its allocation guard
            // stays live until all its destructors finish, including unwind.
            unsafe { ptr::drop_in_place(slice) };
            drop(allocation);
        }
    }

    #[cfg(test)]
    fn check(&self) {
        assert!(PAGE > 0 && self.head < PAGE);
        let end = self
            .head
            .checked_add(self.len)
            .expect("slot interval overflow");
        let slots = self
            .pages
            .len()
            .checked_mul(PAGE)
            .expect("page capacity overflow");
        assert!(slots >= end);
        assert!(self.pages.iter().all(|page| page.as_ptr().is_aligned()));
        if core::mem::size_of::<T>() != 0 {
            for (i, page) in self.pages.iter().enumerate() {
                assert!(self.pages.iter().skip(i + 1).all(|other| page != other));
            }
        }
        // Initialization and allocation provenance cannot be inspected
        // directly; reading exactly the interval exercises them under Miri.
        for (i, page) in self.pages().enumerate() {
            let start = if i == 0 { self.head } else { 0 };
            assert!(!page.is_empty());
            assert_eq!(page.len(), PAGE.min(end - i * PAGE) - start);
        }
        assert_eq!(self.pages().map(<[T]>::len).sum::<usize>(), self.len);
        assert_eq!(self.iter().count(), self.len);
    }
}

impl<T, const PAGE: usize> Drop for PagedDeque<T, PAGE> {
    fn drop(&mut self) {
        struct Guard<'a, T, const PAGE: usize>(&'a mut PagedDeque<T, PAGE>);
        impl<T, const PAGE: usize> Drop for Guard<'_, T, PAGE> {
            fn drop(&mut self) {
                self.0.drop_pages();
            }
        }
        let guard = Guard(self);
        guard.0.drop_pages();
    }
}

#[track_caller]
fn range_indices(range: impl RangeBounds<usize>, len: usize) -> (usize, usize) {
    let start = match range.start_bound() {
        Bound::Included(&n) => n,
        Bound::Excluded(&n) => n.checked_add(1).expect("range start index overflow"),
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&n) => n.checked_add(1).expect("range end index overflow"),
        Bound::Excluded(&n) => n,
        Bound::Unbounded => len,
    };
    assert!(
        start <= end,
        "range start (is {start}) should be <= end (is {end})"
    );
    assert!(
        end <= len,
        "range end (is {end}) should be <= len (is {len})"
    );
    (start, end)
}

// The cursor borrows only the directory, never an entire element page.
// Each entry is removed before its exact sub-slice is yielded, so mixed
// front/back consumption never yields overlapping regions.
struct PageCursor<'a, T, const PAGE: usize> {
    pages: vec_deque::Iter<'a, NonNull<T>>,
    front: usize,
    back: usize,
}

impl<T, const PAGE: usize> Clone for PageCursor<'_, T, PAGE> {
    fn clone(&self) -> Self {
        Self {
            pages: self.pages.clone(),
            front: self.front,
            back: self.back,
        }
    }
}

impl<T, const PAGE: usize> PageCursor<'_, T, PAGE> {
    fn next(&mut self) -> Option<*mut [T]> {
        let last = self.pages.len() == 1;
        let page = *self.pages.next()?;
        let end = if last { self.back } else { PAGE };
        let start = core::mem::replace(&mut self.front, 0);
        Some(PagedDeque::<T, PAGE>::page_slice(page, start, end))
    }

    fn next_back(&mut self) -> Option<*mut [T]> {
        let last = self.pages.len() == 1;
        let page = *self.pages.next_back()?;
        let start = if last { self.front } else { 0 };
        let end = core::mem::replace(&mut self.back, PAGE);
        Some(PagedDeque::<T, PAGE>::page_slice(page, start, end))
    }
}

/// Shared iterator over the initialized, non-empty parts of deque pages.
pub struct Pages<'a, T, const PAGE: usize = DEFAULT_PAGE> {
    cursor: PageCursor<'a, T, PAGE>,
    _borrow: PhantomData<&'a T>,
}

/// Mutable iterator over disjoint, initialized, non-empty page parts.
pub struct PagesMut<'a, T, const PAGE: usize = DEFAULT_PAGE> {
    cursor: PageCursor<'a, T, PAGE>,
    _borrow: PhantomData<&'a mut T>,
}

// SAFETY: invariants 2 and 4 and the cursor borrow keep allocations alive;
// only shared T references are yielded, requiring T: Sync for transfer.
unsafe impl<T: Sync, const PAGE: usize> Send for Pages<'_, T, PAGE> {}
// SAFETY: invariants 2 and 4 keep initialized allocations alive; sharing
// the iterator grants only shared access to T and requires T: Sync.
unsafe impl<T: Sync, const PAGE: usize> Sync for Pages<'_, T, PAGE> {}
// SAFETY: invariants 2 and 4 and the exclusive borrow grant ownership of
// access to disjoint T slots, which may be transferred when T: Send.
unsafe impl<T: Send, const PAGE: usize> Send for PagesMut<'_, T, PAGE> {}
// SAFETY: invariants 2 and 4 keep allocations live; shared access to this
// iterator cannot advance it or mutate T, and requires T: Sync.
unsafe impl<T: Sync, const PAGE: usize> Sync for PagesMut<'_, T, PAGE> {}

impl<T, const PAGE: usize> Clone for Pages<'_, T, PAGE> {
    fn clone(&self) -> Self {
        Self {
            cursor: self.cursor.clone(),
            _borrow: PhantomData,
        }
    }
}

macro_rules! page_iter {
    ($name:ident<$lt:lifetime, $t:ident>, $item:ty, $convert:expr) => {
        impl<$lt, $t, const PAGE: usize> Iterator for $name<$lt, $t, PAGE> {
            type Item = $item;
            fn next(&mut self) -> Option<Self::Item> {
                let slice = self.cursor.next()?;
                Some($convert(slice))
            }
            fn size_hint(&self) -> (usize, Option<usize>) {
                let len = self.cursor.pages.len();
                (len, Some(len))
            }
        }
        impl<$lt, $t, const PAGE: usize> DoubleEndedIterator for $name<$lt, $t, PAGE> {
            fn next_back(&mut self) -> Option<Self::Item> {
                let slice = self.cursor.next_back()?;
                Some($convert(slice))
            }
        }
        impl<$t, const PAGE: usize> ExactSizeIterator for $name<'_, $t, PAGE> {}
        impl<$t, const PAGE: usize> FusedIterator for $name<'_, $t, PAGE> {}
    };
}

page_iter!(Pages<'a, T>, &'a [T], |slice: *mut [T]| {
    // SAFETY: invariants 2 and 4 give precisely the initialized sub-slice;
    // the Pages borrow keeps it alive and forbids mutable deque access.
    unsafe { &*slice }
});
page_iter!(PagesMut<'a, T>, &'a mut [T], |slice: *mut [T]| {
    // SAFETY: invariants 2 and 4 give initialized slots; the exclusive
    // PagesMut borrow and cursor removal ensure each region is yielded once.
    unsafe { &mut *slice }
});

macro_rules! element_iter {
    ($(#[$doc:meta])* $name:ident<$lt:lifetime, $t:ident>, $item:ty, $pages:ident, $slice_iter:ty, $into:expr) => {
        $(#[$doc])*
        pub struct $name<$lt, $t, const PAGE: usize = DEFAULT_PAGE> {
            pages: $pages<$lt, $t, PAGE>,
            front: $slice_iter,
            back: $slice_iter,
            remaining: usize,
        }
        impl<$lt, $t, const PAGE: usize> Iterator for $name<$lt, $t, PAGE> {
            type Item = $item;
            fn next(&mut self) -> Option<Self::Item> {
                loop {
                    if let Some(value) = self.front.next() {
                        self.remaining -= 1;
                        return Some(value);
                    }
                    match self.pages.next() {
                        Some(page) => self.front = $into(page),
                        None => {
                            let value = self.back.next()?;
                            self.remaining -= 1;
                            return Some(value);
                        }
                    }
                }
            }
            fn size_hint(&self) -> (usize, Option<usize>) {
                (self.remaining, Some(self.remaining))
            }
        }
        impl<$lt, $t, const PAGE: usize> DoubleEndedIterator for $name<$lt, $t, PAGE> {
            fn next_back(&mut self) -> Option<Self::Item> {
                loop {
                    if let Some(value) = self.back.next_back() {
                        self.remaining -= 1;
                        return Some(value);
                    }
                    match self.pages.next_back() {
                        Some(page) => self.back = $into(page),
                        None => {
                            let value = self.front.next_back()?;
                            self.remaining -= 1;
                            return Some(value);
                        }
                    }
                }
            }
        }
        impl<$t, const PAGE: usize> ExactSizeIterator for $name<'_, $t, PAGE> {}
        impl<$t, const PAGE: usize> FusedIterator for $name<'_, $t, PAGE> {}
    };
}

element_iter!(
    /// Shared, double-ended element iterator, also used for ranges.
    Iter<'a, T>, &'a T, Pages, core::slice::Iter<'a, T>, |page: &'a [T]| page.iter()
);
element_iter!(
    /// Mutable, double-ended element iterator, also used for ranges.
    IterMut<'a, T>, &'a mut T, PagesMut, core::slice::IterMut<'a, T>, |page: &'a mut [T]| page.iter_mut()
);

impl<T, const PAGE: usize> Clone for Iter<'_, T, PAGE> {
    fn clone(&self) -> Self {
        Self {
            pages: self.pages.clone(),
            front: self.front.clone(),
            back: self.back.clone(),
            remaining: self.remaining,
        }
    }
}

impl<T: fmt::Debug, const PAGE: usize> fmt::Debug for Iter<'_, T, PAGE> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

/// Owning double-ended iterator. Dropping it drops all unconsumed elements
/// and releases their pages.
pub struct IntoIter<T, const PAGE: usize = DEFAULT_PAGE> {
    deque: PagedDeque<T, PAGE>,
}

impl<T, const PAGE: usize> Iterator for IntoIter<T, PAGE> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        self.deque.pop_front()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.deque.len, Some(self.deque.len))
    }
}
impl<T, const PAGE: usize> DoubleEndedIterator for IntoIter<T, PAGE> {
    fn next_back(&mut self) -> Option<T> {
        self.deque.pop_back()
    }
}
impl<T, const PAGE: usize> ExactSizeIterator for IntoIter<T, PAGE> {}
impl<T, const PAGE: usize> FusedIterator for IntoIter<T, PAGE> {}

impl<T, const PAGE: usize> Default for PagedDeque<T, PAGE> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone, const PAGE: usize> Clone for PagedDeque<T, PAGE> {
    fn clone(&self) -> Self {
        let mut result = Self::with_capacity(self.len);
        for value in self {
            result.push_back(value.clone());
        }
        result
    }
}

impl<T: fmt::Debug, const PAGE: usize> fmt::Debug for PagedDeque<T, PAGE> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// Element-wise equality, independent of page size and head offset.
impl<T: PartialEq<U>, U, const PAGE: usize, const PAGE2: usize> PartialEq<PagedDeque<U, PAGE2>>
    for PagedDeque<T, PAGE>
{
    fn eq(&self, other: &PagedDeque<U, PAGE2>) -> bool {
        self.len == other.len && self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}
impl<T: Eq, const PAGE: usize> Eq for PagedDeque<T, PAGE> {}

impl<T: PartialOrd, const PAGE: usize, const PAGE2: usize> PartialOrd<PagedDeque<T, PAGE2>>
    for PagedDeque<T, PAGE>
{
    fn partial_cmp(&self, other: &PagedDeque<T, PAGE2>) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}
impl<T: Ord, const PAGE: usize> Ord for PagedDeque<T, PAGE> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

/// Hashes the length, then each element, as PagedVec does.
impl<T: Hash, const PAGE: usize> Hash for PagedDeque<T, PAGE> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.len);
        for value in self {
            value.hash(state);
        }
    }
}

impl<T, const PAGE: usize> Index<usize> for PagedDeque<T, PAGE> {
    type Output = T;
    #[track_caller]
    fn index(&self, index: usize) -> &T {
        self.get(index).unwrap_or_else(|| {
            panic!(
                "index out of bounds: the len is {} but the index is {index}",
                self.len
            )
        })
    }
}
impl<T, const PAGE: usize> IndexMut<usize> for PagedDeque<T, PAGE> {
    #[track_caller]
    fn index_mut(&mut self, index: usize) -> &mut T {
        let len = self.len;
        self.get_mut(index).unwrap_or_else(|| {
            panic!("index out of bounds: the len is {len} but the index is {index}")
        })
    }
}

impl<T, const PAGE: usize> Extend<T> for PagedDeque<T, PAGE> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        self.reserve(iter.size_hint().0);
        for value in iter {
            self.push_back(value);
        }
    }
}
impl<'a, T: Copy + 'a, const PAGE: usize> Extend<&'a T> for PagedDeque<T, PAGE> {
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        self.extend(iter.into_iter().copied());
    }
}
impl<T, const PAGE: usize> FromIterator<T> for PagedDeque<T, PAGE> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut result = Self::new();
        result.extend(iter);
        result
    }
}
impl<T, const PAGE: usize> From<Vec<T>> for PagedDeque<T, PAGE> {
    fn from(values: Vec<T>) -> Self {
        values.into_iter().collect()
    }
}
impl<T, const N: usize, const PAGE: usize> From<[T; N]> for PagedDeque<T, PAGE> {
    fn from(values: [T; N]) -> Self {
        values.into_iter().collect()
    }
}
impl<'a, T, const PAGE: usize> IntoIterator for &'a PagedDeque<T, PAGE> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T, PAGE>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
impl<'a, T, const PAGE: usize> IntoIterator for &'a mut PagedDeque<T, PAGE> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T, PAGE>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}
impl<T, const PAGE: usize> IntoIterator for PagedDeque<T, PAGE> {
    type Item = T;
    type IntoIter = IntoIter<T, PAGE>;
    fn into_iter(self) -> Self::IntoIter {
        IntoIter { deque: self }
    }
}

#[cfg(feature = "serde")]
mod serde_impl {
    use super::*;
    use serde::de::{Deserialize, Deserializer, SeqAccess, Visitor};
    use serde::ser::{Serialize, Serializer};

    impl<T: Serialize, const PAGE: usize> Serialize for PagedDeque<T, PAGE> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            serializer.collect_seq(self.iter())
        }
    }

    impl<'de, T: Deserialize<'de>, const PAGE: usize> Deserialize<'de> for PagedDeque<T, PAGE> {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            struct SeqVisitor<T, const PAGE: usize>(PhantomData<T>);
            impl<'de, T: Deserialize<'de>, const PAGE: usize> Visitor<'de> for SeqVisitor<T, PAGE> {
                type Value = PagedDeque<T, PAGE>;
                fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    f.write_str("a sequence")
                }
                fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                    let budget = usize::try_from(1_u32 << 20).unwrap_or(usize::MAX);
                    let bytes = PAGE
                        .saturating_mul(core::mem::size_of::<T>())
                        .saturating_add(core::mem::size_of::<NonNull<T>>());
                    let max_hint = (budget / bytes).saturating_mul(PAGE);
                    let hint = seq.size_hint().unwrap_or(0).min(max_hint);
                    let mut result = PagedDeque::with_capacity(hint);
                    while let Some(value) = seq.next_element()? {
                        result.push_back(value);
                    }
                    Ok(result)
                }
            }
            deserializer.deserialize_seq(SeqVisitor::<T, PAGE>(PhantomData))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::rc::Rc;
    use alloc::vec;
    use core::cell::{Cell, RefCell};

    fn ends<const PAGE: usize>() {
        let mut deque = PagedDeque::<usize, PAGE>::new();
        let mut model = VecDeque::new();
        deque.check();
        for i in 0..35 {
            if i % 3 == 0 {
                deque.push_front(i);
                model.push_front(i);
            } else {
                deque.push_back(i);
                model.push_back(i);
            }
            deque.check();
            assert!(deque.iter().eq(model.iter()));
        }
        for i in 0..40 {
            if i % 3 == 0 {
                assert_eq!(deque.pop_back(), model.pop_back());
            } else {
                assert_eq!(deque.pop_front(), model.pop_front());
            }
            deque.check();
            assert!(deque.iter().eq(model.iter()));
        }
        // Empty with a nonzero head, including shrinking and reuse.
        deque.push_back(1);
        deque.check();
        assert_eq!(deque.pop_front(), Some(1));
        deque.check();
        deque.shrink_to_fit();
        deque.check();
        assert_eq!(deque.allocated_pages(), 0);
        deque.push_front(2);
        deque.check();
        assert_eq!(deque.pop_back(), Some(2));
        deque.check();
    }

    #[test]
    fn both_ends_across_small_pages() {
        ends::<1>();
        ends::<2>();
        ends::<3>();
        ends::<4>();
    }

    fn stable<const PAGE: usize>() {
        let mut deque: PagedDeque<usize, PAGE> = (0..19).collect();
        let mut addresses: Vec<_> = deque.iter().map(|x| (*x, ptr::from_ref(x))).collect();
        for i in 19..90 {
            match i % 6 {
                0 => deque.push_front(i),
                1 => deque.push_back(i),
                2 => {
                    deque.pop_front();
                }
                3 => {
                    deque.pop_back();
                }
                4 => deque.reserve(9),
                _ => deque.shrink_to_fit(),
            }
            deque.check();
            for value in &deque {
                if let Some((_, address)) = addresses.iter().find(|(id, _)| id == value) {
                    assert!(ptr::eq(value, *address));
                } else {
                    addresses.push((*value, ptr::from_ref(value)));
                }
            }
        }
    }

    #[test]
    fn survivor_addresses_never_change() {
        stable::<1>();
        stable::<2>();
        stable::<3>();
        stable::<4>();
    }

    #[test]
    fn pushes_preserve_old_raw_mutable_pointers() {
        let mut back: PagedDeque<u32, 4> = PagedDeque::new();
        back.push_back(1);
        back.check();
        let old = back.pages_mut().next().unwrap().as_mut_ptr();
        back.push_back(2); // same page
        back.push_back(3);
        back.reserve(12); // directory growth
        back.push_front(0); // another page
        // SAFETY: invariants 2 and 4 keep the original slot initialized and
        // allocated; pushes and reserve touched no reference to this slot.
        unsafe { old.write(99) };
        back.check();
        assert_eq!(back[1], 99);

        let mut front: PagedDeque<u32, 4> = PagedDeque::new();
        front.push_front(1);
        front.check();
        let old = front.get_mut(0).unwrap() as *mut u32;
        front.push_front(2); // same page
        front.push_front(3);
        front.push_back(4);
        // SAFETY: invariants 2 and 4 keep the old slot initialized and live;
        // all pushes wrote only other individual slots without retagging it.
        unsafe { old.write(88) };
        front.check();
        assert_eq!(front[2], 88);
    }

    #[test]
    fn old_pointers_survive_rotation_recycling_and_shrinking() {
        // Meaningful under Miri: the writes through `kept` would be flagged
        // if any step below formed a reference covering that slot.
        let mut deque: PagedDeque<u32, 2> = (0..6).collect();
        let kept = deque.get_mut(5).unwrap() as *mut u32;
        // Rotates two emptied front pages to the back as spares.
        for expected in 0..4 {
            assert_eq!(deque.pop_front(), Some(expected));
        }
        // SAFETY: element 5 is still in the deque at the same address, and
        // no reference to its slot is live.
        unsafe { kept.write(50) };
        deque.push_front(9); // recycles a spare at the front
        deque.push_back(7); // uses the other spare
        deque.swap(0, 1); // touches other slots only
        // SAFETY: as above.
        unsafe { kept.write(51) };
        deque.pop_back();
        deque.reserve(8);
        deque.shrink_to_fit(); // frees spares, shrinks the directory
        // SAFETY: as above.
        unsafe { kept.write(52) };
        deque.check();
        assert!(deque.iter().copied().eq([4, 9, 52]));
        assert!(ptr::eq(kept, &deque[2]));
    }

    #[test]
    fn partial_range_borrows_only_requested_slots() {
        let mut deque: PagedDeque<u32, 4> = [1, 2, 3, 4].into();
        let first = deque.get_mut(0).unwrap() as *mut u32;
        let last = deque.range_mut(2..3).next().unwrap();
        // SAFETY: invariants 2 and 4 keep slot 0 live; range_mut created a
        // reference only to slot 2, leaving the old slot-0 pointer valid.
        unsafe { first.write(10) };
        *last = 30;
        deque.check();
        assert!(deque.iter().copied().eq([10, 2, 30, 4]));

        let mut iter = deque.iter_mut();
        let a = iter.next().unwrap();
        let d = iter.next_back().unwrap();
        let b = iter.next().unwrap();
        let c = iter.next_back().unwrap();
        *a += 1;
        *b += 1;
        *c += 1;
        *d += 1;
        assert!(iter.next().is_none());
        deque.check();
    }

    #[test]
    fn mutable_page_slices_can_all_stay_live() {
        let mut deque: PagedDeque<u32, 3> = (0..11).collect();
        deque.pop_front();
        deque.check();
        let mut pages = deque.pages_mut();
        let first = pages.next().unwrap();
        let last = pages.next_back().unwrap();
        let middle: Vec<_> = pages.collect();
        for x in first
            .iter_mut()
            .chain(middle.into_iter().flatten())
            .chain(last)
        {
            *x += 10;
        }
        deque.check();
        assert!(deque.iter().copied().eq(11..21));
    }

    fn queue<const PAGE: usize>() {
        let mut deque: PagedDeque<usize, PAGE> = (0..2 * PAGE + 1).collect();
        deque.reserve(PAGE);
        deque.check();
        let pages = deque.allocated_pages();
        for i in 0..100 {
            deque.push_back(i);
            deque.check();
            deque.pop_front();
            deque.check();
            assert_eq!(deque.allocated_pages(), pages);
        }
        for i in 0..100 {
            deque.push_front(i);
            deque.check();
            deque.pop_back();
            deque.check();
            assert_eq!(deque.allocated_pages(), pages);
        }
    }

    #[test]
    fn steady_queues_reuse_pages() {
        queue::<1>();
        queue::<2>();
        queue::<3>();
        queue::<4>();
    }

    #[derive(Default)]
    struct Book {
        drops: RefCell<Vec<usize>>,
        clone_budget: Cell<Option<usize>>,
        drop_bomb: Cell<Option<usize>>,
    }

    impl Book {
        fn make(self: &Rc<Self>, value: usize) -> Item {
            let id = self.drops.borrow().len();
            self.drops.borrow_mut().push(0);
            Item {
                id,
                value,
                book: self.clone(),
            }
        }

        fn all_dropped(&self) {
            assert!(self.drops.borrow().iter().all(|&count| count == 1));
        }

        fn live<const PAGE: usize>(&self, deque: &PagedDeque<Item, PAGE>) {
            let mut expected = vec![1; self.drops.borrow().len()];
            for item in deque {
                assert_eq!(expected[item.id], 1, "duplicate live identity");
                expected[item.id] = 0;
            }
            assert_eq!(*self.drops.borrow(), expected);
            deque.check();
        }
    }

    struct Item {
        id: usize,
        value: usize,
        book: Rc<Book>,
    }

    impl Clone for Item {
        fn clone(&self) -> Self {
            if let Some(budget) = self.book.clone_budget.get() {
                assert!(budget > 0, "clone bomb");
                self.book.clone_budget.set(Some(budget - 1));
            }
            self.book.make(self.value)
        }
    }

    impl Drop for Item {
        fn drop(&mut self) {
            self.book.drops.borrow_mut()[self.id] += 1;
            if self.book.drop_bomb.get() == Some(self.id) {
                self.book.drop_bomb.set(None);
                panic!("drop bomb");
            }
        }
    }

    fn ownership<const PAGE: usize>() {
        let book = Rc::new(Book::default());
        let mut deque: PagedDeque<_, PAGE> = (0..12).map(|i| book.make(i)).collect();
        book.live(&deque);
        deque.insert(2, book.make(20));
        book.live(&deque);
        drop(deque.remove(3));
        book.live(&deque);
        drop(deque.swap_remove_front(2));
        book.live(&deque);
        drop(deque.swap_remove_back(4));
        book.live(&deque);
        deque.truncate(8);
        book.live(&deque);
        deque.truncate_front(6);
        book.live(&deque);
        deque.retain(|x| x.value % 2 == 0);
        book.live(&deque);
        deque.retain_mut(|x| {
            x.value += 1;
            x.value % 3 != 0
        });
        book.live(&deque);
        let mut other: PagedDeque<_, PAGE> = (30..36).map(|i| book.make(i)).collect();
        other.check();
        let pages = other.allocated_pages();
        deque.append(&mut other);
        assert_eq!(other.allocated_pages(), pages);
        assert!(other.is_empty());
        other.check();
        book.live(&deque);
        let mut iter = deque.into_iter();
        drop(iter.next());
        iter.deque.check();
        drop(iter.next_back());
        iter.deque.check();
        drop(iter);
        book.all_dropped();
        let mut deque: PagedDeque<_, PAGE> = (0..7).map(|i| book.make(i)).collect();
        book.live(&deque);
        deque.clear();
        book.live(&deque);
        deque.push_front(book.make(99));
        book.live(&deque);
        drop(deque);
        book.all_dropped();
    }

    #[test]
    fn every_value_is_dropped_exactly_once() {
        ownership::<1>();
        ownership::<2>();
        ownership::<3>();
        ownership::<4>();
    }

    #[test]
    fn zero_sized_and_aligned_elements() {
        let mut deque = PagedDeque::<(), 4>::new();
        for _ in 0..10 {
            deque.push_front(());
            deque.check();
            deque.push_back(());
            deque.check();
        }
        deque.insert(5, ());
        deque.check();
        assert_eq!(deque.remove(3), Some(()));
        deque.check();
        assert_eq!(deque.pages_mut().flatten().count(), 20);
        deque.check();
        deque.truncate_front(9);
        deque.check();
        deque.shrink_to_fit();
        deque.check();
        assert_eq!(deque.into_iter().rev().count(), 9);

        #[repr(align(128))]
        struct Aligned(u8);
        let mut deque: PagedDeque<Aligned, 3> = PagedDeque::new();
        deque.push_back(Aligned(1));
        deque.check();
        deque.push_front(Aligned(2));
        deque.check();
        assert_eq!(deque[0].0 + deque[1].0, 3);

        // A ZST with drop glue must still be dropped once per logical slot.
        static DROPS: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);
        struct Zst;
        impl Drop for Zst {
            fn drop(&mut self) {
                DROPS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            }
        }
        let mut deque = PagedDeque::<Zst, 3>::new();
        for _ in 0..11 {
            deque.push_front(Zst);
            deque.check();
        }
        drop(deque.pop_back());
        deque.check();
        drop(deque);
        assert_eq!(DROPS.load(core::sync::atomic::Ordering::Relaxed), 11);
    }

    #[test]
    fn send_sync_bounds_match_standard_iterators() {
        fn send<T: Send>() {}
        fn sync<T: Sync>() {}
        send::<PagedDeque<Cell<u32>>>();
        sync::<PagedDeque<u32>>();
        send::<Pages<'_, u32>>();
        sync::<Pages<'_, u32>>();
        send::<PagesMut<'_, Cell<u32>>>();
        sync::<PagesMut<'_, u32>>();
        send::<Iter<'_, u32>>();
        sync::<Iter<'_, u32>>();
        send::<IterMut<'_, Cell<u32>>>();
        sync::<IterMut<'_, u32>>();
        send::<IntoIter<Cell<u32>>>();
        sync::<IntoIter<u32>>();
    }

    #[cfg(feature = "std")]
    mod unwind {
        use super::*;
        use std::panic::{AssertUnwindSafe, catch_unwind};

        fn panic_cases<const PAGE: usize>() {
            for bomb in 0..8 {
                for mutable in [false, true] {
                    let book = Rc::new(Book::default());
                    let mut deque: PagedDeque<_, PAGE> = (0..8).map(|i| book.make(i)).collect();
                    let mut calls = 0;
                    assert!(
                        catch_unwind(AssertUnwindSafe(|| {
                            if mutable {
                                deque.retain_mut(|x| {
                                    let call = calls;
                                    calls += 1;
                                    x.value += 100;
                                    assert_ne!(call, bomb, "predicate bomb");
                                    x.value % 2 == 0
                                });
                            } else {
                                deque.retain(|x| {
                                    let call = calls;
                                    calls += 1;
                                    assert_ne!(call, bomb, "predicate bomb");
                                    x.value % 2 == 0
                                });
                            }
                        }))
                        .is_err()
                    );
                    book.live(&deque);
                    let expected: Vec<_> = (0..8)
                        .filter(|&i| i >= bomb || i % 2 == 0)
                        .map(|i| i + if mutable && i <= bomb { 100 } else { 0 })
                        .collect();
                    assert!(deque.iter().map(|x| x.value).eq(expected));
                    drop(deque);
                    book.all_dropped();
                }

                let book = Rc::new(Book::default());
                let mut deque: PagedDeque<_, PAGE> = (0..3).map(|i| book.make(i)).collect();
                let mut calls = 0;
                assert!(
                    catch_unwind(AssertUnwindSafe(|| {
                        deque.resize_with(12, || {
                            let call = calls;
                            calls += 1;
                            assert_ne!(call, bomb, "factory bomb");
                            book.make(100 + call)
                        });
                    }))
                    .is_err()
                );
                assert_eq!(deque.len(), 3 + bomb);
                book.live(&deque);
                drop(deque);
                book.all_dropped();

                let book = Rc::new(Book::default());
                let deque: PagedDeque<_, PAGE> = (0..8).map(|i| book.make(i)).collect();
                book.clone_budget.set(Some(bomb));
                assert!(catch_unwind(AssertUnwindSafe(|| deque.clone())).is_err());
                book.live(&deque);
                drop(deque);
                book.all_dropped();

                let book = Rc::new(Book::default());
                let mut deque = PagedDeque::<_, PAGE>::new();
                let mut calls = 0;
                let iter = core::iter::from_fn(|| {
                    let call = calls;
                    calls += 1;
                    assert_ne!(call, bomb, "iterator bomb");
                    Some(book.make(call))
                });
                assert!(catch_unwind(AssertUnwindSafe(|| deque.extend(iter))).is_err());
                assert_eq!(deque.len(), bomb);
                book.live(&deque);
                drop(deque);
                book.all_dropped();
            }
        }

        #[test]
        fn user_code_panics_preserve_ownership() {
            panic_cases::<1>();
            panic_cases::<2>();
            panic_cases::<3>();
            panic_cases::<4>();
        }

        fn dropping_panics<const PAGE: usize>() {
            for operation in 0..5 {
                for bomb in 0..8 {
                    let book = Rc::new(Book::default());
                    let mut deque: PagedDeque<_, PAGE> = (0..8).map(|i| book.make(i)).collect();
                    // A partial first page, a partial last page, and spares.
                    deque.push_front(book.make(8));
                    deque.check();
                    deque.reserve(2 * PAGE);
                    deque.check();
                    book.drop_bomb.set(Some(bomb));
                    if operation == 4 {
                        assert!(catch_unwind(AssertUnwindSafe(|| drop(deque))).is_err());
                    } else {
                        assert!(
                            catch_unwind(AssertUnwindSafe(|| match operation {
                                0 => deque.clear(),
                                1 => deque.truncate(0),
                                2 => deque.truncate_front(0),
                                _ => deque.retain(|_| false),
                            }))
                            .is_err()
                        );
                        book.live(&deque);
                        deque.push_back(book.make(99));
                        book.live(&deque);
                        drop(deque);
                    }
                    book.all_dropped();
                }
            }
        }

        #[test]
        fn destructor_panics_leave_valid_slots_and_free_every_page() {
            dropping_panics::<1>();
            dropping_panics::<2>();
            dropping_panics::<3>();
            dropping_panics::<4>();
        }

        #[test]
        fn invalid_bounds_and_capacity_overflow_do_not_mutate() {
            let mut deque: PagedDeque<u32, 3> = [1, 2, 3].into();
            deque.reserve(12);
            for index in [3, 4, usize::MAX] {
                assert_eq!(deque.get(index), None);
                assert_eq!(deque.get_mut(index), None);
                assert_eq!(deque.remove(index), None);
                assert_eq!(deque.swap_remove_back(index), None);
                assert_eq!(deque.swap_remove_front(index), None);
                assert!(catch_unwind(|| deque[index]).is_err());
                assert!(catch_unwind(AssertUnwindSafe(|| deque[index] = 0)).is_err());
                assert!(catch_unwind(AssertUnwindSafe(|| deque.swap(0, index))).is_err());
                deque.check();
            }
            assert!(catch_unwind(AssertUnwindSafe(|| deque.insert(4, 0))).is_err());
            for bounds in [
                (Bound::Included(2), Bound::Excluded(1)),
                (Bound::Included(0), Bound::Excluded(4)),
                (Bound::Excluded(usize::MAX), Bound::Unbounded),
                (Bound::Unbounded, Bound::Included(usize::MAX)),
            ] {
                assert!(catch_unwind(|| deque.range(bounds)).is_err());
                assert!(
                    catch_unwind(AssertUnwindSafe(|| {
                        deque.range_mut(bounds);
                    }))
                    .is_err()
                );
                deque.check();
            }
            assert!(deque.try_reserve(usize::MAX).is_err());
            deque.check();
            assert!(catch_unwind(AssertUnwindSafe(|| deque.reserve(usize::MAX))).is_err());
            deque.check();
            assert!(deque.iter().copied().eq([1, 2, 3]));
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_sequence_round_trip() {
        let mut deque: PagedDeque<u32, 3> = (0..8).collect();
        deque.pop_front();
        deque.check();
        deque.push_front(99);
        deque.check();
        deque.reserve(12);
        deque.check();
        let json = serde_json::to_string(&deque).unwrap();
        assert_eq!(json, "[99,1,2,3,4,5,6,7]");
        let mut back: PagedDeque<u32, 3> = serde_json::from_str(&json).unwrap();
        back.check();
        assert_eq!(back, deque);
        back.push_front(0);
        back.check();
        back.push_back(8);
        back.check();
        assert_eq!(back.len(), 10);
    }
}
