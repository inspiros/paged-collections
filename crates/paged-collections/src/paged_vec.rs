//! [`PagedVec`]: a growable vector stored in fixed-size pages.
//!
//! Elements live in pages of `PAGE` elements. A page is allocated once, with
//! room for at least `PAGE` elements, and never reallocated, so growing the
//! vector never moves or copies existing elements:
//!
//! - an element keeps its address while it stays at the same index, through
//!   any number of [`push`](PagedVec::push)es, [`extend`](Extend::extend)s
//!   and [`reserve`](PagedVec::reserve)s;
//! - growth costs one page allocation every `PAGE` pushes instead of an
//!   O(n) copy of the elements. The page directory (one `Vec` header per
//!   page) still grows like a `Vec`, but it is `PAGE` times shorter, so the
//!   occasional copy and the momentary doubling are small.
//!
//! Operations that change which element sits at an index ([`insert`],
//! [`remove`], [`swap`], [`swap_remove`], [`retain`]) move elements between
//! addresses, as they do in a `Vec`.
//!
//! The price is that the elements are not one contiguous slice: there is no
//! `as_slice` or `Deref<Target = [T]>`. Use [`pages`](PagedVec::pages) for
//! slice access one page at a time, or
//! [`with_contiguous_mut`](PagedVec::with_contiguous_mut) to run a slice
//! algorithm on all elements at the cost of a temporary buffer.
//!
//! Sorting goes through that buffer, so it uses the standard library's slice
//! sorts. Searching needs no buffer: a binary search over the pages finds the
//! page, then the slice search runs inside it.
//!
//! | Operation | Cost |
//! |---|---|
//! | [`pop`](PagedVec::pop), [`get`](PagedVec::get), indexing | O(1) |
//! | [`push`](PagedVec::push) | amortized O(1): a page allocation every `PAGE` pushes, rare directory growth |
//! | [`insert`], [`remove`] | O(len − index): elements shift across pages |
//! | [`truncate`](PagedVec::truncate), [`clear`](PagedVec::clear) | O(removed) drops; pages are kept |
//! | [`binary_search`](PagedVec::binary_search), [`partition_point`](PagedVec::partition_point) | O(log n), no allocation |
//! | [`sort`](PagedVec::sort) and variants | the slice sort's cost, plus moving all elements out and back and a temporary buffer of `len` elements |
//!
//! The name follows Rust's `Vec`; other languages may know it as a list.
//!
//! ```
//! use paged_collections::paged_vec::PagedVec;
//!
//! let mut v: PagedVec<u64, 4> = PagedVec::new();
//! v.push(10);
//! let first: *const u64 = &v[0];
//! v.extend(0..1000);
//! // The first page was never reallocated.
//! assert!(core::ptr::eq(first, &v[0]));
//! assert_eq!(v.len(), 1001);
//! assert_eq!(v.pages().next(), Some(&[10, 0, 1, 2][..]));
//! ```
//!
//! [`insert`]: PagedVec::insert
//! [`remove`]: PagedVec::remove
//! [`swap`]: PagedVec::swap
//! [`swap_remove`]: PagedVec::swap_remove
//! [`retain`]: PagedVec::retain

use alloc::collections::TryReserveError;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::iter::FusedIterator;
use core::ops::{Index, IndexMut};

use crate::page::{DEFAULT_PAGE, split};

/// A growable vector whose elements never move while the vector grows.
///
/// `PAGE` is the number of elements per page; it must be non-zero. See the
/// [module docs](crate::paged_vec).
///
/// Invariants, for `n = len`:
///
/// - page `i` holds elements `i * PAGE ..` and has length
///   `min(PAGE, n.saturating_sub(i * PAGE))`, so pages after the last used
///   one are empty spares;
/// - every page has capacity at least `PAGE`. Every page is created by
///   `new_page`/`try_new_page` and never grows past `PAGE` elements, so it
///   never reallocates (`Vec` guarantees that `push` and `insert` do not
///   reallocate when the capacity suffices).
pub struct PagedVec<T, const PAGE: usize = DEFAULT_PAGE> {
    pages: Vec<Vec<T>>,
    len: usize,
}

impl<T, const PAGE: usize> PagedVec<T, PAGE> {
    /// Number of elements per page.
    pub const PAGE: usize = PAGE;

    /// An empty vector. Does not allocate.
    #[inline]
    pub const fn new() -> Self {
        const { assert!(PAGE > 0, "page size must be non-zero") };
        Self {
            pages: Vec::new(),
            len: 0,
        }
    }

    /// An empty vector with pages allocated for `capacity` elements.
    pub fn with_capacity(capacity: usize) -> Self {
        let mut v = Self::new();
        v.reserve(capacity);
        v
    }

    #[inline]
    fn new_page() -> Vec<T> {
        Vec::with_capacity(PAGE)
    }

    fn try_new_page() -> Result<Vec<T>, TryReserveError> {
        let mut page = Vec::new();
        page.try_reserve_exact(PAGE)?;
        Ok(page)
    }

    /// Number of pages that hold at least one element.
    #[inline]
    fn used_pages(&self) -> usize {
        self.len.div_ceil(PAGE)
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

    /// Number of elements the allocated pages hold.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.pages.len().saturating_mul(PAGE)
    }

    /// Number of allocated pages, including empty spares.
    #[inline]
    pub fn allocated_pages(&self) -> usize {
        self.pages.len()
    }

    /// Allocates pages so that `additional` more elements fit without
    /// allocating. Pages are allocated now, not on first use.
    ///
    /// # Panics
    ///
    /// Panics if the length overflows `usize` or an allocation size
    /// overflows `isize::MAX` bytes.
    pub fn reserve(&mut self, additional: usize) {
        let needed = self
            .len
            .checked_add(additional)
            .expect("capacity overflow")
            .div_ceil(PAGE);
        if let Some(missing) = needed.checked_sub(self.pages.len()) {
            self.pages.reserve(missing);
            while self.pages.len() < needed {
                self.pages.push(Self::new_page());
            }
        }
    }

    /// Like [`reserve`](PagedVec::reserve), but returns an error instead of
    /// panicking or aborting. On error, pages allocated so far are kept.
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        let Some(total) = self.len.checked_add(additional) else {
            return Err(capacity_overflow());
        };
        let needed = total.div_ceil(PAGE);
        if let Some(missing) = needed.checked_sub(self.pages.len()) {
            self.pages.try_reserve(missing)?;
            while self.pages.len() < needed {
                self.pages.push(Self::try_new_page()?);
            }
        }
        Ok(())
    }

    /// Frees the spare pages after the last used one. Never moves elements:
    /// the last used page keeps its full capacity.
    pub fn shrink_to_fit(&mut self) {
        self.pages.truncate(self.used_pages());
        self.pages.shrink_to_fit();
    }

    /// Appends an element.
    #[inline]
    pub fn push(&mut self, value: T) {
        let _ = self.push_mut(value);
    }

    /// Appends an element and returns a reference to it.
    ///
    /// # Panics
    ///
    /// Panics if the length would overflow `usize` (possible only for
    /// zero-sized `T`).
    pub fn push_mut(&mut self, value: T) -> &mut T {
        let new_len = self.len.checked_add(1).expect("PagedVec length overflow");
        let (p, _) = split::<PAGE>(self.len);
        if p == self.pages.len() {
            self.pages.push(Self::new_page());
        }
        self.len = new_len;
        let page = &mut self.pages[p];
        debug_assert!(page.len() < PAGE && page.capacity() >= PAGE);
        page.push(value);
        match page.last_mut() {
            Some(last) => last,
            None => unreachable!(),
        }
    }

    /// Removes and returns the last element.
    pub fn pop(&mut self) -> Option<T> {
        let last = self.len.checked_sub(1)?;
        let value = self.pages[last / PAGE].pop();
        debug_assert!(value.is_some());
        self.len = last;
        value
    }

    /// The element at `index`.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&T> {
        let (p, offset) = split::<PAGE>(index);
        self.pages.get(p)?.get(offset)
    }

    /// The element at `index`, mutably.
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        let (p, offset) = split::<PAGE>(index);
        self.pages.get_mut(p)?.get_mut(offset)
    }

    /// The first element.
    #[inline]
    pub fn first(&self) -> Option<&T> {
        self.get(0)
    }

    /// The first element, mutably.
    #[inline]
    pub fn first_mut(&mut self) -> Option<&mut T> {
        self.get_mut(0)
    }

    /// The last element.
    #[inline]
    pub fn last(&self) -> Option<&T> {
        self.get(self.len.checked_sub(1)?)
    }

    /// The last element, mutably.
    #[inline]
    pub fn last_mut(&mut self) -> Option<&mut T> {
        self.get_mut(self.len.checked_sub(1)?)
    }

    /// Swaps the elements at `a` and `b`.
    ///
    /// # Panics
    ///
    /// Panics if `a` or `b` is out of bounds.
    pub fn swap(&mut self, a: usize, b: usize) {
        assert!(
            a < self.len && b < self.len,
            "swap indices (are {a} and {b}) should be < len (is {})",
            self.len
        );
        let (pa, oa) = split::<PAGE>(a);
        let (pb, ob) = split::<PAGE>(b);
        match pa.cmp(&pb) {
            Ordering::Equal => self.pages[pa].swap(oa, ob),
            Ordering::Less => {
                let (low, high) = self.pages.split_at_mut(pb);
                core::mem::swap(&mut low[pa][oa], &mut high[0][ob]);
            }
            Ordering::Greater => {
                let (low, high) = self.pages.split_at_mut(pa);
                core::mem::swap(&mut high[0][oa], &mut low[pb][ob]);
            }
        }
    }

    /// Inserts `value` at `index`, shifting the elements after it up by one.
    /// Costs O(len − index), plus a page allocation every `PAGE` elements.
    ///
    /// # Panics
    ///
    /// Panics if `index > len`.
    pub fn insert(&mut self, index: usize, value: T) {
        assert!(
            index <= self.len,
            "insertion index (is {index}) should be <= len (is {})",
            self.len
        );
        // Allocate first, so nothing below can fail.
        let new_len = self.len.checked_add(1).expect("PagedVec length overflow");
        let (last_page, _) = split::<PAGE>(self.len);
        if last_page == self.pages.len() {
            self.pages.push(Self::new_page());
        }
        let (mut p, mut offset) = split::<PAGE>(index);
        let mut carry = value;
        // Only moves from here on: no user code runs, so nothing can panic.
        loop {
            let page = &mut self.pages[p];
            if page.len() < PAGE {
                page.insert(offset, carry);
                break;
            }
            // A full page passes its last element to the front of the next.
            let Some(out) = page.pop() else {
                unreachable!()
            };
            page.insert(offset, carry);
            carry = out;
            p += 1;
            offset = 0;
        }
        self.len = new_len;
    }

    /// Removes and returns the element at `index`, shifting the elements
    /// after it down by one. Costs O(len − index).
    ///
    /// # Panics
    ///
    /// Panics if `index >= len`.
    pub fn remove(&mut self, index: usize) -> T {
        assert!(
            index < self.len,
            "removal index (is {index}) should be < len (is {})",
            self.len
        );
        let used = self.used_pages();
        let (p, offset) = split::<PAGE>(index);
        let removed = self.pages[p].remove(offset);
        // Each later page passes its first element to the end of the one
        // before, which has just dropped to `PAGE - 1` elements.
        for q in p + 1..used {
            let first = self.pages[q].remove(0);
            self.pages[q - 1].push(first);
        }
        self.len -= 1;
        removed
    }

    /// Removes and returns the element at `index`, replacing it with the
    /// last element. O(1).
    ///
    /// # Panics
    ///
    /// Panics if `index >= len`.
    pub fn swap_remove(&mut self, index: usize) -> T {
        assert!(
            index < self.len,
            "swap_remove index (is {index}) should be < len (is {})",
            self.len
        );
        let last = self.len - 1;
        if index != last {
            self.swap(index, last);
        }
        match self.pop() {
            Some(value) => value,
            None => unreachable!(),
        }
    }

    /// Shortens the vector to `len` elements, dropping the rest one at a
    /// time from the back. Does nothing if `len >= self.len()`. Pages are
    /// kept.
    ///
    /// If a drop panics, the panicking element is gone and the elements
    /// before it stay in the vector.
    pub fn truncate(&mut self, len: usize) {
        if core::mem::needs_drop::<T>() {
            // Each element leaves the vector before it drops, so a panicking
            // drop leaves the vector consistent.
            while self.len > len {
                drop(self.pop());
            }
            return;
        }
        // Nothing to drop, so nothing can panic: cut whole pages at once.
        while self.len > len {
            let (p, _) = split::<PAGE>(self.len - 1);
            let start = p * PAGE;
            let new_len = len.max(start);
            self.len = new_len;
            self.pages[p].truncate(new_len - start);
        }
    }

    /// Drops every element. Pages are kept; see
    /// [`shrink_to_fit`](PagedVec::shrink_to_fit).
    pub fn clear(&mut self) {
        self.truncate(0);
    }

    /// Resizes to `new_len`, filling new slots with `f()` or truncating.
    /// If `f` panics, the elements pushed so far stay.
    pub fn resize_with(&mut self, new_len: usize, mut f: impl FnMut() -> T) {
        if new_len <= self.len {
            self.truncate(new_len);
            return;
        }
        self.reserve(new_len - self.len);
        while self.len < new_len {
            self.push(f());
        }
    }

    /// Resizes to `new_len`, filling new slots with clones of `value` or
    /// truncating.
    pub fn resize(&mut self, new_len: usize, value: T)
    where
        T: Clone,
    {
        self.resize_with(new_len, || value.clone());
    }

    /// Keeps only the elements for which `keep` returns `true`, in order.
    ///
    /// Rejected elements are dropped together at the end, not as they are
    /// visited. If `keep` panics, the elements already rejected are dropped
    /// and the rest (kept, the one being tested, and the unvisited) stay in
    /// order. If dropping a rejected element panics, the rejected elements
    /// not yet dropped stay at the end of the vector.
    pub fn retain(&mut self, mut keep: impl FnMut(&T) -> bool) {
        self.retain_mut(|x| keep(x));
    }

    /// Like [`retain`](PagedVec::retain), with mutable access to each
    /// element.
    pub fn retain_mut(&mut self, mut keep: impl FnMut(&mut T) -> bool) {
        /// Kept elements are compacted into `..write`; rejected ones collect
        /// in `write..read`; `read..` is unvisited.
        struct Guard<'a, T, const PAGE: usize> {
            vec: &'a mut PagedVec<T, PAGE>,
            read: usize,
            write: usize,
        }

        impl<T, const PAGE: usize> Drop for Guard<'_, T, PAGE> {
            fn drop(&mut self) {
                // Move the unvisited tail down past the kept prefix, then
                // drop the rejected elements, which are now at the end. On
                // the normal path `read == len` and this only truncates.
                let mut write = self.write;
                for read in self.read..self.vec.len {
                    self.vec.swap(write, read);
                    write += 1;
                }
                self.vec.truncate(write);
            }
        }

        let len = self.len;
        let mut guard = Guard {
            vec: self,
            read: 0,
            write: 0,
        };
        while guard.read < len {
            let read = guard.read;
            let Some(x) = guard.vec.get_mut(read) else {
                unreachable!()
            };
            if keep(x) {
                if guard.write != read {
                    guard.vec.swap(guard.write, read);
                }
                guard.write += 1;
            }
            guard.read += 1;
        }
    }

    /// Runs `f` on all elements as one contiguous mutable slice, then puts
    /// them back in their new order.
    ///
    /// The elements are moved into a temporary buffer of `len` elements and
    /// back into the same pages, so the pages keep their addresses (as with
    /// any reordering, an element's address follows its index, not the
    /// element). The buffer costs `len * size_of::<T>()` bytes for the
    /// duration of the call.
    ///
    /// If `f` panics, the elements are put back in whatever order `f` left
    /// them, and the panic continues.
    ///
    /// ```
    /// use paged_collections::paged_vec::PagedVec;
    ///
    /// let mut v: PagedVec<u32, 2> = [3, 1, 2, 5, 4].into();
    /// v.with_contiguous_mut(|s| s.rotate_left(2));
    /// assert!(v.iter().copied().eq([2, 5, 4, 3, 1]));
    /// ```
    pub fn with_contiguous_mut<R>(&mut self, f: impl FnOnce(&mut [T]) -> R) -> R {
        /// Moves the buffer back into the pages, `PAGE` elements each, on
        /// return and on unwind.
        struct Refill<'a, T, const PAGE: usize> {
            pages: &'a mut [Vec<T>],
            buf: Vec<T>,
        }

        impl<T, const PAGE: usize> Drop for Refill<'_, T, PAGE> {
            fn drop(&mut self) {
                // One `push` at a time: `Vec` guarantees that `push` does
                // not reallocate when the capacity suffices (bulk methods
                // make no such promise), each page kept its capacity of at
                // least `PAGE` (`Vec` never shrinks on its own), and each
                // gets at most `PAGE` elements. So nothing here allocates or
                // panics, and no page moves.
                let mut rest = self.buf.drain(..);
                for page in self.pages.iter_mut() {
                    debug_assert!(page.is_empty() && page.capacity() >= PAGE);
                    for value in rest.by_ref().take(PAGE) {
                        page.push(value);
                    }
                }
                debug_assert_eq!(rest.len(), 0);
            }
        }

        // Allocate before moving anything, so a failed allocation changes
        // nothing. The guard exists before the first move; gathering only
        // pushes into `buf`, which has room for every element, so it cannot
        // panic either.
        let used = self.used_pages();
        let mut refill = Refill::<T, PAGE> {
            pages: &mut self.pages[..used],
            buf: Vec::with_capacity(self.len),
        };
        for page in refill.pages.iter_mut() {
            for value in page.drain(..) {
                refill.buf.push(value);
            }
        }
        f(&mut refill.buf)
    }

    /// Sorts stably, with the standard library's stable sort. See
    /// [`with_contiguous_mut`](PagedVec::with_contiguous_mut) for the
    /// temporary buffer and panic behaviour: if a comparison panics, the
    /// vector holds the same elements in an unspecified order.
    pub fn sort(&mut self)
    where
        T: Ord,
    {
        self.with_contiguous_mut(|s| s.sort());
    }

    /// Sorts stably with a comparator. See [`sort`](PagedVec::sort).
    pub fn sort_by(&mut self, compare: impl FnMut(&T, &T) -> Ordering) {
        self.with_contiguous_mut(|s| s.sort_by(compare));
    }

    /// Sorts stably by a key. See [`sort`](PagedVec::sort).
    pub fn sort_by_key<K: Ord>(&mut self, key: impl FnMut(&T) -> K) {
        self.with_contiguous_mut(|s| s.sort_by_key(key));
    }

    /// Sorts stably by a key computed at most once per element. See
    /// [`sort`](PagedVec::sort).
    pub fn sort_by_cached_key<K: Ord>(&mut self, key: impl FnMut(&T) -> K) {
        self.with_contiguous_mut(|s| s.sort_by_cached_key(key));
    }

    /// Sorts without preserving the order of equal elements, with the
    /// standard library's unstable sort. See [`sort`](PagedVec::sort).
    pub fn sort_unstable(&mut self)
    where
        T: Ord,
    {
        self.with_contiguous_mut(|s| s.sort_unstable());
    }

    /// Unstable sort with a comparator. See [`sort`](PagedVec::sort).
    pub fn sort_unstable_by(&mut self, compare: impl FnMut(&T, &T) -> Ordering) {
        self.with_contiguous_mut(|s| s.sort_unstable_by(compare));
    }

    /// Unstable sort by a key. See [`sort`](PagedVec::sort).
    pub fn sort_unstable_by_key<K: Ord>(&mut self, key: impl FnMut(&T) -> K) {
        self.with_contiguous_mut(|s| s.sort_unstable_by_key(key));
    }

    /// Reverses the order of the elements in place, without a buffer.
    pub fn reverse(&mut self) {
        let n = self.len;
        for i in 0..n / 2 {
            self.swap(i, n - 1 - i);
        }
    }

    /// Whether some element equals `x`. O(len).
    pub fn contains(&self, x: &T) -> bool
    where
        T: PartialEq,
    {
        self.pages().any(|page| page.contains(x))
    }

    /// Binary search in a sorted vector, like
    /// [`slice::binary_search`]: `Ok` with the index of a matching element
    /// (any one, if several match), or `Err` with the index where `x` would
    /// be inserted to keep the order.
    pub fn binary_search(&self, x: &T) -> Result<usize, usize>
    where
        T: Ord,
    {
        self.binary_search_by(|y| y.cmp(x))
    }

    /// Binary search with a comparator that returns the order of the
    /// element relative to the target, like [`slice::binary_search_by`].
    /// The result is unspecified, but not undefined, if the vector is not
    /// sorted consistently with `f`.
    pub fn binary_search_by<'a>(
        &'a self,
        mut f: impl FnMut(&'a T) -> Ordering,
    ) -> Result<usize, usize> {
        let pages: &'a [Vec<T>] = &self.pages[..self.used_pages()];
        // The first page whose last element is not less than the target is
        // the only one that can hold the first match or the insertion point.
        // Searched by hand: `partition_point`'s closure gets a reference
        // too short-lived to pass to `f`.
        let (mut lo, mut hi) = (0, pages.len());
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let page: &'a Vec<T> = &pages[mid];
            if page.last().is_some_and(|y| f(y) == Ordering::Less) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let p = lo;
        let Some(page) = pages.get(p) else {
            return Err(self.len);
        };
        let base = p * PAGE;
        match page.binary_search_by(f) {
            Ok(i) => Ok(base + i),
            Err(i) => Err(base + i),
        }
    }

    /// Binary search by a key, like [`slice::binary_search_by_key`].
    pub fn binary_search_by_key<'a, K: Ord>(
        &'a self,
        key: &K,
        mut f: impl FnMut(&'a T) -> K,
    ) -> Result<usize, usize> {
        self.binary_search_by(|y| f(y).cmp(key))
    }

    /// The index of the first element for which `pred` is false, given that
    /// `pred` is true for a prefix and false for the rest, like
    /// [`slice::partition_point`].
    pub fn partition_point(&self, mut pred: impl FnMut(&T) -> bool) -> usize {
        let pages = &self.pages[..self.used_pages()];
        let p = pages.partition_point(|page| page.last().is_some_and(&mut pred));
        match pages.get(p) {
            Some(page) => p * PAGE + page.partition_point(pred),
            None => self.len,
        }
    }

    /// Iterates over the elements.
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            pages: self.pages[..self.used_pages()].iter(),
            front: [].iter(),
            back: [].iter(),
            remaining: self.len,
        }
    }

    /// Iterates over the elements mutably.
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        let used = self.used_pages();
        IterMut {
            pages: self.pages[..used].iter_mut(),
            front: [].iter_mut(),
            back: [].iter_mut(),
            remaining: self.len,
        }
    }

    /// The non-empty pages as slices, in order. Every page but the last is
    /// full.
    pub fn pages(
        &self,
    ) -> impl ExactSizeIterator<Item = &[T]> + DoubleEndedIterator + FusedIterator + Clone {
        self.pages[..self.used_pages()].iter().map(Vec::as_slice)
    }

    /// The non-empty pages as mutable slices, in order.
    pub fn pages_mut(
        &mut self,
    ) -> impl ExactSizeIterator<Item = &mut [T]> + DoubleEndedIterator + FusedIterator {
        let used = self.used_pages();
        self.pages[..used].iter_mut().map(Vec::as_mut_slice)
    }
}

fn capacity_overflow() -> TryReserveError {
    // `try_reserve` has no public constructor for its error; a request over
    // `isize::MAX` bytes always fails with the capacity-overflow kind.
    match Vec::<u8>::new().try_reserve(usize::MAX) {
        Err(e) => e,
        Ok(()) => unreachable!(),
    }
}

impl<T, const PAGE: usize> Default for PagedVec<T, PAGE> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Clones page by page. Each new page gets the full capacity `PAGE`, so the
/// clone keeps the stable-address guarantee.
impl<T: Clone, const PAGE: usize> Clone for PagedVec<T, PAGE> {
    fn clone(&self) -> Self {
        let mut pages = Vec::with_capacity(self.used_pages());
        for page in self.pages() {
            let mut new = Self::new_page();
            new.extend_from_slice(page);
            pages.push(new);
        }
        Self {
            pages,
            len: self.len,
        }
    }
}

impl<T: fmt::Debug, const PAGE: usize> fmt::Debug for PagedVec<T, PAGE> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// Element-wise, regardless of page size.
impl<T: PartialEq<U>, U, const PAGE: usize, const PAGE2: usize> PartialEq<PagedVec<U, PAGE2>>
    for PagedVec<T, PAGE>
{
    fn eq(&self, other: &PagedVec<U, PAGE2>) -> bool {
        self.len == other.len && self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl<T: Eq, const PAGE: usize> Eq for PagedVec<T, PAGE> {}

/// Lexicographic, like slices.
impl<T: PartialOrd, const PAGE: usize> PartialOrd for PagedVec<T, PAGE> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<T: Ord, const PAGE: usize> Ord for PagedVec<T, PAGE> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

/// Hashes the length, then each element. Consistent with `Eq` across page
/// sizes; not guaranteed to equal the hash of a slice with the same elements.
impl<T: Hash, const PAGE: usize> Hash for PagedVec<T, PAGE> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.len);
        for x in self {
            x.hash(state);
        }
    }
}

impl<T, const PAGE: usize> Index<usize> for PagedVec<T, PAGE> {
    type Output = T;

    #[inline]
    #[track_caller]
    fn index(&self, index: usize) -> &T {
        match self.get(index) {
            Some(x) => x,
            None => panic!(
                "index out of bounds: the len is {} but the index is {index}",
                self.len
            ),
        }
    }
}

impl<T, const PAGE: usize> IndexMut<usize> for PagedVec<T, PAGE> {
    #[inline]
    #[track_caller]
    fn index_mut(&mut self, index: usize) -> &mut T {
        let len = self.len;
        match self.get_mut(index) {
            Some(x) => x,
            None => panic!("index out of bounds: the len is {len} but the index is {index}"),
        }
    }
}

impl<T, const PAGE: usize> Extend<T> for PagedVec<T, PAGE> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        self.reserve(iter.size_hint().0);
        for x in iter {
            self.push(x);
        }
    }
}

impl<'a, T: Copy + 'a, const PAGE: usize> Extend<&'a T> for PagedVec<T, PAGE> {
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        self.extend(iter.into_iter().copied());
    }
}

impl<T, const PAGE: usize> FromIterator<T> for PagedVec<T, PAGE> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut v = Self::new();
        v.extend(iter);
        v
    }
}

impl<T, const PAGE: usize> From<Vec<T>> for PagedVec<T, PAGE> {
    fn from(v: Vec<T>) -> Self {
        v.into_iter().collect()
    }
}

impl<T, const N: usize, const PAGE: usize> From<[T; N]> for PagedVec<T, PAGE> {
    fn from(a: [T; N]) -> Self {
        a.into_iter().collect()
    }
}

impl<'a, T, const PAGE: usize> IntoIterator for &'a PagedVec<T, PAGE> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

impl<'a, T, const PAGE: usize> IntoIterator for &'a mut PagedVec<T, PAGE> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> IterMut<'a, T> {
        self.iter_mut()
    }
}

impl<T, const PAGE: usize> IntoIterator for PagedVec<T, PAGE> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(mut self) -> IntoIter<T> {
        let used = self.used_pages();
        self.pages.truncate(used);
        IntoIter {
            pages: self.pages.into_iter(),
            front: Vec::new().into_iter(),
            back: Vec::new().into_iter(),
            remaining: self.len,
        }
    }
}

/// Defines a flattening iterator over pages. `$pages` yields pages and
/// `$elems(page)` turns one into an element iterator; `front` and `back`
/// hold the partly consumed pages at either end.
macro_rules! paged_iter {
    ($(#[$m:meta])* $name:ident<$($lt:lifetime,)? $t:ident>, $item:ty, $pages:ty, $elems:ty, $into:expr) => {
        $(#[$m])*
        pub struct $name<$($lt,)? $t> {
            pages: $pages,
            front: $elems,
            back: $elems,
            remaining: usize,
        }

        impl<$($lt,)? $t> Iterator for $name<$($lt,)? $t> {
            type Item = $item;

            fn next(&mut self) -> Option<$item> {
                loop {
                    if let Some(x) = self.front.next() {
                        self.remaining -= 1;
                        return Some(x);
                    }
                    match self.pages.next() {
                        Some(page) => self.front = $into(page),
                        None => {
                            let x = self.back.next()?;
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
                loop {
                    if let Some(x) = self.back.next_back() {
                        self.remaining -= 1;
                        return Some(x);
                    }
                    match self.pages.next_back() {
                        Some(page) => self.back = $into(page),
                        None => {
                            let x = self.front.next_back()?;
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

paged_iter!(
    /// Iterator over references to the elements of a [`PagedVec`].
    Iter<'a, T>, &'a T, core::slice::Iter<'a, Vec<T>>, core::slice::Iter<'a, T>,
    |page: &'a Vec<T>| page.iter()
);

paged_iter!(
    /// Iterator over mutable references to the elements of a [`PagedVec`].
    IterMut<'a, T>, &'a mut T, core::slice::IterMut<'a, Vec<T>>, core::slice::IterMut<'a, T>,
    |page: &'a mut Vec<T>| page.iter_mut()
);

paged_iter!(
    /// Owning iterator over the elements of a [`PagedVec`].
    IntoIter<T>, T, alloc::vec::IntoIter<Vec<T>>, alloc::vec::IntoIter<T>,
    |page: Vec<T>| page.into_iter()
);

impl<T> Clone for Iter<'_, T> {
    fn clone(&self) -> Self {
        Self {
            pages: self.pages.clone(),
            front: self.front.clone(),
            back: self.back.clone(),
            remaining: self.remaining,
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for Iter<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

#[cfg(feature = "serde")]
mod serde_impl {
    use super::*;
    use core::marker::PhantomData;
    use serde::de::{Deserialize, Deserializer, SeqAccess, Visitor};
    use serde::ser::{Serialize, Serializer};

    /// Serializes as a sequence of elements.
    impl<T: Serialize, const PAGE: usize> Serialize for PagedVec<T, PAGE> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            serializer.collect_seq(self.iter())
        }
    }

    /// Deserializes from a sequence of elements. Memory is linear in the
    /// number of elements read; the length hint is capped.
    impl<'de, T: Deserialize<'de>, const PAGE: usize> Deserialize<'de> for PagedVec<T, PAGE> {
        fn deserialize<De: Deserializer<'de>>(deserializer: De) -> Result<Self, De::Error> {
            struct SeqVisitor<T, const PAGE: usize>(PhantomData<T>);

            impl<'de, T: Deserialize<'de>, const PAGE: usize> Visitor<'de> for SeqVisitor<T, PAGE> {
                type Value = PagedVec<T, PAGE>;

                fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    f.write_str("a sequence")
                }

                fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                    // Preallocate at most about 1 MiB of whole pages (elements
                    // plus directory entry) so a hostile length hint cannot
                    // force a huge allocation; none if one page is larger.
                    let budget = usize::try_from(1_u32 << 20).unwrap_or(usize::MAX);
                    let page_bytes = PAGE
                        .saturating_mul(core::mem::size_of::<T>())
                        .saturating_add(core::mem::size_of::<Vec<T>>());
                    let max_hint = (budget / page_bytes).saturating_mul(PAGE);
                    let hint = seq.size_hint().unwrap_or(0).min(max_hint);
                    let mut v = PagedVec::with_capacity(hint);
                    while let Some(x) = seq.next_element()? {
                        v.push(x);
                    }
                    Ok(v)
                }
            }

            deserializer.deserialize_seq(SeqVisitor::<T, PAGE>(PhantomData))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn addresses<T, const PAGE: usize>(v: &PagedVec<T, PAGE>) -> Vec<*const T> {
        v.iter().map(|x| x as *const T).collect()
    }

    #[test]
    fn push_never_moves_elements() {
        let mut v: PagedVec<u32, 3> = PagedVec::new();
        v.extend(0..7);
        let before = addresses(&v);
        for i in 7..1000 {
            v.push(i);
        }
        v.reserve(5000);
        assert_eq!(&addresses(&v)[..7], &before[..]);
        // A clone keeps full-capacity pages, so its partial page is stable.
        let mut c: PagedVec<u32, 3> = (0..4).collect::<PagedVec<_, 3>>().clone();
        let at3: *const u32 = &c[3];
        c.extend(4..100);
        assert!(core::ptr::eq(at3, &c[3]));
    }

    #[test]
    fn insert_and_remove_shift_across_pages() {
        let mut v: PagedVec<u32, 3> = (0..7).collect();
        let mut model: Vec<u32> = (0..7).collect();
        for (i, x) in [(0, 100), (3, 101), (9, 102), (5, 103), (2, 104)] {
            v.insert(i, x);
            model.insert(i, x);
            assert!(v.iter().eq(model.iter()));
        }
        for i in [0, 4, 9, 3, 0] {
            assert_eq!(v.remove(i), model.remove(i));
            assert!(v.iter().eq(model.iter()));
        }
        assert_eq!(v.swap_remove(1), model.swap_remove(1));
        assert!(v.iter().eq(model.iter()));
        let pages: Vec<&[u32]> = v.pages().collect();
        assert!(pages[..pages.len() - 1].iter().all(|p| p.len() == 3));
    }

    #[test]
    fn truncate_retain_and_iterators() {
        let mut v: PagedVec<u32, 4> = (0..10).collect();
        v.retain(|x| x % 3 != 0);
        assert!(v.iter().copied().eq([1, 2, 4, 5, 7, 8]));
        assert!(v.iter().rev().copied().eq([8, 7, 5, 4, 2, 1]));
        let mut it = v.iter();
        assert_eq!(
            (it.next(), it.next_back(), it.len()),
            (Some(&1), Some(&8), 4)
        );
        for x in &mut v {
            *x *= 10;
        }
        v.truncate(3);
        assert_eq!(v.into_iter().collect::<Vec<_>>(), vec![10, 20, 40]);

        let mut v: PagedVec<u32, 4> = (0..10).collect();
        assert_eq!(v.allocated_pages(), 3);
        v.clear();
        assert!(v.is_empty() && v.iter().next().is_none());
        assert_eq!(v.allocated_pages(), 3);
        v.shrink_to_fit();
        assert_eq!(v.allocated_pages(), 0);
        v.resize(5, 7);
        assert_eq!(v.len(), 5);
        assert_eq!((v.first(), v.last()), (Some(&7), Some(&7)));
        assert_eq!(v.get(5), None);
    }

    #[test]
    fn equality_order_and_debug_ignore_page_size() {
        let a: PagedVec<u32, 2> = (0..5).collect();
        let b: PagedVec<u32, 3> = (0..5).collect();
        assert_eq!(a, b);
        assert!(a < (1..2).collect::<PagedVec<u32, 2>>());
        assert_eq!(alloc::format!("{a:?}"), "[0, 1, 2, 3, 4]");
        assert_eq!(a.clone(), a);
    }

    #[test]
    fn zero_sized_elements() {
        let mut v: PagedVec<(), 4> = PagedVec::new();
        v.resize(10, ());
        assert_eq!(v.len(), 10);
        v.insert(3, ());
        v.remove(0);
        assert_eq!(v.iter().count(), 10);
    }

    #[cfg(feature = "std")]
    #[test]
    fn retain_panic_keeps_unvisited_in_order() {
        extern crate std;
        let mut v: PagedVec<u32, 3> = (0..10).collect();
        let mut calls = 0;
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            v.retain(|&x| {
                calls += 1;
                assert!(calls != 6, "boom");
                x % 2 == 0
            })
        }));
        assert!(r.is_err());
        // 0..5 visited (1 and 3 rejected), 5 was being tested.
        assert!(v.iter().copied().eq([0, 2, 4, 5, 6, 7, 8, 9]));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_round_trip() {
        let v: PagedVec<u32, 3> = (0..8).collect();
        let json = serde_json::to_string(&v).unwrap();
        assert_eq!(json, "[0,1,2,3,4,5,6,7]");
        let back: PagedVec<u32, 3> = serde_json::from_str(&json).unwrap();
        assert_eq!(back, v);
    }
}
