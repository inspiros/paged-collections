//! A stable-address vector with shared appends and exclusive editing.

use core::fmt;
use core::iter::{Flatten, FusedIterator};
use core::ops::{Index, IndexMut};

use super::append::{AppendLog, DenseIter, PagesMut};
use crate::page::DEFAULT_PAGE;
use crate::paged_vec::PagedVec;

/// A paged vector that supports appending through shared references.
///
/// Each push claims a distinct index and publishes its value without waiting
/// for earlier writers. Existing elements keep their addresses while shared.
/// [`get`](Self::get) can see every completed push, including values beyond
/// the hole-free prefix reported by [`len`](Self::len). Iteration and page
/// slices snapshot that prefix when called. When no push is in flight,
/// `len() == reserved_len()`.
///
/// Removing, moving or mutating elements requires exclusive access. Pages
/// stay allocated for reuse until [`shrink_to_fit`](Self::shrink_to_fit) or
/// destruction. Uses `core` and `alloc` only.
/// `Send` requires `T: Send`; `Sync` requires `T: Send + Sync`.
pub struct ConcurrentPagedVec<T, const PAGE: usize = DEFAULT_PAGE> {
    log: AppendLog<T, PAGE>,
}

// SAFETY: moving the vector transfers all owned cells and their destruction;
// references into it cannot outlive their borrow.
unsafe impl<T: Send, const PAGE: usize> Send for ConcurrentPagedVec<T, PAGE> {}
// SAFETY: the append log gives each writer a distinct slot and publishes
// immutable values; readers acquire publication and removal is exclusive.
unsafe impl<T: Send + Sync, const PAGE: usize> Sync for ConcurrentPagedVec<T, PAGE> {}

impl<T, const PAGE: usize> ConcurrentPagedVec<T, PAGE> {
    /// Number of elements per page.
    pub const PAGE: usize = PAGE;

    /// Creates an empty vector without allocating.
    ///
    /// # Panics
    ///
    /// Panics if `PAGE` is zero or a page allocation would exceed `isize::MAX`.
    pub fn new() -> Self {
        Self {
            log: AppendLog::new(),
        }
    }

    /// Appends `value` and returns its distinct index.
    ///
    /// Does not wait for an earlier writer. No user code runs between claim
    /// and publication, so a panic cannot leave a claimed, unwritten slot.
    ///
    /// # Panics
    ///
    /// Panics on length or allocation capacity overflow; allocation can fail.
    pub fn push(&self, value: T) -> usize {
        self.log
            .push(value, usize::MAX)
            .unwrap_or_else(|_| panic!("ConcurrentPagedVec length overflow"))
    }

    /// Appends every item with an individual [`push`](Self::push).
    /// Other writers may interleave their values. If iteration panics, values
    /// already pushed remain visible and no reservation is left unwritten.
    pub fn push_all(&self, iter: impl IntoIterator<Item = T>) {
        for value in iter {
            self.push(value);
        }
    }

    /// The element at `index`, once its push has published it.
    /// This can return a value beyond [`len`](Self::len).
    pub fn get(&self, index: usize) -> Option<&T> {
        self.log.get(index)
    }

    /// Length of the hole-free, written prefix.
    /// Can briefly lag behind completed pushes while an earlier push is in
    /// flight. Exact, and equal to [`reserved_len`](Self::reserved_len),
    /// when no push is in flight.
    pub fn len(&self) -> usize {
        self.log.len()
    }

    /// Whether the hole-free prefix is empty; subject to the lag of `len`.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of claimed positions, including writes still in flight.
    /// This count does not prove that any particular element is readable.
    pub fn reserved_len(&self) -> usize {
        self.log.reserved_len()
    }

    /// Iterates over a snapshot of the committed prefix in index order.
    pub fn iter(&self) -> DenseIter<'_, T, PAGE> {
        self.log.iter()
    }

    /// Nonempty slices of a snapshot of the committed prefix, in page order.
    /// Every slice except the last contains `PAGE` elements.
    pub fn pages(
        &self,
    ) -> impl DoubleEndedIterator<Item = &[T]> + ExactSizeIterator + FusedIterator {
        self.log.pages()
    }

    /// Mutable access to an element under exclusive access.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.log.get_mut(index)
    }

    /// Iterates mutably over all elements, in index order.
    pub fn iter_mut(&mut self) -> VecIterMut<'_, T> {
        VecIterMut::new(&mut self.log)
    }

    /// Nonempty mutable page slices, in index order.
    ///
    /// The iterator is `Send` only if `T` is:
    ///
    /// ```
    /// use paged_collections::concurrent::ConcurrentPagedVec;
    /// let mut vector = ConcurrentPagedVec::<u32, 2>::new();
    /// vector.push(0);
    /// std::thread::scope(|scope| {
    ///     let mut pages = vector.pages_mut();
    ///     scope.spawn(move || pages.next().unwrap()[0] = 1);
    /// });
    /// assert_eq!(vector[0], 1);
    /// ```
    ///
    /// ```compile_fail
    /// use paged_collections::concurrent::ConcurrentPagedVec;
    /// use std::{cell::Cell, rc::Rc};
    /// let mut vector = ConcurrentPagedVec::<Rc<Cell<u32>>, 2>::new();
    /// vector.push(Rc::new(Cell::new(0)));
    /// std::thread::scope(|scope| {
    ///     let mut pages = vector.pages_mut();
    ///     scope.spawn(move || pages.next().unwrap()[0].set(1));
    /// });
    /// ```
    pub fn pages_mut(&mut self) -> impl Iterator<Item = &mut [T]> {
        self.log.pages_mut()
    }

    /// Removes and returns the last element. Keeps allocated pages.
    pub fn pop(&mut self) -> Option<T> {
        self.log.pop()
    }

    /// Drops elements from the back until at most `len` remain. Keeps pages.
    /// If a destructor panics, that element is gone and earlier ones remain.
    pub fn truncate(&mut self, len: usize) {
        self.log.truncate(len);
    }

    /// Drops every element and clears readiness for subsequent reuse.
    /// Keeps allocated pages; destructor panics behave as in `truncate`.
    pub fn clear(&mut self) {
        self.log.clear();
    }

    /// Frees pages wholly beyond the end of the vector, keeping directory chunks.
    /// Existing elements keep their addresses. Shared appends can allocate the
    /// released pages again.
    pub fn shrink_to_fit(&mut self) {
        self.log.shrink_to_fit();
    }

    /// Clones the committed prefix into a plain [`PagedVec`] in O(prefix length).
    /// Like [`len`](Self::len) and [`iter`](Self::iter), this can omit
    /// completed pushes beyond an earlier unpublished reservation. The
    /// prefix forms a valid plain vector.
    pub fn snapshot(&self) -> PagedVec<T, PAGE>
    where
        T: Clone,
    {
        self.iter().cloned().collect()
    }

    /// Moves the elements into a plain vector in index order, in O(len).
    /// Element addresses can change; no cloning is required.
    pub fn into_inner(self) -> PagedVec<T, PAGE> {
        self.log.into_inner()
    }
}

impl<'a, T, const PAGE: usize> IntoIterator for &'a ConcurrentPagedVec<T, PAGE> {
    type Item = &'a T;
    type IntoIter = DenseIter<'a, T, PAGE>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, const PAGE: usize> IntoIterator for &'a mut ConcurrentPagedVec<T, PAGE> {
    type Item = &'a mut T;
    type IntoIter = VecIterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

/// Mutably iterates over the elements of a [`ConcurrentPagedVec`] in index order.
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct VecIterMut<'a, T> {
    inner: Flatten<PagesMut<'a, T>>,
}

impl<'a, T> VecIterMut<'a, T> {
    pub(super) fn new<const PAGE: usize>(log: &'a mut AppendLog<T, PAGE>) -> Self {
        Self {
            inner: log.pages_mut().flatten(),
        }
    }
}

impl<'a, T> Iterator for VecIterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T> fmt::Debug for VecIterMut<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VecIterMut").finish_non_exhaustive()
    }
}

/// Clones the committed prefix via [`ConcurrentPagedVec::snapshot`],
/// which can omit completed pushes beyond an unpublished reservation.
impl<T: Clone, const PAGE: usize> Clone for ConcurrentPagedVec<T, PAGE> {
    fn clone(&self) -> Self {
        Self::from(self.snapshot())
    }
}

impl<T, const PAGE: usize> Default for ConcurrentPagedVec<T, PAGE> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: fmt::Debug, const PAGE: usize> fmt::Debug for ConcurrentPagedVec<T, PAGE> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T, const PAGE: usize> Index<usize> for ConcurrentPagedVec<T, PAGE> {
    type Output = T;

    #[track_caller]
    fn index(&self, index: usize) -> &T {
        self.get(index)
            .expect("index not visible in ConcurrentPagedVec")
    }
}

impl<T, const PAGE: usize> IndexMut<usize> for ConcurrentPagedVec<T, PAGE> {
    #[track_caller]
    fn index_mut(&mut self, index: usize) -> &mut T {
        self.get_mut(index)
            .expect("index out of bounds in ConcurrentPagedVec")
    }
}

impl<T, const PAGE: usize> Extend<T> for ConcurrentPagedVec<T, PAGE> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.push_all(iter);
    }
}

impl<T, const PAGE: usize> FromIterator<T> for ConcurrentPagedVec<T, PAGE> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let result = Self::new();
        result.push_all(iter);
        result
    }
}

impl<T, const PAGE: usize> From<PagedVec<T, PAGE>> for ConcurrentPagedVec<T, PAGE> {
    fn from(inner: PagedVec<T, PAGE>) -> Self {
        inner.into_iter().collect()
    }
}

#[cfg(feature = "serde")]
mod serde_impl {
    use super::*;
    use serde::de::{Deserialize, Deserializer};
    use serde::ser::{Serialize, Serializer};

    /// Serializes the committed prefix as a valid plain vector sequence
    /// in O(prefix length). The prefix can omit completed pushes beyond an
    /// earlier unpublished reservation, as in [`ConcurrentPagedVec::iter`].
    impl<T: Serialize, const PAGE: usize> Serialize for ConcurrentPagedVec<T, PAGE> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            serializer.collect_seq(self.iter())
        }
    }

    /// Reads the plain [`PagedVec`] format and converts in O(len).
    impl<'de, T: Deserialize<'de>, const PAGE: usize> Deserialize<'de> for ConcurrentPagedVec<T, PAGE> {
        fn deserialize<De: Deserializer<'de>>(deserializer: De) -> Result<Self, De::Error> {
            PagedVec::deserialize(deserializer).map(Self::from)
        }
    }
}
