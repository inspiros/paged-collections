//! Page directories: where the pages of a paged index live.
//!
//! A paged index splits slot indices into a page number and an offset
//! (`index / PAGE`, `index % PAGE`). Pages are fixed-size arrays allocated on
//! first write. The *directory* maps page numbers to pages and comes in two
//! strategies, picked at the type level with a [`Directory`] marker:
//!
//! | Marker | Store | Page lookup | Directory memory |
//! |---|---|---|---|
//! | [`Flat`] | `Vec<Option<Box<[T; PAGE]>>>` | O(1), one indirection | one pointer per page up to the highest page used |
//! | [`Tree`] | `BTreeMap<usize, Box<[T; PAGE]>>` | O(log pages) | proportional to allocated pages only |
//!
//! Use [`Flat`] (the default) when indices stay below a few billion, for
//! example `u32` keys: the directory then costs at most
//! `8 * 2^32 / PAGE` bytes (32 MiB for the default page size). Use [`Tree`]
//! for indices spread over the 64-bit range, where a flat directory would be
//! too large to allocate.

use alloc::boxed::Box;
use alloc::collections::{BTreeMap, btree_map};
use alloc::vec::Vec;
use core::fmt;
use core::iter::{Enumerate, FilterMap, Map, Zip};
use core::ops::{Bound, Range, RangeInclusive};
use core::slice;

/// Default number of slots per page.
///
/// With 4-byte slots this makes 4 KiB pages, the usual OS page size.
pub const DEFAULT_PAGE: usize = 1024;

/// Splits a slot index into `(page number, offset within the page)`.
///
/// Fails to compile when `PAGE` is zero.
#[inline]
pub const fn split<const PAGE: usize>(index: usize) -> (usize, usize) {
    const { assert!(PAGE > 0, "page size must be non-zero") };
    (index / PAGE, index % PAGE)
}

/// Allocates one page with every slot set to `T::default()`.
///
/// The page is built directly on the heap, so large pages never pass through
/// the stack.
fn new_page<T: Default, const PAGE: usize>() -> Box<[T; PAGE]> {
    let slots: Box<[T]> = (0..PAGE).map(|_| T::default()).collect();
    match slots.try_into() {
        Ok(page) => page,
        Err(_) => unreachable!("iterator yielded exactly PAGE slots"),
    }
}

/// Storage for the pages of one paged index.
///
/// Implemented by [`FlatPages`] and [`TreePages`]; collections reach them
/// through a [`Directory`] marker.
///
/// # Contract for other implementations
///
/// [`get`](Self::get), [`get_mut`](Self::get_mut) and
/// [`allocated`](Self::allocated) must not panic. Collections call them while
/// they update their dense arrays and index together, where a panic would
/// leave the two out of step. As with [`Key`](crate::key::Key), breaking this
/// cannot cause undefined behaviour, only wrong results or panics.
///
/// The page iterators ([`pages`](Self::pages), [`pages_mut`](Self::pages_mut),
/// [`pages_in`](Self::pages_in), [`pages_in_mut`](Self::pages_in_mut),
/// [`into_pages`](Self::into_pages)) must yield every allocated page (for the
/// `_in` ones, every allocated page numbered in the range, and no other)
/// exactly once, in ascending page-number order: ordered collections rely on
/// it.
pub trait PageStore<T, const PAGE: usize>: Default {
    /// Iterator over `(page number, page)` from [`pages`](Self::pages).
    type Pages<'a>: DoubleEndedIterator<Item = (usize, &'a [T; PAGE])> + Clone
    where
        Self: 'a,
        T: 'a;

    /// Iterator over `(page number, page)` from [`pages_mut`](Self::pages_mut).
    type PagesMut<'a>: DoubleEndedIterator<Item = (usize, &'a mut [T; PAGE])>
    where
        Self: 'a,
        T: 'a;

    /// Owning iterator over `(page number, page)` from
    /// [`into_pages`](Self::into_pages).
    type IntoPages: DoubleEndedIterator<Item = (usize, Box<[T; PAGE]>)>;

    /// The page with this number, if it is allocated.
    fn get(&self, page: usize) -> Option<&[T; PAGE]>;

    /// Mutable access to the page with this number, if it is allocated.
    fn get_mut(&mut self, page: usize) -> Option<&mut [T; PAGE]>;

    /// The page with this number, allocating it (all slots `T::default()`)
    /// if needed.
    ///
    /// # Panics
    ///
    /// Panics if the directory cannot grow to hold this page number (for
    /// [`FlatPages`], when the directory length would overflow `usize` or
    /// the allocation fails).
    fn get_or_alloc(&mut self, page: usize) -> &mut [T; PAGE]
    where
        T: Default;

    /// Removes the page with this number and returns it, if it was allocated.
    fn remove(&mut self, page: usize) -> Option<Box<[T; PAGE]>>;

    /// Frees every page.
    fn clear(&mut self);

    /// Number of allocated pages.
    fn allocated(&self) -> usize;

    /// Releases directory memory that no allocated page needs.
    fn shrink_to_fit(&mut self);

    /// The allocated pages in ascending page-number order.
    fn pages(&self) -> Self::Pages<'_>;

    /// The allocated pages in ascending page-number order, mutably.
    fn pages_mut(&mut self) -> Self::PagesMut<'_>;

    /// The allocated pages numbered in `numbers`, in ascending order. An
    /// empty range (start above end) yields nothing.
    ///
    /// Reaching the first page costs O(1) for [`FlatPages`] and O(log pages)
    /// for [`TreePages`]; [`FlatPages`] then steps over the unallocated
    /// directory entries in the range.
    fn pages_in(&self, numbers: RangeInclusive<usize>) -> Self::Pages<'_>;

    /// Like [`pages_in`](Self::pages_in), mutably.
    fn pages_in_mut(&mut self, numbers: RangeInclusive<usize>) -> Self::PagesMut<'_>;

    /// Consumes the store and returns its pages in ascending page-number
    /// order.
    fn into_pages(self) -> Self::IntoPages;
}

/// Chooses the page store a collection uses.
///
/// This is a type-level switch: the collection names a marker ([`Flat`] or
/// [`Tree`]) and the marker names the store type for any slot type and page
/// size.
pub trait Directory {
    /// The page store for slots of type `T` in pages of `PAGE` slots.
    type Store<T, const PAGE: usize>: PageStore<T, PAGE>;
}

/// Marker for the flat, vector-backed directory ([`FlatPages`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Flat;

/// Marker for the tree-backed directory ([`TreePages`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Tree;

impl Directory for Flat {
    type Store<T, const PAGE: usize> = FlatPages<T, PAGE>;
}

impl Directory for Tree {
    type Store<T, const PAGE: usize> = TreePages<T, PAGE>;
}

/// Pages in a vector indexed by page number.
pub struct FlatPages<T, const PAGE: usize> {
    pages: Vec<Option<Box<[T; PAGE]>>>,
    allocated: usize,
}

impl<T, const PAGE: usize> Default for FlatPages<T, PAGE> {
    fn default() -> Self {
        Self {
            pages: Vec::new(),
            allocated: 0,
        }
    }
}

impl<T: Clone, const PAGE: usize> Clone for FlatPages<T, PAGE> {
    fn clone(&self) -> Self {
        Self {
            pages: self.pages.clone(),
            allocated: self.allocated,
        }
    }
}

impl<T, const PAGE: usize> fmt::Debug for FlatPages<T, PAGE> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FlatPages")
            .field("page_size", &PAGE)
            .field("directory_len", &self.pages.len())
            .field("allocated", &self.allocated)
            .finish()
    }
}

type FlatEntry<T, const PAGE: usize> = Option<Box<[T; PAGE]>>;

impl<T, const PAGE: usize> FlatPages<T, PAGE> {
    /// The directory positions of the page numbers in `numbers`.
    fn span(&self, numbers: RangeInclusive<usize>) -> Range<usize> {
        let end = numbers.end().saturating_add(1).min(self.pages.len());
        (*numbers.start()).min(end)..end
    }
}

impl<T, const PAGE: usize> PageStore<T, PAGE> for FlatPages<T, PAGE> {
    type Pages<'a>
        = FilterMap<
        Zip<Range<usize>, slice::Iter<'a, FlatEntry<T, PAGE>>>,
        fn((usize, &'a FlatEntry<T, PAGE>)) -> Option<(usize, &'a [T; PAGE])>,
    >
    where
        T: 'a;

    type PagesMut<'a>
        = FilterMap<
        Zip<Range<usize>, slice::IterMut<'a, FlatEntry<T, PAGE>>>,
        fn((usize, &'a mut FlatEntry<T, PAGE>)) -> Option<(usize, &'a mut [T; PAGE])>,
    >
    where
        T: 'a;

    type IntoPages = FilterMap<
        Enumerate<alloc::vec::IntoIter<FlatEntry<T, PAGE>>>,
        fn((usize, FlatEntry<T, PAGE>)) -> Option<(usize, Box<[T; PAGE]>)>,
    >;

    #[inline]
    fn get(&self, page: usize) -> Option<&[T; PAGE]> {
        self.pages.get(page)?.as_deref()
    }

    #[inline]
    fn get_mut(&mut self, page: usize) -> Option<&mut [T; PAGE]> {
        self.pages.get_mut(page)?.as_deref_mut()
    }

    fn get_or_alloc(&mut self, page: usize) -> &mut [T; PAGE]
    where
        T: Default,
    {
        if page >= self.pages.len() {
            let len = page.checked_add(1).expect("page directory overflow");
            self.pages.resize_with(len, || None);
        }
        let entry = &mut self.pages[page];
        if entry.is_none() {
            // Build the page first: `T::default()` may panic.
            *entry = Some(new_page::<T, PAGE>());
            self.allocated += 1;
        }
        match entry {
            Some(page) => page,
            None => unreachable!("page was just allocated"),
        }
    }

    fn remove(&mut self, page: usize) -> Option<Box<[T; PAGE]>> {
        let removed = self.pages.get_mut(page)?.take();
        if removed.is_some() {
            self.allocated -= 1;
        }
        removed
    }

    fn clear(&mut self) {
        // Reset the count first: dropping a page runs `T::drop`, which may
        // panic after the vector is already emptied.
        self.allocated = 0;
        self.pages.clear();
    }

    #[inline]
    fn allocated(&self) -> usize {
        self.allocated
    }

    fn shrink_to_fit(&mut self) {
        while matches!(self.pages.last(), Some(None)) {
            self.pages.pop();
        }
        self.pages.shrink_to_fit();
    }

    fn pages(&self) -> Self::Pages<'_> {
        self.pages_in(0..=usize::MAX)
    }

    fn pages_mut(&mut self) -> Self::PagesMut<'_> {
        self.pages_in_mut(0..=usize::MAX)
    }

    fn pages_in(&self, numbers: RangeInclusive<usize>) -> Self::Pages<'_> {
        let span = self.span(numbers);
        span.clone()
            .zip(self.pages[span].iter())
            .filter_map(|(n, page)| Some((n, page.as_deref()?)))
    }

    fn pages_in_mut(&mut self, numbers: RangeInclusive<usize>) -> Self::PagesMut<'_> {
        let span = self.span(numbers);
        span.clone()
            .zip(self.pages[span].iter_mut())
            .filter_map(|(n, page)| Some((n, page.as_deref_mut()?)))
    }

    fn into_pages(self) -> Self::IntoPages {
        self.pages
            .into_iter()
            .enumerate()
            .filter_map(|(n, page)| Some((n, page?)))
    }
}

/// `numbers` as bounds `BTreeMap::range` accepts: it panics on a start above
/// the end, so an empty range becomes `0..0`.
fn nonempty(numbers: RangeInclusive<usize>) -> (Bound<usize>, Bound<usize>) {
    if numbers.is_empty() {
        (Bound::Included(0), Bound::Excluded(0))
    } else {
        (
            Bound::Included(*numbers.start()),
            Bound::Included(*numbers.end()),
        )
    }
}

/// Pages in a B-tree keyed by page number.
pub struct TreePages<T, const PAGE: usize> {
    pages: BTreeMap<usize, Box<[T; PAGE]>>,
}

impl<T, const PAGE: usize> Default for TreePages<T, PAGE> {
    fn default() -> Self {
        Self {
            pages: BTreeMap::new(),
        }
    }
}

impl<T: Clone, const PAGE: usize> Clone for TreePages<T, PAGE> {
    fn clone(&self) -> Self {
        Self {
            pages: self.pages.clone(),
        }
    }
}

impl<T, const PAGE: usize> fmt::Debug for TreePages<T, PAGE> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TreePages")
            .field("page_size", &PAGE)
            .field("allocated", &self.pages.len())
            .finish()
    }
}

impl<T, const PAGE: usize> PageStore<T, PAGE> for TreePages<T, PAGE> {
    type Pages<'a>
        = Map<
        btree_map::Range<'a, usize, Box<[T; PAGE]>>,
        fn((&'a usize, &'a Box<[T; PAGE]>)) -> (usize, &'a [T; PAGE]),
    >
    where
        T: 'a;

    type PagesMut<'a>
        = Map<
        btree_map::RangeMut<'a, usize, Box<[T; PAGE]>>,
        fn((&'a usize, &'a mut Box<[T; PAGE]>)) -> (usize, &'a mut [T; PAGE]),
    >
    where
        T: 'a;

    type IntoPages = btree_map::IntoIter<usize, Box<[T; PAGE]>>;

    #[inline]
    fn get(&self, page: usize) -> Option<&[T; PAGE]> {
        self.pages.get(&page).map(|p| &**p)
    }

    #[inline]
    fn get_mut(&mut self, page: usize) -> Option<&mut [T; PAGE]> {
        self.pages.get_mut(&page).map(|p| &mut **p)
    }

    fn get_or_alloc(&mut self, page: usize) -> &mut [T; PAGE]
    where
        T: Default,
    {
        self.pages.entry(page).or_insert_with(new_page::<T, PAGE>)
    }

    fn remove(&mut self, page: usize) -> Option<Box<[T; PAGE]>> {
        self.pages.remove(&page)
    }

    fn clear(&mut self) {
        self.pages.clear();
    }

    #[inline]
    fn allocated(&self) -> usize {
        self.pages.len()
    }

    fn shrink_to_fit(&mut self) {}

    fn pages(&self) -> Self::Pages<'_> {
        self.pages_in(0..=usize::MAX)
    }

    fn pages_mut(&mut self) -> Self::PagesMut<'_> {
        self.pages_in_mut(0..=usize::MAX)
    }

    fn pages_in(&self, numbers: RangeInclusive<usize>) -> Self::Pages<'_> {
        self.pages
            .range(nonempty(numbers))
            .map(|(&n, page)| (n, &**page))
    }

    fn pages_in_mut(&mut self, numbers: RangeInclusive<usize>) -> Self::PagesMut<'_> {
        self.pages
            .range_mut(nonempty(numbers))
            .map(|(&n, page)| (n, &mut **page))
    }

    fn into_pages(self) -> Self::IntoPages {
        self.pages.into_iter()
    }
}
