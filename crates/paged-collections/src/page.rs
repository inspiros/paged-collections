//! Page directories: where the pages of a paged index live.
//!
//! A paged index splits slot indices into a page number and an offset
//! (`index / PAGE`, `index % PAGE`). Pages are fixed-size arrays allocated on
//! first write. The *directory* maps page numbers to pages and comes in three
//! strategies, picked at the type level with a [`Directory`] marker:
//!
//! | Marker | Store | Page lookup | Directory memory |
//! |---|---|---|---|
//! | [`Flat`] | `Vec<Option<Box<[T; PAGE]>>>` | O(1), one indirection | one pointer per page up to the highest page used |
//! | [`Radix`] | path-compressed 64-ary trie | O(depth), at most ceil(usize bits / 6) branches | proportional to allocated pages only |
//! | [`Tree`] | `BTreeMap<usize, Box<[T; PAGE]>>` | O(log pages) | proportional to allocated pages only |
//!
//! Use [`Flat`] (the default) when indices stay below a few billion, for
//! example `u32` keys: the directory then holds at most `ceil(2^32 / PAGE)`
//! pointer-sized entries (32 MiB for the default page size, plus up to as
//! much again in `Vec` growth capacity until
//! [`shrink_to_fit`](PageStore::shrink_to_fit)). Use [`Radix`]
//! for indices spread over the 64-bit range, where a flat directory would be
//! too large to allocate. Use [`Tree`] for a standard B-tree structure, or
//! when benchmarks show it wins for your key pattern.

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
/// Implemented by [`FlatPages`], [`RadixPages`] and [`TreePages`];
/// collections reach them through a [`Directory`] marker.
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
    /// Panics if `T::default()` panics, or if the directory cannot grow to hold this page number (for
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
    /// empty range (start above end, or exhausted by iteration) yields
    /// nothing.
    ///
    /// Reaching the first page costs O(1) for [`FlatPages`] and O(log pages)
    /// for [`TreePages`], and O(depth) for [`RadixPages`]; [`FlatPages`] then
    /// steps over the unallocated directory entries in the range.
    fn pages_in(&self, numbers: RangeInclusive<usize>) -> Self::Pages<'_>;

    /// Like [`pages_in`](Self::pages_in), mutably.
    fn pages_in_mut(&mut self, numbers: RangeInclusive<usize>) -> Self::PagesMut<'_>;

    /// Consumes the store and returns its pages in ascending page-number
    /// order.
    fn into_pages(self) -> Self::IntoPages;
}

/// Chooses the page store a collection uses.
///
/// This is a type-level switch: the collection names a marker ([`Flat`],
/// [`Radix`] or [`Tree`]) and the marker names the store type for any slot
/// type and page size.
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

/// Marker for the path-compressed radix directory ([`RadixPages`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Radix;

impl Directory for Radix {
    type Store<T, const PAGE: usize> = RadixPages<T, PAGE>;
}

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
        // An exhausted range keeps its endpoints, so check it first.
        if numbers.is_empty() {
            return 0..0;
        }
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

/// Pages in a bitmapped, path-compressed radix directory.
///
/// Page numbers may use the full `usize` range. Each branch examines six
/// bits, and shared paths are compressed, so lookup costs O(depth) and
/// directory memory is proportional to the number of allocated pages.
pub struct RadixPages<T, const PAGE: usize> {
    root: Option<Child<T, PAGE>>,
    allocated: usize,
}

#[derive(Clone)]
enum Child<T, const PAGE: usize> {
    Leaf { number: usize, page: Box<[T; PAGE]> },
    Node(Box<Node<T, PAGE>>),
}

#[derive(Clone)]
struct Node<T, const PAGE: usize> {
    shift: u32,
    prefix: usize,
    bitmap: u64,
    children: Vec<Child<T, PAGE>>,
}

fn chunk(number: usize, shift: u32) -> u32 {
    ((number >> shift) & 63) as u32
}

fn prefix(number: usize, shift: u32) -> usize {
    number.checked_shr(shift + 6).unwrap_or(0)
}

fn position(bitmap: u64, chunk: u32) -> usize {
    (bitmap & ((1_u64 << chunk) - 1)).count_ones() as usize
}

impl<T, const PAGE: usize> Node<T, PAGE> {
    /// Children whose chunks can contain page numbers in these bounds.
    fn span(&self, lo: usize, hi: usize) -> Range<usize> {
        let low = self.prefix.checked_shl(self.shift + 6).unwrap_or(0);
        let mask = usize::MAX
            .checked_shr(usize::BITS.saturating_sub(self.shift + 6))
            .unwrap_or(0);
        let high = low | mask;
        if hi < low || lo > high {
            return 0..0;
        }
        let first = if lo > low { chunk(lo, self.shift) } else { 0 };
        let last = if hi < high { chunk(hi, self.shift) } else { 63 };
        let start = position(self.bitmap, first);
        let end = (self.bitmap & (u64::MAX >> (63 - last))).count_ones() as usize;
        start..end
    }
}

impl<T, const PAGE: usize> Child<T, PAGE> {
    /// A number with the same prefix as every leaf below this child.
    fn representative(&self) -> usize {
        match self {
            Self::Leaf { number, .. } => *number,
            Self::Node(node) => node.prefix.checked_shl(node.shift + 6).unwrap_or(0),
        }
    }

    fn get(&self, number: usize) -> Option<&[T; PAGE]> {
        let mut child = self;
        loop {
            match child {
                Self::Leaf { number: n, page } => return (*n == number).then_some(page),
                Self::Node(node) => {
                    let c = chunk(number, node.shift);
                    if prefix(number, node.shift) != node.prefix || node.bitmap & (1_u64 << c) == 0
                    {
                        return None;
                    }
                    child = &node.children[position(node.bitmap, c)];
                }
            }
        }
    }

    fn get_mut(&mut self, number: usize) -> Option<&mut [T; PAGE]> {
        let mut child = self;
        loop {
            match child {
                Self::Leaf { number: n, page } => return (*n == number).then_some(page),
                Self::Node(node) => {
                    let c = chunk(number, node.shift);
                    if prefix(number, node.shift) != node.prefix || node.bitmap & (1_u64 << c) == 0
                    {
                        return None;
                    }
                    child = &mut node.children[position(node.bitmap, c)];
                }
            }
        }
    }

    /// The page of a leaf child.
    fn leaf_page(&mut self) -> &mut [T; PAGE] {
        match self {
            Self::Leaf { page, .. } => page,
            Self::Node(_) => unreachable!("child was just inserted as a leaf"),
        }
    }

    /// Inserts an absent page, after its defaults have all been built, and
    /// returns it.
    fn insert(&mut self, number: usize, page: Box<[T; PAGE]>) -> &mut [T; PAGE] {
        // Decide first, so the borrow of `self` below ends on every path.
        let descend = matches!(self, Self::Node(node) if prefix(number, node.shift) == node.prefix);
        if descend {
            let Self::Node(node) = self else {
                unreachable!("checked above")
            };
            let c = chunk(number, node.shift);
            let pos = position(node.bitmap, c);
            if node.bitmap & (1_u64 << c) == 0 {
                node.children.insert(pos, Self::Leaf { number, page });
                node.bitmap |= 1_u64 << c;
                return node.children[pos].leaf_page();
            }
            return node.children[pos].insert(number, page);
        }

        let existing = self.representative();
        let shift = (usize::BITS - 1 - (existing ^ number).leading_zeros()) / 6 * 6;
        let old_chunk = chunk(existing, shift);
        let new_chunk = chunk(number, shift);
        // Allocate before replacing the old child. No user code runs while
        // the new branch is being linked into the directory.
        let branch = Box::new(Node {
            shift,
            prefix: prefix(number, shift),
            bitmap: (1_u64 << old_chunk) | (1_u64 << new_chunk),
            children: Vec::with_capacity(2),
        });
        let old = core::mem::replace(self, Self::Node(branch));
        let Self::Node(node) = self else {
            unreachable!("branch was just installed")
        };
        let new = Self::Leaf { number, page };
        let pos = if old_chunk < new_chunk {
            node.children.push(old);
            node.children.push(new);
            1
        } else {
            node.children.push(new);
            node.children.push(old);
            0
        };
        node.children[pos].leaf_page()
    }

    /// Removes a known leaf and collapses every branch left with one child.
    fn remove(self, number: usize) -> (Option<Self>, Box<[T; PAGE]>) {
        match self {
            Self::Leaf { page, .. } => (None, page),
            Self::Node(mut node) => {
                let c = chunk(number, node.shift);
                let pos = position(node.bitmap, c);
                let (replacement, page) = node.children.remove(pos).remove(number);
                if let Some(child) = replacement {
                    node.children.insert(pos, child);
                } else {
                    node.bitmap &= !(1_u64 << c);
                }
                let remaining = if node.children.len() == 1 {
                    node.children.pop()
                } else {
                    Some(Self::Node(node))
                };
                (remaining, page)
            }
        }
    }

    fn shrink_to_fit(&mut self) {
        if let Self::Node(node) = self {
            for child in &mut node.children {
                child.shrink_to_fit();
            }
            node.children.shrink_to_fit();
        }
    }

    fn depth(&self) -> usize {
        match self {
            Self::Leaf { .. } => 0,
            Self::Node(node) => 1 + node.children.iter().map(Self::depth).max().unwrap_or(0),
        }
    }

    fn into_pages(self, pages: &mut Vec<(usize, Box<[T; PAGE]>)>) {
        match self {
            Self::Leaf { number, page } => pages.push((number, page)),
            Self::Node(node) => {
                for child in node.children {
                    child.into_pages(pages);
                }
            }
        }
    }
}

impl<T, const PAGE: usize> Default for RadixPages<T, PAGE> {
    fn default() -> Self {
        Self {
            root: None,
            allocated: 0,
        }
    }
}

impl<T: Clone, const PAGE: usize> Clone for RadixPages<T, PAGE> {
    fn clone(&self) -> Self {
        Self {
            root: self.root.clone(),
            allocated: self.allocated,
        }
    }
}

impl<T, const PAGE: usize> fmt::Debug for RadixPages<T, PAGE> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RadixPages")
            .field("page_size", &PAGE)
            .field("allocated", &self.allocated)
            .field("depth", &self.root.as_ref().map_or(0, Child::depth))
            .finish()
    }
}

/// Most slice iterators a radix walker holds at once: the root slice plus one
/// per node on each end's current path. Node shifts strictly decrease along a
/// path and are multiples of six, so a path has at most `ceil(BITS / 6)`
/// nodes.
const LEVELS: usize = 2 * (usize::BITS as usize).div_ceil(6) + 1;

/// Fixed-capacity deque of slice iterators, so creating a walker (which
/// `next_occupant`, `pop_first` and friends do on every call) never
/// allocates.
#[derive(Clone)]
struct Chain<I> {
    slots: [Option<I>; LEVELS],
    head: usize,
    len: usize,
}

impl<I> Chain<I> {
    fn new() -> Self {
        Self {
            slots: [const { None }; LEVELS],
            head: 0,
            len: 0,
        }
    }

    fn back_index(&self) -> usize {
        (self.head + self.len - 1) % LEVELS
    }

    fn front_mut(&mut self) -> Option<&mut I> {
        if self.len == 0 {
            return None;
        }
        self.slots[self.head].as_mut()
    }

    fn back_mut(&mut self) -> Option<&mut I> {
        if self.len == 0 {
            return None;
        }
        let back = self.back_index();
        self.slots[back].as_mut()
    }

    fn push_front(&mut self, iter: I) {
        assert!(self.len < LEVELS, "radix walker deeper than the key width");
        self.head = (self.head + LEVELS - 1) % LEVELS;
        self.slots[self.head] = Some(iter);
        self.len += 1;
    }

    fn push_back(&mut self, iter: I) {
        assert!(self.len < LEVELS, "radix walker deeper than the key width");
        self.len += 1;
        let back = self.back_index();
        self.slots[back] = Some(iter);
    }

    fn pop_front(&mut self) {
        if self.len > 0 {
            self.slots[self.head] = None;
            self.head = (self.head + 1) % LEVELS;
            self.len -= 1;
        }
    }

    fn pop_back(&mut self) {
        if self.len > 0 {
            let back = self.back_index();
            self.slots[back] = None;
            self.len -= 1;
        }
    }
}

/// Shared iterator over radix pages in ascending page-number order.
///
/// The two ends share a chain of slice iterators, so interleaving `next`
/// and `next_back` visits each page exactly once. The chain is stored
/// inline: creating or cloning the iterator never allocates.
pub struct RadixIter<'a, T, const PAGE: usize> {
    levels: Chain<slice::Iter<'a, Child<T, PAGE>>>,
    lo: usize,
    hi: usize,
    done: bool,
}

impl<T, const PAGE: usize> Clone for RadixIter<'_, T, PAGE> {
    fn clone(&self) -> Self {
        Self {
            levels: self.levels.clone(),
            lo: self.lo,
            hi: self.hi,
            done: self.done,
        }
    }
}

/// Mutable iterator over radix pages in ascending page-number order.
///
/// Pages yielded from either end are disjoint, including when `next` and
/// `next_back` are interleaved. The chain is stored inline: creating the
/// iterator never allocates.
pub struct RadixIterMut<'a, T, const PAGE: usize> {
    levels: Chain<slice::IterMut<'a, Child<T, PAGE>>>,
    lo: usize,
    hi: usize,
    done: bool,
}

// Both walkers split slices through their iterators. In particular, the
// mutable walker never borrows a child again after yielding one of its pages.
macro_rules! radix_iterator {
    ($name:ident, $lt:lifetime, $reference:ty, $iter:ident) => {
        impl<$lt, T, const PAGE: usize> $name<$lt, T, PAGE> {
            fn step(&mut self, back: bool) -> Option<(usize, $reference)> {
                while !self.done {
                    let level = if back {
                        self.levels.back_mut()
                    } else {
                        self.levels.front_mut()
                    };
                    let Some(level) = level else {
                        self.done = true;
                        return None;
                    };
                    let child = if back {
                        level.next_back()
                    } else {
                        level.next()
                    };
                    let Some(child) = child else {
                        if back {
                            self.levels.pop_back();
                        } else {
                            self.levels.pop_front();
                        }
                        continue;
                    };
                    match child {
                        Child::Leaf { number, page } => {
                            if (!back && *number > self.hi) || (back && *number < self.lo) {
                                self.done = true;
                                return None;
                            }
                            if *number >= self.lo && *number <= self.hi {
                                return Some((*number, page));
                            }
                        }
                        Child::Node(node) => {
                            let span = node.span(self.lo, self.hi);
                            if !span.is_empty() {
                                let children = node.children[span].$iter();
                                if back {
                                    self.levels.push_back(children);
                                } else {
                                    self.levels.push_front(children);
                                }
                            }
                        }
                    }
                }
                None
            }
        }

        impl<$lt, T, const PAGE: usize> Iterator for $name<$lt, T, PAGE> {
            type Item = (usize, $reference);

            fn next(&mut self) -> Option<Self::Item> {
                self.step(false)
            }
        }

        impl<T, const PAGE: usize> DoubleEndedIterator for $name<'_, T, PAGE> {
            fn next_back(&mut self) -> Option<Self::Item> {
                self.step(true)
            }
        }
    };
}

radix_iterator!(RadixIter, 'a, &'a [T; PAGE], iter);
radix_iterator!(RadixIterMut, 'a, &'a mut [T; PAGE], iter_mut);

impl<T, const PAGE: usize> PageStore<T, PAGE> for RadixPages<T, PAGE> {
    type Pages<'a>
        = RadixIter<'a, T, PAGE>
    where
        T: 'a;
    type PagesMut<'a>
        = RadixIterMut<'a, T, PAGE>
    where
        T: 'a;
    type IntoPages = alloc::vec::IntoIter<(usize, Box<[T; PAGE]>)>;

    #[inline]
    fn get(&self, page: usize) -> Option<&[T; PAGE]> {
        self.root.as_ref()?.get(page)
    }

    #[inline]
    fn get_mut(&mut self, page: usize) -> Option<&mut [T; PAGE]> {
        self.root.as_mut()?.get_mut(page)
    }

    fn get_or_alloc(&mut self, page: usize) -> &mut [T; PAGE]
    where
        T: Default,
    {
        if self.get(page).is_some() {
            return self.get_mut(page).expect("page was just found");
        }
        // Build the page first: `T::default()` may panic.
        let new = new_page::<T, PAGE>();
        self.allocated += 1;
        if self.root.is_none() {
            return self
                .root
                .insert(Child::Leaf {
                    number: page,
                    page: new,
                })
                .leaf_page();
        }
        match &mut self.root {
            Some(root) => root.insert(page, new),
            None => unreachable!("root checked above"),
        }
    }

    fn remove(&mut self, page: usize) -> Option<Box<[T; PAGE]>> {
        self.get(page)?;
        let root = self.root.take().expect("page is allocated");
        let (remaining, removed) = root.remove(page);
        self.root = remaining;
        self.allocated -= 1;
        Some(removed)
    }

    fn clear(&mut self) {
        // Empty the store before dropping pages: `T::drop()` may panic.
        self.allocated = 0;
        let root = self.root.take();
        drop(root);
    }

    #[inline]
    fn allocated(&self) -> usize {
        self.allocated
    }

    fn shrink_to_fit(&mut self) {
        if let Some(root) = &mut self.root {
            root.shrink_to_fit();
        }
    }

    fn pages(&self) -> Self::Pages<'_> {
        self.pages_in(0..=usize::MAX)
    }

    fn pages_mut(&mut self) -> Self::PagesMut<'_> {
        self.pages_in_mut(0..=usize::MAX)
    }

    fn pages_in(&self, numbers: RangeInclusive<usize>) -> Self::Pages<'_> {
        let mut levels = Chain::new();
        if let Some(root) = self.root.as_ref().filter(|_| !numbers.is_empty()) {
            levels.push_back(slice::from_ref(root).iter());
        }
        RadixIter {
            levels,
            lo: *numbers.start(),
            hi: *numbers.end(),
            done: numbers.is_empty(),
        }
    }

    fn pages_in_mut(&mut self, numbers: RangeInclusive<usize>) -> Self::PagesMut<'_> {
        let mut levels = Chain::new();
        if let Some(root) = self.root.as_mut().filter(|_| !numbers.is_empty()) {
            levels.push_back(slice::from_mut(root).iter_mut());
        }
        RadixIterMut {
            levels,
            lo: *numbers.start(),
            hi: *numbers.end(),
            done: numbers.is_empty(),
        }
    }

    fn into_pages(self) -> Self::IntoPages {
        let mut pages = Vec::with_capacity(self.allocated);
        if let Some(root) = self.root {
            root.into_pages(&mut pages);
        }
        pages.into_iter()
    }
}

#[cfg(test)]
impl<T, const PAGE: usize> RadixPages<T, PAGE> {
    fn check(&self) {
        fn visit<T, const PAGE: usize>(child: &Child<T, PAGE>, numbers: &mut Vec<usize>) {
            match child {
                Child::Leaf { number, .. } => numbers.push(*number),
                Child::Node(node) => {
                    assert!(node.children.len() >= 2);
                    assert_eq!(node.shift % 6, 0);
                    assert!(node.shift < usize::BITS);
                    assert_eq!(node.bitmap.count_ones() as usize, node.children.len());
                    let mut chunks = (0..64).filter(|c| node.bitmap & (1_u64 << c) != 0);
                    for child in &node.children {
                        if let Child::Node(inner) = child {
                            assert!(inner.shift < node.shift);
                        }
                        let start = numbers.len();
                        visit(child, numbers);
                        let c = chunks.next().expect("one bit per child");
                        for &n in &numbers[start..] {
                            assert_eq!(prefix(n, node.shift), node.prefix);
                            assert_eq!(chunk(n, node.shift), c);
                        }
                    }
                }
            }
        }
        let mut numbers = Vec::new();
        if let Some(root) = &self.root {
            visit(root, &mut numbers);
        }
        assert_eq!(numbers.len(), self.allocated);
        assert!(numbers.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(self.pages().map(|(n, _)| n).collect::<Vec<_>>(), numbers);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn numbers<S: PageStore<u8, 2>>(store: &mut S, range: RangeInclusive<usize>) -> Vec<usize> {
        let shared: Vec<usize> = store.pages_in(range.clone()).map(|(n, _)| n).collect();
        let unique: Vec<usize> = store.pages_in_mut(range).map(|(n, _)| n).collect();
        assert_eq!(shared, unique);
        shared
    }

    fn ranges<S: PageStore<u8, 2> + Default>() {
        let mut store = S::default();
        for n in [0, 2, 3] {
            store.get_or_alloc(n);
        }
        assert_eq!(numbers(&mut store, 0..=usize::MAX), [0, 2, 3]);
        assert_eq!(numbers(&mut store, 1..=2), [2]);
        #[allow(clippy::reversed_empty_ranges)]
        let inverted = 3..=1;
        assert_eq!(numbers(&mut store, inverted), [0; 0]);
        assert_eq!(numbers(&mut store, 4..=usize::MAX), [0; 0]);
        assert_eq!(numbers(&mut store, usize::MAX..=usize::MAX), [0; 0]);
        // An exhausted range keeps its endpoints but is empty.
        let mut exhausted = 0..=3;
        exhausted.by_ref().for_each(drop);
        assert_eq!((*exhausted.start(), *exhausted.end()), (3, 3));
        assert_eq!(numbers(&mut store, exhausted), [0; 0]);
        let mut back = 0..=0;
        back.next_back();
        assert_eq!(numbers(&mut store, back), [0; 0]);
    }

    #[test]
    fn pages_in_honours_empty_and_exhausted_ranges() {
        ranges::<FlatPages<u8, 2>>();
        ranges::<TreePages<u8, 2>>();
        ranges::<RadixPages<u8, 2>>();
    }

    #[test]
    fn radix_compresses_and_collapses_every_level() {
        let mut store = RadixPages::<u8, 2>::default();
        let keys: Vec<_> = (0..usize::BITS)
            .step_by(6)
            .flat_map(|shift| {
                let n = 1_usize << shift;
                [n - 1, n, n + 1]
            })
            .chain([usize::MAX - 1, usize::MAX])
            .collect();
        store.check();
        for &n in &keys {
            store.get_or_alloc(n)[0] = n as u8;
            store.check();
        }
        store.shrink_to_fit();
        store.check();
        let mut clone = store.clone();
        clone.check();
        for &n in keys.iter().rev() {
            clone.remove(n);
            clone.check();
        }
        assert_eq!(clone.allocated(), 0);
        for &n in &keys {
            store.remove(n);
            store.check();
        }
        assert!(store.root.is_none());
    }

    #[test]
    fn radix_shared_iterator_clone_needs_no_slot_clone() {
        #[derive(Default)]
        struct NotClone;
        let mut store = RadixPages::<NotClone, 1>::default();
        store.get_or_alloc(usize::MAX);
        let iter = store.pages();
        assert_eq!(iter.clone().next().map(|(n, _)| n), Some(usize::MAX));
        store.check();
    }

    #[test]
    fn radix_invariants_after_every_operation() {
        let config = proptest::test_runner::Config {
            cases: if cfg!(miri) { 4 } else { 256 },
            failure_persistence: None,
            ..proptest::test_runner::Config::default()
        };
        let ops = proptest::collection::vec(
            (
                0_u8..12,
                proptest::prelude::any::<usize>(),
                0_u8..4,
                0_u8..11,
            ),
            1..if cfg!(miri) { 16 } else { 200 },
        );
        proptest::test_runner::TestRunner::new(config)
            .run(&ops, |ops| {
                let mut store = RadixPages::<u8, 2>::default();
                for (op, value, region, level) in ops {
                    let shift = (u32::from(level) * 6).min((usize::BITS - 1) / 6 * 6);
                    let boundary = 1_usize << shift;
                    let n = match region {
                        0 => value % 300,
                        1 => usize::MAX - value % 300,
                        2 => boundary.saturating_add(value % 3).saturating_sub(1),
                        _ => value,
                    };
                    match op {
                        0..=3 => {
                            store.get_or_alloc(n);
                        }
                        4 => {
                            store.remove(n);
                        }
                        5 => store.clear(),
                        6 => store.shrink_to_fit(),
                        7 => {
                            store.clone().check();
                        }
                        8 => {
                            store.get(n);
                        }
                        9 => {
                            if let Some(page) = store.get_mut(n) {
                                page[0] ^= 1;
                            }
                        }
                        10 => {
                            for (_, page) in store.pages_in_mut(n..=usize::MAX) {
                                page[1] ^= 1;
                            }
                        }
                        _ => {
                            store.clone().into_pages().for_each(drop);
                        }
                    }
                    store.check();
                }
                Ok(())
            })
            .unwrap();
    }
}
