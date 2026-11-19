//! [`PagedBitSet`]: a set that stores one bit per key slot in fixed-size pages.
//!
//! Each page is allocated on first use. A lookup reads one bit at the slot
//! [`Key::index`] names, and iteration visits set bits in ascending slot-index
//! order. The default page holds 512 words (4 KiB, or 32,768 bits).
//!
//! The slot index must determine the whole key (`DirectKey<Tag = ()>`).
//! Use [`PagedSparseSet`](crate::sparse_set::PagedSparseSet) for generational
//! keys or for keys scattered thinly across pages.
//!
//! | Operation | Cost |
//! |---|---|
//! | [`contains`](PagedBitSet::contains), [`insert`](PagedBitSet::insert), [`remove`](PagedBitSet::remove), [`toggle`](PagedBitSet::toggle) | One page lookup, then one word access. A newly allocated page takes O(`PAGE`) to initialize. |
//! | Iteration, [`clear`](PagedBitSet::clear), [`retain`](PagedBitSet::retain) | Scan allocated words, plus gaps in a [`Flat`] directory. Iteration and retain also visit each set bit. |
//! | [`range`](PagedBitSet::range), [`count_range`](PagedBitSet::count_range), ordered queries | Seek to the first or last page of the range and scan words in the range as needed. |
//! | [`len`](PagedBitSet::len) | O(1) |
//!
//! Empty pages stay allocated for reuse. [`shrink_to_fit`](PagedBitSet::shrink_to_fit)
//! frees empty pages and [`reset`](PagedBitSet::reset) frees every page.
//!
//! ```
//! use paged_collections::bit_set::PagedBitSet;
//!
//! let mut set: PagedBitSet<usize, 1> = [130, 2, 5].into_iter().collect();
//! assert_eq!(set.iter().collect::<Vec<_>>(), [2, 5, 130]);
//! assert_eq!(set.next_occupant(3), Some(5));
//! assert_eq!(set.pop_last(), Some(130));
//! assert_eq!(set.allocated_pages(), 2);
//! set.shrink_to_fit();
//! assert_eq!(set.allocated_pages(), 1);
//! ```

use alloc::vec::Vec;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::iter::{Flatten, Fuse, FusedIterator, Peekable, once};
use core::marker::PhantomData;
use core::ops::{self, Bound, Deref, RangeBounds, RangeInclusive};

use crate::key::{DirectKey, Key};
use crate::page::{Directory, Flat, PageStore, split};

/// The default number of 64-bit words in a bit-set page (4 KiB).
pub const DEFAULT_BIT_PAGE: usize = 512;

const WORD_BITS: usize = u64::BITS as usize;

type Store<const PAGE: usize, D> = <D as Directory>::Store<u64, PAGE>;
type Pages<'a, const PAGE: usize, D> = <Store<PAGE, D> as PageStore<u64, PAGE>>::Pages<'a>;
type IntoPages<const PAGE: usize, D> = <Store<PAGE, D> as PageStore<u64, PAGE>>::IntoPages;

/// A set of keys stored as bits at their slot indices.
///
/// - `PAGE` is the number of 64-bit words per page. It must be non-zero.
/// - `D` is the page directory: [`Flat`] (default),
///   [`Radix`](crate::page::Radix) for wide keys, or [`Tree`](crate::page::Tree).
///
/// The key's [`DirectKey::Tag`] must be `()`. Use
/// [`PagedSparseSet`](crate::sparse_set::PagedSparseSet) for generational keys.
/// Set algebra and equality accept different directories with the same `PAGE`.
pub struct PagedBitSet<K, const PAGE: usize = DEFAULT_BIT_PAGE, D: Directory = Flat> {
    // Invariant: len is the number of set bits across all pages. Commit
    // the word and its count together before allocating or calling user code.
    pages: Store<PAGE, D>,
    len: usize,
    _key: PhantomData<fn() -> K>,
}

impl<K: DirectKey<Tag = ()>, const PAGE: usize, D: Directory> PagedBitSet<K, PAGE, D> {
    /// The number of 64-bit words per page.
    pub const PAGE_SIZE: usize = PAGE;

    /// The number of bits per page. The product must fit in `usize`.
    pub const BITS_PER_PAGE: usize = PAGE * WORD_BITS;

    /// Creates an empty set. The first insertion allocates the first page.
    pub fn new() -> Self {
        Self {
            pages: Default::default(),
            len: 0,
            _key: PhantomData,
        }
    }

    /// Returns the number of keys in the set.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the set holds no keys.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of allocated pages, including empty ones.
    pub fn allocated_pages(&self) -> usize {
        self.pages.allocated()
    }

    /// Returns whether the set contains `key`.
    pub fn contains(&self, key: K) -> bool {
        let (number, offset, mask) = locate::<PAGE>(key.index());
        self.pages
            .get(number)
            .is_some_and(|page| page[offset] & mask != 0)
    }

    /// Inserts `key` and returns whether it was newly added.
    pub fn insert(&mut self, key: K) -> bool {
        let (number, offset, mask) = locate::<PAGE>(key.index());
        let word = &mut self.pages.get_or_alloc(number)[offset];
        if *word & mask != 0 {
            return false;
        }
        let len = checked_len(self.len, 1);
        *word |= mask;
        self.len = len;
        true
    }

    /// Removes `key` and returns whether it was present.
    pub fn remove(&mut self, key: K) -> bool {
        let (number, offset, mask) = locate::<PAGE>(key.index());
        let Some(page) = self.pages.get_mut(number) else {
            return false;
        };
        let word = &mut page[offset];
        if *word & mask == 0 {
            return false;
        }
        *word &= !mask;
        self.len -= 1;
        true
    }

    /// Flips the bit for `key` and returns whether it is now present.
    pub fn toggle(&mut self, key: K) -> bool {
        let (number, offset, mask) = locate::<PAGE>(key.index());
        let word = &mut self.pages.get_or_alloc(number)[offset];
        let present = *word & mask == 0;
        let len = if present {
            checked_len(self.len, 1)
        } else {
            self.len - 1
        };
        *word ^= mask;
        self.len = len;
        present
    }

    /// Sets the bits in the range and returns the number of newly added keys.
    ///
    /// The bounds refer to [`Key::index`]. Every slot in the range must
    /// represent a valid key of type `K`. Unbounded endpoints refer to slot
    /// zero and `usize::MAX`; use an explicit upper bound for narrow keys.
    /// An empty or inverted range adds zero keys.
    ///
    /// # Panics
    ///
    /// Panics if the set would contain more than `usize::MAX` keys or the
    /// directory cannot allocate a page.
    ///
    /// ```
    /// use paged_collections::bit_set::PagedBitSet;
    ///
    /// let mut set: PagedBitSet<u8, 1> = PagedBitSet::new();
    /// assert_eq!(set.insert_range(63..=66), 4);
    /// assert_eq!(set.insert_range(64..=66), 0);
    /// assert_eq!(set.count_range(..=64), 2);
    /// assert_eq!(set.remove_range(65..), 2);
    /// assert_eq!(set.range(..).collect::<Vec<_>>(), [63, 64]);
    /// ```
    pub fn insert_range(&mut self, keys: impl RangeBounds<K>) -> usize {
        let Some(span) = Span::new(&index_bounds(&keys)) else {
            return 0;
        };
        let before = self.len;
        // The full slot domain has one more slot than len can represent.
        checked_len(span.last - span.first, 1);
        for number in span.pages::<PAGE>() {
            let page = self.pages.get_or_alloc(number);
            let start = page_start::<PAGE>(number);
            for offset in span.offsets(start, PAGE) {
                let word = &mut page[offset];
                let new = *word | span.mask(start + offset);
                write_word(&mut self.len, word, new);
            }
        }
        self.len - before
    }

    /// Clears the bits in the range and returns the number of removed keys.
    ///
    /// The bounds refer to [`Key::index`]. An empty or inverted range removes
    /// zero keys. Pages stay allocated.
    pub fn remove_range(&mut self, keys: impl RangeBounds<K>) -> usize {
        let Some(span) = Span::new(&index_bounds(&keys)) else {
            return 0;
        };
        let before = self.len;
        for (number, page) in self.pages.pages_in_mut(span.pages::<PAGE>()) {
            let start = page_start::<PAGE>(number);
            for offset in span.offsets(start, PAGE) {
                let word = &mut page[offset];
                let new = *word & !span.mask(start + offset);
                write_word(&mut self.len, word, new);
            }
        }
        before - self.len
    }

    /// Counts the set bits in the range.
    ///
    /// The bounds refer to [`Key::index`]. An empty or inverted range has
    /// count zero.
    pub fn count_range(&self, keys: impl RangeBounds<K>) -> usize {
        let Some(span) = Span::new(&index_bounds(&keys)) else {
            return 0;
        };
        WordPages {
            pages: self.pages.pages_in(span.pages::<PAGE>()),
            span,
        }
        .flatten()
        .map(|word| word.bits.count_ones() as usize)
        .sum()
    }

    /// Iterates over the keys in the range in ascending slot-index order.
    ///
    /// The bounds refer to [`Key::index`]. An empty or inverted range yields
    /// no keys.
    pub fn range(&self, keys: impl RangeBounds<K>) -> Range<'_, K, PAGE, D> {
        self.slot_range(&index_bounds(&keys))
    }

    fn slot_range(&self, indices: &impl RangeBounds<usize>) -> Range<'_, K, PAGE, D> {
        let span = Span::new(indices);
        let numbers = span.map_or_else(|| RangeInclusive::new(1, 0), Span::pages::<PAGE>);
        let remaining = if span.is_some() { self.len } else { 0 };
        let words = WordPages {
            pages: self.pages.pages_in(numbers),
            span: span.unwrap_or(Span::FULL),
        };
        Range(Cursor::new(words.flatten(), remaining), PhantomData)
    }

    /// Returns the key with the smallest slot index.
    pub fn first(&self) -> Option<K> {
        self.iter().next()
    }

    /// Returns the key with the largest slot index.
    pub fn last(&self) -> Option<K> {
        self.iter().next_back()
    }

    /// Removes and returns the key with the smallest slot index.
    pub fn pop_first(&mut self) -> Option<K> {
        let key = self.first()?;
        self.remove(key);
        Some(key)
    }

    /// Removes and returns the key with the largest slot index.
    pub fn pop_last(&mut self) -> Option<K> {
        let key = self.last()?;
        self.remove(key);
        Some(key)
    }

    /// Returns the first key at slot `index` or after it.
    pub fn next_occupant(&self, index: usize) -> Option<K> {
        self.slot_range(&(index..)).next()
    }

    /// Returns the last key at slot `index` or before it.
    pub fn prev_occupant(&self, index: usize) -> Option<K> {
        self.slot_range(&(..=index)).next_back()
    }

    /// Keeps keys for which `keep` returns true, visiting them in ascending
    /// slot-index order.
    ///
    /// If `keep` panics, the current key and those still to be visited stay
    /// in the set.
    pub fn retain(&mut self, mut keep: impl FnMut(K) -> bool) {
        let mut remaining = self.len;
        for (number, page) in self.pages.pages_mut() {
            let start = page_start::<PAGE>(number);
            for (offset, word) in page.iter_mut().enumerate() {
                if remaining == 0 {
                    return;
                }
                let mut bits = *word;
                while bits != 0 {
                    let bit = bits.trailing_zeros();
                    let mask = 1 << bit;
                    if !keep(K::from_parts(bit_index(start + offset, bit), ())) {
                        *word &= !mask;
                        self.len -= 1;
                    }
                    bits &= !mask;
                    remaining -= 1;
                }
            }
        }
    }

    /// Removes every key and returns them in ascending slot-index order.
    /// Pages stay allocated, as with [`clear`](Self::clear).
    ///
    /// Keys are removed as the iterator yields them; dropping it removes the
    /// rest. If it is leaked instead (`mem::forget`), the keys not yet
    /// yielded stay in the set.
    pub fn drain(&mut self) -> Drain<'_, K, PAGE, D> {
        Drain {
            set: self,
            next: Some(0),
        }
    }

    /// Removes the keys for which `pred` returns `true` and yields them,
    /// visiting the keys in ascending slot-index order.
    ///
    /// Keys are tested and removed as the iterator advances: those it has not
    /// reached when it is dropped stay in the set. Use
    /// [`retain`](Self::retain) if the removed keys are not needed.
    ///
    /// ```
    /// use paged_collections::bit_set::PagedBitSet;
    ///
    /// let mut set: PagedBitSet<u32> = (0..6).collect();
    /// assert_eq!(set.extract_if(|k| k % 2 == 1).collect::<Vec<_>>(), [1, 3, 5]);
    /// assert_eq!(set.drain().collect::<Vec<_>>(), [0, 2, 4]);
    /// assert!(set.is_empty());
    /// ```
    pub fn extract_if<F: FnMut(K) -> bool>(&mut self, pred: F) -> ExtractIf<'_, K, F, PAGE, D> {
        ExtractIf {
            set: self,
            next: Some(0),
            pred,
        }
    }

    /// Removes all keys, keeping the allocated pages for reuse.
    pub fn clear(&mut self) {
        for (_, page) in self.pages.pages_mut() {
            let removed: usize = page
                .iter_mut()
                .map(|word| core::mem::take(word).count_ones() as usize)
                .sum();
            self.len -= removed;
        }
    }

    /// Removes all keys and frees all pages.
    pub fn reset(&mut self) {
        let pages = core::mem::take(&mut self.pages);
        self.len = 0;
        drop(pages);
    }

    /// Frees empty pages and shrinks the page directory.
    pub fn shrink_to_fit(&mut self) {
        if self.is_empty() {
            self.pages.clear();
        } else {
            let empty: Vec<_> = self
                .pages
                .pages()
                .filter(|(_, page)| page.iter().all(|&word| word == 0))
                .map(|(number, _)| number)
                .collect();
            for number in empty {
                self.pages.remove(number);
            }
        }
        self.pages.shrink_to_fit();
    }

    /// Iterates over all keys in ascending slot-index order.
    pub fn iter(&self) -> Iter<'_, K, PAGE, D> {
        let words = WordPages {
            pages: self.pages.pages(),
            span: Span::FULL,
        };
        Iter(Cursor::new(words.flatten(), self.len), PhantomData)
    }
}

fn locate<const PAGE: usize>(index: usize) -> (usize, usize, u64) {
    let (number, offset) = split::<PAGE>(index / WORD_BITS);
    (number, offset, 1 << (index % WORD_BITS))
}

fn page_start<const PAGE: usize>(number: usize) -> usize {
    // Pages are allocated only for real slot indices, so their first word
    // always has a representable word and bit index.
    debug_assert!(number <= usize::MAX / WORD_BITS / PAGE);
    number * PAGE
}

fn bit_index(word: usize, bit: u32) -> usize {
    // A set bit came from a real slot index. Range masks also clip the last
    // page to the last representable word.
    debug_assert!(word <= usize::MAX / WORD_BITS && bit < u64::BITS);
    word * WORD_BITS + bit as usize
}

fn checked_len(len: usize, added: usize) -> usize {
    len.checked_add(added).expect("PagedBitSet length overflow")
}

fn write_word(len: &mut usize, word: &mut u64, new: u64) {
    let updated = checked_len(*len - word.count_ones() as usize, new.count_ones() as usize);
    *word = new;
    *len = updated;
}

fn index_bounds<K: Key>(keys: &impl RangeBounds<K>) -> (Bound<usize>, Bound<usize>) {
    (
        keys.start_bound().map(|key| key.index()),
        keys.end_bound().map(|key| key.index()),
    )
}

// The inclusive slot bounds avoid a special case for an end at usize::MAX.
#[derive(Clone, Copy)]
struct Span {
    first: usize,
    last: usize,
}

impl Span {
    const FULL: Self = Self {
        first: 0,
        last: usize::MAX,
    };

    fn new(indices: &impl RangeBounds<usize>) -> Option<Self> {
        let first = match indices.start_bound() {
            Bound::Included(&index) => index,
            Bound::Excluded(&index) => index.checked_add(1)?,
            Bound::Unbounded => 0,
        };
        let last = match indices.end_bound() {
            Bound::Included(&index) => index,
            Bound::Excluded(&index) => index.checked_sub(1)?,
            Bound::Unbounded => usize::MAX,
        };
        (first <= last).then_some(Self { first, last })
    }

    fn pages<const PAGE: usize>(self) -> RangeInclusive<usize> {
        let (first, _) = split::<PAGE>(self.first / WORD_BITS);
        let (last, _) = split::<PAGE>(self.last / WORD_BITS);
        first..=last
    }

    fn offsets(self, start: usize, page_len: usize) -> ops::Range<usize> {
        let first = (self.first / WORD_BITS).saturating_sub(start).min(page_len);
        let end = (self.last / WORD_BITS + 1)
            .saturating_sub(start)
            .min(page_len);
        first..end
    }

    fn mask(self, word: usize) -> u64 {
        let mut mask = u64::MAX;
        if word == self.first / WORD_BITS {
            mask &= u64::MAX << (self.first % WORD_BITS);
        }
        if word == self.last / WORD_BITS {
            mask &= u64::MAX >> (WORD_BITS - 1 - self.last % WORD_BITS);
        }
        mask
    }
}

#[derive(Clone, Copy, Default)]
struct WordBits {
    index: usize,
    bits: u64,
}

impl Iterator for WordBits {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.bits == 0 {
            return None;
        }
        let bit = self.bits.trailing_zeros();
        self.bits &= self.bits - 1;
        Some(bit_index(self.index, bit))
    }
}

impl DoubleEndedIterator for WordBits {
    fn next_back(&mut self) -> Option<usize> {
        if self.bits == 0 {
            return None;
        }
        let bit = u64::BITS - 1 - self.bits.leading_zeros();
        self.bits &= !(1 << bit);
        Some(bit_index(self.index, bit))
    }
}

#[derive(Clone)]
struct PageWords<P, const PAGE: usize> {
    page: P,
    start: usize,
    offsets: ops::Range<usize>,
    span: Span,
}

impl<P: Deref<Target = [u64; PAGE]>, const PAGE: usize> PageWords<P, PAGE> {
    fn new(number: usize, page: P, span: Span) -> Self {
        let start = page_start::<PAGE>(number);
        Self {
            page,
            start,
            offsets: span.offsets(start, PAGE),
            span,
        }
    }

    fn word(&self, offset: usize) -> WordBits {
        let index = self.start + offset;
        WordBits {
            index,
            bits: self.page[offset] & self.span.mask(index),
        }
    }
}

impl<P: Deref<Target = [u64; PAGE]>, const PAGE: usize> Iterator for PageWords<P, PAGE> {
    type Item = WordBits;

    fn next(&mut self) -> Option<WordBits> {
        let offset = self.offsets.next()?;
        Some(self.word(offset))
    }
}

impl<P: Deref<Target = [u64; PAGE]>, const PAGE: usize> DoubleEndedIterator for PageWords<P, PAGE> {
    fn next_back(&mut self) -> Option<WordBits> {
        let offset = self.offsets.next_back()?;
        Some(self.word(offset))
    }
}

#[derive(Clone)]
struct WordPages<I, const PAGE: usize> {
    pages: I,
    span: Span,
}

impl<I, P, const PAGE: usize> Iterator for WordPages<I, PAGE>
where
    I: Iterator<Item = (usize, P)>,
    P: Deref<Target = [u64; PAGE]>,
{
    type Item = PageWords<P, PAGE>;

    fn next(&mut self) -> Option<Self::Item> {
        let (number, page) = self.pages.next()?;
        Some(PageWords::new(number, page, self.span))
    }
}

impl<I, P, const PAGE: usize> DoubleEndedIterator for WordPages<I, PAGE>
where
    I: DoubleEndedIterator<Item = (usize, P)>,
    P: Deref<Target = [u64; PAGE]>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        let (number, page) = self.pages.next_back()?;
        Some(PageWords::new(number, page, self.span))
    }
}

#[derive(Clone)]
struct Cursor<W: Iterator<Item = WordBits>> {
    // Flatten shares the last page and the last word between the front and
    // back. Each WordBits value clears yielded bits from that shared copy.
    bits: Flatten<W>,
    remaining: usize,
}

impl<W: DoubleEndedIterator<Item = WordBits>> Cursor<W> {
    fn new(words: W, remaining: usize) -> Self {
        Self {
            bits: words.flatten(),
            remaining,
        }
    }

    fn next(&mut self) -> Option<usize> {
        if self.remaining == 0 {
            return None;
        }
        let index = self.bits.next();
        self.remaining = if index.is_some() {
            self.remaining - 1
        } else {
            0
        };
        index
    }

    fn next_back(&mut self) -> Option<usize> {
        if self.remaining == 0 {
            return None;
        }
        let index = self.bits.next_back();
        self.remaining = if index.is_some() {
            self.remaining - 1
        } else {
            0
        };
        index
    }
}

/// The keys removed by [`PagedBitSet::drain`], in slot-index order.
pub struct Drain<
    'a,
    K: DirectKey<Tag = ()>,
    const PAGE: usize = DEFAULT_BIT_PAGE,
    D: Directory = Flat,
> {
    set: &'a mut PagedBitSet<K, PAGE, D>,
    // The slot index to resume from; `None` once the last slot is passed.
    next: Option<usize>,
}

impl<K: DirectKey<Tag = ()>, const PAGE: usize, D: Directory> Iterator for Drain<'_, K, PAGE, D> {
    type Item = K;

    fn next(&mut self) -> Option<K> {
        let Some(key) = self.set.next_occupant(self.next?) else {
            self.next = None;
            return None;
        };
        let next = key.index().checked_add(1);
        self.set.remove(key);
        // Commit the cursor only after the removal, so that every key
        // before it is gone even if a custom `Key::index` panics.
        self.next = next;
        Some(key)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // Every key before the cursor has been removed.
        (self.set.len, Some(self.set.len))
    }
}

impl<K: DirectKey<Tag = ()>, const PAGE: usize, D: Directory> ExactSizeIterator
    for Drain<'_, K, PAGE, D>
{
}

impl<K: DirectKey<Tag = ()>, const PAGE: usize, D: Directory> FusedIterator
    for Drain<'_, K, PAGE, D>
{
}

impl<K: DirectKey<Tag = ()>, const PAGE: usize, D: Directory> Drop for Drain<'_, K, PAGE, D> {
    fn drop(&mut self) {
        self.for_each(drop);
    }
}

impl<K: DirectKey<Tag = ()> + fmt::Debug, const PAGE: usize, D: Directory> fmt::Debug
    for Drain<'_, K, PAGE, D>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Every key still in the set is yet to be yielded.
        f.debug_list().entries(self.set.iter()).finish()
    }
}

/// The keys removed by [`PagedBitSet::extract_if`], in slot-index order.
pub struct ExtractIf<
    'a,
    K: DirectKey<Tag = ()>,
    F,
    const PAGE: usize = DEFAULT_BIT_PAGE,
    D: Directory = Flat,
> {
    set: &'a mut PagedBitSet<K, PAGE, D>,
    next: Option<usize>,
    pred: F,
}

impl<K: DirectKey<Tag = ()>, F: FnMut(K) -> bool, const PAGE: usize, D: Directory> Iterator
    for ExtractIf<'_, K, F, PAGE, D>
{
    type Item = K;

    fn next(&mut self) -> Option<K> {
        loop {
            let Some(key) = self.set.next_occupant(self.next?) else {
                self.next = None;
                return None;
            };
            // Advance first: if `pred` panics, this key stays.
            self.next = key.index().checked_add(1);
            if (self.pred)(key) {
                self.set.remove(key);
                return Some(key);
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(if self.next.is_some() { self.set.len } else { 0 }))
    }
}

impl<K: DirectKey<Tag = ()>, F: FnMut(K) -> bool, const PAGE: usize, D: Directory> FusedIterator
    for ExtractIf<'_, K, F, PAGE, D>
{
}

impl<K: DirectKey<Tag = ()>, F, const PAGE: usize, D: Directory> fmt::Debug
    for ExtractIf<'_, K, F, PAGE, D>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExtractIf")
            .field("next", &self.next)
            .finish_non_exhaustive()
    }
}

/// A borrowed iterator over all the keys of a [`PagedBitSet`], in slot-index order.
pub struct Iter<'a, K, const PAGE: usize = DEFAULT_BIT_PAGE, D: Directory = Flat>(
    Cursor<Flatten<WordPages<Pages<'a, PAGE, D>, PAGE>>>,
    PhantomData<fn() -> K>,
)
where
    D: 'a;

/// A borrowed iterator over the keys in a range, in slot-index order.
pub struct Range<'a, K, const PAGE: usize = DEFAULT_BIT_PAGE, D: Directory = Flat>(
    Cursor<Flatten<WordPages<Pages<'a, PAGE, D>, PAGE>>>,
    PhantomData<fn() -> K>,
)
where
    D: 'a;

/// An owning iterator over all the keys of a [`PagedBitSet`], in slot-index order.
pub struct IntoIter<K, const PAGE: usize = DEFAULT_BIT_PAGE, D: Directory = Flat>(
    Cursor<Flatten<WordPages<IntoPages<PAGE, D>, PAGE>>>,
    PhantomData<fn() -> K>,
);

macro_rules! impl_iterator {
    ($name:ident $(<$lt:lifetime>)?, $exact:literal) => {
        impl<$($lt,)? K: DirectKey<Tag = ()>, const PAGE: usize, D: Directory> Iterator
            for $name<$($lt,)? K, PAGE, D>
            $(where D: $lt)?
        {
            type Item = K;

            fn next(&mut self) -> Option<K> {
                self.0.next().map(|index| K::from_parts(index, ()))
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                let remaining = self.0.remaining;
                (if $exact { remaining } else { 0 }, Some(remaining))
            }
        }

        impl<$($lt,)? K: DirectKey<Tag = ()>, const PAGE: usize, D: Directory> DoubleEndedIterator
            for $name<$($lt,)? K, PAGE, D>
            $(where D: $lt)?
        {
            fn next_back(&mut self) -> Option<K> {
                self.0.next_back().map(|index| K::from_parts(index, ()))
            }
        }

        impl<$($lt,)? K: DirectKey<Tag = ()>, const PAGE: usize, D: Directory> FusedIterator
            for $name<$($lt,)? K, PAGE, D>
            $(where D: $lt)?
        {}
    };
}

impl_iterator!(Iter<'a>, true);
impl_iterator!(Range<'a>, false);
impl_iterator!(IntoIter, true);

impl<'a, K: DirectKey<Tag = ()>, const PAGE: usize, D: Directory> ExactSizeIterator
    for Iter<'a, K, PAGE, D>
where
    D: 'a,
{
}

impl<K: DirectKey<Tag = ()>, const PAGE: usize, D: Directory> ExactSizeIterator
    for IntoIter<K, PAGE, D>
{
}

impl<'a, K, const PAGE: usize, D: Directory> Clone for Iter<'a, K, PAGE, D>
where
    D: 'a,
{
    fn clone(&self) -> Self {
        Self(self.0.clone(), PhantomData)
    }
}

impl<'a, K, const PAGE: usize, D: Directory> Clone for Range<'a, K, PAGE, D>
where
    D: 'a,
{
    fn clone(&self) -> Self {
        Self(self.0.clone(), PhantomData)
    }
}

impl<K: DirectKey<Tag = ()>, const PAGE: usize, D: Directory> Default for PagedBitSet<K, PAGE, D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: DirectKey<Tag = ()>, const PAGE: usize, D: Directory> Clone for PagedBitSet<K, PAGE, D>
where
    Store<PAGE, D>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            pages: self.pages.clone(),
            len: self.len,
            _key: PhantomData,
        }
    }
}

impl<K: DirectKey<Tag = ()> + fmt::Debug, const PAGE: usize, D: Directory> fmt::Debug
    for PagedBitSet<K, PAGE, D>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self).finish()
    }
}

impl<K: DirectKey<Tag = ()>, const PAGE: usize, D: Directory> Extend<K>
    for PagedBitSet<K, PAGE, D>
{
    fn extend<I: IntoIterator<Item = K>>(&mut self, iter: I) {
        for key in iter {
            self.insert(key);
        }
    }
}

impl<'a, K: DirectKey<Tag = ()> + 'a, const PAGE: usize, D: Directory> Extend<&'a K>
    for PagedBitSet<K, PAGE, D>
{
    fn extend<I: IntoIterator<Item = &'a K>>(&mut self, iter: I) {
        self.extend(iter.into_iter().copied());
    }
}

impl<K: DirectKey<Tag = ()>, const PAGE: usize, D: Directory> FromIterator<K>
    for PagedBitSet<K, PAGE, D>
{
    fn from_iter<I: IntoIterator<Item = K>>(iter: I) -> Self {
        let mut set = Self::new();
        set.extend(iter);
        set
    }
}

impl<'a, K: DirectKey<Tag = ()>, const PAGE: usize, D: Directory> IntoIterator
    for &'a PagedBitSet<K, PAGE, D>
{
    type Item = K;
    type IntoIter = Iter<'a, K, PAGE, D>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<K: DirectKey<Tag = ()>, const PAGE: usize, D: Directory> IntoIterator
    for PagedBitSet<K, PAGE, D>
{
    type Item = K;
    type IntoIter = IntoIter<K, PAGE, D>;

    fn into_iter(self) -> Self::IntoIter {
        let words = WordPages {
            pages: self.pages.into_pages(),
            span: Span::FULL,
        };
        IntoIter(Cursor::new(words.flatten(), self.len), PhantomData)
    }
}

#[derive(Clone, Copy)]
enum Algebra {
    Union,
    Intersection,
    Difference,
    SymmetricDifference,
}

impl Algebra {
    fn word(self, left: u64, right: u64) -> u64 {
        match self {
            Self::Union => left | right,
            Self::Intersection => left & right,
            Self::Difference => left & !right,
            Self::SymmetricDifference => left ^ right,
        }
    }
}

type Page<'a, const PAGE: usize> = (usize, &'a [u64; PAGE]);

struct PageMerge<'a, const PAGE: usize, D: Directory, D2: Directory>
where
    D: 'a,
    D2: 'a,
{
    left: Peekable<Fuse<Pages<'a, PAGE, D>>>,
    right: Peekable<Fuse<Pages<'a, PAGE, D2>>>,
    op: Algebra,
}

impl<'a, const PAGE: usize, D: Directory, D2: Directory> Clone for PageMerge<'a, PAGE, D, D2>
where
    D: 'a,
    D2: 'a,
{
    fn clone(&self) -> Self {
        Self {
            left: self.left.clone(),
            right: self.right.clone(),
            op: self.op,
        }
    }
}

impl<'a, const PAGE: usize, D: Directory, D2: Directory> PageMerge<'a, PAGE, D, D2>
where
    D: 'a,
    D2: 'a,
{
    fn new(left: Pages<'a, PAGE, D>, right: Pages<'a, PAGE, D2>, op: Algebra) -> Self {
        Self {
            left: left.fuse().peekable(),
            right: right.fuse().peekable(),
            op,
        }
    }
}

#[derive(Clone)]
struct CombinedWords<'a, const PAGE: usize> {
    number: usize,
    left: Option<&'a [u64; PAGE]>,
    right: Option<&'a [u64; PAGE]>,
    op: Algebra,
    offsets: ops::Range<usize>,
}

impl<const PAGE: usize> Iterator for CombinedWords<'_, PAGE> {
    type Item = (usize, u64);

    fn next(&mut self) -> Option<Self::Item> {
        let offset = self.offsets.next()?;
        let left = self.left.map_or(0, |page| page[offset]);
        let right = self.right.map_or(0, |page| page[offset]);
        Some((offset, self.op.word(left, right)))
    }
}

impl<'a, const PAGE: usize, D: Directory, D2: Directory> Iterator for PageMerge<'a, PAGE, D, D2>
where
    D: 'a,
    D2: 'a,
{
    type Item = CombinedWords<'a, PAGE>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (left, right): (Option<Page<'a, PAGE>>, Option<Page<'a, PAGE>>) =
                match (self.left.peek(), self.right.peek()) {
                    (Some((left, _)), Some((right, _))) => match left.cmp(right) {
                        Ordering::Less => (self.left.next(), None),
                        Ordering::Equal => (self.left.next(), self.right.next()),
                        Ordering::Greater => (None, self.right.next()),
                    },
                    (Some(_), None) => (self.left.next(), None),
                    (None, Some(_)) => (None, self.right.next()),
                    (None, None) => return None,
                };
            let number = left.or(right)?.0;
            match self.op {
                Algebra::Intersection if left.is_none() || right.is_none() => continue,
                Algebra::Difference if left.is_none() => continue,
                _ => {}
            }
            return Some(CombinedWords {
                number,
                left: left.map(|(_, page)| page),
                right: right.map(|(_, page)| page),
                op: self.op,
                offsets: 0..PAGE,
            });
        }
    }
}

struct AlgebraIter<'a, K, const PAGE: usize, D: Directory, D2: Directory>
where
    D: 'a,
    D2: 'a,
{
    pages: PageMerge<'a, PAGE, D, D2>,
    words: Option<CombinedWords<'a, PAGE>>,
    bits: WordBits,
    key: PhantomData<fn() -> K>,
}

impl<'a, K, const PAGE: usize, D: Directory, D2: Directory> Clone
    for AlgebraIter<'a, K, PAGE, D, D2>
where
    D: 'a,
    D2: 'a,
{
    fn clone(&self) -> Self {
        Self {
            pages: self.pages.clone(),
            words: self.words.clone(),
            bits: self.bits,
            key: PhantomData,
        }
    }
}

impl<'a, K: DirectKey<Tag = ()>, const PAGE: usize, D: Directory, D2: Directory> Iterator
    for AlgebraIter<'a, K, PAGE, D, D2>
where
    D: 'a,
    D2: 'a,
{
    type Item = K;

    fn next(&mut self) -> Option<K> {
        loop {
            if let Some(index) = self.bits.next() {
                return Some(K::from_parts(index, ()));
            }
            if let Some(words) = &mut self.words {
                if let Some((offset, bits)) = words.find(|&(_, bits)| bits != 0) {
                    self.bits = WordBits {
                        index: page_start::<PAGE>(words.number) + offset,
                        bits,
                    };
                    continue;
                }
            }
            self.words = Some(self.pages.next()?);
        }
    }
}

impl<'a, K: DirectKey<Tag = ()>, const PAGE: usize, D: Directory, D2: Directory> FusedIterator
    for AlgebraIter<'a, K, PAGE, D, D2>
where
    D: 'a,
    D2: 'a,
{
}

macro_rules! algebra_iterator {
    ($($(#[$doc:meta])* $name:ident,)*) => {$(
        $(#[$doc])*
        pub struct $name<'a, K, const PAGE: usize, D: Directory, D2: Directory>(
            AlgebraIter<'a, K, PAGE, D, D2>,
        )
        where
            D: 'a,
            D2: 'a;

        impl<'a, K: DirectKey<Tag = ()>, const PAGE: usize, D: Directory, D2: Directory> Iterator
            for $name<'a, K, PAGE, D, D2>
        where
            D: 'a,
            D2: 'a,
        {
            type Item = K;

            #[inline]
            fn next(&mut self) -> Option<K> {
                self.0.next()
            }
        }

        impl<'a, K: DirectKey<Tag = ()>, const PAGE: usize, D: Directory, D2: Directory>
            FusedIterator for $name<'a, K, PAGE, D, D2>
        where
            D: 'a,
            D2: 'a,
        {
        }

        impl<'a, K, const PAGE: usize, D: Directory, D2: Directory> Clone
            for $name<'a, K, PAGE, D, D2>
        where
            D: 'a,
            D2: 'a,
        {
            fn clone(&self) -> Self {
                Self(self.0.clone())
            }
        }

        impl<'a, K, const PAGE: usize, D: Directory, D2: Directory> fmt::Debug
            for $name<'a, K, PAGE, D, D2>
        where
            D: 'a,
            D2: 'a,
        {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct(stringify!($name)).finish_non_exhaustive()
            }
        }
    )*};
}

algebra_iterator! {
    /// Iterator over the keys in either of two [`PagedBitSet`]s, in ascending
    /// slot-index order, from [`PagedBitSet::union`].
    Union,
    /// Iterator over the keys in both of two [`PagedBitSet`]s, in ascending
    /// slot-index order, from [`PagedBitSet::intersection`].
    Intersection,
    /// Iterator over the keys in one [`PagedBitSet`] and not another, in
    /// ascending slot-index order, from [`PagedBitSet::difference`].
    Difference,
    /// Iterator over the keys in exactly one of two [`PagedBitSet`]s, in
    /// ascending slot-index order, from [`PagedBitSet::symmetric_difference`].
    SymmetricDifference,
}

impl<K: DirectKey<Tag = ()>, const PAGE: usize, D: Directory> PagedBitSet<K, PAGE, D> {
    /// Returns whether the two sets have no key in common.
    pub fn is_disjoint<D2: Directory>(&self, other: &PagedBitSet<K, PAGE, D2>) -> bool {
        self.pages.pages().all(|(number, left)| {
            other
                .pages
                .get(number)
                .is_none_or(|right| left.iter().zip(right).all(|(a, b)| a & b == 0))
        })
    }

    /// Returns whether every key in this set is also in `other`.
    pub fn is_subset<D2: Directory>(&self, other: &PagedBitSet<K, PAGE, D2>) -> bool {
        self.len <= other.len
            && self.pages.pages().all(|(number, left)| {
                if let Some(right) = other.pages.get(number) {
                    left.iter().zip(right).all(|(a, b)| a & !b == 0)
                } else {
                    left.iter().all(|&word| word == 0)
                }
            })
    }

    /// Returns whether every key in `other` is also in this set.
    pub fn is_superset<D2: Directory>(&self, other: &PagedBitSet<K, PAGE, D2>) -> bool {
        other.is_subset(self)
    }

    /// Iterates over keys in either set, in ascending slot-index order.
    ///
    /// ```
    /// use paged_collections::bit_set::PagedBitSet;
    /// use paged_collections::page::Tree;
    ///
    /// let a: PagedBitSet<usize, 2> = [1, 5].into_iter().collect();
    /// let b: PagedBitSet<usize, 2, Tree> = [2, 5].into_iter().collect();
    /// assert_eq!(a.union(&b).collect::<Vec<_>>(), [1, 2, 5]);
    /// assert_eq!((&a & &b).iter().collect::<Vec<_>>(), [5]);
    /// ```
    pub fn union<'a, D2: Directory>(
        &'a self,
        other: &'a PagedBitSet<K, PAGE, D2>,
    ) -> Union<'a, K, PAGE, D, D2> {
        Union(self.algebra(other, Algebra::Union))
    }

    /// Iterates over keys in both sets, in ascending slot-index order.
    pub fn intersection<'a, D2: Directory>(
        &'a self,
        other: &'a PagedBitSet<K, PAGE, D2>,
    ) -> Intersection<'a, K, PAGE, D, D2> {
        Intersection(self.algebra(other, Algebra::Intersection))
    }

    /// Iterates over keys in this set that are absent from `other`, in
    /// ascending slot-index order.
    pub fn difference<'a, D2: Directory>(
        &'a self,
        other: &'a PagedBitSet<K, PAGE, D2>,
    ) -> Difference<'a, K, PAGE, D, D2> {
        Difference(self.algebra(other, Algebra::Difference))
    }

    /// Iterates over keys in exactly one set, in ascending slot-index order.
    pub fn symmetric_difference<'a, D2: Directory>(
        &'a self,
        other: &'a PagedBitSet<K, PAGE, D2>,
    ) -> SymmetricDifference<'a, K, PAGE, D, D2> {
        SymmetricDifference(self.algebra(other, Algebra::SymmetricDifference))
    }

    fn algebra<'a, D2: Directory>(
        &'a self,
        other: &'a PagedBitSet<K, PAGE, D2>,
        op: Algebra,
    ) -> AlgebraIter<'a, K, PAGE, D, D2> {
        AlgebraIter {
            pages: PageMerge::new(self.pages.pages(), other.pages.pages(), op),
            words: None,
            bits: WordBits::default(),
            key: PhantomData,
        }
    }

    fn combined<D2: Directory>(&self, other: &PagedBitSet<K, PAGE, D2>, op: Algebra) -> Self {
        let mut result = Self::new();
        for words in PageMerge::<PAGE, D, D2>::new(self.pages.pages(), other.pages.pages(), op) {
            let number = words.number;
            let mut nonzero = words.filter(|&(_, word)| word != 0);
            if let Some(first) = nonzero.next() {
                let page = result.pages.get_or_alloc(number);
                for (offset, word) in once(first).chain(nonzero) {
                    write_word(&mut result.len, &mut page[offset], word);
                }
            }
        }
        result
    }

    fn assign<D2: Directory>(&mut self, other: &PagedBitSet<K, PAGE, D2>, op: Algebra) {
        match op {
            Algebra::Union | Algebra::SymmetricDifference => {
                for (number, right) in other.pages.pages() {
                    if right.iter().all(|&word| word == 0) {
                        continue;
                    }
                    let left = self.pages.get_or_alloc(number);
                    for (word, &right) in left.iter_mut().zip(right) {
                        let new = op.word(*word, right);
                        write_word(&mut self.len, word, new);
                    }
                }
            }
            Algebra::Intersection | Algebra::Difference => {
                for (number, left) in self.pages.pages_mut() {
                    let right = other.pages.get(number);
                    if right.is_none() && matches!(op, Algebra::Difference) {
                        continue;
                    }
                    for (offset, word) in left.iter_mut().enumerate() {
                        let new = op.word(*word, right.map_or(0, |page| page[offset]));
                        write_word(&mut self.len, word, new);
                    }
                }
            }
        }
    }
}

macro_rules! impl_algebra {
    ($binary:ident, $binary_fn:ident, $assign:ident, $assign_fn:ident, $op:ident) => {
        impl<K: DirectKey<Tag = ()>, const PAGE: usize, D: Directory, D2: Directory>
            ops::$binary<&PagedBitSet<K, PAGE, D2>> for &PagedBitSet<K, PAGE, D>
        {
            type Output = PagedBitSet<K, PAGE, D>;

            fn $binary_fn(self, other: &PagedBitSet<K, PAGE, D2>) -> Self::Output {
                self.combined(other, Algebra::$op)
            }
        }

        impl<K: DirectKey<Tag = ()>, const PAGE: usize, D: Directory, D2: Directory>
            ops::$assign<&PagedBitSet<K, PAGE, D2>> for PagedBitSet<K, PAGE, D>
        {
            fn $assign_fn(&mut self, other: &PagedBitSet<K, PAGE, D2>) {
                self.assign(other, Algebra::$op);
            }
        }
    };
}

impl_algebra!(BitOr, bitor, BitOrAssign, bitor_assign, Union);
impl_algebra!(BitAnd, bitand, BitAndAssign, bitand_assign, Intersection);
impl_algebra!(Sub, sub, SubAssign, sub_assign, Difference);
impl_algebra!(
    BitXor,
    bitxor,
    BitXorAssign,
    bitxor_assign,
    SymmetricDifference
);

/// Compares the set bits, ignoring empty pages and the directory layout.
impl<K: DirectKey<Tag = ()>, const PAGE: usize, D: Directory, D2: Directory>
    PartialEq<PagedBitSet<K, PAGE, D2>> for PagedBitSet<K, PAGE, D>
{
    fn eq(&self, other: &PagedBitSet<K, PAGE, D2>) -> bool {
        self.len == other.len && self.is_subset(other)
    }
}

impl<K: DirectKey<Tag = ()>, const PAGE: usize, D: Directory> Eq for PagedBitSet<K, PAGE, D> {}

/// Hashes the length and the nonzero words in ascending word-index order.
impl<K: DirectKey<Tag = ()>, const PAGE: usize, D: Directory> Hash for PagedBitSet<K, PAGE, D> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.len.hash(state);
        for (number, page) in self.pages.pages() {
            let start = page_start::<PAGE>(number);
            for (offset, &word) in page.iter().enumerate() {
                if word != 0 {
                    (start + offset, word).hash(state);
                }
            }
        }
    }
}

#[cfg(feature = "rayon")]
pub use rayon_impl::ParIter;

#[cfg(feature = "rayon")]
mod rayon_impl {
    use super::*;
    use rayon::iter::plumbing::UnindexedConsumer;
    use rayon::prelude::*;

    /// An unindexed parallel iterator over the keys of a [`PagedBitSet`].
    ///
    /// Iteration splits by page. Collecting into a `Vec` preserves
    /// slot-index order.
    pub struct ParIter<'a, K, const PAGE: usize = DEFAULT_BIT_PAGE> {
        pages: Vec<Page<'a, PAGE>>,
        key: PhantomData<fn() -> K>,
    }

    impl<'a, K: DirectKey<Tag = ()> + Send, const PAGE: usize, D: Directory> IntoParallelIterator
        for &'a PagedBitSet<K, PAGE, D>
    {
        type Item = K;
        type Iter = ParIter<'a, K, PAGE>;

        fn into_par_iter(self) -> Self::Iter {
            ParIter {
                pages: self.pages.pages().collect(),
                key: PhantomData,
            }
        }
    }

    impl<K: DirectKey<Tag = ()> + Send, const PAGE: usize> ParallelIterator for ParIter<'_, K, PAGE> {
        type Item = K;

        fn drive_unindexed<C: UnindexedConsumer<K>>(self, consumer: C) -> C::Result {
            self.pages
                .into_par_iter()
                .flat_map_iter(|(number, page)| {
                    PageWords::new(number, page, Span::FULL)
                        .flatten()
                        .map(|index| K::from_parts(index, ()))
                })
                .drive_unindexed(consumer)
        }
    }
}

#[cfg(feature = "serde")]
mod serde_impl {
    use super::*;
    use serde::de::{Deserialize, Deserializer, SeqAccess, Visitor};
    use serde::ser::{Serialize, Serializer};

    /// Serializes as a sequence of keys in ascending slot-index order.
    impl<K: DirectKey<Tag = ()> + Serialize, const PAGE: usize, D: Directory> Serialize
        for PagedBitSet<K, PAGE, D>
    {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            serializer.collect_seq(self.iter())
        }
    }

    /// Deserializes from a sequence of keys, skipping duplicates.
    ///
    /// Memory follows the key indices. With the [`Flat`] directory one large
    /// index allocates a directory entry for every page below it. For
    /// untrusted input, use [`Radix`](crate::page::Radix) or
    /// [`Tree`](crate::page::Tree), or validate the keys first.
    impl<'de, K: DirectKey<Tag = ()> + Deserialize<'de>, const PAGE: usize, D: Directory>
        Deserialize<'de> for PagedBitSet<K, PAGE, D>
    {
        fn deserialize<De: Deserializer<'de>>(deserializer: De) -> Result<Self, De::Error> {
            struct SeqVisitor<K, const PAGE: usize, D>(PhantomData<(K, D)>);

            impl<'de, K: DirectKey<Tag = ()> + Deserialize<'de>, const PAGE: usize, D: Directory>
                Visitor<'de> for SeqVisitor<K, PAGE, D>
            {
                type Value = PagedBitSet<K, PAGE, D>;

                fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    f.write_str("a sequence of keys")
                }

                fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                    let mut set = PagedBitSet::new();
                    while let Some(key) = seq.next_element()? {
                        set.insert(key);
                    }
                    Ok(set)
                }
            }

            deserializer.deserialize_seq(SeqVisitor::<K, PAGE, D>(PhantomData))
        }
    }
}
