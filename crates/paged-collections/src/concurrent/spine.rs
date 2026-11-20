//! Stable pages, published once and reclaimed only under exclusive access.

use alloc::boxed::Box;
use core::marker::PhantomData;
use core::ops::Range;
use core::{ptr, slice};

use super::atomic::{AtomicPtr, Ordering};

#[cfg(not(loom))]
const BASE: usize = 32;
#[cfg(loom)]
const BASE: usize = 2;

pub(crate) struct Spine<P, const CHUNKS: usize> {
    chunks: [AtomicPtr<AtomicPtr<P>>; CHUNKS],
    owner: PhantomData<Box<P>>,
}

// SAFETY: the directory owns its pages; moving it transfers that ownership,
// and no page reference can outlive the borrow of the directory.
unsafe impl<P: Send, const CHUNKS: usize> Send for Spine<P, CHUNKS> {}
// SAFETY: shared operations only publish immutable ownership links; pages
// are shared as &P, and a losing installer may drop its own unpublished P.
unsafe impl<P: Send + Sync, const CHUNKS: usize> Sync for Spine<P, CHUNKS> {}

impl<P, const CHUNKS: usize> Spine<P, CHUNKS> {
    #[cfg(not(loom))]
    pub(crate) const fn new() -> Self {
        Self {
            chunks: [const { AtomicPtr::new(ptr::null_mut()) }; CHUNKS],
            owner: PhantomData,
        }
    }

    #[cfg(loom)]
    pub(crate) fn new() -> Self {
        Self {
            chunks: core::array::from_fn(|_| AtomicPtr::new(ptr::null_mut())),
            owner: PhantomData,
        }
    }

    fn location(page: usize) -> Option<(usize, usize)> {
        // Equivalent to log2(page + BASE), without overflowing near usize::MAX.
        let group = (page / BASE).checked_add(1)?;
        let k = group.ilog2() as usize;
        if k >= CHUNKS {
            return None;
        }
        let size = BASE.checked_mul(1usize.checked_shl(k as u32)?)?;
        Some((k, page - (size - BASE)))
    }

    /// Whether a page number fits the directory and its chunk fits in an allocation.
    pub(super) fn can_allocate_page(page: usize) -> bool {
        Self::location(page).is_some_and(|(k, _)| {
            (BASE << k) <= isize::MAX as usize / core::mem::size_of::<AtomicPtr<P>>()
        })
    }

    pub(crate) fn get(&self, page: usize) -> Option<&P> {
        let (k, offset) = Self::location(page)?;
        // Acquire pairs with chunk publication, including all initialized cells.
        let chunk = self.chunks[k].load(Ordering::Acquire);
        if chunk.is_null() {
            return None;
        }
        // SAFETY: an installed chunk has BASE << k initialized cells, offset
        // is within it, and chunks cannot be freed during this shared borrow.
        let cell = unsafe { &*chunk.wrapping_add(offset) };
        // Acquire pairs with page publication, making its initialization visible.
        let page = cell.load(Ordering::Acquire);
        // SAFETY: non-null installed pointers own initialized P values which
        // remain allocated for the entire shared borrow of this directory.
        unsafe { page.as_ref() }
    }

    pub(crate) fn get_or_insert_with(&self, page: usize, make: impl FnOnce() -> Box<P>) -> &P {
        let (k, offset) = Self::location(page).expect("Spine: page exceeds directory capacity");
        // Acquire observes the initialization published by another installer.
        let mut chunk = self.chunks[k].load(Ordering::Acquire);
        if chunk.is_null() {
            let cells: Box<[_]> = (0..(BASE << k))
                .map(|_| AtomicPtr::<P>::default())
                .collect();
            let raw = Box::into_raw(cells) as *mut AtomicPtr<P>;
            // Release publishes all cells; Acquire on either outcome observes
            // the winner's initialized chunk. Only null -> non-null is shared.
            chunk = match self.chunks[k].compare_exchange(
                ptr::null_mut(),
                raw,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => raw,
                Err(winner) => {
                    let slice = ptr::slice_from_raw_parts_mut(raw, BASE << k);
                    // SAFETY: this losing allocation was never published; its
                    // pointer and length are exactly those of the original Box.
                    drop(unsafe { Box::from_raw(slice) });
                    winner
                }
            };
        }
        // SAFETY: the initialized chunk is installed, offset is in bounds,
        // and shared borrows prevent its reclamation.
        let cell = unsafe { &*chunk.wrapping_add(offset) };
        // Acquire observes any page already published in this cell.
        let mut installed = cell.load(Ordering::Acquire);
        if installed.is_null() {
            let raw = Box::into_raw(make());
            // Release publishes P; Acquire observes the winning P on failure.
            installed = match cell.compare_exchange(
                ptr::null_mut(),
                raw,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => raw,
                Err(winner) => {
                    // SAFETY: the losing Box<P> was never published and this
                    // thread still owns its sole allocation pointer.
                    drop(unsafe { Box::from_raw(raw) });
                    winner
                }
            };
        }
        // SAFETY: either CAS outcome supplies an initialized, installed P;
        // installed pages cannot be reclaimed during this shared borrow.
        unsafe { &*installed }
    }

    /// Installed pages in ascending order, weakly consistent across pages.
    pub(crate) fn iter(&self) -> Iter<'_, P, CHUNKS> {
        Iter {
            spine: self,
            next_chunk: 0,
            pages: 0..0,
        }
    }

    /// Installed pages in ascending order, with exclusive access to each.
    pub(crate) fn iter_mut(&mut self) -> IterMut<'_, P> {
        IterMut {
            chunks: self.chunks.iter_mut().enumerate(),
            cells: slice::IterMut::default().enumerate(),
            base: 0,
            marker: PhantomData,
        }
    }

    /// Takes page `page` out of the directory under exclusive access. Its
    /// chunk stays allocated. A later shared `get_or_insert_with` may install
    /// a new page there.
    pub(crate) fn remove(&mut self, page: usize) -> Option<Box<P>> {
        let (k, offset) = Self::location(page)?;
        // Relaxed suffices: exclusive access rules out readers/installers.
        let chunk = self.chunks[k].load(Ordering::Relaxed);
        if chunk.is_null() {
            return None;
        }
        // SAFETY: an installed chunk has BASE << k initialized cells, offset
        // is within it, and &mut self excludes every other access.
        let cell = unsafe { &*chunk.wrapping_add(offset) };
        // Relaxed suffices under the directory's exclusive borrow.
        let raw = cell.swap(ptr::null_mut(), Ordering::Relaxed);
        if raw.is_null() {
            return None;
        }
        // SAFETY: a non-null cell uniquely owns one Box<P>; the swap above
        // unlinked it, and exclusive access ends every borrow of it.
        Some(unsafe { Box::from_raw(raw) })
    }

    /// Frees every page and chunk, including after one page destructor panics.
    pub(crate) fn clear(&mut self) {
        Reclaim {
            chunks: &mut self.chunks,
            next_chunk: 0,
            cells: None,
            next_page: 0,
        }
        .drain();
    }
}

/// Owns the detached chunk and tracks progress before each page drop.
struct Reclaim<'a, P> {
    chunks: &'a mut [AtomicPtr<AtomicPtr<P>>],
    next_chunk: usize,
    cells: Option<Box<[AtomicPtr<P>]>>,
    next_page: usize,
}

impl<P> Reclaim<'_, P> {
    fn drain(&mut self) {
        loop {
            if let Some(cells) = &self.cells {
                while let Some(cell) = cells.get(self.next_page) {
                    self.next_page += 1;
                    // Relaxed suffices under exclusive access.
                    let page = cell.load(Ordering::Relaxed);
                    if !page.is_null() {
                        // SAFETY: this cell owns a distinct Box<P> in the
                        // detached chunk. Advancing first prevents a second
                        // drop of this page if its destructor panics.
                        drop(unsafe { Box::from_raw(page) });
                    }
                }
                self.cells = None;
            }
            let k = self.next_chunk;
            let Some(cell) = self.chunks.get_mut(k) else {
                break;
            };
            self.next_chunk += 1;
            // Relaxed suffices: exclusive access rules out readers/installers.
            let raw = cell.swap(ptr::null_mut(), Ordering::Relaxed);
            if !raw.is_null() {
                let slice = ptr::slice_from_raw_parts_mut(raw, BASE << k);
                // SAFETY: the swap transferred the sole ownership of this
                // allocation, which contains exactly BASE << k cells.
                self.cells = Some(unsafe { Box::from_raw(slice) });
                self.next_page = 0;
            }
        }
    }
}

impl<P> Drop for Reclaim<'_, P> {
    fn drop(&mut self) {
        self.drain();
    }
}

/// Shared traversal over installed spine pages.
pub(crate) struct Iter<'a, P, const CHUNKS: usize> {
    spine: &'a Spine<P, CHUNKS>,
    next_chunk: usize,
    pages: Range<usize>,
}

impl<'a, P, const CHUNKS: usize> Iterator for Iter<'a, P, CHUNKS> {
    type Item = (usize, &'a P);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(page) = self
                .pages
                .find_map(|number| self.spine.get(number).map(|page| (number, page)))
            {
                return Some(page);
            }
            let k = self.next_chunk;
            let cell = self.spine.chunks.get(k)?;
            self.next_chunk += 1;
            // Acquire observes publication; chunks remain installed while shared.
            if !cell.load(Ordering::Acquire).is_null() {
                let len = BASE << k;
                let base = len - BASE;
                self.pages = base..base + len;
            }
        }
    }
}

/// Exclusive traversal over installed spine pages.
pub(crate) struct IterMut<'a, P> {
    chunks: core::iter::Enumerate<slice::IterMut<'a, AtomicPtr<AtomicPtr<P>>>>,
    cells: core::iter::Enumerate<slice::IterMut<'a, AtomicPtr<P>>>,
    base: usize,
    marker: PhantomData<&'a mut P>,
}

impl<'a, P> Iterator for IterMut<'a, P> {
    type Item = (usize, &'a mut P);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            for (offset, cell) in self.cells.by_ref() {
                // Relaxed suffices under the directory's exclusive borrow.
                let page = cell.load(Ordering::Relaxed);
                // SAFETY: installed pages are distinct owned Boxes. Exclusive
                // iteration visits each cell once and prevents shared access.
                if let Some(page) = unsafe { page.as_mut() } {
                    return Some((self.base + offset, page));
                }
            }
            let (k, cell) = self.chunks.next()?;
            let chunk = cell.load(Ordering::Relaxed);
            if !chunk.is_null() {
                let len = BASE << k;
                self.base = len - BASE;
                // SAFETY: this installed allocation contains len cells. The
                // exclusive directory borrow and one visit per chunk make
                // this the only slice borrowing these cells.
                let cells = unsafe { slice::from_raw_parts_mut(chunk, len) };
                self.cells = cells.iter_mut().enumerate();
            }
        }
    }
}

impl<P, const CHUNKS: usize> Drop for Spine<P, CHUNKS> {
    fn drop(&mut self) {
        self.clear();
    }
}
