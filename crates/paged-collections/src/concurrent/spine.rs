//! Stable pages, published once and reclaimed only under exclusive access.

use alloc::boxed::Box;
use core::marker::PhantomData;
use core::ptr;

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
    pub(crate) fn iter(&self) -> impl Iterator<Item = (usize, &P)> {
        self.chunks.iter().enumerate().flat_map(move |(k, cell)| {
            // Acquire observes publication; chunks remain installed while shared.
            let len = if cell.load(Ordering::Acquire).is_null() {
                0
            } else {
                BASE << k
            };
            (0..len).filter_map(move |offset| {
                let number = (BASE << k) - BASE + offset;
                self.get(number).map(|page| (number, page))
            })
        })
    }

    /// Installed pages in ascending order, with exclusive access to each.
    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = (usize, &mut P)> {
        // The adapter below only stores atomic pointers, which are `Send`
        // and `Sync` whatever `P` is. The marker ties the iterator's auto
        // traits to `&mut P`, which it hands out.
        ExclusivePages {
            inner: self.iter_mut_raw(),
            marker: PhantomData,
        }
    }

    fn iter_mut_raw(&mut self) -> impl Iterator<Item = (usize, &mut P)> {
        self.chunks.iter_mut().enumerate().flat_map(|(k, cell)| {
            // Relaxed suffices: exclusive access rules out publication races.
            let chunk = cell.load(Ordering::Relaxed);
            let len = if chunk.is_null() { 0 } else { BASE << k };
            let cells = if len == 0 {
                &mut [][..]
            } else {
                // SAFETY: this installed allocation contains len cells; &mut
                // self excludes every other reference into its pages or cells.
                unsafe { core::slice::from_raw_parts_mut(chunk, len) }
            };
            cells
                .iter_mut()
                .enumerate()
                .filter_map(move |(offset, cell)| {
                    // Relaxed suffices under the directory's exclusive borrow.
                    let page = cell.load(Ordering::Relaxed);
                    // SAFETY: installed pages are distinct owned Boxes; exclusive
                    // iteration visits each cell once and prevents shared access.
                    unsafe { page.as_mut() }.map(|page| ((BASE << k) - BASE + offset, page))
                })
        })
    }

    /// Frees every page and chunk. If a page's destructor panics, the
    /// remaining pages of that chunk are leaked (safe, not unsound).
    pub(crate) fn clear(&mut self) {
        for (k, cell) in self.chunks.iter_mut().enumerate() {
            // Relaxed suffices: exclusive access rules out readers/installers.
            let raw = cell.swap(ptr::null_mut(), Ordering::Relaxed);
            if raw.is_null() {
                continue;
            }
            let slice = ptr::slice_from_raw_parts_mut(raw, BASE << k);
            // SAFETY: this cell uniquely owned the installed chunk with exactly
            // BASE << k entries; exclusive access permits taking it back.
            let cells = unsafe { Box::from_raw(slice) };
            for cell in cells.iter() {
                // Relaxed suffices: no concurrent access exists during clear.
                let page = cell.load(Ordering::Relaxed);
                if !page.is_null() {
                    // SAFETY: every non-null cell uniquely owns one Box<P>,
                    // no page is aliased, and exclusive access ends all borrows.
                    drop(unsafe { Box::from_raw(page) });
                }
            }
        }
    }
}

/// Carries `&mut P`'s auto traits for [`Spine::iter_mut`].
struct ExclusivePages<'a, P, I> {
    inner: I,
    marker: PhantomData<&'a mut P>,
}

impl<'a, P: 'a, I: Iterator<Item = (usize, &'a mut P)>> Iterator for ExclusivePages<'a, P, I> {
    type Item = (usize, &'a mut P);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<P, const CHUNKS: usize> Drop for Spine<P, CHUNKS> {
    fn drop(&mut self) {
        self.clear();
    }
}
