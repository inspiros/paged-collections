//! Append ownership, per-slot publication, and a helped, hole-free watermark.
//!
//! Every position below committed is initialized and ready. Ready positions
//! above it are independently readable. With no push in flight, committed ==
//! reserved. Removal and moving values require exclusive access.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::iter::FusedIterator;
use core::mem::MaybeUninit;
use core::ptr;

use super::atomic::{AtomicU64, AtomicUsize, Ordering, UnsafeCell};
use super::spine::Spine;
use crate::paged_vec::PagedVec;

// Enough directory levels for usize positions on either target and with
// loom's smaller spine base. Unrepresentable allocations fail before claiming.
const CHUNKS: usize = usize::BITS as usize;

struct Page<T> {
    ready: Box<[AtomicU64]>,
    slots: Box<[UnsafeCell<MaybeUninit<T>>]>,
}

impl<T> Page<T> {
    fn new(capacity: usize) -> Self {
        Self {
            ready: (0..capacity.div_ceil(64))
                .map(|_| AtomicU64::new(0))
                .collect(),
            slots: (0..capacity)
                .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
                .collect(),
        }
    }

    fn ready(&self, offset: usize, ordering: Ordering) -> bool {
        // Callers pass Acquire, which observes the value write published by
        // the ready RMW.
        self.ready[offset / 64].load(ordering) & (1 << (offset % 64)) != 0
    }

    fn drop_ready(&mut self) {
        // Only called while the page is destroyed: skip the scan entirely
        // for element types without drop glue.
        if !core::mem::needs_drop::<T>() {
            return;
        }
        for (offset, slot) in self.slots.iter_mut().enumerate() {
            let mask = 1 << (offset % 64);
            // Relaxed: exclusive destruction. Clear before user drop code so
            // unwinding cleanup cannot drop the same value again.
            if self.ready[offset / 64].fetch_and(!mask, Ordering::Relaxed) & mask != 0 {
                slot.with_mut(|cell| {
                    // SAFETY: the ready bit owns one initialized value;
                    // exclusive page destruction excludes readers and writers.
                    unsafe { ptr::drop_in_place(cell.cast::<T>()) };
                });
            }
        }
    }
}

impl<T> Drop for Page<T> {
    fn drop(&mut self) {
        // Continue through remaining ready values if one destructor panics.
        struct Rest<'a, T>(&'a mut Page<T>);
        impl<T> Drop for Rest<'_, T> {
            fn drop(&mut self) {
                self.0.drop_ready();
            }
        }
        let rest = Rest(self);
        rest.0.drop_ready();
        core::mem::forget(rest);
    }
}

/// Stable append storage with independently visible slots and a committed prefix.
pub(crate) struct AppendLog<T, const PAGE: usize> {
    pages: Spine<Page<T>, CHUNKS>,
    reserved: AtomicUsize,
    committed: AtomicUsize,
    /// Only its chain of read-modify-writes matters; the value may wrap.
    sync: AtomicUsize,
}

// SAFETY: moving the log transfers its owned cells, including their drops;
// references cannot outlive the borrow from which they were obtained.
unsafe impl<T: Send, const PAGE: usize> Send for AppendLog<T, PAGE> {}
// SAFETY: claims uniquely own writes, ready/committed acquire publication,
// and shared access never moves or drops values. Losing allocations may drop
// on another thread; sharing initialized values also requires Sync.
unsafe impl<T: Send + Sync, const PAGE: usize> Sync for AppendLog<T, PAGE> {}

impl<T, const PAGE: usize> AppendLog<T, PAGE> {
    pub(crate) fn new() -> Self {
        assert!(
            PAGE > 0
                && PAGE
                    <= isize::MAX as usize
                        / core::mem::size_of::<UnsafeCell<MaybeUninit<T>>>().max(1)
                && PAGE.div_ceil(64) <= isize::MAX as usize / core::mem::size_of::<AtomicU64>(),
            "AppendLog: invalid page capacity"
        );
        Self {
            pages: Spine::new(),
            reserved: AtomicUsize::new(0),
            committed: AtomicUsize::new(0),
            sync: AtomicUsize::new(0),
        }
    }

    fn claim(&self, limit: usize) -> Option<(usize, &Page<T>)> {
        // Acquire: a claim only needs the latest count it can observe; the
        // CAS below decides ownership.
        let mut pos = self.reserved.load(Ordering::Acquire);
        loop {
            if pos >= limit {
                return None;
            }
            // Allocate BEFORE claiming: capacity/allocation panics cannot
            // strand a hole. No user code runs between a claim and ready;
            // values are passed in, and all storage is already initialized.
            let page = self
                .pages
                .get_or_insert_with(pos / PAGE, || Box::new(Page::new(PAGE)));
            // AcqRel: the winner uniquely owns `pos`. Failure acquires the
            // intervening reservations before trying their next position.
            match self
                .reserved
                .compare_exchange(pos, pos + 1, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => return Some((pos, page)),
                Err(actual) => pos = actual,
            }
        }
    }

    fn publish(&self, pos: usize, page: &Page<T>, value: T) {
        let offset = pos % PAGE;
        page.slots[offset].with_mut(|cell| {
            // SAFETY: the reservation CAS uniquely owns this uninitialized
            // slot; shared operations never overwrite an initialized slot.
            unsafe { ptr::write(cell.cast::<T>(), value) };
        });
        // Release publishes the write to anyone who acquires this bit, or
        // acquires something that follows it (the sync in `help`, a sparse
        // slot stored after this push).
        page.ready[offset / 64].fetch_or(1 << (offset % 64), Ordering::Release);
        self.help();
    }

    fn help(&self) {
        // Every publisher does one read-modify-write on `sync` AFTER
        // setting its ready bit and BEFORE scanning. These RMWs are totally
        // ordered and each acquires the previous one, so the publisher whose
        // sync comes last has acquired every other publisher's page install,
        // write and ready bit. Its scan below therefore cannot stop early:
        // with no push in flight, committed == reserved.
        //
        // This replaces the textbook "set my bit, then read yours" check,
        // which is a store-buffering pattern that only SeqCst forbids (and
        // that loom cannot model). The RMW chain needs AcqRel only.
        //
        // A separate counter that really changes, not `fetch_add(0)` on the
        // watermark: compilers may weaken an RMW that leaves its value
        // unchanged.
        self.sync.fetch_add(1, Ordering::AcqRel);
        loop {
            // Acquire: sees every slot below the returned watermark.
            let c = self.committed.load(Ordering::Acquire);
            // Acquire observes the write behind each set bit; a missing page
            // is not ready. Advancing over the whole ready run at once costs
            // one contended CAS per run rather than per position.
            let mut end = c;
            while self.ready(end, Ordering::Acquire) {
                end += 1;
            }
            if end == c {
                break;
            }
            // AcqRel publishes the extended prefix and carries what this
            // helper acquired (the bits in c..end) to later readers of
            // `committed`. On failure another helper advanced: rescan.
            let _ = self
                .committed
                .compare_exchange(c, end, Ordering::AcqRel, Ordering::Acquire);
        }
    }

    pub(crate) fn push(&self, value: T, limit: usize) -> Result<usize, T> {
        let Some((pos, page)) = self.claim(limit) else {
            return Err(value);
        };
        self.publish(pos, page, value);
        Ok(pos)
    }

    /// Claims storage before taking a pending value from the caller.
    pub(super) fn push_from(&self, value: &mut Option<T>, limit: usize) -> Option<usize> {
        assert!(value.is_some(), "missing pending append");
        let (pos, page) = self.claim(limit)?;
        self.publish(pos, page, value.take().expect("pending append"));
        Some(pos)
    }

    /// Appends all `values` at consecutive positions with one reservation
    /// and one helping pass, returning the first position. The caller
    /// retains the values if storage preparation panics or the limit would
    /// be exceeded. An empty batch reserves nothing.
    pub(super) fn push_many_from(&self, values: &mut Vec<T>, limit: usize) -> Option<usize> {
        let n = values.len();
        // Acquire: as in `claim`, only the CAS decides ownership.
        let mut pos = self.reserved.load(Ordering::Acquire);
        if n == 0 {
            return Some(pos);
        }
        loop {
            if pos > limit || limit - pos < n {
                return None;
            }
            // Allocate every page of the block BEFORE claiming it: capacity or
            // allocation panics cannot strand holes. Pages installed for a
            // lost CAS stay for later claims.
            for page in pos / PAGE..=(pos + n - 1) / PAGE {
                self.pages
                    .get_or_insert_with(page, || Box::new(Page::new(PAGE)));
            }
            // AcqRel: the winner uniquely owns pos..pos + n.
            match self
                .reserved
                .compare_exchange(pos, pos + n, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => break,
                Err(actual) => pos = actual,
            }
        }
        // No user code runs from the claim until every slot is ready: moving
        // out of a Vec and raw writes cannot panic.
        for (i, value) in values.drain(..).enumerate() {
            let at = pos + i;
            let page = self.pages.get(at / PAGE).expect("claimed page installed");
            page.slots[at % PAGE].with_mut(|cell| {
                // SAFETY: the block reservation uniquely owns this
                // uninitialized slot; shared operations never overwrite an
                // initialized slot.
                unsafe { ptr::write(cell.cast::<T>(), value) };
            });
        }
        let (mut at, end) = (pos, pos + n);
        while at < end {
            let page = self.pages.get(at / PAGE).expect("claimed page installed");
            let offset = at % PAGE;
            let bit = offset % 64;
            // Bits of this word within both the block and the page.
            let count = (64 - bit).min(end - at).min(PAGE - offset);
            let mask = if count == 64 {
                u64::MAX
            } else {
                ((1u64 << count) - 1) << bit
            };
            // Release publishes these writes, as the single-bit RMW in
            // `publish` does for one.
            page.ready[offset / 64].fetch_or(mask, Ordering::Release);
            at += count;
        }
        self.help();
        Some(pos)
    }

    #[cfg(all(test, loom))]
    pub(super) fn push_many(&self, mut values: Vec<T>, limit: usize) -> Result<usize, Vec<T>> {
        self.push_many_from(&mut values, limit).ok_or(values)
    }

    fn ready(&self, pos: usize, ordering: Ordering) -> bool {
        self.pages
            .get(pos / PAGE)
            .is_some_and(|page| page.ready(pos % PAGE, ordering))
    }

    pub(crate) fn len(&self) -> usize {
        // Acquire observes every write in the prefix, transitively through
        // the helpers' acquiring and releasing watermark CASes.
        self.committed.load(Ordering::Acquire)
    }

    pub(crate) fn reserved_len(&self) -> usize {
        // Relaxed: claims only; this does not grant access to any value.
        self.reserved.load(Ordering::Relaxed)
    }

    pub(crate) fn get(&self, pos: usize) -> Option<&T> {
        let prefix = self.len();
        let page = self.pages.get(pos / PAGE)?;
        // Acquire independently observes a finished push above the prefix.
        if pos >= prefix && !page.ready(pos % PAGE, Ordering::Acquire) {
            return None;
        }
        Some(page.slots[pos % PAGE].with(|cell| {
            // SAFETY: acquiring committed or this slot's ready bit proves
            // initialization. Shared access cannot remove or modify this T.
            unsafe { &*cell.cast::<T>() }
        }))
    }

    pub(crate) fn iter(
        &self,
    ) -> impl DoubleEndedIterator<Item = &T> + ExactSizeIterator + FusedIterator {
        let len = self.len();
        (0..len).map(move |pos| self.get(pos).expect("committed slot is ready"))
    }

    #[cfg(not(loom))]
    pub(crate) fn pages(
        &self,
    ) -> impl DoubleEndedIterator<Item = &[T]> + ExactSizeIterator + FusedIterator {
        let len = self.len();
        (0..len.div_ceil(PAGE)).map(move |p| {
            let page = self.pages.get(p).expect("committed page exists");
            let used = PAGE.min(len - p * PAGE);
            // SAFETY: the acquired prefix initializes exactly these used
            // slots. The production shim, UnsafeCell and MaybeUninit are
            // transparent, so the boxed array has T's stride/alignment.
            // Derive the pointer from the whole allocation, not one cell.
            unsafe { core::slice::from_raw_parts(page.slots.as_ptr().cast::<T>(), used) }
        })
    }

    // Loom's cells contain tracking metadata and are not contiguous T. Visit
    // the SAME snapshotted prefix as singleton slices, checking every cell's
    // publication through the shim instead of reinterpreting that metadata.
    #[cfg(loom)]
    pub(crate) fn pages(
        &self,
    ) -> impl DoubleEndedIterator<Item = &[T]> + ExactSizeIterator + FusedIterator {
        self.iter().map(core::slice::from_ref)
    }

    pub(crate) fn get_mut(&mut self, pos: usize) -> Option<&mut T> {
        if pos >= self.len() {
            return None;
        }
        let page = self.pages.get(pos / PAGE).expect("committed page exists");
        Some(page.slots[pos % PAGE].with_mut(|cell| {
            // SAFETY: exclusivity implies quiescence (committed == reserved),
            // initialization, and no outstanding references to this slot.
            unsafe { &mut *cell.cast::<T>() }
        }))
    }

    #[cfg(not(loom))]
    pub(crate) fn pages_mut(&mut self) -> impl Iterator<Item = &mut [T]> {
        let len = self.len();
        self.pages.iter_mut().filter_map(move |(p, page)| {
            let used = PAGE.min(len.saturating_sub(p * PAGE));
            if used == 0 {
                return None;
            }
            // SAFETY: quiescence initializes the prefix; each page is yielded
            // once under exclusive access. Transparent production cells give
            // the array T's layout; no unwritten tail is in the slice.
            Some(unsafe {
                core::slice::from_raw_parts_mut(page.slots.as_mut_ptr().cast::<T>(), used)
            })
        })
    }

    #[cfg(loom)]
    pub(crate) fn pages_mut(&mut self) -> impl Iterator<Item = &mut [T]> {
        let len = self.len();
        self.pages.iter_mut().flat_map(move |(p, page)| {
            let used = PAGE.min(len.saturating_sub(p * PAGE));
            page.slots.iter_mut().take(used).map(|slot| {
                slot.with_mut(|cell| {
                    // SAFETY: exclusive iteration yields each initialized
                    // prefix slot once. Singleton slices retain loom tracking
                    // without assuming its cell metadata has T's layout.
                    core::slice::from_mut(unsafe { &mut *cell.cast::<T>() })
                })
            })
        })
    }

    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.pages_mut().flatten()
    }

    pub(crate) fn pop(&mut self) -> Option<T> {
        let pos = self.len().checked_sub(1)?;
        let page = self.pages.get(pos / PAGE).expect("committed page exists");
        let offset = pos % PAGE;
        // Relaxed throughout: &mut self excludes all shared operations. Clear
        // ownership metadata before returning a value whose drop may panic.
        page.ready[offset / 64].fetch_and(!(1 << (offset % 64)), Ordering::Relaxed);
        self.committed.store(pos, Ordering::Relaxed);
        self.reserved.store(pos, Ordering::Relaxed);
        Some(page.slots[offset].with_mut(|cell| {
            // SAFETY: this exclusively owned, initialized slot has just been
            // removed from both counters and ready bits; transfer it once.
            unsafe { ptr::read(cell.cast::<T>()) }
        }))
    }

    pub(crate) fn swap_remove(&mut self, pos: usize) -> T {
        let last = self.len().checked_sub(1).expect("nonempty append log");
        assert!(pos <= last, "append log removal out of bounds");
        if pos == last {
            return self.pop().expect("last slot exists");
        }
        let value = self.pop().expect("last slot exists");
        core::mem::replace(self.get_mut(pos).expect("remaining slot exists"), value)
    }

    pub(crate) fn truncate(&mut self, len: usize) {
        while self.len() > len {
            drop(self.pop());
        }
    }

    pub(crate) fn clear(&mut self) {
        self.truncate(0);
    }

    pub(crate) fn into_inner(mut self) -> PagedVec<T, PAGE> {
        let mut result = PagedVec::with_capacity(self.len());
        while let Some(value) = self.pop() {
            result.push(value);
        }
        result.reverse();
        result
    }

    #[cfg(all(test, loom))]
    pub(super) fn reserve_for_test(&self, limit: usize) -> usize {
        self.claim(limit).expect("test reservation fits").0
    }

    #[cfg(all(test, loom))]
    pub(super) fn finish_for_test(&self, pos: usize, value: T) {
        let page = self.pages.get(pos / PAGE).expect("test claimed this slot");
        self.publish(pos, page, value);
    }
}

impl<T, const PAGE: usize> Drop for AppendLog<T, PAGE> {
    fn drop(&mut self) {
        // Quiescence gives committed == reserved. Empty values in the log's
        // normal clear order, and finish that order on unwind.
        // Page::drop remains a fallback for every ready bit, independently of
        // the counters (also useful for deliberately split loom fixtures).
        struct Rest<'a, T, const PAGE: usize>(&'a mut AppendLog<T, PAGE>);
        impl<T, const PAGE: usize> Drop for Rest<'_, T, PAGE> {
            fn drop(&mut self) {
                self.0.clear();
            }
        }
        // Nothing to run for element types without drop glue.
        if !core::mem::needs_drop::<T>() {
            return;
        }
        let rest = Rest(self);
        rest.0.clear();
        core::mem::forget(rest);
    }
}
