//! Segmented FIFO adapted from [Crossbeam's SegQueue] (crossbeam-queue
//! 0.3.12, MIT OR Apache-2.0).
//!
//! The slot-claim CASes compare the monotonically increasing 63-bit
//! positions while shared. A thread dereferences its cached block
//! pointer only after winning a position in that block. A last-slot reader
//! can also inspect its successor while the head is still at the boundary,
//! which prevents that successor from being consumed. Reusing an address
//! therefore cannot make a stale position win a claim. The initial pointer
//! CAS only installs a block into a null tail, which stays non-null while
//! shared.
//!
//! [Crossbeam's SegQueue]: https://docs.rs/crossbeam-queue/0.3.12/src/crossbeam_queue/seg_queue.rs.html

// Portions Copyright (c) 2019 The Crossbeam Project Developers
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#[cfg(not(loom))]
use alloc::alloc::{Layout, alloc_zeroed, handle_alloc_error};
use alloc::boxed::Box;
use alloc::vec::{self, Vec};
use core::fmt;
use core::iter::FusedIterator;
use core::mem::{MaybeUninit, needs_drop};
use core::ptr;

use super::atomic::{AtomicPtr, AtomicU32, AtomicU64, CachePadded, Ordering, UnsafeCell, fence};
use super::free_stack::{FreeStack, NONE};
use super::index::{INDEX_CHUNKS, retry};
use super::spine::Spine;
use crate::page::DEFAULT_PAGE;
use crate::paged_deque::PagedDeque;

const WRITE: u32 = 1;
const READ: u32 = 2;
const DESTROY: u32 = 4;
const POOLED: u32 = 1;

const SHIFT: u32 = 1;
const HAS_NEXT: u64 = 1;
const STEP: u64 = 1 << SHIFT;

struct Slot<T> {
    value: UnsafeCell<MaybeUninit<T>>,
    state: AtomicU32,
}

impl<T> Slot<T> {
    fn wait_write(&self) {
        let mut spins = 0;
        while self.state.load(Ordering::Acquire) & WRITE == 0 {
            retry(&mut spins);
        }
    }

    /// Moves the initialized value out of a slot owned by the caller.
    unsafe fn take(&self) -> T {
        self.value.with(|cell| {
            // SAFETY: the caller has the sole read claim to the initialized
            // slot. The queue will not use these bytes as a T again.
            unsafe { cell.cast::<T>().read() }
        })
    }
}

struct Block<T, const PAGE: usize> {
    next: AtomicPtr<Self>,
    pool_next: AtomicU32,
    pool_state: AtomicU32,
    number: u32,
    slots: [Slot<T>; PAGE],
}

impl<T, const PAGE: usize> Block<T, PAGE> {
    #[cfg(not(loom))]
    fn new(number: u32) -> Box<Self> {
        let layout = Layout::new::<Self>();
        // SAFETY: the atomic header makes the block layout nonzero.
        let raw = unsafe { alloc_zeroed(layout) };
        if raw.is_null() {
            handle_alloc_error(layout);
        }
        // SAFETY: the allocation has the block's exact layout. Null is valid
        // for AtomicPtr, zero is valid for the integer and atomic fields, and
        // every value cell holds MaybeUninit<T> through a transparent wrapper.
        let mut block: Box<Self> = unsafe { Box::from_raw(raw.cast()) };
        block.number = number;
        block
    }

    #[cfg(loom)]
    fn new(number: u32) -> Box<Self> {
        // Loom's atomics and cells carry tracking metadata and must be built
        // with their constructors.
        Box::new(Self {
            next: AtomicPtr::new(ptr::null_mut()),
            pool_next: AtomicU32::new(0),
            pool_state: AtomicU32::new(0),
            number,
            slots: core::array::from_fn(|_| Slot {
                value: UnsafeCell::new(MaybeUninit::uninit()),
                state: AtomicU32::new(0),
            }),
        })
    }

    fn wait_next(&self) -> *mut Self {
        let mut spins = 0;
        loop {
            let next = self.next.load(Ordering::Acquire);
            if !next.is_null() {
                return next;
            }
            retry(&mut spins);
        }
    }
}

struct Position<T, const PAGE: usize> {
    index: AtomicU64,
    block: AtomicPtr<Block<T, PAGE>>,
}

/// An unbounded multi-producer, multi-consumer FIFO queue with recycled pages.
///
/// Producers and consumers claim positions using atomic compare-and-exchange.
/// A pop that claims a slot whose push is still writing waits for that push
/// to publish its value. At a page boundary, operations can also wait for the
/// thread advancing the head or tail to finish linking or installing a page.
///
/// A shared reader cannot pin a slot against a concurrent pop. The queue
/// therefore omits peek, shared iteration, cloning, snapshots and
/// serialization. Convert an owned queue to a [`PagedDeque`] to serialize it.
/// With the `serde` feature, the queue can deserialize that format directly.
///
/// ```
/// use paged_collections::concurrent::ConcurrentQueue;
///
/// let queue = ConcurrentQueue::<_, 2>::new();
/// queue.push("first");
/// queue.push("second");
/// assert_eq!(queue.pop(), Some("first"));
/// assert_eq!(queue.pop(), Some("second"));
/// assert_eq!(queue.pop(), None);
/// ```
pub struct ConcurrentQueue<T, const PAGE: usize = DEFAULT_PAGE> {
    head: CachePadded<Position<T, PAGE>>,
    tail: CachePadded<Position<T, PAGE>>,
    blocks: Spine<Block<T, PAGE>, INDEX_CHUNKS>,
    pool: FreeStack,
    allocated: AtomicU32,
}

// SAFETY: moving the queue transfers ownership of its blocks and values.
// Exclusive borrows of values cannot survive the move.
unsafe impl<T: Send, const PAGE: usize> Send for ConcurrentQueue<T, PAGE> {}
// SAFETY: the position CASes give unique writer and reader claims, WRITE
// publishes initialized values, and the destroy protocol ends all slot
// accesses before recycling. Shared operations only move T between threads.
unsafe impl<T: Send, const PAGE: usize> Sync for ConcurrentQueue<T, PAGE> {}

impl<T, const PAGE: usize> ConcurrentQueue<T, PAGE> {
    const BLOCK_CAP: u64 = PAGE as u64;
    const LAP: u64 = Self::BLOCK_CAP + 1;

    /// Creates an empty queue without allocating. `PAGE` must be nonzero.
    #[cfg(not(loom))]
    pub const fn new() -> Self {
        const { assert!(PAGE > 0, "page size must be non-zero") };
        Self {
            head: CachePadded(Position {
                index: AtomicU64::new(0),
                block: AtomicPtr::new(ptr::null_mut()),
            }),
            tail: CachePadded(Position {
                index: AtomicU64::new(0),
                block: AtomicPtr::new(ptr::null_mut()),
            }),
            blocks: Spine::new(),
            pool: FreeStack::new(),
            allocated: AtomicU32::new(0),
        }
    }

    /// Creates an empty queue without allocating. `PAGE` must be nonzero.
    #[cfg(loom)]
    pub fn new() -> Self {
        const { assert!(PAGE > 0, "page size must be non-zero") };
        Self {
            head: CachePadded(Position {
                index: AtomicU64::new(0),
                block: AtomicPtr::new(ptr::null_mut()),
            }),
            tail: CachePadded(Position {
                index: AtomicU64::new(0),
                block: AtomicPtr::new(ptr::null_mut()),
            }),
            blocks: Spine::new(),
            pool: FreeStack::new(),
            allocated: AtomicU32::new(0),
        }
    }

    /// Creates an empty queue with `capacity.div_ceil(PAGE)` pooled pages.
    ///
    /// A push filling a page also needs its successor, even if that page is
    /// still empty.
    ///
    /// # Panics
    ///
    /// Panics if the requested number of pages exceeds the directory capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        let queue = Self::new();
        let pages = capacity.div_ceil(PAGE);
        if pages > 0 {
            assert!(
                pages <= NONE as usize
                    && Spine::<Block<T, PAGE>, INDEX_CHUNKS>::can_allocate_page(pages - 1),
                "ConcurrentQueue page capacity overflow"
            );
        }
        for _ in 0..pages {
            let number = queue.pool.fresh(NONE, 1).expect("checked page capacity");
            queue.pool_block(queue.allocate_block(number));
        }
        queue
    }

    fn block(&self, number: u32) -> &Block<T, PAGE> {
        self.blocks
            .get(number as usize)
            .expect("pooled queue block")
    }

    fn allocate_block(&self, number: u32) -> &Block<T, PAGE> {
        let page = number as usize;
        assert!(
            Spine::<Block<T, PAGE>, INDEX_CHUNKS>::can_allocate_page(page),
            "ConcurrentQueue page capacity overflow"
        );
        let block = self.blocks.get_or_insert_with(page, || Block::new(number));
        // Relaxed: only counts allocated pages, whose publication is in Spine.
        self.allocated.fetch_add(1, Ordering::Relaxed);
        block
    }

    fn claim_pooled(&self, number: u32) -> Option<&Block<T, PAGE>> {
        let block = self.block(number);
        // A free-stack pop is a hint. Only this CAS grants control of the
        // block. Acquire sees the reset published by the recycler.
        block
            .pool_state
            .compare_exchange(POOLED, 0, Ordering::AcqRel, Ordering::Acquire)
            .ok()
            .map(|_| block)
    }

    fn take_block(&self) -> &Block<T, PAGE> {
        loop {
            // Reserve fresh numbers one at a time. With a block size of one
            // there is no pending local range to take ahead of stealing.
            let (number, pooled) = self
                .pool
                .next_index(|n| &self.block(n).pool_next, || None, NONE, 1)
                .expect("ConcurrentQueue block numbers exhausted");
            if !pooled {
                return self.allocate_block(number);
            }
            if let Some(block) = self.claim_pooled(number) {
                #[cfg(all(test, loom))]
                REUSED_BLOCK.store(true, core::sync::atomic::Ordering::Relaxed);
                return block;
            }
        }
    }

    fn pool_block(&self, block: &Block<T, PAGE>) {
        // Release publishes the reset (or the untouched empty block) to a
        // successful claim. The free stack owns only the atomic pool link.
        block.pool_state.store(POOLED, Ordering::Release);
        self.pool.push(block.number, &block.pool_next);
    }

    fn recycle_block(&self, block: &Block<T, PAGE>) {
        for slot in &block.slots {
            slot.state.store(0, Ordering::Relaxed);
        }
        block.next.store(ptr::null_mut(), Ordering::Relaxed);
        self.pool_block(block);
    }

    fn destroy(&self, block: &Block<T, PAGE>, start: usize) {
        // The last-slot reader starts destruction, so it needs no READ bit.
        for slot in &block.slots[start..PAGE - 1] {
            if slot.state.load(Ordering::Acquire) & READ == 0
                && slot.state.fetch_or(DESTROY, Ordering::AcqRel) & READ == 0
            {
                // That slot's reader now owns the rest of destruction.
                return;
            }
        }

        // SAFETY: the head and tail have advanced past this block. The last
        // reader read next and finished its value access before starting
        // destruction. Acquire loads/RMWs observed READ for the other slots,
        // or handed destruction to the reader whose AcqRel RMW observed
        // DESTROY. The destroyer is now the only user of the slots and next.
        self.recycle_block(block);
    }

    /// Appends a value to the tail.
    ///
    /// # Panics
    ///
    /// Panics if page capacity is exhausted or the 63-bit position would
    /// overflow. An allocation failure can also abort.
    pub fn push(&self, value: T) {
        let mut spins = 0;
        let mut tail = self.tail.0.index.load(Ordering::Acquire);
        let mut block = self.tail.0.block.load(Ordering::Acquire);
        let mut spare = SpareBlock {
            queue: self,
            block: None,
        };

        loop {
            let offset = (tail >> SHIFT) % Self::LAP;
            if offset == Self::BLOCK_CAP {
                #[cfg(all(test, loom))]
                BOUNDARY_WAITED.store(true, core::sync::atomic::Ordering::Relaxed);
                retry(&mut spins);
                tail = self.tail.0.index.load(Ordering::Acquire);
                block = self.tail.0.block.load(Ordering::Acquire);
                continue;
            }

            let last = offset + 1 == Self::BLOCK_CAP;
            let advance = if last { 2 * STEP } else { STEP };
            // Skipping the boundary must also fit. Wrapping could misalign
            // offsets when LAP does not divide the 63-bit position range.
            assert!(
                tail <= u64::MAX - advance,
                "ConcurrentQueue position overflow"
            );

            // Prepare the successor before reserving a position. In
            // particular, PAGE == 1 can require both blocks on the first push.
            if last && spare.block.is_none() {
                spare.block = Some(self.take_block());
            }

            if block.is_null() {
                let first = self.take_block();
                let new = ptr::from_ref(first).cast_mut();
                if self
                    .tail
                    .0
                    .block
                    .compare_exchange(block, new, Ordering::Release, Ordering::Relaxed)
                    .is_ok()
                {
                    self.head.0.block.store(new, Ordering::Release);
                    block = new;
                } else {
                    self.pool_block(first);
                    tail = self.tail.0.index.load(Ordering::Acquire);
                    block = self.tail.0.block.load(Ordering::Acquire);
                    continue;
                }
            }

            let new_tail = tail + STEP;
            match self.tail.0.index.compare_exchange_weak(
                tail,
                new_tail,
                Ordering::SeqCst,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    // SAFETY: the matching position CAS grants this slot
                    // in the cached block. Its reader waits for WRITE, and
                    // the block cannot recycle until that reader finishes.
                    let block = unsafe { &*block };
                    if last {
                        let next = ptr::from_ref(spare.block.take().unwrap()).cast_mut();
                        self.tail.0.block.store(next, Ordering::Release);
                        self.tail.0.index.store(new_tail + STEP, Ordering::Release);
                        block.next.store(next, Ordering::Release);
                    }

                    let slot = &block.slots[offset as usize];
                    slot.value.with_mut(|cell| {
                        // SAFETY: the successful CAS grants this producer
                        // exclusive access to the unwritten value cell.
                        unsafe { cell.write(MaybeUninit::new(value)) };
                    });
                    slot.state.fetch_or(WRITE, Ordering::Release);
                    return;
                }
                Err(actual) => {
                    tail = actual;
                    block = self.tail.0.block.load(Ordering::Acquire);
                    retry(&mut spins);
                }
            }
        }
    }

    /// Removes the oldest value, or returns `None` if the queue is empty.
    ///
    /// If the claimed slot's producer is still writing, this call waits
    /// until the value is published.
    pub fn pop(&self) -> Option<T> {
        self.pop_inner(
            #[cfg(all(test, loom))]
            |_| {},
        )
    }

    fn pop_inner(
        &self,
        #[cfg(all(test, loom))] mut checkpoint: impl FnMut(PopStep<T, PAGE>),
    ) -> Option<T> {
        let mut spins = 0;
        let mut head = self.head.0.index.load(Ordering::Acquire);
        let mut block = self.head.0.block.load(Ordering::Acquire);

        loop {
            let offset = (head >> SHIFT) % Self::LAP;
            if offset == Self::BLOCK_CAP {
                retry(&mut spins);
                head = self.head.0.index.load(Ordering::Acquire);
                block = self.head.0.block.load(Ordering::Acquire);
                continue;
            }

            // At the final encodable position the queue must be empty.
            // Check for emptiness before adding to the index.
            let mut new_head = head;
            if head & HAS_NEXT == 0 {
                fence(Ordering::SeqCst);
                let tail = self.tail.0.index.load(Ordering::Relaxed);
                if head >> SHIFT == tail >> SHIFT {
                    return None;
                }
                if (head >> SHIFT) / Self::LAP != (tail >> SHIFT) / Self::LAP {
                    new_head |= HAS_NEXT;
                }
            }

            if block.is_null() {
                retry(&mut spins);
                head = self.head.0.index.load(Ordering::Acquire);
                block = self.head.0.block.load(Ordering::Acquire);
                continue;
            }

            new_head += STEP;
            #[cfg(all(test, loom))]
            checkpoint(PopStep::BeforeClaim { head, block });
            match self.head.0.index.compare_exchange_weak(
                head,
                new_head,
                Ordering::SeqCst,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    // SAFETY: the matching position CAS grants this reader
                    // its slot. Destruction cannot finish until it is done.
                    let block = unsafe { &*block };
                    let last = offset + 1 == Self::BLOCK_CAP;
                    if last {
                        let next = block.wait_next();
                        let mut next_index = (new_head & !HAS_NEXT) + STEP;
                        // SAFETY: the head is still at the boundary. No
                        // reader can advance past this successor and recycle
                        // it before we publish the new head position.
                        if !unsafe { &*next }.next.load(Ordering::Relaxed).is_null() {
                            next_index |= HAS_NEXT;
                        }
                        self.head.0.block.store(next, Ordering::Release);
                        self.head.0.index.store(next_index, Ordering::Release);
                    }

                    let slot = &block.slots[offset as usize];
                    slot.wait_write();
                    // SAFETY: this reader won the position and acquired WRITE.
                    let value = unsafe { slot.take() };

                    if last {
                        self.destroy(block, 0);
                    } else if slot.state.fetch_or(READ, Ordering::AcqRel) & DESTROY != 0 {
                        #[cfg(all(test, loom))]
                        checkpoint(PopStep::DestroyHandOff);
                        self.destroy(block, offset as usize + 1);
                    }
                    return Some(value);
                }
                Err(actual) => {
                    head = actual;
                    block = self.head.0.block.load(Ordering::Acquire);
                    retry(&mut spins);
                }
            }
        }
    }

    /// Reports whether the queue is empty. Exact when no operation is in flight.
    pub fn is_empty(&self) -> bool {
        let head = self.head.0.index.load(Ordering::SeqCst);
        let tail = self.tail.0.index.load(Ordering::SeqCst);
        head >> SHIFT == tail >> SHIFT
    }

    /// Returns the number of queued values. Exact when no operation is in flight.
    pub fn len(&self) -> usize {
        loop {
            let tail = self.tail.0.index.load(Ordering::SeqCst);
            let head = self.head.0.index.load(Ordering::SeqCst);
            if self.tail.0.index.load(Ordering::SeqCst) == tail {
                // Count values only: lap k, offset o maps to k * PAGE + o,
                // so a boundary marker equals the next lap's first slot.
                let logical = |position: u64| {
                    let position = position >> SHIFT;
                    position - position / Self::LAP
                };
                // The head can be seen ahead of both tail loads: a reader
                // that skipped the tail check through HAS_NEXT may have
                // advanced it, while the boundary tail store is only
                // Release. Report an empty queue then.
                return logical(tail).saturating_sub(logical(head)) as usize;
            }
        }
    }

    /// Returns the number of allocated pages, including pooled pages.
    pub fn allocated_pages(&self) -> usize {
        self.allocated.load(Ordering::Relaxed) as usize
    }

    // Equivalent to Crossbeam's exclusive IntoIter step. The spine owns
    // every page, so a finished page can be pooled or left for Spine::drop.
    fn pop_exclusive(&mut self, recycle: bool) -> Option<T> {
        let head = self.head.0.index.load(Ordering::Relaxed);
        let tail = self.tail.0.index.load(Ordering::Relaxed);
        if head >> SHIFT == tail >> SHIFT {
            return None;
        }

        let offset = (head >> SHIFT) % Self::LAP;
        let raw = self.head.0.block.load(Ordering::Relaxed);
        // SAFETY: under exclusive access a nonempty queue has an initialized
        // head slot in a live block. No push or pop can be in flight.
        let block = unsafe { &*raw };
        // SAFETY: the exclusive head slot holds the next live value.
        let value = unsafe { block.slots[offset as usize].take() };
        if offset + 1 == Self::BLOCK_CAP {
            let next = block.next.load(Ordering::Relaxed);
            self.head.0.block.store(next, Ordering::Relaxed);
            self.head
                .0
                .index
                .store((head & !HAS_NEXT) + 2 * STEP, Ordering::Relaxed);
            if recycle {
                self.recycle_block(block);
            }
        } else {
            self.head.0.index.store(head + STEP, Ordering::Relaxed);
        }
        Some(value)
    }

    // Detaches the active block of an empty queue under exclusive access.
    fn detach_empty_block(&mut self) -> *mut Block<T, PAGE> {
        self.tail.0.index.store(0, Ordering::Relaxed);
        self.head.0.index.store(0, Ordering::Relaxed);
        self.tail.0.block.store(ptr::null_mut(), Ordering::Relaxed);
        self.head.0.block.swap(ptr::null_mut(), Ordering::Relaxed)
    }

    fn pool_empty_block(&mut self) {
        let empty = self.detach_empty_block();
        // SAFETY: the detached block, if any, still belongs to the spine;
        // the queue was empty and exclusive access rules out other users.
        if let Some(block) = unsafe { empty.as_ref() } {
            self.recycle_block(block);
        }
    }

    /// Drops all values in FIFO order and keeps the allocated pages for reuse.
    pub fn clear(&mut self) {
        while let Some(value) = self.pop_exclusive(true) {
            drop(value);
        }
        self.pool_empty_block();
    }

    /// Frees every page the current values do not need.
    ///
    /// An empty queue releases all pages. A nonempty queue moves its values
    /// out in FIFO order, frees all pages and the page directory, and pushes
    /// the values back, in O(len) time. Page numbers restart from zero, so
    /// repeated shrinking does not grow the directory.
    pub fn shrink_to_fit(&mut self) {
        let values: Vec<T> = core::iter::from_fn(|| self.pop_exclusive(false)).collect();
        // Every value is out, so no slot is initialized: freeing the pages
        // drops no T. Pooled and finished pages go with them.
        self.detach_empty_block();
        self.pool = FreeStack::new();
        self.blocks.clear();
        self.allocated.store(0, Ordering::Relaxed);
        self.extend(values);
    }

    /// Iterates over mutable references to queued values in FIFO order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        let mut head = self.head.0.index.load(Ordering::Relaxed) >> SHIFT;
        let tail = self.tail.0.index.load(Ordering::Relaxed) >> SHIFT;
        let mut block = self.head.0.block.load(Ordering::Relaxed);
        core::iter::from_fn(move || {
            if head == tail {
                return None;
            }

            let offset = head % Self::LAP;
            // SAFETY: the exclusive queue borrow keeps this block alive.
            // Project through the raw pointer because references to earlier
            // slots in the same block may still be live.
            let slots = unsafe { &raw const (*block).slots };
            // SAFETY: the offset is in bounds; earlier yielded references
            // point to distinct slots. Borrow just this slot.
            let slot = unsafe { &*slots.cast::<Slot<T>>().wrapping_add(offset as usize) };
            head += 1;
            if offset + 1 == Self::BLOCK_CAP {
                // SAFETY: next belongs to the same live block and its
                // storage is disjoint from all yielded values.
                let next = unsafe { &(*block).next };
                block = next.load(Ordering::Relaxed);
                head += 1;
            }
            Some(slot.value.with_mut(|cell| {
                // SAFETY: this slot is initialized, visited once, and its
                // contents are protected by the exclusive queue borrow.
                unsafe { &mut *cell.cast::<T>() }
            }))
        })
    }

    /// Empties the queue and returns its values in FIFO order.
    ///
    /// The drain owns the values immediately. Dropping it drops any unread
    /// values. The queue retains its pages for reuse.
    pub fn drain(&mut self) -> vec::IntoIter<T> {
        let mut values = Vec::with_capacity(self.len());
        while let Some(value) = self.pop_exclusive(true) {
            values.push(value);
        }
        self.pool_empty_block();
        values.into_iter()
    }
}

// Preallocation can race with another producer, or be followed by a failed
// first-block allocation. Return any unused successor on every exit path.
struct SpareBlock<'a, T, const PAGE: usize> {
    queue: &'a ConcurrentQueue<T, PAGE>,
    block: Option<&'a Block<T, PAGE>>,
}

impl<T, const PAGE: usize> Drop for SpareBlock<'_, T, PAGE> {
    fn drop(&mut self) {
        if let Some(block) = self.block.take() {
            self.queue.pool_block(block);
        }
    }
}

impl<T, const PAGE: usize> Drop for ConcurrentQueue<T, PAGE> {
    fn drop(&mut self) {
        // If a value's destructor panics, keep dropping the rest during the
        // unwind. `pop_exclusive` advances the head before returning a value,
        // so no value is dropped twice.
        struct DropRest<'a, T, const PAGE: usize>(&'a mut ConcurrentQueue<T, PAGE>);
        impl<T, const PAGE: usize> Drop for DropRest<'_, T, PAGE> {
            fn drop(&mut self) {
                while let Some(value) = self.0.pop_exclusive(false) {
                    drop(value);
                }
            }
        }

        if needs_drop::<T>() {
            let rest = DropRest(self);
            while let Some(value) = rest.0.pop_exclusive(false) {
                drop(value);
            }
            core::mem::forget(rest);
        }
        // The spine drops all allocated blocks, including pooled blocks.
    }
}

impl<T, const PAGE: usize> Default for ConcurrentQueue<T, PAGE> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const PAGE: usize> fmt::Debug for ConcurrentQueue<T, PAGE> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConcurrentQueue")
            .field("len", &self.len())
            .finish()
    }
}

impl<T, const PAGE: usize> Extend<T> for ConcurrentQueue<T, PAGE> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for value in iter {
            self.push(value);
        }
    }
}

impl<T, const PAGE: usize> FromIterator<T> for ConcurrentQueue<T, PAGE> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut queue = Self::new();
        queue.extend(iter);
        queue
    }
}

/// An owning FIFO iterator over a concurrent queue.
pub struct IntoIter<T, const PAGE: usize> {
    queue: ConcurrentQueue<T, PAGE>,
}

impl<T, const PAGE: usize> Iterator for IntoIter<T, PAGE> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        self.queue.pop_exclusive(false)
    }
}

impl<T, const PAGE: usize> FusedIterator for IntoIter<T, PAGE> {}

impl<T, const PAGE: usize> IntoIterator for ConcurrentQueue<T, PAGE> {
    type Item = T;
    type IntoIter = IntoIter<T, PAGE>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter { queue: self }
    }
}

/// Moves the values from front to back in O(len) time.
impl<T, const PAGE: usize> From<PagedDeque<T, PAGE>> for ConcurrentQueue<T, PAGE> {
    fn from(deque: PagedDeque<T, PAGE>) -> Self {
        deque.into_iter().collect()
    }
}

/// Moves the values from head to tail in O(len) time.
impl<T, const PAGE: usize> From<ConcurrentQueue<T, PAGE>> for PagedDeque<T, PAGE> {
    fn from(queue: ConcurrentQueue<T, PAGE>) -> Self {
        queue.into_iter().collect()
    }
}

/// Reads the plain deque format and moves the values into the queue.
#[cfg(feature = "serde")]
impl<'de, T: serde::Deserialize<'de>, const PAGE: usize> serde::Deserialize<'de>
    for ConcurrentQueue<T, PAGE>
{
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        <PagedDeque<T, PAGE> as serde::Deserialize>::deserialize(deserializer).map(Self::from)
    }
}

/// Set when a push waits at a boundary marker (a plain atomic outside the
/// loom model): the producer race model asserts it is reached.
#[cfg(all(test, loom))]
static BOUNDARY_WAITED: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Set when a push claims a pooled block (a plain atomic outside the loom
/// model): the cross-thread reuse model asserts it is reached.
#[cfg(all(test, loom))]
static REUSED_BLOCK: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

#[cfg(all(test, loom))]
enum PopStep<T, const PAGE: usize> {
    BeforeClaim {
        head: u64,
        block: *mut Block<T, PAGE>,
    },
    DestroyHandOff,
}

#[cfg(all(test, loom))]
#[path = "loom_queue.rs"]
mod loom_queue;
