//! Intrusive free indices and bounded fresh-index reservation, sharded by
//! thread.
//!
//! Each shard holds a tagged LIFO stack of free indices and a block of fresh
//! indices reserved from the shared counter. A thread pushes to and pops
//! from its own shard, so threads that free and reuse their own indices do
//! not share a cache line. A thread whose shard is empty steals another
//! shard's whole stack in one CAS (a bit mask says which shards may hold
//! indices) before it reserves fresh indices. Under `std` a thread's shard
//! is its round-robin thread number (shared with [`super::count`]); without
//! `std` there is one shard; loom models set it explicitly.
//!
//! Fresh indices come from the shared counter one at a time until a shard
//! sees contention there (a failed reservation CAS). From then on that shard
//! reserves aligned blocks of `block` indices and hands them out locally.
//! Indices reserved in a block but not handed out yet are below
//! [`FreeStack::index_bound`] without ever having been used; exclusive
//! conversions list them with [`FreeStack::unissued`].
//!
//! Callers must push an index only while owning it, and at most once before
//! it is popped again. Links must stay allocated while shared access exists.
//! A tag increments on every pop and steal, which defeats ABA unless 2^32 of
//! them happen inside one stalled pop. After such a wrap a pop may return an
//! index that is not free. So **a pop is only a hint**: the caller must claim
//! the index with a compare-exchange on that slot's own state and discard the
//! index if the claim fails. Soundness must never rest on the tag. A wrap can
//! also leave a stack linked into a cycle, so callers that pop until a claim
//! succeeds may spin forever (a liveness failure, never unsoundness).

use alloc::vec::Vec;
use core::ops::Range;

use super::atomic::{AtomicU32, AtomicU64, CachePadded, Ordering, fence, spin_loop};

pub(super) const NONE: u32 = u32::MAX;

// The steal scan rotates a 32-bit mask; shard numbers must wrap with it, so
// SHARDS must divide 32 (a power of two up to 32).
const _: () = assert!(SHARDS <= 32 && SHARDS.is_power_of_two());

#[cfg(all(feature = "std", not(loom)))]
const SHARDS: usize = 32;
#[cfg(loom)]
const SHARDS: usize = 2;
#[cfg(all(not(feature = "std"), not(loom)))]
const SHARDS: usize = 1;

/// Block word of a shard that has not seen contention: take fresh indices
/// one at a time from the shared counter.
const SINGLE: u64 = 0;
/// Block word while one thread reserves a new block for the shard. Cursor 1
/// above end 0 is no real block: every real block ends above zero.
const REFILLING: u64 = 1;

const INDEX: u64 = u32::MAX as u64;

fn retry(spins: &mut u32) {
    *spins = spins.saturating_add(1);
    #[cfg(all(feature = "std", not(loom)))]
    if *spins >= 64 {
        std::thread::yield_now();
        return;
    }
    spin_loop();
}

/// The fresh-block size for pages of `page` slots: one page, within limits
/// that amortize the shared counter and bound the never-issued indices.
pub(crate) const fn block_for(page: usize) -> u32 {
    if page < 64 {
        64
    } else if page > 4096 {
        4096
    } else {
        page as u32
    }
}

#[cfg(all(feature = "std", not(loom)))]
#[inline]
fn shard_index() -> usize {
    super::count::thread_index() % SHARDS
}

#[cfg(all(not(feature = "std"), not(loom)))]
#[inline]
fn shard_index() -> usize {
    0
}

#[cfg(loom)]
loom::thread_local! {
    static LOOM_SHARD: core::cell::Cell<usize> = core::cell::Cell::new(0);
}

#[cfg(loom)]
fn shard_index() -> usize {
    LOOM_SHARD.with(|shard| shard.get())
}

/// Set when `install` retries its CAS with a cached chain tail (a plain
/// atomic outside loom's model): loom models assert they reach that path.
#[cfg(loom)]
pub(super) static SPLICE_RETRIED: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Makes the calling loom thread use `shard` (default 0).
#[cfg(all(test, loom))]
pub(super) fn set_loom_shard(shard: usize) {
    assert!(shard < SHARDS);
    LOOM_SHARD.with(|cell| cell.set(shard));
}

fn pack(cursor: u32, end: u32) -> u64 {
    (u64::from(end) << 32) | u64::from(cursor)
}

/// A shard's stack head and fresh block, written mostly by its own threads.
#[repr(align(128))]
struct Shard {
    /// `(tag: u32) << 32 | index: u32`; index `NONE` means empty.
    head: AtomicU64,
    /// `SINGLE`, `REFILLING`, or `(end: u32) << 32 | cursor: u32`; the block
    /// is used up when `cursor == end`.
    block: AtomicU64,
}

impl Shard {
    #[cfg(not(loom))]
    const fn new() -> Self {
        Self {
            head: AtomicU64::new(NONE as u64),
            block: AtomicU64::new(SINGLE),
        }
    }

    #[cfg(loom)]
    fn new() -> Self {
        Self {
            head: AtomicU64::new(NONE as u64),
            block: AtomicU64::new(SINGLE),
        }
    }

    /// Hands out the next index of this shard's block, or returns the block
    /// word that showed it empty (SINGLE, REFILLING, or used up).
    fn take_block(&self) -> Result<u32, u64> {
        // Relaxed suffices: the block word only assigns distinct indices;
        // pages and slot states use their own synchronization.
        let mut word = self.block.load(Ordering::Relaxed);
        loop {
            let (cursor, end) = (word as u32, (word >> 32) as u32);
            // Also rejects SINGLE (0 >= 0) and REFILLING (1 > 0).
            if cursor >= end {
                return Err(word);
            }
            match self.block.compare_exchange_weak(
                word,
                word + 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Ok(cursor),
                Err(actual) => word = actual,
            }
        }
    }
}

pub(crate) struct FreeStack {
    shards: [Shard; SHARDS],
    /// Bit `s` is set while shard `s` may hold free indices. A pusher sets
    /// it after pushing; a thief that finds the shard empty clears it and
    /// then rechecks. SeqCst on both sides rules out a nonempty shard with
    /// a clear bit once both finish (store-buffering pattern).
    nonempty: CachePadded<AtomicU32>,
    next_fresh: CachePadded<AtomicU32>,
}

impl FreeStack {
    #[cfg(not(loom))]
    pub(crate) const fn new() -> Self {
        Self {
            shards: [const { Shard::new() }; SHARDS],
            nonempty: CachePadded(AtomicU32::new(0)),
            next_fresh: CachePadded(AtomicU32::new(0)),
        }
    }

    #[cfg(loom)]
    pub(crate) fn new() -> Self {
        Self {
            shards: core::array::from_fn(|_| Shard::new()),
            nonempty: CachePadded(AtomicU32::new(0)),
            next_fresh: CachePadded(AtomicU32::new(0)),
        }
    }

    /// Pushes an owned free index on the calling thread's shard.
    pub(crate) fn push(&self, index: u32, next: &AtomicU32) {
        debug_assert_ne!(index, NONE);
        let shard = shard_index();
        let head = &self.shards[shard].head;
        // Acquire observes the previously published stack and its links.
        let mut word = head.load(Ordering::Acquire);
        let mut spins = 0;
        loop {
            // Release publishes the link; only the owning pusher writes it.
            // Stalled poppers may still read it, so it must remain atomic.
            next.store(word as u32, Ordering::Release);
            let new = (word & !INDEX) | u64::from(index);
            // Release publishes the link and prior slot state. Acquire (also
            // on failure) observes the new head before preparing another
            // link. SeqCst orders the push before the mask check in `mark`.
            match head.compare_exchange_weak(word, new, Ordering::SeqCst, Ordering::Acquire) {
                Ok(_) => break,
                Err(actual) => word = actual,
            }
            retry(&mut spins);
        }
        self.mark(shard);
    }

    /// Sets `shard`'s bit after a push to it, unless already set.
    fn mark(&self, shard: usize) {
        if SHARDS == 1 {
            return;
        }
        let bit = 1 << shard;
        // SeqCst: a thief clears the bit, then rechecks the head; this
        // thread pushed, then checks the bit. At least one sees the other.
        // Loading first keeps the line shared while the bit stays set.
        // C++20 SC ordering alone rules out both missing each other (the
        // push CAS, this load, the clear and the recheck form one total
        // order). Loom weakens SeqCst accesses to AcqRel, so its models add
        // the equivalent SeqCst fences.
        #[cfg(loom)]
        fence(Ordering::SeqCst);
        if self.nonempty.0.load(Ordering::SeqCst) & bit == 0 {
            self.nonempty.0.fetch_or(bit, Ordering::SeqCst);
        }
    }

    /// Pops from the calling thread's shard.
    #[cfg(test)]
    pub(crate) fn pop<'a>(&self, next: impl Fn(u32) -> &'a AtomicU32) -> Option<u32> {
        self.pop_at(shard_index(), &next)
    }

    fn pop_at<'a>(&self, shard: usize, next: &impl Fn(u32) -> &'a AtomicU32) -> Option<u32> {
        let head = &self.shards[shard].head;
        // Acquire observes links and slot state published by push.
        let mut word = head.load(Ordering::Acquire);
        let mut spins = 0;
        loop {
            let index = word as u32;
            if index == NONE {
                return None;
            }
            // Acquire reads the published link. It may already be stale after
            // another pop/push, but the changed tag makes our CAS fail.
            let following = next(index).load(Ordering::Acquire);
            let tag = ((word >> 32) as u32).wrapping_add(1);
            let new = (u64::from(tag) << 32) | u64::from(following);
            // Acquire transfers ownership and sees the pusher's slot writes;
            // Release carries that publication to later head readers/poppers.
            match head.compare_exchange_weak(word, new, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => return Some(index),
                Err(actual) => word = actual,
            }
            retry(&mut spins);
        }
    }

    /// Takes the whole stack of another shard that may hold indices,
    /// returns its top, and moves the rest onto the calling thread's shard.
    #[cfg(test)]
    pub(crate) fn steal<'a>(&self, next: impl Fn(u32) -> &'a AtomicU32) -> Option<u32> {
        self.steal_at(shard_index(), &next)
    }

    #[inline(never)]
    fn steal_at<'a>(&self, own: usize, next: &impl Fn(u32) -> &'a AtomicU32) -> Option<u32> {
        if SHARDS == 1 {
            return None;
        }
        // SeqCst: part of the mask protocol (see `nonempty`).
        let mut mask = self.nonempty.0.load(Ordering::SeqCst) & !(1 << own);
        while mask != 0 {
            // Visit shards after our own first, so thieves spread out.
            let rotated = mask.rotate_right(own as u32 + 1);
            // SHARDS is 1 without std, but then this returned above.
            #[allow(clippy::modulo_one)]
            let victim = (rotated.trailing_zeros() as usize + own + 1) % SHARDS;
            mask &= !(1 << victim);
            if let Some(index) = self.steal_from(victim, own, next) {
                return Some(index);
            }
        }
        None
    }

    fn steal_from<'a>(
        &self,
        victim: usize,
        own: usize,
        next: &impl Fn(u32) -> &'a AtomicU32,
    ) -> Option<u32> {
        let head = &self.shards[victim].head;
        // Acquire observes links and slot state published by pushes.
        let mut word = head.load(Ordering::Acquire);
        let mut spins = 0;
        loop {
            if word as u32 == NONE {
                let bit = 1 << victim;
                // SeqCst: clear, then recheck; a concurrent pusher either is
                // seen here or sees the clear bit and sets it again.
                self.nonempty.0.fetch_and(!bit, Ordering::SeqCst);
                #[cfg(loom)]
                fence(Ordering::SeqCst);
                word = head.load(Ordering::SeqCst);
                if word as u32 == NONE {
                    return None;
                }
                self.nonempty.0.fetch_or(bit, Ordering::SeqCst);
                continue;
            }
            // Detach the whole chain. The tag bump makes stalled pops of the
            // victim fail, exactly as a pop's would.
            let tag = ((word >> 32) as u32).wrapping_add(1);
            let detached = (u64::from(tag) << 32) | u64::from(NONE);
            // Acquire takes ownership of every index in the chain and sees
            // their links and slot states; Release keeps the head's release
            // sequence intact for later poppers of the victim.
            match head.compare_exchange_weak(word, detached, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => break,
                Err(actual) => word = actual,
            }
            retry(&mut spins);
        }
        let first = word as u32;
        // Acquire: the link was published by the push that the detach saw.
        let rest = next(first).load(Ordering::Acquire);
        if rest != NONE {
            self.install(own, rest, next);
        }
        Some(first)
    }

    /// Pushes an owned, `NONE`-terminated chain starting at `first` onto
    /// shard `own`. Finds the chain's tail only if that shard is not empty.
    fn install<'a>(&self, own: usize, first: u32, next: &impl Fn(u32) -> &'a AtomicU32) {
        let head = &self.shards[own].head;
        let mut tail: Option<&AtomicU32> = None;
        let mut word = head.load(Ordering::Acquire);
        let mut spins = 0;
        loop {
            let below = word as u32;
            if below != NONE || tail.is_some() {
                let link = match tail {
                    Some(link) => link,
                    None => match self.tail(first, next) {
                        Some(link) => *tail.insert(link),
                        // A cycle (only after a tag wrap): leak the chain.
                        None => return,
                    },
                };
                // Release publishes the link, as in `push`.
                link.store(below, Ordering::Release);
            }
            let new = (word & !INDEX) | u64::from(first);
            // As in `push`: Release publishes the chain (its links were
            // acquired by the detach), SeqCst precedes the mask check.
            match head.compare_exchange_weak(word, new, Ordering::SeqCst, Ordering::Acquire) {
                Ok(_) => break,
                Err(actual) => word = actual,
            }
            #[cfg(loom)]
            if tail.is_some() {
                SPLICE_RETRIED.store(true, core::sync::atomic::Ordering::Relaxed);
            }
            retry(&mut spins);
        }
        self.mark(own);
    }

    /// The last link of an owned chain, or `None` if it is longer than the
    /// number of indices ever reserved (a cycle).
    fn tail<'a>(&self, first: u32, next: &impl Fn(u32) -> &'a AtomicU32) -> Option<&'a AtomicU32> {
        let mut link = next(first);
        for _ in 0..=self.index_bound() {
            // Acquire: links were published by the pushes we acquired.
            let following = link.load(Ordering::Acquire);
            if following == NONE {
                return Some(link);
            }
            link = next(following);
        }
        None
    }

    /// The next index to claim and whether it is reused (from a stack or
    /// `recycle`) rather than fresh. Order: the calling thread's stack, its
    /// fresh block, other shards' stacks, `recycle`, then fresh reservation.
    ///
    /// Inlined fast paths, one per steady state: a shard with indices left
    /// in its block takes one with a CAS (the contended case); a shard that
    /// has never contended (SINGLE) while no other shard is marked nonempty
    /// has no block and nothing to steal, so it goes from `recycle` to one
    /// counter CAS, as the unsharded stack did. Everything else runs out of
    /// line, so this stays small enough to inline into the callers' claim
    /// loops (with the general path inline, the T=1 fresh claim got ~17-25%
    /// slower: `next_index`, `recycle` and `fresh_at` became calls).
    #[inline]
    pub(crate) fn next_index<'a>(
        &self,
        next: impl Fn(u32) -> &'a AtomicU32,
        recycle: impl FnOnce() -> Option<u32>,
        limit: u32,
        block: u32,
    ) -> Option<(u32, bool)> {
        let shard = shard_index();
        if let Some(index) = self.pop_at(shard, &next) {
            return Some((index, true));
        }
        let own = &self.shards[shard];
        // Relaxed, as in take_block: the block word only assigns indices.
        let word = own.block.load(Ordering::Relaxed);
        let (cursor, end) = (word as u32, (word >> 32) as u32);
        // Also inline: the own block, the hot path once a shard contended.
        // One attempt; a shard-mate's take sends us to the slow path.
        if cursor < end {
            if own
                .block
                .compare_exchange_weak(word, word + 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return Some((cursor, false));
            }
            return self.next_index_slow(shard, &next, recycle, limit, block);
        }
        // SeqCst, as in steal_at. Our own bit may be stale after the pop.
        if word != SINGLE
            || (SHARDS > 1 && self.nonempty.0.load(Ordering::SeqCst) & !(1 << shard) != 0)
        {
            return self.next_index_slow(shard, &next, recycle, limit, block);
        }
        if let Some(index) = recycle() {
            return Some((index, true));
        }
        // Relaxed, as in fresh_at: the counter only assigns distinct indices.
        let n = self.next_fresh.0.load(Ordering::Relaxed);
        if n < limit
            && self
                .next_fresh
                .0
                .compare_exchange(n, n + 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
        {
            return Some((n, false));
        }
        // A failed CAS is contention: switch to blocks at once, as fresh_at
        // does (retrying single indices would keep contended threads on the
        // shared counter). At the limit, drain other shards' blocks.
        self.fresh_from(shard, limit, block, n < limit)
            .map(|index| (index, false))
    }

    /// `next_index` after the own stack, for a shard with a block (or in
    /// REFILLING) or while another shard may hold free indices.
    #[inline(never)]
    fn next_index_slow<'a>(
        &self,
        shard: usize,
        next: &impl Fn(u32) -> &'a AtomicU32,
        recycle: impl FnOnce() -> Option<u32>,
        limit: u32,
        block: u32,
    ) -> Option<(u32, bool)> {
        let own = &self.shards[shard];
        // Relaxed: a hint; take_block rechecks with a CAS.
        let word = own.block.load(Ordering::Relaxed);
        if (word as u32) < (word >> 32) as u32 {
            if let Ok(index) = own.take_block() {
                return Some((index, false));
            }
        }
        // SeqCst, as in steal_at (which loads the mask again).
        if SHARDS > 1 && self.nonempty.0.load(Ordering::SeqCst) & !(1 << shard) != 0 {
            if let Some(index) = self.steal_at(shard, next) {
                return Some((index, true));
            }
        }
        if let Some(index) = recycle() {
            return Some((index, true));
        }
        self.fresh_at(shard, limit, block)
            .map(|index| (index, false))
    }

    /// Reserves a fresh index below `limit`: from the calling thread's
    /// block, else from the shared counter (one index, or a new block of up
    /// to `block` indices once this shard has seen contention), else from
    /// another shard's block once the counter reaches `limit`.
    pub(crate) fn fresh(&self, limit: u32, block: u32) -> Option<u32> {
        self.fresh_at(shard_index(), limit, block)
    }

    fn fresh_at(&self, shard: usize, limit: u32, block: u32) -> Option<u32> {
        self.fresh_from(shard, limit, block, false)
    }

    /// `fresh_at`; `contended` says a single-index CAS from SINGLE has
    /// already failed, so a SINGLE shard switches to blocks without another.
    #[inline(never)]
    fn fresh_from(&self, shard: usize, limit: u32, block: u32, contended: bool) -> Option<u32> {
        debug_assert!(block > 0);
        let shard = &self.shards[shard];
        let mut spins = 0;
        // Relaxed suffices for the block word and the shared counter: they
        // only assign distinct indices; pages and slot versions are
        // published by their own synchronization. The one exception is the
        // hand-off at the limit (see `reserve` and `take_other_block`).
        let mut word = shard.block.load(Ordering::Relaxed);
        loop {
            let (cursor, end) = (word as u32, (word >> 32) as u32);
            if cursor < end {
                match shard.block.compare_exchange_weak(
                    word,
                    word + 1,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return Some(cursor),
                    Err(actual) => word = actual,
                }
                continue;
            }
            // One index from the shared counter, unless a SINGLE shard has
            // already seen contention (REFILLING always tries: a shard-mate
            // owns the refill).
            if word == REFILLING || (word == SINGLE && !contended) {
                let n = self.next_fresh.0.load(Ordering::Relaxed);
                if n >= limit {
                    return self.take_other_block();
                }
                if self
                    .next_fresh
                    .0
                    .compare_exchange(n, n + 1, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
                {
                    return Some(n);
                }
                if word == REFILLING {
                    // A shard-mate is reserving a block; do not wait for it.
                    retry(&mut spins);
                    word = shard.block.load(Ordering::Relaxed);
                    continue;
                }
                // Contention: switch this shard to blocks (for good).
            }
            // Own the refill, so the shard never holds two blocks at once.
            if let Err(actual) =
                shard
                    .block
                    .compare_exchange(word, REFILLING, Ordering::Relaxed, Ordering::Relaxed)
            {
                // A shard-mate refilled or is refilling.
                word = actual;
                retry(&mut spins);
                continue;
            }
            return match self.reserve(limit, block) {
                Some((start, end)) => {
                    // Nothing can panic between taking REFILLING and here.
                    shard.block.store(pack(start + 1, end), Ordering::Relaxed);
                    Some(start)
                }
                None => {
                    // Keep the shard in block mode, used up.
                    shard.block.store(pack(limit, limit), Ordering::Relaxed);
                    self.take_other_block()
                }
            };
        }
    }

    /// Reserves `[start, end)` from the shared counter, ending at the next
    /// multiple of `block` (or at `limit`).
    fn reserve(&self, limit: u32, block: u32) -> Option<(u32, u32)> {
        let mut start = self.next_fresh.0.load(Ordering::Relaxed);
        loop {
            if start >= limit {
                return None;
            }
            let aligned = (u64::from(start) / u64::from(block) + 1) * u64::from(block);
            let end = aligned.min(u64::from(limit)) as u32;
            // Release: a thread that later reads the counter at the limit
            // (its Acquire fence is in `take_other_block`) must see this
            // shard's REFILLING, written before this CAS. Every counter write
            // is a read-modify-write, so any value at the limit lies in this
            // CAS's release sequence. Otherwise it could read a stale used-up
            // block word and report exhaustion while this block holds
            // indices. Failure: Relaxed, the counter only assigns indices.
            match self.next_fresh.0.compare_exchange_weak(
                start,
                end,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Some((start, end)),
                Err(actual) => start = actual,
            }
        }
    }

    /// Takes an index from any shard's block, once the counter is used up.
    /// Call it only after reading the counter at or above the limit.
    ///
    /// A shard caught in REFILLING may have reserved a block it has not
    /// published yet; reporting exhaustion then would hide that whole block.
    /// So this waits for such refills (each finishes within a few
    /// instructions; no panic is possible inside). Only reachable at the
    /// index limit.
    ///
    /// Why one pass that sees no REFILLING proves exhaustion: the counter
    /// never moves back, so no reservation can succeed after the read at the
    /// limit, and every earlier successful `reserve` set its shard's
    /// REFILLING before its Release CAS. The Acquire fence below makes that
    /// REFILLING visible, so each shard's word read here is REFILLING or a
    /// later value: the published block, or a used-up one whose indices were
    /// all handed out. A shard's pending flag must come from the same read
    /// that found it empty: a second scan could miss a block published in
    /// between.
    fn take_other_block(&self) -> Option<u32> {
        // Acquire: pairs with `reserve`'s Release CAS through the counter
        // value at the limit that the caller read (Relaxed).
        fence(Ordering::Acquire);
        let mut spins = 0;
        loop {
            let mut pending = false;
            for shard in &self.shards {
                match shard.take_block() {
                    Ok(index) => return Some(index),
                    Err(word) => pending |= word == REFILLING,
                }
            }
            if !pending {
                return None;
            }
            retry(&mut spins);
        }
    }

    pub(crate) fn index_bound(&self) -> u32 {
        // Relaxed: a reservation bound, not a page-initialization barrier.
        self.next_fresh.0.load(Ordering::Relaxed)
    }

    /// The indices reserved in blocks but never handed out, in ascending
    /// order, under exclusive access. Their slots have never been used.
    pub(crate) fn unissued(&mut self) -> Unissued {
        let mut ranges: Vec<Range<u32>> = self
            .shards
            .iter_mut()
            .filter_map(|shard| {
                // Relaxed: exclusive access. REFILLING never outlives a
                // shared call (nothing can unwind while it is set).
                let word = shard.block.load(Ordering::Relaxed);
                debug_assert_ne!(word, REFILLING);
                let (cursor, end) = (word as u32, (word >> 32) as u32);
                (cursor < end).then_some(cursor..end)
            })
            .collect();
        ranges.sort_unstable_by_key(|range| range.start);
        Unissued { ranges, at: 0 }
    }

    /// A weakly consistent view of fresh block tails, in ascending order.
    /// Reads one word per shard and skips refills in progress.
    pub(crate) fn unissued_shared(&self) -> Unissued {
        let mut ranges: Vec<_> = self
            .shards
            .iter()
            .filter_map(|shard| {
                // Acquire samples each block word once; block writers can
                // race this load. If an observed index is then handed out,
                // emitting version 2 spends its first generation in the fork.
                let word = shard.block.load(Ordering::Acquire);
                let (cursor, end) = (word as u32, (word >> 32) as u32);
                // SINGLE and REFILLING both have cursor >= end.
                (cursor < end).then_some(cursor..end)
            })
            .collect();
        ranges.sort_unstable_by_key(|range| range.start);
        Unissued { ranges, at: 0 }
    }

    /// Empties every stack under exclusive access; fresh blocks and the
    /// fresh bound are kept. No pop can be in flight, so no stale link can
    /// be followed later.
    pub(crate) fn reset(&mut self) {
        for shard in &mut self.shards {
            // Relaxed: exclusive access rules out concurrent pushes and pops.
            shard.head.store(NONE as u64, Ordering::Relaxed);
        }
        self.nonempty.0.store(0, Ordering::Relaxed);
    }

    /// Pops every free index under exclusive access, each shard's stack in
    /// LIFO order, the calling thread's shard first.
    /// Takes `&self` so `next` may borrow the owner; call it only with
    /// exclusive access to the owner.
    pub(crate) fn drain<'a>(&self, next: impl Fn(u32) -> &'a AtomicU32) -> Vec<u32> {
        let mut out = Vec::new();
        let shard = shard_index();
        while let Some(index) = self
            .pop_at(shard, &next)
            .or_else(|| self.steal_at(shard, &next))
        {
            out.push(index);
        }
        out
    }

    /// Sets the fresh-index boundary during exclusive construction only.
    pub(crate) fn set_next_fresh(&mut self, bound: u32) {
        debug_assert_eq!(self.index_bound(), 0);
        // Relaxed: the newly built owner has not been shared or reserved from.
        self.next_fresh.0.store(bound, Ordering::Relaxed);
    }
}

#[cfg(test)]
impl FreeStack {
    /// How many shards have left SINGLE mode.
    #[cfg(loom)]
    pub(super) fn shards_in_block_mode(&self) -> usize {
        self.shards
            .iter()
            .filter(|shard| shard.block.load(Ordering::Relaxed) != SINGLE)
            .count()
    }

    /// Puts the calling thread's shard in block mode, with its block used up.
    pub(super) fn force_blocks(&self) {
        self.shards[shard_index()]
            .block
            .store(pack(1, 1), Ordering::Relaxed);
    }
}

/// Never-issued block indices, queried in ascending order.
pub(crate) struct Unissued {
    ranges: Vec<Range<u32>>,
    at: usize,
}

impl Unissued {
    /// Tests the captured ranges in O(log shards), in any query order.
    pub(crate) fn contains_shared(&self, index: u32) -> bool {
        let at = self.ranges.partition_point(|range| range.end <= index);
        self.ranges
            .get(at)
            .is_some_and(|range| range.contains(&index))
    }

    /// Whether `index` was reserved but never handed out. Successive calls
    /// must pass ascending indices.
    pub(crate) fn contains(&mut self, index: u32) -> bool {
        while let Some(range) = self.ranges.get(self.at) {
            if index < range.end {
                return index >= range.start;
            }
            self.at += 1;
        }
        false
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;

    fn links(n: usize) -> Vec<AtomicU32> {
        (0..n).map(|_| AtomicU32::new(0)).collect()
    }

    #[test]
    fn single_thread_reserves_one_index_at_a_time() {
        let stack = FreeStack::new();
        assert_eq!(stack.fresh(10, 4), Some(0));
        assert_eq!(stack.fresh(10, 4), Some(1));
        assert_eq!(stack.index_bound(), 2);
    }

    #[test]
    fn blocks_are_aligned_clipped_and_listed_as_unissued() {
        let mut stack = FreeStack::new();
        let shard = &stack.shards[shard_index()];
        stack.next_fresh.0.store(3, Ordering::Relaxed);
        // Pretend contention was seen: a used-up block forces a refill.
        shard.block.store(pack(1, 1), Ordering::Relaxed);
        assert_eq!(stack.fresh(10, 4), Some(3)); // block [3, 4): aligned end
        assert_eq!(stack.fresh(10, 4), Some(4)); // block [4, 8)
        assert_eq!(stack.fresh(10, 4), Some(5));
        assert_eq!(stack.index_bound(), 8);
        let mut unissued = stack.unissued();
        assert!(!unissued.contains(5));
        assert!(unissued.contains(6));
        assert!(unissued.contains(7));
        assert!(!unissued.contains(8));
        for expected in [6, 7, 8, 9] {
            assert_eq!(stack.fresh(10, 4), Some(expected));
        }
        assert_eq!(stack.fresh(10, 4), None);
        assert_eq!(stack.index_bound(), 10);
    }

    #[cfg(feature = "std")]
    #[test]
    fn another_thread_steals_the_whole_stack_in_order() {
        let links = links(4);
        let stack = FreeStack::new();
        for i in 0..4 {
            stack.push(i, &links[i as usize]);
        }
        let next = |i: u32| &links[i as usize];
        let popped = std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    // A fresh thread number: in a test binary other threads
                    // may share it, so only check the order, not the shard.
                    let mut got = Vec::new();
                    while let Some(i) = stack.pop(next).or_else(|| stack.steal(next)) {
                        got.push(i);
                    }
                    got
                })
                .join()
                .unwrap()
        });
        assert_eq!(popped, [3, 2, 1, 0]);
        assert_eq!(stack.pop(next), None);
    }

    #[cfg(feature = "std")]
    #[test]
    fn stolen_chain_splices_under_a_nonempty_stack() {
        let links = links(6);
        let next = |i: u32| &links[i as usize];
        let stack = FreeStack::new();
        let own = shard_index();
        let victim = (own + 1) % SHARDS;
        // Pushed indices are always below the bound; the tail walk relies on it.
        stack.next_fresh.0.store(5, Ordering::Relaxed);
        // Stack [2, 1, 0] on the victim, [4, 3] on our shard.
        for i in 0..3 {
            push_to(&stack, victim, i, &links[i as usize]);
        }
        for i in 3..5 {
            stack.push(i, &links[i as usize]);
        }
        assert_eq!(stack.steal_from(victim, own, &next), Some(2));
        let mut rest = Vec::new();
        while let Some(i) = stack.pop(next) {
            rest.push(i);
        }
        assert_eq!(rest, [1, 0, 4, 3]);
        assert_eq!(stack.steal(next), None);
        assert_eq!(stack.nonempty.0.load(Ordering::Relaxed) & (1 << victim), 0);
    }

    /// Pushes onto a chosen shard, as a thread mapped to it would.
    #[cfg(feature = "std")]
    fn push_to(stack: &FreeStack, shard: usize, index: u32, next: &AtomicU32) {
        let head = &stack.shards[shard].head;
        let word = head.load(Ordering::Relaxed);
        next.store(word as u32, Ordering::Relaxed);
        head.store((word & !INDEX) | u64::from(index), Ordering::Relaxed);
        stack.mark(shard);
    }

    /// The real limit is `u32::MAX`, which is also `NONE` and the top of the
    /// packed block word's cursor half: no index may reach it, and a block
    /// ending there must not carry into the end half.
    #[test]
    fn blocks_and_single_reservations_stop_just_below_u32_max() {
        const LIMIT: u32 = u32::MAX;
        assert_eq!(NONE, LIMIT);
        // Block mode: the last aligned block [2^32 - 4096, 2^32) is clipped
        // to end at the limit, so it holds 4095 indices.
        let mut stack = FreeStack::new();
        let start = LIMIT - 4095;
        stack.set_next_fresh(start - 2);
        stack.force_blocks();
        assert_eq!(stack.fresh(LIMIT, 4096), Some(start - 2)); // [start - 2, start)
        assert_eq!(stack.fresh(LIMIT, 4096), Some(start - 1));
        assert_eq!(stack.fresh(LIMIT, 4096), Some(start)); // the clipped block
        assert_eq!(stack.index_bound(), LIMIT);
        let own = &stack.shards[shard_index()];
        assert_eq!(own.block.load(Ordering::Relaxed), pack(start + 1, LIMIT));
        let mut unissued = stack.unissued();
        assert!(!unissued.contains(start));
        assert!(unissued.contains(start + 1));
        assert!(unissued.contains(LIMIT - 1));
        assert!(!unissued.contains(LIMIT));
        for expected in start + 1..LIMIT {
            assert_eq!(stack.fresh(LIMIT, 4096), Some(expected));
        }
        // The cursor reached the end without overflowing into it.
        let own = &stack.shards[shard_index()];
        assert_eq!(own.block.load(Ordering::Relaxed), pack(LIMIT, LIMIT));
        assert_eq!(stack.fresh(LIMIT, 4096), None);
        assert_eq!(stack.fresh(LIMIT, 4096), None);
        assert_eq!(stack.index_bound(), LIMIT);
        assert!(!stack.unissued().contains(LIMIT - 1));

        // SINGLE mode through the inlined claim path.
        let links = links(2);
        let mut stack = FreeStack::new();
        stack.set_next_fresh(LIMIT - 2);
        let claim =
            |stack: &FreeStack| stack.next_index(|i| &links[i as usize], || None, LIMIT, 4096);
        assert_eq!(claim(&stack), Some((LIMIT - 2, false)));
        assert_eq!(claim(&stack), Some((LIMIT - 1, false)));
        assert_eq!(claim(&stack), None);
        assert_eq!(stack.index_bound(), LIMIT);
        assert_eq!(
            stack.shards[shard_index()].block.load(Ordering::Relaxed),
            SINGLE
        );
    }

    #[test]
    fn exhausted_counter_falls_back_to_other_blocks() {
        let stack = FreeStack::new();
        let other = &stack.shards[(shard_index() + 1) % SHARDS];
        if SHARDS == 1 {
            return;
        }
        other.block.store(pack(5, 7), Ordering::Relaxed);
        stack.next_fresh.0.store(10, Ordering::Relaxed);
        assert_eq!(stack.fresh(10, 4), Some(5));
        assert_eq!(stack.fresh(10, 4), Some(6));
        assert_eq!(stack.fresh(10, 4), None);
    }
}
