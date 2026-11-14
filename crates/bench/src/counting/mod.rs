//! A counting global allocator shared by the memory binaries.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

pub(crate) struct Counting;
pub(crate) static CURRENT: AtomicUsize = AtomicUsize::new(0);
pub(crate) static PEAK: AtomicUsize = AtomicUsize::new(0);
fn add(n: usize) {
    let now = CURRENT.fetch_add(n, Ordering::Relaxed) + n;
    PEAK.fetch_max(now, Ordering::Relaxed);
}
// SAFETY: all requests are forwarded unchanged to System, and counters never dereference pointers.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller provides a valid layout.
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            add(layout.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
        // SAFETY: the caller supplies the original pointer and layout.
        unsafe {
            System.dealloc(p, layout);
        }
    }
    unsafe fn realloc(&self, p: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the caller supplies a valid allocation and new size.
        let result = unsafe { System.realloc(p, layout, new_size) };
        if !result.is_null() {
            if new_size >= layout.size() {
                add(new_size - layout.size());
            } else {
                CURRENT.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        result
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;
