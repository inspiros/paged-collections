//! Per-page locks. The spin implementation permits writer starvation.

#[cfg(all(feature = "std", not(loom)))]
mod implementation {
    use std::sync::PoisonError;

    pub(crate) struct RwLock<T>(std::sync::RwLock<T>);
    pub(crate) type ReadGuard<'a, T> = std::sync::RwLockReadGuard<'a, T>;
    pub(crate) type WriteGuard<'a, T> = std::sync::RwLockWriteGuard<'a, T>;

    impl<T> RwLock<T> {
        pub(crate) fn new(value: T) -> Self {
            Self(std::sync::RwLock::new(value))
        }

        pub(crate) fn read(&self) -> ReadGuard<'_, T> {
            self.0.read().unwrap_or_else(PoisonError::into_inner)
        }

        pub(crate) fn write(&self) -> WriteGuard<'_, T> {
            self.0.write().unwrap_or_else(PoisonError::into_inner)
        }

        pub(crate) fn get_mut(&mut self) -> &mut T {
            self.0.get_mut().unwrap_or_else(PoisonError::into_inner)
        }

        pub(crate) fn into_inner(self) -> T {
            self.0.into_inner().unwrap_or_else(PoisonError::into_inner)
        }
    }
}

#[cfg(any(not(feature = "std"), loom))]
mod implementation {
    use core::marker::PhantomData;
    use core::ops::{Deref, DerefMut};

    use super::super::atomic::{AtomicU32, Ordering, UnsafeCell};
    use super::super::index::retry;

    const WRITER: u32 = 1 << 31;

    // Readers may continually overtake a waiting writer; no fairness is promised.
    pub(crate) struct RwLock<T> {
        state: AtomicU32,
        value: UnsafeCell<T>,
    }

    // SAFETY: moving the lock transfers its exclusively owned T.
    unsafe impl<T: Send> Send for RwLock<T> {}
    // SAFETY: the state grants either shared access to T or one exclusive
    // accessor. Acquire/Release synchronizes successive accesses.
    unsafe impl<T: Send + Sync> Sync for RwLock<T> {}

    // Guards are `!Send` like `std`'s, so the containers' guard and entry
    // types have the same auto traits with and without the `std` feature
    // (the feature stays additive). `Sync` still follows `&RwLock<T>`.
    struct NotSend(PhantomData<*const ()>);
    // SAFETY: a zero-sized marker with no data; it only removes `Send`.
    unsafe impl Sync for NotSend {}

    pub(crate) struct ReadGuard<'a, T> {
        lock: &'a RwLock<T>,
        _not_send: NotSend,
        #[cfg(loom)]
        access: Option<loom::cell::ConstPtr<T>>,
    }

    pub(crate) struct WriteGuard<'a, T> {
        lock: &'a RwLock<T>,
        _not_send: NotSend,
        #[cfg(loom)]
        access: Option<loom::cell::MutPtr<T>>,
    }

    impl<T> RwLock<T> {
        pub(crate) fn new(value: T) -> Self {
            Self {
                state: AtomicU32::new(0),
                value: UnsafeCell::new(value),
            }
        }

        pub(crate) fn read(&self) -> ReadGuard<'_, T> {
            let mut spins = 0;
            loop {
                let state = self.state.load(Ordering::Relaxed);
                if state & WRITER == 0 {
                    assert!(state < WRITER - 1, "RwLock: reader count overflow");
                    if self
                        .state
                        .compare_exchange_weak(
                            state,
                            state + 1,
                            Ordering::Acquire,
                            Ordering::Relaxed,
                        )
                        .is_ok()
                    {
                        return ReadGuard {
                            lock: self,
                            _not_send: NotSend(PhantomData),
                            #[cfg(loom)]
                            access: Some(self.value.read()),
                        };
                    }
                }
                retry(&mut spins);
            }
        }

        pub(crate) fn write(&self) -> WriteGuard<'_, T> {
            let mut spins = 0;
            loop {
                if self
                    .state
                    .compare_exchange_weak(0, WRITER, Ordering::Acquire, Ordering::Relaxed)
                    .is_ok()
                {
                    return WriteGuard {
                        lock: self,
                        _not_send: NotSend(PhantomData),
                        #[cfg(loom)]
                        access: Some(self.value.write()),
                    };
                }
                retry(&mut spins);
            }
        }

        pub(crate) fn get_mut(&mut self) -> &mut T {
            // SAFETY: exclusive access to the lock excludes every guard.
            self.value.with_mut(|value| unsafe { &mut *value })
        }

        pub(crate) fn into_inner(self) -> T {
            self.value.into_inner()
        }
    }

    impl<T> Deref for ReadGuard<'_, T> {
        type Target = T;

        fn deref(&self) -> &T {
            #[cfg(loom)]
            // SAFETY: the reader count excludes writers; tracking lives until
            // this guard's drop, before the read lock is released.
            unsafe {
                self.access.as_ref().unwrap().deref()
            }
            #[cfg(not(loom))]
            // SAFETY: this guard holds a reader count, excluding mutation.
            self.lock.value.with(|value| unsafe { &*value })
        }
    }

    impl<T> Deref for WriteGuard<'_, T> {
        type Target = T;

        fn deref(&self) -> &T {
            #[cfg(loom)]
            {
                self.access.as_ref().unwrap().with(|value| {
                    // SAFETY: WRITER grants exclusive access for this guard's lifetime.
                    unsafe { &*value }
                })
            }
            #[cfg(not(loom))]
            // SAFETY: WRITER excludes all other access to the value.
            self.lock.value.with(|value| unsafe { &*value })
        }
    }

    impl<T> DerefMut for WriteGuard<'_, T> {
        fn deref_mut(&mut self) -> &mut T {
            #[cfg(loom)]
            // SAFETY: this exclusive guard is uniquely borrowed, and its cell
            // tracking covers the entire mutable reference's lifetime.
            unsafe {
                self.access.as_ref().unwrap().deref()
            }
            #[cfg(not(loom))]
            // SAFETY: WRITER and &mut self provide exclusive value access.
            self.lock.value.with_mut(|value| unsafe { &mut *value })
        }
    }

    impl<T> Drop for ReadGuard<'_, T> {
        fn drop(&mut self) {
            #[cfg(loom)]
            drop(self.access.take());
            self.lock.state.fetch_sub(1, Ordering::Release);
        }
    }

    impl<T> Drop for WriteGuard<'_, T> {
        fn drop(&mut self) {
            #[cfg(loom)]
            drop(self.access.take());
            self.lock.state.store(0, Ordering::Release);
        }
    }

    #[cfg(all(test, not(loom)))]
    mod tests {
        use super::RwLock;

        #[test]
        fn readers_writer_and_exclusive_access() {
            let mut lock = RwLock::new(1);
            let a = lock.read();
            let b = lock.read();
            assert_eq!((*a, *b), (1, 1));
            drop((a, b));
            *lock.write() = 2;
            assert_eq!(*lock.read(), 2);
            *lock.get_mut() = 3;
            assert_eq!(lock.into_inner(), 3);
        }

        // Compiles only if the guards are `!Send` (otherwise the two impls
        // make the inferred parameter ambiguous) and `Sync` for `Sync` data.
        #[test]
        fn guards_are_not_send_like_std() {
            trait AmbiguousIfSend<A> {
                fn check() {}
            }
            impl<T: ?Sized> AmbiguousIfSend<()> for T {}
            impl<T: ?Sized + Send> AmbiguousIfSend<u8> for T {}
            <super::ReadGuard<'static, u32> as AmbiguousIfSend<_>>::check();
            <super::WriteGuard<'static, u32> as AmbiguousIfSend<_>>::check();
            fn sync<T: Sync>() {}
            sync::<super::ReadGuard<'static, u32>>();
            sync::<super::WriteGuard<'static, u32>>();
        }
    }
}

pub(super) use implementation::{ReadGuard, RwLock, WriteGuard};
