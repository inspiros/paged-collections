//! Directory selection for shared page installation.

use crate::page::{Directory, Flat, Radix};

pub(super) use sealed::{PageIter, PageIterMut, SharedPages};

/// Chooses a concurrent page directory, using the plain directory markers.
///
/// [`Flat`] is the default: lookups cost O(1), directory memory follows the
/// highest page number, and indices outside its finite spine range are
/// rejected. [`Radix`] accepts every `usize` index, uses directory memory
/// proportional to installed pages, and looks up pages in O(depth).
///
/// This trait is sealed: only `Flat` and `Radix` implement it.
/// [`Tree`](crate::page::Tree) has no concurrent form. Conversions to plain
/// collections preserve the directory marker through [`Directory`].
pub trait SharedDirectory: Directory + sealed::Sealed {
    #[doc(hidden)]
    type Pages<P>: sealed::SharedPages<P>;
}

// Public items in this private module are unnameable outside the crate.
// The wrappers keep the public GAT from exposing crate-private spine types.
mod sealed {
    use alloc::boxed::Box;
    use alloc::vec::Vec;

    use super::super::index::{INDEX_CHUNKS, page_number};
    use super::super::radix::{self, RadixSpine};
    use super::super::spine::{self, Spine};

    #[doc(hidden)]
    pub trait Sealed {}

    #[doc(hidden)]
    pub trait SharedPages<P>: Sized {
        #[doc(hidden)]
        fn new() -> Self;
        #[doc(hidden)]
        fn get(&self, page: usize) -> Option<&P>;
        #[doc(hidden)]
        fn get_or_insert_with(&self, page: usize, make: impl FnOnce() -> Box<P>) -> &P;
        #[doc(hidden)]
        fn iter(&self) -> PageIter<'_, P>;
        #[doc(hidden)]
        fn iter_mut(&mut self) -> PageIterMut<'_, P>;
        #[doc(hidden)]
        fn remove(&mut self, page: usize);
        #[doc(hidden)]
        fn retain(&mut self, mut keep: impl FnMut(usize, &mut P) -> bool) {
            let removed: Vec<_> = self
                .iter_mut()
                .filter_map(|(number, page)| (!keep(number, page)).then_some(number))
                .collect();
            for number in removed {
                self.remove(number);
            }
        }
        #[doc(hidden)]
        fn clear(&mut self);
        #[doc(hidden)]
        fn page_number<const PAGE: usize>(index: usize) -> Option<usize>;
    }

    #[doc(hidden)]
    pub struct FlatPages<P>(Spine<P, INDEX_CHUNKS>);
    #[doc(hidden)]
    pub struct RadixPages<P>(RadixSpine<P>);

    #[doc(hidden)]
    pub struct PageIter<'a, P>(PageIterInner<'a, P>);

    enum PageIterInner<'a, P> {
        Flat(spine::Iter<'a, P, INDEX_CHUNKS>),
        Radix(radix::SharedPages<'a, P>),
    }

    impl<'a, P> Iterator for PageIter<'a, P> {
        type Item = (usize, &'a P);

        fn next(&mut self) -> Option<Self::Item> {
            match &mut self.0 {
                PageIterInner::Flat(iter) => iter.next(),
                PageIterInner::Radix(iter) => iter.next(),
            }
        }
    }

    #[doc(hidden)]
    pub struct PageIterMut<'a, P>(PageIterMutInner<'a, P>);

    enum PageIterMutInner<'a, P> {
        Flat(spine::IterMut<'a, P>),
        Radix(radix::ExclusivePages<'a, P>),
    }

    impl<'a, P> Iterator for PageIterMut<'a, P> {
        type Item = (usize, &'a mut P);

        fn next(&mut self) -> Option<Self::Item> {
            match &mut self.0 {
                PageIterMutInner::Flat(iter) => iter.next(),
                PageIterMutInner::Radix(iter) => iter.next(),
            }
        }
    }

    impl<P> SharedPages<P> for FlatPages<P> {
        fn new() -> Self {
            Self(Spine::new())
        }

        fn get(&self, page: usize) -> Option<&P> {
            self.0.get(page)
        }

        fn get_or_insert_with(&self, page: usize, make: impl FnOnce() -> Box<P>) -> &P {
            self.0.get_or_insert_with(page, make)
        }

        fn iter(&self) -> PageIter<'_, P> {
            PageIter(PageIterInner::Flat(self.0.iter()))
        }

        fn iter_mut(&mut self) -> PageIterMut<'_, P> {
            PageIterMut(PageIterMutInner::Flat(self.0.iter_mut()))
        }

        fn remove(&mut self, page: usize) {
            drop(self.0.remove(page));
        }

        fn clear(&mut self) {
            self.0.clear();
        }

        fn page_number<const PAGE: usize>(index: usize) -> Option<usize> {
            page_number::<P, PAGE>(index)
        }
    }

    impl<P> SharedPages<P> for RadixPages<P> {
        fn new() -> Self {
            Self(RadixSpine::new())
        }

        fn get(&self, page: usize) -> Option<&P> {
            self.0.get(page)
        }

        fn get_or_insert_with(&self, page: usize, make: impl FnOnce() -> Box<P>) -> &P {
            self.0.get_or_insert_with(page, make)
        }

        fn iter(&self) -> PageIter<'_, P> {
            PageIter(PageIterInner::Radix(self.0.iter()))
        }

        fn iter_mut(&mut self) -> PageIterMut<'_, P> {
            PageIterMut(PageIterMutInner::Radix(self.0.iter_mut()))
        }

        fn remove(&mut self, page: usize) {
            self.0.remove(page);
        }

        fn clear(&mut self) {
            self.0.clear();
        }

        fn page_number<const PAGE: usize>(index: usize) -> Option<usize> {
            Some(index / PAGE)
        }
    }
}

impl sealed::Sealed for Flat {}
impl sealed::Sealed for Radix {}

impl SharedDirectory for Flat {
    type Pages<P> = sealed::FlatPages<P>;
}

impl SharedDirectory for Radix {
    type Pages<P> = sealed::RadixPages<P>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::concurrent::atomic::AtomicPtr;
    use crate::concurrent::index::INDEX_CHUNKS;

    #[test]
    fn page_numbers_preserve_flat_bounds_and_accept_wide_radix_indices() {
        #[cfg(not(loom))]
        let base = 32;
        #[cfg(loom)]
        let base = 2;
        let mut bound = 0;
        for k in 0..INDEX_CHUNKS {
            let Some(cells) = 1usize
                .checked_shl(k as u32)
                .and_then(|n| n.checked_mul(base))
            else {
                break;
            };
            if cells > isize::MAX as usize / core::mem::size_of::<AtomicPtr<()>>() {
                break;
            }
            bound += cells;
        }
        assert_eq!(
            <Flat as SharedDirectory>::Pages::<()>::page_number::<1>(bound - 1),
            Some(bound - 1)
        );
        assert_eq!(
            <Flat as SharedDirectory>::Pages::<()>::page_number::<1>(bound),
            None
        );
        for index in [0, 1, 31, 32, bound - 1, bound, usize::MAX - 1, usize::MAX] {
            assert_eq!(
                <Flat as SharedDirectory>::Pages::<()>::page_number::<3>(index),
                super::super::index::page_number::<(), 3>(index)
            );
            assert_eq!(
                <Radix as SharedDirectory>::Pages::<()>::page_number::<1>(index),
                Some(index)
            );
            assert_eq!(
                <Radix as SharedDirectory>::Pages::<()>::page_number::<3>(index),
                Some(index / 3)
            );
        }
    }
}
