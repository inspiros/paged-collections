//! Directory selection for shared page installation.

use crate::page::{Directory, Flat, Radix};

pub(super) use sealed::SharedPages;

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

    use super::super::index::{INDEX_CHUNKS, page_number};
    use super::super::radix::RadixSpine;
    use super::super::spine::Spine;

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
        fn iter<'a>(&'a self) -> impl Iterator<Item = (usize, &'a P)>
        where
            P: 'a;
        #[doc(hidden)]
        fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = (usize, &'a mut P)>
        where
            P: 'a;
        #[doc(hidden)]
        fn clear(&mut self);
        #[doc(hidden)]
        fn page_number<const PAGE: usize>(index: usize) -> Option<usize>;
    }

    #[doc(hidden)]
    pub struct FlatPages<P>(Spine<P, INDEX_CHUNKS>);
    #[doc(hidden)]
    pub struct RadixPages<P>(RadixSpine<P>);

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

        fn iter<'a>(&'a self) -> impl Iterator<Item = (usize, &'a P)>
        where
            P: 'a,
        {
            self.0.iter()
        }

        fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = (usize, &'a mut P)>
        where
            P: 'a,
        {
            self.0.iter_mut()
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

        fn iter<'a>(&'a self) -> impl Iterator<Item = (usize, &'a P)>
        where
            P: 'a,
        {
            self.0.iter()
        }

        fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = (usize, &'a mut P)>
        where
            P: 'a,
        {
            self.0.iter_mut()
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
