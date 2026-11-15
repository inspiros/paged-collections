//! Runtime page-size selection. Forwarders dispatch once and expose common
//! slices/iterators, so comparisons never match two size enums together.

use crate::convert::{Key, PyKey};
use paged_collections::direct_map::PagedDirectMap;
use paged_collections::hive::{Hive, RawPage, RawPagesError};
use paged_collections::id_alloc::Id;
use paged_collections::paged_deque::PagedDeque;
use paged_collections::paged_vec::PagedVec;
use paged_collections::priority_queue::PagedPriorityQueue;
use paged_collections::sparse_map::PagedSparseMap;
use paged_collections::sparse_set::PagedSparseSet;
use std::collections::TryReserveError;
use std::ops::{Index, IndexMut, RangeBounds};

paged_enum!(pub(crate) enum SetInner [K: PyKey] [K] = PagedSparseSet [K] []);
paged_enum!(pub(crate) enum MapInner [K: PyKey, V] [K, V] = PagedSparseMap [K, V] []);
paged_enum!(pub(crate) enum DirectInner [V] [V] = PagedDirectMap [Key, V] []);
paged_enum!(pub(crate) enum ListInner [T] [T] = PagedVec [T] []);
paged_enum!(pub(crate) enum DequeInner [T] [T] = PagedDeque [T] []);
paged_enum!(pub(crate) enum QueueInner [P] [P] = PagedPriorityQueue [Key, P] []);
paged_enum!(pub(crate) enum HiveInner [T] [T] = Hive [T] []);

impl<K: PyKey> SetInner<K> {
    pub(crate) fn len(&self) -> usize {
        paged!(Self, self, inner => inner.len())
    }
    pub(crate) fn is_empty(&self) -> bool {
        paged!(Self, self, inner => inner.is_empty())
    }
    pub(crate) fn allocated_pages(&self) -> usize {
        paged!(Self, self, inner => inner.allocated_pages())
    }
    pub(crate) fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        paged!(Self, self, inner => inner.try_reserve(additional))
    }
    pub(crate) fn shrink_to_fit(&mut self) {
        paged!(Self, self, inner => inner.shrink_to_fit())
    }
    pub(crate) fn clear(&mut self) {
        paged!(Self, self, inner => inner.clear())
    }
    pub(crate) fn as_slice(&self) -> &[K] {
        paged!(Self, self, inner => inner.as_slice())
    }
    pub(crate) fn iter(&self) -> std::slice::Iter<'_, K> {
        paged!(Self, self, inner => inner.iter())
    }
    pub(crate) fn contains(&self, key: K) -> bool {
        paged!(Self, self, inner => inner.contains(key))
    }
    pub(crate) fn insert(&mut self, key: K) -> bool {
        paged!(Self, self, inner => inner.insert(key))
    }
    pub(crate) fn remove(&mut self, key: K) -> bool {
        paged!(Self, self, inner => inner.remove(key))
    }
    pub(crate) fn extend(&mut self, keys: impl IntoIterator<Item = K>) {
        paged!(Self, self, inner => inner.extend(keys))
    }
    pub(crate) fn sort_unstable(&mut self) {
        paged!(Self, self, inner => inner.sort_unstable())
    }
}

impl<K: PyKey, V> MapInner<K, V> {
    pub(crate) fn len(&self) -> usize {
        paged!(Self, self, inner => inner.len())
    }
    pub(crate) fn is_empty(&self) -> bool {
        paged!(Self, self, inner => inner.is_empty())
    }
    pub(crate) fn allocated_pages(&self) -> usize {
        paged!(Self, self, inner => inner.allocated_pages())
    }
    pub(crate) fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        paged!(Self, self, inner => inner.try_reserve(additional))
    }
    pub(crate) fn shrink_to_fit(&mut self) {
        paged!(Self, self, inner => inner.shrink_to_fit())
    }
    pub(crate) fn clear(&mut self) {
        paged!(Self, self, inner => inner.clear())
    }
    pub(crate) fn same_size_empty(&self) -> Self {
        Self::new(self.page_size()).expect("existing page size")
    }
    pub(crate) fn contains_key(&self, key: K) -> bool {
        paged!(Self, self, inner => inner.contains_key(key))
    }
    pub(crate) fn get(&self, key: K) -> Option<&V> {
        paged!(Self, self, inner => inner.get(key))
    }
    pub(crate) fn get_mut(&mut self, key: K) -> Option<&mut V> {
        paged!(Self, self, inner => inner.get_mut(key))
    }
    pub(crate) fn remove(&mut self, key: K) -> Option<V> {
        paged!(Self, self, inner => inner.remove(key))
    }
    pub(crate) fn insert(&mut self, key: K, value: V) -> Option<V> {
        paged!(Self, self, inner => inner.insert(key, value))
    }
    pub(crate) fn key_slice(&self) -> &[K] {
        paged!(Self, self, inner => inner.key_slice())
    }
    pub(crate) fn value_slice(&self) -> &[V] {
        paged!(Self, self, inner => inner.value_slice())
    }
    pub(crate) fn values(&self) -> std::slice::Iter<'_, V> {
        paged!(Self, self, inner => inner.values())
    }
    pub(crate) fn values_mut(&mut self) -> std::slice::IterMut<'_, V> {
        paged!(Self, self, inner => inner.values_mut())
    }
    pub(crate) fn iter(&self) -> paged_collections::sparse_map::Iter<'_, K, V> {
        paged!(Self, self, inner => inner.iter())
    }
    pub(crate) fn sort_unstable_keys(&mut self) {
        paged!(Self, self, inner => inner.sort_unstable_keys())
    }
}

impl<V> DirectInner<V> {
    pub(crate) fn len(&self) -> usize {
        paged!(Self, self, inner => inner.len())
    }
    pub(crate) fn is_empty(&self) -> bool {
        paged!(Self, self, inner => inner.is_empty())
    }
    pub(crate) fn allocated_pages(&self) -> usize {
        paged!(Self, self, inner => inner.allocated_pages())
    }
    pub(crate) fn shrink_to_fit(&mut self) {
        paged!(Self, self, inner => inner.shrink_to_fit())
    }
    pub(crate) fn clear(&mut self) {
        paged!(Self, self, inner => inner.clear())
    }
    pub(crate) fn same_size_empty(&self) -> Self {
        Self::new(self.page_size()).expect("existing page size")
    }
    pub(crate) fn contains_key(&self, key: Key) -> bool {
        paged!(Self, self, inner => inner.contains_key(key))
    }
    pub(crate) fn get(&self, key: Key) -> Option<&V> {
        paged!(Self, self, inner => inner.get(key))
    }
    pub(crate) fn get_mut(&mut self, key: Key) -> Option<&mut V> {
        paged!(Self, self, inner => inner.get_mut(key))
    }
    pub(crate) fn remove(&mut self, key: Key) -> Option<V> {
        paged!(Self, self, inner => inner.remove(key))
    }
    pub(crate) fn insert(&mut self, key: Key, value: V) -> Option<V> {
        paged!(Self, self, inner => inner.insert(key, value))
    }
    pub(crate) fn next_occupant(&self, index: usize) -> Option<(Key, &V)> {
        paged!(Self, self, inner => inner.next_occupant(index))
    }
    pub(crate) fn prev_occupant(&self, index: usize) -> Option<(Key, &V)> {
        paged!(Self, self, inner => inner.prev_occupant(index))
    }
    pub(crate) fn first_key_value(&self) -> Option<(Key, &V)> {
        paged!(Self, self, inner => inner.first_key_value())
    }
    pub(crate) fn last_key_value(&self) -> Option<(Key, &V)> {
        paged!(Self, self, inner => inner.last_key_value())
    }
    pub(crate) fn pop_first(&mut self) -> Option<(Key, V)> {
        paged!(Self, self, inner => inner.pop_first())
    }
    pub(crate) fn pop_last(&mut self) -> Option<(Key, V)> {
        paged!(Self, self, inner => inner.pop_last())
    }
    pub(crate) fn iter(&self) -> impl ExactSizeIterator<Item = (Key, &V)> + DoubleEndedIterator {
        paged_iter!(Self, self, inner => inner.iter())
    }
    pub(crate) fn keys(&self) -> impl ExactSizeIterator<Item = Key> + DoubleEndedIterator {
        paged_iter!(Self, self, inner => inner.keys())
    }
    pub(crate) fn values(&self) -> impl ExactSizeIterator<Item = &V> + DoubleEndedIterator {
        paged_iter!(Self, self, inner => inner.values())
    }
    pub(crate) fn slot_range(
        &self,
        range: impl RangeBounds<usize>,
    ) -> impl DoubleEndedIterator<Item = (Key, &V)> {
        paged_iter!(Self, self, inner => inner.slot_range(range))
    }
}

impl<T> ListInner<T> {
    pub(crate) fn len(&self) -> usize {
        paged!(Self, self, inner => inner.len())
    }
    pub(crate) fn is_empty(&self) -> bool {
        paged!(Self, self, inner => inner.is_empty())
    }
    pub(crate) fn allocated_pages(&self) -> usize {
        paged!(Self, self, inner => inner.allocated_pages())
    }
    pub(crate) fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        paged!(Self, self, inner => inner.try_reserve(additional))
    }
    pub(crate) fn shrink_to_fit(&mut self) {
        paged!(Self, self, inner => inner.shrink_to_fit())
    }
    pub(crate) fn clear(&mut self) {
        paged!(Self, self, inner => inner.clear())
    }
    pub(crate) fn capacity(&self) -> usize {
        paged!(Self, self, inner => inner.capacity())
    }
    pub(crate) fn get(&self, index: usize) -> Option<&T> {
        paged!(Self, self, inner => inner.get(index))
    }
    pub(crate) fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        paged!(Self, self, inner => inner.get_mut(index))
    }
    pub(crate) fn remove(&mut self, index: usize) -> T {
        paged!(Self, self, inner => inner.remove(index))
    }
    pub(crate) fn insert(&mut self, index: usize, value: T) {
        paged!(Self, self, inner => inner.insert(index, value))
    }
    pub(crate) fn iter(&self) -> impl ExactSizeIterator<Item = &T> + DoubleEndedIterator {
        paged_iter!(Self, self, inner => inner.iter())
    }
    pub(crate) fn push(&mut self, value: T) {
        paged!(Self, self, inner => inner.push(value))
    }
    pub(crate) fn pop(&mut self) -> Option<T> {
        paged!(Self, self, inner => inner.pop())
    }
    pub(crate) fn extend(&mut self, values: impl IntoIterator<Item = T>) {
        paged!(Self, self, inner => inner.extend(values))
    }
    pub(crate) fn reverse(&mut self) {
        paged!(Self, self, inner => inner.reverse())
    }
    pub(crate) fn sort_by(&mut self, compare: impl FnMut(&T, &T) -> std::cmp::Ordering) {
        paged!(Self, self, inner => inner.sort_by(compare))
    }
    pub(crate) fn pages(&self) -> impl ExactSizeIterator<Item = &[T]> + DoubleEndedIterator {
        paged_iter!(Self, self, inner => inner.pages())
    }
}

impl<T> DequeInner<T> {
    pub(crate) fn len(&self) -> usize {
        paged!(Self, self, inner => inner.len())
    }
    pub(crate) fn is_empty(&self) -> bool {
        paged!(Self, self, inner => inner.is_empty())
    }
    pub(crate) fn allocated_pages(&self) -> usize {
        paged!(Self, self, inner => inner.allocated_pages())
    }
    pub(crate) fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        paged!(Self, self, inner => inner.try_reserve(additional))
    }
    pub(crate) fn shrink_to_fit(&mut self) {
        paged!(Self, self, inner => inner.shrink_to_fit())
    }
    pub(crate) fn capacity(&self) -> usize {
        paged!(Self, self, inner => inner.capacity())
    }
    pub(crate) fn get(&self, index: usize) -> Option<&T> {
        paged!(Self, self, inner => inner.get(index))
    }
    pub(crate) fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        paged!(Self, self, inner => inner.get_mut(index))
    }
    pub(crate) fn remove(&mut self, index: usize) -> Option<T> {
        paged!(Self, self, inner => inner.remove(index))
    }
    pub(crate) fn insert(&mut self, index: usize, value: T) {
        paged!(Self, self, inner => inner.insert(index, value))
    }
    pub(crate) fn iter(&self) -> impl ExactSizeIterator<Item = &T> + DoubleEndedIterator {
        paged_iter!(Self, self, inner => inner.iter())
    }
    pub(crate) fn push_back(&mut self, value: T) {
        paged!(Self, self, inner => inner.push_back(value))
    }
    pub(crate) fn push_front(&mut self, value: T) {
        paged!(Self, self, inner => inner.push_front(value))
    }
    pub(crate) fn pop_back(&mut self) -> Option<T> {
        paged!(Self, self, inner => inner.pop_back())
    }
    pub(crate) fn pop_front(&mut self) -> Option<T> {
        paged!(Self, self, inner => inner.pop_front())
    }
    pub(crate) fn swap(&mut self, a: usize, b: usize) {
        paged!(Self, self, inner => inner.swap(a, b))
    }
}

impl<P> QueueInner<P> {
    pub(crate) fn len(&self) -> usize {
        paged!(Self, self, inner => inner.len())
    }
    pub(crate) fn is_empty(&self) -> bool {
        paged!(Self, self, inner => inner.is_empty())
    }
    pub(crate) fn allocated_pages(&self) -> usize {
        paged!(Self, self, inner => inner.allocated_pages())
    }
    pub(crate) fn shrink_to_fit(&mut self) {
        paged!(Self, self, inner => inner.shrink_to_fit())
    }
    pub(crate) fn clear(&mut self) {
        paged!(Self, self, inner => inner.clear())
    }
    pub(crate) fn capacity(&self) -> usize {
        paged!(Self, self, inner => inner.capacity())
    }
    pub(crate) fn contains_key(&self, key: &Key) -> bool {
        paged!(Self, self, inner => inner.contains_key(key))
    }
    pub(crate) fn priority(&self, key: &Key) -> Option<&P> {
        paged!(Self, self, inner => inner.priority(key))
    }
    pub(crate) fn peek(&self) -> Option<(&Key, &P)> {
        paged!(Self, self, inner => inner.peek())
    }
    pub(crate) fn keys(&self) -> std::slice::Iter<'_, Key> {
        paged!(Self, self, inner => inner.keys())
    }
    pub(crate) fn iter(&self) -> paged_collections::priority_queue::Iter<'_, Key, P> {
        paged!(Self, self, inner => inner.iter())
    }
    pub(crate) fn reserve(&mut self, additional: usize) {
        paged!(Self, self, inner => inner.reserve(additional))
    }
}

impl<T> HiveInner<T> {
    pub(crate) fn len(&self) -> usize {
        paged!(Self, self, inner => inner.len())
    }
    pub(crate) fn allocated_pages(&self) -> usize {
        paged!(Self, self, inner => inner.allocated_pages())
    }
    pub(crate) fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        paged!(Self, self, inner => inner.try_reserve(additional))
    }
    pub(crate) fn shrink_to_fit(&mut self) {
        paged!(Self, self, inner => inner.shrink_to_fit())
    }
    pub(crate) fn capacity(&self) -> usize {
        paged!(Self, self, inner => inner.capacity())
    }
    pub(crate) fn index_bound(&self) -> usize {
        paged!(Self, self, inner => inner.index_bound())
    }
    pub(crate) fn is_exhausted(&self) -> bool {
        paged!(Self, self, inner => inner.is_exhausted())
    }
    pub(crate) fn contains(&self, id: Id) -> bool {
        paged!(Self, self, inner => inner.contains(id))
    }
    pub(crate) fn get(&self, id: Id) -> Option<&T> {
        paged!(Self, self, inner => inner.get(id))
    }
    pub(crate) fn get_mut(&mut self, id: Id) -> Option<&mut T> {
        paged!(Self, self, inner => inner.get_mut(id))
    }
    pub(crate) fn remove(&mut self, id: Id) -> Option<T> {
        paged!(Self, self, inner => inner.remove(id))
    }
    pub(crate) fn occupant(&self, index: u32) -> Option<Id> {
        paged!(Self, self, inner => inner.occupant(index))
    }
    pub(crate) fn try_insert(&mut self, value: T) -> Result<Id, T> {
        paged!(Self, self, inner => inner.try_insert(value))
    }
    pub(crate) fn iter(&self) -> impl ExactSizeIterator<Item = (Id, &T)> {
        paged!(Self, self, inner => inner.iter())
    }
    pub(crate) fn values(&self) -> impl ExactSizeIterator<Item = &T> {
        paged!(Self, self, inner => inner.values())
    }
    pub(crate) fn ids(&self) -> impl ExactSizeIterator<Item = Id> {
        paged!(Self, self, inner => inner.ids())
    }
    pub(crate) fn raw_pages(&self) -> impl ExactSizeIterator<Item = RawPage<&T>> {
        paged_iter!(Self, self, inner => inner.raw_pages())
    }
    pub(crate) fn drain(&mut self) -> impl ExactSizeIterator<Item = (Id, T)> + use<'_, T> {
        paged_iter!(Self, self, inner => inner.drain())
    }
}

impl<P: Ord> QueueInner<P> {
    pub(crate) fn push(&mut self, key: Key, priority: P) -> Option<P> {
        paged!(Self, self, inner => inner.push(key, priority))
    }
    pub(crate) fn pop(&mut self) -> Option<(Key, P)> {
        paged!(Self, self, inner => inner.pop())
    }
    pub(crate) fn remove(&mut self, key: &Key) -> Option<(Key, P)> {
        paged!(Self, self, inner => inner.remove(key))
    }
    pub(crate) fn into_sorted_iter(self) -> impl Iterator<Item = (Key, P)> {
        paged_iter!(Self, self, inner => inner.into_sorted_iter())
    }
}

impl<T> HiveInner<T> {
    pub(crate) fn try_from_raw_pages(
        page_size: usize,
        pages: impl IntoIterator<Item = RawPage<T>>,
    ) -> Result<Self, RawPagesError> {
        let empty = Self::new(page_size).expect("validated page size");
        paged_wrap!(Self, empty, _inner => Hive::try_from_raw_pages(pages))
    }
}

impl<T> Index<usize> for ListInner<T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        self.get(index).expect("in bounds")
    }
}
impl<T> IndexMut<usize> for ListInner<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        self.get_mut(index).expect("in bounds")
    }
}

impl<T> Index<usize> for DequeInner<T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        self.get(index).expect("in bounds")
    }
}
impl<T> IndexMut<usize> for DequeInner<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        self.get_mut(index).expect("in bounds")
    }
}

impl<K: PyKey> PartialEq for SetInner<K> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().all(|&key| other.contains(key))
    }
}
impl<K: PyKey, V: PartialEq> PartialEq for MapInner<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len()
            && self
                .iter()
                .all(|(&key, value)| other.get(key) == Some(value))
    }
}
impl<V: PartialEq> PartialEq for DirectInner<V> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len()
            && self
                .iter()
                .all(|(key, value)| other.get(key) == Some(value))
    }
}

impl<K: PyKey> SetInner<K> {
    pub(crate) fn set_page_size(&mut self, page_size: usize) -> pyo3::PyResult<()> {
        if page_size == self.page_size() {
            return Ok(());
        }
        let mut replacement = Self::new(page_size)?;
        replacement
            .try_reserve(self.len())
            .map_err(crate::reserve_error)?;
        let entries = self.as_slice().to_vec();
        replacement.extend(entries);
        *self = replacement;
        Ok(())
    }
}

impl<K: PyKey, V> MapInner<K, V> {
    pub(crate) fn set_page_size(&mut self, page_size: usize) -> pyo3::PyResult<()> {
        if page_size == self.page_size() {
            return Ok(());
        }
        let mut replacement = Self::new(page_size)?;
        replacement
            .try_reserve(self.len())
            .map_err(crate::reserve_error)?;
        let old = std::mem::replace(self, replacement);
        let entries = paged!(Self, old, inner => inner.into_iter().collect::<Vec<_>>());
        for (key, value) in entries {
            self.insert(key, value);
        }
        Ok(())
    }
}

impl<V> DirectInner<V> {
    pub(crate) fn set_page_size(&mut self, page_size: usize) -> pyo3::PyResult<()> {
        if page_size == self.page_size() {
            return Ok(());
        }
        let replacement = Self::new(page_size)?;

        let old = std::mem::replace(self, replacement);
        let entries = paged!(Self, old, inner => inner.into_iter().collect::<Vec<_>>());
        for (key, value) in entries {
            self.insert(key, value);
        }
        Ok(())
    }
}
