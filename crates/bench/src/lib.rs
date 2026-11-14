use std::collections::{BTreeMap, BTreeSet};
use std::hash::BuildHasherDefault;

use paged_collections::{
    direct_map::PagedDirectMap,
    page::{Directory, Radix},
    sparse_map::PagedSparseMap,
    sparse_set::PagedSparseSet,
};
use rustc_hash::FxHasher;

pub const SIZES: [usize; 3] = [1_000, 100_000, 10_000_000];
pub const PATTERNS: [&str; 3] = ["dense", "clustered", "uniform"];
/// Uniform random keys put almost every key in its own page, so the paged
/// containers would need tens of GiB at the largest size; stop at 1e5.
pub fn skipped(pattern: &str, n: usize) -> bool {
    pattern == "uniform" && n > 100_000
}
pub type FxMap = hashbrown::HashMap<u32, u64, BuildHasherDefault<FxHasher>>;
pub type FxSet = hashbrown::HashSet<u32, BuildHasherDefault<FxHasher>>;

// An invertible permutation gives unique keys and a disjoint miss set.
fn permute(mut x: u32) -> u32 {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    x
}
pub fn keys(pattern: &str, n: usize, miss: bool) -> Vec<u32> {
    (0..n)
        .map(|i| match pattern {
            "dense" => i as u32 + if miss { n as u32 } else { 0 },
            "clustered" => {
                let block = (i / 64) as u32 + if miss { 1 << 25 } else { 0 };
                // Permute the 26-bit block number, preserving its range.
                let mask = (1 << 26) - 1;
                let mut b = block;
                b ^= (b << 7) & mask;
                b ^= b >> 9;
                b ^= (b << 8) & mask;
                b * 64 + (i % 64) as u32
            }
            "uniform" => permute(i as u32 + if miss { 1 << 31 } else { 0 }),
            _ => unreachable!(),
        })
        .collect()
}
pub fn shuffle<T>(items: &mut [T]) {
    let mut state = 0x1234_5678u32;
    for i in (1..items.len()).rev() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        items.swap(i, state as usize % (i + 1));
    }
}
pub fn dense_len(keys: &[u32]) -> Option<usize> {
    let max = keys.iter().copied().max().unwrap_or(0);
    (max <= (1 << 26)).then_some(max as usize + 1)
}

pub trait Map: Default {
    fn for_keys(_keys: &[u32]) -> Self {
        Self::default()
    }
    fn put(&mut self, key: u32, value: u64);
    fn lookup(&self, key: u32) -> Option<u64>;
    fn delete(&mut self, key: u32);
    fn visit(&self) -> u64;
}
impl Map for PagedSparseMap<u32, u64> {
    fn put(&mut self, k: u32, v: u64) {
        let _ = self.insert(k, v);
    }
    fn lookup(&self, k: u32) -> Option<u64> {
        self.get(k).copied()
    }
    fn delete(&mut self, k: u32) {
        let _ = self.remove(k);
    }
    fn visit(&self) -> u64 {
        self.iter().fold(0u64, |a, (_, v)| a.wrapping_add(*v))
    }
}
impl<D: Directory> Map for PagedDirectMap<u32, u64, 1024, D> {
    fn put(&mut self, k: u32, v: u64) {
        let _ = self.insert(k, v);
    }
    fn lookup(&self, k: u32) -> Option<u64> {
        self.get(k).copied()
    }
    fn delete(&mut self, k: u32) {
        let _ = self.remove(k);
    }
    fn visit(&self) -> u64 {
        self.iter().fold(0u64, |a, (_, v)| a.wrapping_add(*v))
    }
}
pub type Direct = PagedDirectMap<u32, u64, 1024>;
pub type DirectRadix = PagedDirectMap<u32, u64, 1024, Radix>;
macro_rules! map_impl {
    ($t:ty) => {
        impl Map for $t {
            fn put(&mut self, k: u32, v: u64) {
                let _ = self.insert(k, v);
            }
            fn lookup(&self, k: u32) -> Option<u64> {
                self.get(&k).copied()
            }
            fn delete(&mut self, k: u32) {
                let _ = self.remove(&k);
            }
            fn visit(&self) -> u64 {
                self.values().fold(0u64, |a, v| a.wrapping_add(*v))
            }
        }
    };
}
map_impl!(FxMap);
map_impl!(BTreeMap<u32,u64>);
#[derive(Default)]
pub struct DenseMap(pub Vec<Option<u64>>);
impl Map for DenseMap {
    fn for_keys(keys: &[u32]) -> Self {
        Self(vec![None; dense_len(keys).expect("dense key bound")])
    }
    fn put(&mut self, k: u32, v: u64) {
        let i = k as usize;
        if i >= self.0.len() {
            self.0.resize(i + 1, None);
        }
        self.0[i] = Some(v);
    }
    fn lookup(&self, k: u32) -> Option<u64> {
        self.0.get(k as usize).copied().flatten()
    }
    fn delete(&mut self, k: u32) {
        if let Some(v) = self.0.get_mut(k as usize) {
            *v = None;
        }
    }
    fn visit(&self) -> u64 {
        self.0
            .iter()
            .flatten()
            .fold(0u64, |a, v| a.wrapping_add(*v))
    }
}
pub trait Set: Default {
    fn for_keys(_keys: &[u32]) -> Self {
        Self::default()
    }
    fn put(&mut self, key: u32);
    fn lookup(&self, key: u32) -> bool;
    fn delete(&mut self, key: u32);
    fn visit(&self) -> u64;
}
impl Set for PagedSparseSet<u32> {
    fn put(&mut self, k: u32) {
        let _ = self.insert(k);
    }
    fn lookup(&self, k: u32) -> bool {
        self.contains(k)
    }
    fn delete(&mut self, k: u32) {
        let _ = self.remove(k);
    }
    fn visit(&self) -> u64 {
        self.iter().fold(0u64, |a, k| a.wrapping_add(*k as u64))
    }
}
macro_rules! set_impl {
    ($t:ty) => {
        impl Set for $t {
            fn put(&mut self, k: u32) {
                let _ = self.insert(k);
            }
            fn lookup(&self, k: u32) -> bool {
                self.contains(&k)
            }
            fn delete(&mut self, k: u32) {
                let _ = self.remove(&k);
            }
            fn visit(&self) -> u64 {
                self.iter().fold(0u64, |a, k| a.wrapping_add(*k as u64))
            }
        }
    };
}
set_impl!(FxSet);
set_impl!(BTreeSet<u32>);
#[derive(Default)]
pub struct DenseSet(pub Vec<bool>);
impl Set for DenseSet {
    fn for_keys(keys: &[u32]) -> Self {
        Self(vec![false; dense_len(keys).expect("dense key bound")])
    }
    fn put(&mut self, k: u32) {
        let i = k as usize;
        if i >= self.0.len() {
            self.0.resize(i + 1, false);
        }
        self.0[i] = true;
    }
    fn lookup(&self, k: u32) -> bool {
        self.0.get(k as usize).copied().unwrap_or(false)
    }
    fn delete(&mut self, k: u32) {
        if let Some(v) = self.0.get_mut(k as usize) {
            *v = false;
        }
    }
    fn visit(&self) -> u64 {
        self.0
            .iter()
            .enumerate()
            .filter(|(_, v)| **v)
            .fold(0u64, |a, (i, _)| a.wrapping_add(i as u64))
    }
}
pub fn build_map<M: Map>(keys: &[u32]) -> M {
    let mut m = M::for_keys(keys);
    for &k in keys {
        m.put(k, k as u64);
    }
    m
}
pub fn build_set<S: Set>(keys: &[u32]) -> S {
    let mut s = S::for_keys(keys);
    for &k in keys {
        s.put(k);
    }
    s
}

pub trait IdStore: Default {
    type Key: Copy;
    fn add(&mut self, value: u64) -> Self::Key;
    fn fetch(&self, key: Self::Key) -> Option<u64>;
    fn erase(&mut self, key: Self::Key);
    fn visit(&self) -> u64;
}
impl IdStore for paged_collections::hive::Hive<u64> {
    type Key = paged_collections::id_alloc::Id;
    fn add(&mut self, v: u64) -> Self::Key {
        self.insert(v)
    }
    fn fetch(&self, k: Self::Key) -> Option<u64> {
        self.get(k).copied()
    }
    fn erase(&mut self, k: Self::Key) {
        let _ = self.remove(k);
    }
    fn visit(&self) -> u64 {
        self.values().fold(0u64, |a, v| a.wrapping_add(*v))
    }
}
#[derive(Default)]
pub struct IdMap {
    alloc: paged_collections::id_alloc::IdAllocator,
    map: paged_collections::sparse_map::PagedSparseMap<paged_collections::id_alloc::Id, u64>,
}
impl IdStore for IdMap {
    type Key = paged_collections::id_alloc::Id;
    fn add(&mut self, v: u64) -> Self::Key {
        let k = self.alloc.alloc();
        let _ = self.map.insert(k, v);
        k
    }
    fn fetch(&self, k: Self::Key) -> Option<u64> {
        self.map.get(k).copied()
    }
    fn erase(&mut self, k: Self::Key) {
        let _ = self.map.remove(k);
        self.alloc.free(k);
    }
    fn visit(&self) -> u64 {
        self.map.values().fold(0u64, |a, v| a.wrapping_add(*v))
    }
}
macro_rules! slot_store {
    ($t:ty) => {
        impl IdStore for $t {
            type Key = slotmap::DefaultKey;
            fn add(&mut self, v: u64) -> Self::Key {
                self.insert(v)
            }
            fn fetch(&self, k: Self::Key) -> Option<u64> {
                self.get(k).copied()
            }
            fn erase(&mut self, k: Self::Key) {
                let _ = self.remove(k);
            }
            fn visit(&self) -> u64 {
                self.iter().fold(0u64, |a, (_, v)| a.wrapping_add(*v))
            }
        }
    };
}
slot_store!(slotmap::SlotMap<slotmap::DefaultKey,u64>);
slot_store!(slotmap::DenseSlotMap<slotmap::DefaultKey,u64>);
impl IdStore for slab::Slab<u64> {
    type Key = usize;
    fn add(&mut self, v: u64) -> Self::Key {
        self.insert(v)
    }
    fn fetch(&self, k: Self::Key) -> Option<u64> {
        self.get(k).copied()
    }
    fn erase(&mut self, k: Self::Key) {
        let _ = self.remove(k);
    }
    fn visit(&self) -> u64 {
        self.iter().fold(0u64, |a, (_, v)| a.wrapping_add(*v))
    }
}
impl IdStore for thunderdome::Arena<u64> {
    type Key = thunderdome::Index;
    fn add(&mut self, v: u64) -> Self::Key {
        self.insert(v)
    }
    fn fetch(&self, k: Self::Key) -> Option<u64> {
        self.get(k).copied()
    }
    fn erase(&mut self, k: Self::Key) {
        let _ = self.remove(k);
    }
    fn visit(&self) -> u64 {
        self.iter().fold(0u64, |a, (_, v)| a.wrapping_add(*v))
    }
}
pub fn build_ids<S: IdStore>(n: usize) -> (S, Vec<S::Key>) {
    let mut s = S::default();
    let mut ids = Vec::with_capacity(n);
    for i in 0..n {
        ids.push(s.add(i as u64));
    }
    (s, ids)
}
