use std::{
    mem::size_of,
    sync::atomic::{AtomicU16, AtomicU32, AtomicU64, AtomicU8, Ordering},
};

use arrayvec::ArrayVec;

use crate::{
    board::evaluation::MINIMUM_TB_WIN_SCORE,
    chessmove::Move,
    util::depth::{CompactDepthStorage, Depth},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Bound {
    None = 0,
    Upper = 1,
    Lower = 2,
    Exact = 3,
}

macro_rules! impl_from_bound {
    ($t:ty) => {
        impl From<Bound> for $t {
            fn from(hflag: Bound) -> Self {
                hflag as Self
            }
        }
    };
}

impl_from_bound!(u8);
impl_from_bound!(i32);

fn divide_into_chunks<T>(slice: &[T], n_chunks: usize) -> impl Iterator<Item = &[T]> {
    let chunk_size = slice.len() / n_chunks + 1; // +1 to avoid 0
    slice.chunks(chunk_size)
}

const MAX_AGE: i32 = 1 << 5; // must be power of 2
const AGE_MASK: i32 = MAX_AGE - 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedInfo {
    data: u8,
}

impl PackedInfo {
    const fn new(age: u8, flag: Bound, pv: bool) -> Self {
        Self { data: (age << 3) | (pv as u8) << 2 | flag as u8 }
    }

    const fn age(self) -> u8 {
        self.data >> 3
    }

    fn flag(self) -> Bound {
        match self.data & 0b11 {
            0 => Bound::None,
            1 => Bound::Upper,
            2 => Bound::Lower,
            3 => Bound::Exact,
            _ => unreachable!(),
        }
    }

    const fn pv(self) -> bool {
        self.data & 0b100 != 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C, align(8))]
pub struct TTEntry {
    pub key: u16,                   // 16 bits
    pub m: Option<Move>,            // 16 bits
    pub score: i16,                 // 16 bits
    pub depth: CompactDepthStorage, // 8 bits, wrapper around a u8
    pub info: PackedInfo,           // 5 + 1 + 2 bits, wrapper around a u8
    pub evaluation: i16,            // 16 bits
}

impl TTEntry {
    #[allow(dead_code)]
    fn as_bits(self) -> [u8; 10] {
        // SAFETY: TTEntry can be safely turned into bits.
        unsafe { TTEntryReadTarget { entry: self }.bits.to_ne_bytes()[0..10].try_into().unwrap() }
    }
}

const CLUSTER_SIZE: usize = 3;

/// Object representing the backing memory used to store tt entries.
#[derive(Debug, Default)]
#[repr(C, align(32))]
struct TTClusterMemory {
    entry1_block1: AtomicU64,
    entry1_block2: AtomicU16,
    entry2_block1: AtomicU16,
    entry2_block2: AtomicU32,
    entry2_block3: AtomicU32,
    entry3_block1: AtomicU32,
    entry3_block2: AtomicU64,
}

#[repr(C)]
union TTEntryReadTarget {
    bits: u128,
    entry: TTEntry,
}

impl TTClusterMemory {
    pub fn load(&self, idx: usize) -> TTEntry {
        // SAFETY: All bitpatterns of TTEntry are valid.
        unsafe {
            let bits = match idx {
                // INTERNAL IMPLEMENTATION DETAIL:
                // We swap the access pattern of the second and third entries, accessing the entry that is third
                // in terms of memory layout when index = 1 is passed, and accessing the entry that is second
                // w.r.t. memory layout when index = 2 is passed. This is because we will often be looping
                // through indexes 0..3, and have a chance to short-circuit before reaching index 2.
                // As such, we save the costliest access for last.
                // (the memory-layout-second entry requires three atomic operations, because of address alignment)
                0 => {
                    let part1 = self.entry1_block1.load(Ordering::Relaxed);
                    let part2 = self.entry1_block2.load(Ordering::Relaxed);
                    // [ 01, 23, 45, 67 ] + [ 89 ] => 10 byte struct.
                    u128::from(part1) | u128::from(part2) << 64
                }
                2 => {
                    let part1 = self.entry2_block1.load(Ordering::Relaxed);
                    let part2 = self.entry2_block2.load(Ordering::Relaxed);
                    let part3 = self.entry2_block3.load(Ordering::Relaxed);
                    // [ 01 ] + [ 23, 45 ] + [ 67, 89 ] => 10 byte struct.
                    u128::from(part1) | u128::from(part2) << 16 | u128::from(part3) << 48
                }
                1 => {
                    let part1 = self.entry3_block1.load(Ordering::Relaxed);
                    let part2 = self.entry3_block2.load(Ordering::Relaxed);
                    // [ 01, 23 ] + [ 45, 67, 89, __ ] => 10 byte struct.
                    u128::from(part1) | u128::from(part2) << 32
                }
                _ => panic!("Index out of bounds!"),
            };
            TTEntryReadTarget { bits }.entry
        }
    }

    pub fn store(&self, idx: usize, entry: TTEntry) {
        #![allow(clippy::cast_possible_truncation)]
        let mut memory = TTEntryReadTarget { bits: 0 };
        memory.entry = entry;
        // SAFETY: This is all initialised.
        let bits = unsafe { memory.bits };
        match idx {
            // INTERNAL IMPLEMENTATION DETAIL:
            // We swap the access pattern of the second and third entries, accessing the entry that is third
            // in terms of memory layout when index = 1 is passed, and accessing the entry that is second
            // w.r.t. memory layout when index = 2 is passed. This is because we will often be looping
            // through indexes 0..3, and have a chance to short-circuit before reaching index 2.
            // As such, we save the costliest access for last.
            // (the memory-layout-second entry requires three atomic operations, because of address alignment)
            0 => {
                self.entry1_block1.store(bits as u64, Ordering::Relaxed);
                self.entry1_block2.store((bits >> 64) as u16, Ordering::Relaxed);
            }
            2 => {
                self.entry2_block1.store(bits as u16, Ordering::Relaxed);
                self.entry2_block2.store((bits >> 16) as u32, Ordering::Relaxed);
                self.entry2_block3.store((bits >> 48) as u32, Ordering::Relaxed);
            }
            1 => {
                self.entry3_block1.store(bits as u32, Ordering::Relaxed);
                self.entry3_block2.store((bits >> 32) as u64, Ordering::Relaxed);
            }
            _ => panic!("Index out of bounds!"),
        }
    }

    pub fn clear(&self) {
        self.entry1_block1.store(0, Ordering::Relaxed);
        self.entry1_block2.store(0, Ordering::Relaxed);
        self.entry2_block1.store(0, Ordering::Relaxed);
        self.entry2_block2.store(0, Ordering::Relaxed);
        self.entry2_block3.store(0, Ordering::Relaxed);
        self.entry3_block1.store(0, Ordering::Relaxed);
        self.entry3_block2.store(0, Ordering::Relaxed);
    }
}

// const _TT_ENTRY_SIZE: () = assert!(size_of::<TTEntry>() == 10);
const _CLUSTER_SIZE: () = assert!(size_of::<TTClusterMemory>() == 32, "TT Cluster size is suboptimal.");

#[derive(Debug)]
pub struct TT {
    table: Vec<TTClusterMemory>,
    age: AtomicU8,
}

#[derive(Debug, Clone, Copy)]
pub struct TTView<'a> {
    table: &'a [TTClusterMemory],
    age: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct TTHit {
    pub mov: Option<Move>,
    pub depth: Depth,
    pub bound: Bound,
    pub value: i32,
    pub eval: i32,
    pub was_pv: bool,
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub struct ClusterIndex {
    idx: u8,
}

impl ClusterIndex {
    pub const NONE: Self = Self { idx: 4 };
}

impl TT {
    pub const fn new() -> Self {
        Self { table: Vec::new(), age: AtomicU8::new(0) }
    }

    pub fn resize(&mut self, bytes: usize) {
        let new_len = bytes / size_of::<TTClusterMemory>();
        // dealloc the old table:
        self.table = Vec::new();
        // construct a new vec:
        // SAFETY: zeroed memory is a legal bitpattern for AtomicU64.
        unsafe {
            let layout = std::alloc::Layout::array::<TTClusterMemory>(new_len).unwrap();
            let ptr = std::alloc::alloc_zeroed(layout);
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            self.table = Vec::from_raw_parts(ptr.cast(), new_len, new_len);
        }
    }

    pub fn clear(&self, threads: usize) {
        #[allow(clippy::collection_is_never_read)]
        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(threads);
            for chunk in divide_into_chunks(&self.table, threads) {
                let handle = s.spawn(move || {
                    for entry in chunk {
                        entry.clear();
                    }
                });
                handles.push(handle);
            }
        });
    }

    const fn pack_key(key: u64) -> u16 {
        #![allow(clippy::cast_possible_truncation)]
        key as u16
    }

    pub fn view(&self) -> TTView {
        TTView { table: &self.table, age: self.age.load(Ordering::Relaxed) }
    }

    pub fn increase_age(&self) {
        #![allow(clippy::cast_possible_truncation)]
        let new_age = (self.age.load(Ordering::Relaxed) + 1) & AGE_MASK as u8; // keep age in range [0, MAX_AGE]
        self.age.store(new_age, Ordering::Relaxed);
    }

    pub fn size(&self) -> usize {
        self.table.len() * size_of::<TTClusterMemory>()
    }
}

impl<'a> TTView<'a> {
    fn wrap_key(&self, key: u64) -> usize {
        #![allow(clippy::cast_possible_truncation)]
        let key = u128::from(key);
        let len = self.table.len() as u128;
        // fixed-point multiplication trick!
        ((key * len) >> 64) as usize
    }

    #[allow(clippy::too_many_arguments)]
    pub fn store(
        &self,
        key: u64,
        ply: usize,
        mut best_move: Option<Move>,
        score: i32,
        eval: i32,
        flag: Bound,
        depth: Depth,
        pv: bool,
        saved_idx: ClusterIndex,
    ) {
        // get index into the table:
        let index = self.wrap_key(key);
        // create a small key from the full key:
        let key = TT::pack_key(key);
        // get current table age:
        let tt_age = i32::from(self.age);
        // load the cluster:
        let cluster = &self.table[index];

        let tte;
        let idx;
        // select the entry:
        if saved_idx == ClusterIndex::NONE {
            panic!("Got invalid ClusterIndex!");
        } else {
            idx = saved_idx.idx as usize;
            tte = cluster.load(idx);
        }

        if best_move.is_none() && tte.key == key {
            // if we don't have a best move, and the entry is for the same position,
            // then we should retain the best move from the previous entry.
            best_move = tte.m;
        }

        // normalise mate / TB scores:
        let score = normalise_gt_truth_score(score, ply);

        // give entries a bonus for type:
        // exact = 3, lower = 2, upper = 1
        let insert_flag_bonus = i32::from(flag);
        let record_flag_bonus = i32::from(tte.info.flag());

        // preferentially overwrite entries that are from searches on previous positions in the game.
        let age_differential = (MAX_AGE + tt_age - i32::from(tte.info.age())) & AGE_MASK;

        // we use quadratic scaling of the age to allow entries that aren't too old to be kept,
        // but to ensure that *really* old entries are overwritten even if they are of high depth.
        let insert_priority = depth + insert_flag_bonus + (age_differential * age_differential) / 4 + Depth::from(pv);
        let record_prority = Depth::from(tte.depth) + record_flag_bonus;

        // replace the entry:
        // 1. unconditionally if we're in the root node (holdover from TT-pv probing)
        // 2. if the entry is for a different position
        // 3. if it's an exact entry, and the old entry is not exact
        // 4. if the new entry is of higher priority than the old entry
        if tte.key != key
            || flag == Bound::Exact && tte.info.flag() != Bound::Exact
            || insert_priority * 3 >= record_prority * 2
        {
            let write = TTEntry {
                key,
                m: best_move,
                score: score.try_into().expect(
                    "attempted to store a score with value outwith [i16::MIN, i16::MAX] in the transposition table",
                ),
                depth: depth.try_into().unwrap(),
                info: PackedInfo::new(self.age, flag, pv),
                evaluation: eval.try_into().expect(
                    "attempted to store an eval with value outwith [i16::MIN, i16::MAX] in the transposition table",
                ),
            };
            self.table[index].store(idx, write);
        }
    }

    pub fn probe(&self, key: u64, ply: usize) -> (Option<TTHit>, ClusterIndex) {
        let index = self.wrap_key(key);
        let key = TT::pack_key(key);

        // load the entry:
        let cluster = &self.table[index];
        let mut cache = ArrayVec::<_, 3>::new();

        for i in 0..CLUSTER_SIZE {
            let entry = cluster.load(i);

            if entry.key != key {
                cache.push(entry);
                continue;
            }

            let tt_move = entry.m;
            let tt_depth = entry.depth.into();
            let tt_bound = entry.info.flag();

            // we can't store the score in a tagged union,
            // because we need to do mate score preprocessing.
            let tt_value = reconstruct_gt_truth_score(entry.score.into(), ply);

            #[allow(clippy::cast_possible_truncation)]
            return (
                Some(TTHit {
                    mov: tt_move,
                    depth: tt_depth,
                    bound: tt_bound,
                    value: tt_value,
                    eval: entry.evaluation.into(),
                    was_pv: entry.info.pv(),
                }),
                ClusterIndex { idx: i as u8 },
            );
        }

        let tt_age = i32::from(self.age);

        let mut tte = cache[0];
        let mut idx = 0;
        if !(tte.key == 0 || tte.key == key) {
            for i in 1..CLUSTER_SIZE {
                let entry = cache[i];

                if entry.key == 0 || entry.key == key {
                    idx = i;
                    break;
                }

                if i32::from(tte.depth.inner()) - ((MAX_AGE + tt_age - i32::from(tte.info.age())) & AGE_MASK) * 4
                    > i32::from(entry.depth.inner()) - ((MAX_AGE + tt_age - i32::from(entry.info.age())) & AGE_MASK) * 4
                {
                    tte = entry;
                    idx = i;
                }
            }
        }

        #[allow(clippy::cast_possible_truncation)]
        (None, ClusterIndex { idx: idx as u8 })
    }

    pub fn prefetch(&self, key: u64) {
        // SAFETY: The pointer we construct is in-bounds, and _mm_prefetch
        // doesn't really do anything particularly dangerous anyway.
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};

            // get a reference to the entry in the table:
            let index = self.wrap_key(key);
            let entry = &self.table[index];

            // prefetch the entry:
            _mm_prefetch(std::ptr::from_ref::<TTClusterMemory>(entry).cast::<i8>(), _MM_HINT_T0);
        }
    }

    pub fn probe_for_provisional_info(&self, key: u64) -> Option<(Option<Move>, i32)> {
        self.probe(key, 0).0.map(|TTHit { mov, value, .. }| (mov, value))
    }

    pub fn hashfull(&self) -> usize {
        let mut hit = 0;
        for i in 0..2000 {
            let cluster = &self.table[i];
            for i in 0..CLUSTER_SIZE {
                let entry = cluster.load(i);
                if entry.key != 0 && entry.info.age() == self.age {
                    hit += 1;
                }
            }
        }
        hit / 2 * CLUSTER_SIZE
    }
}

const fn normalise_gt_truth_score(mut score: i32, ply: usize) -> i32 {
    #![allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    if score >= MINIMUM_TB_WIN_SCORE {
        score += ply as i32;
    } else if score <= -MINIMUM_TB_WIN_SCORE {
        score -= ply as i32;
    }
    score
}

const fn reconstruct_gt_truth_score(mut score: i32, ply: usize) -> i32 {
    #![allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    if score >= MINIMUM_TB_WIN_SCORE {
        score -= ply as i32;
    } else if score <= -MINIMUM_TB_WIN_SCORE {
        score += ply as i32;
    }
    score
}

mod tests {
    #![allow(unused_imports)]
    use crate::{
        piece::PieceType,
        util::{depth::ZERO_PLY, Square},
    };

    use super::*;

    #[test]
    fn tt_entry_roundtrip() {
        #![allow(clippy::cast_possible_wrap)]
        fn format_slice_hex(slice: &[u8]) -> String {
            let inner = slice.iter().map(|x| format!("{x:02X}")).collect::<Vec<_>>();
            let inner = inner.join(", ");
            format!("[{inner}]")
        }
        let entry = TTEntry {
            key: 0x1234,
            m: Some(Move::new(Square::new_clamped(0x1A), Square::new_clamped(0x1B))),
            score: 0xAB,
            depth: Depth::new(0x13).try_into().unwrap(),
            info: PackedInfo::new(31, Bound::Exact, true),
            evaluation: 0xCDEFu16 as i16,
        };
        let cluster_memory = TTClusterMemory::default();
        for i in 0..3 {
            cluster_memory.store(i, entry);
            let loaded = cluster_memory.load(i);
            println!("Slot {i}");
            println!(" Stored: {}", format_slice_hex(&entry.as_bits()));
            println!(" Loaded: {}", format_slice_hex(&loaded.as_bits()));
            assert_eq!(entry.as_bits(), loaded.as_bits(), "Assertion failed for slot {i}!");
        }
    }
}
