use std::{
    mem::{MaybeUninit, size_of},
    sync::atomic::{AtomicU8, AtomicU64, Ordering},
};

use crate::{
    chess::chessmove::Move,
    evaluation::MINIMUM_TB_WIN_SCORE,
    threadpool::{self, ScopeExt},
    util::{self, MEGABYTE, depth::CompactDepthStorage},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Bound {
    None = 0,
    Upper = 1,
    Lower = 2,
    Exact = 3,
}

impl Bound {
    pub fn is_lower(self) -> bool {
        self as u8 & 0b10 != 0
    }

    pub fn is_upper(self) -> bool {
        self as u8 & 0b01 != 0
    }
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

fn divide_into_chunks<T>(slice: &[T], chunks: usize) -> impl Iterator<Item = &[T]> {
    let chunk_size = slice.len() / chunks + 1; // +1 to avoid 0
    slice.chunks(chunk_size)
}

unsafe fn threaded_memset_zero(
    ptr: *mut MaybeUninit<u8>,
    len: usize,
    threads: &[threadpool::WorkerThread],
) {
    #[allow(clippy::collection_is_never_read)]
    std::thread::scope(|s| {
        let thread_count = threads.len();
        let chunk_size = len / thread_count + 64;
        let mut handles = Vec::with_capacity(thread_count);
        for (thread_idx, thread) in threads.iter().enumerate() {
            let start = thread_idx * chunk_size;
            let end = ((thread_idx + 1) * chunk_size).min(len);
            if start > end {
                // with many threads we can hit this
                break;
            }
            // Safety: Resultant pointer is in-bounds.
            let slice_ptr = unsafe { ptr.add(start) };
            let slice_len = end.checked_sub(start).unwrap();
            // launder address
            let addr = slice_ptr as usize;
            handles.push(s.spawn_into(
                move || {
                    let slice_ptr = addr as *mut u8;
                    // Safety: Slice is in-bounds and is disjoint with the other
                    // threads' slices.
                    unsafe { std::ptr::write_bytes(slice_ptr, 0, slice_len) };
                },
                thread,
            ));
        }
        for handle in handles {
            handle.join();
        }
    });
}

const MAX_AGE: i32 = 1 << 5; // must be power of 2
const AGE_MASK: i32 = MAX_AGE - 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedInfo {
    data: u8,
}

impl PackedInfo {
    const fn new(age: u8, flag: Bound, pv: bool) -> Self {
        Self {
            data: (age << 3) | ((pv as u8) << 2) | flag as u8,
        }
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
#[repr(C)]
pub struct TTEntry {
    pub key: u16,                   // 2 bytes
    pub m: Option<Move>,            // 2 bytes
    pub score: i16,                 // 2 bytes
    pub depth: CompactDepthStorage, // 1 byte, wrapper around a u8
    pub info: PackedInfo,           // 1 byte (5 + 1 + 2 bits), wrapper around a u8
    pub evaluation: i16,            // 2 bytes
}

#[repr(C)]
union TTEntryReadTarget {
    bytes: u128,
    entry: TTEntry,
}

impl TTEntry {
    #[allow(dead_code)]
    fn to_ne_bytes(self) -> [u8; 10] {
        let mut memory = TTEntryReadTarget { bytes: 0 };
        memory.entry = self;
        // SAFETY: TTEntry can be safely reinterpreted as bytes, and the
        // whole u128 is initialised.
        let bytes = unsafe { memory.bytes };
        bytes.to_ne_bytes()[0..10].try_into().unwrap()
    }
}

const CLUSTER_SIZE: usize = 3;

/// Object representing the backing memory used to store tt entries.
#[derive(Debug, Default)]
#[repr(C, align(32))]
struct TTClusterMemory {
    memory: [AtomicU64; 4],
}

#[repr(C, align(32))]
struct TTCluster {
    entries: [TTEntry; 3],
    padding: [u8; 2],
}

impl TTClusterMemory {
    pub fn load(&self) -> TTCluster {
        let a = self.memory[0].load(Ordering::Relaxed);
        let b = self.memory[1].load(Ordering::Relaxed);
        let c = self.memory[2].load(Ordering::Relaxed);
        let d = self.memory[3].load(Ordering::Relaxed);
        // Safety: TTCluster is POD.
        unsafe { std::mem::transmute::<[u64; 4], TTCluster>([a, b, c, d]) }
    }

    pub fn store(&self, cluster: TTCluster) {
        // Safety: [u64; 4] is POD.
        let memory = unsafe { std::mem::transmute::<TTCluster, [u64; 4]>(cluster) };
        self.memory[0].store(memory[0], Ordering::Relaxed);
        self.memory[1].store(memory[1], Ordering::Relaxed);
        self.memory[2].store(memory[2], Ordering::Relaxed);
        self.memory[3].store(memory[3], Ordering::Relaxed);
    }

    pub fn clear(&self) {
        self.memory[0].store(0, Ordering::Relaxed);
        self.memory[1].store(0, Ordering::Relaxed);
        self.memory[2].store(0, Ordering::Relaxed);
        self.memory[3].store(0, Ordering::Relaxed);
    }
}

const _CLUSTER_SIZE: () = assert!(
    size_of::<TTClusterMemory>() == 32,
    "TT Cluster size is suboptimal."
);

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
    pub depth: i32,
    pub bound: Bound,
    pub value: i32,
    pub eval: i32,
    pub was_pv: bool,
}

impl TT {
    pub const fn new() -> Self {
        Self {
            table: Vec::new(),
            age: AtomicU8::new(0),
        }
    }

    pub fn resize(&mut self, bytes: usize, threads: &[threadpool::WorkerThread]) {
        let start = std::time::Instant::now();
        let new_len = bytes / size_of::<TTClusterMemory>();
        // dealloc the old table:
        self.table = Vec::new();
        // construct a new vec:
        // SAFETY: zeroed memory is a legal bitpattern for AtomicUXX.
        unsafe {
            let layout = std::alloc::Layout::array::<TTClusterMemory>(new_len).unwrap();
            let ptr = std::alloc::alloc(layout);
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            threaded_memset_zero(ptr.cast(), new_len * size_of::<TTClusterMemory>(), threads);
            self.table = Vec::from_raw_parts(ptr.cast(), new_len, new_len);
        }
        println!(
            "info string hash initialisation of {}mb complete in {}us",
            bytes / MEGABYTE,
            start.elapsed().as_micros()
        );
    }

    pub fn clear(&self, threads: &[threadpool::WorkerThread]) {
        #[allow(clippy::collection_is_never_read)]
        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(threads.len());
            for (chunk, worker) in
                divide_into_chunks(&self.table, threads.len()).zip(threads.iter().cycle())
            {
                let handle = s.spawn_into(
                    move || {
                        for entry in chunk {
                            entry.clear();
                        }
                    },
                    worker,
                );
                handles.push(handle);
            }
            for handle in handles {
                handle.join();
            }
        });
    }

    pub const fn pack_key(key: u64) -> u16 {
        #![allow(clippy::cast_possible_truncation)]
        key as u16
    }

    pub fn view(&self) -> TTView<'_> {
        TTView {
            table: &self.table,
            age: self.age.load(Ordering::Relaxed),
        }
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

impl TTView<'_> {
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
        depth: i32,
        pv: bool,
    ) {
        // get index into the table:
        let cluster_index = self.wrap_key(key);
        // create a small key from the full key:
        let key = TT::pack_key(key);
        // get current table age:
        let tt_age = i32::from(self.age);
        // load the cluster:
        let mut cluster = self.table[cluster_index].load();
        let mut tte = cluster.entries[0];
        let mut idx = 0;

        // select the entry:
        if !(tte.key == 0 || tte.key == key) {
            for i in 1..CLUSTER_SIZE {
                let entry = cluster.entries[i];

                if entry.key == 0 || entry.key == key {
                    tte = entry;
                    idx = i;
                    break;
                }

                if i32::from(tte.depth.inner())
                    - ((MAX_AGE + tt_age - i32::from(tte.info.age())) & AGE_MASK) * 4
                    > i32::from(entry.depth.inner())
                        - ((MAX_AGE + tt_age - i32::from(entry.info.age())) & AGE_MASK) * 4
                {
                    tte = entry;
                    idx = i;
                }
            }
        }

        if best_move.is_none() && tte.key == key {
            // if we don't have a best move, and the entry is for the same position,
            // then we should retain the best move from the previous entry.
            best_move = tte.m;
        }

        // give entries a bonus for type:
        // exact = 3, lower = 2, upper = 1
        let insert_flag_bonus = i32::from(flag);
        let record_flag_bonus = i32::from(tte.info.flag());

        // preferentially overwrite entries that are from searches on previous positions in the game.
        let age_differential = (MAX_AGE + tt_age - i32::from(tte.info.age())) & AGE_MASK;

        // we use quadratic scaling of the age to allow entries that aren't too old to be kept,
        // but to ensure that *really* old entries are overwritten even if they are of high depth.
        let insert_priority =
            depth + insert_flag_bonus + (age_differential * age_differential) / 4 + i32::from(pv);
        let record_prority = i32::from(tte.depth) + record_flag_bonus;

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
                // normalise mate / TB scores:
                score: normalise_gt_truth_score(score, ply).try_into().expect(
                    "attempted to store a score with value outwith [i16::MIN, i16::MAX] in the transposition table",
                ),
                depth: depth.try_into().unwrap(),
                info: PackedInfo::new(self.age, flag, pv),
                evaluation: eval.try_into().expect(
                    "attempted to store an eval with value outwith [i16::MIN, i16::MAX] in the transposition table",
                ),
            };
            cluster.entries[idx] = write;
            self.table[cluster_index].store(cluster);
        }
    }

    pub fn probe(&self, key: u64, ply: usize) -> Option<TTHit> {
        let index = self.wrap_key(key);
        let key = TT::pack_key(key);

        let cluster = self.table[index].load();

        for i in 0..CLUSTER_SIZE {
            let entry = cluster.entries[i];

            if entry.key != key {
                continue;
            }

            return Some(TTHit {
                mov: entry.m,
                depth: entry.depth.into(),
                bound: entry.info.flag(),
                value: reconstruct_gt_truth_score(entry.score.into(), ply),
                eval: entry.evaluation.into(),
                was_pv: entry.info.pv(),
            });
        }

        None
    }

    pub fn prefetch(&self, key: u64) {
        // SAFETY: The pointer we construct is in-bounds, and _mm_prefetch
        // doesn't really do anything particularly dangerous anyway.
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};

            // get a reference to the entry in the table:
            let index = self.wrap_key(key);
            let entry = &self.table[index];

            // prefetch the entry:
            _mm_prefetch(
                util::from_ref::<TTClusterMemory>(entry).cast::<i8>(),
                _MM_HINT_T0,
            );
        }
    }

    pub fn probe_move(&self, key: u64) -> Option<(Option<Move>, i32)> {
        self.probe(key, 0)
            .map(|TTHit { mov, value, .. }| (mov, value))
    }

    pub fn hashfull(&self) -> usize {
        let mut hit = 0;
        for i in 0..2000 {
            let cluster = self.table[i].load();
            for i in 0..CLUSTER_SIZE {
                let entry = cluster.entries[i];
                if entry.key != 0 && entry.info.age() == self.age {
                    hit += 1;
                }
            }
        }
        hit / (2 * CLUSTER_SIZE)
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
    use crate::{chess::piece::PieceType, chess::types::Square};

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
            m: Some(Move::new(
                Square::new_clamped(0x1A),
                Square::new_clamped(0x1B),
            )),
            score: 0xAB,
            depth: 0x13.try_into().unwrap(),
            info: PackedInfo::new(31, Bound::Exact, true),
            evaluation: 0xCDEFu16 as i16,
        };
        let cluster_memory = TTClusterMemory::default();
        for i in 0..3 {
            let mut cluster = cluster_memory.load();
            cluster.entries[i] = entry;
            cluster_memory.store(cluster);
            let loaded = cluster_memory.load().entries[i];
            println!("Slot {i}");
            println!(" Stored: {}", format_slice_hex(&entry.to_ne_bytes()));
            println!(" Loaded: {}", format_slice_hex(&loaded.to_ne_bytes()));
            assert_eq!(
                entry.to_ne_bytes(),
                loaded.to_ne_bytes(),
                "Assertion failed for slot {i}!"
            );
        }
    }

    #[test]
    fn memset_correct() {
        #![allow(clippy::undocumented_unsafe_blocks)]
        let mut x = vec![1u8; 2048];
        let pool = threadpool::make_worker_threads(1);
        unsafe {
            threaded_memset_zero(x.as_mut_ptr().cast(), x.len(), &pool);
        }
        for (i, v) in x.iter().enumerate() {
            assert_eq!(*v, 0, "unset at index {i}");
        }

        x = vec![1u8; 2048];
        let pool = threadpool::make_worker_threads(2);
        unsafe {
            threaded_memset_zero(x.as_mut_ptr().cast(), x.len(), &pool);
        }
        for (i, v) in x.iter().enumerate() {
            assert_eq!(*v, 0, "unset at index {i}");
        }

        x = vec![1u8; 2048];
        let pool = threadpool::make_worker_threads(7);
        unsafe {
            threaded_memset_zero(x.as_mut_ptr().cast(), x.len(), &pool);
        }
        for (i, v) in x.iter().enumerate() {
            assert_eq!(*v, 0, "unset at index {i}");
        }

        x = vec![1u8; 2048];
        let pool = threadpool::make_worker_threads(1337);
        unsafe {
            threaded_memset_zero(x.as_mut_ptr().cast(), x.len(), &pool);
        }
        for (i, v) in x.iter().enumerate() {
            assert_eq!(*v, 0, "unset at index {i}");
        }

        x = vec![1u8; 2048];
        let pool = threadpool::make_worker_threads(5555);
        unsafe {
            threaded_memset_zero(x.as_mut_ptr().cast(), x.len(), &pool);
        }
        for (i, v) in x.iter().enumerate() {
            assert_eq!(*v, 0, "unset at index {i}");
        }
    }
}
