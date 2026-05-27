use std::{
    mem::{MaybeUninit, size_of},
    ptr::slice_from_raw_parts_mut,
    sync::atomic::{AtomicU8, AtomicU64, Ordering},
};

use crate::{
    chess::chessmove::Move,
    evaluation::{MATE_SCORE, MINIMUM_MATE_SCORE, MINIMUM_TB_WIN_SCORE},
    threadpool::{self, ScopeExt},
    util::{MEGABYTE, VALUE_NONE},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Bound {
    Empty = 0,
    Upper = 1,
    Lower = 2,
    Exact = 3,
}

impl Bound {
    pub fn is_lower(self) -> bool {
        self as u8 & 0b10 != 0
    }

    #[expect(unused)]
    pub fn is_upper(self) -> bool {
        self as u8 & 0b01 != 0
    }

    pub fn invert(self) -> Self {
        match self {
            Self::Upper => Self::Lower,
            Self::Lower => Self::Upper,
            x => x,
        }
    }
}

const MAX_AGE: i32 = 1 << 5; // must be power of 2
const AGE_MASK: i32 = MAX_AGE - 1;

unsafe fn threaded_memset_zero(
    ptr: *mut MaybeUninit<u8>,
    len: usize,
    threads: &[threadpool::WorkerThread],
) {
    std::thread::scope(|s| {
        let chunk_size = len / threads.len() + 64;
        let mut handles = Vec::with_capacity(threads.len());
        for (thread_idx, thread) in threads.iter().enumerate() {
            let start = thread_idx * chunk_size;
            let end = ((thread_idx + 1) * chunk_size).min(len);
            if start > end {
                // with many threads we can hit this
                break;
            }
            // launder address
            // Safety: Resultant pointer is in-bounds.
            let addr = unsafe { ptr.add(start) } as usize;
            let work = move || {
                let slice_ptr = addr as *mut u8;
                // Safety: Slice is in-bounds and is disjoint with the other
                // threads' slices.
                unsafe { std::ptr::write_bytes(slice_ptr, 0, end.checked_sub(start).unwrap()) };
            };
            handles.push(s.spawn_into(work, thread));
        }
        for handle in handles {
            handle.join();
        }
    });
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedMeta {
    data: u8,
}

impl PackedMeta {
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
            0 => Bound::Empty,
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
pub struct CacheEntry {
    pub tag: u16,         // 2 bytes
    pub depth: u8,        // 1 byte
    pub info: PackedMeta, // 1 byte (5 + 1 + 2 bits), wrapper around a u8
    pub m: Option<Move>,  // 2 bytes
    pub score: i16,       // 2 bytes
    pub evaluation: i16,  // 2 bytes
}

const CLUSTER_SIZE: usize = 3;

/// Object representing the backing memory used to store cache sets.
#[derive(Debug, Default)]
#[repr(C, align(32))]
struct RawCacheSet {
    memory: [AtomicU64; 4],
}

/// A set in the cache.
#[repr(C, align(32))]
struct CacheSet {
    entries: [CacheEntry; 3],
    coherer: u16,
}

impl CacheSet {
    pub fn checksum(&self) -> u16 {
        self.entries[0].tag ^ self.entries[1].tag ^ self.entries[2].tag
    }
}

impl RawCacheSet {
    /// Read a `CacheSet` out of this backing memory.
    pub fn load(&self) -> CacheSet {
        let a = self.memory[0].load(Ordering::Relaxed);
        let b = self.memory[1].load(Ordering::Relaxed);
        let c = self.memory[2].load(Ordering::Relaxed);
        let d = self.memory[3].load(Ordering::Relaxed);
        // Safety: TTCluster is POD.
        unsafe { std::mem::transmute::<[u64; 4], CacheSet>([a, b, c, d]) }
    }

    /// Write a `CacheSet` to backing memory.
    pub fn store(&self, cluster: CacheSet) {
        // Safety: [u64; 4] is POD.
        let memory = unsafe { std::mem::transmute::<CacheSet, [u64; 4]>(cluster) };
        self.memory[0].store(memory[0], Ordering::Relaxed);
        self.memory[1].store(memory[1], Ordering::Relaxed);
        self.memory[2].store(memory[2], Ordering::Relaxed);
        self.memory[3].store(memory[3], Ordering::Relaxed);
    }

    /// Zero out this `RawCacheSet`.
    pub fn clear(&self) {
        self.memory[0].store(0, Ordering::Relaxed);
        self.memory[1].store(0, Ordering::Relaxed);
        self.memory[2].store(0, Ordering::Relaxed);
        self.memory[3].store(0, Ordering::Relaxed);
    }
}

const _CLUSTER_SIZE: () = assert!(
    size_of::<RawCacheSet>() == 32,
    "TT Cluster size is suboptimal."
);

/// The cache for Viridithas’s search. SMP threads communicate by reading and writing this.
#[derive(Debug)]
pub struct Cache {
    table: Vec<RawCacheSet>,
    age: AtomicU8,
}

/// A borrowed view into the cache.
#[derive(Debug, Clone, Copy)]
pub struct CacheView<'a> {
    table: &'a [RawCacheSet],
    age: u8,
}

/// The result of probing the cache for an entry.
#[derive(Debug, Clone, Copy)]
pub struct CacheResult {
    pub mov: Option<Move>,
    pub depth: i32,
    pub bound: Bound,
    pub value: i32,
    pub eval: i32,
    pub was_pv: bool,
}

impl Cache {
    pub const fn new() -> Self {
        Self {
            table: Vec::new(),
            age: AtomicU8::new(0),
        }
    }

    pub fn resize(&mut self, bytes: usize, threads: &[threadpool::WorkerThread]) {
        let start = std::time::Instant::now();
        let new_len = bytes / size_of::<RawCacheSet>();
        // dealloc the old table:
        self.table = Vec::new();
        // construct a new vec:
        // SAFETY: zeroed memory is a legal bitpattern for AtomicUXX.
        unsafe {
            let layout = std::alloc::Layout::array::<RawCacheSet>(new_len).unwrap();
            let ptr = std::alloc::alloc(layout);
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            threaded_memset_zero(ptr.cast(), new_len * size_of::<RawCacheSet>(), threads);
            self.table = Box::from_raw(slice_from_raw_parts_mut(ptr.cast(), new_len)).into();
        }
        println!(
            "info string hash initialisation of {}mb complete in {}µs",
            bytes / MEGABYTE,
            start.elapsed().as_micros()
        );
    }

    pub fn clear(&self, threads: &[threadpool::WorkerThread]) {
        fn divide_into_chunks<T>(slice: &[T], chunks: usize) -> impl Iterator<Item = &[T]> {
            let chunk_size = slice.len() / chunks + 1; // +1 to avoid 0
            slice.chunks(chunk_size)
        }
        #[allow(clippy::collection_is_never_read)]
        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(threads.len());
            for (chunk, worker) in
                divide_into_chunks(&self.table, threads.len()).zip(threads.iter().cycle())
            {
                let work = move || {
                    for entry in chunk {
                        entry.clear();
                    }
                };
                handles.push(s.spawn_into(work, worker));
            }
            for handle in handles {
                handle.join();
            }
        });
    }

    pub fn view(&self) -> CacheView<'_> {
        CacheView {
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
        self.table.len() * size_of::<RawCacheSet>()
    }
}

impl CacheView<'_> {
    /// Given a Zobrist key for a position, derive an index into the cache,
    /// and a tag for the corresponding entry.
    /// The index is computed using Daniel Lemire’s fast alternative to the
    /// modulo reduction, and as such is dependent on the high bits† of the key.
    ///
    /// As such, we must use the low bits of the key to tag the entry.
    ///
    /// †This can be seen intuïtively if one construes the trick as multiplying
    /// the length of the cache by a value in `[0, 1]`, by treating the 0–2⁶⁴ key
    /// as being a fixed-point multiplier with value of unity = 2⁶⁴.
    /// Clearly, the high bits of a multiplier control where the output lands.
    fn derive_index_tag(&self, key: u64) -> (usize, u16) {
        #![expect(clippy::cast_possible_truncation, reason = "deliberately truncating")]

        // fixed-point multiplication trick!
        let index = ((u128::from(key) * self.table.len() as u128) >> 64) as usize;
        // take low bits:
        let tag = key as u16;

        (index, tag)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn store(
        &self,
        key: u64,
        height: usize,
        mut best_move: Option<Move>,
        score: i32,
        eval: i32,
        flag: Bound,
        depth: i32,
        pv: bool,
    ) {
        let (cluster_index, tag) = self.derive_index_tag(key);
        // get current table age:
        let tt_age = i32::from(self.age);
        // load the cluster:
        let mut cluster = self.table[cluster_index].load();
        let mut tte = cluster.entries[0];
        let mut idx = 0;

        // select the entry:
        if !(tte.tag == 0 || tte.tag == tag) {
            for i in 1..CLUSTER_SIZE {
                let entry = cluster.entries[i];

                if entry.tag == 0 || entry.tag == tag {
                    tte = entry;
                    idx = i;
                    break;
                }

                if i32::from(tte.depth)
                    - ((MAX_AGE + tt_age - i32::from(tte.info.age())) & AGE_MASK) * 4
                    > i32::from(entry.depth)
                        - ((MAX_AGE + tt_age - i32::from(entry.info.age())) & AGE_MASK) * 4
                {
                    tte = entry;
                    idx = i;
                }
            }
        }

        if best_move.is_none() && tte.tag == tag {
            // if we don't have a best move, and the entry is for the same position,
            // then we should retain the best move from the previous entry.
            best_move = tte.m;
        }

        // give entries a bonus for type:
        // exact = 3, lower = 2, upper = 1
        let insert_flag_bonus = flag as i32;
        let record_flag_bonus = tte.info.flag() as i32;

        // preferentially overwrite entries that are from searches on previous positions in the game.
        let age_differential = (MAX_AGE + tt_age - i32::from(tte.info.age())) & AGE_MASK;

        // we use quadratic scaling of the age to allow entries that aren't too old to be kept,
        // but to ensure that *really* old entries are overwritten even if they are of high depth.
        let insert_priority =
            depth + insert_flag_bonus + (age_differential * age_differential) / 4 + i32::from(pv);
        let record_prority = i32::from(tte.depth) + record_flag_bonus;

        // replace the entry:
        // 1. if the entry is for a different position
        // 2. if it's an exact entry, and the old entry is not exact
        // 3. if the new entry is of higher priority than the old entry
        if tte.tag != tag
            || flag == Bound::Exact && tte.info.flag() != Bound::Exact
            || insert_priority * 3 >= record_prority * 2
        {
            let write = CacheEntry {
                tag,
                m: best_move,
                // normalise mate / TB scores:
                score: normalise_gt_truth_score(score, height)
                    .try_into()
                    .expect("score with value outwith i16"),
                depth: depth.try_into().unwrap(),
                info: PackedMeta::new(self.age, flag, pv),
                evaluation: eval.try_into().expect("eval with value outwith i16"),
            };
            cluster.entries[idx] = write;
            cluster.coherer = cluster.checksum();
            self.table[cluster_index].store(cluster);
        }
    }

    pub fn probe(&self, key: u64, ply: usize, clock: u8) -> Option<CacheResult> {
        let (index, tag) = self.derive_index_tag(key);

        let cluster = self.table[index].load();

        if cluster.checksum() != cluster.coherer {
            return None;
        }

        for entry in cluster.entries {
            if entry.tag != tag {
                continue;
            }

            return Some(CacheResult {
                mov: entry.m,
                depth: entry.depth.into(),
                bound: entry.info.flag(),
                value: reconstruct_gt_truth_score(entry.score.into(), ply, clock),
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
            let (index, _) = self.derive_index_tag(key);
            let entry = &self.table[index];

            // prefetch the entry:
            _mm_prefetch(
                std::ptr::from_ref::<RawCacheSet>(entry).cast::<i8>(),
                _MM_HINT_T0,
            );
        }
        #[cfg(target_arch = "aarch64")]
        {
            // Silence warnings on ARM, which lacks a prefetch equivalent.
            let _ = self;
            let _ = key;
        }
    }

    pub fn probe_move(&self, key: u64) -> Option<(Option<Move>, i32)> {
        self.probe(key, 0, 0)
            .map(|CacheResult { mov, value, .. }| (mov, value))
    }

    // TODO: rename and fix impl.
    pub fn hashfull(&self) -> usize {
        let mut hit = 0;
        for i in 0..2000 {
            let cluster = self.table[i].load();
            for i in 0..CLUSTER_SIZE {
                let entry = cluster.entries[i];
                if entry.tag != 0 && entry.info.age() == self.age {
                    hit += 1;
                }
            }
        }
        hit / (2 * CLUSTER_SIZE)
    }
}

/// Normalise a game-theoretic score for storage into the cache.
///
/// Positions in the tree can be reached through different sequences,
/// and the distance of a solved node from the root is relevant to its
/// value (#2 is better than #33). When storing and later loading from
/// the cache, the loading node may have a different height than the
/// storing node, which means that naïvely preserving game-theoretic
/// scores is incorrect.
///
/// Instead, scores are *shifted*. If a node is yielding mate-in-8-ply,
/// and is found 5 ply from the root, it is really mate-in-⟨8 – 5 = 3⟩-ply.
/// Then, when loading, game-theoretic scores are unshifted, so mate-in-3-ply
/// two steps from the root becomes mate-in-⟨3 + 2 = 5⟩-ply.
/// This is a fairly simple rerooting mechanism for such scores.
const fn normalise_gt_truth_score(mut score: i32, height: usize) -> i32 {
    #![allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    if score == VALUE_NONE {
        return VALUE_NONE;
    }
    if score >= MINIMUM_TB_WIN_SCORE {
        score += height as i32;
    } else if score <= -MINIMUM_TB_WIN_SCORE {
        score -= height as i32;
    }
    score
}

/// Reconstruct a game-theoretic score probed from the cache by rerooting
/// it to the current branch.
const fn reconstruct_gt_truth_score(score: i32, height: usize, clock: u8) -> i32 {
    #![allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    if score == VALUE_NONE {
        return VALUE_NONE;
    }
    if score >= MINIMUM_TB_WIN_SCORE {
        // If the score is mating, but it will take more moves than we have
        // halfmove clock in which to mate, return a wonderful but non-winning value.
        if score >= MINIMUM_MATE_SCORE && MATE_SCORE - score > 100 - clock as i32 {
            return MINIMUM_TB_WIN_SCORE - 1;
        }
        // If the score is TB-winning, but it will take more moves than we have
        // halfmove clock in which to TB-win, return a wonderful but non-winning value.
        if MINIMUM_TB_WIN_SCORE - score > 100 - clock as i32 {
            return MINIMUM_TB_WIN_SCORE - 1;
        }
        // re-root the score to the current branch.
        return score - height as i32;
    }
    if score <= -MINIMUM_TB_WIN_SCORE {
        // If the score is mated, but it will take more moves than we have
        // halfmove clock in which to mate, return a terrible but non-losing value.
        if score <= -MINIMUM_MATE_SCORE && MATE_SCORE + score > 100 - clock as i32 {
            return -MINIMUM_TB_WIN_SCORE + 1;
        }
        // If the score is TB-losing, but it will take more moves than we have
        // halfmove clock in which to TB-lose, return a terrible but non-losing value.
        if -MINIMUM_TB_WIN_SCORE + score > 100 - clock as i32 {
            return -MINIMUM_TB_WIN_SCORE + 1;
        }
        // re-root the score to the current branch.
        return score + height as i32;
    }
    score
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports)]
    use crate::{chess::piece::PieceType, chess::types::Square};

    use super::*;

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
