use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

use crate::{
    board::evaluation::MINIMUM_TB_WIN_SCORE,
    chessmove::Move,
    util::{depth::CompactDepthStorage, depth::Depth},
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgeAndFlag {
    data: u8,
}

impl AgeAndFlag {
    const NULL: Self = Self { data: 0 };

    const fn new(age: u8, flag: Bound) -> Self {
        Self { data: (age << 2) | flag as u8 }
    }

    const fn age(self) -> u8 {
        self.data >> 2
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct TTEntry {
    pub key: u16,                   // 16 bits
    pub m: Option<Move>,            // 16 bits
    pub score: i16,                 // 16 bits
    pub depth: CompactDepthStorage, // 8 bits, wrapper around a u8
    pub age_and_flag: AgeAndFlag,   // 6 + 2 bits, wrapper around a u8
    pub evaluation: i16,            // 16 bits
    pub pv: u8,                     // 8 bits
    pub dummy: [u8; 5],             // 40 bits
}

const _TT_ENTRIES_ARE_ONE_WORD: () = assert!(std::mem::size_of::<TTEntry>() == 16, "TT entry is not one word");

impl TTEntry {
    pub const NULL: Self = Self {
        key: 0,
        m: None,
        score: 0,
        depth: CompactDepthStorage::NULL,
        age_and_flag: AgeAndFlag::NULL,
        evaluation: 0,
        pv: 0,
        dummy: [0; 5],
    };
}

impl From<[u64; 2]> for TTEntry {
    fn from(data: [u64; 2]) -> Self {
        // SAFETY: This is safe because all fields of TTEntry are (at base) integral types,
        // and TTEntry is repr(C).
        unsafe { std::mem::transmute(data) }
    }
}

impl From<TTEntry> for [u64; 2] {
    fn from(entry: TTEntry) -> Self {
        // SAFETY: This is safe because all bitpatterns of `u64` are valid.
        unsafe { std::mem::transmute(entry) }
    }
}

const TT_ENTRY_SIZE: usize = std::mem::size_of::<TTEntry>();

#[derive(Debug)]
pub struct TT {
    table: Vec<[AtomicU64; 2]>,
    age: AtomicU8,
}

#[derive(Debug, Clone, Copy)]
pub struct TTView<'a> {
    table: &'a [[AtomicU64; 2]],
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

impl TT {
    const NULL_VALUE: u64 = 0;

    pub const fn new() -> Self {
        Self { table: Vec::new(), age: AtomicU8::new(0) }
    }

    pub fn resize(&mut self, bytes: usize) {
        let new_len = bytes / TT_ENTRY_SIZE;
        // dealloc the old table:
        self.table = Vec::new();
        // construct a new vec:
        // SAFETY: zeroed memory is a legal bitpattern for AtomicU64.
        unsafe {
            let layout = std::alloc::Layout::array::<[AtomicU64; 2]>(new_len).unwrap();
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
                        entry[0].store(Self::NULL_VALUE, Ordering::Relaxed);
                        entry[1].store(Self::NULL_VALUE, Ordering::Relaxed);
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
        let new_age = (self.age.load(Ordering::Relaxed) + 1) & 0b11_1111; // keep age in range [0, 63]
        self.age.store(new_age, Ordering::Relaxed);
    }

    pub fn size(&self) -> usize {
        self.table.len() * TT_ENTRY_SIZE
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
    ) {
        // get index into the table:
        let index = self.wrap_key(key);
        // create a small key from the full key:
        let key = TT::pack_key(key);
        // load the entry:
        let parts = [self.table[index][0].load(Ordering::Relaxed), self.table[index][1].load(Ordering::Relaxed)];
        let entry: TTEntry = parts.into();

        if best_move.is_none() && entry.key == key {
            // if we don't have a best move, and the entry is for the same position,
            // then we should retain the best move from the previous entry.
            best_move = entry.m;
        }

        // normalise mate / TB scores:
        let score = normalise_gt_truth_score(score, ply);

        // give entries a bonus for type:
        // exact = 3, lower = 2, upper = 1
        let insert_flag_bonus = i32::from(flag);
        let record_flag_bonus = i32::from(entry.age_and_flag.flag());

        // preferentially overwrite entries that are from searches on previous positions in the game.
        let age_differential = (i32::from(self.age) + 64 - i32::from(entry.age_and_flag.age())) & 0b11_1111;

        // we use quadratic scaling of the age to allow entries that aren't too old to be kept,
        // but to ensure that *really* old entries are overwritten even if they are of high depth.
        let insert_priority = depth + insert_flag_bonus + (age_differential * age_differential) / 4 + Depth::from(pv);
        let record_prority = Depth::from(entry.depth) + record_flag_bonus;

        // replace the entry:
        // 1. unconditionally if we're in the root node (holdover from TT-pv probing)
        // 2. if the entry is for a different position
        // 3. if it's an exact entry, and the old entry is not exact
        // 4. if the new entry is of higher priority than the old entry
        if entry.key != key
            || flag == Bound::Exact && entry.age_and_flag.flag() != Bound::Exact
            || insert_priority * 3 >= record_prority * 2
        {
            let write: [u64; 2] = TTEntry {
                key,
                m: best_move,
                score: score.try_into().expect(
                    "attempted to store a score with value outwith [i16::MIN, i16::MAX] in the transposition table",
                ),
                depth: depth.try_into().unwrap(),
                age_and_flag: AgeAndFlag::new(self.age, flag),
                evaluation: eval.try_into().expect(
                    "attempted to store an eval with value outwith [i16::MIN, i16::MAX] in the transposition table",
                ),
                pv: pv.into(),
                dummy: Default::default(),
            }
            .into();
            self.table[index][0].store(write[0], Ordering::Relaxed);
            self.table[index][1].store(write[1], Ordering::Relaxed);
        }
    }

    pub fn probe(&self, key: u64, ply: usize) -> Option<TTHit> {
        let index = self.wrap_key(key);
        let key = TT::pack_key(key);

        // load the entry:
        let parts = [self.table[index][0].load(Ordering::Relaxed), self.table[index][1].load(Ordering::Relaxed)];
        let entry: TTEntry = parts.into();

        if entry.key != key {
            return None;
        }

        let tt_move = entry.m;
        let tt_depth = entry.depth.into();
        let tt_bound = entry.age_and_flag.flag();

        // we can't store the score in a tagged union,
        // because we need to do mate score preprocessing.
        let tt_value = reconstruct_gt_truth_score(entry.score.into(), ply);

        Some(TTHit {
            mov: tt_move,
            depth: tt_depth,
            bound: tt_bound,
            value: tt_value,
            eval: entry.evaluation.into(),
            was_pv: entry.pv != 0,
        })
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
            _mm_prefetch(std::ptr::from_ref::<[AtomicU64; 2]>(entry).cast::<i8>(), _MM_HINT_T0);
        }
    }

    pub fn probe_for_provisional_info(&self, key: u64) -> Option<(Option<Move>, i32)> {
        self.probe(key, 0).map(|TTHit { mov, value, .. }| (mov, value))
    }

    pub fn hashfull(&self) -> usize {
        self.table.iter().take(1000).filter(|e| e[0].load(Ordering::Relaxed) != 0).count()
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
        let entry = TTEntry {
            key: 0x1234,
            m: Some(Move::new(Square::A1, Square::A2)),
            score: 0,
            depth: ZERO_PLY.try_into().unwrap(),
            age_and_flag: AgeAndFlag::new(63, Bound::Exact),
            evaluation: 1337,
            pv: 1,
            dummy: [0; 5],
        };
        let packed: [u64; 2] = entry.into();
        let unpacked: TTEntry = packed.into();
        assert_eq!(entry, unpacked);
    }

    #[test]
    fn null_tt_entry_is_zero() {
        let entry = TTEntry::NULL;
        let packed: [u64; 2] = entry.into();
        assert_eq!(packed, [TT::NULL_VALUE; 2]);
    }
}
