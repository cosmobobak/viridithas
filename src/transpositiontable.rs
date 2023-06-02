use std::sync::atomic::{AtomicU64, Ordering};

use crate::{
    board::evaluation::MINIMUM_TB_WIN_SCORE,
    chessmove::Move,
    definitions::{
        depth::Depth,
        depth::{CompactDepthStorage, ZERO_PLY},
        INFINITY, MAX_DEPTH,
    },
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
    pub m: Move,                    // 16 bits
    pub score: i16,                 // 16 bits
    pub depth: CompactDepthStorage, // 8 bits, wrapper around a u8
    pub age_and_flag: AgeAndFlag,   // 6 + 2 bits, wrapper around a u8
}

const _TT_ENTRIES_ARE_ONE_WORD: () =
    assert!(std::mem::size_of::<TTEntry>() == 8, "TT entry is not one word");

impl TTEntry {
    pub const NULL: Self = Self {
        key: 0,
        m: Move::NULL,
        score: 0,
        depth: CompactDepthStorage::NULL,
        age_and_flag: AgeAndFlag::NULL,
    };
}

impl From<u64> for TTEntry {
    fn from(data: u64) -> Self {
        // SAFETY: This is safe because all fields of TTEntry are (at base) integral types,
        // and TTEntry is repr(C).
        unsafe { std::mem::transmute(data) }
    }
}

impl From<TTEntry> for u64 {
    fn from(entry: TTEntry) -> Self {
        // SAFETY: This is safe because all bitpatterns of `u64` are valid.
        unsafe { std::mem::transmute(entry) }
    }
}

const TT_ENTRY_SIZE: usize = std::mem::size_of::<TTEntry>();

#[derive(Debug)]
pub struct TT {
    table: Vec<AtomicU64>,
    age: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct TTView<'a> {
    table: &'a [AtomicU64],
    age: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct TTHit {
    pub tt_move: Move,
    pub tt_depth: Depth,
    pub tt_bound: Bound,
    pub tt_value: i32,
}

pub enum ProbeResult {
    Cutoff(TTHit),
    Hit(TTHit),
    Nothing,
}

impl TT {
    const NULL_VALUE: u64 = 0;

    pub const fn new() -> Self {
        Self { table: Vec::new(), age: 0 }
    }

    pub fn resize(&mut self, bytes: usize) {
        let new_len = bytes / TT_ENTRY_SIZE;
        self.table.resize_with(new_len, || AtomicU64::new(Self::NULL_VALUE));
        self.table.shrink_to_fit();
        self.table.iter_mut().for_each(|x| x.store(Self::NULL_VALUE, Ordering::SeqCst));
    }

    pub fn clear(&self) {
        self.table.iter().for_each(|x| x.store(Self::NULL_VALUE, Ordering::SeqCst));
    }

    const fn pack_key(key: u64) -> u16 {
        #![allow(clippy::cast_possible_truncation)]
        key as u16
    }

    pub fn view(&self) -> TTView {
        TTView { table: &self.table, age: self.age }
    }

    pub fn increase_age(&mut self) {
        self.age = (self.age + 1) & 0b11_1111; // keep age in range [0, 63]
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

    pub fn store<const ROOT: bool>(
        &self,
        key: u64,
        ply: usize,
        mut best_move: Move,
        score: i32,
        flag: Bound,
        depth: Depth,
    ) {
        debug_assert!((ZERO_PLY..=MAX_DEPTH).contains(&depth), "depth: {depth}");
        debug_assert!(score >= -INFINITY);
        debug_assert!((0..=MAX_DEPTH.ply_to_horizon()).contains(&ply));

        let index = self.wrap_key(key);
        let key = TT::pack_key(key);
        let entry: TTEntry = self.table[index].load(Ordering::SeqCst).into();

        if best_move.is_null() {
            best_move = entry.m;
        }

        let score = normalise_gt_truth_score(score, ply);

        let record_depth: Depth = entry.depth.into();

        // give entries a bonus for type:
        // exact = 3, lower = 2, upper = 1
        let insert_flag_bonus = i32::from(flag);
        let record_flag_bonus = i32::from(entry.age_and_flag.flag());

        // preferentially overwrite entries that are from searches on previous positions in the game.
        let age_differential =
            (i32::from(self.age) + 64 - i32::from(entry.age_and_flag.age())) & 0b11_1111;

        // we use quadratic scaling of the age to allow entries that aren't too old to be kept,
        // but to ensure that *really* old entries are overwritten even if they are of high depth.
        let insert_priority = depth + insert_flag_bonus + (age_differential * age_differential) / 4;
        let record_prority = record_depth + record_flag_bonus;

        if ROOT
            || entry.key != key
            || flag == Bound::Exact && entry.age_and_flag.flag() != Bound::Exact
            || insert_priority * 3 >= record_prority * 2
        {
            let write = TTEntry {
                key,
                m: best_move,
                score: score.try_into().expect("attempted to store a score with value outwith [i16::MIN, i16::MAX] in the transposition table"),
                depth: depth.try_into().unwrap(),
                age_and_flag: AgeAndFlag::new(self.age, flag),
            };
            self.table[index].store(write.into(), Ordering::SeqCst);
        }
    }

    pub fn probe(
        &self,
        key: u64,
        ply: usize,
        alpha: i32,
        beta: i32,
        depth: Depth,
        do_not_cut: bool,
    ) -> ProbeResult {
        let index = self.wrap_key(key);
        let key = TT::pack_key(key);

        debug_assert!((ZERO_PLY..=MAX_DEPTH).contains(&depth), "depth: {depth}");
        debug_assert!(alpha < beta);
        debug_assert!(alpha >= -INFINITY);
        debug_assert!(beta >= -INFINITY);
        debug_assert!((0..=MAX_DEPTH.ply_to_horizon()).contains(&ply));

        let entry: TTEntry = self.table[index].load(Ordering::SeqCst).into();

        if entry.key != key {
            return ProbeResult::Nothing;
        }

        let tt_move = entry.m;
        let tt_depth = entry.depth.into();
        let tt_bound = entry.age_and_flag.flag();

        debug_assert!((ZERO_PLY..=MAX_DEPTH).contains(&tt_depth), "depth: {tt_depth}");

        // we can't store the score in a tagged union,
        // because we need to do mate score preprocessing.
        let tt_value = reconstruct_gt_truth_score(entry.score.into(), ply);

        if tt_depth < depth {
            return ProbeResult::Hit(TTHit { tt_move, tt_depth, tt_bound, tt_value });
        }

        debug_assert!(tt_value >= -INFINITY);
        match tt_bound {
            Bound::None => ProbeResult::Nothing, // this only gets hit when the hashkey manages to have all zeroes in the lower 16 bits.
            Bound::Upper => {
                if tt_value <= alpha && !do_not_cut {
                    ProbeResult::Cutoff(TTHit {
                        tt_move,
                        tt_depth,
                        tt_bound: Bound::Upper,
                        tt_value,
                    }) // never cutoff at root.
                } else {
                    ProbeResult::Hit(TTHit { tt_move, tt_depth, tt_bound: Bound::Upper, tt_value })
                }
            }
            Bound::Lower => {
                if tt_value >= beta && !do_not_cut {
                    ProbeResult::Cutoff(TTHit {
                        tt_move,
                        tt_depth,
                        tt_bound: Bound::Lower,
                        tt_value,
                    }) // never cutoff at root.
                } else {
                    ProbeResult::Hit(TTHit { tt_move, tt_depth, tt_bound: Bound::Lower, tt_value })
                }
            }
            Bound::Exact => {
                if do_not_cut {
                    ProbeResult::Hit(TTHit { tt_move, tt_depth, tt_bound: Bound::Exact, tt_value })
                } else {
                    ProbeResult::Cutoff(TTHit {
                        tt_move,
                        tt_depth,
                        tt_bound: Bound::Exact,
                        tt_value,
                    }) // never cutoff at root.
                }
            }
        }
    }

    pub fn prefetch(&self, key: u64) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};

            // get a reference to the entry in the table:
            let index = self.wrap_key(key);
            let entry = &self.table[index];

            // prefetch the entry:
            _mm_prefetch((entry as *const AtomicU64).cast::<i8>(), _MM_HINT_T0);
        }
    }

    pub fn probe_for_provisional_info(&self, key: u64) -> Option<(Move, i32)> {
        let result = self.probe(key, 0, -INFINITY, INFINITY, ZERO_PLY, true);
        match result {
            ProbeResult::Hit(TTHit { tt_move, tt_value, .. }) => Some((tt_move, tt_value)),
            _ => None,
        }
    }

    pub fn hashfull(&self) -> usize {
        self.table.iter().take(1000).filter(|e| e.load(Ordering::Relaxed) != 0).count()
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
    use crate::{definitions::Square, piece::PieceType};

    use super::*;

    #[test]
    fn tt_entry_roundtrip() {
        let entry = TTEntry {
            key: 0x1234,
            m: Move::new(Square::A1, Square::A2),
            score: 0,
            depth: ZERO_PLY.try_into().unwrap(),
            age_and_flag: AgeAndFlag::new(63, Bound::Exact),
        };
        let packed: u64 = entry.into();
        let unpacked: TTEntry = packed.into();
        assert_eq!(entry, unpacked);
    }

    #[test]
    fn null_tt_entry_is_zero() {
        let entry = TTEntry::NULL;
        let packed: u64 = entry.into();
        assert_eq!(packed, TT::NULL_VALUE);
    }
}
