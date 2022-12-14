use std::sync::atomic::{AtomicU64, Ordering};

use crate::{
    board::evaluation::MINIMUM_MATE_SCORE,
    chessmove::Move,
    definitions::{depth::{CompactDepthStorage, ZERO_PLY}, depth::Depth, INFINITY, MAX_DEPTH},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum HFlag {
    None = 0,
    UpperBound = 1,
    LowerBound = 2,
    Exact = 3,
}

macro_rules! impl_from_hflag {
    ($t:ty) => {
        impl From<HFlag> for $t {
            fn from(hflag: HFlag) -> Self {
                hflag as Self
            }
        }
    };
}

impl_from_hflag!(u8);
impl_from_hflag!(i32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgeAndFlag {
    data: u8,
}

impl AgeAndFlag {
    const NULL: Self = Self { data: 0 };

    const fn new(age: u8, flag: HFlag) -> Self {
        Self {
            data: (age << 2) | flag as u8,
        }
    }

    const fn age(self) -> u8 {
        self.data >> 2
    }

    fn flag(self) -> HFlag {
        match self.data & 0b11 {
            0 => HFlag::None,
            1 => HFlag::UpperBound,
            2 => HFlag::LowerBound,
            3 => HFlag::Exact,
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
    pub depth: CompactDepthStorage, // 8 bits
    pub age_and_flag: AgeAndFlag,   // 6 + 2 bits
}

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
        // SAFETY: This is safe because all fields of `TTEntry` are (at base) integral types.
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
pub struct TranspositionTable {
    table: Vec<AtomicU64>,
    age: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct TranspositionTableView<'a> {
    table: &'a [AtomicU64],
    age: u8,
}

pub struct TTHit {
    pub tt_move: Move,
    pub tt_depth: Depth,
    pub tt_bound: HFlag,
    pub tt_value: i32,
}

pub enum ProbeResult {
    Cutoff(i32),
    Hit(TTHit),
    Nothing,
}

impl TranspositionTable {
    pub const fn new() -> Self {
        Self { table: Vec::new(), age: 0 }
    }

    pub fn resize(&mut self, bytes: usize) {
        let new_len = bytes / TT_ENTRY_SIZE;
        self.table.resize_with(new_len, || AtomicU64::new(TTEntry::NULL.into()));
        self.table.shrink_to_fit();
        self.table.iter_mut().for_each(|x| x.store(TTEntry::NULL.into(), Ordering::SeqCst));
    }

    pub fn clear(&self) {
        self.table.iter().for_each(|x| x.store(TTEntry::NULL.into(), Ordering::SeqCst));
    }

    const fn pack_key(key: u64) -> u16 {
        #![allow(clippy::cast_possible_truncation)]
        key as u16
    }

    pub fn view(&self) -> TranspositionTableView {
        TranspositionTableView { table: &self.table, age: self.age }
    }

    pub fn increase_age(&mut self) {
        self.age = (self.age + 1) & 0b11_1111; // keep age in range [0, 63]
    }
}

impl<'a> TranspositionTableView<'a> {
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
        flag: HFlag,
        depth: Depth,
    ) {
        use HFlag::Exact;

        debug_assert!((0i32.into()..=MAX_DEPTH).contains(&depth), "depth: {depth}");
        debug_assert!(score >= -INFINITY);
        debug_assert!((0..=MAX_DEPTH.ply_to_horizon()).contains(&ply));

        let index = self.wrap_key(key);
        let key = TranspositionTable::pack_key(key);
        let entry: TTEntry = self.table[index].load(Ordering::SeqCst).into();

        if best_move.is_null() {
            best_move = entry.m;
        }

        let score = normalise_mate_score(score, ply);

        let record_depth: Depth = entry.depth.into();

        // give entries a bonus for type:
        // exact = 3, lower = 2, upper = 1
        let insert_flag_bonus = i32::from(flag);
        let record_flag_bonus = i32::from(entry.age_and_flag.flag());

        // preferentially overwrite entries that are from searches on previous positions in the game.
        let age_differential = (i32::from(self.age) + 64 - i32::from(entry.age_and_flag.age())) & 0b11_1111;

        let insert_depth = depth + insert_flag_bonus;
        let record_depth = record_depth + record_flag_bonus;

        if ROOT
            || entry.key != key
            || age_differential >= 2
            || flag == Exact && entry.age_and_flag.flag() != Exact
            || insert_depth * 3 >= record_depth * 2
        {
            let write = TTEntry {
                key,
                m: best_move,
                score: score.try_into().unwrap(),
                depth: depth.try_into().unwrap(),
                age_and_flag: AgeAndFlag::new(self.age, flag),
            };
            self.table[index].store(write.into(), Ordering::SeqCst);
        }
    }

    pub fn probe<const ROOT: bool>(
        &self,
        key: u64,
        ply: usize,
        alpha: i32,
        beta: i32,
        depth: Depth,
    ) -> ProbeResult {
        let index = self.wrap_key(key);
        let key = TranspositionTable::pack_key(key);

        debug_assert!((0i32.into()..=MAX_DEPTH).contains(&depth), "depth: {depth}");
        debug_assert!(alpha < beta);
        debug_assert!(alpha >= -INFINITY);
        debug_assert!(beta >= -INFINITY);
        debug_assert!((0..=MAX_DEPTH.ply_to_horizon()).contains(&ply));

        let entry: TTEntry = self.table[index].load(Ordering::SeqCst).into();

        if entry.key != key {
            return ProbeResult::Nothing;
        }

        let m = entry.m;
        let e_depth = entry.depth.into();

        debug_assert!((0i32.into()..=MAX_DEPTH).contains(&e_depth), "depth: {e_depth}");

        if e_depth < depth {
            return ProbeResult::Hit(TTHit {
                tt_move: m,
                tt_depth: e_depth,
                tt_bound: entry.age_and_flag.flag(),
                tt_value: entry.score.into(),
            });
        }

        // we can't store the score in a tagged union,
        // because we need to do mate score preprocessing.
        let score = reconstruct_mate_score(entry.score.into(), ply);

        debug_assert!(score >= -INFINITY);
        match entry.age_and_flag.flag() {
            HFlag::None => ProbeResult::Nothing, // this only gets hit when the hashkey manages to have all zeroes in the lower 16 bits.
            HFlag::UpperBound => {
                if !ROOT && score <= alpha {
                    ProbeResult::Cutoff(alpha) // never cutoff at root.
                } else {
                    ProbeResult::Hit(TTHit {
                        tt_move: m,
                        tt_depth: e_depth,
                        tt_bound: HFlag::UpperBound,
                        tt_value: entry.score.into(),
                    })
                }
            }
            HFlag::LowerBound => {
                if !ROOT && score >= beta {
                    ProbeResult::Cutoff(beta) // never cutoff at root.
                } else {
                    ProbeResult::Hit(TTHit {
                        tt_move: m,
                        tt_depth: e_depth,
                        tt_bound: HFlag::LowerBound,
                        tt_value: entry.score.into(),
                    })
                }
            }
            HFlag::Exact => {
                if ROOT {
                    ProbeResult::Hit(TTHit {
                        tt_move: m,
                        tt_depth: e_depth,
                        tt_bound: HFlag::Exact,
                        tt_value: entry.score.into(),
                    })
                } else {
                    ProbeResult::Cutoff(score) // never cutoff at root.
                }
            }
        }
    }

    pub fn probe_for_provisional_info(&self, key: u64) -> Option<(Move, i32)> {
        let result = self.probe::<true>(key, 0, -INFINITY, INFINITY, ZERO_PLY);
        match result {
            ProbeResult::Hit(TTHit {
                tt_move,
                tt_value,
                ..
            }) => Some((tt_move, tt_value)),
            _ => None,
        }
    }

    pub fn hashfull(&self) -> usize {
        self.table
            .iter()
            .take(1000)
            .filter(|e| <u64 as Into<TTEntry>>::into(e.load(Ordering::Relaxed)).key != 0)
            .count()
    }
}

const fn normalise_mate_score(mut score: i32, ply: usize) -> i32 {
    #![allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    if score > MINIMUM_MATE_SCORE {
        score += ply as i32;
    } else if score < -MINIMUM_MATE_SCORE {
        score -= ply as i32;
    }
    score
}

const fn reconstruct_mate_score(mut score: i32, ply: usize) -> i32 {
    #![allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    if score > MINIMUM_MATE_SCORE {
        score -= ply as i32;
    } else if score < -MINIMUM_MATE_SCORE {
        score += ply as i32;
    }
    score
}

mod tests {
    #![allow(unused_imports)]
    use crate::definitions::Square;

    use super::*;

    #[test]
    fn tt_entries_are_one_word() {
        assert_eq!(std::mem::size_of::<TTEntry>(), 8);
    }

    #[test]
    fn tt_entry_roundtrip() {
        let entry = TTEntry {
            key: 0x1234,
            m: Move::new(Square::A1, Square::A2, 0, 0),
            score: 0,
            depth: ZERO_PLY.try_into().unwrap(),
            age_and_flag: AgeAndFlag::new(0, HFlag::Exact),
        };
        let packed: u64 = entry.into();
        let unpacked: TTEntry = packed.into();
        assert_eq!(entry, unpacked);
    }
}
