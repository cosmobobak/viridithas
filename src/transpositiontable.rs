#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    dead_code
)]

use crate::{
    board::evaluation::IS_MATE_SCORE,
    chessmove::Move,
    definitions::{CompactDepthStorage, Depth, INFINITY, MAX_DEPTH},
    macros,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HFlag {
    None = 0,
    Alpha = 1,
    Beta = 2,
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
pub struct TTEntry {
    pub key: u64,
    pub m: Move,
    pub score: i32,
    pub depth: CompactDepthStorage,
    pub flag: HFlag,
}

impl TTEntry {
    pub const NULL: Self = Self {
        key: 0,
        m: Move::NULL,
        score: 0,
        depth: CompactDepthStorage::NULL,
        flag: HFlag::None,
    };
}

const TASTY_PRIME_NUMBER: usize = 12_582_917;

const MEGABYTE: usize = 1024 * 1024;
const TT_ENTRY_SIZE: usize = std::mem::size_of::<TTEntry>();

/// One option is to use 4MB of memory for the hashtable,
/// as my i5 has 6mb of L3 cache, so this endeavours to keep the
/// entire hashtable in L3 cache.
pub const IN_CACHE_TABLE_SIZE: usize = MEGABYTE * 4 / TT_ENTRY_SIZE;
/// Another option is just to use a ton of memory,
/// wahoooooooo
pub const BIG_TABLE_SIZE: usize = MEGABYTE * 4096 / TT_ENTRY_SIZE;
/// Middle-ground between the two.
pub const MEDIUM_TABLE_SIZE: usize = MEGABYTE * 512 / TT_ENTRY_SIZE;
/// Prime sized table that's around 256-512 megabytes.
pub const PRIME_TABLE_SIZE: usize = TASTY_PRIME_NUMBER;

pub const DEFAULT_TABLE_SIZE: usize = PRIME_TABLE_SIZE * 2;

#[derive(Debug)]
pub struct TranspositionTable<const SIZE: usize> {
    table: Vec<TTEntry>,
}

pub type DefaultTT = TranspositionTable<DEFAULT_TABLE_SIZE>;

pub struct TTHit { pub tt_move: Move, pub tt_depth: Depth, pub tt_bound: HFlag, pub tt_value: i32 }

pub enum ProbeResult {
    Cutoff(i32),
    Hit(TTHit),
    Nothing,
}

impl<const SIZE: usize> TranspositionTable<SIZE> {
    pub const fn new() -> Self {
        Self { table: Vec::new() }
    }

    pub fn clear(&mut self) {
        if self.table.is_empty() {
            self.table.resize(SIZE, TTEntry::NULL);
        } else {
            self.table.fill(TTEntry::NULL);
        }
    }

    pub fn clear_for_search(&mut self) {
        if self.table.is_empty() {
            self.table.resize(SIZE, TTEntry::NULL);
        } else {
            // do nothing.
        }
    }

    pub fn store(
        &mut self,
        key: u64,
        ply: usize,
        best_move: Move,
        score: i32,
        flag: HFlag,
        depth: Depth,
    ) {
        use HFlag::Exact;

        debug_assert!((0i32.into()..=MAX_DEPTH).contains(&depth), "depth: {depth}");
        debug_assert!(score >= -INFINITY);
        debug_assert!((0..=MAX_DEPTH.ply_to_horizon()).contains(&ply));

        let index = (key % SIZE as u64) as usize;
        let slot = &mut self.table[index];

        let score = normalise_mate_score(score, ply);

        let entry = TTEntry {
            key,
            m: best_move,
            score,
            depth: depth.try_into().unwrap(),
            flag,
        };

        let record_depth: Depth = slot.depth.into();

        let insert_flag_bonus = i32::from(flag);
        let record_flag_bonus = i32::from(slot.flag);

        let insert_depth = depth + insert_flag_bonus;
        let record_depth = record_depth + record_flag_bonus;

        if flag == Exact && slot.flag != Exact || insert_depth * 3 >= record_depth * 2 {
            *slot = entry;
        }
    }

    pub fn probe(
        &mut self,
        key: u64,
        ply: usize,
        alpha: i32,
        beta: i32,
        depth: Depth,
    ) -> ProbeResult {
        let index = (key % (SIZE as u64)) as usize;

        debug_assert!((0i32.into()..=MAX_DEPTH).contains(&depth), "depth: {depth}");
        debug_assert!(alpha < beta);
        debug_assert!(alpha >= -INFINITY);
        debug_assert!(beta >= -INFINITY);
        debug_assert!((0..=MAX_DEPTH.ply_to_horizon()).contains(&ply));

        let entry = &self.table[index];

        if entry.key != key { 
            return ProbeResult::Nothing;
        }

        let m = entry.m;
        let e_depth = entry.depth.into();

        debug_assert!((0i32.into()..=MAX_DEPTH).contains(&e_depth), "depth: {e_depth}");

        if e_depth < depth {
            return ProbeResult::Hit(TTHit { tt_move: m, tt_depth: e_depth, tt_bound: entry.flag, tt_value: entry.score });
        }

        // we can't store the score in a tagged union,
        // because we need to do mate score preprocessing.
        let score = reconstruct_mate_score(entry.score, ply);

        debug_assert!(score >= -INFINITY);
        match entry.flag {
            HFlag::None => unsafe { macros::inconceivable!() },
            HFlag::Alpha => {
                if score <= alpha {
                    ProbeResult::Cutoff(alpha)
                } else {
                    ProbeResult::Hit(TTHit { tt_move: m, tt_depth: e_depth, tt_bound: HFlag::Alpha, tt_value: entry.score })
                }
            }
            HFlag::Beta => {
                if score >= beta {
                    ProbeResult::Cutoff(beta)
                } else {
                    ProbeResult::Hit(TTHit { tt_move: m, tt_depth: e_depth, tt_bound: HFlag::Beta, tt_value: entry.score })
                }
            }
            HFlag::Exact => {
                ProbeResult::Cutoff(score)
            }
        }
    }
}

const fn normalise_mate_score(mut score: i32, ply: usize) -> i32 {
    if score > IS_MATE_SCORE {
        score += ply as i32;
    } else if score < -IS_MATE_SCORE {
        score -= ply as i32;
    }
    score
}

const fn reconstruct_mate_score(mut score: i32, ply: usize) -> i32 {
    if score > IS_MATE_SCORE {
        score -= ply as i32;
    } else if score < -IS_MATE_SCORE {
        score += ply as i32;
    }
    score
}
