#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    dead_code
)]

use crate::{
    board::evaluation::IS_MATE_SCORE,
    chessmove::Move,
    definitions::{INFINITY, MAX_DEPTH},
    opt,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HFlag {
    None,
    Alpha,
    Beta,
    Exact,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TTEntry {
    pub key: u64,
    pub m: Move,
    pub score: i32,
    pub depth: i16,
    pub flag: HFlag,
}

impl TTEntry {
    pub const NULL: Self = Self {
        key: 0,
        m: Move::NULL,
        score: 0,
        depth: 0,
        flag: HFlag::None,
    };
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Bucket {
    pub depth_preferred: TTEntry,
    pub always_replace: TTEntry,
}

impl Bucket {
    pub const NULL: Self = Self {
        depth_preferred: TTEntry::NULL,
        always_replace: TTEntry::NULL,
    };
}

const TASTY_PRIME_NUMBER: usize = 12_582_917;

const MEGABYTE: usize = 1024 * 1024;
const TT_ENTRY_SIZE: usize = std::mem::size_of::<Bucket>();

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

pub const DEFAULT_TABLE_SIZE: usize = PRIME_TABLE_SIZE;

#[derive(Debug)]
pub struct TranspositionTable<const SIZE: usize> {
    table: Vec<Bucket>,
}

pub type DefaultTT = TranspositionTable<DEFAULT_TABLE_SIZE>;

pub enum ProbeResult {
    Cutoff(i32),
    BestMove(Move),
    Nothing,
}

impl<const SIZE: usize> TranspositionTable<SIZE> {
    pub fn new() -> Self {
        Self {
            table: vec![Bucket::NULL; SIZE],
        }
    }

    pub fn clear(&mut self) {
        self.table.fill(Bucket::NULL);
    }

    pub fn store(
        &mut self,
        key: u64,
        ply: usize,
        best_move: Move,
        score: i32,
        flag: HFlag,
        depth: i32,
    ) {
        let index = (key % SIZE as u64) as usize;

        debug_assert!((1..=MAX_DEPTH).contains(&depth));
        debug_assert!(score >= -INFINITY);
        debug_assert!((0..=MAX_DEPTH as usize).contains(&ply));

        let mut score = score;
        if score > IS_MATE_SCORE {
            score += ply as i32;
        } else if score < -IS_MATE_SCORE {
            score -= ply as i32;
        }

        let slot = &mut self.table[index];

        let entry = TTEntry {
            key,
            m: best_move,
            score,
            depth: depth.try_into().unwrap(),
            flag,
        };

        if depth >= i32::from(slot.depth_preferred.depth) {
            slot.depth_preferred = entry;
        } else {
            slot.always_replace = entry;
        }
    }

    pub fn probe(
        &mut self,
        key: u64,
        ply: usize,
        alpha: i32,
        beta: i32,
        depth: i32,
    ) -> ProbeResult {
        let index = (key % (SIZE as u64)) as usize;

        debug_assert!((1..=MAX_DEPTH).contains(&depth));
        debug_assert!(alpha < beta);
        debug_assert!(alpha >= -INFINITY);
        debug_assert!(beta >= -INFINITY);
        debug_assert!((0..=MAX_DEPTH as usize).contains(&ply));

        let slot = &self.table[index];
        let e1 = &slot.depth_preferred;
        let e2 = &slot.always_replace;

        if e1.key == key || e2.key == key {
            let entry = if e1.key == key { e1 } else { e2 };
            let m = entry.m;
            let e_depth = i32::from(entry.depth);
            if e_depth >= depth {
                debug_assert!((1..=MAX_DEPTH).contains(&e_depth));

                // we can't store the score in a tagged union,
                // because we need to do mate score preprocessing.
                let mut score = entry.score;
                if score > IS_MATE_SCORE {
                    score -= ply as i32;
                } else if score < -IS_MATE_SCORE {
                    score += ply as i32;
                }

                debug_assert!(score >= -INFINITY);
                match entry.flag {
                    HFlag::None => unsafe { opt::impossible!() },
                    HFlag::Alpha => {
                        if score <= alpha {
                            return ProbeResult::Cutoff(alpha);
                        }
                    }
                    HFlag::Beta => {
                        if score >= beta {
                            return ProbeResult::Cutoff(beta);
                        }
                    }
                    HFlag::Exact => {
                        return ProbeResult::Cutoff(score);
                    }
                }
            }
            return ProbeResult::BestMove(m);
        }

        ProbeResult::Nothing
    }

    pub fn clear_for_search(&mut self) {
        let _ = self;
    }
}
