#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]

use crate::{
    chessmove::Move,
    definitions::{INFINITY, MAX_DEPTH},
    evaluation::IS_MATE_SCORE,
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
    pub depth: usize,
    /// encode in tagged union instead.
    pub flag: HFlag,
}

pub const TT_ENTRY_SIZE: usize = std::mem::size_of::<TTEntry>();

const TASTY_PRIME_NUMBER: usize = 12_582_917;

const MEGABYTE: usize = 1024 * 1024;

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranspositionTable<const SIZE: usize, const REPLACEMENT_STRATEGY: u8> {
    table: Vec<TTEntry>,
    cutoffs: u64,
    entries: u64,
    new_writes: u64,
    overwrites: u64,
    hits: u64,
}

pub type DefaultTT = TranspositionTable<DEFAULT_TABLE_SIZE, DEPTH_PREFERRED>;

pub enum ProbeResult {
    Cutoff(i32),
    BestMove(Move),
    Nothing,
}

pub const ALWAYS_OVERWRITE: u8 = 0;
pub const DEPTH_PREFERRED: u8 = 1;

impl<const SIZE: usize, const REPLACEMENT_STRATEGY: u8>
    TranspositionTable<SIZE, REPLACEMENT_STRATEGY>
{
    pub fn new() -> Self {
        Self {
            table: vec![
                TTEntry {
                    key: 0,
                    m: Move::null(),
                    score: 0,
                    depth: 0,
                    flag: HFlag::None,
                };
                SIZE
            ],
            cutoffs: 0,
            entries: 0,
            new_writes: 0,
            overwrites: 0,
            hits: 0,
        }
    }

    pub fn store(
        &mut self,
        key: u64,
        ply: usize,
        best_move: Move,
        score: i32,
        flag: HFlag,
        depth: usize,
    ) {
        let index = (key % SIZE as u64) as usize;

        debug_assert!((1..=MAX_DEPTH).contains(&depth));
        debug_assert!(score >= -INFINITY);
        debug_assert!((0..=MAX_DEPTH).contains(&ply));

        if self.table[index].key == 0 {
            self.new_writes += 1;
        } else {
            self.overwrites += 1;
        }

        let mut score = score;
        if score > IS_MATE_SCORE {
            score += ply as i32;
        } else if score < -IS_MATE_SCORE {
            score -= ply as i32;
        }

        if REPLACEMENT_STRATEGY == ALWAYS_OVERWRITE {
            self.table[index] = TTEntry {
                key,
                m: best_move,
                score,
                depth,
                flag,
            };
        } else if REPLACEMENT_STRATEGY == DEPTH_PREFERRED {
            if depth >= self.table[index].depth {
                self.table[index] = TTEntry {
                    key,
                    m: best_move,
                    score,
                    depth,
                    flag,
                };
            }
        } else {
            panic!("Unknown strategy");
        }
    }

    pub fn probe(
        &mut self,
        key: u64,
        ply: usize,
        alpha: i32,
        beta: i32,
        depth: usize,
    ) -> ProbeResult {
        let index = (key % (SIZE as u64)) as usize;

        debug_assert!((1..=MAX_DEPTH).contains(&depth));
        debug_assert!(alpha < beta);
        debug_assert!(alpha >= -INFINITY);
        debug_assert!(beta >= -INFINITY);
        debug_assert!((0..=MAX_DEPTH).contains(&ply));

        let entry = &self.table[index];

        if entry.key == key {
            let m = entry.m;
            if entry.depth >= depth {
                self.hits += 1;

                debug_assert!(entry.depth >= 1 && entry.depth <= MAX_DEPTH);

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
                    HFlag::None => unreachable!(),
                    HFlag::Alpha => {
                        if score <= alpha {
                            score = alpha;
                            self.cutoffs += 1;
                            return ProbeResult::Cutoff(score);
                        }
                    }
                    HFlag::Beta => {
                        if score >= beta {
                            score = beta;
                            self.cutoffs += 1;
                            return ProbeResult::Cutoff(score);
                        }
                    }
                    HFlag::Exact => {
                        self.cutoffs += 1;
                        return ProbeResult::Cutoff(score);
                    }
                }
            }
            return ProbeResult::BestMove(m);
        }

        ProbeResult::Nothing
    }

    pub fn clear_for_search(&mut self) {
        self.overwrites = 0;
        self.hits = 0;
        self.cutoffs = 0;
    }
}
