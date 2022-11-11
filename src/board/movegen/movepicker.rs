use crate::{chessmove::Move, board::Board, threadlocal::ThreadData, lookups, definitions::PAWN};

use super::{MoveList, MoveListEntry};

pub const TT_MOVE_SCORE: i32 = 20_000_000;
const FIRST_ORDER_KILLER_SCORE: i32 = 9_000_000;
const SECOND_ORDER_KILLER_SCORE: i32 = 8_000_000;
const COUNTER_MOVE_SCORE: i32 = 2_000_000;
const THIRD_ORDER_KILLER_SCORE: i32 = 1_000_000;
const WINNING_CAPTURE_SCORE: i32 = 10_000_000;
const MOVEGEN_SEE_THRESHOLD: i32 = 0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    TTMove,
    GenerateMoves,
    YieldMoves,
}

pub struct MovePicker<const CAPTURES_ONLY: bool, const DO_SEE: bool> {
    movelist: MoveList,
    index: usize,
    skip_ordering: bool,
    stage: Stage,
    tt_move: Move,
}

impl<const CAPTURES_ONLY: bool, const DO_SEE: bool> MovePicker<CAPTURES_ONLY, DO_SEE> {
    pub const fn new(tt_move: Move) -> Self {
        Self { 
            movelist: MoveList::new(), 
            index: 0, 
            skip_ordering: false, 
            tt_move, 
            stage: Stage::TTMove,
        }
    }

    pub fn moves_made(&self) -> &[MoveListEntry] {
        &self.movelist.moves[..self.index]
    }

    pub fn skip_ordering(&mut self) {
        self.skip_ordering = true;
    }
    
    /// Select the next move to try. Usually executes one iteration of partial insertion sort.
    pub fn next(&mut self, position: &mut Board, t: &ThreadData) -> Option<MoveListEntry> {
        if self.stage == Stage::TTMove {
            self.stage = Stage::GenerateMoves;
            if position.is_pseudo_legal(self.tt_move) {
                return Some(MoveListEntry { mov: self.tt_move, score: TT_MOVE_SCORE });
            }
        }
        if self.stage == Stage::GenerateMoves {
            self.stage = Stage::YieldMoves;
            if CAPTURES_ONLY {
                position.generate_captures::<DO_SEE>(&mut self.movelist);
            } else {
                position.generate_moves::<DO_SEE>(&mut self.movelist);
            }
        }
        // If we have already tried all moves, return None.
        if self.index == self.movelist.count {
            return None;
        } else if self.skip_ordering {
            // If we are skipping ordering, just return the next move.
            let entry = unsafe { self.movelist.moves.get_unchecked_mut(self.index) };
            let score = Self::score_entry(t, position, entry);
            self.index += 1;
            if entry.mov == self.tt_move {
                return self.next(position, t);
            }
            return Some(MoveListEntry { mov: entry.mov, score });
        }

        // SAFETY: self.index is always in bounds.
        let first_entry = unsafe { self.movelist.moves.get_unchecked_mut(self.index) };
        let mut best_score = Self::score_entry(t, position, first_entry);
        let mut best_num = self.index;

        // find the best move in the unsorted portion of the movelist.
        for index in self.index + 1..self.movelist.count {
            // SAFETY: self.count is always less than 256, and self.index is always in bounds.
            let m = unsafe { self.movelist.moves.get_unchecked_mut(index) };
            let score = Self::score_entry(t, position, m);
            if score > best_score {
                best_score = score;
                best_num = index;
            }
        }

        debug_assert!(self.index < self.movelist.count);
        debug_assert!(best_num < self.movelist.count);
        debug_assert!(best_num >= self.index);

        // SAFETY: best_num is drawn from self.index..self.count, which is always in bounds.
        let &m = unsafe { self.movelist.moves.get_unchecked(best_num) };

        // swap the best move with the first unsorted move.
        // SAFETY: best_num is drawn from self.index..self.count, which is always in bounds.
        // and self.index is always in bounds.
        unsafe {
            *self.movelist.moves.get_unchecked_mut(best_num) = *self.movelist.moves.get_unchecked(self.index);
        }

        self.index += 1;

        if m.mov == self.tt_move {
            self.next(position, t)
        } else {
            Some(m)
        }
    }

    pub fn score_entry(t: &ThreadData, pos: &Board, entry: &mut MoveListEntry) -> i32 {
        if entry.score != MoveList::UNSCORED {
            return entry.score;
        }
        let m = entry.mov;
        let score = if m.is_quiet() {
            Self::score_quiet(t, pos, m)
        } else {
            Self::score_capture(t, pos, m)
        };
        entry.score = score;
        score
    }

    pub fn score_quiet(t: &ThreadData, pos: &Board, m: Move) -> i32 {
        let killer_entry = t.killer_move_table[pos.height];

        if killer_entry[0] == m {
            FIRST_ORDER_KILLER_SCORE
        } else if killer_entry[1] == m {
            SECOND_ORDER_KILLER_SCORE
        } else if t.is_countermove(pos, m) {
            // move that refuted the previous move
            COUNTER_MOVE_SCORE
        } else if t.is_third_order_killer(pos, m) {
            // killer from two moves ago
            THIRD_ORDER_KILLER_SCORE
        } else {
            let history = t.history_score(pos, m);
            let followup_history = t.followup_history_score(pos, m);
            history + 2 * followup_history
        }
    }

    pub fn score_capture(_t: &ThreadData, pos: &Board, m: Move) -> i32 {
        if m.is_promo() {
            let mut score = lookups::get_mvv_lva_score(m.promotion(), PAWN);
            if !DO_SEE || pos.static_exchange_eval(m, MOVEGEN_SEE_THRESHOLD) {
                score += WINNING_CAPTURE_SCORE;
            }
            score
        } else if m.is_ep() {
            let mut score = 1050; // the score for PxP in MVVLVA
            if !DO_SEE || pos.static_exchange_eval(m, MOVEGEN_SEE_THRESHOLD) {
                score += WINNING_CAPTURE_SCORE;
            }
            score
        } else {
            let mut score = lookups::get_mvv_lva_score(m.capture(), pos.piece_at(m.from()));
            if !DO_SEE || pos.static_exchange_eval(m, MOVEGEN_SEE_THRESHOLD) {
                score += WINNING_CAPTURE_SCORE;
            }
            score
        }
    }
}
