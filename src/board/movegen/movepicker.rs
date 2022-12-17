use crate::{
    board::Board,
    chessmove::Move,
    definitions::{make_piece, PAWN, QUEEN},
    lookups,
    threadlocal::ThreadData,
};

use super::{MoveList, MoveListEntry};

pub const TT_MOVE_SCORE: i32 = 20_000_000;
const FIRST_ORDER_KILLER_SCORE: i32 = 9_000_000;
const SECOND_ORDER_KILLER_SCORE: i32 = 8_000_000;
const COUNTER_MOVE_SCORE: i32 = 2_000_000;
const THIRD_ORDER_KILLER_SCORE: i32 = 1_000_000;
const WINNING_CAPTURE_SCORE: i32 = 10_000_000;
const MOVEGEN_SEE_THRESHOLD: i32 = 0;
pub const ILLEGAL_MOVE_SCORE: i32 = -TT_MOVE_SCORE;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stage {
    TTMove,
    GenerateMoves,
    YieldMoves,
}

pub struct MovePicker<const CAPTURES_ONLY: bool, const DO_SEE: bool, const ROOT: bool> {
    movelist: MoveList,
    index: usize,
    stage: Stage,
    tt_move: Move,
    killers: [Move; 3],
    pub skip_quiets: bool,
}

pub type MainMovePicker<const ROOT: bool> = MovePicker<false, true, ROOT>;
pub type CapturePicker = MovePicker<true, true, false>;

impl<const CAPTURES_ONLY: bool, const DO_SEE: bool, const ROOT: bool>
    MovePicker<CAPTURES_ONLY, DO_SEE, ROOT>
{
    pub const fn new(tt_move: Move, killers: [Move; 3]) -> Self {
        Self {
            movelist: MoveList::new(),
            index: 0,
            stage: Stage::TTMove,
            killers,
            tt_move,
            skip_quiets: false,
        }
    }

    pub fn was_tried_lazily(&self, m: Move) -> bool {
        m == self.tt_move
    }

    /// Select the next move to try. Usually executes one iteration of partial insertion sort.
    pub fn next(&mut self, position: &Board, t: &ThreadData) -> Option<MoveListEntry> {
        if self.stage == Stage::TTMove {
            self.stage = Stage::GenerateMoves;
            // let mut temp_ml = MoveList::new();
            // position.generate_moves(&mut temp_ml);
            if position.is_pseudo_legal(self.tt_move) {
                // debug_assert!(temp_ml.iter().any(|&mov| mov == self.tt_move), "tt_move is not pseudo legal: got {} in position {}", self.tt_move, position.fen());
                return Some(MoveListEntry { mov: self.tt_move, score: TT_MOVE_SCORE });
            }
            // debug_assert!(temp_ml.iter().all(|&mov| mov != self.tt_move), "tt_move is pseudo legal but was rejected: got {} in position {}", self.tt_move, position.fen());
        }
        if self.stage == Stage::GenerateMoves {
            self.stage = Stage::YieldMoves;
            if CAPTURES_ONLY {
                position.generate_captures(&mut self.movelist);
                for entry in &mut self.movelist.moves[..self.movelist.count] {
                    entry.score = Self::score_capture(t, position, entry.mov);
                }
            } else {
                position.generate_moves(&mut self.movelist);
                if ROOT {
                    // Root move generation is special because we need to sort the moves by score.
                    for e in &mut self.movelist.moves[..self.movelist.count] {
                        let normal_ordering_score = if e.score == MoveListEntry::TACTICAL_SENTINEL {
                            Self::score_capture(t, position, e.mov)
                        } else {
                            Self::score_quiet(self.killers, t, position, e.mov)
                        };
                        let root_score = t.score_at_root(e.mov);
                        e.score = root_score.checked_add(normal_ordering_score).unwrap();
                    }
                } else {
                    for e in &mut self.movelist.moves[..self.movelist.count] {
                        e.score = if e.score == MoveListEntry::TACTICAL_SENTINEL {
                            Self::score_capture(t, position, e.mov)
                        } else {
                            Self::score_quiet(self.killers, t, position, e.mov)
                        };
                    }
                }
            }
        }
        // If we have already tried all moves, return None.
        if self.index == self.movelist.count {
            return None;
        }

        // SAFETY: self.index is always in bounds.
        let mut best_score = unsafe { self.movelist.moves.get_unchecked(self.index).score };
        let mut best_num = self.index;

        // find the best move in the unsorted portion of the movelist.
        for index in self.index + 1..self.movelist.count {
            // SAFETY: self.count is always less than 256, and self.index is always in bounds.
            let score = unsafe { self.movelist.moves.get_unchecked(index).score };
            if score > best_score {
                best_score = score;
                best_num = index;
            }
        }

        // SAFETY: best_num is drawn from self.index..self.count, which is always in bounds.
        let &m = unsafe { self.movelist.moves.get_unchecked(best_num) };

        // swap the best move with the first unsorted move.
        // SAFETY: best_num is drawn from self.index..self.count, which is always in bounds.
        // and self.index is always in bounds.
        unsafe {
            *self.movelist.moves.get_unchecked_mut(best_num) =
                *self.movelist.moves.get_unchecked(self.index);
        }

        self.index += 1;

        if self.was_tried_lazily(m.mov) || (m.score < WINNING_CAPTURE_SCORE && self.skip_quiets) {
            self.next(position, t)
        } else {
            Some(m)
        }
    }

    pub fn score_quiet(killers: [Move; 3], t: &ThreadData, pos: &Board, m: Move) -> i32 {
        if killers[0] == m {
            FIRST_ORDER_KILLER_SCORE
        } else if killers[1] == m {
            SECOND_ORDER_KILLER_SCORE
        } else if t.is_countermove(pos, m) {
            // move that refuted the previous move
            COUNTER_MOVE_SCORE
        } else if killers[2] == m {
            // killer from two moves ago
            THIRD_ORDER_KILLER_SCORE
        } else {
            let history = t.history_score(pos, m);
            let followup_history = t.followup_history_score(pos, m);
            history + followup_history
        }
    }

    pub fn score_capture(_t: &ThreadData, pos: &Board, m: Move) -> i32 {
        let turn = pos.turn();
        let mut score;
        if m.is_promo() {
            score = lookups::get_mvv_lva_score(make_piece(turn, m.promotion_type()), PAWN);
        } else if m.is_ep() {
            score = 1050; // the score for PxP in MVVLVA
        } else {
            score = lookups::get_mvv_lva_score(pos.captured_piece(m), pos.piece_at(m.from()));
        }
        if !DO_SEE || pos.static_exchange_eval(m, MOVEGEN_SEE_THRESHOLD) {
            score += WINNING_CAPTURE_SCORE;
        }
        if m.is_promo() && m.promotion_type() == QUEEN {
            score += WINNING_CAPTURE_SCORE / 2;
        }
        score
    }
}
