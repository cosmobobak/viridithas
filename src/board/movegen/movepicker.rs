use crate::{
    board::Board,
    chessmove::Move,
    lookups,
    threadlocal::ThreadData, piece::PieceType,
};

use super::{MoveList, MoveListEntry};

pub const TT_MOVE_SCORE: i32 = 20_000_000;
const FIRST_ORDER_KILLER_SCORE: i32 = 9_000_000;
const SECOND_ORDER_KILLER_SCORE: i32 = 8_000_000;
const COUNTER_MOVE_SCORE: i32 = 2_000_000;
pub const WINNING_CAPTURE_SCORE: i32 = 10_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stage {
    TTMove,
    GenerateMoves,
    YieldMoves,
    Done,
}

pub struct MovePicker<const CAPTURES_ONLY: bool, const DO_SEE: bool, const ROOT: bool> {
    movelist: MoveList,
    index: usize,
    pub stage: Stage,
    tt_move: Move,
    killers: [Move; 2],
    pub skip_quiets: bool,
    see_threshold: i32
}

pub type MainMovePicker<const ROOT: bool> = MovePicker<false, true, ROOT>;
pub type CapturePicker = MovePicker<true, true, false>;

impl<const CAPTURES_ONLY: bool, const DO_SEE: bool, const ROOT: bool>
    MovePicker<CAPTURES_ONLY, DO_SEE, ROOT>
{
    pub const fn new(tt_move: Move, killers: [Move; 2], see_threshold: i32) -> Self {
        Self {
            movelist: MoveList::new(),
            index: 0,
            stage: Stage::TTMove,
            killers,
            tt_move,
            skip_quiets: false,
            see_threshold
        }
    }

    pub fn was_tried_lazily(&self, m: Move) -> bool {
        m == self.tt_move
    }

    /// Select the next move to try. Usually executes one iteration of partial insertion sort.
    pub fn next(&mut self, position: &Board, t: &ThreadData) -> Option<MoveListEntry> {
        if self.stage == Stage::Done {
            return None;
        }
        if self.stage == Stage::TTMove {
            self.stage = Stage::GenerateMoves;
            if position.is_pseudo_legal(self.tt_move) {
                return Some(MoveListEntry { mov: self.tt_move, score: TT_MOVE_SCORE });
            }
        }
        if self.stage == Stage::GenerateMoves {
            self.stage = Stage::YieldMoves;
            if CAPTURES_ONLY {
                position.generate_captures(&mut self.movelist);
                for entry in &mut self.movelist.moves[..self.movelist.count] {
                    entry.score = Self::score_capture(t, position, entry.mov, self.see_threshold);
                }
            } else {
                position.generate_moves(&mut self.movelist);
                for e in &mut self.movelist.moves[..self.movelist.count] {
                    e.score = if e.score == MoveListEntry::TACTICAL_SENTINEL {
                        Self::score_capture(t, position, e.mov, self.see_threshold)
                    } else {
                        Self::score_quiet(self.killers, t, position, e.mov)
                    };
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

        let not_winning = m.score < WINNING_CAPTURE_SCORE;
        if self.was_tried_lazily(m.mov) || self.skip_quiets && not_winning {
            self.next(position, t)
        } else {
            Some(m)
        }
    }

    pub fn score_quiet(killers: [Move; 2], t: &ThreadData, pos: &Board, m: Move) -> i32 {
        if killers[0] == m {
            FIRST_ORDER_KILLER_SCORE
        } else if killers[1] == m {
            SECOND_ORDER_KILLER_SCORE
        } else if t.is_countermove(pos, m) {
            COUNTER_MOVE_SCORE
        } else {
            let history = t.history_score(pos, m);
            let followup_history = t.followup_history_score(pos, m);
            i32::from(history + followup_history)
        }
    }

    pub fn score_capture(_t: &ThreadData, pos: &Board, m: Move, see_threshold: i32) -> i32 {
        const QUEEN_PROMO_BONUS: i32 = lookups::get_mvv_lva_score(PieceType::QUEEN, PieceType::PAWN);
        let mut score = if m.is_ep() {
            lookups::get_mvv_lva_score(PieceType::PAWN, PieceType::PAWN)
        } else {
            lookups::get_mvv_lva_score(pos.captured_piece(m).piece_type(), pos.moved_piece(m).piece_type())
        };
        if m.is_promo() {
            if m.promotion_type() == PieceType::QUEEN {
                score += QUEEN_PROMO_BONUS;
            } else {
                return -WINNING_CAPTURE_SCORE; // basically no point looking at these.
            }
        }
        if !DO_SEE || pos.static_exchange_eval(m, see_threshold) {
            score += WINNING_CAPTURE_SCORE;
        }
        score
    }
}
