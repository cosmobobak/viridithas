use crate::{board::Board, chessmove::Move, lookups, piece::PieceType, threadlocal::ThreadData};

use super::{MoveList, MoveListEntry};

pub const TT_MOVE_SCORE: i32 = 20_000_000;
const FIRST_ORDER_KILLER_SCORE: i32 = 9_000_000;
const SECOND_ORDER_KILLER_SCORE: i32 = 8_000_000;
const COUNTER_MOVE_SCORE: i32 = 2_000_000;
pub const WINNING_CAPTURE_SCORE: i32 = 10_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stage {
    TTMove,
    GenerateCaptures,
    YieldGoodCaptures,
    GenerateQuiets,
    YieldRemaining,
    Done,
}

pub struct MovePicker<const CAPTURES_ONLY: bool, const DO_SEE: bool, const ROOT: bool> {
    movelist: MoveList,
    index: usize,
    pub stage: Stage,
    tt_move: Move,
    killers: [Move; 2],
    pub skip_quiets: bool,
    see_threshold: i32,
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
            see_threshold,
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
            self.stage = Stage::GenerateCaptures;
            if position.is_pseudo_legal(self.tt_move) {
                return Some(MoveListEntry { mov: self.tt_move, score: TT_MOVE_SCORE });
            }
        }
        if self.stage == Stage::GenerateCaptures {
            self.stage = Stage::YieldGoodCaptures;
            debug_assert_eq!(self.movelist.count, 0, "movelist not empty before capture generation");
            position.generate_captures(&mut self.movelist);
            for entry in &mut self.movelist.moves[..self.movelist.count] {
                entry.score = Self::score_capture(t, position, entry.mov, self.see_threshold);
            }
        }
        if self.stage == Stage::YieldGoodCaptures {
            if let Some(m) = self.yield_once() {
                return Some(m);
            }
            self.stage = if CAPTURES_ONLY {
                Stage::Done
            } else {
                Stage::GenerateQuiets
            };
        }
        if self.stage == Stage::GenerateQuiets {
            self.stage = Stage::YieldRemaining;
            let start = self.movelist.count;
            position.generate_quiets(&mut self.movelist);
            for entry in &mut self.movelist.moves[start..self.movelist.count] {
                entry.score = Self::score_quiet(self.killers, t, position, entry.mov);
            }
        }
        if self.stage == Stage::YieldRemaining {
            if let Some(m) = self.yield_once() {
                return Some(m);
            }
            self.stage = Stage::Done;
        }
        None
    }

    fn yield_once(&mut self) -> Option<MoveListEntry> {
        // If we have already tried all moves, return None.
        if self.index == self.movelist.count {
            return None;
        }

        let mut best_score = self.movelist.moves[self.index].score;
        let mut best_num = self.index;

        // find the best move in the unsorted portion of the movelist.
        for index in self.index + 1..self.movelist.count {
            // SAFETY: self.count is always less than 256, and self.index is always in bounds.
            let score = self.movelist.moves[index].score;
            if score > best_score {
                best_score = score;
                best_num = index;
            }
        }

        // SAFETY: best_num is drawn from self.index..self.count, which is always in bounds.
        let m = self.movelist.moves[best_num];

        // swap the best move with the first unsorted move.
        self.movelist.moves.swap(best_num, self.index);

        self.index += 1;

        let not_winning = m.score < WINNING_CAPTURE_SCORE;
        if self.was_tried_lazily(m.mov) || self.skip_quiets && not_winning {
            self.yield_once()
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
            let counter_move_history = t.counter_move_history_score(pos, m);
            i32::from(history) + i32::from(followup_history) + i32::from(counter_move_history)
        }
    }

    pub fn score_capture(_t: &ThreadData, pos: &Board, m: Move, see_threshold: i32) -> i32 {
        const QUEEN_PROMO_BONUS: i32 =
            lookups::get_mvv_lva_score(PieceType::QUEEN, PieceType::PAWN);
        let mut score = if m.is_ep() {
            lookups::get_mvv_lva_score(PieceType::PAWN, PieceType::PAWN)
        } else {
            lookups::get_mvv_lva_score(
                pos.captured_piece(m).piece_type(),
                pos.moved_piece(m).piece_type(),
            )
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
