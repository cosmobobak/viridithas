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

pub struct MovePicker<const CAPTURES_ONLY: bool> {
    movelist: MoveList,
    index: usize,
    pub stage: Stage,
    tt_move: Move,
    killers: [Move; 2],
    pub skip_quiets: bool,
    see_threshold: i32,
}

pub type MainMovePicker = MovePicker<false>;
pub type CapturePicker = MovePicker<true>;

impl<const QSEARCH: bool> MovePicker<QSEARCH> {
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

    /// Returns true if a move was already yielded by the movepicker.
    pub fn was_tried_lazily(&self, m: Move) -> bool {
        m == self.tt_move
    }

    /// Select the next move to try. Returns None if there are no more moves to try.
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
            debug_assert_eq!(
                self.movelist.count, 0,
                "movelist not empty before capture generation"
            );
            position.generate_captures::<QSEARCH>(&mut self.movelist);
            for entry in &mut self.movelist.moves[..self.movelist.count] {
                entry.score = Self::score_capture(t, position, entry.mov, self.see_threshold);
            }
        }
        if self.stage == Stage::YieldGoodCaptures {
            if let Some(m) = self.yield_once() {
                if m.score >= WINNING_CAPTURE_SCORE {
                    return Some(m);
                }
                // the move was not winning, so we're going to
                // generate quiet moves next. As such, we decrement
                // the index so we can try this move again.
                self.index -= 1;
            }
            self.stage = if QSEARCH { Stage::Done } else { Stage::GenerateQuiets };
        }
        if self.stage == Stage::GenerateQuiets {
            self.stage = Stage::YieldRemaining;
            let start = self.movelist.count;
            position.generate_quiets(&mut self.movelist);
            let quiets = &mut self.movelist.moves[start..self.movelist.count];
            Self::score_quiets(self.killers, t, position, quiets);
        }
        if self.stage == Stage::YieldRemaining {
            if let Some(m) = self.yield_once() {
                return Some(m);
            }
            self.stage = Stage::Done;
        }
        None
    }

    /// Perform iterations of partial insertion sort.
    /// Extracts the best move from the unsorted portion of the movelist,
    /// or returns None if there are no more moves to try.
    ///
    /// Usually only one iteration is performed, but in the case where
    /// the best move has already been tried or doesn't meet SEE requirements,
    /// we will continue to iterate until we find a move that is valid.
    fn yield_once(&mut self) -> Option<MoveListEntry> {
        // If we have already tried all moves, return None.
        if self.index == self.movelist.count {
            return None;
        }

        let mut best_score = self.movelist.moves[self.index].score;
        let mut best_num = self.index;

        // find the best move in the unsorted portion of the movelist.
        for index in self.index + 1..self.movelist.count {
            let score = self.movelist.moves[index].score;
            if score > best_score {
                best_score = score;
                best_num = index;
            }
        }

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

    pub fn score_quiets(killers: [Move; 2], t: &ThreadData, pos: &Board, ms: &mut [MoveListEntry]) {
        // zero-out the ordering scores
        for m in ms.iter_mut() {
            m.score = 0;
        }

        t.get_history_scores(pos, ms);
        t.get_counter_move_history_scores(pos, ms);
        t.get_followup_history_scores(pos, ms);

        let counter_move = t.get_counter_move(pos);

        for m in ms {
            if killers[0] == m.mov {
                m.score = FIRST_ORDER_KILLER_SCORE;
            } else if killers[1] == m.mov {
                m.score = SECOND_ORDER_KILLER_SCORE;
            } else if counter_move == m.mov {
                m.score = COUNTER_MOVE_SCORE;
            }
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
        if pos.static_exchange_eval(m, see_threshold) {
            score += WINNING_CAPTURE_SCORE;
        }
        score
    }
}
