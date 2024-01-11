use crate::{
    board::{
        history,
        movegen::bitboards,
        Board,
    },
    chessmove::Move,
    piece::PieceType,
    threadlocal::ThreadData,
};

use super::{MoveList, MoveListEntry};

pub const TT_MOVE_SCORE: i32 = 20_000_000;
pub const FIRST_KILLER_SCORE: i32 = 9_000_000;
pub const SECOND_KILLER_SCORE: i32 = 8_000_000;
pub const COUNTER_MOVE_SCORE: i32 = 2_000_000;
pub const WINNING_CAPTURE_SCORE: i32 = 10_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stage {
    TTMove,
    GenerateCaptures,
    YieldGoodCaptures,
    YieldKiller1,
    YieldKiller2,
    YieldCounterMove,
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
    counter_move: Move,
    pub skip_quiets: bool,
    see_threshold: i32,
}

pub type MainMovePicker = MovePicker<false>;
pub type CapturePicker = MovePicker<true>;

impl<const QSEARCH: bool> MovePicker<QSEARCH> {
    pub fn new(tt_move: Move, killers: [Move; 2], counter_move: Move, see_threshold: i32) -> Self {
        debug_assert!(
            killers[0].is_null() || killers[0] != killers[1],
            "Killers are both {}",
            killers[0]
        );
        Self {
            movelist: MoveList::new(),
            index: 0,
            stage: Stage::TTMove,
            tt_move,
            killers,
            counter_move,
            skip_quiets: false,
            see_threshold,
        }
    }

    /// Returns true if a move was already yielded by the movepicker.
    pub fn was_tried_lazily(&self, m: Move) -> bool {
        m == self.tt_move || m == self.killers[0] || m == self.killers[1] || m == self.counter_move
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
            Self::score_captures(t, position, self.movelist.as_slice_mut(), self.see_threshold);
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
            self.stage = if QSEARCH { Stage::Done } else { Stage::YieldKiller1 };
        }
        if self.stage == Stage::YieldKiller1 {
            self.stage = Stage::YieldKiller2;
            if !self.skip_quiets
                && self.killers[0] != self.tt_move
                && position.is_pseudo_legal(self.killers[0])
            {
                return Some(MoveListEntry { mov: self.killers[0], score: FIRST_KILLER_SCORE });
            }
        }
        if self.stage == Stage::YieldKiller2 {
            self.stage = Stage::YieldCounterMove;
            if !self.skip_quiets
                && self.killers[1] != self.tt_move
                && position.is_pseudo_legal(self.killers[1])
            {
                return Some(MoveListEntry { mov: self.killers[1], score: SECOND_KILLER_SCORE });
            }
        }
        if self.stage == Stage::YieldCounterMove {
            self.stage = Stage::GenerateQuiets;
            if !self.skip_quiets
                && self.counter_move != self.tt_move
                && self.counter_move != self.killers[0]
                && self.counter_move != self.killers[1]
                && position.is_pseudo_legal(self.counter_move)
            {
                return Some(MoveListEntry { mov: self.counter_move, score: COUNTER_MOVE_SCORE });
            }
        }
        if self.stage == Stage::GenerateQuiets {
            self.stage = Stage::YieldRemaining;
            if !self.skip_quiets {
                let start = self.movelist.count;
                position.generate_quiets(&mut self.movelist);
                let quiets = &mut self.movelist.moves[start..self.movelist.count];
                Self::score_quiets(t, position, quiets);
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

        // as the scores of positive-SEE moves can be pushed below
        // WINNING_CAPTURE_SCORE if their capture history is particularly
        // bad, this implicitly filters out moves with bad history scores.
        let not_winning = m.score < WINNING_CAPTURE_SCORE;

        if self.skip_quiets && not_winning {
            // the best we could find wasn't winning,
            // and we're skipping quiet moves, so we're done.
            return None;
        }
        if self.was_tried_lazily(m.mov) {
            self.yield_once()
        } else {
            Some(m)
        }
    }

    pub fn score_quiets(t: &ThreadData, pos: &Board, ms: &mut [MoveListEntry]) {
        const NEAR_PROMOTION_BONUSES: [i32; 8] = [0, 0, 0, 0, 500, 2000, 8000, 0];
        // zero-out the ordering scores
        for m in &mut *ms {
            m.score = 0;
        }

        t.get_history_scores(pos, ms);
        if pos.height > 0 {
            ThreadData::get_continuation_history_scores(
                pos,
                ms,
                t.conthist_indices[pos.height - 1],
                &t.counter_move_history,
            );
        }
        if pos.height > 1 {
            ThreadData::get_continuation_history_scores(
                pos,
                ms,
                t.conthist_indices[pos.height - 2],
                &t.followup_history,
            );
        }

        let bb = &pos.pieces;
        let th = &pos.threats;
        let our_pieces = bb.occupied_co(pos.turn());
        let them = bb.occupied_co(pos.turn().flip());

        // add bonuses / maluses for position-specific features
        // this stuff is pretty direct from caissa
        for MoveListEntry { mov, score } in ms {
            let from_sq = mov.from();
            let to_sq = mov.to();
            let moved_piece = pos.piece_at(from_sq);
            match moved_piece.piece_type() {
                PieceType::PAWN => {
                    // add bonus for threatening promotions
                    let relative_tgt = mov.to().relative_to(pos.turn());
                    *score += NEAR_PROMOTION_BONUSES[relative_tgt.rank() as usize];
                    // check if pushed pawn is protected by other pawn
                    if (bitboards::pawn_attacks_runtime(mov.to().as_set(), pos.turn().flip())
                        & our_pieces)
                        .non_empty()
                    {
                        // bonus for creating threats
                        let pawn_attacks =
                            bitboards::pawn_attacks_runtime(mov.to().as_set(), pos.turn()) & them;
                        match () {
                            () if (pawn_attacks & bb.all_kings()).non_empty() => *score += 10000,
                            () if (pawn_attacks & bb.all_pawns()).non_empty() => *score += 1000,
                            () if (pawn_attacks & bb.all_queens()).non_empty() => *score += 8000,
                            () if (pawn_attacks & bb.all_rooks()).non_empty() => *score += 6000,
                            () if (pawn_attacks & bb.all_bishops()).non_empty() => *score += 4000,
                            () if (pawn_attacks & bb.all_knights()).non_empty() => *score += 4000,
                            () => {}
                        }
                    }
                }
                PieceType::KNIGHT | PieceType::BISHOP => {
                    if th.pawn.contains_square(from_sq) {
                        *score += 4000;
                    }
                    if th.pawn.contains_square(to_sq) {
                        *score -= 4000;
                    }
                }
                PieceType::ROOK => {
                    if (th.pawn | th.minor).contains_square(from_sq) {
                        *score += 8000;
                    }
                    if (th.pawn | th.minor).contains_square(to_sq) {
                        *score -= 8000;
                    }
                }
                PieceType::QUEEN => {
                    if (th.pawn | th.minor | th.rook).contains_square(from_sq) {
                        *score += 12000;
                    }
                    if (th.pawn | th.minor | th.rook).contains_square(to_sq) {
                        *score -= 12000;
                    }
                }
                _ => {}
            }
        }
    }

    pub fn score_captures(
        t: &ThreadData,
        pos: &Board,
        moves: &mut [MoveListEntry],
        see_threshold: i32,
    ) {
        const MVV_SCORE: [i32; 5] = [0, 2400, 2400, 4800, 9600];
        // zero-out the ordering scores
        for m in &mut *moves {
            m.score = 0;
        }

        t.get_tactical_history_scores(pos, moves);
        for MoveListEntry { mov, score } in moves {
            *score += MVV_SCORE[history::caphist_piece_type(pos, *mov).index()];
            if pos.static_exchange_eval(*mov, see_threshold) {
                *score += WINNING_CAPTURE_SCORE;
            }
        }
    }
}
