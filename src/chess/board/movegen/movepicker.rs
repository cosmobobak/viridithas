use crate::{
    chess::board::{history, Board},
    chess::chessmove::Move,
    historytable::MAX_HISTORY,
    threadlocal::ThreadData,
};

use super::{MoveList, MoveListEntry};

pub const TT_MOVE_SCORE: i32 = 20_000_000;
pub const FIRST_KILLER_SCORE: i32 = 9_000_000;
pub const SECOND_KILLER_SCORE: i32 = 8_000_000;
pub const COUNTER_MOVE_SCORE: i32 = 2_000_000;
pub const WINNING_CAPTURE_SCORE: i32 = 10_000_000;

pub trait MovePickerMode {
    const CAPTURES_ONLY: bool;
}

pub struct QSearch;
impl MovePickerMode for QSearch {
    const CAPTURES_ONLY: bool = true;
}
pub struct MainSearch;
impl MovePickerMode for MainSearch {
    const CAPTURES_ONLY: bool = false;
}

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

pub struct MovePicker {
    movelist: MoveList,
    index: usize,
    pub stage: Stage,
    tt_move: Option<Move>,
    killers: [Option<Move>; 2],
    counter_move: Option<Move>,
    pub skip_quiets: bool,
    see_threshold: i32,
}

impl MovePicker {
    pub fn new(
        tt_move: Option<Move>,
        killers: [Option<Move>; 2],
        counter_move: Option<Move>,
        see_threshold: i32,
    ) -> Self {
        debug_assert!(
            killers[0].is_none() || killers[0] != killers[1],
            "Killers are both {:?}",
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
        let m = Some(m);
        m == self.tt_move || m == self.killers[0] || m == self.killers[1] || m == self.counter_move
    }

    /// Select the next move to try. Returns None if there are no more moves to try.
    #[allow(clippy::cognitive_complexity)]
    pub fn next(&mut self, position: &Board, t: &ThreadData) -> Option<MoveListEntry> {
        if self.stage == Stage::Done {
            return None;
        }
        if self.stage == Stage::TTMove {
            self.stage = Stage::GenerateCaptures;
            if let Some(tt_move) = self.tt_move {
                if position.is_pseudo_legal(tt_move) {
                    return Some(MoveListEntry {
                        mov: tt_move,
                        score: TT_MOVE_SCORE,
                    });
                }
            }
        }
        if self.stage == Stage::GenerateCaptures {
            self.stage = Stage::YieldGoodCaptures;
            debug_assert_eq!(
                self.movelist.len(),
                0,
                "movelist not empty before capture generation"
            );
            // when we're in check, we want to generate enough moves to prove we're not mated.
            if self.skip_quiets {
                position.generate_captures::<QSearch>(&mut self.movelist);
            } else {
                position.generate_captures::<MainSearch>(&mut self.movelist);
            }
            Self::score_captures(t, position, &mut self.movelist);
        }
        if self.stage == Stage::YieldGoodCaptures {
            if let Some(m) = self.yield_once(position) {
                if m.score >= WINNING_CAPTURE_SCORE {
                    return Some(m);
                }
                // the move was not winning, so we're going to
                // generate quiet moves next. As such, we decrement
                // the index so we can try this move again.
                self.index -= 1;
            }
            self.stage = if self.skip_quiets {
                Stage::Done
            } else {
                Stage::YieldKiller1
            };
        }
        if self.stage == Stage::YieldKiller1 {
            self.stage = Stage::YieldKiller2;
            if !self.skip_quiets && self.killers[0] != self.tt_move {
                if let Some(killer) = self.killers[0] {
                    if position.is_pseudo_legal(killer) {
                        return Some(MoveListEntry {
                            mov: killer,
                            score: FIRST_KILLER_SCORE,
                        });
                    }
                }
            }
        }
        if self.stage == Stage::YieldKiller2 {
            self.stage = Stage::YieldCounterMove;
            if !self.skip_quiets && self.killers[1] != self.tt_move {
                if let Some(killer) = self.killers[1] {
                    if position.is_pseudo_legal(killer) {
                        return Some(MoveListEntry {
                            mov: killer,
                            score: SECOND_KILLER_SCORE,
                        });
                    }
                }
            }
        }
        if self.stage == Stage::YieldCounterMove {
            self.stage = Stage::GenerateQuiets;
            if !self.skip_quiets
                && self.counter_move != self.tt_move
                && self.counter_move != self.killers[0]
                && self.counter_move != self.killers[1]
            {
                if let Some(counter) = self.counter_move {
                    if position.is_pseudo_legal(counter) {
                        return Some(MoveListEntry {
                            mov: counter,
                            score: COUNTER_MOVE_SCORE,
                        });
                    }
                }
            }
        }
        if self.stage == Stage::GenerateQuiets {
            self.stage = Stage::YieldRemaining;
            if !self.skip_quiets {
                let start = self.movelist.len();
                position.generate_quiets(&mut self.movelist);
                let quiets = &mut self.movelist[start..];
                Self::score_quiets(t, position, quiets);
            }
        }
        if self.stage == Stage::YieldRemaining {
            if let Some(m) = self.yield_once(position) {
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
    fn yield_once(&mut self, pos: &Board) -> Option<MoveListEntry> {
        loop {
            // If we have already tried all moves, return None.
            if self.index == self.movelist.len() {
                return None;
            }

            let mut best_score = self.movelist[self.index].score;
            let mut best_num = self.index;

            // find the best move in the unsorted portion of the movelist.
            for index in self.index + 1..self.movelist.len() {
                let score = self.movelist[index].score;
                if score > best_score {
                    best_score = score;
                    best_num = index;
                }
            }

            let m = &mut self.movelist[best_num];

            // test if this is a potentially-winning capture that's yet to be SEE-ed:
            if m.score >= (WINNING_CAPTURE_SCORE - i32::from(MAX_HISTORY))
                && !pos.static_exchange_eval(m.mov, self.see_threshold)
            {
                // if it fails SEE, then we want to try the next best move, and de-mark this one.
                m.score -= WINNING_CAPTURE_SCORE;
                continue;
            }

            let m = *m;

            // swap the best move with the first unsorted move.
            self.movelist.swap(best_num, self.index);

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
            if !self.was_tried_lazily(m.mov) {
                return Some(m);
            }
        }
    }

    pub fn score_quiets(t: &ThreadData, pos: &Board, ms: &mut [MoveListEntry]) {
        // zero-out the ordering scores
        for m in &mut *ms {
            m.score = 0;
        }

        t.get_history_scores(pos, ms);
        t.get_continuation_history_scores(pos, ms, 0);
        t.get_continuation_history_scores(pos, ms, 1);
        // t.get_continuation_history_scores(pos, ms, 3);
    }

    pub fn score_captures(t: &ThreadData, pos: &Board, moves: &mut [MoveListEntry]) {
        const MVV_SCORE: [i32; 6] = [0, 2400, 2400, 4800, 9600, 0];

        // provisionally set the WINNING_CAPTURE offset, for lazily SEE-guarding stuff later.
        for m in &mut *moves {
            m.score = WINNING_CAPTURE_SCORE;
        }

        t.get_tactical_history_scores(pos, moves);
        for MoveListEntry { mov, score } in moves {
            *score += MVV_SCORE[history::caphist_piece_type(pos, *mov)];
        }
    }
}
