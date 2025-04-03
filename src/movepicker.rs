use std::cell::Cell;

use crate::{
    chess::{
        board::{
            movegen::{AllMoves, MoveList, MoveListEntry, SkipQuiets},
            Board,
        },
        chessmove::Move,
    },
    history,
    historytable::MAX_HISTORY,
    searchinfo::SearchInfo,
    threadlocal::ThreadData,
};

pub const TT_MOVE_SCORE: i32 = 20_000_000;
pub const KILLER_SCORE: i32 = 9_000_000;
pub const COUNTER_MOVE_SCORE: i32 = 2_000_000;
pub const WINNING_CAPTURE_SCORE: i32 = 10_000_000;
pub const MIN_WINNING_SEE_SCORE: i32 = WINNING_CAPTURE_SCORE - MAX_HISTORY as i32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stage {
    TTMove,
    GenerateCaptures,
    YieldGoodCaptures,
    YieldKiller,
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
    killer: Option<Move>,
    counter_move: Option<Move>,
    pub skip_quiets: bool,
    see_threshold: i32,
}

impl MovePicker {
    pub fn new(
        tt_move: Option<Move>,
        killer: Option<Move>,
        counter_move: Option<Move>,
        see_threshold: i32,
    ) -> Self {
        Self {
            movelist: MoveList::new(),
            index: 0,
            stage: Stage::TTMove,
            tt_move,
            killer,
            counter_move,
            skip_quiets: false,
            see_threshold,
        }
    }

    /// Select the next move to try. Returns None if there are no more moves to try.
    #[allow(clippy::cognitive_complexity)]
    pub fn next(
        &mut self,
        position: &Board,
        t: &ThreadData,
        info: &SearchInfo,
    ) -> Option<MoveListEntry> {
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
                position.generate_captures::<SkipQuiets>(&mut self.movelist);
            } else {
                position.generate_captures::<AllMoves>(&mut self.movelist);
            }
            Self::score_captures(t, position, &mut self.movelist);
        }
        if self.stage == Stage::YieldGoodCaptures {
            if let Some(m) = self.yield_once(info, position) {
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
                Stage::YieldKiller
            };
        }
        if self.stage == Stage::YieldKiller {
            self.stage = Stage::YieldCounterMove;
            if !self.skip_quiets && self.killer != self.tt_move {
                if let Some(killer) = self.killer {
                    if position.is_pseudo_legal(killer) {
                        debug_assert!(!position.is_tactical(killer));
                        return Some(MoveListEntry {
                            mov: killer,
                            score: KILLER_SCORE,
                        });
                    }
                }
            }
        }
        if self.stage == Stage::YieldCounterMove {
            self.stage = Stage::GenerateQuiets;
            if !self.skip_quiets
                && self.counter_move != self.tt_move
                && self.counter_move != self.killer
            {
                if let Some(counter) = self.counter_move {
                    if position.is_pseudo_legal(counter) {
                        debug_assert!(!position.is_tactical(counter));
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
            if let Some(m) = self.yield_once(info, position) {
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
    fn yield_once(&mut self, info: &SearchInfo, pos: &Board) -> Option<MoveListEntry> {
        let mut remaining =
            Cell::as_slice_of_cells(Cell::from_mut(&mut self.movelist[self.index..]));
        while let Some(best_entry_ref) =
            remaining
                .iter()
                .reduce(|a, b| if a.get().score >= b.get().score { a } else { b })
        {
            let best = best_entry_ref.get();
            debug_assert!(
                best.score < WINNING_CAPTURE_SCORE / 2 || best.score >= MIN_WINNING_SEE_SCORE,
                "{}'s score is {}, lower bound is {}, this is too close.",
                best.mov.display(false),
                best.score,
                MIN_WINNING_SEE_SCORE
            );
            // test if this is a potentially-winning capture that's yet to be SEE-ed:
            if best.score >= MIN_WINNING_SEE_SCORE
                && !pos.static_exchange_eval(info, best.mov, self.see_threshold)
            {
                // if it fails SEE, then we want to try the next best move, and de-mark this one.
                best_entry_ref.set(MoveListEntry {
                    score: best.score - WINNING_CAPTURE_SCORE,
                    mov: best.mov,
                });
                continue;
            }

            // swap the best move with the first unsorted move.
            best_entry_ref.set(remaining[0].get());
            remaining[0].set(best);
            remaining = &remaining[1..];

            self.index += 1;

            if self.skip_quiets && best.score < MIN_WINNING_SEE_SCORE {
                // the best we could find wasn't winning,
                // and we're skipping quiet moves, so we're done.
                return None;
            }
            if !(Some(best.mov) == self.tt_move
                || Some(best.mov) == self.killer
                || Some(best.mov) == self.counter_move)
            {
                return Some(best);
            }
        }

        // If we have already tried all moves, return None.
        None
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
