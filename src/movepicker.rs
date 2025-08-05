use std::cell::Cell;

use crate::{
    chess::{
        board::{
            movegen::{pawn_attacks_by, AllMoves, MoveList, MoveListEntry, SkipQuiets},
            Board,
        },
        chessmove::Move,
        piece::PieceType,
        squareset::SquareSet,
    },
    history,
    historytable::MAX_HISTORY,
    search::static_exchange_eval,
    searchinfo::SearchInfo,
    threadlocal::ThreadData,
};

pub const WINNING_CAPTURE_BONUS: i32 = 10_000_000;
pub const MIN_WINNING_SEE_SCORE: i32 = WINNING_CAPTURE_BONUS - MAX_HISTORY as i32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stage {
    TTMove,
    GenerateCaptures,
    YieldGoodCaptures,
    YieldKiller,
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
    pub skip_quiets: bool,
    see_threshold: i32,
}

impl MovePicker {
    pub fn new(tt_move: Option<Move>, killer: Option<Move>, see_threshold: i32) -> Self {
        Self {
            movelist: MoveList::new(),
            index: 0,
            stage: Stage::TTMove,
            tt_move,
            killer,
            skip_quiets: false,
            see_threshold,
        }
    }

    /// Select the next move to try. Returns None if there are no more moves to try.
    #[allow(clippy::cognitive_complexity)]
    pub fn next(&mut self, position: &Board, t: &ThreadData, info: &SearchInfo) -> Option<Move> {
        if self.stage == Stage::Done {
            return None;
        }
        if self.stage == Stage::TTMove {
            self.stage = Stage::GenerateCaptures;
            if let Some(tt_move) = self.tt_move {
                if position.is_pseudo_legal(tt_move) {
                    return Some(tt_move);
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
                if m.score >= WINNING_CAPTURE_BONUS {
                    return Some(m.mov);
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
            self.stage = Stage::GenerateQuiets;
            if !self.skip_quiets && self.killer != self.tt_move {
                if let Some(killer) = self.killer {
                    if position.is_pseudo_legal(killer) {
                        debug_assert!(!position.is_tactical(killer));
                        return Some(killer);
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
                return Some(m.mov);
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
                best.score < WINNING_CAPTURE_BONUS / 2 || best.score >= MIN_WINNING_SEE_SCORE,
                "{}'s score is {}, lower bound is {}, this is too close.",
                best.mov.display(false),
                best.score,
                MIN_WINNING_SEE_SCORE
            );
            // test if this is a potentially-winning capture that's yet to be SEE-ed:
            if best.score >= MIN_WINNING_SEE_SCORE
                && !static_exchange_eval(pos, info, best.mov, self.see_threshold)
            {
                // if it fails SEE, then we want to try the next best move, and de-mark this one.
                best_entry_ref.set(MoveListEntry {
                    score: best.score - WINNING_CAPTURE_BONUS,
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
            if !(Some(best.mov) == self.tt_move || Some(best.mov) == self.killer) {
                return Some(best);
            }
        }

        // If we have already tried all moves, return None.
        None
    }

    pub fn score_quiets(t: &ThreadData, pos: &Board, ms: &mut [MoveListEntry]) {
        let cont_block_0 = pos
            .height()
            .checked_sub(1)
            .and_then(|i| t.ss.get(i))
            .map(|ss| t.continuation_history.get_index(ss.conthist_index));
        let cont_block_1 = pos
            .height()
            .checked_sub(2)
            .and_then(|i| t.ss.get(i))
            .map(|ss| t.continuation_history.get_index(ss.conthist_index));

        let threats = pos.state.threats.all;
        for m in ms {
            let from = m.mov.from();
            let piece = pos.state.mailbox[from].unwrap();
            let to = m.mov.history_to_square();
            let from_threat = usize::from(threats.contains_square(from));
            let to_threat = usize::from(threats.contains_square(to));

            let mut score = 0;

            score += i32::from(t.main_history[from_threat][to_threat][piece][to]);
            if let Some(cmh_block) = cont_block_0 {
                score += i32::from(cmh_block[piece][to]);
            }
            if let Some(cmh_block) = cont_block_1 {
                score += i32::from(cmh_block[piece][to]);
            }

            match piece.piece_type() {
                PieceType::Pawn => {
                    let turn = pos.turn();
                    let us = pos.state.bbs.colours[turn];
                    let them = pos.state.bbs.colours[!turn];

                    let our_pawns = pos.state.bbs.pieces[PieceType::Pawn] & us;
                    let their_king = pos.state.bbs.pieces[PieceType::King] & them;
                    let their_queens = pos.state.bbs.pieces[PieceType::Queen] & them;
                    let their_rooks = pos.state.bbs.pieces[PieceType::Rook] & them;
                    let their_minors = (pos.state.bbs.pieces[PieceType::Bishop]
                        | pos.state.bbs.pieces[PieceType::Knight])
                        & them;
                    let their_pawns = pos.state.bbs.pieces[PieceType::Pawn] & them;

                    if pawn_attacks_by(to.as_set(), !turn) & our_pawns != SquareSet::EMPTY {
                        // bonus for creating threats
                        let pawn_attacks = pawn_attacks_by(to.as_set(), turn);
                        if pawn_attacks & their_king != SquareSet::EMPTY {
                            score += 10_000;
                        } else if pawn_attacks & their_queens != SquareSet::EMPTY {
                            score += 8_000;
                        } else if pawn_attacks & their_rooks != SquareSet::EMPTY {
                            score += 6_000;
                        } else if pawn_attacks & their_minors != SquareSet::EMPTY {
                            score += 4_000;
                        } else if pawn_attacks & their_pawns != SquareSet::EMPTY {
                            score += 1_000;
                        }
                    }
                }
                PieceType::Knight | PieceType::Bishop => {
                    if pos.state.threats.leq_pawn.contains_square(from) {
                        score += 4000;
                    }
                    if pos.state.threats.leq_pawn.contains_square(to) {
                        score -= 4000;
                    }
                }
                PieceType::Rook => {
                    if pos.state.threats.leq_minor.contains_square(from) {
                        score += 8000;
                    }
                    if pos.state.threats.leq_minor.contains_square(to) {
                        score -= 8000;
                    }
                }
                PieceType::Queen => {
                    if pos.state.threats.leq_rook.contains_square(from) {
                        score += 12000;
                    }
                    if pos.state.threats.leq_rook.contains_square(to) {
                        score -= 12000;
                    }
                }
                PieceType::King => {}
            }

            m.score = score;
        }
    }

    pub fn score_captures(t: &ThreadData, pos: &Board, moves: &mut [MoveListEntry]) {
        const MVV_SCORE: [i32; 6] = [0, 2400, 2400, 4800, 9600, 0];

        for m in moves {
            let from = m.mov.from();
            let to = m.mov.to();
            let piece = pos.state.mailbox[from].unwrap();
            let capture = history::caphist_piece_type(pos, m.mov);

            // optimistically initialised with the winning-SEE score.
            // lazily checked during yield_once.
            let mut score = WINNING_CAPTURE_BONUS;

            score += MVV_SCORE[capture];
            score += i32::from(t.tactical_history[capture][piece][to]);

            m.score = score;
        }
    }
}
