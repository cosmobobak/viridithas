use std::cell::Cell;

use crate::{
    chess::{
        board::movegen::{pawn_attacks_by, AllMoves, MoveList, MoveListEntry, SkipQuiets},
        chessmove::Move,
        piece::PieceType,
        squareset::SquareSet,
    },
    history,
    historytable::MAX_HISTORY,
    search::static_exchange_eval,
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
    pub fn next(&mut self, t: &ThreadData) -> Option<Move> {
        if self.stage == Stage::Done {
            return None;
        }
        if self.stage == Stage::TTMove {
            self.stage = Stage::GenerateCaptures;
            if let Some(tt_move) = self.tt_move {
                if t.board.is_pseudo_legal(tt_move) {
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
                t.board.generate_captures::<SkipQuiets>(&mut self.movelist);
            } else {
                t.board.generate_captures::<AllMoves>(&mut self.movelist);
            }
            Self::score_captures(t, &mut self.movelist);
        }
        if self.stage == Stage::YieldGoodCaptures {
            if let Some(m) = self.yield_once(t) {
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
                    if t.board.is_pseudo_legal(killer) {
                        debug_assert!(!t.board.is_tactical(killer));
                        return Some(killer);
                    }
                }
            }
        }
        if self.stage == Stage::GenerateQuiets {
            self.stage = Stage::YieldRemaining;
            if !self.skip_quiets {
                let start = self.movelist.len();
                t.board.generate_quiets(&mut self.movelist);
                let quiets = &mut self.movelist[start..];
                Self::score_quiets(t, quiets);
            }
        }
        if self.stage == Stage::YieldRemaining {
            if let Some(m) = self.yield_once(t) {
                return Some(m.mov);
            }
            self.stage = Stage::Done;
        }
        None
    }

    #[inline(never)]
    fn fast_select(entries: &[Cell<MoveListEntry>]) -> Option<&Cell<MoveListEntry>> {
        #![allow(clippy::cast_possible_truncation)]
        fn to_u64(e: MoveListEntry) -> u64 {
            #![allow(clippy::cast_sign_loss)]
            let widened = i64::from(e.score);
            let offset = widened - i64::from(i32::MIN);
            (offset as u64) << 32
        }
        let best = entries.first()?.get();
        let mut best = to_u64(best) | 256;
        for i in 1..entries.len() {
            let curr = entries[i].get();
            let curr = to_u64(curr) | (256 - i as u64);
            best = std::cmp::max(best, curr);
        }
        let best_idx = 256 - (best & 0xFFFF_FFFF);
        let best_idx = best_idx as usize;
        // SAFETY: best_idx is guaranteed to be in-bounds.
        unsafe { Some(entries.get_unchecked(best_idx)) }
    }

    /// Perform iterations of partial insertion sort.
    /// Extracts the best move from the unsorted portion of the movelist,
    /// or returns None if there are no more moves to try.
    ///
    /// Usually only one iteration is performed, but in the case where
    /// the best move has already been tried or doesn't meet SEE requirements,
    /// we will continue to iterate until we find a move that is valid.
    fn yield_once(&mut self, t: &ThreadData) -> Option<MoveListEntry> {
        let mut remaining =
            Cell::as_slice_of_cells(Cell::from_mut(&mut self.movelist[self.index..]));
        while let Some(best_entry_ref) = Self::fast_select(remaining) {
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
                && !static_exchange_eval(&t.board, &t.info, best.mov, self.see_threshold)
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

    pub fn score_quiets(t: &ThreadData, ms: &mut [MoveListEntry]) {
        let cont_block_0 = t
            .board
            .height()
            .checked_sub(1)
            .and_then(|i| t.ss.get(i))
            .map(|ss| t.continuation_history.get_index(ss.conthist_index));
        let cont_block_1 = t
            .board
            .height()
            .checked_sub(2)
            .and_then(|i| t.ss.get(i))
            .map(|ss| t.continuation_history.get_index(ss.conthist_index));

        let threats = t.board.state.threats.all;
        for m in ms {
            let from = m.mov.from();
            let piece = t.board.state.mailbox[from].unwrap();
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
                    let turn = t.board.turn();
                    let us = t.board.state.bbs.colours[turn];
                    let them = t.board.state.bbs.colours[!turn];

                    let our_pawns = t.board.state.bbs.pieces[PieceType::Pawn] & us;
                    let their_king = t.board.state.bbs.pieces[PieceType::King] & them;
                    let their_queens = t.board.state.bbs.pieces[PieceType::Queen] & them;
                    let their_rooks = t.board.state.bbs.pieces[PieceType::Rook] & them;
                    let their_minors = (t.board.state.bbs.pieces[PieceType::Bishop]
                        | t.board.state.bbs.pieces[PieceType::Knight])
                        & them;
                    let their_pawns = t.board.state.bbs.pieces[PieceType::Pawn] & them;

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
                    if t.board.state.threats.leq_pawn.contains_square(from) {
                        score += 4000;
                    }
                    if t.board.state.threats.leq_pawn.contains_square(to) {
                        score -= 4000;
                    }
                }
                PieceType::Rook => {
                    if t.board.state.threats.leq_minor.contains_square(from) {
                        score += 8000;
                    }
                    if t.board.state.threats.leq_minor.contains_square(to) {
                        score -= 8000;
                    }
                }
                PieceType::Queen => {
                    if t.board.state.threats.leq_rook.contains_square(from) {
                        score += 12000;
                    }
                    if t.board.state.threats.leq_rook.contains_square(to) {
                        score -= 12000;
                    }
                }
                PieceType::King => {}
            }

            m.score = score;
        }
    }

    pub fn score_captures(t: &ThreadData, moves: &mut [MoveListEntry]) {
        const MVV_SCORE: [i32; 6] = [0, 2400, 2400, 4800, 9600, 0];

        let threats = t.board.state.threats.all;
        for m in moves {
            let from = m.mov.from();
            let to = m.mov.to();
            let threat_to = threats.contains_square(to);
            let piece = t.board.state.mailbox[from].unwrap();
            let capture = history::caphist_piece_type(&t.board, m.mov);

            // optimistically initialised with the winning-SEE score.
            // lazily checked during yield_once.
            let mut score = WINNING_CAPTURE_BONUS;

            score += MVV_SCORE[capture];
            score += i32::from(t.tactical_history[usize::from(threat_to)][capture][piece][to]);

            m.score = score;
        }
    }
}
