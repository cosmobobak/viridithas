use std::cell::Cell;

use arrayvec::ArrayVec;

use crate::{
    chess::{
        board::{
            Board, Rules,
            movegen::{AllMoves, MAX_POSITION_MOVES, MoveList, SkipQuiets, pawn_attacks_by},
        },
        chessmove::Move,
        piece::PieceType,
        squareset::SquareSet,
    },
    history,
    historytable::{HASH_HISTORY_SIZE, MAX_HISTORY},
    search::static_exchange_eval,
    stack::StackFrame,
    threadlocal::{Histories, ThreadData},
    util::MAX_DEPTH,
};

pub const WINNING_CAPTURE_BONUS: i32 = 10_000_000;
pub const MIN_WINNING_SEE_SCORE: i32 = WINNING_CAPTURE_BONUS - MAX_HISTORY;

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

#[derive(Debug)]
pub struct MovePicker {
    moves: MoveList,
    scores: ArrayVec<i32, MAX_POSITION_MOVES>,
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
            moves: MoveList::new(),
            scores: ArrayVec::new(),
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
                return Some(tt_move);
            }
        }
        if self.stage == Stage::GenerateCaptures {
            self.stage = Stage::YieldGoodCaptures;
            debug_assert_eq!(
                self.moves.len(),
                0,
                "movelist not empty before capture generation"
            );
            // when we're in check, we want to generate enough moves to prove we're not mated.
            if self.skip_quiets {
                t.board.generate_captures::<SkipQuiets>(&mut self.moves);
            } else {
                t.board.generate_captures::<AllMoves>(&mut self.moves);
            }
            Self::score_captures(&t.board, &t.histories, &self.moves, &mut self.scores);
        }
        if self.stage == Stage::YieldGoodCaptures {
            if let Some((m, s)) = self.yield_once(t) {
                if s >= WINNING_CAPTURE_BONUS {
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
            self.stage = Stage::GenerateQuiets;
            if !self.skip_quiets
                && self.killer != self.tt_move
                && let Some(killer) = self.killer
                && t.board.is_pseudo_legal(killer)
            {
                debug_assert!(!t.board.is_tactical(killer));
                return Some(killer);
            }
        }
        if self.stage == Stage::GenerateQuiets {
            self.stage = Stage::YieldRemaining;
            if !self.skip_quiets {
                let start = self.moves.len();
                t.board.generate_quiets(&mut self.moves);
                let quiets = &self.moves[start..];
                Self::score_quiets(&t.board, &t.histories, &t.ss, quiets, &mut self.scores);
            }
        }
        if self.stage == Stage::YieldRemaining {
            if let Some((m, _)) = self.yield_once(t) {
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
    fn yield_once(&mut self, t: &ThreadData) -> Option<(Move, i32)> {
        let r_moves = &mut self.moves[self.index..];
        let r_scores = &mut self.scores[self.index..];
        let mut r_moves = Cell::as_slice_of_cells(Cell::from_mut(r_moves));
        let mut r_scores = Cell::as_slice_of_cells(Cell::from_mut(r_scores));
        while let Some((best_mv_entry, best_score_entry)) =
            r_moves.iter().zip(r_scores).max_by_key(|(_, s)| s.get())
        {
            let best_mv = best_mv_entry.get();
            let best_score = best_score_entry.get();
            debug_assert!(
                !(WINNING_CAPTURE_BONUS / 2..MIN_WINNING_SEE_SCORE).contains(&best_score),
                "{}'s score is {}, lower bound is {}, this is too close.",
                best_mv.display(Rules::Classical),
                best_score,
                MIN_WINNING_SEE_SCORE
            );
            // test if this is a potentially-winning capture that's yet to be SEE-ed:
            if best_score >= MIN_WINNING_SEE_SCORE
                && !static_exchange_eval(&t.board, &t.info.conf, best_mv, self.see_threshold)
            {
                // if it fails SEE, then we want to try the next best move, and de-mark this one.
                best_score_entry.set(best_score - WINNING_CAPTURE_BONUS);
                continue;
            }

            // swap the best move with the first unsorted move.
            best_mv_entry.set(r_moves[0].get());
            best_score_entry.set(r_scores[0].get());
            r_moves[0].set(best_mv);
            r_scores[0].set(best_score);
            r_moves = &r_moves[1..];
            r_scores = &r_scores[1..];

            self.index += 1;

            if self.skip_quiets && best_score < MIN_WINNING_SEE_SCORE {
                // the best we could find wasn't winning,
                // and we're skipping quiet moves, so we're done.
                return None;
            }
            if !(Some(best_mv) == self.tt_move || Some(best_mv) == self.killer) {
                return Some((best_mv, best_score));
            }
        }

        // If we have already tried all moves, return None.
        None
    }

    pub fn score_quiets(
        board: &Board,
        histories: &Histories,
        ss: &[StackFrame; MAX_DEPTH + 1],
        ms: &[Move],
        scores: &mut ArrayVec<i32, MAX_POSITION_MOVES>,
    ) {
        let height = board.height();

        let cont_blocks =
            [1, 2].map(|i| (height > i).then(|| &histories.continuation[ss[height - i].ch_idx]));

        let threats = board.state.threats.all;
        #[expect(clippy::cast_possible_truncation)]
        let pawn_index = (board.state.keys.pawn % HASH_HISTORY_SIZE as u64) as usize;

        let turn = board.turn();
        let us = board.state.bbs.colours[turn];
        let them = board.state.bbs.colours[!turn];
        let our_pawns = board.state.bbs.pieces[PieceType::Pawn] & us;
        let their_king = board.state.bbs.pieces[PieceType::King] & them;
        let their_queens = board.state.bbs.pieces[PieceType::Queen] & them;
        let their_rooks = board.state.bbs.pieces[PieceType::Rook] & them;
        let their_minors = (board.state.bbs.pieces[PieceType::Bishop]
            | board.state.bbs.pieces[PieceType::Knight])
            & them;
        let their_pawns = board.state.bbs.pieces[PieceType::Pawn] & them;

        for m in ms {
            let from = m.from();
            let piece = board.state.mailbox[from].unwrap();
            let to = m.history_to_square();
            let from_threat = usize::from(threats.contains_square(from));
            let to_threat = usize::from(threats.contains_square(to));

            let mut score = 0;

            score += i32::midpoint(
                i32::from(histories.piece_to[from_threat][to_threat][piece][to]),
                i32::from(histories.from_to[from_threat][to_threat][from][to]),
            );
            for block in cont_blocks {
                score += block.map_or(0, |b| i32::from(b[piece][to]));
            }
            score += i32::from(histories.pawn[pawn_index][piece][to]);

            score += 10_000 * i32::from(board.gives_check(*m));

            match piece.piece_type() {
                PieceType::Pawn => {
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
                    if board.state.threats.leq_pawn.contains_square(from) {
                        score += 4000;
                    }
                    if board.state.threats.leq_pawn.contains_square(to) {
                        score -= 4000;
                    }
                }
                PieceType::Rook => {
                    if board.state.threats.leq_minor.contains_square(from) {
                        score += 8000;
                    }
                    if board.state.threats.leq_minor.contains_square(to) {
                        score -= 8000;
                    }
                }
                PieceType::Queen => {
                    if board.state.threats.leq_rook.contains_square(from) {
                        score += 12000;
                    }
                    if board.state.threats.leq_rook.contains_square(to) {
                        score -= 12000;
                    }
                }
                PieceType::King => {}
            }

            scores.push(score);
        }
    }

    pub fn score_captures(
        board: &Board,
        histories: &Histories,
        moves: &[Move],
        scores: &mut ArrayVec<i32, MAX_POSITION_MOVES>,
    ) {
        const MVV_SCORE: [i32; 6] = [0, 2400, 2400, 4800, 9600, 0];

        let threats = board.state.threats.all;
        for m in moves {
            let from = m.from();
            let to = m.to();
            let threat_to = threats.contains_square(to);
            let piece = board.state.mailbox[from].unwrap();
            let capture = history::caphist_piece_type(board, *m);

            // optimistically initialised with the winning-SEE score.
            // lazily checked during yield_once.
            let mut score = WINNING_CAPTURE_BONUS;

            score += MVV_SCORE[capture];
            score += i32::from(histories.tactical[usize::from(threat_to)][capture][piece][to]);

            scores.push(score);
        }
    }
}
