// SPDX-License-Identifier: AGPL-3.0-only

use std::cell::Cell;

use crate::{
    chess::{
        board::{
            Board, Rules,
            movegen::{AllMoves, MoveList, MoveListEntry, SkipQuiets, pawn_attacks_by},
        },
        chessmove::Move,
        piece::PieceType,
        squareset::SquareSet,
    },
    history,
    historytable::{HASH_HISTORY_SIZE, MAX_HISTORY},
    search::{parameters::Config, static_exchange_eval},
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
    index: usize,
    pub stage: Stage,
    tt_move: Option<Move>,
    killer: Option<Move>,
    pub skip_quiets: bool,
    see_threshold: i32,
}

#[cfg(target_feature = "avx512f")]
fn fast_select(entries: &[Cell<MoveListEntry>]) -> Option<&Cell<MoveListEntry>> {
    use crate::nnue::simd;

    if entries.is_empty() {
        return None;
    }

    // SAFETY: This is largely an explicit rewrite of the intended
    // vectorisation of the old version of this function, which can
    // be found here:
    // https://github.com/cosmobobak/viridithas/blob/e605537c8a0ffe78e250d9419924ff6b23e98bad/src/movepicker.rs#L48
    //
    // We do load OOB, but mask.
    let best = unsafe {
        // base ptr
        #[expect(clippy::cast_ptr_alignment, reason = "unaligned loads are fine")]
        let base = entries.as_ptr().cast::<u64>();
        let sign = simd::splat_u64(1 << 63);
        let step = simd::splat_u64(simd::U64_CHUNK as u64);
        // indexes for mixing in to scores.
        // starts at [0, 1, 2, 3, …]
        let mut index = simd::iota_u64();
        // accumulator for maximal elements
        let mut best = simd::zero_u64();
        let mut i = 0;
        while i < entries.len() {
            // mask off a potential OOB tail:
            let remaining = entries.len() - i;
            let mask = if remaining >= simd::U64_CHUNK {
                u8::MAX
            } else {
                (1 << remaining) - 1
            };
            // load elements
            let loaded = simd::maskz_loadu_u64(mask, base.add(i));
            // we want this:
            // > key = (score - i32::MIN) << 32 | index
            // subtracting i32::MIN shifts the i32 range up
            // into the u32 range, which makes it safe to use
            // as the top bits of our comparison target.
            // N.B. we XOR by 1 << 63 instead of subtracting
            // i32::MIN, which does the same thing.
            let keys = simd::or_u64(simd::xor_u64(simd::shl_u64::<32>(loaded), sign), index);
            // max the valid lanes
            best = simd::mask_max_u64(best, mask, best, keys);
            // increments
            index = simd::add_u64(index, step);
            i += simd::U64_CHUNK;
        }
        simd::reduce_max_u64(best)
    };

    #[expect(clippy::cast_possible_truncation)]
    let best_idx = (best & 0xFFFF_FFFF) as usize;
    // SAFETY: by construction, the low bits are a valid index.
    unsafe { Some(entries.get_unchecked(best_idx)) }
}

#[cfg(not(target_feature = "avx512f"))]
fn fast_select(entries: &[Cell<MoveListEntry>]) -> Option<&Cell<MoveListEntry>> {
    let (first, rest) = entries.split_first()?;
    let mut best = first;
    let mut best_score = first.get().score;
    for entry in rest {
        let score = entry.get().score;
        if score >= best_score {
            best_score = score;
            best = entry;
        }
    }
    Some(best)
}

impl MovePicker {
    pub fn new(tt_move: Option<Move>, killer: Option<Move>, see_threshold: i32) -> Self {
        Self {
            moves: MoveList::new(),
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
            Self::score_captures(&t.board, &t.histories, &mut self.moves);
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
                let quiets = &mut self.moves[start..];
                Self::score_quiets(&t.board, &t.info.conf, &t.histories, &t.ss, quiets);
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

    /// Perform iterations of partial insertion sort.
    /// Extracts the best move from the unsorted portion of the movelist,
    /// or returns None if there are no more moves to try.
    ///
    /// Usually only one iteration is performed, but in the case where
    /// the best move has already been tried or doesn't meet SEE requirements,
    /// we will continue to iterate until we find a move that is valid.
    fn yield_once(&mut self, t: &ThreadData) -> Option<MoveListEntry> {
        let remaining = &mut self.moves[self.index..];
        let mut remaining = Cell::as_slice_of_cells(Cell::from_mut(remaining));
        while let Some(best_entry_ref) = fast_select(remaining) {
            let best = best_entry_ref.get();
            debug_assert!(
                best.score < WINNING_CAPTURE_BONUS / 2 || best.score >= MIN_WINNING_SEE_SCORE,
                "{}'s score is {}, lower bound is {}, this is too close.",
                best.mov.display(Rules::Classical),
                best.score,
                MIN_WINNING_SEE_SCORE
            );
            // test if this is a potentially-winning capture that's yet to be SEE-ed:
            if best.score >= MIN_WINNING_SEE_SCORE
                && !static_exchange_eval(&t.board, &t.info.conf, best.mov, self.see_threshold)
            {
                // if it fails SEE, then we want to try the next best move, and de-mark this one.
                best_entry_ref.set(MoveListEntry::new(
                    best.mov,
                    best.score - WINNING_CAPTURE_BONUS,
                ));
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

    pub fn score_quiets(
        board: &Board,
        conf: &Config,
        histories: &Histories,
        ss: &[StackFrame; MAX_DEPTH + 1],
        ms: &mut [MoveListEntry],
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
            let from = m.mov.from();
            let piece = board.state.mailbox[from].unwrap();
            let to = m.mov.history_to_square();
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

            if board.gives_check(m.mov)
                && static_exchange_eval(board, conf, m.mov, -conf.quiet_check_see_margin)
            {
                score += 10_000;
            }

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

            m.score = score;
        }
    }

    pub fn score_captures(board: &Board, histories: &Histories, moves: &mut [MoveListEntry]) {
        const MVV_SCORE: [i32; 6] = [0, 2400, 2400, 4800, 9600, 0];

        let threats = board.state.threats.all;
        for m in moves {
            let from = m.mov.from();
            let to = m.mov.to();
            let threat_to = threats.contains_square(to);
            let piece = board.state.mailbox[from].unwrap();
            let capture = history::caphist_piece_type(board, m.mov);

            // optimistically initialised with the winning-SEE score.
            // lazily checked during yield_once.
            let mut score = WINNING_CAPTURE_BONUS;

            score += MVV_SCORE[capture];
            score += i32::from(histories.tactical[usize::from(threat_to)][capture][piece][to]);

            m.score = score;
        }
    }
}
