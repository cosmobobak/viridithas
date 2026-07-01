// SPDX-License-Identifier: GPL-3.0-only

use crate::{
    chess::{
        chessmove::Move,
        piece::{Colour, Piece, PieceType},
        squareset::SquareSet,
        types::Square,
    },
    historytable::{
        CORRECTION_HISTORY_MAX, HASH_HISTORY_SIZE, history_delta, update_cont_history,
        update_correction, update_history,
    },
    lookups::PIECE_KEYS,
    searchinfo::SearchInfo,
    stack::StackFrame,
    threadlocal::{Histories, ThreadData},
    util::MAX_DEPTH,
};

use crate::chess::board::Board;

#[derive(Clone, Copy)]
pub struct UpdateCtx<'a> {
    pub board: &'a Board,
    pub info: &'a SearchInfo<'a>,
}

macro_rules! ctx {
    ($t:ident) => {
        $crate::history::UpdateCtx {
            board: &$t.board,
            info: &$t.info,
        }
    };
}
pub(crate) use ctx;

impl Histories {
    /// Apply a delta to the main history counters (piece-to & from-to) for a single move.
    pub fn update_main_history_single(
        &mut self,
        from: Square,
        to: Square,
        moved: Piece,
        threats: SquareSet,
        delta: i32,
    ) {
        let ft = threats.contains_square(from);
        let tt = threats.contains_square(to);
        update_history(&mut self.piece_to.get_mut(ft, tt)[moved][to], delta);
        update_history(&mut self.from_to.get_mut(ft, tt)[from][to], delta);
    }

    /// Apply a delta to the main history for the inbound edge into a node,
    /// i.e. a move that has already been made on `board`.
    pub fn update_inbound_edge(&mut self, board: &Board, mov: Move, delta: i32) {
        let from = mov.from();
        let to = mov.history_to_square();
        let moved = board.state.mailbox[to].expect("Cannot fail, move has been made.");
        debug_assert_eq!(moved.colour(), !board.turn());
        let threats = board.history().last().unwrap().threats.all;
        self.update_main_history_single(from, to, moved, threats, delta);
    }

    /// Apply a delta to the pawn-structure history counter for a single move.
    #[inline]
    pub fn update_pawn_history_single(
        &mut self,
        ctx: UpdateCtx,
        to: Square,
        moved: Piece,
        delta: i32,
    ) {
        #[expect(clippy::cast_possible_truncation)]
        let pawn_index = (ctx.board.state.keys.pawn % HASH_HISTORY_SIZE as u64) as usize;
        update_history(&mut self.pawn[pawn_index][moved][to], delta);
    }

    /// Update the continuation history counters for a single move.
    pub fn update_cont_hist_single(
        &mut self,
        ctx: UpdateCtx,
        ss: &[StackFrame; MAX_DEPTH + 1],
        to: Square,
        piece: Piece,
        depth: i32,
        height: usize,
        good: bool,
    ) {
        let conf = &ctx.info.conf;
        let plies_back = [
            (1, conf.cont1_stat_score_mul, &conf.cont1_history),
            (2, conf.cont2_stat_score_mul, &conf.cont2_history),
            // (3, conf.cont3_stat_score_mul, &conf.cont3_history),
            (4, conf.cont4_stat_score_mul, &conf.cont4_history),
            // (5, conf.cont5_stat_score_mul, &conf.cont5_history),
            // (6, conf.cont6_stat_score_mul, &conf.cont6_history),
        ];

        let mut sum = 0;
        for &(index, mul, _) in &plies_back {
            if height < index {
                break;
            }
            sum += mul * i32::from(self.continuation[ss[height - index].ch_idx][piece][to]);
        }
        sum /= 32;

        for &(index, _, history_conf) in &plies_back {
            if height < index {
                break;
            }
            let val = &mut self.continuation[ss[height - index].ch_idx][piece][to];
            update_cont_history(val, sum, history_delta(history_conf, depth, good));
        }
    }

    /// Update the main, continuation, and pawn-structure history tables for a single quiet move.
    #[expect(
        clippy::inline_always,
        reason = "called in a loop from which the prologue can be hoisted"
    )]
    #[inline(always)]
    pub fn update_quiet_history_single(
        &mut self,
        ctx: UpdateCtx,
        ss: &[StackFrame; MAX_DEPTH + 1],
        m: Move,
        depth: i32,
        height: usize,
        good: bool,
    ) {
        let conf = &ctx.info.conf;
        let from = m.from();
        let to = m.history_to_square();
        let moved = ctx.board.state.mailbox[from].unwrap();
        let threats = ctx.board.state.threats.all;

        let main_delta = history_delta(&conf.main_history, depth, good);
        let pawn_delta = history_delta(&conf.pawn_history, depth, good);

        self.update_main_history_single(from, to, moved, threats, main_delta);
        self.update_cont_hist_single(ctx, ss, to, moved, depth, height, good);
        self.update_pawn_history_single(ctx, to, moved, pawn_delta);
    }

    /// Update the main, continuation, and pawn-structure history tables for a batch of quiet moves.
    pub fn update_quiet_history(
        &mut self,
        ctx: UpdateCtx,
        ss: &[StackFrame; MAX_DEPTH + 1],
        moves: &[Move],
        best_move: Move,
        depth: i32,
    ) {
        let height = ctx.board.height();
        for &m in moves {
            self.update_quiet_history_single(ctx, ss, m, depth, height, m == best_move);
        }
    }

    /// Update the tactical history counters of a batch of moves.
    pub fn update_tactical_history(
        &mut self,
        ctx: UpdateCtx,
        moves: &[Move],
        best_move: Move,
        depth: i32,
    ) {
        let threats = ctx.board.state.threats.all;
        let deltas =
            [false, true].map(|good| history_delta(&ctx.info.conf.tactical_history, depth, good));
        for &m in moves {
            let piece_moved = ctx.board.state.mailbox[m.from()].unwrap();
            let capture = caphist_piece_type(ctx.board, m);
            let to = m.to();
            let to_threat = threats.contains_square(to);
            let val = &mut self.tactical[usize::from(to_threat)][capture][piece_moved][to];
            update_history(val, deltas[usize::from(m == best_move)]);
        }
    }
}

impl ThreadData<'_> {
    /// Add a killer move.
    pub fn insert_killer(&mut self, m: Move) {
        debug_assert!(self.board.height() < MAX_DEPTH);
        let idx = self.board.height();
        self.killer_move_table[idx] = Some(m);
    }

    /// Update the correction history for a pawn pattern.
    pub fn update_correction_history(&mut self, depth: i32, tt_complexity: i32, diff: i32) {
        #![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

        use Colour::{Black, White};

        let us = self.board.turn();
        let height = self.board.height();

        // wow! floating point in a chess engine!
        let tt_complexity_factor =
            ((1.0 + (tt_complexity as f32 + 1.0).log2() / 10.0) * 8.0) as i32;

        let bonus = i32::clamp(
            diff * depth * tt_complexity_factor / 64,
            -CORRECTION_HISTORY_MAX / 4,
            CORRECTION_HISTORY_MAX / 4,
        );

        let keys = &self.board.state.keys;

        let pawn = self.pawn_corrhist.get_mut(us, keys.pawn);
        let [nonpawn_white, nonpawn_black] = &mut self.nonpawn_corrhist;
        let nonpawn_white = nonpawn_white.get_mut(us, keys.non_pawn[White]);
        let nonpawn_black = nonpawn_black.get_mut(us, keys.non_pawn[Black]);
        let minor = self.minor_corrhist.get_mut(us, keys.minor);
        let major = self.major_corrhist.get_mut(us, keys.major);

        let update = move |entry: &mut i16| {
            update_correction(entry, bonus);
        };

        update(pawn);
        update(nonpawn_white);
        update(nonpawn_black);
        update(minor);
        update(major);

        if height > 2 {
            let index = cont_corrhist_index(&self.ss, height, 2);
            update(self.cont_corrhist.get_mut(us, index));
        }

        if height > 4 {
            let index = cont_corrhist_index(&self.ss, height, 4);
            update(self.cont_corrhist.get_mut(us, index));
        }
    }

    /// Adjust a raw evaluation using statistics from the correction history.
    #[allow(clippy::cast_possible_truncation)]
    pub fn correction(&self) -> i32 {
        use Colour::{Black, White};

        let keys = &self.board.state.keys;
        let us = self.board.turn();
        let height = self.board.height();

        let pawn = self.pawn_corrhist.get(us, keys.pawn);
        let [white, black] = &self.nonpawn_corrhist;
        let white = white.get(us, keys.non_pawn[White]);
        let black = black.get(us, keys.non_pawn[Black]);
        let minor = self.minor_corrhist.get(us, keys.minor);
        let major = self.major_corrhist.get(us, keys.major);

        let cont12 = if height > 2 {
            self.cont_corrhist
                .get(us, cont_corrhist_index(&self.ss, height, 2))
        } else {
            0
        };

        let cont14 = if height > 4 {
            self.cont_corrhist
                .get(us, cont_corrhist_index(&self.ss, height, 4))
        } else {
            0
        };

        let adjustment = pawn * i64::from(self.info.conf.pawn_corrhist_weight)
            + major * i64::from(self.info.conf.major_corrhist_weight)
            + minor * i64::from(self.info.conf.minor_corrhist_weight)
            + (white + black) * i64::from(self.info.conf.nonpawn_corrhist_weight)
            + cont12 * i64::from(self.info.conf.continuation_12_corrhist_weight)
            + cont14 * i64::from(self.info.conf.continuation_14_corrhist_weight);

        (adjustment * 12 / 0x40000) as i32
    }
}

/// Compute the continuation-correction-history key from
/// the move `1` ply back and the move `back` plies back.
fn cont_corrhist_index(ss: &[StackFrame; MAX_DEPTH + 1], height: usize, back: usize) -> u64 {
    let ch1 = ss[height - 1].ch_idx;
    let ch2 = ss[height - back].ch_idx;
    PIECE_KEYS[ch1.piece][ch1.to] ^ PIECE_KEYS[ch2.piece][ch2.to]
}

pub fn caphist_piece_type(pos: &Board, mv: Move) -> PieceType {
    if mv.is_ep() || mv.is_promo() {
        // it's fine to make all promos of type PAWN,
        // because you'd never usually capture pawns on
        // the back ranks, so these slots are free in
        // the capture history table.
        PieceType::Pawn
    } else {
        debug_assert!(!mv.is_castle(), "shouldn't be using caphist for castling.");
        pos.state.mailbox[mv.to()]
            .expect("you weren't capturing anything!")
            .piece_type()
    }
}
