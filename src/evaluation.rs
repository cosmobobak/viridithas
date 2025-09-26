// The granularity of evaluation in this engine is in centipawns.

use crate::{
    chess::{
        board::Board,
        chessmove::Move,
        piece::{Colour, PieceType},
        squareset::SquareSet,
    },
    nnue::network,
    search::draw_score,
    searchinfo::SearchInfo,
    threadlocal::ThreadData,
    util::MAX_DEPTH,
};

/// The value of checkmate.
/// To recover depth-to-mate, we subtract depth (ply) from this value.
/// e.g. if white has a mate in two ply, the output from a depth-5 search will be
/// two less than `MATE_SCORE`.
pub const MATE_SCORE: i32 = i16::MAX as i32 - 367;
pub const fn mate_in(ply: usize) -> i32 {
    #![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    debug_assert!(ply <= MAX_DEPTH);
    MATE_SCORE - ply as i32
}
pub const fn mated_in(ply: usize) -> i32 {
    #![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    debug_assert!(ply <= MAX_DEPTH);
    -MATE_SCORE + ply as i32
}
pub const TB_WIN_SCORE: i32 = MATE_SCORE - 1000;
pub const fn tb_win_in(ply: usize) -> i32 {
    #![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    debug_assert!(ply <= MAX_DEPTH);
    TB_WIN_SCORE - ply as i32
}
pub const fn tb_loss_in(ply: usize) -> i32 {
    #![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    debug_assert!(ply <= MAX_DEPTH);
    -TB_WIN_SCORE + ply as i32
}

/// A threshold over which scores must be mate.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub const MINIMUM_MATE_SCORE: i32 = MATE_SCORE - 2 * MAX_DEPTH as i32 - 44;
/// A threshold over which scores must be a TB win (or mate).
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub const MINIMUM_TB_WIN_SCORE: i32 = TB_WIN_SCORE - 2 * MAX_DEPTH as i32 - 44;

pub const fn is_mate_score(score: i32) -> bool {
    score.abs() >= MINIMUM_MATE_SCORE
}
pub const fn is_decisive(score: i32) -> bool {
    score.abs() >= MINIMUM_TB_WIN_SCORE
}

pub const SEE_PAWN_VALUE: i32 = 277;
pub const SEE_KNIGHT_VALUE: i32 = 477;
pub const SEE_BISHOP_VALUE: i32 = 442;
pub const SEE_ROOK_VALUE: i32 = 709;
pub const SEE_QUEEN_VALUE: i32 = 1300;
pub const MATERIAL_SCALE_BASE: i32 = 905;

impl Board {
    pub fn material(&self, info: &SearchInfo) -> i32 {
        #![allow(clippy::cast_possible_wrap)]
        let b = &self.state.bbs;
        (info.conf.see_knight_value * b.pieces[PieceType::Knight].count() as i32
            + info.conf.see_bishop_value * b.pieces[PieceType::Bishop].count() as i32
            + info.conf.see_rook_value * b.pieces[PieceType::Rook].count() as i32
            + info.conf.see_queen_value * b.pieces[PieceType::Queen].count() as i32)
            / 32
    }

    pub fn zugzwang_unlikely(&self) -> bool {
        let stm = self.turn();
        let us = self.state.bbs.colours[stm];
        let kings = self.state.bbs.pieces[PieceType::King];
        let pawns = self.state.bbs.pieces[PieceType::Pawn];
        (us & (kings | pawns)) != us
    }

    pub fn estimated_see(&self, info: &SearchInfo, m: Move) -> i32 {
        // initially take the value of the thing on the target square
        let mut value = self.state.mailbox[m.to()].map_or(0, |p| see_value(p.piece_type(), info));

        if let Some(promo) = m.promotion_type() {
            // if it's a promo, swap a pawn for the promoted piece type
            value += see_value(promo, info) - info.conf.see_pawn_value;
        } else if m.is_ep() {
            // for e.p. we will miss a pawn because the target square is empty
            value = info.conf.see_pawn_value;
        }

        value
    }
}

pub fn evaluate_nnue(t: &ThreadData) -> i32 {
    // get the raw network output
    let output_bucket = network::output_bucket(&t.board);
    let v = t
        .nnue
        .evaluate(t.nnue_params, t.board.turn(), output_bucket);

    // clamp the value into the valid range.
    // this basically never comes up, but the network will
    // occasionally output OOB values in crazy positions with
    // massive material imbalances.
    v.clamp(-MINIMUM_TB_WIN_SCORE + 1024, MINIMUM_TB_WIN_SCORE - 1024)
}

pub fn evaluate(t: &mut ThreadData, nodes: u64) -> i32 {
    // detect draw by insufficient material
    if t.board.state.bbs.pieces[PieceType::Pawn] == SquareSet::EMPTY
        && t.board.state.bbs.is_material_draw()
    {
        return if t.board.turn() == Colour::White {
            draw_score(t, nodes, t.board.turn())
        } else {
            -draw_score(t, nodes, t.board.turn())
        };
    }
    // apply all in-waiting updates to generate a valid
    // neural network accumulator state.
    t.nnue.force(&t.board, t.nnue_params);
    // run the neural network evaluation
    evaluate_nnue(t)
}

pub const fn see_value(piece_type: PieceType, info: &SearchInfo) -> i32 {
    match piece_type {
        PieceType::Pawn => info.conf.see_pawn_value,
        PieceType::Knight => info.conf.see_knight_value,
        PieceType::Bishop => info.conf.see_bishop_value,
        PieceType::Rook => info.conf.see_rook_value,
        PieceType::Queen => info.conf.see_queen_value,
        PieceType::King => 0,
    }
}
