// The granularity of evaluation in this engine is in centipawns.

use crate::{
    board::Board,
    chessmove::Move,
    piece::{Colour, PieceType},
    search::draw_score,
    threadlocal::ThreadData,
    util::MAX_DEPTH,
};

/// The value of checkmate.
/// To recover depth-to-mate, we subtract depth (ply) from this value.
/// e.g. if white has a mate in two ply, the output from a depth-5 search will be
/// two less than `MATE_SCORE`.
pub const MATE_SCORE: i32 = i16::MAX as i32 - 300;
pub const fn mate_in(ply: usize) -> i32 {
    #![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    debug_assert!(ply <= MAX_DEPTH.ply_to_horizon());
    MATE_SCORE - ply as i32
}
pub const fn mated_in(ply: usize) -> i32 {
    #![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    debug_assert!(ply <= MAX_DEPTH.ply_to_horizon());
    -MATE_SCORE + ply as i32
}
pub const TB_WIN_SCORE: i32 = MATE_SCORE - 1000;
pub const fn tb_win_in(ply: usize) -> i32 {
    #![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    debug_assert!(ply <= MAX_DEPTH.ply_to_horizon());
    TB_WIN_SCORE - ply as i32
}
pub const fn tb_loss_in(ply: usize) -> i32 {
    #![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    debug_assert!(ply <= MAX_DEPTH.ply_to_horizon());
    -TB_WIN_SCORE + ply as i32
}

/// A threshold over which scores must be mate.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
const MINIMUM_MATE_SCORE: i32 = MATE_SCORE - MAX_DEPTH.ply_to_horizon() as i32;
/// A threshold over which scores must be a TB win (or mate).
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub const MINIMUM_TB_WIN_SCORE: i32 = TB_WIN_SCORE - MAX_DEPTH.ply_to_horizon() as i32;

pub const fn is_mate_score(score: i32) -> bool {
    score.abs() >= MINIMUM_MATE_SCORE
}
pub const fn is_game_theoretic_score(score: i32) -> bool {
    score.abs() >= MINIMUM_TB_WIN_SCORE
}

impl Board {
    const fn material_scale(&self) -> i32 {
        #![allow(clippy::cast_possible_wrap)]
        700 + (PieceType::KNIGHT.see_value() * self.pieces.all_knights().count() as i32
            + PieceType::BISHOP.see_value() * self.pieces.all_bishops().count() as i32
            + PieceType::ROOK.see_value() * self.pieces.all_rooks().count() as i32
            + PieceType::QUEEN.see_value() * self.pieces.all_queens().count() as i32)
            / 32
    }

    pub fn evaluate_nnue(&self, t: &ThreadData) -> i32 {
        // get the raw network output
        let v = t.nnue.evaluate(self.side);

        // scale down the value estimate when there's not much
        // material left - this will incentivize keeping material
        // on the board if we have winning chances, and trading
        // material off if the position is worse for us.
        let v = v * self.material_scale() / 1024;

        // scale down the value when the fifty-move counter is high.
        // this goes some way toward making viri realise when he's not
        // making progress in a position.
        let v = v * (200 - i32::from(self.fifty_move_counter)) / 200;

        // clamp the value into the valid range.
        // this basically never comes up, but the network will
        // occasionally output OOB values in crazy positions with
        // massive material imbalances.
        v.clamp(-MINIMUM_TB_WIN_SCORE + 1, MINIMUM_TB_WIN_SCORE - 1)
    }

    pub fn evaluate(&self, t: &mut ThreadData, nodes: u64) -> i32 {
        // detect draw by insufficient material
        if !self.pieces.any_pawns() && self.pieces.is_material_draw() {
            return if self.side == Colour::WHITE {
                draw_score(t, nodes, self.turn())
            } else {
                -draw_score(t, nodes, self.turn())
            };
        }
        // apply all in-waiting updates to generate a valid
        // neural network accumulator state.
        t.nnue.force(self);
        // run the neural network evaluation
        self.evaluate_nnue(t)
    }

    pub fn zugzwang_unlikely(&self) -> bool {
        // TODO: this can be done without even looking at the king / pawn BBs
        let stm = self.turn();
        let us = self.pieces.occupied_co(stm);
        let kings = self.pieces.all_kings();
        let pawns = self.pieces.all_pawns();
        (us & (kings | pawns)) != us
    }

    pub fn estimated_see(&self, m: Move) -> i32 {
        // initially take the value of the thing on the target square
        let mut value = self.piece_at(m.to()).piece_type().see_value();

        if m.is_promo() {
            // if it's a promo, swap a pawn for the promoted piece type
            value += m.promotion_type().see_value() - PieceType::PAWN.see_value();
        } else if m.is_ep() {
            // for e.p. we will miss a pawn because the target square is empty
            value = PieceType::PAWN.see_value();
        }

        value
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct KingDangerInfo {
    attack_units_on_white: i32,
    attack_units_on_black: i32,
}
