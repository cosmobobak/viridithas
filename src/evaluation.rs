// The granularity of evaluation in this engine is in centipawns.

use crate::{
    chess::board::Board,
    chess::chessmove::Move,
    chess::piece::{Colour, Piece, PieceType},
    nnue::network,
    search::draw_score,
    threadlocal::ThreadData,
    util::{MAX_DEPTH, MAX_PLY},
};

/// The value of checkmate.
/// To recover depth-to-mate, we subtract depth (ply) from this value.
/// e.g. if white has a mate in two ply, the output from a depth-5 search will be
/// two less than `MATE_SCORE`.
pub const MATE_SCORE: i32 = i16::MAX as i32 - 300;
pub const fn mate_in(ply: usize) -> i32 {
    #![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    debug_assert!(ply <= MAX_PLY);
    MATE_SCORE - ply as i32
}
pub const fn mated_in(ply: usize) -> i32 {
    #![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    debug_assert!(ply <= MAX_PLY);
    -MATE_SCORE + ply as i32
}
pub const TB_WIN_SCORE: i32 = MATE_SCORE - 1000;
pub const fn tb_win_in(ply: usize) -> i32 {
    #![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    debug_assert!(ply <= MAX_PLY);
    TB_WIN_SCORE - ply as i32
}
pub const fn tb_loss_in(ply: usize) -> i32 {
    #![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    debug_assert!(ply <= MAX_PLY);
    -TB_WIN_SCORE + ply as i32
}

/// A threshold over which scores must be mate.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
const MINIMUM_MATE_SCORE: i32 = MATE_SCORE - MAX_DEPTH;
/// A threshold over which scores must be a TB win (or mate).
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub const MINIMUM_TB_WIN_SCORE: i32 = TB_WIN_SCORE - MAX_DEPTH;

pub const fn is_mate_score(score: i32) -> bool {
    score.abs() >= MINIMUM_MATE_SCORE
}
pub const fn is_game_theoretic_score(score: i32) -> bool {
    score.abs() >= MINIMUM_TB_WIN_SCORE
}

impl Board {
    fn material_scale(&self) -> i32 {
        #![allow(clippy::cast_possible_wrap)]
        let b = self.pieces();
        700 + (PieceType::Knight.see_value() * b.all_knights().count() as i32
            + PieceType::Bishop.see_value() * b.all_bishops().count() as i32
            + PieceType::Rook.see_value() * b.all_rooks().count() as i32
            + PieceType::Queen.see_value() * b.all_queens().count() as i32)
            / 32
    }

    pub fn evaluate_nnue(&self, t: &ThreadData) -> i32 {
        // get the raw network output
        let output_bucket = network::output_bucket(self);
        let v = t.nnue.evaluate(t.nnue_params, self.turn(), output_bucket);

        // scale down the value estimate when there's not much
        // material left - this will incentivize keeping material
        // on the board if we have winning chances, and trading
        // material off if the position is worse for us.
        let v = v * self.material_scale() / 1024;

        // scale down the value when the fifty-move counter is high.
        // this goes some way toward making viri realise when he's not
        // making progress in a position.
        let v = v * (200 - i32::from(self.fifty_move_counter())) / 200;

        // clamp the value into the valid range.
        // this basically never comes up, but the network will
        // occasionally output OOB values in crazy positions with
        // massive material imbalances.
        v.clamp(-MINIMUM_TB_WIN_SCORE + 1, MINIMUM_TB_WIN_SCORE - 1)
    }

    pub fn evaluate(&self, t: &mut ThreadData, nodes: u64) -> i32 {
        // detect draw by insufficient material
        if !self.pieces().any_pawns() && self.pieces().is_material_draw() {
            return if self.turn() == Colour::White {
                draw_score(t, nodes, self.turn())
            } else {
                -draw_score(t, nodes, self.turn())
            };
        }
        // apply all in-waiting updates to generate a valid
        // neural network accumulator state.
        t.nnue.force(self, t.nnue_params);
        // run the neural network evaluation
        self.evaluate_nnue(t)
    }

    pub fn zugzwang_unlikely(&self) -> bool {
        let stm = self.turn();
        let us = self.pieces().occupied_co(stm);
        let kings = self.pieces().all_kings();
        let pawns = self.pieces().all_pawns();
        (us & (kings | pawns)) != us
    }

    pub fn estimated_see(&self, m: Move) -> i32 {
        // initially take the value of the thing on the target square
        let mut value = self
            .piece_at(m.to())
            .map_or(0, |p| PieceType::see_value(Piece::piece_type(p)));

        if let Some(promo) = m.promotion_type() {
            // if it's a promo, swap a pawn for the promoted piece type
            value += promo.see_value() - PieceType::Pawn.see_value();
        } else if m.is_ep() {
            // for e.p. we will miss a pawn because the target square is empty
            value = PieceType::Pawn.see_value();
        }

        value
    }
}
