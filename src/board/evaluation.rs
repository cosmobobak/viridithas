// The granularity of evaluation in this engine is in centipawns.

use crate::{
    board::Board,
    chessmove::Move,
    piece::{Colour, Piece, PieceType},
    search::draw_score,
    squareset::SquareSet,
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
pub const MINIMUM_MATE_SCORE: i32 = MATE_SCORE - MAX_DEPTH.ply_to_horizon() as i32;
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

    pub fn evaluate_nnue(&self, t: &ThreadData, nodes: u64) -> i32 {
        if !self.pieces.any_pawns() && self.is_material_draw() {
            return if self.side == Colour::WHITE {
                draw_score(t, nodes, self.turn())
            } else {
                -draw_score(t, nodes, self.turn())
            };
        }

        let v = t.nnue.evaluate(self.side);
        let v = v * self.material_scale() / 1024;

        let v = v * (200 - i32::from(self.fifty_move_counter)) / 200;

        v.clamp(-MINIMUM_TB_WIN_SCORE + 1, MINIMUM_TB_WIN_SCORE - 1)
    }

    pub fn evaluate(&self, t: &ThreadData, nodes: u64) -> i32 {
        self.evaluate_nnue(t, nodes)
    }

    fn unwinnable_for<const IS_WHITE: bool>(&self) -> bool {
        if self.pieces.majors::<IS_WHITE>() != SquareSet::EMPTY {
            return false;
        }
        if self.pieces.minors::<IS_WHITE>().count() > 1 {
            return false;
        }
        if self.pieces.pawns::<IS_WHITE>() & self.pieces.our_pieces::<IS_WHITE>()
            != SquareSet::EMPTY
        {
            return false;
        }

        true
    }

    fn is_material_draw(&self) -> bool {
        if self.num_pt(PieceType::ROOK) == 0 && self.num_pt(PieceType::QUEEN) == 0 {
            if self.num_pt(PieceType::BISHOP) == 0 {
                if self.num(Piece::WN) < 3 && self.num(Piece::BN) < 3 {
                    return true;
                }
            } else if (self.num_pt(PieceType::KNIGHT) == 0
                && self.num(Piece::WB).abs_diff(self.num(Piece::BB)) < 2)
                || (self.num(Piece::WB) + self.num(Piece::WN) == 1
                    && self.num(Piece::BB) + self.num(Piece::BN) == 1)
            {
                return true;
            }
        } else if self.num_pt(PieceType::QUEEN) == 0 {
            if self.num(Piece::WR) == 1 && self.num(Piece::BR) == 1 {
                if (self.num(Piece::WN) + self.num(Piece::WB)) < 2
                    && (self.num(Piece::BN) + self.num(Piece::BB)) < 2
                {
                    return true;
                }
            } else if self.num(Piece::WR) == 1 && self.num(Piece::BR) == 0 {
                if (self.num(Piece::WN) + self.num(Piece::WB)) == 0
                    && ((self.num(Piece::BN) + self.num(Piece::BB)) == 1
                        || (self.num(Piece::BN) + self.num(Piece::BB)) == 2)
                {
                    return true;
                }
            } else if self.num(Piece::WR) == 0
                && self.num(Piece::BR) == 1
                && (self.num(Piece::BN) + self.num(Piece::BB)) == 0
                && ((self.num(Piece::WN) + self.num(Piece::WB)) == 1
                    || (self.num(Piece::WN) + self.num(Piece::WB)) == 2)
            {
                return true;
            }
        }
        false
    }

    #[allow(dead_code)]
    fn preprocess_drawish_scores(&self, t: &ThreadData, score: i32, nodes: u64) -> i32 {
        // if we can't win with our material, we clamp the eval to zero.
        let drawscore = draw_score(t, nodes, self.turn());
        if score > drawscore && self.unwinnable_for::<true>()
            || score < drawscore && self.unwinnable_for::<false>()
        {
            drawscore
        } else {
            score
        }
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
