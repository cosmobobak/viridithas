#![allow(unused_imports)]

use crate::{
    nnue::network::NNUEState,
    piece::{Colour, Piece},
    searchinfo::SearchInfo,
    util::{Rank, Square},
};

#[cfg(debug_assertions)]
use crate::errors::PositionValidityError;

use super::{movegen::bitboards::BitLoop, Board};

impl Board {
    #[cfg(debug_assertions)]
    #[allow(clippy::cognitive_complexity, clippy::too_many_lines)]
    pub fn check_validity(&self) -> Result<(), PositionValidityError> {
        #![allow(clippy::similar_names, clippy::cast_possible_truncation)]

        // check turn
        if self.side != Colour::WHITE && self.side != Colour::BLACK {
            return Err(format!("invalid side: {:?}", self.side));
        }

        // check bitboard / piece array coherency
        for sq in Square::all() {
            let piece = self.piece_array[sq.index()];
            if self.pieces.piece_at(sq) != piece {
                return Err(format!(
                    "bitboard / piece array coherency corrupt: expected square {} to be '{:?}' but was '{:?}'",
                    sq,
                    piece,
                    self.piece_at(sq)
                ));
            }
        }

        if !(self.side == Colour::WHITE || self.side == Colour::BLACK) {
            return Err(format!("side is corrupt: expected WHITE or BLACK, got {:?}", self.side));
        }
        if self.generate_pos_key() != self.key {
            return Err(format!(
                "key is corrupt: expected {:?}, got {:?}",
                self.generate_pos_key(),
                self.key
            ));
        }

        if !(self.ep_sq == Square::NO_SQUARE
            || (self.ep_sq.rank() == Rank::RANK_6 && self.side == Colour::WHITE)
            || (self.ep_sq.rank() == Rank::RANK_3 && self.side == Colour::BLACK))
        {
            return Err(format!("en passant square is corrupt: expected square to be {} or to be on ranks 6 or 3, got {} (Rank {})", Square::NO_SQUARE, self.ep_sq, self.ep_sq.rank()));
        }

        // the fifty-move counter is allowed to be *exactly* 100, to allow a finished game to be
        // created.
        if self.fifty_move_counter > 100 {
            return Err(format!(
                "fifty move counter is corrupt: expected 0-100, got {}",
                self.fifty_move_counter
            ));
        }

        // check there are the correct number of kings for each side
        if self.pieces.piece_bb(Piece::WK).count() != 1 {
            return Err(format!(
                "white king count is corrupt: expected 1, got {}",
                self.pieces.piece_bb(Piece::WK).count()
            ));
        }
        if self.pieces.piece_bb(Piece::BK).count() != 1 {
            return Err(format!(
                "black king count is corrupt: expected 1, got {}",
                self.pieces.piece_bb(Piece::BK).count()
            ));
        }

        if self.piece_at(self.king_sq(Colour::WHITE)) != Piece::WK {
            return Err(format!(
                "white king square is corrupt: expected white king, got {:?}",
                self.piece_at(self.king_sq(Colour::WHITE))
            ));
        }
        if self.piece_at(self.king_sq(Colour::BLACK)) != Piece::BK {
            return Err(format!(
                "black king square is corrupt: expected black king, got {:?}",
                self.piece_at(self.king_sq(Colour::BLACK))
            ));
        }

        Ok(())
    }
}
