#![allow(unused_imports)]

use crate::{
    chess::{
        piece::{Colour, Piece},
        types::{Rank, Square},
    },
    nnue::network::NNUEState,
    searchinfo::SearchInfo,
};

#[cfg(debug_assertions)]
use crate::errors::PositionValidityError;

use super::{movegen::piecelayout::SquareIter, Board};

impl Board {
    #[cfg(debug_assertions)]
    #[allow(clippy::cognitive_complexity, clippy::too_many_lines)]
    pub fn check_validity(&self) -> Result<(), PositionValidityError> {
        #![allow(clippy::similar_names, clippy::cast_possible_truncation)]

        // check turn
        if self.side != Colour::White && self.side != Colour::Black {
            return Err(format!("invalid side: {:?}", self.side));
        }

        // check square-set / piece array coherency
        for sq in Square::all() {
            let piece = self.piece_array[sq];
            if self.pieces.piece_at(sq) != piece {
                return Err(format!(
                    "square-set / piece array coherency corrupt: expected square {} to be '{:?}' but was '{:?}'",
                    sq,
                    piece,
                    self.piece_at(sq)
                ));
            }
        }

        if !(self.side == Colour::White || self.side == Colour::Black) {
            return Err(format!(
                "side is corrupt: expected WHITE or BLACK, got {:?}",
                self.side
            ));
        }
        if self.generate_pos_keys() != self.all_keys() {
            return Err(format!(
                "key is corrupt: expected {:?}, got {:?}",
                self.generate_pos_keys(),
                self.all_keys()
            ));
        }

        if !(self.ep_sq.is_none()
            || (self.ep_sq.unwrap().rank() == Rank::Six && self.side == Colour::White)
            || (self.ep_sq.unwrap().rank() == Rank::Three && self.side == Colour::Black))
        {
            return Err(format!(
                "en passant square is corrupt: expected square to be None or to be on ranks 6 or 3, got {} ({:?})",
                self.ep_sq.unwrap(),
                self.ep_sq.unwrap().rank()
            ));
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

        if self.piece_at(self.king_sq(Colour::White)) != Some(Piece::WK) {
            return Err(format!(
                "white king square is corrupt: expected white king, got {:?}",
                self.piece_at(self.king_sq(Colour::White))
            ));
        }
        if self.piece_at(self.king_sq(Colour::Black)) != Some(Piece::BK) {
            return Err(format!(
                "black king square is corrupt: expected black king, got {:?}",
                self.piece_at(self.king_sq(Colour::Black))
            ));
        }

        Ok(())
    }
}
