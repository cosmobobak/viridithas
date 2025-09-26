#![allow(unused_imports)]

use crate::{
    chess::{
        board::Board,
        piece::{Colour, Piece},
        types::{Rank, Square},
    },
    nnue::network::NNUEState,
    searchinfo::SearchInfo,
};

#[cfg(debug_assertions)]
use crate::errors::PositionValidityError;

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
            let piece_ss = self.state.bbs.piece_at(sq);
            let piece_mb = self.state.mailbox[sq];
            if piece_ss != piece_mb {
                return Err(format!(
                    "square-set / piece array coherency corrupt: expected square {sq} to be '{piece_ss:?}' but was '{piece_mb:?}'",
                ));
            }
        }

        if !(self.side == Colour::White || self.side == Colour::Black) {
            return Err(format!(
                "side is corrupt: expected WHITE or BLACK, got {:?}",
                self.side
            ));
        }
        if self.state.generate_pos_keys(self.side) != self.state.keys {
            return Err(format!(
                "key is corrupt: expected {:?}, got {:?}",
                self.state.generate_pos_keys(self.side),
                self.state.keys
            ));
        }

        if !(self.state.ep_square.is_none()
            || (self.state.ep_square.unwrap().rank() == Rank::Six && self.side == Colour::White)
            || (self.state.ep_square.unwrap().rank() == Rank::Three && self.side == Colour::Black))
        {
            return Err(format!(
                "en passant square is corrupt: expected square to be None or to be on ranks 6 or 3, got {} ({:?})",
                self.state.ep_square.unwrap(),
                self.state.ep_square.unwrap().rank()
            ));
        }

        // the fifty-move counter is allowed to be *exactly* 100, to allow a finished game to be
        // created.
        if self.state.fifty_move_counter > 100 {
            return Err(format!(
                "fifty move counter is corrupt: expected 0-100, got {}",
                self.state.fifty_move_counter
            ));
        }

        // check there are the correct number of kings for each side
        if self.state.bbs.piece_bb(Piece::WK).count() != 1 {
            return Err(format!(
                "white king count is corrupt: expected 1, got {}",
                self.state.bbs.piece_bb(Piece::WK).count()
            ));
        }
        if self.state.bbs.piece_bb(Piece::BK).count() != 1 {
            return Err(format!(
                "black king count is corrupt: expected 1, got {}",
                self.state.bbs.piece_bb(Piece::BK).count()
            ));
        }

        if self.state.mailbox[self.state.bbs.king_sq(Colour::White)] != Some(Piece::WK) {
            return Err(format!(
                "white king square is corrupt: expected white king, got {:?}",
                self.state.mailbox[self.state.bbs.king_sq(Colour::White)]
            ));
        }
        if self.state.mailbox[self.state.bbs.king_sq(Colour::Black)] != Some(Piece::BK) {
            return Err(format!(
                "black king square is corrupt: expected black king, got {:?}",
                self.state.mailbox[self.state.bbs.king_sq(Colour::Black)]
            ));
        }

        Ok(())
    }
}
