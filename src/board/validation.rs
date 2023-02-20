#![allow(unused_imports)]

use crate::{
    board::evaluation::score::S,
    definitions::{
        Rank::{RANK_3, RANK_6},
        Square,
    },
    lookups::{PIECE_BIG, PIECE_MAJ},
    nnue::NNUEState, piece::{Colour, Piece},
};

#[cfg(debug_assertions)]
use crate::{errors::PositionValidityError, lookups::PIECE_MIN};

use super::{evaluation::get_eval_params, movegen::bitboards::BitLoop, Board};

impl Board {
    #[cfg(debug_assertions)]
    #[allow(clippy::cognitive_complexity, clippy::too_many_lines)]
    pub fn check_validity(&self) -> Result<(), PositionValidityError> {
        #![allow(clippy::similar_names, clippy::cast_possible_truncation)]
        let mut big_pce = [0, 0];
        let mut maj_pce = [0, 0];
        let mut min_pce = [0, 0];
        let mut material = [S(0, 0), S(0, 0)];

        // check turn
        if self.side != Colour::WHITE && self.side != Colour::BLACK {
            return Err(format!("invalid side: {:?}", self.side));
        }

        // check piece count and other counters
        for sq in Square::all() {
            let piece = self.piece_at(sq);
            if piece == Piece::EMPTY {
                continue;
            }
            let colour = piece.colour();
            if PIECE_BIG[piece.index()] {
                big_pce[colour.index()] += 1;
            }
            if PIECE_MAJ[piece.index()] {
                maj_pce[colour.index()] += 1;
            }
            if PIECE_MIN[piece.index()] {
                min_pce[colour.index()] += 1;
            }
            material[colour.index()] += get_eval_params().piece_values[piece.index()];
        }

        // check bitboard / piece array coherency
        for piece in Piece::all() {
            let bb = self.pieces.piece_bb(piece);
            for sq in BitLoop::new(bb) {
                if self.piece_at(sq) != piece {
                    return Err(format!(
                        "bitboard / piece array coherency corrupt: expected square {} to be '{:?}' but was '{:?}'",
                        sq,
                        piece,
                        self.piece_at(sq)
                    ));
                }
            }
        }

        if material[Colour::WHITE.index()].0 != self.material[Colour::WHITE.index()].0 {
            return Err(format!(
                "white midgame material is corrupt: expected {:?}, got {:?}",
                material[Colour::WHITE.index()].0, self.material[Colour::WHITE.index()].0
            ));
        }
        if material[Colour::WHITE.index()].1 != self.material[Colour::WHITE.index()].1 {
            return Err(format!(
                "white endgame material is corrupt: expected {:?}, got {:?}",
                material[Colour::WHITE.index()].1, self.material[Colour::WHITE.index()].1
            ));
        }
        if material[Colour::BLACK.index()].0 != self.material[Colour::BLACK.index()].0 {
            return Err(format!(
                "black midgame material is corrupt: expected {:?}, got {:?}",
                material[Colour::BLACK.index()].0, self.material[Colour::BLACK.index()].0
            ));
        }
        if material[Colour::BLACK.index()].1 != self.material[Colour::BLACK.index()].1 {
            return Err(format!(
                "black endgame material is corrupt: expected {:?}, got {:?}",
                material[Colour::BLACK.index()].1, self.material[Colour::BLACK.index()].1
            ));
        }
        if min_pce[Colour::WHITE.index()] != self.minor_piece_counts[Colour::WHITE.index()] {
            return Err(format!(
                "white minor piece count is corrupt: expected {:?}, got {:?}",
                min_pce[Colour::WHITE.index()], self.minor_piece_counts[Colour::WHITE.index()]
            ));
        }
        if min_pce[Colour::BLACK.index()] != self.minor_piece_counts[Colour::BLACK.index()] {
            return Err(format!(
                "black minor piece count is corrupt: expected {:?}, got {:?}",
                min_pce[Colour::BLACK.index()], self.minor_piece_counts[Colour::BLACK.index()]
            ));
        }
        if maj_pce[Colour::WHITE.index()] != self.major_piece_counts[Colour::WHITE.index()] {
            return Err(format!(
                "white major piece count is corrupt: expected {:?}, got {:?}",
                maj_pce[Colour::WHITE.index()], self.major_piece_counts[Colour::WHITE.index()]
            ));
        }
        if maj_pce[Colour::BLACK.index()] != self.major_piece_counts[Colour::BLACK.index()] {
            return Err(format!(
                "black major piece count is corrupt: expected {:?}, got {:?}",
                maj_pce[Colour::BLACK.index()], self.major_piece_counts[Colour::BLACK.index()]
            ));
        }
        if big_pce[Colour::WHITE.index()] != self.big_piece_counts[Colour::WHITE.index()] {
            return Err(format!(
                "white big piece count is corrupt: expected {:?}, got {:?}",
                big_pce[Colour::WHITE.index()], self.big_piece_counts[Colour::WHITE.index()]
            ));
        }
        if big_pce[Colour::BLACK.index()] != self.big_piece_counts[Colour::BLACK.index()] {
            return Err(format!(
                "black big piece count is corrupt: expected {:?}, got {:?}",
                big_pce[Colour::BLACK.index()], self.big_piece_counts[Colour::BLACK.index()]
            ));
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
            || (self.ep_sq.rank() == RANK_6 && self.side == Colour::WHITE)
            || (self.ep_sq.rank() == RANK_3 && self.side == Colour::BLACK))
        {
            return Err(format!("en passant square is corrupt: expected square to be {} or to be on ranks 6 or 3, got {} (Rank {})", Square::NO_SQUARE, self.ep_sq, self.ep_sq.rank()));
        }

        if self.fifty_move_counter >= 100 {
            return Err(format!(
                "fifty move counter is corrupt: expected 0-99, got {}",
                self.fifty_move_counter
            ));
        }

        // check there are the correct number of kings for each side
        if self.num(Piece::WK) != 1 {
            return Err(format!("white king count is corrupt: expected 1, got {}", self.num(Piece::WK)));
        }
        if self.num(Piece::BK) != 1 {
            return Err(format!("black king count is corrupt: expected 1, got {}", self.num(Piece::BK)));
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
    
    pub fn check_nnue_coherency(&self, nn: &NNUEState) -> bool {
        #[cfg(debug_assertions)]
        for (colour, piece_type, square) in
            nn.active_features().map(NNUEState::feature_loc_to_parts)
        {
            let piece_on_board = self.piece_at(square);
            let actual_colour = piece_on_board.colour();
            if colour != actual_colour {
                eprintln!(
                    "coherency check failed: feature on sq {square} has colour {colour:?}, but piece on board is {piece_on_board:?}"
                );
                eprintln!("fen: {}", self.fen());
                return false;
            }
            let actual_piece_type = piece_on_board.piece_type();
            if piece_type != actual_piece_type {
                eprintln!(
                    "coherency check failed: feature on sq {square} has piece type {piece_type:?}, but piece on board is {piece_on_board:?}"
                );
                eprintln!("fen: {}", self.fen());
                return false;
            }
        }
        #[cfg(not(debug_assertions))]
        let _ = (self, nn);
        true
    }
}
