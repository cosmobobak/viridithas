use std::fmt::Display;

use crate::chess::{
    board::movegen::{bishop_attacks, king_attacks, knight_attacks, pawn_attacks, rook_attacks},
    piece::{Black, Col, Colour, Piece, PieceType, White},
    squareset::SquareSet,
    types::{File, Rank, Square},
};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub struct PieceLayout {
    pub pieces: [SquareSet; 6],
    pub colours: [SquareSet; 2],
}

impl PieceLayout {
    pub fn king<C: Col>(&self) -> SquareSet {
        self.pieces[PieceType::King] & self.colours[C::COLOUR]
    }

    pub fn pawns<C: Col>(&self) -> SquareSet {
        self.pieces[PieceType::Pawn] & self.colours[C::COLOUR]
    }

    pub fn orthos<C: Col>(&self) -> SquareSet {
        (self.pieces[PieceType::Rook] | self.pieces[PieceType::Queen]) & self.colours[C::COLOUR]
    }

    pub fn diags<C: Col>(&self) -> SquareSet {
        (self.pieces[PieceType::Bishop] | self.pieces[PieceType::Queen]) & self.colours[C::COLOUR]
    }

    pub fn empty(&self) -> SquareSet {
        !self.occupied()
    }

    pub fn occupied(&self) -> SquareSet {
        self.colours[Colour::White] | self.colours[Colour::Black]
    }

    pub fn knights<C: Col>(&self) -> SquareSet {
        self.pieces[PieceType::Knight] & self.colours[C::COLOUR]
    }

    #[cfg(any(feature = "datagen", test))]
    pub fn rooks<C: Col>(&self) -> SquareSet {
        self.pieces[PieceType::Rook] & self.colours[C::COLOUR]
    }

    #[cfg(any(feature = "datagen", test))]
    pub fn bishops<C: Col>(&self) -> SquareSet {
        self.pieces[PieceType::Bishop] & self.colours[C::COLOUR]
    }

    #[cfg(any(feature = "datagen", test))]
    pub fn queens<C: Col>(&self) -> SquareSet {
        self.pieces[PieceType::Queen] & self.colours[C::COLOUR]
    }

    pub fn move_piece(&mut self, from: Square, to: Square, piece: Piece) {
        let from_to_bb = from.as_set() | to.as_set();
        self.pieces[piece.piece_type()] ^= from_to_bb;
        self.colours[piece.colour()] ^= from_to_bb;
    }

    pub fn set_piece_at(&mut self, sq: Square, piece: Piece) {
        let sq_bb = sq.as_set();
        self.pieces[piece.piece_type()] |= sq_bb;
        self.colours[piece.colour()] |= sq_bb;
    }

    pub fn clear_piece_at(&mut self, sq: Square, piece: Piece) {
        let sq_bb = sq.as_set();
        self.pieces[piece.piece_type()] &= !sq_bb;
        self.colours[piece.colour()] &= !sq_bb;
    }

    pub fn piece_bb(&self, piece: Piece) -> SquareSet {
        SquareSet::intersection(
            self.pieces[piece.piece_type()],
            self.colours[piece.colour()],
        )
    }

    pub fn all_attackers_to_sq(&self, sq: Square, occupied: SquareSet) -> SquareSet {
        let sq_bb = sq.as_set();
        let black_pawn_attackers = pawn_attacks::<White>(sq_bb) & self.pawns::<Black>();
        let white_pawn_attackers = pawn_attacks::<Black>(sq_bb) & self.pawns::<White>();
        let knight_attackers = knight_attacks(sq) & (self.pieces[PieceType::Knight]);
        let diag_attackers = bishop_attacks(sq, occupied)
            & (self.pieces[PieceType::Bishop] | self.pieces[PieceType::Queen]);
        let orth_attackers = rook_attacks(sq, occupied)
            & (self.pieces[PieceType::Rook] | self.pieces[PieceType::Queen]);
        let king_attackers = king_attacks(sq) & (self.pieces[PieceType::King]);
        black_pawn_attackers
            | white_pawn_attackers
            | knight_attackers
            | diag_attackers
            | orth_attackers
            | king_attackers
    }

    pub fn piece_at(&self, sq: Square) -> Option<Piece> {
        let sq_bb = sq.as_set();
        let colour = if (self.colours[Colour::White] & sq_bb) != SquareSet::EMPTY {
            Colour::White
        } else if (self.colours[Colour::Black] & sq_bb) != SquareSet::EMPTY {
            Colour::Black
        } else {
            return None;
        };
        for piece in PieceType::all() {
            if (self.pieces[piece] & sq_bb) != SquareSet::EMPTY {
                return Some(Piece::new(colour, piece));
            }
        }
        panic!("Bit set in colour square-set for {colour:?} but not in piece square-sets! square is {sq}");
    }

    fn any_bbs_overlapping(&self) -> bool {
        if (self.colours[0] & self.colours[1]) != SquareSet::EMPTY {
            return true;
        }
        for i in 0..self.pieces.len() {
            for j in i + 1..self.pieces.len() {
                if (self.pieces[i] & self.pieces[j]) != SquareSet::EMPTY {
                    return true;
                }
            }
        }
        false
    }

    /// Calls `callback` for each piece that is added or removed from `self` to `target`.
    pub fn update_iter(&self, target: Self, mut callback: impl FnMut(Square, Piece, bool)) {
        for colour in Colour::all() {
            for piece_type in PieceType::all() {
                let piece = Piece::new(colour, piece_type);
                let source_bb = self.pieces[piece_type] & self.colours[colour];
                let target_bb = target.pieces[piece_type] & target.colours[colour];
                let added = target_bb & !source_bb;
                let removed = source_bb & !target_bb;
                for sq in added {
                    callback(sq, piece, true);
                }
                for sq in removed {
                    callback(sq, piece, false);
                }
            }
        }
    }

    pub fn visit_pieces(&self, mut callback: impl FnMut(Square, Piece)) {
        for colour in Colour::all() {
            for piece_type in PieceType::all() {
                let piece = Piece::new(colour, piece_type);
                let bb = self.pieces[piece_type] & self.colours[colour];
                for sq in bb {
                    callback(sq, piece);
                }
            }
        }
    }

    /// Returns true if the current position *would* be a draw by insufficient material,
    /// if there were no pawns on the board.
    pub fn is_material_draw(&self) -> bool {
        let white = self.colours[Colour::White];
        let black = self.colours[Colour::Black];
        let bishops = self.pieces[PieceType::Bishop];
        let knights = self.pieces[PieceType::Knight];
        let rooks = self.pieces[PieceType::Rook];
        let queens = self.pieces[PieceType::Queen];
        let wb = bishops & white;
        let bb = bishops & black;
        let wn = knights & white;
        let bn = knights & black;
        let wr = rooks & white;
        let br = rooks & black;
        if queens != SquareSet::EMPTY {
            return false;
        }
        if rooks == SquareSet::EMPTY {
            if bishops == SquareSet::EMPTY {
                if wn.count() < 3 && bn.count() < 3 {
                    return true;
                }
            } else if knights == SquareSet::EMPTY && wb.count().abs_diff(bb.count()) < 2
                || (wb | wn).count() == 1 && (bb | bn).count() == 1
            {
                return true;
            }
        } else if wr.count() == 1 && br.count() == 1 {
            if (wn | wb).count() < 2 && (bn | bb).count() < 2 {
                return true;
            }
        } else if wr.count() == 1 && br == SquareSet::EMPTY {
            if (wn | wb) == SquareSet::EMPTY && ((bn | bb).count() == 1 || (bn | bb).count() == 2) {
                return true;
            }
        } else if wr == SquareSet::EMPTY
            && br.count() == 1
            && (bn | bb) == SquareSet::EMPTY
            && ((wn | wb).count() == 1 || (wn | wb).count() == 2)
        {
            return true;
        }
        false
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub struct Threats {
    pub all: SquareSet,
    // pub pawn: SquareSet,
    // pub minor: SquareSet,
    // pub rook: SquareSet,
    pub checkers: SquareSet,
}

impl Display for PieceLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for rank in Rank::all().rev() {
            for file in File::all() {
                let sq = Square::from_rank_file(rank, file);
                if let Some(piece) = self.piece_at(sq) {
                    write!(f, " {}", piece.char())?;
                } else {
                    write!(f, " .")?;
                }
            }
            writeln!(f)?;
        }
        if self.any_bbs_overlapping() {
            writeln!(f, "WARNING: Some square-sets are overlapping")?;
        }
        Ok(())
    }
}
