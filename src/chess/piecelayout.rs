use std::fmt::Display;

use crate::chess::{
    board::movegen::{bishop_attacks, king_attacks, knight_attacks, pawn_attacks, rook_attacks},
    piece::{Black, Col, Colour, Piece, PieceType, White},
    squareset::SquareSet,
    types::{File, Rank, Square},
};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub struct PieceLayout {
    pieces: [SquareSet; 6],
    colours: [SquareSet; 2],
}

impl PieceLayout {
    pub fn king<C: Col>(&self) -> SquareSet {
        self.all_kings() & self.our_pieces::<C>()
    }

    pub fn pawns<C: Col>(&self) -> SquareSet {
        self.all_pawns() & self.our_pieces::<C>()
    }

    pub fn occupied_co(&self, colour: Colour) -> SquareSet {
        self.colours[colour]
    }

    pub fn their_pieces<C: Col>(&self) -> SquareSet {
        self.colours[C::Opposite::COLOUR]
    }

    pub fn our_pieces<C: Col>(&self) -> SquareSet {
        self.colours[C::COLOUR]
    }

    pub fn orthos<C: Col>(&self) -> SquareSet {
        (self.all_rooks() | self.all_queens()) & self.our_pieces::<C>()
    }

    pub fn diags<C: Col>(&self) -> SquareSet {
        (self.all_bishops() | self.all_queens()) & self.our_pieces::<C>()
    }

    pub fn empty(&self) -> SquareSet {
        !self.occupied()
    }

    pub fn occupied(&self) -> SquareSet {
        self.colours[Colour::White] | self.colours[Colour::Black]
    }

    pub fn knights<C: Col>(&self) -> SquareSet {
        self.all_knights() & self.our_pieces::<C>()
    }

    #[cfg(any(feature = "datagen", test))]
    pub fn rooks<C: Col>(&self) -> SquareSet {
        self.all_rooks() & self.our_pieces::<C>()
    }

    #[cfg(any(feature = "datagen", test))]
    pub fn bishops<C: Col>(&self) -> SquareSet {
        self.all_bishops() & self.our_pieces::<C>()
    }

    #[cfg(any(feature = "datagen", test))]
    pub fn queens<C: Col>(&self) -> SquareSet {
        self.all_queens() & self.our_pieces::<C>()
    }

    pub fn all_pawns(&self) -> SquareSet {
        self.pieces[PieceType::Pawn]
    }

    pub fn all_knights(&self) -> SquareSet {
        self.pieces[PieceType::Knight]
    }

    pub fn all_bishops(&self) -> SquareSet {
        self.pieces[PieceType::Bishop]
    }

    pub fn all_rooks(&self) -> SquareSet {
        self.pieces[PieceType::Rook]
    }

    pub fn all_queens(&self) -> SquareSet {
        self.pieces[PieceType::Queen]
    }

    pub fn all_kings(&self) -> SquareSet {
        self.pieces[PieceType::King]
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

    pub fn any_pawns(&self) -> bool {
        self.all_pawns().non_empty()
    }

    pub fn piece_bb(&self, piece: Piece) -> SquareSet {
        SquareSet::intersection(
            self.pieces[piece.piece_type()],
            self.colours[piece.colour()],
        )
    }

    pub fn of_type(&self, piece_type: PieceType) -> SquareSet {
        self.pieces[piece_type]
    }

    pub fn all_attackers_to_sq(&self, sq: Square, occupied: SquareSet) -> SquareSet {
        let sq_bb = sq.as_set();
        let black_pawn_attackers = pawn_attacks::<White>(sq_bb) & self.pawns::<Black>();
        let white_pawn_attackers = pawn_attacks::<Black>(sq_bb) & self.pawns::<White>();
        let knight_attackers = knight_attacks(sq) & (self.all_knights());
        let diag_attackers =
            bishop_attacks(sq, occupied) & (self.all_bishops() | self.all_queens());
        let orth_attackers = rook_attacks(sq, occupied) & (self.all_rooks() | self.all_queens());
        let king_attackers = king_attacks(sq) & (self.all_kings());
        black_pawn_attackers
            | white_pawn_attackers
            | knight_attackers
            | diag_attackers
            | orth_attackers
            | king_attackers
    }

    pub fn piece_at(&self, sq: Square) -> Option<Piece> {
        let sq_bb = sq.as_set();
        let colour = if (self.our_pieces::<White>() & sq_bb).non_empty() {
            Colour::White
        } else if (self.our_pieces::<Black>() & sq_bb).non_empty() {
            Colour::Black
        } else {
            return None;
        };
        for piece in PieceType::all() {
            if (self.pieces[piece] & sq_bb).non_empty() {
                return Some(Piece::new(colour, piece));
            }
        }
        panic!("Bit set in colour square-set for {colour:?} but not in piece square-sets! square is {sq}");
    }

    fn any_bbs_overlapping(&self) -> bool {
        if (self.colours[0] & self.colours[1]).non_empty() {
            return true;
        }
        for i in 0..self.pieces.len() {
            for j in i + 1..self.pieces.len() {
                if (self.pieces[i] & self.pieces[j]).non_empty() {
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
        if self.of_type(PieceType::Rook).is_empty() && self.of_type(PieceType::Queen).is_empty() {
            if self.of_type(PieceType::Bishop).is_empty() {
                if self.piece_bb(Piece::WN).count() < 3 && self.piece_bb(Piece::BN).count() < 3 {
                    return true;
                }
            } else if (self.of_type(PieceType::Knight).is_empty()
                && self
                    .piece_bb(Piece::WB)
                    .count()
                    .abs_diff(self.piece_bb(Piece::BB).count())
                    < 2)
                || SquareSet::union(self.piece_bb(Piece::WB), self.piece_bb(Piece::WN)).count() == 1
                    && SquareSet::union(self.piece_bb(Piece::BB), self.piece_bb(Piece::BN)).count()
                        == 1
            {
                return true;
            }
        } else if self.of_type(PieceType::Queen).is_empty() {
            if self.piece_bb(Piece::WR).count() == 1 && self.piece_bb(Piece::BR).count() == 1 {
                if SquareSet::union(self.piece_bb(Piece::WN), self.piece_bb(Piece::WB)).count() < 2
                    && SquareSet::union(self.piece_bb(Piece::BN), self.piece_bb(Piece::BB)).count()
                        < 2
                {
                    return true;
                }
            } else if self.piece_bb(Piece::WR).count() == 1 && self.piece_bb(Piece::BR).is_empty() {
                if SquareSet::union(self.piece_bb(Piece::WN), self.piece_bb(Piece::WB)).is_empty()
                    && (SquareSet::union(self.piece_bb(Piece::BN), self.piece_bb(Piece::BB))
                        .count()
                        == 1
                        || SquareSet::union(self.piece_bb(Piece::BN), self.piece_bb(Piece::BB))
                            .count()
                            == 2)
                {
                    return true;
                }
            } else if self.piece_bb(Piece::WR).is_empty()
                && self.piece_bb(Piece::BR).count() == 1
                && SquareSet::union(self.piece_bb(Piece::BN), self.piece_bb(Piece::BB)).is_empty()
                && (SquareSet::union(self.piece_bb(Piece::WN), self.piece_bb(Piece::WB)).count()
                    == 1
                    || SquareSet::union(self.piece_bb(Piece::WN), self.piece_bb(Piece::WB)).count()
                        == 2)
            {
                return true;
            }
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
