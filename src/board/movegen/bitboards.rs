use std::fmt::Display;

use crate::{
    lookups, magic,
    nnue::network::FeatureUpdate,
    piece::{Black, Col, Colour, Piece, PieceType, White},
    squareset::SquareSet,
    util::Square,
};

/// Iterator over the squares of a bitboard.
/// The squares are returned in increasing order.
pub struct BitLoop {
    value: u64,
}

impl BitLoop {
    pub const fn new(value: u64) -> Self {
        Self { value }
    }
}

impl Iterator for BitLoop {
    type Item = Square;

    fn next(&mut self) -> Option<Self::Item> {
        if self.value == 0 {
            None
        } else {
            // faster if we have bmi (maybe)
            // SOUNDNESS: the trailing_zeros of a u64 cannot exceed 64, which is less than u8::MAX
            #[allow(clippy::cast_possible_truncation)]
            let lsb: u8 = self.value.trailing_zeros() as u8;
            self.value &= self.value - 1;
            Some(Square::new(lsb))
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BitBoard {
    pieces: [SquareSet; 6],
    colours: [SquareSet; 2],
}

impl BitBoard {
    pub const NULL: Self = Self::new(
        SquareSet::EMPTY,
        SquareSet::EMPTY,
        SquareSet::EMPTY,
        SquareSet::EMPTY,
        SquareSet::EMPTY,
        SquareSet::EMPTY,
        SquareSet::EMPTY,
        SquareSet::EMPTY,
    );

    #[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
    pub const fn new(
        p: SquareSet,
        n: SquareSet,
        b: SquareSet,
        r: SquareSet,
        q: SquareSet,
        k: SquareSet,
        white: SquareSet,
        black: SquareSet,
    ) -> Self {
        Self { pieces: [p, n, b, r, q, k], colours: [white, black] }
    }

    pub fn king<C: Col>(&self) -> SquareSet {
        self.all_kings() & self.our_pieces::<C>()
    }

    pub fn pawns<C: Col>(&self) -> SquareSet {
        self.all_pawns() & self.our_pieces::<C>()
    }

    pub const fn occupied_co(&self, colour: Colour) -> SquareSet {
        self.colours[colour.index()]
    }

    pub const fn their_pieces<C: Col>(&self) -> SquareSet {
        self.colours[C::Opposite::COLOUR.index()]
    }

    pub const fn our_pieces<C: Col>(&self) -> SquareSet {
        self.colours[C::COLOUR.index()]
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
        self.colours[Colour::WHITE.index()] | self.colours[Colour::BLACK.index()]
    }

    pub fn knights<C: Col>(&self) -> SquareSet {
        self.all_knights() & self.our_pieces::<C>()
    }

    pub fn rooks<C: Col>(&self) -> SquareSet {
        self.all_rooks() & self.our_pieces::<C>()
    }

    pub fn bishops<C: Col>(&self) -> SquareSet {
        self.all_bishops() & self.our_pieces::<C>()
    }

    pub fn queens<C: Col>(&self) -> SquareSet {
        self.all_queens() & self.our_pieces::<C>()
    }

    pub const fn all_pawns(&self) -> SquareSet {
        self.pieces[PieceType::PAWN.index()]
    }

    pub const fn all_knights(&self) -> SquareSet {
        self.pieces[PieceType::KNIGHT.index()]
    }

    pub const fn all_bishops(&self) -> SquareSet {
        self.pieces[PieceType::BISHOP.index()]
    }

    pub const fn all_rooks(&self) -> SquareSet {
        self.pieces[PieceType::ROOK.index()]
    }

    pub const fn all_queens(&self) -> SquareSet {
        self.pieces[PieceType::QUEEN.index()]
    }

    pub const fn all_kings(&self) -> SquareSet {
        self.pieces[PieceType::KING.index()]
    }

    pub fn reset(&mut self) {
        *self = Self::NULL;
    }

    pub fn move_piece(&mut self, from: Square, to: Square, piece: Piece) {
        let from_to_bb = from.as_set() | to.as_set();
        self.pieces[piece.piece_type().index()] ^= from_to_bb;
        self.colours[piece.colour().index()] ^= from_to_bb;
    }

    pub fn set_piece_at(&mut self, sq: Square, piece: Piece) {
        let sq_bb = sq.as_set();
        self.pieces[piece.piece_type().index()] |= sq_bb;
        self.colours[piece.colour().index()] |= sq_bb;
    }

    pub fn clear_piece_at(&mut self, sq: Square, piece: Piece) {
        let sq_bb = sq.as_set();
        self.pieces[piece.piece_type().index()] &= !sq_bb;
        self.colours[piece.colour().index()] &= !sq_bb;
    }

    pub const fn any_pawns(&self) -> bool {
        self.all_pawns().non_empty()
    }

    pub const fn piece_bb(&self, piece: Piece) -> SquareSet {
        SquareSet::intersection(self.pieces[piece.piece_type().index()], self.colours[piece.colour().index()])
    }

    pub const fn of_type(&self, piece_type: PieceType) -> SquareSet {
        self.pieces[piece_type.index()]
    }

    pub fn all_attackers_to_sq(&self, sq: Square, occupied: SquareSet) -> SquareSet {
        let sq_bb = sq.as_set();
        let black_pawn_attackers = pawn_attacks::<White>(sq_bb) & self.pawns::<Black>();
        let white_pawn_attackers = pawn_attacks::<Black>(sq_bb) & self.pawns::<White>();
        let knight_attackers = knight_attacks(sq) & (self.all_knights());
        let diag_attackers = bishop_attacks(sq, occupied) & (self.all_bishops() | self.all_queens());
        let orth_attackers = rook_attacks(sq, occupied) & (self.all_rooks() | self.all_queens());
        let king_attackers = king_attacks(sq) & (self.all_kings());
        black_pawn_attackers
            | white_pawn_attackers
            | knight_attackers
            | diag_attackers
            | orth_attackers
            | king_attackers
    }

    pub fn piece_at(&self, sq: Square) -> Piece {
        let sq_bb = sq.as_set();
        let colour = if (self.our_pieces::<White>() & sq_bb).non_empty() {
            Colour::WHITE
        } else if (self.our_pieces::<Black>() & sq_bb).non_empty() {
            Colour::BLACK
        } else {
            return Piece::EMPTY;
        };
        for piece in PieceType::all() {
            if (self.pieces[piece.index()] & sq_bb).non_empty() {
                return Piece::new(colour, piece);
            }
        }
        panic!("Bit set in colour bitboard for {colour:?} but not in piece bitboards! square is {sq}");
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
    pub fn update_iter(&self, target: Self, mut callback: impl FnMut(FeatureUpdate, bool)) {
        for colour in Colour::all() {
            for piece_type in PieceType::all() {
                let piece = Piece::new(colour, piece_type);
                let source_bb = self.pieces[piece_type.index()] & self.colours[colour.index()];
                let target_bb = target.pieces[piece_type.index()] & target.colours[colour.index()];
                let added = target_bb & !source_bb;
                let removed = source_bb & !target_bb;
                for sq in added {
                    callback(FeatureUpdate { sq, piece }, true);
                }
                for sq in removed {
                    callback(FeatureUpdate { sq, piece }, false);
                }
            }
        }
    }

    pub fn visit_pieces(&self, mut callback: impl FnMut(Square, Piece)) {
        for colour in Colour::all() {
            for piece_type in PieceType::all() {
                let piece = Piece::new(colour, piece_type);
                let bb = self.pieces[piece_type.index()] & self.colours[colour.index()];
                for sq in bb {
                    callback(sq, piece);
                }
            }
        }
    }

    /// Returns true if the current position *would* be a draw by insufficient material,
    /// if there were no pawns on the board.
    pub const fn is_material_draw(&self) -> bool {
        if self.of_type(PieceType::ROOK).is_empty() && self.of_type(PieceType::QUEEN).is_empty() {
            if self.of_type(PieceType::BISHOP).is_empty() {
                if self.piece_bb(Piece::WN).count() < 3 && self.piece_bb(Piece::BN).count() < 3 {
                    return true;
                }
            } else if (self.of_type(PieceType::KNIGHT).is_empty()
                && self.piece_bb(Piece::WB).count().abs_diff(self.piece_bb(Piece::BB).count()) < 2)
                || SquareSet::union(self.piece_bb(Piece::WB), self.piece_bb(Piece::WN)).count() == 1
                    && SquareSet::union(self.piece_bb(Piece::BB), self.piece_bb(Piece::BN)).count() == 1
            {
                return true;
            }
        } else if self.of_type(PieceType::QUEEN).is_empty() {
            if self.piece_bb(Piece::WR).count() == 1 && self.piece_bb(Piece::BR).count() == 1 {
                if SquareSet::union(self.piece_bb(Piece::WN), self.piece_bb(Piece::WB)).count() < 2
                    && SquareSet::union(self.piece_bb(Piece::BN), self.piece_bb(Piece::BB)).count() < 2
                {
                    return true;
                }
            } else if self.piece_bb(Piece::WR).count() == 1 && self.piece_bb(Piece::BR).is_empty() {
                if SquareSet::union(self.piece_bb(Piece::WN), self.piece_bb(Piece::WB)).is_empty()
                    && (SquareSet::union(self.piece_bb(Piece::BN), self.piece_bb(Piece::BB)).count() == 1
                        || SquareSet::union(self.piece_bb(Piece::BN), self.piece_bb(Piece::BB)).count() == 2)
                {
                    return true;
                }
            } else if self.piece_bb(Piece::WR).is_empty()
                && self.piece_bb(Piece::BR).count() == 1
                && SquareSet::union(self.piece_bb(Piece::BN), self.piece_bb(Piece::BB)).is_empty()
                && (SquareSet::union(self.piece_bb(Piece::WN), self.piece_bb(Piece::WB)).count() == 1
                    || SquareSet::union(self.piece_bb(Piece::WN), self.piece_bb(Piece::WB)).count() == 2)
            {
                return true;
            }
        }
        false
    }
}

pub fn bishop_attacks(sq: Square, blockers: SquareSet) -> SquareSet {
    magic::get_diagonal_attacks(sq, blockers)
}
pub fn rook_attacks(sq: Square, blockers: SquareSet) -> SquareSet {
    magic::get_orthogonal_attacks(sq, blockers)
}
// pub fn queen_attacks(sq: Square, blockers: SquareSet) -> SquareSet {
//     magic::get_diagonal_attacks(sq, blockers) | magic::get_orthogonal_attacks(sq, blockers)
// }
pub fn knight_attacks(sq: Square) -> SquareSet {
    lookups::get_knight_attacks(sq)
}
pub fn king_attacks(sq: Square) -> SquareSet {
    lookups::get_king_attacks(sq)
}
pub fn pawn_attacks<C: Col>(bb: SquareSet) -> SquareSet {
    if C::WHITE {
        bb.north_east_one() | bb.north_west_one()
    } else {
        bb.south_east_one() | bb.south_west_one()
    }
}

pub fn attacks_by_type(pt: PieceType, sq: Square, blockers: SquareSet) -> SquareSet {
    match pt {
        PieceType::BISHOP => magic::get_diagonal_attacks(sq, blockers),
        PieceType::ROOK => magic::get_orthogonal_attacks(sq, blockers),
        PieceType::QUEEN => magic::get_diagonal_attacks(sq, blockers) | magic::get_orthogonal_attacks(sq, blockers),
        PieceType::KNIGHT => lookups::get_knight_attacks(sq),
        PieceType::KING => lookups::get_king_attacks(sq),
        _ => panic!("Invalid piece type: {pt:?}"),
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

impl Display for BitBoard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for rank in (0..=7).rev() {
            for file in 0..=7 {
                let sq = Square::from_rank_file(rank, file);
                let piece = self.piece_at(sq);
                let piece_char = piece.char();
                if let Some(symbol) = piece_char {
                    write!(f, " {symbol}")?;
                } else {
                    write!(f, " .")?;
                }
            }
            writeln!(f)?;
        }
        if self.any_bbs_overlapping() {
            writeln!(f, "WARNING: Some bitboards are overlapping")?;
        }
        Ok(())
    }
}
