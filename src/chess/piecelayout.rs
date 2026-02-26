use std::fmt::Display;

use crate::chess::{
    board::movegen::{
        self, RAY_BETWEEN, bishop_attacks, king_attacks, knight_attacks, pawn_attacks, rook_attacks,
    },
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
    pub fn occupied(&self) -> SquareSet {
        self.colours[Colour::White] | self.colours[Colour::Black]
    }

    pub fn empty(&self) -> SquareSet {
        !self.occupied()
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
        let black_pawn_attackers = pawn_attacks::<White>(sq_bb)
            & self.pieces[PieceType::Pawn]
            & self.colours[Colour::Black];
        let white_pawn_attackers = pawn_attacks::<Black>(sq_bb)
            & self.pieces[PieceType::Pawn]
            & self.colours[Colour::White];
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

    /// Determines if `sq` is attacked by `side`
    pub fn sq_attacked(&self, sq: Square, side: Colour) -> bool {
        match side {
            Colour::White => self.sq_attacked_by::<White>(sq),
            Colour::Black => self.sq_attacked_by::<Black>(sq),
        }
    }

    pub fn sq_attacked_by<C: Col>(&self, sq: Square) -> bool {
        use PieceType::{Bishop, King, Knight, Pawn, Queen, Rook};

        let attackers = self.colours[C::COLOUR];

        // pawns
        if pawn_attacks::<C>(self.pieces[Pawn] & attackers).contains_square(sq) {
            return true;
        }

        // knights
        if (attackers & self.pieces[Knight]) & movegen::knight_attacks(sq) != SquareSet::EMPTY {
            return true;
        }

        let blockers = attackers | self.colours[!C::COLOUR];

        // bishops, queens
        let diags = attackers & (self.pieces[Queen] | self.pieces[Bishop]);
        if diags & movegen::bishop_attacks(sq, blockers) != SquareSet::EMPTY {
            return true;
        }

        // rooks, queens
        let orthos = attackers & (self.pieces[Queen] | self.pieces[Rook]);
        if orthos & movegen::rook_attacks(sq, blockers) != SquareSet::EMPTY {
            return true;
        }

        // king
        if (attackers & self.pieces[King]) & movegen::king_attacks(sq) != SquareSet::EMPTY {
            return true;
        }

        false
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
        panic!(
            "Bit set in colour square-set for {colour:?} but not in piece square-sets! square is {sq}"
        );
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

    pub fn king_sq(&self, colour: Colour) -> Square {
        let king_bb = self.pieces[PieceType::King] & self.colours[colour];
        debug_assert_eq!(king_bb.count(), 1);
        king_bb.first().unwrap()
    }

    pub fn generate_pinned(&self, side: Colour) -> (SquareSet, SquareSet) {
        use PieceType::{Bishop, Queen, Rook};

        let mut pinned = SquareSet::EMPTY;
        let mut pinners = SquareSet::EMPTY;

        let king = self.king_sq(side);

        let us = self.colours[side];
        let them = self.colours[!side];

        let their_diags = (self.pieces[Queen] | self.pieces[Bishop]) & them;
        let their_orthos = (self.pieces[Queen] | self.pieces[Rook]) & them;

        let potential_attackers =
            bishop_attacks(king, them) & their_diags | rook_attacks(king, them) & their_orthos;

        for potential_attacker in potential_attackers {
            let maybe_pinned = us & RAY_BETWEEN[king][potential_attacker];
            if maybe_pinned.one() {
                pinned |= maybe_pinned;
                pinners |= potential_attacker.as_set();
            }
        }

        (pinned, pinners)
    }

    pub fn generate_threats(&self, side: Colour) -> Threats {
        let mut checkers = SquareSet::EMPTY;

        let us = self.colours[side];
        let them = self.colours[!side];
        let their_pawns = self.pieces[PieceType::Pawn] & them;
        let their_knights = self.pieces[PieceType::Knight] & them;
        let their_bishops = self.pieces[PieceType::Bishop] & them;
        let their_rooks = self.pieces[PieceType::Rook] & them;
        let their_queens = self.pieces[PieceType::Queen] & them;
        let their_king = (self.pieces[PieceType::King] & them).first().unwrap();
        let blockers = us | them;

        // compute threats
        let leq_pawn = match side {
            Colour::White => their_pawns.south_east_one() | their_pawns.south_west_one(),
            Colour::Black => their_pawns.north_east_one() | their_pawns.north_west_one(),
        };

        let mut leq_minor = leq_pawn;
        for sq in their_knights {
            leq_minor |= knight_attacks(sq);
        }
        for sq in their_bishops {
            leq_minor |= bishop_attacks(sq, blockers);
        }
        let mut leq_rook = leq_minor;
        for sq in their_rooks {
            leq_rook |= rook_attacks(sq, blockers);
        }
        let mut all_threats = leq_rook;
        for sq in their_queens {
            all_threats |= bishop_attacks(sq, blockers);
            all_threats |= rook_attacks(sq, blockers);
        }
        all_threats |= king_attacks(their_king);

        // compute checkers
        let our_king_bb = us & self.pieces[PieceType::King];
        let our_king_sq = our_king_bb.first().unwrap();
        let backwards_from_king = match side {
            Colour::White => our_king_bb.north_east_one() | our_king_bb.north_west_one(),
            Colour::Black => our_king_bb.south_east_one() | our_king_bb.south_west_one(),
        };
        checkers |= backwards_from_king & their_pawns;
        let knight_attacks = knight_attacks(our_king_sq);
        checkers |= knight_attacks & their_knights;
        let diag_attacks = bishop_attacks(our_king_sq, blockers);
        checkers |= diag_attacks & (their_bishops | their_queens);
        let ortho_attacks = rook_attacks(our_king_sq, blockers);
        checkers |= ortho_attacks & (their_rooks | their_queens);

        Threats {
            all: all_threats,
            leq_pawn,
            leq_minor,
            leq_rook,
            checkers,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub struct Threats {
    pub all: SquareSet,
    pub leq_pawn: SquareSet,
    pub leq_minor: SquareSet,
    pub leq_rook: SquareSet,
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
