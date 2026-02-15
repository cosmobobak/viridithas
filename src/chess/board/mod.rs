pub mod movegen;
mod san;
pub mod validation;

use std::{
    fmt::{self, Debug, Display, Formatter, Write},
    sync::atomic::Ordering,
};

use arrayvec::ArrayVec;
use movegen::{MAX_POSITION_MOVES, RAY_BETWEEN, RAY_FULL};

use crate::{
    chess::{
        CHESS960,
        board::movegen::{MoveList, bishop_attacks, pawn_attacks, pawn_attacks_by, rook_attacks},
        chessmove::{Move, MoveFlags},
        fen::Fen,
        piece::{Black, Col, Colour, Piece, PieceType, White},
        quick::Quick,
        squareset::SquareSet,
        types::{CastlingRights, CheckState, File, Rank, Square, State},
    },
    cuckoo,
    errors::MoveParseError,
    lookups::{CASTLE_KEYS, EP_KEYS, HM_CLOCK_KEYS, PIECE_KEYS, SIDE_KEY},
    nnue::network::{FeatureUpdate, MovedPiece, NNUEState, UpdateBuffer},
    search::pv::PVariation,
};

#[derive(PartialEq, Eq, Clone)]
pub struct Board {
    /// Copyable state for the board.
    pub(crate) state: State,
    /// The side to move.
    side: Colour,
    /// The number of half moves made since the start of the game.
    ply: usize,

    height: usize,
    history: Vec<State>,
}

impl Debug for Board {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Board")
            .field("piece_array", &self.state.mailbox)
            .field("side", &self.side)
            .field("ep_sq", &self.state.ep_square)
            .field("fifty_move_counter", &self.state.fifty_move_counter)
            .field("height", &self.height)
            .field("ply", &self.ply)
            .field("key", &self.state.keys.zobrist)
            .field("threats", &self.state.threats)
            .field("castle_perm", &self.state.castle_perm)
            .finish_non_exhaustive()
    }
}

impl Board {
    pub const STARTING_FEN: &'static str =
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    pub const STARTING_FEN_960: &'static str =
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w AHah - 0 1";

    pub fn empty() -> Self {
        let mut out = Self {
            state: State::default(),
            side: Colour::White,
            height: 0,
            ply: 0,
            history: Vec::new(),
        };
        out.reset();
        out
    }

    pub const fn ep_sq(&self) -> Option<Square> {
        self.state.ep_square
    }

    pub fn history(&self) -> &[State] {
        &self.history
    }

    #[cfg(feature = "datagen")]
    pub fn ep_sq_mut(&mut self) -> &mut Option<Square> {
        &mut self.state.ep_square
    }

    #[cfg(feature = "datagen")]
    pub fn turn_mut(&mut self) -> &mut Colour {
        &mut self.side
    }

    #[cfg(feature = "datagen")]
    pub fn halfmove_clock_mut(&mut self) -> &mut u8 {
        &mut self.state.fifty_move_counter
    }

    #[cfg(feature = "datagen")]
    pub fn set_fullmove_clock(&mut self, fullmove_clock: u16) {
        self.ply = (fullmove_clock as usize - 1) * 2 + usize::from(self.side == Colour::Black);
    }

    pub const fn ply(&self) -> usize {
        self.ply
    }

    pub fn in_check(&self) -> bool {
        self.state.threats.checkers != SquareSet::EMPTY
    }

    pub fn zero_height(&mut self) {
        self.height = 0;
    }

    pub const fn height(&self) -> usize {
        self.height
    }

    pub const fn turn(&self) -> Colour {
        self.side
    }

    pub const fn castling_rights(&self) -> CastlingRights {
        self.state.castle_perm
    }

    #[cfg(feature = "datagen")]
    pub fn castling_rights_mut(&mut self) -> &mut CastlingRights {
        &mut self.state.castle_perm
    }

    #[cfg(feature = "datagen")]
    pub fn regenerate_zobrist(&mut self) {
        self.state.keys = self.state.generate_pos_keys(self.side);
    }

    #[cfg(feature = "datagen")]
    pub fn regenerate_threats(&mut self) {
        self.state.threats = self.state.bbs.generate_threats(self.side);
    }

    pub fn reset(&mut self) {
        self.state = State::default();
        self.side = Colour::White;
        self.height = 0;
        self.ply = 0;
        self.history.clear();
    }

    pub fn set_frc_idx(&mut self, scharnagl: usize) {
        #![allow(clippy::cast_possible_truncation)]
        assert!(scharnagl < 960, "scharnagl index out of range");
        let backrank = Self::get_scharnagl_backrank(scharnagl);
        self.reset();
        for (&piece_type, file) in backrank.iter().zip(File::all()) {
            let sq = Square::from_rank_file(Rank::One, file);
            self.add_piece(sq, Piece::new(Colour::White, piece_type));
        }
        for file in File::all() {
            // add pawns
            let sq = Square::from_rank_file(Rank::Two, file);
            self.add_piece(sq, Piece::new(Colour::White, PieceType::Pawn));
        }
        for (&piece_type, file) in backrank.iter().zip(File::all()) {
            let sq = Square::from_rank_file(Rank::Eight, file);
            self.add_piece(sq, Piece::new(Colour::Black, piece_type));
        }
        for file in File::all() {
            // add pawns
            let sq = Square::from_rank_file(Rank::Seven, file);
            self.add_piece(sq, Piece::new(Colour::Black, PieceType::Pawn));
        }
        let mut rook_indices = backrank.iter().enumerate().filter_map(|(i, &piece)| {
            if piece == PieceType::Rook {
                Some(i)
            } else {
                None
            }
        });
        let queenside_file = rook_indices.next().unwrap();
        let kingside_file = rook_indices.next().unwrap();
        self.state.castle_perm = CastlingRights::new(
            Some(File::from_index(kingside_file as u8).unwrap()),
            Some(File::from_index(queenside_file as u8).unwrap()),
            Some(File::from_index(kingside_file as u8).unwrap()),
            Some(File::from_index(queenside_file as u8).unwrap()),
        );
        self.state.keys = self.state.generate_pos_keys(self.side);
        self.state.threats = self.state.bbs.generate_threats(self.side);
        self.state.pinned = [
            self.state.bbs.generate_pinned(Colour::White),
            self.state.bbs.generate_pinned(Colour::Black),
        ];
    }

    pub fn set_dfrc_idx(&mut self, scharnagl: usize) {
        #![allow(clippy::cast_possible_truncation)]
        assert!(scharnagl < 960 * 960, "double scharnagl index out of range");
        let white_backrank = Self::get_scharnagl_backrank(scharnagl % 960);
        let black_backrank = Self::get_scharnagl_backrank(scharnagl / 960);
        self.reset();
        for (&piece_type, file) in white_backrank.iter().zip(File::all()) {
            let sq = Square::from_rank_file(Rank::One, file);
            self.add_piece(sq, Piece::new(Colour::White, piece_type));
        }
        for file in File::all() {
            // add pawns
            let sq = Square::from_rank_file(Rank::Two, file);
            self.add_piece(sq, Piece::new(Colour::White, PieceType::Pawn));
        }
        for (&piece_type, file) in black_backrank.iter().zip(File::all()) {
            let sq = Square::from_rank_file(Rank::Eight, file);
            self.add_piece(sq, Piece::new(Colour::Black, piece_type));
        }
        for file in File::all() {
            // add pawns
            let sq = Square::from_rank_file(Rank::Seven, file);
            self.add_piece(sq, Piece::new(Colour::Black, PieceType::Pawn));
        }
        let mut white_rook_indices = white_backrank.iter().enumerate().filter_map(|(i, &piece)| {
            if piece == PieceType::Rook {
                Some(i)
            } else {
                None
            }
        });
        let white_queenside_file = white_rook_indices.next().unwrap();
        let white_kingside_file = white_rook_indices.next().unwrap();
        let mut black_rook_indices = black_backrank.iter().enumerate().filter_map(|(i, &piece)| {
            if piece == PieceType::Rook {
                Some(i)
            } else {
                None
            }
        });
        let black_queenside_file = black_rook_indices.next().unwrap();
        let black_kingside_file = black_rook_indices.next().unwrap();
        self.state.castle_perm = CastlingRights::new(
            Some(File::from_index(white_kingside_file as u8).unwrap()),
            Some(File::from_index(white_queenside_file as u8).unwrap()),
            Some(File::from_index(black_kingside_file as u8).unwrap()),
            Some(File::from_index(black_queenside_file as u8).unwrap()),
        );
        self.state.keys = self.state.generate_pos_keys(self.side);
        self.state.threats = self.state.bbs.generate_threats(self.side);
        self.state.pinned = [
            self.state.bbs.generate_pinned(Colour::White),
            self.state.bbs.generate_pinned(Colour::Black),
        ];
    }

    pub fn get_scharnagl_backrank(scharnagl: usize) -> [PieceType; 8] {
        // White's starting array can be derived from its number N (0 ... 959) as follows (https://en.wikipedia.org/wiki/Fischer_random_chess_numbering_scheme#Direct_derivation):
        // A. Divide N by 4, yielding quotient N2 and remainder B1. Place a Bishop upon the bright square corresponding to B1 (0=b, 1=d, 2=f, 3=h).
        // B. Divide N2 by 4 again, yielding quotient N3 and remainder B2. Place a second Bishop upon the dark square corresponding to B2 (0=a, 1=c, 2=e, 3=g).
        // C. Divide N3 by 6, yielding quotient N4 and remainder Q. Place the Queen according to Q, where 0 is the first free square starting from a, 1 is the second, etc.
        // D. N4 will be a single digit, 0 ... 9. Ignoring Bishops and Queen, find the positions of two Knights within the remaining five spaces.
        //    Place the Knights according to its value by consulting the following N5N table:
        // DIGIT | Knight Positioning
        //   0   | N N - - -
        //   1   | N - N - -
        //   2   | N - - N -
        //   3   | N - - - N
        //   4   | - N N - -
        //   5   | - N - N -
        //   6   | - N - - N
        //   7   | - - N N -
        //   8   | - - N - N
        //   9   | - - - N N
        // E. There are three blank squares remaining; place a Rook in each of the outer two and the King in the middle one.
        let mut out = [None; 8];
        let n = scharnagl;
        let (n2, b1) = (n / 4, n % 4);
        match b1 {
            0 => out[File::B] = Some(PieceType::Bishop),
            1 => out[File::D] = Some(PieceType::Bishop),
            2 => out[File::F] = Some(PieceType::Bishop),
            3 => out[File::H] = Some(PieceType::Bishop),
            _ => unreachable!(),
        }
        let (n3, b2) = (n2 / 4, n2 % 4);
        match b2 {
            0 => out[File::A] = Some(PieceType::Bishop),
            1 => out[File::C] = Some(PieceType::Bishop),
            2 => out[File::E] = Some(PieceType::Bishop),
            3 => out[File::G] = Some(PieceType::Bishop),
            _ => unreachable!(),
        }
        let (n4, mut q) = (n3 / 6, n3 % 6);
        for (idx, &piece) in out.iter().enumerate() {
            if piece.is_none() {
                if q == 0 {
                    out[idx] = Some(PieceType::Queen);
                    break;
                }
                q -= 1;
            }
        }
        let remaining_slots = out.iter_mut().filter(|piece| piece.is_none());
        let selection = match n4 {
            0 => [0, 1],
            1 => [0, 2],
            2 => [0, 3],
            3 => [0, 4],
            4 => [1, 2],
            5 => [1, 3],
            6 => [1, 4],
            7 => [2, 3],
            8 => [2, 4],
            9 => [3, 4],
            _ => unreachable!(),
        };
        for (i, slot) in remaining_slots.enumerate() {
            if i == selection[0] || i == selection[1] {
                *slot = Some(PieceType::Knight);
            }
        }

        out.iter_mut()
            .filter(|piece| piece.is_none())
            .zip([PieceType::Rook, PieceType::King, PieceType::Rook])
            .for_each(|(slot, piece)| *slot = Some(piece));

        out.map(Option::unwrap)
    }

    pub fn set_from_fen(&mut self, fen: &Fen) {
        self.reset();

        self.state.bbs = fen.board;

        for sq in Square::all() {
            self.state.mailbox[sq] = fen.board.piece_at(sq);
        }

        self.side = fen.turn;
        self.state.castle_perm = fen.castling;
        self.state.ep_square = fen.ep;
        self.state.fifty_move_counter = fen.halfmove;
        self.ply = (fen.fullmove.get() - 1) * 2;
        if self.side == Colour::Black {
            self.ply += 1;
        }

        // generate derived state
        self.state.keys = self.state.generate_pos_keys(self.side);
        self.state.threats = self.state.bbs.generate_threats(self.side);
        self.state.pinned = [
            self.state.bbs.generate_pinned(Colour::White),
            self.state.bbs.generate_pinned(Colour::Black),
        ];

        // clear illegal en-passant squares:
        let can_attack = self
            .state
            .ep_square
            .into_iter()
            .flat_map(|sq| {
                let sources = pawn_attacks_by(sq.as_set(), !self.side);
                let our_pawns =
                    self.state.bbs.colours[self.side] & self.state.bbs.pieces[PieceType::Pawn];

                (sources & our_pawns).into_iter().zip(std::iter::repeat(sq))
            })
            .map(|(from, to)| Move::new_with_flags(from, to, MoveFlags::EnPassant))
            .any(|mv| self.is_pseudo_legal(mv) && self.is_legal(mv));

        if !can_attack {
            self.state.ep_square = None;
            self.state.keys = self.state.generate_pos_keys(self.side);
        }
    }

    pub fn set_startpos(&mut self) {
        let starting_fen = if CHESS960.load(Ordering::SeqCst) {
            Self::STARTING_FEN_960
        } else {
            Self::STARTING_FEN
        };
        let fen = Fen::parse(starting_fen).expect("STARTING_FEN is broken");
        self.set_from_fen(&fen);
    }

    pub fn set_from_quick(&mut self, quick: &Quick) {
        self.reset();

        self.state.bbs = quick.board;

        for sq in Square::all() {
            self.state.mailbox[sq] = quick.board.piece_at(sq);
        }

        self.side = quick.turn;
        self.state.castle_perm = quick.rights;
        if self.side == Colour::Black {
            self.ply += 1;
        }

        self.state.keys = self.state.generate_pos_keys(self.side);
        self.state.threats = self.state.bbs.generate_threats(self.side);
        self.state.pinned = [
            self.state.bbs.generate_pinned(Colour::White),
            self.state.bbs.generate_pinned(Colour::Black),
        ];
    }

    #[cfg(test)]
    pub fn from_fen(fen: &str) -> Result<Self, crate::errors::FenParseError> {
        let parsed = Fen::parse_relaxed(fen)?;
        let mut out = Self::empty();
        out.set_from_fen(&parsed);
        Ok(out)
    }

    #[cfg(test)]
    pub fn from_frc_idx(scharnagl: usize) -> Self {
        let mut out = Self::empty();
        out.set_frc_idx(scharnagl);
        out
    }

    #[cfg(test)]
    pub fn from_dfrc_idx(scharnagl: usize) -> Self {
        let mut out = Self::empty();
        out.set_dfrc_idx(scharnagl);
        out
    }

    #[cfg(test)]
    pub fn from_quick(record: &str) -> Result<Self, crate::errors::QuickParseError> {
        let parsed = Quick::parse(record.trim_ascii())?;
        let mut out = Self::empty();
        out.set_from_quick(&parsed);
        Ok(out)
    }

    /// Determines if `sq` is attacked by `side`
    pub fn sq_attacked(&self, sq: Square, side: Colour) -> bool {
        match side {
            Colour::White => self.sq_attacked_by::<White>(sq),
            Colour::Black => self.sq_attacked_by::<Black>(sq),
        }
    }

    pub fn sq_attacked_by<C: Col>(&self, sq: Square) -> bool {
        // we remove this check because the board actually *can*
        // be in an inconsistent state when we call this, as it's
        // used to determine if a move is legal, and we'd like to
        // only do a lot of the make_move work *after* we've
        // determined that the move is legal.
        // #[cfg(debug_assertions)]
        // self.check_validity();

        if C::WHITE == (self.side == Colour::Black) {
            return self.state.threats.all.contains_square(sq);
        }

        self.state.bbs.sq_attacked_by::<C>(sq)
    }

    /// Checks whether a move is pseudo-legal.
    /// This means that it is a legal move, except for the fact that it might leave the king in check.
    pub fn is_pseudo_legal(&self, m: Move) -> bool {
        if m.is_castle() {
            return self.is_pseudo_legal_castling(m);
        }

        let from = m.from();
        let to = m.to();

        let moved_piece = self.state.mailbox[from];
        let captured_piece = self.state.mailbox[to];

        let Some(moved_piece) = moved_piece else {
            return false;
        };

        if moved_piece.colour() != self.side {
            return false;
        }

        if captured_piece.is_some_and(|piece| piece.colour() == self.side) {
            return false;
        }

        if captured_piece.is_some()
            && moved_piece.piece_type() == PieceType::Pawn
            && from.file() == to.file()
        {
            return false;
        }

        if moved_piece.piece_type() == PieceType::Pawn {
            let should_be_promoting = to > Square::H7 || to < Square::A2;
            if should_be_promoting && !m.is_promo() {
                return false;
            }
            if m.is_ep() {
                return Some(to) == self.state.ep_square;
            } else if (SquareSet::RANK_4 | SquareSet::RANK_5).contains_square(to)
                && (SquareSet::RANK_2 | SquareSet::RANK_7).contains_square(from)
            {
                if from.relative_to(self.side).rank() != Rank::Two {
                    return false;
                }
                let Some(one_forward) = from.pawn_push(self.side) else {
                    return false;
                };
                return self.state.mailbox[one_forward].is_none()
                    && Some(to) == one_forward.pawn_push(self.side);
            } else if captured_piece.is_none() {
                return Some(to) == from.pawn_push(self.side);
            }
            // pawn capture
            return match self.side {
                Colour::White => pawn_attacks::<White>(from.as_set()).contains_square(to),
                Colour::Black => pawn_attacks::<Black>(from.as_set()).contains_square(to),
            };
        }

        // not a pawn move, but is somehow ep/promo?
        if m.is_ep() || m.is_promo() {
            return false;
        }

        if moved_piece.piece_type() == PieceType::King && self.state.threats.all.contains_square(to)
        {
            return false;
        }

        movegen::attacks_by_type(moved_piece.piece_type(), from, self.state.bbs.occupied())
            .contains_square(to)
    }

    pub fn is_pseudo_legal_castling(&self, m: Move) -> bool {
        // illegal if:
        // - we're not moving the king
        // - we're not doing everything on the home rank
        // - we don't have castling rights on the target square
        // - we're in check
        // - there are pieces between the king and the rook
        // - the king passes through a square that is attacked by the opponent
        // - the king ends up in check (not checked here)
        let Some(moved) = self.state.mailbox[m.from()] else {
            return false;
        };
        if moved.piece_type() != PieceType::King {
            return false;
        }
        let home_rank = match self.side {
            Colour::White => SquareSet::RANK_1,
            Colour::Black => SquareSet::RANK_8,
        };
        if !home_rank.contains_square(m.to()) {
            return false;
        }
        if !home_rank.contains_square(m.from()) {
            return false;
        }
        let (king_dst, rook_dst) = if m.to() > m.from() {
            // kingside castling.
            if self.state.castle_perm.kingside(self.side) != Some(m.to().file()) {
                // the to-square doesn't match the castling rights
                // (it goes to the wrong place, or the rights don't exist)
                return false;
            }
            (
                Square::G1.relative_to(self.side),
                Square::F1.relative_to(self.side),
            )
        } else {
            // queenside castling.
            if self.state.castle_perm.queenside(self.side) != Some(m.to().file()) {
                // the to-square doesn't match the castling rights
                // (it goes to the wrong place, or the rights don't exist)
                return false;
            }
            (
                Square::C1.relative_to(self.side),
                Square::D1.relative_to(self.side),
            )
        };

        // king_path is the path the king takes to get to its destination.
        let king_path = RAY_BETWEEN[m.from()][king_dst];
        // rook_path is the path the rook takes to get to its destination.
        let rook_path = RAY_BETWEEN[m.from()][m.to()];
        // castle_occ is the occupancy that "counts" for castling.
        let castle_occ = self.state.bbs.occupied() ^ m.from().as_set() ^ m.to().as_set();

        castle_occ & (king_path | rook_path | king_dst.as_set() | rook_dst.as_set())
            == SquareSet::EMPTY
            && !self.any_attacked(king_path | m.from().as_set(), self.side.flip())
    }

    /// Checks whether a given pseudo-legal move is legal in the current position.
    pub fn is_legal(&self, m: Move) -> bool {
        debug_assert!(
            self.is_pseudo_legal(m),
            "got {} in position: {}",
            m.display(false),
            self
        );

        let turn = self.turn();
        let bbs = &self.state.bbs;

        let from = m.from();
        let to = m.to();

        let us = bbs.colours[turn];
        let our_king_bb = bbs.pieces[PieceType::King] & us;
        let king = our_king_bb.first().unwrap();

        let them = bbs.colours[!turn];
        let their_queens = bbs.pieces[PieceType::Queen] & them;
        let their_bishops = bbs.pieces[PieceType::Bishop] & them;
        let their_rooks = bbs.pieces[PieceType::Rook] & them;

        if m.is_castle() {
            let king_to = m.history_to_square();
            // TODO: determine necessity of first conditional component
            return !(self.state.threats.all.contains_square(king_to)
                || CHESS960.load(Ordering::Relaxed)
                    && self.state.pinned[turn].contains_square(to));
        } else if m.is_ep() {
            let rank = to.rank();
            let file = to.file();

            let rank = if rank == Rank::Three {
                Rank::Four
            } else {
                Rank::Five
            };

            let cap_sq = Square::from_rank_file(rank, file);

            let occ_after = bbs.occupied() ^ to.as_set() ^ from.as_set() ^ cap_sq.as_set();

            return bishop_attacks(king, occ_after) & (their_queens | their_bishops)
                == SquareSet::EMPTY
                && rook_attacks(king, occ_after) & (their_queens | their_rooks)
                    == SquareSet::EMPTY;
        }

        let moving = self.state.mailbox[from].unwrap();

        if moving.piece_type() == PieceType::King {
            let without_king = bbs.occupied() ^ our_king_bb;

            let diags = their_queens | their_bishops;
            let orthos = their_queens | their_rooks;
            let moving_into_check = bishop_attacks(to, without_king) & diags != SquareSet::EMPTY
                || rook_attacks(to, without_king) & orthos != SquareSet::EMPTY;
            return !moving_into_check;
        }

        // moving anything other than the king
        // is illegal when in double-check.
        if self.state.threats.checkers.many() {
            return false;
        }

        // if your piece is pinned, you can only
        // move it along the direction that it is
        // pinned in.
        if self.state.pinned[turn].contains_square(from)
            && !RAY_FULL[from][to].contains_square(king)
        {
            return false;
        }

        // not in check and not moving the king? a-ok.
        if self.state.threats.checkers == SquareSet::EMPTY {
            return true;
        }

        // single checker, you have to be
        // capturing it or blocking the check.
        let checker = self.state.threats.checkers.first().unwrap();
        (RAY_BETWEEN[king][checker] | self.state.threats.checkers).contains_square(to)
    }

    pub fn any_attacked(&self, squares: SquareSet, by: Colour) -> bool {
        if by == self.side.flip() {
            squares & self.state.threats.all != SquareSet::EMPTY
        } else {
            for sq in squares {
                if self.sq_attacked(sq, by) {
                    return true;
                }
            }
            false
        }
    }

    pub fn add_piece(&mut self, sq: Square, piece: Piece) {
        self.state.bbs.set_piece_at(sq, piece);
        self.state.mailbox[sq] = Some(piece);
    }

    /// Gets the piece that will be captured by the given move.
    pub fn captured_piece(&self, m: Move) -> Option<Piece> {
        if m.is_castle() {
            return None;
        }
        let idx = m.to();
        self.state.mailbox[idx]
    }

    /// Determines whether this move would be a capture in the current position.
    pub fn is_capture(&self, m: Move) -> bool {
        self.captured_piece(m).is_some()
    }

    /// Determines whether this move would be tactical in the current position.
    pub fn is_tactical(&self, m: Move) -> bool {
        m.is_promo() || m.is_ep() || self.is_capture(m)
    }

    pub fn make_move_simple(&mut self, m: Move) {
        self.make_move_base(m, &mut UpdateBuffer::default());
    }

    #[allow(clippy::cognitive_complexity, clippy::too_many_lines)]
    pub fn make_move_base(&mut self, m: Move, update_buffer: &mut UpdateBuffer) {
        debug_assert!(self.is_pseudo_legal(m));
        debug_assert!(self.is_legal(m));

        #[cfg(debug_assertions)]
        self.check_validity();

        self.history.push(self.state.clone());

        let from = m.from();
        let mut to = m.to();
        let castle = m.is_castle();
        let side = self.side;
        let piece = self.state.mailbox[from].unwrap();
        let captured = if castle { None } else { self.state.mailbox[to] };
        let mut castling_perm = self.state.castle_perm;

        // from, to, and piece are valid unless this is a castling move,
        // as castling is encoded as king-captures-rook.
        // we sort out castling in a branch later, dw about it.
        if !castle {
            if m.is_promo() {
                // just remove the source piece, as a different piece will be arriving here
                update_buffer.clear_piece(from, piece);
            } else {
                update_buffer.move_piece(from, to, piece);
            }
        }

        if m.is_ep() {
            let clear_at = match side {
                Colour::White => to.sub(8),
                Colour::Black => to.add(8),
            }
            .unwrap();
            let to_clear = Piece::new(side.flip(), PieceType::Pawn);
            self.state.bbs.clear_piece_at(clear_at, to_clear);
            update_buffer.clear_piece(clear_at, to_clear);
        } else if castle {
            self.state.bbs.clear_piece_at(from, piece);
            let to_file = Some(to.file());
            let rook_from = to;
            let rook_to = if to_file == castling_perm.kingside(side) {
                to = Square::G1.relative_to(side);
                Square::F1.relative_to(side)
            } else if to_file == castling_perm.queenside(side) {
                to = Square::C1.relative_to(side);
                Square::D1.relative_to(side)
            } else {
                unreachable!()
            };
            if from != to {
                update_buffer.move_piece(from, to, piece);
            }
            if rook_from != rook_to {
                let rook = Piece::new(side, PieceType::Rook);
                self.state.bbs.move_piece(rook_from, rook_to, rook);
                update_buffer.move_piece(rook_from, rook_to, rook);
            }
        }

        self.state.fifty_move_counter += 1;

        if let Some(captured) = captured {
            self.state.fifty_move_counter = 0;
            self.state.bbs.clear_piece_at(to, captured);
            update_buffer.clear_piece(to, captured);
        }

        if let Some(ep_sq) = self.state.ep_square {
            self.state.keys.zobrist ^= EP_KEYS[ep_sq];
        }
        self.state.ep_square = None;
        if piece.piece_type() == PieceType::Pawn {
            self.state.fifty_move_counter = 0;
            if m.is_double_pawn_push_ranks()
                && (m.to().as_set().west_one() | m.to().as_set().east_one())
                    & self.state.bbs.pieces[PieceType::Pawn]
                    & self.state.bbs.colours[side.flip()]
                    != SquareSet::EMPTY
            {
                if side == Colour::White {
                    self.state.ep_square = from.add(8);
                    debug_assert!(self.state.ep_square.unwrap().rank() == Rank::Three);
                } else {
                    self.state.ep_square = from.sub(8);
                    debug_assert!(self.state.ep_square.unwrap().rank() == Rank::Six);
                }
            }
        }
        if let Some(ep_sq) = self.state.ep_square {
            self.state.keys.zobrist ^= EP_KEYS[ep_sq];
        }

        if let Some(promo) = m.promotion_type() {
            let promo = Piece::new(side, promo);
            debug_assert!(promo.piece_type().legal_promo());
            self.state.bbs.clear_piece_at(from, piece);
            self.state.bbs.set_piece_at(to, promo);
            update_buffer.add_piece(to, promo);
        } else if castle {
            self.state.bbs.set_piece_at(to, piece); // stupid hack for piece-swapping
        } else {
            self.state.bbs.move_piece(from, to, piece);
        }

        self.side = self.side.flip();

        // hash out the castling to insert it again after updating rights.
        self.state.keys.zobrist ^= CASTLE_KEYS[castling_perm.hashkey_index()];
        // update castling rights
        if piece == Piece::WR && from.rank() == Rank::One {
            if Some(from.file()) == castling_perm.kingside(Colour::White) {
                castling_perm.clear_side::<true, White>();
            } else if Some(from.file()) == castling_perm.queenside(Colour::White) {
                castling_perm.clear_side::<false, White>();
            }
        } else if piece == Piece::BR && from.rank() == Rank::Eight {
            if Some(from.file()) == castling_perm.kingside(Colour::Black) {
                castling_perm.clear_side::<true, Black>();
            } else if Some(from.file()) == castling_perm.queenside(Colour::Black) {
                castling_perm.clear_side::<false, Black>();
            }
        } else if piece == Piece::WK {
            castling_perm.clear::<White>();
        } else if piece == Piece::BK {
            castling_perm.clear::<Black>();
        }
        if to.rank() == Rank::One {
            castling_perm.remove::<White>(to.file());
        } else if to.rank() == Rank::Eight {
            castling_perm.remove::<Black>(to.file());
        }
        self.state.keys.zobrist ^= CASTLE_KEYS[castling_perm.hashkey_index()];
        self.state.castle_perm = castling_perm;

        // apply all the updates to the zobrist hash
        self.state.keys.zobrist ^= SIDE_KEY;
        for &FeatureUpdate { sq, piece } in update_buffer.subs() {
            self.state.mailbox[sq] = None;
            let piece_key = PIECE_KEYS[piece][sq];
            self.state.keys.zobrist ^= piece_key;
            if piece.piece_type() == PieceType::Pawn {
                self.state.keys.pawn ^= piece_key;
            } else {
                self.state.keys.non_pawn[piece.colour()] ^= piece_key;
                if piece.piece_type() == PieceType::King {
                    self.state.keys.major ^= piece_key;
                    self.state.keys.minor ^= piece_key;
                } else if matches!(piece.piece_type(), PieceType::Queen | PieceType::Rook) {
                    self.state.keys.major ^= piece_key;
                } else {
                    self.state.keys.minor ^= piece_key;
                }
            }
        }
        for &FeatureUpdate { sq, piece } in update_buffer.adds() {
            self.state.mailbox[sq] = Some(piece);
            let piece_key = PIECE_KEYS[piece][sq];
            self.state.keys.zobrist ^= piece_key;
            if piece.piece_type() == PieceType::Pawn {
                self.state.keys.pawn ^= piece_key;
            } else {
                self.state.keys.non_pawn[piece.colour()] ^= piece_key;
                if piece.piece_type() == PieceType::King {
                    self.state.keys.major ^= piece_key;
                    self.state.keys.minor ^= piece_key;
                } else if matches!(piece.piece_type(), PieceType::Queen | PieceType::Rook) {
                    self.state.keys.major ^= piece_key;
                } else {
                    self.state.keys.minor ^= piece_key;
                }
            }
        }

        self.ply += 1;
        self.height += 1;

        self.state.threats = self.state.bbs.generate_threats(self.side);
        self.state.pinned = [
            self.state.bbs.generate_pinned(Colour::White),
            self.state.bbs.generate_pinned(Colour::Black),
        ];

        #[cfg(debug_assertions)]
        self.check_validity();
    }

    pub fn unmake_move_base(&mut self) {
        // we remove this check because the board actually *can*
        // be in an inconsistent state when we call this, as we
        // may be unmaking a move that was determined to be
        // illegal, and as such the full make_move hasn't been
        // run yet.
        // #[cfg(debug_assertions)]
        // self.check_validity();

        self.height -= 1;
        self.ply -= 1;
        self.side = self.side.flip();
        self.state = self.history.pop().expect("No move to unmake!");

        #[cfg(debug_assertions)]
        self.check_validity();
    }

    pub fn make_nullmove(&mut self) {
        #[cfg(debug_assertions)]
        self.check_validity();
        debug_assert!(!self.in_check());

        self.history.push(self.state.clone());

        let mut key = self.state.keys.zobrist;
        if let Some(ep_sq) = self.state.ep_square {
            key ^= EP_KEYS[ep_sq];
        }
        key ^= SIDE_KEY;
        self.state.keys.zobrist = key;

        self.state.ep_square = None;
        self.side = self.side.flip();
        self.ply += 1;
        self.height += 1;

        self.state.threats = self.state.bbs.generate_threats(self.side);

        #[cfg(debug_assertions)]
        self.check_validity();
    }

    pub fn unmake_nullmove(&mut self) {
        #[cfg(debug_assertions)]
        self.check_validity();

        self.height -= 1;
        self.ply -= 1;
        self.side = self.side.flip();

        let State {
            ep_square,
            threats,
            keys,
            pinned,
            ..
        } = self.history.last().expect("No move to unmake!");

        self.state.ep_square = *ep_square;
        self.state.threats = *threats;
        self.state.pinned = *pinned;
        self.state.keys.zobrist = keys.zobrist;

        self.history.pop();

        #[cfg(debug_assertions)]
        self.check_validity();
    }

    pub fn make_move_nnue(&mut self, m: Move, nnue: &mut NNUEState) {
        let mut update_buffer = UpdateBuffer::default();
        let piece = self.state.mailbox[m.from()].unwrap();

        self.make_move_base(m, &mut update_buffer);

        nnue.accumulators[nnue.current_acc].mv = MovedPiece {
            from: m.from(),
            to: m.to(),
            piece,
        };
        nnue.accumulators[nnue.current_acc].update_buffer = update_buffer;
        nnue.current_acc += 1;
        nnue.accumulators[nnue.current_acc].correct = [false; 2];
    }

    pub fn unmake_move_nnue(&mut self, nnue: &mut NNUEState) {
        self.unmake_move_base();
        nnue.current_acc -= 1;
    }

    pub fn make_move(&mut self, m: Move, nnue: &mut NNUEState) {
        self.make_move_nnue(m, nnue);
    }

    pub fn unmake_move(&mut self, nnue: &mut NNUEState) {
        self.unmake_move_nnue(nnue);
    }

    /// Makes a guess about the new position key after a move.
    /// This is a cheap estimate, and will fail for special moves such as castling.
    pub fn key_after(&self, m: Move) -> u64 {
        let src = m.from();
        let tgt = m.to();
        let piece = self.state.mailbox[src].unwrap();
        let captured = self.state.mailbox[tgt];
        let is_pawn = piece.piece_type() == PieceType::Pawn;
        let src_piece = piece;
        let dst_piece = m
            .promotion_type()
            .map_or(piece, |promo| Piece::new(src_piece.colour(), promo));

        let mut new_key = self.state.keys.zobrist;
        new_key ^= PIECE_KEYS[src_piece][src];
        new_key ^= PIECE_KEYS[dst_piece][tgt];

        if let Some(captured) = captured {
            new_key ^= PIECE_KEYS[captured][tgt];
        }

        new_key ^= SIDE_KEY;

        let new_hmc = if captured.is_some() || is_pawn {
            0
        } else {
            self.state.fifty_move_counter + 1
        };

        new_key ^ HM_CLOCK_KEYS[new_hmc as usize]
    }

    pub fn key_after_null_move(&self) -> u64 {
        self.state.keys.zobrist ^ SIDE_KEY
    }

    /// Parses a move in the UCI format and returns a move or a reason why it couldn't be parsed.
    pub fn parse_uci(&self, uci: &str) -> Result<Move, MoveParseError> {
        use MoveParseError::{
            IllegalMove, InvalidFromSquareFile, InvalidFromSquareRank, InvalidLength,
            InvalidPromotionPiece, InvalidToSquareFile, InvalidToSquareRank, Unknown,
        };
        let san_bytes = uci.as_bytes();
        if !(4..=5).contains(&san_bytes.len()) {
            return Err(InvalidLength(san_bytes.len()));
        }
        if !(b'a'..=b'h').contains(&san_bytes[0]) {
            return Err(InvalidFromSquareFile(san_bytes[0] as char));
        }
        if !(b'1'..=b'8').contains(&san_bytes[1]) {
            return Err(InvalidFromSquareRank(san_bytes[1] as char));
        }
        if !(b'a'..=b'h').contains(&san_bytes[2]) {
            return Err(InvalidToSquareFile(san_bytes[2] as char));
        }
        if !(b'1'..=b'8').contains(&san_bytes[3]) {
            return Err(InvalidToSquareRank(san_bytes[3] as char));
        }
        if san_bytes.len() == 5 && ![b'n', b'b', b'r', b'q', b'k'].contains(&san_bytes[4]) {
            return Err(InvalidPromotionPiece(san_bytes[4] as char));
        }

        let from = Square::from_rank_file(
            Rank::from_index(san_bytes[1] - b'1').ok_or(Unknown)?,
            File::from_index(san_bytes[0] - b'a').ok_or(Unknown)?,
        );
        let to = Square::from_rank_file(
            Rank::from_index(san_bytes[3] - b'1').ok_or(Unknown)?,
            File::from_index(san_bytes[2] - b'a').ok_or(Unknown)?,
        );

        let frc_cleanup = !CHESS960.load(Ordering::Relaxed);

        self.legal_moves()
            .into_iter()
            .find(|&m| {
                let m_to = if frc_cleanup && m.is_castle() {
                    // if we're in normal UCI mode, we'll rework our castling moves into the
                    // standard format.
                    match m.to() {
                        Square::A1 => Square::C1,
                        Square::H1 => Square::G1,
                        Square::A8 => Square::C8,
                        Square::H8 => Square::G8,
                        _ => m.to(),
                    }
                } else {
                    m.to()
                };
                m.from() == from
                    && m_to == to
                    && (san_bytes.len() == 4
                        || m.promotion_type().and_then(PieceType::promo_char).unwrap()
                            == san_bytes[4] as char)
            })
            .ok_or_else(|| IllegalMove(uci.to_string()))
    }

    pub fn gives(&self, m: Move) -> CheckState {
        debug_assert!(self.is_pseudo_legal(m));
        debug_assert!(self.is_legal(m));
        let Self {
            state,
            side,
            ply,
            height,
            ..
        } = self;
        let mut playout = Self {
            state: state.clone(),
            side: *side,
            ply: *ply,
            height: *height,
            history: Vec::new(),
        };
        playout.make_move_simple(m);
        let gives_check = playout.in_check();
        if gives_check {
            let mut ml = MoveList::new();
            playout.generate_moves(&mut ml);
            for &m in ml.iter_moves() {
                if !playout.is_legal(m) {
                    continue;
                }
                // we found a legal move, so m does not give checkmate.
                return CheckState::Check;
            }
            // we didn't return, so there were no legal moves,
            // so m gives checkmate.
            return CheckState::Checkmate;
        }
        CheckState::None
    }

    /// Has the current position occurred before in the current game?
    pub fn is_repetition(&self) -> bool {
        let mut counter = 0;
        // distance to the last irreversible move
        let moves_since_zeroing = self.fifty_move_counter() as usize;
        // a repetition is first possible at four ply back:
        for (dist_back, u) in self
            .history
            .iter()
            .rev()
            .enumerate()
            .take(moves_since_zeroing)
            .skip(3)
            .step_by(2)
        {
            if u.keys.zobrist == self.state.keys.zobrist {
                // in-tree, can twofold:
                if dist_back < self.height {
                    return true;
                }
                // partially materialised, proper threefold:
                counter += 1;
                if counter >= 2 {
                    return true;
                }
            }
        }
        false
    }

    /// Should we consider the current position a draw?
    pub fn is_draw(&self) -> bool {
        (self.state.fifty_move_counter >= 100 || self.is_repetition()) && self.height != 0
    }

    pub fn pv_san(&self, pv: &PVariation) -> Result<String, fmt::Error> {
        let mut playout = self.clone();
        let mut out = String::new();
        for &m in pv.moves() {
            let san = playout.san(m).expect("illegal move in PV");
            write!(out, "{san} ")?;
            playout.make_move_simple(m);
        }
        Ok(out)
    }

    pub fn legal_moves(&self) -> ArrayVec<Move, MAX_POSITION_MOVES> {
        let mut legal_moves = ArrayVec::default();
        let mut move_list = MoveList::new();
        self.generate_moves(&mut move_list);
        for &m in move_list.iter_moves() {
            if self.is_legal(m) {
                legal_moves.push(m);
            }
        }
        legal_moves
    }

    pub const fn fifty_move_counter(&self) -> u8 {
        self.state.fifty_move_counter
    }

    #[cfg(any(feature = "datagen", test))]
    pub fn has_insufficient_material<C: Col>(&self) -> bool {
        use PieceType::{Bishop, King, Knight, Pawn, Queen, Rook};

        let bbs = &self.state.bbs;

        let us = bbs.colours[C::COLOUR];
        let them = bbs.colours[!C::COLOUR];

        if us & (bbs.pieces[Pawn] | bbs.pieces[Rook] | bbs.pieces[Queen]) != SquareSet::EMPTY {
            return false;
        }

        if us & bbs.pieces[Knight] != SquareSet::EMPTY {
            // this approach renders KNNvK as *not* being insufficient material.
            // this is because the losing side can in theory help the winning side
            // into a checkmate, despite it being impossible to /force/ mate.
            let kings = bbs.pieces[King];
            let queens = bbs.pieces[Queen];
            return us.count() <= 2 && them & !kings & !queens == SquareSet::EMPTY;
        }

        if us & bbs.pieces[Bishop] != SquareSet::EMPTY {
            let bishops = bbs.pieces[Bishop];
            let pawns = bbs.pieces[Pawn];
            let knights = bbs.pieces[Knight];
            return pawns == SquareSet::EMPTY
                && knights == SquareSet::EMPTY
                && (bishops & SquareSet::DARK_SQUARES == SquareSet::EMPTY
                    || bishops & SquareSet::LIGHT_SQUARES == SquareSet::EMPTY);
        }

        true
    }

    #[cfg(feature = "datagen")]
    pub const fn full_move_number(&self) -> usize {
        self.ply / 2 + 1
    }

    pub fn has_game_cycle(&self, ply: usize) -> bool {
        let end = std::cmp::min(self.fifty_move_counter() as usize, self.history.len());

        if end < 3 {
            return false;
        }

        let old_key = |i: usize| self.history[self.history.len() - i].keys.zobrist;

        let occ = self.state.bbs.occupied();
        let original_key = self.state.keys.zobrist;

        let mut other = !(original_key ^ old_key(1));

        for i in (3..=end).step_by(2) {
            let curr_key = old_key(i);

            other ^= !(curr_key ^ old_key(i - 1));
            if other != 0 {
                continue;
            }

            #[allow(clippy::cast_possible_truncation)]
            let diff = original_key ^ curr_key;

            let mut slot = cuckoo::h1(diff);

            if diff != cuckoo::KEYS[slot] {
                slot = cuckoo::h2(diff);
            }

            if diff != cuckoo::KEYS[slot] {
                continue;
            }

            let mv = cuckoo::MOVES[slot].unwrap();

            if (occ & RAY_BETWEEN[mv.from()][mv.to()]) == SquareSet::EMPTY {
                // repetition is after root, done:
                if ply > i {
                    return true;
                }

                let mut piece = self.state.mailbox[mv.from()];
                if piece.is_none() {
                    piece = self.state.mailbox[mv.to()];
                }

                return piece.unwrap().colour() == self.side;
            }
        }

        false
    }

    #[cfg(any(feature = "datagen", test))]
    pub fn is_insufficient_material(&self) -> bool {
        self.has_insufficient_material::<White>() && self.has_insufficient_material::<Black>()
    }

    #[cfg(any(feature = "datagen", test))]
    pub fn outcome(&self) -> Option<GameOutcome> {
        use DrawType::{FiftyMoves, InsufficientMaterial, Repetition, Stalemate};
        use GameOutcome::{BlackWin, Draw, WhiteWin};
        if self.state.fifty_move_counter >= 100 {
            return Some(Draw(FiftyMoves));
        }
        let mut reps = 1;
        for undo in self.history.iter().rev().skip(1).step_by(2) {
            if undo.keys.zobrist == self.state.keys.zobrist {
                reps += 1;
                if reps == 3 {
                    return Some(Draw(Repetition));
                }
            }
            // optimisation: if the fifty move counter was zeroed, then any prior positions will not be repetitions.
            if undo.fifty_move_counter == 0 {
                break;
            }
        }
        if self.is_insufficient_material() {
            return Some(Draw(InsufficientMaterial));
        }
        let mut move_list = MoveList::new();
        self.generate_moves(&mut move_list);
        let mut legal_moves = false;
        for &m in move_list.iter_moves() {
            if self.is_legal(m) {
                legal_moves = true;
                break;
            }
        }
        if legal_moves {
            None
        } else if self.in_check() {
            match self.side {
                Colour::White => Some(BlackWin(WinType::Mate)),
                Colour::Black => Some(WhiteWin(WinType::Mate)),
            }
        } else {
            Some(Draw(Stalemate))
        }
    }

    #[cfg(debug_assertions)]
    pub fn assert_mated(&self) {
        assert!(self.in_check());
        let mut move_list = MoveList::new();
        self.generate_moves(&mut move_list);
        for &mv in move_list.iter_moves() {
            assert!(!self.is_legal(mv));
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GameOutcome {
    BlackWin(WinType),
    Draw(DrawType),
    WhiteWin(WinType),
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WinType {
    Mate,
    TB,
    Adjudication,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DrawType {
    TB,
    FiftyMoves,
    Repetition,
    Stalemate,
    InsufficientMaterial,
    Adjudication,
}

#[cfg(feature = "datagen")]
impl GameOutcome {
    pub const fn as_packed_u8(self) -> u8 {
        // 0 for black win, 1 for draw, 2 for white win
        match self {
            Self::BlackWin(_) => 0,
            Self::Draw(_) => 1,
            Self::WhiteWin(_) => 2,
        }
    }
}

impl Default for Board {
    fn default() -> Self {
        let mut out = Self::empty();
        out.set_startpos();
        out
    }
}

impl Display for Board {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        let mut counter = 0;
        for rank in Rank::all().rev() {
            for file in File::all() {
                let sq = Square::from_rank_file(rank, file);
                let piece = self.state.mailbox[sq];
                if let Some(piece) = piece {
                    if counter != 0 {
                        write!(f, "{counter}")?;
                    }
                    counter = 0;
                    write!(f, "{piece}")?;
                } else {
                    counter += 1;
                }
            }
            if counter != 0 {
                write!(f, "{counter}")?;
            }
            counter = 0;
            if rank != Rank::One {
                write!(f, "/")?;
            }
        }

        match self.side {
            Colour::White => write!(f, " w")?,
            Colour::Black => write!(f, " b")?,
        }
        write!(f, " ")?;
        if self.state.castle_perm == CastlingRights::default() {
            write!(f, "-")?;
        } else {
            for (_, ch) in [
                self.state.castle_perm.kingside(Colour::White),
                self.state.castle_perm.queenside(Colour::White),
                self.state.castle_perm.kingside(Colour::Black),
                self.state.castle_perm.queenside(Colour::Black),
            ]
            .into_iter()
            .zip("KQkq".chars())
            .filter(|(m, _)| m.is_some())
            {
                write!(f, "{ch}")?;
            }
        }
        if let Some(ep_sq) = self.state.ep_square {
            write!(f, " {ep_sq}")?;
        } else {
            write!(f, " -")?;
        }
        write!(f, " {}", self.state.fifty_move_counter)?;
        write!(f, " {}", self.ply / 2 + 1)?;

        Ok(())
    }
}

impl std::fmt::UpperHex for Board {
    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
        for rank in Rank::all().rev() {
            write!(f, "{} ", rank as u8 + 1)?;
            for file in File::all() {
                let sq = Square::from_rank_file(rank, file);
                if let Some(piece) = self.state.mailbox[sq] {
                    write!(f, "{piece} ")?;
                } else {
                    write!(f, ". ")?;
                }
            }
            writeln!(f)?;
        }

        writeln!(f, "  a b c d e f g h")?;
        write!(f, "FEN: {self}")?;

        Ok(())
    }
}

mod tests {
    #[test]
    fn game_end_states() {
        use super::Board;
        use super::{DrawType, GameOutcome};
        use crate::{chess::chessmove::Move, chess::types::Square};
        use DrawType::{FiftyMoves, InsufficientMaterial, Repetition, Stalemate};
        use GameOutcome::Draw;

        let fiftymove_draw =
            Board::from_fen("rnbqkb1r/pppppppp/5n2/8/3N4/8/PPPPPPPP/RNBQKB1R b KQkq - 100 2")
                .unwrap();
        assert_eq!(fiftymove_draw.outcome(), Some(Draw(FiftyMoves)));
        let mut draw_repetition = Board::default();
        assert_eq!(draw_repetition.outcome(), None);
        draw_repetition.make_move_simple(Move::new(Square::G1, Square::F3));
        draw_repetition.make_move_simple(Move::new(Square::B8, Square::C6));
        assert_eq!(draw_repetition.outcome(), None);
        draw_repetition.make_move_simple(Move::new(Square::F3, Square::G1));
        draw_repetition.make_move_simple(Move::new(Square::C6, Square::B8));
        assert_eq!(draw_repetition.outcome(), None);
        draw_repetition.make_move_simple(Move::new(Square::G1, Square::F3));
        draw_repetition.make_move_simple(Move::new(Square::B8, Square::C6));
        assert_eq!(draw_repetition.outcome(), None);
        draw_repetition.make_move_simple(Move::new(Square::F3, Square::G1));
        draw_repetition.make_move_simple(Move::new(Square::C6, Square::B8));
        assert_eq!(draw_repetition.outcome(), Some(Draw(Repetition)));
        let stalemate = Board::from_fen("7k/8/6Q1/8/8/8/8/K7 b - - 0 1").unwrap();
        assert_eq!(stalemate.outcome(), Some(Draw(Stalemate)));
        let insufficient_material_bare_kings =
            Board::from_fen("8/8/5k2/8/8/2K5/8/8 b - - 0 1").unwrap();
        assert_eq!(
            insufficient_material_bare_kings.outcome(),
            Some(Draw(InsufficientMaterial))
        );
        let insufficient_material_knights =
            Board::from_fen("8/8/5k2/8/2N5/2K2N2/8/8 b - - 0 1").unwrap();
        // using FIDE rules.
        assert_eq!(insufficient_material_knights.outcome(), None);
    }

    #[test]
    fn fen_round_trip() {
        use crate::chess::board::Board;
        use std::{
            fs::File,
            io::{BufRead, BufReader},
        };

        let fens = BufReader::new(File::open("epds/perftsuite.epd").unwrap())
            .lines()
            .map(|l| l.unwrap().split_once(';').unwrap().0.trim().to_owned())
            .collect::<Vec<_>>();
        for fen in fens {
            let board = Board::from_fen(&fen).expect("from_fen failed.");
            let fen_2 = board.to_string();
            assert_eq!(fen, fen_2);
        }
    }

    #[test]
    fn scharnagl_backrank_works() {
        use super::Board;
        use crate::chess::piece::PieceType;
        let normal_chess_arrangement = Board::get_scharnagl_backrank(518);
        assert_eq!(
            normal_chess_arrangement,
            [
                PieceType::Rook,
                PieceType::Knight,
                PieceType::Bishop,
                PieceType::Queen,
                PieceType::King,
                PieceType::Bishop,
                PieceType::Knight,
                PieceType::Rook
            ]
        );
    }

    #[test]
    fn scharnagl_full_works() {
        #![allow(clippy::similar_names)]
        use super::Board;
        let normal = Board::from_fen(Board::STARTING_FEN).unwrap();
        let frc = Board::from_frc_idx(518);
        let dfrc = Board::from_dfrc_idx(518 * 960 + 518);
        assert_eq!(normal, frc);
        assert_eq!(normal, dfrc);
    }

    #[test]
    fn castling_pseudolegality() {
        use super::Board;
        use crate::chess::chessmove::{Move, MoveFlags};
        use crate::chess::types::Square;
        let board =
            Board::from_fen("1r2k2r/2pb1pp1/2pp4/p1n5/2P4p/PP2P2P/1qB2PP1/R2QKN1R w KQk - 0 20")
                .unwrap();
        let kingside_castle = Move::new_with_flags(Square::E1, Square::H1, MoveFlags::Castle);
        assert!(!board.is_pseudo_legal(kingside_castle));
    }

    #[test]
    fn threat_generation_white() {
        use super::Board;
        use crate::chess::squareset::SquareSet;

        let board = Board::from_fen("3k4/8/8/5N2/8/1P6/8/K1Q1RB2 b - - 0 1").unwrap();
        assert_eq!(
            board.state.threats.all,
            SquareSet::from_inner(0x1454_9d56_bddd_5f3f)
        );
    }

    #[test]
    fn threat_generation_black() {
        use super::Board;
        use crate::chess::squareset::SquareSet;

        let board = Board::from_fen("2br1q1k/8/6p1/8/2n5/8/8/4K3 w - - 0 1").unwrap();
        assert_eq!(
            board.state.threats.all,
            SquareSet::from_inner(0xfcfa_bbbd_6ab9_2a28)
        );
    }

    #[test]
    fn key_after_works_for_simple_moves() {
        use super::Board;
        use crate::chess::chessmove::Move;
        use crate::chess::types::Square;
        let mut board = Board::default();
        let mv = Move::new(Square::E2, Square::E3);
        let key = board.key_after(mv);
        board.make_move_simple(mv);
        assert_eq!(board.state.keys.zobrist, key);
    }

    #[test]
    fn key_after_works_for_captures() {
        use super::Board;
        use crate::chess::chessmove::Move;
        use crate::chess::types::Square;
        let mut board =
            Board::from_fen("r1bqkb1r/ppp2ppp/2n5/3np1N1/2B5/8/PPPP1PPP/RNBQK2R w KQkq - 0 6")
                .unwrap();
        // Nxf7!!
        let mv = Move::new(Square::G5, Square::F7);
        let key = board.key_after(mv);
        board.make_move_simple(mv);
        assert_eq!(board.state.keys.zobrist, key);
    }

    #[test]
    fn key_after_works_for_nullmove() {
        use super::Board;
        let mut board = Board::default();
        let key = board.key_after_null_move();
        board.make_nullmove();
        assert_eq!(board.state.keys.zobrist, key);
    }

    #[test]
    fn ep_square_edge_case() {
        use super::Board;
        use crate::chess::chessmove::Move;
        use crate::chess::piece::Piece;
        use crate::chess::types::Square;
        use crate::lookups::{EP_KEYS, PIECE_KEYS, SIDE_KEY};
        let mut not_ep_capturable =
            Board::from_fen("rnbqkbnr/ppppp1pp/8/5p2/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2")
                .unwrap();
        let mut ep_capturable =
            Board::from_fen("rnbqkbnr/ppppp1pp/8/4Pp2/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2").unwrap();
        let d5 = Move::new(Square::D7, Square::D5);
        let mut not_ep_capturable_key = not_ep_capturable.state.keys.zobrist;
        let mut ep_capturable_key = ep_capturable.state.keys.zobrist;
        not_ep_capturable_key ^= SIDE_KEY;
        ep_capturable_key ^= SIDE_KEY;
        not_ep_capturable_key ^= PIECE_KEYS[Piece::BP][Square::D7];
        ep_capturable_key ^= PIECE_KEYS[Piece::BP][Square::D7];
        not_ep_capturable_key ^= PIECE_KEYS[Piece::BP][Square::D5];
        ep_capturable_key ^= PIECE_KEYS[Piece::BP][Square::D5];

        ep_capturable_key ^= EP_KEYS[Square::D6];

        assert!(not_ep_capturable.is_legal(d5));
        not_ep_capturable.make_move_simple(d5);
        assert!(ep_capturable.is_legal(d5));
        ep_capturable.make_move_simple(d5);

        assert_eq!(not_ep_capturable.state.ep_square, None);
        assert_eq!(ep_capturable.state.ep_square, Some(Square::D6));

        assert_eq!(not_ep_capturable.state.keys.zobrist, not_ep_capturable_key);
        assert_eq!(ep_capturable.state.keys.zobrist, ep_capturable_key);
    }

    #[test]
    fn other_ep_edge_case() {
        use super::Board;
        use crate::chess::chessmove::Move;
        use crate::chess::types::Square;
        let mut board =
            Board::from_fen("rnbqkbnr/1ppppppp/p7/P7/8/8/1PPPPPPP/RNBQKBNR b KQkq - 0 2").unwrap();
        assert!(board.is_legal(Move::new(Square::B7, Square::B5)));
        board.make_move_simple(Move::new(Square::B7, Square::B5));
        assert_eq!(board.state.ep_square, Some(Square::B6));
    }

    #[test]
    fn reset_properly() {
        use super::Board;
        use crate::chess::chessmove::Move;
        use crate::chess::types::Square;
        let mut board = Board::default();
        assert!(board.is_legal(Move::new(Square::E2, Square::E4)));
        board.make_move_simple(Move::new(Square::E2, Square::E4));
        assert!(board.is_legal(Move::new(Square::E7, Square::E5)));
        board.make_move_simple(Move::new(Square::E7, Square::E5));
        board.set_startpos();
        let board2 = Board::default();
        assert_eq!(board, board2);
    }

    #[test]
    fn illegal_ep_construction() {
        use super::Board;
        use crate::chess::types::Square;

        let illegal =
            Board::from_fen("rnbq1bnr/p1ppkppp/8/4p3/1pP5/BP3PP1/P2PP2P/RN1QKBNR b KQ c3 0 5")
                .unwrap();
        assert!(illegal.ep_sq().is_none());

        let legal =
            Board::from_fen("r1bqkbnr/pppp1p1p/2n5/4pPp1/4P3/8/PPPP2PP/RNBQKBNR w KQkq g6 0 4")
                .unwrap();
        assert_eq!(legal.ep_sq(), Some(Square::G6));
    }

    #[test]
    fn quick_sanity_check() {
        use crate::chess::board::Board;

        let board = Board::from_quick("RxlGB:R:::::9GK:::::Dk::::::O:::::::::::").unwrap();

        assert_eq!(board.to_string(), "8/8/6q1/8/p2R4/P3P3/1P1K1k2/8 b - - 0 1");
    }
}
