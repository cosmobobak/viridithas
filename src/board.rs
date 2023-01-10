pub mod evaluation;
mod history;
pub mod movegen;
pub mod validation;

use std::{
    fmt::{Debug, Display, Formatter, Write},
    sync::Once,
};

use regex::Regex;

use crate::{
    board::{
        evaluation::get_eval_params,
        movegen::{
            bitboards::{
                self, pawn_attacks, BitShiftExt, BB_ALL, BB_FILES, BB_NONE, BB_RANKS, BB_RANK_2,
                BB_RANK_4, BB_RANK_5, BB_RANK_7,
            },
            MoveList,
        },
    },
    chessmove::Move,
    definitions::{
        File,
        Rank::{self, RANK_3, RANK_6},
        Square, Undo, BKCA, BQCA, INFINITY,
        MAX_DEPTH, WKCA, WQCA, CheckState,
    },
    errors::{FenParseError, MoveParseError},
    lookups::{PIECE_BIG, PIECE_MAJ},
    macros,
    makemove::{hash_castling, hash_ep, hash_piece, hash_side, CASTLE_PERM_MASKS},
    nnue::{ACTIVATE, DEACTIVATE},
    piecesquaretable::pst_value,
    threadlocal::ThreadData,
    transpositiontable::{ProbeResult, TTHit, TTView},
    piece::{Piece, Colour, PieceType},
};

use self::{evaluation::score::S, movegen::bitboards::BitBoard};

static SAN_REGEX_INIT: Once = Once::new();
static mut SAN_REGEX: Option<Regex> = None;
fn get_san_regex() -> &'static Regex {
    unsafe {
        SAN_REGEX_INIT.call_once(|| {
            SAN_REGEX = Some(
                Regex::new(
                    r"^([NBKRQ])?([a-h])?([1-8])?[\-x]?([a-h][1-8])(=?[nbrqkNBRQK])?[\+#]?$",
                )
                .unwrap(),
            );
        });
        SAN_REGEX.as_ref().unwrap()
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct Board {
    /// The bitboards of all the pieces on the board.
    pub(crate) pieces: BitBoard,
    /// An array to accelerate piece_at().
    piece_array: [Piece; 64],
    /// The side to move.
    side: Colour,
    /// The en passant square.
    ep_sq: Square,
    /// The castling permissions.
    castle_perm: u8,
    /// The number of half moves made since the last capture or pawn advance.
    fifty_move_counter: u8,
    /// The number of half moves made since the start of the game.
    ply: usize,

    /// The Zobrist hash of the board.
    key: u64,

    /* Incrementally updated features used to accelerate various queries */
    big_piece_counts: [u8; 2],
    major_piece_counts: [u8; 2],
    minor_piece_counts: [u8; 2],
    material: [S; 2],
    pst_vals: S,

    height: usize,
    history: Vec<Undo>,
    repetition_cache: Vec<u64>,

    principal_variation: Vec<Move>,
}

impl Debug for Board {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Board")
            .field("piece_array", &self.piece_array)
            .field("side", &self.side)
            .field("ep_sq", &self.ep_sq)
            .field("fifty_move_counter", &self.fifty_move_counter)
            .field("height", &self.height)
            .field("ply", &self.ply)
            .field("key", &self.key)
            .field("big_piece_counts", &self.big_piece_counts)
            .field("major_piece_counts", &self.major_piece_counts)
            .field("minor_piece_counts", &self.minor_piece_counts)
            .field("material", &self.material)
            .field("castle_perm", &self.castle_perm)
            .field("pst_vals", &self.pst_vals)
            .finish_non_exhaustive()
    }
}

impl Board {
    pub const STARTING_FEN: &'static str =
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    pub fn new() -> Self {
        let mut out = Self {
            pieces: BitBoard::NULL,
            piece_array: [Piece::EMPTY; 64],
            side: Colour::WHITE,
            ep_sq: Square::NO_SQUARE,
            fifty_move_counter: 0,
            height: 0,
            ply: 0,
            key: 0,
            big_piece_counts: [0; 2],
            major_piece_counts: [0; 2],
            minor_piece_counts: [0; 2],
            material: [S(0, 0); 2],
            castle_perm: 0,
            history: Vec::new(),
            repetition_cache: Vec::new(),
            principal_variation: Vec::new(),
            pst_vals: S(0, 0),
        };
        out.reset();
        out
    }

    pub const fn ep_sq(&self) -> Square {
        self.ep_sq
    }

    pub const fn hashkey(&self) -> u64 {
        self.key
    }

    pub fn principal_variation(&self) -> &[Move] {
        &self.principal_variation
    }

    // pub const fn n_men(&self) -> u8 {
    //     #![allow(clippy::cast_possible_truncation)]
    //     self.pieces.occupied().count_ones() as u8
    // }

    pub fn king_sq(&self, side: Colour) -> Square {
        debug_assert!(side == Colour::WHITE || side == Colour::BLACK);
        debug_assert_eq!(self.pieces.king::<true>().count_ones(), 1);
        debug_assert_eq!(self.pieces.king::<false>().count_ones(), 1);
        let sq = match side {
            Colour::WHITE => self.pieces.king::<true>().first_square(),
            Colour::BLACK => self.pieces.king::<false>().first_square(),
            _ => unsafe { macros::inconceivable!() },
        };
        debug_assert!(sq < Square::NO_SQUARE);
        debug_assert_eq!(self.piece_at(sq).colour(), side);
        debug_assert_eq!(self.piece_at(sq).piece_type(), PieceType::KING);
        sq
    }

    pub const US: u8 = 0;
    pub const THEM: u8 = 1;
    pub fn in_check<const SIDE: u8>(&self) -> bool {
        if SIDE == Self::US {
            let king_sq = self.king_sq(self.side);
            self.sq_attacked(king_sq, self.side.flip())
        } else {
            let king_sq = self.king_sq(self.side.flip());
            self.sq_attacked(king_sq, self.side)
        }
    }

    #[allow(dead_code)]
    pub fn maybe_gives_check(&self, mov: Move) -> bool {
        let piece = self.piece_at(mov.from());
        let piece_type = piece.piece_type();
        let stm = self.side;

        let opponent_king = self.king_sq(stm.flip());
        let blockers = self.pieces.occupied() ^ mov.bitboard();

        // direct check:
        let attacks_from_dest = if piece_type == PieceType::PAWN {
            if stm == Colour::WHITE {
                pawn_attacks::<true>(mov.to().bitboard())
            } else {
                pawn_attacks::<false>(mov.to().bitboard())
            }
        } else {
            bitboards::attacks_by_type(piece_type, mov.to(), blockers)
        };
        if attacks_from_dest & opponent_king.bitboard() != 0 {
            return true;
        }

        // discovered check:
        let ortho = bitboards::attacks::<{ PieceType::ROOK.inner() }>(opponent_king, blockers);
        let diag = bitboards::attacks::<{ PieceType::BISHOP.inner() }>(opponent_king, blockers);
        let our_orthos = if stm == Colour::WHITE {
            self.pieces.rookqueen::<true>()
        } else {
            self.pieces.rookqueen::<false>()
        };
        let our_diags = if stm == Colour::WHITE {
            self.pieces.bishopqueen::<true>()
        } else {
            self.pieces.bishopqueen::<false>()
        };
        if ortho & our_orthos != 0 {
            return true;
        }
        if diag & our_diags != 0 {
            return true;
        }

        false
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

    pub fn generate_pos_key(&self) -> u64 {
        #![allow(clippy::cast_possible_truncation)]
        let mut key = 0;
        for (sq, &piece) in self.piece_array.iter().enumerate() {
            let sq = Square::new(sq.try_into().unwrap());
            if !piece.is_empty() {
                hash_piece(&mut key, piece, sq);
            }
        }

        if self.side == Colour::WHITE {
            hash_side(&mut key);
        }

        if self.ep_sq != Square::NO_SQUARE {
            debug_assert!(self.ep_sq.on_board());
            hash_ep(&mut key, self.ep_sq);
        }

        debug_assert!(self.castle_perm <= 15);
        hash_castling(&mut key, self.castle_perm);

        debug_assert!(self.fifty_move_counter < 100);

        key
    }

    pub fn reset(&mut self) {
        self.pieces.reset();
        self.piece_array = [Piece::EMPTY; 64];
        self.big_piece_counts.fill(0);
        self.major_piece_counts.fill(0);
        self.minor_piece_counts.fill(0);
        self.material.fill(S(0, 0));
        self.side = Colour::WHITE;
        self.ep_sq = Square::NO_SQUARE;
        self.fifty_move_counter = 0;
        self.height = 0;
        self.ply = 0;
        self.castle_perm = 0;
        self.key = 0;
        self.pst_vals = S(0, 0);
        self.history.clear();
        self.repetition_cache.clear();
    }

    pub fn set_from_fen(&mut self, fen: &str) -> Result<(), FenParseError> {
        if !fen.is_ascii() {
            return Err(format!("FEN string is not ASCII: {fen}"));
        }

        let mut rank = Rank::RANK_8;
        let mut file = File::FILE_A;

        self.reset();

        let fen_chars = fen.as_bytes();
        let split_idx = fen_chars
            .iter()
            .position(|&c| c == b' ')
            .ok_or_else(|| format!("FEN string is missing space: {fen}"))?;
        let (board_part, info_part) = fen_chars.split_at(split_idx);

        for &c in board_part {
            let mut count = 1;
            let piece;
            match c {
                b'P' => piece = Piece::WP,
                b'R' => piece = Piece::WR,
                b'N' => piece = Piece::WN,
                b'B' => piece = Piece::WB,
                b'Q' => piece = Piece::WQ,
                b'K' => piece = Piece::WK,
                b'p' => piece = Piece::BP,
                b'r' => piece = Piece::BR,
                b'n' => piece = Piece::BN,
                b'b' => piece = Piece::BB,
                b'q' => piece = Piece::BQ,
                b'k' => piece = Piece::BK,
                b'1'..=b'8' => {
                    piece = Piece::EMPTY;
                    count = c - b'0';
                }
                b'/' => {
                    rank -= 1;
                    file = File::FILE_A;
                    continue;
                }
                c => {
                    return Err(format!(
                        "FEN string is invalid, got unexpected character: \"{}\"",
                        c as char
                    ));
                }
            }

            for _ in 0..count {
                let sq = Square::from_rank_file(rank, file);
                if piece != Piece::EMPTY {
                    // this is only ever run once, as count is 1 for non-empty pieces.
                    self.add_piece(sq, piece);
                }
                file += 1;
            }
        }

        let mut info_parts = info_part[1..].split(|&c| c == b' ');

        self.set_side(info_parts.next())?;
        self.set_castling(info_parts.next())?;
        self.set_ep(info_parts.next())?;
        self.set_halfmove(info_parts.next())?;
        self.set_fullmove(info_parts.next())?;

        self.key = self.generate_pos_key();

        Ok(())
    }

    pub fn set_startpos(&mut self) {
        self.set_from_fen(Self::STARTING_FEN)
            .expect("for some reason, STARTING_FEN is now broken.");
        debug_assert_eq!(
            self.material[Colour::WHITE.index()].0, self.material[Colour::BLACK.index()].0,
            "mg_material is not equal, white: {}, black: {}",
            self.material[Colour::WHITE.index()].0, self.material[Colour::BLACK.index()].0
        );
        debug_assert_eq!(
            self.material[Colour::WHITE.index()].1, self.material[Colour::BLACK.index()].1,
            "eg_material is not equal, white: {}, black: {}",
            self.material[Colour::WHITE.index()].1, self.material[Colour::BLACK.index()].1
        );
        debug_assert_eq!(
            self.pst_vals.0, 0,
            "midgame pst value is not 0, it is: {}",
            self.pst_vals.0
        );
        debug_assert_eq!(
            self.pst_vals.1, 0,
            "endgame pst value is not 0, it is: {}",
            self.pst_vals.1
        );
    }

    #[allow(dead_code)]
    pub fn from_fen(fen: &str) -> Result<Self, FenParseError> {
        let mut out = Self::new();
        out.set_from_fen(fen)?;
        Ok(out)
    }

    pub fn fen(&self) -> String {
        let mut fen = String::with_capacity(60);

        let mut counter = 0;
        for rank in (Rank::RANK_1..=Rank::RANK_8).rev() {
            for file in File::FILE_A..=File::FILE_H {
                let sq = Square::from_rank_file(rank, file);
                let piece = self.piece_at(sq);
                if piece == Piece::EMPTY {
                    counter += 1;
                } else {
                    if counter != 0 {
                        write!(fen, "{counter}").unwrap();
                    }
                    counter = 0;
                    fen.push(piece.char().unwrap());
                }
            }
            if counter != 0 {
                write!(fen, "{counter}").unwrap();
            }
            counter = 0;
            if rank != 0 {
                fen.push('/');
            }
        }

        fen.push(' ');
        fen.push(if self.side == Colour::WHITE { 'w' } else { 'b' });
        fen.push(' ');
        if self.castle_perm == 0 {
            fen.push('-');
        } else {
            [WKCA, WQCA, BKCA, BQCA]
                .into_iter()
                .zip(['K', 'Q', 'k', 'q'])
                .filter(|(m, _)| self.castle_perm & m != 0)
                .for_each(|(_, ch)| fen.push(ch));
        }
        fen.push(' ');
        if self.ep_sq == Square::NO_SQUARE {
            fen.push('-');
        } else {
            fen.push_str(&self.ep_sq.to_string());
        }
        write!(fen, " {}", self.fifty_move_counter).unwrap();
        write!(fen, " {}", self.ply / 2 + 1).unwrap();

        fen
    }

    fn set_side(&mut self, side_part: Option<&[u8]>) -> Result<(), FenParseError> {
        self.side = match side_part {
            Some([b'w']) => Colour::WHITE,
            Some([b'b']) => Colour::BLACK,
            Some(other) => {
                return Err(format!(
                    "FEN string is invalid, expected side to be 'w' or 'b', got \"{}\"",
                    std::str::from_utf8(other).unwrap_or("<invalid utf8>")
                ))
            }
            None => return Err("FEN string is invalid, expected side part.".into()),
        };
        Ok(())
    }

    fn set_castling(&mut self, castling_part: Option<&[u8]>) -> Result<(), FenParseError> {
        match castling_part {
            None => return Err("FEN string is invalid, expected castling part.".into()),
            Some([b'-']) => self.castle_perm = 0,
            Some(castling) => {
                for &c in castling {
                    match c {
                        b'K' => self.castle_perm |= WKCA,
                        b'Q' => self.castle_perm |= WQCA,
                        b'k' => self.castle_perm |= BKCA,
                        b'q' => self.castle_perm |= BQCA,
                        _ => return Err(format!("FEN string is invalid, expected castling part to be of the form 'KQkq', got \"{}\"", std::str::from_utf8(castling).unwrap_or("<invalid utf8>"))),
                    }
                }
            }
        }

        Ok(())
    }

    fn set_ep(&mut self, ep_part: Option<&[u8]>) -> Result<(), FenParseError> {
        match ep_part {
            None => return Err("FEN string is invalid, expected en passant part.".to_string()),
            Some([b'-']) => self.ep_sq = Square::NO_SQUARE,
            Some(ep_sq) => {
                if ep_sq.len() != 2 {
                    return Err(format!("FEN string is invalid, expected en passant part to be of the form 'a1', got \"{}\"", std::str::from_utf8(ep_sq).unwrap_or("<invalid utf8>")));
                }
                let file = ep_sq[0] - b'a';
                let rank = ep_sq[1] - b'1';
                if !((File::FILE_A..=File::FILE_H).contains(&file)
                    && (Rank::RANK_1..=Rank::RANK_8).contains(&rank))
                {
                    return Err(format!("FEN string is invalid, expected en passant part to be of the form 'a1', got \"{}\"", std::str::from_utf8(ep_sq).unwrap_or("<invalid utf8>")));
                }
                self.ep_sq = Square::from_rank_file(rank, file);
            }
        }

        Ok(())
    }

    fn set_halfmove(&mut self, halfmove_part: Option<&[u8]>) -> Result<(), FenParseError> {
        match halfmove_part {
            None => return Err("FEN string is invalid, expected halfmove clock part.".to_string()),
            Some(halfmove_clock) => {
                self.fifty_move_counter = std::str::from_utf8(halfmove_clock)
                    .map_err(|_| {
                        "FEN string is invalid, expected halfmove clock part to be valid UTF-8"
                    })?
                    .parse::<u8>()
                    .map_err(|err| {
                        format!("FEN string is invalid, expected halfmove clock part to be a number, got \"{}\", {err}", std::str::from_utf8(halfmove_clock).unwrap_or("<invalid utf8>"))
                    })?;
            }
        }

        Ok(())
    }

    fn set_fullmove(&mut self, fullmove_part: Option<&[u8]>) -> Result<(), FenParseError> {
        match fullmove_part {
            None => return Err("FEN string is invalid, expected fullmove number part.".to_string()),
            Some(fullmove_number) => {
                let fullmove_number = std::str::from_utf8(fullmove_number)
                    .map_err(|_| {
                        "FEN string is invalid, expected fullmove number part to be valid UTF-8"
                    })?
                    .parse::<usize>()
                    .map_err(|_| {
                        "FEN string is invalid, expected fullmove number part to be a number"
                    })?;
                self.ply = (fullmove_number - 1) * 2;
                if self.side == Colour::BLACK {
                    self.ply += 1;
                }
            }
        }

        Ok(())
    }

    /// Determines if `sq` is attacked by `side`
    pub fn sq_attacked(&self, sq: Square, side: Colour) -> bool {
        if side == Colour::WHITE {
            self.sq_attacked_by::<true>(sq)
        } else {
            self.sq_attacked_by::<false>(sq)
        }
    }

    pub fn sq_attacked_by<const IS_WHITE: bool>(&self, sq: Square) -> bool {
        debug_assert!(sq.on_board());
        #[cfg(debug_assertions)]
        self.check_validity().unwrap();

        let sq_bb = sq.bitboard();
        let our_pawns = self.pieces.pawns::<IS_WHITE>();
        let our_knights = self.pieces.knights::<IS_WHITE>();
        let our_diags = self.pieces.bishopqueen::<IS_WHITE>();
        let our_orthos = self.pieces.rookqueen::<IS_WHITE>();
        let our_king = self.pieces.king::<IS_WHITE>();
        let blockers = self.pieces.occupied();

        // pawns
        let attacks = pawn_attacks::<IS_WHITE>(our_pawns);
        if attacks & sq_bb != 0 {
            return true;
        }

        // knights
        let knight_attacks_from_this_square = bitboards::attacks::<{ PieceType::KNIGHT.inner() }>(sq, BB_NONE);
        if our_knights & knight_attacks_from_this_square != BB_NONE {
            return true;
        }

        // bishops, queens
        let diag_attacks_from_this_square = bitboards::attacks::<{ PieceType::BISHOP.inner() }>(sq, blockers);
        if our_diags & diag_attacks_from_this_square != BB_NONE {
            return true;
        }

        // rooks, queens
        let ortho_attacks_from_this_square = bitboards::attacks::<{ PieceType::ROOK.inner() }>(sq, blockers);
        if our_orthos & ortho_attacks_from_this_square != BB_NONE {
            return true;
        }

        // king
        let king_attacks_from_this_square = bitboards::attacks::<{ PieceType::KING.inner() }>(sq, BB_NONE);
        if our_king & king_attacks_from_this_square != BB_NONE {
            return true;
        }

        false
    }

    /// Checks whether a move is pseudo-legal
    /// This means that it is a legal move, except for the fact that it might leave the king in check.
    pub fn is_pseudo_legal(&self, m: Move) -> bool {
        if m.is_null() {
            return false;
        }

        let from = m.from();
        let to = m.to();

        let moved_piece = self.piece_at(from);
        let captured_piece = self.piece_at(to);
        let is_capture = captured_piece != Piece::EMPTY;
        let is_pawn_double_push = self.is_double_pawn_push(m);

        if moved_piece == Piece::EMPTY {
            return false;
        }

        if moved_piece.colour() != self.side {
            return false;
        }

        if is_capture && captured_piece.colour() == self.side {
            return false;
        }

        if moved_piece.piece_type() != PieceType::PAWN && (is_pawn_double_push || m.is_ep() || m.is_promo()) {
            return false;
        }

        if moved_piece.piece_type() != PieceType::KING && m.is_castle() {
            return false;
        }

        if is_capture && is_pawn_double_push {
            return false;
        }

        if m.is_castle() {
            return self.is_pseudo_legal_castling(to);
        }

        if moved_piece.piece_type() == PieceType::PAWN {
            let should_be_promoting = to > Square::H7 || to < Square::A2;
            if should_be_promoting && !m.is_promo() {
                return false;
            }
            if m.is_ep() {
                return to == self.ep_sq;
            } else if is_pawn_double_push {
                let one_forward = from.pawn_push(self.side);
                return self.piece_at(one_forward) == Piece::EMPTY
                    && to == one_forward.pawn_push(self.side);
            } else if !is_capture {
                return to == from.pawn_push(self.side) && captured_piece == Piece::EMPTY;
            }
            // pawn capture
            if self.side == Colour::WHITE {
                return pawn_attacks::<true>(from.bitboard()) & to.bitboard() != 0;
            }
            return pawn_attacks::<false>(from.bitboard()) & to.bitboard() != 0;
        }

        to.bitboard()
            & bitboards::attacks_by_type(moved_piece.piece_type(), from, self.pieces.occupied())
            != BB_NONE
    }

    pub fn is_pseudo_legal_castling(&self, to: Square) -> bool {
        const WK_FREESPACE: u64 = Square::F1.bitboard() | Square::G1.bitboard();
        const WQ_FREESPACE: u64 =
            Square::B1.bitboard() | Square::C1.bitboard() | Square::D1.bitboard();
        const BK_FREESPACE: u64 = Square::F8.bitboard() | Square::G8.bitboard();
        const BQ_FREESPACE: u64 =
            Square::B8.bitboard() | Square::C8.bitboard() | Square::D8.bitboard();
        let occupied = self.pieces.occupied();

        assert!(to == Square::C1 || to == Square::G1 || to == Square::C8 || to == Square::G8);

        // illegal if
        // - we don't have castling rights on the target square
        // - we're in check
        // - there are pieces between the king and the rook
        // - the king passes through a square that is attacked by the opponent
        // - the king ends up in check (not checked here)

        let (target_castling_perm, occ_mask, from_sq, next_sq) = match (self.side, to) {
            (Colour::WHITE, Square::C1) => (self.castle_perm & WQCA, WQ_FREESPACE, Square::E1, Square::D1),
            (Colour::WHITE, Square::G1) => (self.castle_perm & WKCA, WK_FREESPACE, Square::E1, Square::F1),
            (Colour::BLACK, Square::C8) => (self.castle_perm & BQCA, BQ_FREESPACE, Square::E8, Square::D8),
            (Colour::BLACK, Square::G8) => (self.castle_perm & BKCA, BK_FREESPACE, Square::E8, Square::F8),
            (_, _) => return false,
        };
        if target_castling_perm == 0 {
            return false;
        }
        if occupied & occ_mask != 0 {
            return false;
        }
        if self.sq_attacked(from_sq, self.side.flip()) {
            return false;
        }
        if self.sq_attacked(next_sq, self.side.flip()) {
            return false;
        }

        true
    }

    /// Checks if a move is legal in the current position.
    /// Because moves must be played and unplayed, this method
    /// requires a mutable reference to the position.
    /// Despite this, you should expect the state of a Board
    /// object to be the same before and after calling [`Board::is_legal`].
    #[allow(clippy::wrong_self_convention)]
    pub fn is_legal(&mut self, move_to_check: Move) -> bool {
        let mut list = MoveList::new();
        self.generate_moves(&mut list);

        for &m in list.iter() {
            if !self.make_move_hce(m) {
                continue;
            }
            self.unmake_move_hce();
            if m == move_to_check {
                return true;
            }
        }

        false
    }

    fn clear_piece(&mut self, sq: Square) {
        debug_assert!(sq.on_board());
        let piece = self.piece_at(sq);
        debug_assert!(
            !piece.is_empty(),
            "Invalid piece at {}: {:?}, board {}, last move was {}",
            sq,
            piece,
            self.fen(),
            self.history.last().map_or("None".to_string(), |h| h.m.to_string())
        );

        self.pieces.clear_piece_at(sq, piece);

        let colour = piece.colour();

        hash_piece(&mut self.key, piece, sq);

        *self.piece_at_mut(sq) = Piece::EMPTY;
        self.material[colour.index()] -= get_eval_params().piece_values[piece.index()];
        self.pst_vals -= pst_value(piece, sq, &get_eval_params().piece_square_tables);

        if PIECE_BIG[piece.index()] {
            self.big_piece_counts[colour.index()] -= 1;
            if PIECE_MAJ[piece.index()] {
                self.major_piece_counts[colour.index()] -= 1;
            } else {
                self.minor_piece_counts[colour.index()] -= 1;
            }
        }
    }

    fn add_piece(&mut self, sq: Square, piece: Piece) {
        debug_assert!(sq.on_board());

        self.pieces.set_piece_at(sq, piece);

        let colour = piece.colour();

        hash_piece(&mut self.key, piece, sq);

        *self.piece_at_mut(sq) = piece;
        self.material[colour.index()] += get_eval_params().piece_values[piece.index()];
        self.pst_vals += pst_value(piece, sq, &get_eval_params().piece_square_tables);

        if PIECE_BIG[piece.index()] {
            self.big_piece_counts[colour.index()] += 1;
            if PIECE_MAJ[piece.index()] {
                self.major_piece_counts[colour.index()] += 1;
            } else {
                self.minor_piece_counts[colour.index()] += 1;
            }
        }
    }

    fn move_piece(&mut self, from: Square, to: Square) {
        debug_assert!(from.on_board());
        debug_assert!(to.on_board());

        let piece_moved = self.piece_at(from);

        let from_to_bb = from.bitboard() | to.bitboard();
        self.pieces.move_piece(from_to_bb, piece_moved);

        hash_piece(&mut self.key, piece_moved, from);
        hash_piece(&mut self.key, piece_moved, to);

        *self.piece_at_mut(from) = Piece::EMPTY;
        *self.piece_at_mut(to) = piece_moved;
        self.pst_vals -= pst_value(piece_moved, from, &get_eval_params().piece_square_tables);
        self.pst_vals += pst_value(piece_moved, to, &get_eval_params().piece_square_tables);
    }

    /// Gets the piece that will be moved by the given move.
    pub fn moved_piece(&self, m: Move) -> Piece {
        debug_assert!(m.from().on_board());
        unsafe { *self.piece_array.get_unchecked(m.from().index()) }
    }

    /// Gets the piece that will be captured by the given move.
    pub fn captured_piece(&self, m: Move) -> Piece {
        debug_assert!(m.to().on_board());
        unsafe { *self.piece_array.get_unchecked(m.to().index()) }
    }

    /// Determines whether this move would be a capture in the current position.
    pub fn is_capture(&self, m: Move) -> bool {
        debug_assert!(m.from().on_board());
        debug_assert!(m.to().on_board());
        self.captured_piece(m) != Piece::EMPTY
    }

    /// Determines whether this move would be a double pawn push in the current position.
    pub fn is_double_pawn_push(&self, m: Move) -> bool {
        debug_assert!(m.from().on_board());
        debug_assert!(m.to().on_board());
        let from_bb = m.from().bitboard();
        if from_bb & (BB_RANK_2 | BB_RANK_7) == 0 {
            return false;
        }
        let to_bb = m.to().bitboard();
        if to_bb & (BB_RANK_4 | BB_RANK_5) == 0 {
            return false;
        }
        let piece_moved = self.moved_piece(m);
        piece_moved.piece_type() == PieceType::PAWN
    }

    /// Determines whether this move would be tactical in the current position.
    pub fn is_tactical(&self, m: Move) -> bool {
        m.is_promo() || m.is_ep() || self.is_capture(m)
    }

    /// Gets the piece at the given square.
    pub fn piece_at(&self, sq: Square) -> Piece {
        debug_assert!(sq.on_board());
        unsafe { *self.piece_array.get_unchecked(sq.index()) }
    }

    /// Gets a mutable reference to the piece at the given square.
    pub fn piece_at_mut(&mut self, sq: Square) -> &mut Piece {
        debug_assert!(sq.on_board());
        unsafe { self.piece_array.get_unchecked_mut(sq.index()) }
    }

    #[allow(clippy::cognitive_complexity)]
    pub fn make_move_hce(&mut self, m: Move) -> bool {
        #[cfg(debug_assertions)]
        self.check_validity().unwrap();

        let from = m.from();
        let to = m.to();
        let side = self.side;
        let piece = self.moved_piece(m);
        let captured = self.captured_piece(m);

        debug_assert!(from.on_board());
        debug_assert!(to.on_board());

        let saved_key = self.key;

        if m.is_ep() {
            if side == Colour::WHITE {
                self.clear_piece(to.sub(8));
            } else {
                self.clear_piece(to.add(8));
            }
        } else if m.is_castle() {
            match to {
                Square::C1 => self.move_piece(Square::A1, Square::D1),
                Square::C8 => self.move_piece(Square::A8, Square::D8),
                Square::G1 => self.move_piece(Square::H1, Square::F1),
                Square::G8 => self.move_piece(Square::H8, Square::F8),
                _ => {
                    panic!("Invalid castle move");
                }
            }
        }

        if self.ep_sq != Square::NO_SQUARE {
            hash_ep(&mut self.key, self.ep_sq);
        }

        // hash out the castling to insert it again after updating rights.
        hash_castling(&mut self.key, self.castle_perm);

        self.history.push(Undo {
            m,
            castle_perm: self.castle_perm,
            ep_square: self.ep_sq,
            fifty_move_counter: self.fifty_move_counter,
            capture: captured,
        });
        self.repetition_cache.push(saved_key);

        self.castle_perm &= unsafe { *CASTLE_PERM_MASKS.get_unchecked(from.index()) };
        self.castle_perm &= unsafe { *CASTLE_PERM_MASKS.get_unchecked(to.index()) };
        self.ep_sq = Square::NO_SQUARE;

        // reinsert the castling rights
        hash_castling(&mut self.key, self.castle_perm);

        self.fifty_move_counter += 1;

        if captured != Piece::EMPTY {
            self.clear_piece(to);
            self.fifty_move_counter = 0;
        }

        self.ply += 1;
        self.height += 1;

        if piece.piece_type() == PieceType::PAWN {
            self.fifty_move_counter = 0;
            if self.is_double_pawn_push(m) {
                if side == Colour::WHITE {
                    self.ep_sq = from.add(8);
                    debug_assert!(self.ep_sq.rank() == RANK_3);
                } else {
                    self.ep_sq = from.sub(8);
                    debug_assert!(self.ep_sq.rank() == RANK_6);
                }
                hash_ep(&mut self.key, self.ep_sq);
            }
        }

        if m.is_promo() {
            let promo = Piece::new(side, m.promotion_type());
            debug_assert!(promo.piece_type().legal_promo());
            self.clear_piece(from);
            self.add_piece(to, promo);
        } else {
            self.move_piece(from, to);
        }

        self.side = self.side.flip();
        hash_side(&mut self.key);

        #[cfg(debug_assertions)]
        self.check_validity().unwrap();

        // reversed in_check fn, as we have now swapped sides
        if self.in_check::<{ Self::THEM }>() {
            self.unmake_move_hce();
            return false;
        }

        true
    }

    pub fn unmake_move_hce(&mut self) {
        #[cfg(debug_assertions)]
        self.check_validity().unwrap();

        self.height -= 1;
        self.ply -= 1;

        let Undo { m, castle_perm, ep_square, fifty_move_counter, capture } =
            self.history.pop().expect("No move to unmake!");

        let from = m.from();
        let to = m.to();

        if self.ep_sq != Square::NO_SQUARE {
            hash_ep(&mut self.key, self.ep_sq);
        }

        // hash out the castling to insert it again after updating rights.
        hash_castling(&mut self.key, self.castle_perm);

        self.castle_perm = castle_perm;
        self.ep_sq = ep_square;
        self.fifty_move_counter = fifty_move_counter;

        if self.ep_sq != Square::NO_SQUARE {
            hash_ep(&mut self.key, self.ep_sq);
        }

        // reinsert the castling rights
        hash_castling(&mut self.key, self.castle_perm);

        self.side = self.side.flip();
        hash_side(&mut self.key);

        if m.is_ep() {
            if self.side == Colour::WHITE {
                self.add_piece(to.sub(8), Piece::BP);
            } else {
                self.add_piece(to.add(8), Piece::WP);
            }
        } else if m.is_castle() {
            match to {
                Square::C1 => self.move_piece(Square::D1, Square::A1),
                Square::C8 => self.move_piece(Square::D8, Square::A8),
                Square::G1 => self.move_piece(Square::F1, Square::H1),
                Square::G8 => self.move_piece(Square::F8, Square::H8),
                _ => {
                    panic!("Invalid castle move");
                }
            }
        }

        if m.is_promo() {
            let promotion = Piece::new(self.side, m.promotion_type());
            debug_assert!(promotion.piece_type().legal_promo());
            debug_assert_eq!(promotion.colour(), self.piece_at(to).colour());
            self.clear_piece(to);
            self.add_piece(from, if self.side == Colour::WHITE { Piece::WP } else { Piece::BP });
        } else {
            self.move_piece(to, from);
        }

        if capture != Piece::EMPTY {
            self.add_piece(to, capture);
        }

        let key = self.repetition_cache.pop().expect("No key to unmake!");
        debug_assert_eq!(key, self.key);

        #[cfg(debug_assertions)]
        self.check_validity().unwrap();
    }

    pub fn make_nullmove(&mut self) {
        #[cfg(debug_assertions)]
        self.check_validity().unwrap();
        debug_assert!(!self.in_check::<{ Self::US }>());

        self.history.push(Undo {
            m: Move::NULL,
            castle_perm: self.castle_perm,
            ep_square: self.ep_sq,
            fifty_move_counter: self.fifty_move_counter,
            capture: Piece::EMPTY,
        });

        if self.ep_sq != Square::NO_SQUARE {
            hash_ep(&mut self.key, self.ep_sq);
        }

        self.ep_sq = Square::NO_SQUARE;

        self.side = self.side.flip();
        self.ply += 1;
        self.height += 1;
        hash_side(&mut self.key);

        #[cfg(debug_assertions)]
        self.check_validity().unwrap();
    }

    pub fn unmake_nullmove(&mut self) {
        #[cfg(debug_assertions)]
        self.check_validity().unwrap();

        self.height -= 1;
        self.ply -= 1;

        if self.ep_sq != Square::NO_SQUARE {
            // this might be unreachable, but that's ok.
            // the branch predictor will hopefully figure it out.
            hash_ep(&mut self.key, self.ep_sq);
        }

        let Undo { m: _, castle_perm, ep_square, fifty_move_counter, capture } =
            self.history.pop().expect("No move to unmake!");

        debug_assert_eq!(capture, Piece::EMPTY);

        self.castle_perm = castle_perm;
        self.ep_sq = ep_square;
        self.fifty_move_counter = fifty_move_counter;

        if self.ep_sq != Square::NO_SQUARE {
            hash_ep(&mut self.key, self.ep_sq);
        }

        self.side = self.side.flip();
        hash_side(&mut self.key);

        #[cfg(debug_assertions)]
        self.check_validity().unwrap();
    }

    pub fn make_move_nnue(&mut self, m: Move, t: &mut ThreadData) -> bool {
        let piece_type = self.moved_piece(m).piece_type();
        let colour = self.turn();
        let capture = self.captured_piece(m);
        let res = self.make_move_hce(m);
        if !res {
            return false;
        }
        let from = m.from();
        let to = m.to();
        t.nnue.push_acc();
        if m.is_ep() {
            let ep_sq = if colour == Colour::WHITE { to.sub(8) } else { to.add(8) };
            t.nnue.efficiently_update_manual::<DEACTIVATE>(PieceType::PAWN, colour.flip(), ep_sq);
        } else if m.is_castle() {
            match to {
                Square::C1 => {
                    t.nnue.efficiently_update_from_move(PieceType::ROOK, colour, Square::A1, Square::D1);
                }
                Square::C8 => {
                    t.nnue.efficiently_update_from_move(PieceType::ROOK, colour, Square::A8, Square::D8);
                }
                Square::G1 => {
                    t.nnue.efficiently_update_from_move(PieceType::ROOK, colour, Square::H1, Square::F1);
                }
                Square::G8 => {
                    t.nnue.efficiently_update_from_move(PieceType::ROOK, colour, Square::H8, Square::F8);
                }
                _ => {
                    panic!("Invalid castle move");
                }
            }
        }

        if capture != Piece::EMPTY {
            t.nnue.efficiently_update_manual::<DEACTIVATE>(capture.piece_type(), colour.flip(), to);
        }

        if m.is_promo() {
            let promo = m.promotion_type();
            debug_assert!(promo.legal_promo());
            t.nnue.efficiently_update_manual::<DEACTIVATE>(PieceType::PAWN, colour, from);
            t.nnue.efficiently_update_manual::<ACTIVATE>(promo, colour, to);
        } else {
            t.nnue.efficiently_update_from_move(piece_type, colour, from, to);
        }

        true
    }

    pub fn unmake_move_nnue(&mut self, t: &mut ThreadData) {
        let m = self.history.last().unwrap().m;
        self.unmake_move_hce();
        let piece = self.moved_piece(m).piece_type();
        let from = m.from();
        let to = m.to();
        let colour = self.turn();
        t.nnue.pop_acc();
        if m.is_ep() {
            let ep_sq = if colour == Colour::WHITE { to.sub(8) } else { to.add(8) };
            t.nnue.update_pov_manual::<ACTIVATE>(PieceType::PAWN, colour.flip(), ep_sq);
        } else if m.is_castle() {
            match to {
                Square::C1 => t.nnue.update_pov_move(PieceType::ROOK, colour, Square::D1, Square::A1),
                Square::G8 => t.nnue.update_pov_move(PieceType::ROOK, colour, Square::F8, Square::H8),
                Square::C8 => t.nnue.update_pov_move(PieceType::ROOK, colour, Square::D8, Square::A8),
                Square::G1 => t.nnue.update_pov_move(PieceType::ROOK, colour, Square::F1, Square::H1),
                _ => {
                    panic!("Invalid castle move");
                }
            }
        }
        if m.is_promo() {
            let promo = m.promotion_type();
            debug_assert!(promo.legal_promo());
            t.nnue.update_pov_manual::<DEACTIVATE>(promo, colour, to);
            t.nnue.update_pov_manual::<ACTIVATE>(PieceType::PAWN, colour, from);
        } else {
            t.nnue.update_pov_move(piece, colour, to, from);
        }
        let capture = self.captured_piece(m);
        if capture != Piece::EMPTY {
            t.nnue.update_pov_manual::<ACTIVATE>(capture.piece_type(), colour.flip(), to);
        }
    }

    pub fn make_move<const USE_NNUE: bool>(&mut self, m: Move, t: &mut ThreadData) -> bool {
        if USE_NNUE {
            self.make_move_nnue(m, t)
        } else {
            self.make_move_hce(m)
        }
    }

    pub fn unmake_move<const USE_NNUE: bool>(&mut self, t: &mut ThreadData) {
        if USE_NNUE {
            self.unmake_move_nnue(t);
        } else {
            self.unmake_move_hce();
        }
    }

    pub fn last_move_was_nullmove(&self) -> bool {
        if let Some(Undo { m, .. }) = self.history.last() {
            m.is_null()
        } else {
            false
        }
    }

    /// Parses a move in the UCI format and returns a move or a reason why it couldn't be parsed.
    pub fn parse_uci(&self, uci: &str) -> Result<Move, MoveParseError> {
        use crate::errors::MoveParseError::{
            IllegalMove, InvalidFromSquareFile, InvalidFromSquareRank, InvalidLength,
            InvalidPromotionPiece, InvalidToSquareFile, InvalidToSquareRank,
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

        let from = Square::from_rank_file(san_bytes[1] - b'1', san_bytes[0] - b'a');
        let to = Square::from_rank_file(san_bytes[3] - b'1', san_bytes[2] - b'a');

        let mut list = MoveList::new();
        self.generate_moves(&mut list);

        let res = list
            .iter()
            .copied()
            .find(|&m| {
                m.from() == from
                    && m.to() == to
                    && (san_bytes.len() == 4
                        || m.safe_promotion_type().promo_char().unwrap() == san_bytes[4] as char)
            })
            .ok_or_else(|| IllegalMove(uci.to_string()));

        res
    }

    #[allow(clippy::too_many_lines)]
    pub fn parse_san(&mut self, san: &str) -> Result<Move, MoveParseError> {
        use crate::errors::MoveParseError::{AmbiguousSAN, IllegalMove, InvalidSAN};

        if ["O-O", "O-O+", "O-O#", "0-0", "0-0+", "0-0#"].contains(&san) {
            let mut ml = MoveList::new();
            self.generate_castling_moves(&mut ml);
            let m = ml
                .iter()
                .copied()
                .find(|&m| m.is_kingside_castling())
                .ok_or_else(|| IllegalMove(san.to_string()));
            return m;
        }
        if ["O-O-O", "O-O-O+", "O-O-O#", "0-0-0", "0-0-0+", "0-0-0#"].contains(&san) {
            let mut ml = MoveList::new();
            self.generate_castling_moves(&mut ml);
            let m = ml
                .iter()
                .copied()
                .find(|&m| m.is_queenside_castling())
                .ok_or_else(|| IllegalMove(san.to_string()));
            return m;
        }

        let regex = get_san_regex();
        let reg_match = regex.captures(san);
        if reg_match.is_none() {
            if ["--", "Z0", "0000", "@@@@"].contains(&san) {
                return Ok(Move::NULL);
            }
            return Err(InvalidSAN(san.to_string()));
        }
        let reg_match = reg_match.unwrap();

        let to_sq_name = reg_match.get(4).unwrap().as_str();
        let to_square = to_sq_name.parse::<Square>().unwrap();
        let to_bb = to_square.bitboard();
        let mut from_bb = BB_ALL;

        let promo = reg_match.get(5).map(|promo| {
            b"..NBRQ.".iter().position(|&c| c == *promo.as_str().as_bytes().last().unwrap()).unwrap()
        });
        if promo.is_some() {
            let legal_mask = if self.side == Colour::WHITE { BB_RANK_7 } else { BB_RANK_2 };
            from_bb &= legal_mask;
        }

        if let Some(file) = reg_match.get(2) {
            let fname = file.as_str().as_bytes()[0];
            let file = usize::from(fname - b'a');
            from_bb &= BB_FILES[file];
        }

        if let Some(rank) = reg_match.get(3) {
            let rname = rank.as_str().as_bytes()[0];
            let rank = usize::from(rname - b'1');
            from_bb &= BB_RANKS[rank];
        }

        if let Some(piece) = reg_match.get(1) {
            let piece_char = piece.as_str().as_bytes()[0];
            let piece_type = PieceType::from_symbol(piece_char).unwrap();
            let whitepbb = self.pieces.piece_bb(Piece::new(Colour::WHITE, piece_type));
            let blackpbb = self.pieces.piece_bb(Piece::new(Colour::BLACK, piece_type));
            from_bb &= whitepbb | blackpbb;
        } else {
            from_bb &= self.pieces.pawns::<true>() | self.pieces.pawns::<false>();
            if reg_match.get(2).is_none() {
                from_bb &= BB_FILES[to_square.file() as usize];
            }
        }

        if from_bb == 0 {
            return Err(IllegalMove(san.to_string()));
        }

        let mut ml = MoveList::new();
        self.generate_moves(&mut ml);

        let mut legal_move = None;
        for &m in ml.iter() {
            if !self.make_move_hce(m) {
                continue;
            }
            self.unmake_move_hce();
            let legal_promo_t = m.safe_promotion_type();
            if legal_promo_t.index() != promo.unwrap_or(0) {
                continue;
            }
            let m_from_bb = m.from().bitboard();
            let m_to_bb = m.to().bitboard();
            if (m_from_bb & from_bb) != 0 && (m_to_bb & to_bb) != 0 {
                if legal_move.is_some() {
                    return Err(AmbiguousSAN(san.to_string()));
                }
                legal_move = Some(m);
            }
        }
        legal_move.ok_or_else(|| IllegalMove(san.to_string()))
    }

    pub fn san(&mut self, m: Move) -> Option<String> {
        let to_sq = m.to();
        let is_capture = !self.piece_at(to_sq).is_empty();
        let moved_piece = self.piece_at(m.from());
        let piece_prefix = match moved_piece.piece_type() {
            PieceType::PAWN if !is_capture => "",
            PieceType::PAWN => &"abcdefgh"[m.from().file() as usize..=m.from().file() as usize],
            PieceType::KNIGHT => "N",
            PieceType::BISHOP => "B",
            PieceType::ROOK => "R",
            PieceType::QUEEN => "Q",
            PieceType::KING => "K",
            PieceType::NO_PIECE_TYPE => return None,
            _ => unreachable!(),
        };
        let possible_ambiguous_attackers = if moved_piece.piece_type() == PieceType::PAWN {
            0
        } else {
            bitboards::attacks_by_type(
                moved_piece.piece_type(), 
                to_sq,
                self.pieces.occupied(),
            ) & self.pieces.piece_bb(moved_piece)
        };
        let needs_disambiguation = possible_ambiguous_attackers.count_ones() > 1 && moved_piece.piece_type() != PieceType::PAWN;
        let from_file = BB_FILES[m.from().file() as usize];
        let from_rank = BB_RANKS[m.from().rank() as usize];
        let can_be_disambiguated_by_file = (possible_ambiguous_attackers & from_file).count_ones() == 1;
        let can_be_disambiguated_by_rank = (possible_ambiguous_attackers & from_rank).count_ones() == 1;
        let needs_both = !can_be_disambiguated_by_file && !can_be_disambiguated_by_rank;
        let must_be_disambiguated_by_file = needs_both || can_be_disambiguated_by_file;
        let must_be_disambiguated_by_rank = needs_both || (can_be_disambiguated_by_rank && !can_be_disambiguated_by_file);
        let disambiguator1 = if needs_disambiguation && must_be_disambiguated_by_file {
            &"abcdefgh"[m.from().file() as usize..=m.from().file() as usize]
        } else {
            ""
        };
        let disambiguator2 = if needs_disambiguation && must_be_disambiguated_by_rank {
            &"12345678"[m.from().rank() as usize..=m.from().rank() as usize]
        } else {
            ""
        };
        let capture_sigil = if is_capture { "x" } else { "" };
        let promo_str = match m.safe_promotion_type() {
            PieceType::KNIGHT => "=N",
            PieceType::BISHOP => "=B",
            PieceType::ROOK => "=R",
            PieceType::QUEEN => "=Q",
            PieceType::NO_PIECE_TYPE => "",
            _ => unreachable!(),
        };
        let check_char = match self.gives(m) {
            CheckState::None => "",
            CheckState::Check => "+",
            CheckState::Checkmate => "#",
        };
        let san = format!("{piece_prefix}{disambiguator1}{disambiguator2}{capture_sigil}{to_sq}{promo_str}{check_char}");
        Some(san)
    }

    pub fn gives(&mut self, m: Move) -> CheckState {
        if !self.make_move_hce(m) {
            return CheckState::None;
        }
        let gives_check = self.in_check::<{ Self::US }>();
        if gives_check {
            let mut ml = MoveList::new();
            self.generate_moves(&mut ml);
            for &m in ml.iter() {
                if !self.make_move_hce(m) {
                    continue;
                }
                // we found a legal move, so m does not give checkmate.
                self.unmake_move_hce();
                self.unmake_move_hce();
                return CheckState::Check;
            }
            // we didn't return, so there were no legal moves,
            // so m gives checkmate.
            self.unmake_move_hce();
            return CheckState::Checkmate;
        }
        self.unmake_move_hce();
        CheckState::None
    }

    /// Has the current position occurred before in the current game?
    pub fn is_repetition(&self) -> bool {
        for (key, undo) in
            self.repetition_cache.iter().rev().zip(self.history.iter().rev()).skip(1).step_by(2)
        {
            if *key == self.key {
                return true;
            }
            // optimisation: if the fifty move counter was zeroed, then any prior positions will not be repetitions.
            if undo.fifty_move_counter == 0 {
                return false;
            }
        }
        false
    }

    /// Should we consider the current position a draw?
    pub fn is_draw(&self) -> bool {
        (self.fifty_move_counter >= 100 || self.is_repetition()) && self.height != 0
    }

    pub const fn num(&self, piece: Piece) -> u8 {
        #![allow(clippy::cast_possible_truncation)]
        self.pieces.piece_bb(piece).count_ones() as u8
    }

    pub const fn num_pt(&self, pt: PieceType) -> u8 {
        self.num(Piece::new(Colour::WHITE, pt)) + self.num(Piece::new(Colour::BLACK, pt))
    }

    pub fn regenerate_pv_line(&mut self, depth: i32, tt: TTView) {
        self.principal_variation.clear();

        while let ProbeResult::Hit(TTHit { tt_move, .. }) =
            tt.probe(self.key, 0, -INFINITY, INFINITY, MAX_DEPTH, true)
        {
            if self.principal_variation.len() < depth.try_into().unwrap()
                && self.is_legal(tt_move)
                && !self.is_draw()
            {
                self.make_move_hce(tt_move);
                self.principal_variation.push(tt_move);
            } else {
                break;
            }
        }

        for _ in 0..self.principal_variation.len() {
            self.unmake_move_hce();
        }
    }

    fn get_pv_line(&self) -> &[Move] {
        &self.principal_variation
    }

    pub fn print_pv(&self) {
        for &m in self.get_pv_line() {
            print!("{m} ");
        }
    }

    pub fn pv_san(&mut self) -> String {
        #![allow(clippy::unnecessary_to_owned)] // needed for ownership stuff
        let mut out = String::new();
        let mut moves_made = 0;
        for m in self.get_pv_line().to_vec() {
            write!(out, "{} ", self.san(m).unwrap_or_else(|| "???".to_string())).unwrap();
            self.make_move_hce(m);
            moves_made += 1;
        }
        for _ in 0..moves_made {
            self.unmake_move_hce();
        }
        out
    }

    pub fn predicted_moves_left(&self) -> u64 {
        #![allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        static WEIGHTS: [f64; 6] = [
            0.322_629_598_871_821,
            -0.789_691_260_856_047_8,
            -0.001_186_089_580_333_777,
            -0.000_701_857_880_801_681_8,
            0.002_348_458_515_297_663_4,
            88.189_977_391_814_35,
        ];
        let half_moves_since_game_start = self.ply;
        let phase = self.phase();
        let (a, b) = (f64::from(phase), half_moves_since_game_start as f64);
        let features = [a, b, a * b, a * a, b * b, 1.0];
        let prediction = WEIGHTS.iter().zip(features.iter()).map(|(w, f)| w * f).sum::<f64>();
        prediction.round().max(2.0) as u64
    }

    pub fn legal_moves(&mut self) -> Vec<Move> {
        let mut move_list = MoveList::new();
        self.generate_moves(&mut move_list);
        let mut legal_moves = Vec::new();
        for &m in move_list.iter() {
            if self.make_move_hce(m) {
                self.unmake_move_hce();
                legal_moves.push(m);
            }
        }
        legal_moves
    }

    pub const fn fifty_move_counter(&self) -> u8 {
        self.fifty_move_counter
    }
}

impl Default for Board {
    fn default() -> Self {
        let mut out = Self::new();
        out.set_startpos();
        out
    }
}

impl Display for Board {
    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
        for rank in (Rank::RANK_1..=Rank::RANK_8).rev() {
            write!(f, "{} ", rank + 1)?;
            for file in File::FILE_A..=File::FILE_H {
                let sq = Square::from_rank_file(rank, file);
                let piece = self.piece_at(sq);
                write!(f, "{piece} ")?;
            }
            writeln!(f)?;
        }

        writeln!(f, "  a b c d e f g h")?;
        writeln!(f, "FEN: {}", self.fen())?;

        Ok(())
    }
}

mod tests {
    #[test]
    fn read_fen_validity() {
        use super::Board;

        crate::magic::initialise();

        let mut board_1 = Board::new();
        board_1.set_from_fen(Board::STARTING_FEN).expect("setfen failed.");
        board_1.check_validity().unwrap();

        let board_2 = Board::from_fen(Board::STARTING_FEN).expect("setfen failed.");
        board_2.check_validity().unwrap();

        assert_eq!(board_1, board_2);
    }

    #[test]
    fn fen_round_trip() {
        use crate::board::Board;
        use std::{
            fs::File,
            io::{BufRead, BufReader},
        };
        crate::magic::initialise();
        let fens = BufReader::new(File::open("epds/perftsuite.epd").unwrap())
            .lines()
            .map(|l| l.unwrap().split_once(';').unwrap().0.trim().to_owned())
            .collect::<Vec<_>>();
        let mut board = Board::new();
        for fen in fens {
            board.set_from_fen(&fen).expect("setfen failed.");
            let fen_2 = board.fen();
            assert_eq!(fen, fen_2);
        }
    }

    #[test]
    fn test_num_pt() {
        use super::Board;
        use crate::piece::Piece;
        use crate::piece::PieceType;
        let board =
            Board::from_fen("rnbqkbnr/pppppppp/1n1q2n1/8/8/RR3B1R/PPPPPPP1/RNBQKBNR w KQkq - 0 1")
                .unwrap();

        for ((p1, p2), pt) in Piece::all().take(6)
            .into_iter()
            .zip(Piece::all().skip(6).take(6))
            .zip(PieceType::all())
        {
            assert_eq!(board.num_pt(pt), board.num(p1) + board.num(p2));
        }
    }

    #[test]
    fn test_san() {
        use super::Board;
        use std::{
            fs::File,
            io::{BufRead, BufReader},
        };
        
        let f = File::open("epds/perftsuite.epd").unwrap();
        let mut pos = Board::new();
        let mut ml = crate::board::movegen::MoveList::new();
        for line in BufReader::new(f).lines() {
            let line = line.unwrap();
            let fen = line.split_once(';').unwrap().0.trim();
            pos.set_from_fen(fen).unwrap();
            pos.generate_moves(&mut ml);
            for &m in ml.iter() {
                if !pos.make_move_hce(m) {
                    continue;
                }
                pos.unmake_move_hce();
                let san_repr = pos.san(m);
                let san_repr = san_repr.unwrap();
                let parsed_move = pos.parse_san(&san_repr);
                assert_eq!(
                    parsed_move, 
                    Ok(m), 
                    "{san_repr} != {m} in fen {}",
                    pos.fen(),
                );
            }
        }
    }
}
