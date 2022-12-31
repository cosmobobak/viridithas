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
        colour_of, make_piece, type_of, Colour, File,
        Rank::{self, RANK_3, RANK_6},
        Square, Undo, BB, BISHOP, BK, BKCA, BLACK, BN, BP, BQ, BQCA, BR, INFINITY, KING, KNIGHT,
        MAX_DEPTH, PAWN, PIECE_EMPTY, ROOK, WB, WHITE, WK, WKCA, WN, WP, WQ, WQCA, WR,
    },
    errors::{FenParseError, MoveParseError},
    lookups::{piece_char, PIECE_BIG, PIECE_MAJ, PROMO_CHAR_LOOKUP},
    macros,
    makemove::{hash_castling, hash_ep, hash_piece, hash_side, CASTLE_PERM_MASKS},
    nnue::{ACTIVATE, DEACTIVATE},
    piecesquaretable::pst_value,
    threadlocal::ThreadData,
    transpositiontable::{ProbeResult, TTHit, TranspositionTableView},
    validate::{piece_type_valid, piece_valid, side_valid},
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

#[derive(Clone)]
pub struct Board {
    /// The bitboards of all the pieces on the board.
    pub(crate) pieces: BitBoard,
    /// An array to accelerate piece_at().
    piece_array: [u8; 64],
    /// The side to move.
    side: u8,
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

impl PartialEq for Board {
    fn eq(&self, other: &Self) -> bool {
        #[cfg(debug_assertions)]
        self.check_validity().unwrap();
        #[cfg(debug_assertions)]
        other.check_validity().unwrap();
        self.pieces == other.pieces
            && self.side == other.side
            && self.ep_sq == other.ep_sq
            && self.fifty_move_counter == other.fifty_move_counter
            && self.castle_perm == other.castle_perm
            && self.key == other.key
    }
}

impl Board {
    pub const STARTING_FEN: &'static str =
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    pub fn new() -> Self {
        let mut out = Self {
            pieces: BitBoard::NULL,
            piece_array: [0; 64],
            side: 0,
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

    pub fn king_sq(&self, side: u8) -> Square {
        debug_assert!(side == WHITE || side == BLACK);
        debug_assert_eq!(self.pieces.king::<true>().count_ones(), 1);
        debug_assert_eq!(self.pieces.king::<false>().count_ones(), 1);
        let sq = match side {
            WHITE => self.pieces.king::<true>().first_square(),
            BLACK => self.pieces.king::<false>().first_square(),
            _ => unsafe { macros::inconceivable!() },
        };
        debug_assert!(sq < Square::NO_SQUARE);
        debug_assert_eq!(colour_of(self.piece_at(sq)), side);
        debug_assert_eq!(type_of(self.piece_at(sq)), KING);
        sq
    }

    pub const US: u8 = 0;
    pub const THEM: u8 = 1;
    pub fn in_check<const SIDE: u8>(&self) -> bool {
        if SIDE == Self::US {
            let king_sq = self.king_sq(self.side);
            self.sq_attacked(king_sq, self.side ^ 1)
        } else {
            let king_sq = self.king_sq(self.side ^ 1);
            self.sq_attacked(king_sq, self.side)
        }
    }

    #[allow(dead_code)]
    pub fn maybe_gives_check(&self, mov: Move) -> bool {
        let piece = self.piece_at(mov.from());
        let piece_type = type_of(piece);
        let stm = self.side;

        let opponent_king = self.king_sq(stm ^ 1);
        let blockers = self.pieces.occupied() ^ mov.bitboard();

        // direct check:
        let attacks_from_dest = if piece_type == PAWN {
            if stm == WHITE {
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
        let ortho = bitboards::attacks::<ROOK>(opponent_king, blockers);
        let diag = bitboards::attacks::<BISHOP>(opponent_king, blockers);
        let our_orthos = if stm == WHITE {
            self.pieces.rookqueen::<true>()
        } else {
            self.pieces.rookqueen::<false>()
        };
        let our_diags = if stm == WHITE {
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

    pub const fn turn(&self) -> u8 {
        self.side
    }

    pub fn generate_pos_key(&self) -> u64 {
        #![allow(clippy::cast_possible_truncation)]
        let mut key = 0;
        for (sq, &piece) in self.piece_array.iter().enumerate() {
            let sq = Square::new(sq.try_into().unwrap());
            if piece != PIECE_EMPTY {
                debug_assert!((WP..=BK).contains(&piece));
                hash_piece(&mut key, piece, sq);
            }
        }

        if self.side == WHITE {
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
        self.piece_array = [0; 64];
        self.big_piece_counts.fill(0);
        self.major_piece_counts.fill(0);
        self.minor_piece_counts.fill(0);
        self.material.fill(S(0, 0));
        self.side = Colour::White as u8;
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
                b'P' => piece = WP,
                b'R' => piece = WR,
                b'N' => piece = WN,
                b'B' => piece = WB,
                b'Q' => piece = WQ,
                b'K' => piece = WK,
                b'p' => piece = BP,
                b'r' => piece = BR,
                b'n' => piece = BN,
                b'b' => piece = BB,
                b'q' => piece = BQ,
                b'k' => piece = BK,
                b'1'..=b'8' => {
                    piece = PIECE_EMPTY;
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
                if piece != PIECE_EMPTY {
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
            self.material[WHITE as usize].0, self.material[BLACK as usize].0,
            "mg_material is not equal, white: {}, black: {}",
            self.material[WHITE as usize].0, self.material[BLACK as usize].0
        );
        debug_assert_eq!(
            self.material[WHITE as usize].1, self.material[BLACK as usize].1,
            "eg_material is not equal, white: {}, black: {}",
            self.material[WHITE as usize].1, self.material[BLACK as usize].1
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
        for rank in (0..8).rev() {
            for file in 0..8 {
                let sq = Square::from_rank_file(rank, file);
                let piece = self.piece_at(sq);
                if piece == PIECE_EMPTY {
                    counter += 1;
                } else {
                    if counter != 0 {
                        write!(fen, "{counter}").unwrap();
                    }
                    counter = 0;
                    fen.push(piece_char(piece).unwrap());
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
        fen.push(if self.side == WHITE { 'w' } else { 'b' });
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
            None => return Err("FEN string is invalid, expected side part.".into()),
            Some([b'w']) => WHITE,
            Some([b'b']) => BLACK,
            Some(other) => {
                return Err(format!(
                    "FEN string is invalid, expected side to be 'w' or 'b', got \"{}\"",
                    std::str::from_utf8(other).unwrap_or("<invalid utf8>")
                ))
            }
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
                if self.side == BLACK {
                    self.ply += 1;
                }
            }
        }

        Ok(())
    }

    /// Determines if `sq` is attacked by `side`
    pub fn sq_attacked(&self, sq: Square, side: u8) -> bool {
        if side == WHITE {
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
        let knight_attacks_from_this_square = bitboards::attacks::<KNIGHT>(sq, BB_NONE);
        if our_knights & knight_attacks_from_this_square != BB_NONE {
            return true;
        }

        // bishops, queens
        let diag_attacks_from_this_square = bitboards::attacks::<BISHOP>(sq, blockers);
        if our_diags & diag_attacks_from_this_square != BB_NONE {
            return true;
        }

        // rooks, queens
        let ortho_attacks_from_this_square = bitboards::attacks::<ROOK>(sq, blockers);
        if our_orthos & ortho_attacks_from_this_square != BB_NONE {
            return true;
        }

        // king
        let king_attacks_from_this_square = bitboards::attacks::<KING>(sq, BB_NONE);
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
        let is_capture = captured_piece != PIECE_EMPTY;
        let is_pawn_double_push = self.is_double_pawn_push(m);

        if moved_piece == PIECE_EMPTY {
            return false;
        }

        if colour_of(moved_piece) != self.side {
            return false;
        }

        if is_capture && colour_of(captured_piece) == self.side {
            return false;
        }

        if type_of(moved_piece) != PAWN && (is_pawn_double_push || m.is_ep() || m.is_promo()) {
            return false;
        }

        if type_of(moved_piece) != KING && m.is_castle() {
            return false;
        }

        if is_capture && is_pawn_double_push {
            return false;
        }

        if m.is_castle() {
            return self.is_pseudo_legal_castling(to);
        }

        if type_of(moved_piece) == PAWN {
            let should_be_promoting = to > Square::H7 || to < Square::A2;
            if should_be_promoting && !m.is_promo() {
                return false;
            }
            if m.is_ep() {
                return to == self.ep_sq;
            } else if is_pawn_double_push {
                let one_forward = from.pawn_push(self.side);
                return self.piece_at(one_forward) == PIECE_EMPTY
                    && to == one_forward.pawn_push(self.side);
            } else if !is_capture {
                return to == from.pawn_push(self.side) && captured_piece == PIECE_EMPTY;
            }
            // pawn capture
            if self.side == WHITE {
                return pawn_attacks::<true>(from.bitboard()) & to.bitboard() != 0;
            }
            return pawn_attacks::<false>(from.bitboard()) & to.bitboard() != 0;
        }

        to.bitboard()
            & bitboards::attacks_by_type(type_of(moved_piece), from, self.pieces.occupied())
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
            (WHITE, Square::C1) => (self.castle_perm & WQCA, WQ_FREESPACE, Square::E1, Square::D1),
            (WHITE, Square::G1) => (self.castle_perm & WKCA, WK_FREESPACE, Square::E1, Square::F1),
            (BLACK, Square::C8) => (self.castle_perm & BQCA, BQ_FREESPACE, Square::E8, Square::D8),
            (BLACK, Square::G8) => (self.castle_perm & BKCA, BK_FREESPACE, Square::E8, Square::F8),
            (_, _) => return false,
        };
        if target_castling_perm == 0 {
            return false;
        }
        if occupied & occ_mask != 0 {
            return false;
        }
        if self.sq_attacked(from_sq, 1 ^ self.side) {
            return false;
        }
        if self.sq_attacked(next_sq, 1 ^ self.side) {
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
            piece_valid(piece),
            "Invalid piece at {}: {}, board {}, last move was {}",
            sq,
            piece,
            self.fen(),
            self.history.last().map_or("None".to_string(), |h| h.m.to_string())
        );
        let piece_type = type_of(piece);
        debug_assert!(piece_type_valid(piece_type));

        self.pieces.clear_piece_at(sq, piece);

        let colour = colour_of(piece);

        hash_piece(&mut self.key, piece, sq);

        *self.piece_at_mut(sq) = PIECE_EMPTY;
        self.material[colour as usize] -= get_eval_params().piece_values[piece as usize];
        self.pst_vals -= pst_value(piece, sq, &get_eval_params().piece_square_tables);

        if PIECE_BIG[piece as usize] {
            self.big_piece_counts[colour as usize] -= 1;
            if PIECE_MAJ[piece as usize] {
                self.major_piece_counts[colour as usize] -= 1;
            } else {
                self.minor_piece_counts[colour as usize] -= 1;
            }
        }
    }

    fn add_piece(&mut self, sq: Square, piece: u8) {
        debug_assert!(piece_valid(piece));
        debug_assert!(sq.on_board());

        self.pieces.set_piece_at(sq, piece);

        let colour = colour_of(piece);

        hash_piece(&mut self.key, piece, sq);

        *self.piece_at_mut(sq) = piece;
        self.material[colour as usize] += get_eval_params().piece_values[piece as usize];
        self.pst_vals += pst_value(piece, sq, &get_eval_params().piece_square_tables);

        if PIECE_BIG[piece as usize] {
            self.big_piece_counts[colour as usize] += 1;
            if PIECE_MAJ[piece as usize] {
                self.major_piece_counts[colour as usize] += 1;
            } else {
                self.minor_piece_counts[colour as usize] += 1;
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

        *self.piece_at_mut(from) = PIECE_EMPTY;
        *self.piece_at_mut(to) = piece_moved;
        self.pst_vals -= pst_value(piece_moved, from, &get_eval_params().piece_square_tables);
        self.pst_vals += pst_value(piece_moved, to, &get_eval_params().piece_square_tables);
    }

    /// Gets the piece that will be moved by the given move.
    pub fn moved_piece(&self, m: Move) -> u8 {
        debug_assert!(m.from().on_board());
        unsafe { *self.piece_array.get_unchecked(m.from().index()) }
    }

    /// Gets the piece that will be captured by the given move.
    pub fn captured_piece(&self, m: Move) -> u8 {
        debug_assert!(m.to().on_board());
        unsafe { *self.piece_array.get_unchecked(m.to().index()) }
    }

    /// Determines whether this move would be a capture in the current position.
    pub fn is_capture(&self, m: Move) -> bool {
        debug_assert!(m.from().on_board());
        debug_assert!(m.to().on_board());
        self.captured_piece(m) != PIECE_EMPTY
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
        type_of(piece_moved) == PAWN
    }

    /// Determines whether this move would be tactical in the current position.
    pub fn is_tactical(&self, m: Move) -> bool {
        m.is_promo() || m.is_ep() || self.is_capture(m)
    }

    /// Gets the piece at the given square.
    pub fn piece_at(&self, sq: Square) -> u8 {
        debug_assert!(sq.on_board());
        unsafe { *self.piece_array.get_unchecked(sq.index()) }
    }

    /// Gets a mutable reference to the piece at the given square.
    pub fn piece_at_mut(&mut self, sq: Square) -> &mut u8 {
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
        debug_assert!(side_valid(side));
        debug_assert!(piece_valid(piece), "piece: {piece:?}");
        debug_assert!(captured == PIECE_EMPTY || piece_valid(captured), "captured: {captured:?}");

        let saved_key = self.key;

        if m.is_ep() {
            if side == WHITE {
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

        if captured != PIECE_EMPTY {
            debug_assert!(piece_valid(captured));
            self.clear_piece(to);
            self.fifty_move_counter = 0;
        }

        self.ply += 1;
        self.height += 1;

        if piece == WP || piece == BP {
            self.fifty_move_counter = 0;
            if self.is_double_pawn_push(m) {
                if side == WHITE {
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
            let promo = make_piece(side, m.promotion_type());
            debug_assert!(piece_valid(promo));
            debug_assert!(promo != WP && promo != BP && promo != WK && promo != BK);
            self.clear_piece(from);
            self.add_piece(to, promo);
        } else {
            self.move_piece(from, to);
        }

        self.side ^= 1;
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

        self.side ^= 1;
        hash_side(&mut self.key);

        if m.is_ep() {
            if self.side == WHITE {
                self.add_piece(to.sub(8), BP);
            } else {
                self.add_piece(to.add(8), WP);
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
            let promotion = make_piece(self.side, m.promotion_type());
            debug_assert!(piece_valid(promotion));
            debug_assert!(promotion != WP && promotion != BP && promotion != WK && promotion != BK);
            debug_assert_eq!(colour_of(promotion), colour_of(self.piece_at(to)));
            self.clear_piece(to);
            self.add_piece(from, if self.side == WHITE { WP } else { BP });
        } else {
            self.move_piece(to, from);
        }

        if capture != PIECE_EMPTY {
            debug_assert!(piece_valid(capture));
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
            capture: PIECE_EMPTY,
        });

        if self.ep_sq != Square::NO_SQUARE {
            hash_ep(&mut self.key, self.ep_sq);
        }

        self.ep_sq = Square::NO_SQUARE;

        self.side ^= 1;
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

        debug_assert_eq!(capture, PIECE_EMPTY);

        self.castle_perm = castle_perm;
        self.ep_sq = ep_square;
        self.fifty_move_counter = fifty_move_counter;

        if self.ep_sq != Square::NO_SQUARE {
            hash_ep(&mut self.key, self.ep_sq);
        }

        self.side ^= 1;
        hash_side(&mut self.key);

        #[cfg(debug_assertions)]
        self.check_validity().unwrap();
    }

    pub fn make_move_nnue(&mut self, m: Move, t: &mut ThreadData) -> bool {
        let piece = type_of(self.moved_piece(m));
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
            let ep_sq = if colour == WHITE { to.sub(8) } else { to.add(8) };
            t.nnue.efficiently_update_manual::<DEACTIVATE>(PAWN, 1 ^ colour, ep_sq);
        } else if m.is_castle() {
            match to {
                Square::C1 => {
                    t.nnue.efficiently_update_from_move(ROOK, colour, Square::A1, Square::D1);
                }
                Square::C8 => {
                    t.nnue.efficiently_update_from_move(ROOK, colour, Square::A8, Square::D8);
                }
                Square::G1 => {
                    t.nnue.efficiently_update_from_move(ROOK, colour, Square::H1, Square::F1);
                }
                Square::G8 => {
                    t.nnue.efficiently_update_from_move(ROOK, colour, Square::H8, Square::F8);
                }
                _ => {
                    panic!("Invalid castle move");
                }
            }
        }

        if capture != PIECE_EMPTY {
            t.nnue.efficiently_update_manual::<DEACTIVATE>(type_of(capture), 1 ^ colour, to);
        }

        if m.is_promo() {
            let promo = m.promotion_type();
            debug_assert!(promo != WP && promo != BP && promo != WK && promo != BK);
            t.nnue.efficiently_update_manual::<DEACTIVATE>(PAWN, colour, from);
            t.nnue.efficiently_update_manual::<ACTIVATE>(promo, colour, to);
        } else {
            t.nnue.efficiently_update_from_move(piece, colour, from, to);
        }

        true
    }

    pub fn unmake_move_nnue(&mut self, t: &mut ThreadData) {
        let m = self.history.last().unwrap().m;
        self.unmake_move_hce();
        let piece = type_of(self.moved_piece(m));
        let from = m.from();
        let to = m.to();
        let colour = self.turn();
        t.nnue.pop_acc();
        if m.is_ep() {
            let ep_sq = if colour == WHITE { to.sub(8) } else { to.add(8) };
            t.nnue.update_pov_manual::<ACTIVATE>(PAWN, 1 ^ colour, ep_sq);
        } else if m.is_castle() {
            match to {
                Square::C1 => t.nnue.update_pov_move(ROOK, colour, Square::D1, Square::A1),
                Square::G8 => t.nnue.update_pov_move(ROOK, colour, Square::F8, Square::H8),
                Square::C8 => t.nnue.update_pov_move(ROOK, colour, Square::D8, Square::A8),
                Square::G1 => t.nnue.update_pov_move(ROOK, colour, Square::F1, Square::H1),
                _ => {
                    panic!("Invalid castle move");
                }
            }
        }
        if m.is_promo() {
            let promo = m.promotion_type();
            debug_assert!(piece_valid(promo));
            debug_assert!(promo != WP && promo != BP && promo != WK && promo != BK);
            t.nnue.update_pov_manual::<DEACTIVATE>(promo, colour, to);
            t.nnue.update_pov_manual::<ACTIVATE>(PAWN, colour, from);
        } else {
            t.nnue.update_pov_move(piece, colour, to, from);
        }
        let capture = self.captured_piece(m);
        if capture != PIECE_EMPTY {
            t.nnue.update_pov_manual::<ACTIVATE>(type_of(capture), 1 ^ colour, to);
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
                        || PROMO_CHAR_LOOKUP[m.safe_promotion_type() as usize] == san_bytes[4])
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
            b"pnbrqk".iter().position(|&c| c == *promo.as_str().as_bytes().last().unwrap()).unwrap()
        });
        if promo.is_some() {
            let legal_mask = if self.side == WHITE { BB_RANK_7 } else { BB_RANK_2 };
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
            let piece = piece.as_str().as_bytes()[0];
            let piece: u8 =
                b".PNBRQK".iter().position(|&c| c == piece).unwrap().try_into().unwrap();
            let whitepbb = self.pieces.piece_bb(piece);
            let blackpbb = self.pieces.piece_bb(piece + 6);
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
            let promotion = if m.is_promo() {
                make_piece(self.side, m.safe_promotion_type())
            } else {
                PIECE_EMPTY
            };
            if promotion as usize != promo.unwrap_or(0) {
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

    pub const fn num(&self, piece: u8) -> u8 {
        #![allow(clippy::cast_possible_truncation)]
        self.pieces.piece_bb(piece).count_ones() as u8
    }

    pub const fn num_pt(&self, pt: u8) -> u8 {
        self.num(pt) + self.num(pt + 6)
    }

    pub const fn num_ct<const PIECE: u8>(&self) -> u8 {
        #![allow(clippy::cast_possible_truncation)]
        self.pieces.piece_bb(PIECE).count_ones() as u8
    }

    pub const fn num_pt_ct<const PIECE_TYPE: u8>(&self) -> u8 {
        #![allow(clippy::cast_possible_truncation)]
        self.pieces.piece_bb(PIECE_TYPE).count_ones() as u8
            + self.pieces.piece_bb(PIECE_TYPE + 6).count_ones() as u8
    }

    pub fn regenerate_pv_line(&mut self, depth: i32, tt: TranspositionTableView) {
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
        println!();
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
        writeln!(f, "Game Board:")?;

        for rank in ((Rank::RANK_1)..=(Rank::RANK_8)).rev() {
            write!(f, "{} ", rank + 1)?;
            for file in (File::FILE_A)..=(File::FILE_H) {
                let sq = Square::from_rank_file(rank, file);
                let piece = self.piece_at(sq);
                write!(f, "{} ", piece_char(piece).unwrap())?;
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
        let fens = BufReader::new(File::open("perftsuite.epd").unwrap())
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
        use crate::definitions::{
            BB, BISHOP, BK, BN, BP, BQ, BR, KING, KNIGHT, PAWN, QUEEN, ROOK, WB, WK, WN, WP, WQ, WR,
        };
        let board =
            Board::from_fen("rnbqkbnr/pppppppp/1n1q2n1/8/8/RR3B1R/PPPPPPP1/RNBQKBNR w KQkq - 0 1")
                .unwrap();

        for ((p1, p2), pt) in [WP, WN, WB, WR, WQ, WK]
            .into_iter()
            .zip([BP, BN, BB, BR, BQ, BK])
            .zip([PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING])
        {
            assert_eq!(board.num_pt(pt), board.num(p1) + board.num(p2));
        }
    }
}
