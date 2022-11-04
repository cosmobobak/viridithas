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
    board::movegen::{
        bitboards::{
            self, pawn_attacks, BB_ALL, BB_FILES, BB_NONE, BB_RANKS, BB_RANK_2, BB_RANK_7,
        },
        MoveList,
    },
    chessmove::Move,
    definitions::{
        colour_of,
        depth::Depth,
        type_of, Colour, File,
        Rank::{self, RANK_3, RANK_6},
        Square,
        Undo, BB, BISHOP, BK, BKCA, BLACK, BN, BP, BQ, BQCA, BR, INFINITY, KING,
        KNIGHT, MAX_DEPTH, PAWN, PIECE_EMPTY, ROOK, WB, WHITE, WK, WKCA, WN, WP, WQ, WQCA, WR,
    },
    errors::{FenParseError, MoveParseError},
    historytable::{DoubleHistoryTable, HistoryTable, MoveTable},
    lookups::{
        piece_char, PIECE_BIG, PIECE_MAJ, PROMO_CHAR_LOOKUP,
    },
    macros,
    makemove::{hash_castling, hash_ep, hash_piece, hash_side, CASTLE_PERM_MASKS},
    nnue::{ACTIVATE, DEACTIVATE},
    piecelist::PieceList,
    piecesquaretable::pst_value,
    search::{self, parameters::SearchParams, AspirationWindow},
    searchinfo::SearchInfo,
    threadlocal::ThreadData,
    transpositiontable::{HFlag, ProbeResult, TTHit, TranspositionTable},
    uci::format_score,
    validate::{piece_type_valid, piece_valid, side_valid},
};

const UPPER_BOUND: u8 = 1;
const LOWER_BOUND: u8 = 2;
const EXACT: u8 = 3;

use self::{
    evaluation::{is_mate_score, score::S},
    movegen::bitboards::BitBoard,
};

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

pub struct Board {
    /// The bitboards of all the pieces on the board.
    pub(crate) pieces: BitBoard,
    /// An array to accelerate piece_at().
    piece_array: [u8; 64],
    /// Piece lists that allow pieces to be located quickly.
    piece_lists: [PieceList; 13],
    /// The side to move.
    side: u8,
    /// The en passant square.
    ep_sq: Square,
    /// The castling permissions.
    castle_perm: u8,
    /// The number of half moves made since the last capture or pawn advance.
    fifty_move_counter: u8,
    /// The number of half made since the start of the game.
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

    history_table: HistoryTable,
    killer_move_table: [[Move; 2]; MAX_DEPTH.ply_to_horizon()],
    counter_move_table: MoveTable,
    followup_history: DoubleHistoryTable,
    tt: TranspositionTable,
    hashtable_bytes: usize,

    eparams: evaluation::parameters::EvalParams,
    pub sparams: SearchParams,
    pub lmr_table: search::LMRTable,

    movegen_ready: bool,
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
            piece_lists: [PieceList::new(); 13],
            principal_variation: Vec::new(),
            history_table: HistoryTable::new(),
            killer_move_table: [[Move::NULL; 2]; MAX_DEPTH.ply_to_horizon()],
            counter_move_table: MoveTable::new(),
            followup_history: DoubleHistoryTable::new(),
            pst_vals: S(0, 0),
            tt: TranspositionTable::new(),
            hashtable_bytes: 4 * 1024 * 1024,
            eparams: evaluation::parameters::EvalParams::default(),
            sparams: SearchParams::default(),
            lmr_table: search::LMRTable::new(&SearchParams::default()),
            movegen_ready: false,
        };
        out.reset();
        out
    }

    pub fn set_search_params(&mut self, config: SearchParams) {
        self.sparams = config;
        self.lmr_table = search::LMRTable::new(&self.sparams);
    }

    pub fn tt_store<const ROOT: bool>(&mut self, best_move: Move, score: i32, flag: HFlag, depth: Depth) {
        self.tt.store::<ROOT>(self.key, self.height, best_move, score, flag, depth);
    }

    pub fn tt_probe<const ROOT: bool>(&self, alpha: i32, beta: i32, depth: Depth) -> ProbeResult {
        self.tt.probe::<ROOT>(self.key, self.height, alpha, beta, depth)
    }

    /// Nuke the transposition table.
    /// This wipes all entries in the table, don't call it during a search.
    pub fn clear_tt(&mut self) {
        self.tt.resize(self.hashtable_bytes);
    }

    pub const fn ep_sq(&self) -> Square {
        self.ep_sq
    }

    pub fn set_hash_size(&mut self, mb: usize) {
        self.hashtable_bytes = mb * 1024 * 1024;
        self.clear_tt();
    }

    pub fn king_sq(&self, side: u8) -> Square {
        debug_assert!(side == WHITE || side == BLACK);
        debug_assert_eq!(self.pieces.king::<true>().count_ones(), 1);
        debug_assert_eq!(self.pieces.king::<false>().count_ones(), 1);
        debug_assert_eq!(self.piece_lists[WK as usize].len(), 1);
        debug_assert_eq!(self.piece_lists[BK as usize].len(), 1);
        let sq = match side {
            WHITE => self.piece_lists[WK as usize][0],
            BLACK => self.piece_lists[BK as usize][0],
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

        key
    }

    pub fn reset(&mut self) {
        self.pieces.reset();
        self.piece_array = [0; 64];
        self.big_piece_counts.fill(0);
        self.major_piece_counts.fill(0);
        self.minor_piece_counts.fill(0);
        self.material.fill(S(0, 0));
        self.piece_lists.iter_mut().for_each(PieceList::clear);
        self.side = Colour::Both as u8;
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
            return Err(format!("FEN string is not ASCII: {}", fen));
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
                    fen.push(piece_char(piece).unwrap() as char);
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
                let file = ep_sq[0] as u8 - b'a';
                let rank = ep_sq[1] as u8 - b'1';
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

        if moved_piece == PIECE_EMPTY {
            return false;
        }

        if colour_of(moved_piece) != self.side {
            return false;
        }

        let captured_piece = self.piece_at(to);

        if colour_of(captured_piece) == self.side {
            return false;
        }

        if captured_piece != m.capture() {
            return false;
        }

        if m.is_castle() {
            return self.is_pseudo_legal_castling(to);
        }

        if type_of(moved_piece) == PAWN {
            if m.is_ep() {
                return to == self.ep_sq;
            } else if m.is_pawn_start() {
                let one_forward = from.pawn_push(self.side);
                return self.piece_at(one_forward) == PIECE_EMPTY && to == one_forward.pawn_push(self.side);
            } else if captured_piece == PIECE_EMPTY {
                return to == from.pawn_push(self.side) && self.piece_at(to) == PIECE_EMPTY;
            }
            // pawn capture
            if self.side == WHITE {
                return pawn_attacks::<true>(from.bitboard()) & self.pieces.their_pieces::<true>() != 0;
            }
            return pawn_attacks::<false>(from.bitboard()) & self.pieces.their_pieces::<false>() != 0;
        }

        to.bitboard() & bitboards::attacks_by_type(type_of(moved_piece), from, self.pieces.occupied()) != BB_NONE
    }

    pub fn is_pseudo_legal_castling(&self, to: Square) -> bool {
        const WK_FREESPACE: u64 = Square::F1.bitboard() | Square::G1.bitboard();
        const WQ_FREESPACE: u64 = Square::B1.bitboard() | Square::C1.bitboard() | Square::D1.bitboard();
        const BK_FREESPACE: u64 = Square::F8.bitboard() | Square::G8.bitboard();
        const BQ_FREESPACE: u64 = Square::B8.bitboard() | Square::C8.bitboard() | Square::D8.bitboard();
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
            (colour, target_sq) => panic!("Invalid castling target square {} for colour {}", target_sq, colour),
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
            if !self.make_move(m) {
                continue;
            }
            self.unmake_move();
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
        self.material[colour as usize] -= self.eparams.piece_values[piece as usize];
        self.pst_vals -= pst_value(piece, sq, &self.eparams.piece_square_tables);

        if PIECE_BIG[piece as usize] {
            self.big_piece_counts[colour as usize] -= 1;
            if PIECE_MAJ[piece as usize] {
                self.major_piece_counts[colour as usize] -= 1;
            } else {
                self.minor_piece_counts[colour as usize] -= 1;
            }
        }

        self.piece_lists[piece as usize].remove(sq);
    }

    fn add_piece(&mut self, sq: Square, piece: u8) {
        debug_assert!(piece_valid(piece));
        debug_assert!(sq.on_board());

        self.pieces.set_piece_at(sq, piece);

        let colour = colour_of(piece);

        hash_piece(&mut self.key, piece, sq);

        *self.piece_at_mut(sq) = piece;
        self.material[colour as usize] += self.eparams.piece_values[piece as usize];
        self.pst_vals += pst_value(piece, sq, &self.eparams.piece_square_tables);

        if PIECE_BIG[piece as usize] {
            self.big_piece_counts[colour as usize] += 1;
            if PIECE_MAJ[piece as usize] {
                self.major_piece_counts[colour as usize] += 1;
            } else {
                self.minor_piece_counts[colour as usize] += 1;
            }
        }

        self.piece_lists[piece as usize].insert(sq);
    }

    fn move_piece(&mut self, from: Square, to: Square) {
        debug_assert!(from.on_board());
        debug_assert!(to.on_board());

        let piece_moved = self.piece_at(from);

        let from_to_bb = from.bitboard() | to.bitboard();
        self.pieces.move_piece(from_to_bb, piece_moved);

        // if we're in debug mode, check that we actually find a matching piecelist entry.
        #[cfg(debug_assertions)]
        let mut t_piece_num = false;

        hash_piece(&mut self.key, piece_moved, from);
        hash_piece(&mut self.key, piece_moved, to);

        *self.piece_at_mut(from) = PIECE_EMPTY;
        *self.piece_at_mut(to) = piece_moved;
        self.pst_vals -= pst_value(piece_moved, from, &self.eparams.piece_square_tables);
        self.pst_vals += pst_value(piece_moved, to, &self.eparams.piece_square_tables);

        for sq in self.piece_lists[piece_moved as usize].iter_mut() {
            if *sq == from {
                *sq = to;
                #[cfg(debug_assertions)]
                {
                    t_piece_num = true;
                }
                break;
            }
        }

        #[cfg(debug_assertions)]
        {
            debug_assert!(t_piece_num);
        }
    }

    /// Gets the piece that will be moved by the given move.
    pub fn moved_piece(&self, m: Move) -> u8 {
        debug_assert!(m.from().on_board());
        unsafe { *self.piece_array.get_unchecked(m.from().index()) }
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
    pub fn make_move(&mut self, m: Move) -> bool {
        #[cfg(debug_assertions)]
        self.check_validity().unwrap();

        let from = m.from();
        let to = m.to();
        let side = self.side;
        let piece = self.moved_piece(m);

        debug_assert!(from.on_board());
        debug_assert!(to.on_board());
        debug_assert!(side_valid(side));
        debug_assert!(piece_valid(piece), "piece: {:?}", piece);

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
        });
        self.repetition_cache.push(saved_key);

        self.castle_perm &= unsafe { *CASTLE_PERM_MASKS.get_unchecked(from.index()) };
        self.castle_perm &= unsafe { *CASTLE_PERM_MASKS.get_unchecked(to.index()) };
        self.ep_sq = Square::NO_SQUARE;

        // reinsert the castling rights
        hash_castling(&mut self.key, self.castle_perm);

        let captured = m.capture();
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
            if m.is_pawn_start() {
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

        self.move_piece(from, to);

        let promoted_piece = m.promotion();

        if promoted_piece != PIECE_EMPTY {
            debug_assert!(piece_valid(promoted_piece));
            debug_assert!(promoted_piece != WP && promoted_piece != BP);
            self.clear_piece(to);
            self.add_piece(to, promoted_piece);
        }

        self.side ^= 1;
        hash_side(&mut self.key);

        #[cfg(debug_assertions)]
        self.check_validity().unwrap();

        // reversed in_check fn, as we have now swapped sides
        if self.in_check::<{ Self::THEM }>() {
            self.unmake_move();
            return false;
        }

        true
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

    pub fn last_move_was_nullmove(&self) -> bool {
        if let Some(Undo { m, .. }) = self.history.last() {
            *m == Move::NULL
        } else {
            false
        }
    }

    pub fn unmake_move(&mut self) {
        #[cfg(debug_assertions)]
        self.check_validity().unwrap();

        self.height -= 1;
        self.ply -= 1;

        let Undo { m, castle_perm, ep_square, fifty_move_counter } =
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

        self.move_piece(to, from);

        let captured = m.capture();
        if captured != PIECE_EMPTY {
            debug_assert!(piece_valid(captured));
            self.add_piece(to, captured);
        }

        if m.promotion() != PIECE_EMPTY {
            debug_assert!(piece_valid(m.promotion()));
            debug_assert!(m.promotion() != WP && m.promotion() != BP);
            self.clear_piece(from);
            self.add_piece(from, if colour_of(m.promotion()) == WHITE { WP } else { BP });
        }

        let key = self.repetition_cache.pop().expect("No key to unmake!");
        debug_assert_eq!(key, self.key);

        #[cfg(debug_assertions)]
        self.check_validity().unwrap();
    }

    pub fn make_move_nnue(&mut self, m: Move, t: &mut ThreadData) -> bool {
        let piece = type_of(self.moved_piece(m));
        let colour = self.turn();
        let res = self.make_move(m);
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
                Square::C1 => t.nnue.efficiently_update_from_move(ROOK, colour, Square::A1, Square::D1),
                Square::C8 => t.nnue.efficiently_update_from_move(ROOK, colour, Square::A8, Square::D8),
                Square::G1 => t.nnue.efficiently_update_from_move(ROOK, colour, Square::H1, Square::F1),
                Square::G8 => t.nnue.efficiently_update_from_move(ROOK, colour, Square::H8, Square::F8),
                _ => {
                    panic!("Invalid castle move");
                }
            }
        }

        if m.is_capture() {
            let captured = type_of(m.capture());
            t.nnue.efficiently_update_manual::<DEACTIVATE>(captured, 1 ^ colour, to);
        }

        t.nnue.efficiently_update_from_move(piece, colour, from, to);

        if m.is_promo() {
            let promo = type_of(m.promotion());
            t.nnue.efficiently_update_manual::<DEACTIVATE>(PAWN, colour, to);
            t.nnue.efficiently_update_manual::<ACTIVATE>(promo, colour, to);
        }

        true
    }

    pub fn unmake_move_nnue(&mut self, t: &mut ThreadData) {
        let m = self.history.last().unwrap().m;
        self.unmake_move();
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
            let promo = type_of(m.promotion());
            debug_assert!(piece_valid(promo));
            debug_assert!(promo != WP && promo != BP);
            t.nnue.update_pov_manual::<DEACTIVATE>(promo, colour, to);
            t.nnue.update_pov_manual::<ACTIVATE>(PAWN, colour, to);
        }
        t.nnue.update_pov_move(piece, colour, to, from);
        let captured = m.capture();
        if captured != PIECE_EMPTY {
            t.nnue.update_pov_manual::<ACTIVATE>(type_of(captured), 1 ^ colour, to);
        }
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

        let Undo { m: _, castle_perm, ep_square, fifty_move_counter } =
            self.history.pop().expect("No move to unmake!");

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
                        || PROMO_CHAR_LOOKUP[m.promotion() as usize] == san_bytes[4])
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
            let file = b"abcdefgh".iter().position(|&c| c == fname).unwrap();
            from_bb &= BB_FILES[file];
        }

        if let Some(rank) = reg_match.get(3) {
            let rname = rank.as_str().as_bytes()[0];
            let rank = b"12345678".iter().position(|&c| c == rname).unwrap();
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
            if !self.make_move(m) {
                continue;
            }
            self.unmake_move();
            if m.promotion() as usize != promo.unwrap_or(0) {
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
        // search backward because
        for (key, undo) in self.repetition_cache.iter().rev().zip(self.history.iter().rev()) {
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
        self.piece_lists[piece as usize].len()
    }

    pub const fn num_pt(&self, pt: u8) -> u8 {
        self.num(pt) + self.num(pt + 6)
    }

    pub const fn num_ct<const PIECE: u8>(&self) -> u8 {
        self.piece_lists[PIECE as usize].len()
    }

    pub const fn num_pt_ct<const PIECE_TYPE: u8>(&self) -> u8 {
        self.piece_lists[PIECE_TYPE as usize].len()
            + self.piece_lists[PIECE_TYPE as usize + 6].len()
    }

    pub fn setup_tables_for_search(&mut self) {
        self.history_table.age_entries();
        self.followup_history.age_entries();
        self.killer_move_table.fill([Move::NULL; 2]);
        self.counter_move_table.clear();
        self.height = 0;
        self.tt.clear_for_search(self.hashtable_bytes);
    }

    pub fn alloc_tables(&mut self) {
        self.history_table.clear();
        self.followup_history.clear();
        self.counter_move_table.clear();
        self.killer_move_table.fill([Move::NULL; 2]);
        self.tt.clear_for_search(self.hashtable_bytes);
        self.height = 0;
        self.movegen_ready = true;
    }

    fn regenerate_pv_line(&mut self, depth: i32) {
        self.principal_variation.clear();

        while let ProbeResult::Hit(TTHit { tt_move, .. }) =
            self.tt_probe::<true>(-INFINITY, INFINITY, MAX_DEPTH)
        {
            if self.principal_variation.len() < depth.try_into().unwrap()
                && self.is_legal(tt_move)
                && !self.is_draw()
            {
                self.make_move(tt_move);
                self.principal_variation.push(tt_move);
            } else {
                break;
            }
        }

        for _ in 0..self.principal_variation.len() {
            self.unmake_move();
        }
    }

    fn get_pv_line(&self) -> &[Move] {
        &self.principal_variation
    }

    fn print_pv(&self) {
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

    /// Performs the root search. Returns the score of the position, from white's perspective, and the best move.
    #[allow(clippy::too_many_lines)]
    pub fn search_position<const USE_NNUE: bool>(
        &mut self,
        info: &mut SearchInfo,
        thread_data: &mut [ThreadData],
    ) -> (i32, Move) {
        self.setup_tables_for_search();
        info.setup_for_search();

        let legal_moves = self.legal_moves();
        if legal_moves.is_empty() {
            return (0, Move::NULL);
        }
        if info.in_game() && legal_moves.len() == 1 {
            info.set_time_window(0);
        }

        let mut most_recent_move = legal_moves[0];
        let mut most_recent_score = 0;
        let mut aspiration_window = AspirationWindow::new();
        let max_depth = info.limit.depth().unwrap_or(MAX_DEPTH - 1).round();
        let mut mate_counter = 0;
        let mut forcing_time_reduction = false;
        let mut fail_increment = false;
        'deepening: for i_depth in 1..=max_depth {
            // consider stopping early if we've neatly completed a depth:
            if i_depth > 8 && info.in_game() && info.is_past_opt_time() {
                break 'deepening;
            }
            // aspiration loop:
            loop {
                let score = self.alpha_beta::<true, true, USE_NNUE>(
                    info,
                    &mut thread_data[0],
                    Depth::new(i_depth),
                    aspiration_window.alpha,
                    aspiration_window.beta,
                );
                info.check_up();
                if info.stopped {
                    break 'deepening;
                }
                if is_mate_score(score) && i_depth > 10 {
                    mate_counter += 1;
                    if info.in_game() && mate_counter >= 3 {
                        break 'deepening;
                    }
                } else {
                    mate_counter = 0;
                }

                let score_string = format_score(score, self.turn());

                most_recent_score = score;
                self.regenerate_pv_line(i_depth);
                most_recent_move = *self.principal_variation.first().unwrap_or(&most_recent_move);

                if i_depth > 8 && !forcing_time_reduction && info.in_game() {
                    let saved_seldepth = info.seldepth;
                    let forced = self.is_forced::<200>(
                        info,
                        &mut thread_data[0],
                        most_recent_move,
                        most_recent_score,
                        Depth::new(i_depth),
                    );
                    info.seldepth = saved_seldepth;
                    if forced {
                        forcing_time_reduction = true;
                        info.multiply_time_window(0.25);
                    }
                    info.check_up();
                    if info.stopped {
                        break 'deepening;
                    }
                }

                if aspiration_window.alpha != -INFINITY && score <= aspiration_window.alpha {
                    // fail low
                    if info.print_to_stdout {
                        // this is an upper bound, because we're going to widen the window downward,
                        // and find a lower score (in theory).
                        self.readout_info::<UPPER_BOUND>(&score_string, i_depth, info);
                    }
                    aspiration_window.widen_down();
                    if !fail_increment && info.in_game() {
                        fail_increment = true;
                        info.multiply_time_window(1.5);
                    }
                    continue;
                }
                if aspiration_window.beta != INFINITY && score >= aspiration_window.beta {
                    // fail high
                    if info.print_to_stdout {
                        // this is a lower bound, because we're going to widen the window upward,
                        // and find a higher score (in theory).
                        self.readout_info::<LOWER_BOUND>(&score_string, i_depth, info);
                    }
                    aspiration_window.widen_up();
                    continue;
                }
                if info.print_to_stdout {
                    self.readout_info::<EXACT>(&score_string, i_depth, info);
                }

                break; // we got an exact score, so we can stop the aspiration loop.
            }

            if i_depth > 4 {
                aspiration_window = AspirationWindow::from_last_score(most_recent_score);
            } else {
                aspiration_window = AspirationWindow::new();
            }
        }

        if info.print_to_stdout {
            println!("bestmove {most_recent_move}");
        }

        (if self.side == WHITE { most_recent_score } else { -most_recent_score }, most_recent_move)
    }

    fn readout_info<const BOUND: u8>(&self, sstring: &str, depth: i32, info: &SearchInfo) {
        #![allow(clippy::cast_precision_loss, clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let nps = (info.nodes as f64 / info.start_time.elapsed().as_secs_f64()) as u64;
        let mut bound = BOUND;
        if self.turn() == BLACK {
            bound = match bound {
                UPPER_BOUND => LOWER_BOUND,
                LOWER_BOUND => UPPER_BOUND,
                _ => EXACT,
            };
        }
        if bound == UPPER_BOUND {
            print!(
                "info score {sstring} upperbound depth {depth} seldepth {} nodes {} time {} nps {nps} hashfull {} pv ",
                info.seldepth.ply_to_horizon(),
                info.nodes,
                info.start_time.elapsed().as_millis(),
                self.tt.hashfull(),
            );
        } else if bound == LOWER_BOUND {
            print!(
                "info score {sstring} lowerbound depth {depth} seldepth {} nodes {} time {} nps {nps} hashfull {} pv ",
                info.seldepth.ply_to_horizon(),
                info.nodes,
                info.start_time.elapsed().as_millis(),
                self.tt.hashfull(),
            );
        } else {
            print!(
                "info score {sstring} depth {depth} seldepth {} nodes {} time {} nps {nps} hashfull {} pv ",
                info.seldepth.ply_to_horizon(),
                info.nodes,
                info.start_time.elapsed().as_millis(),
                self.tt.hashfull(),
            );
        }
        self.print_pv();
        // #[allow(clippy::cast_precision_loss)]
        // let move_ordering_percentage = info.failhigh_first as f64 * 100.0 / info.failhigh as f64;
        // eprintln!("move ordering quality: {:.2}%", move_ordering_percentage);
    }

    pub fn legal_moves(&mut self) -> Vec<Move> {
        let mut move_list = MoveList::new();
        self.generate_moves(&mut move_list);
        let mut legal_moves = Vec::new();
        for &m in move_list.iter() {
            if self.make_move(m) {
                self.unmake_move();
                legal_moves.push(m);
            }
        }
        legal_moves
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
                write!(f, "{} ", piece_char(piece).unwrap() as char)?;
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
