pub mod evaluation;
mod history;
pub mod movegen;
pub mod validation;

use std::{
    fmt::{self, Debug, Display, Formatter, Write},
    sync::{atomic::Ordering, OnceLock},
};

use rand::prelude::SliceRandom;
use rand::rngs::ThreadRng;
use regex::Regex;

use crate::{
    board::movegen::{
        bitboards::{
            self, bishop_attacks, king_attacks, knight_attacks, pawn_attacks, rook_attacks,
        },
        MoveList,
    },
    chessmove::Move,
    errors::{FenParseError, MoveParseError},
    lookups::{PIECE_BIG, PIECE_MAJ},
    makemove::{hash_castling, hash_ep, hash_piece, hash_side},
    nnue::network::{self, Activate, Deactivate, Update},
    piece::{Colour, Piece, PieceType},
    piecesquaretable::pst_value,
    search::pv::PVariation,
    searchinfo::SearchInfo,
    squareset::{self, SquareSet},
    threadlocal::ThreadData,
    uci::CHESS960,
    util::{CastlingRights, CheckState, File, Rank, Square, Undo, HORIZONTAL_RAY_BETWEEN},
};

use self::{
    evaluation::score::S,
    movegen::{bitboards::BitBoard, MoveListEntry},
};

static SAN_REGEX: OnceLock<Regex> = OnceLock::new();
fn get_san_regex() -> &'static Regex {
    SAN_REGEX.get_or_init(|| {
        Regex::new(r"^([NBKRQ])?([a-h])?([1-8])?[\-x]?([a-h][1-8])(=?[nbrqkNBRQK])?[\+#]?$")
            .unwrap()
    })
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
    /// A mask of the rooks that can castle.
    castle_perm: CastlingRights,
    /// The number of half moves made since the last capture or pawn advance.
    fifty_move_counter: u8,
    /// The number of half moves made since the start of the game.
    ply: usize,

    /// The Zobrist hash of the board.
    key: u64,

    /// All squares that the opponent attacks
    threats: SquareSet,

    /* Incrementally updated features used to accelerate various queries */
    big_piece_counts: [u8; 2],
    major_piece_counts: [u8; 2],
    minor_piece_counts: [u8; 2],
    material: [S; 2],
    pst_vals: S,

    height: usize,
    history: Vec<Undo>,
    repetition_cache: Vec<u64>,
}

/// Check that two boards are equal.
/// This is used for debugging.
#[allow(dead_code, clippy::cognitive_complexity)]
pub fn check_eq(lhs: &Board, rhs: &Board, msg: &str) {
    assert_eq!(lhs.pieces.all_pawns(), rhs.pieces.all_pawns(), "pawn bitboards {msg}");
    assert_eq!(lhs.pieces.all_knights(), rhs.pieces.all_knights(), "knight bitboards {msg}");
    assert_eq!(lhs.pieces.all_bishops(), rhs.pieces.all_bishops(), "bishop bitboards {msg}");
    assert_eq!(lhs.pieces.all_rooks(), rhs.pieces.all_rooks(), "rook bitboards {msg}");
    assert_eq!(lhs.pieces.all_queens(), rhs.pieces.all_queens(), "queen bitboards {msg}");
    assert_eq!(lhs.pieces.all_kings(), rhs.pieces.all_kings(), "king bitboards {msg}");
    assert_eq!(
        lhs.pieces.occupied_co(Colour::WHITE),
        rhs.pieces.occupied_co(Colour::WHITE),
        "white bitboards {msg}"
    );
    assert_eq!(
        lhs.pieces.occupied_co(Colour::BLACK),
        rhs.pieces.occupied_co(Colour::BLACK),
        "black bitboards {msg}"
    );
    for sq in Square::all() {
        assert_eq!(lhs.piece_at(sq), rhs.piece_at(sq), "piece_at({sq:?}) {msg}");
    }
    assert_eq!(lhs.side, rhs.side, "side {msg}");
    assert_eq!(lhs.ep_sq, rhs.ep_sq, "ep_sq {msg}");
    assert_eq!(lhs.castle_perm, rhs.castle_perm, "castle_perm {msg}");
    assert_eq!(lhs.fifty_move_counter, rhs.fifty_move_counter, "fifty_move_counter {msg}");
    assert_eq!(lhs.ply, rhs.ply, "ply {msg}");
    assert_eq!(lhs.key, rhs.key, "key {msg}");
    assert_eq!(lhs.threats, rhs.threats, "threats {msg}");
    assert_eq!(lhs.big_piece_counts, rhs.big_piece_counts, "big_piece_counts {msg}");
    assert_eq!(lhs.major_piece_counts, rhs.major_piece_counts, "major_piece_counts {msg}");
    assert_eq!(lhs.minor_piece_counts, rhs.minor_piece_counts, "minor_piece_counts {msg}");
    assert_eq!(lhs.material, rhs.material, "material {msg}");
    assert_eq!(lhs.pst_vals, rhs.pst_vals, "pst_vals {msg}");
    assert_eq!(lhs.height, rhs.height, "height {msg}");
    assert_eq!(lhs.history, rhs.history, "history {msg}");
    assert_eq!(lhs.repetition_cache, rhs.repetition_cache, "repetition_cache {msg}");
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
            .field("threats", &self.threats)
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
            threats: SquareSet::EMPTY,
            big_piece_counts: [0; 2],
            major_piece_counts: [0; 2],
            minor_piece_counts: [0; 2],
            material: [S(0, 0); 2],
            castle_perm: CastlingRights::NONE,
            history: Vec::new(),
            repetition_cache: Vec::new(),
            pst_vals: S(0, 0),
        };
        out.reset();
        out
    }

    pub const fn ep_sq(&self) -> Square {
        self.ep_sq
    }

    pub fn ep_sq_mut(&mut self) -> &mut Square {
        &mut self.ep_sq
    }

    pub fn turn_mut(&mut self) -> &mut Colour {
        &mut self.side
    }

    pub fn halfmove_clock_mut(&mut self) -> &mut u8 {
        &mut self.fifty_move_counter
    }

    pub fn set_fullmove_clock(&mut self, fullmove_clock: u16) {
        self.ply = (fullmove_clock as usize - 1) * 2 + usize::from(self.side == Colour::BLACK);
    }

    pub const fn hashkey(&self) -> u64 {
        self.key
    }

    pub fn n_men(&self) -> u8 {
        #![allow(clippy::cast_possible_truncation)]
        self.pieces.occupied().count() as u8
    }

    pub const fn ply(&self) -> usize {
        self.ply
    }

    pub fn king_sq(&self, side: Colour) -> Square {
        debug_assert!(side == Colour::WHITE || side == Colour::BLACK);
        debug_assert_eq!(self.pieces.king::<true>().count(), 1);
        debug_assert_eq!(self.pieces.king::<false>().count(), 1);
        let sq = match side {
            Colour::WHITE => self.pieces.king::<true>().first(),
            Colour::BLACK => self.pieces.king::<false>().first(),
            _ => unreachable!(),
        };
        debug_assert!(sq < Square::NO_SQUARE);
        debug_assert_eq!(self.piece_at(sq).colour(), side);
        debug_assert_eq!(self.piece_at(sq).piece_type(), PieceType::KING);
        sq
    }

    pub fn in_check(&self) -> bool {
        self.threats.contains_square(self.king_sq(self.side))
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
        self.castle_perm
    }

    pub fn castling_rights_mut(&mut self) -> &mut CastlingRights {
        &mut self.castle_perm
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

        hash_castling(&mut key, self.castle_perm);

        debug_assert!(self.fifty_move_counter <= 100);

        key
    }

    pub fn regenerate_zobrist(&mut self) {
        self.key = self.generate_pos_key();
    }

    pub fn regenerate_threats(&mut self) {
        self.threats = self.generate_threats(self.side.flip());
    }

    pub fn generate_threats(&self, side: Colour) -> SquareSet {
        if side == Colour::WHITE {
            self.generate_threats_from::<true>()
        } else {
            self.generate_threats_from::<false>()
        }
    }

    pub fn generate_threats_from<const IS_WHITE: bool>(&self) -> SquareSet {
        let mut threats = SquareSet::EMPTY;

        let their_pawns = self.pieces.pawns::<IS_WHITE>();
        let their_knights = self.pieces.knights::<IS_WHITE>();
        let their_diags = self.pieces.bishopqueen::<IS_WHITE>();
        let their_orthos = self.pieces.rookqueen::<IS_WHITE>();
        let their_king = self.king_sq(if IS_WHITE { Colour::WHITE } else { Colour::BLACK });
        let blockers = self.pieces.occupied();

        threats |= pawn_attacks::<IS_WHITE>(their_pawns);

        their_knights.iter().for_each(|sq| threats |= knight_attacks(sq));
        their_diags.iter().for_each(|sq| threats |= bishop_attacks(sq, blockers));
        their_orthos.iter().for_each(|sq| threats |= rook_attacks(sq, blockers));

        threats |= king_attacks(their_king);

        threats
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
        self.castle_perm = CastlingRights::NONE;
        self.key = 0;
        self.threats = SquareSet::EMPTY;
        self.pst_vals = S(0, 0);
        self.history.clear();
        self.repetition_cache.clear();
    }

    pub fn set_frc_idx(&mut self, scharnagl: usize) {
        assert!(scharnagl < 960, "scharnagl index out of range");
        let backrank = Self::get_scharnagl_backrank(scharnagl);
        self.reset();
        for (file, &piece_type) in backrank.iter().enumerate() {
            let sq = Square::from_rank_file(Rank::RANK_1, file.try_into().unwrap());
            self.add_piece(sq, Piece::new(Colour::WHITE, piece_type));
        }
        for file in 0..8 {
            // add pawns
            let sq = Square::from_rank_file(Rank::RANK_2, file.try_into().unwrap());
            self.add_piece(sq, Piece::new(Colour::WHITE, PieceType::PAWN));
        }
        for (file, &piece_type) in backrank.iter().enumerate() {
            let sq = Square::from_rank_file(Rank::RANK_8, file.try_into().unwrap());
            self.add_piece(sq, Piece::new(Colour::BLACK, piece_type));
        }
        for file in 0..8 {
            // add pawns
            let sq = Square::from_rank_file(Rank::RANK_7, file.try_into().unwrap());
            self.add_piece(sq, Piece::new(Colour::BLACK, PieceType::PAWN));
        }
        let mut rook_indices = backrank.iter().enumerate().filter_map(|(i, &piece)| {
            if piece == PieceType::ROOK {
                Some(i)
            } else {
                None
            }
        });
        let queenside_file = rook_indices.next().unwrap();
        let kingside_file = rook_indices.next().unwrap();
        self.castle_perm = CastlingRights {
            wk: Square::from_rank_file(Rank::RANK_1, kingside_file.try_into().unwrap()),
            wq: Square::from_rank_file(Rank::RANK_1, queenside_file.try_into().unwrap()),
            bk: Square::from_rank_file(Rank::RANK_8, kingside_file.try_into().unwrap()),
            bq: Square::from_rank_file(Rank::RANK_8, queenside_file.try_into().unwrap()),
        };
        self.key = self.generate_pos_key();
        self.threats = self.generate_threats(self.side.flip());
    }

    pub fn set_dfrc_idx(&mut self, scharnagl: usize) {
        assert!(scharnagl < 960 * 960, "double scharnagl index out of range");
        let white_backrank = Self::get_scharnagl_backrank(scharnagl % 960);
        let black_backrank = Self::get_scharnagl_backrank(scharnagl / 960);
        self.reset();
        for (file, &piece_type) in white_backrank.iter().enumerate() {
            let sq = Square::from_rank_file(Rank::RANK_1, file.try_into().unwrap());
            self.add_piece(sq, Piece::new(Colour::WHITE, piece_type));
        }
        for file in 0..8 {
            // add pawns
            let sq = Square::from_rank_file(Rank::RANK_2, file.try_into().unwrap());
            self.add_piece(sq, Piece::new(Colour::WHITE, PieceType::PAWN));
        }
        for (file, &piece_type) in black_backrank.iter().enumerate() {
            let sq = Square::from_rank_file(Rank::RANK_8, file.try_into().unwrap());
            self.add_piece(sq, Piece::new(Colour::BLACK, piece_type));
        }
        for file in 0..8 {
            // add pawns
            let sq = Square::from_rank_file(Rank::RANK_7, file.try_into().unwrap());
            self.add_piece(sq, Piece::new(Colour::BLACK, PieceType::PAWN));
        }
        let mut white_rook_indices = white_backrank.iter().enumerate().filter_map(|(i, &piece)| {
            if piece == PieceType::ROOK {
                Some(i)
            } else {
                None
            }
        });
        let white_queenside_file = white_rook_indices.next().unwrap();
        let white_kingside_file = white_rook_indices.next().unwrap();
        let mut black_rook_indices = black_backrank.iter().enumerate().filter_map(|(i, &piece)| {
            if piece == PieceType::ROOK {
                Some(i)
            } else {
                None
            }
        });
        let black_queenside_file = black_rook_indices.next().unwrap();
        let black_kingside_file = black_rook_indices.next().unwrap();
        self.castle_perm = CastlingRights {
            wk: Square::from_rank_file(Rank::RANK_1, white_kingside_file.try_into().unwrap()),
            wq: Square::from_rank_file(Rank::RANK_1, white_queenside_file.try_into().unwrap()),
            bk: Square::from_rank_file(Rank::RANK_8, black_kingside_file.try_into().unwrap()),
            bq: Square::from_rank_file(Rank::RANK_8, black_queenside_file.try_into().unwrap()),
        };
        self.key = self.generate_pos_key();
        self.threats = self.generate_threats(self.side.flip());
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
        let mut out = [PieceType::NONE; 8];
        let n = scharnagl;
        let (n2, b1) = (n / 4, n % 4);
        match b1 {
            0 => out[File::FILE_B as usize] = PieceType::BISHOP,
            1 => out[File::FILE_D as usize] = PieceType::BISHOP,
            2 => out[File::FILE_F as usize] = PieceType::BISHOP,
            3 => out[File::FILE_H as usize] = PieceType::BISHOP,
            _ => unreachable!(),
        }
        let (n3, b2) = (n2 / 4, n2 % 4);
        match b2 {
            0 => out[File::FILE_A as usize] = PieceType::BISHOP,
            1 => out[File::FILE_C as usize] = PieceType::BISHOP,
            2 => out[File::FILE_E as usize] = PieceType::BISHOP,
            3 => out[File::FILE_G as usize] = PieceType::BISHOP,
            _ => unreachable!(),
        }
        let (n4, mut q) = (n3 / 6, n3 % 6);
        for (idx, &piece) in out.iter().enumerate() {
            if piece == PieceType::NONE {
                if q == 0 {
                    out[idx] = PieceType::QUEEN;
                    break;
                }
                q -= 1;
            }
        }
        let remaining_slots = out.iter_mut().filter(|piece| **piece == PieceType::NONE);
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
                *slot = PieceType::KNIGHT;
            }
        }

        *out.iter_mut().find(|piece| **piece == PieceType::NONE).unwrap() = PieceType::ROOK;
        *out.iter_mut().find(|piece| **piece == PieceType::NONE).unwrap() = PieceType::KING;
        *out.iter_mut().find(|piece| **piece == PieceType::NONE).unwrap() = PieceType::ROOK;

        out
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
        self.threats = self.generate_threats(self.side.flip());

        Ok(())
    }

    pub fn set_startpos(&mut self) {
        self.set_from_fen(Self::STARTING_FEN)
            .expect("for some reason, STARTING_FEN is now broken.");
        debug_assert_eq!(
            self.material[Colour::WHITE.index()].0,
            self.material[Colour::BLACK.index()].0,
            "mg_material is not equal, white: {}, black: {}",
            self.material[Colour::WHITE.index()].0,
            self.material[Colour::BLACK.index()].0
        );
        debug_assert_eq!(
            self.material[Colour::WHITE.index()].1,
            self.material[Colour::BLACK.index()].1,
            "eg_material is not equal, white: {}, black: {}",
            self.material[Colour::WHITE.index()].1,
            self.material[Colour::BLACK.index()].1
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

    #[allow(dead_code)]
    pub fn from_frc_idx(scharnagl: usize) -> Self {
        let mut out = Self::new();
        out.set_frc_idx(scharnagl);
        out
    }

    #[allow(dead_code)]
    pub fn from_dfrc_idx(scharnagl: usize) -> Self {
        let mut out = Self::new();
        out.set_dfrc_idx(scharnagl);
        out
    }

    pub fn fen(&self) -> String {
        let mut out = Vec::with_capacity(60);
        self.write_fen_into(&mut out).expect("something terrible happened while writing FEN");
        // SAFETY: we know that the string is valid UTF-8, because we only write ASCII characters.
        unsafe { String::from_utf8_unchecked(out) }
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
            Some(b"-") => self.castle_perm = CastlingRights::NONE,
            Some(castling) if !CHESS960.load(Ordering::SeqCst) => {
                for &c in castling {
                    match c {
                        b'K' => self.castle_perm.wk = Square::H1,
                        b'Q' => self.castle_perm.wq = Square::A1,
                        b'k' => self.castle_perm.bk = Square::H8,
                        b'q' => self.castle_perm.bq = Square::A8,
                        _ => return Err(format!("FEN string is invalid, expected castling part to be of the form 'KQkq', got \"{}\"", std::str::from_utf8(castling).unwrap_or("<invalid utf8>"))),
                    }
                }
            }
            Some(shredder_castling) => {
                // valid shredder castling strings are of the form "AHah", "Bd"
                let white_king = self.king_sq(Colour::WHITE);
                let black_king = self.king_sq(Colour::BLACK);
                if white_king.rank() != Rank::RANK_1
                    && shredder_castling.iter().any(u8::is_ascii_uppercase)
                {
                    return Err(format!("FEN string is invalid, white king is not on the back rank, but got uppercase castling characters, implying present castling rights, got \"{}\"", std::str::from_utf8(shredder_castling).unwrap_or("<invalid utf8>")));
                }
                if black_king.rank() != Rank::RANK_8
                    && shredder_castling.iter().any(u8::is_ascii_lowercase)
                {
                    return Err(format!("FEN string is invalid, black king is not on the back rank, but got lowercase castling characters, implying present castling rights, got \"{}\"", std::str::from_utf8(shredder_castling).unwrap_or("<invalid utf8>")));
                }
                for &c in shredder_castling {
                    match c {
                        c if c.is_ascii_uppercase() => {
                            let file = c - b'A';
                            let king_file = white_king.file();
                            if file == king_file {
                                return Err(format!("FEN string is invalid, white king is on file {}, but got castling rights on that file - got \"{}\"", king_file, std::str::from_utf8(shredder_castling).unwrap_or("<invalid utf8>")));
                            }
                            let sq = Square::from_rank_file(Rank::RANK_1, file);
                            if file > king_file {
                                // castling rights are to the right of the king, so it's "kingside" castling rights.
                                self.castle_perm.wk = sq;
                            } else {
                                // castling rights are to the left of the king, so it's "queenside" castling rights.
                                self.castle_perm.wq = sq;
                            }
                        }
                        c if c.is_ascii_lowercase() => {
                            let file = c - b'a';
                            let king_file = black_king.file();
                            if file == king_file {
                                return Err(format!("FEN string is invalid, black king is on file {}, but got castling rights on that file - got \"{}\"", king_file, std::str::from_utf8(shredder_castling).unwrap_or("<invalid utf8>")));
                            }
                            let sq = Square::from_rank_file(Rank::RANK_8, file);
                            if file > king_file {
                                // castling rights are to the right of the king, so it's "kingside" castling rights.
                                self.castle_perm.bk = sq;
                            } else {
                                // castling rights are to the left of the king, so it's "queenside" castling rights.
                                self.castle_perm.bq = sq;
                            }
                        }
                        _ => {
                            return Err(format!("FEN string is invalid, expected castling part to be of the form 'AHah', 'Bd', or '-', got \"{}\"", std::str::from_utf8(shredder_castling).unwrap_or("<invalid utf8>")));
                        }
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

        if IS_WHITE == (self.side == Colour::BLACK) {
            return self.threats.contains_square(sq);
        }

        let sq_bb = sq.as_set();
        let our_pawns = self.pieces.pawns::<IS_WHITE>();
        let our_knights = self.pieces.knights::<IS_WHITE>();
        let our_diags = self.pieces.bishopqueen::<IS_WHITE>();
        let our_orthos = self.pieces.rookqueen::<IS_WHITE>();
        let our_king = self.pieces.king::<IS_WHITE>();
        let blockers = self.pieces.occupied();

        // pawns
        let attacks = pawn_attacks::<IS_WHITE>(our_pawns);
        if (attacks & sq_bb).non_empty() {
            return true;
        }

        // knights
        let knight_attacks_from_this_square = bitboards::knight_attacks(sq);
        if (our_knights & knight_attacks_from_this_square).non_empty() {
            return true;
        }

        // bishops, queens
        let diag_attacks_from_this_square = bitboards::bishop_attacks(sq, blockers);
        if (our_diags & diag_attacks_from_this_square).non_empty() {
            return true;
        }

        // rooks, queens
        let ortho_attacks_from_this_square = bitboards::rook_attacks(sq, blockers);
        if (our_orthos & ortho_attacks_from_this_square).non_empty() {
            return true;
        }

        // king
        let king_attacks_from_this_square = bitboards::king_attacks(sq);
        if (our_king & king_attacks_from_this_square).non_empty() {
            return true;
        }

        false
    }

    /// Checks whether a move is pseudo-legal.
    /// This means that it is a legal move, except for the fact that it might leave the king in check.
    pub fn is_pseudo_legal(&self, m: Move) -> bool {
        if m.is_null() {
            return false;
        }

        let from = m.from();
        let to = m.to();

        let moved_piece = self.piece_at(from);
        let captured_piece = if m.is_castle() { Piece::EMPTY } else { self.piece_at(to) };
        let is_capture = !captured_piece.is_empty();
        let is_pawn_double_push = self.is_double_pawn_push(m);

        if moved_piece.is_empty() {
            return false;
        }

        if moved_piece.colour() != self.side {
            return false;
        }

        if is_capture && captured_piece.colour() == self.side {
            return false;
        }

        if moved_piece.piece_type() != PieceType::PAWN
            && (is_pawn_double_push || m.is_ep() || m.is_promo())
        {
            return false;
        }

        if moved_piece.piece_type() != PieceType::KING && m.is_castle() {
            return false;
        }

        if is_capture && is_pawn_double_push {
            return false;
        }

        if m.is_castle() {
            return self.is_pseudo_legal_castling(m);
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
                return (pawn_attacks::<true>(from.as_set()) & to.as_set()).non_empty();
            }
            return (pawn_attacks::<false>(from.as_set()) & to.as_set()).non_empty();
        }

        (to.as_set()
            & bitboards::attacks_by_type(moved_piece.piece_type(), from, self.pieces.occupied()))
        .non_empty()
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
        let moved = self.piece_at(m.from());
        if moved.piece_type() != PieceType::KING {
            return false;
        }
        let home_rank =
            if self.side == Colour::WHITE { SquareSet::RANK_1 } else { SquareSet::RANK_8 };
        if (m.to().as_set() & home_rank).is_empty() {
            return false;
        }
        if (m.from().as_set() & home_rank).is_empty() {
            return false;
        }
        let (king_dst, rook_dst) = if m.to() > m.from() {
            // kingside castling.
            if m.to()
                != (if self.side == Colour::BLACK {
                    self.castle_perm.bk
                } else {
                    self.castle_perm.wk
                })
            {
                // the to-square doesn't match the castling rights
                // (it goes to the wrong place, or the rights don't exist)
                return false;
            }
            if self.side == Colour::BLACK {
                (Square::G8, Square::F8)
            } else {
                (Square::G1, Square::F1)
            }
        } else {
            // queenside castling.
            if m.to()
                != (if self.side == Colour::BLACK {
                    self.castle_perm.bq
                } else {
                    self.castle_perm.wq
                })
            {
                // the to-square doesn't match the castling rights
                // (it goes to the wrong place, or the rights don't exist)
                return false;
            }
            if self.side == Colour::BLACK {
                (Square::C8, Square::D8)
            } else {
                (Square::C1, Square::D1)
            }
        };

        // king_path is the path the king takes to get to its destination.
        let king_path = HORIZONTAL_RAY_BETWEEN[m.from().index()][king_dst.index()];
        // rook_path is the path the rook takes to get to its destination.
        let rook_path = HORIZONTAL_RAY_BETWEEN[m.from().index()][m.to().index()];
        // castle_occ is the occupancy that "counts" for castling.
        let castle_occ = self.pieces.occupied() ^ m.from().as_set() ^ m.to().as_set();

        (castle_occ & (king_path | rook_path | king_dst.as_set() | rook_dst.as_set())).is_empty()
            && !self.any_attacked(king_path | m.from().as_set(), self.side.flip())
    }

    pub fn any_attacked(&self, squares: SquareSet, by: Colour) -> bool {
        if by == self.side.flip() {
            (squares & self.threats).non_empty()
        } else {
            for sq in squares.iter() {
                if self.sq_attacked(sq, by) {
                    return true;
                }
            }
            false
        }
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

        if PIECE_BIG[piece.index()] {
            self.big_piece_counts[colour.index()] -= 1;
            if PIECE_MAJ[piece.index()] {
                self.major_piece_counts[colour.index()] -= 1;
            } else {
                self.minor_piece_counts[colour.index()] -= 1;
            }
        }
    }

    pub fn activate_psqt(&mut self, info: &SearchInfo, pt: PieceType, colour: Colour, sq: Square) {
        let piece = Piece::new(colour, pt);
        self.material[colour.index()] += info.eval_params.piece_values[piece.index()];
        self.pst_vals += pst_value(piece, sq, &info.eval_params.piece_square_tables);
    }

    pub fn deactivate_psqt(
        &mut self,
        info: &SearchInfo,
        pt: PieceType,
        colour: Colour,
        sq: Square,
    ) {
        let piece = Piece::new(colour, pt);
        self.material[colour.index()] -= info.eval_params.piece_values[piece.index()];
        self.pst_vals -= pst_value(piece, sq, &info.eval_params.piece_square_tables);
    }

    pub fn move_psqt(
        &mut self,
        info: &SearchInfo,
        pt: PieceType,
        colour: Colour,
        from: Square,
        to: Square,
    ) {
        let piece = Piece::new(colour, pt);
        self.pst_vals -= pst_value(piece, from, &info.eval_params.piece_square_tables);
        self.pst_vals += pst_value(piece, to, &info.eval_params.piece_square_tables);
    }

    pub fn refresh_psqt(&mut self, info: &SearchInfo) {
        for sq in Square::all() {
            let piece = self.piece_at(sq);
            if piece == Piece::EMPTY {
                continue;
            }
            let colour = piece.colour();
            self.material[colour.index()] += info.eval_params.piece_values[piece.index()];
            self.pst_vals += pst_value(piece, sq, &info.eval_params.piece_square_tables);
        }
    }

    pub fn add_piece(&mut self, sq: Square, piece: Piece) {
        debug_assert!(sq.on_board());

        self.pieces.set_piece_at(sq, piece);

        let colour = piece.colour();

        hash_piece(&mut self.key, piece, sq);

        *self.piece_at_mut(sq) = piece;

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
        debug_assert!(
            self.piece_at(from) != Piece::EMPTY,
            "from: {}, to: {}, board: {}, history: {:?}",
            from,
            to,
            self.fen(),
            self.history,
        );
        debug_assert!(
            self.piece_at(to) == Piece::EMPTY,
            "from: {}, to: {}, board: {}, history: {:?}",
            from,
            to,
            self.fen(),
            self.history,
        );
        if from == to {
            return;
        }

        let piece_moved = self.piece_at(from);

        let from_to_bb = from.as_set() | to.as_set();
        self.pieces.move_piece(from_to_bb, piece_moved);

        hash_piece(&mut self.key, piece_moved, from);
        hash_piece(&mut self.key, piece_moved, to);

        *self.piece_at_mut(from) = Piece::EMPTY;
        *self.piece_at_mut(to) = piece_moved;
    }

    /// Gets the piece that will be moved by the given move.
    pub fn moved_piece(&self, m: Move) -> Piece {
        debug_assert!(m.from().on_board());
        let idx = m.from().index();
        self.piece_array[idx]
    }

    /// Gets the piece that will be captured by the given move.
    pub fn captured_piece(&self, m: Move) -> Piece {
        debug_assert!(m.to().on_board());
        if m.is_castle() {
            return Piece::EMPTY;
        }
        let idx = m.to().index();
        self.piece_array[idx]
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
        let from_bb = m.from().as_set();
        if (from_bb & (SquareSet::RANK_2 | SquareSet::RANK_7)).is_empty() {
            return false;
        }
        let to_bb = m.to().as_set();
        if (to_bb & (SquareSet::RANK_4 | SquareSet::RANK_5)).is_empty() {
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
        self.piece_array[sq.index()]
    }

    /// Gets a mutable reference to the piece at the given square.
    pub fn piece_at_mut(&mut self, sq: Square) -> &mut Piece {
        debug_assert!(sq.on_board());
        &mut self.piece_array[sq.index()]
    }

    #[allow(clippy::cognitive_complexity, clippy::too_many_lines)]
    pub fn make_move_base(&mut self, m: Move) -> bool {
        #[cfg(debug_assertions)]
        self.check_validity().unwrap();

        let from = m.from();
        let mut to = m.to();
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
            self.clear_piece(from);
            match to {
                _ if to == self.castle_perm.wk => {
                    self.move_piece(self.castle_perm.wk, Square::F1);
                    to = Square::G1;
                }
                _ if to == self.castle_perm.wq => {
                    self.move_piece(self.castle_perm.wq, Square::D1);
                    to = Square::C1;
                }
                _ if to == self.castle_perm.bk => {
                    self.move_piece(self.castle_perm.bk, Square::F8);
                    to = Square::G8;
                }
                _ if to == self.castle_perm.bq => {
                    self.move_piece(self.castle_perm.bq, Square::D8);
                    to = Square::C8;
                }
                _ => {
                    panic!("Invalid castle move, to: {}, castle_perm: {}", to, self.castle_perm);
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
            threats: self.threats,
        });
        self.repetition_cache.push(saved_key);

        // update castling rights
        let mut new_rights = self.castle_perm;
        if piece == Piece::WR {
            if from == self.castle_perm.wk {
                new_rights.wk = Square::NO_SQUARE;
            } else if from == self.castle_perm.wq {
                new_rights.wq = Square::NO_SQUARE;
            }
        } else if piece == Piece::BR {
            if from == self.castle_perm.bk {
                new_rights.bk = Square::NO_SQUARE;
            } else if from == self.castle_perm.bq {
                new_rights.bq = Square::NO_SQUARE;
            }
        } else if piece == Piece::WK {
            new_rights.wk = Square::NO_SQUARE;
            new_rights.wq = Square::NO_SQUARE;
        } else if piece == Piece::BK {
            new_rights.bk = Square::NO_SQUARE;
            new_rights.bq = Square::NO_SQUARE;
        }
        new_rights.remove(to);
        self.castle_perm = new_rights;

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
                    debug_assert!(self.ep_sq.rank() == Rank::RANK_3);
                } else {
                    self.ep_sq = from.sub(8);
                    debug_assert!(self.ep_sq.rank() == Rank::RANK_6);
                }
                hash_ep(&mut self.key, self.ep_sq);
            }
        }

        if m.is_promo() {
            let promo = Piece::new(side, m.promotion_type());
            debug_assert!(promo.piece_type().legal_promo());
            self.clear_piece(from);
            self.add_piece(to, promo);
        } else if m.is_castle() {
            self.add_piece(to, piece); // stupid hack for piece-swapping
        } else {
            self.move_piece(from, to);
        }

        self.side = self.side.flip();
        hash_side(&mut self.key);

        self.threats = self.generate_threats(self.side.flip());

        #[cfg(debug_assertions)]
        self.check_validity().unwrap();

        // reversed in_check fn, as we have now swapped sides
        if self.sq_attacked(self.king_sq(self.side.flip()), self.side) {
            self.unmake_move_base();
            return false;
        }

        true
    }

    pub fn unmake_move_base(&mut self) {
        #[cfg(debug_assertions)]
        self.check_validity().unwrap();

        self.height -= 1;
        self.ply -= 1;

        let Undo { m, castle_perm, ep_square, fifty_move_counter, capture, threats } =
            self.history.pop().expect("No move to unmake!");

        let from = m.from();
        let mut to = m.to();

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

        self.threats = threats;

        if m.is_ep() {
            if self.side == Colour::WHITE {
                self.add_piece(to.sub(8), Piece::BP);
            } else {
                self.add_piece(to.add(8), Piece::WP);
            }
        } else if m.is_castle() {
            match to {
                _ if to == self.castle_perm.wk => {
                    to = Square::G1;
                    self.clear_piece(to);
                    self.move_piece(Square::F1, self.castle_perm.wk);
                }
                _ if to == self.castle_perm.wq => {
                    to = Square::C1;
                    self.clear_piece(to);
                    self.move_piece(Square::D1, self.castle_perm.wq);
                }
                _ if to == self.castle_perm.bk => {
                    to = Square::G8;
                    self.clear_piece(to);
                    self.move_piece(Square::F8, self.castle_perm.bk);
                }
                _ if to == self.castle_perm.bq => {
                    to = Square::C8;
                    self.clear_piece(to);
                    self.move_piece(Square::D8, self.castle_perm.bq);
                }
                _ => {
                    panic!("Invalid castle move, to: {}, castle_perm: {}", to, self.castle_perm);
                }
            }
        }

        if m.is_promo() {
            let promotion = Piece::new(self.side, m.promotion_type());
            debug_assert!(promotion.piece_type().legal_promo());
            debug_assert_eq!(promotion.colour(), self.piece_at(to).colour());
            self.clear_piece(to);
            self.add_piece(from, if self.side == Colour::WHITE { Piece::WP } else { Piece::BP });
        } else if m.is_castle() {
            self.add_piece(from, Piece::new(self.side, PieceType::KING));
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
        debug_assert!(!self.in_check());

        self.history.push(Undo {
            m: Move::NULL,
            castle_perm: self.castle_perm,
            ep_square: self.ep_sq,
            fifty_move_counter: self.fifty_move_counter,
            capture: Piece::EMPTY,
            threats: self.threats,
        });

        if self.ep_sq != Square::NO_SQUARE {
            hash_ep(&mut self.key, self.ep_sq);
        }

        self.ep_sq = Square::NO_SQUARE;

        self.side = self.side.flip();
        self.ply += 1;
        self.height += 1;
        hash_side(&mut self.key);

        self.threats = self.generate_threats(self.side.flip());

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

        let Undo { m: _, castle_perm, ep_square, fifty_move_counter, capture, threats } =
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

        self.threats = threats;

        #[cfg(debug_assertions)]
        self.check_validity().unwrap();
    }

    #[allow(clippy::too_many_lines)]
    pub fn make_move_nnue(&mut self, m: Move, t: &mut ThreadData) -> bool {
        let piece_type = self.moved_piece(m).piece_type();
        let colour = self.turn();
        let capture = self.captured_piece(m);
        let saved_castle_perm = self.castle_perm;
        let res = self.make_move_base(m);
        if !res {
            return false;
        }
        let from = m.from();
        let mut to = m.to();
        t.nnue.push_acc();

        let king_moved = piece_type == PieceType::KING;
        let ue = if network::BUCKETS != 1 && king_moved {
            let refresh = Update::colour(colour);
            t.nnue.refresh_accumulators(self, refresh);
            refresh.opposite()
        } else {
            Update::BOTH
        };

        let white_king = self.king_sq(Colour::WHITE);
        let black_king = self.king_sq(Colour::BLACK);

        if m.is_ep() {
            let ep_sq = if colour == Colour::WHITE { to.sub(8) } else { to.add(8) };
            t.nnue.update_feature::<Deactivate>(
                white_king,
                black_king,
                PieceType::PAWN,
                colour.flip(),
                ep_sq,
                ue,
            );
        } else if m.is_castle() {
            match () {
                _ if to == saved_castle_perm.wk => {
                    t.nnue.move_feature(
                        white_king,
                        black_king,
                        PieceType::ROOK,
                        colour,
                        saved_castle_perm.wk,
                        Square::F1,
                        ue,
                    );
                    to = Square::G1;
                }
                _ if to == saved_castle_perm.wq => {
                    t.nnue.move_feature(
                        white_king,
                        black_king,
                        PieceType::ROOK,
                        colour,
                        saved_castle_perm.wq,
                        Square::D1,
                        ue,
                    );
                    to = Square::C1;
                }
                _ if to == saved_castle_perm.bk => {
                    t.nnue.move_feature(
                        white_king,
                        black_king,
                        PieceType::ROOK,
                        colour,
                        saved_castle_perm.bk,
                        Square::F8,
                        ue,
                    );
                    to = Square::G8;
                }
                _ if to == saved_castle_perm.bq => {
                    t.nnue.move_feature(
                        white_king,
                        black_king,
                        PieceType::ROOK,
                        colour,
                        saved_castle_perm.bq,
                        Square::D8,
                        ue,
                    );
                    to = Square::C8;
                }
                _ => {
                    panic!("Invalid castle move {m} with castle_perm {saved_castle_perm} in position {fen}", fen = self.fen());
                }
            }
        }

        if capture != Piece::EMPTY {
            t.nnue.update_feature::<Deactivate>(
                white_king,
                black_king,
                capture.piece_type(),
                colour.flip(),
                to,
                ue,
            );
        }

        if m.is_promo() {
            let promo = m.promotion_type();
            debug_assert!(promo.legal_promo());
            t.nnue.update_feature::<Deactivate>(
                white_king,
                black_king,
                PieceType::PAWN,
                colour,
                from,
                ue,
            );
            t.nnue.update_feature::<Activate>(white_king, black_king, promo, colour, to, ue);
        } else {
            t.nnue.move_feature(white_king, black_king, piece_type, colour, from, to, ue);
        }

        true
    }

    pub fn make_move_hce(&mut self, m: Move, info: &SearchInfo) -> bool {
        debug_assert!(self.check_hce_coherency(info));
        let piece_type = self.moved_piece(m).piece_type();
        let colour = self.turn();
        let capture = self.captured_piece(m);
        let saved_castle_perm = self.castle_perm;
        let res = self.make_move_base(m);
        if !res {
            return false;
        }
        let from = m.from();
        let mut to = m.to();
        if m.is_ep() {
            let ep_sq = if colour == Colour::WHITE { to.sub(8) } else { to.add(8) };
            // deactivate pawn on ep_sq
            self.deactivate_psqt(info, PieceType::PAWN, colour.flip(), ep_sq);
        } else if m.is_castle() {
            match to {
                _ if to == saved_castle_perm.wk => {
                    self.move_psqt(info, PieceType::ROOK, colour, saved_castle_perm.wk, Square::F1);
                    to = Square::G1;
                }
                _ if to == saved_castle_perm.wq => {
                    self.move_psqt(info, PieceType::ROOK, colour, saved_castle_perm.wq, Square::D1);
                    to = Square::C1;
                }
                _ if to == saved_castle_perm.bk => {
                    self.move_psqt(info, PieceType::ROOK, colour, saved_castle_perm.bk, Square::F8);
                    to = Square::G8;
                }
                _ if to == saved_castle_perm.bq => {
                    self.move_psqt(info, PieceType::ROOK, colour, saved_castle_perm.bq, Square::D8);
                    to = Square::C8;
                }
                _ => {
                    panic!("Invalid castle move");
                }
            }
        }

        if capture != Piece::EMPTY {
            self.deactivate_psqt(info, capture.piece_type(), colour.flip(), to);
        }

        if m.is_promo() {
            let promo = m.promotion_type();
            debug_assert!(promo.legal_promo());
            self.deactivate_psqt(info, PieceType::PAWN, colour, from);
            self.activate_psqt(info, promo, colour, to);
        } else {
            self.move_psqt(info, piece_type, colour, from, to);
        }

        debug_assert!(self.check_hce_coherency(info));

        true
    }

    pub fn unmake_move_nnue(&mut self, t: &mut ThreadData) {
        #[cfg(debug_assertions)]
        let m = self.history.last().unwrap().m;
        self.unmake_move_base();
        t.nnue.pop_acc();
        #[cfg(debug_assertions)]
        {
            let piece = self.moved_piece(m).piece_type();
            let from = m.from();
            let mut to = m.to();
            let colour = self.turn();
            if m.is_ep() {
                let ep_sq = if colour == Colour::WHITE { to.sub(8) } else { to.add(8) };
                t.nnue.update_pov_manual::<Activate>(PieceType::PAWN, colour.flip(), ep_sq);
            } else if m.is_castle() {
                let (rook_from, rook_to) = match () {
                    _ if to == self.castle_perm.wk => {
                        to = Square::G1;
                        (Square::F1, self.castle_perm.wk)
                    }
                    _ if to == self.castle_perm.wq => {
                        to = Square::C1;
                        (Square::D1, self.castle_perm.wq)
                    }
                    _ if to == self.castle_perm.bk => {
                        to = Square::G8;
                        (Square::F8, self.castle_perm.bk)
                    }
                    _ if to == self.castle_perm.bq => {
                        to = Square::C8;
                        (Square::D8, self.castle_perm.bq)
                    }
                    _ => {
                        panic!("Invalid castle move {m} with castle_perm {castle_perm} in position {fen}", castle_perm = self.castle_perm, fen = self.fen());
                    }
                };
                t.nnue.update_pov_move(PieceType::ROOK, colour, rook_from, rook_to);
            }
            if m.is_promo() {
                let promo = m.promotion_type();
                debug_assert!(promo.legal_promo());
                t.nnue.update_pov_manual::<Deactivate>(promo, colour, to);
                t.nnue.update_pov_manual::<Activate>(PieceType::PAWN, colour, from);
            } else {
                t.nnue.update_pov_move(piece, colour, to, from);
            }
            let capture = self.captured_piece(m);
            if capture != Piece::EMPTY {
                t.nnue.update_pov_manual::<Activate>(capture.piece_type(), colour.flip(), to);
            }
        }
    }

    pub fn unmake_move_hce(&mut self, info: &SearchInfo) {
        debug_assert!(self.check_hce_coherency(info));
        let m = self.history.last().unwrap().m;
        self.unmake_move_base();
        let piece = self.moved_piece(m).piece_type();
        let from = m.from();
        let to = m.to();
        let colour = self.turn();
        if m.is_ep() {
            let ep_sq = if colour == Colour::WHITE { to.sub(8) } else { to.add(8) };
            self.activate_psqt(info, PieceType::PAWN, colour.flip(), ep_sq);
        } else if m.is_castle() {
            match to {
                _ if to == self.castle_perm.wk => {
                    self.move_psqt(info, PieceType::ROOK, colour, Square::F1, self.castle_perm.wk);
                }
                _ if to == self.castle_perm.wq => {
                    self.move_psqt(info, PieceType::ROOK, colour, Square::D1, self.castle_perm.wq);
                }
                _ if to == self.castle_perm.bk => {
                    self.move_psqt(info, PieceType::ROOK, colour, Square::F8, self.castle_perm.bk);
                }
                _ if to == self.castle_perm.bq => {
                    self.move_psqt(info, PieceType::ROOK, colour, Square::D8, self.castle_perm.bq);
                }
                _ => panic!("Invalid castle move"),
            };
        }
        if m.is_promo() {
            let promo = m.promotion_type();
            debug_assert!(promo.legal_promo());
            self.deactivate_psqt(info, promo, colour, to);
            self.activate_psqt(info, PieceType::PAWN, colour, from);
        } else if m.is_castle() {
            let king_to_sq = m.history_to_square();
            self.move_psqt(info, PieceType::KING, colour, king_to_sq, from);
        } else {
            self.move_psqt(info, piece, colour, to, from);
        }
        let capture = self.captured_piece(m);
        if capture != Piece::EMPTY {
            self.activate_psqt(info, capture.piece_type(), colour.flip(), to);
        }
        debug_assert!(self.check_hce_coherency(info));
    }

    pub fn make_move<const USE_NNUE: bool>(
        &mut self,
        m: Move,
        t: &mut ThreadData,
        info: &SearchInfo,
    ) -> bool {
        if USE_NNUE {
            self.make_move_nnue(m, t)
        } else {
            self.make_move_hce(m, info)
        }
    }

    pub fn unmake_move<const USE_NNUE: bool>(&mut self, t: &mut ThreadData, info: &SearchInfo) {
        if USE_NNUE {
            self.unmake_move_nnue(t);
        } else {
            self.unmake_move_hce(info);
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

        let frc_cleanup = !CHESS960.load(Ordering::Relaxed);
        let res = list
            .iter()
            .copied()
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
        let to_bb = to_square.as_set();
        let mut from_bb = SquareSet::FULL;

        let promo = reg_match.get(5).map(|promo| {
            b".NBRQ.."
                .iter()
                .position(|&c| c == *promo.as_str().as_bytes().last().unwrap())
                .unwrap()
        });
        if promo.is_some() {
            let legal_mask =
                if self.side == Colour::WHITE { SquareSet::RANK_7 } else { SquareSet::RANK_2 };
            from_bb &= legal_mask;
        }

        if let Some(file) = reg_match.get(2) {
            let fname = file.as_str().as_bytes()[0];
            let file = usize::from(fname - b'a');
            from_bb &= squareset::BB_FILES[file];
        }

        if let Some(rank) = reg_match.get(3) {
            let rname = rank.as_str().as_bytes()[0];
            let rank = usize::from(rname - b'1');
            from_bb &= squareset::BB_RANKS[rank];
        }

        if let Some(piece) = reg_match.get(1) {
            let piece_char = piece.as_str().as_bytes()[0];
            let piece_type = PieceType::from_symbol(piece_char).unwrap();
            let whitepbb = self.pieces.piece_bb(Piece::new(Colour::WHITE, piece_type));
            let blackpbb = self.pieces.piece_bb(Piece::new(Colour::BLACK, piece_type));
            from_bb &= whitepbb | blackpbb;
        } else {
            from_bb &= self.pieces.all_pawns();
            if reg_match.get(2).is_none() {
                from_bb &= squareset::BB_FILES[to_square.file() as usize];
            }
        }

        if from_bb.is_empty() {
            return Err(IllegalMove(san.to_string()));
        }

        let mut ml = MoveList::new();
        self.generate_moves(&mut ml);

        let mut legal_move = None;
        for &m in ml.iter() {
            if !self.make_move_base(m) {
                continue;
            }
            self.unmake_move_base();
            let legal_promo_t = m.safe_promotion_type();
            if legal_promo_t.index() != promo.unwrap_or(PieceType::NONE.index()) {
                continue;
            }
            let m_from_bb = m.from().as_set();
            let m_to_bb = m.to().as_set();
            if (m_from_bb & from_bb).non_empty() && (m_to_bb & to_bb).non_empty() {
                if legal_move.is_some() {
                    return Err(AmbiguousSAN(san.to_string()));
                }
                legal_move = Some(m);
            }
        }
        legal_move.ok_or_else(|| IllegalMove(san.to_string()))
    }

    pub fn san(&mut self, m: Move) -> Option<String> {
        let check_char = match self.gives(m) {
            CheckState::None => "",
            CheckState::Check => "+",
            CheckState::Checkmate => "#",
        };
        if m.is_castle() {
            match () {
                _ if m.to() > m.from() => return Some(format!("O-O{check_char}")),
                _ if m.to() < m.from() => return Some(format!("O-O-O{check_char}")),
                _ => unreachable!(),
            }
        }
        let to_sq = m.to();
        let moved_piece = self.piece_at(m.from());
        let is_capture = self.is_capture(m)
            || (moved_piece.piece_type() == PieceType::PAWN && to_sq == self.ep_sq);
        let piece_prefix = match moved_piece.piece_type() {
            PieceType::PAWN if !is_capture => "",
            PieceType::PAWN => &"abcdefgh"[m.from().file() as usize..=m.from().file() as usize],
            PieceType::KNIGHT => "N",
            PieceType::BISHOP => "B",
            PieceType::ROOK => "R",
            PieceType::QUEEN => "Q",
            PieceType::KING => "K",
            PieceType::NONE => return None,
            _ => unreachable!(),
        };
        let possible_ambiguous_attackers = if moved_piece.piece_type() == PieceType::PAWN {
            SquareSet::EMPTY
        } else {
            bitboards::attacks_by_type(moved_piece.piece_type(), to_sq, self.pieces.occupied())
                & self.pieces.piece_bb(moved_piece)
        };
        let needs_disambiguation =
            possible_ambiguous_attackers.count() > 1 && moved_piece.piece_type() != PieceType::PAWN;
        let from_file = squareset::BB_FILES[m.from().file() as usize];
        let from_rank = squareset::BB_RANKS[m.from().rank() as usize];
        let can_be_disambiguated_by_file = (possible_ambiguous_attackers & from_file).count() == 1;
        let can_be_disambiguated_by_rank = (possible_ambiguous_attackers & from_rank).count() == 1;
        let needs_both = !can_be_disambiguated_by_file && !can_be_disambiguated_by_rank;
        let must_be_disambiguated_by_file = needs_both || can_be_disambiguated_by_file;
        let must_be_disambiguated_by_rank =
            needs_both || (can_be_disambiguated_by_rank && !can_be_disambiguated_by_file);
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
            PieceType::NONE => "",
            _ => unreachable!(),
        };
        let san = format!("{piece_prefix}{disambiguator1}{disambiguator2}{capture_sigil}{to_sq}{promo_str}{check_char}");
        Some(san)
    }

    pub fn gives(&mut self, m: Move) -> CheckState {
        if !self.make_move_base(m) {
            return CheckState::None;
        }
        let gives_check = self.in_check();
        if gives_check {
            let mut ml = MoveList::new();
            self.generate_moves(&mut ml);
            for &m in ml.iter() {
                if !self.make_move_base(m) {
                    continue;
                }
                // we found a legal move, so m does not give checkmate.
                self.unmake_move_base();
                self.unmake_move_base();
                return CheckState::Check;
            }
            // we didn't return, so there were no legal moves,
            // so m gives checkmate.
            self.unmake_move_base();
            return CheckState::Checkmate;
        }
        self.unmake_move_base();
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

    pub fn num(&self, piece: Piece) -> u8 {
        #![allow(clippy::cast_possible_truncation)]
        self.pieces.piece_bb(piece).count() as u8
    }

    pub fn num_pt(&self, pt: PieceType) -> u8 {
        self.num(Piece::new(Colour::WHITE, pt)) + self.num(Piece::new(Colour::BLACK, pt))
    }

    pub fn pv_san(&mut self, pv: &PVariation) -> Result<String, fmt::Error> {
        let mut out = String::new();
        let mut moves_made = 0;
        for &m in pv.moves() {
            write!(out, "{} ", self.san(m).unwrap_or_else(|| "???".to_string()))?;
            self.make_move_base(m);
            moves_made += 1;
        }
        for _ in 0..moves_made {
            self.unmake_move_base();
        }
        Ok(out)
    }

    pub fn legal_moves(&mut self) -> Vec<Move> {
        let mut move_list = MoveList::new();
        self.generate_moves(&mut move_list);
        let mut legal_moves = Vec::new();
        for &m in move_list.iter() {
            if self.make_move_base(m) {
                self.unmake_move_base();
                legal_moves.push(m);
            }
        }
        legal_moves
    }

    pub const fn fifty_move_counter(&self) -> u8 {
        self.fifty_move_counter
    }

    pub fn has_insufficient_material<const IS_WHITE: bool>(&self) -> bool {
        if (self.pieces.pawns::<IS_WHITE>()
            | self.pieces.rooks::<IS_WHITE>()
            | self.pieces.queens::<IS_WHITE>())
        .non_empty()
        {
            return false;
        }

        if self.pieces.knights::<IS_WHITE>().non_empty() {
            // this approach renders KNNvK as *not* being insufficient material.
            // this is because the losing side can in theory help the winning side
            // into a checkmate, despite it being impossible to /force/ mate.
            let kings = self.pieces.all_kings();
            let queens = self.pieces.all_queens();
            return self.pieces.our_pieces::<IS_WHITE>().count() <= 2
                && (self.pieces.their_pieces::<IS_WHITE>() & !kings & !queens).is_empty();
        }

        if self.pieces.bishops::<IS_WHITE>().non_empty() {
            let bishops = self.pieces.all_bishops();
            let same_color = (bishops & SquareSet::DARK_SQUARES).is_empty()
                || (bishops & SquareSet::LIGHT_SQUARES).is_empty();
            let pawns = self.pieces.all_pawns();
            let knights = self.pieces.all_knights();
            return same_color && pawns.is_empty() && knights.is_empty();
        }

        true
    }

    pub fn write_fen_into(&self, mut f: impl std::io::Write) -> std::io::Result<usize> {
        #![allow(clippy::cast_possible_truncation)]
        let mut bytes_written = 0;
        let mut counter = 0;
        for rank in (Rank::RANK_1..=Rank::RANK_8).rev() {
            for file in File::FILE_A..=File::FILE_H {
                let sq = Square::from_rank_file(rank, file);
                let piece = self.piece_at(sq);
                if piece == Piece::EMPTY {
                    counter += 1;
                } else {
                    if counter != 0 {
                        bytes_written += f.write(&[counter + b'0'])?;
                    }
                    counter = 0;
                    let char = piece.byte_char();
                    bytes_written += f.write(&[char])?;
                }
            }
            if counter != 0 {
                bytes_written += f.write(&[counter + b'0'])?;
            }
            counter = 0;
            if rank != 0 {
                bytes_written += f.write(b"/")?;
            }
        }

        bytes_written += f.write(b" ")?;
        if self.side == Colour::WHITE {
            bytes_written += f.write(b"w")?;
        } else {
            bytes_written += f.write(b"b")?;
        }
        bytes_written += f.write(b" ")?;
        if self.castle_perm == CastlingRights::NONE {
            bytes_written += f.write(b"-")?;
        } else {
            bytes_written += [
                self.castle_perm.wk,
                self.castle_perm.wq,
                self.castle_perm.bk,
                self.castle_perm.bq,
            ]
            .into_iter()
            .zip(b"KQkq")
            .filter(|(m, _)| *m != Square::NO_SQUARE)
            .try_fold(0, |acc, (_, &ch)| f.write(&[ch]).map(|n| acc + n))?;
        }
        bytes_written += f.write(b" ")?;
        if self.ep_sq == Square::NO_SQUARE {
            bytes_written += f.write(b"-")?;
        } else {
            bytes_written += f.write(self.ep_sq.name().unwrap().as_bytes())?;
        }
        let hc = self.fifty_move_counter;
        let hundreds = hc / 100;
        let tens = (hc % 100) / 10;
        let ones = hc % 10;
        bytes_written += f.write(b" ")?;
        if hundreds != 0 {
            bytes_written += f.write(&[hundreds + b'0'])?;
        }
        if tens != 0 || hundreds != 0 {
            bytes_written += f.write(&[tens + b'0'])?;
        }
        bytes_written += f.write(&[ones + b'0'])?;
        let ply = self.ply / 2 + 1;
        let hundreds = (ply / 100) as u8;
        let tens = ((ply % 100) / 10) as u8;
        let ones = (ply % 10) as u8;
        bytes_written += f.write(b" ")?;
        if hundreds != 0 {
            bytes_written += f.write(&[hundreds + b'0'])?;
        }
        if tens != 0 || hundreds != 0 {
            bytes_written += f.write(&[tens + b'0'])?;
        }
        bytes_written += f.write(&[ones + b'0'])?;

        Ok(bytes_written)
    }

    pub const fn full_move_number(&self) -> usize {
        self.ply / 2 + 1
    }

    #[allow(dead_code /* for datagen */)]
    pub fn make_random_move<const NNUE: bool>(
        &mut self,
        rng: &mut ThreadRng,
        t: &mut ThreadData,
        info: &SearchInfo,
    ) -> Option<Move> {
        let mut ml = MoveList::new();
        self.generate_moves(&mut ml);
        let Some(MoveListEntry { mov, .. }) = ml.as_slice().choose(rng) else {
            return None;
        };
        self.make_move::<NNUE>(*mov, t, info);
        Some(*mov)
    }

    #[allow(dead_code /* for datagen */)]
    pub fn is_insufficient_material(&self) -> bool {
        self.has_insufficient_material::<true>() && self.has_insufficient_material::<false>()
    }

    #[allow(dead_code /* for datagen */)]
    pub fn outcome(&mut self) -> GameOutcome {
        if self.fifty_move_counter >= 100 {
            return GameOutcome::DrawFiftyMoves;
        }
        let mut reps = 1;
        for (key, undo) in
            self.repetition_cache.iter().rev().zip(self.history.iter().rev()).skip(1).step_by(2)
        {
            if *key == self.key {
                reps += 1;
                if reps == 3 {
                    return GameOutcome::DrawRepetition;
                }
            }
            // optimisation: if the fifty move counter was zeroed, then any prior positions will not be repetitions.
            if undo.fifty_move_counter == 0 {
                break;
            }
        }
        if self.is_insufficient_material() {
            return GameOutcome::DrawInsufficientMaterial;
        }
        let mut move_list = MoveList::new();
        self.generate_moves(&mut move_list);
        let mut legal_moves = false;
        for &m in move_list.iter() {
            if self.make_move_base(m) {
                self.unmake_move_base();
                legal_moves = true;
                break;
            }
        }
        if legal_moves {
            GameOutcome::Ongoing
        } else if self.in_check() {
            match self.side {
                Colour::WHITE => GameOutcome::BlackWinMate,
                Colour::BLACK => GameOutcome::WhiteWinMate,
                _ => unreachable!(),
            }
        } else {
            GameOutcome::DrawStalemate
        }
    }
}

#[allow(dead_code /* for datagen */)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GameOutcome {
    WhiteWinMate,
    BlackWinMate,
    WhiteWinTB,
    BlackWinTB,
    DrawFiftyMoves,
    DrawRepetition,
    DrawStalemate,
    DrawInsufficientMaterial,
    DrawTB,
    WhiteWinAdjudication,
    BlackWinAdjudication,
    DrawAdjudication,
    Ongoing,
}

#[allow(dead_code /* for datagen */)]
impl GameOutcome {
    pub const fn as_float_str(self) -> &'static str {
        match self {
            Self::WhiteWinMate | Self::WhiteWinTB | Self::WhiteWinAdjudication => "1.0",
            Self::BlackWinMate | Self::BlackWinTB | Self::BlackWinAdjudication => "0.0",
            Self::DrawFiftyMoves
            | Self::DrawRepetition
            | Self::DrawStalemate
            | Self::DrawInsufficientMaterial
            | Self::DrawTB
            | Self::DrawAdjudication => "0.5",
            Self::Ongoing => panic!("Game is not over!"),
        }
    }

    pub const fn as_packed_u8(self) -> u8 {
        // 0 for black win, 1 for draw, 2 for white win
        match self {
            Self::WhiteWinMate | Self::WhiteWinTB | Self::WhiteWinAdjudication => 2,
            Self::BlackWinMate | Self::BlackWinTB | Self::BlackWinAdjudication => 0,
            Self::DrawFiftyMoves
            | Self::DrawRepetition
            | Self::DrawStalemate
            | Self::DrawInsufficientMaterial
            | Self::DrawTB
            | Self::DrawAdjudication => 1,
            Self::Ongoing => panic!("Game is not over!"),
        }
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

        let mut board_1 = Board::new();
        board_1.set_from_fen(Board::STARTING_FEN).expect("setfen failed.");
        board_1.check_validity().unwrap();

        let board_2 = Board::from_fen(Board::STARTING_FEN).expect("setfen failed.");
        board_2.check_validity().unwrap();

        assert_eq!(board_1, board_2);
    }

    #[test]
    fn game_end_states() {
        use super::Board;
        use super::GameOutcome;
        use crate::{chessmove::Move, util::Square};

        let mut fiftymove_draw =
            Board::from_fen("rnbqkb1r/pppppppp/5n2/8/3N4/8/PPPPPPPP/RNBQKB1R b KQkq - 100 2")
                .unwrap();
        assert_eq!(fiftymove_draw.outcome(), GameOutcome::DrawFiftyMoves);
        let mut draw_repetition = Board::default();
        assert_eq!(draw_repetition.outcome(), GameOutcome::Ongoing);
        draw_repetition.make_move_base(Move::new(Square::G1, Square::F3));
        draw_repetition.make_move_base(Move::new(Square::B8, Square::C6));
        assert_eq!(draw_repetition.outcome(), GameOutcome::Ongoing);
        draw_repetition.make_move_base(Move::new(Square::F3, Square::G1));
        draw_repetition.make_move_base(Move::new(Square::C6, Square::B8));
        assert_eq!(draw_repetition.outcome(), GameOutcome::Ongoing);
        draw_repetition.make_move_base(Move::new(Square::G1, Square::F3));
        draw_repetition.make_move_base(Move::new(Square::B8, Square::C6));
        assert_eq!(draw_repetition.outcome(), GameOutcome::Ongoing);
        draw_repetition.make_move_base(Move::new(Square::F3, Square::G1));
        draw_repetition.make_move_base(Move::new(Square::C6, Square::B8));
        assert_eq!(draw_repetition.outcome(), GameOutcome::DrawRepetition);
        let mut stalemate = Board::from_fen("7k/8/6Q1/8/8/8/8/K7 b - - 0 1").unwrap();
        assert_eq!(stalemate.outcome(), GameOutcome::DrawStalemate);
        let mut insufficient_material_bare_kings =
            Board::from_fen("8/8/5k2/8/8/2K5/8/8 b - - 0 1").unwrap();
        assert_eq!(
            insufficient_material_bare_kings.outcome(),
            GameOutcome::DrawInsufficientMaterial
        );
        let mut insufficient_material_knights =
            Board::from_fen("8/8/5k2/8/2N5/2K2N2/8/8 b - - 0 1").unwrap();
        assert_eq!(insufficient_material_knights.outcome(), GameOutcome::Ongoing);
        // using FIDE rules.
    }

    #[test]
    fn fen_round_trip() {
        use crate::board::Board;
        use std::{
            fs::File,
            io::{BufRead, BufReader},
        };

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

        for ((p1, p2), pt) in
            Piece::all().take(6).zip(Piece::all().skip(6).take(6)).zip(PieceType::all())
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
                if !pos.make_move_base(m) {
                    continue;
                }
                pos.unmake_move_base();
                let san_repr = pos.san(m);
                let san_repr = san_repr.unwrap();
                let parsed_move = pos.parse_san(&san_repr);
                assert_eq!(parsed_move, Ok(m), "{san_repr} != {m} in fen {}", pos.fen());
            }
        }
    }

    #[test]
    fn scharnagl_backrank_works() {
        use super::Board;
        use crate::piece::PieceType;
        let normal_chess_arrangement = Board::get_scharnagl_backrank(518);
        assert_eq!(
            normal_chess_arrangement,
            [
                PieceType::ROOK,
                PieceType::KNIGHT,
                PieceType::BISHOP,
                PieceType::QUEEN,
                PieceType::KING,
                PieceType::BISHOP,
                PieceType::KNIGHT,
                PieceType::ROOK
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
        use crate::chessmove::Move;
        use crate::util::Square;
        let board =
            Board::from_fen("1r2k2r/2pb1pp1/2pp4/p1n5/2P4p/PP2P2P/1qB2PP1/R2QKN1R w KQk - 0 20")
                .unwrap();
        let kingside_castle = Move::new_with_flags(Square::E1, Square::H1, Move::CASTLE_FLAG);
        assert!(!board.is_pseudo_legal(kingside_castle));
    }

    #[test]
    fn threat_generation_white() {
        use super::Board;
        use crate::squareset::SquareSet;

        let board = Board::from_fen("3k4/8/8/5N2/8/1P6/8/K1Q1RB2 b - - 0 1").unwrap();
        assert_eq!(board.threats, SquareSet::from_inner(0x1454_9d56_bddd_5f3f));
    }

    #[test]
    fn threat_generation_black() {
        use super::Board;
        use crate::squareset::SquareSet;

        let board = Board::from_fen("2br1q1k/8/6p1/8/2n5/8/8/4K3 w - - 0 1").unwrap();
        assert_eq!(board.threats, SquareSet::from_inner(0xfcfa_bbbd_6ab9_2a28));
    }
}
