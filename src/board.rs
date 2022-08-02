#![allow(
    clippy::collapsible_else_if,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]

pub mod evaluation;
pub mod movegen;
mod history;

use std::{
    collections::HashSet,
    fmt::{Debug, Display, Formatter, Write},
};

use crate::{
    board::movegen::{
            bitboards::{
                self, north_east_one, north_west_one, south_east_one, south_west_one, BitLoop,
                BB_NONE,
            },
            MoveList,
        },
    chessmove::Move,
    definitions::{
        colour_of, square_name, type_of, Colour, Depth, File,
        Rank::{self, RANK_3, RANK_6},
        Square::{A1, A8, C1, C8, D1, D8, F1, F8, G1, G8, H1, H8, NO_SQUARE},
        Undo, BB, BISHOP, BK, BKCA, BLACK, BN, BOARD_N_SQUARES, BP, BQ, BQCA, BR, INFINITY, KING,
        KNIGHT, MAX_DEPTH, PIECE_EMPTY, ROOK, WB, WHITE, WK, WKCA, WN, WP, WQ, WQCA, WR,
    },
    errors::{FenParseError, MoveParseError, PositionValidityError},
    lookups::{
        filerank_to_square, piece_char, rank, PIECE_BIG, PIECE_MAJ, PIECE_MIN, PROMO_CHAR_LOOKUP,
        SQUARE_NAMES,
    },
    macros,
    makemove::{hash_castling, hash_ep, hash_piece, hash_side, CASTLE_PERM_MASKS},
    piecelist::PieceList,
    piecesquaretable::pst_value,
    search::{self, ASPIRATION_WINDOW, Stack},
    searchinfo::SearchInfo,
    transpositiontable::{DefaultTT, HFlag, ProbeResult, TTHit},
    uci::format_score,
    validate::{piece_type_valid, piece_valid, side_valid, square_on_board}, historytable::{DoubleHistoryTable, HistoryTable, MoveTable},
};

use self::{evaluation::score::S, movegen::bitboards::BitBoard};

pub struct Board {
    /// The bitboards of all the pieces on the board.
    pieces: BitBoard,
    /// An array to accelerate piece_at().
    piece_array: [u8; 64],
    side: u8,
    ep_sq: u8,
    fifty_move_counter: u8,
    height: usize,
    ply: usize,
    key: u64,
    big_piece_counts: [u8; 2],
    major_piece_counts: [u8; 2],
    minor_piece_counts: [u8; 2],
    material: [S; 2],
    castle_perm: u8,
    history: Vec<Undo>,
    repetition_cache: HashSet<u64>,
    piece_lists: [PieceList; 13],

    principal_variation: Vec<Move>,

    history_table: HistoryTable,
    killer_move_table: [[Move; 2]; MAX_DEPTH.ply_to_horizon()],
    counter_move_table: MoveTable,
    followup_history: DoubleHistoryTable,
    tt: DefaultTT,

    pst_vals: S,

    eval_params: evaluation::parameters::Parameters,
    pub search_params: search::Config,
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
            ep_sq: NO_SQUARE,
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
            repetition_cache: HashSet::new(),
            piece_lists: [PieceList::new(); 13],
            principal_variation: Vec::new(),
            history_table: HistoryTable::new(),
            killer_move_table: [[Move::NULL; 2]; MAX_DEPTH.ply_to_horizon()],
            counter_move_table: MoveTable::new(),
            followup_history: DoubleHistoryTable::new(),
            pst_vals: S(0, 0),
            tt: DefaultTT::new(),
            eval_params: evaluation::parameters::Parameters::default(),
            search_params: search::Config::default(),
            lmr_table: search::LMRTable::new(&search::Config::default()),
            movegen_ready: false,
        };
        out.reset();
        out
    }

    pub fn set_search_config(&mut self, config: search::Config) {
        self.search_params = config;
        self.lmr_table = search::LMRTable::new(&self.search_params);
    }

    pub fn tt_store(&mut self, best_move: Move, score: i32, flag: HFlag, depth: Depth) {
        self.tt
            .store(self.key, self.height, best_move, score, flag, depth);
    }

    pub fn tt_probe(&mut self, alpha: i32, beta: i32, depth: Depth) -> ProbeResult {
        self.tt.probe(self.key, self.height, alpha, beta, depth)
    }

    /// Nuke the transposition table.
    /// This wipes all entries in the table, don't call it during a search.
    pub fn clear_tt(&mut self) {
        self.tt.clear();
    }

    pub fn king_sq(&self, side: u8) -> u8 {
        debug_assert!(side == WHITE || side == BLACK);
        debug_assert_eq!(self.pieces.king::<true>().count_ones(), 1);
        debug_assert_eq!(self.pieces.king::<false>().count_ones(), 1);
        debug_assert_eq!(self.piece_lists[WK as usize].len(), 1);
        debug_assert_eq!(self.piece_lists[BK as usize].len(), 1);
        let sq = match side {
            WHITE => *self.piece_lists[WK as usize].first().unwrap(),
            BLACK => *self.piece_lists[BK as usize].first().unwrap(),
            _ => unsafe { macros::inconceivable!() },
        };
        debug_assert!(sq < 64);
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
            if piece != PIECE_EMPTY {
                debug_assert!((WP..=BK).contains(&piece));
                hash_piece(&mut key, piece, sq.try_into().unwrap());
            }
        }

        if self.side == WHITE {
            hash_side(&mut key);
        }

        if self.ep_sq != NO_SQUARE {
            debug_assert!(self.ep_sq < BOARD_N_SQUARES.try_into().unwrap());
            hash_ep(&mut key, self.ep_sq as u8);
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
        self.ep_sq = NO_SQUARE;
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
            .ok_or_else(|| format!("FEN string is missing space: {}", fen))?;
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
                let sq = rank * 8 + file;
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
                let sq = rank * 8 + file;
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
            #[rustfmt::skip]
            write!(
                fen,
                "{}{}{}{}",
                if self.castle_perm & WKCA == 0 { "" } else { "K" },
                if self.castle_perm & WQCA == 0 { "" } else { "Q" },
                if self.castle_perm & BKCA == 0 { "" } else { "k" },
                if self.castle_perm & BQCA == 0 { "" } else { "q" },
            ).unwrap();
        }
        fen.push(' ');
        if self.ep_sq == NO_SQUARE {
            fen.push('-');
        } else {
            fen.push_str(SQUARE_NAMES[self.ep_sq as usize]);
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
            Some([b'-']) => self.ep_sq = NO_SQUARE,
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
                self.ep_sq = filerank_to_square(file, rank);
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
                    .map_err(|_| {
                        "FEN string is invalid, expected halfmove clock part to be a number"
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

    #[allow(clippy::cognitive_complexity, clippy::too_many_lines, dead_code)]
    pub fn check_validity(&self) -> Result<(), PositionValidityError> {
        #![allow(clippy::similar_names, clippy::cast_possible_truncation)]
        use Colour::{Black, White};
        let mut piece_num = [0u8; 13];
        let mut big_pce = [0, 0];
        let mut maj_pce = [0, 0];
        let mut min_pce = [0, 0];
        let mut material = [S(0, 0), S(0, 0)];

        // check piece lists
        for piece in WP..=BK {
            for &sq in self.piece_lists[piece as usize].iter() {
                if self.piece_at(sq) != piece {
                    return Err(format!(
                        "piece list corrupt: expected square {} to be '{}' but was '{}'",
                        square_name(sq).unwrap_or(&format!("offboard: {}", sq)),
                        piece_char(piece)
                            .map(|c| c.to_string())
                            .unwrap_or(format!("unknown piece: {}", piece)),
                        piece_char(self.piece_at(sq))
                            .map(|c| c.to_string())
                            .unwrap_or(format!("unknown piece: {}", self.piece_at(sq)))
                    ));
                }
            }
        }

        // check turn
        if self.side != WHITE && self.side != BLACK {
            return Err(format!("invalid side: {}", self.side));
        }

        // check piece count and other counters
        for sq in 0..64 {
            let piece = self.piece_at(sq);
            if piece == PIECE_EMPTY {
                continue;
            }
            piece_num[piece as usize] += 1;
            let colour = colour_of(piece);
            if PIECE_BIG[piece as usize] {
                big_pce[colour as usize] += 1;
            }
            if PIECE_MAJ[piece as usize] {
                maj_pce[colour as usize] += 1;
            }
            if PIECE_MIN[piece as usize] {
                min_pce[colour as usize] += 1;
            }
            material[colour as usize] += self.eval_params.piece_values[piece as usize];
        }

        if piece_num[1..].to_vec()
            != self.piece_lists[1..]
                .iter()
                .map(PieceList::len)
                .collect::<Vec<_>>()
        {
            return Err(format!(
                "piece counts are corrupt: expected {:?}, got {:?}",
                &piece_num[1..],
                &self.piece_lists[1..]
                    .iter()
                    .map(PieceList::len)
                    .collect::<Vec<_>>()
            ));
        }

        // check bitboard / piece array coherency
        for piece in WP..=BK {
            let bb = self.pieces.piece_bb(piece);
            for sq in BitLoop::new(bb) {
                if self.piece_at(sq) != piece {
                    return Err(format!(
                        "bitboard / piece array coherency corrupt: expected square {} to be '{}' but was '{}'",
                        square_name(sq).unwrap_or(&format!("offboard: {}", sq)),
                        piece_char(piece).map(|c| c.to_string()).unwrap_or(format!("unknown piece: {}", piece)),
                        piece_char(self.piece_at(sq)).map(|c| c.to_string()).unwrap_or(format!("unknown piece: {}", self.piece_at(sq)))
                    ));
                }
            }
        }

        if material[White as usize].0 != self.material[White as usize].0 {
            return Err(format!(
                "white midgame material is corrupt: expected {:?}, got {:?}",
                material[White as usize].0, self.material[White as usize].0
            ));
        }
        if material[White as usize].1 != self.material[White as usize].1 {
            return Err(format!(
                "white endgame material is corrupt: expected {:?}, got {:?}",
                material[White as usize].1, self.material[White as usize].1
            ));
        }
        if material[Black as usize].0 != self.material[Black as usize].0 {
            return Err(format!(
                "black midgame material is corrupt: expected {:?}, got {:?}",
                material[Black as usize].0, self.material[Black as usize].0
            ));
        }
        if material[Black as usize].1 != self.material[Black as usize].1 {
            return Err(format!(
                "black endgame material is corrupt: expected {:?}, got {:?}",
                material[Black as usize].1, self.material[Black as usize].1
            ));
        }
        if min_pce[White as usize] != self.minor_piece_counts[White as usize] {
            return Err(format!(
                "white minor piece count is corrupt: expected {:?}, got {:?}",
                min_pce[White as usize], self.minor_piece_counts[White as usize]
            ));
        }
        if min_pce[Black as usize] != self.minor_piece_counts[Black as usize] {
            return Err(format!(
                "black minor piece count is corrupt: expected {:?}, got {:?}",
                min_pce[Black as usize], self.minor_piece_counts[Black as usize]
            ));
        }
        if maj_pce[White as usize] != self.major_piece_counts[White as usize] {
            return Err(format!(
                "white major piece count is corrupt: expected {:?}, got {:?}",
                maj_pce[White as usize], self.major_piece_counts[White as usize]
            ));
        }
        if maj_pce[Black as usize] != self.major_piece_counts[Black as usize] {
            return Err(format!(
                "black major piece count is corrupt: expected {:?}, got {:?}",
                maj_pce[Black as usize], self.major_piece_counts[Black as usize]
            ));
        }
        if big_pce[White as usize] != self.big_piece_counts[White as usize] {
            return Err(format!(
                "white big piece count is corrupt: expected {:?}, got {:?}",
                big_pce[White as usize], self.big_piece_counts[White as usize]
            ));
        }
        if big_pce[Black as usize] != self.big_piece_counts[Black as usize] {
            return Err(format!(
                "black big piece count is corrupt: expected {:?}, got {:?}",
                big_pce[Black as usize], self.big_piece_counts[Black as usize]
            ));
        }

        if !(self.side == WHITE || self.side == BLACK) {
            return Err(format!(
                "side is corrupt: expected WHITE or BLACK, got {:?}",
                self.side
            ));
        }
        if self.generate_pos_key() != self.key {
            return Err(format!(
                "key is corrupt: expected {:?}, got {:?}",
                self.generate_pos_key(),
                self.key
            ));
        }

        if !(self.ep_sq == NO_SQUARE
            || (rank(self.ep_sq) == RANK_6 && self.side == WHITE)
            || (rank(self.ep_sq) == RANK_3 && self.side == BLACK))
        {
            return Err(format!("en passant square is corrupt: expected square to be {} (NoSquare) or to be on ranks 6 or 3, got {} (Rank {})", NO_SQUARE, self.ep_sq, rank(self.ep_sq)));
        }

        if self.fifty_move_counter >= 100 {
            return Err(format!(
                "fifty move counter is corrupt: expected 0-99, got {}",
                self.fifty_move_counter
            ));
        }

        // check there are the correct number of kings for each side
        assert_eq!(self.num(WK), 1);
        assert_eq!(self.num(BK), 1);

        if self.piece_at(self.king_sq(WHITE)) != WK {
            return Err(format!(
                "white king square is corrupt: expected white king, got {:?}",
                self.piece_at(self.king_sq(WHITE))
            ));
        }
        if self.piece_at(self.king_sq(BLACK)) != BK {
            return Err(format!(
                "black king square is corrupt: expected black king, got {:?}",
                self.piece_at(self.king_sq(BLACK))
            ));
        }

        Ok(())
    }

    /// Determines if `sq` is attacked by `side`
    pub fn sq_attacked(&self, sq: u8, side: u8) -> bool {
        debug_assert!(side_valid(side));
        debug_assert!(square_on_board(sq));
        #[cfg(debug_assertions)]
        self.check_validity().unwrap();

        // pawns
        if side == WHITE {
            let our_pawns = self.pieces.pawns::<true>();
            let west_attacks = north_west_one(our_pawns);
            let east_attacks = north_east_one(our_pawns);
            let attacks = west_attacks | east_attacks;
            if attacks & (1 << sq) != 0 {
                return true;
            }
        } else {
            let our_pawns = self.pieces.pawns::<false>();
            let west_attacks = south_west_one(our_pawns);
            let east_attacks = south_east_one(our_pawns);
            let attacks = west_attacks | east_attacks;
            if attacks & (1 << sq) != 0 {
                return true;
            }
        }

        // knights
        let knights = if side == WHITE {
            self.pieces.knights::<true>()
        } else {
            self.pieces.knights::<false>()
        };
        let knight_attacks_from_this_square = bitboards::attacks::<{ KNIGHT }>(sq, BB_NONE);
        if (knights & knight_attacks_from_this_square) != BB_NONE {
            return true;
        }

        // rooks, queens
        let our_rooks_queens = if side == WHITE {
            self.pieces.rookqueen::<true>()
        } else {
            self.pieces.rookqueen::<false>()
        };
        let ortho_attacks_from_this_square =
            bitboards::attacks::<{ ROOK }>(sq, self.pieces.occupied());
        if (our_rooks_queens & ortho_attacks_from_this_square) != BB_NONE {
            return true;
        }

        // bishops, queens
        let our_bishops_queens = if side == WHITE {
            self.pieces.bishopqueen::<true>()
        } else {
            self.pieces.bishopqueen::<false>()
        };
        let diag_attacks_from_this_square =
            bitboards::attacks::<{ BISHOP }>(sq, self.pieces.occupied());
        if (our_bishops_queens & diag_attacks_from_this_square) != BB_NONE {
            return true;
        }

        // king
        let our_king = if side == WHITE {
            self.pieces.king::<true>()
        } else {
            self.pieces.king::<false>()
        };
        let king_attacks_from_this_square = bitboards::attacks::<{ KING }>(sq, BB_NONE);
        if (our_king & king_attacks_from_this_square) != BB_NONE {
            return true;
        }

        false
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

    fn clear_piece(&mut self, sq: u8) {
        debug_assert!(square_on_board(sq));
        let piece = self.piece_at(sq);
        debug_assert!(piece_valid(piece));
        let piece_type = type_of(piece);
        debug_assert!(piece_type_valid(piece_type));

        self.pieces.clear_piece_at(sq, piece);

        let colour = colour_of(piece);

        hash_piece(&mut self.key, piece, sq);

        *self.piece_at_mut(sq) = PIECE_EMPTY;
        self.material[colour as usize] -= self.eval_params.piece_values[piece as usize];
        self.pst_vals -= pst_value(piece, sq, &self.eval_params.piece_square_tables);

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

    fn add_piece(&mut self, sq: u8, piece: u8) {
        debug_assert!(piece_valid(piece));
        debug_assert!(square_on_board(sq));

        self.pieces.set_piece_at(sq, piece);

        let colour = colour_of(piece);

        hash_piece(&mut self.key, piece, sq);

        *self.piece_at_mut(sq) = piece;
        self.material[colour as usize] += self.eval_params.piece_values[piece as usize];
        self.pst_vals += pst_value(piece, sq, &self.eval_params.piece_square_tables);

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

    fn move_piece(&mut self, from: u8, to: u8) {
        debug_assert!(square_on_board(from));
        debug_assert!(square_on_board(to));

        let piece_moved = self.piece_at(from);

        let from_to_bb = 1 << from | 1 << to;
        self.pieces.move_piece(from_to_bb, piece_moved);

        // if we're in debug mode, check that we actually find a matching piecelist entry.
        #[cfg(debug_assertions)]
        let mut t_piece_num = false;

        hash_piece(&mut self.key, piece_moved, from);
        hash_piece(&mut self.key, piece_moved, to);

        *self.piece_at_mut(from) = PIECE_EMPTY;
        *self.piece_at_mut(to) = piece_moved;
        self.pst_vals -= pst_value(piece_moved, from, &self.eval_params.piece_square_tables);
        self.pst_vals += pst_value(piece_moved, to, &self.eval_params.piece_square_tables);

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
        debug_assert!(square_on_board(m.from()));
        unsafe { *self.piece_array.get_unchecked(m.from() as usize) }
    }

    /// Gets the piece at the given square.
    pub fn piece_at(&self, sq: u8) -> u8 {
        debug_assert!((sq as usize) < BOARD_N_SQUARES);
        unsafe { *self.piece_array.get_unchecked(sq as usize) }
    }

    /// Gets a mutable reference to the piece at the given square.
    pub fn piece_at_mut(&mut self, sq: u8) -> &mut u8 {
        debug_assert!((sq as usize) < BOARD_N_SQUARES);
        unsafe { self.piece_array.get_unchecked_mut(sq as usize) }
    }

    #[allow(clippy::cognitive_complexity)]
    pub fn make_move(&mut self, m: Move) -> bool {
        #[cfg(debug_assertions)]
        self.check_validity().unwrap();

        let from = m.from();
        let to = m.to();
        let side = self.side;
        let piece = self.moved_piece(m);

        debug_assert!(square_on_board(from));
        debug_assert!(square_on_board(to));
        debug_assert!(side_valid(side));
        debug_assert!(piece_valid(piece), "piece: {:?}", piece);

        let saved_key = self.key;

        if m.is_ep() {
            if side == WHITE {
                self.clear_piece(to - 8);
            } else {
                self.clear_piece(to + 8);
            }
        } else if m.is_castle() {
            match to {
                C1 => self.move_piece(A1, D1),
                C8 => self.move_piece(A8, D8),
                G1 => self.move_piece(H1, F1),
                G8 => self.move_piece(H8, F8),
                _ => {
                    panic!("Invalid castle move");
                }
            }
        }

        if self.ep_sq != NO_SQUARE {
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
        self.repetition_cache.insert(saved_key);

        self.castle_perm &= unsafe { *CASTLE_PERM_MASKS.get_unchecked(from as usize) };
        self.castle_perm &= unsafe { *CASTLE_PERM_MASKS.get_unchecked(to as usize) };
        self.ep_sq = NO_SQUARE;

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
                    self.ep_sq = from + 8;
                    debug_assert!(rank(self.ep_sq) == RANK_3);
                } else {
                    self.ep_sq = from - 8;
                    debug_assert!(rank(self.ep_sq) == RANK_6);
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

        if self.ep_sq != NO_SQUARE {
            hash_ep(&mut self.key, self.ep_sq);
        }

        self.ep_sq = NO_SQUARE;

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

        let Undo {
            m,
            castle_perm,
            ep_square,
            fifty_move_counter,
        } = self.history.pop().expect("No move to unmake!");

        let from = m.from();
        let to = m.to();

        if self.ep_sq != NO_SQUARE {
            hash_ep(&mut self.key, self.ep_sq);
        }

        // hash out the castling to insert it again after updating rights.
        hash_castling(&mut self.key, self.castle_perm);

        self.castle_perm = castle_perm;
        self.ep_sq = ep_square;
        self.fifty_move_counter = fifty_move_counter;

        if self.ep_sq != NO_SQUARE {
            hash_ep(&mut self.key, self.ep_sq);
        }

        // reinsert the castling rights
        hash_castling(&mut self.key, self.castle_perm);

        self.side ^= 1;
        hash_side(&mut self.key);

        if m.is_ep() {
            if self.side == WHITE {
                self.add_piece(to - 8, BP);
            } else {
                self.add_piece(to + 8, WP);
            }
        } else if m.is_castle() {
            match to {
                C1 => self.move_piece(D1, A1),
                C8 => self.move_piece(D8, A8),
                G1 => self.move_piece(F1, H1),
                G8 => self.move_piece(F8, H8),
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
            self.add_piece(
                from,
                if colour_of(m.promotion()) == WHITE {
                    WP
                } else {
                    BP
                },
            );
        }

        let something_removed = self.repetition_cache.remove(&self.key);
        debug_assert!(something_removed);

        #[cfg(debug_assertions)]
        self.check_validity().unwrap();
    }

    pub fn unmake_nullmove(&mut self) {
        #[cfg(debug_assertions)]
        self.check_validity().unwrap();

        self.height -= 1;
        self.ply -= 1;

        if self.ep_sq != NO_SQUARE {
            // this might be unreachable, but that's ok.
            // the branch predictor will hopefully figure it out.
            hash_ep(&mut self.key, self.ep_sq);
        }

        let Undo {
            m: _,
            castle_perm,
            ep_square,
            fifty_move_counter,
        } = self.history.pop().expect("No move to unmake!");

        self.castle_perm = castle_perm;
        self.ep_sq = ep_square;
        self.fifty_move_counter = fifty_move_counter;

        if self.ep_sq != NO_SQUARE {
            hash_ep(&mut self.key, self.ep_sq);
        }

        self.side ^= 1;
        hash_side(&mut self.key);

        #[cfg(debug_assertions)]
        self.check_validity().unwrap();
    }

    /// Parses Standard Algebraic Notation (SAN) and returns a move or a reason why it couldn't be parsed.
    pub fn parse_san(&self, san: &str) -> Result<Move, MoveParseError> {
        use crate::errors::MoveParseError::{
            IllegalMove, InvalidFromSquareFile, InvalidFromSquareRank, InvalidLength,
            InvalidPromotionPiece, InvalidToSquareFile, InvalidToSquareRank,
        };
        let san_bytes = san.as_bytes();
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

        let from = filerank_to_square(san_bytes[0] - b'a', san_bytes[1] - b'1');
        let to = filerank_to_square(san_bytes[2] - b'a', san_bytes[3] - b'1');

        let mut list = MoveList::new();
        // nasty hack: we can't do movegen when
        self.generate_moves(&mut list); 

        let res = list.iter()
            .copied()
            .find(|&m| {
                m.from() == from
                    && m.to() == to
                    && (san_bytes.len() == 4
                        || PROMO_CHAR_LOOKUP[m.promotion() as usize] == san_bytes[4])
            })
            .ok_or_else(|| IllegalMove(san.to_string()));

        res
    }

    /// Has the current position occurred before in the current game?
    pub fn is_repetition(&self) -> bool {
        self.repetition_cache.contains(&self.key)
    }

    /// Should we consider the current position a draw?
    pub fn is_draw(&self) -> bool {
        (self.fifty_move_counter >= 100 || self.is_repetition()) && self.height != 0
    }

    pub const fn num(&self, piece: u8) -> u8 {
        self.piece_lists[piece as usize].len()
    }

    pub fn reset_tables(&mut self) {
        self.history_table.clear();
        self.killer_move_table.fill([Move::NULL; 2]);
        self.counter_move_table.clear();
        self.followup_history.clear();
        self.height = 0;
        self.tt.clear_for_search();
        self.movegen_ready = true;
    }

    fn regenerate_pv_line(&mut self, depth: i32) {
        self.principal_variation.clear();

        while let ProbeResult::Hit(TTHit{ tt_move, .. }) = self.tt_probe(-INFINITY, INFINITY, MAX_DEPTH) {
            if self.principal_variation.len() < depth as usize
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

    /// Performs the root search. Returns the score of the position, from white's perspective.
    pub fn search_position(&mut self, info: &mut SearchInfo) -> i32 {
        self.reset_tables();
        info.clear_for_search();

        let first_legal = self.get_first_legal_move().unwrap_or(Move::NULL);

        let mut most_recent_move = first_legal;
        let mut most_recent_score = 0;
        let mut best_depth = 1;
        let (mut alpha, mut beta) = (-INFINITY, INFINITY);
        let max_depth = std::cmp::min(info.depth, MAX_DEPTH - 1).round();
        for i_depth in 0..=max_depth {
            let depth = Depth::new(i_depth);
            // main search
            assert!(self.height == 0, "height != 0 before aspiration search");
            let mut score = Self::alpha_beta::<true>(self, info, &mut Stack::new(), depth, alpha, beta);

            info.check_up();
            if info.stopped {
                break;
            }

            if score <= alpha || score >= beta {
                let score_string = format_score(score, self.turn());
                let boundstr = ["lowerbound", "upperbound"][usize::from(score <= alpha)];
                print!(
                    "info score {score_string} {boundstr} depth {i_depth} seldepth {} nodes {} time {} pv ",
                    info.seldepth.ply_to_horizon(),
                    info.nodes,
                    info.start_time.elapsed().as_millis()
                );
                self.regenerate_pv_line(best_depth);
                self.print_pv();
                // recalculate the score with a full window, as we failed either low or high.
                assert!(self.height == 0, "height != 0 before fullwindow search");
                score = Self::alpha_beta::<true>(self, info, &mut Stack::new(), depth, -INFINITY, INFINITY);
                info.check_up();
                if info.stopped {
                    break;
                }
            }

            most_recent_score = score;
            best_depth = i_depth;
            if !evaluation::is_mate_score(score) && i_depth > 4 {
                alpha = score - ASPIRATION_WINDOW;
                beta = score + ASPIRATION_WINDOW;
            } else {
                alpha = -INFINITY;
                beta = INFINITY;
            }
            self.regenerate_pv_line(best_depth);
            most_recent_move = *self
                .principal_variation
                .first()
                .unwrap_or(&most_recent_move);

            let score_string = format_score(most_recent_score, self.turn());
            print!(
                "info score {score_string} depth {i_depth} seldepth {} nodes {} time {} pv ",
                info.seldepth.ply_to_horizon(),
                info.nodes,
                info.start_time.elapsed().as_millis()
            );
            self.print_pv();
        }
        let score_string = format_score(most_recent_score, self.turn());
        print!(
            "info score {score_string} depth {best_depth} seldepth {} nodes {} time {} pv ",
            info.seldepth.ply_to_horizon(),
            info.nodes,
            info.start_time.elapsed().as_millis()
        );
        self.regenerate_pv_line(best_depth);
        self.print_pv();
        println!("bestmove {most_recent_move}");
        if self.side == WHITE {
            most_recent_score
        } else {
            -most_recent_score
        }
    }

    fn get_first_legal_move(&mut self) -> Option<Move> {
        let mut move_list = MoveList::new();
        self.generate_moves(&mut move_list);
        let mut first_legal = None;
        for &m in move_list.iter() {
            if self.make_move(m) {
                self.unmake_move();
                first_legal = Some(m);
            }
        }
        first_legal
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
                let sq = filerank_to_square(file, rank);
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
        board_1
            .set_from_fen(Board::STARTING_FEN)
            .expect("setfen failed.");
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
}
