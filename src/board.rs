#![allow(dead_code)]
#![allow(
    clippy::collapsible_else_if,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]

use std::fmt::{Debug, Display, Formatter};

use crate::{
    attack::{
        BLACK_JUMPERS, BLACK_SLIDERS, B_DIR, IS_BISHOPQUEEN, IS_KING, IS_KNIGHT, IS_ROOKQUEEN,
        K_DIRS, N_DIRS, Q_DIR, R_DIR, WHITE_JUMPERS, WHITE_SLIDERS,
    },
    bitboard::pop_lsb,
    chessmove::Move,
    definitions::{
        Castling, Colour, File, Rank, Square120, Undo, A1, A8, BB, BK, BLACK, BN, BOARD_N_SQUARES,
        BOTH, BP, BQ, BR, C1, C8, D1, D8, F1, F8, FIRST_ORDER_KILLER_SCORE, G1, G8, H1, H8,
        INFINITY, MAX_DEPTH, MAX_GAME_MOVES, NO_SQUARE, OFF_BOARD, PIECE_EMPTY, RANK_3, RANK_6,
        SECOND_ORDER_KILLER_SCORE, WB, WHITE, WK, WN, WP, WQ, WR,
    },
    errors::{FenParseError, MoveParseError, PositionValidityError},
    evaluation::{
        MoveCounter, BISHOP_PAIR_BONUS, BLACK_PASSED_BB, DOUBLED_PAWN_MALUS, DRAW_SCORE,
        EG_PIECE_VALUES, FILE_BB, ISOLATED_BB, ISOLATED_PAWN_MALUS, LAZY_THRESHOLD_1,
        MG_PIECE_VALUES, PASSED_PAWN_BONUS, PIECE_DANGER_VALUES, SHIELD_BONUS, WHITE_PASSED_BB,
    },
    lookups::{
        filerank_to_square, FILES_BOARD, MVV_LVA_SCORE, PIECE_BIG, PIECE_COL, PIECE_MAJ, PIECE_MIN,
        PROMO_CHAR_LOOKUP, RANKS_BOARD, SQ120_TO_SQ64, SQ64_TO_SQ120,
    },
    makemove::{hash_castling, hash_ep, hash_piece, hash_side, CASTLE_PERM_MASKS},
    movegen::{offset_square_offboard, MoveConsumer, MoveList},
    piecelist::PieceList,
    piecesquaretable::{endgame_pst_value, midgame_pst_value, ENDGAME_PST, MIDGAME_PST},
    search::alpha_beta,
    searchinfo::SearchInfo,
    transpositiontable::{DefaultTT, HFlag, ProbeResult},
    uci::format_score,
    validate::{piece_valid, piece_valid_empty, side_valid, square_on_board},
};

#[derive(Clone, PartialEq)]
pub struct Board {
    pieces: [u8; BOARD_N_SQUARES],
    pawns: [u64; 3],
    king_sq: [u8; 2],
    side: u8,
    ep_sq: u8,
    fifty_move_counter: u8,
    ply: usize,
    hist_ply: usize,
    key: u64,
    big_piece_counts: [u8; 2],
    major_piece_counts: [u8; 2],
    minor_piece_counts: [u8; 2],
    mg_material: [i32; 2],
    eg_material: [i32; 2],
    castle_perm: u8,
    history: Vec<Undo>,
    piece_lists: [PieceList; 13],

    principal_variation: Vec<Move>,

    history_table: [[i32; BOARD_N_SQUARES]; 13],
    killer_move_table: [[Move; 2]; MAX_DEPTH],
    tt: DefaultTT,

    pst_vals: [i32; 2],
}

impl Board {
    pub const STARTING_FEN: &'static str =
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    pub fn new() -> Self {
        let mut out = Self {
            pieces: [NO_SQUARE; BOARD_N_SQUARES],
            pawns: [0; 3],
            king_sq: [NO_SQUARE; 2],
            side: 0,
            ep_sq: NO_SQUARE,
            fifty_move_counter: 0,
            ply: 0,
            hist_ply: 0,
            key: 0,
            big_piece_counts: [0; 2],
            major_piece_counts: [0; 2],
            minor_piece_counts: [0; 2],
            mg_material: [0; 2],
            eg_material: [0; 2],
            castle_perm: 0,
            history: Vec::with_capacity(MAX_GAME_MOVES),
            piece_lists: [PieceList::new(); 13],
            principal_variation: Vec::with_capacity(MAX_DEPTH),
            history_table: [[0; BOARD_N_SQUARES]; 13],
            killer_move_table: [[Move::null(); 2]; MAX_DEPTH],
            pst_vals: [0; 2],
            tt: DefaultTT::new(),
        };
        out.reset();
        out
    }

    pub fn tt_store(&mut self, best_move: Move, score: i32, flag: HFlag, depth: usize) {
        self.tt
            .store(self.key, self.ply, best_move, score, flag, depth);
    }

    pub fn tt_probe(&mut self, alpha: i32, beta: i32, depth: usize) -> ProbeResult {
        self.tt.probe(self.key, self.ply, alpha, beta, depth)
    }

    pub fn clear_tt(&mut self) {
        self.tt.clear();
    }

    pub fn insert_killer(&mut self, m: Move) {
        debug_assert!(self.ply < MAX_DEPTH);
        let entry = unsafe { self.killer_move_table.get_unchecked_mut(self.ply) };
        entry[1] = entry[0];
        entry[0] = m;
    }

    pub fn insert_history(&mut self, m: Move, score: i32) {
        let from = m.from() as usize;
        let piece_moved = unsafe { *self.pieces.get_unchecked(from) as usize };
        let history_board = unsafe { self.history_table.get_unchecked_mut(piece_moved) };
        let to = m.to() as usize;
        unsafe {
            *history_board.get_unchecked_mut(to) += score;
        }
    }

    pub const US: u8 = 0;
    pub const THEM: u8 = 1;
    pub fn in_check<const SIDE: u8>(&self) -> bool {
        if SIDE == Self::US {
            let king_sq = unsafe { *self.king_sq.get_unchecked(self.side as usize) };
            self.sq_attacked(king_sq, self.side ^ 1)
        } else {
            let king_sq = unsafe { *self.king_sq.get_unchecked((self.side ^ 1) as usize) };
            self.sq_attacked(king_sq, self.side)
        }
    }

    pub fn zero_ply(&mut self) {
        self.ply = 0;
    }

    pub const fn ply(&self) -> usize {
        self.ply
    }

    pub const fn turn(&self) -> u8 {
        self.side
    }

    #[allow(clippy::wrong_self_convention)]
    pub fn is_terminal(&mut self) -> bool {
        self.is_checkmate() || self.is_stalemate() || self.is_draw()
    }

    #[allow(clippy::wrong_self_convention)]
    pub fn is_checkmate(&mut self) -> bool {
        if !self.in_check::<{ Self::US }>() {
            return false;
        }

        let mut moves = MoveList::new();
        self.generate_moves(&mut moves);

        for &m in moves.iter() {
            if !self.make_move(m) {
                continue;
            }
            self.unmake_move();
            return false;
        }

        true
    }

    #[allow(clippy::wrong_self_convention)]
    pub fn is_stalemate(&mut self) -> bool {
        if self.in_check::<{ Self::US }>() {
            return false;
        }

        let mut moves = MoveList::new();
        self.generate_moves(&mut moves);

        for &m in moves.iter() {
            if !self.make_move(m) {
                continue;
            }
            self.unmake_move();
            return false;
        }

        true
    }

    pub fn generate_pos_key(&self) -> u64 {
        #![allow(clippy::cast_possible_truncation)]
        let mut key = 0;
        for sq in 0..(BOARD_N_SQUARES as u8) {
            let piece = self.piece_at(sq);
            if piece != PIECE_EMPTY && piece != OFF_BOARD && piece != NO_SQUARE {
                debug_assert!((WP..=BK).contains(&piece));
                hash_piece(&mut key, piece, sq);
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
        self.pieces.fill(OFF_BOARD);
        for &sq in &SQ64_TO_SQ120 {
            *self.piece_at_mut(sq) = PIECE_EMPTY;
        }
        self.big_piece_counts.fill(0);
        self.major_piece_counts.fill(0);
        self.minor_piece_counts.fill(0);
        self.mg_material.fill(0);
        self.eg_material.fill(0);
        self.pawns.fill(0);
        self.piece_lists.iter_mut().for_each(PieceList::clear);
        self.king_sq = [NO_SQUARE, NO_SQUARE];
        self.side = Colour::Both as u8;
        self.ep_sq = NO_SQUARE;
        self.fifty_move_counter = 0;
        self.ply = 0;
        self.hist_ply = 0;
        self.castle_perm = 0;
        self.key = 0;
        self.pst_vals.fill(0);
        self.history.clear();
    }

    pub fn set_from_fen(&mut self, fen: &str) -> Result<(), FenParseError> {
        if !fen.is_ascii() {
            return Err(format!("FEN string is not ASCII: {}", fen));
        }

        let mut rank = Rank::Rank8 as u8;
        let mut file = File::FileA as u8;

        self.reset();

        let fen_chars = fen.as_bytes();
        let split_idx = fen_chars.iter().position(|&c| c == b' ').unwrap();
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
                    file = File::FileA as u8;
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
                let sq64 = rank * 8 + file;
                let sq120 = SQ64_TO_SQ120[sq64 as usize];
                if piece != PIECE_EMPTY {
                    *self.piece_at_mut(sq120) = piece;
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

        self.set_up_incremental_trackers();

        Ok(())
    }

    pub fn set_startpos(&mut self) {
        self.set_from_fen(Self::STARTING_FEN)
            .expect("for some reason, STARTING_FEN is now broken.");
    }

    pub fn from_fen(fen: &str) -> Result<Self, FenParseError> {
        let mut out = Self::new();
        out.set_from_fen(fen)?;
        Ok(out)
    }

    fn set_side(&mut self, side_part: Option<&[u8]>) -> Result<(), FenParseError> {
        self.side = match side_part {
            None => return Err("FEN string is invalid, expected side part.".to_string()),
            Some([b'w']) => WHITE,
            Some([b'b']) => BLACK,
            Some(other) => {
                return Err(format!(
                    "FEN string is invalid, expected side to be 'w' or 'b', got \"{}\"",
                    std::str::from_utf8(other).unwrap()
                ))
            }
        };
        Ok(())
    }

    fn set_castling(&mut self, castling_part: Option<&[u8]>) -> Result<(), FenParseError> {
        match castling_part {
            None => return Err("FEN string is invalid, expected castling part.".to_string()),
            Some([b'-']) => self.castle_perm = 0,
            Some(castling) => {
                for &c in castling {
                    match c {
                        b'K' => self.castle_perm |= Castling::WK as u8,
                        b'Q' => self.castle_perm |= Castling::WQ as u8,
                        b'k' => self.castle_perm |= Castling::BK as u8,
                        b'q' => self.castle_perm |= Castling::BQ as u8,
                        _ => return Err(format!("FEN string is invalid, expected castling part to be of the form 'KQkq', got \"{}\"", castling.iter().map(|&c| c as char).collect::<String>())),
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
                    return Err(format!("FEN string is invalid, expected en passant part to be of the form 'a1', got \"{}\"", ep_sq.iter().map(|&c| c as char).collect::<String>()));
                }
                let file = ep_sq[0] as u8 - b'a';
                let rank = ep_sq[1] as u8 - b'1';
                if !(file >= File::FileA as u8
                    && file <= File::FileH as u8
                    && rank >= Rank::Rank1 as u8
                    && rank <= Rank::Rank8 as u8)
                {
                    return Err(format!("FEN string is invalid, expected en passant part to be of the form 'a1', got \"{}\"", ep_sq.iter().map(|&c| c as char).collect::<String>()));
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
                self.ply = std::str::from_utf8(fullmove_number)
                    .map_err(|_| {
                        "FEN string is invalid, expected fullmove number part to be valid UTF-8"
                    })?
                    .parse::<usize>()
                    .map_err(|_| {
                        "FEN string is invalid, expected fullmove number part to be a number"
                    })?
                    * 2;
                if self.side == BLACK {
                    self.ply += 1;
                }
            }
        }

        Ok(())
    }

    fn set_up_incremental_trackers(&mut self) {
        for index in 0..BOARD_N_SQUARES {
            let sq: u8 = index.try_into().unwrap();
            let piece = self.piece_at(sq);
            if piece != OFF_BOARD && piece != PIECE_EMPTY {
                let colour = PIECE_COL[piece as usize];

                if PIECE_BIG[piece as usize] {
                    self.big_piece_counts[colour as usize] += 1;
                }
                if PIECE_MIN[piece as usize] {
                    self.minor_piece_counts[colour as usize] += 1;
                }
                if PIECE_MAJ[piece as usize] {
                    self.major_piece_counts[colour as usize] += 1;
                }

                self.mg_material[colour as usize] += MG_PIECE_VALUES[piece as usize];
                self.eg_material[colour as usize] += EG_PIECE_VALUES[piece as usize];

                self.piece_lists[piece as usize].insert(sq);

                if piece == WK || piece == BK {
                    self.king_sq[colour as usize] = sq;
                }

                if piece == WP || piece == BP {
                    self.pawns[colour as usize] |= 1 << SQ120_TO_SQ64[sq as usize];
                    self.pawns[Colour::Both as usize] |= 1 << SQ120_TO_SQ64[sq as usize];
                }
            }
        }
        (self.pst_vals[0], self.pst_vals[1]) = self.generate_pst_value();
    }

    #[allow(clippy::cognitive_complexity, clippy::too_many_lines)]
    pub fn check_validity(&self) -> Result<(), PositionValidityError> {
        #![allow(clippy::similar_names)]
        use Colour::{Black, Both, White};
        let mut piece_num = [0u8; 13];
        let mut big_pce = [0, 0];
        let mut maj_pce = [0, 0];
        let mut min_pce = [0, 0];
        let mut mg_material = [0, 0];
        let mut eg_material = [0, 0];

        let mut pawns = self.pawns;

        // check piece lists
        for piece in WP..=BK {
            for &sq120 in self.piece_lists[piece as usize].iter() {
                if self.piece_at(sq120) != piece {
                    return Err(format!(
                        "piece list corrupt: expected slot {} to be {} but was {}",
                        sq120,
                        piece,
                        self.piece_at(sq120)
                    ));
                }
            }
        }

        // check piece count and other counters
        for &sq120 in &SQ64_TO_SQ120 {
            let piece = self.piece_at(sq120);
            piece_num[piece as usize] += 1;
            let colour = PIECE_COL[piece as usize];
            if PIECE_BIG[piece as usize] {
                big_pce[colour as usize] += 1;
            }
            if PIECE_MAJ[piece as usize] {
                maj_pce[colour as usize] += 1;
            }
            if PIECE_MIN[piece as usize] {
                min_pce[colour as usize] += 1;
            }
            if colour != Both {
                mg_material[colour as usize] += MG_PIECE_VALUES[piece as usize];
                eg_material[colour as usize] += EG_PIECE_VALUES[piece as usize];
            }
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

        // check bitboards' count
        if pawns[White as usize].count_ones() != u32::from(self.num(WP)) {
            return Err(format!(
                "white pawn bitboard is corrupt: expected {:?}, got {:?}",
                self.num(WP),
                pawns[White as usize].count_ones()
            ));
        }
        if pawns[Black as usize].count_ones() != u32::from(self.num(BP)) {
            return Err(format!(
                "black pawn bitboard is corrupt: expected {:?}, got {:?}",
                self.num(BP),
                pawns[Black as usize].count_ones()
            ));
        }
        if pawns[Both as usize].count_ones() != u32::from(self.num(WP)) + u32::from(self.num(BP)) {
            return Err(format!(
                "both pawns bitboard is corrupt: expected {:?}, got {:?}",
                self.num(WP) + self.num(BP),
                pawns[Both as usize].count_ones()
            ));
        }

        // check bitboards' squares
        while pawns[White as usize] > 0 {
            let sq64 = pop_lsb(&mut pawns[White as usize]);
            if self.piece_at(SQ64_TO_SQ120[sq64 as usize]) != WP {
                return Err(format!(
                    "white pawn bitboard is corrupt: expected white pawn, got {:?}",
                    self.piece_at(SQ64_TO_SQ120[sq64 as usize])
                ));
            }
        }

        while pawns[Black as usize] > 0 {
            let sq64 = pop_lsb(&mut pawns[Black as usize]);
            if self.piece_at(SQ64_TO_SQ120[sq64 as usize]) != BP {
                return Err(format!(
                    "black pawn bitboard is corrupt: expected black pawn, got {:?}",
                    self.piece_at(SQ64_TO_SQ120[sq64 as usize])
                ));
            }
        }

        while pawns[Both as usize] > 0 {
            let sq64 = pop_lsb(&mut pawns[Both as usize]);
            if !(self.piece_at(SQ64_TO_SQ120[sq64 as usize]) == WP
                || self.piece_at(SQ64_TO_SQ120[sq64 as usize]) == BP)
            {
                return Err(format!(
                    "both pawns bitboard is corrupt: expected white or black pawn, got {:?}",
                    self.piece_at(SQ64_TO_SQ120[sq64 as usize])
                ));
            }
        }

        if mg_material[White as usize] != self.mg_material[White as usize] {
            return Err(format!(
                "white midgame material is corrupt: expected {:?}, got {:?}",
                mg_material[White as usize], self.mg_material[White as usize]
            ));
        }
        if eg_material[White as usize] != self.eg_material[White as usize] {
            return Err(format!(
                "white endgame material is corrupt: expected {:?}, got {:?}",
                eg_material[White as usize], self.eg_material[White as usize]
            ));
        }
        if mg_material[Black as usize] != self.mg_material[Black as usize] {
            return Err(format!(
                "black midgame material is corrupt: expected {:?}, got {:?}",
                mg_material[Black as usize], self.mg_material[Black as usize]
            ));
        }
        if eg_material[Black as usize] != self.eg_material[Black as usize] {
            return Err(format!(
                "black endgame material is corrupt: expected {:?}, got {:?}",
                eg_material[Black as usize], self.eg_material[Black as usize]
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
            || (RANKS_BOARD[self.ep_sq as usize] == Rank::Rank6 as u8 && self.side == WHITE)
            || (RANKS_BOARD[self.ep_sq as usize] == Rank::Rank3 as u8 && self.side == BLACK))
        {
            return Err(format!("en passant square is corrupt: expected square to be {} (NoSquare) or to be on ranks 6 or 3, got {} (Rank {})", NO_SQUARE, self.ep_sq, RANKS_BOARD[self.ep_sq as usize]));
        }

        if self.fifty_move_counter >= 100 {
            return Err(format!(
                "fifty move counter is corrupt: expected 0-99, got {}",
                self.fifty_move_counter
            ));
        }

        if self.piece_at(self.king_sq[White as usize]) != WK {
            return Err(format!(
                "white king square is corrupt: expected white king, got {:?}",
                self.piece_at(self.king_sq[White as usize])
            ));
        }
        if self.piece_at(self.king_sq[Black as usize]) != BK {
            return Err(format!(
                "black king square is corrupt: expected black king, got {:?}",
                self.piece_at(self.king_sq[Black as usize])
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
            if self.piece_at(sq - 11) == WP || self.piece_at(sq - 9) == WP {
                return true;
            }
        } else {
            if self.piece_at(sq + 11) == BP || self.piece_at(sq + 9) == BP {
                return true;
            }
        }

        // knights
        for &dir in &N_DIRS {
            let t_sq = (sq as i8 + dir) as u8;
            let p = self.piece_at(t_sq);
            if p != OFF_BOARD && IS_KNIGHT[p as usize] && PIECE_COL[p as usize] as u8 == side {
                return true;
            }
        }

        // rooks, queens
        for &dir in &R_DIR {
            let mut t_sq = sq as i8 + dir;
            let mut piece = self.piece_at(t_sq as u8);
            while piece != OFF_BOARD {
                if piece != PIECE_EMPTY {
                    if IS_ROOKQUEEN[piece as usize] && PIECE_COL[piece as usize] as u8 == side {
                        return true;
                    }
                    break;
                }
                t_sq += dir;
                piece = self.piece_at(t_sq as u8);
            }
        }

        // bishops, queens
        for &dir in &B_DIR {
            let mut t_sq = sq as i8 + dir;
            let mut piece = self.piece_at(t_sq as u8);
            while piece != OFF_BOARD {
                if piece != PIECE_EMPTY {
                    if IS_BISHOPQUEEN[piece as usize] && PIECE_COL[piece as usize] as u8 == side {
                        return true;
                    }
                    break;
                }
                t_sq += dir;
                piece = self.piece_at(t_sq as u8);
            }
        }

        // king
        for &dir in &K_DIRS {
            let t_sq = (sq as i8 + dir) as u8;
            let p = self.piece_at(t_sq);
            if p != OFF_BOARD && IS_KING[p as usize] && PIECE_COL[p as usize] as u8 == side {
                return true;
            }
        }

        false
    }

    #[inline]
    fn add_quiet_move(&self, m: Move, move_list: &mut impl MoveConsumer) {
        let _ = self;
        debug_assert!(square_on_board(m.from()));
        debug_assert!(square_on_board(m.to()));

        let killer_entry = unsafe { self.killer_move_table.get_unchecked(self.ply) };

        let score = if killer_entry[0] == m {
            FIRST_ORDER_KILLER_SCORE
        } else if killer_entry[1] == m {
            SECOND_ORDER_KILLER_SCORE
        } else {
            let from = m.from() as usize;
            let to = m.to() as usize;
            let piece_moved = unsafe { *self.pieces.get_unchecked(from) as usize };
            unsafe {
                *self
                    .history_table
                    .get_unchecked(piece_moved)
                    .get_unchecked(to)
            }
        };

        move_list.push(m, score);
    }

    #[inline]
    fn add_capture_move(&self, m: Move, move_list: &mut impl MoveConsumer) {
        debug_assert!(square_on_board(m.from()));
        debug_assert!(square_on_board(m.to()));
        debug_assert!(piece_valid(m.capture()));

        let capture = m.capture() as usize;
        let from = m.from() as usize;
        let piece_moved = unsafe { *self.pieces.get_unchecked(from) as usize };
        let mmvlva = unsafe {
            *MVV_LVA_SCORE
                .get_unchecked(capture)
                .get_unchecked(piece_moved)
        };

        let score = mmvlva + 10_000_000;
        move_list.push(m, score);
    }

    fn add_ep_move(&self, m: Move, move_list: &mut impl MoveConsumer) {
        let _ = self;
        move_list.push(m, 1050 + 10_000_000);
    }

    fn add_pawn_cap_move<const SIDE: u8, MC: MoveConsumer>(
        &self,
        from: u8,
        to: u8,
        cap: u8,
        move_list: &mut MC,
    ) {
        debug_assert!(piece_valid_empty(cap));
        debug_assert!(square_on_board(from));
        debug_assert!(square_on_board(to));
        let promo_rank = if SIDE == WHITE {
            Rank::Rank7 as u8
        } else {
            Rank::Rank2 as u8
        };
        if RANKS_BOARD[from as usize] == promo_rank {
            if SIDE == WHITE {
                for &promo in &[WQ, WN, WR, WB] {
                    self.add_capture_move(Move::new(from, to, cap, promo, 0), move_list);
                }
            } else {
                for &promo in &[BQ, BN, BR, BB] {
                    self.add_capture_move(Move::new(from, to, cap, promo, 0), move_list);
                }
            };
        } else {
            self.add_capture_move(Move::new(from, to, cap, PIECE_EMPTY, 0), move_list);
        }
    }

    fn add_pawn_move<const SIDE: u8, MC: MoveConsumer>(
        &self,
        from: u8,
        to: u8,
        move_list: &mut MC,
    ) {
        debug_assert!(square_on_board(from));
        debug_assert!(square_on_board(to));
        let promo_rank = if SIDE == WHITE {
            Rank::Rank7 as u8
        } else {
            Rank::Rank2 as u8
        };
        if RANKS_BOARD[from as usize] == promo_rank {
            if SIDE == WHITE {
                for &promo in &[WQ, WN, WR, WB] {
                    self.add_quiet_move(Move::new(from, to, PIECE_EMPTY, promo, 0), move_list);
                }
            } else {
                for &promo in &[BQ, BN, BR, BB] {
                    self.add_quiet_move(Move::new(from, to, PIECE_EMPTY, promo, 0), move_list);
                }
            };
        } else {
            self.add_quiet_move(Move::new(from, to, PIECE_EMPTY, PIECE_EMPTY, 0), move_list);
        }
    }

    fn generate_pawn_caps<const SIDE: u8, MC: MoveConsumer>(&self, sq: u8, move_list: &mut MC) {
        let left_sq = if SIDE == WHITE { sq + 9 } else { sq - 9 };
        let right_sq = if SIDE == WHITE { sq + 11 } else { sq - 11 };
        if square_on_board(left_sq) && PIECE_COL[self.piece_at(left_sq) as usize] as u8 == SIDE ^ 1
        {
            self.add_pawn_cap_move::<SIDE, MC>(sq, left_sq, self.piece_at(left_sq), move_list);
        }
        if square_on_board(right_sq)
            && PIECE_COL[self.piece_at(right_sq) as usize] as u8 == SIDE ^ 1
        {
            self.add_pawn_cap_move::<SIDE, MC>(sq, right_sq, self.piece_at(right_sq), move_list);
        }
    }

    fn generate_ep<const SIDE: u8, MC: MoveConsumer>(&self, sq: u8, move_list: &mut MC) {
        if self.ep_sq == NO_SQUARE {
            return;
        }
        let left_sq = if SIDE == WHITE { sq + 9 } else { sq - 9 };
        let right_sq = if SIDE == WHITE { sq + 11 } else { sq - 11 };
        if left_sq == self.ep_sq {
            self.add_ep_move(
                Move::new(sq, left_sq, PIECE_EMPTY, PIECE_EMPTY, Move::EP_MASK),
                move_list,
            );
        }
        if right_sq == self.ep_sq {
            self.add_ep_move(
                Move::new(sq, right_sq, PIECE_EMPTY, PIECE_EMPTY, Move::EP_MASK),
                move_list,
            );
        }
    }

    fn generate_pawn_forward<const SIDE: u8, MC: MoveConsumer>(&self, sq: u8, move_list: &mut MC) {
        let start_rank = if SIDE == WHITE {
            Rank::Rank2 as u8
        } else {
            Rank::Rank7 as u8
        };
        let offset_sq = if SIDE == WHITE { sq + 10 } else { sq - 10 };
        if self.piece_at(offset_sq) == PIECE_EMPTY {
            self.add_pawn_move::<SIDE, MC>(sq, offset_sq, move_list);
            let double_sq = if SIDE == WHITE { sq + 20 } else { sq - 20 };
            if RANKS_BOARD[sq as usize] == start_rank && self.piece_at(double_sq) == PIECE_EMPTY {
                self.add_quiet_move(
                    Move::new(
                        sq,
                        double_sq,
                        PIECE_EMPTY,
                        PIECE_EMPTY,
                        Move::PAWN_START_MASK,
                    ),
                    move_list,
                );
            }
        }
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    pub fn generate_moves<MC: MoveConsumer>(&self, move_list: &mut MC) {
        #[cfg(debug_assertions)]
        self.check_validity().unwrap();

        if MC::DO_PAWN_MOVEGEN {
            if self.side == WHITE {
                for &sq in self.piece_lists[WP as usize].iter() {
                    debug_assert!(square_on_board(sq));
                    self.generate_pawn_forward::<{ WHITE }, MC>(sq, move_list);
                    self.generate_pawn_caps::<{ WHITE }, MC>(sq, move_list);
                    self.generate_ep::<{ WHITE }, MC>(sq, move_list);
                }
            } else {
                for &sq in self.piece_lists[BP as usize].iter() {
                    debug_assert!(square_on_board(sq));
                    self.generate_pawn_forward::<{ BLACK }, MC>(sq, move_list);
                    self.generate_pawn_caps::<{ BLACK }, MC>(sq, move_list);
                    self.generate_ep::<{ BLACK }, MC>(sq, move_list);
                }
            }
        }

        let jumpers = if self.side == WHITE {
            &WHITE_JUMPERS
        } else {
            &BLACK_JUMPERS
        };
        for &piece in jumpers {
            let dirs = if piece == WN || piece == BN {
                &N_DIRS
            } else {
                &K_DIRS
            };
            for &sq in self.piece_lists[piece as usize].iter() {
                debug_assert!(square_on_board(sq));
                for &offset in dirs {
                    let t_sq = sq as i8 + offset;
                    if offset_square_offboard(t_sq) {
                        continue;
                    }

                    // now safe to convert to u8
                    // as offset_square_offboard() is false
                    let t_sq: u8 = unsafe { t_sq.try_into().unwrap_unchecked() };

                    if self.piece_at(t_sq) != PIECE_EMPTY {
                        if PIECE_COL[self.piece_at(t_sq) as usize] as u8 == self.side ^ 1 {
                            self.add_capture_move(
                                Move::new(sq, t_sq, self.piece_at(t_sq), PIECE_EMPTY, 0),
                                move_list,
                            );
                        }
                    } else {
                        self.add_quiet_move(
                            Move::new(sq, t_sq, PIECE_EMPTY, PIECE_EMPTY, 0),
                            move_list,
                        );
                    }
                }
            }
        }

        let sliders = if self.side == WHITE {
            &WHITE_SLIDERS
        } else {
            &BLACK_SLIDERS
        };
        for &piece in sliders {
            debug_assert!(piece_valid(piece));
            let dirs: &[i8] = match piece {
                WB | BB => &B_DIR,
                WR | BR => &R_DIR,
                WQ | BQ => &Q_DIR,
                _ => unsafe { std::hint::unreachable_unchecked() },
            };
            for &sq in self.piece_lists[piece as usize].iter() {
                debug_assert!(square_on_board(sq));

                for &dir in dirs {
                    let mut slider = sq as i8 + dir;
                    while !offset_square_offboard(slider) {
                        // now safe to convert to u8
                        // as offset_square_offboard() is false
                        let t_sq: u8 = unsafe { slider.try_into().unwrap_unchecked() };

                        if self.piece_at(t_sq) != PIECE_EMPTY {
                            if PIECE_COL[self.piece_at(t_sq) as usize] as u8 == self.side ^ 1 {
                                self.add_capture_move(
                                    Move::new(sq, t_sq, self.piece_at(t_sq), PIECE_EMPTY, 0),
                                    move_list,
                                );
                            }
                            break;
                        }
                        self.add_quiet_move(
                            Move::new(sq, t_sq, PIECE_EMPTY, PIECE_EMPTY, 0),
                            move_list,
                        );
                        slider += dir;
                    }
                }
            }
        }

        self.generate_castling_moves(move_list);
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    pub fn generate_captures<MC: MoveConsumer>(&self, move_list: &mut MC) {
        #[cfg(debug_assertions)]
        self.check_validity().unwrap();

        // both pawn moves and captures
        if self.side == WHITE {
            for &sq in self.piece_lists[WP as usize].iter() {
                debug_assert!(square_on_board(sq));
                self.generate_pawn_caps::<{ WHITE }, MC>(sq, move_list);
                self.generate_ep::<{ WHITE }, MC>(sq, move_list);
            }
        } else {
            for &sq in self.piece_lists[BP as usize].iter() {
                debug_assert!(square_on_board(sq));
                self.generate_pawn_caps::<{ BLACK }, MC>(sq, move_list);
                self.generate_ep::<{ BLACK }, MC>(sq, move_list);
            }
        }

        let jumpers = if self.side == WHITE {
            &WHITE_JUMPERS
        } else {
            &BLACK_JUMPERS
        };
        for &piece in jumpers {
            let dirs = if piece == WN || piece == BN {
                &N_DIRS
            } else {
                &K_DIRS
            };
            for &sq in self.piece_lists[piece as usize].iter() {
                debug_assert!(square_on_board(sq));
                for &offset in dirs {
                    let t_sq = sq as i8 + offset;
                    if offset_square_offboard(t_sq) {
                        continue;
                    }

                    // now safe to convert to u8
                    // as offset_square_offboard() is false
                    let t_sq: u8 = unsafe { t_sq.try_into().unwrap_unchecked() };

                    if self.piece_at(t_sq) != PIECE_EMPTY
                        && PIECE_COL[self.piece_at(t_sq) as usize] as u8 == self.side ^ 1
                    {
                        self.add_capture_move(
                            Move::new(sq, t_sq, self.piece_at(t_sq), PIECE_EMPTY, 0),
                            move_list,
                        );
                    }
                }
            }
        }

        let sliders = if self.side == WHITE {
            &WHITE_SLIDERS
        } else {
            &BLACK_SLIDERS
        };
        for &piece in sliders {
            debug_assert!(piece_valid(piece));
            let dirs: &[i8] = match piece {
                WB | BB => &B_DIR,
                WR | BR => &R_DIR,
                WQ | BQ => &Q_DIR,
                _ => unsafe { std::hint::unreachable_unchecked() },
            };
            for &sq in self.piece_lists[piece as usize].iter() {
                debug_assert!(square_on_board(sq));

                for &dir in dirs {
                    let mut slider = sq as i8 + dir;
                    while !offset_square_offboard(slider) {
                        // now safe to convert to u8
                        // as offset_square_offboard() is false
                        let t_sq: u8 = unsafe { slider.try_into().unwrap_unchecked() };

                        if self.piece_at(t_sq) != PIECE_EMPTY {
                            if PIECE_COL[self.piece_at(t_sq) as usize] as u8 == self.side ^ 1 {
                                self.add_capture_move(
                                    Move::new(sq, t_sq, self.piece_at(t_sq), PIECE_EMPTY, 0),
                                    move_list,
                                );
                            }
                            break;
                        }
                        slider += dir;
                    }
                }
            }
        }
    }

    fn generate_castling_moves(&self, move_list: &mut impl MoveConsumer) {
        if self.side == WHITE {
            if (self.castle_perm & Castling::WK as u8) != 0
                && self.piece_at(Square120::F1 as u8) == PIECE_EMPTY
                && self.piece_at(Square120::G1 as u8) == PIECE_EMPTY
                && !self.sq_attacked(Square120::E1 as u8, BLACK)
                && !self.sq_attacked(Square120::F1 as u8, BLACK)
            {
                self.add_quiet_move(
                    Move::new(
                        Square120::E1 as u8,
                        Square120::G1 as u8,
                        PIECE_EMPTY,
                        PIECE_EMPTY,
                        Move::CASTLE_MASK,
                    ),
                    move_list,
                );
            }

            if (self.castle_perm & Castling::WQ as u8) != 0
                && self.piece_at(Square120::D1 as u8) == PIECE_EMPTY
                && self.piece_at(Square120::C1 as u8) == PIECE_EMPTY
                && self.piece_at(Square120::B1 as u8) == PIECE_EMPTY
                && !self.sq_attacked(Square120::E1 as u8, BLACK)
                && !self.sq_attacked(Square120::D1 as u8, BLACK)
            {
                self.add_quiet_move(
                    Move::new(
                        Square120::E1 as u8,
                        Square120::C1 as u8,
                        PIECE_EMPTY,
                        PIECE_EMPTY,
                        Move::CASTLE_MASK,
                    ),
                    move_list,
                );
            }
        } else {
            if (self.castle_perm & Castling::BK as u8) != 0
                && self.piece_at(Square120::F8 as u8) == PIECE_EMPTY
                && self.piece_at(Square120::G8 as u8) == PIECE_EMPTY
                && !self.sq_attacked(Square120::E8 as u8, WHITE)
                && !self.sq_attacked(Square120::F8 as u8, WHITE)
            {
                self.add_quiet_move(
                    Move::new(
                        Square120::E8 as u8,
                        Square120::G8 as u8,
                        PIECE_EMPTY,
                        PIECE_EMPTY,
                        Move::CASTLE_MASK,
                    ),
                    move_list,
                );
            }

            if (self.castle_perm & Castling::BQ as u8) != 0
                && self.piece_at(Square120::D8 as u8) == PIECE_EMPTY
                && self.piece_at(Square120::C8 as u8) == PIECE_EMPTY
                && self.piece_at(Square120::B8 as u8) == PIECE_EMPTY
                && !self.sq_attacked(Square120::E8 as u8, WHITE)
                && !self.sq_attacked(Square120::D8 as u8, WHITE)
            {
                self.add_quiet_move(
                    Move::new(
                        Square120::E8 as u8,
                        Square120::C8 as u8,
                        PIECE_EMPTY,
                        PIECE_EMPTY,
                        Move::CASTLE_MASK,
                    ),
                    move_list,
                );
            }
        }
    }

    /// Checks if a move is legal in the current position.
    /// Because moves must be played and unplayed, this method
    /// requires a mutable reference to the position.
    /// Despite this, you should expect the state of a Board
    /// object to be the same before and after calling [`is_legal`].
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

    /// An immutable version of [`is_legal`].
    /// Because the entire board state must be copied, this method
    /// is much less efficient than [`is_legal`], and is only provided
    /// for convenience in the case where you need to to do legality
    /// checking with an immutable Board object.
    #[deprecated(note = "prefer Board::is_legal")]
    pub fn is_legal_immutable(&self, move_to_check: Move) -> bool {
        let mut board = self.clone();

        let mut list = MoveList::new();
        board.generate_moves(&mut list);

        for &m in list.iter() {
            if !board.make_move(m) {
                continue;
            }
            board.unmake_move();
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

        let colour = PIECE_COL[piece as usize] as u8;

        hash_piece(&mut self.key, piece, sq);

        *self.piece_at_mut(sq) = PIECE_EMPTY;
        self.mg_material[colour as usize] -= MG_PIECE_VALUES[piece as usize];
        self.eg_material[colour as usize] -= EG_PIECE_VALUES[piece as usize];
        self.pst_vals[0] -= midgame_pst_value(piece, sq);
        self.pst_vals[1] -= endgame_pst_value(piece, sq);

        if PIECE_BIG[piece as usize] {
            self.big_piece_counts[colour as usize] -= 1;
            if PIECE_MAJ[piece as usize] {
                self.major_piece_counts[colour as usize] -= 1;
            } else {
                self.minor_piece_counts[colour as usize] -= 1;
            }
        } else {
            let sq64 = SQ120_TO_SQ64[sq as usize];
            self.pawns[colour as usize] &= !(1 << sq64);
            self.pawns[Colour::Both as usize] &= !(1 << sq64);
        }

        self.piece_lists[piece as usize].remove(sq);
    }

    fn add_piece(&mut self, sq: u8, piece: u8) {
        debug_assert!(piece_valid(piece));
        debug_assert!(square_on_board(sq));

        let colour = PIECE_COL[piece as usize] as u8;

        hash_piece(&mut self.key, piece, sq);

        *self.piece_at_mut(sq) = piece;
        self.mg_material[colour as usize] += MG_PIECE_VALUES[piece as usize];
        self.eg_material[colour as usize] += EG_PIECE_VALUES[piece as usize];
        self.pst_vals[0] += midgame_pst_value(piece, sq);
        self.pst_vals[1] += endgame_pst_value(piece, sq);

        if PIECE_BIG[piece as usize] {
            self.big_piece_counts[colour as usize] += 1;
            if PIECE_MAJ[piece as usize] {
                self.major_piece_counts[colour as usize] += 1;
            } else {
                self.minor_piece_counts[colour as usize] += 1;
            }
        } else {
            let sq64 = SQ120_TO_SQ64[sq as usize];
            self.pawns[colour as usize] |= 1 << sq64;
            self.pawns[Colour::Both as usize] |= 1 << sq64;
        }

        self.piece_lists[piece as usize].insert(sq);
    }

    fn move_piece(&mut self, from: u8, to: u8) {
        debug_assert!(square_on_board(from));
        debug_assert!(square_on_board(to));

        let piece = self.piece_at(from);
        let colour = PIECE_COL[piece as usize] as u8;

        // if we're in debug mode, check that we actually find a matching piecelist entry.
        #[cfg(debug_assertions)]
        let mut t_piece_num = false;

        hash_piece(&mut self.key, piece, from);
        hash_piece(&mut self.key, piece, to);

        *self.piece_at_mut(from) = PIECE_EMPTY;
        self.pst_vals[0] -= midgame_pst_value(piece, from);
        self.pst_vals[1] -= endgame_pst_value(piece, from);
        *self.piece_at_mut(to) = piece;
        self.pst_vals[0] += midgame_pst_value(piece, to);
        self.pst_vals[1] += endgame_pst_value(piece, to);

        if piece == WP || piece == BP {
            let sq64from = SQ120_TO_SQ64[from as usize];
            self.pawns[colour as usize] &= !(1 << sq64from);
            self.pawns[Colour::Both as usize] &= !(1 << sq64from);

            let sq64to = SQ120_TO_SQ64[to as usize];
            self.pawns[colour as usize] |= 1 << sq64to;
            self.pawns[Colour::Both as usize] |= 1 << sq64to;
        }

        for sq in self.piece_lists[piece as usize].iter_mut() {
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

    #[inline]
    pub fn moved_piece(&self, m: Move) -> u8 {
        debug_assert!(square_on_board(m.from()));
        unsafe { *self.pieces.get_unchecked(m.from() as usize) }
    }

    #[inline]
    pub fn piece_at(&self, sq: u8) -> u8 {
        debug_assert!((sq as usize) < BOARD_N_SQUARES);
        unsafe { *self.pieces.get_unchecked(sq as usize) }
    }

    #[inline]
    pub fn piece_at_mut(&mut self, sq: u8) -> &mut u8 {
        debug_assert!((sq as usize) < BOARD_N_SQUARES);
        unsafe { self.pieces.get_unchecked_mut(sq as usize) }
    }

    #[allow(clippy::cognitive_complexity)]
    pub fn make_move(&mut self, m: Move) -> bool {
        #[cfg(debug_assertions)]
        self.check_validity().unwrap();

        let from = m.from();
        let to = m.to();
        let side = self.side;
        let piece = self.piece_at(from);

        debug_assert!(square_on_board(from));
        debug_assert!(square_on_board(to));
        debug_assert!(side_valid(side));
        debug_assert!(piece_valid(piece), "piece: {:?}", piece);

        let saved_key = self.key;

        if m.is_ep() {
            if side == WHITE {
                self.clear_piece(to - 10);
            } else {
                self.clear_piece(to + 10);
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
            position_key: saved_key,
        });

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

        self.hist_ply += 1;
        self.ply += 1;

        if piece == WP || piece == BP {
            self.fifty_move_counter = 0;
            if m.is_pawn_start() {
                if side == WHITE {
                    self.ep_sq = from + 10;
                    debug_assert!(RANKS_BOARD[self.ep_sq as usize] == RANK_3);
                } else {
                    self.ep_sq = from - 10;
                    debug_assert!(RANKS_BOARD[self.ep_sq as usize] == RANK_6);
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

        if piece == WK || piece == BK {
            unsafe {
                *self.king_sq.get_unchecked_mut(side as usize) = to;
            }
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

        self.ply += 1;
        self.history.push(Undo {
            m: Move::null(),
            castle_perm: self.castle_perm,
            ep_square: self.ep_sq,
            fifty_move_counter: self.fifty_move_counter,
            position_key: self.key,
        });

        if self.ep_sq != NO_SQUARE {
            hash_ep(&mut self.key, self.ep_sq);
        }

        self.ep_sq = NO_SQUARE;

        self.side ^= 1;
        self.hist_ply += 1;
        hash_side(&mut self.key);

        #[cfg(debug_assertions)]
        self.check_validity().unwrap();
    }

    pub fn unmake_move(&mut self) {
        #[cfg(debug_assertions)]
        self.check_validity().unwrap();

        self.ply -= 1;
        self.hist_ply -= 1;

        let Undo {
            m,
            castle_perm,
            ep_square,
            fifty_move_counter,
            position_key: _, // this is sus.
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
                self.add_piece(to - 10, BP);
            } else {
                self.add_piece(to + 10, WP);
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

        let from_piece = self.piece_at(from);
        if from_piece == WK || from_piece == BK {
            self.king_sq[self.side as usize] = from;
        }

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
                if PIECE_COL[m.promotion() as usize] as u8 == WHITE {
                    WP
                } else {
                    BP
                },
            );
        }

        #[cfg(debug_assertions)]
        self.check_validity().unwrap();
    }

    pub fn unmake_nullmove(&mut self) {
        #[cfg(debug_assertions)]
        self.check_validity().unwrap();

        self.ply -= 1;
        self.hist_ply -= 1;

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
            position_key: _, // this is sus.
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

    pub const fn zugzwang_unlikely(&self) -> bool {
        self.big_piece_counts[self.side as usize] > 0
    }

    /// Parses Standard Algebraic Notation (SAN) and returns a move or a reason why it couldn't be parsed.
    pub fn parse_san(&self, san: &str) -> Result<Move, MoveParseError> {
        use crate::errors::MoveParseError::{
            IllegalMove, InvalidFromSquareFile, InvalidFromSquareRank, InvalidLength,
            InvalidPromotionPiece, InvalidToSquareFile, InvalidToSquareRank,
        };
        let san_bytes = san.as_bytes();
        if !(4..=5).contains(&san_bytes.len()) {
            return Err(InvalidLength);
        }
        if !(b'a'..=b'h').contains(&san_bytes[0]) {
            return Err(InvalidFromSquareFile);
        }
        if !(b'1'..=b'8').contains(&san_bytes[1]) {
            return Err(InvalidFromSquareRank);
        }
        if !(b'a'..=b'h').contains(&san_bytes[2]) {
            return Err(InvalidToSquareFile);
        }
        if !(b'1'..=b'8').contains(&san_bytes[3]) {
            return Err(InvalidToSquareRank);
        }
        if san_bytes.len() == 5 && ![b'n', b'b', b'r', b'q', b'k'].contains(&san_bytes[4]) {
            return Err(InvalidPromotionPiece);
        }

        let from = filerank_to_square(san_bytes[0] - b'a', san_bytes[1] - b'1');
        let to = filerank_to_square(san_bytes[2] - b'a', san_bytes[3] - b'1');

        let mut list = MoveList::new();
        self.generate_moves(&mut list);

        let mut moves = list.iter();

        moves
            .find(|&m| {
                m.from() == from
                    && m.to() == to
                    && (san_bytes.len() == 4
                        || PROMO_CHAR_LOOKUP[m.promotion() as usize] == san_bytes[4])
            })
            .ok_or(IllegalMove)
            .copied()
    }

    pub fn is_repetition(&self) -> bool {
        for entry in self.history.iter().rev() {
            if entry.position_key == self.key {
                return true;
            }
            if entry.fifty_move_counter == 0 {
                break;
            }
        }
        false
    }

    pub fn is_draw(&self) -> bool {
        (self.fifty_move_counter >= 100 || self.is_repetition()) && self.ply != 0
    }

    fn get_pv_line(&mut self, depth: usize) -> usize {
        self.principal_variation.clear();

        if depth >= MAX_DEPTH {
            return 0;
        }

        let mut entry = self.tt_probe(-INFINITY, INFINITY, MAX_DEPTH);

        let mut moves_done = 0;

        while let ProbeResult::BestMove(pv_move) = entry {
            if self.is_legal(pv_move) && moves_done < depth {
                self.make_move(pv_move);
                self.principal_variation.push(pv_move);
                moves_done += 1;
            } else {
                break;
            }
            entry = self.tt_probe(-INFINITY, INFINITY, MAX_DEPTH);
        }

        for _ in 0..moves_done {
            self.unmake_move();
        }

        moves_done
    }

    // pub fn eval_terms(&mut self) -> EvalTerms {
    //     let material = self.material[WHITE as usize] - self.material[BLACK as usize];
    //     let pawns = self.piece_num[WP as usize] as usize + self.piece_num[BP as usize] as usize;
    //     let knights = self.piece_num[WN as usize] as usize + self.piece_num[BN as usize] as usize;
    //     let bishops = self.piece_num[WB as usize] as usize + self.piece_num[BB as usize] as usize;
    //     let rooks = self.piece_num[WR as usize] as usize + self.piece_num[BR as usize] as usize;
    //     let queens = self.piece_num[WQ as usize] as usize + self.piece_num[BQ as usize] as usize;
    //     let phase = crate::evaluation::game_phase(pawns, knights, bishops, rooks, queens);
    //     let mut mid_pst_counter = [[0.0; 64]; 13];
    //     let mut end_pst_counter = [[0.0; 64]; 13];
    //     for piece in (WP as usize)..=(WK as usize) {
    //         let pnum = self.piece_num[piece] as usize;
    //         for &sq in self.piece_list[piece][..pnum].iter() {
    //             mid_pst_counter[piece][sq as usize] += 1.0 - phase;
    //             end_pst_counter[piece][sq as usize] += phase;
    //         }
    //     }
    //     for piece in (BP as usize)..=(BK as usize) {
    //         let pnum = self.piece_num[piece] as usize;
    //         for &sq in self.piece_list[piece][..pnum].iter() {
    //             mid_pst_counter[piece][sq as usize] += 1.0 - phase;
    //             end_pst_counter[piece][sq as usize] += phase;
    //         }
    //     }
    //     let mut doubled_pawns = 0;
    //     let mut isolated_pawns = 0;
    //     let mut passed_pawns = 0;
    //     // file counters are padded with zeros to simplify the code.
    //     let mut w_file_counters = [0; 10];
    //     for &wp_loc in self.piece_list[WP as usize][..self.piece_num[WP as usize] as usize].iter() {
    //         let file = FILES_BOARD[wp_loc as usize] as usize;
    //         unsafe { *w_file_counters.get_unchecked_mut(file + 1) += 1 };
    //     }
    //     let mut b_file_counters = [0; 10];
    //     for &bp_loc in self.piece_list[BP as usize][..self.piece_num[BP as usize] as usize].iter() {
    //         let file = FILES_BOARD[bp_loc as usize] as usize;
    //         unsafe { *b_file_counters.get_unchecked_mut(file + 1) += 1 };
    //     }

    //     for index in 1..9 {
    //         let w_file_count = unsafe { *w_file_counters.get_unchecked(index) };
    //         let w_left = unsafe { *w_file_counters.get_unchecked(index - 1) };
    //         let w_right = unsafe { *w_file_counters.get_unchecked(index + 1) };
    //         let b_file_count = unsafe { *b_file_counters.get_unchecked(index) };
    //         let b_left = unsafe { *b_file_counters.get_unchecked(index - 1) };
    //         let b_right = unsafe { *b_file_counters.get_unchecked(index + 1) };
    //         if w_file_count > 0 && b_left == 0 && b_right == 0 && b_file_count == 0 {
    //             passed_pawns += 1;
    //         } else if b_file_count > 0 && w_left == 0 && w_right == 0 && w_file_count == 0 {
    //             passed_pawns -= 1;
    //         }
    //         if w_file_count > 0 && w_left == 0 && w_right == 0 {
    //             isolated_pawns += 1 * w_file_count as i32;
    //         }
    //         if b_file_count > 0 && b_left == 0 && b_right == 0 {
    //             isolated_pawns += 1 * b_file_count as i32;
    //         }
    //         if w_file_count >= 2 {
    //             doubled_pawns += 1 * (w_file_count - 1) as i32;
    //         }
    //         if b_file_count >= 2 {
    //             doubled_pawns -= 1 * (b_file_count - 1) as i32;
    //         }
    //     }

    //     EvalTerms { phase, material, mobility, kingsafety, bishop_pair: (), passed_pawns: (), isolated_pawns: (), doubled_pawns: (), counters: () }
    // }

    /// Computes a score for the position, from the point of view of the side to move.
    /// This function should strive to be as cheap to call as possible, relying on
    /// incremental updates in make-unmake to avoid recomputation.
    pub fn evaluate(&mut self) -> i32 {
        #![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

        if self.pawns[BOTH as usize] == 0 && self.is_material_draw() {
            return if self.side == WHITE {
                DRAW_SCORE
            } else {
                -DRAW_SCORE
            };
        }

        let game_phase = self.phase();
        let midgame_material = self.mg_material[WHITE as usize] - self.mg_material[BLACK as usize];
        let endgame_material = self.eg_material[WHITE as usize] - self.eg_material[BLACK as usize];
        let material = (midgame_material as f32 * (1.0 - game_phase)) as i32
            + (endgame_material as f32 * game_phase) as i32;
        let pst_val = self.pst_value(game_phase);

        let mut score = material + pst_val;

        // if the score is already outwith a threshold, we can stop here.
        if score.abs() > LAZY_THRESHOLD_1 {
            // score = self.clamp_score(score);
            if self.side == WHITE {
                return score;
            }
            return -score;
        }

        let pawn_val = self.pawn_structure_term(); // INCREMENTAL UPDATE.
        let bishop_pair_val = self.bishop_pair_term();
        let mobility_val = self.mobility();
        let king_safety_val = self.pawn_shield_term(game_phase);

        score += pawn_val;
        score += bishop_pair_val;
        score += mobility_val;
        score += king_safety_val;

        // println!(
        //     "material: {} pst: {} pawn: {} bishop pair: {} mobility: {}",
        //     score, pst_val, pawn_val, bishop_pair_val, mobility_val
        // );

        score = self.clamp_score(score);

        if self.side == WHITE {
            score
        } else {
            -score
        }
    }

    /// Determines whether a given position is quiescent (no checks or captures).
    #[allow(clippy::wrong_self_convention)]
    pub fn is_quiet_position(&mut self) -> bool {
        let mut move_list = MoveList::new();
        self.generate_moves(&mut move_list);

        for &m in move_list.iter() {
            if !self.make_move(m) {
                continue;
            }
            if self.in_check::<{ Self::US }>() {
                self.unmake_move();
                return false;
            }
            self.unmake_move();
            if m.is_capture() {
                return false;
            }
        }

        true
    }

    #[inline]
    fn clamp_score(&mut self, score: i32) -> i32 {
        // if we can't win with our material, we clamp the eval to be at best one millipawn.
        if score > 0 && self.unwinnable_for::<{ WHITE }>() {
            1
        } else if score < 0 && self.unwinnable_for::<{ BLACK }>() {
            -1
        } else {
            score
        }
    }

    fn pst_value(&self, phase: f32) -> i32 {
        #![allow(
            clippy::cast_possible_truncation,
            clippy::cast_precision_loss,
            clippy::similar_names
        )]
        let mg_val = self.pst_vals[0] as f32;
        let eg_val = self.pst_vals[1] as f32;
        eg_val.mul_add(phase, (1.0 - phase) * mg_val) as i32
    }

    const fn bishop_pair_term(&self) -> i32 {
        let w_count = self.num(WB);
        let b_count = self.num(BB);
        if w_count == b_count {
            return 0;
        }
        if w_count >= 2 {
            return BISHOP_PAIR_BONUS;
        }
        if b_count >= 2 {
            return -BISHOP_PAIR_BONUS;
        }
        0
    }

    fn pawn_shield_term(&self, phase: f32) -> i32 {
        #![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

        let white_kingloc = self.king_sq[WHITE as usize];
        let black_kingloc = self.king_sq[BLACK as usize];

        let mut white_shield = 0;
        for loc in white_kingloc + 9..=white_kingloc + 11 {
            if self.piece_at(loc) == WP {
                white_shield += 1;
            }
        }

        let mut black_shield = 0;
        for loc in black_kingloc - 11..=black_kingloc - 9 {
            if self.piece_at(loc) == BP {
                black_shield += 1;
            }
        }

        (((SHIELD_BONUS[white_shield] - SHIELD_BONUS[black_shield]) as f32) * (1.0 - phase)) as i32
    }

    fn king_tropism_term(&self) -> i32 {
        #![allow(clippy::similar_names)]
        let white_king_square = self.king_sq[WHITE as usize] as usize;
        let (wkr, wkf) = (
            RANKS_BOARD[white_king_square],
            FILES_BOARD[white_king_square],
        );
        let black_king_square = self.king_sq[BLACK as usize] as usize;
        let (bkr, bkf) = (
            RANKS_BOARD[black_king_square],
            FILES_BOARD[black_king_square],
        );
        let mut score = 0;
        for piece_type in BP..=BQ {
            let piece_type = piece_type as usize;
            let danger = PIECE_DANGER_VALUES[piece_type];
            for &sq in self.piece_lists[piece_type].iter() {
                let rank = RANKS_BOARD[sq as usize];
                let file = FILES_BOARD[sq as usize];
                let dist = i32::from(wkr.abs_diff(rank) + wkf.abs_diff(file));
                score -= danger / dist;
            }
        }
        for piece_type in WP..=WQ {
            let piece_type = piece_type as usize;
            let danger = PIECE_DANGER_VALUES[piece_type];
            for &sq in self.piece_lists[piece_type].iter() {
                let rank = RANKS_BOARD[sq as usize];
                let file = FILES_BOARD[sq as usize];
                let dist = i32::from(bkr.abs_diff(rank) + bkf.abs_diff(file));
                score += danger / dist;
            }
        }

        score
    }

    fn pawn_structure_term(&self) -> i32 {
        let mut w_score = 0;
        for &white_pawn_loc in self.piece_lists[WP as usize].iter() {
            let sq64 = SQ120_TO_SQ64[white_pawn_loc as usize] as usize;
            if ISOLATED_BB[sq64] & self.pawns[WHITE as usize] == 0 {
                w_score -= ISOLATED_PAWN_MALUS;
            }

            if WHITE_PASSED_BB[sq64] & self.pawns[BLACK as usize] == 0 {
                let rank = RANKS_BOARD[white_pawn_loc as usize] as usize;
                w_score += PASSED_PAWN_BONUS[rank];
            }

            let file = FILES_BOARD[white_pawn_loc as usize] as usize;
            if (FILE_BB[file] & self.pawns[WHITE as usize]).count_ones() > 1 {
                w_score -= DOUBLED_PAWN_MALUS / 2; // we double-count the malus, so halve it.
            }
        }

        let mut b_score = 0;
        for &black_pawn_loc in self.piece_lists[BP as usize].iter() {
            let sq64 = SQ120_TO_SQ64[black_pawn_loc as usize] as usize;
            if ISOLATED_BB[sq64] & self.pawns[BLACK as usize] == 0 {
                b_score -= ISOLATED_PAWN_MALUS;
            }

            if BLACK_PASSED_BB[sq64] & self.pawns[WHITE as usize] == 0 {
                let rank = RANKS_BOARD[black_pawn_loc as usize] as usize;
                b_score += PASSED_PAWN_BONUS[7 - rank];
            }

            let file = FILES_BOARD[black_pawn_loc as usize] as usize;
            if (FILE_BB[file] & self.pawns[BLACK as usize]).count_ones() > 1 {
                b_score -= DOUBLED_PAWN_MALUS / 2; // we double-count the malus, so halve it.
            }
        }

        w_score - b_score
    }

    /// `phase` computes a number between 0.0 and 1.0, which is the phase of the game. 0.0 is the opening, 1.0 is the endgame.
    fn phase(&self) -> f32 {
        let pawns = self.num(WP) + self.num(BP);
        let knights = self.num(WN) + self.num(BN);
        let bishops = self.num(WB) + self.num(BB);
        let rooks = self.num(WR) + self.num(BR);
        let queens = self.num(WQ) + self.num(BQ);
        crate::evaluation::game_phase(pawns, knights, bishops, rooks, queens)
    }

    fn generate_pst_value(&self) -> (i32, i32) {
        #![allow(clippy::needless_range_loop, clippy::similar_names)]
        let mut mg_pst_val = 0;
        let mut eg_pst_val = 0;
        for piece in (WP as usize)..=(WK as usize) {
            for &sq in self.piece_lists[piece].iter() {
                let mg = MIDGAME_PST[piece][sq as usize];
                let eg = ENDGAME_PST[piece][sq as usize];
                // println!("adding {} for {} at {}", mg, PIECE_NAMES[piece], SQUARE_NAMES[SQ120_TO_SQ64[sq as usize] as usize]);
                mg_pst_val += mg;
                eg_pst_val += eg;
            }
        }

        for piece in (BP as usize)..=(BK as usize) {
            for &sq in self.piece_lists[piece].iter() {
                let mg = MIDGAME_PST[piece][sq as usize];
                let eg = ENDGAME_PST[piece][sq as usize];
                // println!("adding {} for {} at {}", mg, PIECE_NAMES[piece], SQUARE_NAMES[SQ120_TO_SQ64[sq as usize] as usize]);
                mg_pst_val += mg;
                eg_pst_val += eg;
            }
        }
        (mg_pst_val, eg_pst_val)
    }

    fn mobility(&mut self) -> i32 {
        #![allow(clippy::cast_possible_truncation)]
        let is_check = self.in_check::<{ Self::US }>();
        if is_check {
            match self.side {
                WHITE => return -50,
                BLACK => return 50,
                _ => unsafe { std::hint::unreachable_unchecked() },
            }
        }

        let mut list = MoveCounter::new(self);
        self.generate_moves(&mut list);

        let our_pseudo_legal_moves = list.score();

        self.make_nullmove();
        let mut list = MoveCounter::new(self);
        self.generate_moves(&mut list);
        let their_pseudo_legal_moves = list.score();
        self.unmake_nullmove();

        if self.side == WHITE {
            our_pseudo_legal_moves as i32 - their_pseudo_legal_moves as i32
        } else {
            their_pseudo_legal_moves as i32 - our_pseudo_legal_moves as i32
        }
    }

    #[inline]
    pub const fn num(&self, piece: u8) -> u8 {
        self.piece_lists[piece as usize].len()
    }

    const fn unwinnable_for<const SIDE: u8>(&self) -> bool {
        assert!(
            SIDE == WHITE || SIDE == BLACK,
            "unwinnable_for called with invalid side"
        );

        if SIDE == WHITE {
            if self.major_piece_counts[WHITE as usize] != 0 {
                return false;
            }
            if self.minor_piece_counts[WHITE as usize] > 1 {
                return false;
            }
            if self.num(WP) != 0 {
                return false;
            }
            true
        } else {
            if self.major_piece_counts[BLACK as usize] != 0 {
                return false;
            }
            if self.minor_piece_counts[BLACK as usize] > 1 {
                return false;
            }
            if self.num(BP) != 0 {
                return false;
            }
            true
        }
    }

    const fn is_material_draw(&self) -> bool {
        if self.num(WR) == 0 && self.num(BR) == 0 && self.num(WQ) == 0 && self.num(BQ) == 0 {
            if self.num(WB) == 0 && self.num(BB) == 0 {
                if self.num(WN) < 3 && self.num(BN) < 3 {
                    return true;
                }
            } else if (self.num(WN) == 0
                && self.num(BN) == 0
                && self.num(WB).abs_diff(self.num(BB)) < 2)
                || (self.num(WB) + self.num(WN) == 1 && self.num(BB) + self.num(BN) == 1)
            {
                return true;
            }
        } else if self.num(WQ) == 0 && self.num(BQ) == 0 {
            if self.num(WR) == 1 && self.num(BR) == 1 {
                if (self.num(WN) + self.num(WB)) < 2 && (self.num(BN) + self.num(BB)) < 2 {
                    return true;
                }
            } else if self.num(WR) == 1 && self.num(BR) == 0 {
                if (self.num(WN) + self.num(WB)) == 0
                    && ((self.num(BN) + self.num(BB)) == 1 || (self.num(BN) + self.num(BB)) == 2)
                {
                    return true;
                }
            } else if self.num(WR) == 0
                && self.num(BR) == 1
                && (self.num(BN) + self.num(BB)) == 0
                && ((self.num(WN) + self.num(WB)) == 1 || (self.num(WN) + self.num(WB)) == 2)
            {
                return true;
            }
        }
        false
    }

    fn clear_for_search(&mut self) {
        self.history_table.iter_mut().for_each(|h| h.fill(0));
        self.killer_move_table.fill([Move::null(); 2]);
        self.ply = 0;
        self.tt.clear_for_search();
    }

    pub fn search_position(&mut self, info: &mut SearchInfo) {
        self.clear_for_search();
        info.clear_for_search();

        let first_legal = self.get_first_legal_move().unwrap_or(Move::null());

        let mut most_recent_move = first_legal;
        let mut most_recent_score = 0;
        let mut pv_length = 0;
        let mut best_depth = 1;
        for depth in 0..=info.depth {
            let score = alpha_beta(self, info, depth, -INFINITY, INFINITY);

            info.check_up();

            if info.stopped {
                break;
            }

            most_recent_score = score;
            best_depth = depth;
            pv_length = self.get_pv_line(depth);
            most_recent_move = *self.principal_variation.get(0).unwrap_or(&first_legal);

            let score_string = format_score(most_recent_score);

            print!(
                "info score {} depth {} nodes {} time {} pv ",
                score_string,
                depth,
                info.nodes,
                info.start_time.elapsed().as_millis()
            );
            for &m in &self.principal_variation[..pv_length] {
                print!("{m} ");
            }
            println!();
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        }
        let score_string = format_score(most_recent_score);
        print!(
            "info score {} depth {} nodes {} time {} pv ",
            score_string,
            best_depth,
            info.nodes,
            info.start_time.elapsed().as_millis()
        );
        for &m in &self.principal_variation[..pv_length] {
            print!("{m} ");
        }
        println!();
        println!("bestmove {}", most_recent_move);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
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
        Self::from_fen(Self::STARTING_FEN).expect("for some reason, STARTING_FEN is now broken.")
    }
}

impl Display for Board {
    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
        static PIECE_CHAR: [u8; 13] = *b".PNBRQKpnbrqk";
        static SIDE_CHAR: [u8; 3] = *b"wb-";
        static RANK_CHAR: [u8; 8] = *b"12345678";
        static FILE_CHAR: [u8; 8] = *b"abcdefgh";

        writeln!(f, "Game Board:")?;

        for rank in ((Rank::Rank1 as u8)..=(Rank::Rank8 as u8)).rev() {
            write!(f, "{} ", rank + 1)?;
            for file in (File::FileA as u8)..=(File::FileH as u8) {
                let sq = filerank_to_square(file, rank);
                let piece = self.piece_at(sq);
                write!(f, "{} ", PIECE_CHAR[piece as usize] as char)?;
            }
            writeln!(f)?;
        }

        writeln!(f, "  a b c d e f g h")?;
        writeln!(f, "side: {}", SIDE_CHAR[self.side as usize] as char)?;

        Ok(())
    }
}

impl Debug for Board {
    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "{}", self)?;
        writeln!(f, "ep-square: {}", self.ep_sq)?;
        writeln!(f, "castling: {:b}", self.castle_perm)?;
        writeln!(f, "fifty-move-counter: {}", self.fifty_move_counter)?;
        writeln!(f, "ply: {}", self.ply)?;
        writeln!(f, "hash: {:x}", self.key)?;
        // write_bb(self.pawns[Colour::White as usize], f)?;
        // writeln!(f)?;
        // write_bb(self.pawns[Colour::Black as usize], f)?;
        // writeln!(f)?;
        // write_bb(self.pawns[Colour::Both as usize], f)?;
        Ok(())
    }
}

mod tests {
    #[test]
    fn read_fen_validity() {
        use super::*;
        let mut board_1 = Board::new();
        board_1
            .set_from_fen(Board::STARTING_FEN)
            .expect("setfen failed.");
        board_1.check_validity().unwrap();

        let board_2 = Board::from_fen(Board::STARTING_FEN).expect("setfen failed.");
        board_2.check_validity().unwrap();

        assert_eq!(board_1, board_2);
    }
}
