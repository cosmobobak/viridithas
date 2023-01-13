use std::fmt::Display;

use crate::{
    definitions::Square,
    lookups, magic, piece::{PieceType, Piece, Colour},
};

pub const BB_RANK_1: u64 = 0x0000_0000_0000_00FF;
pub const BB_RANK_2: u64 = 0x0000_0000_0000_FF00;
pub const BB_RANK_3: u64 = 0x0000_0000_00FF_0000;
pub const BB_RANK_4: u64 = 0x0000_0000_FF00_0000;
pub const BB_RANK_5: u64 = 0x0000_00FF_0000_0000;
pub const BB_RANK_6: u64 = 0x0000_FF00_0000_0000;
pub const BB_RANK_7: u64 = 0x00FF_0000_0000_0000;
pub const BB_RANK_8: u64 = 0xFF00_0000_0000_0000;
pub const BB_FILE_A: u64 = 0x0101_0101_0101_0101;
pub const BB_FILE_B: u64 = 0x0202_0202_0202_0202;
pub const BB_FILE_C: u64 = 0x0404_0404_0404_0404;
pub const BB_FILE_D: u64 = 0x0808_0808_0808_0808;
pub const BB_FILE_E: u64 = 0x1010_1010_1010_1010;
pub const BB_FILE_F: u64 = 0x2020_2020_2020_2020;
pub const BB_FILE_G: u64 = 0x4040_4040_4040_4040;
pub const BB_FILE_H: u64 = 0x8080_8080_8080_8080;
pub const BB_NONE: u64 = 0x0000_0000_0000_0000;
pub const BB_ALL: u64 = 0xFFFF_FFFF_FFFF_FFFF;
pub const BB_LIGHT_SQUARES: u64 = 0x55AA_55AA_55AA_55AA;
pub const BB_DARK_SQUARES: u64 = 0xAA55_AA55_AA55_AA55;

pub const LIGHT_SQUARE: bool = true;
pub const DARK_SQUARE: bool = false;

pub static BB_RANKS: [u64; 8] =
    [BB_RANK_1, BB_RANK_2, BB_RANK_3, BB_RANK_4, BB_RANK_5, BB_RANK_6, BB_RANK_7, BB_RANK_8];

pub static BB_FILES: [u64; 8] =
    [BB_FILE_A, BB_FILE_B, BB_FILE_C, BB_FILE_D, BB_FILE_E, BB_FILE_F, BB_FILE_G, BB_FILE_H];

/// least significant bit of a u64
/// ```
/// assert_eq!(3, bitboard::lsb(0b00001000));
/// ```
pub const fn lsb(x: u64) -> u64 {
    x.trailing_zeros() as u64
}

/// first set square of a u64
pub const fn first_square(x: u64) -> Square {
    #![allow(clippy::cast_possible_truncation)]
    Square::new(x.trailing_zeros() as u8)
}

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
            self.value ^= 1 << lsb;
            Some(Square::new(lsb))
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BitBoard {
    w_pawns: u64,
    w_knights: u64,
    w_bishops: u64,
    w_rooks: u64,
    w_queens: u64,
    w_king: u64,

    b_pawns: u64,
    b_knights: u64,
    b_bishops: u64,
    b_rooks: u64,
    b_queens: u64,
    b_king: u64,

    white: u64,
    black: u64,
    occupied: u64,
}

impl BitBoard {
    pub const NULL: Self = Self::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        bp: u64,
        bn: u64,
        bb: u64,
        br: u64,
        bq: u64,
        bk: u64,
        wp: u64,
        wn: u64,
        wb: u64,
        wr: u64,
        wq: u64,
        wk: u64,
    ) -> Self {
        let white = wp | wk | wb | wr | wq;
        let black = bp | bk | bb | br | bq;
        let occupied = white | black;
        Self {
            w_pawns: wp,
            w_knights: wn,
            w_bishops: wb,
            w_rooks: wr,
            w_queens: wq,
            w_king: wk,

            b_pawns: bp,
            b_knights: bn,
            b_bishops: bb,
            b_rooks: br,
            b_queens: bq,
            b_king: bk,

            white,
            black,
            occupied,
        }
    }

    #[rustfmt::skip]
    pub const fn king<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE { self.w_king } else { self.b_king }
    }

    #[rustfmt::skip]
    pub const fn pawns<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE { self.w_pawns } else { self.b_pawns }
    }

    pub fn occupied_co(&self, colour: Colour) -> u64 {
        if colour == Colour::WHITE {
            self.white
        } else {
            self.black
        }
    }

    #[rustfmt::skip]
    pub const fn their_pieces<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE { self.black } else { self.white }
    }

    #[rustfmt::skip]
    pub const fn rookqueen<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE { self.w_rooks | self.w_queens } else { self.b_rooks | self.b_queens }
    }

    #[rustfmt::skip]
    pub const fn bishopqueen<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE { self.w_bishops | self.w_queens } else { self.b_bishops | self.b_queens }
    }

    #[rustfmt::skip]
    pub const fn minors<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE { self.w_bishops | self.w_knights } else { self.b_bishops | self.b_knights }
    }

    #[rustfmt::skip]
    pub const fn majors<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE { self.w_rooks | self.w_queens } else { self.b_rooks | self.b_queens }
    }

    pub const fn empty(&self) -> u64 {
        !self.occupied
    }

    pub const fn occupied(&self) -> u64 {
        self.occupied
    }

    pub const fn knights<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE {
            self.w_knights
        } else {
            self.b_knights
        }
    }

    pub const fn rooks<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE {
            self.w_rooks
        } else {
            self.b_rooks
        }
    }

    pub const fn bishops<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE {
            self.w_bishops
        } else {
            self.b_bishops
        }
    }

    pub const fn queens<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE {
            self.w_queens
        } else {
            self.b_queens
        }
    }

    pub const fn bishops_sqco<const IS_WHITE: bool, const IS_LSB: bool>(&self) -> u64 {
        if IS_WHITE {
            if IS_LSB {
                self.w_bishops & BB_LIGHT_SQUARES
            } else {
                self.w_bishops & BB_DARK_SQUARES
            }
        } else if IS_LSB {
            self.b_bishops & BB_LIGHT_SQUARES
        } else {
            self.b_bishops & BB_DARK_SQUARES
        }
    }

    pub fn reset(&mut self) {
        *self = Self::NULL;
    }

    pub fn move_piece(&mut self, from_to_bb: u64, piece: Piece) {
        self.occupied ^= from_to_bb;
        if piece.colour() == Colour::WHITE {
            self.white ^= from_to_bb;
            match piece.piece_type() {
                PieceType::PAWN => self.w_pawns ^= from_to_bb,
                PieceType::KNIGHT => self.w_knights ^= from_to_bb,
                PieceType::BISHOP => self.w_bishops ^= from_to_bb,
                PieceType::ROOK => self.w_rooks ^= from_to_bb,
                PieceType::QUEEN => self.w_queens ^= from_to_bb,
                _ => self.w_king ^= from_to_bb,
            }
        } else {
            self.black ^= from_to_bb;
            match piece.piece_type() {
                PieceType::PAWN => self.b_pawns ^= from_to_bb,
                PieceType::KNIGHT => self.b_knights ^= from_to_bb,
                PieceType::BISHOP => self.b_bishops ^= from_to_bb,
                PieceType::ROOK => self.b_rooks ^= from_to_bb,
                PieceType::QUEEN => self.b_queens ^= from_to_bb,
                _ => self.b_king ^= from_to_bb,
            }
        }
    }

    pub fn set_piece_at(&mut self, sq: Square, piece: Piece) {
        let sq_bb = sq.bitboard();
        self.occupied |= sq_bb;
        if piece.colour() == Colour::WHITE {
            self.white |= sq_bb;
            match piece.piece_type() {
                PieceType::PAWN => self.w_pawns |= sq_bb,
                PieceType::KNIGHT => self.w_knights |= sq_bb,
                PieceType::BISHOP => self.w_bishops |= sq_bb,
                PieceType::ROOK => self.w_rooks |= sq_bb,
                PieceType::QUEEN => self.w_queens |= sq_bb,
                _ => self.w_king |= sq_bb,
            }
        } else {
            self.black |= sq_bb;
            match piece.piece_type() {
                PieceType::PAWN => self.b_pawns |= sq_bb,
                PieceType::KNIGHT => self.b_knights |= sq_bb,
                PieceType::BISHOP => self.b_bishops |= sq_bb,
                PieceType::ROOK => self.b_rooks |= sq_bb,
                PieceType::QUEEN => self.b_queens |= sq_bb,
                _ => self.b_king |= sq_bb,
            }
        }
    }

    pub fn clear_piece_at(&mut self, sq: Square, piece: Piece) {
        let sq_bb = sq.bitboard();
        self.occupied &= !sq_bb;
        if piece.colour() == Colour::WHITE {
            self.white &= !sq_bb;
            match piece.piece_type() {
                PieceType::PAWN => self.w_pawns &= !sq_bb,
                PieceType::KNIGHT => self.w_knights &= !sq_bb,
                PieceType::BISHOP => self.w_bishops &= !sq_bb,
                PieceType::ROOK => self.w_rooks &= !sq_bb,
                PieceType::QUEEN => self.w_queens &= !sq_bb,
                _ => self.w_king &= !sq_bb,
            }
        } else {
            self.black &= !sq_bb;
            match piece.piece_type() {
                PieceType::PAWN => self.b_pawns &= !sq_bb,
                PieceType::KNIGHT => self.b_knights &= !sq_bb,
                PieceType::BISHOP => self.b_bishops &= !sq_bb,
                PieceType::ROOK => self.b_rooks &= !sq_bb,
                PieceType::QUEEN => self.b_queens &= !sq_bb,
                _ => self.b_king &= !sq_bb,
            }
        }
    }

    pub const fn any_pawns(&self) -> bool {
        self.w_pawns | self.b_pawns != 0
    }

    pub const fn piece_bb(&self, piece: Piece) -> u64 {
        match piece {
            Piece::WP => self.w_pawns,
            Piece::WN => self.w_knights,
            Piece::WB => self.w_bishops,
            Piece::WR => self.w_rooks,
            Piece::WQ => self.w_queens,
            Piece::WK => self.w_king,
            Piece::BP => self.b_pawns,
            Piece::BN => self.b_knights,
            Piece::BB => self.b_bishops,
            Piece::BR => self.b_rooks,
            Piece::BQ => self.b_queens,
            _ => self.b_king,
        }
    }

    pub const fn of_type(&self, piece_type: PieceType) -> u64 {
        match piece_type {
            PieceType::PAWN => self.w_pawns | self.b_pawns,
            PieceType::KNIGHT => self.w_knights | self.b_knights,
            PieceType::BISHOP => self.w_bishops | self.b_bishops,
            PieceType::ROOK => self.w_rooks | self.b_rooks,
            PieceType::QUEEN => self.w_queens | self.b_queens,
            _ => self.w_king | self.b_king,
        }
    }

    pub fn pawn_attacks<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE {
            self.w_pawns.north_east_one() | self.w_pawns.north_west_one()
        } else {
            self.b_pawns.south_east_one() | self.b_pawns.south_west_one()
        }
    }

    pub fn all_attackers_to_sq(&self, sq: Square, occupied: u64) -> u64 {
        let sq_bb = sq.bitboard();
        let black_pawn_attackers = pawn_attacks::<true>(sq_bb) & self.b_pawns;
        let white_pawn_attackers = pawn_attacks::<false>(sq_bb) & self.w_pawns;
        let knight_attackers = attacks::<{ PieceType::KNIGHT.inner() }>(sq, BB_NONE) & (self.w_knights | self.b_knights);
        let diag_attackers = attacks::<{ PieceType::BISHOP.inner() }>(sq, occupied)
            & (self.w_bishops | self.b_bishops | self.w_queens | self.b_queens);
        let orth_attackers = attacks::<{ PieceType::ROOK.inner() }>(sq, occupied)
            & (self.w_rooks | self.b_rooks | self.w_queens | self.b_queens);
        let king_attackers = attacks::<{ PieceType::KING.inner() }>(sq, BB_NONE) & (self.w_king | self.b_king);
        black_pawn_attackers
            | white_pawn_attackers
            | knight_attackers
            | diag_attackers
            | orth_attackers
            | king_attackers
    }

    const fn piece_at(&self, sq: Square) -> Piece {
        let sq_bb = sq.bitboard();
        if self.w_pawns & sq_bb != 0 {
            Piece::WP
        } else if self.w_knights & sq_bb != 0 {
            Piece::WN
        } else if self.w_bishops & sq_bb != 0 {
            Piece::WB
        } else if self.w_rooks & sq_bb != 0 {
            Piece::WR
        } else if self.w_queens & sq_bb != 0 {
            Piece::WQ
        } else if self.w_king & sq_bb != 0 {
            Piece::WK
        } else if self.b_pawns & sq_bb != 0 {
            Piece::BP
        } else if self.b_knights & sq_bb != 0 {
            Piece::BN
        } else if self.b_bishops & sq_bb != 0 {
            Piece::BB
        } else if self.b_rooks & sq_bb != 0 {
            Piece::BR
        } else if self.b_queens & sq_bb != 0 {
            Piece::BQ
        } else if self.b_king & sq_bb != 0 {
            Piece::BK
        } else {
            Piece::EMPTY
        }
    }

    fn any_bbs_overlapping(&self) -> bool {
        let bbs = [
            self.w_pawns,
            self.w_knights,
            self.w_bishops,
            self.w_rooks,
            self.w_queens,
            self.w_king,
            self.b_pawns,
            self.b_knights,
            self.b_bishops,
            self.b_rooks,
            self.b_queens,
            self.b_king,
        ];
        for i in 0..bbs.len() {
            for j in i + 1..bbs.len() {
                if bbs[i] & bbs[j] != 0 {
                    return true;
                }
            }
        }
        false
    }
}

pub trait BitShiftExt {
    fn north_east_one(self) -> Self;
    fn north_west_one(self) -> Self;
    fn south_east_one(self) -> Self;
    fn south_west_one(self) -> Self;
    fn east_one(self) -> Self;
    fn west_one(self) -> Self;
    fn north_one(self) -> Self;
    fn south_one(self) -> Self;
    fn first_square(self) -> Square;
}

impl BitShiftExt for u64 {
    fn north_east_one(self) -> Self {
        (self << 9) & !BB_FILE_A
    }
    fn north_west_one(self) -> Self {
        (self << 7) & !BB_FILE_H
    }
    fn south_east_one(self) -> Self {
        (self >> 7) & !BB_FILE_A
    }
    fn south_west_one(self) -> Self {
        (self >> 9) & !BB_FILE_H
    }
    fn east_one(self) -> Self {
        (self >> 1) & !BB_FILE_A
    }
    fn west_one(self) -> Self {
        (self << 1) & !BB_FILE_H
    }
    fn north_one(self) -> Self {
        self << 8
    }
    fn south_one(self) -> Self {
        self >> 8
    }
    fn first_square(self) -> Square {
        #![allow(clippy::cast_possible_truncation)]
        debug_assert!(self != 0, "Tried to get first square of empty bitboard");
        Square::new(self.trailing_zeros() as u8)
    }
}

pub fn attacks<const PIECE_TYPE: u8>(sq: Square, blockers: u64) -> u64 {
    debug_assert!(PieceType::new(PIECE_TYPE) != PieceType::PAWN);
    match PieceType::new(PIECE_TYPE) {
        PieceType::BISHOP => magic::get_bishop_attacks(sq, blockers),
        PieceType::ROOK => magic::get_rook_attacks(sq, blockers),
        PieceType::QUEEN => magic::get_bishop_attacks(sq, blockers) | magic::get_rook_attacks(sq, blockers),
        PieceType::KNIGHT => lookups::get_jumping_piece_attack::<{ PieceType::KNIGHT.inner() }>(sq),
        PieceType::KING => lookups::get_jumping_piece_attack::<{ PieceType::KING.inner() }>(sq),
        _ => panic!("Invalid piece type"),
    }
}

pub fn attacks_by_type(pt: PieceType, sq: Square, blockers: u64) -> u64 {
    match pt {
        PieceType::BISHOP => magic::get_bishop_attacks(sq, blockers),
        PieceType::ROOK => magic::get_rook_attacks(sq, blockers),
        PieceType::QUEEN => magic::get_bishop_attacks(sq, blockers) | magic::get_rook_attacks(sq, blockers),
        PieceType::KNIGHT => lookups::get_jumping_piece_attack::<{ PieceType::KNIGHT.inner() }>(sq),
        PieceType::KING => lookups::get_jumping_piece_attack::<{ PieceType::KING.inner() }>(sq),
        _ => panic!("Invalid piece type: {pt:?}"),
    }
}

pub fn pawn_attacks<const IS_WHITE: bool>(bb: u64) -> u64 {
    if IS_WHITE {
        bb.north_east_one() | bb.north_west_one()
    } else {
        bb.south_east_one() | bb.south_west_one()
    }
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
