#![allow(dead_code)]

use crate::{
    definitions::{
        colour_of, type_of, BB, BISHOP, BK, BLACK, BN, BP, BQ, BR, KING, KNIGHT, PAWN, QUEEN, ROOK,
        WB, WHITE, WK, WN, WP, WQ, WR,
    },
    lookups, macros, magic,
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

/// least significant bit of a u64
/// ```
/// assert_eq!(3, bitboard::lsb(0b00001000));
/// ```
pub const fn lsb(x: u64) -> u64 {
    x.trailing_zeros() as u64
}

/// Iterator over the squares of a bitboard.
/// The squares are returned in increasing order.
/// ```
/// let bb = 0b010110;
/// let squares = BitLoop::new(bb).collect::<Vec<_>>();
/// assert_eq!(squares, vec![1, 2, 4]);
/// ```
pub struct BitLoop {
    value: u64,
}

impl BitLoop {
    pub const fn new(value: u64) -> Self {
        Self { value }
    }
}

impl Iterator for BitLoop {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        if self.value == 0 {
            None
        } else {
            // faster if we have bmi (maybe)
            let lsb: u8 = unsafe { self.value.trailing_zeros().try_into().unwrap_unchecked() };
            self.value ^= 1 << lsb;
            Some(lsb)
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
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
    pub const fn enemy_king<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE { self.b_king } else { self.w_king }
    }

    #[rustfmt::skip]
    pub const fn pawns<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE { self.w_pawns } else { self.b_pawns }
    }

    #[rustfmt::skip]
    pub const fn our_pieces<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE { self.white } else { self.black }
    }

    #[rustfmt::skip]
    pub const fn their_pieces<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE { self.black } else { self.white }
    }

    #[rustfmt::skip]
    pub const fn enemy_rookqueen<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE { self.b_rooks | self.b_queens } else { self.w_rooks | self.w_queens }
    }

    #[rustfmt::skip]
    pub const fn rookqueen<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE { self.w_rooks | self.w_queens } else { self.b_rooks | self.b_queens }
    }

    #[rustfmt::skip]
    pub const fn enemy_bishopqueen<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE { self.b_bishops | self.b_queens } else { self.w_bishops | self.w_queens }
    }

    #[rustfmt::skip]
    pub const fn bishopqueen<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE { self.w_bishops | self.w_queens } else { self.b_bishops | self.b_queens }
    }

    #[rustfmt::skip]
    pub const fn enemy_or_empty<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE { !self.white } else { !self.black }
    }

    pub const fn empty(&self) -> u64 {
        !self.occupied
    }

    pub const fn occupied(&self) -> u64 {
        self.occupied
    }

    #[rustfmt::skip]
    pub const fn knights<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE { self.w_knights } else { self.b_knights }
    }

    #[rustfmt::skip]
    pub const fn rooks<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE { self.w_rooks } else { self.b_rooks }
    }

    #[rustfmt::skip]
    pub const fn bishops<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE { self.w_bishops } else { self.b_bishops }
    }

    #[rustfmt::skip]
    pub const fn queens<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE { self.w_queens } else { self.b_queens }
    }

    pub fn reset(&mut self) {
        *self = Self::NULL;
    }

    pub fn move_piece_comptime<const PIECE: u8>(&mut self, from_to_bb: u64) {
        match PIECE {
            WP => self.w_pawns ^= from_to_bb,
            WN => self.w_knights ^= from_to_bb,
            WB => self.w_bishops ^= from_to_bb,
            WR => self.w_rooks ^= from_to_bb,
            WQ => self.w_queens ^= from_to_bb,
            WK => self.w_king ^= from_to_bb,
            BP => self.b_pawns ^= from_to_bb,
            BN => self.b_knights ^= from_to_bb,
            BB => self.b_bishops ^= from_to_bb,
            BR => self.b_rooks ^= from_to_bb,
            BQ => self.b_queens ^= from_to_bb,
            BK => self.b_king ^= from_to_bb,
            _ => unsafe { macros::impossible!() },
        }
        match colour_of(PIECE) {
            WHITE => self.white ^= from_to_bb,
            BLACK => self.black ^= from_to_bb,
            _ => unsafe { macros::impossible!() },
        }
        self.occupied ^= from_to_bb;
    }

    pub fn set_piece_comptime<const PIECE: u8>(&mut self, sq: u8) {
        match PIECE {
            WP => self.w_pawns |= 1u64 << sq,
            WN => self.w_knights |= 1u64 << sq,
            WB => self.w_bishops |= 1u64 << sq,
            WR => self.w_rooks |= 1u64 << sq,
            WQ => self.w_queens |= 1u64 << sq,
            WK => self.w_king |= 1u64 << sq,
            BP => self.b_pawns |= 1u64 << sq,
            BN => self.b_knights |= 1u64 << sq,
            BB => self.b_bishops |= 1u64 << sq,
            BR => self.b_rooks |= 1u64 << sq,
            BQ => self.b_queens |= 1u64 << sq,
            BK => self.b_king |= 1u64 << sq,
            _ => unsafe { macros::impossible!() },
        }
        match colour_of(PIECE) {
            WHITE => self.white |= 1u64 << sq,
            BLACK => self.black |= 1u64 << sq,
            _ => unsafe { macros::impossible!() },
        }
        self.occupied |= 1u64 << sq;
    }

    pub fn clear_piece_comptime<const PIECE: u8>(&mut self, sq: u8) {
        match PIECE {
            WP => self.w_pawns &= !(1u64 << sq),
            WN => self.w_knights &= !(1u64 << sq),
            WB => self.w_bishops &= !(1u64 << sq),
            WR => self.w_rooks &= !(1u64 << sq),
            WQ => self.w_queens &= !(1u64 << sq),
            WK => self.w_king &= !(1u64 << sq),
            BP => self.b_pawns &= !(1u64 << sq),
            BN => self.b_knights &= !(1u64 << sq),
            BB => self.b_bishops &= !(1u64 << sq),
            BR => self.b_rooks &= !(1u64 << sq),
            BQ => self.b_queens &= !(1u64 << sq),
            BK => self.b_king &= !(1u64 << sq),
            _ => unsafe { macros::impossible!() },
        }
        match colour_of(PIECE) {
            WHITE => self.white &= !(1u64 << sq),
            BLACK => self.black &= !(1u64 << sq),
            _ => unsafe { macros::impossible!() },
        }
        self.occupied &= !(1u64 << sq);
    }

    pub fn move_piece(&mut self, from_to_bb: u64, piece: u8) {
        self.occupied ^= from_to_bb;
        if colour_of(piece) == WHITE {
            self.white ^= from_to_bb;
            match type_of(piece) {
                PAWN => self.w_pawns ^= from_to_bb,
                KNIGHT => self.w_knights ^= from_to_bb,
                BISHOP => self.w_bishops ^= from_to_bb,
                ROOK => self.w_rooks ^= from_to_bb,
                QUEEN => self.w_queens ^= from_to_bb,
                KING => self.w_king ^= from_to_bb,
                _ => unsafe { macros::impossible!() },
            }
        } else {
            self.black ^= from_to_bb;
            match type_of(piece) {
                PAWN => self.b_pawns ^= from_to_bb,
                KNIGHT => self.b_knights ^= from_to_bb,
                BISHOP => self.b_bishops ^= from_to_bb,
                ROOK => self.b_rooks ^= from_to_bb,
                QUEEN => self.b_queens ^= from_to_bb,
                KING => self.b_king ^= from_to_bb,
                _ => unsafe { macros::impossible!() },
            }
        }
    }

    pub fn set_piece_at(&mut self, sq: u8, piece: u8) {
        self.occupied |= 1u64 << sq;
        if colour_of(piece) == WHITE {
            self.white |= 1u64 << sq;
            match type_of(piece) {
                PAWN => self.w_pawns |= 1u64 << sq,
                KNIGHT => self.w_knights |= 1u64 << sq,
                BISHOP => self.w_bishops |= 1u64 << sq,
                ROOK => self.w_rooks |= 1u64 << sq,
                QUEEN => self.w_queens |= 1u64 << sq,
                KING => self.w_king |= 1u64 << sq,
                _ => unsafe { macros::impossible!() },
            }
        } else {
            self.black |= 1u64 << sq;
            match type_of(piece) {
                PAWN => self.b_pawns |= 1u64 << sq,
                KNIGHT => self.b_knights |= 1u64 << sq,
                BISHOP => self.b_bishops |= 1u64 << sq,
                ROOK => self.b_rooks |= 1u64 << sq,
                QUEEN => self.b_queens |= 1u64 << sq,
                KING => self.b_king |= 1u64 << sq,
                _ => unsafe { macros::impossible!() },
            }
        }
    }

    pub fn clear_piece_at(&mut self, sq: u8, piece: u8) {
        self.occupied &= !(1u64 << sq);
        if colour_of(piece) == WHITE {
            self.white &= !(1u64 << sq);
            match type_of(piece) {
                PAWN => self.w_pawns &= !(1u64 << sq),
                KNIGHT => self.w_knights &= !(1u64 << sq),
                BISHOP => self.w_bishops &= !(1u64 << sq),
                ROOK => self.w_rooks &= !(1u64 << sq),
                QUEEN => self.w_queens &= !(1u64 << sq),
                KING => self.w_king &= !(1u64 << sq),
                _ => unsafe { macros::impossible!() },
            }
        } else {
            self.black &= !(1u64 << sq);
            match type_of(piece) {
                PAWN => self.b_pawns &= !(1u64 << sq),
                KNIGHT => self.b_knights &= !(1u64 << sq),
                BISHOP => self.b_bishops &= !(1u64 << sq),
                ROOK => self.b_rooks &= !(1u64 << sq),
                QUEEN => self.b_queens &= !(1u64 << sq),
                KING => self.b_king &= !(1u64 << sq),
                _ => unsafe { macros::impossible!() },
            }
        }
    }

    pub const fn any_pawns(&self) -> bool {
        self.w_pawns | self.b_pawns != 0
    }

    pub const fn piece_bb_comptime<const PIECE: u8>(&self) -> u64 {
        match PIECE {
            WP => self.w_pawns,
            WN => self.w_knights,
            WB => self.w_bishops,
            WR => self.w_rooks,
            WQ => self.w_queens,
            WK => self.w_king,
            BP => self.b_pawns,
            BN => self.b_knights,
            BB => self.b_bishops,
            BR => self.b_rooks,
            BQ => self.b_queens,
            BK => self.b_king,
            _ => unsafe { macros::impossible!() },
        }
    }

    pub const fn piece_bb(&self, piece: u8) -> u64 {
        match piece {
            WP => self.w_pawns,
            WN => self.w_knights,
            WB => self.w_bishops,
            WR => self.w_rooks,
            WQ => self.w_queens,
            WK => self.w_king,
            BP => self.b_pawns,
            BN => self.b_knights,
            BB => self.b_bishops,
            BR => self.b_rooks,
            BQ => self.b_queens,
            BK => self.b_king,
            _ => unsafe { macros::impossible!() },
        }
    }

    pub const fn pawn_attacks<const IS_WHITE: bool>(&self) -> u64 {
        if IS_WHITE {
            north_east_one(self.w_pawns) | north_west_one(self.w_pawns)
        } else {
            south_east_one(self.b_pawns) | south_west_one(self.b_pawns)
        }
    }
}

pub const fn north_east_one(b: u64) -> u64 {
    (b << 9) & !BB_FILE_A
}
pub const fn south_east_one(b: u64) -> u64 {
    (b >> 7) & !BB_FILE_A
}
pub const fn south_west_one(b: u64) -> u64 {
    (b >> 9) & !BB_FILE_H
}
pub const fn north_west_one(b: u64) -> u64 {
    (b << 7) & !BB_FILE_H
}

pub fn attacks<const PIECE_TYPE: u8>(sq: u8, blockers: u64) -> u64 {
    debug_assert!(PIECE_TYPE != PAWN);
    match PIECE_TYPE {
        BISHOP => magic::get_bishop_attacks(sq, blockers),
        ROOK => magic::get_rook_attacks(sq, blockers),
        QUEEN => magic::get_bishop_attacks(sq, blockers) | magic::get_rook_attacks(sq, blockers),
        KNIGHT => lookups::get_jumping_piece_attack::<KNIGHT>(sq),
        KING => lookups::get_jumping_piece_attack::<KING>(sq),
        _ => unsafe { macros::impossible!() },
    }
}

pub fn print_bb(bb: u64) {
    for rank in (0..=7).rev() {
        for file in 0..=7 {
            let sq = crate::lookups::filerank_to_square(file, rank);
            assert!(
                sq < 64,
                "sq64: {}, sq: {}, file: {}, rank: {}",
                sq,
                sq,
                file,
                rank
            );
            if bb & (1 << sq) != 0 {
                print!(" X");
            } else {
                print!(" .");
            }
        }
        println!();
    }
}
