use crate::lookups::{filerank_to_square, SQ120_TO_SQ64};



pub const BOARD_N_SQUARES: usize = 120;
pub const MAX_GAME_MOVES: usize = 2048;
pub const OFF_BOARD: u8 = 120;
pub const NO_SQUARE: u8 = 121;

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Piece {
    Empty = 0,
    WP = 0b0001, WN = 0b0010, WB = 0b0011, WR = 0b0100, WQ = 0b0101, WK = 0b0110,
    BP = 0b1001, BN = 0b1010, BB = 0b1011, BR = 0b1100, BQ = 0b1101, BK = 0b1110,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum File {
    FileA = 0, FileB, FileC, FileD, FileE, FileF, FileG, FileH, None
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Rank {
    Rank1 = 0, Rank2, Rank3, Rank4, Rank5, Rank6, Rank7, Rank8, None
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Colour {
    White = 0, Black, Both
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Square120 {
    A1 = 21, B1, C1, D1, E1, F1, G1, H1,
    A2 = 31, B2, C2, D2, E2, F2, G2, H2,
    A3 = 41, B3, C3, D3, E3, F3, G3, H3,
    A4 = 51, B4, C4, D4, E4, F4, G4, H4,
    A5 = 61, B5, C5, D5, E5, F5, G5, H5,
    A6 = 71, B6, C6, D6, E6, F6, G6, H6,
    A7 = 81, B7, C7, D7, E7, F7, G7, H7,
    A8 = 91, B8, C8, D8, E8, F8, G8, H8,
    None
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Square64 {
    A1 = 0, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum Castling {
    WK = 0b0001, WQ = 0b0010, BK = 0b0100, BQ = 0b1000,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Undo {
    pub m: u16,
    pub castle_perm: u8,
    pub ep_square: u8,
    pub fifty_move_counter: u8,
    pub position_key: u64,
}

#[inline]
pub fn pop_lsb(bb: &mut u64) -> u32 {
    let lsb = bb.trailing_zeros();
    *bb &= *bb - 1;
    lsb
}

pub fn print_bb(bb: u64) {
    for rank in (0..=7).rev() {
        for file in 0..=7 {
            let sq = filerank_to_square(file, rank);
            let sq64 = SQ120_TO_SQ64[sq as usize];
            assert!(sq64 < 64, "sq64: {}, sq: {}, file: {}, rank: {}", sq64, sq, file, rank);
            if ((1 << sq64) & bb) == 0 {
                print!(".");
            } else {
                print!("X");
            }
        }
        println!();
    }
}