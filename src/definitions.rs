use crate::lookups::{filerank_to_square, SQ120_TO_SQ64, SQUARE_NAMES};

pub const BOARD_N_SQUARES: usize = 120;
pub const MAX_GAME_MOVES: usize = 1024;

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Piece {
    Empty = 0,
    WP, WN, WB, WR, WQ, WK,
    BP, BN, BB, BR, BQ, BK,
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

pub const WHITE: u8 = Colour::White as u8;
pub const BLACK: u8 = Colour::Black as u8;

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
    NoSquare, OffBoard
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
pub enum Castling {
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

pub fn square120_name(sq: u8) -> Option<&'static str> {
    let sq64 = SQ120_TO_SQ64[sq as usize];
    square64_name(sq64)
}

pub fn square64_name(sq: u8) -> Option<&'static str> {
    SQUARE_NAMES.get(sq as usize).copied()
}