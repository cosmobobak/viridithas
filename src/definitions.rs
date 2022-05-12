use crate::{
    chessmove::Move,
    evaluation::{MATE_SCORE, MG_PAWN_VALUE},
    lookups::{SQ120_TO_SQ64, SQUARE_NAMES},
};

pub const BOARD_N_SQUARES: usize = 120;
pub const MAX_GAME_MOVES: usize = 1024;
pub const MAX_DEPTH: usize = 512;
pub const INFINITY: i32 = MATE_SCORE * 2;

pub const SECOND_ORDER_KILLER_SCORE: i32 = 8_000_000;
pub const FIRST_ORDER_KILLER_SCORE: i32 = 9_000_000;

pub const FUTILITY_MARGIN: i32 = 2 * MG_PAWN_VALUE;

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
#[rustfmt::skip]
enum Piece {
    Empty = 0,
    WP, WN, WB, WR, WQ, WK,
    BP, BN, BB, BR, BQ, BK,
}

pub const PIECE_EMPTY: u8 = Piece::Empty as u8;
pub const WP: u8 = Piece::WP as u8;
pub const WN: u8 = Piece::WN as u8;
pub const WB: u8 = Piece::WB as u8;
pub const WR: u8 = Piece::WR as u8;
pub const WQ: u8 = Piece::WQ as u8;
pub const WK: u8 = Piece::WK as u8;
pub const BP: u8 = Piece::BP as u8;
pub const BN: u8 = Piece::BN as u8;
pub const BB: u8 = Piece::BB as u8;
pub const BR: u8 = Piece::BR as u8;
pub const BQ: u8 = Piece::BQ as u8;
pub const BK: u8 = Piece::BK as u8;

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum File {
    FileA = 0,
    FileB,
    FileC,
    FileD,
    FileE,
    FileF,
    FileG,
    FileH,
    None,
}

pub const FILE_A: u8 = File::FileA as u8;
pub const FILE_B: u8 = File::FileB as u8;
pub const FILE_C: u8 = File::FileC as u8;
pub const FILE_D: u8 = File::FileD as u8;
pub const FILE_E: u8 = File::FileE as u8;
pub const FILE_F: u8 = File::FileF as u8;
pub const FILE_G: u8 = File::FileG as u8;
pub const FILE_H: u8 = File::FileH as u8;

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Rank {
    Rank1 = 0,
    Rank2,
    Rank3,
    Rank4,
    Rank5,
    Rank6,
    Rank7,
    Rank8,
    None,
}

pub const RANK_1: u8 = Rank::Rank1 as u8;
pub const RANK_2: u8 = Rank::Rank2 as u8;
pub const RANK_3: u8 = Rank::Rank3 as u8;
pub const RANK_4: u8 = Rank::Rank4 as u8;
pub const RANK_5: u8 = Rank::Rank5 as u8;
pub const RANK_6: u8 = Rank::Rank6 as u8;
pub const RANK_7: u8 = Rank::Rank7 as u8;
pub const RANK_8: u8 = Rank::Rank8 as u8;

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Colour {
    White = 0,
    Black,
    Both,
}

pub const WHITE: u8 = Colour::White as u8;
pub const BLACK: u8 = Colour::Black as u8;
pub const BOTH: u8 = Colour::Both as u8;

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
#[rustfmt::skip]
pub enum Square120 {
    A1 = 21, B1, C1, D1, E1, F1, G1, H1,
    A2 = 31, B2, C2, D2, E2, F2, G2, H2,
    A3 = 41, B3, C3, D3, E3, F3, G3, H3,
    A4 = 51, B4, C4, D4, E4, F4, G4, H4,
    A5 = 61, B5, C5, D5, E5, F5, G5, H5,
    A6 = 71, B6, C6, D6, E6, F6, G6, H6,
    A7 = 81, B7, C7, D7, E7, F7, G7, H7,
    A8 = 91, B8, C8, D8, E8, F8, G8, H8,
    NoSquare,
    OffBoard,
}

pub const NO_SQUARE: u8 = Square120::NoSquare as u8;
pub const OFF_BOARD: u8 = Square120::OffBoard as u8;
pub const A1: u8 = Square120::A1 as u8;
pub const B1: u8 = Square120::B1 as u8;
pub const C1: u8 = Square120::C1 as u8;
pub const D1: u8 = Square120::D1 as u8;
pub const E1: u8 = Square120::E1 as u8;
pub const F1: u8 = Square120::F1 as u8;
pub const G1: u8 = Square120::G1 as u8;
pub const H1: u8 = Square120::H1 as u8;
pub const A2: u8 = Square120::A2 as u8;
pub const B2: u8 = Square120::B2 as u8;
pub const C2: u8 = Square120::C2 as u8;
pub const D2: u8 = Square120::D2 as u8;
pub const E2: u8 = Square120::E2 as u8;
pub const F2: u8 = Square120::F2 as u8;
pub const G2: u8 = Square120::G2 as u8;
pub const H2: u8 = Square120::H2 as u8;
pub const A3: u8 = Square120::A3 as u8;
pub const B3: u8 = Square120::B3 as u8;
pub const C3: u8 = Square120::C3 as u8;
pub const D3: u8 = Square120::D3 as u8;
pub const E3: u8 = Square120::E3 as u8;
pub const F3: u8 = Square120::F3 as u8;
pub const G3: u8 = Square120::G3 as u8;
pub const H3: u8 = Square120::H3 as u8;
pub const A4: u8 = Square120::A4 as u8;
pub const B4: u8 = Square120::B4 as u8;
pub const C4: u8 = Square120::C4 as u8;
pub const D4: u8 = Square120::D4 as u8;
pub const E4: u8 = Square120::E4 as u8;
pub const F4: u8 = Square120::F4 as u8;
pub const G4: u8 = Square120::G4 as u8;
pub const H4: u8 = Square120::H4 as u8;
pub const A5: u8 = Square120::A5 as u8;
pub const B5: u8 = Square120::B5 as u8;
pub const C5: u8 = Square120::C5 as u8;
pub const D5: u8 = Square120::D5 as u8;
pub const E5: u8 = Square120::E5 as u8;
pub const F5: u8 = Square120::F5 as u8;
pub const G5: u8 = Square120::G5 as u8;
pub const H5: u8 = Square120::H5 as u8;
pub const A6: u8 = Square120::A6 as u8;
pub const B6: u8 = Square120::B6 as u8;
pub const C6: u8 = Square120::C6 as u8;
pub const D6: u8 = Square120::D6 as u8;
pub const E6: u8 = Square120::E6 as u8;
pub const F6: u8 = Square120::F6 as u8;
pub const G6: u8 = Square120::G6 as u8;
pub const H6: u8 = Square120::H6 as u8;
pub const A7: u8 = Square120::A7 as u8;
pub const B7: u8 = Square120::B7 as u8;
pub const C7: u8 = Square120::C7 as u8;
pub const D7: u8 = Square120::D7 as u8;
pub const E7: u8 = Square120::E7 as u8;
pub const F7: u8 = Square120::F7 as u8;
pub const G7: u8 = Square120::G7 as u8;
pub const H7: u8 = Square120::H7 as u8;
pub const A8: u8 = Square120::A8 as u8;
pub const B8: u8 = Square120::B8 as u8;
pub const C8: u8 = Square120::C8 as u8;
pub const D8: u8 = Square120::D8 as u8;
pub const E8: u8 = Square120::E8 as u8;
pub const F8: u8 = Square120::F8 as u8;
pub const G8: u8 = Square120::G8 as u8;
pub const H8: u8 = Square120::H8 as u8;

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
#[rustfmt::skip]
pub enum Square64 {
    A1, B1, C1, D1, E1, F1, G1, H1,
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
    WK = 0b0001,
    WQ = 0b0010,
    BK = 0b0100,
    BQ = 0b1000,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Undo {
    pub m: Move,
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

pub const STARTING_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

pub fn print_bitboard(bb: u64) {
    for row in 0..8 {
        for col in 0..8 {
            if bb & (1 << (row * 8 + col)) != 0 {
                print!("X");
            } else {
                print!(".");
            }
        }
        println!();
    }
}
