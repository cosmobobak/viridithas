#![allow(dead_code)]

use std::{
    fmt::Display,
    ops::{Add, AddAssign, Sub, SubAssign},
};

use crate::{
    board::evaluation::MATE_SCORE,
    chessmove::Move,
    lookups::{file, rank, SQUARE_NAMES},
    macros,
};

pub const BOARD_N_SQUARES: usize = 64;
pub const MAX_GAME_MOVES: usize = 1024;
pub const MAX_DEPTH: Depth = Depth::new(128);
pub const INFINITY: i32 = MATE_SCORE * 2;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Depth(i32);

impl Depth {
    const INNER_INCREMENT_PER_PLY: i32 = 100;

    pub const fn new(depth: i32) -> Self {
        Self(depth * Self::INNER_INCREMENT_PER_PLY)
    }

    pub const fn from_raw(raw: i32) -> Self {
        Self(raw)
    }

    pub const fn n_ply(self) -> usize {
        #![allow(clippy::cast_sign_loss)]
        if self.0 <= 0 {
            0
        } else {
            (self.0 / Self::INNER_INCREMENT_PER_PLY) as usize
        }
    }

    pub const fn round(self) -> i32 {
        self.0 / Self::INNER_INCREMENT_PER_PLY
    }

    pub const fn nearest_full_ply(self) -> Self {
        let x = self.0;
        let x = x + Self::INNER_INCREMENT_PER_PLY / 2;
        Self(x - x % Self::INNER_INCREMENT_PER_PLY)
    }

    pub const fn is_exact_ply(self) -> bool {
        self.0 % Self::INNER_INCREMENT_PER_PLY == 0
    }

    pub const fn raw_inner(self) -> i32 {
        self.0
    }
}

impl Add<Self> for Depth {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        debug_assert!(
            self.is_exact_ply() && other.is_exact_ply(),
            "Depth::add: only exact ply allowed, got {} and {}",
            self,
            other
        );
        Self(self.0 + other.0)
    }
}
impl AddAssign<Self> for Depth {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}
impl Add<i32> for Depth {
    type Output = Self;
    fn add(self, other: i32) -> Self::Output {
        debug_assert!(
            self.is_exact_ply(),
            "Depth::add: only exact ply allowed, got {}",
            self
        );
        Self(self.0 + other * Self::INNER_INCREMENT_PER_PLY)
    }
}
impl AddAssign<i32> for Depth {
    fn add_assign(&mut self, other: i32) {
        *self = *self + other;
    }
}
impl Sub<Self> for Depth {
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        debug_assert!(
            self.is_exact_ply() && other.is_exact_ply(),
            "Depth::sub: only exact ply allowed, got {} and {}",
            self,
            other
        );
        Self(self.0 - other.0)
    }
}
impl SubAssign<Self> for Depth {
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}
impl Sub<i32> for Depth {
    type Output = Self;
    fn sub(self, other: i32) -> Self::Output {
        debug_assert!(
            self.is_exact_ply(),
            "Depth::sub: only exact ply allowed, got {}",
            self
        );
        Self(self.0 - other * Self::INNER_INCREMENT_PER_PLY)
    }
}
impl SubAssign<i32> for Depth {
    fn sub_assign(&mut self, other: i32) {
        *self = *self - other;
    }
}
impl From<i32> for Depth {
    fn from(depth: i32) -> Self {
        Self::new(depth)
    }
}
impl From<f32> for Depth {
    fn from(depth: f32) -> Self {
        #![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        let inner_depth = depth * Self::INNER_INCREMENT_PER_PLY as f32;
        Self::from_raw(inner_depth as i32)
    }
}
impl From<Depth> for f32 {
    fn from(depth: Depth) -> Self {
        #![allow(clippy::cast_precision_loss)]
        depth.0 as Self / Depth::INNER_INCREMENT_PER_PLY as Self
    }
}
impl TryFrom<Depth> for i16 {
    type Error = <Self as std::convert::TryFrom<i32>>::Error;
    fn try_from(depth: Depth) -> Result<Self, Self::Error> {
        depth.0.try_into()
    }
}
impl From<i16> for Depth {
    fn from(depth: i16) -> Self {
        Self::from_raw(i32::from(depth))
    }
}
impl Display for Depth {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let sign = match self.0.signum() {
            1 | 0 => "",
            -1 => "-",
            _ => unreachable!(),
        };
        write!(
            f,
            "{}{}.{}",
            sign,
            self.0.abs() / Self::INNER_INCREMENT_PER_PLY,
            self.0.abs() % Self::INNER_INCREMENT_PER_PLY
        )
    }
}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct CompactDepthStorage(i16);
impl CompactDepthStorage {
    pub const NULL: Self = Self(0);
}
impl TryFrom<Depth> for CompactDepthStorage {
    type Error = <i16 as std::convert::TryFrom<i32>>::Error;
    fn try_from(depth: Depth) -> Result<Self, Self::Error> {
        let inner = depth.0.try_into()?;
        Ok(Self(inner))
    }
}
impl From<CompactDepthStorage> for Depth {
    fn from(depth: CompactDepthStorage) -> Self {
        Self::from_raw(i32::from(depth.0))
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
#[rustfmt::skip]
enum Piece {
    Empty = 0,
    WP, WN, WB, WR, WQ, WK,
    BP, BN, BB, BR, BQ, BK,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
#[rustfmt::skip]
enum PieceType {
    Empty = 0,
    Pawn, Knight, Bishop, Rook, Queen, King,
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
pub const PAWN: u8 = PieceType::Pawn as u8;
pub const KNIGHT: u8 = PieceType::Knight as u8;
pub const BISHOP: u8 = PieceType::Bishop as u8;
pub const ROOK: u8 = PieceType::Rook as u8;
pub const QUEEN: u8 = PieceType::Queen as u8;
pub const KING: u8 = PieceType::King as u8;

pub const fn type_of(piece: u8) -> u8 {
    match piece {
        WP | BP => PAWN,
        WN | BN => KNIGHT,
        WB | BB => BISHOP,
        WR | BR => ROOK,
        WQ | BQ => QUEEN,
        WK | BK => KING,
        _ => PIECE_EMPTY,
    }
}

pub const fn colour_of(piece: u8) -> u8 {
    match piece {
        WP | WN | WB | WR | WQ | WK => WHITE,
        BP | BN | BB | BR | BQ | BK => BLACK,
        _ => unsafe { macros::impossible!() },
    }
}

pub const fn u8max(a: u8, b: u8) -> u8 {
    if a > b {
        a
    } else {
        b
    }
}

pub const fn square_distance(a: u8, b: u8) -> u8 {
    u8max(file(a).abs_diff(file(b)), rank(a).abs_diff(rank(b)))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

impl Display for File {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::FileA => "a",
                Self::FileB => "b",
                Self::FileC => "c",
                Self::FileD => "d",
                Self::FileE => "e",
                Self::FileF => "f",
                Self::FileG => "g",
                Self::FileH => "h",
                Self::None => "NO_FILE",
            }
        )
    }
}

pub const FILE_A: u8 = File::FileA as u8;
pub const FILE_B: u8 = File::FileB as u8;
pub const FILE_C: u8 = File::FileC as u8;
pub const FILE_D: u8 = File::FileD as u8;
pub const FILE_E: u8 = File::FileE as u8;
pub const FILE_F: u8 = File::FileF as u8;
pub const FILE_G: u8 = File::FileG as u8;
pub const FILE_H: u8 = File::FileH as u8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

impl Display for Rank {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Rank1 => "1",
                Self::Rank2 => "2",
                Self::Rank3 => "3",
                Self::Rank4 => "4",
                Self::Rank5 => "5",
                Self::Rank6 => "6",
                Self::Rank7 => "7",
                Self::Rank8 => "8",
                Self::None => "NO_RANK",
            }
        )
    }
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
pub enum Square64 {
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
    NoSquare
}

pub const A1: u8 = Square64::A1 as u8;
pub const B1: u8 = Square64::B1 as u8;
pub const C1: u8 = Square64::C1 as u8;
pub const D1: u8 = Square64::D1 as u8;
pub const E1: u8 = Square64::E1 as u8;
pub const F1: u8 = Square64::F1 as u8;
pub const G1: u8 = Square64::G1 as u8;
pub const H1: u8 = Square64::H1 as u8;
pub const A2: u8 = Square64::A2 as u8;
pub const B2: u8 = Square64::B2 as u8;
pub const C2: u8 = Square64::C2 as u8;
pub const D2: u8 = Square64::D2 as u8;
pub const E2: u8 = Square64::E2 as u8;
pub const F2: u8 = Square64::F2 as u8;
pub const G2: u8 = Square64::G2 as u8;
pub const H2: u8 = Square64::H2 as u8;
pub const A3: u8 = Square64::A3 as u8;
pub const B3: u8 = Square64::B3 as u8;
pub const C3: u8 = Square64::C3 as u8;
pub const D3: u8 = Square64::D3 as u8;
pub const E3: u8 = Square64::E3 as u8;
pub const F3: u8 = Square64::F3 as u8;
pub const G3: u8 = Square64::G3 as u8;
pub const H3: u8 = Square64::H3 as u8;
pub const A4: u8 = Square64::A4 as u8;
pub const B4: u8 = Square64::B4 as u8;
pub const C4: u8 = Square64::C4 as u8;
pub const D4: u8 = Square64::D4 as u8;
pub const E4: u8 = Square64::E4 as u8;
pub const F4: u8 = Square64::F4 as u8;
pub const G4: u8 = Square64::G4 as u8;
pub const H4: u8 = Square64::H4 as u8;
pub const A5: u8 = Square64::A5 as u8;
pub const B5: u8 = Square64::B5 as u8;
pub const C5: u8 = Square64::C5 as u8;
pub const D5: u8 = Square64::D5 as u8;
pub const E5: u8 = Square64::E5 as u8;
pub const F5: u8 = Square64::F5 as u8;
pub const G5: u8 = Square64::G5 as u8;
pub const H5: u8 = Square64::H5 as u8;
pub const A6: u8 = Square64::A6 as u8;
pub const B6: u8 = Square64::B6 as u8;
pub const C6: u8 = Square64::C6 as u8;
pub const D6: u8 = Square64::D6 as u8;
pub const E6: u8 = Square64::E6 as u8;
pub const F6: u8 = Square64::F6 as u8;
pub const G6: u8 = Square64::G6 as u8;
pub const H6: u8 = Square64::H6 as u8;
pub const A7: u8 = Square64::A7 as u8;
pub const B7: u8 = Square64::B7 as u8;
pub const C7: u8 = Square64::C7 as u8;
pub const D7: u8 = Square64::D7 as u8;
pub const E7: u8 = Square64::E7 as u8;
pub const F7: u8 = Square64::F7 as u8;
pub const G7: u8 = Square64::G7 as u8;
pub const H7: u8 = Square64::H7 as u8;
pub const A8: u8 = Square64::A8 as u8;
pub const B8: u8 = Square64::B8 as u8;
pub const C8: u8 = Square64::C8 as u8;
pub const D8: u8 = Square64::D8 as u8;
pub const E8: u8 = Square64::E8 as u8;
pub const F8: u8 = Square64::F8 as u8;
pub const G8: u8 = Square64::G8 as u8;
pub const H8: u8 = Square64::H8 as u8;
pub const NO_SQUARE: u8 = Square64::NoSquare as u8;

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Castling {
    WK = 0b0001,
    WQ = 0b0010,
    BK = 0b0100,
    BQ = 0b1000,
}

pub const WKCA: u8 = Castling::WK as u8;
pub const WQCA: u8 = Castling::WQ as u8;
pub const BKCA: u8 = Castling::BK as u8;
pub const BQCA: u8 = Castling::BQ as u8;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Undo {
    pub m: Move,
    pub castle_perm: u8,
    pub ep_square: u8,
    pub fifty_move_counter: u8,
    pub position_key: u64,
}

pub fn square_name(sq: u8) -> Option<&'static str> {
    SQUARE_NAMES.get(sq as usize).copied()
}

pub const STARTING_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
