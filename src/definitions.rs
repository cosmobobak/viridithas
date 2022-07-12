use std::{
    fmt::Display,
    ops::{Add, AddAssign, Sub, SubAssign},
    str::FromStr,
};

use crate::{
    board::evaluation::MATE_SCORE,
    chessmove::Move,
    lookups::{file, rank, SQUARE_NAMES}, macros,
};

pub const BOARD_N_SQUARES: usize = 64;
pub const MAX_DEPTH: Depth = Depth::new(128);
pub const INFINITY: i32 = MATE_SCORE * 2;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Depth(i32);

impl Depth {
    pub const ONE_PLY: i32 = 100;

    pub const fn new(depth: i32) -> Self {
        Self(depth * Self::ONE_PLY)
    }

    pub const fn from_raw(raw: i32) -> Self {
        Self(raw)
    }

    pub const fn n_ply(self) -> usize {
        #![allow(clippy::cast_sign_loss)]
        if self.0 <= 0 {
            0
        } else {
            (self.0 / Self::ONE_PLY) as usize
        }
    }

    pub const fn round(self) -> i32 {
        self.0 / Self::ONE_PLY
    }

    pub const fn nearest_full_ply(self) -> Self {
        let x = self.0;
        let x = x + Self::ONE_PLY / 2;
        Self(x - x % Self::ONE_PLY)
    }

    pub const fn is_exact_ply(self) -> bool {
        self.0 % Self::ONE_PLY == 0
    }

    pub const fn raw_inner(self) -> i32 {
        self.0
    }
}

impl Add<Self> for Depth {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
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
        Self(self.0 + other * Self::ONE_PLY)
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
        Self(self.0 - other * Self::ONE_PLY)
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
        let inner_depth = depth * Self::ONE_PLY as f32;
        Self::from_raw(inner_depth as i32)
    }
}
impl From<Depth> for f32 {
    fn from(depth: Depth) -> Self {
        #![allow(clippy::cast_precision_loss)]
        depth.0 as Self / Depth::ONE_PLY as Self
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
            self.0.abs() / Self::ONE_PLY,
            self.0.abs() % Self::ONE_PLY
        )
    }
}
impl FromStr for Depth {
    type Err = std::num::ParseFloatError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let floating_repr: f32 = s.parse()?;
        Ok(Self::from(floating_repr))
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
    Pawn = 1, Knight, Bishop, Rook, Queen, King,
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

pub const fn piece_index(piece: u8) -> usize {
    match piece {
        WP | BP => 0,
        WN | BN => 1,
        WB | BB => 2,
        WR | BR => 3,
        WQ | BQ => 4,
        WK | BK => 5,
        _ => unsafe { macros::impossible!() },
    }
}

pub const fn colour_of(piece: u8) -> u8 {
    match piece {
        WP | WN | WB | WR | WQ | WK => WHITE,
        _ => BLACK,
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

#[allow(non_snake_case, dead_code)]
pub mod File {
    pub const FILE_A: u8 = 0;
    pub const FILE_B: u8 = 1;
    pub const FILE_C: u8 = 2;
    pub const FILE_D: u8 = 3;
    pub const FILE_E: u8 = 4;
    pub const FILE_F: u8 = 5;
    pub const FILE_G: u8 = 6;
    pub const FILE_H: u8 = 7;
}

#[allow(non_snake_case, dead_code)]
pub mod Rank {
    pub const RANK_1: u8 = 0;
    pub const RANK_2: u8 = 1;
    pub const RANK_3: u8 = 2;
    pub const RANK_4: u8 = 3;
    pub const RANK_5: u8 = 4;
    pub const RANK_6: u8 = 5;
    pub const RANK_7: u8 = 6;
    pub const RANK_8: u8 = 7;
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Colour {
    White = 0,
    Black,
    Both,
}

pub const WHITE: u8 = Colour::White as u8;
pub const BLACK: u8 = Colour::Black as u8;

#[allow(non_snake_case, dead_code)]
pub mod Square {
    pub const A1: u8 = 0;
    pub const B1: u8 = 1;
    pub const C1: u8 = 2;
    pub const D1: u8 = 3;
    pub const E1: u8 = 4;
    pub const F1: u8 = 5;
    pub const G1: u8 = 6;
    pub const H1: u8 = 7;
    pub const A2: u8 = 8;
    pub const B2: u8 = 9;
    pub const C2: u8 = 10;
    pub const D2: u8 = 11;
    pub const E2: u8 = 12;
    pub const F2: u8 = 13;
    pub const G2: u8 = 14;
    pub const H2: u8 = 15;
    pub const A3: u8 = 16;
    pub const B3: u8 = 17;
    pub const C3: u8 = 18;
    pub const D3: u8 = 19;
    pub const E3: u8 = 20;
    pub const F3: u8 = 21;
    pub const G3: u8 = 22;
    pub const H3: u8 = 23;
    pub const A4: u8 = 24;
    pub const B4: u8 = 25;
    pub const C4: u8 = 26;
    pub const D4: u8 = 27;
    pub const E4: u8 = 28;
    pub const F4: u8 = 29;
    pub const G4: u8 = 30;
    pub const H4: u8 = 31;
    pub const A5: u8 = 32;
    pub const B5: u8 = 33;
    pub const C5: u8 = 34;
    pub const D5: u8 = 35;
    pub const E5: u8 = 36;
    pub const F5: u8 = 37;
    pub const G5: u8 = 38;
    pub const H5: u8 = 39;
    pub const A6: u8 = 40;
    pub const B6: u8 = 41;
    pub const C6: u8 = 42;
    pub const D6: u8 = 43;
    pub const E6: u8 = 44;
    pub const F6: u8 = 45;
    pub const G6: u8 = 46;
    pub const H6: u8 = 47;
    pub const A7: u8 = 48;
    pub const B7: u8 = 49;
    pub const C7: u8 = 50;
    pub const D7: u8 = 51;
    pub const E7: u8 = 52;
    pub const F7: u8 = 53;
    pub const G7: u8 = 54;
    pub const H7: u8 = 55;
    pub const A8: u8 = 56;
    pub const B8: u8 = 57;
    pub const C8: u8 = 58;
    pub const D8: u8 = 59;
    pub const E8: u8 = 60;
    pub const F8: u8 = 61;
    pub const G8: u8 = 62;
    pub const H8: u8 = 63;
    pub const NO_SQUARE: u8 = 64;
}

pub const WKCA: u8 = 0b0001;
pub const WQCA: u8 = 0b0010;
pub const BKCA: u8 = 0b0100;
pub const BQCA: u8 = 0b1000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Undo {
    pub m: Move,
    pub castle_perm: u8,
    pub ep_square: u8,
    pub fifty_move_counter: u8,
}

pub fn square_name(sq: u8) -> Option<&'static str> {
    SQUARE_NAMES.get(sq as usize).copied()
}

pub const fn flip_rank(sq: u8) -> u8 {
    sq ^ 56
}

pub const fn flip_file(sq: u8) -> u8 {
    sq ^ 7
}

#[allow(dead_code)]
pub const STARTING_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

mod tests {
    #[test]
    fn square_flipping() {
        use super::{
            flip_file, flip_rank,
            Square::{A1, A8, H1, H8},
        };

        assert_eq!(flip_rank(A1), A8);
        assert_eq!(flip_rank(H1), H8);
        assert_eq!(flip_rank(A8), A1);
        assert_eq!(flip_rank(H8), H1);

        assert_eq!(flip_file(A1), H1);
        assert_eq!(flip_file(H1), A1);
        assert_eq!(flip_file(A8), H8);
        assert_eq!(flip_file(H8), A8);
    }
}
