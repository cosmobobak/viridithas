pub mod depth;

use std::{
    fmt::{self, Display},
    str::FromStr,
};

use crate::{
    board::evaluation::MATE_SCORE,
    chessmove::Move,
    piece::{Colour, Piece},
};

pub const BOARD_N_SQUARES: usize = 64;
pub const MAX_DEPTH: depth::Depth = depth::Depth::new(128);
pub const MAX_PLY: usize = MAX_DEPTH.ply_to_horizon();
pub const INFINITY: i32 = MATE_SCORE * 2;
pub const MEGABYTE: usize = 1024 * 1024;

pub struct File;

#[allow(dead_code)]
impl File {
    pub const FILE_A: u8 = 0;
    pub const FILE_B: u8 = 1;
    pub const FILE_C: u8 = 2;
    pub const FILE_D: u8 = 3;
    pub const FILE_E: u8 = 4;
    pub const FILE_F: u8 = 5;
    pub const FILE_G: u8 = 6;
    pub const FILE_H: u8 = 7;
}

pub struct Rank;

#[allow(dead_code)]
impl Rank {
    pub const RANK_1: u8 = 0;
    pub const RANK_2: u8 = 1;
    pub const RANK_3: u8 = 2;
    pub const RANK_4: u8 = 3;
    pub const RANK_5: u8 = 4;
    pub const RANK_6: u8 = 5;
    pub const RANK_7: u8 = 6;
    pub const RANK_8: u8 = 7;

    pub fn double_pawn_push_rank(colour: Colour) -> u8 {
        if colour == Colour::WHITE {
            Self::RANK_4
        } else {
            Self::RANK_5
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug, PartialOrd, Ord)]
pub struct Square(u8);

static SQUARE_NAMES: [&str; 64] = [
    "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1", "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
    "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3", "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
    "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5", "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
    "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7", "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
];

#[allow(clippy::unusual_byte_groupings)]
impl Square {
    pub const A1: Self = Self(0);
    pub const B1: Self = Self(1);
    pub const C1: Self = Self(2);
    pub const D1: Self = Self(3);
    pub const E1: Self = Self(4);
    pub const F1: Self = Self(5);
    pub const G1: Self = Self(6);
    pub const H1: Self = Self(7);
    pub const A2: Self = Self(8);
    pub const B2: Self = Self(9);
    pub const C2: Self = Self(10);
    pub const D2: Self = Self(11);
    pub const E2: Self = Self(12);
    pub const F2: Self = Self(13);
    pub const G2: Self = Self(14);
    pub const H2: Self = Self(15);
    pub const A3: Self = Self(16);
    pub const B3: Self = Self(17);
    pub const C3: Self = Self(18);
    pub const D3: Self = Self(19);
    pub const E3: Self = Self(20);
    pub const F3: Self = Self(21);
    pub const G3: Self = Self(22);
    pub const H3: Self = Self(23);
    pub const A4: Self = Self(24);
    pub const B4: Self = Self(25);
    pub const C4: Self = Self(26);
    pub const D4: Self = Self(27);
    pub const E4: Self = Self(28);
    pub const F4: Self = Self(29);
    pub const G4: Self = Self(30);
    pub const H4: Self = Self(31);
    pub const A5: Self = Self(32);
    pub const B5: Self = Self(33);
    pub const C5: Self = Self(34);
    pub const D5: Self = Self(35);
    pub const E5: Self = Self(36);
    pub const F5: Self = Self(37);
    pub const G5: Self = Self(38);
    pub const H5: Self = Self(39);
    pub const A6: Self = Self(40);
    pub const B6: Self = Self(41);
    pub const C6: Self = Self(42);
    pub const D6: Self = Self(43);
    pub const E6: Self = Self(44);
    pub const F6: Self = Self(45);
    pub const G6: Self = Self(46);
    pub const H6: Self = Self(47);
    pub const A7: Self = Self(48);
    pub const B7: Self = Self(49);
    pub const C7: Self = Self(50);
    pub const D7: Self = Self(51);
    pub const E7: Self = Self(52);
    pub const F7: Self = Self(53);
    pub const G7: Self = Self(54);
    pub const H7: Self = Self(55);
    pub const A8: Self = Self(56);
    pub const B8: Self = Self(57);
    pub const C8: Self = Self(58);
    pub const D8: Self = Self(59);
    pub const E8: Self = Self(60);
    pub const F8: Self = Self(61);
    pub const G8: Self = Self(62);
    pub const H8: Self = Self(63);
    pub const NO_SQUARE: Self = Self(64);

    pub const fn from_rank_file(rank: u8, file: u8) -> Self {
        let inner = rank * 8 + file;
        debug_assert!(inner <= 64);
        Self(inner)
    }

    pub const fn new(inner: u8) -> Self {
        debug_assert!(inner <= 64);
        Self(inner)
    }

    pub const fn flip_rank(self) -> Self {
        Self(self.0 ^ 0b111_000)
    }

    pub const fn flip_file(self) -> Self {
        Self(self.0 ^ 0b000_111)
    }

    /// The file that this square is on.
    pub const fn file(self) -> u8 {
        self.0 % 8
    }
    /// The rank that this square is on.
    pub const fn rank(self) -> u8 {
        self.0 / 8
    }

    pub const fn distance(a: Self, b: Self) -> u8 {
        max!(a.file().abs_diff(b.file()), a.rank().abs_diff(b.rank()))
    }

    pub const fn signed_inner(self) -> i8 {
        #![allow(clippy::cast_possible_wrap)]
        self.0 as i8
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }

    pub const fn add(self, offset: u8) -> Self {
        #![allow(
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap,
            clippy::cast_sign_loss
        )]
        let res = self.0 + offset;
        debug_assert!(res < 64, "Square::add overflowed");
        Self(res)
    }

    pub const fn sub(self, offset: u8) -> Self {
        #![allow(
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap,
            clippy::cast_sign_loss
        )]
        let res = self.0 - offset;
        debug_assert!(res < 64, "Square::sub overflowed");
        Self(res)
    }

    pub const fn on_board(self) -> bool {
        self.0 < 64
    }

    pub const fn bitboard(self) -> u64 {
        1 << self.0
    }

    pub fn pawn_push(self, side: Colour) -> Self {
        if side == Colour::WHITE {
            self.add(8)
        } else {
            self.sub(8)
        }
    }

    pub fn pawn_right(self, side: Colour) -> Self {
        if side == Colour::WHITE {
            self.add(9)
        } else {
            self.sub(7)
        }
    }

    pub fn pawn_left(self, side: Colour) -> Self {
        if side == Colour::WHITE {
            self.add(7)
        } else {
            self.sub(9)
        }
    }

    #[rustfmt::skip]
    pub const fn le(self, other: Self) -> bool { self.0 <= other.0 }
    #[rustfmt::skip]
    pub const fn ge(self, other: Self) -> bool { self.0 >= other.0 }
    #[rustfmt::skip]
    pub const fn lt(self, other: Self) -> bool { self.0 < other.0  }
    #[rustfmt::skip]
    pub const fn gt(self, other: Self) -> bool { self.0 > other.0  }

    pub fn all() -> impl Iterator<Item = Self> {
        (0..64).map(Self::new)
    }

    pub fn name(self) -> Option<&'static str> {
        SQUARE_NAMES.get(self.index()).copied()
    }
}

impl Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = SQUARE_NAMES.get(self.index()).copied();
        if let Some(name) = name {
            write!(f, "{name}")
        } else if self.0 == 64 {
            write!(f, "NO_SQUARE")
        } else {
            write!(f, "ILLEGAL: Square({})", self.0)
        }
    }
}
impl FromStr for Square {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        SQUARE_NAMES
            .iter()
            .position(|&name| name == s)
            .and_then(|index| -> Option<u8> { index.try_into().ok() })
            .map(Self::new)
            .ok_or("Invalid square name")
    }
}
impl From<Square> for u16 {
    fn from(square: Square) -> Self {
        Self::from(square.0)
    }
}

pub const WKCA: u8 = 0b0001;
pub const WQCA: u8 = 0b0010;
pub const BKCA: u8 = 0b0100;
pub const BQCA: u8 = 0b1000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Undo {
    pub m: Move,
    pub castle_perm: u8,
    pub ep_square: Square,
    pub fifty_move_counter: u8,
    pub capture: Piece,
}

pub enum CheckState {
    None,
    Check,
    Checkmate,
}

pub struct StackVec<T: Copy, const CAPACITY: usize> {
    data: [T; CAPACITY],
    len: usize,
}

impl<T: Copy, const CAPACITY: usize> StackVec<T, CAPACITY> {
    pub const fn from_default(default: T) -> Self {
        Self { data: [default; CAPACITY], len: 0 }
    }

    pub fn push(&mut self, item: T) {
        self.data[self.len] = item;
        self.len += 1;
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data[..self.len]
    }
}

mod tests {
    #[test]
    fn square_flipping() {
        use super::Square;

        assert_eq!(Square::A1.flip_rank(), Square::A8);
        assert_eq!(Square::H1.flip_rank(), Square::H8);
        assert_eq!(Square::A8.flip_rank(), Square::A1);
        assert_eq!(Square::H8.flip_rank(), Square::H1);

        assert_eq!(Square::A1.flip_file(), Square::H1);
        assert_eq!(Square::H1.flip_file(), Square::A1);
        assert_eq!(Square::A8.flip_file(), Square::H8);
        assert_eq!(Square::H8.flip_file(), Square::A8);
    }
}
