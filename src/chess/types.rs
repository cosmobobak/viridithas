use std::{
    fmt::{self, Display},
    mem::size_of,
    ops::{Index, IndexMut},
    str::FromStr,
};

use crate::chess::{
    piece::{Colour, Piece},
    piecelayout::{PieceLayout, Threats},
    squareset::SquareSet,
};

#[derive(PartialEq, Eq, Clone, Copy, PartialOrd, Ord, Hash, Debug)]
#[repr(u8)]
pub enum File {
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
}

const _FILE_ASSERT: () = assert!(size_of::<File>() == size_of::<Option<File>>());

impl File {
    pub const ALL: [Self; 8] = [
        Self::A,
        Self::B,
        Self::C,
        Self::D,
        Self::E,
        Self::F,
        Self::G,
        Self::H,
    ];

    pub const fn abs_diff(self, other: Self) -> u8 {
        (self as u8).abs_diff(other as u8)
    }

    pub const fn from_index(index: u8) -> Option<Self> {
        if index < 8 {
            // SAFETY: inner is less than 8, so it corresponds to a valid enum variant.
            Some(unsafe { std::mem::transmute::<u8, Self>(index) })
        } else {
            None
        }
    }

    pub const fn add(self, diff: u8) -> Option<Self> {
        Self::from_index(self as u8 + diff)
    }

    pub const fn sub(self, diff: u8) -> Option<Self> {
        #![allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
        Self::from_index((self as i8 - diff as i8) as u8)
    }
}

impl<T> Index<File> for [T; 8] {
    type Output = T;

    fn index(&self, index: File) -> &Self::Output {
        // SAFETY: the legal values for this type are all in bounds.
        unsafe { self.get_unchecked(index as usize) }
    }
}

impl<T> IndexMut<File> for [T; 8] {
    fn index_mut(&mut self, index: File) -> &mut Self::Output {
        // SAFETY: the legal values for this type are all in bounds.
        unsafe { self.get_unchecked_mut(index as usize) }
    }
}

#[derive(PartialEq, Eq, Clone, Copy, PartialOrd, Ord, Hash, Debug)]
#[repr(u8)]
pub enum Rank {
    One,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
}

const _RANK_ASSERT: () = assert!(size_of::<Rank>() == size_of::<Option<Rank>>());

impl Rank {
    pub const ALL: [Self; 8] = [
        Self::One,
        Self::Two,
        Self::Three,
        Self::Four,
        Self::Five,
        Self::Six,
        Self::Seven,
        Self::Eight,
    ];

    pub const fn abs_diff(self, other: Self) -> u8 {
        (self as u8).abs_diff(other as u8)
    }

    pub const fn from_index(index: u8) -> Option<Self> {
        if index < 8 {
            // SAFETY: inner is less than 8, so it corresponds to a valid enum variant.
            Some(unsafe { std::mem::transmute::<u8, Self>(index) })
        } else {
            None
        }
    }

    pub const fn add(self, diff: u8) -> Option<Self> {
        Self::from_index(self as u8 + diff)
    }

    pub const fn sub(self, diff: u8) -> Option<Self> {
        #![allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
        Self::from_index((self as i8 - diff as i8) as u8)
    }
}

impl<T> Index<Rank> for [T; 8] {
    type Output = T;

    fn index(&self, index: Rank) -> &Self::Output {
        // SAFETY: the legal values for this type are all in bounds.
        unsafe { self.get_unchecked(index as usize) }
    }
}

impl<T> IndexMut<Rank> for [T; 8] {
    fn index_mut(&mut self, index: Rank) -> &mut Self::Output {
        // SAFETY: the legal values for this type are all in bounds.
        unsafe { self.get_unchecked_mut(index as usize) }
    }
}

#[rustfmt::skip]
#[derive(PartialEq, Eq, Clone, Copy, PartialOrd, Ord, Hash, Debug)]
#[repr(u8)]
pub enum Square {
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
}

const _SQUARE_ASSERT: () = assert!(size_of::<Square>() == size_of::<Option<Square>>());

impl<T> Index<Square> for [T; 64] {
    type Output = T;

    fn index(&self, index: Square) -> &Self::Output {
        // SAFETY: the legal values for this type are all in bounds.
        unsafe { self.get_unchecked(index as usize) }
    }
}

impl<T> IndexMut<Square> for [T; 64] {
    fn index_mut(&mut self, index: Square) -> &mut Self::Output {
        // SAFETY: the legal values for this type are all in bounds.
        unsafe { self.get_unchecked_mut(index as usize) }
    }
}

static SQUARE_NAMES: [&str; 64] = [
    "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1", "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
    "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3", "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
    "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5", "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
    "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7", "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
];

#[allow(clippy::unusual_byte_groupings)]
impl Square {
    pub const fn from_rank_file(rank: Rank, file: File) -> Self {
        let inner = rank as u8 * 8 + file as u8;
        // SAFETY: Rank and File are constrained such that inner is always < 64.
        unsafe { std::mem::transmute(inner) }
    }

    pub const fn new(inner: u8) -> Option<Self> {
        if inner < 64 {
            // SAFETY: inner is less than 64, so it corresponds to a valid enum variant.
            Some(unsafe { std::mem::transmute::<u8, Self>(inner) })
        } else {
            None
        }
    }

    pub const fn new_clamped(inner: u8) -> Self {
        let inner = min!(inner, 63);
        let maybe_square = Self::new(inner);
        if let Some(sq) = maybe_square {
            sq
        } else {
            panic!()
        }
    }

    /// SAFETY: you may only call this function with value of `inner` less than 64.
    pub const unsafe fn new_unchecked(inner: u8) -> Self {
        debug_assert!(inner < 64);
        std::mem::transmute(inner)
    }

    pub const fn flip_rank(self) -> Self {
        // SAFETY: given the precondition that `self as u8` is less than 64,
        // this operation cannot construct a value >= 64.
        unsafe { std::mem::transmute(self as u8 ^ 0b111_000) }
    }

    pub const fn flip_file(self) -> Self {
        // SAFETY: given the precondition that `self as u8` is less than 64,
        // this operation cannot construct a value >= 64.
        unsafe { std::mem::transmute(self as u8 ^ 0b000_111) }
    }

    pub const fn relative_to(self, side: Colour) -> Self {
        if matches!(side, Colour::White) {
            self
        } else {
            self.flip_rank()
        }
    }

    /// The file that this square is on.
    pub const fn file(self) -> File {
        // SAFETY: `self as u8` is less than 64, and this operation can only
        // decrease the value, so cannot construct a value >= 64.
        unsafe { std::mem::transmute(self as u8 % 8) }
    }

    /// The rank that this square is on.
    pub const fn rank(self) -> Rank {
        // SAFETY: `self as u8` is less than 64, and this operation can only
        // decrease the value, so cannot construct a value >= 64.
        unsafe { std::mem::transmute(self as u8 / 8) }
    }

    pub const fn distance(a: Self, b: Self) -> u8 {
        max!(a.file().abs_diff(b.file()), a.rank().abs_diff(b.rank()))
    }

    pub const fn signed_inner(self) -> i8 {
        #![allow(clippy::cast_possible_wrap)]
        self as i8
    }

    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn inner(self) -> u8 {
        self as u8
    }

    pub const fn add(self, offset: u8) -> Option<Self> {
        #![allow(
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap,
            clippy::cast_sign_loss
        )]
        let res = self as u8 + offset;
        Self::new(res)
    }

    pub const fn saturating_add(self, offset: u8) -> Self {
        #![allow(
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap,
            clippy::cast_sign_loss
        )]
        let res = self as u8 + offset;
        let inner = min!(res, 63);
        let maybe_square = Self::new(inner);
        if let Some(sq) = maybe_square {
            sq
        } else {
            panic!()
        }
    }

    /// SAFETY: You may not call this function with a square and offset such that
    /// `square as u8 + offset` is outwith `0..64`.
    pub const unsafe fn add_unchecked(self, offset: u8) -> Self {
        #![allow(
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap,
            clippy::cast_sign_loss
        )]
        let res = self as u8 + offset;
        Self::new_unchecked(res)
    }

    /// SAFETY: You may not call this function with a square and offset such that
    /// `square as u8 - offset` is outwith `0..64`.
    pub const fn sub(self, offset: u8) -> Option<Self> {
        #![allow(
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap,
            clippy::cast_sign_loss
        )]
        let res = self as u8 - offset;
        Self::new(res)
    }

    pub const fn as_set(self) -> SquareSet {
        SquareSet::from_inner(1 << self as u8)
    }

    pub fn pawn_push(self, side: Colour) -> Option<Self> {
        if side == Colour::White {
            self.add(8)
        } else {
            self.sub(8)
        }
    }

    pub fn pawn_right(self, side: Colour) -> Option<Self> {
        if side == Colour::White {
            self.add(9)
        } else {
            self.sub(7)
        }
    }

    pub fn pawn_left(self, side: Colour) -> Option<Self> {
        if side == Colour::White {
            self.add(7)
        } else {
            self.sub(9)
        }
    }

    #[rustfmt::skip]
    pub const fn le(self, other: Self) -> bool { self as u8 <= other as u8 }
    #[rustfmt::skip]
    pub const fn ge(self, other: Self) -> bool { self as u8 >= other as u8 }
    #[rustfmt::skip]
    pub const fn lt(self, other: Self) -> bool { (self as u8) < other as u8  }
    #[rustfmt::skip]
    pub const fn gt(self, other: Self) -> bool { self as u8 > other as u8  }

    pub fn all() -> impl Iterator<Item = Self> {
        // SAFETY: all values are within `0..64`.
        (0..64u8).map(|i| unsafe { std::mem::transmute(i) })
    }

    pub fn name(self) -> &'static str {
        SQUARE_NAMES[self]
    }
}

impl Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", SQUARE_NAMES[*self])
    }
}

impl FromStr for Square {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        SQUARE_NAMES
            .iter()
            .position(|&name| name == s)
            .and_then(|index| -> Option<u8> { index.try_into().ok() })
            .and_then(Self::new)
            .ok_or("Invalid square name")
    }
}

impl From<Square> for u16 {
    fn from(square: Square) -> Self {
        square as Self
    }
}

pub const WKCA: u8 = 0b0001;

pub const WQCA: u8 = 0b0010;

pub const BKCA: u8 = 0b0100;

pub const BQCA: u8 = 0b1000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContHistIndex {
    pub piece: Piece,
    pub square: Square,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Undo {
    pub castle_perm: CastlingRights,
    pub ep_square: Option<Square>,
    pub fifty_move_counter: u8,
    pub threats: Threats,
    pub cont_hist_index: Option<ContHistIndex>,
    pub piece_layout: PieceLayout,
    pub piece_array: [Option<Piece>; 64],
    /// The Zobrist hash of the board.
    pub key: u64,
    /// The Zobrist hash of the pawns on the board.
    pub pawn_key: u64,
    /// The Zobrist hash of the non-pawns on the board, split by side.
    pub non_pawn_key: [u64; 2],
    /// The Zobrist hash of the minor pieces on the board.
    pub minor_key: u64,
    /// The Zobrist hash of the major pieces on the board.
    pub major_key: u64,
}

impl Default for Undo {
    fn default() -> Self {
        Self {
            castle_perm: CastlingRights::NONE,
            ep_square: None,
            fifty_move_counter: 0,
            threats: Threats {
                all: SquareSet::EMPTY,
                checkers: SquareSet::EMPTY,
            },
            cont_hist_index: None,
            piece_layout: PieceLayout::NULL,
            piece_array: [None; 64],
            key: 0,
            pawn_key: 0,
            non_pawn_key: [0; 2],
            minor_key: 0,
            major_key: 0,
        }
    }
}

pub enum CheckState {
    None,
    Check,
    Checkmate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CastlingRights {
    pub wk: Option<Square>,
    pub wq: Option<Square>,
    pub bk: Option<Square>,
    pub bq: Option<Square>,
}

impl CastlingRights {
    pub const NONE: Self = Self {
        wk: None,
        wq: None,
        bk: None,
        bq: None,
    };

    pub const fn hashkey_index(self) -> usize {
        let mut index = 0;
        if self.wk.is_some() {
            index |= WKCA;
        }
        if self.wq.is_some() {
            index |= WQCA;
        }
        if self.bk.is_some() {
            index |= BKCA;
        }
        if self.bq.is_some() {
            index |= BQCA;
        }
        index as usize
    }

    pub fn remove(&mut self, sq: Square) {
        let sq = Some(sq);
        if self.wk == sq {
            self.wk = None;
        } else if self.wq == sq {
            self.wq = None;
        } else if self.bk == sq {
            self.bk = None;
        } else if self.bq == sq {
            self.bq = None;
        }
    }

    pub fn kingside(self, side: Colour) -> Option<Square> {
        if side == Colour::White {
            self.wk
        } else {
            self.bk
        }
    }

    #[allow(dead_code)]
    pub fn kingside_mut(&mut self, side: Colour) -> &mut Option<Square> {
        if side == Colour::White {
            &mut self.wk
        } else {
            &mut self.bk
        }
    }

    pub fn queenside(self, side: Colour) -> Option<Square> {
        if side == Colour::White {
            self.wq
        } else {
            self.bq
        }
    }

    #[allow(dead_code)]
    pub fn queenside_mut(&mut self, side: Colour) -> &mut Option<Square> {
        if side == Colour::White {
            &mut self.wq
        } else {
            &mut self.bq
        }
    }

    pub const fn display(&self, chess_960: bool) -> CastlingRightsDisplay {
        CastlingRightsDisplay {
            rights: self,
            chess_960,
        }
    }
}

pub struct CastlingRightsDisplay<'a> {
    rights: &'a CastlingRights,
    chess_960: bool,
}

impl Display for CastlingRightsDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const FILE_NAMES: [u8; 8] = *b"abcdefgh";
        if self.chess_960 {
            if let Some(right) = self.rights.wk {
                write!(
                    f,
                    "{}",
                    FILE_NAMES[right.file()].to_ascii_uppercase() as char
                )?;
            }
            if let Some(right) = self.rights.wq {
                write!(
                    f,
                    "{}",
                    FILE_NAMES[right.file()].to_ascii_uppercase() as char
                )?;
            }
            if let Some(right) = self.rights.bk {
                write!(f, "{}", FILE_NAMES[right.file()] as char)?;
            }
            if let Some(right) = self.rights.bq {
                write!(f, "{}", FILE_NAMES[right.file()] as char)?;
            }
        } else {
            if self.rights.wk.is_some() {
                write!(f, "K")?;
            }
            if self.rights.wq.is_some() {
                write!(f, "Q")?;
            }
            if self.rights.bk.is_some() {
                write!(f, "k")?;
            }
            if self.rights.bq.is_some() {
                write!(f, "q")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
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

    #[test]
    fn square_relative_to() {
        use super::{Colour, Square};

        assert_eq!(Square::A1.relative_to(Colour::White), Square::A1);
        assert_eq!(Square::A1.relative_to(Colour::Black), Square::A8);
        assert_eq!(Square::A8.relative_to(Colour::White), Square::A8);
        assert_eq!(Square::A8.relative_to(Colour::Black), Square::A1);
    }
}
