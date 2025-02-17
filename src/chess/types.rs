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

use super::piece::Col;

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

    pub fn all() -> impl DoubleEndedIterator<Item = Self> {
        // SAFETY: all values are within `0..64`.
        (0..8u8).map(|i| unsafe { std::mem::transmute(i) })
    }

    pub const fn with(self, rank: Rank) -> Square {
        Square::from_rank_file(rank, self)
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

    pub fn all() -> impl DoubleEndedIterator<Item = Self> {
        // SAFETY: all values are within `0..8`.
        (0..8u8).map(|i| unsafe { std::mem::transmute(i) })
    }

    pub const fn with(self, file: File) -> Square {
        Square::from_rank_file(self, file)
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
#[derive(PartialEq, Eq, Clone, Copy, PartialOrd, Ord, Hash, Debug, Default)]
#[repr(u8)]
pub enum Square {
    #[default]
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

    pub fn all() -> impl DoubleEndedIterator<Item = Self> {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct ContHistIndex {
    pub piece: Piece,
    pub square: Square,
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
#[repr(C)]
/// Zobrist keys for a position.
///
/// `repr(C)` because this actually how i want it to be laid out in memory.
pub struct Keys {
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

/// Full state for a chess position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct State {
    /// Which rooks can castle.
    pub castle_perm: CastlingRights,
    /// The en passant square.
    pub ep_square: Option<Square>,
    /// The number of half moves made since the last capture or pawn advance.
    pub fifty_move_counter: u8,
    /// Squares that the opponent attacks
    pub threats: Threats,
    /// The square-sets of all the pieces on the board.
    pub piece_layout: PieceLayout,
    /// An array to accelerate `Board::piece_at()`.
    pub piece_array: [Option<Piece>; 64],
    /// Zobrist hashes.
    pub keys: Keys,
}

impl Default for State {
    fn default() -> Self {
        Self {
            // curse thee array autoimpls
            piece_array: [None; 64],
            castle_perm: CastlingRights::default(),
            ep_square: None,
            fifty_move_counter: Default::default(),
            threats: Threats::default(),
            piece_layout: PieceLayout::default(),
            keys: Keys::default(),
        }
    }
}

pub enum CheckState {
    None,
    Check,
    Checkmate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CastlingRights {
    // packed representation:
    // each file requires 8 values - three bits.
    // three times four is twelve, and then
    // the bottom four bits are used to represent presence.
    // [ 3 | wk ][ 3 | wq ][ 3 | bk ][ 3 | bq ][ 4 | flags ]
    data: u16,
}

#[allow(clippy::unusual_byte_groupings)]
impl CastlingRights {
    pub const WKCA: u16 = 0b0001;
    pub const WQCA: u16 = 0b0010;
    pub const BKCA: u16 = 0b0100;
    pub const BQCA: u16 = 0b1000;
    pub const WK_MASK: u16 = 0b111_000_000_000_0000;
    pub const WQ_MASK: u16 = 0b000_111_000_000_0000;
    pub const BK_MASK: u16 = 0b000_000_111_000_0000;
    pub const BQ_MASK: u16 = 0b000_000_000_111_0000;
    pub const WK_SHIFT: u8 = 4 + 3 + 3 + 3;
    pub const WQ_SHIFT: u8 = 4 + 3 + 3;
    pub const BK_SHIFT: u8 = 4 + 3;
    pub const BQ_SHIFT: u8 = 4;
    pub const KEY_MASK: u16 = 0b1111;

    pub const fn new(
        wk: Option<File>,
        wq: Option<File>,
        bk: Option<File>,
        bq: Option<File>,
    ) -> Self {
        let mut data = 0;

        if let Some(wk) = wk {
            data |= ((wk as u16) << Self::WK_SHIFT) | Self::WKCA;
        }
        if let Some(wq) = wq {
            data |= ((wq as u16) << Self::WQ_SHIFT) | Self::WQCA;
        }
        if let Some(bk) = bk {
            data |= ((bk as u16) << Self::BK_SHIFT) | Self::BKCA;
        }
        if let Some(bq) = bq {
            data |= ((bq as u16) << Self::BQ_SHIFT) | Self::BQCA;
        }

        Self { data }
    }

    pub const fn hashkey_index(self) -> usize {
        (self.data & Self::KEY_MASK) as usize
    }

    pub fn clear<C: Col>(&mut self) {
        self.data &= if C::WHITE {
            !(Self::WK_MASK | Self::WQ_MASK | Self::WKCA | Self::WQCA)
        } else {
            !(Self::BK_MASK | Self::BQ_MASK | Self::BKCA | Self::BQCA)
        };
    }

    pub fn clear_side<const IS_KINGSIDE: bool, C: Col>(&mut self) {
        #![allow(clippy::collapsible_else_if)]
        self.data &= !if C::WHITE {
            if IS_KINGSIDE {
                Self::WK_MASK | Self::WKCA
            } else {
                Self::WQ_MASK | Self::WQCA
            }
        } else {
            if IS_KINGSIDE {
                Self::BK_MASK | Self::BKCA
            } else {
                Self::BQ_MASK | Self::BQCA
            }
        };
    }

    pub fn remove<C: Col>(&mut self, file: File) {
        if self.kingside(C::COLOUR) == Some(file) {
            self.clear_side::<true, C>();
        } else if self.queenside(C::COLOUR) == Some(file) {
            self.clear_side::<false, C>();
        }
    }

    pub fn kingside(self, side: Colour) -> Option<File> {
        #![allow(clippy::cast_possible_truncation)]
        let presence = [Self::WKCA, Self::BKCA][side];
        if self.data & presence == 0 {
            return None;
        }
        let shift = [Self::WK_SHIFT, Self::BK_SHIFT][side];
        let mask = [Self::WK_MASK, Self::BK_MASK][side];
        let value = (self.data & mask) >> shift;
        File::from_index(value as u8)
    }

    pub fn queenside(self, side: Colour) -> Option<File> {
        #![allow(clippy::cast_possible_truncation)]
        let presence = [Self::WQCA, Self::BQCA][side];
        if self.data & presence == 0 {
            return None;
        }
        let shift = [Self::WQ_SHIFT, Self::BQ_SHIFT][side];
        let mask = [Self::WQ_MASK, Self::BQ_MASK][side];
        let value = (self.data & mask) >> shift;
        File::from_index(value as u8)
    }

    pub fn set_kingside(&mut self, side: Colour, file: File) {
        let presence = [Self::WKCA, Self::BKCA][side];
        let shift = [Self::WK_SHIFT, Self::BK_SHIFT][side];
        let mask = [!Self::WK_MASK, !Self::BK_MASK][side];
        let value = file as u16;
        self.data &= mask;
        self.data |= (value << shift) | presence;
    }

    pub fn set_queenside(&mut self, side: Colour, file: File) {
        let presence = [Self::WQCA, Self::BQCA][side];
        let shift = [Self::WQ_SHIFT, Self::BQ_SHIFT][side];
        let mask = [!Self::WQ_MASK, !Self::BQ_MASK][side];
        let value = file as u16;
        self.data &= mask;
        self.data |= (value << shift) | presence;
    }
}

#[cfg(test)]
mod tests {
    use crate::chess::piece::{Black, White};

    use super::*;

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

    #[test]
    fn test_kingside_getters_and_setters() {
        let mut rights = CastlingRights::default();

        // Test white kingside
        assert_eq!(rights.kingside(Colour::White), None);
        rights.set_kingside(Colour::White, File::H);
        assert_eq!(rights.kingside(Colour::White), Some(File::H));

        // Test black kingside
        assert_eq!(rights.kingside(Colour::Black), None);
        rights.set_kingside(Colour::Black, File::H);
        assert_eq!(rights.kingside(Colour::Black), Some(File::H));

        // Test overwriting existing rights
        rights.set_kingside(Colour::White, File::G);
        assert_eq!(rights.kingside(Colour::White), Some(File::G));
    }

    #[test]
    fn test_queenside_getters_and_setters() {
        let mut rights = CastlingRights::default();

        // Test white queenside
        assert_eq!(rights.queenside(Colour::White), None);
        rights.set_queenside(Colour::White, File::A);
        assert_eq!(rights.queenside(Colour::White), Some(File::A));

        // Test black queenside
        assert_eq!(rights.queenside(Colour::Black), None);
        rights.set_queenside(Colour::Black, File::A);
        assert_eq!(rights.queenside(Colour::Black), Some(File::A));

        // Test overwriting existing rights
        rights.set_queenside(Colour::White, File::B);
        assert_eq!(rights.queenside(Colour::White), Some(File::B));
    }

    #[test]
    fn test_clear_rights() {
        let mut rights =
            CastlingRights::new(Some(File::H), Some(File::A), Some(File::H), Some(File::A));

        assert_eq!(rights.kingside(Colour::White), Some(File::H));
        assert_eq!(rights.queenside(Colour::White), Some(File::A));
        assert_eq!(rights.kingside(Colour::Black), Some(File::H));
        assert_eq!(rights.queenside(Colour::Black), Some(File::A));

        // Test clearing white rights
        rights.clear::<White>();
        assert_eq!(rights.kingside(Colour::White), None);
        assert_eq!(rights.queenside(Colour::White), None);
        assert_eq!(rights.kingside(Colour::Black), Some(File::H));
        assert_eq!(rights.queenside(Colour::Black), Some(File::A));

        // Test clearing black rights
        rights.clear::<Black>();
        assert_eq!(rights.kingside(Colour::Black), None);
        assert_eq!(rights.queenside(Colour::Black), None);
    }

    #[test]
    fn test_remove_specific_rights() {
        let mut rights =
            CastlingRights::new(Some(File::H), Some(File::A), Some(File::H), Some(File::A));

        // Remove white kingside by file
        rights.remove::<White>(File::H);
        assert_eq!(rights.kingside(Colour::White), None);
        assert_eq!(rights.queenside(Colour::White), Some(File::A));

        // Remove black queenside by file
        rights.remove::<Black>(File::A);
        assert_eq!(rights.kingside(Colour::Black), Some(File::H));
        assert_eq!(rights.queenside(Colour::Black), None);

        // Removing non-existent rights should have no effect
        rights.remove::<White>(File::G);
        assert_eq!(rights.queenside(Colour::White), Some(File::A));
    }
}
