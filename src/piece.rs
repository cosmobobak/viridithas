use std::{
    fmt::{Debug, Display},
    mem::size_of,
    ops::{Index, IndexMut},
};

pub trait Col {
    type Opposite: Col;
    const WHITE: bool;
    const COLOUR: Colour;
}

pub struct White;
pub struct Black;

impl Col for White {
    type Opposite = Black;
    const WHITE: bool = true;
    const COLOUR: Colour = Colour::White;
}

impl Col for Black {
    type Opposite = White;
    const WHITE: bool = false;
    const COLOUR: Colour = Colour::Black;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Colour {
    White,
    Black,
}

const _COLOUR_ASSERT: () = assert!(size_of::<Colour>() == size_of::<Option<Colour>>());

impl Display for Colour {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::White => write!(f, "White"),
            Self::Black => write!(f, "Black"),
        }
    }
}

#[allow(clippy::module_name_repetitions)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[repr(u8)]
pub enum PieceType {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

const _PIECE_TYPE_ASSERT: () = assert!(size_of::<PieceType>() == size_of::<Option<PieceType>>());

impl Display for PieceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pawn => write!(f, "Pawn"),
            Self::Knight => write!(f, "Knight"),
            Self::Bishop => write!(f, "Bishop"),
            Self::Rook => write!(f, "Rook"),
            Self::Queen => write!(f, "Queen"),
            Self::King => write!(f, "King"),
        }
    }
}

#[rustfmt::skip]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[repr(u8)]
pub enum Piece {
    WP, WN, WB, WR, WQ, WK,
    BP, BN, BB, BR, BQ, BK,
}

const _PIECE_ASSERT: () = assert!(size_of::<Piece>() == size_of::<Option<Piece>>());

impl Display for Piece {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.char())
    }
}

impl Colour {
    pub const fn new(v: bool) -> Self {
        if v {
            Self::Black
        } else {
            Self::White
        }
    }

    pub const fn flip(self) -> Self {
        match self {
            Self::White => Self::Black,
            Self::Black => Self::White,
        }
    }

    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn inner(self) -> u8 {
        self as u8
    }

    pub fn all() -> impl Iterator<Item = Self> {
        [Self::White, Self::Black].iter().copied()
    }
}

impl PieceType {
    pub const fn new(v: u8) -> Option<Self> {
        if v < 6 {
            // SAFETY: inner is less than 6, so it corresponds to a valid enum variant.
            Some(unsafe { std::mem::transmute::<u8, Self>(v) })
        } else {
            None
        }
    }

    pub const unsafe fn from_index_unchecked(v: u8) -> Self {
        std::mem::transmute(v)
    }

    pub const fn inner(self) -> u8 {
        self as u8
    }

    pub const fn legal_promo(self) -> bool {
        // self == Self::QUEEN || self == Self::KNIGHT || self == Self::BISHOP || self == Self::ROOK
        matches!(self, Self::Queen | Self::Knight | Self::Bishop | Self::Rook)
    }

    pub const fn promo_char(self) -> Option<char> {
        match self {
            Self::Queen => Some('q'),
            Self::Knight => Some('n'),
            Self::Bishop => Some('b'),
            Self::Rook => Some('r'),
            _ => None,
        }
    }

    pub const fn all() -> PieceTypesIterator {
        PieceTypesIterator::new()
    }

    pub const fn index(self) -> usize {
        self as usize
    }

    pub fn from_symbol(c: u8) -> Option<Self> {
        const SYMBOLS: [u8; 7] = *b"PNBRQK.";
        SYMBOLS.iter().position(|&x| x == c).and_then(|x| Self::new(x.try_into().ok()?))
    }

    pub fn see_value(self) -> i32 {
        const SEE_PIECE_VALUES: [i32; 6] = [161, 445, 463, 704, 1321, 0];
        SEE_PIECE_VALUES[self]
    }
}

#[allow(clippy::module_name_repetitions)]
pub struct PieceTypesIterator {
    v: u8,
}
impl PieceTypesIterator {
    const fn new() -> Self {
        Self { v: 0 }
    }
}

impl Iterator for PieceTypesIterator {
    type Item = PieceType;

    fn next(&mut self) -> Option<Self::Item> {
        let v = self.v;
        self.v += 1;
        PieceType::new(v)
    }
}

impl Piece {
    pub const fn new(colour: Colour, piece_type: PieceType) -> Self {
        let index = colour as u8 * 6 + piece_type as u8;
        // SAFETY: Colour is {0, 1}, piece_type is {0, 1, 2, 3, 4, 5}.
        // colour * 6 + piece_type is therefore at most 11, which corresponds
        // to a valid enum variant.
        unsafe { std::mem::transmute(index) }
    }

    pub const fn from_index(v: u8) -> Option<Self> {
        if v < 12 {
            // SAFETY: inner is less than 12, so it corresponds to a valid enum variant.
            Some(unsafe { std::mem::transmute::<u8, Self>(v) })
        } else {
            None
        }
    }

    pub const fn colour(self) -> Colour {
        if (self as u8) < 6 {
            Colour::White
        } else {
            Colour::Black
        }
    }

    pub const fn piece_type(self) -> PieceType {
        let pt_index = self as u8 % 6;
        // SAFETY: pt_index is always within the bounds of the type.
        unsafe { PieceType::from_index_unchecked(pt_index) }
    }

    pub const fn char(self) -> char {
        match self {
            Self::WP => 'P',
            Self::WN => 'N',
            Self::WB => 'B',
            Self::WR => 'R',
            Self::WQ => 'Q',
            Self::WK => 'K',
            Self::BP => 'p',
            Self::BN => 'n',
            Self::BB => 'b',
            Self::BR => 'r',
            Self::BQ => 'q',
            Self::BK => 'k',
        }
    }

    pub const fn byte_char(self) -> u8 {
        match self {
            Self::WP => b'P',
            Self::WN => b'N',
            Self::WB => b'B',
            Self::WR => b'R',
            Self::WQ => b'Q',
            Self::WK => b'K',
            Self::BP => b'p',
            Self::BN => b'n',
            Self::BB => b'b',
            Self::BR => b'r',
            Self::BQ => b'q',
            Self::BK => b'k',
        }
    }

    #[allow(dead_code)]
    pub const fn all() -> PiecesIterator {
        PiecesIterator::new()
    }

    #[allow(dead_code)]
    pub const fn inner(self) -> u8 {
        self as u8
    }
}

pub struct PiecesIterator {
    v: u8,
}

impl PiecesIterator {
    const fn new() -> Self {
        Self { v: 0 }
    }
}

impl Iterator for PiecesIterator {
    type Item = Piece;

    fn next(&mut self) -> Option<Self::Item> {
        let v = self.v;
        self.v += 1;
        Piece::from_index(v)
    }
}

impl<T> Index<Colour> for [T; 2] {
    type Output = T;

    fn index(&self, index: Colour) -> &Self::Output {
        // SAFETY: the legal values for this type are all in bounds.
        unsafe { self.get_unchecked(index as usize) }
    }
}

impl<T> IndexMut<Colour> for [T; 2] {
    fn index_mut(&mut self, index: Colour) -> &mut Self::Output {
        // SAFETY: the legal values for this type are all in bounds.
        unsafe { self.get_unchecked_mut(index as usize) }
    }
}

impl<T> Index<PieceType> for [T; 6] {
    type Output = T;

    fn index(&self, index: PieceType) -> &Self::Output {
        // SAFETY: the legal values for this type are all in bounds.
        unsafe { self.get_unchecked(index as usize) }
    }
}

impl<T> IndexMut<PieceType> for [T; 6] {
    fn index_mut(&mut self, index: PieceType) -> &mut Self::Output {
        // SAFETY: the legal values for this type are all in bounds.
        unsafe { self.get_unchecked_mut(index as usize) }
    }
}

impl<T> Index<Piece> for [T; 12] {
    type Output = T;

    fn index(&self, index: Piece) -> &Self::Output {
        // SAFETY: the legal values for this type are all in bounds.
        unsafe { self.get_unchecked(index as usize) }
    }
}

impl<T> IndexMut<Piece> for [T; 12] {
    fn index_mut(&mut self, index: Piece) -> &mut Self::Output {
        // SAFETY: the legal values for this type are all in bounds.
        unsafe { self.get_unchecked_mut(index as usize) }
    }
}
