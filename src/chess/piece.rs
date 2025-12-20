use std::{
    fmt::{Debug, Display},
    mem::size_of,
    ops::{Index, IndexMut, Not},
};

pub trait Col {
    type Opposite: Col;
    const WHITE: bool;
    const COLOUR: Colour;

    const PAWN_LEFT_OFFSET: i8;
    const PAWN_FWD_OFFSET: i8;
    const PAWN_RIGHT_OFFSET: i8;
    const PAWN_DOUBLE_OFFSET: i8;
}

pub struct White;
pub struct Black;

impl Col for White {
    type Opposite = Black;
    const WHITE: bool = true;
    const COLOUR: Colour = Colour::White;

    const PAWN_LEFT_OFFSET: i8 = 7;
    const PAWN_FWD_OFFSET: i8 = 8;
    const PAWN_RIGHT_OFFSET: i8 = 9;
    const PAWN_DOUBLE_OFFSET: i8 = 16;
}

impl Col for Black {
    type Opposite = White;
    const WHITE: bool = false;
    const COLOUR: Colour = Colour::Black;

    const PAWN_LEFT_OFFSET: i8 = -9;
    const PAWN_FWD_OFFSET: i8 = -8;
    const PAWN_RIGHT_OFFSET: i8 = -7;
    const PAWN_DOUBLE_OFFSET: i8 = -16;
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
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
#[repr(u8)]
pub enum Piece {
    #[default]
    WP, BP,
    WN, BN,
    WB, BB,
    WR, BR,
    WQ, BQ,
    WK, BK,
}

const _PIECE_ASSERT: () = assert!(size_of::<Piece>() == size_of::<Option<Piece>>());

impl Display for Piece {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.char())
    }
}

impl Colour {
    pub const fn new(v: bool) -> Self {
        if v { Self::Black } else { Self::White }
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

    pub fn all() -> impl DoubleEndedIterator<Item = Self> {
        [Self::White, Self::Black].into_iter()
    }
}

impl Not for Colour {
    type Output = Self;

    fn not(self) -> Self::Output {
        self.flip()
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

    /// SAFETY: you may only call this function with value of `inner` less than 6.
    pub const unsafe fn from_index_unchecked(v: u8) -> Self {
        debug_assert!(v < 6);
        // Safety: caller's precondition.
        unsafe { std::mem::transmute(v) }
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

    pub fn all() -> impl DoubleEndedIterator<Item = Self> {
        // SAFETY: all values are within `0..6`.
        (0..6u8).map(|i| unsafe { std::mem::transmute(i) })
    }

    pub const fn index(self) -> usize {
        self as usize
    }

    pub fn from_symbol(c: u8) -> Option<Self> {
        const SYMBOLS: [u8; 7] = *b"PNBRQK.";
        SYMBOLS
            .iter()
            .position(|&x| x == c)
            .and_then(|x| Self::new(x.try_into().ok()?))
    }
}

impl Piece {
    pub const fn new(colour: Colour, piece_type: PieceType) -> Self {
        let index = colour as u8 | (piece_type as u8) << 1;
        // SAFETY: Colour is {0, 1}, piece_type is {0, 1, 2, 3, 4, 5}.
        // colour | piece_type << 1 is therefore at most 11, which corresponds
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
        if (self as u8) & 1 == 0 {
            Colour::White
        } else {
            Colour::Black
        }
    }

    pub const fn piece_type(self) -> PieceType {
        let pt_index = self as u8 >> 1;
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

    pub fn byte_char(self) -> u8 {
        b"PpNnBbRrQqKk"[self]
    }

    pub fn all() -> impl DoubleEndedIterator<Item = Self> {
        // SAFETY: all values are within `0..12`.
        (0..12u8).map(|i| unsafe { std::mem::transmute(i) })
    }

    #[allow(dead_code)]
    pub const fn inner(self) -> u8 {
        self as u8
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn piece_construction_and_decomposition() {
        // Test that we can construct all pieces and decompose them correctly
        for colour in Colour::all() {
            for piece_type in PieceType::all() {
                let piece = Piece::new(colour, piece_type);
                assert_eq!(
                    piece.colour(),
                    colour,
                    "Colour mismatch for {colour:?} {piece_type:?}"
                );
                assert_eq!(
                    piece.piece_type(),
                    piece_type,
                    "PieceType mismatch for {colour:?} {piece_type:?}"
                );
            }
        }
    }

    #[test]
    fn piece_round_trip_construction() {
        // Test that decomposing and reconstructing gives the same piece
        for piece in Piece::all() {
            let reconstructed = Piece::new(piece.colour(), piece.piece_type());
            assert_eq!(piece, reconstructed, "Round-trip failed for {piece:?}");
        }
    }

    #[test]
    fn piece_from_index() {
        // Test that all valid indices produce Some(piece)
        for i in 0..12 {
            assert!(
                Piece::from_index(i).is_some(),
                "from_index({i}) should be Some"
            );
        }

        // Test that invalid indices produce None
        for i in 12..=255 {
            assert!(
                Piece::from_index(i).is_none(),
                "from_index({i}) should be None"
            );
        }
    }

    #[test]
    fn piece_index_round_trip() {
        // Test that inner() and from_index() are inverses
        for piece in Piece::all() {
            let index = piece.inner();
            assert!(
                index < 12,
                "inner() returned out-of-range value {index} for {piece:?}"
            );
            assert_eq!(
                Piece::from_index(index),
                Some(piece),
                "Round-trip failed for {piece:?}"
            );
        }
    }

    #[test]
    fn piece_char_case_convention() {
        // White pieces should be uppercase, black pieces lowercase
        for piece in Piece::all() {
            let c = piece.char();
            if piece.colour() == Colour::White {
                assert!(c.is_uppercase(), "{piece:?} should have uppercase char");
            } else {
                assert!(c.is_lowercase(), "{piece:?} should have lowercase char");
            }
        }
    }

    #[test]
    fn piece_all_iterator() {
        let pieces: Vec<_> = Piece::all().collect();
        assert_eq!(pieces.len(), 12, "all() should return 12 pieces");

        // Verify all pieces are unique
        for i in 0..pieces.len() {
            for j in (i + 1)..pieces.len() {
                assert_ne!(
                    pieces[i], pieces[j],
                    "Duplicate piece found at indices {i} and {j}"
                );
            }
        }

        // Verify the order matches the enum definition
        assert_eq!(pieces[0], Piece::WP);
        assert_eq!(pieces[1], Piece::BP);
        assert_eq!(pieces[2], Piece::WN);
        assert_eq!(pieces[3], Piece::BN);
        assert_eq!(pieces[4], Piece::WB);
        assert_eq!(pieces[5], Piece::BB);
        assert_eq!(pieces[6], Piece::WR);
        assert_eq!(pieces[7], Piece::BR);
        assert_eq!(pieces[8], Piece::WQ);
        assert_eq!(pieces[9], Piece::BQ);
        assert_eq!(pieces[10], Piece::WK);
        assert_eq!(pieces[11], Piece::BK);
    }

    #[test]
    fn piece_array_indexing() {
        // Test that we can use Piece to index into arrays
        let mut arr = [0; 12];

        for piece in Piece::all() {
            arr[piece] = i32::from(piece.inner());
        }

        for piece in Piece::all() {
            assert_eq!(arr[piece], i32::from(piece.inner()));
        }
    }

    #[test]
    fn specific_piece_constructions() {
        // Test specific known pieces
        assert_eq!(Piece::new(Colour::White, PieceType::Pawn), Piece::WP);
        assert_eq!(Piece::new(Colour::Black, PieceType::Pawn), Piece::BP);
        assert_eq!(Piece::new(Colour::White, PieceType::Knight), Piece::WN);
        assert_eq!(Piece::new(Colour::Black, PieceType::Knight), Piece::BN);
        assert_eq!(Piece::new(Colour::White, PieceType::Bishop), Piece::WB);
        assert_eq!(Piece::new(Colour::Black, PieceType::Bishop), Piece::BB);
        assert_eq!(Piece::new(Colour::White, PieceType::Rook), Piece::WR);
        assert_eq!(Piece::new(Colour::Black, PieceType::Rook), Piece::BR);
        assert_eq!(Piece::new(Colour::White, PieceType::Queen), Piece::WQ);
        assert_eq!(Piece::new(Colour::Black, PieceType::Queen), Piece::BQ);
        assert_eq!(Piece::new(Colour::White, PieceType::King), Piece::WK);
        assert_eq!(Piece::new(Colour::Black, PieceType::King), Piece::BK);
    }
}
