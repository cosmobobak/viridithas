use std::fmt::{Debug, Display};

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
    const COLOUR: Colour = Colour::WHITE;
}

impl Col for Black {
    type Opposite = White;
    const WHITE: bool = false;
    const COLOUR: Colour = Colour::BLACK;
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Colour {
    v: bool,
}

impl Debug for Colour {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::WHITE => write!(f, "Colour::WHITE"),
            Self::BLACK => write!(f, "Colour::BLACK"),
        }
    }
}

impl Display for Colour {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::WHITE => write!(f, "White"),
            Self::BLACK => write!(f, "Black"),
        }
    }
}

#[allow(clippy::module_name_repetitions)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct PieceType {
    v: u8,
}

impl Debug for PieceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.v {
            0 => write!(f, "PieceType::PAWN"),
            1 => write!(f, "PieceType::KNIGHT"),
            2 => write!(f, "PieceType::BISHOP"),
            3 => write!(f, "PieceType::ROOK"),
            4 => write!(f, "PieceType::QUEEN"),
            5 => write!(f, "PieceType::KING"),
            6 => write!(f, "PieceType::NO_PIECE_TYPE"),
            _ => write!(f, "PieceType::INVALID({})", self.v),
        }
    }
}

impl Display for PieceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.v {
            0 => write!(f, "Pawn"),
            1 => write!(f, "Knight"),
            2 => write!(f, "Bishop"),
            3 => write!(f, "Rook"),
            4 => write!(f, "Queen"),
            5 => write!(f, "King"),
            6 => write!(f, "NoPieceType"),
            _ => write!(f, "?"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Piece {
    v: u8,
}

impl Debug for Piece {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.v {
            0 => write!(f, "Piece::WHITE_PAWN"),
            1 => write!(f, "Piece::WHITE_KNIGHT"),
            2 => write!(f, "Piece::WHITE_BISHOP"),
            3 => write!(f, "Piece::WHITE_ROOK"),
            4 => write!(f, "Piece::WHITE_QUEEN"),
            5 => write!(f, "Piece::WHITE_KING"),
            6 => write!(f, "Piece::BLACK_PAWN"),
            7 => write!(f, "Piece::BLACK_KNIGHT"),
            8 => write!(f, "Piece::BLACK_BISHOP"),
            9 => write!(f, "Piece::BLACK_ROOK"),
            10 => write!(f, "Piece::BLACK_QUEEN"),
            11 => write!(f, "Piece::BLACK_KING"),
            12 => write!(f, "Piece::EMPTY"),
            _ => write!(f, "Piece::INVALID({})", self.v),
        }
    }
}

impl Display for Piece {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.v {
            0..=11 => write!(f, "{}", self.char().unwrap()),
            12 => write!(f, "."),
            _ => write!(f, "?"),
        }
    }
}

impl Colour {
    pub const WHITE: Self = Self { v: false };
    pub const BLACK: Self = Self { v: true };

    pub const fn new(v: bool) -> Self {
        Self { v }
    }

    pub const fn flip(self) -> Self {
        Self::new(!self.v)
    }

    pub const fn index(self) -> usize {
        self.v as usize
    }

    pub const fn inner(self) -> bool {
        self.v
    }

    pub fn all() -> impl Iterator<Item = Self> {
        [Self::WHITE, Self::BLACK].iter().copied()
    }
}

impl PieceType {
    pub const PAWN: Self = Self { v: 0 };
    pub const KNIGHT: Self = Self { v: 1 };
    pub const BISHOP: Self = Self { v: 2 };
    pub const ROOK: Self = Self { v: 3 };
    pub const QUEEN: Self = Self { v: 4 };
    pub const KING: Self = Self { v: 5 };
    pub const NONE: Self = Self { v: 6 };

    pub const fn new(v: u8) -> Self {
        debug_assert!(v < 7);
        Self { v }
    }

    pub const fn inner(self) -> u8 {
        self.v
    }

    pub const fn legal_promo(self) -> bool {
        self.v == Self::QUEEN.v || self.v == Self::KNIGHT.v || self.v == Self::BISHOP.v || self.v == Self::ROOK.v
    }

    pub const fn promo_char(self) -> Option<char> {
        match self {
            Self::QUEEN => Some('q'),
            Self::KNIGHT => Some('n'),
            Self::BISHOP => Some('b'),
            Self::ROOK => Some('r'),
            _ => None,
        }
    }

    pub const fn all() -> PieceTypesIterator {
        PieceTypesIterator::new()
    }

    pub const fn index(self) -> usize {
        self.v as usize
    }

    pub fn from_symbol(c: u8) -> Option<Self> {
        const SYMBOLS: [u8; 7] = *b"PNBRQK.";
        SYMBOLS.iter().position(|&x| x == c).and_then(|x| Some(Self::new(x.try_into().ok()?)))
    }

    pub const fn see_value(self) -> i32 {
        const SEE_PIECE_VALUES: [i32; 7] = [161, 445, 463, 704, 1321, 0, 0];
        SEE_PIECE_VALUES[self.index()]
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
        if self.v == 6 {
            None
        } else {
            let v = self.v;
            self.v += 1;
            Some(PieceType::new(v))
        }
    }
}

impl Piece {
    pub const WP: Self = Self::new(Colour::WHITE, PieceType::PAWN);
    pub const WN: Self = Self::new(Colour::WHITE, PieceType::KNIGHT);
    pub const WB: Self = Self::new(Colour::WHITE, PieceType::BISHOP);
    pub const WR: Self = Self::new(Colour::WHITE, PieceType::ROOK);
    pub const WQ: Self = Self::new(Colour::WHITE, PieceType::QUEEN);
    pub const WK: Self = Self::new(Colour::WHITE, PieceType::KING);

    pub const BP: Self = Self::new(Colour::BLACK, PieceType::PAWN);
    pub const BN: Self = Self::new(Colour::BLACK, PieceType::KNIGHT);
    pub const BB: Self = Self::new(Colour::BLACK, PieceType::BISHOP);
    pub const BR: Self = Self::new(Colour::BLACK, PieceType::ROOK);
    pub const BQ: Self = Self::new(Colour::BLACK, PieceType::QUEEN);
    pub const BK: Self = Self::new(Colour::BLACK, PieceType::KING);

    pub const EMPTY: Self = Self { v: 12 };

    pub const fn new(colour: Colour, piece_type: PieceType) -> Self {
        debug_assert!(piece_type.v < 7);
        Self { v: colour.v as u8 * 6 + piece_type.v }
    }

    pub const fn colour(self) -> Colour {
        if self.v < 6 {
            Colour::WHITE
        } else {
            Colour::BLACK
        }
    }

    pub const fn piece_type(self) -> PieceType {
        match self {
            Self::WP | Self::BP => PieceType::PAWN,
            Self::WN | Self::BN => PieceType::KNIGHT,
            Self::WB | Self::BB => PieceType::BISHOP,
            Self::WR | Self::BR => PieceType::ROOK,
            Self::WQ | Self::BQ => PieceType::QUEEN,
            Self::WK | Self::BK => PieceType::KING,
            _ => PieceType::NONE,
        }
    }

    pub const fn index(self) -> usize {
        self.v as usize
    }

    pub const fn hist_table_offset(self) -> usize {
        debug_assert!(!self.is_empty());
        self.v as usize
    }

    pub const fn char(self) -> Option<char> {
        match self {
            Self::WP => Some('P'),
            Self::WN => Some('N'),
            Self::WB => Some('B'),
            Self::WR => Some('R'),
            Self::WQ => Some('Q'),
            Self::WK => Some('K'),
            Self::BP => Some('p'),
            Self::BN => Some('n'),
            Self::BB => Some('b'),
            Self::BR => Some('r'),
            Self::BQ => Some('q'),
            Self::BK => Some('k'),
            _ => None,
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
            _ => b'.',
        }
    }

    pub const fn is_empty(self) -> bool {
        self.v == Self::EMPTY.v
    }

    #[allow(dead_code)]
    pub const fn all() -> PiecesIterator {
        PiecesIterator::new()
    }

    #[allow(dead_code)]
    pub const fn inner(self) -> u8 {
        self.v
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
        if self.v == 12 {
            None
        } else {
            let v = self.v;
            self.v += 1;
            Some(Piece { v })
        }
    }
}
