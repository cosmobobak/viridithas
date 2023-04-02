use std::fmt::{Debug, Display};

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Colour {
    v: u8,
}

impl Debug for Colour {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::WHITE => write!(f, "Colour::WHITE"),
            Self::BLACK => write!(f, "Colour::BLACK"),
            _ => write!(f, "Colour::INVALID({})", self.v),
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
            0 => write!(f, "PieceType::NO_PIECE_TYPE"),
            1 => write!(f, "PieceType::PAWN"),
            2 => write!(f, "PieceType::KNIGHT"),
            3 => write!(f, "PieceType::BISHOP"),
            4 => write!(f, "PieceType::ROOK"),
            5 => write!(f, "PieceType::QUEEN"),
            6 => write!(f, "PieceType::KING"),
            _ => write!(f, "PieceType::INVALID({})", self.v),
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
            0 => write!(f, "Piece::EMPTY"),
            1 => write!(f, "Piece::WHITE_PAWN"),
            2 => write!(f, "Piece::WHITE_KNIGHT"),
            3 => write!(f, "Piece::WHITE_BISHOP"),
            4 => write!(f, "Piece::WHITE_ROOK"),
            5 => write!(f, "Piece::WHITE_QUEEN"),
            6 => write!(f, "Piece::WHITE_KING"),
            7 => write!(f, "Piece::BLACK_PAWN"),
            8 => write!(f, "Piece::BLACK_KNIGHT"),
            9 => write!(f, "Piece::BLACK_BISHOP"),
            10 => write!(f, "Piece::BLACK_ROOK"),
            11 => write!(f, "Piece::BLACK_QUEEN"),
            12 => write!(f, "Piece::BLACK_KING"),
            _ => write!(f, "Piece::INVALID({})", self.v),
        }
    }
}

impl Display for Piece {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.v {
            0 => write!(f, "."),
            1..=12 => write!(f, "{}", self.char().unwrap()),
            _ => write!(f, "?"),
        }
    }
}

impl Colour {
    pub const WHITE: Self = Self { v: 0 };
    pub const BLACK: Self = Self { v: 1 };

    pub const fn new(v: u8) -> Self {
        debug_assert!(v < 2);
        Self { v }
    }

    pub const fn flip(self) -> Self {
        Self::new(self.v ^ 1)
    }

    pub const fn index(self) -> usize {
        self.v as usize
    }

    pub const fn inner(self) -> u8 {
        self.v
    }
}

impl PieceType {
    pub const NO_PIECE_TYPE: Self = Self { v: 0 };
    pub const PAWN: Self = Self { v: 1 };
    pub const KNIGHT: Self = Self { v: 2 };
    pub const BISHOP: Self = Self { v: 3 };
    pub const ROOK: Self = Self { v: 4 };
    pub const QUEEN: Self = Self { v: 5 };
    pub const KING: Self = Self { v: 6 };

    pub const fn new(v: u8) -> Self {
        debug_assert!(v < 7);
        Self { v }
    }

    pub const fn inner(self) -> u8 {
        self.v
    }

    pub const fn legal_promo(self) -> bool {
        self.v == Self::QUEEN.v
            || self.v == Self::KNIGHT.v
            || self.v == Self::BISHOP.v
            || self.v == Self::ROOK.v
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
        const SYMBOLS: [u8; 7] = *b".PNBRQK";
        SYMBOLS.iter().position(|&x| x == c).and_then(|x| Some(Self::new(x.try_into().ok()?)))
    }
}

#[allow(clippy::module_name_repetitions)]
pub struct PieceTypesIterator {
    v: u8,
}
impl PieceTypesIterator {
    const fn new() -> Self {
        Self { v: 1 }
    }
}

impl Iterator for PieceTypesIterator {
    type Item = PieceType;

    fn next(&mut self) -> Option<Self::Item> {
        if self.v == 7 {
            None
        } else {
            let v = self.v;
            self.v += 1;
            Some(PieceType::new(v))
        }
    }
}

impl Piece {
    pub const EMPTY: Self = Self { v: 0 };

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

    pub const fn new(colour: Colour, piece_type: PieceType) -> Self {
        debug_assert!(colour.v < 2);
        debug_assert!(piece_type.v < 7);
        Self { v: colour.v * 6 + piece_type.v }
    }

    pub const fn colour(self) -> Colour {
        match self {
            Self::WP | Self::WN | Self::WB | Self::WR | Self::WQ | Self::WK => Colour::WHITE,
            _ => Colour::BLACK,
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
            _ => PieceType::NO_PIECE_TYPE,
        }
    }

    pub const fn index(self) -> usize {
        self.v as usize
    }

    pub const fn hist_table_offset(self) -> usize {
        debug_assert!(!self.is_empty());
        self.v as usize - 1
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

    pub const fn all() -> PiecesIterator {
        PiecesIterator::new()
    }
}

pub struct PiecesIterator {
    v: u8,
}
impl PiecesIterator {
    const fn new() -> Self {
        Self { v: 1 }
    }
}

impl Iterator for PiecesIterator {
    type Item = Piece;

    fn next(&mut self) -> Option<Self::Item> {
        if self.v == 13 {
            None
        } else {
            let v = self.v;
            self.v += 1;
            Some(Piece { v })
        }
    }
}
