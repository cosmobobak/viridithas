use std::fmt::{Debug, Display, Formatter};

use crate::{
    definitions::Square, piece::PieceType,
};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Move {
    pub data: u16,
}

impl Move {
    const FROM_MASK: u16 = 0b0000_0000_0011_1111;
    const TO_MASK: u16 = 0b0000_1111_1100_0000;
    const PROMO_MASK: u16 = 0b0011_0000_0000_0000;
    pub const EP_FLAG: u16 = 0b0100_0000_0000_0000;
    pub const CASTLE_FLAG: u16 = 0b1000_0000_0000_0000;
    pub const PROMO_FLAG: u16 = 0b1100_0000_0000_0000;
    pub const NULL: Self = Self { data: 0 };

    pub fn new(from: Square, to: Square, promotion: PieceType, flags: u16) -> Self {
        debug_assert!((flags & (Self::EP_FLAG | Self::CASTLE_FLAG | Self::PROMO_FLAG)) == flags);
        debug_assert!(u16::from(from) & 0b11_1111 == u16::from(from));
        debug_assert!(u16::from(to) & 0b11_1111 == u16::from(to));
        debug_assert!(
            promotion == PieceType::NO_PIECE_TYPE && flags != Self::PROMO_FLAG
                || promotion.legal_promo() && flags == Self::PROMO_FLAG
        );
        let promotion = promotion.inner().wrapping_sub(2) & 0b11; // can't promote to NO_PIECE or PAWN
        Self { data: u16::from(from) | (u16::from(to) << 6) | (u16::from(promotion) << 12) | flags }
    }

    pub const fn from(self) -> Square {
        Square::new((self.data & Self::FROM_MASK) as u8)
    }

    pub const fn to(self) -> Square {
        Square::new((((self.data & Self::TO_MASK) >> 6) & 0b11_1111) as u8)
    }

    pub fn promotion_type(self) -> PieceType {
        debug_assert!(self.is_promo());
        let output = PieceType::new((((self.data & Self::PROMO_MASK) >> 12) & 0b11) as u8 + 2);
        debug_assert!(output.legal_promo());
        output
    }

    pub fn safe_promotion_type(self) -> PieceType {
        if self.is_promo() {
            self.promotion_type()
        } else {
            PieceType::NO_PIECE_TYPE
        }
    }

    pub const fn is_promo(self) -> bool {
        (self.data & Self::PROMO_FLAG) == Self::PROMO_FLAG
    }

    pub const fn is_ep(self) -> bool {
        (self.data & Self::EP_FLAG) != 0 && self.data & Self::CASTLE_FLAG == 0
    }

    pub const fn is_castle(self) -> bool {
        (self.data & Self::CASTLE_FLAG) != 0 && self.data & Self::EP_FLAG == 0
    }

    pub const fn is_null(self) -> bool {
        self.data == 0
    }

    pub const fn is_kingside_castling(self) -> bool {
        self.is_castle() && matches!(self.to(), Square::G1 | Square::G8)
    }

    pub const fn is_queenside_castling(self) -> bool {
        self.is_castle() && matches!(self.to(), Square::C1 | Square::C8)
    }

    #[allow(dead_code)]
    pub const fn bitboard(self) -> u64 {
        self.from().bitboard() | self.to().bitboard()
    }

    pub fn is_valid(self) -> bool {
        let promotion = self.safe_promotion_type();
        if promotion != PieceType::NO_PIECE_TYPE && !self.is_promo() {
            // promotion type is set but not a promotion move
            return false;
        }
        promotion == PieceType::NO_PIECE_TYPE
            || promotion.legal_promo()
    }
}

impl Display for Move {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        if self.is_null() {
            return write!(f, "null");
        }

        if self.is_promo() {
            let pchar = self.promotion_type().promo_char().unwrap_or('?');
            write!(f, "{}{}{pchar}", self.from(), self.to())?;
        } else {
            write!(f, "{}{}", self.from(), self.to())?;
        }

        Ok(())
    }
}

impl Debug for Move {
    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
        write!(
            f,
            "move from {} ({:?}) to {} ({:?}), promo {}, ispromo {}, ep {}, castle {}",
            self.from(),
            self.from(),
            self.to(),
            self.to(),
            self.safe_promotion_type().promo_char().unwrap_or('X'),
            self.is_promo(),
            self.is_ep(),
            self.is_castle()
        )
    }
}

mod tests {
    #[test]
    fn test_simple_move() {
        use super::*;
        let m = Move::new(Square::A1, Square::B2, PieceType::NO_PIECE_TYPE, 0);
        println!("{m:?}");
        println!("bitpattern: {:016b}", m.data);
        assert_eq!(m.from(), Square::A1);
        assert_eq!(m.to(), Square::B2);
        assert!(!m.is_ep());
        assert!(!m.is_castle());
        assert!(!m.is_null());
        assert!(!m.is_promo());
        assert!(m.is_valid());
    }

    #[test]
    fn test_promotion() {
        use super::*;
        let m = Move::new(Square::A7, Square::A8, PieceType::QUEEN, Move::PROMO_FLAG);
        println!("{m:?}");
        println!("bitpattern: {:016b}", m.data);
        assert_eq!(m.from(), Square::A7);
        assert_eq!(m.to(), Square::A8);
        assert!(m.is_promo());
        assert!(!m.is_ep());
        assert!(!m.is_castle());
        assert!(!m.is_null());
        assert_eq!(m.promotion_type(), PieceType::QUEEN);
        assert!(m.is_valid());
    }

    #[test]
    fn test_all_square_combinations() {
        use crate::board::movegen::BitLoop;
        use crate::board::movegen::bitboards::BB_ALL;
        use super::*;
        for from in BitLoop::new(BB_ALL) {
            for to in BitLoop::new(BB_ALL) {
                let m = Move::new(from, to, PieceType::NO_PIECE_TYPE, 0);
                assert_eq!(m.from(), from);
                assert_eq!(m.to(), to);
                assert!(!m.is_promo());
                assert!(!m.is_ep());
                assert!(!m.is_castle());
                assert!(m.is_valid());
            }
        }
    }
}
