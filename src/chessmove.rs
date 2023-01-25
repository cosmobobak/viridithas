use std::fmt::{Debug, Display, Formatter};

use crate::{definitions::Square, piece::PieceType};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Move {
    pub data: u16,
}

impl Move {
    const SQ_MASK: u16 = 0b11_1111;
    const TO_SHIFT: usize = 6;
    const PROMO_MASK: u16 = 0b11;
    const PROMO_SHIFT: usize = 12;
    const PROMO_FLAG: u16 = 0b1100_0000_0000_0000;
    pub const EP_FLAG: u16 = 0b0100_0000_0000_0000;
    pub const CASTLE_FLAG: u16 = 0b1000_0000_0000_0000;
    pub const NULL: Self = Self { data: 0 };

    pub fn new_with_promo(from: Square, to: Square, promotion: PieceType) -> Self {
        debug_assert!(u16::from(from) & Self::SQ_MASK == u16::from(from));
        debug_assert!(u16::from(to) & Self::SQ_MASK == u16::from(to));
        debug_assert_ne!(
            promotion,
            PieceType::NO_PIECE_TYPE,
            "attempted to construct promotion to NO_PIECE_TYPE"
        );
        debug_assert_ne!(promotion, PieceType::PAWN, "attempted to construct promotion to pawn");
        debug_assert_ne!(promotion, PieceType::KING, "attempted to construct promotion to king");
        let promotion = u16::from(promotion.inner()).wrapping_sub(2) & Self::PROMO_MASK; // can't promote to NO_PIECE or PAWN
        Self {
            data: u16::from(from)
                | (u16::from(to) << Self::TO_SHIFT)
                | (promotion << Self::PROMO_SHIFT)
                | Self::PROMO_FLAG,
        }
    }

    pub fn new_with_flags(from: Square, to: Square, flags: u16) -> Self {
        debug_assert_ne!(
            flags & Self::PROMO_FLAG,
            Self::PROMO_FLAG,
            "promotion flag set without piece type"
        );
        debug_assert_ne!(flags, 0, "attempted to construct move with no flags");
        debug_assert!(u16::from(from) & Self::SQ_MASK == u16::from(from));
        debug_assert!(u16::from(to) & Self::SQ_MASK == u16::from(to));
        Self { data: u16::from(from) | (u16::from(to) << Self::TO_SHIFT) | flags }
    }

    pub fn new(from: Square, to: Square) -> Self {
        debug_assert!(u16::from(from) & Self::SQ_MASK == u16::from(from));
        debug_assert!(u16::from(to) & Self::SQ_MASK == u16::from(to));
        Self { data: u16::from(from) | (u16::from(to) << Self::TO_SHIFT) }
    }

    pub const fn from(self) -> Square {
        Square::new((self.data & Self::SQ_MASK) as u8)
    }

    pub const fn to(self) -> Square {
        Square::new(((self.data >> Self::TO_SHIFT) & Self::SQ_MASK) as u8)
    }

    pub fn promotion_type(self) -> PieceType {
        debug_assert!(self.is_promo());
        let output =
            PieceType::new(((self.data >> Self::PROMO_SHIFT) & Self::PROMO_MASK) as u8 + 2);
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

    pub fn is_valid(self) -> bool {
        let promotion = self.safe_promotion_type();
        if promotion != PieceType::NO_PIECE_TYPE && !self.is_promo() {
            // promotion type is set but not a promotion move
            return false;
        }
        promotion == PieceType::NO_PIECE_TYPE || promotion.legal_promo()
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
        let m = Move::new(Square::A1, Square::B2);
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
        let m = Move::new_with_promo(Square::A7, Square::A8, PieceType::QUEEN);
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
        use super::*;
        use crate::board::movegen::bitboards::BB_ALL;
        use crate::board::movegen::BitLoop;
        for from in BitLoop::new(BB_ALL) {
            for to in BitLoop::new(BB_ALL) {
                let m = Move::new(from, to);
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
