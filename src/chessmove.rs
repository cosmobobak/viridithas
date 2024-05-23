use std::{
    fmt::{Debug, Display, Formatter},
    num::NonZeroU16,
    sync::atomic::Ordering,
};

use crate::{
    piece::PieceType,
    uci::CHESS960,
    util::{File, Square},
};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Move {
    data: NonZeroU16,
}

const _: () = assert!(std::mem::size_of::<Move>() == std::mem::size_of::<Option<Move>>());

const PROMO_FLAG_BITS: u16 = 0b1100_0000_0000_0000;
const EP_FLAG_BITS: u16 = 0b0100_0000_0000_0000;
const CASTLE_FLAG_BITS: u16 = 0b1000_0000_0000_0000;

#[repr(u16)]
#[derive(PartialEq, Eq, Debug)]
pub enum MoveFlags {
    Promotion = PROMO_FLAG_BITS,
    EnPassant = EP_FLAG_BITS,
    Castle = CASTLE_FLAG_BITS,
}

impl Move {
    const SQ_MASK: u16 = 0b11_1111;
    const TO_SHIFT: usize = 6;
    const PROMO_MASK: u16 = 0b11;
    const PROMO_SHIFT: usize = 12;

    pub fn new_with_promo(from: Square, to: Square, promotion: PieceType) -> Self {
        debug_assert!(u16::from(from) & Self::SQ_MASK == u16::from(from));
        debug_assert!(u16::from(to) & Self::SQ_MASK == u16::from(to));
        debug_assert_ne!(promotion, PieceType::Pawn, "attempted to construct promotion to pawn");
        debug_assert_ne!(promotion, PieceType::King, "attempted to construct promotion to king");
        let promotion = u16::from(promotion.inner()).wrapping_sub(1) & Self::PROMO_MASK; // can't promote to NO_PIECE or PAWN
        let data = u16::from(from)
            | (u16::from(to) << Self::TO_SHIFT)
            | (promotion << Self::PROMO_SHIFT)
            | MoveFlags::Promotion as u16;
        // SAFETY: data is always OR-ed with MoveFlags::Promotion, and so is always non-zero.
        let data = unsafe { NonZeroU16::new_unchecked(data) };
        Self { data }
    }

    pub fn new_with_flags(from: Square, to: Square, flags: MoveFlags) -> Self {
        debug_assert_ne!(flags, MoveFlags::Promotion, "promotion flag set without piece type");
        debug_assert!(u16::from(from) & Self::SQ_MASK == u16::from(from));
        debug_assert!(u16::from(to) & Self::SQ_MASK == u16::from(to));
        debug_assert_ne!(from, to);
        let data = u16::from(from) | (u16::from(to) << Self::TO_SHIFT) | flags as u16;
        // SAFETY: data is always OR-ed with a MoveFlag, which are all non-zero, and so is always non-zero.
        let data = unsafe { NonZeroU16::new_unchecked(data) };
        Self { data }
    }

    pub fn new(from: Square, to: Square) -> Self {
        debug_assert!(u16::from(from) & Self::SQ_MASK == u16::from(from));
        debug_assert!(u16::from(to) & Self::SQ_MASK == u16::from(to));
        debug_assert_ne!(from, to);
        let data = u16::from(from) | (u16::from(to) << Self::TO_SHIFT);
        // SAFETY: this function is only called from within the movegen routines,
        // where we never create A1 -> A1 moves. This function is technically unsound
        // if called as Move::new(Square::A1, Square::A1).
        let data = unsafe { NonZeroU16::new_unchecked(data) };
        Self { data }
    }

    pub const fn from(self) -> Square {
        // SAFETY: SQ_MASK guarantees that this is in bounds.
        unsafe { Square::new_unchecked((self.data.get() & Self::SQ_MASK) as u8) }
    }

    pub const fn to(self) -> Square {
        // SAFETY: SQ_MASK guarantees that this is in bounds.
        unsafe { Square::new_unchecked(((self.data.get() >> Self::TO_SHIFT) & Self::SQ_MASK) as u8) }
    }

    pub fn promotion_type(self) -> Option<PieceType> {
        if self.is_promo() {
            // SAFETY: out-of-range values are made impossible by the mask.
            let output = unsafe {
                PieceType::from_index_unchecked(((self.data.get() >> Self::PROMO_SHIFT) & Self::PROMO_MASK) as u8 + 1)
            };
            debug_assert!(output.legal_promo());
            Some(output)
        } else {
            None
        }
    }

    pub const fn is_promo(self) -> bool {
        (self.data.get() & PROMO_FLAG_BITS) == PROMO_FLAG_BITS
    }

    pub const fn is_ep(self) -> bool {
        (self.data.get() & EP_FLAG_BITS) != 0 && self.data.get() & CASTLE_FLAG_BITS == 0
    }

    pub const fn is_castle(self) -> bool {
        (self.data.get() & CASTLE_FLAG_BITS) != 0 && self.data.get() & EP_FLAG_BITS == 0
    }

    pub fn is_kingside_castling(self) -> bool {
        self.is_castle() && self.to() > self.from()
    }

    pub fn is_queenside_castling(self) -> bool {
        self.is_castle() && self.to() < self.from()
    }

    /// Handles castling moves, which are a bit weird.
    pub fn history_to_square(self) -> Square {
        if self.is_castle() {
            let to_rank = self.from().rank();
            if self.to() > self.from() {
                // kingside castling, king goes to the G file
                Square::from_rank_file(to_rank, File::G)
            } else {
                // queenside castling, king goes to the C file
                Square::from_rank_file(to_rank, File::C)
            }
        } else {
            self.to()
        }
    }

    pub fn is_valid(self) -> bool {
        let promotion = self.promotion_type();
        if promotion.is_some() && !self.is_promo() {
            // promotion type is set but not a promotion move
            return false;
        }
        promotion.is_none() || promotion.unwrap().legal_promo()
    }

    pub const fn inner(self) -> u16 {
        self.data.get()
    }

    pub fn from_raw(data: u16) -> Option<Self> {
        NonZeroU16::new(data).map(|nz| Self { data: nz })
    }
}

impl Display for Move {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        if CHESS960.load(Ordering::Relaxed) {
            if let Some(promo) = self.promotion_type() {
                let pchar = promo.promo_char().unwrap_or('?');
                write!(f, "{}{}{pchar}", self.from(), self.to())?;
            } else {
                write!(f, "{}{}", self.from(), self.to())?;
            }
        } else {
            let mut to = self.to();
            // fix up castling moves for normal UCI.
            if self.is_castle() {
                to = match to {
                    Square::H1 => Square::G1,
                    Square::A1 => Square::C1,
                    Square::H8 => Square::G8,
                    Square::A8 => Square::C8,
                    _ => unreachable!(),
                }
            }
            if let Some(promo) = self.promotion_type() {
                let pchar = promo.promo_char().unwrap_or('?');
                write!(f, "{}{}{}", self.from(), to, pchar)?;
            } else {
                write!(f, "{}{}", self.from(), to)?;
            }
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
            self.promotion_type().and_then(PieceType::promo_char).unwrap_or('X'),
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
        assert!(!m.is_promo());
        assert!(m.is_valid());
    }

    #[test]
    fn test_promotion() {
        use super::*;
        let m = Move::new_with_promo(Square::A7, Square::A8, PieceType::Queen);
        println!("{m:?}");
        println!("bitpattern: {:016b}", m.data);
        assert_eq!(m.from(), Square::A7);
        assert_eq!(m.to(), Square::A8);
        assert!(m.is_promo());
        assert!(!m.is_ep());
        assert!(!m.is_castle());
        assert_eq!(m.promotion_type(), Some(PieceType::Queen));
        assert!(m.is_valid());
    }

    #[test]
    fn test_all_square_combinations() {
        use super::*;
        use crate::squareset::SquareSet;
        for from in SquareSet::FULL {
            for to in SquareSet::FULL.iter().filter(|s| *s < from) {
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
