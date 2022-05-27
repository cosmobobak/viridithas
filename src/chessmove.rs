use std::fmt::{Debug, Display, Formatter};

use crate::{
    definitions::{square120_name, square64_name},
    lookups::PROMO_CHAR_LOOKUP,
};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Move {
    pub data: u32,
}

impl Move {
    const FROM_MASK: u32 = 0b0111_1111;
    const TO_MASK: u32 = 0b0011_1111_1000_0000;
    const CAPTURE_MASK: u32 = 0b0011_1100_0000_0000_0000;
    const PROMO_MASK: u32 = 0b1111_0000_0000_0000_0000_0000;
    pub const EP_MASK: u32 = 0b0100_0000_0000_0000_0000;
    pub const PAWN_START_MASK: u32 = 0b1000_0000_0000_0000_0000;
    pub const CASTLE_MASK: u32 = 0b0001_0000_0000_0000_0000_0000_0000;
    pub const NULL: Self = Self { data: 0 };

    pub fn new(from: u8, to: u8, capture: u8, promotion: u8, flags: u32) -> Self {
        debug_assert!(
            (flags & (Self::EP_MASK | Self::PAWN_START_MASK | Self::CASTLE_MASK)) == flags
        );
        Self {
            data: u32::from(from)
                | (u32::from(to) << 7)
                | (u32::from(capture) << 14)
                | (u32::from(promotion) << 20)
                | flags,
        }
    }

    pub const fn from(self) -> u8 {
        (self.data & Self::FROM_MASK) as u8
    }

    pub const fn to(self) -> u8 {
        (((self.data & Self::TO_MASK) >> 7) & 0x7F) as u8
    }

    pub const fn capture(self) -> u8 {
        (((self.data & Self::CAPTURE_MASK) >> 14) & 0xF) as u8
    }

    pub const fn promotion(self) -> u8 {
        (((self.data & Self::PROMO_MASK) >> 20) & 0xF) as u8
    }

    pub const fn is_promo(self) -> bool {
        ((self.data & Self::PROMO_MASK) != 0) as bool
    }

    pub const fn is_ep(self) -> bool {
        (self.data & Self::EP_MASK) != 0
    }

    pub const fn is_pawn_start(self) -> bool {
        (self.data & Self::PAWN_START_MASK) != 0
    }

    pub const fn is_castle(self) -> bool {
        (self.data & Self::CASTLE_MASK) != 0
    }

    pub const fn is_capture(self) -> bool {
        (self.data & Self::CAPTURE_MASK) != 0
    }

    pub const fn is_null(self) -> bool {
        self.data == 0
    }
}

impl Display for Move {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        if self.is_null() {
            return write!(f, "null");
        }

        let from_square =
            square120_name(self.from()).unwrap_or_else(|| panic!("Invalid square {}", self.from()));
        let to_square =
            square120_name(self.to()).unwrap_or_else(|| panic!("Invalid square {}", self.to()));

        if self.is_promo() {
            let pchar = PROMO_CHAR_LOOKUP[self.promotion() as usize];
            write!(f, "{}{}{}", from_square, to_square, pchar as char)?;
        } else {
            write!(f, "{}{}", from_square, to_square)?;
        }

        Ok(())
    }
}

impl Debug for Move {
    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
        write!(
            f,
            "move from {} ({}) to {} ({}), capture {}, promo {}, ispromo {}, ep {}, pawn start {}, castle {}",
            self.from(),
            square64_name(self.from()).unwrap_or("NONE"), 
            self.to(),
            square64_name(self.to()).unwrap_or("NONE"),
            self.capture(),
            self.promotion(),
            self.is_promo(),
            self.is_ep(),
            self.is_pawn_start(),
            self.is_castle()
        )
    }
}
