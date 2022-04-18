use std::fmt::{Display, Formatter, Debug};

use crate::{definitions::{square120_name, square64_name}, attack::{IS_KNIGHT, IS_ROOKQUEEN, IS_BISHOPQUEEN}};

pub struct Move {
    pub data: u32
}

impl Move {
    const FROM_MASK: u32 = 0b0111_1111;
    const TO_MASK: u32 = 0b0011_1111_1000_0000;
    const CAPTURE_MASK: u32 = 0b0011_1100_0000_0000_0000;
    const EP_MASK: u32 = 0b0100_0000_0000_0000_0000;
    const PAWN_START_MASK: u32 = 0b1000_0000_0000_0000_0000;
    const PROMO_MASK: u32 = 0b1111_0000_0000_0000_0000_0000;
    const CASTLE_MASK: u32 = 0b0001_0000_0000_0000_0000_0000_0000;

    pub fn new(from: u8, to: u8, capture: u8, promotion: u8, ep: bool, pawn_start: bool, castle: bool) -> Self {
        let mut data = u32::from(from)
         | (u32::from(to) << 7)
         | (u32::from(capture) << 14)
         | (u32::from(promotion) << 20);
        if ep {
            data |= Self::EP_MASK;
        }
        if pawn_start {
            data |= Self::PAWN_START_MASK;
        }
        if castle {
            data |= Self::CASTLE_MASK;
        }
        Self { data }
    }

    pub fn from(&self) -> u8 {
        (self.data & Self::FROM_MASK) as u8
    }

    pub fn to(&self) -> u8 {
        (((self.data & Self::TO_MASK) >> 7) & 0x7F) as u8
    }

    pub fn capture(&self) -> u8 {
        (((self.data & Self::CAPTURE_MASK) >> 14) & 0xF) as u8
    }

    pub fn promotion(&self) -> u8 {
        (((self.data & Self::PROMO_MASK) >> 20) & 0xF) as u8
    }

    pub fn is_promo(&self) -> bool {
        ((self.data & Self::PROMO_MASK) != 0) as bool
    }

    pub fn is_ep(&self) -> bool {
        (self.data & Self::EP_MASK) != 0
    }

    pub fn is_pawn_start(&self) -> bool {
        (self.data & Self::PAWN_START_MASK) != 0
    }

    pub fn is_castle(&self) -> bool {
        (self.data & Self::CASTLE_MASK) != 0
    }
}

impl Display for Move {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        let from_square = square64_name(self.from()).expect("Invalid square");
        let to_square = square64_name(self.to()).expect("Invalid square");

        if self.is_promo() {
            let pchar = if IS_KNIGHT[self.promotion() as usize] {
                'n'
            } else if IS_ROOKQUEEN[self.promotion() as usize] && !IS_BISHOPQUEEN[self.promotion() as usize] {
                'r'
            } else if IS_BISHOPQUEEN[self.promotion() as usize] && !IS_ROOKQUEEN[self.promotion() as usize] {
                'b'
            } else {
                'q'
            };

            write!(f, "{}{}{}", from_square, to_square, pchar)?;
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