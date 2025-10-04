use std::{
    fmt::Display,
    ops::{
        BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr, Sub,
        SubAssign,
    },
};

use crate::chess::types::Square;

use super::piece::Colour;

/// A set of squares, with support for very fast set operations and in-order iteration.
/// Most chess engines call this type `Bitboard`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct SquareSet {
    inner: u64,
}

impl SquareSet {
    pub const EMPTY: Self = Self { inner: 0 };
    pub const FULL: Self = Self { inner: !0 };

    pub const RANK_1: Self = Self {
        inner: 0x0000_0000_0000_00FF,
    };
    pub const RANK_2: Self = Self {
        inner: 0x0000_0000_0000_FF00,
    };
    pub const RANK_3: Self = Self {
        inner: 0x0000_0000_00FF_0000,
    };
    pub const RANK_4: Self = Self {
        inner: 0x0000_0000_FF00_0000,
    };
    pub const RANK_5: Self = Self {
        inner: 0x0000_00FF_0000_0000,
    };
    pub const RANK_6: Self = Self {
        inner: 0x0000_FF00_0000_0000,
    };
    pub const RANK_7: Self = Self {
        inner: 0x00FF_0000_0000_0000,
    };
    pub const RANK_8: Self = Self {
        inner: 0xFF00_0000_0000_0000,
    };
    pub const FILE_A: Self = Self {
        inner: 0x0101_0101_0101_0101,
    };
    pub const FILE_B: Self = Self {
        inner: 0x0202_0202_0202_0202,
    };
    pub const FILE_C: Self = Self {
        inner: 0x0404_0404_0404_0404,
    };
    pub const FILE_D: Self = Self {
        inner: 0x0808_0808_0808_0808,
    };
    pub const FILE_E: Self = Self {
        inner: 0x1010_1010_1010_1010,
    };
    pub const FILE_F: Self = Self {
        inner: 0x2020_2020_2020_2020,
    };
    pub const FILE_G: Self = Self {
        inner: 0x4040_4040_4040_4040,
    };
    pub const FILE_H: Self = Self {
        inner: 0x8080_8080_8080_8080,
    };
    pub const LIGHT_SQUARES: Self = Self {
        inner: 0x55AA_55AA_55AA_55AA,
    };
    pub const DARK_SQUARES: Self = Self {
        inner: 0xAA55_AA55_AA55_AA55,
    };

    pub const RANKS: [Self; 8] = [
        Self::RANK_1,
        Self::RANK_2,
        Self::RANK_3,
        Self::RANK_4,
        Self::RANK_5,
        Self::RANK_6,
        Self::RANK_7,
        Self::RANK_8,
    ];

    pub const FILES: [Self; 8] = [
        Self::FILE_A,
        Self::FILE_B,
        Self::FILE_C,
        Self::FILE_D,
        Self::FILE_E,
        Self::FILE_F,
        Self::FILE_G,
        Self::FILE_H,
    ];

    pub const BACK_RANKS: Self = Self::union(Self::RANK_1, Self::RANK_8);

    pub const fn from_inner(inner: u64) -> Self {
        Self { inner }
    }

    pub const fn inner(self) -> u64 {
        self.inner
    }

    pub const fn count(self) -> u32 {
        self.inner.count_ones()
    }

    pub const fn intersection(self, other: Self) -> Self {
        Self {
            inner: self.inner & other.inner,
        }
    }

    pub const fn contains(self, other: Self) -> bool {
        (self.inner & other.inner) == other.inner
    }

    pub const fn contains_square(self, square: Square) -> bool {
        (self.inner & (1 << square.index())) != 0
    }

    pub const fn union(self, other: Self) -> Self {
        Self {
            inner: self.inner | other.inner,
        }
    }

    pub const fn add_square(self, square: Square) -> Self {
        Self {
            inner: self.inner | (1 << square.index()),
        }
    }

    pub const fn remove(self, other: Self) -> Self {
        Self {
            inner: self.inner & !other.inner,
        }
    }

    pub const fn remove_square(self, square: Square) -> Self {
        Self {
            inner: self.inner & !(1 << square.index()),
        }
    }

    pub const fn toggle(self, other: Self) -> Self {
        Self {
            inner: self.inner ^ other.inner,
        }
    }

    pub const fn toggle_square(self, square: Square) -> Self {
        Self {
            inner: self.inner ^ (1 << square.index()),
        }
    }

    #[allow(clippy::missing_const_for_fn)]
    pub fn iter(self) -> SquareIter {
        SquareIter::new(self.inner)
    }

    #[allow(clippy::cast_possible_truncation)]
    pub const fn first(self) -> Option<Square> {
        Square::new(self.inner.trailing_zeros() as u8)
    }

    pub const fn from_square(square: Square) -> Self {
        Self {
            inner: 1 << square.index(),
        }
    }

    pub fn north_east_one(self) -> Self {
        (self << 9) & !Self::FILE_A
    }
    pub fn north_west_one(self) -> Self {
        (self << 7) & !Self::FILE_H
    }
    pub fn south_east_one(self) -> Self {
        (self >> 7) & !Self::FILE_A
    }
    pub fn south_west_one(self) -> Self {
        (self >> 9) & !Self::FILE_H
    }
    pub fn east_one(self) -> Self {
        (self << 1) & !Self::FILE_A
    }
    pub fn west_one(self) -> Self {
        (self >> 1) & !Self::FILE_H
    }
    pub fn north_one(self) -> Self {
        self << 8
    }
    pub fn south_one(self) -> Self {
        self >> 8
    }

    pub fn isolate_lsb(self) -> Self {
        self & (Self::from_inner(0u64.wrapping_sub(self.inner())))
    }

    pub fn without_lsb(self) -> Self {
        self & (Self::from_inner(self.inner().wrapping_sub(1)))
    }

    pub fn one(self) -> bool {
        self != Self::EMPTY && self.without_lsb() == Self::EMPTY
    }

    pub fn many(self) -> bool {
        self.without_lsb() != Self::EMPTY
    }

    pub fn relative_to(self, colour: Colour) -> Self {
        if colour == Colour::White {
            self
        } else {
            Self {
                inner: self.inner.swap_bytes(),
            }
        }
    }
}

/// Iterator over the squares of a square-set.
/// The squares are returned in increasing order.
pub struct SquareIter {
    value: u64,
}

impl SquareIter {
    pub const fn new(value: u64) -> Self {
        Self { value }
    }
}

impl Iterator for SquareIter {
    type Item = Square;

    fn next(&mut self) -> Option<Self::Item> {
        if self.value == 0 {
            None
        } else {
            // faster if we have bmi (maybe)
            #[allow(clippy::cast_possible_truncation)]
            let lsb: u8 = self.value.trailing_zeros() as u8;
            self.value &= self.value - 1;
            // SAFETY: u64::trailing_zeros can only return values within `0..64`,
            // all of which correspond to valid enum variants of Square.
            Some(unsafe { Square::new_unchecked(lsb) })
        }
    }
}

impl IntoIterator for SquareSet {
    type Item = Square;
    type IntoIter = SquareIter;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl BitOr for SquareSet {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner | rhs.inner,
        }
    }
}

impl BitOrAssign for SquareSet {
    fn bitor_assign(&mut self, rhs: Self) {
        self.inner |= rhs.inner;
    }
}

impl BitAnd for SquareSet {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner & rhs.inner,
        }
    }
}

impl BitAndAssign for SquareSet {
    fn bitand_assign(&mut self, rhs: Self) {
        self.inner &= rhs.inner;
    }
}

impl BitXor for SquareSet {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner ^ rhs.inner,
        }
    }
}

impl BitXorAssign for SquareSet {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.inner ^= rhs.inner;
    }
}

impl Sub for SquareSet {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner & !rhs.inner,
        }
    }
}

impl SubAssign for SquareSet {
    fn sub_assign(&mut self, rhs: Self) {
        self.inner &= !rhs.inner;
    }
}

impl Not for SquareSet {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self { inner: !self.inner }
    }
}

impl Shr<u8> for SquareSet {
    type Output = Self;

    fn shr(self, rhs: u8) -> Self::Output {
        Self {
            inner: self.inner >> rhs,
        }
    }
}

impl Shl<u8> for SquareSet {
    type Output = Self;

    fn shl(self, rhs: u8) -> Self::Output {
        Self {
            inner: self.inner << rhs,
        }
    }
}

impl Display for SquareSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for rank in (0..8).rev() {
            for file in 0..8 {
                let bit = 1u64 << (rank * 8 + file);
                write!(f, "{}", if self.inner & bit != 0 { '1' } else { '0' })?;
            }
            if rank > 0 {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::chess::{squareset::SquareSet, types::Square};

    #[test]
    fn counters() {
        let empty = SquareSet::EMPTY;
        assert_eq!(empty, SquareSet::EMPTY);
        assert!(!empty.one());
        assert!(!empty.many());

        let one = Square::E4.as_set();
        assert_ne!(one, SquareSet::EMPTY);
        assert!(one.one());
        assert!(!one.many());

        let two = one.add_square(Square::E5);
        assert_ne!(two, SquareSet::EMPTY);
        assert!(!two.one());
        assert!(two.many());
    }
}
