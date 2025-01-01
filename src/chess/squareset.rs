use std::ops::{
    BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr, Sub, SubAssign,
};

use crate::chess::types::Square;

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

    pub const fn from_inner(inner: u64) -> Self {
        Self { inner }
    }

    pub const fn inner(self) -> u64 {
        self.inner
    }

    pub const fn count(self) -> u32 {
        self.inner.count_ones()
    }

    pub const fn is_empty(self) -> bool {
        self.inner == 0
    }

    pub const fn is_full(self) -> bool {
        self.inner == !0
    }

    pub const fn non_empty(self) -> bool {
        self.inner != 0
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
    pub const fn first(self) -> Square {
        debug_assert!(
            self.inner != 0,
            "Tried to get first square of empty square-set"
        );
        // SAFETY: u64::trailing_zeros can only return values within `0..64`,
        // all of which correspond to valid enum variants of Square.
        unsafe { Square::new_unchecked(self.inner.trailing_zeros() as u8) }
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
