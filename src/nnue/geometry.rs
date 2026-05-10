#[cfg(target_feature = "avx512vbmi")]
mod vbmi;
use std::ops::{BitAnd, Not};

#[cfg(target_feature = "avx512vbmi")]
pub use vbmi::*;

#[cfg(not(any(target_feature = "neon", target_feature = "avx512vbmi")))]
mod avx2;
#[cfg(not(any(target_feature = "neon", target_feature = "avx512vbmi")))]
pub use avx2::*;

#[cfg(target_feature = "neon")]
mod neon;
#[cfg(target_feature = "neon")]
pub use neon::*;

use crate::{cfor, chess::piece::Piece};

/// Bits that can be queried in order to extract masks from `BitRays`.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Bit(u8);

#[rustfmt::skip]
impl Bit {
    const WHITE_PAWN : Self = Self(0x01);
    const BLACK_PAWN : Self = Self(0x02);
    const KNIGHT     : Self = Self(0x04);
    const BISHOP     : Self = Self(0x08);
    const ROOK       : Self = Self(0x10);
    const QUEEN      : Self = Self(0x20);
    pub const KING   : Self = Self(0x40);
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct BitRays(u64);

impl BitRays {
    pub const NON_KNIGHT: BitRays = BitRays(0xFEFE_FEFE_FEFE_FEFE);

    #[allow(dead_code)]
    pub fn inner(self) -> u64 {
        self.0
    }

    pub fn count_ones(self) -> u32 {
        self.0.count_ones()
    }

    pub fn flip(self) -> Self {
        Self(self.0.rotate_right(32))
    }
}

impl BitAnd for BitRays {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        BitRays(self.0 & rhs.0)
    }
}

impl Not for BitRays {
    type Output = Self;

    fn not(self) -> Self::Output {
        BitRays(!self.0)
    }
}

impl IntoIterator for BitRays {
    type Item = usize;
    type IntoIter = BitRaysIter;

    fn into_iter(self) -> Self::IntoIter {
        BitRaysIter(self.0)
    }
}

pub struct BitRaysIter(u64);

impl Iterator for BitRaysIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.0 == 0 {
            None
        } else {
            let lsb = self.0.trailing_zeros();
            self.0 &= self.0 - 1;
            Some(lsb as usize)
        }
    }
}

/// Given a square, return a permutation that allows the transformation
/// of a mailbox into a ray-vector, as specified in
/// <https://87flowers.com/byteboard-attack-tables-1/#perm>
#[expect(clippy::cast_possible_truncation)]
pub const PERMUTATION: [[u8; 64]; 64] = {
    let offsets = [
        0x1F, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, // N
        0x21, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, // NE
        0x12, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, // E
        0xF2, 0xF1, 0xE2, 0xD3, 0xC4, 0xB5, 0xA6, 0x97, // SE
        0xE1, 0xF0, 0xE0, 0xD0, 0xC0, 0xB0, 0xA0, 0x90, // S
        0xDF, 0xEF, 0xDE, 0xCD, 0xBC, 0xAB, 0x9A, 0x89, // SW
        0xEE, 0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xFA, 0xF9, // W
        0x0E, 0x0F, 0x1E, 0x2D, 0x3C, 0x4B, 0x5A, 0x69, // NW
    ];

    let mut permutations = [[0; 64]; 64];

    cfor!(let mut focus = 0; focus < 64; focus += 1; {
        cfor!(let mut i = 0; i < 64; i += 1; {
            let wide_focus = focus + (focus & 0x38);
            let wide_result = offsets[i] + wide_focus;
            let result = ((wide_result & 0x70) >> 1) | (wide_result & 0x07);
            let valid = (wide_result & 0x88) == 0;
            permutations[focus][i] = if valid { result as u8 } else { 0x80 };
        });
    });

    permutations
};

/// Given a piece, get the `Bit` mask used to extract a mask from `BitRays`.
pub const PIECE_TO_BIT: [Bit; 16] = {
    let mut lut = [Bit(0); 16];

    lut[Piece::WP as usize] = Bit::WHITE_PAWN;
    lut[Piece::BP as usize] = Bit::BLACK_PAWN;
    lut[Piece::WN as usize] = Bit::KNIGHT;
    lut[Piece::BN as usize] = Bit::KNIGHT;
    lut[Piece::WB as usize] = Bit::BISHOP;
    lut[Piece::BB as usize] = Bit::BISHOP;
    lut[Piece::WR as usize] = Bit::ROOK;
    lut[Piece::BR as usize] = Bit::ROOK;
    lut[Piece::WQ as usize] = Bit::QUEEN;
    lut[Piece::BQ as usize] = Bit::QUEEN;
    lut[Piece::WK as usize] = Bit::KING;
    lut[Piece::BK as usize] = Bit::KING;

    lut
};

/// The ray-vector format attacks of a piece.
/// Read <https://87flowers.com/byteboard-attack-tables-1/>
/// for an introduction to this format.
///
/// The layout is
///
/// ```text
/// knight [     north ray attacks]
/// knight [north-east ray attacks]
/// knight [      east ray attacks]
/// knight [south-east ray attacks]
/// knight [     south ray attacks]
/// knight [south-west ray attacks]
/// knight [      west ray attacks]
/// knight [north-west ray attacks]
/// ```
///
/// As such, white pawns use `0x02_00_00_00_00_00_02_00`,
/// encoding a bit in index [1] of second and last rows,
/// or index [0] in the north-east & north-west ray attacks.
///
/// Similarly, knights use `0x01_01_01_01_01_01_01_01`, as
/// they fill all the knight attack slots and none of the
/// ray slots.
///
/// King threats are zeroed as a quirk of the particular
/// feature-set we use, which fully excludes king threats
/// – ordinarily, they’d be `0x02_02_02_02_02_02_02_02`.
const OUTGOING_THREATS: [BitRays; 12] = {
    let mut lut = [BitRays(0); 12];
    lut[Piece::WP as usize] = BitRays(0x02_00_00_00_00_00_02_00);
    lut[Piece::BP as usize] = BitRays(0x00_00_02_00_02_00_00_00);
    lut[Piece::WN as usize] = BitRays(0x01_01_01_01_01_01_01_01);
    lut[Piece::BN as usize] = BitRays(0x01_01_01_01_01_01_01_01);
    lut[Piece::WB as usize] = BitRays(0xFE_00_FE_00_FE_00_FE_00);
    lut[Piece::BB as usize] = BitRays(0xFE_00_FE_00_FE_00_FE_00);
    lut[Piece::WR as usize] = BitRays(0x00_FE_00_FE_00_FE_00_FE);
    lut[Piece::BR as usize] = BitRays(0x00_FE_00_FE_00_FE_00_FE);
    lut[Piece::WQ as usize] = BitRays(0xFE_FE_FE_FE_FE_FE_FE_FE);
    lut[Piece::BQ as usize] = BitRays(0xFE_FE_FE_FE_FE_FE_FE_FE);
    lut[Piece::WK as usize] = BitRays(0); // ignore king threats
    lut[Piece::BK as usize] = BitRays(0); // ignore king threats
    lut
};

/// Given a ray-vector index, which sorts of piece can attack it?
pub const INCOMING_THREATS_MASK: [Bit; 64] = {
    const HORS: Bit = Bit::KNIGHT;
    const ORTH: Bit = Bit(Bit::QUEEN.0 | Bit::ROOK.0);
    const DIAG: Bit = Bit(Bit::QUEEN.0 | Bit::BISHOP.0);
    const ORNR: Bit = ORTH;
    const WPNR: Bit = Bit(DIAG.0 | Bit::WHITE_PAWN.0);
    const BPNR: Bit = Bit(DIAG.0 | Bit::BLACK_PAWN.0);

    [
        HORS, ORNR, ORTH, ORTH, ORTH, ORTH, ORTH, ORTH, // N
        HORS, BPNR, DIAG, DIAG, DIAG, DIAG, DIAG, DIAG, // NE
        HORS, ORNR, ORTH, ORTH, ORTH, ORTH, ORTH, ORTH, // E
        HORS, WPNR, DIAG, DIAG, DIAG, DIAG, DIAG, DIAG, // SE
        HORS, ORNR, ORTH, ORTH, ORTH, ORTH, ORTH, ORTH, // S
        HORS, WPNR, DIAG, DIAG, DIAG, DIAG, DIAG, DIAG, // SW
        HORS, ORNR, ORTH, ORTH, ORTH, ORTH, ORTH, ORTH, // W
        HORS, BPNR, DIAG, DIAG, DIAG, DIAG, DIAG, DIAG, // NW
    ]
};

/// Given a ray-vector index, which sliders can attack it?
pub const INCOMING_SLIDERS_MASK: [Bit; 64] = {
    const ORTH: Bit = Bit(Bit::QUEEN.0 | Bit::ROOK.0);
    const DIAG: Bit = Bit(Bit::QUEEN.0 | Bit::BISHOP.0);
    const NULL: Bit = Bit(0x80);

    [
        NULL, ORTH, ORTH, ORTH, ORTH, ORTH, ORTH, ORTH, // N
        NULL, DIAG, DIAG, DIAG, DIAG, DIAG, DIAG, DIAG, // NE
        NULL, ORTH, ORTH, ORTH, ORTH, ORTH, ORTH, ORTH, // E
        NULL, DIAG, DIAG, DIAG, DIAG, DIAG, DIAG, DIAG, // SE
        NULL, ORTH, ORTH, ORTH, ORTH, ORTH, ORTH, ORTH, // S
        NULL, DIAG, DIAG, DIAG, DIAG, DIAG, DIAG, DIAG, // SW
        NULL, ORTH, ORTH, ORTH, ORTH, ORTH, ORTH, ORTH, // W
        NULL, DIAG, DIAG, DIAG, DIAG, DIAG, DIAG, DIAG, // NW
    ]
};

/// Given some bit-rays, selects every ray that has a nonzero
/// element, and fills the whole ray.
#[inline]
pub fn ray_fill(br: BitRays) -> BitRays {
    let br = (br.0.wrapping_add(0x7E_7E_7E_7E_7E_7E_7E_7E)) & 0x80_80_80_80_80_80_80_80;
    BitRays(br.wrapping_sub(br >> 7))
}

/// Determine which of `closest` are seen by this piece.
#[inline]
pub fn outgoing_threats(piece: Piece, closest: BitRays) -> BitRays {
    OUTGOING_THREATS[piece as usize] & closest
}
