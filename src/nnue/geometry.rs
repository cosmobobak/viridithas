#[cfg(target_feature = "avx512vbmi")]
mod vbmi;
#[cfg(target_feature = "avx512vbmi")]
pub use vbmi::*;

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512vbmi")))]
mod avx2;
#[cfg(all(target_feature = "avx2", not(target_feature = "avx512vbmi")))]
pub use avx2::*;

#[cfg(target_feature = "neon")]
mod neon;
#[cfg(target_feature = "neon")]
pub use neon::*;

use crate::{cfor, chess::piece::Piece};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub(crate) struct Bit(u8);

#[rustfmt::skip]
impl Bit {
    const WHITE_PAWN : Self = Self(0x01);
    const BLACK_PAWN : Self = Self(0x02);
    const KNIGHT     : Self = Self(0x04);
    const BISHOP     : Self = Self(0x08);
    const ROOK       : Self = Self(0x10);
    const QUEEN      : Self = Self(0x20);
    const KING       : Self = Self(0x40);
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct BitRays(pub u64);

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

const OUTGOING_THREATS: [BitRays; 12] = {
    let mut lut = [BitRays(0); 12];
    lut[Piece::WP as usize].0 = 0x02_00_00_00_00_00_02_00;
    lut[Piece::BP as usize].0 = 0x00_00_02_00_02_00_00_00;
    lut[Piece::WN as usize].0 = 0x01_01_01_01_01_01_01_01;
    lut[Piece::BN as usize].0 = 0x01_01_01_01_01_01_01_01;
    lut[Piece::WB as usize].0 = 0xFE_00_FE_00_FE_00_FE_00;
    lut[Piece::BB as usize].0 = 0xFE_00_FE_00_FE_00_FE_00;
    lut[Piece::WR as usize].0 = 0x00_FE_00_FE_00_FE_00_FE;
    lut[Piece::BR as usize].0 = 0x00_FE_00_FE_00_FE_00_FE;
    lut[Piece::WQ as usize].0 = 0xFE_FE_FE_FE_FE_FE_FE_FE;
    lut[Piece::BQ as usize].0 = 0xFE_FE_FE_FE_FE_FE_FE_FE;
    lut[Piece::WK as usize].0 = 0; // ignore king threats
    lut[Piece::BK as usize].0 = 0; // ignore king threats
    lut
};

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

#[inline]
pub fn ray_fill(mut br: BitRays) -> BitRays {
    br.0 = (br.0.wrapping_add(0x7E_7E_7E_7E_7E_7E_7E_7E)) & 0x80_80_80_80_80_80_80_80;
    BitRays(br.0.wrapping_sub(br.0 >> 7))
}

#[inline]
pub fn outgoing_threats(piece: Piece, closest: BitRays) -> BitRays {
    BitRays(OUTGOING_THREATS[piece as usize].0 & closest.0)
}
