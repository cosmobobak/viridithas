#![expect(clippy::cast_sign_loss, clippy::undocumented_unsafe_blocks)]

use std::arch::x86_64::*;

use crate::chess::{piece::Piece, types::Square};

use super::{BitRays, INCOMING_SLIDERS_MASK, INCOMING_THREATS_MASK, PERMUTATION, PIECE_TO_BIT};

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Vector {
    pub raw: __m512i,
}

impl Vector {
    pub fn flip(self) -> Self {
        unsafe {
            Self {
                raw: _mm512_shuffle_i64x2(self.raw, self.raw, 0b01_00_11_10),
            }
        }
    }
}

pub struct Permutation {
    pub indices: Vector,
    pub valid: u64,
}

pub fn permutation_for(focus: Square) -> Permutation {
    unsafe {
        let indices = _mm512_loadu_si512(PERMUTATION[focus.index()].as_ptr().cast());
        let valid = _mm512_testn_epi8_mask(indices, _mm512_set1_epi8(0x80u8 as i8));
        Permutation {
            indices: Vector { raw: indices },
            valid,
        }
    }
}

pub fn permute_mailbox(
    permutation: &Permutation,
    mailbox: &[Option<Piece>; 64],
) -> (Vector, Vector) {
    unsafe {
        let lut = _mm512_broadcast_i32x4(_mm_loadu_si128(PIECE_TO_BIT.as_ptr().cast::<__m128i>()));

        let masked_mailbox = _mm512_loadu_si512(mailbox.as_ptr().cast());
        let permuted = _mm512_permutexvar_epi8(permutation.indices.raw, masked_mailbox);
        let bits = _mm512_maskz_shuffle_epi8(permutation.valid, lut, permuted);
        (Vector { raw: permuted }, Vector { raw: bits })
    }
}

pub fn permute_mailbox_ignoring(
    permutation: &Permutation,
    mailbox: &[Option<Piece>; 64],
    ignore: Square,
) -> (Vector, Vector) {
    unsafe {
        let lut = _mm512_broadcast_i32x4(_mm_loadu_si128(PIECE_TO_BIT.as_ptr().cast::<__m128i>()));

        let ignore_mask: u64 = 1 << ignore.inner();
        let masked_mailbox = _mm512_mask_blend_epi8(
            ignore_mask,
            _mm512_loadu_si512(mailbox.as_ptr().cast()),
            // None<Piece> is represented as 12 due to niche optimisation.
            _mm512_set1_epi8(12),
        );
        let permuted = _mm512_permutexvar_epi8(permutation.indices.raw, masked_mailbox);
        let bits = _mm512_maskz_shuffle_epi8(permutation.valid, lut, permuted);
        (Vector { raw: permuted }, Vector { raw: bits })
    }
}

pub fn closest_occupied(bits: Vector) -> BitRays {
    unsafe {
        let occupied = _mm512_test_epi8_mask(bits.raw, bits.raw);
        let o = occupied | 0x8181818181818181;
        (o ^ o.wrapping_sub(0x0303030303030303)) & occupied
    }
}

pub fn incoming_attackers(bits: Vector, closest: BitRays) -> BitRays {
    unsafe {
        let mask = _mm512_loadu_si512(INCOMING_THREATS_MASK.as_ptr().cast());
        _mm512_test_epi8_mask(bits.raw, mask) & closest
    }
}

pub fn incoming_sliders(bits: Vector, closest: BitRays) -> BitRays {
    unsafe {
        let mask = _mm512_loadu_si512(INCOMING_SLIDERS_MASK.as_ptr().cast());
        _mm512_test_epi8_mask(bits.raw, mask) & closest & 0xFEFEFEFEFEFEFEFE
    }
}
