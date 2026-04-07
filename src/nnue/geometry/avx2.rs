#![expect(clippy::cast_sign_loss, clippy::undocumented_unsafe_blocks)]

use std::arch::x86_64::*;

use crate::{
    chess::{piece::Piece, types::Square},
    util::Align64,
};

use super::{BitRays, INCOMING_SLIDERS_MASK, INCOMING_THREATS_MASK, PERMUTATION, PIECE_TO_BIT};

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Vector {
    pub raw: [__m256i; 2],
}

impl Vector {
    pub fn flip(self) -> Self {
        Self {
            raw: [self.raw[1], self.raw[0]],
        }
    }

    fn mask(self) -> BitRays {
        unsafe {
            let a = u64::from(_mm256_movemask_epi8(self.raw[0]) as u32);
            let b = u64::from(_mm256_movemask_epi8(self.raw[1]) as u32);
            a | (b << 32)
        }
    }

    unsafe fn load(ptr: *const u8) -> Self {
        #![expect(clippy::cast_ptr_alignment)]
        unsafe {
            Self {
                raw: [
                    _mm256_loadu_si256(ptr.cast::<__m256i>()),
                    _mm256_loadu_si256(ptr.cast::<__m256i>().add(1)),
                ],
            }
        }
    }

    unsafe fn cast<T>(v: &T) -> Self {
        unsafe { Self::load((v as *const T).cast::<u8>()) }
    }
}

pub struct Permutation {
    pub indices: Vector,
    pub invalid: Vector,
}

pub fn permutation_for(focus: Square) -> Permutation {
    unsafe {
        let indices = Vector::cast(&PERMUTATION[focus.index()]);
        let invalid = Vector {
            raw: [
                _mm256_cmpeq_epi8(indices.raw[0], _mm256_set1_epi8(0x80u8 as i8)),
                _mm256_cmpeq_epi8(indices.raw[1], _mm256_set1_epi8(0x80u8 as i8)),
            ],
        };
        Permutation { indices, invalid }
    }
}

fn permute_mailbox_inner(permutation: &Permutation, masked_mailbox: Vector) -> (Vector, Vector) {
    unsafe {
        let lut =
            _mm256_broadcastsi128_si256(_mm_loadu_si128(PIECE_TO_BIT.as_ptr().cast::<__m128i>()));

        let half_swizzle = |bytes0: __m256i, bytes1: __m256i, idxs: __m256i| -> __m256i {
            let mask0 = _mm256_slli_epi64(idxs, 2);
            let mask1 = _mm256_slli_epi64(idxs, 3);

            let lolo0 = _mm256_shuffle_epi8(_mm256_permute2x128_si256(bytes0, bytes0, 0x00), idxs);
            let hihi0 = _mm256_shuffle_epi8(_mm256_permute2x128_si256(bytes0, bytes0, 0x11), idxs);
            let x = _mm256_blendv_epi8(lolo0, hihi0, mask1);

            let lolo1 = _mm256_shuffle_epi8(_mm256_permute2x128_si256(bytes1, bytes1, 0x00), idxs);
            let hihi1 = _mm256_shuffle_epi8(_mm256_permute2x128_si256(bytes1, bytes1, 0x11), idxs);
            let y = _mm256_blendv_epi8(lolo1, hihi1, mask1);

            _mm256_blendv_epi8(x, y, mask0)
        };

        let permuted = Vector {
            raw: [
                half_swizzle(
                    masked_mailbox.raw[0],
                    masked_mailbox.raw[1],
                    permutation.indices.raw[0],
                ),
                half_swizzle(
                    masked_mailbox.raw[0],
                    masked_mailbox.raw[1],
                    permutation.indices.raw[1],
                ),
            ],
        };
        let bits = Vector {
            raw: [
                _mm256_andnot_si256(
                    permutation.invalid.raw[0],
                    _mm256_shuffle_epi8(lut, permuted.raw[0]),
                ),
                _mm256_andnot_si256(
                    permutation.invalid.raw[1],
                    _mm256_shuffle_epi8(lut, permuted.raw[1]),
                ),
            ],
        };

        (permuted, bits)
    }
}

pub fn permute_mailbox(
    permutation: &Permutation,
    mailbox: &[Option<Piece>; 64],
) -> (Vector, Vector) {
    let mb = unsafe { Vector::load(mailbox.as_ptr().cast::<u8>()) };
    permute_mailbox_inner(permutation, mb)
}

pub fn permute_mailbox_ignoring(
    permutation: &Permutation,
    mailbox: &[Option<Piece>; 64],
    ignore: Square,
) -> (Vector, Vector) {
    unsafe {
        let iota = Align64([
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
            46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        ]);
        let iota = Vector::load(iota.0.as_ptr());
        let ignore_vec = _mm256_set1_epi8(ignore.inner() as i8);
        // None<Piece> is represented as 12 due to niche optimisation.
        let none_vec = _mm256_set1_epi8(12);
        let mb = Vector::load(mailbox.as_ptr().cast::<u8>());
        let masked_mailbox = Vector {
            raw: [
                _mm256_blendv_epi8(
                    mb.raw[0],
                    none_vec,
                    _mm256_cmpeq_epi8(iota.raw[0], ignore_vec),
                ),
                _mm256_blendv_epi8(
                    mb.raw[1],
                    none_vec,
                    _mm256_cmpeq_epi8(iota.raw[1], ignore_vec),
                ),
            ],
        };
        permute_mailbox_inner(permutation, masked_mailbox)
    }
}

pub fn closest_occupied(bits: Vector) -> BitRays {
    unsafe {
        let unoccupied = Vector {
            raw: [
                _mm256_cmpeq_epi8(bits.raw[0], _mm256_setzero_si256()),
                _mm256_cmpeq_epi8(bits.raw[1], _mm256_setzero_si256()),
            ],
        };
        let occupied = !unoccupied.mask();
        let o = occupied | 0x8181_8181_8181_8181;
        (o ^ o.wrapping_sub(0x0303_0303_0303_0303)) & occupied
    }
}

pub fn incoming_attackers(bits: Vector, closest: BitRays) -> BitRays {
    unsafe {
        let mask = Vector::cast(&INCOMING_THREATS_MASK);
        let v = Vector {
            raw: [
                _mm256_cmpeq_epi8(
                    _mm256_and_si256(bits.raw[0], mask.raw[0]),
                    _mm256_setzero_si256(),
                ),
                _mm256_cmpeq_epi8(
                    _mm256_and_si256(bits.raw[1], mask.raw[1]),
                    _mm256_setzero_si256(),
                ),
            ],
        };
        !v.mask() & closest
    }
}

pub fn incoming_sliders(bits: Vector, closest: BitRays) -> BitRays {
    unsafe {
        let mask = Vector::cast(&INCOMING_SLIDERS_MASK);
        let v = Vector {
            raw: [
                _mm256_cmpeq_epi8(
                    _mm256_and_si256(bits.raw[0], mask.raw[0]),
                    _mm256_setzero_si256(),
                ),
                _mm256_cmpeq_epi8(
                    _mm256_and_si256(bits.raw[1], mask.raw[1]),
                    _mm256_setzero_si256(),
                ),
            ],
        };
        !v.mask() & closest & 0xFEFE_FEFE_FEFE_FEFE
    }
}
