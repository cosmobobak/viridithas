#![expect(clippy::undocumented_unsafe_blocks)]

use std::arch::aarch64::*;

use crate::{
    chess::{piece::Piece, types::Square},
    util::Align64,
};

use super::{BitRays, INCOMING_SLIDERS_MASK, INCOMING_THREATS_MASK, PERMUTATION, PIECE_TO_BIT};

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Vector {
    pub raw: [uint8x16_t; 4],
}

impl Vector {
    pub fn flip(self) -> Self {
        Self {
            raw: [self.raw[2], self.raw[3], self.raw[0], self.raw[1]],
        }
    }

    fn to_mask(&self) -> u64 {
        #[rustfmt::skip]
        const MASK: [u8; 16] = [
            0x01, 0x02, 0x04, 0x08,
            0x10, 0x20, 0x40, 0x80,
            0x01, 0x02, 0x04, 0x08,
            0x10, 0x20, 0x40, 0x80,
        ];
        unsafe {
            let mask = vld1q_u8(MASK.as_ptr());
            let v = vpaddq_u8(
                vpaddq_u8(vandq_u8(self.raw[0], mask), vandq_u8(self.raw[1], mask)),
                vpaddq_u8(vandq_u8(self.raw[2], mask), vandq_u8(self.raw[3], mask)),
            );
            vgetq_lane_u64(vreinterpretq_u64_u8(vpaddq_u8(v, v)), 0)
        }
    }

    unsafe fn load(ptr: *const u8) -> Self {
        unsafe {
            let x = vld1q_u8_x4(ptr);
            Self {
                raw: [x.0, x.1, x.2, x.3],
            }
        }
    }

    unsafe fn cast<T>(v: &T) -> Self {
        unsafe { Self::load((v as *const T).cast::<u8>()) }
    }
}

pub struct Permutation {
    pub indices: Vector,
    pub valid: Vector,
}

pub fn permutation_for(focus: Square) -> Permutation {
    unsafe {
        let indices = Vector::load(PERMUTATION[focus.index()].as_ptr());
        let valid = Vector {
            raw: [
                vmvnq_u8(vreinterpretq_u8_s8(vshrq_n_s8(
                    vreinterpretq_s8_u8(indices.raw[0]),
                    7,
                ))),
                vmvnq_u8(vreinterpretq_u8_s8(vshrq_n_s8(
                    vreinterpretq_s8_u8(indices.raw[1]),
                    7,
                ))),
                vmvnq_u8(vreinterpretq_u8_s8(vshrq_n_s8(
                    vreinterpretq_s8_u8(indices.raw[2]),
                    7,
                ))),
                vmvnq_u8(vreinterpretq_u8_s8(vshrq_n_s8(
                    vreinterpretq_s8_u8(indices.raw[3]),
                    7,
                ))),
            ],
        };
        Permutation { indices, valid }
    }
}

fn permute_mailbox_inner(permutation: &Permutation, mailbox: &Vector) -> (Vector, Vector) {
    unsafe {
        let lut = vld1q_u8(PIECE_TO_BIT.as_ptr().cast::<u8>());

        let mb = uint8x16x4_t(
            mailbox.raw[0],
            mailbox.raw[1],
            mailbox.raw[2],
            mailbox.raw[3],
        );
        let permuted = Vector {
            raw: [
                vqtbl4q_u8(mb, permutation.indices.raw[0]),
                vqtbl4q_u8(mb, permutation.indices.raw[1]),
                vqtbl4q_u8(mb, permutation.indices.raw[2]),
                vqtbl4q_u8(mb, permutation.indices.raw[3]),
            ],
        };
        let bits = Vector {
            raw: [
                vandq_u8(vqtbl1q_u8(lut, permuted.raw[0]), permutation.valid.raw[0]),
                vandq_u8(vqtbl1q_u8(lut, permuted.raw[1]), permutation.valid.raw[1]),
                vandq_u8(vqtbl1q_u8(lut, permuted.raw[2]), permutation.valid.raw[2]),
                vandq_u8(vqtbl1q_u8(lut, permuted.raw[3]), permutation.valid.raw[3]),
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
    permute_mailbox_inner(permutation, &mb)
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
        // None<Piece> is represented as 12 due to niche optimisation.
        let none_vec = vdupq_n_u8(12);
        let ignore_vec = vdupq_n_u8(ignore.inner());

        let mb = Vector::load(mailbox.as_ptr().cast::<u8>());
        let masked_mailbox = Vector {
            raw: [
                vbslq_u8(vceqq_u8(iota.raw[0], ignore_vec), none_vec, mb.raw[0]),
                vbslq_u8(vceqq_u8(iota.raw[1], ignore_vec), none_vec, mb.raw[1]),
                vbslq_u8(vceqq_u8(iota.raw[2], ignore_vec), none_vec, mb.raw[2]),
                vbslq_u8(vceqq_u8(iota.raw[3], ignore_vec), none_vec, mb.raw[3]),
            ],
        };
        permute_mailbox_inner(permutation, &masked_mailbox)
    }
}

pub fn closest_occupied(bits: Vector) -> BitRays {
    unsafe {
        let occupied_vec = Vector {
            raw: [
                vtstq_u8(bits.raw[0], bits.raw[0]),
                vtstq_u8(bits.raw[1], bits.raw[1]),
                vtstq_u8(bits.raw[2], bits.raw[2]),
                vtstq_u8(bits.raw[3], bits.raw[3]),
            ],
        };
        let occupied = occupied_vec.to_mask();
        let o = occupied | 0x8181_8181_8181_8181;
        (o ^ o.wrapping_sub(0x0303_0303_0303_0303)) & occupied
    }
}

pub fn incoming_attackers(bits: Vector, closest: BitRays) -> BitRays {
    unsafe {
        let mask = Vector::cast(&INCOMING_THREATS_MASK);
        let v = Vector {
            raw: [
                vtstq_u8(bits.raw[0], mask.raw[0]),
                vtstq_u8(bits.raw[1], mask.raw[1]),
                vtstq_u8(bits.raw[2], mask.raw[2]),
                vtstq_u8(bits.raw[3], mask.raw[3]),
            ],
        };
        v.to_mask() & closest
    }
}

pub fn incoming_sliders(bits: Vector, closest: BitRays) -> BitRays {
    unsafe {
        let mask = Vector::cast(&INCOMING_SLIDERS_MASK);
        let v = Vector {
            raw: [
                vtstq_u8(bits.raw[0], mask.raw[0]),
                vtstq_u8(bits.raw[1], mask.raw[1]),
                vtstq_u8(bits.raw[2], mask.raw[2]),
                vtstq_u8(bits.raw[3], mask.raw[3]),
            ],
        };
        v.to_mask() & closest & 0xFEFE_FEFE_FEFE_FEFE
    }
}

pub fn king_positions(bits: Vector) -> BitRays {
    unsafe {
        let king_mask = vdupq_n_u8(super::Bit::KING.0);
        let v = Vector {
            raw: [
                vtstq_u8(bits.raw[0], king_mask),
                vtstq_u8(bits.raw[1], king_mask),
                vtstq_u8(bits.raw[2], king_mask),
                vtstq_u8(bits.raw[3], king_mask),
            ],
        };
        v.to_mask()
    }
}
