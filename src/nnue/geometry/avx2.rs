#![expect(clippy::cast_sign_loss, clippy::undocumented_unsafe_blocks)]

use std::arch::x86_64::*;

use crate::nnue::geometry::BitRays;

#[repr(transparent)]
struct Vector {
    raw: [__m256i; 2],
}

impl Vector {
    fn flip(self) -> Self {
        Self {
            raw: [self.raw[1], self.raw[0]],
        }
    }

    fn mask(self) -> BitRays {
        unsafe {
            let a = _mm256_movemask_epi8(self.raw[0]) as u64;
            let b = _mm256_movemask_epi8(self.raw[1]) as u64;
            BitRays(a | b << 32)
        }
    }

    unsafe fn load(ptr: *const u8) -> Self {
        #![expect(clippy::cast_ptr_alignment)]
        Self {
            raw: unsafe {
                [
                    _mm256_loadu_si256(ptr.cast::<__m256i>().add(0)),
                    _mm256_loadu_si256(ptr.cast::<__m256i>().add(1)),
                ]
            },
        }
    }
}
