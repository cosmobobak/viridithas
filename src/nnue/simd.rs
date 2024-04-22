#![allow(clippy::all, clippy::nursery, clippy::pedantic, dead_code)]

use super::network::Align64;

#[derive(Clone, Copy)]
pub struct Vector16 {
    #[cfg(target_feature = "avx512")]
    data: std::arch::x86_64::__m512i,
    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512")))]
    data: std::arch::x86_64::__m256i,
    #[cfg(target_feature = "neon")]
    data: std::arch::aarch64::int16x8_t,
    #[cfg(not(any(target_feature = "avx512", target_feature = "avx2", target_feature = "neon")))]
    data: i16,
}

#[derive(Clone, Copy)]
pub struct Vector32 {
    #[cfg(target_feature = "avx512")]
    data: std::arch::x86_64::__m512i,
    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512")))]
    data: std::arch::x86_64::__m256i,
    #[cfg(target_feature = "neon")]
    data: std::arch::aarch64::int32x4_t,
    #[cfg(not(any(target_feature = "avx512", target_feature = "avx2", target_feature = "neon")))]
    data: i32,
}

impl Vector16 {
    pub const SIZE: usize = std::mem::size_of::<Self>();
    pub const COUNT: usize = Self::SIZE / std::mem::size_of::<i16>();

    pub fn new(
        #[cfg(target_feature = "avx512")]
        data: std::arch::x86_64::__m512i,
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512")))]
        data: std::arch::x86_64::__m256i,
        #[cfg(target_feature = "neon")]
        data: std::arch::aarch64::int16x8_t,
        #[cfg(not(any(target_feature = "avx512", target_feature = "avx2", target_feature = "neon")))]
        data: i16,
    ) -> Self {
        Self { data }
    }

    #[inline]
    pub unsafe fn load_at<const VEC_SIZE: usize>(memory: &Align64<[i16; VEC_SIZE]>, start_idx: usize) -> Self {
        #[cfg(target_feature = "avx512")]
        { Self { data: std::arch::x86_64::_mm512_load_si512(memory.0.as_ptr().add(start_idx).cast()) } }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512")))]
        { Self { data: std::arch::x86_64::_mm256_load_si256(memory.0.as_ptr().add(start_idx).cast()) } }
        #[cfg(target_feature = "neon")]
        { Self { data: std::arch::aarch64::vld1q_s16(memory.0.as_ptr().add(start_idx).cast()) } }
        #[cfg(not(any(target_feature = "avx512", target_feature = "avx2", target_feature = "neon")))]
        { Self { data: *memory.get_unchecked(start_idx) } }
    }

    #[inline]
    pub unsafe fn min(a: Self, b: Self) -> Self {
        #[cfg(target_feature = "avx512")]
        { Self { data: std::arch::x86_64::_mm512_min_epi16(a.data, b.data) } }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512")))]
        { Self { data: std::arch::x86_64::_mm256_min_epi16(a.data, b.data) } }
        #[cfg(target_feature = "neon")]
        { Self { data: std::arch::aarch64::vminq_s16(a.data, b.data) } }
        #[cfg(not(any(target_feature = "avx512", target_feature = "avx2", target_feature = "neon")))]
        { Self { data: std::cmp::min(a.data, b.data) } }
    }

    #[inline]
    pub unsafe fn max(a: Self, b: Self) -> Self {
        #[cfg(target_feature = "avx512")]
        { Self { data: std::arch::x86_64::_mm512_max_epi16(a.data, b.data) } }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512")))]
        { Self { data: std::arch::x86_64::_mm256_max_epi16(a.data, b.data) } }
        #[cfg(target_feature = "neon")]
        { Self { data: std::arch::aarch64::vmaxq_s16(a.data, b.data) } }
        #[cfg(not(any(target_feature = "avx512", target_feature = "avx2", target_feature = "neon")))]
        { Self { data: std::cmp::max(a.data, b.data) } }
    }

    #[inline]
    pub unsafe fn mul_truncating(a: Self, b: Self) -> Self {
        #[cfg(target_feature = "avx512")]
        { Self { data: std::arch::x86_64::_mm512_mullo_epi16(a.data, b.data) } }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512")))]
        { Self { data: std::arch::x86_64::_mm256_mullo_epi16(a.data, b.data) } }
        #[cfg(target_feature = "neon")]
        { Self { data: std::arch::aarch64::vmulq_s16(a.data, b.data) } }
        #[cfg(not(any(target_feature = "avx512", target_feature = "avx2", target_feature = "neon")))]
        { Self { data: a.data * b.data } }
    }

    #[inline]
    pub unsafe fn mul_widening(a: Self, b: Self) -> Vector32 {
        #[cfg(target_feature = "avx512")]
        { Vector32 { data: std::arch::x86_64::_mm512_madd_epi16(a.data, b.data) } }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512")))]
        { Vector32 { data: std::arch::x86_64::_mm256_madd_epi16(a.data, b.data) } }
        #[cfg(target_feature = "neon")]
        {
            let a_lo = std::arch::aarch64::vget_low_s16(a.data);
            let b_lo = std::arch::aarch64::vget_low_s16(b.data);
            let a_hi = std::arch::aarch64::vget_high_s16(a.data);
            let b_hi = std::arch::aarch64::vget_high_s16(b.data);
            let product_lo = std::arch::aarch64::vmull_s16(a_lo, b_lo);
            let product_hi = std::arch::aarch64::vmull_s16(a_hi, b_hi);
            Vector32 { data: std::arch::aarch64::vaddq_s32(product_lo, product_hi) }
        }
        #[cfg(not(any(target_feature = "avx512", target_feature = "avx2", target_feature = "neon")))]
        { Vector32 { data: i32::from(a.data) * i32::from(b.data) } }
    }

    #[inline]
    pub unsafe fn splat(value: i16) -> Self {
        #[cfg(target_feature = "avx512")]
        { Self { data: std::arch::x86_64::_mm512_set1_epi16(value) } }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512")))]
        { Self { data: std::arch::x86_64::_mm256_set1_epi16(value) } }
        #[cfg(target_feature = "neon")]
        { Self { data: std::arch::aarch64::vld1q_dup_s16(&value) } }
        #[cfg(not(any(target_feature = "avx512", target_feature = "avx2", target_feature = "neon")))]
        { Self { data: value } }
    }

    #[inline]
    pub unsafe fn zero() -> Self {
        #[cfg(target_feature = "avx512")]
        { Self { data: std::arch::x86_64::_mm512_setzero_si512() } }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512")))]
        { Self { data: std::arch::x86_64::_mm256_setzero_si256() } }
        #[cfg(target_feature = "neon")]
        { Self { data: std::arch::aarch64::vld1q_dup_s16(&0) } }
        #[cfg(not(any(target_feature = "avx512", target_feature = "avx2", target_feature = "neon")))]
        { Self { data: 0 } }
    }
}

impl Vector32 {
    pub const SIZE: usize = std::mem::size_of::<Self>();
    pub const COUNT: usize = Self::SIZE / std::mem::size_of::<i32>();

    #[inline]
    pub unsafe fn add(a: Self, b: Self) -> Self {
        #[cfg(target_feature = "avx512")]
        { Self { data: std::arch::x86_64::_mm512_add_epi32(a.data, b.data) } }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512")))]
        { Self { data: std::arch::x86_64::_mm256_add_epi32(a.data, b.data) } }
        #[cfg(target_feature = "neon")]
        { Self { data: std::arch::aarch64::vaddq_s32(a.data, b.data) } }
        #[cfg(not(any(target_feature = "avx512", target_feature = "avx2", target_feature = "neon")))]
        { Self { data: a.data + b.data } }
    }

    #[inline]
    pub unsafe fn sum(a: Self) -> i32 {
        #[cfg(target_feature = "avx512")]
        {
            let high_256 = std::arch::x86_64::_mm512_extracti64x4_epi64(a.data);
            let low_256 = std::arch::x86_64::_mm512_castsi512_si256(a.data);
            let sum_256 = std::arch::x86_64::_mm256_add_epi32(low_256, high_256);
            let upper_128 = std::arch::x86_64::_mm256_extracti128_si256::<1>(sum_256);
            let lower_128 = std::arch::x86_64::_mm256_castsi256_si128(sum_256);
            let sum_128 = std::arch::x86_64::_mm_add_epi32(upper_128, lower_128);
            let upper_64 = std::arch::x86_64::_mm_unpackhi_epi64(sum_128, sum_128);
            let sum_64 = std::arch::x86_64::_mm_add_epi32(upper_64, sum_128);
            let upper_32 = std::arch::x86_64::_mm_shuffle_epi32::<0b00_00_00_01>(sum_64);
            let sum_32 = std::arch::x86_64::_mm_add_epi32(upper_32, sum_64);

            std::arch::x86_64::_mm_cvtsi128_si32(sum_32)
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512")))]
        {
            let upper_128 = std::arch::x86_64::_mm256_extracti128_si256::<1>(a.data);
            let lower_128 = std::arch::x86_64::_mm256_castsi256_si128(a.data);
            let sum_128 = std::arch::x86_64::_mm_add_epi32(upper_128, lower_128);
            let upper_64 = std::arch::x86_64::_mm_unpackhi_epi64(sum_128, sum_128);
            let sum_64 = std::arch::x86_64::_mm_add_epi32(upper_64, sum_128);
            let upper_32 = std::arch::x86_64::_mm_shuffle_epi32::<0b00_00_00_01>(sum_64);
            let sum_32 = std::arch::x86_64::_mm_add_epi32(upper_32, sum_64);

            std::arch::x86_64::_mm_cvtsi128_si32(sum_32)
        }
        #[cfg(target_feature = "neon")]
        { std::arch::aarch64::vaddlvq_s32(a.data) as i32 }
        #[cfg(not(any(target_feature = "avx512", target_feature = "avx2", target_feature = "neon")))]
        { a.data }
    }

    #[inline]
    pub unsafe fn zero() -> Self {
        #[cfg(target_feature = "avx512")]
        { Self { data: std::arch::x86_64::_mm512_setzero_si512() } }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512")))]
        { Self { data: std::arch::x86_64::_mm256_setzero_si256() } }
        #[cfg(target_feature = "neon")]
        { Self { data: std::arch::aarch64::vld1q_dup_s32(&0) } }
        #[cfg(not(any(target_feature = "avx512", target_feature = "avx2", target_feature = "neon")))]
        { Self { data: 0 } }
    }
}