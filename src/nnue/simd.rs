#![allow(clippy::all, clippy::nursery, clippy::pedantic, dead_code)]

/////////////////////////////////////////////////////////////////////////////////////////////////////////
///                                                                                                   ///
///    This place is a message... and part of a system of messages... pay attention to it!            ///
///                                                                                                   ///
///    Sending this message was important to us. We considered ourselves to be a powerful culture.    ///
///                                                                                                   ///
///    This place is not a place of honor...                                                          ///
///    no highly esteemed deed is commemorated here...                                                ///
///    nothing valued is here.                                                                        ///
///                                                                                                   ///
///    What is here was dangerous and repulsive to us. This message is a warning about danger.        ///
///                                                                                                   ///
///    The danger is in a particular location... it increases towards a center...                     ///
///    the center of danger is here... of a particular size and shape, and below us.                  ///
///                                                                                                   ///
///    The danger is still present, in your time, as it was in ours.                                  ///
///                                                                                                   ///
///    The danger is to your memory, and it can kill.                                                 ///
///                                                                                                   ///
///    The form of the danger is an emanation of SIMD.                                                ///
///                                                                                                   ///
///    The danger is unleashed only if you substantially disturb this place physically.               ///
///    This place is best shunned and left uninhabited.                                               ///
///                                                                                                   ///
/////////////////////////////////////////////////////////////////////////////////////////////////////////
use super::network::Align64;

#[derive(Clone, Copy)]
pub struct VectorI16 {
    #[cfg(target_feature = "avx512f")]
    data: std::arch::x86_64::__m512i,
    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
    data: std::arch::x86_64::__m256i,
    #[cfg(target_feature = "neon")]
    data: std::arch::aarch64::int16x8_t,
    #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2", target_feature = "neon")))]
    data: i16,
}

impl VectorI16 {
    pub const SIZE: usize = std::mem::size_of::<Self>();
    pub const COUNT: usize = Self::SIZE / std::mem::size_of::<i16>();

    pub fn new(
        #[cfg(target_feature = "avx512f")] data: std::arch::x86_64::__m512i,
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))] data: std::arch::x86_64::__m256i,
        #[cfg(target_feature = "neon")] data: std::arch::aarch64::int16x8_t,
        #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2", target_feature = "neon")))] data: i16,
    ) -> Self {
        Self { data }
    }

    #[inline]
    pub unsafe fn load_at<const VEC_SIZE: usize>(memory: &Align64<[i16; VEC_SIZE]>, start_idx: usize) -> Self {
        debug_assert!(start_idx % Self::COUNT == 0);
        debug_assert!(start_idx + Self::COUNT <= memory.0.len());
        #[cfg(target_feature = "avx512f")]
        {
            Self { data: std::arch::x86_64::_mm512_load_si512(memory.0.as_ptr().add(start_idx).cast()) }
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
        {
            Self { data: std::arch::x86_64::_mm256_load_si256(memory.0.as_ptr().add(start_idx).cast()) }
        }
        #[cfg(target_feature = "neon")]
        {
            Self { data: std::arch::aarch64::vld1q_s16(memory.0.as_ptr().add(start_idx).cast()) }
        }
        #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2", target_feature = "neon")))]
        {
            Self { data: *memory.get_unchecked(start_idx) }
        }
    }

    #[inline]
    pub unsafe fn store_at<const VEC_SIZE: usize>(
        memory: &mut Align64<[i16; VEC_SIZE]>,
        value: Self,
        start_idx: usize,
    ) {
        debug_assert!(start_idx % Self::COUNT == 0);
        debug_assert!(start_idx + Self::COUNT <= memory.0.len());
        #[cfg(target_feature = "avx512f")]
        {
            std::arch::x86_64::_mm512_store_si512(memory.0.as_mut_ptr().add(start_idx).cast(), value.data)
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
        {
            std::arch::x86_64::_mm256_store_si256(memory.0.as_mut_ptr().add(start_idx).cast(), value.data)
        }
        #[cfg(target_feature = "neon")]
        {
            std::arch::aarch64::vst1q_s16(memory.0.as_mut_ptr().add(start_idx).cast(), value.data)
        }
        #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2", target_feature = "neon")))]
        {
            *memory.get_unchecked_mut(start_idx) = value.data
        }
    }

    #[inline]
    pub unsafe fn min(a: Self, b: Self) -> Self {
        #[cfg(target_feature = "avx512f")]
        {
            Self { data: std::arch::x86_64::_mm512_min_epi16(a.data, b.data) }
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
        {
            Self { data: std::arch::x86_64::_mm256_min_epi16(a.data, b.data) }
        }
        #[cfg(target_feature = "neon")]
        {
            Self { data: std::arch::aarch64::vminq_s16(a.data, b.data) }
        }
        #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2", target_feature = "neon")))]
        {
            Self { data: std::cmp::min(a.data, b.data) }
        }
    }

    #[inline]
    pub unsafe fn max(a: Self, b: Self) -> Self {
        #[cfg(target_feature = "avx512f")]
        {
            Self { data: std::arch::x86_64::_mm512_max_epi16(a.data, b.data) }
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
        {
            Self { data: std::arch::x86_64::_mm256_max_epi16(a.data, b.data) }
        }
        #[cfg(target_feature = "neon")]
        {
            Self { data: std::arch::aarch64::vmaxq_s16(a.data, b.data) }
        }
        #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2", target_feature = "neon")))]
        {
            Self { data: std::cmp::max(a.data, b.data) }
        }
    }

    #[inline]
    pub unsafe fn add(a: Self, b: Self) -> Self {
        #[cfg(target_feature = "avx512f")]
        {
            Self { data: std::arch::x86_64::_mm512_add_epi16(a.data, b.data) }
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
        {
            Self { data: std::arch::x86_64::_mm256_add_epi16(a.data, b.data) }
        }
        #[cfg(target_feature = "neon")]
        {
            Self { data: std::arch::aarch64::vaddq_s16(a.data, b.data) }
        }
        #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2", target_feature = "neon")))]
        {
            Self { data: a.data + b.data }
        }
    }

    #[inline]
    pub unsafe fn sub(a: Self, b: Self) -> Self {
        #[cfg(target_feature = "avx512f")]
        {
            Self { data: std::arch::x86_64::_mm512_sub_epi16(a.data, b.data) }
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
        {
            Self { data: std::arch::x86_64::_mm256_sub_epi16(a.data, b.data) }
        }
        #[cfg(target_feature = "neon")]
        {
            Self { data: std::arch::aarch64::vsubq_s16(a.data, b.data) }
        }
        #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2", target_feature = "neon")))]
        {
            Self { data: a.data - b.data }
        }
    }

    #[inline]
    pub unsafe fn mul_truncating(a: Self, b: Self) -> Self {
        #[cfg(target_feature = "avx512f")]
        {
            Self { data: std::arch::x86_64::_mm512_mullo_epi16(a.data, b.data) }
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
        {
            Self { data: std::arch::x86_64::_mm256_mullo_epi16(a.data, b.data) }
        }
        #[cfg(target_feature = "neon")]
        {
            Self { data: std::arch::aarch64::vmulq_s16(a.data, b.data) }
        }
        #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2", target_feature = "neon")))]
        {
            Self { data: a.data * b.data }
        }
    }

    #[inline]
    pub unsafe fn mul_widening(a: Self, b: Self) -> VectorI32 {
        #[cfg(target_feature = "avx512f")]
        {
            VectorI32 { data: std::arch::x86_64::_mm512_madd_epi16(a.data, b.data) }
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
        {
            VectorI32 { data: std::arch::x86_64::_mm256_madd_epi16(a.data, b.data) }
        }
        #[cfg(target_feature = "neon")]
        {
            // lo
            let a_lo = std::arch::aarch64::vget_low_s16(a.data);
            let b_lo = std::arch::aarch64::vget_low_s16(b.data);
            let lo_prod = std::arch::aarch64::vmull_s16(b_lo, a_lo);
            // hi
            let hi_prod = std::arch::aarch64::vmull_high_s16(a.data, b.data);
            // sum
            let data = std::arch::aarch64::vpaddq_s32(lo_prod, hi_prod);
            VectorI32 { data }
        }
        #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2", target_feature = "neon")))]
        {
            VectorI32 { data: i32::from(a.data) * i32::from(b.data) }
        }
    }

    #[inline]
    pub unsafe fn splat(value: i16) -> Self {
        #[cfg(target_feature = "avx512f")]
        {
            Self { data: std::arch::x86_64::_mm512_set1_epi16(value) }
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
        {
            Self { data: std::arch::x86_64::_mm256_set1_epi16(value) }
        }
        #[cfg(target_feature = "neon")]
        {
            Self { data: std::arch::aarch64::vld1q_dup_s16(&value) }
        }
        #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2", target_feature = "neon")))]
        {
            Self { data: value }
        }
    }

    #[inline]
    pub fn zero() -> Self {
        // SAFETY: All of these functions are actually perfectly fine to call on their own.
        #[allow(unused_unsafe)]
        unsafe {
            #[cfg(target_feature = "avx512f")]
            {
                Self { data: std::arch::x86_64::_mm512_setzero_si512() }
            }
            #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
            {
                Self { data: std::arch::x86_64::_mm256_setzero_si256() }
            }
            #[cfg(target_feature = "neon")]
            {
                Self { data: std::arch::aarch64::vld1q_dup_s16(&0) }
            }
            #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2", target_feature = "neon")))]
            {
                Self { data: 0 }
            }
        }
    }
}

#[derive(Clone, Copy)]
pub struct VectorI32 {
    #[cfg(target_feature = "avx512f")]
    data: std::arch::x86_64::__m512i,
    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
    data: std::arch::x86_64::__m256i,
    #[cfg(target_feature = "neon")]
    data: std::arch::aarch64::int32x4_t,
    #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2", target_feature = "neon")))]
    data: i32,
}

impl VectorI32 {
    pub const SIZE: usize = std::mem::size_of::<Self>();
    pub const COUNT: usize = Self::SIZE / std::mem::size_of::<i32>();

    #[inline]
    pub unsafe fn add(a: Self, b: Self) -> Self {
        #[cfg(target_feature = "avx512f")]
        {
            Self { data: std::arch::x86_64::_mm512_add_epi32(a.data, b.data) }
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
        {
            Self { data: std::arch::x86_64::_mm256_add_epi32(a.data, b.data) }
        }
        #[cfg(target_feature = "neon")]
        {
            Self { data: std::arch::aarch64::vaddq_s32(a.data, b.data) }
        }
        #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2", target_feature = "neon")))]
        {
            Self { data: a.data + b.data }
        }
    }

    #[inline]
    pub unsafe fn sum(a: Self) -> i32 {
        #[cfg(target_feature = "avx512f")]
        {
            let high_256 = std::arch::x86_64::_mm512_extracti64x4_epi64::<1>(a.data);
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
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
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
        {
            std::arch::aarch64::vaddlvq_s32(a.data) as i32
        }
        #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2", target_feature = "neon")))]
        {
            a.data
        }
    }

    #[inline]
    pub unsafe fn zero() -> Self {
        #[cfg(target_feature = "avx512f")]
        {
            Self { data: std::arch::x86_64::_mm512_setzero_si512() }
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
        {
            Self { data: std::arch::x86_64::_mm256_setzero_si256() }
        }
        #[cfg(target_feature = "neon")]
        {
            Self { data: std::arch::aarch64::vld1q_dup_s32(&0) }
        }
        #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2", target_feature = "neon")))]
        {
            Self { data: 0 }
        }
    }
}

#[derive(Clone, Copy)]
pub struct VectorF32 {
    #[cfg(target_feature = "avx512f")]
    data: std::arch::x86_64::__m512,
    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
    data: std::arch::x86_64::__m256,
    #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2")))]
    data: f32,
}

impl VectorF32 {
    pub const SIZE: usize = std::mem::size_of::<Self>();
    pub const COUNT: usize = Self::SIZE / std::mem::size_of::<f32>();

    #[inline]
    pub unsafe fn load_at<const VEC_SIZE: usize>(memory: &Align64<[f32; VEC_SIZE]>, start_idx: usize) -> Self {
        debug_assert!(start_idx % Self::COUNT == 0);
        debug_assert!(start_idx + Self::COUNT <= memory.0.len());
        #[cfg(target_feature = "avx512f")]
        {
            Self { data: std::arch::x86_64::_mm512_load_ps(memory.0.as_ptr().add(start_idx).cast()) }
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
        {
            Self { data: std::arch::x86_64::_mm256_load_ps(memory.0.as_ptr().add(start_idx).cast()) }
        }
        #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2")))]
        {
            Self { data: *memory.get_unchecked(start_idx) }
        }
    }

    #[inline]
    pub unsafe fn store_at<const VEC_SIZE: usize>(
        memory: &mut Align64<[f32; VEC_SIZE]>,
        value: Self,
        start_idx: usize,
    ) {
        debug_assert!(start_idx % Self::COUNT == 0);
        debug_assert!(start_idx + Self::COUNT <= memory.0.len());
        #[cfg(target_feature = "avx512f")]
        {
            std::arch::x86_64::_mm512_store_ps(memory.0.as_mut_ptr().add(start_idx).cast(), value.data)
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
        {
            std::arch::x86_64::_mm256_store_ps(memory.0.as_mut_ptr().add(start_idx).cast(), value.data)
        }
        #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2")))]
        {
            *memory.get_unchecked_mut(start_idx) = value.data
        }
    }

    #[inline]
    pub fn zero() -> Self {
        #[cfg(target_feature = "avx512f")]
        {
            Self { data: std::arch::x86_64::_mm512_setzero_ps() }
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
        {
            Self { data: std::arch::x86_64::_mm256_setzero_ps() }
        }
        #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2")))]
        {
            Self { data: 0.0 }
        }
    }

    #[inline]
    pub unsafe fn splat(value: f32) -> Self {
        #[cfg(target_feature = "avx512f")]
        {
            Self { data: std::arch::x86_64::_mm512_set1_ps(value) }
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
        {
            Self { data: std::arch::x86_64::_mm256_set1_ps(value) }
        }
        #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2")))]
        {
            Self { data: value }
        }
    }

    #[inline]
    pub unsafe fn from_i32s(value: VectorI32) -> Self {
        #[cfg(target_feature = "avx512f")]
        {
            Self { data: std::arch::x86_64::_mm512_cvtepi32_ps(value.data) }
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
        {
            Self { data: std::arch::x86_64::_mm256_cvtepi32_ps(value.data) }
        }
        #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2")))]
        {
            Self { data: value.data as f32 }
        }
    }

    #[inline]
    pub unsafe fn div(a: Self, b: Self) -> Self {
        #[cfg(target_feature = "avx512f")]
        {
            Self { data: std::arch::x86_64::_mm512_div_ps(a.data, b.data) }
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
        {
            Self { data: std::arch::x86_64::_mm256_div_ps(a.data, b.data) }
        }
        #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2")))]
        {
            Self { data: a.data / b.data }
        }
    }

    #[inline]
    pub unsafe fn mul(a: Self, b: Self) -> Self {
        #[cfg(target_feature = "avx512f")]
        {
            Self { data: std::arch::x86_64::_mm512_mul_ps(a.data, b.data) }
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
        {
            Self { data: std::arch::x86_64::_mm256_mul_ps(a.data, b.data) }
        }
        #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2")))]
        {
            Self { data: a.data * b.data }
        }
    }

    #[inline]
    pub unsafe fn add(a: Self, b: Self) -> Self {
        #[cfg(target_feature = "avx512f")]
        {
            Self { data: std::arch::x86_64::_mm512_add_ps(a.data, b.data) }
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
        {
            Self { data: std::arch::x86_64::_mm256_add_ps(a.data, b.data) }
        }
        #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2")))]
        {
            Self { data: a.data + b.data }
        }
    }

    #[inline]
    pub unsafe fn max(a: Self, b: Self) -> Self {
        #[cfg(target_feature = "avx512f")]
        {
            Self { data: std::arch::x86_64::_mm512_max_ps(a.data, b.data) }
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
        {
            Self { data: std::arch::x86_64::_mm256_max_ps(a.data, b.data) }
        }
        #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2")))]
        {
            Self { data: a.data.max(b.data) }
        }
    }

    #[inline]
    pub unsafe fn min(a: Self, b: Self) -> Self {
        #[cfg(target_feature = "avx512f")]
        {
            Self { data: std::arch::x86_64::_mm512_min_ps(a.data, b.data) }
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
        {
            Self { data: std::arch::x86_64::_mm256_min_ps(a.data, b.data) }
        }
        #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2")))]
        {
            Self { data: a.data.min(b.data) }
        }
    }
}

#[derive(Clone, Copy)]
pub struct VectorI8 {
    #[cfg(target_feature = "avx512f")]
    data: std::arch::x86_64::__m512i,
    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
    data: std::arch::x86_64::__m256i,
    #[cfg(target_feature = "neon")]
    data: std::arch::aarch64::int8x8_t,
    #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2", target_feature = "neon")))]
    data: i8,
}

#[cfg(target_feature = "avx512f")]
mod avx512 {
    use std::arch::x86_64::*;

    type vepi8 = __m512i;
    type vepi16 = __m512i;
    type vepi32 = __m512i;
    type vps32 = __m512;

    #[inline]
    unsafe fn vec_zero_epi16() -> vepi16 { _mm512_setzero_si512() }
    #[inline]
    unsafe fn vec_zero_epi32() -> vepi32 { _mm512_setzero_si512() }
    #[inline]
    unsafe fn vec_set1_epi16(value: i16) -> vepi16 { _mm512_set1_epi16(value) }
    #[inline]
    unsafe fn vec_load_epi(src: &vepi16) -> vepi16 { _mm512_load_si512(src) }
    #[inline]
    unsafe fn vec_store_epi(dst: &mut vepi16, src: vepi16) { _mm512_store_si512(dst, src) }
    #[inline]
    unsafe fn vec_max_epi16(a: vepi16, b: vepi16) -> vepi16 { _mm512_max_epi16(a, b) }
    #[inline]
    unsafe fn vec_min_epi16(a: vepi16, b: vepi16) -> vepi16 { _mm512_min_epi16(a, b) }
    #[inline]
    unsafe fn vec_mullo_epi16(a: vepi16, b: vepi16) -> vepi16 { _mm512_mullo_epi16(a, b) }
    #[inline]
    unsafe fn vec_srli_epi16(a: vepi16, shift: i32) -> vepi16 { _mm512_srli_epi16(a, shift) }
    #[inline]
    unsafe fn vec_packus_permute_epi16(a: vepi16, b: vepi16) -> vepi8 {
        let packed = _mm512_packus_epi16(a, b);
        _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), packed)
    }

    #[inline]
    unsafe fn vec_dpbusd_epi32(sum: vepi32, a: vepi8, b: vepi8) -> vepi32 {
        #[cfg(target_feature = "avx512vnni")]
        {
            _mm512_dpbusd_epi32(sum, a, b)
        }
        #[cfg(not(target_feature = "avx512vnni"))]
        {
            let product16 = _mm512_maddubs_epi16(a, b);
            let product32 = _mm512_madd_epi16(product16, _mm512_set1_epi16(1));
            _mm512_add_epi32(sum, product32)
        }
    }

    #[inline]
    unsafe fn vec_set1_epi32(value: i32) -> vepi32 { _mm512_set1_epi32(value) }
    #[inline]
    unsafe fn vec_cvtepi32_ps(a: vepi32) -> vps32 { _mm512_cvtepi32_ps(a) }

    #[inline]
    unsafe fn vec_zero_ps() -> vps32 { _mm512_setzero_ps() }
    #[inline]
    unsafe fn vec_set1_ps(value: f32) -> vps32 { _mm512_set1_ps(value) }
    #[inline]
    unsafe fn vec_load_ps(src: &vps32) -> vps32 { _mm512_load_ps(src) }
    #[inline]
    unsafe fn vec_store_ps(dst: &mut vps32, src: vps32) { _mm512_store_ps(dst, src) }
    #[inline]
    unsafe fn vec_add_ps(a: vps32, b: vps32) -> vps32 { _mm512_add_ps(a, b) }
    #[inline]
    unsafe fn vec_mul_ps(a: vps32, b: vps32) -> vps32 { _mm512_mul_ps(a, b) }
    #[inline]
    unsafe fn vec_div_ps(a: vps32, b: vps32) -> vps32 { _mm512_div_ps(a, b) }
    #[inline]
    unsafe fn vec_min_ps(a: vps32, b: vps32) -> vps32 { _mm512_min_ps(a, b) }
    #[inline]
    unsafe fn vec_max_ps(a: vps32, b: vps32) -> vps32 { _mm512_max_ps(a, b) }
    #[inline]
    unsafe fn vec_mul_add_ps(a: vps32, b: vps32, c: vps32) -> vps32 { _mm512_fmadd_ps(a, b, c) }
    #[inline]
    unsafe fn vec_reduce_add_ps(a: vps32) -> f32 { _mm512_reduce_add_ps(a) }
}

#[cfg(target_feature = "avx2")]
mod avx2 {
    use std::arch::x86_64::*;

    type vepi8 = __m256i;
    type vepi16 = __m256i;
    type vepi32 = __m256i;
    type vps32 = __m256;

    #[inline]
    unsafe fn vec_zero_epi16() -> vepi16 { _mm256_setzero_si256() }
    #[inline]
    unsafe fn vec_zero_epi32() -> vepi32 { _mm256_setzero_si256() }
    #[inline]
    unsafe fn vec_set1_epi16(value: i16) -> vepi16 { _mm256_set1_epi16(value) }
    #[inline]
    unsafe fn vec_load_epi(src: &vepi16) -> vepi16 { _mm256_load_si256(src) }
    #[inline]
    unsafe fn vec_store_epi(dst: &mut vepi16, src: vepi16) { _mm256_store_si256(dst, src) }
    #[inline]
    unsafe fn vec_max_epi16(a: vepi16, b: vepi16) -> vepi16 { _mm256_max_epi16(a, b) }
    #[inline]
    unsafe fn vec_min_epi16(a: vepi16, b: vepi16) -> vepi16 { _mm256_min_epi16(a, b) }
    #[inline]
    unsafe fn vec_mullo_epi16(a: vepi16, b: vepi16) -> vepi16 { _mm256_mullo_epi16(a, b) }
    #[inline]
    unsafe fn vec_srli_epi16(a: vepi16, shift: i32) -> vepi16 { _mm256_srli_epi16(a, shift) }
    #[inline]
    unsafe fn vec_packus_permute_epi16(a: vepi16, b: vepi16) -> vepi8 {
        let packed = _mm256_packus_epi16(a, b);
        _mm256_permute4x64_epi64(packed, _MM_SHUFFLE(3, 1, 2, 0))
    }

    #[inline]
    unsafe fn vec_dpbusd_epi32(sum: vepi32, a: vepi8, b: vepi8) -> vepi32 {
        let product16 = _mm256_maddubs_epi16(a, b);
        let product32 = _mm256_madd_epi16(product16, _mm256_set1_epi16(1));
        _mm256_add_epi32(sum, product32)
    }

    #[inline]
    unsafe fn vec_set1_epi32(value: i32) -> vepi32 { _mm256_set1_epi32(value) }

    #[inline]
    unsafe fn vec_cvtepi32_ps(value: vepi32) -> vps32 { _mm256_cvtepi32_ps(value) }

    #[inline]
    unsafe fn vec_zero_ps() -> vps32 { _mm256_setzero_ps() }
    #[inline]
    unsafe fn vec_set1_ps(value: f32) -> vps32 { _mm256_set1_ps(value) }
    #[inline]
    unsafe fn vec_load_ps(src: &f32) -> vps32 { _mm256_load_ps(src) }
    #[inline]
    unsafe fn vec_store_ps(dst: &mut f32, src: vps32) { _mm256_store_ps(dst, src) }
    #[inline]
    unsafe fn vec_add_ps(a: vps32, b: vps32) -> vps32 { _mm256_add_ps(a, b) }
    #[inline]
    unsafe fn vec_mul_ps(a: vps32, b: vps32) -> vps32 { _mm256_mul_ps(a, b) }
    #[inline]
    unsafe fn vec_div_ps(a: vps32, b: vps32) -> vps32 { _mm256_div_ps(a, b) }
    #[inline]
    unsafe fn vec_min_ps(a: vps32, b: vps32) -> vps32 { _mm256_min_ps(a, b) }
    #[inline]
    unsafe fn vec_max_ps(a: vps32, b: vps32) -> vps32 { _mm256_max_ps(a, b) }
    #[inline]
    unsafe fn vec_mul_add_ps(a: vps32, b: vps32, c: vps32) -> vps32 { _mm256_fmadd_ps(a, b, c) }
    #[inline]
    unsafe fn vec_reduce_add_ps(a: vps32) -> f32 { _mm256_reduce_add_ps(a) }
}