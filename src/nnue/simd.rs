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
use super::network::{Align64, INPUT, LAYER_1_SIZE};

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
        #[cfg(target_feature = "avx512")] data: std::arch::x86_64::__m512i,
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512")))] data: std::arch::x86_64::__m256i,
        #[cfg(target_feature = "neon")] data: std::arch::aarch64::int16x8_t,
        #[cfg(not(any(target_feature = "avx512", target_feature = "avx2", target_feature = "neon")))] data: i16,
    ) -> Self {
        Self { data }
    }

    #[inline]
    pub unsafe fn load_at<const VEC_SIZE: usize>(memory: &Align64<[i16; VEC_SIZE]>, start_idx: usize) -> Self {
        #[cfg(target_feature = "avx512")]
        {
            Self { data: std::arch::x86_64::_mm512_load_si512(memory.0.as_ptr().add(start_idx).cast()) }
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512")))]
        {
            Self { data: std::arch::x86_64::_mm256_load_si256(memory.0.as_ptr().add(start_idx).cast()) }
        }
        #[cfg(target_feature = "neon")]
        {
            Self { data: std::arch::aarch64::vld1q_s16(memory.0.as_ptr().add(start_idx).cast()) }
        }
        #[cfg(not(any(target_feature = "avx512", target_feature = "avx2", target_feature = "neon")))]
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
        #[cfg(target_feature = "avx512")]
        {
            std::arch::x86_64::_mm512_store_si512(memory.0.as_mut_ptr().add(start_idx).cast(), value.data)
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512")))]
        {
            std::arch::x86_64::_mm256_store_si256(memory.0.as_mut_ptr().add(start_idx).cast(), value.data)
        }
        #[cfg(target_feature = "neon")]
        {
            std::arch::aarch64::vst1q_s16(memory.0.as_mut_ptr().add(start_idx).cast(), value.data)
        }
        #[cfg(not(any(target_feature = "avx512", target_feature = "avx2", target_feature = "neon")))]
        {
            *memory.get_unchecked_mut(start_idx) = value.data
        }
    }

    #[inline]
    pub unsafe fn min(a: Self, b: Self) -> Self {
        #[cfg(target_feature = "avx512")]
        {
            Self { data: std::arch::x86_64::_mm512_min_epi16(a.data, b.data) }
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512")))]
        {
            Self { data: std::arch::x86_64::_mm256_min_epi16(a.data, b.data) }
        }
        #[cfg(target_feature = "neon")]
        {
            Self { data: std::arch::aarch64::vminq_s16(a.data, b.data) }
        }
        #[cfg(not(any(target_feature = "avx512", target_feature = "avx2", target_feature = "neon")))]
        {
            Self { data: std::cmp::min(a.data, b.data) }
        }
    }

    #[inline]
    pub unsafe fn max(a: Self, b: Self) -> Self {
        #[cfg(target_feature = "avx512")]
        {
            Self { data: std::arch::x86_64::_mm512_max_epi16(a.data, b.data) }
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512")))]
        {
            Self { data: std::arch::x86_64::_mm256_max_epi16(a.data, b.data) }
        }
        #[cfg(target_feature = "neon")]
        {
            Self { data: std::arch::aarch64::vmaxq_s16(a.data, b.data) }
        }
        #[cfg(not(any(target_feature = "avx512", target_feature = "avx2", target_feature = "neon")))]
        {
            Self { data: std::cmp::max(a.data, b.data) }
        }
    }

    #[inline]
    pub unsafe fn add(a: Self, b: Self) -> Self {
        #[cfg(target_feature = "avx512")]
        {
            Self { data: std::arch::x86_64::_mm512_add_epi16(a.data, b.data) }
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512")))]
        {
            Self { data: std::arch::x86_64::_mm256_add_epi16(a.data, b.data) }
        }
        #[cfg(target_feature = "neon")]
        {
            Self { data: std::arch::aarch64::vaddq_s16(a.data, b.data) }
        }
        #[cfg(not(any(target_feature = "avx512", target_feature = "avx2", target_feature = "neon")))]
        {
            Self { data: a.data + b.data }
        }
    }

    #[inline]
    pub unsafe fn sub(a: Self, b: Self) -> Self {
        #[cfg(target_feature = "avx512")]
        {
            Self { data: std::arch::x86_64::_mm512_sub_epi16(a.data, b.data) }
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512")))]
        {
            Self { data: std::arch::x86_64::_mm256_sub_epi16(a.data, b.data) }
        }
        #[cfg(target_feature = "neon")]
        {
            Self { data: std::arch::aarch64::vsubq_s16(a.data, b.data) }
        }
        #[cfg(not(any(target_feature = "avx512", target_feature = "avx2", target_feature = "neon")))]
        {
            Self { data: a.data - b.data }
        }
    }

    #[inline]
    pub unsafe fn mul_truncating(a: Self, b: Self) -> Self {
        #[cfg(target_feature = "avx512")]
        {
            Self { data: std::arch::x86_64::_mm512_mullo_epi16(a.data, b.data) }
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512")))]
        {
            Self { data: std::arch::x86_64::_mm256_mullo_epi16(a.data, b.data) }
        }
        #[cfg(target_feature = "neon")]
        {
            Self { data: std::arch::aarch64::vmulq_s16(a.data, b.data) }
        }
        #[cfg(not(any(target_feature = "avx512", target_feature = "avx2", target_feature = "neon")))]
        {
            Self { data: a.data * b.data }
        }
    }

    #[inline]
    pub unsafe fn mul_widening(a: Self, b: Self) -> Vector32 {
        #[cfg(target_feature = "avx512")]
        {
            Vector32 { data: std::arch::x86_64::_mm512_madd_epi16(a.data, b.data) }
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512")))]
        {
            Vector32 { data: std::arch::x86_64::_mm256_madd_epi16(a.data, b.data) }
        }
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
        {
            Vector32 { data: i32::from(a.data) * i32::from(b.data) }
        }
    }

    #[inline]
    pub unsafe fn splat(value: i16) -> Self {
        #[cfg(target_feature = "avx512")]
        {
            Self { data: std::arch::x86_64::_mm512_set1_epi16(value) }
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512")))]
        {
            Self { data: std::arch::x86_64::_mm256_set1_epi16(value) }
        }
        #[cfg(target_feature = "neon")]
        {
            Self { data: std::arch::aarch64::vld1q_dup_s16(&value) }
        }
        #[cfg(not(any(target_feature = "avx512", target_feature = "avx2", target_feature = "neon")))]
        {
            Self { data: value }
        }
    }

    #[inline]
    pub unsafe fn zero() -> Self {
        #[cfg(target_feature = "avx512")]
        {
            Self { data: std::arch::x86_64::_mm512_setzero_si512() }
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512")))]
        {
            Self { data: std::arch::x86_64::_mm256_setzero_si256() }
        }
        #[cfg(target_feature = "neon")]
        {
            Self { data: std::arch::aarch64::vld1q_dup_s16(&0) }
        }
        #[cfg(not(any(target_feature = "avx512", target_feature = "avx2", target_feature = "neon")))]
        {
            Self { data: 0 }
        }
    }
}

impl Vector32 {
    pub const SIZE: usize = std::mem::size_of::<Self>();
    pub const COUNT: usize = Self::SIZE / std::mem::size_of::<i32>();

    #[inline]
    pub unsafe fn add(a: Self, b: Self) -> Self {
        #[cfg(target_feature = "avx512")]
        {
            Self { data: std::arch::x86_64::_mm512_add_epi32(a.data, b.data) }
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512")))]
        {
            Self { data: std::arch::x86_64::_mm256_add_epi32(a.data, b.data) }
        }
        #[cfg(target_feature = "neon")]
        {
            Self { data: std::arch::aarch64::vaddq_s32(a.data, b.data) }
        }
        #[cfg(not(any(target_feature = "avx512", target_feature = "avx2", target_feature = "neon")))]
        {
            Self { data: a.data + b.data }
        }
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
        {
            std::arch::aarch64::vaddlvq_s32(a.data) as i32
        }
        #[cfg(not(any(target_feature = "avx512", target_feature = "avx2", target_feature = "neon")))]
        {
            a.data
        }
    }

    #[inline]
    pub unsafe fn zero() -> Self {
        #[cfg(target_feature = "avx512")]
        {
            Self { data: std::arch::x86_64::_mm512_setzero_si512() }
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512")))]
        {
            Self { data: std::arch::x86_64::_mm256_setzero_si256() }
        }
        #[cfg(target_feature = "neon")]
        {
            Self { data: std::arch::aarch64::vld1q_dup_s32(&0) }
        }
        #[cfg(not(any(target_feature = "avx512", target_feature = "avx2", target_feature = "neon")))]
        {
            Self { data: 0 }
        }
    }
}

unsafe fn slice_to_aligned(slice: &[i16]) -> &Align64<[i16; LAYER_1_SIZE]> {
    unsafe {
        // don't immediately cast to Align64, as we want to check the alignment first.
        let ptr = slice.as_ptr();
        debug_assert_eq!(ptr.align_offset(64), 0);
        // alignments are sensible, so we can safely cast.
        #[allow(clippy::cast_ptr_alignment)]
        &*ptr.cast()
    }
}

/// Vector-accelerated memcopy.
#[inline]
pub fn copy(src: &Align64<[i16; LAYER_1_SIZE]>, dst: &mut Align64<[i16; LAYER_1_SIZE]>) {
    // hard to beat memcpy, isn't it.
    unsafe { std::ptr::copy_nonoverlapping(src, dst, 1) };
}

/// Apply add/subtract updates in place.
pub fn vector_update_inplace(
    input: &mut Align64<[i16; LAYER_1_SIZE]>,
    bucket: &Align64<[i16; INPUT * LAYER_1_SIZE]>,
    adds: &[usize],
    subs: &[usize],
) {
    unsafe {
        for &sub_index in subs {
            let sub_index = sub_index * LAYER_1_SIZE;
            let sub_block = slice_to_aligned(bucket.get_unchecked(sub_index..sub_index + LAYER_1_SIZE));
            for i in 0..LAYER_1_SIZE / Vector16::COUNT {
                let x = Vector16::load_at(input, i * Vector16::COUNT);
                let w = Vector16::load_at(sub_block, i * Vector16::COUNT);
                let r = Vector16::sub(x, w);
                Vector16::store_at(input, r, i * Vector16::COUNT);
            }
        }
        for &add_index in adds {
            let add_index = add_index * LAYER_1_SIZE;
            let add_block = slice_to_aligned(bucket.get_unchecked(add_index..add_index + LAYER_1_SIZE));
            for i in 0..LAYER_1_SIZE / Vector16::COUNT {
                let x = Vector16::load_at(input, i * Vector16::COUNT);
                let w = Vector16::load_at(add_block, i * Vector16::COUNT);
                let r = Vector16::add(x, w);
                Vector16::store_at(input, r, i * Vector16::COUNT);
            }
        }
    }
}

/// Move a feature from one square to another.
pub fn vector_add_sub(
    input: &Align64<[i16; LAYER_1_SIZE]>,
    output: &mut Align64<[i16; LAYER_1_SIZE]>,
    bucket: &Align64<[i16; INPUT * LAYER_1_SIZE]>,
    feature_idx_add: usize,
    feature_idx_sub: usize,
) {
    let offset_add = feature_idx_add * LAYER_1_SIZE;
    let offset_sub = feature_idx_sub * LAYER_1_SIZE;
    let s_block = unsafe { slice_to_aligned(&bucket[offset_sub..offset_sub + LAYER_1_SIZE]) };
    let a_block = unsafe { slice_to_aligned(&bucket[offset_add..offset_add + LAYER_1_SIZE]) };
    for i in 0..LAYER_1_SIZE / Vector16::COUNT {
        unsafe {
            let x = Vector16::load_at(input, i * Vector16::COUNT);
            let w_sub = Vector16::load_at(s_block, i * Vector16::COUNT);
            let w_add = Vector16::load_at(a_block, i * Vector16::COUNT);
            let t = Vector16::sub(x, w_sub);
            let t = Vector16::add(t, w_add);
            Vector16::store_at(output, t, i * Vector16::COUNT);
        }
    }
}

/// Subtract two features and add one feature all at once.
pub fn vector_add_sub2(
    input: &Align64<[i16; LAYER_1_SIZE]>,
    output: &mut Align64<[i16; LAYER_1_SIZE]>,
    bucket: &Align64<[i16; INPUT * LAYER_1_SIZE]>,
    feature_idx_add: usize,
    feature_idx_sub1: usize,
    feature_idx_sub2: usize,
) {
    let offset_add = feature_idx_add * LAYER_1_SIZE;
    let offset_sub1 = feature_idx_sub1 * LAYER_1_SIZE;
    let offset_sub2 = feature_idx_sub2 * LAYER_1_SIZE;
    let a_block = unsafe { slice_to_aligned(&bucket[offset_add..offset_add + LAYER_1_SIZE]) };
    let s_block1 = unsafe { slice_to_aligned(&bucket[offset_sub1..offset_sub1 + LAYER_1_SIZE]) };
    let s_block2 = unsafe { slice_to_aligned(&bucket[offset_sub2..offset_sub2 + LAYER_1_SIZE]) };
    for i in 0..LAYER_1_SIZE / Vector16::COUNT {
        unsafe {
            let x = Vector16::load_at(input, i * Vector16::COUNT);
            let w_sub1 = Vector16::load_at(s_block1, i * Vector16::COUNT);
            let w_sub2 = Vector16::load_at(s_block2, i * Vector16::COUNT);
            let w_add = Vector16::load_at(a_block, i * Vector16::COUNT);
            let t = Vector16::sub(x, w_sub1);
            let t = Vector16::sub(t, w_sub2);
            let t = Vector16::add(t, w_add);
            Vector16::store_at(output, t, i * Vector16::COUNT);
        }
    }
}

/// Add two features and subtract two features all at once.
pub fn vector_add2_sub2(
    input: &Align64<[i16; LAYER_1_SIZE]>,
    output: &mut Align64<[i16; LAYER_1_SIZE]>,
    bucket: &Align64<[i16; INPUT * LAYER_1_SIZE]>,
    feature_idx_add1: usize,
    feature_idx_add2: usize,
    feature_idx_sub1: usize,
    feature_idx_sub2: usize,
) {
    let offset_add1 = feature_idx_add1 * LAYER_1_SIZE;
    let offset_add2 = feature_idx_add2 * LAYER_1_SIZE;
    let offset_sub1 = feature_idx_sub1 * LAYER_1_SIZE;
    let offset_sub2 = feature_idx_sub2 * LAYER_1_SIZE;
    let a_block1 = unsafe { slice_to_aligned(&bucket[offset_add1..offset_add1 + LAYER_1_SIZE]) };
    let a_block2 = unsafe { slice_to_aligned(&bucket[offset_add2..offset_add2 + LAYER_1_SIZE]) };
    let s_block1 = unsafe { slice_to_aligned(&bucket[offset_sub1..offset_sub1 + LAYER_1_SIZE]) };
    let s_block2 = unsafe { slice_to_aligned(&bucket[offset_sub2..offset_sub2 + LAYER_1_SIZE]) };
    for i in 0..LAYER_1_SIZE / Vector16::COUNT {
        unsafe {
            let x = Vector16::load_at(input, i * Vector16::COUNT);
            let w_sub1 = Vector16::load_at(s_block1, i * Vector16::COUNT);
            let w_sub2 = Vector16::load_at(s_block2, i * Vector16::COUNT);
            let w_add1 = Vector16::load_at(a_block1, i * Vector16::COUNT);
            let w_add2 = Vector16::load_at(a_block2, i * Vector16::COUNT);
            let t = Vector16::sub(x, w_sub1);
            let t = Vector16::sub(t, w_sub2);
            let t = Vector16::add(t, w_add1);
            let t = Vector16::add(t, w_add2);
            Vector16::store_at(output, t, i * Vector16::COUNT);
        }
    }
}
