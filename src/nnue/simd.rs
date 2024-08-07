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

#[inline]
pub const fn mm_shuffle(z: i32, y: i32, x: i32, w: i32) -> i32 {
    ((z) << 6) | ((y) << 4) | ((x) << 2) | (w)
}

#[cfg(target_feature = "avx2")]
mod avx2 {
    #![allow(non_camel_case_types)]
    use std::arch::x86_64::*;

    pub type vepi8 = __m256i;
    pub type vepi16 = __m256i;
    pub type vepi32 = __m256i;
    pub type vepi64 = __m256i;

    pub type vps32 = __m256;

    #[inline] pub unsafe fn vec_zero_epi16() -> vepi16 { return _mm256_setzero_si256(); }
    #[inline] pub unsafe fn vec_zero_epi32() -> vepi32 { return _mm256_setzero_si256(); }
    #[inline] pub unsafe fn vec_set1_epi16(n: i16)  -> vepi16 { return _mm256_set1_epi16(n); }
    #[inline] pub unsafe fn vec_set1_epi32(n: i32)  -> vepi32 { return _mm256_set1_epi32(n); }
    #[inline] pub unsafe fn vec_load_epi8(src: &i8) -> vepi8 { 
        // check alignment in debug mode
        debug_assert!((std::ptr::from_ref(src) as usize) % std::mem::align_of::<vepi16>() == 0);
        return _mm256_load_si256(std::ptr::from_ref(src).cast());
    }
    #[inline] pub unsafe fn vec_store_epi8(dst: &mut i8, vec: vepi8) { 
        // check alignment in debug mode
        debug_assert!((std::ptr::from_ref(dst) as usize) % std::mem::align_of::<vepi8>() == 0);
        _mm256_store_si256(std::ptr::from_mut(dst).cast(), vec);
    }
    #[inline] pub unsafe fn vec_load_epiu8(src: &u8) -> vepi8 { 
        // check alignment in debug mode
        debug_assert!((std::ptr::from_ref(src) as usize) % std::mem::align_of::<vepi16>() == 0);
        return _mm256_load_si256(std::ptr::from_ref(src).cast());
    }
    #[inline] pub unsafe fn vec_store_epiu8(dst: &mut u8, vec: vepi8) { 
        // check alignment in debug mode
        debug_assert!((std::ptr::from_ref(dst) as usize) % std::mem::align_of::<vepi8>() == 0);
        _mm256_store_si256(std::ptr::from_mut(dst).cast(), vec);
    }
    #[inline] pub unsafe fn vec_load_epi16(src: &i16) -> vepi16 { 
        // check alignment in debug mode
        debug_assert!((std::ptr::from_ref(src) as usize) % std::mem::align_of::<vepi16>() == 0);
        return _mm256_load_si256(std::ptr::from_ref(src).cast());
    }
    #[inline] pub unsafe fn vec_store_epi16(dst: &mut i16, vec: vepi16) { 
        // check alignment in debug mode
        debug_assert!((std::ptr::from_ref(dst) as usize) % std::mem::align_of::<vepi16>() == 0);
        _mm256_store_si256(std::ptr::from_mut(dst).cast(), vec);
    }
    #[inline] pub unsafe fn vec_load_epi32(src: &i32) -> vepi16 { 
        // check alignment in debug mode
        debug_assert!((std::ptr::from_ref(src) as usize) % std::mem::align_of::<vepi16>() == 0);
        return _mm256_load_si256(std::ptr::from_ref(src).cast());
    }
    #[inline] pub unsafe fn vec_store_epi32(dst: &mut i32, vec: vepi16) { 
        // check alignment in debug mode
        debug_assert!((std::ptr::from_ref(dst) as usize) % std::mem::align_of::<vepi32>() == 0);
        _mm256_store_si256(std::ptr::from_mut(dst).cast(), vec);
    }
    #[inline] pub unsafe fn vec_store_epiu32(dst: &mut u32, vec: vepi32) { 
        // check alignment in debug mode
        debug_assert!((std::ptr::from_ref(dst) as usize) % std::mem::align_of::<vepi32>() == 0);
        _mm256_storeu_si256(std::ptr::from_mut(dst).cast(), vec);
    }
    #[inline] pub unsafe fn vec_max_epi16(vec0: vepi16, vec1: vepi16)   -> vepi16  { return _mm256_max_epi16(vec0, vec1); }
    #[inline] pub unsafe fn vec_min_epi16  (vec0: vepi16, vec1: vepi16) -> vepi16 { return _mm256_min_epi16(vec0, vec1); }
    #[inline] pub unsafe fn vec_add_epi16(vec0: vepi16, vec1: vepi16) -> vepi16 { return _mm256_add_epi16(vec0, vec1); }
    #[inline] pub unsafe fn vec_sub_epi16(vec0: vepi16, vec1: vepi16) -> vepi16 { return _mm256_sub_epi16(vec0, vec1); }
    #[inline] pub unsafe fn vec_add_epi32(vec0: vepi32, vec1: vepi32) -> vepi32 { return _mm256_add_epi32(vec0, vec1); }
    #[inline] pub unsafe fn vec_mulhi_epi16(vec0: vepi16, vec1: vepi16) -> vepi16 { return _mm256_mulhi_epi16(vec0, vec1); }
    #[inline] pub unsafe fn vec_slli_epi16<const SHIFT: i32>(vec: vepi16)  -> vepi16  { return _mm256_slli_epi16(vec, SHIFT); }
    #[inline] pub unsafe fn vec_nnz_mask(vec: vepi32) -> u16 { return _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(vec, _mm256_setzero_si256()))) as u16; }
    #[inline] pub unsafe fn vec_packus_permute_epi16(vec0: vepi16, vec1: vepi16) -> vepi8 {
        let packed = _mm256_packus_epi16(vec0, vec1);
        return _mm256_permute4x64_epi64(packed, super::mm_shuffle(3, 1, 2, 0));
    }

    #[inline] pub unsafe fn vec_dpbusd_epi32(sum: vepi32, vec0: vepi8, vec1: vepi8) -> vepi32 {
        let product16 = _mm256_maddubs_epi16(vec0, vec1);
        let product32 = _mm256_madd_epi16(product16, _mm256_set1_epi16(1));
        return _mm256_add_epi32(sum, product32);
    }

    #[inline] pub unsafe fn vec_cvtepi32_ps(vec: vepi32) -> vps32 { return _mm256_cvtepi32_ps(vec); }

    #[inline] pub unsafe fn vec_zero_ps () -> vps32 { return _mm256_setzero_ps(); }
    #[inline] pub unsafe fn vec_set1_ps (n: f32) -> vps32 { return _mm256_set1_ps(n); }
    #[inline] pub unsafe fn vec_load_ps (src: &f32) -> vps32 { 
        // check alignment in debug mode
        debug_assert!((std::ptr::from_ref(src) as usize) % std::mem::align_of::<vps32>() == 0);
        return _mm256_load_ps(src);
    }
    #[inline] pub unsafe fn vec_store_ps(dst: &mut f32, vec: vps32) { 
        // check alignment in debug mode
        debug_assert!((std::ptr::from_ref(dst) as usize) % std::mem::align_of::<vps32>() == 0);
        _mm256_store_ps(dst, vec);
    }
    #[inline] pub unsafe fn vec_add_ps(vec0: vps32, vec1: vps32) -> vps32 { return _mm256_add_ps(vec0, vec1); }
    #[inline] pub unsafe fn vec_mul_ps(vec0: vps32, vec1: vps32) -> vps32 { return _mm256_mul_ps(vec0, vec1); }
    #[inline] pub unsafe fn vec_div_ps(vec0: vps32, vec1: vps32) -> vps32 { return _mm256_div_ps(vec0, vec1); }
    #[inline] pub unsafe fn vec_max_ps(vec0: vps32, vec1: vps32) -> vps32 { return _mm256_max_ps(vec0, vec1); }
    #[inline] pub unsafe fn vec_min_ps(vec0: vps32, vec1: vps32) -> vps32 { return _mm256_min_ps(vec0, vec1); }
    #[inline] pub unsafe fn vec_mul_add_ps(vec0: vps32, vec1: vps32, vec2: vps32) -> vps32 { return _mm256_fmadd_ps(vec0, vec1, vec2); }
    #[inline] pub unsafe fn vec_reduce_add_ps(vec: vps32) -> f32 {
        let upper_128 = _mm256_extractf128_ps(vec, 1);
        let lower_128 = _mm256_castps256_ps128(vec);
        let sum_128 = _mm_add_ps(upper_128, lower_128);

        let upper_64 = _mm_movehl_ps(sum_128, sum_128);
        let sum_64 = _mm_add_ps(upper_64, sum_128);

        let upper_32 = _mm_shuffle_ps(sum_64, sum_64, 1);
        let sum_32 = _mm_add_ss(upper_32, sum_64);

        return _mm_cvtss_f32(sum_32);
    }

    pub const U8_CHUNK_SIZE: usize = std::mem::size_of::<vepi8>() / std::mem::size_of::<u8>();
    pub const I8_CHUNK_SIZE_I32: usize = std::mem::size_of::<i32>() / std::mem::size_of::<u8>();
    pub const I16_CHUNK_SIZE: usize = std::mem::size_of::<vepi16>() / std::mem::size_of::<i16>();
    pub const I32_CHUNK_SIZE: usize = std::mem::size_of::<vepi32>() / std::mem::size_of::<i32>();
    pub const F32_CHUNK_SIZE: usize = std::mem::size_of::<vps32>() / std::mem::size_of::<f32>();
}

#[cfg(target_feature = "avx2")]
pub use avx2::*;