#![allow(clippy::all, clippy::nursery, clippy::pedantic, dead_code)]
#![allow(clippy::undocumented_unsafe_blocks)]

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

#[inline(always)]
pub const fn mm_shuffle(z: i32, y: i32, x: i32, w: i32) -> i32 {
    ((z) << 6) | ((y) << 4) | ((x) << 2) | (w)
}

/// Given a regular type and a SIMD register type, and the new type name, create a new type that wraps the register type.
#[allow(unused_macros)]
macro_rules! wrap_simd_register {
    ($register_type:ty, $held_type:ty, $new_type:ident) => {
        #[repr(transparent)]
        #[derive(Debug, Clone, Copy)]
        pub struct $new_type($register_type);
        impl $new_type {
            #[inline(always)]
            pub const fn from_raw(value: $register_type) -> Self {
                Self(value)
            }
            #[inline(always)]
            pub const fn inner(self) -> $register_type {
                self.0
            }
        }
    };
}

#[cfg(target_feature = "avx512f")]
mod avx512 {
    #![allow(non_camel_case_types)]
    use std::arch::x86_64::*;

    pub const INNER_ARCH: &str = "avx512";

    wrap_simd_register!(__m512i, i8, VecI8);
    wrap_simd_register!(__m512i, i16, VecI16);
    wrap_simd_register!(__m512i, i32, VecI32);
    wrap_simd_register!(__m512i, i64, VecI64);
    wrap_simd_register!(__m512, f32, VecF32);

    #[inline(always)]
    pub unsafe fn zero_i16() -> VecI16 {
        unsafe {
            return VecI16::from_raw(_mm512_setzero_si512());
        }
    }
    #[inline(always)]
    pub unsafe fn zero_i32() -> VecI32 {
        unsafe {
            return VecI32::from_raw(_mm512_setzero_si512());
        }
    }
    #[inline(always)]
    pub unsafe fn splat_i16(n: i16) -> VecI16 {
        unsafe {
            return VecI16::from_raw(_mm512_set1_epi16(n));
        }
    }
    #[inline(always)]
    pub unsafe fn splat_i32(n: i32) -> VecI32 {
        unsafe {
            return VecI32::from_raw(_mm512_set1_epi32(n));
        }
    }
    #[inline(always)]
    pub unsafe fn load_i8(src: *const i8) -> VecI8 {
        unsafe {
            // check alignment in debug mode
            debug_assert!((src as usize) % std::mem::align_of::<VecI8>() == 0);
            return VecI8::from_raw(_mm512_load_si512(src.cast()));
        }
    }
    #[inline(always)]
    pub unsafe fn store_i8(dst: *mut i8, vec: VecI8) {
        unsafe {
            // check alignment in debug mode
            debug_assert!((dst as usize) % std::mem::align_of::<VecI8>() == 0);
            _mm512_store_si512(dst.cast(), vec.inner());
        }
    }
    #[inline(always)]
    pub unsafe fn load_u8(src: *const u8) -> VecI8 {
        unsafe {
            // check alignment in debug mode
            debug_assert!((src as usize) % std::mem::align_of::<VecI8>() == 0);
            return VecI8::from_raw(_mm512_load_si512(src.cast()));
        }
    }
    #[inline(always)]
    pub unsafe fn store_u8(dst: *mut u8, vec: VecI8) {
        unsafe {
            // check alignment in debug mode
            debug_assert!((dst as usize) % std::mem::align_of::<VecI8>() == 0);
            _mm512_store_si512(dst.cast(), vec.inner());
        }
    }
    #[inline(always)]
    pub unsafe fn load_i16(src: *const i16) -> VecI16 {
        unsafe {
            // check alignment in debug mode
            debug_assert!((src as usize) % std::mem::align_of::<VecI16>() == 0);
            return VecI16::from_raw(_mm512_load_si512(src.cast()));
        }
    }
    #[inline(always)]
    pub unsafe fn store_i16(dst: *mut i16, vec: VecI16) {
        unsafe {
            // check alignment in debug mode
            debug_assert!((dst as usize) % std::mem::align_of::<VecI16>() == 0);
            _mm512_store_si512(dst.cast(), vec.inner());
        }
    }
    #[inline(always)]
    pub unsafe fn load_i32(src: *const i32) -> VecI32 {
        unsafe {
            // check alignment in debug mode
            debug_assert!((src as usize) % std::mem::align_of::<VecI32>() == 0);
            return VecI32::from_raw(_mm512_load_si512(src.cast()));
        }
    }
    #[inline(always)]
    pub unsafe fn store_i32(dst: *mut i32, vec: VecI32) {
        unsafe {
            // check alignment in debug mode
            debug_assert!((dst as usize) % std::mem::align_of::<VecI32>() == 0);
            _mm512_store_si512(dst.cast(), vec.inner());
        }
    }
    #[inline(always)]
    pub unsafe fn store_u32(dst: *mut u32, vec: VecI32) {
        unsafe {
            // check alignment in debug mode
            debug_assert!((dst as usize) % std::mem::align_of::<VecI32>() == 0);
            _mm512_storeu_si512(dst.cast(), vec.inner());
        }
    }
    #[inline(always)]
    pub unsafe fn max_i16(vec0: VecI16, vec1: VecI16) -> VecI16 {
        unsafe {
            return VecI16::from_raw(_mm512_max_epi16(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn min_i16(vec0: VecI16, vec1: VecI16) -> VecI16 {
        unsafe {
            return VecI16::from_raw(_mm512_min_epi16(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn add_i16(vec0: VecI16, vec1: VecI16) -> VecI16 {
        unsafe {
            return VecI16::from_raw(_mm512_add_epi16(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn sub_i16(vec0: VecI16, vec1: VecI16) -> VecI16 {
        unsafe {
            return VecI16::from_raw(_mm512_sub_epi16(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn add_i32(vec0: VecI32, vec1: VecI32) -> VecI32 {
        unsafe {
            return VecI32::from_raw(_mm512_add_epi32(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn mul_high_i16(vec0: VecI16, vec1: VecI16) -> VecI16 {
        unsafe {
            return VecI16::from_raw(_mm512_mulhi_epi16(vec0.inner(), vec1.inner()));
        }
    }
    // stupid hack for the different intrinsics
    pub type S = u32;
    #[inline(always)]
    pub unsafe fn shl_i16<const SHIFT: u32>(vec: VecI16) -> VecI16 {
        unsafe {
            return VecI16::from_raw(_mm512_slli_epi16(vec.inner(), SHIFT));
        }
    }
    #[inline(always)]
    pub unsafe fn nonzero_mask_i32(vec: VecI32) -> u16 {
        unsafe {
            return _mm512_cmpgt_epi32_mask(vec.inner(), _mm512_setzero_si512()) as u16;
        }
    }
    #[inline(always)]
    pub unsafe fn pack_i16_to_u8(vec0: VecI16, vec1: VecI16) -> VecI8 {
        unsafe {
            let packed = _mm512_packus_epi16(vec0.inner(), vec1.inner());
            // return VecI8::from_raw(_mm512_permutexvar_epi64(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), packed));
            return VecI8::from_raw(packed);
        }
    }
    #[inline(always)]
    pub unsafe fn mul_add_u8_to_i32(sum: VecI32, vec0: VecI8, vec1: VecI8) -> VecI32 {
        unsafe {
            #[cfg(target_feature = "avx512vnni")]
            {
                return VecI32::from_raw(_mm512_dpbusd_epi32(
                    sum.inner(),
                    vec0.inner(),
                    vec1.inner(),
                ));
            }
            #[cfg(not(target_feature = "avx512vnni"))]
            {
                let product16 = _mm512_maddubs_epi16(vec0.inner(), vec1.inner());
                let product32 = _mm512_madd_epi16(product16, _mm512_set1_epi16(1));
                return VecI32::from_raw(_mm512_add_epi32(sum.inner(), product32));
            }
        }
    }
    #[inline(always)]
    pub unsafe fn mul_add_2xu8_to_i32(
        sum: VecI32,
        vec0: VecI8,
        vec1: VecI8,
        vec2: VecI8,
        vec3: VecI8,
    ) -> VecI32 {
        unsafe {
            #[cfg(target_feature = "avx512vnni")]
            {
                return VecI32::from_raw(_mm512_dpbusd_epi32(
                    _mm512_dpbusd_epi32(sum.inner(), vec0.inner(), vec1.inner()),
                    vec2.inner(),
                    vec3.inner(),
                ));
            }
            #[cfg(not(target_feature = "avx512vnni"))]
            {
                let product16a = _mm512_maddubs_epi16(vec0.inner(), vec1.inner());
                let product16b = _mm512_maddubs_epi16(vec2.inner(), vec3.inner());
                let product32 = _mm512_madd_epi16(
                    _mm512_add_epi16(product16a, product16b),
                    _mm512_set1_epi16(1),
                );
                return VecI32::from_raw(_mm512_add_epi32(sum.inner(), product32));
            }
        }
    }
    #[inline(always)]
    pub unsafe fn i32_to_f32(vec: VecI32) -> VecF32 {
        unsafe {
            return VecF32::from_raw(_mm512_cvtepi32_ps(vec.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn zero_f32() -> VecF32 {
        unsafe {
            return VecF32::from_raw(_mm512_setzero_ps());
        }
    }
    #[inline(always)]
    pub unsafe fn splat_f32(n: f32) -> VecF32 {
        unsafe {
            return VecF32::from_raw(_mm512_set1_ps(n));
        }
    }
    #[inline(always)]
    pub unsafe fn load_f32(src: *const f32) -> VecF32 {
        unsafe {
            // check alignment in debug mode
            debug_assert!((src as usize) % std::mem::align_of::<VecF32>() == 0);
            return VecF32::from_raw(_mm512_load_ps(src));
        }
    }
    #[inline(always)]
    pub unsafe fn store_f32(dst: *mut f32, vec: VecF32) {
        unsafe {
            // check alignment in debug mode
            debug_assert!((dst as usize) % std::mem::align_of::<VecF32>() == 0);
            _mm512_store_ps(dst, vec.inner());
        }
    }
    #[inline(always)]
    pub unsafe fn add_f32(vec0: VecF32, vec1: VecF32) -> VecF32 {
        unsafe {
            return VecF32::from_raw(_mm512_add_ps(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn mul_f32(vec0: VecF32, vec1: VecF32) -> VecF32 {
        unsafe {
            return VecF32::from_raw(_mm512_mul_ps(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn div_f32(vec0: VecF32, vec1: VecF32) -> VecF32 {
        unsafe {
            return VecF32::from_raw(_mm512_div_ps(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn max_f32(vec0: VecF32, vec1: VecF32) -> VecF32 {
        unsafe {
            return VecF32::from_raw(_mm512_max_ps(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn min_f32(vec0: VecF32, vec1: VecF32) -> VecF32 {
        unsafe {
            return VecF32::from_raw(_mm512_min_ps(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn mul_add_f32(vec0: VecF32, vec1: VecF32, vec2: VecF32) -> VecF32 {
        unsafe {
            return VecF32::from_raw(_mm512_fmadd_ps(vec0.inner(), vec1.inner(), vec2.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn sum_f32(vec: VecF32) -> f32 {
        unsafe {
            return _mm512_reduce_add_ps(vec.inner());
        }
    }
    #[inline(always)]
    pub unsafe fn reduce_add_f32s(vec: &[VecF32; 1]) -> f32 {
        unsafe {
            return _mm512_reduce_add_ps(vec.get_unchecked(0).inner());
        }
    }

    pub const U8_CHUNK_SIZE: usize = std::mem::size_of::<VecI8>() / std::mem::size_of::<u8>();
    pub const I8_CHUNK_SIZE_I32: usize = std::mem::size_of::<i32>() / std::mem::size_of::<u8>();
    pub const I16_CHUNK_SIZE: usize = std::mem::size_of::<VecI16>() / std::mem::size_of::<i16>();
    pub const I32_CHUNK_SIZE: usize = std::mem::size_of::<VecI32>() / std::mem::size_of::<i32>();
    pub const F32_CHUNK_SIZE: usize = std::mem::size_of::<VecF32>() / std::mem::size_of::<f32>();
}

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
mod avx2 {
    #![allow(non_camel_case_types)]
    use std::arch::x86_64::*;

    pub const INNER_ARCH: &str = "avx2";

    wrap_simd_register!(__m256i, i8, VecI8);
    wrap_simd_register!(__m256i, i16, VecI16);
    wrap_simd_register!(__m256i, i32, VecI32);
    wrap_simd_register!(__m256i, i64, VecI64);
    wrap_simd_register!(__m256, f32, VecF32);

    #[inline(always)]
    pub unsafe fn zero_i16() -> VecI16 {
        unsafe {
            return VecI16::from_raw(_mm256_setzero_si256());
        }
    }
    #[inline(always)]
    pub unsafe fn zero_i32() -> VecI32 {
        unsafe {
            return VecI32::from_raw(_mm256_setzero_si256());
        }
    }
    #[inline(always)]
    pub unsafe fn splat_i16(n: i16) -> VecI16 {
        unsafe {
            return VecI16::from_raw(_mm256_set1_epi16(n));
        }
    }
    #[inline(always)]
    pub unsafe fn splat_i32(n: i32) -> VecI32 {
        unsafe {
            return VecI32::from_raw(_mm256_set1_epi32(n));
        }
    }
    #[inline(always)]
    pub unsafe fn load_i8(src: *const i8) -> VecI8 {
        unsafe {
            // check alignment in debug mode
            debug_assert!((src as usize) % std::mem::align_of::<VecI8>() == 0);
            return VecI8::from_raw(_mm256_load_si256(src.cast()));
        }
    }
    #[inline(always)]
    pub unsafe fn store_i8(dst: *mut i8, vec: VecI8) {
        unsafe {
            // check alignment in debug mode
            debug_assert!((dst as usize) % std::mem::align_of::<VecI8>() == 0);
            _mm256_store_si256(dst.cast(), vec.inner());
        }
    }
    #[inline(always)]
    pub unsafe fn load_u8(src: *const u8) -> VecI8 {
        unsafe {
            // check alignment in debug mode
            debug_assert!((src as usize) % std::mem::align_of::<VecI8>() == 0);
            return VecI8::from_raw(_mm256_load_si256(src.cast()));
        }
    }
    #[inline(always)]
    pub unsafe fn store_u8(dst: *mut u8, vec: VecI8) {
        unsafe {
            // check alignment in debug mode
            debug_assert!((dst as usize) % std::mem::align_of::<VecI8>() == 0);
            _mm256_store_si256(dst.cast(), vec.inner());
        }
    }
    #[inline(always)]
    pub unsafe fn load_i16(src: *const i16) -> VecI16 {
        unsafe {
            // check alignment in debug mode
            debug_assert!((src as usize) % std::mem::align_of::<VecI16>() == 0);
            return VecI16::from_raw(_mm256_load_si256(src.cast()));
        }
    }
    #[inline(always)]
    pub unsafe fn store_i16(dst: *mut i16, vec: VecI16) {
        unsafe {
            // check alignment in debug mode
            debug_assert!((dst as usize) % std::mem::align_of::<VecI16>() == 0);
            _mm256_store_si256(dst.cast(), vec.inner());
        }
    }
    #[inline(always)]
    pub unsafe fn load_i32(src: *const i32) -> VecI32 {
        unsafe {
            // check alignment in debug mode
            debug_assert!((src as usize) % std::mem::align_of::<VecI32>() == 0);
            return VecI32::from_raw(_mm256_load_si256(src.cast()));
        }
    }
    #[inline(always)]
    pub unsafe fn store_i32(dst: *mut i32, vec: VecI32) {
        unsafe {
            // check alignment in debug mode
            debug_assert!((dst as usize) % std::mem::align_of::<VecI32>() == 0);
            _mm256_store_si256(dst.cast(), vec.inner());
        }
    }
    #[inline(always)]
    pub unsafe fn store_u32(dst: *mut u32, vec: VecI32) {
        unsafe {
            // check alignment in debug mode
            debug_assert!((dst as usize) % std::mem::align_of::<VecI32>() == 0);
            _mm256_storeu_si256(dst.cast(), vec.inner());
        }
    }
    #[inline(always)]
    pub unsafe fn max_i16(vec0: VecI16, vec1: VecI16) -> VecI16 {
        unsafe {
            return VecI16::from_raw(_mm256_max_epi16(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn min_i16(vec0: VecI16, vec1: VecI16) -> VecI16 {
        unsafe {
            return VecI16::from_raw(_mm256_min_epi16(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn add_i16(vec0: VecI16, vec1: VecI16) -> VecI16 {
        unsafe {
            return VecI16::from_raw(_mm256_add_epi16(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn sub_i16(vec0: VecI16, vec1: VecI16) -> VecI16 {
        unsafe {
            return VecI16::from_raw(_mm256_sub_epi16(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn add_i32(vec0: VecI32, vec1: VecI32) -> VecI32 {
        unsafe {
            return VecI32::from_raw(_mm256_add_epi32(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn mul_high_i16(vec0: VecI16, vec1: VecI16) -> VecI16 {
        unsafe {
            return VecI16::from_raw(_mm256_mulhi_epi16(vec0.inner(), vec1.inner()));
        }
    }
    // stupid hack for the different intrinsics
    pub type S = i32;
    #[inline(always)]
    pub unsafe fn shl_i16<const SHIFT: i32>(vec: VecI16) -> VecI16 {
        unsafe {
            return VecI16::from_raw(_mm256_slli_epi16(vec.inner(), SHIFT));
        }
    }
    #[inline(always)]
    pub unsafe fn nonzero_mask_i32(vec: VecI32) -> u16 {
        unsafe {
            return _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(
                vec.inner(),
                _mm256_setzero_si256(),
            ))) as u16;
        }
    }
    #[inline(always)]
    pub unsafe fn pack_i16_to_u8(vec0: VecI16, vec1: VecI16) -> VecI8 {
        unsafe {
            let packed = _mm256_packus_epi16(vec0.inner(), vec1.inner());
            // return VecI8::from_raw(_mm256_permute4x64_epi64(packed, super::mm_shuffle(3, 1, 2, 0)));
            return VecI8::from_raw(packed);
        }
    }
    #[inline(always)]
    pub unsafe fn mul_add_u8_to_i32(sum: VecI32, vec0: VecI8, vec1: VecI8) -> VecI32 {
        unsafe {
            let product16 = _mm256_maddubs_epi16(vec0.inner(), vec1.inner());
            let product32 = _mm256_madd_epi16(product16, _mm256_set1_epi16(1));
            return VecI32::from_raw(_mm256_add_epi32(sum.inner(), product32));
        }
    }
    #[inline(always)]
    pub unsafe fn mul_add_2xu8_to_i32(
        sum: VecI32,
        vec0: VecI8,
        vec1: VecI8,
        vec2: VecI8,
        vec3: VecI8,
    ) -> VecI32 {
        unsafe {
            let product16a = _mm256_maddubs_epi16(vec0.inner(), vec1.inner());
            let product16b = _mm256_maddubs_epi16(vec2.inner(), vec3.inner());
            let product32 = _mm256_madd_epi16(
                _mm256_add_epi16(product16a, product16b),
                _mm256_set1_epi16(1),
            );
            return VecI32::from_raw(_mm256_add_epi32(sum.inner(), product32));
        }
    }
    #[inline(always)]
    pub unsafe fn i32_to_f32(vec: VecI32) -> VecF32 {
        unsafe {
            return VecF32::from_raw(_mm256_cvtepi32_ps(vec.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn zero_f32() -> VecF32 {
        unsafe {
            return VecF32::from_raw(_mm256_setzero_ps());
        }
    }
    #[inline(always)]
    pub unsafe fn splat_f32(n: f32) -> VecF32 {
        unsafe {
            return VecF32::from_raw(_mm256_set1_ps(n));
        }
    }
    #[inline(always)]
    pub unsafe fn load_f32(src: *const f32) -> VecF32 {
        unsafe {
            // check alignment in debug mode
            debug_assert!((src as usize) % std::mem::align_of::<VecF32>() == 0);
            return VecF32::from_raw(_mm256_load_ps(src));
        }
    }
    #[inline(always)]
    pub unsafe fn store_f32(dst: *mut f32, vec: VecF32) {
        unsafe {
            // check alignment in debug mode
            debug_assert!((dst as usize) % std::mem::align_of::<VecF32>() == 0);
            _mm256_store_ps(dst, vec.inner());
        }
    }
    #[inline(always)]
    pub unsafe fn add_f32(vec0: VecF32, vec1: VecF32) -> VecF32 {
        unsafe {
            return VecF32::from_raw(_mm256_add_ps(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn mul_f32(vec0: VecF32, vec1: VecF32) -> VecF32 {
        unsafe {
            return VecF32::from_raw(_mm256_mul_ps(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn div_f32(vec0: VecF32, vec1: VecF32) -> VecF32 {
        unsafe {
            return VecF32::from_raw(_mm256_div_ps(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn max_f32(vec0: VecF32, vec1: VecF32) -> VecF32 {
        unsafe {
            return VecF32::from_raw(_mm256_max_ps(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn min_f32(vec0: VecF32, vec1: VecF32) -> VecF32 {
        unsafe {
            return VecF32::from_raw(_mm256_min_ps(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn mul_add_f32(vec0: VecF32, vec1: VecF32, vec2: VecF32) -> VecF32 {
        unsafe {
            return VecF32::from_raw(_mm256_fmadd_ps(vec0.inner(), vec1.inner(), vec2.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn sum_f32(vec: VecF32) -> f32 {
        unsafe {
            let upper_128 = _mm256_extractf128_ps(vec.inner(), 1);
            let lower_128 = _mm256_castps256_ps128(vec.inner());
            let sum_128 = _mm_add_ps(upper_128, lower_128);

            let upper_64 = _mm_movehl_ps(sum_128, sum_128);
            let sum_64 = _mm_add_ps(upper_64, sum_128);

            let upper_32 = _mm_shuffle_ps(sum_64, sum_64, 1);
            let sum_32 = _mm_add_ss(upper_32, sum_64);

            return _mm_cvtss_f32(sum_32);
        }
    }
    #[inline(always)]
    pub unsafe fn reduce_add_f32s(vec: &[VecF32; 2]) -> f32 {
        unsafe {
            let vec = _mm256_add_ps(vec.get_unchecked(0).inner(), vec.get_unchecked(1).inner());

            let upper_128 = _mm256_extractf128_ps(vec, 1);
            let lower_128 = _mm256_castps256_ps128(vec);
            let sum_128 = _mm_add_ps(upper_128, lower_128);

            let upper_64 = _mm_movehl_ps(sum_128, sum_128);
            let sum_64 = _mm_add_ps(upper_64, sum_128);

            let upper_32 = _mm_shuffle_ps(sum_64, sum_64, 1);
            let sum_32 = _mm_add_ss(upper_32, sum_64);

            return _mm_cvtss_f32(sum_32);
        }
    }

    pub const U8_CHUNK_SIZE: usize = std::mem::size_of::<VecI8>() / std::mem::size_of::<u8>();
    pub const I8_CHUNK_SIZE_I32: usize = std::mem::size_of::<i32>() / std::mem::size_of::<u8>();
    pub const I16_CHUNK_SIZE: usize = std::mem::size_of::<VecI16>() / std::mem::size_of::<i16>();
    pub const I32_CHUNK_SIZE: usize = std::mem::size_of::<VecI32>() / std::mem::size_of::<i32>();
    pub const F32_CHUNK_SIZE: usize = std::mem::size_of::<VecF32>() / std::mem::size_of::<f32>();
}

#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "avx2"),
    not(target_feature = "avx512f")
))]
mod sse2 {
    #![allow(non_camel_case_types)]
    use std::arch::x86_64::*;

    pub const INNER_ARCH: &str = "sse2";

    wrap_simd_register!(__m128i, i8, VecI8);
    wrap_simd_register!(__m128i, i16, VecI16);
    wrap_simd_register!(__m128i, i32, VecI32);
    wrap_simd_register!(__m128i, i64, VecI64);
    wrap_simd_register!(__m128, f32, VecF32);

    #[inline(always)]
    pub unsafe fn zero_i16() -> VecI16 {
        unsafe {
            return VecI16::from_raw(_mm_setzero_si128());
        }
    }
    #[inline(always)]
    pub unsafe fn zero_i32() -> VecI32 {
        unsafe {
            return VecI32::from_raw(_mm_setzero_si128());
        }
    }
    #[inline(always)]
    pub unsafe fn splat_i16(n: i16) -> VecI16 {
        unsafe {
            return VecI16::from_raw(_mm_set1_epi16(n));
        }
    }
    #[inline(always)]
    pub unsafe fn splat_i32(n: i32) -> VecI32 {
        unsafe {
            return VecI32::from_raw(_mm_set1_epi32(n));
        }
    }
    #[inline(always)]
    pub unsafe fn load_i8(src: *const i8) -> VecI8 {
        unsafe {
            // check alignment in debug mode
            debug_assert!((src as usize) % std::mem::align_of::<VecI8>() == 0);
            return VecI8::from_raw(_mm_load_si128(src.cast()));
        }
    }
    #[inline(always)]
    pub unsafe fn store_i8(dst: *mut i8, vec: VecI8) {
        unsafe {
            // check alignment in debug mode
            debug_assert!((dst as usize) % std::mem::align_of::<VecI8>() == 0);
            _mm_store_si128(dst.cast(), vec.inner());
        }
    }
    #[inline(always)]
    pub unsafe fn load_u8(src: *const u8) -> VecI8 {
        unsafe {
            // check alignment in debug mode
            debug_assert!((src as usize) % std::mem::align_of::<VecI8>() == 0);
            return VecI8::from_raw(_mm_load_si128(src.cast()));
        }
    }
    #[inline(always)]
    pub unsafe fn store_u8(dst: *mut u8, vec: VecI8) {
        unsafe {
            // check alignment in debug mode
            debug_assert!((dst as usize) % std::mem::align_of::<VecI8>() == 0);
            _mm_store_si128(dst.cast(), vec.inner());
        }
    }
    #[inline(always)]
    pub unsafe fn load_i16(src: *const i16) -> VecI16 {
        unsafe {
            // check alignment in debug mode
            debug_assert!((src as usize) % std::mem::align_of::<VecI16>() == 0);
            return VecI16::from_raw(_mm_load_si128(src.cast()));
        }
    }
    #[inline(always)]
    pub unsafe fn store_i16(dst: *mut i16, vec: VecI16) {
        unsafe {
            // check alignment in debug mode
            debug_assert!((dst as usize) % std::mem::align_of::<VecI16>() == 0);
            _mm_store_si128(dst.cast(), vec.inner());
        }
    }
    #[inline(always)]
    pub unsafe fn load_i32(src: *const i32) -> VecI32 {
        unsafe {
            // check alignment in debug mode
            debug_assert!((src as usize) % std::mem::align_of::<VecI32>() == 0);
            return VecI32::from_raw(_mm_load_si128(src.cast()));
        }
    }
    #[inline(always)]
    pub unsafe fn store_i32(dst: *mut i32, vec: VecI32) {
        unsafe {
            // check alignment in debug mode
            debug_assert!((dst as usize) % std::mem::align_of::<VecI32>() == 0);
            _mm_store_si128(dst.cast(), vec.inner());
        }
    }
    #[inline(always)]
    pub unsafe fn store_u32(dst: *mut u32, vec: VecI32) {
        unsafe {
            // check alignment in debug mode
            debug_assert!((dst as usize) % std::mem::align_of::<VecI32>() == 0);
            _mm_storeu_si128(dst.cast(), vec.inner());
        }
    }
    #[inline(always)]
    pub unsafe fn max_i16(vec0: VecI16, vec1: VecI16) -> VecI16 {
        unsafe {
            return VecI16::from_raw(_mm_max_epi16(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn min_i16(vec0: VecI16, vec1: VecI16) -> VecI16 {
        unsafe {
            return VecI16::from_raw(_mm_min_epi16(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn add_i16(vec0: VecI16, vec1: VecI16) -> VecI16 {
        unsafe {
            return VecI16::from_raw(_mm_add_epi16(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn sub_i16(vec0: VecI16, vec1: VecI16) -> VecI16 {
        unsafe {
            return VecI16::from_raw(_mm_sub_epi16(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn add_i32(vec0: VecI32, vec1: VecI32) -> VecI32 {
        unsafe {
            return VecI32::from_raw(_mm_add_epi32(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn mul_high_i16(vec0: VecI16, vec1: VecI16) -> VecI16 {
        unsafe {
            return VecI16::from_raw(_mm_mulhi_epi16(vec0.inner(), vec1.inner()));
        }
    }
    // stupid hack for the different intrinsics
    pub type S = i32;
    #[inline(always)]
    pub unsafe fn shl_i16<const SHIFT: i32>(vec: VecI16) -> VecI16 {
        unsafe {
            return VecI16::from_raw(_mm_slli_epi16(vec.inner(), SHIFT));
        }
    }
    #[inline(always)]
    pub unsafe fn nonzero_mask_i32(vec: VecI32) -> u16 {
        unsafe {
            return _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpgt_epi32(
                vec.inner(),
                _mm_setzero_si128(),
            ))) as u16;
        }
    }
    #[inline(always)]
    pub unsafe fn pack_i16_to_u8(vec0: VecI16, vec1: VecI16) -> VecI8 {
        unsafe {
            let packed = _mm_packus_epi16(vec0.inner(), vec1.inner());
            return VecI8::from_raw(packed);
        }
    }
    #[inline(always)]
    pub unsafe fn mul_add_u8_to_i32(sum: VecI32, vec0: VecI8, vec1: VecI8) -> VecI32 {
        unsafe {
            let product16 = _mm_maddubs_epi16(vec0.inner(), vec1.inner());
            let product32 = _mm_madd_epi16(product16, _mm_set1_epi16(1));
            return VecI32::from_raw(_mm_add_epi32(sum.inner(), product32));
        }
    }
    #[inline(always)]
    pub unsafe fn mul_add_2xu8_to_i32(
        sum: VecI32,
        vec0: VecI8,
        vec1: VecI8,
        vec2: VecI8,
        vec3: VecI8,
    ) -> VecI32 {
        unsafe {
            let product16a = _mm_maddubs_epi16(vec0.inner(), vec1.inner());
            let product16b = _mm_maddubs_epi16(vec2.inner(), vec3.inner());
            let product32 =
                _mm_madd_epi16(_mm_add_epi16(product16a, product16b), _mm_set1_epi16(1));
            return VecI32::from_raw(_mm_add_epi32(sum.inner(), product32));
        }
    }
    #[inline(always)]
    pub unsafe fn i32_to_f32(vec: VecI32) -> VecF32 {
        unsafe {
            return VecF32::from_raw(_mm_cvtepi32_ps(vec.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn zero_f32() -> VecF32 {
        unsafe {
            return VecF32::from_raw(_mm_setzero_ps());
        }
    }
    #[inline(always)]
    pub unsafe fn splat_f32(n: f32) -> VecF32 {
        unsafe {
            return VecF32::from_raw(_mm_set1_ps(n));
        }
    }
    #[inline(always)]
    pub unsafe fn load_f32(src: *const f32) -> VecF32 {
        unsafe {
            // check alignment in debug mode
            debug_assert!((src as usize) % std::mem::align_of::<VecF32>() == 0);
            return VecF32::from_raw(_mm_load_ps(src));
        }
    }
    #[inline(always)]
    pub unsafe fn store_f32(dst: *mut f32, vec: VecF32) {
        unsafe {
            // check alignment in debug mode
            debug_assert!((dst as usize) % std::mem::align_of::<VecF32>() == 0);
            _mm_store_ps(dst, vec.inner());
        }
    }
    #[inline(always)]
    pub unsafe fn add_f32(vec0: VecF32, vec1: VecF32) -> VecF32 {
        unsafe {
            return VecF32::from_raw(_mm_add_ps(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn mul_f32(vec0: VecF32, vec1: VecF32) -> VecF32 {
        unsafe {
            return VecF32::from_raw(_mm_mul_ps(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn div_f32(vec0: VecF32, vec1: VecF32) -> VecF32 {
        unsafe {
            return VecF32::from_raw(_mm_div_ps(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn max_f32(vec0: VecF32, vec1: VecF32) -> VecF32 {
        unsafe {
            return VecF32::from_raw(_mm_max_ps(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn min_f32(vec0: VecF32, vec1: VecF32) -> VecF32 {
        unsafe {
            return VecF32::from_raw(_mm_min_ps(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn mul_add_f32(vec0: VecF32, vec1: VecF32, vec2: VecF32) -> VecF32 {
        unsafe {
            return VecF32::from_raw(_mm_fmadd_ps(vec0.inner(), vec1.inner(), vec2.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn sum_f32(vec: VecF32) -> f32 {
        unsafe {
            let upper_64 = _mm_movehl_ps(vec.inner(), vec.inner());
            let sum_64 = _mm_add_ps(vec.inner(), upper_64);

            let upper_32 = _mm_shuffle_ps(sum_64, sum_64, 1);
            let sum_32 = _mm_add_ss(upper_32, sum_64);

            return _mm_cvtss_f32(sum_32);
        }
    }
    #[inline(always)]
    pub unsafe fn reduce_add_f32s(vec: &[VecF32; 4]) -> f32 {
        unsafe {
            let vec_a = _mm_add_ps(vec.get_unchecked(0).inner(), vec.get_unchecked(2).inner());
            let vec_b = _mm_add_ps(vec.get_unchecked(1).inner(), vec.get_unchecked(3).inner());
            let vec = _mm_add_ps(vec_a, vec_b);
            let upper_64 = _mm_movehl_ps(vec, vec);
            let sum_64 = _mm_add_ps(vec, upper_64);

            let upper_32 = _mm_shuffle_ps(sum_64, sum_64, 1);
            let sum_32 = _mm_add_ss(upper_32, sum_64);

            return _mm_cvtss_f32(sum_32);
        }
    }

    pub const U8_CHUNK_SIZE: usize = std::mem::size_of::<VecI8>() / std::mem::size_of::<u8>();
    pub const I8_CHUNK_SIZE_I32: usize = std::mem::size_of::<i32>() / std::mem::size_of::<u8>();
    pub const I16_CHUNK_SIZE: usize = std::mem::size_of::<VecI16>() / std::mem::size_of::<i16>();
    pub const I32_CHUNK_SIZE: usize = std::mem::size_of::<VecI32>() / std::mem::size_of::<i32>();
    pub const F32_CHUNK_SIZE: usize = std::mem::size_of::<VecF32>() / std::mem::size_of::<f32>();
}

#[cfg(target_feature = "avx512f")]
pub use avx512::*;

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
pub use avx2::*;

#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "avx2"),
    not(target_feature = "avx512f")
))]
pub use sse2::*;

#[cfg(target_arch = "aarch64")]
pub use neon::*;

#[cfg(target_arch = "aarch64")]
mod neon {
    #![allow(non_camel_case_types)]
    use std::arch::aarch64::*;

    pub const INNER_ARCH: &str = "neon";

    wrap_simd_register!(int8x16_t, i8, VecI8);
    wrap_simd_register!(int16x8_t, i16, VecI16);
    wrap_simd_register!(int32x4_t, i32, VecI32);
    wrap_simd_register!(int64x2_t, i64, VecI64);
    wrap_simd_register!(float32x4_t, f32, VecF32);

    #[inline(always)]
    pub unsafe fn zero_i16() -> VecI16 {
        unsafe {
            return VecI16::from_raw(vdupq_n_s16(0));
        }
    }
    #[inline(always)]
    pub unsafe fn zero_i32() -> VecI32 {
        unsafe {
            return VecI32::from_raw(vdupq_n_s32(0));
        }
    }
    #[inline(always)]
    pub unsafe fn splat_i16(n: i16) -> VecI16 {
        unsafe {
            return VecI16::from_raw(vdupq_n_s16(n));
        }
    }
    #[inline(always)]
    pub unsafe fn splat_i32(n: i32) -> VecI32 {
        unsafe {
            return VecI32::from_raw(vdupq_n_s32(n));
        }
    }
    #[inline(always)]
    pub unsafe fn load_i8(src: *const i8) -> VecI8 {
        unsafe {
            // check alignment in debug mode
            debug_assert!((src as usize) % std::mem::align_of::<VecI8>() == 0);
            return VecI8::from_raw(vld1q_s8(src));
        }
    }
    #[inline(always)]
    pub unsafe fn store_i8(dst: *mut i8, vec: VecI8) {
        unsafe {
            // check alignment in debug mode
            debug_assert!((dst as usize) % std::mem::align_of::<VecI8>() == 0);
            vst1q_s8(dst, vec.inner());
        }
    }
    #[inline(always)]
    pub unsafe fn load_u8(src: *const u8) -> VecI8 {
        unsafe {
            // check alignment in debug mode
            debug_assert!((src as usize) % std::mem::align_of::<VecI8>() == 0);
            return VecI8::from_raw(vreinterpretq_s8_u8(vld1q_u8(src)));
        }
    }
    #[inline(always)]
    pub unsafe fn store_u8(dst: *mut u8, vec: VecI8) {
        unsafe {
            // check alignment in debug mode
            debug_assert!((dst as usize) % std::mem::align_of::<VecI8>() == 0);
            vst1q_u8(dst, vreinterpretq_u8_s8(vec.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn load_i16(src: *const i16) -> VecI16 {
        unsafe {
            // check alignment in debug mode
            debug_assert!((src as usize) % std::mem::align_of::<VecI16>() == 0);
            return VecI16::from_raw(vld1q_s16(src));
        }
    }
    #[inline(always)]
    pub unsafe fn store_i16(dst: *mut i16, vec: VecI16) {
        unsafe {
            // check alignment in debug mode
            debug_assert!((dst as usize) % std::mem::align_of::<VecI16>() == 0);
            vst1q_s16(dst, vec.inner());
        }
    }
    #[inline(always)]
    pub unsafe fn load_i32(src: *const i32) -> VecI32 {
        unsafe {
            // check alignment in debug mode
            debug_assert!((src as usize) % std::mem::align_of::<VecI32>() == 0);
            return VecI32::from_raw(vld1q_s32(src));
        }
    }
    #[inline(always)]
    pub unsafe fn store_i32(dst: *mut i32, vec: VecI32) {
        unsafe {
            // check alignment in debug mode
            debug_assert!((dst as usize) % std::mem::align_of::<VecI32>() == 0);
            vst1q_s32(dst, vec.inner());
        }
    }
    #[inline(always)]
    pub unsafe fn store_u32(dst: *mut u32, vec: VecI32) {
        unsafe {
            // check alignment in debug mode
            debug_assert!((dst as usize) % std::mem::align_of::<VecI32>() == 0);
            vst1q_u32(dst, vreinterpretq_u32_s32(vec.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn max_i16(vec0: VecI16, vec1: VecI16) -> VecI16 {
        unsafe {
            return VecI16::from_raw(vmaxq_s16(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn min_i16(vec0: VecI16, vec1: VecI16) -> VecI16 {
        unsafe {
            return VecI16::from_raw(vminq_s16(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn add_i16(vec0: VecI16, vec1: VecI16) -> VecI16 {
        unsafe {
            return VecI16::from_raw(vaddq_s16(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn sub_i16(vec0: VecI16, vec1: VecI16) -> VecI16 {
        unsafe {
            return VecI16::from_raw(vsubq_s16(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn add_i32(vec0: VecI32, vec1: VecI32) -> VecI32 {
        unsafe {
            return VecI32::from_raw(vaddq_s32(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn mul_high_i16(vec0: VecI16, vec1: VecI16) -> VecI16 {
        unsafe {
            // NEON doesn't have direct high multiply, so we need to emulate it
            let low = vmull_s16(vget_low_s16(vec0.inner()), vget_low_s16(vec1.inner()));
            let high = vmull_high_s16(vec0.inner(), vec1.inner());
            let low_high = vshrn_n_s32(low, 16);
            let high_high = vshrn_n_s32(high, 16);
            return VecI16::from_raw(vcombine_s16(low_high, high_high));
        }
    }
    // stupid hack for the different intrinsics
    pub type S = i32;
    #[inline(always)]
    pub unsafe fn shl_i16<const SHIFT: i32>(vec: VecI16) -> VecI16 {
        unsafe {
            return VecI16::from_raw(vshlq_n_s16(vec.inner(), SHIFT));
        }
    }
    #[inline(always)]
    pub unsafe fn nonzero_mask_i32(vec: VecI32) -> u16 {
        unsafe {
            let zero = vdupq_n_s32(0);
            let mask = vcgtq_s32(vec.inner(), zero);
            let mask_u32 = vreinterpretq_u32_s32(mask);
            // Extract individual lanes and create a bitmask
            let lane0 = if vgetq_lane_u32(mask_u32, 0) != 0 { 1 } else { 0 };
            let lane1 = if vgetq_lane_u32(mask_u32, 1) != 0 { 2 } else { 0 };
            let lane2 = if vgetq_lane_u32(mask_u32, 2) != 0 { 4 } else { 0 };
            let lane3 = if vgetq_lane_u32(mask_u32, 3) != 0 { 8 } else { 0 };
            return (lane0 | lane1 | lane2 | lane3) as u16;
        }
    }
    #[inline(always)]
    pub unsafe fn pack_i16_to_u8(vec0: VecI16, vec1: VecI16) -> VecI8 {
        unsafe {
            let packed_low = vqmovun_s16(vec0.inner());
            let packed_high = vqmovun_s16(vec1.inner());
            return VecI8::from_raw(vreinterpretq_s8_u8(vcombine_u8(packed_low, packed_high)));
        }
    }
    #[inline(always)]
    pub unsafe fn mul_add_u8_to_i32(sum: VecI32, vec0: VecI8, vec1: VecI8) -> VecI32 {
        unsafe {
            // Convert to u8 for dot product
            let u8_vec0 = vreinterpretq_u8_s8(vec0.inner());
            let u8_vec1 = vreinterpretq_u8_s8(vec1.inner());

            // Multiply and accumulate using dot product
            let low_dot = vdotq_u32(vreinterpretq_u32_s32(sum.inner()),
                                  vget_low_u8(u8_vec0), vget_low_u8(u8_vec1));
            let high_dot = vdotq_u32(low_dot,
                                   vget_high_u8(u8_vec0), vget_high_u8(u8_vec1));
            return VecI32::from_raw(vreinterpretq_s32_u32(high_dot));
        }
    }
    #[inline(always)]
    pub unsafe fn mul_add_2xu8_to_i32(
        sum: VecI32,
        vec0: VecI8,
        vec1: VecI8,
        vec2: VecI8,
        vec3: VecI8,
    ) -> VecI32 {
        unsafe {
            let result1 = mul_add_u8_to_i32(sum, vec0, vec1);
            return mul_add_u8_to_i32(result1, vec2, vec3);
        }
    }
    #[inline(always)]
    pub unsafe fn i32_to_f32(vec: VecI32) -> VecF32 {
        unsafe {
            return VecF32::from_raw(vcvtq_f32_s32(vec.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn zero_f32() -> VecF32 {
        unsafe {
            return VecF32::from_raw(vdupq_n_f32(0.0));
        }
    }
    #[inline(always)]
    pub unsafe fn splat_f32(n: f32) -> VecF32 {
        unsafe {
            return VecF32::from_raw(vdupq_n_f32(n));
        }
    }
    #[inline(always)]
    pub unsafe fn load_f32(src: *const f32) -> VecF32 {
        unsafe {
            // check alignment in debug mode
            debug_assert!((src as usize) % std::mem::align_of::<VecF32>() == 0);
            return VecF32::from_raw(vld1q_f32(src));
        }
    }
    #[inline(always)]
    pub unsafe fn store_f32(dst: *mut f32, vec: VecF32) {
        unsafe {
            // check alignment in debug mode
            debug_assert!((dst as usize) % std::mem::align_of::<VecF32>() == 0);
            vst1q_f32(dst, vec.inner());
        }
    }
    #[inline(always)]
    pub unsafe fn add_f32(vec0: VecF32, vec1: VecF32) -> VecF32 {
        unsafe {
            return VecF32::from_raw(vaddq_f32(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn mul_f32(vec0: VecF32, vec1: VecF32) -> VecF32 {
        unsafe {
            return VecF32::from_raw(vmulq_f32(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn div_f32(vec0: VecF32, vec1: VecF32) -> VecF32 {
        unsafe {
            return VecF32::from_raw(vdivq_f32(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn max_f32(vec0: VecF32, vec1: VecF32) -> VecF32 {
        unsafe {
            return VecF32::from_raw(vmaxq_f32(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn min_f32(vec0: VecF32, vec1: VecF32) -> VecF32 {
        unsafe {
            return VecF32::from_raw(vminq_f32(vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn mul_add_f32(vec0: VecF32, vec1: VecF32, vec2: VecF32) -> VecF32 {
        unsafe {
            return VecF32::from_raw(vfmaq_f32(vec2.inner(), vec0.inner(), vec1.inner()));
        }
    }
    #[inline(always)]
    pub unsafe fn sum_f32(vec: VecF32) -> f32 {
        unsafe {
            return vaddvq_f32(vec.inner());
        }
    }
    #[inline(always)]
    pub unsafe fn reduce_add_f32s(vec: &[VecF32; 4]) -> f32 {
        unsafe {
            let sum1 = vaddq_f32(vec.get_unchecked(0).inner(), vec.get_unchecked(1).inner());
            let sum2 = vaddq_f32(vec.get_unchecked(2).inner(), vec.get_unchecked(3).inner());
            let total = vaddq_f32(sum1, sum2);
            return vaddvq_f32(total);
        }
    }

    pub const U8_CHUNK_SIZE: usize = std::mem::size_of::<VecI8>() / std::mem::size_of::<u8>();
    pub const I8_CHUNK_SIZE_I32: usize = std::mem::size_of::<i32>() / std::mem::size_of::<u8>();
    pub const I16_CHUNK_SIZE: usize = std::mem::size_of::<VecI16>() / std::mem::size_of::<i16>();
    pub const I32_CHUNK_SIZE: usize = std::mem::size_of::<VecI32>() / std::mem::size_of::<i32>();
    pub const F32_CHUNK_SIZE: usize = std::mem::size_of::<VecF32>() / std::mem::size_of::<f32>();
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn reinterpret_i32s_as_i8s(vec: VecI32) -> VecI8 {
    VecI8::from_raw(vec.inner())
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn reinterpret_i8s_as_i32s(vec: VecI8) -> VecI32 {
    VecI32::from_raw(vec.inner())
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn reinterpret_i32s_as_i8s(vec: VecI32) -> VecI8 {
    VecI8::from_raw(unsafe { vreinterpretq_s8_s32(vec.inner()) })
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn reinterpret_i8s_as_i32s(vec: VecI8) -> VecI32 {
    VecI32::from_raw(unsafe { vreinterpretq_s32_s8(vec.inner()) })
}

#[cfg(target_arch = "x86_64")]
pub const ARCH: &str = INNER_ARCH;
#[cfg(target_arch = "aarch64")]
pub const ARCH: &str = INNER_ARCH;
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub const ARCH: &str = "generic";
