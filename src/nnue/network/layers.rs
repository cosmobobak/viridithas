#[cfg(not(target_feature = "ssse3"))]
mod generic {
    use super::super::{Align64, L1_SIZE, L2_SIZE, L3_SIZE, QA, QB};

    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::cast_sign_loss)]
    fn activate_ft(us: &Align64<[i16; L1_SIZE]>, them: &Align64<[i16; L1_SIZE]>, output: &mut Align64<[u8; L1_SIZE]>) {
        // this is just autovec'd for the moment.
        for (a, acc) in [us, them].into_iter().enumerate() {
            for i in 0..L1_SIZE / 2 {
                // SAFETY: the largest index into `acc` that we construct is `L1_SIZE / 2 + (L1_SIZE / 2 - 1)`.
                // this is in-bounds.
                unsafe {
                    let l = *acc.get_unchecked(i);
                    let r = *acc.get_unchecked(L1_SIZE / 2 + i);
                    let cl = i32::clamp(i32::from(l), 0, QA);
                    let cr = i32::clamp(i32::from(r), 0, QA);
                    let r = (cl * cr) / QA;
                    *output.get_unchecked_mut(i + a * L1_SIZE / 2) = r as u8;
                }
            }
        }
    }

    #[allow(clippy::needless_range_loop, clippy::cast_precision_loss)]
    fn propagate_l1(
        inputs: &Align64<[u8; L1_SIZE]>,
        weights: &Align64<[i8; L1_SIZE * L2_SIZE]>,
        biases: &Align64<[f32; L2_SIZE]>,
        output: &mut Align64<[f32; L2_SIZE]>,
    ) {
        const SUM_DIV: f32 = (QA * QB) as f32;
        // this is just autovec'd for the moment.
        let mut sums = [0; L2_SIZE];
        for i in 0..L1_SIZE {
            for j in 0..L2_SIZE {
                // SAFETY: `sums` is `L2_SIZE` long, `inputs` is `L1_SIZE` long,
                // and `weights` is `L1_SIZE * L2_SIZE` long. As such, the
                // indices that we construct are valid.
                unsafe {
                    *sums.get_unchecked_mut(j) +=
                        i32::from(*inputs.get_unchecked(i)) * i32::from(*weights.get_unchecked(j * L1_SIZE + i));
                }
            }
        }

        for i in 0..L2_SIZE {
            // convert to f32 and activate L1
            // SAFETY: `sums` is `L2_SIZE` long, and `output` is `L2_SIZE` long.
            // As such, the indices that we construct are valid.
            unsafe {
                let clipped =
                    f32::clamp((*sums.get_unchecked(i) as f32) / SUM_DIV + *biases.get_unchecked(i), 0.0, 1.0);
                *output.get_unchecked_mut(i) = clipped * clipped;
            }
        }
    }

    pub fn activate_ft_and_propagate_l1(
        us: &Align64<[i16; L1_SIZE]>,
        them: &Align64<[i16; L1_SIZE]>,
        weights: &Align64<[i8; L1_SIZE * L2_SIZE]>,
        biases: &Align64<[f32; L2_SIZE]>,
        output: &mut Align64<[f32; L2_SIZE]>,
    ) {
        let mut ft_outputs = Align64([0; L1_SIZE]);
        activate_ft(us, them, &mut ft_outputs);
        propagate_l1(&ft_outputs, weights, biases, output);
    }

    #[allow(clippy::needless_range_loop)]
    pub fn propagate_l2(
        inputs: &Align64<[f32; L2_SIZE]>,
        weights: &Align64<[f32; L2_SIZE * L3_SIZE]>,
        biases: &Align64<[f32; L3_SIZE]>,
        output: &mut Align64<[f32; L3_SIZE]>,
    ) {
        // this is just autovec'd for the moment.
        let mut sums = biases.clone();

        // affine transform for l2
        for i in 0..L2_SIZE {
            for j in 0..L3_SIZE {
                // SAFETY: `sums` is `L3_SIZE` long, `inputs` is `L2_SIZE` long,
                // and `weights` is `L2_SIZE * L3_SIZE` long. As such, the
                // indices that we construct are valid.
                unsafe {
                    *sums.get_unchecked_mut(j) += *inputs.get_unchecked(i) * *weights.get_unchecked(j * L2_SIZE + i);
                }
            }
        }

        // activate l2
        for i in 0..L3_SIZE {
            // SAFETY: `sums` is `L3_SIZE` long, and `output` is `L3_SIZE` long.
            // As such, the indices that we construct are valid.
            unsafe {
                let clipped = f32::clamp(*sums.get_unchecked(i), 0.0, 1.0);
                *output.get_unchecked_mut(i) = clipped * clipped;
            }
        }
    }

    pub fn propagate_l3(
        inputs: &Align64<[f32; L3_SIZE]>,
        weights: &Align64<[f32; L3_SIZE]>,
        bias: f32,
        output: &mut f32,
    ) {
        let mut sum = bias;

        for (i, w) in inputs.iter().zip(weights.iter()) {
            sum += *i * *w;
        }

        *output = sum;
    }
}

#[cfg(target_feature = "ssse3")]
mod x86simd {
    use super::super::{Align64, L1_SIZE, L2_SIZE, L3_SIZE, QA, QB};
    use crate::nnue::{
        network::L1_CHUNK_PER_32,
        simd::{self, VecI32, F32_CHUNK_SIZE, I16_CHUNK_SIZE, I32_CHUNK_SIZE, S, U8_CHUNK_SIZE},
    };
    use std::mem::MaybeUninit;

    const FT_SHIFT: u32 = 10;

    #[derive(Debug, Clone, Copy)]
    #[repr(C, align(16))]
    struct NNZEntry {
        indices: [u16; 8],
    }

    struct NNZTable {
        table: [NNZEntry; 256],
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    const NNZ_TABLE: NNZTable = {
        let mut table = [NNZEntry { indices: [0; 8] }; 256];

        let mut i = 0;
        while i < 256 {
            let mut j = i;
            let mut k = 0;
            while j != 0 {
                table[i].indices[k] = j.trailing_zeros() as u16;
                j &= j - 1;
                k += 1;
            }
            i += 1;
        }

        NNZTable { table }
    };

    // used in only one place, separate function for clarity.
    unsafe fn reinterpret_as_i32s(ptr: &Align64<[MaybeUninit<u8>; L1_SIZE]>) -> &Align64<[i32; L1_SIZE / 4]> {
        let ptr = std::ptr::from_ref(ptr);
        // check that the reference is aligned to the register alignment
        debug_assert!((ptr as usize) % std::mem::align_of::<i32>() == 0);
        debug_assert!((ptr as usize) % std::mem::align_of::<Align64<[i32; L1_SIZE / 4]>>() == 0);
        // cast:
        &*ptr.cast::<Align64<[i32; L1_SIZE / 4]>>()
    }

    unsafe fn find_nnz<'a>(
        input: &Align64<[i32; L1_SIZE / L1_CHUNK_PER_32]>,
        out: &'a mut Align64<[MaybeUninit<u16>; L1_SIZE / L1_CHUNK_PER_32]>,
    ) -> &'a [u16] {
        use std::arch::x86_64::_mm_add_epi16 as vec128_add;
        use std::arch::x86_64::_mm_load_si128 as vec128_load;
        use std::arch::x86_64::_mm_set1_epi16 as vec128_set_16;
        use std::arch::x86_64::_mm_setzero_si128 as vec128_zero;
        use std::arch::x86_64::_mm_storeu_si128 as vec128_storeu;

        const INPUT_SIMD_WIDTH: usize = std::mem::size_of::<VecI32>() / std::mem::size_of::<i32>();
        const CHUNK_SIZE: usize = max!(INPUT_SIMD_WIDTH, 8);
        const NUM_CHUNKS: usize = (L1_SIZE / L1_CHUNK_PER_32) / CHUNK_SIZE;
        const INPUTS_PER_CHUNK: usize = CHUNK_SIZE / INPUT_SIMD_WIDTH;
        const OUTPUTS_PER_CHUNK: usize = CHUNK_SIZE / 8;

        let mut count = 0;
        let mut base = vec128_zero();
        let increment = vec128_set_16(8);
        for i in 0..NUM_CHUNKS {
            // bitmask of nonzero values in this chunk
            let mut nnz = 0;
            for j in 0..INPUTS_PER_CHUNK {
                let input_chunk = simd::load_i32(input.get_unchecked((i * INPUTS_PER_CHUNK + j) * I32_CHUNK_SIZE));
                nnz |= u32::from(simd::nonzero_mask_i32(input_chunk)) << (j * INPUT_SIMD_WIDTH);
            }
            for j in 0..OUTPUTS_PER_CHUNK {
                let lookup = (nnz >> (j * 8)) & 0xFF;
                let offsets = vec128_load(std::ptr::from_ref(NNZ_TABLE.table.get_unchecked(lookup as usize)).cast());
                vec128_storeu(std::ptr::from_mut(out.get_unchecked_mut(count)).cast(), vec128_add(base, offsets));
                count += u32::count_ones(lookup) as usize;
                base = vec128_add(base, increment);
            }
        }

        // SAFETY: we have initialised this region of the array.
        std::slice::from_raw_parts(out.get_unchecked(0).as_ptr().cast(), count)
    }

    #[allow(
        clippy::too_many_lines,
        clippy::identity_op,
        clippy::erasing_op,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_ptr_alignment,
        clippy::cast_possible_wrap,
        clippy::needless_range_loop,
        clippy::similar_names
    )]
    pub fn activate_ft_and_propagate_l1(
        us: &Align64<[i16; L1_SIZE]>,
        them: &Align64<[i16; L1_SIZE]>,
        weights: &Align64<[i8; L1_SIZE * L2_SIZE]>,
        biases: &Align64<[f32; L2_SIZE]>,
        output: &mut Align64<[f32; L2_SIZE]>,
    ) {
        const L1_PAIR_COUNT: usize = L1_SIZE / 2;
        const L1_MUL: f32 = (1 << FT_SHIFT) as f32 / (QA * QA * QB) as f32;

        // SAFETY: Breaking it down by unsafe operations:
        // 1. get_unchecked[_mut]: We only ever index at most
        // div_ceil(L1_PAIR_COUNT - 1, I16_CHUNK_SIZE * 2) + I16_CHUNK_SIZE + L1_PAIR_COUNT
        // into the `acc` array. This is in bounds, as `acc` has length L1_PAIR_COUNT * 2.
        // Additionally, we only ever indexx at most div_ceil(L1_PAIR_COUNT - 1, I16_CHUNK_SIZE * 2) + L1_PAIR_COUNT
        // into the `ft_outputs` array. This is in bounds, as `ft_outputs` has length L1_PAIR_COUNT * 2.
        // 2. SIMD instructions: All of our loads and stores are aligned.
        // 3. Use of MaybeUninit: We always store into the entirety of `ft_outputs`, before
        // reading from it. Additionally, find_nnz returns a slice into `nnz` that it has
        // initialised, so we can soundly read from it.
        unsafe {
            let ft_zero = simd::zero_i16();
            let ft_one = simd::splat_i16(QA as i16);

            let mut ft_outputs: Align64<[MaybeUninit<u8>; L1_SIZE]> = MaybeUninit::uninit().assume_init();

            let mut offset = 0;
            for acc in [us, them] {
                for i in (0..L1_PAIR_COUNT).step_by(I16_CHUNK_SIZE * 2) {
                    let input0a = simd::load_i16(acc.get_unchecked(i + 0 + 0));
                    let input0b = simd::load_i16(acc.get_unchecked(i + I16_CHUNK_SIZE + 0));
                    let input1a = simd::load_i16(acc.get_unchecked(i + 0 + L1_PAIR_COUNT));
                    let input1b = simd::load_i16(acc.get_unchecked(i + I16_CHUNK_SIZE + L1_PAIR_COUNT));

                    let clipped0a = simd::min_i16(simd::max_i16(input0a, ft_zero), ft_one);
                    let clipped0b = simd::min_i16(simd::max_i16(input0b, ft_zero), ft_one);
                    let clipped1a = simd::min_i16(input1a, ft_one);
                    let clipped1b = simd::min_i16(input1b, ft_one);

                    let producta = simd::mul_high_i16(simd::shl_i16::<{ 16 - FT_SHIFT as S }>(clipped0a), clipped1a);
                    let productb = simd::mul_high_i16(simd::shl_i16::<{ 16 - FT_SHIFT as S }>(clipped0b), clipped1b);
                    simd::store_u8(
                        std::ptr::from_mut(ft_outputs.get_unchecked_mut(offset + i)).cast(),
                        simd::pack_i16_to_unsigned_and_permute(producta, productb),
                    );
                }
                offset += L1_PAIR_COUNT;
            }

            // logging for permutation
            #[cfg(feature = "nnz-counts")]
            for (i, elem) in ft_outputs.iter().enumerate() {
                let elem = elem.assume_init();
                let nnz = elem != 0;
                if nnz {
                    super::NNZ_COUNTS[i].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }

            // &Align64<[MaybeUninit<u8>; L1_SIZE]>) -> &Align64<[i32; L1_SIZE / 4]>
            let input32 = reinterpret_as_i32s(&ft_outputs);

            // Compute the non-zero indices.
            let mut nnz: Align64<[MaybeUninit<u16>; L1_SIZE / L1_CHUNK_PER_32]> = MaybeUninit::uninit().assume_init();
            let nnz_slice = find_nnz(input32, &mut nnz);

            let mut sums = [0; L2_SIZE];

            for &i in nnz_slice {
                // load the non-zero activation, and splat it into a SIMD register.
                let input = simd::splat_i32(*input32.get_unchecked(i as usize));
                // compute the index into the weights matrix.
                let w_offset = i as usize * L2_SIZE * L1_CHUNK_PER_32;
                // for each SIMD-block in the row, compute the product
                // of the non-zero activation with the corresponding
                // weight, and add it to the accumulator.
                for k in 0..L2_SIZE / F32_CHUNK_SIZE {
                    simd::store_i32(
                        sums.get_unchecked_mut(k * F32_CHUNK_SIZE),
                        simd::mul_add_u8_to_i32(
                            simd::load_i32(sums.get_unchecked(k * F32_CHUNK_SIZE)),
                            simd::reinterpret_i32s_as_i8s(input),
                            simd::load_i8(weights.get_unchecked(w_offset + k * U8_CHUNK_SIZE)),
                        ),
                    );
                }
            }

            let zero = simd::zero_f32();
            let one = simd::splat_f32(1.0);
            let sum_mul = simd::splat_f32(L1_MUL);
            for i in 0..L2_SIZE / F32_CHUNK_SIZE {
                // Convert into floats, and activate L1
                let bias = simd::load_f32(biases.get_unchecked(i * F32_CHUNK_SIZE));
                let sum = simd::mul_add_f32(
                    simd::i32_to_f32(simd::load_i32(sums.get_unchecked(i * F32_CHUNK_SIZE))),
                    sum_mul,
                    bias,
                );
                let clipped = simd::min_f32(simd::max_f32(sum, zero), one);
                let squared = simd::mul_f32(clipped, clipped);
                simd::store_f32(output.get_unchecked_mut(i * F32_CHUNK_SIZE), squared);
            }
        }
    }

    #[allow(clippy::needless_range_loop, clippy::cast_ptr_alignment)]
    pub fn propagate_l2(
        inputs: &Align64<[f32; L2_SIZE]>,
        weights: &Align64<[f32; L2_SIZE * L3_SIZE]>,
        biases: &Align64<[f32; L3_SIZE]>,
        output: &mut Align64<[f32; L3_SIZE]>,
    ) {
        // SAFETY: Breaking it down by unsafe operations:
        // 1. get_unchecked[_mut]: We only ever index at most (L3_SIZE / F32_CHUNK_SIZE - 1) * F32_CHUNK_SIZE
        // into the `sums` and `biases` arrays. This is in bounds, as `sums` has length L3_SIZE and
        // `biases` has length L3_SIZE. We only ever index at most
        // (L2_SIZE - 1) * L3_SIZE + (L3_SIZE / F32_CHUNK_SIZE - 1) * F32_CHUNK_SIZE
        // into the `weights` array. This is in bounds, as `weights` has length L2_SIZE * L3_SIZE.
        // We only ever index at most L2_SIZE - 1 into the `inputs` array. This is in bounds, as `inputs`
        // has length L2_SIZE.
        // 2. SIMD instructions: All of our loads and stores are aligned.
        unsafe {
            let mut sums = [0.0; L3_SIZE];

            for i in 0..L3_SIZE / F32_CHUNK_SIZE {
                simd::store_f32(
                    sums.get_unchecked_mut(i * F32_CHUNK_SIZE),
                    simd::load_f32(biases.get_unchecked(i * F32_CHUNK_SIZE)),
                );
            }

            for i in 0..L2_SIZE {
                let input_vec = simd::splat_f32(*inputs.get_unchecked(i));
                for j in 0..L3_SIZE / F32_CHUNK_SIZE {
                    simd::store_f32(
                        sums.get_unchecked_mut(j * F32_CHUNK_SIZE),
                        simd::mul_add_f32(
                            input_vec,
                            simd::load_f32(weights.get_unchecked(i * L3_SIZE + j * F32_CHUNK_SIZE)),
                            simd::load_f32(sums.get_unchecked(j * F32_CHUNK_SIZE)),
                        ),
                    );
                }
            }

            // Activate L2
            let one = simd::splat_f32(1.0);
            for i in 0..L3_SIZE / F32_CHUNK_SIZE {
                let clipped = simd::min_f32(
                    simd::max_f32(simd::load_f32(sums.get_unchecked(i * F32_CHUNK_SIZE)), simd::zero_f32()),
                    one,
                );
                let squared = simd::mul_f32(clipped, clipped);
                simd::store_f32(output.get_unchecked_mut(i * F32_CHUNK_SIZE), squared);
            }
        }
    }

    pub fn propagate_l3(
        inputs: &Align64<[f32; L3_SIZE]>,
        weights: &Align64<[f32; L3_SIZE]>,
        bias: f32,
        output: &mut f32,
    ) {
        // SAFETY: Breaking it down by unsafe operations:
        // 1. get_unchecked[_mut]: We only ever index at most (L3_SIZE / F32_CHUNK_SIZE - 1) * F32_CHUNK_SIZE
        // into the `weights` and `inputs` arrays. This is in bounds, as `weights` has length L3_SIZE and
        // `inputs` has length L3_SIZE.
        // 2. SIMD instructions: All of our loads and stores are aligned.
        unsafe {
            let mut sum_vec = simd::zero_f32();

            // Affine transform for L3
            for i in (0..L3_SIZE).step_by(F32_CHUNK_SIZE) {
                let weight_vec = simd::load_f32(weights.get_unchecked(i));
                let input_vec = simd::load_f32(inputs.get_unchecked(i));
                sum_vec = simd::mul_add_f32(input_vec, weight_vec, sum_vec);
            }

            *output = bias + simd::sum_f32(sum_vec);
        }
    }
}

#[cfg(target_feature = "ssse3")]
pub use x86simd::*;

#[cfg(not(target_feature = "ssse3"))]
pub use generic::*;

// logging for permutation
#[cfg(feature = "nnz-counts")]
pub static NNZ_COUNTS: [std::sync::atomic::AtomicU64; super::L1_SIZE] =
    { unsafe { std::mem::transmute([0u64; super::L1_SIZE]) } };
