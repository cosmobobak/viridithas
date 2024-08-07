use super::{Align64, FT_SHIFT, L1_SIZE, L2_SIZE, L3_SIZE, QA, QB};

#[cfg(not(target_feature = "avx2"))]
mod generic {
    use super::*;

    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::cast_sign_loss)]
    fn activate_ft(us: &Align64<[i16; L1_SIZE]>, them: &Align64<[i16; L1_SIZE]>, output: &mut Align64<[u8; L1_SIZE]>) {
        // this is just autovec'd for the moment.
        for (a, acc) in [us, them].into_iter().enumerate() {
            for i in 0..L1_SIZE / 2 {
                let l = *acc.get_unchecked(i);
                let r = *acc.get_unchecked(L1_SIZE / 2 + i);
                let cl = i32::clamp(i32::from(l), 0, QA);
                let cr = i32::clamp(i32::from(r), 0, QA);
                let r = (cl * cr) / QA;
                *output.get_unchecked_mut(i + a * L1_SIZE / 2) = r as u8;
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
                *sums.get_unchecked_mut(j) += i32::from(*inputs.get_unchecked(i)) * i32::from(*weights.get_unchecked(j * L1_SIZE + i));
            }
        }

        for i in 0..L2_SIZE {
            // convert to f32 and activate L1
            let clipped = f32::clamp((*sums.get_unchecked(i) as f32) / SUM_DIV + *biases.get_unchecked(i), 0.0, 1.0);
            *output.get_unchecked_mut(i) = clipped * clipped;
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
        let mut sums = biases;

        // affine transform for l2
        for i in 0..L2_SIZE {
            for j in 0..L3_SIZE {
                *sums.get_unchecked_mut(j) += *inputs.get_unchecked(i) * *weights.get_unchecked(j * L2_SIZE + i);
            }
        }

        // activate l2
        for i in 0..L3_SIZE {
            let clipped = f32::clamp(*sums.get_unchecked(i), 0.0, 1.0);
            *output.get_unchecked_mut(i) = clipped * clipped;
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

#[cfg(target_feature = "avx2")]
mod avx2 {
    use super::{Align64, FT_SHIFT, L1_SIZE, L2_SIZE, L3_SIZE, QA, QB};
    use crate::nnue::{
        network::L1_CHUNK_PER_32,
        simd::{
            vec_cvtepi32_ps, vec_dpbusd_epi32, vec_load_epi16, vec_load_ps, vec_max_epi16, vec_max_ps, vec_min_epi16, vec_min_ps, vec_mul_add_ps, vec_mul_ps, vec_mulhi_epi16, vec_packus_permute_epi16, vec_reduce_add_ps, vec_set1_epi16, vec_set1_epi32, vec_set1_ps, vec_slli_epi16, vec_store_epiu8, vec_store_ps, vec_zero_epi16, vec_zero_epi32, vec_zero_ps, vepi8, vps32, F32_CHUNK_SIZE, I16_CHUNK_SIZE
        },
    };

    #[allow(
        clippy::too_many_lines,
        clippy::identity_op,
        clippy::erasing_op,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_ptr_alignment,
        clippy::needless_range_loop,
        clippy::similar_names,
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

        // SAFETY: we never hold multiple mutable references, we never mutate immutable memory, etc.
        unsafe {
            let ft_zero = vec_zero_epi16();
            let ft_one = vec_set1_epi16(QA as i16);

            let mut ft_outputs = Align64([0; L1_SIZE]);

            let mut offset = 0;
            for acc in [us, them] {
                for i in (0..L1_PAIR_COUNT).step_by(I16_CHUNK_SIZE * 2) {
                    let input0a   = vec_load_epi16(acc.get_unchecked(i + 0              + 0));
                    let input0b   = vec_load_epi16(acc.get_unchecked(i + I16_CHUNK_SIZE + 0));
                    let input1a   = vec_load_epi16(acc.get_unchecked(i + 0              + L1_PAIR_COUNT));
                    let input1b   = vec_load_epi16(acc.get_unchecked(i + I16_CHUNK_SIZE + L1_PAIR_COUNT));

                    let clipped0a = vec_min_epi16(vec_max_epi16(input0a, ft_zero), ft_one);
                    let clipped0b = vec_min_epi16(vec_max_epi16(input0b, ft_zero), ft_one);
                    let clipped1a = vec_min_epi16(input1a, ft_one);
                    let clipped1b = vec_min_epi16(input1b, ft_one);

                    let producta  = vec_mulhi_epi16(vec_slli_epi16::<{ 16 - FT_SHIFT }>(clipped0a), clipped1a);
                    let productb  = vec_mulhi_epi16(vec_slli_epi16::<{ 16 - FT_SHIFT }>(clipped0b), clipped1b);
                    vec_store_epiu8(ft_outputs.get_unchecked_mut(offset + i), vec_packus_permute_epi16(producta, productb));
                }
                offset += L1_PAIR_COUNT;
            }

            let mut sums = [vec_zero_epi32(); L2_SIZE / F32_CHUNK_SIZE];
            let inputs32 = std::ptr::from_ref(&ft_outputs).cast::<i32>();
            for i in 0..L1_SIZE / L1_CHUNK_PER_32 {
                let input = vec_set1_epi32(*inputs32.add(i));
                let weight = std::ptr::from_ref(weights.get_unchecked(i * L1_CHUNK_PER_32 * L2_SIZE)).cast::<vepi8>();
                for j in 0..L2_SIZE / F32_CHUNK_SIZE {
                    *sums.get_unchecked_mut(j) = vec_dpbusd_epi32(*sums.get_unchecked(j), input, *weight.add(j));
                }
            }

            let zero    = vec_zero_ps();
            let one     = vec_set1_ps(1.0);
            let sum_mul  = vec_set1_ps(L1_MUL);
            for i in 0..L2_SIZE / F32_CHUNK_SIZE {
                // Convert into floats, and activate L1
                let bias_vec = vec_load_ps(biases.get_unchecked(i * F32_CHUNK_SIZE));
                let sum_ps   = vec_mul_add_ps(vec_cvtepi32_ps(*sums.get_unchecked(i)), sum_mul, bias_vec);
                let clipped  = vec_min_ps(vec_max_ps(sum_ps, zero), one);
                let squared  = vec_mul_ps(clipped, clipped);
                vec_store_ps(output.get_unchecked_mut(i * F32_CHUNK_SIZE), squared);
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
        // SAFETY: we never hold multiple mutable references, we never mutate immutable memory, etc.
        unsafe {
            let mut sum_vecs = [vec_zero_ps(); L3_SIZE / F32_CHUNK_SIZE];

            for i in 0..L3_SIZE / F32_CHUNK_SIZE {
                *sum_vecs.get_unchecked_mut(i) = vec_load_ps(biases.get_unchecked(i * F32_CHUNK_SIZE));
            }

            for i in 0..L2_SIZE {
                let input_vec = vec_set1_ps(*inputs.get_unchecked(i));
                let weight = std::ptr::from_ref(weights.get_unchecked(i * L3_SIZE)).cast::<vps32>();
                for j in 0..L3_SIZE / F32_CHUNK_SIZE {
                    *sum_vecs.get_unchecked_mut(j) = vec_mul_add_ps(input_vec, *weight.add(j), *sum_vecs.get_unchecked(j));
                }
            }

            // Activate L2
            let one = vec_set1_ps(1.0);
            for i in 0..L3_SIZE / F32_CHUNK_SIZE {
                let clipped = vec_min_ps(vec_max_ps(*sum_vecs.get_unchecked(i), vec_zero_ps()), one);
                let squared = vec_mul_ps(clipped, clipped);
                vec_store_ps(output.get_unchecked_mut(i * F32_CHUNK_SIZE), squared);
            }
        }
    }

    pub fn propagate_l3(
        inputs: &Align64<[f32; L3_SIZE]>,
        weights: &Align64<[f32; L3_SIZE]>,
        bias: f32,
        output: &mut f32,
    ) {
        // SAFETY: we never hold multiple mutable references, we never mutate immutable memory, etc.
        unsafe {
            let mut sum_vec = vec_zero_ps();

            // Affine transform for L3
            for i in (0..L3_SIZE).step_by(F32_CHUNK_SIZE) {
                let weight_vec = vec_load_ps(weights.get_unchecked(i));
                let input_vec = vec_load_ps(inputs.get_unchecked(i));
                sum_vec = vec_mul_add_ps(input_vec, weight_vec, sum_vec);
            }

            *output = bias + vec_reduce_add_ps(sum_vec);
        }
    }
}

#[cfg(target_feature = "avx2")]
pub use avx2::*;

#[cfg(not(target_feature = "avx2"))]
pub use generic::*;
