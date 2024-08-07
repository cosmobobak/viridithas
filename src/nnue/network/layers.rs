use super::{Align64, FT_SHIFT, L1_SIZE, L2_SIZE, L3_SIZE, QA, QAB, QB};

#[cfg(not(target_feature = "avx2"))]
mod generic {
    use super::*;

    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::cast_sign_loss)]
    fn activate_ft(us: &Align64<[i16; L1_SIZE]>, them: &Align64<[i16; L1_SIZE]>, output: &mut Align64<[u8; L1_SIZE]>) {
        // this is just autovec'd for the moment.
        for (a, acc) in [us, them].into_iter().enumerate() {
            for i in 0..L1_SIZE / 2 {
                let l = acc.0[i];
                let r = acc.0[L1_SIZE / 2 + i];
                let cl = i32::clamp(i32::from(l), 0, QA);
                let cr = i32::clamp(i32::from(r), 0, QA);
                let r = (cl * cr) / QA;
                output.0[i + a * L1_SIZE / 2] = r as u8;
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
        const SUM_DIV: f32 = QAB as f32;
        // this is just autovec'd for the moment.
        let mut sums = [0; L2_SIZE];
        for i in 0..L1_SIZE {
            for j in 0..L2_SIZE {
                sums[j] += i32::from(inputs.0[i]) * i32::from(weights.0[j * L1_SIZE + i]);
            }
        }

        for i in 0..L2_SIZE {
            // convert to f32 and activate L1
            let clipped = f32::clamp((sums[i] as f32) / SUM_DIV + biases.0[i], 0.0, 1.0);
            output.0[i] = clipped * clipped;
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
        let mut sums = biases.0;

        // affine transform for l2
        for i in 0..L2_SIZE {
            for j in 0..L3_SIZE {
                sums[j] += inputs.0[i] * weights.0[j * L2_SIZE + i];
            }
        }

        // activate l2
        for i in 0..L3_SIZE {
            let clipped = f32::clamp(sums[i], 0.0, 1.0);
            output.0[i] = clipped * clipped;
        }
    }

    pub fn propagate_l3(
        inputs: &Align64<[f32; L3_SIZE]>,
        weights: &Align64<[f32; L3_SIZE]>,
        bias: f32,
        output: &mut f32,
    ) {
        let mut sum = bias;

        for i in 0..L3_SIZE {
            sum += inputs.0[i] * weights.0[i];
        }

        *output = sum;
    }
}

#[cfg(target_feature = "avx2")]
mod avx2 {
    use super::{Align64, FT_SHIFT, L1_SIZE, L2_SIZE, L3_SIZE, QA, QB};
    use crate::nnue::simd::{
        vec_add_epi32, vec_cvtepi32_ps, vec_dpbusd_epi32, vec_load_epi16, vec_load_epi8, vec_load_ps, vec_max_epi16,
        vec_max_ps, vec_min_epi16, vec_min_ps, vec_mul_add_ps, vec_mul_ps, vec_mulhi_epi16, vec_packus_permute_epi16,
        vec_set1_epi16, vec_set1_epi32, vec_set1_ps, vec_slli_epi16, vec_store_epiu8, vec_store_ps, vec_zero_epi16,
        vec_zero_epi32, vec_zero_ps, I16_CHUNK_SIZE, I8_CHUNK_SIZE_I32, U8_CHUNK_SIZE,
    };

    #[allow(
        clippy::too_many_lines,
        clippy::identity_op,
        clippy::erasing_op,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss
    )]
    pub fn activate_ft_and_propagate_l1(
        us: &Align64<[i16; L1_SIZE]>,
        them: &Align64<[i16; L1_SIZE]>,
        weights: &Align64<[i8; L1_SIZE * L2_SIZE]>,
        biases: &Align64<[f32; L2_SIZE]>,
        output: &mut Align64<[f32; L2_SIZE]>,
    ) {
        const L1_PAIR_COUNT: usize = L1_SIZE / 2;

        // SAFETY: we never hold multiple mutable references, we never mutate immutable memory, etc.
        unsafe {
            let mut ft_outputs = Align64([0u8; L1_SIZE]);

            let zero = vec_zero_epi16();
            let ft_one = vec_set1_epi16(QA as i16);

            // stm perspective
            for i in (0..L1_PAIR_COUNT).step_by(I16_CHUNK_SIZE * 4) {
                let mut i1_0 = vec_load_epi16(&us.0[i + I16_CHUNK_SIZE * 0]);
                let mut i1_1 = vec_load_epi16(&us.0[i + I16_CHUNK_SIZE * 1]);
                let mut i1_2 = vec_load_epi16(&us.0[i + I16_CHUNK_SIZE * 2]);
                let mut i1_3 = vec_load_epi16(&us.0[i + I16_CHUNK_SIZE * 3]);

                let mut i2_0 = vec_load_epi16(&us.0[i + L1_PAIR_COUNT + I16_CHUNK_SIZE * 0]);
                let mut i2_1 = vec_load_epi16(&us.0[i + L1_PAIR_COUNT + I16_CHUNK_SIZE * 1]);
                let mut i2_2 = vec_load_epi16(&us.0[i + L1_PAIR_COUNT + I16_CHUNK_SIZE * 2]);
                let mut i2_3 = vec_load_epi16(&us.0[i + L1_PAIR_COUNT + I16_CHUNK_SIZE * 3]);

                i1_0 = vec_min_epi16(i1_0, ft_one);
                i1_1 = vec_min_epi16(i1_1, ft_one);
                i1_2 = vec_min_epi16(i1_2, ft_one);
                i1_3 = vec_min_epi16(i1_3, ft_one);

                i2_0 = vec_min_epi16(i2_0, ft_one);
                i2_1 = vec_min_epi16(i2_1, ft_one);
                i2_2 = vec_min_epi16(i2_2, ft_one);
                i2_3 = vec_min_epi16(i2_3, ft_one);

                i1_0 = vec_max_epi16(i1_0, zero);
                i1_1 = vec_max_epi16(i1_1, zero);
                i1_2 = vec_max_epi16(i1_2, zero);
                i1_3 = vec_max_epi16(i1_3, zero);

                i2_0 = vec_slli_epi16::<6>(i2_0);
                i2_1 = vec_slli_epi16::<6>(i2_1);
                i2_2 = vec_slli_epi16::<6>(i2_2);
                i2_3 = vec_slli_epi16::<6>(i2_3);

                let p_0 = vec_mulhi_epi16(i1_0, i2_0);
                let p_1 = vec_mulhi_epi16(i1_1, i2_1);
                let p_2 = vec_mulhi_epi16(i1_2, i2_2);
                let p_3 = vec_mulhi_epi16(i1_3, i2_3);

                let packed0 = vec_packus_permute_epi16(p_0, p_1);
                let packed1 = vec_packus_permute_epi16(p_2, p_3);

                vec_store_epiu8(&mut ft_outputs.0[i + U8_CHUNK_SIZE * 0], packed0);
                vec_store_epiu8(&mut ft_outputs.0[i + U8_CHUNK_SIZE * 1], packed1);
            }

            // nstm perspective
            for i in (0..L1_PAIR_COUNT).step_by(I16_CHUNK_SIZE * 4) {
                let mut i1_0 = vec_load_epi16(&them.0[i + I16_CHUNK_SIZE * 0]);
                let mut i1_1 = vec_load_epi16(&them.0[i + I16_CHUNK_SIZE * 1]);
                let mut i1_2 = vec_load_epi16(&them.0[i + I16_CHUNK_SIZE * 2]);
                let mut i1_3 = vec_load_epi16(&them.0[i + I16_CHUNK_SIZE * 3]);

                let mut i2_0 = vec_load_epi16(&them.0[i + L1_PAIR_COUNT + I16_CHUNK_SIZE * 0]);
                let mut i2_1 = vec_load_epi16(&them.0[i + L1_PAIR_COUNT + I16_CHUNK_SIZE * 1]);
                let mut i2_2 = vec_load_epi16(&them.0[i + L1_PAIR_COUNT + I16_CHUNK_SIZE * 2]);
                let mut i2_3 = vec_load_epi16(&them.0[i + L1_PAIR_COUNT + I16_CHUNK_SIZE * 3]);

                i1_0 = vec_min_epi16(i1_0, ft_one);
                i1_1 = vec_min_epi16(i1_1, ft_one);
                i1_2 = vec_min_epi16(i1_2, ft_one);
                i1_3 = vec_min_epi16(i1_3, ft_one);

                i2_0 = vec_min_epi16(i2_0, ft_one);
                i2_1 = vec_min_epi16(i2_1, ft_one);
                i2_2 = vec_min_epi16(i2_2, ft_one);
                i2_3 = vec_min_epi16(i2_3, ft_one);

                i1_0 = vec_max_epi16(i1_0, zero);
                i1_1 = vec_max_epi16(i1_1, zero);
                i1_2 = vec_max_epi16(i1_2, zero);
                i1_3 = vec_max_epi16(i1_3, zero);

                i2_0 = vec_slli_epi16::<6>(i2_0);
                i2_1 = vec_slli_epi16::<6>(i2_1);
                i2_2 = vec_slli_epi16::<6>(i2_2);
                i2_3 = vec_slli_epi16::<6>(i2_3);

                let p_0 = vec_mulhi_epi16(i1_0, i2_0);
                let p_1 = vec_mulhi_epi16(i1_1, i2_1);
                let p_2 = vec_mulhi_epi16(i1_2, i2_2);
                let p_3 = vec_mulhi_epi16(i1_3, i2_3);

                let packed0 = vec_packus_permute_epi16(p_0, p_1);
                let packed1 = vec_packus_permute_epi16(p_2, p_3);

                vec_store_epiu8(&mut ft_outputs.0[i + L1_PAIR_COUNT + U8_CHUNK_SIZE * 0], packed0);
                vec_store_epiu8(&mut ft_outputs.0[i + L1_PAIR_COUNT + U8_CHUNK_SIZE * 1], packed1);
            }

            let mut l1_intermediate0 = vec_zero_epi32();
            let mut l1_intermediate1 = vec_zero_epi32();
            let mut l1_intermediate2 = vec_zero_epi32();
            let mut l1_intermediate3 = vec_zero_epi32();

            let ft_outputs_i32 = std::slice::from_raw_parts(
                std::ptr::from_ref(&ft_outputs).cast::<i32>(),
                ft_outputs.len() / I8_CHUNK_SIZE_I32,
            );

            for idx in (0..L1_SIZE).step_by(I8_CHUNK_SIZE_I32 * 4) {
                let weights_start = idx * L2_SIZE;

                let i0 = vec_set1_epi32(ft_outputs_i32[idx / I8_CHUNK_SIZE_I32 + 0]);
                let i1 = vec_set1_epi32(ft_outputs_i32[idx / I8_CHUNK_SIZE_I32 + 1]);
                let i2 = vec_set1_epi32(ft_outputs_i32[idx / I8_CHUNK_SIZE_I32 + 2]);
                let i3 = vec_set1_epi32(ft_outputs_i32[idx / I8_CHUNK_SIZE_I32 + 3]);

                let w0 = vec_load_epi8(&weights.0[weights_start + I8_CHUNK_SIZE_I32 * L2_SIZE * 0]);
                let w1 = vec_load_epi8(&weights.0[weights_start + I8_CHUNK_SIZE_I32 * L2_SIZE * 1]);
                let w2 = vec_load_epi8(&weights.0[weights_start + I8_CHUNK_SIZE_I32 * L2_SIZE * 2]);
                let w3 = vec_load_epi8(&weights.0[weights_start + I8_CHUNK_SIZE_I32 * L2_SIZE * 3]);

                l1_intermediate0 = vec_dpbusd_epi32(l1_intermediate0, i0, w0);
                l1_intermediate1 = vec_dpbusd_epi32(l1_intermediate1, i1, w1);
                l1_intermediate2 = vec_dpbusd_epi32(l1_intermediate2, i2, w2);
                l1_intermediate3 = vec_dpbusd_epi32(l1_intermediate3, i3, w3);
            }

            let l1_half_sums_0 = vec_add_epi32(l1_intermediate0, l1_intermediate1);
            let l1_half_sums_1 = vec_add_epi32(l1_intermediate2, l1_intermediate3);

            let l1_sums_i32 = vec_add_epi32(l1_half_sums_0, l1_half_sums_1);

            let l1b = vec_load_ps(&biases.0[0]);

            let mut l1_sums = vec_cvtepi32_ps(l1_sums_i32);

            let rqf = ((1 << FT_SHIFT) as f32) / (QA * QA * QB) as f32;
            let rq = vec_set1_ps(rqf);

            l1_sums = vec_mul_add_ps(l1_sums, rq, l1b);
            l1_sums = vec_min_ps(l1_sums, vec_set1_ps(1.0));
            l1_sums = vec_max_ps(l1_sums, vec_zero_ps());
            l1_sums = vec_mul_ps(l1_sums, l1_sums);

            vec_store_ps(&mut output.0[0], l1_sums);
        }
    }

    #[allow(clippy::needless_range_loop)]
    pub fn propagate_l2(
        inputs: &Align64<[f32; L2_SIZE]>,
        weights: &Align64<[f32; L2_SIZE * L3_SIZE]>,
        biases: &Align64<[f32; L3_SIZE]>,
        output: &mut Align64<[f32; L3_SIZE]>,
    ) {
        // this is just autovec'd for the moment.
        let mut sums = biases.0;

        // affine transform for l2
        for i in 0..L2_SIZE {
            for j in 0..L3_SIZE {
                sums[j] += inputs.0[i] * weights.0[j * L2_SIZE + i];
            }
        }

        // activate l2
        for i in 0..L3_SIZE {
            let clipped = f32::clamp(sums[i], 0.0, 1.0);
            output.0[i] = clipped * clipped;
        }
    }

    pub fn propagate_l3(
        inputs: &Align64<[f32; L3_SIZE]>,
        weights: &Align64<[f32; L3_SIZE]>,
        bias: f32,
        output: &mut f32,
    ) {
        let mut sum = bias;

        for i in 0..L3_SIZE {
            sum += inputs.0[i] * weights.0[i];
        }

        *output = sum;
    }
}

#[cfg(target_feature = "avx2")]
pub use avx2::*;

#[cfg(not(target_feature = "avx2"))]
pub use generic::*;
