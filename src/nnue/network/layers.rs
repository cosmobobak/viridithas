use super::{Align64, FT_SHIFT, L1_SIZE, L2_SIZE, L3_SIZE, QA, QB};

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
        const SUM_DIV: f32 = (QA * QB) as f32;
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
    use crate::nnue::{
        network::L1_CHUNK_PER_32,
        simd::{
            vec_add_ps, vec_cvtepi32_ps, vec_div_ps, vec_dpbusd_epi32, vec_load_epi16, vec_load_ps, vec_max_epi16,
            vec_max_ps, vec_min_epi16, vec_min_ps, vec_mul_add_ps, vec_mul_ps, vec_mulhi_epi16, vec_nnz_mask,
            vec_packus_permute_epi16, vec_reduce_add_ps, vec_set1_epi16, vec_set1_epi32, vec_set1_ps, vec_slli_epi16,
            vec_store_epi32, vec_store_ps, vec_zero_epi16, vec_zero_epi32, vec_zero_ps, vepi32, vepi8, vps32,
            F32_CHUNK_SIZE, I16_CHUNK_SIZE, I32_CHUNK_SIZE,
        },
    };

    #[derive(Debug, Clone, Copy)]
    struct NNZEntry {
        indices: [u8; 8],
        count: u8,
    }

    struct NNZTable {
        table: [NNZEntry; 256],
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    const NNZ_TABLE: NNZTable = {
        let mut table = [NNZEntry { indices: [0; 8], count: 0 }; 256];

        let mut i = 0u16;
        while i < 256 {
            table[i as usize].count = i.count_ones() as u8;
            let mut j = i;
            let mut k = 0;
            while j != 0 {
                table[i as usize].indices[k] = j.trailing_zeros() as u8;
                j &= j - 1;
                k += 1;
            }
            i += 1;
        }

        NNZTable { table }
    };

    #[allow(
        clippy::too_many_lines,
        clippy::identity_op,
        clippy::erasing_op,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_ptr_alignment,
        clippy::needless_range_loop
    )]
    pub fn activate_ft_and_propagate_l1(
        us: &Align64<[i16; L1_SIZE]>,
        them: &Align64<[i16; L1_SIZE]>,
        weights: &Align64<[i8; L1_SIZE * L2_SIZE]>,
        biases: &Align64<[f32; L2_SIZE]>,
        output: &mut Align64<[f32; L2_SIZE]>,
    ) {
        const L1_PAIR_COUNT: usize = L1_SIZE / 2;
        const L1_DIV: f32 = (QA * QA * QB) as f32 / (1 << FT_SHIFT) as f32;

        // SAFETY: we never hold multiple mutable references, we never mutate immutable memory, etc.
        unsafe {
            let zero = vec_zero_epi16();
            let ft_one = vec_set1_epi16(QA as i16);

            let mut sums = Align64([vec_zero_epi32(); L2_SIZE / I32_CHUNK_SIZE]);
            let mut registers = Align64([0i32; size_of::<vepi32>() / size_of::<i32>()]);

            let mut offset = 0;
            for acc in [us, them] {
                for i in (0..L1_PAIR_COUNT).step_by(I16_CHUNK_SIZE * 2) {
                    let i1_0 = vec_load_epi16(&acc[i]);
                    let i1_1 = vec_load_epi16(&acc[i + I16_CHUNK_SIZE]);

                    let i2_0 = vec_load_epi16(&acc[i + L1_PAIR_COUNT]);
                    let i2_1 = vec_load_epi16(&acc[i + L1_PAIR_COUNT + I16_CHUNK_SIZE]);

                    let c1_0 = vec_max_epi16(vec_min_epi16(i1_0, ft_one), zero);
                    let c1_1 = vec_max_epi16(vec_min_epi16(i1_1, ft_one), zero);

                    let c2_0 = vec_slli_epi16::<{ 16 - FT_SHIFT }>(vec_min_epi16(i2_0, ft_one));
                    let c2_1 = vec_slli_epi16::<{ 16 - FT_SHIFT }>(vec_min_epi16(i2_1, ft_one));

                    let p_0 = vec_mulhi_epi16(c1_0, c2_0);
                    let p_1 = vec_mulhi_epi16(c1_1, c2_1);

                    let product = vec_packus_permute_epi16(p_0, p_1);

                    vec_store_epi32(&mut registers[0], product);

                    let nnz_mask = vec_nnz_mask(product);

                    for lookup in 0..registers.len() / 8 {
                        let mask_slice = (nnz_mask >> (lookup * 8)) & 0xFF;
                        let nnz_entry = NNZ_TABLE.table[mask_slice as usize];
                        for j in 0..nnz_entry.count {
                            let nnz = nnz_entry.indices[j as usize] as usize;
                            let input32 = vec_set1_epi32(registers[nnz + lookup * 8]);
                            let weight = std::ptr::from_ref(
                                &weights[((nnz + 8 * lookup) * L1_CHUNK_PER_32 + i + offset) * L2_SIZE],
                            )
                            .cast::<vepi8>();
                            for k in 0..L2_SIZE / I32_CHUNK_SIZE {
                                sums[k] = vec_dpbusd_epi32(sums[k], input32, *weight.add(k));
                            }
                        }
                    }
                }
                offset += L1_PAIR_COUNT;
            }

            let one = vec_set1_ps(1.0);
            for i in 0..L2_SIZE / F32_CHUNK_SIZE {
                // Convert into floats, and activate L1
                let bias_vec = vec_load_ps(&biases[i * F32_CHUNK_SIZE]);
                let sum_div = vec_set1_ps(L1_DIV);
                let sum_ps = vec_add_ps(vec_div_ps(vec_cvtepi32_ps(sums[i]), sum_div), bias_vec);
                let clipped = vec_min_ps(vec_max_ps(sum_ps, vec_zero_ps()), one);
                let squared = vec_mul_ps(clipped, clipped);
                vec_store_ps(&mut output[i * F32_CHUNK_SIZE], squared);
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
                sum_vecs[i] = vec_load_ps(&biases[i * F32_CHUNK_SIZE]);
            }

            for i in 0..L2_SIZE {
                let input_vec = vec_set1_ps(inputs[i]);
                let weight = std::ptr::from_ref(&weights[i * L3_SIZE]).cast::<vps32>();
                for j in 0..L3_SIZE / F32_CHUNK_SIZE {
                    sum_vecs[j] = vec_mul_add_ps(input_vec, *weight.add(j), sum_vecs[j]);
                }
            }

            // Activate L2
            let one = vec_set1_ps(1.0);
            for i in 0..L3_SIZE / F32_CHUNK_SIZE {
                let clipped = vec_min_ps(vec_max_ps(sum_vecs[i], vec_zero_ps()), one);
                let squared = vec_mul_ps(clipped, clipped);
                vec_store_ps(&mut output[i * F32_CHUNK_SIZE], squared);
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
                let weight_vec = vec_load_ps(&weights[i]);
                let input_vec = vec_load_ps(&inputs[i]);
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
