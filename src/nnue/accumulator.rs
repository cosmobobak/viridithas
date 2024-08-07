// use crate::{board::Board, piece::PieceType};

use crate::piece::Colour;

use super::network::{feature::FeatureIndex, Align64, MovedPiece, PovUpdate, UpdateBuffer, INPUT, L1_SIZE};

/// Activations of the hidden layer.
#[derive(Debug, Clone)]
pub struct Accumulator {
    pub white: Align64<[i16; L1_SIZE]>,
    pub black: Align64<[i16; L1_SIZE]>,

    pub mv: MovedPiece,
    pub update_buffer: UpdateBuffer,
    pub correct: [bool; 2],
}

impl Accumulator {
    /// Initializes the accumulator with the given bias.
    pub fn init(&mut self, bias: &Align64<[i16; L1_SIZE]>, update: PovUpdate) {
        if update.white {
            self.white = bias.clone();
        }
        if update.black {
            self.black = bias.clone();
        }
    }

    /// Select the buffer by colour.
    pub fn select_mut(&mut self, colour: Colour) -> &mut Align64<[i16; L1_SIZE]> {
        match colour {
            Colour::White => &mut self.white,
            Colour::Black => &mut self.black,
        }
    }
}

#[allow(clippy::needless_lifetimes)]
unsafe fn slice_to_aligned<'a>(slice: &'a [i16]) -> &'a Align64<[i16; L1_SIZE]> {
    // don't immediately cast to Align64, as we want to check the alignment first.
    let ptr = slice.as_ptr();
    debug_assert_eq!(ptr.align_offset(64), 0);
    // alignments are sensible, so we can safely cast.
    #[allow(clippy::cast_ptr_alignment)]
    &*ptr.cast()
}

#[cfg(target_feature = "avx2")]
mod avx2 {
use super::*;
use crate::nnue::simd::*;

/// Apply add/subtract updates in place.
pub fn vector_update_inplace(
    input: &mut Align64<[i16; L1_SIZE]>,
    bucket: &Align64<[i16; INPUT * L1_SIZE]>,
    adds: &[FeatureIndex],
    subs: &[FeatureIndex],
) {
    const REGISTERS: usize = 16;
    const UNROLL: usize = I16_CHUNK_SIZE * REGISTERS;
    // SAFETY: we never hold multiple mutable references, we never mutate immutable memory,
    // we use iterators to ensure that we're staying in-bounds, etc.
    unsafe {
        let mut registers = [vec_zero_epi16(); 16];
        for i in 0..L1_SIZE / UNROLL {
            let unroll_offset = i * UNROLL;
            for (r_idx, reg) in registers.iter_mut().enumerate() {
                *reg = vec_load_epi16(input.get_unchecked(unroll_offset + r_idx * I16_CHUNK_SIZE));
            }
            for &sub_index in subs {
                let sub_index = sub_index.index() * L1_SIZE;
                let sub_block = slice_to_aligned(bucket.get_unchecked(sub_index..sub_index + L1_SIZE));
                for (r_idx, reg) in registers.iter_mut().enumerate() {
                    let sub = vec_load_epi16(sub_block.get_unchecked(unroll_offset + r_idx * I16_CHUNK_SIZE));
                    *reg = vec_sub_epi16(*reg, sub);
                }
            }
            for &add_index in adds {
                let add_index = add_index.index() * L1_SIZE;
                let add_block = slice_to_aligned(bucket.get_unchecked(add_index..add_index + L1_SIZE));
                for (r_idx, reg) in registers.iter_mut().enumerate() {
                    let add = vec_load_epi16(add_block.get_unchecked(unroll_offset + r_idx * I16_CHUNK_SIZE));
                    *reg = vec_add_epi16(*reg, add);
                }
            }
            for (r_idx, reg) in registers.iter().enumerate() {
                vec_store_epi16(input.get_unchecked_mut(unroll_offset + r_idx * I16_CHUNK_SIZE), *reg);
            }
        }
    }
}

/// Move a feature from one square to another.
pub fn vector_add_sub(
    input: &Align64<[i16; L1_SIZE]>,
    output: &mut Align64<[i16; L1_SIZE]>,
    bucket: &Align64<[i16; INPUT * L1_SIZE]>,
    feature_idx_add: FeatureIndex,
    feature_idx_sub: FeatureIndex,
) {
    let offset_add = feature_idx_add.index() * L1_SIZE;
    let offset_sub = feature_idx_sub.index() * L1_SIZE;
    let s_block;
    let a_block;
    // SAFETY: offset_{add,sub} are multiples of LAYER_1_SIZE, and so are correctly-aligned.
    // additionally, as they originate from FeatureIndex, the L1-SIZE slices are all in bounds,
    // as FeatureIndex ranges in 0..768.
    unsafe {
        s_block = slice_to_aligned(bucket.get_unchecked(offset_sub..offset_sub + L1_SIZE));
        a_block = slice_to_aligned(bucket.get_unchecked(offset_add..offset_add + L1_SIZE));
    }
    for i in 0..L1_SIZE / I16_CHUNK_SIZE {
        // SAFETY: we never hold multiple mutable references, we never mutate immutable memory, etc.
        unsafe {
            let x = vec_load_epi16(input.get_unchecked(i * I16_CHUNK_SIZE));
            let w_sub = vec_load_epi16(s_block.get_unchecked(i * I16_CHUNK_SIZE));
            let w_add = vec_load_epi16(a_block.get_unchecked(i * I16_CHUNK_SIZE));
            let t = vec_sub_epi16(x, w_sub);
            let t = vec_add_epi16(t, w_add);
            vec_store_epi16(output.get_unchecked_mut(i * I16_CHUNK_SIZE), t);
        }
    }
}

/// Subtract two features and add one feature all at once.
pub fn vector_add_sub2(
    input: &Align64<[i16; L1_SIZE]>,
    output: &mut Align64<[i16; L1_SIZE]>,
    bucket: &Align64<[i16; INPUT * L1_SIZE]>,
    feature_idx_add: FeatureIndex,
    feature_idx_sub1: FeatureIndex,
    feature_idx_sub2: FeatureIndex,
) {
    let offset_add = feature_idx_add.index() * L1_SIZE;
    let offset_sub1 = feature_idx_sub1.index() * L1_SIZE;
    let offset_sub2 = feature_idx_sub2.index() * L1_SIZE;
    let a_block;
    let s_block1;
    let s_block2;
    // SAFETY: offset_{add,sub}{1,2} are all multiples of LAYER_1_SIZE, and so are correctly-aligned.
    // additionally, as they originate from FeatureIndex, the L1-SIZE slices are all in bounds, as
    // FeatureIndex ranges in 0..768.
    unsafe {
        a_block = slice_to_aligned(bucket.get_unchecked(offset_add..offset_add + L1_SIZE));
        s_block1 = slice_to_aligned(bucket.get_unchecked(offset_sub1..offset_sub1 + L1_SIZE));
        s_block2 = slice_to_aligned(bucket.get_unchecked(offset_sub2..offset_sub2 + L1_SIZE));
    }
    for i in 0..L1_SIZE / I16_CHUNK_SIZE {
        // SAFETY: we never hold multiple mutable references, we never mutate immutable memory, etc.
        unsafe {
            let x = vec_load_epi16(input.get_unchecked(i * I16_CHUNK_SIZE));
            let w_sub1 = vec_load_epi16(s_block1.get_unchecked(i * I16_CHUNK_SIZE));
            let w_sub2 = vec_load_epi16(s_block2.get_unchecked(i * I16_CHUNK_SIZE));
            let w_add = vec_load_epi16(a_block.get_unchecked(i * I16_CHUNK_SIZE));
            let t = vec_sub_epi16(x, w_sub1);
            let t = vec_sub_epi16(t, w_sub2);
            let t = vec_add_epi16(t, w_add);
            vec_store_epi16(output.get_unchecked_mut(i * I16_CHUNK_SIZE), t);
        }
    }
}

/// Add two features and subtract two features all at once.
pub fn vector_add2_sub2(
    input: &Align64<[i16; L1_SIZE]>,
    output: &mut Align64<[i16; L1_SIZE]>,
    bucket: &Align64<[i16; INPUT * L1_SIZE]>,
    feature_idx_add1: FeatureIndex,
    feature_idx_add2: FeatureIndex,
    feature_idx_sub1: FeatureIndex,
    feature_idx_sub2: FeatureIndex,
) {
    let offset_add1 = feature_idx_add1.index() * L1_SIZE;
    let offset_add2 = feature_idx_add2.index() * L1_SIZE;
    let offset_sub1 = feature_idx_sub1.index() * L1_SIZE;
    let offset_sub2 = feature_idx_sub2.index() * L1_SIZE;
    let a_block1;
    let a_block2;
    let s_block1;
    let s_block2;
    // SAFETY: offset_{add,sub}{1,2} are all multiples of LAYER_1_SIZE, and so are correctly-aligned.
    // additionally, as they originate from FeatureIndex, the L1-SIZE slices are all in bounds, as
    // FeatureIndex ranges in 0..768.
    unsafe {
        a_block1 = slice_to_aligned(bucket.get_unchecked(offset_add1..offset_add1 + L1_SIZE));
        a_block2 = slice_to_aligned(bucket.get_unchecked(offset_add2..offset_add2 + L1_SIZE));
        s_block1 = slice_to_aligned(bucket.get_unchecked(offset_sub1..offset_sub1 + L1_SIZE));
        s_block2 = slice_to_aligned(bucket.get_unchecked(offset_sub2..offset_sub2 + L1_SIZE));
    }
    for i in 0..L1_SIZE / I16_CHUNK_SIZE {
        // SAFETY: we never hold multiple mutable references, we never mutate immutable memory, etc.
        unsafe {
            let x = vec_load_epi16(input.get_unchecked(i * I16_CHUNK_SIZE));
            let w_sub1 = vec_load_epi16(s_block1.get_unchecked(i * I16_CHUNK_SIZE));
            let w_sub2 = vec_load_epi16(s_block2.get_unchecked(i * I16_CHUNK_SIZE));
            let w_add1 = vec_load_epi16(a_block1.get_unchecked(i * I16_CHUNK_SIZE));
            let w_add2 = vec_load_epi16(a_block2.get_unchecked(i * I16_CHUNK_SIZE));
            let t = vec_sub_epi16(x, w_sub1);
            let t = vec_sub_epi16(t, w_sub2);
            let t = vec_add_epi16(t, w_add1);
            let t = vec_add_epi16(t, w_add2);
            vec_store_epi16(output.get_unchecked_mut(i * I16_CHUNK_SIZE), t);
        }
    }
}
}

#[cfg(not(target_feature = "avx2"))]
mod generic {
use super::*;

/// Apply add/subtract updates in place.
pub fn vector_update_inplace(
    input: &mut Align64<[i16; L1_SIZE]>,
    bucket: &Align64<[i16; INPUT * L1_SIZE]>,
    adds: &[FeatureIndex],
    subs: &[FeatureIndex],
) {
    const REGISTERS: usize = 16;
    const UNROLL: usize = REGISTERS;
    // SAFETY: we never hold multiple mutable references, we never mutate immutable memory,
    // we use iterators to ensure that we're staying in-bounds, etc.
    unsafe {
        let mut registers = [0; 16];
        for i in 0..L1_SIZE / UNROLL {
            let unroll_offset = i * UNROLL;
            for (r_idx, reg) in registers.iter_mut().enumerate() {
                *reg = *input.get_unchecked(unroll_offset + r_idx);
            }
            for &sub_index in subs {
                let sub_index = sub_index.index() * L1_SIZE;
                let sub_block = slice_to_aligned(bucket.get_unchecked(sub_index..sub_index + L1_SIZE));
                for (r_idx, reg) in registers.iter_mut().enumerate() {
                    let sub = *sub_block.get_unchecked(unroll_offset + r_idx);
                    *reg = *reg - sub;
                }
            }
            for &add_index in adds {
                let add_index = add_index.index() * L1_SIZE;
                let add_block = slice_to_aligned(bucket.get_unchecked(add_index..add_index + L1_SIZE));
                for (r_idx, reg) in registers.iter_mut().enumerate() {
                    let add = *add_block.get_unchecked(unroll_offset + r_idx);
                    *reg = *reg + add;
                }
            }
            for (r_idx, reg) in registers.iter().enumerate() {
                *input.get_unchecked_mut(unroll_offset + r_idx) = *reg;
            }
        }
    }
}

/// Move a feature from one square to another.
pub fn vector_add_sub(
    input: &Align64<[i16; L1_SIZE]>,
    output: &mut Align64<[i16; L1_SIZE]>,
    bucket: &Align64<[i16; INPUT * L1_SIZE]>,
    feature_idx_add: FeatureIndex,
    feature_idx_sub: FeatureIndex,
) {
    let offset_add = feature_idx_add.index() * L1_SIZE;
    let offset_sub = feature_idx_sub.index() * L1_SIZE;
    let s_block;
    let a_block;
    // SAFETY: offset_{add,sub} are multiples of LAYER_1_SIZE, and so are correctly-aligned.
    // additionally, as they originate from FeatureIndex, the L1-SIZE slices are all in bounds,
    // as FeatureIndex ranges in 0..768.
    unsafe {
        s_block = slice_to_aligned(bucket.get_unchecked(offset_sub..offset_sub + L1_SIZE));
        a_block = slice_to_aligned(bucket.get_unchecked(offset_add..offset_add + L1_SIZE));
    }
    for i in 0..L1_SIZE {
        // SAFETY: we never hold multiple mutable references, we never mutate immutable memory, etc.
        unsafe {
            let x = *input.get_unchecked(i);
            let w_sub = *s_block.get_unchecked(i);
            let w_add = *a_block.get_unchecked(i);
            let t = x - w_sub;
            let t = t + w_add;
            *output.get_unchecked_mut(i) = t;
        }
    }
}

/// Subtract two features and add one feature all at once.
pub fn vector_add_sub2(
    input: &Align64<[i16; L1_SIZE]>,
    output: &mut Align64<[i16; L1_SIZE]>,
    bucket: &Align64<[i16; INPUT * L1_SIZE]>,
    feature_idx_add: FeatureIndex,
    feature_idx_sub1: FeatureIndex,
    feature_idx_sub2: FeatureIndex,
) {
    let offset_add = feature_idx_add.index() * L1_SIZE;
    let offset_sub1 = feature_idx_sub1.index() * L1_SIZE;
    let offset_sub2 = feature_idx_sub2.index() * L1_SIZE;
    let a_block;
    let s_block1;
    let s_block2;
    // SAFETY: offset_{add,sub}{1,2} are all multiples of LAYER_1_SIZE, and so are correctly-aligned.
    // additionally, as they originate from FeatureIndex, the L1-SIZE slices are all in bounds, as
    // FeatureIndex ranges in 0..768.
    unsafe {
        a_block = slice_to_aligned(bucket.get_unchecked(offset_add..offset_add + L1_SIZE));
        s_block1 = slice_to_aligned(bucket.get_unchecked(offset_sub1..offset_sub1 + L1_SIZE));
        s_block2 = slice_to_aligned(bucket.get_unchecked(offset_sub2..offset_sub2 + L1_SIZE));
    }
    for i in 0..L1_SIZE {
        // SAFETY: we never hold multiple mutable references, we never mutate immutable memory, etc.
        unsafe {
            let x = *input.get_unchecked(i);
            let w_sub1 = *s_block1.get_unchecked(i);
            let w_sub2 = *s_block2.get_unchecked(i);
            let w_add = *a_block.get_unchecked(i);
            let t = x - w_sub1;
            let t = t - w_sub2;
            let t = t + w_add;
            *output.get_unchecked_mut(i) = t;
        }
    }
}

/// Add two features and subtract two features all at once.
pub fn vector_add2_sub2(
    input: &Align64<[i16; L1_SIZE]>,
    output: &mut Align64<[i16; L1_SIZE]>,
    bucket: &Align64<[i16; INPUT * L1_SIZE]>,
    feature_idx_add1: FeatureIndex,
    feature_idx_add2: FeatureIndex,
    feature_idx_sub1: FeatureIndex,
    feature_idx_sub2: FeatureIndex,
) {
    let offset_add1 = feature_idx_add1.index() * L1_SIZE;
    let offset_add2 = feature_idx_add2.index() * L1_SIZE;
    let offset_sub1 = feature_idx_sub1.index() * L1_SIZE;
    let offset_sub2 = feature_idx_sub2.index() * L1_SIZE;
    let a_block1;
    let a_block2;
    let s_block1;
    let s_block2;
    // SAFETY: offset_{add,sub}{1,2} are all multiples of LAYER_1_SIZE, and so are correctly-aligned.
    // additionally, as they originate from FeatureIndex, the L1-SIZE slices are all in bounds, as
    // FeatureIndex ranges in 0..768.
    unsafe {
        a_block1 = slice_to_aligned(bucket.get_unchecked(offset_add1..offset_add1 + L1_SIZE));
        a_block2 = slice_to_aligned(bucket.get_unchecked(offset_add2..offset_add2 + L1_SIZE));
        s_block1 = slice_to_aligned(bucket.get_unchecked(offset_sub1..offset_sub1 + L1_SIZE));
        s_block2 = slice_to_aligned(bucket.get_unchecked(offset_sub2..offset_sub2 + L1_SIZE));
    }
    for i in 0..L1_SIZE {
        // SAFETY: we never hold multiple mutable references, we never mutate immutable memory, etc.
        unsafe {
            let x = *input.get_unchecked(i);
            let w_sub1 = *s_block1.get_unchecked(i);
            let w_sub2 = *s_block2.get_unchecked(i);
            let w_add1 = *a_block1.get_unchecked(i);
            let w_add2 = *a_block2.get_unchecked(i);
            let t = x - w_sub1;
            let t = t - w_sub2;
            let t = t + w_add1;
            let t = t + w_add2;
            *output.get_unchecked_mut(i) = t;
        }
    }
}
}

#[cfg(not(target_feature = "avx2"))]
pub use generic::*;
#[cfg(target_feature = "avx2")]
pub use avx2::*;