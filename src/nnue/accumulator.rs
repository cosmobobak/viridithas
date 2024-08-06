// use crate::{board::Board, piece::PieceType};

use crate::{nnue::simd::VectorI16, piece::Colour};

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

/// Apply add/subtract updates in place.
pub fn vector_update_inplace(
    input: &mut Align64<[i16; L1_SIZE]>,
    bucket: &Align64<[i16; INPUT * L1_SIZE]>,
    adds: &[FeatureIndex],
    subs: &[FeatureIndex],
) {
    const REGISTERS: usize = 16;
    const UNROLL: usize = VectorI16::COUNT * REGISTERS;
    let mut registers = [VectorI16::zero(); 16];
    for i in 0..L1_SIZE / UNROLL {
        let unroll_offset = i * UNROLL;
        // SAFETY: we never hold multiple mutable references, we never mutate immutable memory,
        // we use iterators to ensure that we're staying in-bounds, etc.
        unsafe {
            for (r_idx, reg) in registers.iter_mut().enumerate() {
                *reg = VectorI16::load_at(input, unroll_offset + r_idx * VectorI16::COUNT);
            }
            for &sub_index in subs {
                let sub_index = sub_index.index() * L1_SIZE;
                let sub_block = slice_to_aligned(bucket.get_unchecked(sub_index..sub_index + L1_SIZE));
                for (r_idx, reg) in registers.iter_mut().enumerate() {
                    let sub = VectorI16::load_at(sub_block, unroll_offset + r_idx * VectorI16::COUNT);
                    *reg = VectorI16::sub(*reg, sub);
                }
            }
            for &add_index in adds {
                let add_index = add_index.index() * L1_SIZE;
                let add_block = slice_to_aligned(bucket.get_unchecked(add_index..add_index + L1_SIZE));
                for (r_idx, reg) in registers.iter_mut().enumerate() {
                    let add = VectorI16::load_at(add_block, unroll_offset + r_idx * VectorI16::COUNT);
                    *reg = VectorI16::add(*reg, add);
                }
            }
            for (r_idx, reg) in registers.iter().enumerate() {
                VectorI16::store_at(input, *reg, unroll_offset + r_idx * VectorI16::COUNT);
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
    for i in 0..L1_SIZE / VectorI16::COUNT {
        // SAFETY: we never hold multiple mutable references, we never mutate immutable memory,
        // we use iterators to ensure that we're staying in-bounds, etc.
        unsafe {
            let x = VectorI16::load_at(input, i * VectorI16::COUNT);
            let w_sub = VectorI16::load_at(s_block, i * VectorI16::COUNT);
            let w_add = VectorI16::load_at(a_block, i * VectorI16::COUNT);
            let t = VectorI16::sub(x, w_sub);
            let t = VectorI16::add(t, w_add);
            VectorI16::store_at(output, t, i * VectorI16::COUNT);
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
    for i in 0..L1_SIZE / VectorI16::COUNT {
        // SAFETY: we never hold multiple mutable references, we never mutate immutable memory,
        // we use iterators to ensure that we're staying in-bounds, etc.
        unsafe {
            let x = VectorI16::load_at(input, i * VectorI16::COUNT);
            let w_sub1 = VectorI16::load_at(s_block1, i * VectorI16::COUNT);
            let w_sub2 = VectorI16::load_at(s_block2, i * VectorI16::COUNT);
            let w_add = VectorI16::load_at(a_block, i * VectorI16::COUNT);
            let t = VectorI16::sub(x, w_sub1);
            let t = VectorI16::sub(t, w_sub2);
            let t = VectorI16::add(t, w_add);
            VectorI16::store_at(output, t, i * VectorI16::COUNT);
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
    for i in 0..L1_SIZE / VectorI16::COUNT {
        // SAFETY: we never hold multiple mutable references, we never mutate immutable memory,
        // we use iterators to ensure that we're staying in-bounds, etc.
        unsafe {
            let x = VectorI16::load_at(input, i * VectorI16::COUNT);
            let w_sub1 = VectorI16::load_at(s_block1, i * VectorI16::COUNT);
            let w_sub2 = VectorI16::load_at(s_block2, i * VectorI16::COUNT);
            let w_add1 = VectorI16::load_at(a_block1, i * VectorI16::COUNT);
            let w_add2 = VectorI16::load_at(a_block2, i * VectorI16::COUNT);
            let t = VectorI16::sub(x, w_sub1);
            let t = VectorI16::sub(t, w_sub2);
            let t = VectorI16::add(t, w_add1);
            let t = VectorI16::add(t, w_add2);
            VectorI16::store_at(output, t, i * VectorI16::COUNT);
        }
    }
}
