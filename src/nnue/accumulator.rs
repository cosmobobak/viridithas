use crate::{
    chess::piece::Colour,
    nnue::network::{feature::FeatureIndex, MovedPiece, UpdateBuffer, INPUT, L1_SIZE},
    util::Align64,
};

/// Activations of the hidden layer.
pub struct Accumulator {
    pub white: Align64<[i16; L1_SIZE]>,
    pub black: Align64<[i16; L1_SIZE]>,

    pub mv: MovedPiece,
    pub update_buffer: UpdateBuffer,
    pub correct: [bool; 2],
}

impl Accumulator {
    /// Select the buffer by colour.
    pub const fn select(&self, colour: Colour) -> &Align64<[i16; L1_SIZE]> {
        match colour {
            Colour::White => &self.white,
            Colour::Black => &self.black,
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

#[allow(clippy::inline_always)]
#[inline(always)]
unsafe fn slice_to_aligned(slice: &[i16]) -> &Align64<[i16; L1_SIZE]> {
    debug_assert_eq!(slice.len(), L1_SIZE);
    // don't immediately cast to Align64, as we want to check the alignment first.
    let ptr = slice.as_ptr();
    debug_assert_eq!(ptr.align_offset(64), 0);
    // alignments are sensible, so we can safely cast.
    #[allow(clippy::cast_ptr_alignment)]
    &*ptr.cast()
}

#[cfg(target_arch = "x86_64")]
mod x86simd {
    use arrayvec::ArrayVec;

    use super::{slice_to_aligned, Align64, FeatureIndex, INPUT, L1_SIZE};
    use crate::nnue::simd::{self, I16_CHUNK_SIZE};

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
            let mut add_blocks = ArrayVec::<_, 32>::new();
            let mut sub_blocks = ArrayVec::<_, 32>::new();
            for &add_index in adds {
                let add_index = add_index.index() * L1_SIZE;
                add_blocks.push(slice_to_aligned(
                    bucket.get_unchecked(add_index..add_index + L1_SIZE),
                ));
            }
            for &sub_index in subs {
                let sub_index = sub_index.index() * L1_SIZE;
                sub_blocks.push(slice_to_aligned(
                    bucket.get_unchecked(sub_index..sub_index + L1_SIZE),
                ));
            }
            let mut registers = [simd::zero_i16(); REGISTERS];
            for i in 0..L1_SIZE / UNROLL {
                let unroll_offset = i * UNROLL;
                for (r_idx, reg) in registers.iter_mut().enumerate() {
                    let src = input.as_ptr().add(unroll_offset + r_idx * I16_CHUNK_SIZE);
                    *reg = simd::load_i16(src);
                }
                for &sub_block in &sub_blocks {
                    for (r_idx, reg) in registers.iter_mut().enumerate() {
                        let src = sub_block
                            .as_ptr()
                            .add(unroll_offset + r_idx * I16_CHUNK_SIZE);
                        *reg = simd::sub_i16(*reg, simd::load_i16(src));
                    }
                }
                for &add_block in &add_blocks {
                    for (r_idx, reg) in registers.iter_mut().enumerate() {
                        let src = add_block
                            .as_ptr()
                            .add(unroll_offset + r_idx * I16_CHUNK_SIZE);
                        *reg = simd::add_i16(*reg, simd::load_i16(src));
                    }
                }
                for (r_idx, reg) in registers.iter().enumerate() {
                    let dst = input
                        .as_mut_ptr()
                        .add(unroll_offset + r_idx * I16_CHUNK_SIZE);
                    simd::store_i16(dst, *reg);
                }
            }
        }
    }

    /// Move a feature from one square to another.
    pub fn vector_add_sub(
        input: &Align64<[i16; L1_SIZE]>,
        output: &mut Align64<[i16; L1_SIZE]>,
        bucket: &Align64<[i16; INPUT * L1_SIZE]>,
        add: FeatureIndex,
        sub: FeatureIndex,
    ) {
        let offset_add = add.index() * L1_SIZE;
        let offset_sub = sub.index() * L1_SIZE;
        let s_block;
        let a_block;
        // SAFETY: offset_{add,sub} are multiples of LAYER_1_SIZE, and so are correctly-aligned.
        // additionally, as they originate from FeatureIndex, the L1-SIZE slices are all in bounds,
        // as FeatureIndex ranges in 0..704.
        unsafe {
            s_block = slice_to_aligned(bucket.get_unchecked(offset_sub..offset_sub + L1_SIZE));
            a_block = slice_to_aligned(bucket.get_unchecked(offset_add..offset_add + L1_SIZE));
        }
        for i in 0..L1_SIZE / I16_CHUNK_SIZE {
            // SAFETY: we never hold multiple mutable references, we never mutate immutable memory, etc.
            unsafe {
                let x = simd::load_i16(input.as_ptr().add(i * I16_CHUNK_SIZE));
                let w_sub = simd::load_i16(s_block.as_ptr().add(i * I16_CHUNK_SIZE));
                let w_add = simd::load_i16(a_block.as_ptr().add(i * I16_CHUNK_SIZE));
                let t = simd::sub_i16(x, w_sub);
                let t = simd::add_i16(t, w_add);
                simd::store_i16(output.as_mut_ptr().add(i * I16_CHUNK_SIZE), t);
            }
        }
    }

    /// Subtract two features and add one feature all at once.
    pub fn vector_add_sub2(
        input: &Align64<[i16; L1_SIZE]>,
        output: &mut Align64<[i16; L1_SIZE]>,
        bucket: &Align64<[i16; INPUT * L1_SIZE]>,
        add: FeatureIndex,
        sub1: FeatureIndex,
        sub2: FeatureIndex,
    ) {
        let offset_add = add.index() * L1_SIZE;
        let offset_sub1 = sub1.index() * L1_SIZE;
        let offset_sub2 = sub2.index() * L1_SIZE;
        let a_block;
        let s_block1;
        let s_block2;
        // SAFETY: offset_{add,sub}{1,2} are all multiples of LAYER_1_SIZE, and so are correctly-aligned.
        // additionally, as they originate from FeatureIndex, the L1-SIZE slices are all in bounds, as
        // FeatureIndex ranges in 0..704.
        unsafe {
            a_block = slice_to_aligned(bucket.get_unchecked(offset_add..offset_add + L1_SIZE));
            s_block1 = slice_to_aligned(bucket.get_unchecked(offset_sub1..offset_sub1 + L1_SIZE));
            s_block2 = slice_to_aligned(bucket.get_unchecked(offset_sub2..offset_sub2 + L1_SIZE));
        }
        for i in 0..L1_SIZE / I16_CHUNK_SIZE {
            // SAFETY: we never hold multiple mutable references, we never mutate immutable memory, etc.
            unsafe {
                let x = simd::load_i16(input.as_ptr().add(i * I16_CHUNK_SIZE));
                let w_sub1 = simd::load_i16(s_block1.as_ptr().add(i * I16_CHUNK_SIZE));
                let w_sub2 = simd::load_i16(s_block2.as_ptr().add(i * I16_CHUNK_SIZE));
                let w_add = simd::load_i16(a_block.as_ptr().add(i * I16_CHUNK_SIZE));
                let t = simd::sub_i16(x, w_sub1);
                let t = simd::sub_i16(t, w_sub2);
                let t = simd::add_i16(t, w_add);
                simd::store_i16(output.as_mut_ptr().add(i * I16_CHUNK_SIZE), t);
            }
        }
    }

    /// Add two features and subtract two features all at once.
    pub fn vector_add2_sub2(
        input: &Align64<[i16; L1_SIZE]>,
        output: &mut Align64<[i16; L1_SIZE]>,
        bucket: &Align64<[i16; INPUT * L1_SIZE]>,
        add1: FeatureIndex,
        add2: FeatureIndex,
        sub1: FeatureIndex,
        sub2: FeatureIndex,
    ) {
        let offset_add1 = add1.index() * L1_SIZE;
        let offset_add2 = add2.index() * L1_SIZE;
        let offset_sub1 = sub1.index() * L1_SIZE;
        let offset_sub2 = sub2.index() * L1_SIZE;
        let a_block1;
        let a_block2;
        let s_block1;
        let s_block2;
        // SAFETY: offset_{add,sub}{1,2} are all multiples of LAYER_1_SIZE, and so are correctly-aligned.
        // additionally, as they originate from FeatureIndex, the L1-SIZE slices are all in bounds, as
        // FeatureIndex ranges in 0..704.
        unsafe {
            a_block1 = slice_to_aligned(bucket.get_unchecked(offset_add1..offset_add1 + L1_SIZE));
            a_block2 = slice_to_aligned(bucket.get_unchecked(offset_add2..offset_add2 + L1_SIZE));
            s_block1 = slice_to_aligned(bucket.get_unchecked(offset_sub1..offset_sub1 + L1_SIZE));
            s_block2 = slice_to_aligned(bucket.get_unchecked(offset_sub2..offset_sub2 + L1_SIZE));
        }
        for i in 0..L1_SIZE / I16_CHUNK_SIZE {
            // SAFETY: we never hold multiple mutable references, we never mutate immutable memory, etc.
            unsafe {
                let x = simd::load_i16(input.as_ptr().add(i * I16_CHUNK_SIZE));
                let w_sub1 = simd::load_i16(s_block1.as_ptr().add(i * I16_CHUNK_SIZE));
                let w_sub2 = simd::load_i16(s_block2.as_ptr().add(i * I16_CHUNK_SIZE));
                let w_add1 = simd::load_i16(a_block1.as_ptr().add(i * I16_CHUNK_SIZE));
                let w_add2 = simd::load_i16(a_block2.as_ptr().add(i * I16_CHUNK_SIZE));
                let t = simd::sub_i16(x, w_sub1);
                let t = simd::sub_i16(t, w_sub2);
                let t = simd::add_i16(t, w_add1);
                let t = simd::add_i16(t, w_add2);
                simd::store_i16(output.as_mut_ptr().add(i * I16_CHUNK_SIZE), t);
            }
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
mod generic {
    use arrayvec::ArrayVec;

    use super::{slice_to_aligned, Align64, FeatureIndex, INPUT, L1_SIZE};

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
            let mut add_blocks = ArrayVec::<_, 32>::new();
            let mut sub_blocks = ArrayVec::<_, 32>::new();
            for &add_index in adds {
                let add_index = add_index.index() * L1_SIZE;
                add_blocks.push(slice_to_aligned(
                    bucket.get_unchecked(add_index..add_index + L1_SIZE),
                ));
            }
            for &sub_index in subs {
                let sub_index = sub_index.index() * L1_SIZE;
                sub_blocks.push(slice_to_aligned(
                    bucket.get_unchecked(sub_index..sub_index + L1_SIZE),
                ));
            }
            let mut registers = [0; REGISTERS];
            for i in 0..L1_SIZE / UNROLL {
                let unroll_offset = i * UNROLL;
                for (r_idx, reg) in registers.iter_mut().enumerate() {
                    *reg = *input.get_unchecked(unroll_offset + r_idx);
                }
                for &add_block in &add_blocks {
                    for (r_idx, reg) in registers.iter_mut().enumerate() {
                        let add = *add_block.get_unchecked(unroll_offset + r_idx);
                        *reg += add;
                    }
                }
                for &sub_block in &sub_blocks {
                    for (r_idx, reg) in registers.iter_mut().enumerate() {
                        let sub = *sub_block.get_unchecked(unroll_offset + r_idx);
                        *reg -= sub;
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
        add: FeatureIndex,
        sub: FeatureIndex,
    ) {
        let offset_add = add.index() * L1_SIZE;
        let offset_sub = sub.index() * L1_SIZE;
        let s_block;
        let a_block;
        // SAFETY: offset_{add,sub} are multiples of LAYER_1_SIZE, and so are correctly-aligned.
        // additionally, as they originate from FeatureIndex, the L1-SIZE slices are all in bounds,
        // as FeatureIndex ranges in 0..704.
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
        add: FeatureIndex,
        sub1: FeatureIndex,
        sub2: FeatureIndex,
    ) {
        let offset_add = add.index() * L1_SIZE;
        let offset_sub1 = sub1.index() * L1_SIZE;
        let offset_sub2 = sub2.index() * L1_SIZE;
        let a_block;
        let s_block1;
        let s_block2;
        // SAFETY: offset_{add,sub}{1,2} are all multiples of LAYER_1_SIZE, and so are correctly-aligned.
        // additionally, as they originate from FeatureIndex, the L1-SIZE slices are all in bounds, as
        // FeatureIndex ranges in 0..704.
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
        add1: FeatureIndex,
        add2: FeatureIndex,
        sub1: FeatureIndex,
        sub2: FeatureIndex,
    ) {
        let offset_add1 = add1.index() * L1_SIZE;
        let offset_add2 = add2.index() * L1_SIZE;
        let offset_sub1 = sub1.index() * L1_SIZE;
        let offset_sub2 = sub2.index() * L1_SIZE;
        let a_block1;
        let a_block2;
        let s_block1;
        let s_block2;
        // SAFETY: offset_{add,sub}{1,2} are all multiples of LAYER_1_SIZE, and so are correctly-aligned.
        // additionally, as they originate from FeatureIndex, the L1-SIZE slices are all in bounds, as
        // FeatureIndex ranges in 0..704.
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

#[cfg(target_arch = "x86_64")]
pub use x86simd::*;

#[cfg(not(target_arch = "x86_64"))]
pub use generic::*;
