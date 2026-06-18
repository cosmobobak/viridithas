use crate::{
    nnue::network::{L1_SIZE, PSQT_FEATURES, feature::PsqtFeatureIndex},
    util::Align,
};

/// Pre-activations of l0’s output.
pub struct Accumulator {
    pub halves: [Align<[i16; L1_SIZE]>; 2],
}

#[allow(clippy::inline_always)]
#[inline(always)]
unsafe fn slice_to_aligned<T>(slice: &[T]) -> &Align<[T; L1_SIZE]> {
    debug_assert_eq!(slice.len(), L1_SIZE);
    // don't immediately cast to Align64, as we want to check the alignment first.
    let ptr = slice.as_ptr();
    debug_assert_eq!(ptr.align_offset(64), 0);
    // Safety: alignments are sensible, so we can safely cast.
    #[allow(clippy::cast_ptr_alignment)]
    unsafe {
        &*ptr.cast()
    }
}

mod simd {
    use arrayvec::ArrayVec;

    use super::{Align, L1_SIZE, PSQT_FEATURES, PsqtFeatureIndex, slice_to_aligned};
    use crate::{
        chess::{
            board::{Board, movegen::attacks_by_type},
            piece::{Colour, PieceType},
            types::Square,
        },
        nnue::{
            network::{THREAT_FEATURES, ThreatUpdateBuffer, feature::threat_index},
            simd::{self, I16_CHUNK},
        },
    };

    /// Apply add/subtract PSQT updates in place.
    pub fn vector_update_inplace_psqt(
        input: &mut Align<[i16; L1_SIZE]>,
        bucket: &Align<[i16; PSQT_FEATURES * L1_SIZE]>,
        adds: &[PsqtFeatureIndex],
        subs: &[PsqtFeatureIndex],
    ) {
        const REGISTERS: usize = 16;
        const UNROLL: usize = I16_CHUNK * REGISTERS;
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
                    let src = input.as_ptr().add(unroll_offset + r_idx * I16_CHUNK);
                    *reg = simd::load_i16(src);
                }
                for &sub_block in &sub_blocks {
                    for (r_idx, reg) in registers.iter_mut().enumerate() {
                        let src = sub_block.as_ptr().add(unroll_offset + r_idx * I16_CHUNK);
                        *reg = simd::sub_i16(*reg, simd::load_i16(src));
                    }
                }
                for &add_block in &add_blocks {
                    for (r_idx, reg) in registers.iter_mut().enumerate() {
                        let src = add_block.as_ptr().add(unroll_offset + r_idx * I16_CHUNK);
                        *reg = simd::add_i16(*reg, simd::load_i16(src));
                    }
                }
                for (r_idx, reg) in registers.iter().enumerate() {
                    let dst = input.as_mut_ptr().add(unroll_offset + r_idx * I16_CHUNK);
                    simd::store_i16(dst, *reg);
                }
            }
        }
    }

    /// Apply add/subtract updates in place.
    pub fn vector_update_threats(
        src_acc: &Align<[i16; L1_SIZE]>,
        dst_acc: &mut Align<[i16; L1_SIZE]>,
        weights: &Align<[i8; THREAT_FEATURES * L1_SIZE]>,
        updates: &ThreatUpdateBuffer,
        king: Square,
        colour: Colour,
    ) {
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};

        const REGISTERS: usize = 16;
        const UNROLL: usize = I16_CHUNK * REGISTERS;

        if updates.add.is_empty() && updates.sub.is_empty() {
            dst_acc.copy_from_slice(&**src_acc);
            return;
        }

        // SAFETY: we never hold multiple mutable references, we never mutate immutable memory,
        // we use iterators to ensure that we're staying in-bounds, etc.
        unsafe {
            #![allow(clippy::cast_possible_truncation)]
            let mut add_blocks = ArrayVec::<u32, 128>::new();
            let mut sub_blocks = ArrayVec::<u32, 128>::new();
            for &idx in &updates.add {
                let (good, idx) = idx.index(colour, king);
                let len = add_blocks.len();
                debug_assert!(len < add_blocks.capacity(), "OOB write");
                let loc = add_blocks.as_mut_ptr().add(len);
                *loc = idx * const { L1_SIZE as u32 };
                add_blocks.set_len(len + usize::from(good));
            }
            for &idx in &updates.sub {
                let (good, idx) = idx.index(colour, king);
                let len = sub_blocks.len();
                debug_assert!(len < sub_blocks.capacity(), "OOB write");
                let loc = sub_blocks.as_mut_ptr().add(len);
                *loc = idx * const { L1_SIZE as u32 };
                sub_blocks.set_len(len + usize::from(good));
            }
            for &offset in &add_blocks {
                #[cfg(target_arch = "x86_64")]
                _mm_prefetch(
                    (*weights).as_ptr().add(offset as usize).cast::<i8>(),
                    _MM_HINT_T0,
                );
            }
            for &offset in &sub_blocks {
                #[cfg(target_arch = "x86_64")]
                _mm_prefetch(
                    (*weights).as_ptr().add(offset as usize).cast::<i8>(),
                    _MM_HINT_T0,
                );
            }
            let mut registers = [simd::zero_i16(); REGISTERS];
            for i in 0..L1_SIZE / UNROLL {
                let unroll_offset = i * UNROLL;
                for (r_idx, reg) in registers.iter_mut().enumerate() {
                    let src = src_acc.as_ptr().add(unroll_offset + r_idx * I16_CHUNK);
                    *reg = simd::load_i16(src);
                }
                // todo: is load_extend_i8 the fastest way to do this?
                // check if the compiler is smart enough to load in a sensible way.
                for &sub_block in &sub_blocks {
                    for (r_idx, reg) in registers.iter_mut().enumerate() {
                        let src = (*weights)
                            .as_ptr()
                            .add(sub_block as usize + unroll_offset + r_idx * I16_CHUNK);
                        *reg = simd::sub_i16(*reg, simd::load_extend_i8(src));
                    }
                }
                for &add_block in &add_blocks {
                    for (r_idx, reg) in registers.iter_mut().enumerate() {
                        let src = (*weights)
                            .as_ptr()
                            .add(add_block as usize + unroll_offset + r_idx * I16_CHUNK);
                        *reg = simd::add_i16(*reg, simd::load_extend_i8(src));
                    }
                }
                for (r_idx, reg) in registers.iter().enumerate() {
                    let dst = dst_acc.as_mut_ptr().add(unroll_offset + r_idx * I16_CHUNK);
                    simd::store_i16(dst, *reg);
                }
            }
        }
    }

    pub fn refresh_threats(
        weights: &Align<[i8; THREAT_FEATURES * L1_SIZE]>,
        acc: &mut Align<[i16; L1_SIZE]>,
        board: &Board,
        colour: Colour,
    ) {
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};

        const REGISTERS: usize = 16;
        const UNROLL: usize = I16_CHUNK * REGISTERS;

        let bbs = &board.state.bbs;
        let occ = bbs.occupied();
        let king = board.state.bbs.king_sq(colour);
        let bb = occ & !bbs.pieces[PieceType::King];

        let mut indexes = ArrayVec::<u32, 128>::new();

        // Safety: We know this routine never produces more than 128 features.
        unsafe {
            #![allow(clippy::cast_possible_truncation)]
            for from in bb {
                let attacker = board.state.mailbox[from].unwrap();
                let threats =
                    occ & attacks_by_type(attacker, from, occ) & !bbs.pieces[PieceType::King];
                for to in threats {
                    let victim = board.state.mailbox[to].unwrap();
                    let (good, feature) = threat_index(colour, king, attacker, victim, from, to);
                    let len = indexes.len();
                    debug_assert!(len < indexes.capacity(), "OOB write");
                    let loc = indexes.as_mut_ptr().add(len);
                    *loc = feature * const { L1_SIZE as u32 };
                    indexes.set_len(len + usize::from(good));
                }
            }

            for &offset in &indexes {
                #[cfg(target_arch = "x86_64")]
                _mm_prefetch(
                    (*weights).as_ptr().add(offset as usize).cast::<i8>(),
                    _MM_HINT_T0,
                );
            }

            let mut registers = [simd::zero_i16(); REGISTERS];
            for i in 0..L1_SIZE / UNROLL {
                let unroll_offset = i * UNROLL;
                for reg in &mut registers {
                    *reg = simd::zero_i16();
                }
                for &block in &indexes {
                    for (r_idx, reg) in registers.iter_mut().enumerate() {
                        let src = (*weights)
                            .as_ptr()
                            .add(block as usize + unroll_offset + r_idx * I16_CHUNK);
                        *reg = simd::add_i16(*reg, simd::load_extend_i8(src));
                    }
                }
                for (r_idx, reg) in registers.iter().enumerate() {
                    let dst = acc.as_mut_ptr().add(unroll_offset + r_idx * I16_CHUNK);
                    simd::store_i16(dst, *reg);
                }
            }
        }
    }

    /// Move a PSQT feature from one square to another.
    pub fn vector_add_sub_psqt(
        input: &Align<[i16; L1_SIZE]>,
        output: &mut Align<[i16; L1_SIZE]>,
        bucket: &Align<[i16; PSQT_FEATURES * L1_SIZE]>,
        add: PsqtFeatureIndex,
        sub: PsqtFeatureIndex,
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
        for i in 0..L1_SIZE / I16_CHUNK {
            // SAFETY: we never hold multiple mutable references, we never mutate immutable memory, etc.
            unsafe {
                let x = simd::load_i16(input.as_ptr().add(i * I16_CHUNK));
                let w_sub = simd::load_i16(s_block.as_ptr().add(i * I16_CHUNK));
                let w_add = simd::load_i16(a_block.as_ptr().add(i * I16_CHUNK));
                let t = simd::sub_i16(x, w_sub);
                let t = simd::add_i16(t, w_add);
                simd::store_i16(output.as_mut_ptr().add(i * I16_CHUNK), t);
            }
        }
    }

    /// Subtract two PSQT features and add one PSQT feature all at once.
    pub fn vector_add_sub2_psqt(
        input: &Align<[i16; L1_SIZE]>,
        output: &mut Align<[i16; L1_SIZE]>,
        bucket: &Align<[i16; PSQT_FEATURES * L1_SIZE]>,
        add: PsqtFeatureIndex,
        sub1: PsqtFeatureIndex,
        sub2: PsqtFeatureIndex,
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
        for i in 0..L1_SIZE / I16_CHUNK {
            // SAFETY: we never hold multiple mutable references, we never mutate immutable memory, etc.
            unsafe {
                let x = simd::load_i16(input.as_ptr().add(i * I16_CHUNK));
                let w_sub1 = simd::load_i16(s_block1.as_ptr().add(i * I16_CHUNK));
                let w_sub2 = simd::load_i16(s_block2.as_ptr().add(i * I16_CHUNK));
                let w_add = simd::load_i16(a_block.as_ptr().add(i * I16_CHUNK));
                let t = simd::sub_i16(x, w_sub1);
                let t = simd::sub_i16(t, w_sub2);
                let t = simd::add_i16(t, w_add);
                simd::store_i16(output.as_mut_ptr().add(i * I16_CHUNK), t);
            }
        }
    }

    /// Add two PSQT features and subtract two PSQT features all at once.
    pub fn vector_add2_sub2_psqt(
        input: &Align<[i16; L1_SIZE]>,
        output: &mut Align<[i16; L1_SIZE]>,
        bucket: &Align<[i16; PSQT_FEATURES * L1_SIZE]>,
        add1: PsqtFeatureIndex,
        add2: PsqtFeatureIndex,
        sub1: PsqtFeatureIndex,
        sub2: PsqtFeatureIndex,
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
        for i in 0..L1_SIZE / I16_CHUNK {
            // SAFETY: we never hold multiple mutable references, we never mutate immutable memory, etc.
            unsafe {
                let x = simd::load_i16(input.as_ptr().add(i * I16_CHUNK));
                let w_sub1 = simd::load_i16(s_block1.as_ptr().add(i * I16_CHUNK));
                let w_sub2 = simd::load_i16(s_block2.as_ptr().add(i * I16_CHUNK));
                let w_add1 = simd::load_i16(a_block1.as_ptr().add(i * I16_CHUNK));
                let w_add2 = simd::load_i16(a_block2.as_ptr().add(i * I16_CHUNK));
                let t = simd::sub_i16(x, w_sub1);
                let t = simd::sub_i16(t, w_sub2);
                let t = simd::add_i16(t, w_add1);
                let t = simd::add_i16(t, w_add2);
                simd::store_i16(output.as_mut_ptr().add(i * I16_CHUNK), t);
            }
        }
    }
}

pub use simd::*;
