use std::{
    fmt::{Debug, Display},
    fs::File,
    io::BufReader,
    ops::{Deref, DerefMut},
};

use anyhow::Context;
use arrayvec::ArrayVec;

use crate::{
    board::{movegen::piecelayout::PieceLayout, Board},
    image::{self, Image},
    nnue::simd::{Vector16, Vector32},
    piece::{Colour, Piece, PieceType},
    util::{Square, MAX_DEPTH},
};

use super::{accumulator::Accumulator, simd};

pub mod feature;

/// The size of the input layer of the network.
pub const INPUT: usize = 768;
/// The amount to scale the output of the network by.
/// This is to allow for the sigmoid activation to differentiate positions with
/// a small difference in evaluation.
const SCALE: i32 = 400;
/// The size of one-half of the hidden layer of the network.
pub const L1_SIZE: usize = 2048;
/// The size of the second layer of the network.
pub const L2_SIZE: usize = 16;
/// The size of the third layer of the network.
pub const L3_SIZE: usize = 32;
/// chunking constant for l1
pub const L1_CHUNK_PER_32: usize = size_of::<i32>() / size_of::<i8>();
/// The structure of the king-buckets.
#[rustfmt::skip]
const HALF_BUCKET_MAP: [usize; 32] = [
     0,  1,  2,  3,
     4,  5,  6,  7,
     8,  9, 10, 11,
     8,  9, 10, 11,
    12, 12, 13, 13,
    12, 12, 13, 13,
    14, 14, 15, 15,
    14, 14, 15, 15,
];
/// The number of buckets in the feature transformer.
pub const BUCKETS: usize = max!(HALF_BUCKET_MAP) + 1;
/// The mapping from square to bucket.
const BUCKET_MAP: [usize; 64] = {
    let mut map = [0; 64];
    let mut row = 0;
    while row < 8 {
        let mut col = 0;
        while col < 4 {
            let mirrored = 7 - col;
            map[row * 8 + col] = HALF_BUCKET_MAP[row * 4 + col];
            map[row * 8 + mirrored] = HALF_BUCKET_MAP[row * 4 + col] + BUCKETS;
            col += 1;
        }
        row += 1;
    }
    map
};
/// Get indices into the feature transformer given king positions.
pub fn get_bucket_indices(white_king: Square, black_king: Square) -> (usize, usize) {
    let white_bucket = BUCKET_MAP[white_king];
    let black_bucket = BUCKET_MAP[black_king.flip_rank()];
    (white_bucket, black_bucket)
}
/// The number of output buckets
pub const OUTPUT_BUCKETS: usize = 8;
/// Get index into the output layer given a board state.
pub fn output_bucket(pos: &Board) -> usize {
    #![allow(clippy::cast_possible_truncation)]
    const DIVISOR: usize = (32 + OUTPUT_BUCKETS - 1) / OUTPUT_BUCKETS;
    (pos.n_men() as usize - 2) / DIVISOR
}

const QA: i32 = 255;
const QB: i32 = 64;
// const QAB: i32 = QA * QB;
const FT_SHIFT: i32 = 9;

// read in the binary file containing the network parameters
// have to do some path manipulation to get relative paths to work
pub static COMPRESSED_NNUE: &[u8] = include_bytes!("../../viridithas.nnue.zst");

#[repr(C)]
struct UnquantisedNetwork {
    ft_weights: [f32; INPUT * L1_SIZE * BUCKETS],
    ft_biases: [f32; L1_SIZE],
    l1_weights: [[[f32; L2_SIZE]; OUTPUT_BUCKETS]; L1_SIZE],
    l1_biases: [[f32; L2_SIZE]; OUTPUT_BUCKETS],
    l2_weights: [[[f32; L3_SIZE]; OUTPUT_BUCKETS]; L2_SIZE],
    l2_biases: [[f32; L3_SIZE]; OUTPUT_BUCKETS],
    l3_weights: [[f32; OUTPUT_BUCKETS]; L3_SIZE],
    l3_biases: [f32; OUTPUT_BUCKETS],
}

#[repr(C)]
pub struct NNUEParams {
    pub feature_weights: Align64<[i16; INPUT * L1_SIZE * BUCKETS]>,
    pub feature_bias: Align64<[i16; L1_SIZE]>,
    pub l1_weights: [Align64<[i8; L1_SIZE * L2_SIZE]>; OUTPUT_BUCKETS],
    pub l1_bias: [Align64<[f32; L2_SIZE]>; OUTPUT_BUCKETS],
    pub l2_weights: [Align64<[f32; L2_SIZE * L3_SIZE]>; OUTPUT_BUCKETS],
    pub l2_bias: [Align64<[f32; L3_SIZE]>; OUTPUT_BUCKETS],
    pub l3_weights: [Align64<[f32; L3_SIZE]>; OUTPUT_BUCKETS],
    pub l3_bias: [f32; OUTPUT_BUCKETS],
}

impl UnquantisedNetwork {
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn process(&self, use_simd: bool) -> Box<NNUEParams> {
        let mut net = NNUEParams::zeroed();
        // quantise the feature transformer weights
        for (src, tgt) in self.ft_weights.iter().zip(net.feature_weights.0.iter_mut()) {
            let scaled = *src * QA as f32;
            *tgt = scaled.round() as i16;
        }

        // quantise the feature transformer biases
        for (src, tgt) in self.ft_biases.iter().zip(net.feature_bias.0.iter_mut()) {
            let scaled = *src * QA as f32;
            *tgt = scaled.round() as i16;
        }

        // transpose the L{1,2,3} weights and biases
        for bucket in 0..OUTPUT_BUCKETS {
            // quant the L1 weights
            if use_simd {
                for i in 0..L1_SIZE / L1_CHUNK_PER_32 {
                    for j in 0..L2_SIZE {
                        for k in 0..L1_CHUNK_PER_32 {
                            let v = (f32::round(self.l1_weights[i * L1_CHUNK_PER_32 + k][bucket][j] * QB as f32)) as i8;
                            net.l1_weights[bucket].0[i * L1_CHUNK_PER_32 * L2_SIZE + j * L1_CHUNK_PER_32 + k] = v;
                        }
                    }
                }
            } else {
                for i in 0..L1_SIZE {
                    for j in 0..L2_SIZE {
                        let v = (f32::round(self.l1_weights[i][bucket][j] * QB as f32)) as i8;
                        net.l1_weights[bucket].0[j * L1_SIZE + i] = v;
                    }
                }
            }

            // transfer the L1 biases
            for i in 0..L2_SIZE {
                net.l1_bias[bucket].0[i] = self.l1_biases[bucket][i];
            }

            // transpose the L2 weights
            if use_simd {
                for i in 0..L2_SIZE {
                    for j in 0..L3_SIZE {
                        net.l2_weights[bucket].0[i * L3_SIZE + j] = self.l2_weights[i][bucket][j];
                    }
                }
            } else {
                for i in 0..L2_SIZE {
                    for j in 0..L3_SIZE {
                        net.l2_weights[bucket].0[j * L2_SIZE + i] = self.l2_weights[i][bucket][j];
                    }
                }
            }

            // transfer the L2 biases
            for i in 0..L3_SIZE {
                net.l2_bias[bucket].0[i] = self.l2_biases[bucket][i];
            }

            // transpose the L3 weights
            for i in 0..L3_SIZE {
                net.l3_weights[bucket].0[i] = self.l3_weights[i][bucket];
            }

            // transfer the L3 biases
            net.l3_bias[bucket] = self.l3_biases[bucket];
        }

        net
    }

    fn read(reader: &mut impl std::io::Read) -> anyhow::Result<Box<Self>> {
        // SAFETY: NNUEParams can be zeroed.
        unsafe {
            let layout = std::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc_zeroed(layout);
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            let mem = std::slice::from_raw_parts_mut(ptr.cast(), layout.size());
            reader.read_exact(mem)?;
            Ok(Box::from_raw(ptr.cast()))
        }
    }
}

impl NNUEParams {
    pub fn decompress_and_alloc() -> anyhow::Result<Box<Self>> {
        // SAFETY: NNUEParams can be zeroed.
        unsafe {
            let layout = std::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc_zeroed(layout);
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            #[allow(clippy::cast_ptr_alignment)]
            let mut net = Box::from_raw(ptr.cast::<Self>());
            let mut mem = std::slice::from_raw_parts_mut(std::ptr::from_mut(net.as_mut()).cast::<u8>(), layout.size());
            let expected_bytes = mem.len() as u64;
            let mut decoder = ruzstd::StreamingDecoder::new(COMPRESSED_NNUE)
                .with_context(|| "Failed to construct zstd decoder for NNUE weights.")?;
            let bytes_written = std::io::copy(&mut decoder, &mut mem)
                .with_context(|| "Failed to decompress NNUE weights.")?;
            anyhow::ensure!(bytes_written == expected_bytes, "encountered issue while decompressing NNUE weights, expected {expected_bytes} bytes, but got {bytes_written}");
            Ok(net)
        }
    }

    fn zeroed() -> Box<Self> {
        // SAFETY: NNUEParams can be zeroed.
        unsafe {
            let layout = std::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc_zeroed(layout);
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Box::from_raw(ptr.cast())
        }
    }

    fn write(&self, writer: &mut impl std::io::Write) -> anyhow::Result<()> {
        let ptr = std::ptr::from_ref::<Self>(self).cast::<u8>();
        let len = size_of::<Self>();
        // SAFETY: We're writing a slice of bytes, and we know that the slice is valid.
        writer.write_all(unsafe { std::slice::from_raw_parts(ptr, len) })?;
        Ok(())
    }

    pub fn select_feature_weights(&self, bucket: usize) -> &Align64<[i16; INPUT * L1_SIZE]> {
        // handle mirroring
        let bucket = bucket % BUCKETS;
        let start = bucket * INPUT * L1_SIZE;
        let end = start + INPUT * L1_SIZE;
        let slice = &self.feature_weights[start..end];
        // SAFETY: The resulting slice is indeed INPUT * LAYER_1_SIZE long,
        // and we check that the slice is aligned to 64 bytes.
        // additionally, we're generating the reference from our own data,
        // so we know that the lifetime is valid.
        unsafe {
            // don't immediately cast to Align64, as we want to check the alignment first.
            let ptr = slice.as_ptr();
            assert_eq!(ptr.align_offset(64), 0);
            // alignments are sensible, so we can safely cast.
            #[allow(clippy::cast_ptr_alignment)]
            &*ptr.cast()
        }
    }
}

pub fn quantise(input: &std::path::Path, output: &std::path::Path) -> anyhow::Result<()> {
    let mut reader = BufReader::new(File::open(input)?);
    let mut writer = File::create(output)?;
    let unquantised_net = UnquantisedNetwork::read(&mut reader)?;
    let net = unquantised_net.process(false);
    net.write(&mut writer)?;
    Ok(())
}

/// The size of the stack used to store the activations of the hidden layer.
const ACC_STACK_SIZE: usize = MAX_DEPTH.ply_to_horizon() + 1;

#[derive(Debug, Copy, Clone)]
pub struct PovUpdate {
    pub white: bool,
    pub black: bool,
}
impl PovUpdate {
    pub const BOTH: Self = Self { white: true, black: true };

    pub const fn colour(colour: Colour) -> Self {
        match colour {
            Colour::White => Self { white: true, black: false },
            Colour::Black => Self { white: false, black: true },
        }
    }
}

/// Struct representing some unmaterialised feature update made as part of a move.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FeatureUpdate {
    pub sq: Square,
    pub piece: Piece,
}

impl Display for FeatureUpdate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{piece} on {sq}", piece = self.piece, sq = self.sq)
    }
}

#[derive(PartialEq, Eq, Clone, Debug, Default)]
pub struct UpdateBuffer {
    add: ArrayVec<FeatureUpdate, 2>,
    sub: ArrayVec<FeatureUpdate, 2>,
}

impl UpdateBuffer {
    pub fn move_piece(&mut self, from: Square, to: Square, piece: Piece) {
        self.add.push(FeatureUpdate { sq: to, piece });
        self.sub.push(FeatureUpdate { sq: from, piece });
    }

    pub fn clear_piece(&mut self, sq: Square, piece: Piece) {
        self.sub.push(FeatureUpdate { sq, piece });
    }

    pub fn add_piece(&mut self, sq: Square, piece: Piece) {
        self.add.push(FeatureUpdate { sq, piece });
    }

    pub fn adds(&self) -> &[FeatureUpdate] {
        &self.add[..]
    }

    pub fn subs(&self) -> &[FeatureUpdate] {
        &self.sub[..]
    }
}

/// Stores last-seen accumulators for each bucket, so that we can hopefully avoid
/// having to completely recompute the accumulator for a position, instead
/// partially reconstructing it from the last-seen accumulator.
#[allow(clippy::large_stack_frames)]
#[derive(Clone)]
pub struct BucketAccumulatorCache {
    // both of these are BUCKETS * 2, rather than just BUCKETS,
    // because we use a horizontally-mirrored architecture.
    accs: [Accumulator; BUCKETS * 2],
    board_states: [[PieceLayout; BUCKETS * 2]; 2],
}

impl BucketAccumulatorCache {
    #[allow(clippy::too_many_lines)]
    pub fn load_accumulator_for_position(
        &mut self,
        nnue_params: &NNUEParams,
        board_state: PieceLayout,
        pov_update: PovUpdate,
        acc: &mut Accumulator,
    ) {
        let side_we_care_about = if pov_update.white { Colour::White } else { Colour::Black };
        let wk = board_state.piece_bb(Piece::WK).first();
        let bk = board_state.piece_bb(Piece::BK).first();
        let (white_bucket, black_bucket) = get_bucket_indices(wk, bk);
        let bucket = if side_we_care_about == Colour::White { white_bucket } else { black_bucket };
        let cache_acc = self.accs[bucket].select_mut(side_we_care_about);

        let mut adds = ArrayVec::<_, 32>::new();
        let mut subs = ArrayVec::<_, 32>::new();
        self.board_states[side_we_care_about][bucket].update_iter(board_state, |f, is_add| {
            let (white_idx, black_idx) = feature::indices(wk, bk, f);
            let index = if side_we_care_about == Colour::White { white_idx } else { black_idx };
            if is_add {
                adds.push(index);
            } else {
                subs.push(index);
            }
        });

        let weights = nnue_params.select_feature_weights(bucket);

        simd::vector_update_inplace(cache_acc, weights, &adds, &subs);

        *acc.select_mut(side_we_care_about) = cache_acc.clone();
        acc.correct[side_we_care_about] = true;

        self.board_states[side_we_care_about][bucket] = board_state;
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[repr(C, align(64))]
pub struct Align64<T>(pub T);

impl<T, const SIZE: usize> Deref for Align64<[T; SIZE]> {
    type Target = [T; SIZE];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<T, const SIZE: usize> DerefMut for Align64<[T; SIZE]> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MovedPiece {
    pub from: Square,
    pub to: Square,
    pub piece: Piece,
}

/// State of the partial activations of the NNUE network.
#[allow(clippy::upper_case_acronyms, clippy::large_stack_frames)]
#[derive(Clone)]
pub struct NNUEState {
    /// Accumulators for the first layer.
    pub accumulators: [Accumulator; ACC_STACK_SIZE],
    /// Index of the current accumulator.
    pub current_acc: usize,
    /// Cache of last-seen accumulators for each bucket.
    pub bucket_cache: BucketAccumulatorCache,
}

impl NNUEState {
    /// Create a new `NNUEState`.
    #[allow(clippy::unnecessary_box_returns)]
    pub fn new(board: &Board, nnue_params: &NNUEParams) -> Box<Self> {
        #![allow(clippy::cast_ptr_alignment)]
        // NNUEState is INPUT * 2 * 2 + LAYER_1_SIZE * ACC_STACK_SIZE * 2 * 2 + 8 bytes
        // at time of writing, this adds up to 396,296 bytes.
        // Putting this on the stack will almost certainly blow it, so we box it.
        // Unfortunately, in debug mode `Box::new(Self::new())` will allocate on the stack
        // and then memcpy it to the heap, so we have to do this manually.

        // SAFETY: NNUEState has four fields:
        // {white,black}_pov, which are just arrays of ints, for whom the all-zeroes bitpattern is valid.
        // current_acc, which is just an int, so the all-zeroes bitpattern is valid.
        // accumulators, which is an array of Accumulator<SIZE>.
        //     Accumulator is a struct containing a pair of arrays of ints, so this field is safe for zeroing too.
        // As all fields can be safely initialised to all zeroes, the following code is sound.
        let mut net: Box<Self> = unsafe {
            let layout = std::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc_zeroed(layout);
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Box::from_raw(ptr.cast())
        };

        net.reinit_from(board, nnue_params);

        net
    }

    /// Reinitialise the state from a board.
    pub fn reinit_from(&mut self, board: &Board, nnue_params: &NNUEParams) {
        // set the current accumulator to the first one
        self.current_acc = 0;

        // initalise all the accumulators in the bucket cache to the bias
        for acc in &mut self.bucket_cache.accs {
            acc.init(&nnue_params.feature_bias, PovUpdate::BOTH);
        }
        // initialise all the board states in the bucket cache to the empty board
        for board_state in self.bucket_cache.board_states.iter_mut().flatten() {
            *board_state = PieceLayout::NULL;
        }

        // refresh the first accumulator
        self.bucket_cache.load_accumulator_for_position(
            nnue_params,
            board.pieces,
            PovUpdate { white: true, black: false },
            &mut self.accumulators[0],
        );
        self.bucket_cache.load_accumulator_for_position(
            nnue_params,
            board.pieces,
            PovUpdate { white: false, black: true },
            &mut self.accumulators[0],
        );
    }

    fn requires_refresh(piece: Piece, from: Square, to: Square) -> bool {
        if piece.piece_type() != PieceType::King {
            return false;
        }

        BUCKET_MAP[from] != BUCKET_MAP[to]
    }

    fn can_efficiently_update(&self, view: Colour) -> bool {
        let mut curr_idx = self.current_acc;
        loop {
            curr_idx -= 1;
            let curr = &self.accumulators[curr_idx];

            let mv = curr.mv;
            let from = mv.from.relative_to(view);
            let to = mv.to.relative_to(view);
            let piece = mv.piece;

            if piece.colour() == view && Self::requires_refresh(piece, from, to) {
                return false;
            }
            if curr.correct[view] {
                return true;
            }
        }
    }

    fn apply_lazy_updates(&mut self, nnue_params: &NNUEParams, board: &Board, view: Colour) {
        let mut curr_index = self.current_acc;
        loop {
            curr_index -= 1;

            if self.accumulators[curr_index].correct[view] {
                break;
            }
        }

        let pov_update = PovUpdate::colour(view);
        let white_king = board.king_sq(Colour::White);
        let black_king = board.king_sq(Colour::Black);

        loop {
            self.materialise_new_acc_from(white_king, black_king, pov_update, curr_index + 1, nnue_params);

            self.accumulators[curr_index + 1].correct[view] = true;

            curr_index += 1;
            if curr_index == self.current_acc {
                break;
            }
        }
    }

    /// Apply all in-flight updates, generating all the accumulators up to the current one.
    pub fn force(&mut self, board: &Board, nnue_params: &NNUEParams) {
        for colour in Colour::all() {
            if !self.accumulators[self.current_acc].correct[colour] {
                if self.can_efficiently_update(colour) {
                    self.apply_lazy_updates(nnue_params, board, colour);
                } else {
                    self.bucket_cache.load_accumulator_for_position(
                        nnue_params,
                        board.pieces,
                        PovUpdate::colour(colour),
                        &mut self.accumulators[self.current_acc],
                    );
                }
            }
        }
    }

    pub fn materialise_new_acc_from(
        &mut self,
        white_king: Square,
        black_king: Square,
        pov_update: PovUpdate,
        create_at_idx: usize,
        nnue_params: &NNUEParams,
    ) {
        let (front, back) = self.accumulators.split_at_mut(create_at_idx);
        let src = front.last().unwrap();
        let tgt = back.first_mut().unwrap();

        match (src.update_buffer.adds(), src.update_buffer.subs()) {
            // quiet or promotion
            (&[add], &[sub]) => {
                Self::apply_quiet(white_king, black_king, add, sub, pov_update, src, tgt, nnue_params);
            }
            // capture
            (&[add], &[sub1, sub2]) => {
                Self::apply_capture(white_king, black_king, add, sub1, sub2, pov_update, src, tgt, nnue_params);
            }
            // castling
            (&[add1, add2], &[sub1, sub2]) => {
                Self::apply_castling(white_king, black_king, add1, add2, sub1, sub2, pov_update, src, tgt, nnue_params);
            }
            (_, _) => panic!("invalid update buffer: {:?}", src.update_buffer),
        }
    }

    /// Move a single piece on the board.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_quiet(
        white_king: Square,
        black_king: Square,
        add: FeatureUpdate,
        sub: FeatureUpdate,
        update: PovUpdate,
        source_acc: &Accumulator,
        target_acc: &mut Accumulator,
        nnue_params: &NNUEParams,
    ) {
        let (white_add, black_add) = feature::indices(white_king, black_king, add);
        let (white_sub, black_sub) = feature::indices(white_king, black_king, sub);

        let (white_bucket, black_bucket) = get_bucket_indices(white_king, black_king);

        let white_bucket = nnue_params.select_feature_weights(white_bucket);
        let black_bucket = nnue_params.select_feature_weights(black_bucket);

        if update.white {
            simd::vector_add_sub(&source_acc.white, &mut target_acc.white, white_bucket, white_add, white_sub);
        }
        if update.black {
            simd::vector_add_sub(&source_acc.black, &mut target_acc.black, black_bucket, black_add, black_sub);
        }
    }

    /// Make a capture on the board.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_capture(
        white_king: Square,
        black_king: Square,
        add: FeatureUpdate,
        sub1: FeatureUpdate,
        sub2: FeatureUpdate,
        update: PovUpdate,
        source_acc: &Accumulator,
        target_acc: &mut Accumulator,
        nnue_params: &NNUEParams,
    ) {
        let (white_add, black_add) = feature::indices(white_king, black_king, add);
        let (white_sub1, black_sub1) = feature::indices(white_king, black_king, sub1);
        let (white_sub2, black_sub2) = feature::indices(white_king, black_king, sub2);

        let (white_bucket, black_bucket) = get_bucket_indices(white_king, black_king);

        let white_bucket = nnue_params.select_feature_weights(white_bucket);
        let black_bucket = nnue_params.select_feature_weights(black_bucket);

        if update.white {
            simd::vector_add_sub2(
                &source_acc.white,
                &mut target_acc.white,
                white_bucket,
                white_add,
                white_sub1,
                white_sub2,
            );
        }
        if update.black {
            simd::vector_add_sub2(
                &source_acc.black,
                &mut target_acc.black,
                black_bucket,
                black_add,
                black_sub1,
                black_sub2,
            );
        }
    }

    /// Make a castling move on the board.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_castling(
        white_king: Square,
        black_king: Square,
        add1: FeatureUpdate,
        add2: FeatureUpdate,
        sub1: FeatureUpdate,
        sub2: FeatureUpdate,
        update: PovUpdate,
        source_acc: &Accumulator,
        target_acc: &mut Accumulator,
        nnue_params: &NNUEParams,
    ) {
        let (white_add1, black_add1) = feature::indices(white_king, black_king, add1);
        let (white_add2, black_add2) = feature::indices(white_king, black_king, add2);
        let (white_sub1, black_sub1) = feature::indices(white_king, black_king, sub1);
        let (white_sub2, black_sub2) = feature::indices(white_king, black_king, sub2);

        let (white_bucket, black_bucket) = get_bucket_indices(white_king, black_king);

        let white_bucket = nnue_params.select_feature_weights(white_bucket);
        let black_bucket = nnue_params.select_feature_weights(black_bucket);

        if update.white {
            simd::vector_add2_sub2(
                &source_acc.white,
                &mut target_acc.white,
                white_bucket,
                white_add1,
                white_add2,
                white_sub1,
                white_sub2,
            );
        }
        if update.black {
            simd::vector_add2_sub2(
                &source_acc.black,
                &mut target_acc.black,
                black_bucket,
                black_add1,
                black_add2,
                black_sub1,
                black_sub2,
            );
        }
    }

    /// Evaluate the final layer on the partial activations.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn evaluate(&self, nn: &NNUEParams, stm: Colour, out: usize) -> i32 {
        let acc = &self.accumulators[self.current_acc];

        debug_assert!(acc.correct[0] && acc.correct[1]);

        let (us, them) = if stm == Colour::White { (&acc.white, &acc.black) } else { (&acc.black, &acc.white) };

        let mut ft_outputs = Align64([0; L1_SIZE]);
        let mut l1_outputs = Align64([0.0; L2_SIZE]);
        let mut l2_outputs = Align64([0.0; L3_SIZE]);
        let mut l3_output = 0.0;

        activate_ft(us, them, &mut ft_outputs);
        propagate_l1(&ft_outputs, &nn.l1_weights[out], &nn.l1_bias[out], &mut l1_outputs);
        propagate_l2(&l1_outputs, &nn.l2_weights[out], &nn.l2_bias[out], &mut l2_outputs);
        propagate_l3(&l2_outputs, &nn.l3_weights[out], nn.l3_bias[out], &mut l3_output);

        (l3_output * SCALE as f32) as i32
    }
}

/// Implementation of the forward pass.
#[allow(dead_code)]
fn flatten(us: &Align64<[i16; L1_SIZE]>, them: &Align64<[i16; L1_SIZE]>, weights: &Align64<[i16; L1_SIZE * 2]>) -> i32 {
    const CHUNK: usize = Vector16::COUNT;

    // SAFETY: a great number of unsafe functions are called in the below code, but we're
    // implementing a perfectly safe pattern, just using SIMD equivalents of normal things.
    unsafe {
        let mut sum = Vector32::zero();
        let min = Vector16::zero();
        #[allow(clippy::cast_possible_truncation)]
        let max = Vector16::splat(QA as i16);

        // the following code uses a trick devised by the author of the Lizard chess engine.
        // we're implementing the function f(x) = clamp(x, 0, QA)^2 * w,
        // and we do this in the following manner:
        // 1. load the input, x
        // 2. compute v := clamp(x, 0, QA)
        // 3. load the weight, w
        // 4. compute t := v * w via truncating 16-bit multiply.
        //    this step relies on our invariant that v * w fits in i16.
        // 5. compute product := v * t via horizontally accumulating
        //    expand-to-i32 multiply.
        // 6. add product to the running sum.
        // at this point we've computed clamp(x, 0, QA)^2 * w
        // by doing (clamp(x, 0, QA) * w) * clamp(x, 0, QA).
        // the clever part is step #4, which the compiler cannot know to do.

        // accumulate the first half of the weights
        for i in 0..L1_SIZE / CHUNK {
            let x = Vector16::load_at(us, i * CHUNK);
            let v = Vector16::min(Vector16::max(x, min), max);
            let w = Vector16::load_at(weights, i * CHUNK);
            let product = Vector16::mul_widening(v, Vector16::mul_truncating(v, w));
            sum = Vector32::add(sum, product);
        }

        // accumulate the second half of the weights
        for i in 0..L1_SIZE / CHUNK {
            let x = Vector16::load_at(them, i * CHUNK);
            let v = Vector16::min(Vector16::max(x, min), max);
            let w = Vector16::load_at(weights, L1_SIZE + i * CHUNK);
            let product = Vector16::mul_widening(v, Vector16::mul_truncating(v, w));
            sum = Vector32::add(sum, product);
        }

        Vector32::sum(sum) / QA
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::cast_sign_loss)]
fn activate_ft(us: &Align64<[i16; L1_SIZE]>, them: &Align64<[i16; L1_SIZE]>, output: &mut Align64<[u8; L1_SIZE]>) {
    // this is just autovec'd for the moment.
    for (a, acc) in [us, them].into_iter().enumerate() {
        for i in 0..L1_SIZE / 2 {
            let l = acc.0[i];
            let r = acc.0[L1_SIZE / 2 + i];
            let cl = i32::clamp(i32::from(l), 0, QA);
            let cr = i32::clamp(i32::from(r), 0, QA);
            output.0[i + a * L1_SIZE / 2] = ((cl * cr) >> FT_SHIFT) as u8;
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
    const SUM_DIV: f32 = ((QA * QA * QB) >> FT_SHIFT) as f32;
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

#[allow(clippy::needless_range_loop)]
fn propagate_l2(
    inputs: &Align64<[f32; L2_SIZE]>,
    weights: &Align64<[f32; L2_SIZE * L3_SIZE]>,
    biases: &Align64<[f32; L3_SIZE]>,
    output: &mut Align64<[f32; L3_SIZE]>,
) {
    // this is just autovec'd for the moment.
    let mut sums = [0.0; L3_SIZE];

    sums.copy_from_slice(&biases.0);

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

fn propagate_l3(inputs: &Align64<[f32; L3_SIZE]>, weights: &Align64<[f32; L3_SIZE]>, bias: f32, output: &mut f32) {
    let mut sum = bias;

    for i in 0..L3_SIZE {
        sum += inputs.0[i] * weights.0[i];
    }

    *output = sum;
}

/// Benchmark the inference portion of the NNUE evaluation.
/// (everything after the feature extraction)
pub fn inference_benchmark(state: &NNUEState, nnue_params: &NNUEParams) {
    let start = std::time::Instant::now();
    for _ in 0..1_000_000 {
        std::hint::black_box(state.evaluate(nnue_params, Colour::White, 0));
    }
    let elapsed = start.elapsed();
    let nanos = elapsed.as_nanos();
    let ns_per_eval = nanos / 1_000_000;
    println!("{ns_per_eval} ns per evaluation");
}

pub fn visualise_nnue() -> anyhow::Result<()> {
    let nnue_params = NNUEParams::decompress_and_alloc()?;
    // create folder for the images
    let path = std::path::PathBuf::from("nnue-visualisations");
    std::fs::create_dir_all(&path).with_context(|| "Failed to create NNUE visualisations folder.")?;
    for neuron in 0..crate::nnue::network::L1_SIZE {
        nnue_params.visualise_neuron(neuron, &path);
    }
    let (min, max) = nnue_params.min_max_feature_weight();
    println!("Min / Max FT values: {min} / {max}");
    Ok(())
}

impl NNUEParams {
    pub fn visualise_neuron(&self, neuron: usize, path: &std::path::Path) {
        #![allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        // remap pieces to keep opposite colours together
        static PIECE_REMAPPING: [usize; 12] = [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11];
        assert!(neuron < L1_SIZE);
        let starting_idx = neuron;
        let mut slice = Vec::with_capacity(768);
        for colour in Colour::all() {
            for piece_type in PieceType::all() {
                for square in Square::all() {
                    let feature_indices = feature::indices(
                        Square::H1,
                        Square::H8,
                        FeatureUpdate { sq: square, piece: Piece::new(colour, piece_type) },
                    );
                    let index = feature_indices.0.index() * L1_SIZE + starting_idx;
                    slice.push(self.feature_weights[index]);
                }
            }
        }

        let mut image = Image::zeroed(8 * 6 + 5, 8 * 2 + 1); // + for inter-piece spacing

        for (piece, chunk) in slice.chunks(64).enumerate() {
            let piece = PIECE_REMAPPING[piece];
            let piece_colour = piece % 2;
            let piece_type = piece / 2;
            let (max, min) = if piece_type == 0 {
                let chunk = &chunk[8..56]; // first and last rank are always 0 for pawns
                (*chunk.iter().max().unwrap(), *chunk.iter().min().unwrap())
            } else {
                (*chunk.iter().max().unwrap(), *chunk.iter().min().unwrap())
            };
            let weight_to_colour = |weight: i16| -> u32 {
                let intensity = f32::from(weight - min) / f32::from(max - min);
                let idx = (intensity * 255.0).round() as u8;
                image::inferno_colour_map(idx)
            };
            for (square, &weight) in chunk.iter().enumerate() {
                let row = square / 8;
                let col = square % 8;
                let colour = if (row == 0 || row == 7) && piece_type == 0 {
                    0 // pawns on first and last rank are always 0
                } else {
                    weight_to_colour(weight)
                };
                image.set(col + piece_type * 8 + piece_type, row + piece_colour * 9, colour);
            }
        }

        let path = path.join(format!("neuron_{neuron}.tga"));
        image.save_as_tga(path);
    }

    pub fn min_max_feature_weight(&self) -> (i16, i16) {
        let mut min = i16::MAX;
        let mut max = i16::MIN;
        for &f in &self.feature_weights.0 {
            if f < min {
                min = f;
            }
            if f > max {
                max = f;
            }
        }
        (min, max)
    }
}
