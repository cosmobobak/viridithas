use std::{
    env, mem,
    ops::{Deref, DerefMut},
};

use crate::{
    board::Board,
    image::{self, Image},
    piece::{Colour, Piece, PieceType},
    util::{Square, MAX_DEPTH},
};

use super::accumulator::Accumulator;

/// The size of the input layer of the network.
const INPUT: usize = 768;
/// The amount to scale the output of the network by.
/// This is to allow for the sigmoid activation to differentiate positions with
/// a small difference in evaluation.
const SCALE: i32 = 400;
/// The size of one-half of the hidden layer of the network.
pub const LAYER_1_SIZE: usize = 1536;
/// The number of buckets in the feature transformer.
pub const BUCKETS: usize = 1;
/// The mapping from square to bucket.
#[rustfmt::skip]
pub const BUCKET_MAP: [usize; 64] = [
    0, 0, 0, 0, 1, 1, 1, 1,
    0, 0, 0, 0, 1, 1, 1, 1,
    2, 2, 2, 2, 3, 3, 3, 3,
    2, 2, 2, 2, 3, 3, 3, 3,
    2, 2, 2, 2, 3, 3, 3, 3,
    2, 2, 2, 2, 3, 3, 3, 3,
    2, 2, 2, 2, 3, 3, 3, 3,
    2, 2, 2, 2, 3, 3, 3, 3,
];

const QA: i32 = 181;
const QB: i32 = 64;
const QAB: i32 = QA * QB;

/// The size of the stack used to store the activations of the hidden layer.
const ACC_STACK_SIZE: usize = MAX_DEPTH.ply_to_horizon() + 1;

pub trait Activation {
    const ACTIVATE: bool;
    type Reverse: Activation;
}
pub struct Activate;
impl Activation for Activate {
    const ACTIVATE: bool = true;
    type Reverse = Deactivate;
}
pub struct Deactivate;
impl Activation for Deactivate {
    const ACTIVATE: bool = false;
    type Reverse = Activate;
}
#[derive(Debug, Copy, Clone)]
pub struct PovUpdate {
    pub white: bool,
    pub black: bool,
}
impl PovUpdate {
    pub const BOTH: Self = Self { white: true, black: true };

    pub const fn opposite(self) -> Self {
        Self { white: self.black, black: self.white }
    }

    pub fn colour(colour: Colour) -> Self {
        match colour {
            Colour::WHITE => Self { white: true, black: false },
            Colour::BLACK => Self { white: false, black: true },
            _ => unreachable!(),
        }
    }
}

/// Struct representing some unmaterialised feature update made as part of a move.
pub struct FeatureUpdate {
    from: Square,
    to: Square,
    piece: Piece,
}

impl std::fmt::Debug for FeatureUpdate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FeatureUpdate {{ from: {}, to: {}, piece: {} }}", self.from, self.to, self.piece)
    }
}

impl FeatureUpdate {
    const NULL: Self = Self { from: Square::A1, to: Square::A1, piece: Piece::WP };
}

pub struct UpdateBuffer {
    buffer: [FeatureUpdate; 4],
    count: usize,
}

impl Default for UpdateBuffer {
    fn default() -> Self {
        Self { buffer: [FeatureUpdate::NULL; 4], count: 0 }
    }
}

impl UpdateBuffer {
    pub fn move_piece(&mut self, from: Square, to: Square, piece: Piece) {
        self.buffer[self.count] = FeatureUpdate { from, to, piece };
        self.count += 1;
    }

    pub fn clear_piece(&mut self, sq: Square, piece: Piece) {
        self.buffer[self.count] = FeatureUpdate { from: sq, to: Square::NO_SQUARE, piece };
        self.count += 1;
    }

    pub fn add_piece(&mut self, sq: Square, piece: Piece) {
        self.buffer[self.count] = FeatureUpdate { from: Square::NO_SQUARE, to: sq, piece };
        self.count += 1;
    }

    pub fn updates(&self) -> &[FeatureUpdate] {
        &self.buffer[..self.count]
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
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

// read in the binary file containing the network parameters
// have to do some path manipulation to get relative paths to work
// SAFETY: alignment to u16 is guaranteed because transmute() is a copy operation.
pub static NNUE: NNUEParams =
    unsafe { mem::transmute(*include_bytes!(concat!("../../", env!("EVALFILE"),))) };

#[repr(C)]
pub struct NNUEParams {
    pub feature_weights: Align64<[i16; INPUT * LAYER_1_SIZE * BUCKETS]>,
    pub feature_bias: Align64<[i16; LAYER_1_SIZE]>,
    pub output_weights: Align64<[i16; LAYER_1_SIZE * 2]>,
    pub output_bias: i16,
}

impl NNUEParams {
    pub const fn num_params() -> usize {
        INPUT * LAYER_1_SIZE + LAYER_1_SIZE + LAYER_1_SIZE * 2 + 1 // don't duplicate the feature weights
    }

    pub fn visualise_neuron(&self, neuron: usize, path: &std::path::Path) {
        #![allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        // remap pieces to keep opposite colours together
        static PIECE_REMAPPING: [usize; 12] = [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11];
        assert!(neuron < LAYER_1_SIZE);
        let starting_idx = neuron;
        let mut slice = Vec::with_capacity(768);
        for colour in Colour::all() {
            for piece in PieceType::all() {
                for square in Square::all() {
                    let feature_indices = feature_indices(square, piece, colour);
                    let index = feature_indices.0 * LAYER_1_SIZE + starting_idx;
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

    pub const fn select_feature_weights(
        &self,
        _king_sq: Square,
    ) -> &Align64<[i16; INPUT * LAYER_1_SIZE]> {
        // {
        //     let bucket = BUCKET_MAP[king_sq.index()];
        //     let start = bucket * INPUT * LAYER_1_SIZE;
        //     let end = start + INPUT * LAYER_1_SIZE;
        //     let slice = &self.feature_weights[start..end];
        //     // SAFETY: The resulting slice is indeed INPUT * LAYER_1_SIZE long,
        //     // and we check that the slice is aligned to 64 bytes.
        //     // additionally, we're generating the reference from our own data,
        //     // so we know that the lifetime is valid.
        //     unsafe {
        //         // don't immediately cast to Align64, as we want to check the alignment first.
        //         let ptr = slice.as_ptr();
        //         assert_eq!(ptr.align_offset(64), 0);
        //         // alignments are sensible, so we can safely cast.
        //         #[allow(clippy::cast_ptr_alignment)]
        //         &*ptr.cast()
        //     }
        // }
        &self.feature_weights
    }
}

/// State of the partial activations of the NNUE network.
#[allow(clippy::upper_case_acronyms, clippy::large_stack_frames)]
#[derive(Debug, Clone)]
pub struct NNUEState {
    /// Accumulators for the first layer.
    pub accumulators: [Accumulator; ACC_STACK_SIZE],
    /// Index of the current accumulator.
    pub current_acc: usize,
    /// White king locations.
    pub white_king_locs: [Square; ACC_STACK_SIZE],
    /// Black king locations.
    pub black_king_locs: [Square; ACC_STACK_SIZE],
}

const fn feature_indices(sq: Square, piece_type: PieceType, colour: Colour) -> (usize, usize) {
    const COLOUR_STRIDE: usize = 64 * 6;
    const PIECE_STRIDE: usize = 64;

    let piece_type = piece_type.index();
    let colour = colour.index();

    let white_idx = colour * COLOUR_STRIDE + piece_type * PIECE_STRIDE + sq.index();
    let black_idx =
        (1 ^ colour) * COLOUR_STRIDE + piece_type * PIECE_STRIDE + sq.flip_rank().index();

    (white_idx, black_idx)
}

impl NNUEState {
    /// Create a new `NNUEState`.
    #[allow(clippy::unnecessary_box_returns)]
    pub fn new(board: &Board) -> Box<Self> {
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

        net.reinit_from(board);

        net
    }

    /// Decrement the current accumulator.
    pub fn pop_acc(&mut self) {
        self.current_acc -= 1;
    }

    /// Reinitialise the state from a board.
    pub fn reinit_from(&mut self, board: &Board) {
        self.current_acc = 0;

        self.refresh_accumulators(board, PovUpdate { white: true, black: true });
    }

    /// Make the next accumulator freshly generated from the board state.
    pub fn push_fresh_acc(&mut self, board: &Board, update: PovUpdate) {
        self.current_acc += 1;

        self.refresh_accumulators(board, update);

        unimplemented!("push_fresh_acc");
    }

    fn refresh_accumulators(&mut self, board: &Board, update: PovUpdate) {
        let white_king = board.king_sq(Colour::WHITE);
        let black_king = board.king_sq(Colour::BLACK);

        self.accumulators[self.current_acc].init(&NNUE.feature_bias, update);

        for colour in [Colour::WHITE, Colour::BLACK] {
            for piece_type in PieceType::all() {
                let piece = Piece::new(colour, piece_type);
                let piece_bb = board.pieces.piece_bb(piece);

                for sq in piece_bb.iter() {
                    Self::update_feature_inplace::<Activate>(
                        white_king,
                        black_king,
                        piece_type,
                        colour,
                        sq,
                        update,
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
        update_buffer: &UpdateBuffer,
    ) {
        let (front, back) = self.accumulators.split_at_mut(self.current_acc + 1);
        let old_acc = front.last().unwrap();
        let new_acc = back.first_mut().unwrap();

        let (f_ud, t_uds) = update_buffer.updates().split_first().unwrap();

        if f_ud.from == Square::NO_SQUARE {
            Self::update_feature::<Activate>(
                white_king,
                black_king,
                f_ud.piece.piece_type(),
                f_ud.piece.colour(),
                f_ud.to,
                pov_update,
                old_acc,
                new_acc,
            );
        } else if f_ud.to == Square::NO_SQUARE {
            Self::update_feature::<Deactivate>(
                white_king,
                black_king,
                f_ud.piece.piece_type(),
                f_ud.piece.colour(),
                f_ud.from,
                pov_update,
                old_acc,
                new_acc,
            );
        } else {
            Self::move_feature(
                white_king,
                black_king,
                f_ud.piece.piece_type(),
                f_ud.piece.colour(),
                f_ud.from,
                f_ud.to,
                pov_update,
                old_acc,
                new_acc,
            );
        }

        for t_ud in t_uds {
            if t_ud.from == Square::NO_SQUARE {
                Self::update_feature_inplace::<Activate>(
                    white_king,
                    black_king,
                    t_ud.piece.piece_type(),
                    t_ud.piece.colour(),
                    t_ud.to,
                    pov_update,
                    new_acc,
                );
            } else if t_ud.to == Square::NO_SQUARE {
                Self::update_feature_inplace::<Deactivate>(
                    white_king,
                    black_king,
                    t_ud.piece.piece_type(),
                    t_ud.piece.colour(),
                    t_ud.from,
                    pov_update,
                    new_acc,
                );
            } else {
                Self::move_feature_inplace(
                    white_king,
                    black_king,
                    t_ud.piece.piece_type(),
                    t_ud.piece.colour(),
                    t_ud.from,
                    t_ud.to,
                    pov_update,
                    new_acc,
                );
            }
        }

        self.current_acc += 1;
    }

    /// Update the state from a move.
    #[allow(clippy::too_many_arguments)]
    pub fn move_feature(
        white_king: Square,
        black_king: Square,
        piece_type: PieceType,
        colour: Colour,
        from: Square,
        to: Square,
        update: PovUpdate,
        source_acc: &Accumulator,
        target_acc: &mut Accumulator,
    ) {
        let (white_from, black_from) = feature_indices(from, piece_type, colour);
        let (white_to, black_to) = feature_indices(to, piece_type, colour);

        let white_bucket = NNUEParams::select_feature_weights(&NNUE, white_king);
        let black_bucket = NNUEParams::select_feature_weights(&NNUE, black_king.flip_rank());

        if update.white {
            vector_add_sub(
                &source_acc.white,
                &mut target_acc.white,
                white_bucket,
                white_to,
                white_from,
            );
        }
        if update.black {
            vector_add_sub(
                &source_acc.black,
                &mut target_acc.black,
                black_bucket,
                black_to,
                black_from,
            );
        }
    }

    /// Update the state from a move.
    #[allow(clippy::too_many_arguments)]
    pub fn move_feature_inplace(
        white_king: Square,
        black_king: Square,
        piece_type: PieceType,
        colour: Colour,
        from: Square,
        to: Square,
        update: PovUpdate,
        acc: &mut Accumulator,
    ) {
        let (white_from, black_from) = feature_indices(from, piece_type, colour);
        let (white_to, black_to) = feature_indices(to, piece_type, colour);

        let white_bucket = NNUEParams::select_feature_weights(&NNUE, white_king);
        let black_bucket = NNUEParams::select_feature_weights(&NNUE, black_king.flip_rank());

        if update.white {
            vector_add_sub_inplace(&mut acc.white, white_bucket, white_to, white_from);
        }
        if update.black {
            vector_add_sub_inplace(&mut acc.black, black_bucket, black_to, black_from);
        }
    }

    /// Update by activating or deactivating a piece.
    #[allow(clippy::too_many_arguments)]
    pub fn update_feature<A: Activation>(
        white_king: Square,
        black_king: Square,
        piece_type: PieceType,
        colour: Colour,
        sq: Square,
        update: PovUpdate,
        source_acc: &Accumulator,
        target_acc: &mut Accumulator,
    ) {
        let (white_idx, black_idx) = feature_indices(sq, piece_type, colour);

        let white_bucket = NNUEParams::select_feature_weights(&NNUE, white_king);
        let black_bucket = NNUEParams::select_feature_weights(&NNUE, black_king.flip_rank());

        if A::ACTIVATE {
            if update.white {
                vector_add(&source_acc.white, &mut target_acc.white, white_bucket, white_idx);
            }
            if update.black {
                vector_add(&source_acc.black, &mut target_acc.black, black_bucket, black_idx);
            }
        } else {
            if update.white {
                vector_sub(&source_acc.white, &mut target_acc.white, white_bucket, white_idx);
            }
            if update.black {
                vector_sub(&source_acc.black, &mut target_acc.black, black_bucket, black_idx);
            }
        }
    }

    /// Update by activating or deactivating a piece.
    pub fn update_feature_inplace<A: Activation>(
        white_king: Square,
        black_king: Square,
        piece_type: PieceType,
        colour: Colour,
        sq: Square,
        update: PovUpdate,
        acc: &mut Accumulator,
    ) {
        let (white_idx, black_idx) = feature_indices(sq, piece_type, colour);

        let white_bucket = NNUEParams::select_feature_weights(&NNUE, white_king);
        let black_bucket = NNUEParams::select_feature_weights(&NNUE, black_king.flip_rank());

        if A::ACTIVATE {
            if update.white {
                vector_add_inplace(&mut acc.white, white_bucket, white_idx);
            }
            if update.black {
                vector_add_inplace(&mut acc.black, black_bucket, black_idx);
            }
        } else {
            if update.white {
                vector_sub_inplace(&mut acc.white, white_bucket, white_idx);
            }
            if update.black {
                vector_sub_inplace(&mut acc.black, black_bucket, black_idx);
            }
        }
    }

    /// Evaluate the final layer on the partial activations.
    pub fn evaluate(&self, stm: Colour) -> i32 {
        let acc = &self.accumulators[self.current_acc];

        let (us, them) =
            if stm == Colour::WHITE { (&acc.white, &acc.black) } else { (&acc.black, &acc.white) };

        let output = flatten(us, them, &NNUE.output_weights);

        (output + i32::from(NNUE.output_bias)) * SCALE / QAB
    }
}

/// Move a feature from one square to another.
fn vector_add_sub(
    input: &Align64<[i16; LAYER_1_SIZE]>,
    output: &mut Align64<[i16; LAYER_1_SIZE]>,
    bucket: &Align64<[i16; INPUT * LAYER_1_SIZE]>,
    feature_idx_add: usize,
    feature_idx_sub: usize,
) {
    let offset_add = feature_idx_add * LAYER_1_SIZE;
    let offset_sub = feature_idx_sub * LAYER_1_SIZE;
    let s_block = &bucket[offset_sub..offset_sub + LAYER_1_SIZE];
    let a_block = &bucket[offset_add..offset_add + LAYER_1_SIZE];
    for (((i, o), ds), da) in input.iter().zip(output.iter_mut()).zip(s_block).zip(a_block) {
        *o = *i - *ds + *da;
    }
}

/// Move a feature from one square to another
fn vector_add_sub_inplace(
    input: &mut Align64<[i16; LAYER_1_SIZE]>,
    bucket: &Align64<[i16; INPUT * LAYER_1_SIZE]>,
    feature_idx_add: usize,
    feature_idx_sub: usize,
) {
    let offset_add = feature_idx_add * LAYER_1_SIZE;
    let offset_sub = feature_idx_sub * LAYER_1_SIZE;
    let s_block = &bucket[offset_sub..offset_sub + LAYER_1_SIZE];
    let a_block = &bucket[offset_add..offset_add + LAYER_1_SIZE];
    for ((i, ds), da) in input.iter_mut().zip(s_block).zip(a_block) {
        *i = *i - *ds + *da;
    }
}

/// Add a feature to a square.
fn vector_add(
    input: &Align64<[i16; LAYER_1_SIZE]>,
    output: &mut Align64<[i16; LAYER_1_SIZE]>,
    bucket: &Align64<[i16; INPUT * LAYER_1_SIZE]>,
    feature_idx_add: usize,
) {
    let offset_add = feature_idx_add * LAYER_1_SIZE;
    let a_block = &bucket[offset_add..offset_add + LAYER_1_SIZE];
    for ((i, o), da) in input.iter().zip(output.iter_mut()).zip(a_block) {
        *o = *i + *da;
    }
}

/// Add a feature to a square.
fn vector_add_inplace(
    input: &mut Align64<[i16; LAYER_1_SIZE]>,
    bucket: &Align64<[i16; INPUT * LAYER_1_SIZE]>,
    feature_idx_add: usize,
) {
    let offset_add = feature_idx_add * LAYER_1_SIZE;
    let a_block = &bucket[offset_add..offset_add + LAYER_1_SIZE];
    for (i, d) in input.iter_mut().zip(a_block) {
        *i += *d;
    }
}

/// Subtract a feature from a square.
fn vector_sub(
    input: &Align64<[i16; LAYER_1_SIZE]>,
    output: &mut Align64<[i16; LAYER_1_SIZE]>,
    bucket: &Align64<[i16; INPUT * LAYER_1_SIZE]>,
    feature_idx_sub: usize,
) {
    let offset_sub = feature_idx_sub * LAYER_1_SIZE;
    let s_block = &bucket[offset_sub..offset_sub + LAYER_1_SIZE];
    for ((i, o), ds) in input.iter().zip(output.iter_mut()).zip(s_block) {
        *o = *i - *ds;
    }
}

/// Subtract a feature from a square.
fn vector_sub_inplace(
    input: &mut Align64<[i16; LAYER_1_SIZE]>,
    bucket: &Align64<[i16; INPUT * LAYER_1_SIZE]>,
    feature_idx_sub: usize,
) {
    let offset_sub = feature_idx_sub * LAYER_1_SIZE;
    let s_block = &bucket[offset_sub..offset_sub + LAYER_1_SIZE];
    for (i, d) in input.iter_mut().zip(s_block) {
        *i -= *d;
    }
}

fn flatten(
    us: &Align64<[i16; LAYER_1_SIZE]>,
    them: &Align64<[i16; LAYER_1_SIZE]>,
    weights: &Align64<[i16; LAYER_1_SIZE * 2]>,
) -> i32 {
    #[cfg(target_feature = "avx2")]
    unsafe {
        avx2::flatten(us, them, weights)
    }
    #[cfg(not(target_feature = "avx2"))]
    {
        generic::flatten(us, them, weights)
    }
}

/// Non-SIMD implementation of the forward pass.
#[cfg(not(target_feature = "avx2"))]
mod generic {
    use super::{Align64, LAYER_1_SIZE, QA};

    #[allow(clippy::cast_possible_truncation)]
    fn screlu(x: i16) -> i32 {
        let x = x.clamp(0, QA as i16);
        let x = i32::from(x);
        x * x
    }

    /// Execute an activation on the partial activations,
    /// and accumulate the result into a sum.
    pub fn flatten(
        us: &Align64<[i16; LAYER_1_SIZE]>,
        them: &Align64<[i16; LAYER_1_SIZE]>,
        weights: &Align64<[i16; LAYER_1_SIZE * 2]>,
    ) -> i32 {
        let mut sum: i32 = 0;
        for (&i, &w) in us.iter().zip(&weights[..LAYER_1_SIZE]) {
            sum += screlu(i) * i32::from(w);
        }
        for (&i, &w) in them.iter().zip(&weights[LAYER_1_SIZE..]) {
            sum += screlu(i) * i32::from(w);
        }
        sum / QA
    }
}

/// SIMD implementation of the forward pass.
#[cfg(target_feature = "avx2")]
mod avx2 {
    use super::{Align64, LAYER_1_SIZE, QA};
    use std::arch::x86_64::{
        __m256i, _mm256_add_epi32, _mm256_castsi256_si128, _mm256_extracti128_si256,
        _mm256_load_si256, _mm256_madd_epi16, _mm256_max_epi16, _mm256_min_epi16,
        _mm256_mullo_epi16, _mm256_set1_epi16, _mm256_setzero_si256, _mm_add_epi32,
        _mm_cvtsi128_si32, _mm_shuffle_epi32, _mm_unpackhi_epi64,
    };

    type Vec256 = __m256i;

    #[allow(clippy::cast_possible_truncation)]
    #[inline]
    unsafe fn screlu(mut v: Vec256) -> Vec256 {
        let min = _mm256_setzero_si256();
        let max = _mm256_set1_epi16(QA as i16);
        v = _mm256_min_epi16(_mm256_max_epi16(v, min), max);
        _mm256_mullo_epi16(v, v)
    }

    #[inline]
    unsafe fn load_i16s<const VEC_SIZE: usize>(
        acc: &Align64<[i16; VEC_SIZE]>,
        start_idx: usize,
    ) -> Vec256 {
        _mm256_load_si256(acc.0.as_ptr().add(start_idx).cast())
    }

    #[inline]
    unsafe fn horizontal_sum_i32(sum: Vec256) -> i32 {
        let upper_128 = _mm256_extracti128_si256::<1>(sum);
        let lower_128 = _mm256_castsi256_si128(sum);
        let sum_128 = _mm_add_epi32(upper_128, lower_128);
        let upper_64 = _mm_unpackhi_epi64(sum_128, sum_128);
        let sum_64 = _mm_add_epi32(upper_64, sum_128);
        let upper_32 = _mm_shuffle_epi32::<0b00_00_00_01>(sum_64);
        let sum_32 = _mm_add_epi32(upper_32, sum_64);

        _mm_cvtsi128_si32(sum_32)
    }

    /// Execute an activation on the partial activations,
    /// and accumulate the result into a sum.
    pub unsafe fn flatten(
        us: &Align64<[i16; LAYER_1_SIZE]>,
        them: &Align64<[i16; LAYER_1_SIZE]>,
        weights: &Align64<[i16; LAYER_1_SIZE * 2]>,
    ) -> i32 {
        const CHUNK: usize = 16;

        let mut sum = _mm256_setzero_si256();

        // accumulate the first half of the weights
        for i in 0..LAYER_1_SIZE / CHUNK {
            let v = screlu(load_i16s(us, i * CHUNK));
            let w = load_i16s(weights, i * CHUNK);
            let product = _mm256_madd_epi16(v, w);
            sum = _mm256_add_epi32(sum, product);
        }

        // accumulate the second half of the weights
        for i in 0..LAYER_1_SIZE / CHUNK {
            let v = screlu(load_i16s(them, i * CHUNK));
            let w = load_i16s(weights, LAYER_1_SIZE + i * CHUNK);
            let product = _mm256_madd_epi16(v, w);
            sum = _mm256_add_epi32(sum, product);
        }

        horizontal_sum_i32(sum) / QA
    }
}

/// Benchmark the inference portion of the NNUE evaluation.
/// (everything after the feature extraction)
pub fn inference_benchmark(state: &NNUEState) {
    let start = std::time::Instant::now();
    for _ in 0..1_000_000 {
        std::hint::black_box(state.evaluate(Colour::WHITE));
    }
    let elapsed = start.elapsed();
    let nanos = elapsed.as_nanos();
    let ns_per_eval = nanos / 1_000_000;
    println!("{ns_per_eval} ns per evaluation");
}

pub fn visualise_nnue() {
    // create folder for the images
    let path = std::path::PathBuf::from("nnue-visualisations");
    std::fs::create_dir_all(&path).unwrap();
    for neuron in 0..crate::nnue::network::LAYER_1_SIZE {
        crate::nnue::network::NNUE.visualise_neuron(neuron, &path);
    }
}
