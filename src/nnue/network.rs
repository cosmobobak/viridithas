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
pub struct Update {
    pub white: bool,
    pub black: bool,
}
impl Update {
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
    /// Active features from white's perspective.
    #[cfg(debug_assertions)]
    pub white_pov: Align64<[i16; INPUT]>,
    /// Active features from black's perspective.
    #[cfg(debug_assertions)]
    pub black_pov: Align64<[i16; INPUT]>,

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

    /// Copy the current accumulator to the next accumulator, and increment the current accumulator.
    pub fn push_acc(&mut self) {
        self.accumulators[self.current_acc + 1] = self.accumulators[self.current_acc];
        self.current_acc += 1;
    }

    /// Decrement the current accumulator.
    pub fn pop_acc(&mut self) {
        self.current_acc -= 1;
    }

    /// Reinitialise the state from a board.
    pub fn reinit_from(&mut self, board: &Board) {
        self.current_acc = 0;

        self.refresh_accumulators(board, Update { white: true, black: true });
    }

    pub fn refresh_accumulators(&mut self, board: &Board, update: Update) {
        #[cfg(debug_assertions)]
        if update.white {
            self.white_pov.fill(0);
        }
        #[cfg(debug_assertions)]
        if update.black {
            self.black_pov.fill(0);
        }

        let white_king = board.king_sq(Colour::WHITE);
        let black_king = board.king_sq(Colour::BLACK);

        self.accumulators[self.current_acc].init(&NNUE.feature_bias, update);

        for colour in [Colour::WHITE, Colour::BLACK] {
            for piece_type in PieceType::all() {
                let piece = Piece::new(colour, piece_type);
                let piece_bb = board.pieces.piece_bb(piece);

                for sq in piece_bb.iter() {
                    self.update_feature::<Activate>(
                        white_king, black_king, piece_type, colour, sq, update,
                    );
                }
            }
        }
    }

    /// Update the state from a move.
    #[allow(clippy::too_many_arguments)]
    pub fn move_feature(
        &mut self,
        white_king: Square,
        black_king: Square,
        piece_type: PieceType,
        colour: Colour,
        from: Square,
        to: Square,
        update: Update,
    ) {
        let (white_from, black_from) = feature_indices(from, piece_type, colour);
        let (white_to, black_to) = feature_indices(to, piece_type, colour);

        let acc = &mut self.accumulators[self.current_acc];

        let white_bucket = NNUEParams::select_feature_weights(&NNUE, white_king);
        let black_bucket = NNUEParams::select_feature_weights(&NNUE, black_king.flip_rank());

        if update.white {
            vector_add_sub(&mut acc.white, white_bucket, white_to, white_from);
        }
        if update.black {
            vector_add_sub(&mut acc.black, black_bucket, black_to, black_from);
        }

        #[cfg(debug_assertions)]
        {
            self.assert_state::<Activate>(
                white_from,
                black_from,
                (colour, piece_type, from),
                update,
            );
            self.assert_state::<Deactivate>(white_to, black_to, (colour, piece_type, to), update);
            if update.white {
                self.white_pov[white_from] = 0;
            }
            if update.black {
                self.black_pov[black_from] = 0;
            }
            if update.white {
                self.white_pov[white_to] = 1;
            }
            if update.black {
                self.black_pov[black_to] = 1;
            }
        }
    }

    /// Update by activating or deactivating a piece.
    pub fn update_feature<A: Activation>(
        &mut self,
        white_king: Square,
        black_king: Square,
        piece_type: PieceType,
        colour: Colour,
        sq: Square,
        update: Update,
    ) {
        let (white_idx, black_idx) = feature_indices(sq, piece_type, colour);
        let acc = &mut self.accumulators[self.current_acc];
        let white_bucket = NNUEParams::select_feature_weights(&NNUE, white_king);
        let black_bucket = NNUEParams::select_feature_weights(&NNUE, black_king.flip_rank());

        if A::ACTIVATE {
            if update.white {
                vector_add(&mut acc.white, white_bucket, white_idx);
            }
            if update.black {
                vector_add(&mut acc.black, black_bucket, black_idx);
            }
        } else {
            if update.white {
                vector_sub(&mut acc.white, white_bucket, white_idx);
            }
            if update.black {
                vector_sub(&mut acc.black, black_bucket, black_idx);
            }
        }

        #[cfg(debug_assertions)]
        {
            self.assert_state::<A::Reverse>(white_idx, black_idx, (colour, piece_type, sq), update);
            if update.white {
                self.white_pov[white_idx] = A::ACTIVATE.into();
            }
            if update.black {
                self.black_pov[black_idx] = A::ACTIVATE.into();
            }
        }
    }

    /// Update only the feature planes that are affected by the given move.
    /// This is just for debugging purposes.
    #[cfg(debug_assertions)]
    pub fn update_pov_move(
        &mut self,
        piece_type: PieceType,
        colour: Colour,
        from: Square,
        to: Square,
    ) {
        const COLOUR_STRIDE: usize = 64 * 6;
        const PIECE_STRIDE: usize = 64;

        let piece_type = piece_type.index();
        let colour = colour.index();

        let white_idx_from = colour * COLOUR_STRIDE + piece_type * PIECE_STRIDE + from.index();
        let black_idx_from =
            (1 ^ colour) * COLOUR_STRIDE + piece_type * PIECE_STRIDE + from.flip_rank().index();
        let white_idx_to = colour * COLOUR_STRIDE + piece_type * PIECE_STRIDE + to.index();
        let black_idx_to =
            (1 ^ colour) * COLOUR_STRIDE + piece_type * PIECE_STRIDE + to.flip_rank().index();

        self.white_pov[white_idx_from] = 0;
        self.black_pov[black_idx_from] = 0;
        self.white_pov[white_idx_to] = 1;
        self.black_pov[black_idx_to] = 1;
    }

    /// Update only the feature planes that are affected by the addition or removal of a piece.
    /// This is just for debugging purposes.
    #[cfg(debug_assertions)]
    pub fn update_pov_manual<A: Activation>(
        &mut self,
        piece_type: PieceType,
        colour: Colour,
        sq: Square,
    ) {
        const COLOUR_STRIDE: usize = 64 * 6;
        const PIECE_STRIDE: usize = 64;

        let piece_type = piece_type.index();
        let colour = colour.index();

        let white_idx = colour * COLOUR_STRIDE + piece_type * PIECE_STRIDE + sq.index();
        let black_idx =
            (1 ^ colour) * COLOUR_STRIDE + piece_type * PIECE_STRIDE + sq.flip_rank().index();

        if A::ACTIVATE {
            self.white_pov[white_idx] = 1;
            self.black_pov[black_idx] = 1;
        } else {
            self.white_pov[white_idx] = 0;
            self.black_pov[black_idx] = 0;
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

    /// Get the active features for the current position, from white's perspective.
    #[cfg(debug_assertions)]
    pub fn active_features(&self) -> impl Iterator<Item = usize> + '_ {
        self.white_pov.iter().enumerate().filter(|(_, &x)| x == 1).map(|(i, _)| i)
    }

    /// Go from a feature index to the corresponding (colour, piece type, square) tuple.
    ///
    /// (from white's perspective)
    #[cfg(debug_assertions)]
    pub const fn feature_loc_to_parts(loc: usize) -> (Colour, PieceType, Square) {
        #![allow(clippy::cast_possible_truncation)]
        const COLOUR_STRIDE: usize = 64 * 6;
        const PIECE_STRIDE: usize = 64;
        let colour = (loc / COLOUR_STRIDE) as u8;
        let rem = loc % COLOUR_STRIDE;
        let piece = (rem / PIECE_STRIDE) as u8;
        let sq = (rem % PIECE_STRIDE) as u8;
        (Colour::new(colour), PieceType::new(piece), Square::new(sq))
    }

    /// Assert that the input feature planes (the two boards from white's and black's perspective)
    /// are consistent with what we expect.
    #[cfg(debug_assertions)]
    fn assert_state<A: Activation>(
        &self,
        white: usize,
        black: usize,
        feature: (Colour, PieceType, Square),
        update: Update,
    ) {
        #![allow(clippy::bool_to_int_with_if, clippy::cast_possible_truncation)]
        let (colour, piece_type, sq) = feature;
        let val = if A::ACTIVATE { 1 } else { 0 };
        if update.white {
            assert_eq!(
                self.white_pov[white],
                val,
                "piece: {}, sq: {}",
                Piece::new(colour, piece_type),
                sq,
            );
        }
        if update.black {
            assert_eq!(
                self.black_pov[black],
                val,
                "piece: {}, sq: {}",
                Piece::new(colour, piece_type),
                sq,
            );
        }
    }
}

/// Move a feature from one square to another.
fn vector_add_sub(
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

mod tests {
    #[test]
    fn pov_preserved() {
        let mut board = crate::board::Board::default();
        let mut t = crate::threadlocal::ThreadData::new(0, &board);
        let mut ml = crate::board::movegen::MoveList::new();
        board.generate_moves(&mut ml);
        let initial_white = t.nnue.white_pov;
        let initial_black = t.nnue.black_pov;
        for &m in ml.iter() {
            if !board.make_move_nnue(m, &mut t) {
                continue;
            }
            board.unmake_move_nnue(&mut t);
            assert_eq!(initial_white, t.nnue.white_pov);
            assert_eq!(initial_black, t.nnue.black_pov);
        }
    }

    #[test]
    fn pov_preserved_ep() {
        let mut board = crate::board::Board::from_fen(
            "rnbqkbnr/1pp1ppp1/p7/2PpP2p/8/8/PP1P1PPP/RNBQKBNR w KQkq d6 0 5",
        )
        .unwrap();
        let mut t = crate::threadlocal::ThreadData::new(0, &board);
        let mut ml = crate::board::movegen::MoveList::new();
        board.generate_moves(&mut ml);
        let initial_white = t.nnue.white_pov;
        let initial_black = t.nnue.black_pov;
        for &m in ml.iter() {
            if !board.make_move_nnue(m, &mut t) {
                continue;
            }
            board.unmake_move_nnue(&mut t);
            assert_eq!(initial_white, t.nnue.white_pov);
            assert_eq!(initial_black, t.nnue.black_pov);
        }
    }

    #[test]
    fn pov_preserved_castling() {
        let mut board = crate::board::Board::from_fen(
            "rnbqkbnr/1pp1p3/p4pp1/2PpP2p/8/3B1N2/PP1P1PPP/RNBQK2R w KQkq - 0 7",
        )
        .unwrap();
        let mut t = crate::threadlocal::ThreadData::new(0, &board);
        let mut ml = crate::board::movegen::MoveList::new();
        board.generate_moves(&mut ml);
        let initial_white = t.nnue.white_pov;
        let initial_black = t.nnue.black_pov;
        for &m in ml.iter() {
            if !board.make_move_nnue(m, &mut t) {
                continue;
            }
            board.unmake_move_nnue(&mut t);
            assert_eq!(initial_white, t.nnue.white_pov);
            assert_eq!(initial_black, t.nnue.black_pov);
        }
    }

    #[test]
    fn pov_preserved_promo() {
        use crate::nnue::network::NNUEState;

        let mut board = crate::board::Board::from_fen(
            "rnbqk2r/1pp1p1P1/p4np1/2Pp3p/8/3B1N2/PP1P1PPP/RNBQK2R w KQkq - 1 9",
        )
        .unwrap();
        let mut t = crate::threadlocal::ThreadData::new(0, &board);
        let mut ml = crate::board::movegen::MoveList::new();
        board.generate_moves(&mut ml);
        let initial_white = t.nnue.white_pov;
        let initial_black = t.nnue.black_pov;
        for &m in ml.iter() {
            println!("{m}");
            if !board.make_move_nnue(m, &mut t) {
                continue;
            }
            println!("made move");
            board.unmake_move_nnue(&mut t);
            println!("unmade move");
            for i in 0..768 {
                if initial_white[i] != t.nnue.white_pov[i] {
                    let (colour, piecetype, square) = NNUEState::feature_loc_to_parts(i);
                    eprintln!(
                        "{i}: {} != {} ({colour}, {piecetype}, {square}) in {}",
                        initial_white[i],
                        t.nnue.white_pov[i],
                        board.fen()
                    );
                }
                if initial_black[i] != t.nnue.black_pov[i] {
                    let (colour, piecetype, square) = NNUEState::feature_loc_to_parts(i);
                    eprintln!(
                        "{i}: {} != {} ({colour}, {piecetype}, {square}) in {}",
                        initial_black[i],
                        t.nnue.black_pov[i],
                        board.fen()
                    );
                }
            }
            assert_eq!(initial_white, t.nnue.white_pov);
            assert_eq!(initial_black, t.nnue.black_pov);
        }
    }
}
