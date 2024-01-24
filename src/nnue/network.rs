use std::{
    env,
    fmt::{Debug, Display},
    mem,
    ops::{Deref, DerefMut},
};

use crate::{
    board::{movegen::bitboards::BitBoard, Board},
    image::{self, Image},
    piece::{Colour, Piece, PieceType},
    util::{File, Square, MAX_DEPTH},
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
pub const BUCKETS: usize = 4;
/// The mapping from square to bucket.
pub const fn get_bucket_indices(white_king: Square, black_king: Square) -> (usize, usize) {
    #[rustfmt::skip]
    const BUCKET_MAP: [usize; 64] = [
        0, 0, 0, 0, 4, 4, 4, 4,
        1, 1, 1, 1, 5, 5, 5, 5,
        2, 2, 2, 2, 6, 6, 6, 6,
        2, 2, 2, 2, 6, 6, 6, 6,
        3, 3, 3, 3, 7, 7, 7, 7,
        3, 3, 3, 3, 7, 7, 7, 7,
        3, 3, 3, 3, 7, 7, 7, 7,
        3, 3, 3, 3, 7, 7, 7, 7,
    ];
    let white_bucket = BUCKET_MAP[white_king.index()];
    let black_bucket = BUCKET_MAP[black_king.flip_rank().index()];
    (white_bucket, black_bucket)
}

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
#[derive(Debug, Copy, Clone)]
pub struct FeatureUpdate {
    pub sq: Square,
    pub piece: Piece,
}

impl FeatureUpdate {
    const NULL: Self = Self { sq: Square::NO_SQUARE, piece: Piece::EMPTY };
}

impl Display for FeatureUpdate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{piece} on {sq}", piece = self.piece, sq = self.sq)
    }
}

pub struct UpdateBuffer {
    add: [FeatureUpdate; 2],
    add_count: u8,
    sub: [FeatureUpdate; 2],
    sub_count: u8,
}

impl Debug for UpdateBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "UpdateBuffer {{ adds: [")?;
        for i in 0..self.add_count {
            write!(f, "{}, ", self.add[i as usize])?;
        }
        write!(f, "], subs: [")?;
        for i in 0..self.sub_count {
            write!(f, "{}, ", self.sub[i as usize])?;
        }
        write!(f, "] }}")
    }
}

impl Default for UpdateBuffer {
    fn default() -> Self {
        Self {
            add: [FeatureUpdate::NULL; 2],
            add_count: 0,
            sub: [FeatureUpdate::NULL; 2],
            sub_count: 0,
        }
    }
}

impl UpdateBuffer {
    pub fn move_piece(&mut self, from: Square, to: Square, piece: Piece) {
        self.add[self.add_count as usize] = FeatureUpdate { sq: to, piece };
        self.add_count += 1;
        self.sub[self.sub_count as usize] = FeatureUpdate { sq: from, piece };
        self.sub_count += 1;
    }

    pub fn clear_piece(&mut self, sq: Square, piece: Piece) {
        self.sub[self.sub_count as usize] = FeatureUpdate { sq, piece };
        self.sub_count += 1;
    }

    pub fn add_piece(&mut self, sq: Square, piece: Piece) {
        self.add[self.add_count as usize] = FeatureUpdate { sq, piece };
        self.add_count += 1;
    }
}

/// Stores last-seen accumulators for each bucket, so that we can hopefully avoid
/// having to completely recompute the accumulator for a position, instead
/// partially reconstructing it from the last-seen accumulator.
#[allow(clippy::large_stack_frames)]
#[derive(Clone)]
pub struct BucketAccumulatorCache {
    accs: [[Accumulator; BUCKETS * 2]; 2],
    board_states: [[BitBoard; BUCKETS * 2]; 2],
}

impl BucketAccumulatorCache {
    #[allow(clippy::too_many_lines)]
    pub fn load_accumulator_for_position(
        &mut self,
        board_state: BitBoard,
        pov_update: PovUpdate,
        acc: &mut Accumulator,
    ) {
        debug_assert!(
            matches!(
                pov_update,
                PovUpdate { white: true, black: false } | PovUpdate { white: false, black: true }
            ),
            "invalid pov update: {pov_update:?}"
        );
        #[cfg(debug_assertions)]
        {
            // verify all the cached board states make sense
            for colour in Colour::all() {
                for bucket in 0..BUCKETS {
                    let cached_board_state = self.board_states[colour.index()][bucket];
                    if cached_board_state == unsafe { std::mem::zeroed() } {
                        continue;
                    }
                    let white_king = cached_board_state.piece_bb(Piece::WK).first();
                    let black_king = cached_board_state.piece_bb(Piece::BK).first();
                    let (white_bucket_from_board_position, black_bucket_from_board_position) =
                        get_bucket_indices(white_king, black_king);
                    if colour == Colour::WHITE {
                        assert_eq!(white_bucket_from_board_position, bucket);
                    } else {
                        assert_eq!(black_bucket_from_board_position, bucket);
                    }
                }
            }
        }

        let side_we_care_about = if pov_update.white { Colour::WHITE } else { Colour::BLACK };
        let wk = board_state.piece_bb(Piece::WK).first();
        let bk = board_state.piece_bb(Piece::BK).first();
        let (white_bucket, black_bucket) = get_bucket_indices(wk, bk);
        let bucket = if side_we_care_about == Colour::WHITE { white_bucket } else { black_bucket };
        let cache_acc = &mut self.accs[side_we_care_about.index()][bucket];

        let mut adds = [FeatureUpdate::NULL; 32];
        let mut subs = [FeatureUpdate::NULL; 32];
        let mut add_count = 0;
        let mut sub_count = 0;
        self.board_states[side_we_care_about.index()][bucket].update_iter(
            board_state,
            |f, is_add| {
                if is_add {
                    adds[add_count] = f;
                    add_count += 1;
                } else {
                    subs[sub_count] = f;
                    sub_count += 1;
                }
            },
        );

        for &sub in &subs[..sub_count] {
            NNUEState::update_feature_inplace::<Deactivate>(wk, bk, sub, pov_update, cache_acc);
        }

        for &add in &adds[..add_count] {
            NNUEState::update_feature_inplace::<Activate>(wk, bk, add, pov_update, cache_acc);
        }

        if pov_update.white {
            acc.white = cache_acc.white;
        } else {
            acc.black = cache_acc.black;
        }

        self.board_states[side_we_care_about.index()][bucket] = board_state;
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
            for piece_type in PieceType::all() {
                for square in Square::all() {
                    let feature_indices = feature_indices(
                        Square::H1,
                        Square::H8,
                        FeatureUpdate { sq: square, piece: Piece::new(colour, piece_type) },
                    );
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

    pub fn select_feature_weights(&self, bucket: usize) -> &Align64<[i16; INPUT * LAYER_1_SIZE]> {
        // handle mirroring
        let bucket = bucket % 4;
        let start = bucket * INPUT * LAYER_1_SIZE;
        let end = start + INPUT * LAYER_1_SIZE;
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

const fn feature_indices(
    white_king: Square,
    black_king: Square,
    f: FeatureUpdate,
) -> (usize, usize) {
    const COLOUR_STRIDE: usize = 64 * 6;
    const PIECE_STRIDE: usize = 64;

    let white_sq = if white_king.file() >= File::FILE_E { f.sq.flip_file() } else { f.sq };
    let black_sq = if black_king.file() >= File::FILE_E { f.sq.flip_file() } else { f.sq };

    let piece_type = f.piece.piece_type().index();
    let colour = f.piece.colour().index();

    let white_idx = colour * COLOUR_STRIDE + piece_type * PIECE_STRIDE + white_sq.index();
    let black_idx =
        (1 ^ colour) * COLOUR_STRIDE + piece_type * PIECE_STRIDE + black_sq.flip_rank().index();

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
        // set the current accumulator to the first one
        self.current_acc = 0;

        // initalise all the accumulators in the bucket cache to the bias
        for acc in self.bucket_cache.accs.iter_mut().flatten() {
            acc.init(&NNUE.feature_bias, PovUpdate::BOTH);
        }
        // initialise all the board states in the bucket cache to the empty board
        for board_state in self.bucket_cache.board_states.iter_mut().flatten() {
            *board_state = BitBoard::NULL;
        }

        // refresh the first accumulator
        self.bucket_cache.load_accumulator_for_position(
            board.pieces,
            PovUpdate { white: true, black: false },
            &mut self.accumulators[0],
        );
        self.bucket_cache.load_accumulator_for_position(
            board.pieces,
            PovUpdate { white: false, black: true },
            &mut self.accumulators[0],
        );
    }

    /// Make the next accumulator freshly generated from the board state.
    pub fn push_fresh_acc(&mut self, board: &Board, update: PovUpdate) {
        self.bucket_cache.load_accumulator_for_position(
            board.pieces,
            update,
            &mut self.accumulators[self.current_acc + 1],
        );
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

        match (
            &update_buffer.add[..update_buffer.add_count as usize],
            &update_buffer.sub[..update_buffer.sub_count as usize],
        ) {
            (&[add], &[sub]) => {
                // quiet move
                Self::apply_quiet(white_king, black_king, add, sub, pov_update, old_acc, new_acc);
            }
            (&[add], &[sub1, sub2]) => {
                // capture
                Self::apply_capture(
                    white_king, black_king, add, sub1, sub2, pov_update, old_acc, new_acc,
                );
            }
            (&[add1, add2], &[sub1, sub2]) => {
                // castling
                Self::apply_castling(
                    white_king, black_king, add1, add2, sub1, sub2, pov_update, old_acc, new_acc,
                );
            }
            (_, _) => panic!("invalid update buffer: {update_buffer:?}"),
        }

        self.current_acc += 1;
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
    ) {
        let (white_add, black_add) = feature_indices(white_king, black_king, add);
        let (white_sub, black_sub) = feature_indices(white_king, black_king, sub);

        let (white_bucket, black_bucket) = get_bucket_indices(white_king, black_king);

        let white_bucket = NNUEParams::select_feature_weights(&NNUE, white_bucket);
        let black_bucket = NNUEParams::select_feature_weights(&NNUE, black_bucket);

        if update.white {
            vector_add_sub(
                &source_acc.white,
                &mut target_acc.white,
                white_bucket,
                white_add,
                white_sub,
            );
        }
        if update.black {
            vector_add_sub(
                &source_acc.black,
                &mut target_acc.black,
                black_bucket,
                black_add,
                black_sub,
            );
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
    ) {
        let (white_add, black_add) = feature_indices(white_king, black_king, add);
        let (white_sub1, black_sub1) = feature_indices(white_king, black_king, sub1);
        let (white_sub2, black_sub2) = feature_indices(white_king, black_king, sub2);

        let (white_bucket, black_bucket) = get_bucket_indices(white_king, black_king);

        let white_bucket = NNUEParams::select_feature_weights(&NNUE, white_bucket);
        let black_bucket = NNUEParams::select_feature_weights(&NNUE, black_bucket);

        if update.white {
            vector_add_sub2(
                &source_acc.white,
                &mut target_acc.white,
                white_bucket,
                white_add,
                white_sub1,
                white_sub2,
            );
        }
        if update.black {
            vector_add_sub2(
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
    ) {
        let (white_add1, black_add1) = feature_indices(white_king, black_king, add1);
        let (white_add2, black_add2) = feature_indices(white_king, black_king, add2);
        let (white_sub1, black_sub1) = feature_indices(white_king, black_king, sub1);
        let (white_sub2, black_sub2) = feature_indices(white_king, black_king, sub2);

        let (white_bucket, black_bucket) = get_bucket_indices(white_king, black_king);

        let white_bucket = NNUEParams::select_feature_weights(&NNUE, white_bucket);
        let black_bucket = NNUEParams::select_feature_weights(&NNUE, black_bucket);

        if update.white {
            vector_add2_sub2(
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
            vector_add2_sub2(
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

    /// Update by activating or deactivating a piece.
    pub fn update_feature_inplace<A: Activation>(
        white_king: Square,
        black_king: Square,
        f: FeatureUpdate,
        update: PovUpdate,
        acc: &mut Accumulator,
    ) {
        let (white_idx, black_idx) = feature_indices(white_king, black_king, f);

        let (white_bucket, black_bucket) = get_bucket_indices(white_king, black_king);

        let white_bucket = NNUEParams::select_feature_weights(&NNUE, white_bucket);
        let black_bucket = NNUEParams::select_feature_weights(&NNUE, black_bucket);

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

/// Add two features and subtract two features all at once.
fn vector_add2_sub2(
    input: &Align64<[i16; LAYER_1_SIZE]>,
    output: &mut Align64<[i16; LAYER_1_SIZE]>,
    bucket: &Align64<[i16; INPUT * LAYER_1_SIZE]>,
    feature_idx_add1: usize,
    feature_idx_add2: usize,
    feature_idx_sub1: usize,
    feature_idx_sub2: usize,
) {
    let offset_add1 = feature_idx_add1 * LAYER_1_SIZE;
    let offset_add2 = feature_idx_add2 * LAYER_1_SIZE;
    let offset_sub1 = feature_idx_sub1 * LAYER_1_SIZE;
    let offset_sub2 = feature_idx_sub2 * LAYER_1_SIZE;
    let a_block1 = &bucket[offset_add1..offset_add1 + LAYER_1_SIZE];
    let a_block2 = &bucket[offset_add2..offset_add2 + LAYER_1_SIZE];
    let s_block1 = &bucket[offset_sub1..offset_sub1 + LAYER_1_SIZE];
    let s_block2 = &bucket[offset_sub2..offset_sub2 + LAYER_1_SIZE];
    for i in 0..LAYER_1_SIZE {
        output[i] = input[i] - s_block1[i] - s_block2[i] + a_block1[i] + a_block2[i];
    }
}

/// Subtract two features and add one feature all at once.
fn vector_add_sub2(
    input: &Align64<[i16; LAYER_1_SIZE]>,
    output: &mut Align64<[i16; LAYER_1_SIZE]>,
    bucket: &Align64<[i16; INPUT * LAYER_1_SIZE]>,
    feature_idx_add: usize,
    feature_idx_sub1: usize,
    feature_idx_sub2: usize,
) {
    let offset_add = feature_idx_add * LAYER_1_SIZE;
    let offset_sub1 = feature_idx_sub1 * LAYER_1_SIZE;
    let offset_sub2 = feature_idx_sub2 * LAYER_1_SIZE;
    let a_block = &bucket[offset_add..offset_add + LAYER_1_SIZE];
    let s_block1 = &bucket[offset_sub1..offset_sub1 + LAYER_1_SIZE];
    let s_block2 = &bucket[offset_sub2..offset_sub2 + LAYER_1_SIZE];
    for i in 0..LAYER_1_SIZE {
        output[i] = input[i] - s_block1[i] - s_block2[i] + a_block[i];
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
