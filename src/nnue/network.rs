use std::{
    array::from_mut,
    fs, mem,
    ops::{Deref, DerefMut},
};

use serde_json::Value;

use crate::{
    board::{movegen::BitLoop, Board},
    definitions::Square,
    image::{self, Image},
    piece::{Colour, Piece, PieceType},
};

use super::accumulator::Accumulator;

/// The size of the input layer of the network.
const INPUT: usize = 768;
/// The minimum value for the clipped relu activation.
const CR_MIN: i16 = 0;
/// The maximum value for the clipped relu activation.
const CR_MAX: i16 = 255;
/// The amount to scale the output of the network by.
/// This is to allow for the sigmoid activation to differentiate positions with
/// a small difference in evaluation.
const SCALE: i32 = 400;
/// The size of one-half of the hidden layer of the network.
pub const LAYER_1_SIZE: usize = 512;
// const BUCKETS: usize = 16;

const QA: i32 = 255;
const QB: i32 = 64;
const QAB: i32 = QA * QB;

/// The size of the stack used to store the activations of the hidden layer.
const ACC_STACK_SIZE: usize = 256;

pub trait Activation {
    const ACTIVATE: bool;
}
pub struct Activate;
impl Activation for Activate {
    const ACTIVATE: bool = true;
}
pub struct Deactivate;
impl Activation for Deactivate {
    const ACTIVATE: bool = false;
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(C, align(64))]
pub struct Align<T>(pub T);

impl<T, const SIZE: usize> Deref for Align<[T; SIZE]> {
    type Target = [T; SIZE];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<T, const SIZE: usize> DerefMut for Align<[T; SIZE]> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

// read in bytes from files and transmute them into u16s.
// SAFETY: alignment to u16 is guaranteed because transmute() is a copy operation.
pub static NNUE: NNUEParams = NNUEParams {
    feature_weights: unsafe { mem::transmute(*include_bytes!("../../nnue/flipped_weights.bin")) },
    feature_bias: unsafe { mem::transmute(*include_bytes!("../../nnue/feature_bias.bin")) },
    output_weights: unsafe { mem::transmute(*include_bytes!("../../nnue/output_weights.bin")) },
    output_bias: unsafe { mem::transmute(*include_bytes!("../../nnue/output_bias.bin")) },
};

pub struct NNUEParams {
    pub feature_weights: Align<[i16; INPUT * LAYER_1_SIZE]>,
    pub feature_bias: Align<[i16; LAYER_1_SIZE]>,
    pub output_weights: Align<[i16; LAYER_1_SIZE * 2]>,
    pub output_bias: i16,
}

// pub static NNUE2: NNUEParams2 = NNUEParams2 {
//     input_weights: [Align([0; LAYER_1_SIZE]); INPUT],
//     input_bias: Align([0; LAYER_1_SIZE]),
//     hidden_weights: [Align([0; LAYER_1_SIZE * 2]); BUCKETS],
//     hidden_bias: Align([0; BUCKETS]),
// };

// pub struct NNUEParams2 {
//     pub input_weights: [Align<[i16; LAYER_1_SIZE]>; INPUT],
//     pub input_bias: Align<[i16; LAYER_1_SIZE]>,
//     pub hidden_weights: [Align<[i16; LAYER_1_SIZE * 2]>; BUCKETS],
//     pub hidden_bias: Align<[i16; BUCKETS]>,
// }

impl NNUEParams {
    pub const fn num_params() -> usize {
        INPUT * LAYER_1_SIZE + LAYER_1_SIZE + LAYER_1_SIZE * 2 + 1 // don't duplicate the feature weights
    }

    pub fn visualise_neuron(&self, neuron: usize, path: &std::path::Path) {
        #![allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        // remap pieces to keep opposite colours together
        static PIECE_REMAPPING: [usize; 12] = [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11];
        assert!(neuron < LAYER_1_SIZE);
        let starting_idx = neuron * INPUT;
        let slice = &self.feature_weights[starting_idx..starting_idx + INPUT];

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

    pub fn from_json(path: impl AsRef<std::path::Path>) -> Box<Self> {
        #![allow(clippy::cast_possible_truncation)]
        fn weight<const LEN: usize>(
            weight_relation: &Value,
            weight_array: &mut [i16; LEN],
            stride: usize,
            k: i32,
            flip: bool,
        ) {
            for (i, output) in weight_relation.as_array().unwrap().iter().enumerate() {
                for (j, weight) in output.as_array().unwrap().iter().enumerate() {
                    let index = if flip { j * stride + i } else { i * stride + j };
                    let value = weight.as_f64().unwrap();
                    weight_array[index] = (value * f64::from(k)) as i16;
                }
            }
        }

        fn bias<const LEN: usize>(bias_relation: &Value, bias_array: &mut [i16; LEN], k: i32) {
            for (i, bias) in bias_relation.as_array().unwrap().iter().enumerate() {
                let value = bias.as_f64().unwrap();
                bias_array[i] = (value * f64::from(k)) as i16;
            }
        }

        #[allow(clippy::large_stack_arrays)]
        // we only run this function in release, so it shouldn't blow the stack.
        let mut out = Box::new(Self {
            feature_weights: Align([0; INPUT * LAYER_1_SIZE]),
            feature_bias: Align([0; LAYER_1_SIZE]),
            output_weights: Align([0; LAYER_1_SIZE * 2]),
            output_bias: 0,
        });

        let file = fs::read_to_string(path).unwrap();
        let json: Value = serde_json::from_str(&file).unwrap();

        for (key, value) in json.as_object().unwrap() {
            match key.as_str() {
                "ft.weight" => {
                    // weight(value, &mut out.feature_weights, INPUT, QA, false);
                    weight(value, &mut out.feature_weights, LAYER_1_SIZE, QA, true);
                }
                "ft.bias" => {
                    bias(value, &mut out.feature_bias, QA);
                }
                "out.weight" => {
                    weight(value, &mut out.output_weights, LAYER_1_SIZE * 2, QB, false);
                }
                "out.bias" => {
                    bias(value, from_mut(&mut out.output_bias), QAB);
                }
                _ => {}
            }
        }

        out
    }

    pub fn to_bytes(&self) -> Vec<Vec<u8>> {
        let mut out = Vec::new();

        // let (head, feature_weights, tail) = unsafe { self.feature_weights.align_to::<u8>() };
        // assert!(head.is_empty() && tail.is_empty());
        let (head, feature_weights, tail) = unsafe { self.feature_weights.align_to::<u8>() };
        assert!(head.is_empty() && tail.is_empty());
        let (head, feature_bias, tail) = unsafe { self.feature_bias.align_to::<u8>() };
        assert!(head.is_empty() && tail.is_empty());
        let (head, output_weights, tail) = unsafe { self.output_weights.align_to::<u8>() };
        assert!(head.is_empty() && tail.is_empty());
        let ob = [self.output_bias];
        let (head, output_bias, tail) = unsafe { ob.align_to::<u8>() };
        assert!(head.is_empty() && tail.is_empty());

        // out.push(feature_weights.to_vec());
        out.push(feature_weights.to_vec());
        out.push(feature_bias.to_vec());
        out.push(output_weights.to_vec());
        out.push(output_bias.to_vec());

        out
    }
}

/// State of the partial activations of the NNUE network.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone)]
pub struct NNUEState {
    /// Active features from white's perspective.
    #[cfg(debug_assertions)]
    pub white_pov: Align<[i16; INPUT]>,
    /// Active features from black's perspective.
    #[cfg(debug_assertions)]
    pub black_pov: Align<[i16; INPUT]>,

    /// Accumulators for the first layer.
    pub accumulators: [Accumulator<LAYER_1_SIZE>; ACC_STACK_SIZE],
    /// Index of the current accumulator.
    pub current_acc: usize,
}

impl NNUEState {
    /// Create a new `NNUEState`.
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

        net.refresh_acc(board);

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
    pub fn refresh_acc(&mut self, board: &Board) {
        self.current_acc = 0;

        #[cfg(debug_assertions)]
        self.white_pov.fill(0);
        #[cfg(debug_assertions)]
        self.black_pov.fill(0);

        self.accumulators[self.current_acc].init(&NNUE.feature_bias);

        for colour in [Colour::WHITE, Colour::BLACK] {
            for piece_type in PieceType::all() {
                let piece = Piece::new(colour, piece_type);
                let piece_bb = board.pieces.piece_bb(piece);

                for sq in BitLoop::new(piece_bb) {
                    self.efficiently_update_manual::<Activate>(piece_type, colour, sq);
                }
            }
        }
    }

    /// Update the state from a move.
    pub fn efficiently_update_from_move(
        &mut self,
        piece_type: PieceType,
        colour: Colour,
        from: Square,
        to: Square,
    ) {
        #![allow(clippy::cast_possible_truncation)]
        const COLOUR_STRIDE: usize = 64 * 6;
        const PIECE_STRIDE: usize = 64;

        let piece_type = piece_type.index() - 1; // shift into correct range.
        let colour = colour.index();

        let white_idx_from = colour * COLOUR_STRIDE + piece_type * PIECE_STRIDE + from.index();
        let black_idx_from =
            (1 ^ colour) * COLOUR_STRIDE + piece_type * PIECE_STRIDE + from.flip_rank().index();
        let white_idx_to = colour * COLOUR_STRIDE + piece_type * PIECE_STRIDE + to.index();
        let black_idx_to =
            (1 ^ colour) * COLOUR_STRIDE + piece_type * PIECE_STRIDE + to.flip_rank().index();

        #[cfg(debug_assertions)]
        {
            assert_eq!(
                self.white_pov[white_idx_from],
                1,
                "piece: {}, from: {}, to: {}",
                Piece::new(Colour::new(colour as u8), PieceType::new(piece_type as u8 + 1)),
                from,
                to
            );
            assert_eq!(
                self.black_pov[black_idx_from],
                1,
                "piece: {}, from: {}, to: {}",
                Piece::new(Colour::new(colour as u8), PieceType::new(piece_type as u8 + 1)),
                from,
                to
            );
            assert_eq!(
                self.white_pov[white_idx_to],
                0,
                "piece: {}, from: {}, to: {}",
                Piece::new(Colour::new(colour as u8), PieceType::new(piece_type as u8 + 1)),
                from,
                to
            );
            assert_eq!(
                self.black_pov[black_idx_to],
                0,
                "piece: {}, from: {}, to: {}",
                Piece::new(Colour::new(colour as u8), PieceType::new(piece_type as u8 + 1)),
                from,
                to
            );
            self.white_pov[white_idx_from] = 0;
            self.black_pov[black_idx_from] = 0;
            self.white_pov[white_idx_to] = 1;
            self.black_pov[black_idx_to] = 1;
        }

        let acc = &mut self.accumulators[self.current_acc];

        subtract_and_add_to_all(
            &mut acc.white,
            &NNUE.feature_weights,
            white_idx_from * LAYER_1_SIZE,
            white_idx_to * LAYER_1_SIZE,
        );
        subtract_and_add_to_all(
            &mut acc.black,
            &NNUE.feature_weights,
            black_idx_from * LAYER_1_SIZE,
            black_idx_to * LAYER_1_SIZE,
        );
    }

    /// Update by activating or deactivating a piece.
    pub fn efficiently_update_manual<A: Activation>(
        &mut self,
        piece_type: PieceType,
        colour: Colour,
        sq: Square,
    ) {
        #![allow(clippy::cast_possible_truncation)]
        const COLOUR_STRIDE: usize = 64 * 6;
        const PIECE_STRIDE: usize = 64;

        let sq = sq;
        let piece_type = piece_type.index() - 1; // shift into correct range.
        let colour = colour.index();

        let white_idx = colour * COLOUR_STRIDE + piece_type * PIECE_STRIDE + sq.index();
        let black_idx =
            (1 ^ colour) * COLOUR_STRIDE + piece_type * PIECE_STRIDE + sq.flip_rank().index();

        let acc = &mut self.accumulators[self.current_acc];

        if A::ACTIVATE {
            #[cfg(debug_assertions)]
            {
                debug_assert!(
                    self.white_pov[white_idx] == 0,
                    "piece: {}, sq: {}",
                    Piece::new(Colour::new(colour as u8), PieceType::new(piece_type as u8 + 1)),
                    sq
                );
                debug_assert!(
                    self.black_pov[black_idx] == 0,
                    "piece: {}, sq: {}",
                    Piece::new(Colour::new(colour as u8), PieceType::new(piece_type as u8 + 1)),
                    sq
                );
                self.white_pov[white_idx] = 1;
                self.black_pov[black_idx] = 1;
            }
            add_to_all(&mut acc.white, &NNUE.feature_weights, white_idx * LAYER_1_SIZE);
            add_to_all(&mut acc.black, &NNUE.feature_weights, black_idx * LAYER_1_SIZE);
        } else {
            #[cfg(debug_assertions)]
            {
                debug_assert!(
                    self.white_pov[white_idx] == 1,
                    "piece: {}, sq: {}",
                    Piece::new(Colour::new(colour as u8), PieceType::new(piece_type as u8 + 1)),
                    sq
                );
                debug_assert!(
                    self.black_pov[black_idx] == 1,
                    "piece: {}, sq: {}",
                    Piece::new(Colour::new(colour as u8), PieceType::new(piece_type as u8 + 1)),
                    sq
                );
                self.white_pov[white_idx] = 0;
                self.black_pov[black_idx] = 0;
            }
            sub_from_all(&mut acc.white, &NNUE.feature_weights, white_idx * LAYER_1_SIZE);
            sub_from_all(&mut acc.black, &NNUE.feature_weights, black_idx * LAYER_1_SIZE);
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

        let piece_type = piece_type.index() - 1; // shift into correct range.
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

        let sq = sq;
        let piece_type = piece_type.index() - 1; // shift into correct range.
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

        let output = crelu_flatten(us, them, &NNUE.output_weights);

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
        (Colour::new(colour), PieceType::new(piece + 1), Square::new(sq))
    }
}

/// Move a feature from one square to another.
fn subtract_and_add_to_all<const SIZE: usize, const WEIGHTS: usize>(
    input: &mut [i16; SIZE],
    delta: &[i16; WEIGHTS],
    offset_sub: usize,
    offset_add: usize,
) {
    let s_block = &delta[offset_sub..offset_sub + SIZE];
    let a_block = &delta[offset_add..offset_add + SIZE];
    for ((i, ds), da) in input.iter_mut()
        .zip(s_block)
        .zip(a_block)
    {
        *i = *i - *ds + *da;
    }
}

/// Add a feature to a square.
fn add_to_all<const SIZE: usize, const WEIGHTS: usize>(
    input: &mut [i16; SIZE],
    delta: &[i16; WEIGHTS],
    offset_add: usize,
) {
    let a_block = &delta[offset_add..offset_add + SIZE];
    for (i, d) in input.iter_mut().zip(a_block) {
        *i += *d;
    }
}

/// Subtract a feature from a square.
fn sub_from_all<const SIZE: usize, const WEIGHTS: usize>(
    input: &mut [i16; SIZE],
    delta: &[i16; WEIGHTS],
    offset_sub: usize,
) {
    let s_block = &delta[offset_sub..offset_sub + SIZE];
    for (i, d) in input.iter_mut().zip(s_block) {
        *i -= *d;
    }
}

/// Execute clipped relu on the partial activations,
/// and accumulate the result into a sum.
pub fn crelu_flatten(
    us: &[i16; LAYER_1_SIZE],
    them: &[i16; LAYER_1_SIZE],
    weights: &[i16; LAYER_1_SIZE * 2],
) -> i32 {
    let mut sum: i32 = 0;
    for (&i, &w) in us.iter().zip(weights) {
        sum += i32::from(i.clamp(CR_MIN, CR_MAX)) * i32::from(w);
    }
    for (&i, &w) in them.iter().zip(&weights[LAYER_1_SIZE..]) {
        sum += i32::from(i.clamp(CR_MIN, CR_MAX)) * i32::from(w);
    }
    sum
}

/// Load a network from a JSON file, and emit binary files.
pub fn convert_json_to_binary(
    json_path: impl AsRef<std::path::Path>,
    output_path: impl AsRef<std::path::Path>,
) {
    let nnue = NNUEParams::from_json(json_path);
    let bytes = nnue.to_bytes();
    fs::create_dir(&output_path).unwrap();
    for (fname, byte_vector) in
        ["flipped_weights", "feature_bias", "output_weights", "output_bias"].into_iter().zip(&bytes)
    {
        let mut f =
            fs::File::create(output_path.as_ref().join(fname).with_extension("bin")).unwrap();
        std::io::Write::write_all(&mut f, byte_vector).unwrap();
    }
}

mod tests {
    #[test]
    fn pov_preserved() {
        crate::magic::initialise();
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
        crate::magic::initialise();
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
        crate::magic::initialise();
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
        crate::magic::initialise();
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
                    eprintln!("{i}: {} != {}", initial_white[i], t.nnue.white_pov[i]);
                }
                if initial_black[i] != t.nnue.black_pov[i] {
                    eprintln!("{i}: {} != {}", initial_black[i], t.nnue.black_pov[i]);
                }
            }
            assert_eq!(initial_white, t.nnue.white_pov);
            assert_eq!(initial_black, t.nnue.black_pov);
        }
    }
}
