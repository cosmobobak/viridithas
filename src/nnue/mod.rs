use std::{fs::File, path, io::BufReader};

use serde_json::Value;

use crate::{board::{Board, movegen::BitLoop}, definitions::{flip_rank, BLACK, WHITE, PAWN, KING}};

use self::accumulator::BasicAccumulator;

pub mod convert;
mod accumulator;

const INPUT: usize = 768;
const HIDDEN: usize = 256;
const OUTPUT: usize = 1;
const CR_MIN: i16 = 0;
const CR_MAX: i16 = QA as i16;
const SCALE: i32 = 400;

const QA: i32 = 255;
const QB: i32 = 64;
const QAB: i32 = QA * QB;

const ACC_STACK_SIZE: usize = 256;

const ACTIVATE: bool = true;
const DEACTIVATE: bool = false;

#[derive(Debug)]
pub struct BasicNNUE {
    feature_weights: [i16; INPUT * HIDDEN],
    flipped_weights: [i16; INPUT * HIDDEN],
    feature_bias: [i16; HIDDEN],
    output_weights: [i16; HIDDEN * 2 * OUTPUT],
    output_bias: [i16; OUTPUT],

    white_pov: [i16; INPUT],
    black_pov: [i16; INPUT],

    accumulators: [BasicAccumulator<HIDDEN>; ACC_STACK_SIZE],
    current_acc: usize,

    output: [i32; OUTPUT],
}

impl BasicNNUE {
    pub fn push_acc(&mut self) {
        self.accumulators[self.current_acc + 1] = self.accumulators[self.current_acc];
        self.current_acc += 1;
    }

    pub fn pop_acc(&mut self) {
        self.current_acc -= 1;
    }

    pub fn reset_acc(&mut self) {
        self.current_acc = 0;
    }

    pub fn refresh_acc(&mut self, board: &Board) {
        const COLOUR_STRIDE: usize = 64 * 6;
        const PIECE_STRIDE: usize = 64;
        
        self.white_pov.fill(0);
        self.black_pov.fill(0);

        self.accumulators[self.current_acc].zero_out();

        for colour in [WHITE, BLACK] {
            for piece_type in PAWN..=KING {
                let piece = piece_type + 6 * colour;
                let piece_bb = board.pieces.piece_bb(piece);

                for sq in BitLoop::new(piece_bb) {
                    self.efficiently_update_manual::<ACTIVATE>(piece_type, colour, sq);
                }
            }
        }
    }

    pub fn efficiently_update_from_move(&mut self, piece: u8, colour: u8, from: u8, to: u8) {
        const COLOUR_STRIDE: usize = 64 * 6;
        const PIECE_STRIDE: usize = 64;
        let from = from as usize;
        let to = to as usize;
        let piece = piece as usize - 1; // shift into correct range.
        let colour = colour as usize;

        let white_idx_from = colour * COLOUR_STRIDE + piece * PIECE_STRIDE + from;
        let black_idx_from = (1 ^ colour) * COLOUR_STRIDE + piece * PIECE_STRIDE + flip_rank(from as u8) as usize;
        let white_idx_to = colour * COLOUR_STRIDE + piece * PIECE_STRIDE + to;
        let black_idx_to = (1 ^ colour) * COLOUR_STRIDE + piece * PIECE_STRIDE + flip_rank(to as u8) as usize;

        let acc = &mut self.accumulators[self.current_acc];

        self.white_pov[white_idx_from] = 0;
        self.black_pov[black_idx_from] = 0;
        self.white_pov[white_idx_to] = 1;
        self.black_pov[black_idx_to] = 1;

        subtract_and_add_to_all(
            &mut acc.white, 
            &self.flipped_weights,
            white_idx_from * HIDDEN,
            white_idx_to * HIDDEN,
        );
        subtract_and_add_to_all(
            &mut acc.black, 
            &self.flipped_weights,
            black_idx_from * HIDDEN,
            black_idx_to * HIDDEN,
        );
    }

    fn efficiently_update_manual<const IS_ACTIVATE: bool>(&mut self, piece: u8, colour: u8, sq: u8) {
        const COLOUR_STRIDE: usize = 64 * 6;
        const PIECE_STRIDE: usize = 64;
        let sq = sq as usize;
        let piece = piece as usize - 1; // shift into correct range.
        let colour = colour as usize;
        
        let white_idx = colour * COLOUR_STRIDE + piece * PIECE_STRIDE + sq as usize;
        let black_idx = (1 ^ colour) * COLOUR_STRIDE + piece * PIECE_STRIDE + flip_rank(sq as u8) as usize;

        let acc = &mut self.accumulators[self.current_acc];

        if IS_ACTIVATE {
            self.white_pov[white_idx] = 1;
            self.black_pov[black_idx] = 1;
            add_to_all(&mut acc.white, &self.flipped_weights, white_idx * HIDDEN);
            add_to_all(&mut acc.black, &self.flipped_weights, black_idx * HIDDEN);
        } else {
            self.white_pov[white_idx] = 0;
            self.black_pov[black_idx] = 0;
            sub_from_all(&mut acc.white, &self.flipped_weights, white_idx * HIDDEN);
            sub_from_all(&mut acc.black, &self.flipped_weights, black_idx * HIDDEN);
        }
    }

    pub fn evaluate(&mut self, stm: u8) -> i32 {
        let acc = &self.accumulators[self.current_acc];

        if stm == WHITE {
            clipped_relu_flatten_and_forward(
                &acc.white,
                &acc.black,
                &self.feature_bias,
                &self.output_weights,
                &mut self.output,
                CR_MIN,
                CR_MAX,
            );
        } else {
            clipped_relu_flatten_and_forward(
                &acc.black,
                &acc.white,
                &self.feature_bias,
                &self.output_weights,
                &mut self.output,
                CR_MIN,
                CR_MAX,
            );
        }

        (self.output[0] + i32::from(self.output_bias[0])) * SCALE / QAB
    }

    pub fn from_json(path: &str) -> Box<Self> {
        fn weight(weight_relation: &Value, weight_array: &mut [i16], stride: usize, k: i32, flip: bool) {
            let mut i = 0;
            for output in weight_relation.as_array().unwrap() {
                let mut j = 0;
                for weight in output.as_array().unwrap() {
                    let index = if flip {
                        (j * stride + i) as usize
                    } else {
                        (i * stride + j) as usize
                    };
                    let value = weight.as_f64().unwrap();
                    weight_array[index] = (value * k as f64) as i16;
                    j += 1;
                }
                i += 1;
            }
        }

        fn bias(bias_relation: &Value, bias_array: &mut [i16], k: i32) {
            let mut i = 0;
            for bias in bias_relation.as_array().unwrap() {
                let value = bias.as_f64().unwrap();
                bias_array[i] = (value * k as f64) as i16;
                i += 1;
            }
        }

        let mut out = Box::new(Self {
            feature_weights: [0; INPUT * HIDDEN],
            flipped_weights: [0; INPUT * HIDDEN],
            feature_bias: [0; HIDDEN],
            output_weights: [0; HIDDEN * 2 * OUTPUT],
            output_bias: [0; OUTPUT],
            white_pov: [0; INPUT],
            black_pov: [0; INPUT],
            accumulators: [BasicAccumulator::new(); ACC_STACK_SIZE],
            current_acc: 0,
            output: [0; OUTPUT],
        });

        let file = File::open(path).unwrap();
        let reader = BufReader::new(file);
        let json: Value = serde_json::from_reader(reader).unwrap();

        for property in json.as_object().unwrap() {
            match property.0.as_str() {
                "ft.weight" => {
                    weight(property.1, &mut out.feature_weights, INPUT, QA, false);
                    weight(property.1, &mut out.flipped_weights, HIDDEN, QA, true);
                    println!("feature weights loaded");
                }
                "ft.bias" => {
                    bias(property.1, &mut out.feature_bias, QA);
                    println!("feature bias loaded");
                }
                "out.weight" => {
                    weight(property.1, &mut out.output_weights, HIDDEN * 2, QB, false);
                    println!("output weights loaded");
                }
                "out.bias" => {
                    bias(property.1, &mut out.output_bias, QAB);
                    println!("output bias loaded");
                }
                _ => {}
            }
        }

        println!("nnue loaded");

        out
    }
}

fn subtract_and_add_to_all<const SIZE: usize, const WEIGHTS: usize>(
    input: &mut [i16; SIZE],
    delta: &[i16; WEIGHTS],
    offset_sub: usize,
    offset_add: usize,
) {
    for i in 0..SIZE {
        input[i] = input[i] - delta[offset_sub + i] + delta[offset_add + i];
    }
}

fn add_to_all<const SIZE: usize, const WEIGHTS: usize>(
    input: &mut [i16; SIZE],
    delta: &[i16; WEIGHTS],
    offset_add: usize,
) {
    for i in 0..SIZE {
        input[i] = input[i] + delta[offset_add + i];
    }
}

fn sub_from_all<const SIZE: usize, const WEIGHTS: usize>(
    input: &mut [i16; SIZE],
    delta: &[i16; WEIGHTS],
    offset_sub: usize,
) {
    for i in 0..SIZE {
        input[i] = input[i] - delta[offset_sub + i];
    }
}

pub fn clipped_relu_flatten_and_forward<
    const SIZE: usize, 
    const BIAS: usize, 
    const WEIGHTS: usize,
    const OUTPUT: usize,
>(
    input_us: &[i16; SIZE],
    input_them: &[i16; SIZE],
    bias: &[i16; BIAS],
    weights: &[i16; WEIGHTS],
    output: &mut [i32; OUTPUT],
    min: i16,
    max: i16,
) {
    for i in 0..OUTPUT {
        let mut sum: i32 = 0;
        for j in 0..SIZE {
            let input = input_us;
            let r_idx = j;
            sum += i32::from((input[r_idx] + bias[i]).clamp(min, max)) * i32::from(weights[j]);
        }
        for j in SIZE..(SIZE * 2) {
            let input = input_them;
            let r_idx = j - SIZE;
            sum += i32::from((input[r_idx] + bias[i]).clamp(min, max)) * i32::from(weights[j]);
        }
        output[i] = sum;
    }
}

pub static mut GLOBAL_NNUE: Option<Box<BasicNNUE>> = None;

pub fn initialise() {
    unsafe {
        GLOBAL_NNUE = Some(BasicNNUE::from_json("..\\marlinflow\\trainer\\nn\\sn0016_80.json"));
    }
}

pub fn evaluate(board: &Board) -> i32 {
    unsafe {
        let nnue = GLOBAL_NNUE.as_mut().unwrap();
        nnue.refresh_acc(board);
        nnue.evaluate(board.turn())
    }
}