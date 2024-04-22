// use crate::{board::Board, piece::PieceType};

use super::{network::{Align64, MovedPiece, PovUpdate, UpdateBuffer, LAYER_1_SIZE}, simd};

/// Activations of the hidden layer.
#[derive(Debug, Clone)]
pub struct Accumulator {
    pub white: Align64<[i16; LAYER_1_SIZE]>,
    pub black: Align64<[i16; LAYER_1_SIZE]>,

    pub mv: MovedPiece,
    pub update_buffer: UpdateBuffer,
    pub correct: [bool; 2],
}

impl Accumulator {
    /// Initializes the accumulator with the given bias.
    pub fn init(&mut self, bias: &Align64<[i16; LAYER_1_SIZE]>, update: PovUpdate) {
        if update.white {
            simd::copy(bias, &mut self.white);
        }
        if update.black {
            simd::copy(bias, &mut self.black);
        }
    }
}
