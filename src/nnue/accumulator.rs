// use crate::{board::Board, piece::PieceType};

use crate::piece::Colour;

use super::network::{Align64, MovedPiece, PovUpdate, UpdateBuffer, L1_SIZE};

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
