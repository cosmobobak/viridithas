// use crate::{board::Board, piece::PieceType};

use super::network::Align64;

/// Activations of the hidden layer.
#[derive(Debug, Clone, Copy)]
pub struct Accumulator<const HIDDEN: usize> {
    pub white: Align64<[i16; HIDDEN]>,
    pub black: Align64<[i16; HIDDEN]>,
}

impl<const HIDDEN: usize> Accumulator<HIDDEN> {
    /// Initializes the accumulator with the given bias.
    pub fn init(&mut self, white_bias: &Align64<[i16; HIDDEN]>, black_bias: &Align64<[i16; HIDDEN]>) {
        self.white = *white_bias;
        self.black = *black_bias;
    }
}
