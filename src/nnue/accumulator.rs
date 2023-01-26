// use crate::{board::Board, piece::PieceType};

use super::Align;

#[derive(Debug, Clone, Copy)]
pub struct Accumulator<const HIDDEN: usize> {
    pub white: Align<[i16; HIDDEN]>,
    pub black: Align<[i16; HIDDEN]>,
}

impl<const HIDDEN: usize> Accumulator<HIDDEN> {
    pub const fn new() -> Self {
        Self { white: Align([0; HIDDEN]), black: Align([0; HIDDEN]) }
    }

    pub fn init(&mut self, bias: &[i16; HIDDEN]) {
        self.white.copy_from_slice(bias);
        self.black.copy_from_slice(bias);
    }
}

// #[allow(clippy::module_name_repetitions)]
// #[derive(Debug, Clone, Copy)]
// pub struct Accumulator2<const HIDDEN: usize> {
//     pub white: Align<[i16; HIDDEN]>,
//     pub black: Align<[i16; HIDDEN]>,
//     pub material: usize,
// }

// impl<const HIDDEN: usize> Accumulator2<HIDDEN> {
//     pub const fn new(board: &Board) -> Self {
//         Self {
//             white: Align([0; HIDDEN]),
//             black: Align([0; HIDDEN]),
//             material: board.num_pt(PieceType::PAWN) as usize
//                 + 3 * board.num_pt(PieceType::KNIGHT) as usize
//                 + 3 * board.num_pt(PieceType::BISHOP) as usize
//                 + 5 * board.num_pt(PieceType::ROOK) as usize
//                 + 8 * board.num_pt(PieceType::QUEEN) as usize,
//         }
//     }

//     pub fn init(&mut self, bias: &[i16; HIDDEN]) {
//         self.white.copy_from_slice(bias);
//         self.black.copy_from_slice(bias);
//     }
// }