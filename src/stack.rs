use arrayvec::ArrayVec;

use crate::chess::{board::movegen::MAX_POSITION_MOVES, chessmove::Move, types::ContHistIndex};

/// An out-of-line explicit stack frame, permitting the search
/// to reach up and down the callstack for information on what
/// parent and child nodes are doing.
#[derive(Default, Clone)]
#[allow(clippy::module_name_repetitions)]
pub struct StackFrame {
    pub static_eval: i32,
    pub eval: i32,
    pub excluded: Option<Move>,
    pub best_move: Option<Move>,
    pub searching: Option<Move>,
    pub searching_tactical: bool,
    pub dextensions: i32,
    pub ttpv: bool,
    pub ch_idx: ContHistIndex,
    pub reduction: i32,
    pub quiets_tried: ArrayVec<Move, MAX_POSITION_MOVES>,
    pub tacticals_tried: ArrayVec<Move, 32>,
}
