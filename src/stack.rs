use crate::chess::{chessmove::Move, types::ContHistIndex};

#[derive(Default, Clone)]
#[allow(clippy::module_name_repetitions)]
pub struct StackEntry {
    pub eval: i32,
    pub excluded: Option<Move>,
    pub best_move: Option<Move>,
    pub searching: Option<Move>,
    pub searching_tactical: bool,
    pub dextensions: i32,
    pub ttpv: bool,
    pub conthist_index: ContHistIndex,
    pub reduction: i32,
}
