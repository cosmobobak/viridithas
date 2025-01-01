use crate::chess::chessmove::Move;

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
}
