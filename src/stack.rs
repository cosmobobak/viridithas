use crate::chessmove::Move;

#[derive(Default, Clone)]
#[allow(clippy::module_name_repetitions)]
pub struct StackEntry {
    pub eval: i32,
    pub excluded: Option<Move>,
    pub best_move: Option<Move>,
    pub double_extensions: i32,
    pub ttpv: bool,
}