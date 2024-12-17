use crate::{chessmove::Move, piece::PieceType};

#[derive(Default, Clone)]
#[allow(clippy::module_name_repetitions)]
pub struct StackEntry {
    pub eval: i32,
    pub excluded: Option<Move>,
    pub played: Option<(PieceType, Move)>,
    pub dextensions: i32,
    pub ttpv: bool,
}
