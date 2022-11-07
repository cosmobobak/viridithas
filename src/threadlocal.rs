use crate::{chessmove::Move, definitions::MAX_PLY, nnue};

pub struct ThreadData {
    pub evals: [i32; MAX_PLY],
    pub excluded: [Move; MAX_PLY],
    pub nnue: Box<nnue::NNUEState>,
    pub use_nnue: bool,
}

impl ThreadData {
    pub fn new() -> Self {
        Self {
            evals: [0; MAX_PLY],
            excluded: [Move::NULL; MAX_PLY],
            nnue: Box::new(nnue::NNUEState::new()),
            use_nnue: true,
        }
    }
}
