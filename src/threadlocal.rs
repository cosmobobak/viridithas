use crate::{nnue, chessmove::Move, definitions::MAX_PLY};

pub struct ThreadData {
    pub evals: [i32; MAX_PLY],
    pub excluded: [Move; MAX_PLY],
    pub nnue: Box<nnue::NNUEState>,
    pub root_legal_moves: Vec<Move>,
}

impl ThreadData {
    pub fn new() -> Self {
        Self { 
            evals: [0; MAX_PLY], 
            excluded: [Move::NULL; MAX_PLY],
            nnue: Box::new(nnue::NNUEState::new()),
            root_legal_moves: Vec::new(),
        }
    }
}