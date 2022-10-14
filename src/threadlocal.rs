use crate::{nnue, chessmove::Move, definitions::MAX_PLY};

pub struct ThreadData {
    pub evals: [i32; MAX_PLY],
    pub excluded: [Move; MAX_PLY],
    pub nnue: Box<nnue::NNUEState>,
    pub root_legal_moves: Vec<(Move, u64)>,
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

    pub fn order_root_moves(&mut self, nodecounts: &[(Move, u64)]) {
        let nodecount = |m| nodecounts.iter().find(|(m2, _)| *m2 == m).map(|(_, n)| *n);
        // moves with the highest nodecounts at the front, please.
        for entry in &mut self.root_legal_moves {
            entry.1 = nodecount(entry.0).unwrap_or(entry.1);
        }
        self.root_legal_moves.sort_by_key(|b| std::cmp::Reverse(b.1));
    }
}