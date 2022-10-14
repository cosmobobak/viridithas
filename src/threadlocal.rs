use crate::{chessmove::Move, definitions::MAX_PLY, nnue};

pub struct ThreadData {
    pub evals: [i32; MAX_PLY],
    pub excluded: [Move; MAX_PLY],
    pub nnue: Box<nnue::NNUEState>,
    pub root_moves: Vec<(Move, u64)>,
}

impl ThreadData {
    pub fn new() -> Self {
        Self {
            evals: [0; MAX_PLY],
            excluded: [Move::NULL; MAX_PLY],
            nnue: Box::new(nnue::NNUEState::new()),
            root_moves: Vec::new(),
        }
    }

    pub fn order_root_moves(&mut self, nodecounts: &[(Move, u64)]) {
        // println!("len of nodecounts: {}", nodecounts.len());
        let nodecount = |m| nodecounts.iter().find(|(m2, _)| *m2 == m).map(|(_, n)| *n);
        // moves with the highest nodecounts at the front, please.
        for entry in &mut self.root_moves {
            let nc = nodecount(entry.0);
            entry.1 = if let Some(nc) = nc {
                if nc == 0 {
                    entry.1
                } else {
                    nc
                }
            } else {
                entry.1
            };
        }
        self.root_moves.sort_by_key(|b| std::cmp::Reverse(b.1));
        // for entry in &self.root_moves {
        //     print!("{}: nc {}, ", entry.0, entry.1);
        // }
        // println!();
    }
}
