use crate::{chessmove::Move, definitions::{MAX_PLY, MAX_DEPTH}, nnue, historytable::{HistoryTable, MoveTable, DoubleHistoryTable}, board::movegen::movepicker::{ILLEGAL_MOVE_SCORE, TT_MOVE_SCORE}};

pub struct ThreadData {
    pub evals: [i32; MAX_PLY],
    pub excluded: [Move; MAX_PLY],
    pub do_nmp: bool,
    pub multi_pv_excluded: Vec<Move>,
    pub nnue: Box<nnue::NNUEState>,

    pub history_table: HistoryTable,
    pub killer_move_table: [[Move; 2]; MAX_DEPTH.ply_to_horizon()],
    pub counter_move_table: MoveTable,
    pub followup_history: DoubleHistoryTable,

    /// A record of the size of the game subtree under each root move
    /// in the last iteration of the search. This is used to order the
    /// root moves in the next iteration.
    pub root_move_ordering: Vec<(Move, i32)>,
    /// The previous bestmoves in shallower searches.
    /// idx[0] is the actual best move found last search,
    /// the rest are the previous bestmoves ordered by recency.
    pub previous_bestmoves: Vec<Move>,
}

impl ThreadData {
    pub fn new() -> Self {
        Self {
            evals: [0; MAX_PLY],
            excluded: [Move::NULL; MAX_PLY],
            do_nmp: true,
            multi_pv_excluded: Vec::new(),
            nnue: Box::new(nnue::NNUEState::new()),
            history_table: HistoryTable::new(),
            killer_move_table: [[Move::NULL; 2]; MAX_DEPTH.ply_to_horizon()],
            counter_move_table: MoveTable::new(),
            followup_history: DoubleHistoryTable::new(),
            root_move_ordering: Vec::new(),
            previous_bestmoves: Vec::new(),
        }
    }

    #[allow(dead_code)]
    pub fn add_multipv_excluded(&mut self, m: Move) {
        self.multi_pv_excluded.push(m);
    }

    #[allow(dead_code)]
    pub fn clear_multipv_excluded(&mut self) {
        self.multi_pv_excluded.clear();
    }

    pub fn alloc_tables(&mut self) {
        self.history_table.clear();
        self.followup_history.clear();
        self.killer_move_table.fill([Move::NULL; 2]);
        self.counter_move_table.clear();
    }

    pub fn setup_tables_for_search(&mut self) {
        self.history_table.age_entries();
        self.followup_history.age_entries();
        self.killer_move_table.fill([Move::NULL; 2]);
        self.counter_move_table.clear();
    }

    /// Get the score of a move at the root of the search.
    /// This is used to sort the moves at the root of the search,
    /// and allows a more sophisticated move ordering than just
    /// using SEE and history counters.
    pub fn score_at_root(&self, mov: Move) -> i32 {
        #![allow(clippy::option_if_let_else)]
        if let Some(index) = self.previous_bestmoves.iter().position(|m| *m == mov) {
            let index: i32 = index.try_into().unwrap();
            TT_MOVE_SCORE / (index + 1) // full score for index 0, 1/2 score for index 1, etc.
        } else if let Some(&(_, score)) = self.root_move_ordering.iter().find(|(m, _)| *m == mov) {
            score
        } else {
            ILLEGAL_MOVE_SCORE
        }
    }

    /// Bring a new best-move to the front of the root move order list.
    /// The order of moves after this is preserved, so for example if the list
    /// started as
    /// ```
    /// ["e2e4", "d2d4", "c2c4", "g1f3"]
    /// ```
    /// and we call `inject_new_best_move("c2c4")`, the list will become
    /// ```
    /// ["c2c4", "e2e4", "d2d4", "g1f3"]
    /// ```
    pub fn inject_new_best_move(&mut self, mov: Move) {
        if let Some(index) = self.previous_bestmoves.iter().position(|m| *m == mov) {
            self.previous_bestmoves[..=index].rotate_right(1);
        } else {
            self.previous_bestmoves.insert(0, mov);
        }
    }

    /// Record a subtree nodecount for a root move, to be used for
    /// root move ordering.
    pub fn record_subtree_nodecount(&mut self, mov: Move, nodecount: u64) {
        let adjusted_nodecount: i32 = (nodecount / 55).try_into().unwrap_or(i32::MAX);
        if let Some(index) = self.root_move_ordering.iter().position(|(m, _)| *m == mov) {
            let curr = self.root_move_ordering[index].1;
            self.root_move_ordering[index].1 = curr.saturating_add(adjusted_nodecount);
        } else {
            panic!("Tried to record subtree nodecount for a move that wasn't in the root move ordering list!")
        }
    }

    /// Add a legal root move to the root move ordering list.
    pub fn add_root_move(&mut self, mov: Move) {
        if !self.root_move_ordering.iter().any(|(m, _)| *m == mov) {
            self.root_move_ordering.push((mov, 0));
        }
    }
}

mod tests {
    #![allow(unused_imports)]
    use crate::{chessmove::Move, definitions::Square};
    use super::ThreadData;

    #[test]
    fn inject_best_move_reorder() {
        let mut td = ThreadData::new();
        let ml @ [e4, d4, c4, nf3] = [
            Move::new(Square::E2, Square::E4, 0, 0),
            Move::new(Square::D2, Square::D4, 0, 0),
            Move::new(Square::C2, Square::C4, 0, 0),
            Move::new(Square::G1, Square::F3, 0, 0),
        ];
        td.previous_bestmoves = ml.to_vec();
        td.inject_new_best_move(c4);
        assert_eq!(td.previous_bestmoves, [c4, e4, d4, nf3]);
    }

    #[test]
    fn inject_best_move_insert() {
        let mut td = ThreadData::new();
        let ml = [
            Move::new(Square::E2, Square::E4, 0, 0),
            Move::new(Square::D2, Square::D4, 0, 0),
            Move::new(Square::C2, Square::C4, 0, 0),
            Move::new(Square::G1, Square::F3, 0, 0),
        ];
        td.previous_bestmoves = ml.to_vec();
        let a1a2 = Move::new(Square::A1, Square::A2, 0, 0);
        td.inject_new_best_move(a1a2);
        assert_eq!(td.previous_bestmoves, [a1a2, ml[0], ml[1], ml[2], ml[3]]);
    }
}