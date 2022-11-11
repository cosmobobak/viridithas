use crate::{chessmove::Move, definitions::{MAX_PLY, MAX_DEPTH}, nnue, historytable::{HistoryTable, MoveTable, DoubleHistoryTable}};

pub struct ThreadData {
    pub evals: [i32; MAX_PLY],
    pub excluded: [Move; MAX_PLY],
    pub nnue: Box<nnue::NNUEState>,

    pub history_table: HistoryTable,
    pub killer_move_table: [[Move; 2]; MAX_DEPTH.ply_to_horizon()],
    pub counter_move_table: MoveTable,
    pub followup_history: DoubleHistoryTable,
}

impl ThreadData {
    pub fn new() -> Self {
        Self {
            evals: [0; MAX_PLY],
            excluded: [Move::NULL; MAX_PLY],
            nnue: Box::new(nnue::NNUEState::new()),
            history_table: HistoryTable::new(),
            killer_move_table: [[Move::NULL; 2]; MAX_DEPTH.ply_to_horizon()],
            counter_move_table: MoveTable::new(),
            followup_history: DoubleHistoryTable::new(),
        }
    }

    pub fn alloc_tables(&mut self) {
        self.history_table.clear();
        self.killer_move_table.fill([Move::NULL; 2]);
        self.counter_move_table.clear();
        self.followup_history.clear();
    }

    pub fn setup_tables_for_search(&mut self) {
        self.history_table.age_entries();
        self.followup_history.age_entries();
        self.killer_move_table.fill([Move::NULL; 2]);
        self.counter_move_table.clear();
    }
}
