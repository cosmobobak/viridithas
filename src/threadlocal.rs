use crate::{
    chessmove::Move,
    definitions::{MAX_DEPTH, MAX_PLY},
    historytable::{DoubleHistoryTable, HistoryTable, MoveTable},
    nnue, piece::Colour,
};

#[derive(Clone)]
pub struct ThreadData {
    pub evals: [i32; MAX_PLY],
    pub excluded: [Move; MAX_PLY],
    pub best_moves: [Move; MAX_PLY],
    pub banned_nmp: u8,
    pub multi_pv_excluded: Vec<Move>,
    pub nnue: Box<nnue::NNUEState>,

    pub history_table: HistoryTable,
    pub followup_history: DoubleHistoryTable,
    pub killer_move_table: [[Move; 2]; MAX_DEPTH.ply_to_horizon()],
    pub counter_move_table: MoveTable,

    pub thread_id: usize,
}

impl ThreadData {
    const WHITE_BANNED_NMP: u8 = 0b01;
    const BLACK_BANNED_NMP: u8 = 0b10;

    pub fn new(thread_id: usize) -> Self {
        Self {
            evals: [0; MAX_PLY],
            excluded: [Move::NULL; MAX_PLY],
            best_moves: [Move::NULL; MAX_PLY],
            banned_nmp: 0,
            multi_pv_excluded: Vec::new(),
            nnue: Box::new(nnue::NNUEState::new()),
            history_table: HistoryTable::new(),
            followup_history: DoubleHistoryTable::new(),
            killer_move_table: [[Move::NULL; 2]; MAX_DEPTH.ply_to_horizon()],
            counter_move_table: MoveTable::new(),
            thread_id,
        }
    }

    pub fn ban_nmp_for(&mut self, colour: Colour) {
        self.banned_nmp |= if colour == Colour::WHITE {
            Self::WHITE_BANNED_NMP
        } else {
            Self::BLACK_BANNED_NMP
        };
    }

    pub fn unban_nmp_for(&mut self, colour: Colour) {
        self.banned_nmp &= if colour == Colour::WHITE {
            !Self::WHITE_BANNED_NMP
        } else {
            !Self::BLACK_BANNED_NMP
        };
    }

    pub fn nmp_banned_for(&self, colour: Colour) -> bool {
        self.banned_nmp & if colour == Colour::WHITE {
            Self::WHITE_BANNED_NMP
        } else {
            Self::BLACK_BANNED_NMP
        } != 0
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
}
