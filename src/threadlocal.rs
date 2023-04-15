use crate::{
    board::Board,
    chessmove::Move,
    definitions::{MAX_DEPTH, MAX_PLY},
    historytable::{DoubleHistoryTable, HistoryTable, MoveTable},
    nnue,
    piece::Colour,
    search::PVariation,
};

#[derive(Clone)]
pub struct ThreadData {
    pub evals: [i32; MAX_PLY],
    pub excluded: [Move; MAX_PLY],
    pub best_moves: [Move; MAX_PLY],
    pub double_extensions: [i32; MAX_PLY],
    pub checks: [bool; MAX_PLY],
    pub banned_nmp: u8,
    pub multi_pv_excluded: Vec<Move>,
    pub nnue: Box<nnue::network::NNUEState>,

    pub history_table: HistoryTable,
    pub followup_history: Box<DoubleHistoryTable>,
    pub counter_move_history: Box<DoubleHistoryTable>,
    pub killer_move_table: [[Move; 2]; MAX_DEPTH.ply_to_horizon()],
    pub counter_move_table: MoveTable,

    pub thread_id: usize,

    pub pvs: Vec<PVariation>,
    pub completed: usize,
    pub depth: usize,
}

impl ThreadData {
    const WHITE_BANNED_NMP: u8 = 0b01;
    const BLACK_BANNED_NMP: u8 = 0b10;

    pub fn new(thread_id: usize, board: &Board) -> Self {
        let mut td = Self {
            evals: [0; MAX_PLY],
            excluded: [Move::NULL; MAX_PLY],
            best_moves: [Move::NULL; MAX_PLY],
            double_extensions: [0; MAX_PLY],
            checks: [false; MAX_PLY],
            banned_nmp: 0,
            multi_pv_excluded: Vec::new(),
            nnue: nnue::network::NNUEState::new(board),
            history_table: HistoryTable::new(),
            followup_history: DoubleHistoryTable::boxed(),
            counter_move_history: DoubleHistoryTable::boxed(),
            killer_move_table: [[Move::NULL; 2]; MAX_PLY],
            counter_move_table: MoveTable::new(),
            thread_id,
            pvs: vec![PVariation::default(); MAX_PLY],
            completed: 0,
            depth: 0,
        };

        td.alloc_tables();

        td
    }

    pub fn ban_nmp_for(&mut self, colour: Colour) {
        self.banned_nmp |=
            if colour == Colour::WHITE { Self::WHITE_BANNED_NMP } else { Self::BLACK_BANNED_NMP };
    }

    pub fn unban_nmp_for(&mut self, colour: Colour) {
        self.banned_nmp &=
            if colour == Colour::WHITE { !Self::WHITE_BANNED_NMP } else { !Self::BLACK_BANNED_NMP };
    }

    pub fn nmp_banned_for(&self, colour: Colour) -> bool {
        self.banned_nmp
            & if colour == Colour::WHITE { Self::WHITE_BANNED_NMP } else { Self::BLACK_BANNED_NMP }
            != 0
    }

    fn alloc_tables(&mut self) {
        self.history_table.clear();
        self.followup_history.clear();
        self.counter_move_history.clear();
        self.killer_move_table.fill([Move::NULL; 2]);
        self.counter_move_table.clear();
        self.depth = 0;
        self.completed = 0;
        self.pvs.fill(PVariation::default());
    }

    pub fn setup_tables_for_search(&mut self) {
        self.history_table.age_entries();
        self.followup_history.age_entries();
        self.counter_move_history.age_entries();
        self.killer_move_table.fill([Move::NULL; 2]);
        self.counter_move_table.clear();
        self.depth = 0;
        self.completed = 0;
        self.pvs.fill(PVariation::default());
    }

    pub fn update_best_line(&mut self, pv: &PVariation) {
        self.completed = self.depth;
        self.pvs[self.depth] = pv.clone();
    }

    pub fn revert_best_line(&mut self) {
        self.completed = self.depth - 1;
    }

    pub fn pv_move(&self) -> Option<Move> {
        self.pvs[self.completed].moves().first().copied()
    }

    pub fn pv_score(&self) -> i32 {
        self.pvs[self.completed].score()
    }
}
