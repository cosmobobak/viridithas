use std::array;

use crate::{
    board::Board,
    chessmove::Move,
    historytable::{CaptureHistoryTable, DoubleHistoryTable, MoveTable, ThreatsHistoryTable},
    nnue,
    piece::Colour,
    search::pv::PVariation,
    stack::StackEntry,
    transpositiontable::TTView,
    util::MAX_PLY,
};

#[derive(Clone)]
#[repr(align(64))] // these get stuck in a vec and each thread accesses its own index
pub struct ThreadData<'a> {
    // stack array is right-padded by one because singular verification
    // will try to access the next ply in an edge case.
    pub ss: [StackEntry; MAX_PLY + 1],
    pub banned_nmp: u8,
    pub multi_pv_excluded: Vec<Move>,
    pub nnue: Box<nnue::network::NNUEState>,

    pub main_history: ThreatsHistoryTable,
    pub tactical_history: Box<CaptureHistoryTable>,
    pub cont_hists: [Box<DoubleHistoryTable>; 2],
    pub killer_move_table: [[Option<Move>; 2]; MAX_PLY + 1],
    pub counter_move_table: MoveTable,

    pub thread_id: usize,

    pub pvs: Vec<PVariation>,
    pub completed: usize,
    pub depth: usize,

    pub stm_at_root: Colour,

    pub tt: TTView<'a>,
}

impl<'a> ThreadData<'a> {
    const WHITE_BANNED_NMP: u8 = 0b01;
    const BLACK_BANNED_NMP: u8 = 0b10;

    pub fn new(thread_id: usize, board: &Board, tt: TTView<'a>) -> Self {
        let mut td = Self {
            ss: array::from_fn(|_| StackEntry::default()),
            banned_nmp: 0,
            multi_pv_excluded: Vec::new(),
            nnue: nnue::network::NNUEState::new(board),
            main_history: ThreatsHistoryTable::new(),
            tactical_history: CaptureHistoryTable::boxed(),
            cont_hists: [(); 2].map(|()| DoubleHistoryTable::boxed()),
            killer_move_table: [[None; 2]; MAX_PLY + 1],
            counter_move_table: MoveTable::new(),
            thread_id,
            pvs: vec![PVariation::default(); MAX_PLY],
            completed: 0,
            depth: 0,
            stm_at_root: board.turn(),
            tt,
        };

        td.clear_tables();

        td
    }

    pub fn ban_nmp_for(&mut self, colour: Colour) {
        self.banned_nmp |= if colour == Colour::White { Self::WHITE_BANNED_NMP } else { Self::BLACK_BANNED_NMP };
    }

    pub fn unban_nmp_for(&mut self, colour: Colour) {
        self.banned_nmp &= if colour == Colour::White { !Self::WHITE_BANNED_NMP } else { !Self::BLACK_BANNED_NMP };
    }

    pub fn nmp_banned_for(&self, colour: Colour) -> bool {
        self.banned_nmp & if colour == Colour::White { Self::WHITE_BANNED_NMP } else { Self::BLACK_BANNED_NMP } != 0
    }

    pub fn clear_tables(&mut self) {
        self.main_history.clear();
        self.tactical_history.clear();
        self.cont_hists.iter_mut().for_each(|h| h.clear());
        self.killer_move_table.fill([None; 2]);
        self.counter_move_table.clear();
        self.depth = 0;
        self.completed = 0;
        self.pvs.fill(PVariation::default());
    }

    pub fn set_up_for_search(&mut self, board: &Board) {
        self.main_history.age_entries();
        self.tactical_history.age_entries();
        self.cont_hists.iter_mut().for_each(|h| h.age_entries());
        self.killer_move_table.fill([None; 2]);
        self.counter_move_table.clear();
        self.depth = 0;
        self.completed = 0;
        self.pvs.fill(PVariation::default());
        self.nnue.reinit_from(board);
        self.stm_at_root = board.turn();
    }

    pub fn update_best_line(&mut self, pv: &PVariation) {
        self.completed = self.depth;
        self.pvs[self.depth] = pv.clone();
    }

    pub fn revert_best_line(&mut self) {
        self.completed = self.depth - 1;
    }

    pub fn pv(&self) -> &PVariation {
        &self.pvs[self.completed]
    }
}
