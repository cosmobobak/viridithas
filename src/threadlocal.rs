use std::array;

use crate::{
    chess::board::Board,
    chess::chessmove::Move,
    chess::piece::Colour,
    historytable::{
        CaptureHistoryTable, CorrectionHistoryTable, DoubleHistoryTable, MoveTable,
        ThreatsHistoryTable,
    },
    nnue::{self, network::NNUEParams},
    search::pv::PVariation,
    stack::StackEntry,
    transpositiontable::TTView,
    util::MAX_PLY,
};

#[repr(align(64))] // these get stuck in a vec and each thread accesses its own index
pub struct ThreadData<'a> {
    // stack array is right-padded by one because singular verification
    // will try to access the next ply in an edge case.
    pub ss: [StackEntry; MAX_PLY + 1],
    pub banned_nmp: u8,
    pub nnue: Box<nnue::network::NNUEState>,
    pub nnue_params: &'a NNUEParams,

    pub main_history: ThreatsHistoryTable,
    pub tactical_history: Box<CaptureHistoryTable>,
    pub continuation_history: Box<DoubleHistoryTable>,
    pub killer_move_table: [[Option<Move>; 2]; MAX_PLY + 1],
    pub counter_move_table: MoveTable,
    pub pawn_corrhist: Box<CorrectionHistoryTable>,
    pub nonpawn_corrhist: [Box<CorrectionHistoryTable>; 2],
    pub major_corrhist: Box<CorrectionHistoryTable>,
    pub minor_corrhist: Box<CorrectionHistoryTable>,

    pub thread_id: usize,

    pub pvs: [PVariation; MAX_PLY],
    pub completed: usize,
    pub depth: usize,

    pub stm_at_root: Colour,

    pub tt: TTView<'a>,
}

impl<'a> ThreadData<'a> {
    const WHITE_BANNED_NMP: u8 = 0b01;
    const BLACK_BANNED_NMP: u8 = 0b10;
    const ARRAY_REPEAT_VALUE: PVariation = PVariation::default_const();
    const EMPTY_PV_TABLE: [PVariation; MAX_PLY] = [Self::ARRAY_REPEAT_VALUE; MAX_PLY];

    pub fn new(
        thread_id: usize,
        board: &Board,
        tt: TTView<'a>,
        nnue_params: &'a NNUEParams,
    ) -> Self {
        let mut td = Self {
            ss: array::from_fn(|_| StackEntry::default()),
            banned_nmp: 0,
            nnue: nnue::network::NNUEState::new(board, nnue_params),
            nnue_params,
            main_history: ThreatsHistoryTable::new(),
            tactical_history: CaptureHistoryTable::boxed(),
            continuation_history: DoubleHistoryTable::boxed(),
            killer_move_table: [[None; 2]; MAX_PLY + 1],
            counter_move_table: MoveTable::new(),
            pawn_corrhist: CorrectionHistoryTable::boxed(),
            nonpawn_corrhist: [
                CorrectionHistoryTable::boxed(),
                CorrectionHistoryTable::boxed(),
            ],
            major_corrhist: CorrectionHistoryTable::boxed(),
            minor_corrhist: CorrectionHistoryTable::boxed(),
            thread_id,
            pvs: Self::EMPTY_PV_TABLE,
            completed: 0,
            depth: 0,
            stm_at_root: board.turn(),
            tt,
        };

        td.clear_tables();

        td
    }

    pub fn ban_nmp_for(&mut self, colour: Colour) {
        self.banned_nmp |= if colour == Colour::White {
            Self::WHITE_BANNED_NMP
        } else {
            Self::BLACK_BANNED_NMP
        };
    }

    pub fn unban_nmp_for(&mut self, colour: Colour) {
        self.banned_nmp &= if colour == Colour::White {
            !Self::WHITE_BANNED_NMP
        } else {
            !Self::BLACK_BANNED_NMP
        };
    }

    pub fn nmp_banned_for(&self, colour: Colour) -> bool {
        self.banned_nmp
            & if colour == Colour::White {
                Self::WHITE_BANNED_NMP
            } else {
                Self::BLACK_BANNED_NMP
            }
            != 0
    }

    pub fn clear_tables(&mut self) {
        self.main_history.clear();
        self.tactical_history.clear();
        self.continuation_history.clear();
        self.pawn_corrhist.clear();
        self.nonpawn_corrhist[Colour::White].clear();
        self.nonpawn_corrhist[Colour::Black].clear();
        self.major_corrhist.clear();
        self.minor_corrhist.clear();
        self.killer_move_table.fill([None; 2]);
        self.counter_move_table.clear();
        self.depth = 0;
        self.completed = 0;
        self.pvs = Self::EMPTY_PV_TABLE;
    }

    pub fn set_up_for_search(&mut self, board: &Board) {
        self.main_history.age_entries();
        self.tactical_history.age_entries();
        self.continuation_history.age_entries();
        self.killer_move_table.fill([None; 2]);
        self.counter_move_table.clear();
        self.depth = 0;
        self.completed = 0;
        self.pvs = Self::EMPTY_PV_TABLE;
        self.nnue.reinit_from(board, self.nnue_params);
        self.stm_at_root = board.turn();
    }

    pub fn update_best_line(&mut self, pv: &PVariation) {
        self.completed = self.depth;
        self.pvs[self.depth] = pv.clone();
    }

    pub fn revert_best_line(&mut self) {
        self.completed = self.depth - 1;
    }

    pub const fn pv(&self) -> &PVariation {
        &self.pvs[self.completed]
    }
}
