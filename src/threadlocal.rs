use std::{
    array,
    sync::atomic::{AtomicBool, AtomicU64},
};

use anyhow::Context;
use arrayvec::ArrayVec;
use vec1::Vec1;

use crate::{
    chess::{board::Board, chessmove::Move, piece::Colour},
    historytable::{
        CaptureHistoryTable, CorrectionHistoryTable, DoubleHistoryTable, FromToTable,
        HashHistoryTable, PieceToTable, ThreatsHistoryTable,
    },
    nnue::{self, network::NNUEParams},
    search::pv::PVariation,
    searchinfo::{Control, SearchInfo},
    stack::StackFrame,
    threadpool::{self, ScopeExt},
    transpositiontable::CacheView,
    util::{MAX_DEPTH, VALUE_NONE},
};

pub struct Histories {
    pub piece_to: Box<ThreatsHistoryTable<PieceToTable>>,
    pub from_to: Box<ThreatsHistoryTable<FromToTable>>,
    pub tactical: Box<CaptureHistoryTable>,
    pub continuation: Box<DoubleHistoryTable>,
    pub pawn: Box<HashHistoryTable>,
}

impl Histories {
    pub fn new() -> Self {
        Self {
            piece_to: ThreatsHistoryTable::boxed(),
            from_to: ThreatsHistoryTable::boxed(),
            tactical: CaptureHistoryTable::boxed(),
            continuation: DoubleHistoryTable::boxed(),
            pawn: HashHistoryTable::boxed(),
        }
    }

    pub fn clear(&mut self) {
        self.piece_to.clear();
        self.from_to.clear();
        self.tactical.clear();
        self.continuation.clear();
        self.pawn.clear();
    }
}

#[repr(align(64))]
pub struct ThreadData<'a> {
    // stack array is right-padded by one because singular verification
    // will try to access the next ply in an edge case.
    pub ss: [StackFrame; MAX_DEPTH + 1],
    pub banned_nmp: u8,
    pub nnue: Box<nnue::network::NNUEState>,
    pub nnue_params: &'static NNUEParams,

    pub histories: Histories,
    pub killer_move_table: [Option<Move>; MAX_DEPTH + 1],
    pub pawn_corrhist: Box<CorrectionHistoryTable>,
    pub nonpawn_corrhist: [Box<CorrectionHistoryTable>; 2],
    pub major_corrhist: Box<CorrectionHistoryTable>,
    pub minor_corrhist: Box<CorrectionHistoryTable>,
    pub cont_corrhist: Box<CorrectionHistoryTable>,

    pub thread_id: usize,

    /// principal variations, indexed by ID iteration.
    pub pvs: Vec<PVariation>,
    /// evaluations, indexed by ID iteration.
    pub scores: Vec<i32>,
    /// the iterative deepening loop counter
    pub iteration: usize,
    /// the highest finished ID iteration
    pub completed: usize,
    /// the draft we're actually kicking off searches at
    pub root_depth: i32,

    /// scratch space for value-at-root
    pub score_scratch: i32,
    /// scratch space for PVs as they move up/down the stack.
    pub pv_scratch: Vec<PVariation>,

    pub stm_at_root: Colour,
    pub optimism: [i32; 2],

    pub cache: CacheView<'a>,

    pub board: Board,
    pub info: SearchInfo<'a>,
}

impl<'a> ThreadData<'a> {
    const WHITE_BANNED_NMP: u8 = 0b01;
    const BLACK_BANNED_NMP: u8 = 0b10;

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        thread_id: usize,
        board: Board,
        cache: CacheView<'a>,
        nnue_params: &'static NNUEParams,
        stopped: &'a AtomicBool,
        nodes: &'a AtomicU64,
        tbhits: &'a AtomicU64,
        control: &'a Control,
    ) -> Self {
        let mut td = Self {
            ss: array::from_fn(|_| StackFrame::default()),
            banned_nmp: 0,
            nnue: nnue::network::NNUEState::new(&board, nnue_params),
            nnue_params,
            histories: Histories::new(),
            killer_move_table: [None; MAX_DEPTH + 1],
            pawn_corrhist: CorrectionHistoryTable::boxed(),
            nonpawn_corrhist: [
                CorrectionHistoryTable::boxed(),
                CorrectionHistoryTable::boxed(),
            ],
            major_corrhist: CorrectionHistoryTable::boxed(),
            minor_corrhist: CorrectionHistoryTable::boxed(),
            cont_corrhist: CorrectionHistoryTable::boxed(),
            thread_id,
            pvs: vec![
                PVariation {
                    moves: ArrayVec::new_const(),
                };
                MAX_DEPTH
            ],
            scores: vec![VALUE_NONE; MAX_DEPTH],
            iteration: 0,
            completed: 0,
            root_depth: 0,
            score_scratch: VALUE_NONE,
            pv_scratch: vec![
                PVariation {
                    moves: ArrayVec::new_const(),
                };
                MAX_DEPTH + 1 // reaches forward by one when bootstrapping
            ],
            stm_at_root: board.turn(),
            optimism: [0; 2],
            cache,
            board,
            info: SearchInfo::new(stopped, nodes, tbhits, control),
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
        self.histories.clear();
        self.pawn_corrhist.clear();
        self.nonpawn_corrhist[Colour::White].clear();
        self.nonpawn_corrhist[Colour::Black].clear();
        self.major_corrhist.clear();
        self.minor_corrhist.clear();
        self.cont_corrhist.clear();
        self.killer_move_table.fill(None);
        self.root_depth = 0;
        self.completed = 0;
        self.pvs.fill_with(PVariation::new);
    }

    pub fn set_up_for_search(&mut self) {
        self.killer_move_table.fill(None);
        self.root_depth = 0;
        self.completed = 0;
        self.pvs.fill_with(PVariation::new);
        self.nnue.reïnit_from(&self.board, self.nnue_params);
        self.stm_at_root = self.board.turn();
    }

    pub fn update_best_line(&mut self) {
        self.completed = self.iteration;
        self.pvs[self.iteration] = self.pv_scratch[0].clone();
        self.scores[self.iteration] = self.score_scratch;
    }

    pub fn revert_best_line(&mut self) {
        self.completed = self.iteration - 1;
    }

    pub fn pv(&self) -> &PVariation {
        &self.pvs[self.completed]
    }

    pub fn score(&self) -> i32 {
        self.scores[self.completed]
    }
}

#[allow(clippy::too_many_arguments)]
pub fn make_thread_data<'a>(
    pos: &Board,
    cache: CacheView<'a>,
    nnue_params: &'static NNUEParams,
    stopped: &'a AtomicBool,
    nodes: &'a AtomicU64,
    tbhits: &'a AtomicU64,
    control: &'a Control,
    worker_threads: &[threadpool::WorkerThread],
) -> anyhow::Result<Vec1<Box<ThreadData<'a>>>> {
    std::thread::scope(|s| -> anyhow::Result<Vec1<Box<ThreadData>>> {
        let handles = worker_threads
            .iter()
            .enumerate()
            .map(|(thread_id, worker)| {
                let (tx, rx) = std::sync::mpsc::channel();
                let join_handle = s.spawn_into(
                    move || {
                        #[allow(clippy::unwrap_used)]
                        tx.send(Box::new(ThreadData::new(
                            thread_id,
                            pos.clone(),
                            cache,
                            nnue_params,
                            stopped,
                            nodes,
                            tbhits,
                            control,
                        )))
                        .unwrap();
                    },
                    worker,
                );
                (rx, join_handle)
            })
            .collect::<Vec<_>>();

        let mut thread_data: Vec<Box<ThreadData>> = Vec::with_capacity(handles.len());
        for (rx, handle) in handles {
            let td = rx
                .recv()
                .with_context(|| "Failed to receive thread data from worker thread")?;
            thread_data.push(td);
            handle.join();
        }

        Ok(thread_data.try_into()?)
    })
}
