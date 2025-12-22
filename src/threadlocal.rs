use std::{
    array,
    sync::atomic::{AtomicBool, AtomicI16, AtomicU64, Ordering},
};

use anyhow::Context;

use crate::{
    chess::{
        board::Board,
        chessmove::Move,
        piece::Colour,
        types::{ContHistIndex, Keys},
    },
    historytable::{
        CORRECTION_HISTORY_MAX, CaptureHistoryTable, ContinuationCorrectionHistoryTable,
        CorrectionHistoryTable, DoubleHistoryTable, HashHistoryTable, ThreatsHistoryTable,
        update_correction,
    },
    nnue::{self, network::NNUEParams},
    search::{parameters::Config, pv::PVariation},
    searchinfo::SearchInfo,
    stack::StackEntry,
    threadpool::{self, ScopeExt},
    transpositiontable::TTView,
    util::MAX_DEPTH,
};

pub struct Corrhists {
    pub pawn: Box<CorrectionHistoryTable>,
    pub nonpawn: [Box<CorrectionHistoryTable>; 2],
    pub major: Box<CorrectionHistoryTable>,
    pub minor: Box<CorrectionHistoryTable>,
    pub continuation: Box<ContinuationCorrectionHistoryTable>,
}

impl Corrhists {
    pub fn new() -> Self {
        Self {
            pawn: CorrectionHistoryTable::boxed(),
            nonpawn: [
                CorrectionHistoryTable::boxed(),
                CorrectionHistoryTable::boxed(),
            ],
            major: CorrectionHistoryTable::boxed(),
            minor: CorrectionHistoryTable::boxed(),
            continuation: ContinuationCorrectionHistoryTable::boxed(),
        }
    }

    pub fn clear(&self) {
        self.pawn.clear();
        self.nonpawn[Colour::White].clear();
        self.nonpawn[Colour::Black].clear();
        self.major.clear();
        self.minor.clear();
        self.continuation.clear();
    }

    /// Update the correction history for a position.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn update(
        &self,
        keys: &Keys,
        us: Colour,
        cont_indices: Option<(ContHistIndex, ContHistIndex)>,
        depth: i32,
        tt_complexity: i32,
        diff: i32,
    ) {
        use Colour::{Black, White};

        // wow! floating point in a chess engine!
        let tt_complexity_factor =
            ((1.0 + (tt_complexity as f32 + 1.0).log2() / 10.0) * 8.0) as i32;

        let bonus = i32::clamp(
            diff * depth * tt_complexity_factor / 64,
            -CORRECTION_HISTORY_MAX / 4,
            CORRECTION_HISTORY_MAX / 4,
        );

        let pawn = self.pawn.get_ref(us, keys.pawn);
        let [nonpawn_white, nonpawn_black] = &self.nonpawn;
        let nonpawn_white = nonpawn_white.get_ref(us, keys.non_pawn[White]);
        let nonpawn_black = nonpawn_black.get_ref(us, keys.non_pawn[Black]);
        let minor = self.minor.get_ref(us, keys.minor);
        let major = self.major.get_ref(us, keys.major);

        let update = move |entry: &AtomicI16| {
            update_correction(entry, bonus);
        };

        update(pawn);
        update(nonpawn_white);
        update(nonpawn_black);
        update(minor);
        update(major);

        if let Some((ch1, ch2)) = cont_indices {
            let pt1 = ch1.piece.piece_type();
            let pt2 = ch2.piece.piece_type();
            update(&self.continuation[ch1.to][pt1][ch2.to][pt2][us]);
        }
    }

    /// Compute the correction history adjustment for a position.
    #[expect(clippy::cast_possible_truncation, clippy::similar_names)]
    pub fn correction(
        &self,
        keys: &Keys,
        us: Colour,
        cont_indices: Option<(ContHistIndex, ContHistIndex)>,
        conf: &Config,
    ) -> i32 {
        use Colour::{Black, White};

        let pawn = self.pawn.get(us, keys.pawn);
        let [white, black] = &self.nonpawn;
        let white = white.get(us, keys.non_pawn[White]);
        let black = black.get(us, keys.non_pawn[Black]);
        let minor = self.minor.get(us, keys.minor);
        let major = self.major.get(us, keys.major);

        let cont = if let Some((ch1, ch2)) = cont_indices {
            let pt1 = ch1.piece.piece_type();
            let pt2 = ch2.piece.piece_type();
            i64::from(self.continuation[ch1.to][pt1][ch2.to][pt2][us].load(Ordering::Relaxed))
        } else {
            0
        };

        let adjustment = pawn * i64::from(conf.pawn_corrhist_weight)
            + major * i64::from(conf.major_corrhist_weight)
            + minor * i64::from(conf.minor_corrhist_weight)
            + (white + black) * i64::from(conf.nonpawn_corrhist_weight)
            + cont * i64::from(conf.continuation_corrhist_weight);

        (adjustment * 12 / 0x40000) as i32
    }
}

#[repr(align(64))]
pub struct ThreadData<'a> {
    // stack array is right-padded by one because singular verification
    // will try to access the next ply in an edge case.
    pub ss: [StackEntry; MAX_DEPTH + 1],
    pub banned_nmp: u8,
    pub nnue: Box<nnue::network::NNUEState>,
    pub nnue_params: &'static NNUEParams,

    pub main_hist: ThreatsHistoryTable,
    pub tactical_hist: Box<CaptureHistoryTable>,
    pub cont_hist: Box<DoubleHistoryTable>,
    pub pawn_hist: Box<HashHistoryTable>,
    pub killer_move_table: [Option<Move>; MAX_DEPTH + 1],

    pub corrhists: &'a Corrhists,

    pub thread_id: usize,

    pub pvs: [PVariation; MAX_DEPTH],
    /// the iterative deepening loop counter
    pub iteration: usize,
    /// the highest finished ID iteration
    pub completed: usize,
    /// the draft we're actually kicking off searches at
    pub root_depth: i32,

    pub stm_at_root: Colour,
    pub optimism: [i32; 2],

    pub tt: TTView<'a>,

    pub board: Board,
    pub info: SearchInfo<'a>,
}

impl<'a> ThreadData<'a> {
    const WHITE_BANNED_NMP: u8 = 0b01;
    const BLACK_BANNED_NMP: u8 = 0b10;
    const ARRAY_REPEAT_VALUE: PVariation = PVariation::default_const();

    #[expect(clippy::too_many_arguments)]
    pub fn new(
        thread_id: usize,
        board: Board,
        tt: TTView<'a>,
        corrhists: &'a Corrhists,
        nnue_params: &'static NNUEParams,
        stopped: &'a AtomicBool,
        nodes: &'a AtomicU64,
        tbhits: &'a AtomicU64,
    ) -> Self {
        let mut td = Self {
            ss: array::from_fn(|_| StackEntry::default()),
            banned_nmp: 0,
            nnue: nnue::network::NNUEState::new(&board, nnue_params),
            nnue_params,
            main_hist: ThreatsHistoryTable::new(),
            tactical_hist: CaptureHistoryTable::boxed(),
            cont_hist: DoubleHistoryTable::boxed(),
            pawn_hist: HashHistoryTable::boxed(),
            killer_move_table: [None; MAX_DEPTH + 1],
            corrhists,
            thread_id,
            #[allow(clippy::large_stack_arrays)]
            pvs: [Self::ARRAY_REPEAT_VALUE; MAX_DEPTH],
            iteration: 0,
            completed: 0,
            root_depth: 0,
            stm_at_root: board.turn(),
            optimism: [0; 2],
            tt,
            board,
            info: SearchInfo::new(stopped, nodes, tbhits),
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
        self.main_hist.clear();
        self.tactical_hist.clear();
        self.cont_hist.clear();
        self.pawn_hist.clear();
        self.corrhists.clear();
        self.killer_move_table.fill(None);
        self.root_depth = 0;
        self.completed = 0;
        self.pvs.fill(Self::ARRAY_REPEAT_VALUE);
    }

    pub fn set_up_for_search(&mut self) {
        self.killer_move_table.fill(None);
        self.root_depth = 0;
        self.completed = 0;
        self.pvs.fill(Self::ARRAY_REPEAT_VALUE);
        self.nnue.reinit_from(&self.board, self.nnue_params);
        self.stm_at_root = self.board.turn();
    }

    pub fn update_best_line(&mut self, pv: &PVariation) {
        self.completed = self.iteration;
        self.pvs[self.iteration] = pv.clone();
    }

    pub fn revert_best_line(&mut self) {
        self.completed = self.iteration - 1;
    }

    pub const fn pv(&self) -> &PVariation {
        &self.pvs[self.completed]
    }

    pub const fn pv_mut(&mut self) -> &mut PVariation {
        &mut self.pvs[self.completed]
    }
}

#[expect(clippy::too_many_arguments)]
pub fn make_thread_data<'a>(
    pos: &Board,
    tt: TTView<'a>,
    corrhists: &'a Corrhists,
    nnue_params: &'static NNUEParams,
    stopped: &'a AtomicBool,
    nodes: &'a AtomicU64,
    tbhits: &'a AtomicU64,
    worker_threads: &[threadpool::WorkerThread],
) -> anyhow::Result<Vec<Box<ThreadData<'a>>>> {
    std::thread::scope(|s| -> anyhow::Result<Vec<Box<ThreadData>>> {
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
                            tt,
                            corrhists,
                            nnue_params,
                            stopped,
                            nodes,
                            tbhits,
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

        Ok(thread_data)
    })
}
