use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    mpsc, Mutex,
};

use crate::{
    board::evaluation::parameters::EvalParams,
    search::{parameters::SearchParams, LMTable},
    timemgmt::{SearchLimit, TimeManager},
    uci,
    util::{
        depth::{Depth, ZERO_PLY},
        BatchedAtomicCounter,
    },
};

#[cfg(feature = "stats")]
use crate::board::movegen::MAX_POSITION_MOVES;

#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Debug)]
pub struct SearchInfo<'a> {
    /// The number of nodes searched.
    pub nodes: BatchedAtomicCounter<'a>,
    /// A table storing the number of nodes under the root move(s).
    pub root_move_nodes: [[u64; 64]; 64], // [from][to]
    /// Signal to stop the search.
    pub stopped: &'a AtomicBool,
    /// The highest depth reached (selective depth).
    pub seldepth: Depth,
    /// A handle to a receiver for stdin.
    pub stdin_rx: Option<&'a Mutex<mpsc::Receiver<String>>>,
    /// Whether to print the search info to stdout.
    pub print_to_stdout: bool,
    /// Evaluation parameters for HCE.
    pub eval_params: EvalParams,
    /// Search parameters.
    pub search_params: SearchParams,
    /// LMR + LMP lookup table.
    pub lm_table: LMTable,
    /// The time manager.
    pub time_manager: TimeManager,

    /* Conditionally-compiled stat trackers: */
    /// The number of fail-highs found (beta cutoffs).
    #[cfg(feature = "stats")]
    pub failhigh: u64,
    /// The number of fail-highs that occurred on a given ply.
    #[cfg(feature = "stats")]
    pub failhigh_index: [u64; MAX_POSITION_MOVES],
    /// Tracks fail-highs of different types.
    #[cfg(feature = "stats")]
    pub failhigh_types: [u64; 8],
    /// The number of fail-highs found in quiescence search.
    #[cfg(feature = "stats")]
    pub qfailhigh: u64,
    /// The number of fail-highs that occurred on a given ply in quiescence search.
    #[cfg(feature = "stats")]
    pub qfailhigh_index: [u64; MAX_POSITION_MOVES],
}

impl<'a> SearchInfo<'a> {
    pub fn new(stopped: &'a AtomicBool, nodes: &'a AtomicU64) -> Self {
        let out = Self {
            nodes: BatchedAtomicCounter::new(nodes),
            root_move_nodes: [[0; 64]; 64],
            stopped,
            seldepth: ZERO_PLY,
            stdin_rx: None,
            print_to_stdout: true,
            eval_params: EvalParams::default(),
            search_params: SearchParams::default(),
            lm_table: LMTable::default(),
            time_manager: TimeManager::default(),
            #[cfg(feature = "stats")]
            failhigh: 0,
            #[cfg(feature = "stats")]
            failhigh_index: [0; MAX_POSITION_MOVES],
            #[cfg(feature = "stats")]
            failhigh_types: [0; 8],
            #[cfg(feature = "stats")]
            qfailhigh: 0,
            #[cfg(feature = "stats")]
            qfailhigh_index: [0; MAX_POSITION_MOVES],
        };
        assert!(!out.stopped.load(Ordering::SeqCst));
        out
    }

    pub fn set_up_for_search(&mut self) {
        self.stopped.store(false, Ordering::SeqCst);
        self.nodes.reset();
        self.root_move_nodes = [[0; 64]; 64];
        self.time_manager.reset_for_id();
        #[cfg(feature = "stats")]
        {
            self.failhigh = 0;
            self.failhigh_index = [0; MAX_POSITION_MOVES];
            self.failhigh_types = [0; 8];
            self.qfailhigh = 0;
            self.qfailhigh_index = [0; MAX_POSITION_MOVES];
        }
    }

    pub fn set_stdin(&mut self, stdin_rx: &'a Mutex<mpsc::Receiver<String>>) {
        self.stdin_rx = Some(stdin_rx);
    }

    pub fn check_up(&mut self) -> bool {
        let already_stopped = self.stopped.load(Ordering::SeqCst);
        if already_stopped {
            return true;
        }
        let res = self.time_manager.check_up(self.stopped, self.nodes.get_global());
        if let Some(Ok(cmd)) = self.stdin_rx.map(|m| m.lock().unwrap().try_recv()) {
            self.stopped.store(true, Ordering::SeqCst);
            let cmd = cmd.trim();
            if cmd == "quit" {
                uci::QUIT.store(true, Ordering::SeqCst);
            }
            true
        } else {
            res
        }
    }

    pub fn skip_print(&self) -> bool {
        self.time_manager.time_since_start().as_millis() < 50
    }

    pub fn stopped(&self) -> bool {
        self.stopped.load(Ordering::SeqCst)
    }

    #[cfg(feature = "stats")]
    pub fn log_fail_high<const QSEARCH: bool>(&mut self, move_index: usize, ordering_score: i32) {
        use crate::board::movegen::movepicker::{
            COUNTER_MOVE_SCORE, FIRST_KILLER_SCORE, SECOND_KILLER_SCORE, TT_MOVE_SCORE,
            WINNING_CAPTURE_SCORE,
        };

        if QSEARCH {
            self.qfailhigh += 1;
            self.qfailhigh_index[move_index] += 1;
        } else {
            self.failhigh += 1;
            self.failhigh_index[move_index] += 1;
            let fail_type = if ordering_score == TT_MOVE_SCORE {
                FailHighType::TTMove
            } else if ordering_score >= WINNING_CAPTURE_SCORE {
                FailHighType::GoodTactical
            } else if ordering_score == FIRST_KILLER_SCORE {
                FailHighType::Killer1
            } else if ordering_score == SECOND_KILLER_SCORE {
                FailHighType::Killer2
            } else if ordering_score == COUNTER_MOVE_SCORE {
                FailHighType::CounterMove
            } else if ordering_score > 0 {
                FailHighType::GoodQuiet
            } else {
                FailHighType::BadQuiet
            };
            self.failhigh_types[fail_type as usize] += 1;
        }
    }

    #[cfg(feature = "stats")]
    pub fn print_stats(&self) {
        #[allow(clippy::cast_precision_loss)]
        let fail_high_percentages = self
            .failhigh_index
            .iter()
            .map(|&x| (x as f64 * 100.0) / self.failhigh as f64)
            .take(10)
            .collect::<Vec<_>>();
        #[allow(clippy::cast_precision_loss)]
        let qs_fail_high_percentages = self
            .qfailhigh_index
            .iter()
            .map(|&x| (x as f64 * 100.0) / self.qfailhigh as f64)
            .take(10)
            .collect::<Vec<_>>();
        for ((i1, &x1), (i2, &x2)) in fail_high_percentages
            .iter()
            .enumerate()
            .zip(qs_fail_high_percentages.iter().enumerate())
        {
            println!("failhigh {x1:5.2}% at move {i1}     qfailhigh {x2:5.2}% at move {i2}");
        }
        let type_percentages = self
            .failhigh_types
            .iter()
            .map(|&x| (x as f64 * 100.0) / self.failhigh as f64)
            .collect::<Vec<_>>();
        println!("failhigh ttmove        {:5.2}%", type_percentages[0]);
        println!("failhigh good tactical {:5.2}%", type_percentages[1]);
        println!("failhigh killer1       {:5.2}%", type_percentages[2]);
        println!("failhigh killer2       {:5.2}%", type_percentages[3]);
        println!("failhigh countermove   {:5.2}%", type_percentages[4]);
        println!("failhigh good quiet    {:5.2}%", type_percentages[5]);
        println!("failhigh bad quiet     {:5.2}%", type_percentages[6]);
    }
}

#[cfg(feature = "stats")]
enum FailHighType {
    TTMove,
    GoodTactical,
    Killer1,
    Killer2,
    CounterMove,
    GoodQuiet,
    BadQuiet,
}

mod tests {
    #![allow(unused_imports)]
    use std::{
        array,
        sync::atomic::{AtomicBool, AtomicU64},
    };

    use super::{SearchInfo, SearchLimit};
    use crate::{
        board::{
            evaluation::{mate_in, mated_in},
            Board,
        },
        magic,
        threadlocal::ThreadData,
        timemgmt::TimeManager,
        transpositiontable::TT,
        util::MEGABYTE,
    };

    #[cfg(test)] // while running tests, we don't want multiple concurrent searches
    static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn go_mate_in_2_white() {
        let guard = TEST_LOCK.lock().unwrap();

        let mut position =
            Board::from_fen("r1b2bkr/ppp3pp/2n5/3qp3/2B5/8/PPPP1PPP/RNB1K2R w KQ - 0 9").unwrap();
        let stopped = AtomicBool::new(false);
        let time_manager = TimeManager::default_with_limit(SearchLimit::mate_in(2));
        let nodes = AtomicU64::new(0);
        let mut info = SearchInfo { time_manager, ..SearchInfo::new(&stopped, &nodes) };
        let mut tt = TT::new();
        tt.resize(MEGABYTE, 1);
        let mut t = ThreadData::new(0, &position);
        let (value, mov) =
            position.search_position::<true>(&mut info, array::from_mut(&mut t), tt.view());

        assert!(matches!(position.san(mov).as_deref(), Some("Bxd5+")));
        assert_eq!(value, mate_in(3)); // 3 ply because we're mating.

        drop(guard);
    }

    #[test]
    fn go_mated_in_2_white() {
        let guard = TEST_LOCK.lock().unwrap();

        let mut position =
            Board::from_fen("r1bq1bkr/ppp3pp/2n5/3Qp3/2B5/8/PPPP1PPP/RNB1K2R b KQ - 0 8").unwrap();
        let stopped = AtomicBool::new(false);
        let time_manager = TimeManager::default_with_limit(SearchLimit::mate_in(2));
        let nodes = AtomicU64::new(0);
        let mut info = SearchInfo { time_manager, ..SearchInfo::new(&stopped, &nodes) };
        let mut tt = TT::new();
        tt.resize(MEGABYTE, 1);
        let mut t = ThreadData::new(0, &position);
        let (value, mov) =
            position.search_position::<true>(&mut info, array::from_mut(&mut t), tt.view());

        assert!(matches!(position.san(mov).as_deref(), Some("Qxd5")));
        assert_eq!(value, mate_in(4)); // 4 ply (and positive) because white mates but it's black's turn.

        drop(guard);
    }

    #[test]
    fn go_mated_in_2_black() {
        let guard = TEST_LOCK.lock().unwrap();

        let mut position =
            Board::from_fen("rnb1k2r/pppp1ppp/8/2b5/3qP3/P1N5/1PP3PP/R1BQ1BKR w kq - 0 9").unwrap();
        let stopped = AtomicBool::new(false);
        let time_manager = TimeManager::default_with_limit(SearchLimit::mate_in(2));
        let nodes = AtomicU64::new(0);
        let mut info = SearchInfo { time_manager, ..SearchInfo::new(&stopped, &nodes) };
        let mut tt = TT::new();
        tt.resize(MEGABYTE, 1);
        let mut t = ThreadData::new(0, &position);
        let (value, mov) =
            position.search_position::<true>(&mut info, array::from_mut(&mut t), tt.view());

        assert!(matches!(position.san(mov).as_deref(), Some("Qxd4")));
        assert_eq!(value, -mate_in(4)); // 4 ply (and negative) because black mates but it's white's turn.

        drop(guard);
    }

    #[test]
    fn go_mate_in_2_black() {
        let guard = TEST_LOCK.lock().unwrap();

        let mut position =
            Board::from_fen("rnb1k2r/pppp1ppp/8/2b5/3QP3/P1N5/1PP3PP/R1B2BKR b kq - 0 9").unwrap();
        let stopped = AtomicBool::new(false);
        let time_manager = TimeManager::default_with_limit(SearchLimit::mate_in(2));
        let nodes = AtomicU64::new(0);
        let mut info = SearchInfo { time_manager, ..SearchInfo::new(&stopped, &nodes) };
        let mut tt = TT::new();
        tt.resize(MEGABYTE, 1);
        let mut t = ThreadData::new(0, &position);
        let (value, mov) =
            position.search_position::<true>(&mut info, array::from_mut(&mut t), tt.view());

        assert!(matches!(position.san(mov).as_deref(), Some("Bxd4+")));
        assert_eq!(value, -mate_in(3)); // 3 ply because we're mating.

        drop(guard);
    }
}
