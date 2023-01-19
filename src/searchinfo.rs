use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Mutex,
    },
    time::{Duration, Instant},
};

use crate::{chessmove::Move, definitions::depth::{Depth, ZERO_PLY}, board::evaluation::mate_in, search::parameters::SearchParams};

#[derive(PartialEq, Eq, Clone, Debug)]
pub enum SearchLimit {
    Infinite,
    Depth(Depth),
    Time(u64),
    TimeOrCorrectMoves(u64, Vec<Move>),
    Nodes(u64),
    Mate(usize),
    Dynamic {
        our_clock: u64,
        their_clock: u64,
        our_inc: u64,
        their_inc: u64,
        moves_to_go: u64,
        max_time_window: u64,
        time_window: u64,
    },
}

impl Default for SearchLimit {
    fn default() -> Self {
        Self::Infinite
    }
}

impl SearchLimit {
    pub const fn depth(&self) -> Option<Depth> {
        match self {
            Self::Depth(d) => Some(*d),
            _ => None,
        }
    }

    pub fn compute_time_windows(our_clock: u64, moves_to_go: Option<u64>, our_inc: u64, config: &SearchParams) -> (u64, u64) {
        let max_time = our_clock.saturating_sub(30);
        if let Some(moves_to_go) = moves_to_go {
            let divisor = moves_to_go.clamp(2, config.search_time_fraction);
            let computed_time_window = our_clock / divisor;
            let time_window = computed_time_window.min(max_time);
            let max_time_window = (time_window * 5 / 2).min(max_time);
            return (time_window, max_time_window);
        }
        let computed_time_window = our_clock / config.search_time_fraction + our_inc / 2 - 10;
        let time_window = computed_time_window.min(max_time);
        let max_time_window = (time_window * 5 / 2).min(max_time);
        (time_window, max_time_window)
    }

    #[cfg(test)]
    pub const fn mate_in(moves: usize) -> Self {
        Self::Mate(moves * 2)
    }
}

#[allow(clippy::struct_excessive_bools)]
#[derive(Clone)]
pub struct SearchInfo<'a> {
    /// The starting time of the search.
    pub start_time: Instant,
    /// The number of nodes searched.
    pub nodes: u64,
    /// Signal to quit the search.
    pub quit: bool,
    /// Signal to stop the search.
    pub stopped: &'static AtomicBool,
    /// The number of fail-highs found (beta cutoffs).
    pub failhigh: u64,
    /// The number of fail-highs that occured on the first move searched.
    pub failhigh_first: u64,
    /// The highest depth reached (selective depth).
    pub seldepth: Depth,
    /// A handle to a receiver for stdin.
    pub stdin_rx: Option<&'a Mutex<mpsc::Receiver<String>>>,
    /// Whether to print the search info to stdout.
    pub print_to_stdout: bool,
    /// Form of the search limit.
    pub limit: SearchLimit,
}

static GLOBAL_STOPPED: AtomicBool = AtomicBool::new(false);

impl Default for SearchInfo<'_> {
    fn default() -> Self {
        let out = Self {
            start_time: Instant::now(),
            nodes: 0,
            quit: false,
            stopped: &GLOBAL_STOPPED,
            failhigh: 0,
            failhigh_first: 0,
            seldepth: ZERO_PLY,
            stdin_rx: None,
            print_to_stdout: true,
            limit: SearchLimit::default(),
        };
        debug_assert!(!out.stopped.load(Ordering::SeqCst));
        out
    }
}

impl<'a> SearchInfo<'a> {
    pub fn setup_for_search(&mut self) {
        self.stopped.store(false, Ordering::SeqCst);
        self.nodes = 0;
        self.failhigh = 0;
        self.failhigh_first = 0;
    }

    pub fn set_stdin(&mut self, stdin_rx: &'a Mutex<mpsc::Receiver<String>>) {
        self.stdin_rx = Some(stdin_rx);
    }

    pub fn set_time_window(&mut self, millis: u64) {
        self.start_time = Instant::now();
        match &mut self.limit {
            SearchLimit::Dynamic { time_window, .. } => {
                *time_window = millis;
            }
            other => panic!("Unexpected search limit: {other:?}"),
        }
    }

    pub fn multiply_time_window(&mut self, factor: f64) {
        #![allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        match &mut self.limit {
            SearchLimit::Dynamic { time_window, .. } => {
                *time_window = (*time_window as f64 * factor) as u64;
            }
            other => panic!("Unexpected search limit: {other:?}"),
        }
    }

    pub fn check_up(&mut self) {
        match self.limit {
            SearchLimit::Depth(_) | SearchLimit::Mate(_) | SearchLimit::Infinite => {}
            SearchLimit::Nodes(nodes) => {
                if self.nodes >= nodes {
                    self.stopped.store(true, Ordering::SeqCst);
                }
            }
            SearchLimit::Time(time_window)
            | SearchLimit::Dynamic { time_window, .. }
            | SearchLimit::TimeOrCorrectMoves(time_window, _) => {
                let elapsed = self.start_time.elapsed();
                // this cast is safe to do, because u64::MAX milliseconds is 585K centuries.
                #[allow(clippy::cast_possible_truncation)]
                let elapsed_millis = elapsed.as_millis() as u64;
                if elapsed_millis >= time_window {
                    self.stopped.store(true, Ordering::SeqCst);
                }
            }
        }
        if let Some(Ok(cmd)) = self.stdin_rx.map(|m| m.lock().unwrap().try_recv()) {
            self.stopped.store(true, Ordering::SeqCst);
            let cmd = cmd.trim();
            if cmd == "quit" {
                self.quit = true;
            }
        };
    }

    pub fn stopped(&self) -> bool {
        self.stopped.load(Ordering::SeqCst)
    }

    pub fn check_if_search_condition_met(&mut self, best_move: Move, value: i32, depth: i32) {
        if let SearchLimit::TimeOrCorrectMoves(_, correct_moves) = &self.limit {
            if correct_moves.contains(&best_move) {
                self.stopped.store(true, Ordering::SeqCst);
            }
        } else if let &SearchLimit::Mate(mate_ply) = &self.limit {
            let expected_score = mate_in(mate_ply);
            let is_good_enough = value.abs() >= expected_score;
            #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
            if is_good_enough && depth >= mate_ply as i32 {
                self.stopped.store(true, Ordering::SeqCst);
            }
        }
    }

    /// If we have used enough time that stopping after finishing a depth would be good here.
    pub fn is_past_opt_time(&self) -> bool {
        match self.limit {
            SearchLimit::Dynamic { time_window, .. } => {
                let elapsed = self.start_time.elapsed();
                // this cast is safe to do, because u64::MAX milliseconds is 585K centuries.
                #[allow(clippy::cast_possible_truncation)]
                let elapsed_millis = elapsed.as_millis() as u64;
                let optimistic_time_window = time_window * 6 / 10;
                elapsed_millis >= optimistic_time_window
            }
            _ => false,
        }
    }

    pub const fn in_game(&self) -> bool {
        matches!(self.limit, SearchLimit::Dynamic { .. })
    }

    pub fn time_since_start(&self) -> Duration {
        Instant::now().checked_duration_since(self.start_time).unwrap_or_default()
    }
}

mod tests {
    #![allow(unused_imports)]
    use crate::{board::{Board, evaluation::{mate_in, mated_in}}, threadlocal::ThreadData, transpositiontable::TT, definitions::MEGABYTE, magic};
    use super::{SearchInfo, SearchLimit};

#[cfg(test)] // while running tests, we don't want multiple concurrent searches
static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn go_mate_in_2_white() {
        let guard = TEST_LOCK.lock().unwrap();
        magic::initialise();
        let mut position = Board::from_fen("r1b2bkr/ppp3pp/2n5/3qp3/2B5/8/PPPP1PPP/RNB1K2R w KQ - 0 9").unwrap();
        let mut info = SearchInfo {
            limit: SearchLimit::mate_in(2),
            ..SearchInfo::default()
        };
        let mut tt = TT::new();
        tt.resize(MEGABYTE);
        let mut thread_headers = vec![
            ThreadData::new(0),
        ];
        thread_headers[0].alloc_tables();
        thread_headers[0].nnue.refresh_acc(&position);
        let (value, mov) = position.search_position::<true>(&mut info, &mut thread_headers, tt.view());

        assert!(matches!(position.san(mov).as_deref(), Some("Bxd5+")));
        assert_eq!(value, mate_in(3)); // 3 ply because we're mating.

        drop(guard);
    }

    #[test]
    fn go_mated_in_2_white() {
        let guard = TEST_LOCK.lock().unwrap();
        magic::initialise();
        let mut position = Board::from_fen("r1bq1bkr/ppp3pp/2n5/3Qp3/2B5/8/PPPP1PPP/RNB1K2R b KQ - 0 8").unwrap();
        let mut info = SearchInfo {
            limit: SearchLimit::mate_in(2),
            ..SearchInfo::default()
        };
        let mut tt = TT::new();
        tt.resize(MEGABYTE);
        let mut thread_headers = vec![
            ThreadData::new(0),
        ];
        thread_headers[0].alloc_tables();
        thread_headers[0].nnue.refresh_acc(&position);
        let (value, mov) = position.search_position::<true>(&mut info, &mut thread_headers, tt.view());
        
        assert!(matches!(position.san(mov).as_deref(), Some("Qxd5")));
        assert_eq!(value, mate_in(4)); // 4 ply (and positive) because white mates but it's black's turn.

        drop(guard);
    }

    #[test]
    fn go_mated_in_2_black() {
        let guard = TEST_LOCK.lock().unwrap();
        magic::initialise();
        let mut position = Board::from_fen("rnb1k2r/pppp1ppp/8/2b5/3qP3/P1N5/1PP3PP/R1BQ1BKR w kq - 0 9").unwrap();
        let mut info = SearchInfo {
            limit: SearchLimit::mate_in(2),
            ..SearchInfo::default()
        };
        let mut tt = TT::new();
        tt.resize(MEGABYTE);
        let mut thread_headers = vec![
            ThreadData::new(0),
        ];
        thread_headers[0].alloc_tables();
        thread_headers[0].nnue.refresh_acc(&position);
        let (value, mov) = position.search_position::<true>(&mut info, &mut thread_headers, tt.view());
        
        assert!(matches!(position.san(mov).as_deref(), Some("Qxd4")));
        assert_eq!(value, -mate_in(4)); // 4 ply (and negative) because black mates but it's white's turn.

        drop(guard);
    }

    #[test]
    fn go_mate_in_2_black() {
        let guard = TEST_LOCK.lock().unwrap();
        magic::initialise();
        let mut position = Board::from_fen("rnb1k2r/pppp1ppp/8/2b5/3QP3/P1N5/1PP3PP/R1B2BKR b kq - 0 9").unwrap();
        let mut info = SearchInfo {
            limit: SearchLimit::mate_in(2),
            ..SearchInfo::default()
        };
        let mut tt = TT::new();
        tt.resize(MEGABYTE);
        let mut thread_headers = vec![
            ThreadData::new(0),
        ];
        thread_headers[0].alloc_tables();
        thread_headers[0].nnue.refresh_acc(&position);
        let (value, mov) = position.search_position::<true>(&mut info, &mut thread_headers, tt.view());

        assert!(matches!(position.san(mov).as_deref(), Some("Bxd4+")));
        assert_eq!(value, -mate_in(3)); // 3 ply because we're mating.

        drop(guard);
    }
}