use std::{
    ops::ControlFlow,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

use crate::{
    board::{
        evaluation::{is_mate_score, mate_in},
        Board,
    },
    chessmove::Move,
    definitions::depth::Depth,
    search::{parameters::SearchParams, PVariation},
    threadlocal::ThreadData,
    transpositiontable::{Bound, TTView},
};

const MOVE_OVERHEAD: u64 = 10;

#[derive(PartialEq, Eq, Clone, Debug)]
pub enum SearchLimit {
    Infinite,
    Depth(Depth),
    Time(u64),
    TimeOrCorrectMoves(u64, Vec<Move>),
    Nodes(u64),
    Mate {
        ply: usize,
    },
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

    pub fn compute_time_windows(
        our_clock: u64,
        moves_to_go: Option<u64>,
        our_inc: u64,
        config: &SearchParams,
    ) -> (u64, u64) {
        let max_time = our_clock.saturating_sub(MOVE_OVERHEAD);
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
        Self::Mate { ply: moves * 2 }
    }
}

#[derive(Clone, Debug)]
pub struct TimeManager {
    /// The starting time of the search.
    pub start_time: Instant,
    /// The limit on the search.
    pub limit: SearchLimit,
    /// The maximum time that the search may last for.
    pub max_time: Duration,
    /// The time after which we will stop upon completing a depth.
    pub opt_time: Duration,
    /// The value from the last iteration of search.
    pub prev_score: i32,
    /// The best move from the last iteration of search.
    pub prev_move: Move,
    /// The number of ID iterations for which the best move remained.
    pub stability: usize,
    /// Whether the search has failed low at appreciable depth.
    pub failed_low: bool,
    /// Number of ID iterations that a mate score has remained.
    pub mate_counter: usize,
    /// Whether we have found a forced move.
    pub found_forced_move: bool,
}

impl Default for TimeManager {
    fn default() -> Self {
        Self {
            start_time: Instant::now(),
            limit: SearchLimit::Infinite,
            max_time: Duration::from_secs(0),
            opt_time: Duration::from_secs(0),
            prev_score: 0,
            prev_move: Move::NULL,
            stability: 0,
            failed_low: false,
            mate_counter: 0,
            found_forced_move: false,
        }
    }
}

impl TimeManager {
    pub fn check_up(&mut self, stopped: &AtomicBool, nodes_so_far: u64) -> bool {
        match self.limit {
            SearchLimit::Depth(_) | SearchLimit::Mate { .. } | SearchLimit::Infinite => {
                stopped.load(Ordering::SeqCst)
            }
            SearchLimit::Nodes(nodes) => {
                let past_limit = nodes_so_far >= nodes;
                if past_limit {
                    stopped.store(true, Ordering::SeqCst);
                }
                past_limit
            }
            SearchLimit::Time(time_window)
            | SearchLimit::Dynamic { time_window, .. }
            | SearchLimit::TimeOrCorrectMoves(time_window, _) => {
                let elapsed = self.start_time.elapsed();
                // this cast is safe to do, because u64::MAX milliseconds is 585K centuries.
                #[allow(clippy::cast_possible_truncation)]
                let elapsed_millis = elapsed.as_millis() as u64;
                let past_limit = elapsed_millis >= time_window;
                if past_limit {
                    stopped.store(true, Ordering::SeqCst);
                }
                past_limit
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

    pub fn time_since_start(&self) -> Duration {
        Instant::now().checked_duration_since(self.start_time).unwrap_or_default()
    }

    pub fn report_aspiration_fail(&mut self, depth: Depth, bound: Bound) {
        const FAIL_LOW_UPDATE_THRESHOLD: Depth = Depth::new(10);
        if self.in_game()
            && depth >= FAIL_LOW_UPDATE_THRESHOLD
            && bound == Bound::Upper
            && !self.failed_low
        {
            self.failed_low = true;
            // add 50% to the time limit
            self.max_time += self.max_time / 2;
            self.opt_time += self.opt_time / 2;
            // clamp to under (clock - move overhead)
            let SearchLimit::Dynamic { our_clock, .. } = self.limit else { unreachable!() };
            let hard_time_limit = our_clock.saturating_sub(MOVE_OVERHEAD);
            let hard_time_limit = Duration::from_millis(hard_time_limit);
            self.max_time = self.max_time.min(hard_time_limit);
            self.opt_time = self.opt_time.min(hard_time_limit);
        }
    }

    pub const fn is_test_suite(&self) -> bool {
        matches!(self.limit, SearchLimit::TimeOrCorrectMoves(_, _))
    }

    pub const fn in_game(&self) -> bool {
        matches!(self.limit, SearchLimit::Dynamic { .. })
    }

    pub fn solved_breaker<const MAIN_THREAD: bool>(
        &mut self,
        best_move: Move,
        value: i32,
        depth: usize,
    ) -> ControlFlow<()> {
        if !MAIN_THREAD || depth < 8 {
            return ControlFlow::Continue(());
        }
        if let SearchLimit::TimeOrCorrectMoves(_, correct_moves) = &self.limit {
            if correct_moves.contains(&best_move) {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        } else if let &SearchLimit::Mate { ply } = &self.limit {
            let expected_score = mate_in(ply);
            let is_good_enough = value.abs() >= expected_score;
            #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
            if is_good_enough && depth >= ply {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        } else {
            ControlFlow::Continue(())
        }
    }

    pub fn mate_found_breaker<const MAIN_THREAD: bool>(
        &mut self,
        pv: &PVariation,
        depth: Depth,
    ) -> ControlFlow<()> {
        const MINIMUM_MATE_BREAK_DEPTH: Depth = Depth::new(10);
        if MAIN_THREAD
            && self.in_game()
            && is_mate_score(pv.score())
            && depth > MINIMUM_MATE_BREAK_DEPTH
        {
            self.mate_counter += 1;
            if self.mate_counter >= 3 {
                return ControlFlow::Break(());
            }
        } else if MAIN_THREAD {
            self.mate_counter = 0;
        }
        ControlFlow::Continue(())
    }

    pub fn report_forced_move(
        &mut self,
        board: &mut Board,
        depth: Depth,
        tt: TTView,
        t: &mut ThreadData,
        bestmove: Move,
        score: i32,
    ) -> ControlFlow<()> {
        assert!(!self.found_forced_move);
        self.found_forced_move = true;
        // reduce thinking time by 75%
        self.max_time = self.max_time / 4;
        self.opt_time = self.opt_time / 4;
        ControlFlow::Continue(())
    }

    pub fn check_for_forced_move(&self, depth: Depth) -> bool {
        const FORCED_MOVE_BREAK_DEPTH: Depth = Depth::new(10);
        depth > FORCED_MOVE_BREAK_DEPTH && !self.found_forced_move && self.in_game()
    }

    pub fn notify_one_legal_move(&mut self) {
        self.opt_time = Duration::from_millis(0);
    }
}
