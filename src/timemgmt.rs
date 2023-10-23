use std::{
    ops::ControlFlow,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

use crate::{
    board::evaluation::{is_mate_score, mate_in},
    chessmove::Move,
    search::pv::PVariation,
    transpositiontable::Bound,
    util::depth::Depth,
};

const MOVE_OVERHEAD: u64 = 10;
const DEFAULT_MOVES_TO_GO: u64 = 20;

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum ForcedMoveType {
    OneLegal,
    Strong,
    Weak,
    None,
}

impl ForcedMoveType {
    pub const fn tm_multiplier(self) -> f64 {
        match self {
            Self::OneLegal => 0.01,
            Self::Strong => 0.25,
            Self::Weak => 0.5,
            Self::None => 1.0,
        }
    }
}

#[derive(PartialEq, Eq, Clone, Debug)]
pub enum SearchLimit {
    Infinite,
    Depth(Depth),
    Time(u64),
    Nodes(u64),
    SoftNodes(u64),
    Mate {
        ply: usize,
    },
    Dynamic {
        our_clock: u64,
        their_clock: u64,
        our_inc: u64,
        their_inc: u64,
        moves_to_go: Option<u64>,
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
    ) -> (u64, u64, u64) {
        // The absolute maximum time we could spend without losing on the clock:
        let absolute_maximum = our_clock.saturating_sub(MOVE_OVERHEAD);

        // If we have a moves to go, we can use that to compute a time window.
        if let Some(moves_to_go) = moves_to_go {
            // Use more time if we have fewer moves to go, but not more than DEFAULT_MOVES_TO_GO.
            let divisor = moves_to_go.clamp(2, DEFAULT_MOVES_TO_GO);
            let computed_time_window = our_clock / divisor;
            let hard_time_window = computed_time_window.min(absolute_maximum);
            let optimal_time_window = hard_time_window * 6 / 10;
            return (optimal_time_window, hard_time_window, absolute_maximum);
        }

        // Otherwise, we use DEFAULT_MOVES_TO_GO.
        let computed_time_window =
            our_clock / DEFAULT_MOVES_TO_GO + our_inc * 3 / 4 - MOVE_OVERHEAD;
        let hard_time_window = computed_time_window.min(absolute_maximum);
        let optimal_time_window = hard_time_window * 6 / 10;
        (optimal_time_window, hard_time_window, absolute_maximum)
    }

    #[cfg(test)]
    pub const fn mate_in(moves: usize) -> Self {
        Self::Mate { ply: moves * 2 }
    }
}

#[derive(Clone, Debug)]
pub struct TimeManager {
    /// The starting time of the search.
    start_time: Instant,
    /// The limit on the search.
    limit: SearchLimit,
    /// The maximum time that the search may last for without losing on the clock.
    max_time: Duration,
    /// The time after which search will be halted even mid-search.
    hard_time: Duration,
    /// The time after which we will stop upon completing a depth.
    opt_time: Duration,
    /// The value from the last iteration of search.
    prev_score: i32,
    /// The best move from the last iteration of search.
    prev_move: Move,
    /// The number of ID iterations for which the best move remained.
    stability: usize,
    /// Number of times that we have failed low.
    failed_low: i32,
    /// Number of ID iterations that a mate score has remained.
    mate_counter: usize,
    /// The nature of the forced move (if any)
    found_forced_move: ForcedMoveType,
    /// The last set of multiplicative factors.
    last_factors: [f64; 2],
    /// Fraction of nodes that were underneath the best move.
    best_move_nodes_fraction: Option<f64>,
}

impl Default for TimeManager {
    fn default() -> Self {
        Self {
            start_time: Instant::now(),
            limit: SearchLimit::Infinite,
            max_time: Duration::from_secs(0),
            hard_time: Duration::from_secs(0),
            opt_time: Duration::from_secs(0),
            prev_score: 0,
            prev_move: Move::NULL,
            stability: 0,
            failed_low: 0,
            mate_counter: 0,
            found_forced_move: ForcedMoveType::None,
            last_factors: [1.0, 1.0],
            best_move_nodes_fraction: None,
        }
    }
}

impl TimeManager {
    pub fn set_limit(&mut self, limit: SearchLimit) {
        self.limit = limit;
    }

    pub fn start(&mut self) {
        self.start_time = Instant::now();
    }

    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    pub const fn limit(&self) -> &SearchLimit {
        &self.limit
    }

    pub fn default_with_limit(limit: SearchLimit) -> Self {
        Self { limit, ..Default::default() }
    }

    pub fn reset_for_id(&mut self) {
        self.prev_score = 0;
        self.prev_move = Move::NULL;
        self.stability = 0;
        self.failed_low = 0;
        self.mate_counter = 0;
        self.found_forced_move = ForcedMoveType::None;
        self.last_factors = [1.0, 1.0];
        self.best_move_nodes_fraction = None;

        if let SearchLimit::Dynamic { our_clock, our_inc, moves_to_go, .. } = self.limit {
            let (opt_time, hard_time, max_time) =
                SearchLimit::compute_time_windows(our_clock, moves_to_go, our_inc);
            self.max_time = Duration::from_millis(max_time);
            self.hard_time = Duration::from_millis(hard_time);
            self.opt_time = Duration::from_millis(opt_time);
        }
    }

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
            SearchLimit::Time(millis) => {
                let elapsed = self.start_time.elapsed();
                // this cast is safe to do, because u64::MAX milliseconds is 585K centuries.
                #[allow(clippy::cast_possible_truncation)]
                let elapsed_millis = elapsed.as_millis() as u64;
                let past_limit = elapsed_millis >= millis;
                if past_limit {
                    stopped.store(true, Ordering::SeqCst);
                }
                past_limit
            }
            SearchLimit::Dynamic { .. } => {
                let past_limit = self.time_since_start() >= self.hard_time;
                if past_limit {
                    stopped.store(true, Ordering::SeqCst);
                }
                past_limit
            }
            SearchLimit::SoftNodes(limit) => {
                // this should never *really* return true, but we do this in case of search explosions.
                let hard_limit = limit * 128;
                let past_limit = nodes_so_far >= hard_limit;
                if past_limit {
                    stopped.store(true, Ordering::SeqCst);
                }
                past_limit
            }
        }
    }

    /// If we have used enough time that stopping after finishing a depth would be good here.
    pub fn is_past_opt_time(&self, nodes: u64) -> bool {
        match self.limit {
            SearchLimit::Dynamic { .. } => self.time_since_start() >= self.opt_time,
            SearchLimit::SoftNodes(limit) => nodes >= limit,
            _ => false,
        }
    }

    pub fn time_since_start(&self) -> Duration {
        self.start_time.elapsed()
    }

    pub const fn is_dynamic(&self) -> bool {
        matches!(self.limit, SearchLimit::Dynamic { .. })
    }

    pub const fn is_soft_nodes(&self) -> bool {
        matches!(self.limit, SearchLimit::SoftNodes(_))
    }

    pub fn solved_breaker<const MAIN_THREAD: bool>(
        &mut self,
        value: i32,
        depth: usize,
    ) -> ControlFlow<()> {
        if !MAIN_THREAD || depth < 8 {
            return ControlFlow::Continue(());
        }
        if let &SearchLimit::Mate { ply } = &self.limit {
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
            && self.is_dynamic()
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

    const SLIGHTLY_FORCED: Depth = Depth::new(12);
    const VERY_FORCED: Depth = Depth::new(8);
    pub fn report_forced_move(&mut self, depth: Depth) {
        assert_eq!(self.found_forced_move, ForcedMoveType::None);
        if depth >= Self::SLIGHTLY_FORCED {
            // reduce thinking time by 50%
            self.hard_time /= 2;
            self.opt_time /= 2;
            self.found_forced_move = ForcedMoveType::Weak;
        } else {
            /* depth >= Self::VERY_FORCED */
            // reduce thinking time by 75%
            self.hard_time /= 4;
            self.opt_time /= 4;
            self.found_forced_move = ForcedMoveType::Strong;
        }
    }

    pub fn check_for_forced_move(&self, depth: Depth) -> Option<i32> {
        if self.found_forced_move == ForcedMoveType::None && self.is_dynamic() {
            if depth >= Self::SLIGHTLY_FORCED {
                Some(170)
            } else if depth >= Self::VERY_FORCED {
                Some(400)
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn notify_one_legal_move(&mut self) {
        self.opt_time = Duration::from_millis(0);
        self.found_forced_move = ForcedMoveType::OneLegal;
    }

    fn best_move_stability_multiplier(stability: usize) -> f64 {
        // approach to this is adapted from Stash.
        const VALUES: [f64; 5] = [2.50, 1.20, 0.90, 0.80, 0.75];

        // Clamp stability to the range [0, 4], and convert it to a time scale in
        // the range [0.75, 2.50].
        let stability = stability.min(4);

        VALUES[stability]
    }

    fn best_move_subtree_size_multiplier(nodes_fraction: f64) -> f64 {
        (1.5 - nodes_fraction) * 1.35
    }

    pub fn report_completed_depth(
        &mut self,
        _depth: Depth,
        eval: i32,
        best_move: Move,
        best_move_nodes_fraction: Option<f64>,
    ) {
        if let SearchLimit::Dynamic { our_clock, our_inc, moves_to_go, .. } = self.limit {
            let (opt_time, hard_time, max_time) =
                SearchLimit::compute_time_windows(our_clock, moves_to_go, our_inc);
            let max_time = Duration::from_millis(max_time);
            let hard_time = Duration::from_millis(hard_time);
            let opt_time = Duration::from_millis(opt_time);

            if best_move == self.prev_move {
                self.stability += 1;
            } else {
                self.stability = 0;
            }
            self.best_move_nodes_fraction = best_move_nodes_fraction;

            let stability_multiplier = Self::best_move_stability_multiplier(self.stability);
            // retain time added by windows that failed low
            let failed_low_multiplier = f64::from(self.failed_low).mul_add(0.25, 1.0);
            let forced_move_multiplier = self.found_forced_move.tm_multiplier();
            let subtree_size_multiplier =
                self.best_move_nodes_fraction.map_or(1.0, Self::best_move_subtree_size_multiplier);

            let multiplier = stability_multiplier
                * failed_low_multiplier
                * forced_move_multiplier
                * subtree_size_multiplier;

            let hard_time = Duration::from_secs_f64(hard_time.as_secs_f64() * multiplier);
            let opt_time = Duration::from_secs_f64(opt_time.as_secs_f64() * multiplier);

            self.hard_time = hard_time.min(max_time);
            self.opt_time = opt_time.min(max_time);

            self.last_factors = [stability_multiplier, failed_low_multiplier];
        }

        self.prev_move = best_move;
        self.prev_score = eval;
    }

    pub fn report_aspiration_fail(&mut self, depth: Depth, bound: Bound) {
        const FAIL_LOW_UPDATE_THRESHOLD: Depth = Depth::new(0);
        let SearchLimit::Dynamic { our_clock, our_inc, moves_to_go, .. } = self.limit else {
            return;
        };
        if depth >= FAIL_LOW_UPDATE_THRESHOLD && bound == Bound::Upper && self.failed_low < 2 {
            self.failed_low += 1;

            let (opt_time, hard_time, max_time) =
                SearchLimit::compute_time_windows(our_clock, moves_to_go, our_inc);
            let max_time = Duration::from_millis(max_time);
            let hard_time = Duration::from_millis(hard_time);
            let opt_time = Duration::from_millis(opt_time);

            let stability_multiplier = self.last_factors[0];
            // calculate the failed low multiplier
            let failed_low_multiplier = f64::from(self.failed_low).mul_add(0.25, 1.0);
            let forced_move_multiplier = self.found_forced_move.tm_multiplier();
            let subtree_size_multiplier =
                self.best_move_nodes_fraction.map_or(1.0, Self::best_move_subtree_size_multiplier);

            let multiplier = stability_multiplier
                * failed_low_multiplier
                * forced_move_multiplier
                * subtree_size_multiplier;

            let hard_time = Duration::from_secs_f64(hard_time.as_secs_f64() * multiplier);
            let opt_time = Duration::from_secs_f64(opt_time.as_secs_f64() * multiplier);

            self.hard_time = hard_time.min(max_time);
            self.opt_time = opt_time.min(max_time);

            self.last_factors[1] = failed_low_multiplier;
        }
    }
}
