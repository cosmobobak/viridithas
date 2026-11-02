// SPDX-License-Identifier: AGPL-3.0-only

use std::{
    ops::Deref,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

use crate::{
    chess::chessmove::Move,
    evaluation::{is_mate_score, mate_in},
    search::parameters::Config,
    transpositiontable::Bound,
};

/// A buffer to account for communication between the engine and the client.
const MOVE_OVERHEAD: u64 = 30;
/// The fraction per mille that we are ever allowed to use of the bank.
const MAX_BANK_USABLE: u64 = 600;

pub const STRONG_FORCED_TM_FRAC: u32 = 386;
pub const WEAK_FORCED_TM_FRAC: u32 = 627;
pub const DEFAULT_MOVES_TO_GO: u32 = 24;
pub const HARD_WINDOW_FRAC: u32 = 46;
pub const OPTIMAL_WINDOW_FRAC: u32 = 73;
pub const INCREMENT_FRAC: u32 = 94;
pub const NODE_TM_SUBTREE_MULTIPLIER: u32 = 140;
pub const FAIL_LOW_TM_BONUS: u32 = 340;

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum Forcedness {
    OneLegal,
    Strong,
    Weak,
    None,
}

impl Forcedness {
    pub fn tm_multiplier(self, conf: &Config) -> f64 {
        match self {
            Self::OneLegal => 0.01,
            Self::Strong => f64::from(conf.strong_forced_tm_frac) / 1000.0,
            Self::Weak => f64::from(conf.weak_forced_tm_frac) / 1000.0,
            Self::None => 1.0,
        }
    }
}

#[derive(PartialEq, Eq, Clone, Debug)]
pub struct Dynamic {
    pub our_clock: u64,
    pub their_clock: u64,
    pub our_inc: u64,
    pub their_inc: u64,
    pub moves_to_go: Option<u64>,
}

#[derive(PartialEq, Eq, Clone, Debug)]
pub enum SearchLimit {
    /// search ⟨x⟩ plies only.
    Depth(usize),
    /// search for ⟨x⟩ milliseconds only.
    Time(u64),
    /// search ⟨x⟩ nodes only.
    Nodes(u64),
    /// search until a mate in ⟨x⟩ or fewer ply is found.
    Mate { ply: usize },
    /// search according to a clock.
    Dynamic(Dynamic),
    /// search until ⟨x⟩ nodes have been
    /// explored and a depth has been cleared.
    #[cfg(feature = "datagen")]
    SoftNodes { soft_limit: u64, hard_limit: u64 },
    /// restrict search to these moves only
    Moves(Vec<Move>),
}

impl SearchLimit {
    #[cfg(test)]
    pub const fn mate_in(moves: usize) -> Self {
        Self::Mate { ply: moves * 2 }
    }

    pub fn check_up(
        &self,
        start_time: Instant,
        // TODO: this is weird to pass unconditionally.
        hard_time: Duration,
        nodes_so_far: u64,
    ) -> bool {
        match *self {
            Self::Depth(_) | Self::Mate { .. } => false, // depth & mate are checked externally.
            Self::Nodes(nodes) => nodes_so_far >= nodes,
            Self::Time(millis) => start_time.elapsed().as_millis() >= u128::from(millis),
            Self::Dynamic { .. } => start_time.elapsed() >= hard_time,
            #[cfg(feature = "datagen")]
            Self::SoftNodes { hard_limit, .. } => {
                // this should never *really* return true, but we do this in case of search explosions.
                nodes_so_far >= hard_limit
            }
            Self::Moves(_) => {
                /* nothing to do */
                false
            } //
              // TODO: reïmplement Ponder.
              // SearchLimit::Pondering { .. } => false,
        }
    }

    /// If we have used enough time that stopping after finishing a depth would be good here.
    #[allow(unused_variables)]
    pub fn is_past_opt_time(&self, nodes: u64, start_time: Instant, opt_time: Duration) -> bool {
        match *self {
            Self::Dynamic { .. } => start_time.elapsed() >= opt_time,
            #[cfg(feature = "datagen")]
            Self::SoftNodes { soft_limit, .. } => nodes >= soft_limit,
            _ => false,
        }
    }

    pub const fn is_dynamic(&self) -> bool {
        matches!(*self, Self::Dynamic { .. })
    }
}

/// A conjunctive set of limits to the search.
///
/// Causes search to terminate when one or more limits apply.
///
/// A command like “`go depth 8 nodes 25000`” should terminate
/// the search when depth 8 has been reached or 25000 nodes have
/// been searched, whichever comes first.
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct Limits {
    limits: Vec<SearchLimit>,
}

impl Deref for Limits {
    type Target = [SearchLimit];

    fn deref(&self) -> &Self::Target {
        &self.limits
    }
}

impl From<SearchLimit> for Limits {
    fn from(value: SearchLimit) -> Self {
        Self {
            limits: vec![value],
        }
    }
}

impl Limits {
    pub fn infinite() -> Self {
        Self { limits: Vec::new() }
    }

    pub fn dynamic(&self) -> Option<Dynamic> {
        for limit in self.iter() {
            if let SearchLimit::Dynamic(limit) = limit {
                return Some(limit.clone());
            }
        }

        None
    }

    pub fn depth(&self) -> Option<usize> {
        for limit in self.iter() {
            if let SearchLimit::Depth(limit) = limit {
                return Some(*limit);
            }
        }

        None
    }

    pub fn mate(&self) -> Option<usize> {
        for limit in self.iter() {
            if let SearchLimit::Mate { ply } = limit {
                return Some(*ply);
            }
        }

        None
    }

    /// Add a new limit.
    pub fn push(&mut self, limit: SearchLimit) {
        // note: we might consider deduplicating equal
        //       limit types here, but presently don’t.
        self.limits.push(limit);
    }
}

pub fn compute_time_windows(
    our_clock: u64,
    moves_to_go: Option<u64>,
    our_inc: u64,
    conf: &Config,
) -> (u64, u64, u64) {
    // The very highest amount of time we are willing to search in a position, ever.
    let max_time = (our_clock * MAX_BANK_USABLE / 1000).saturating_sub(MOVE_OVERHEAD);

    // The maximum time we can spend searching before forcibly stopping:
    let hard_time_window = (our_clock * u64::from(conf.hard_window_frac) / 100).min(max_time);

    // If we have a moves to go, we can use that to compute a time window.
    if let Some(moves_to_go) = moves_to_go {
        // Use more time if we have fewer moves to go, but not more than default_moves_to_go.
        let divisor = moves_to_go.clamp(2, u64::from(conf.default_moves_to_go));
        let computed_time_window = our_clock / divisor;
        let optimal_time_window =
            computed_time_window.min(max_time) * u64::from(conf.optimal_window_frac) / 100;
        return (optimal_time_window, hard_time_window, max_time);
    }

    // Otherwise, we use default_moves_to_go.
    let computed_time_window = our_clock / u64::from(conf.default_moves_to_go)
        + our_inc * u64::from(conf.increment_frac) / 100
        - MOVE_OVERHEAD;

    let optimal_time_window =
        (computed_time_window.min(max_time) * u64::from(conf.optimal_window_frac) / 100)
            .min(hard_time_window);

    (optimal_time_window, hard_time_window, max_time)
}

#[derive(Clone, Debug)]
pub struct TimeManager {
    /// The starting time of the search.
    start_time: Instant,
    /// The limit on the search.
    pub limits: Limits,
    /// The time after which search will be halted even mid-search.
    hard_time: Duration,
    /// The time after which we will stop upon completing a depth.
    opt_time: Duration,
    /// The value from the last iteration of search.
    prev_score: i32,
    /// The best move from the last iteration of search.
    prev_move: Option<Move>,
    /// The number of ID iterations for which the best move remained.
    stability: usize,
    /// Number of times that we have failed low.
    failed_low: i32,
    /// Number of ID iterations that a mate score has remained.
    mate_counter: usize,
    /// The nature of the forced move (if any)
    forcedness: Forcedness,
    /// The last set of multiplicative factors.
    last_factors: [f64; 2],
    /// Fraction of nodes that were underneath the best move.
    best_move_nodes_fraction: Option<f64>,
}

impl Default for TimeManager {
    fn default() -> Self {
        Self {
            start_time: Instant::now(),
            limits: Limits::infinite(),
            hard_time: Duration::from_secs(0),
            opt_time: Duration::from_secs(0),
            prev_score: 0,
            prev_move: None,
            stability: 0,
            failed_low: 0,
            mate_counter: 0,
            forcedness: Forcedness::None,
            last_factors: [1.0, 1.0],
            best_move_nodes_fraction: None,
        }
    }
}

impl TimeManager {
    pub fn set_limits(&mut self, limits: Limits) {
        self.limits = limits;
    }

    pub fn start(&mut self) {
        self.start_time = Instant::now();
    }

    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    #[allow(dead_code)]
    pub fn default_with_limit(limit: SearchLimit) -> Self {
        Self {
            limits: limit.into(),
            ..Default::default()
        }
    }

    pub fn check_up(&self, stopped: &AtomicBool, nodes_so_far: u64) -> bool {
        let start_time = self.start_time;
        let hard_time = self.hard_time;
        let hit_limit = self
            .limits
            .iter()
            .any(|l| l.check_up(start_time, hard_time, nodes_so_far));

        if hit_limit {
            stopped.store(true, Ordering::Relaxed);
            true
        } else {
            stopped.load(Ordering::Relaxed)
        }
    }

    pub fn is_past_opt_time(&self, nodes: u64) -> bool {
        self.limits
            .iter()
            .any(|l| l.is_past_opt_time(nodes, self.start_time, self.opt_time))
    }

    pub fn solved_or_mating(&mut self, score: i32) -> bool {
        if let Some(ply) = self.limits.mate()
            && score.abs() >= mate_in(ply)
        {
            return true;
        }

        if self.limits.dynamic().is_some() {
            if is_mate_score(score) {
                self.mate_counter += 1;
                if self.mate_counter >= 3 {
                    return true;
                }
            } else {
                self.mate_counter = 0;
            }
        }

        false
    }

    pub fn reset_for_id(&mut self, conf: &Config) {
        self.prev_score = 0;
        self.prev_move = None;
        self.stability = 0;
        self.failed_low = 0;
        self.mate_counter = 0;
        self.forcedness = Forcedness::None;
        self.last_factors = [1.0, 1.0];
        self.best_move_nodes_fraction = None;

        if let Some(Dynamic {
            our_clock,
            our_inc,
            moves_to_go,
            ..
        }) = self.limits.dynamic()
        {
            let (opt_time, hard_time, _) =
                compute_time_windows(our_clock, moves_to_go, our_inc, conf);
            // TODO: reïmplement Ponder.
            // deal with "ponderhit" arriving while we're stuck on a depth:
            // if matches!(self.limit, SearchLimit::Pondering { .. }) {
            //     hard_time = opt_time;
            // }
            self.hard_time = Duration::from_millis(hard_time);
            self.opt_time = Duration::from_millis(opt_time);
        }
    }

    const WEAKLY_FORCED: i32 = 12;
    const STRONGLY_FORCED: i32 = 8;
    pub fn report_forced_move(&mut self, depth: i32, conf: &Config) {
        assert_eq!(self.forcedness, Forcedness::None);
        if depth >= Self::WEAKLY_FORCED {
            self.hard_time = self.hard_time * conf.weak_forced_tm_frac / 1000;
            self.opt_time = self.opt_time * conf.weak_forced_tm_frac / 1000;
            self.forcedness = Forcedness::Weak;
        } else {
            // depth >= Self::VERY_FORCED
            self.hard_time = self.hard_time * conf.strong_forced_tm_frac / 1000;
            self.opt_time = self.opt_time * conf.strong_forced_tm_frac / 1000;
            self.forcedness = Forcedness::Strong;
        }
    }

    pub fn forcedness_margin(&self, depth: i32) -> Option<i32> {
        if self.forcedness == Forcedness::None && self.limits.iter().any(SearchLimit::is_dynamic) {
            if depth >= Self::WEAKLY_FORCED {
                Some(170)
            } else if depth >= Self::STRONGLY_FORCED {
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
        self.forcedness = Forcedness::OneLegal;
    }

    fn best_move_stability_multiplier(stability: usize) -> f64 {
        // approach to this is adapted from Stash.
        const VALUES: [f64; 5] = [2.50, 1.20, 0.90, 0.80, 0.75];

        // Clamp stability to the range [0, 4], and convert it to a time scale in
        // the range [0.75, 2.50].
        let stability = stability.min(4);

        VALUES[stability]
    }

    fn best_move_subtree_size_multiplier(nodes_fraction: f64, conf: &Config) -> f64 {
        (1.62 - nodes_fraction) * f64::from(conf.node_tm_subtree_multiplier) / 100.0
    }

    pub fn report_completed_depth(
        &mut self,
        _depth: i32,
        eval: i32,
        best_move: Move,
        best_move_nodes_fraction: Option<f64>,
        conf: &Config,
    ) {
        if let Some(Dynamic {
            our_clock,
            our_inc,
            moves_to_go,
            ..
        }) = self.limits.dynamic()
        {
            let (opt_time, hard_time, max_time) =
                compute_time_windows(our_clock, moves_to_go, our_inc, conf);
            let max_time = Duration::from_millis(max_time);
            let hard_time = Duration::from_millis(hard_time);
            let opt_time = Duration::from_millis(opt_time);

            if Some(best_move) == self.prev_move {
                self.stability += 1;
            } else {
                self.stability = 0;
            }
            self.best_move_nodes_fraction = best_move_nodes_fraction;

            let stability_multiplier = Self::best_move_stability_multiplier(self.stability);
            // retain time added by windows that failed low
            let failed_low_multiplier =
                f64::from(self.failed_low).mul_add(f64::from(conf.fail_low_tm_bonus) / 1000.0, 1.0);
            let forced_move_multiplier = self.forcedness.tm_multiplier(conf);
            let subtree_size_multiplier = self.best_move_nodes_fraction.map_or(1.0, |frac| {
                Self::best_move_subtree_size_multiplier(frac, conf)
            });

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

        self.prev_move = Some(best_move);
        self.prev_score = eval;
    }

    pub fn report_aspiration_fail(&mut self, depth: i32, bound: Bound, conf: &Config) {
        const FAIL_LOW_UPDATE_THRESHOLD: i32 = 0;
        let Some(Dynamic {
            our_clock,
            our_inc,
            moves_to_go,
            ..
        }) = self.limits.dynamic()
        else {
            return;
        };
        if depth >= FAIL_LOW_UPDATE_THRESHOLD && bound == Bound::Upper && self.failed_low < 2 {
            self.failed_low += 1;

            let (opt_time, hard_time, max_time) =
                compute_time_windows(our_clock, moves_to_go, our_inc, conf);
            let max_time = Duration::from_millis(max_time);
            let hard_time = Duration::from_millis(hard_time);
            let opt_time = Duration::from_millis(opt_time);

            let stability_multiplier = self.last_factors[0];
            // calculate the failed low multiplier
            let failed_low_multiplier =
                f64::from(self.failed_low).mul_add(f64::from(conf.fail_low_tm_bonus) / 1000.0, 1.0);
            let forced_move_multiplier = self.forcedness.tm_multiplier(conf);
            let subtree_size_multiplier = self.best_move_nodes_fraction.map_or(1.0, |frac| {
                Self::best_move_subtree_size_multiplier(frac, conf)
            });

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
