use std::{time::{Instant, Duration}, sync::atomic::{AtomicBool, Ordering}};

use crate::{definitions::depth::Depth, chessmove::Move, search::parameters::SearchParams};

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
        const MOVE_OVERHEAD: u64 = 10;
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
}