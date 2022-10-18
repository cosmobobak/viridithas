use std::{
    sync::mpsc,
    time::{Duration, Instant},
};

use crate::definitions::depth::Depth;

#[derive(PartialEq, Eq, Clone, Debug)]
pub enum SearchLimit {
    Infinite,
    Depth(Depth),
    Time(u64),
    Nodes(u64),
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

    pub fn compute_time_windows(our_clock: u64, _moves_to_go: u64, our_inc: u64) -> (u64, u64) {
        let computed_time_window = our_clock / 20 + our_inc / 2;
        let time_window = computed_time_window.min(our_clock);
        let max_time_window = (time_window * 5 / 2).min(our_clock);
        (time_window, max_time_window)
    }
}

#[allow(clippy::struct_excessive_bools)]
pub struct SearchInfo<'a> {
    /// The starting time of the search.
    pub start_time: Instant,
    /// The number of nodes searched.
    pub nodes: u64,
    /// Signal to quit the search.
    pub quit: bool,
    /// Signal to stop the search.
    pub stopped: bool,
    /// The number of fail-highs found (beta cutoffs).
    pub failhigh: u64,
    /// The number of fail-highs that occured on the first move searched.
    pub failhigh_first: u64,
    /// The highest depth reached (selective depth).
    pub seldepth: Depth,
    /// A handle to a receiver for stdin.
    pub stdin_rx: Option<&'a mpsc::Receiver<String>>,
    /// Whether to print the search info to stdout.
    pub print_to_stdout: bool,
    /// Form of the search limit.
    pub limit: SearchLimit,
}

impl Default for SearchInfo<'_> {
    fn default() -> Self {
        Self {
            start_time: Instant::now(),
            nodes: 0,
            quit: false,
            stopped: false,
            failhigh: 0,
            failhigh_first: 0,
            seldepth: 0.into(),
            stdin_rx: None,
            print_to_stdout: true,
            limit: SearchLimit::default(),
        }
    }
}

impl<'a> SearchInfo<'a> {
    pub fn setup_for_search(&mut self) {
        self.stopped = false;
        self.nodes = 0;
        self.failhigh = 0;
        self.failhigh_first = 0;
    }

    pub fn set_stdin(&mut self, stdin_rx: &'a mpsc::Receiver<String>) {
        self.stdin_rx = Some(stdin_rx);
    }

    pub fn set_time_window(&mut self, millis: u64) {
        self.start_time = Instant::now();
        match &mut self.limit {
            SearchLimit::Dynamic { time_window, .. } => {
                *time_window = millis;
            }
            SearchLimit::Time(inner_ms) => {
                *inner_ms = millis;
            }
            other => panic!("Unexpected search limit: {:?}", other),
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
            other => panic!("Unexpected search limit: {:?}", other),
        }
    }

    pub fn check_up(&mut self) {
        match self.limit {
            SearchLimit::Depth(_) | SearchLimit::Infinite => {}
            SearchLimit::Nodes(nodes) => {
                if self.nodes >= nodes {
                    self.stopped = true;
                }
            }
            SearchLimit::Time(time_window) | SearchLimit::Dynamic { time_window, .. } => {
                let elapsed = self.start_time.elapsed();
                // this cast is safe to do, because u64::MAX milliseconds is 585K centuries.
                #[allow(clippy::cast_possible_truncation)]
                let elapsed_millis = elapsed.as_millis() as u64;
                if elapsed_millis >= time_window {
                    self.stopped = true;
                }
            }
        }
        if let Some(Ok(cmd)) = self.stdin_rx.map(mpsc::Receiver::try_recv) {
            self.stopped = true;
            let cmd = cmd.trim();
            if cmd == "quit" {
                self.quit = true;
            }
        };
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
