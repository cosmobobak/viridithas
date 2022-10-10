use std::{sync::mpsc, time::{Instant, Duration}};

use crate::definitions::depth::Depth;

enum Limit {
    Infinite,
    Depth(Depth),
    FixedMillis(u64),
    FixedNodes(u64),
    Variable {
        time_on_clock: u64,
        increment: u64,
        moves_to_go: Option<u64>,
    }
}

#[allow(clippy::struct_excessive_bools)]
pub struct SearchInfo<'a> {
    /// The starting time of the search.
    pub start_time: Instant,
    /// The ending time of the search.
    pub stop_time: Instant,

    /// The maximum depth of the search.
    pub depth: Depth,

    pub dyntime_allowed: bool,
    pub time_set: bool,
    pub moves_to_go: usize,
    pub infinite: bool,
    pub nodes: u64,
    pub max_time_window: Duration,

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
}

impl Default for SearchInfo<'_> {
    fn default() -> Self {
        Self {
            start_time: Instant::now(),
            stop_time: Instant::now() + std::time::Duration::from_secs(1),
            depth: 60.into(),
            dyntime_allowed: false,
            time_set: false,
            moves_to_go: 0,
            infinite: false,
            nodes: 0,
            max_time_window: std::time::Duration::from_secs(2),
            quit: false,
            stopped: false,
            failhigh: 0,
            failhigh_first: 0,
            seldepth: 0.into(),
            stdin_rx: None,
            print_to_stdout: true,
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
        self.stop_time = self.start_time + std::time::Duration::from_millis(millis);
    }

    pub fn multiply_time_window(&mut self, factor: f64) {
        assert!(self.dyntime_allowed);
        let secs = (self.stop_time - self.start_time).as_secs_f64();
        let new_secs = secs * factor;
        let new_duration = std::time::Duration::from_secs_f64(new_secs);
        self.stop_time = self.start_time + new_duration.min(self.max_time_window);
    }

    pub fn check_up(&mut self) {
        if self.time_set && Instant::now().checked_duration_since(self.stop_time).is_some() {
            self.stopped = true;
        }
        if let Some(Ok(cmd)) = self.stdin_rx.map(mpsc::Receiver::try_recv) {
            self.stopped = true;
            let cmd = cmd.trim();
            if cmd == "quit" {
                self.quit = true;
            }
        };
    }

    pub const fn in_game(&self) -> bool {
        !self.infinite
    }

    pub fn time_since_start(&self) -> Duration {
        Instant::now().checked_duration_since(self.start_time).unwrap_or_default()
    }
}
