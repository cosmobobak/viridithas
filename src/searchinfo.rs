use std::{sync::mpsc, time::Instant};

#[allow(clippy::struct_excessive_bools)]
pub struct SearchInfo<'a> {
    /// The starting time of the search.
    pub start_time: Instant,
    /// The ending time of the search.
    pub stop_time: Instant,

    /// The maximum depth of the search.
    pub depth: i32,

    pub time_set: bool,
    pub moves_to_go: usize,
    pub infinite: bool,
    pub nodes: u64,

    /// Signal to quit the search.
    pub quit: bool,
    /// Signal to stop the search.
    pub stopped: bool,

    /// The number of fail-highs found (beta cutoffs).
    pub failhigh: f32,
    /// The number of fail-highs that occured on the first move searched.
    pub failhigh_first: f32,
    /// The highest depth reached (selective depth).
    pub seldepth: i32,

    /// A handle to a receiver for stdin.
    pub stdin_rx: Option<&'a mpsc::Receiver<String>>,
}

impl Default for SearchInfo<'_> {
    fn default() -> Self {
        Self {
            start_time: Instant::now(),
            stop_time: Instant::now() + std::time::Duration::from_secs(1),
            depth: 60,
            time_set: false,
            moves_to_go: 0,
            infinite: false,
            nodes: 0,
            quit: false,
            stopped: false,
            failhigh: 0.0,
            failhigh_first: 0.0,
            seldepth: 0,
            stdin_rx: None,
        }
    }
}

impl<'a> SearchInfo<'a> {
    pub fn clear_for_search(&mut self) {
        self.stopped = false;
        self.nodes = 0;
        self.failhigh = 0.0;
        self.failhigh_first = 0.0;
    }

    pub fn set_stdin(&mut self, stdin_rx: &'a mpsc::Receiver<String>) {
        self.stdin_rx = Some(stdin_rx);
    }

    pub fn set_time_window(&mut self, millis: u64) {
        self.start_time = Instant::now();
        self.stop_time = self.start_time + std::time::Duration::from_millis(millis);
    }

    pub fn check_up(&mut self) {
        if self.time_set
            && Instant::now()
                .checked_duration_since(self.stop_time)
                .is_some()
        {
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
}
