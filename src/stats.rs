// SPDX-License-Identifier: GPL-3.0-only

//! Statistics tracking for search value distributions.
//!
//! This module provides infrastructure for tracking the distribution of values
//! during search.
//!
//! Enabled only with the `stats` feature.

use std::{
    fmt::Write,
    process::{Command, Stdio},
    sync::{
        OnceLock,
        atomic::{AtomicI64, AtomicU64, Ordering},
    },
};

use histogram::AtomicHistogram;
use linkme::distributed_slice;

// Bucket configuration for the per-`TrackedValue` histograms.
//
/// `grouping_power = 1` gives 2 sub-buckets per power of two.
const HIST_GROUPING_POWER: u8 = 1;
/// `max_value_power = 32` covers absolute values up to 2³².
const HIST_MAX_VALUE_POWER: u8 = 32;

/// Construct a fresh `AtomicHistogram` with the configuration above. Used by
/// the `OnceLock`s inside `TrackedValue` to lazily initialise on first record.
fn new_histogram() -> AtomicHistogram {
    AtomicHistogram::new(HIST_GROUPING_POWER, HIST_MAX_VALUE_POWER)
        .expect("valid histogram configuration")
}

/// A tracked value with statistics collected via atomic operations.
///
/// Bucketing is delegated to the `histogram` crate's `AtomicHistogram`. Because
/// that type only handles `u64`, we keep two histograms per tracked value: one
/// for non-negative samples and one for the absolute value of negative samples.
pub struct TrackedValue {
    /// Name of the tracked value (`file:line expr`).
    pub name: &'static str,
    /// Number of samples.
    pub count: AtomicU64,
    /// Sum of all values.
    pub total: AtomicI64,
    /// Sum of absolute values.
    pub total_abs: AtomicU64,
    /// Sum of squared values (may overflow for large values, but useful for stddev).
    pub total_sq: AtomicU64,
    /// Minimum value seen.
    pub min: AtomicI64,
    /// Maximum value seen.
    pub max: AtomicI64,
    /// Histogram of non-negative samples.
    pub pos_hist: OnceLock<AtomicHistogram>,
    /// Histogram of `|v|` for negative samples.
    pub neg_hist: OnceLock<AtomicHistogram>,
}

impl TrackedValue {
    /// Create a new tracked value.
    #[must_use]
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            count: AtomicU64::new(0),
            total: AtomicI64::new(0),
            total_abs: AtomicU64::new(0),
            total_sq: AtomicU64::new(0),
            min: AtomicI64::new(i64::MAX),
            max: AtomicI64::new(i64::MIN),
            pos_hist: OnceLock::new(),
            neg_hist: OnceLock::new(),
        }
    }

    /// Record a value. Called from the `track!` macro.
    #[inline]
    pub fn record(&self, v: i64) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.total.fetch_add(v, Ordering::Relaxed);
        self.total_abs
            .fetch_add(v.unsigned_abs(), Ordering::Relaxed);
        #[allow(clippy::cast_sign_loss)]
        self.total_sq
            .fetch_add((v.saturating_mul(v)) as u64, Ordering::Relaxed);
        self.min.fetch_min(v, Ordering::Relaxed);
        self.max.fetch_max(v, Ordering::Relaxed);

        // values outside [0, 2^HIST_MAX_VALUE_POWER) overflow the bucket range.
        let hist = if v >= 0 {
            self.pos_hist.get_or_init(new_histogram)
        } else {
            self.neg_hist.get_or_init(new_histogram)
        };
        let _ = hist.increment(v.unsigned_abs());
    }
}

/// Distributed slice collecting all tracked values.
/// `linkme` is so goddamn cool.
#[distributed_slice]
pub static TRACKED_VALUES: [TrackedValue];

/// A populated histogram bucket, with both bounds carried as signed integers so
/// that negative buckets can be emitted directly.
struct BucketEntry {
    start: i64,
    end: i64,
    count: u64,
}

/// Snapshot the positive and negative histograms of `tv` into a single ordered
/// list of non-empty buckets covering negative samples first, then non-negative.
fn collect_buckets(tv: &TrackedValue) -> Vec<BucketEntry> {
    #![allow(clippy::cast_possible_wrap)]

    let mut buckets = Vec::new();

    let nh = tv.neg_hist.get().map(AtomicHistogram::load);
    for b in nh.iter().flatten().filter(|x| x.count() != 0) {
        buckets.push(BucketEntry {
            start: -(b.end() as i64),
            end: -(b.start() as i64),
            count: b.count(),
        });
    }
    // ideally we’d just place a `.rev()` on the flatten, but
    // histogram hasn’t implemented DoubleEndedIterator :(
    buckets.reverse();

    let ph = tv.pos_hist.get().map(AtomicHistogram::load);
    for b in ph.iter().flatten().filter(|x| x.count() != 0) {
        buckets.push(BucketEntry {
            start: b.start() as i64,
            end: b.end() as i64,
            count: b.count(),
        });
    }

    buckets
}

/// Dump all tracked statistics and invoke the plotter script.
pub fn dump_and_plot() {
    if TRACKED_VALUES.is_empty() {
        return;
    }

    // Build JSON output
    let mut json = String::from("[\n");
    for (i, tv) in TRACKED_VALUES
        .iter()
        .filter(|tv| tv.count.load(Ordering::Relaxed) != 0)
        .enumerate()
    {
        let count = tv.count.load(Ordering::Relaxed);
        let total = tv.total.load(Ordering::Relaxed);
        let total_abs = tv.total_abs.load(Ordering::Relaxed);
        let total_sq = tv.total_sq.load(Ordering::Relaxed);
        let min = tv.min.load(Ordering::Relaxed);
        let max = tv.max.load(Ordering::Relaxed);

        #[allow(clippy::cast_precision_loss)]
        let avg = total as f64 / count as f64;
        #[allow(clippy::cast_precision_loss)]
        let avg_abs = total_abs as f64 / count as f64;
        #[allow(clippy::cast_precision_loss)]
        let variance = avg.mul_add(-avg, total_sq as f64 / count as f64);
        let stddev = variance.max(0.0).sqrt();

        let buckets = collect_buckets(tv);

        if i > 0 {
            json.push_str(",\n");
        }
        write!(
            json,
            r#"  {{
    "name": {:?},
    "count": {count},
    "total": {total},
    "avg": {avg},
    "avg_abs": {avg_abs},
    "stddev": {stddev},
    "min": {min},
    "max": {max},
    "buckets": ["#,
            tv.name
        )
        .unwrap();
        for (j, b) in buckets.iter().enumerate() {
            if j > 0 {
                json.push(',');
            }
            write!(
                json,
                r#"{{"start":{},"end":{},"count":{}}}"#,
                b.start, b.end, b.count
            )
            .unwrap();
        }
        json.push_str("]\n  }");
    }
    json.push_str("\n]\n");

    // run the plotter
    let plotter_path = concat!(env!("CARGO_MANIFEST_DIR"), "/scripts/plotter.py");
    let result = Command::new("uv")
        .args(["run", plotter_path, "--min-bucket-fraction", "0.01"])
        .stdin(Stdio::piped())
        .spawn();

    match result {
        Ok(mut child) => {
            if let Some(mut stdin) = child.stdin.take() {
                let _ = std::io::Write::write_all(&mut stdin, json.as_bytes());
            }

            let _ = child.wait();
        }
        Err(e) => {
            eprintln!("info string failed to run plotter: {e}");
            eprintln!("info string dumping stats to stdout instead:");
            print!("{json}");
        }
    }
}
