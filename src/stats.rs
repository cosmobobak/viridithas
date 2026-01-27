//! Statistics tracking for search value distributions.
//!
//! This module provides infrastructure for tracking the distribution of values
//! during search, such as pruning margins, reductions, and other dynamic parameters.
//! Enabled only with the `stats` feature.

use std::{
    fmt::Write,
    process::{Command, Stdio},
    sync::atomic::{AtomicI64, AtomicU64, Ordering},
};

use linkme::distributed_slice;

/// Number of histogram buckets (64 negative + 64 non-negative).
pub const NUM_BUCKETS: usize = 128;

/// A tracked value with statistics collected via atomic operations.
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
    /// Histogram buckets (√2-scale, buckets 0-63 negative, 64-127 non-negative).
    pub histogram: [AtomicU64; NUM_BUCKETS],
}

impl TrackedValue {
    /// Create a new tracked value.
    #[must_use]
    pub const fn new(name: &'static str) -> Self {
        // can't use array::from_fn in const context :(
        Self {
            name,
            count: AtomicU64::new(0),
            total: AtomicI64::new(0),
            total_abs: AtomicU64::new(0),
            total_sq: AtomicU64::new(0),
            min: AtomicI64::new(i64::MAX),
            max: AtomicI64::new(i64::MIN),
            histogram: [const { AtomicU64::new(0) }; NUM_BUCKETS],
        }
    }

    /// Record a value. Called from the `track!` macro.
    #[inline]
    pub fn record(&self, v: i64) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.total.fetch_add(v, Ordering::Relaxed);
        #[allow(clippy::cast_sign_loss)]
        self.total_abs
            .fetch_add(v.unsigned_abs(), Ordering::Relaxed);
        #[allow(clippy::cast_sign_loss)]
        self.total_sq
            .fetch_add((v.saturating_mul(v)) as u64, Ordering::Relaxed);
        self.min.fetch_min(v, Ordering::Relaxed);
        self.max.fetch_max(v, Ordering::Relaxed);

        let bucket = Self::bucket_index(v);
        self.histogram[bucket].fetch_add(1, Ordering::Relaxed);
    }

    /// Compute the bucket index for a value.
    /// Buckets 0-63: negative values (0 = most negative, 63 = -1)
    /// Bucket 64: zero
    /// Buckets 65-127: positive values (65 = 1, 127 = largest positive)
    #[inline]
    fn bucket_index(v: i64) -> usize {
        if v == 0 {
            return 64;
        }

        let v_abs = v.unsigned_abs();

        // compute floor(2 * log2(v_abs)) = floor(log2(v_abs²))
        // for v_abs up to 2^32, v_abs² fits in u64
        #[allow(clippy::cast_possible_truncation)]
        let fine_log = if v_abs <= (1u64 << 32) {
            let sq = v_abs * v_abs;
            (63 - sq.leading_zeros()) as usize
        } else {
            2 * (63 - v_abs.leading_zeros()) as usize
        };

        if v > 0 {
            (65 + fine_log).min(127)
        } else {
            63_usize.saturating_sub(fine_log)
        }
    }
}

/// Distributed slice collecting all tracked values.
/// `linkme` is so goddamn cool.
#[distributed_slice]
pub static TRACKED_VALUES: [TrackedValue];

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

        // collect histogram
        let histogram = tv
            .histogram
            .iter()
            .map(|b| b.load(Ordering::Relaxed))
            .collect::<Vec<_>>();

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
    "histogram": {histogram:?}
  }}"#,
            tv.name
        )
        .unwrap();
    }
    json.push_str("\n]\n");

    // run the plotter
    let plotter_path = concat!(env!("CARGO_MANIFEST_DIR"), "/scripts/plotter.py");
    let result = Command::new("uv")
        .args(["run", plotter_path])
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
