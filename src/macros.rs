macro_rules! max {
    ($a:expr) => {{
        let mut idx = 0;
        let mut max = $a[0];
        while idx < $a.len() {
            max = max!(max, $a[idx]);
            idx += 1;
        }
        max
    }};
    ($a:expr, $b:expr) => {
        if $a > $b { $a } else { $b }
    };
}

macro_rules! min {
    ($a:expr) => {{
        let mut idx = 0;
        let mut min = $a[0];
        while idx < $a.len() {
            min = min!(min, $a[idx]);
            idx += 1;
        }
        min
    }};
    ($a:expr, $b:expr) => {
        if $a < $b { $a } else { $b }
    };
}

/// Collect statistics on the distribution of a variable.
///
/// When the `stats` feature is enabled, this macro records the value into a
/// global registry with histograms and summary statistics. Call
/// `crate::stats::dump_and_plot()` to visualize all tracked values.
///
/// When the `stats` feature is disabled, this is a no-op identity function.
#[cfg(feature = "stats")]
#[allow(unused_macros)]
macro_rules! track {
    ($v:expr) => {{
        #[allow(clippy::cast_possible_wrap)]
        {
            #[linkme::distributed_slice($crate::stats::TRACKED_VALUES)]
            static ENTRY: $crate::stats::TrackedValue = $crate::stats::TrackedValue::new(
                concat!(file!(), ":", line!(), " ", stringify!($v)),
            );
            let value = $v;
            ENTRY.record(value as i64);
            value
        }
    }};
}

/// No-op version when stats feature is disabled.
#[cfg(not(feature = "stats"))]
#[allow(unused_macros)]
macro_rules! track {
    ($v:expr) => {{
        #[allow(unused)]
        {
            $v
        }
    }};
}

macro_rules! include_bytes_aligned {
    ($path:literal) => {{
        // this assignment is made possible by CoerceUnsized
        static ALIGNED: &$crate::util::Align64<[u8]> = &Align64(*include_bytes!($path));

        &ALIGNED.0
    }};
}
