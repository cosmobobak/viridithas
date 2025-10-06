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

/// Collect statistics on the average value of a variable.
#[allow(unused_macros)]
macro_rules! track {
    ($v:expr) => {{
        {
            #![allow(
                clippy::cast_lossless,
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                clippy::cast_possible_wrap
            )]
            static COUNT: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
            static TOTAL: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
            static TOTAL_ABS: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
            static TOTAL_SQ: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
            static MAX: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(i64::MIN);
            static MIN: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(i64::MAX);
            let v = $v as i64;
            MAX.fetch_max(v, std::sync::atomic::Ordering::Relaxed);
            MIN.fetch_min(v, std::sync::atomic::Ordering::Relaxed);
            TOTAL.fetch_add(v, std::sync::atomic::Ordering::Relaxed);
            TOTAL_ABS.fetch_add(v.abs(), std::sync::atomic::Ordering::Relaxed);
            TOTAL_SQ.fetch_add(v * v, std::sync::atomic::Ordering::Relaxed);
            let count = COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if count % 8192 == 0 {
                let total = TOTAL.load(std::sync::atomic::Ordering::Relaxed);
                let squares_total = SQUARES_TOTAL.load(std::sync::atomic::Ordering::Relaxed);
                let avg = total as f64 / count as f64;
                println!("average value of {}: {}", stringify!($v), avg);
                let total_abs = TOTAL_ABS.load(std::sync::atomic::Ordering::Relaxed);
                let avg_abs = total_abs as f64 / count as f64;
                println!("average absolute value of {}: {}", stringify!($v), avg_abs);
                let total_sq = TOTAL_SQ.load(std::sync::atomic::Ordering::Relaxed);
                let variance = total_sq as f64 / count as f64 - avg * avg;
                let stddev = variance.sqrt();
                println!("stddev of {}: {}", stringify!($v), stddev);
                println!(
                    "min/max value of {}: {}/{}",
                    stringify!($v),
                    MIN.load(std::sync::atomic::Ordering::Relaxed),
                    MAX.load(std::sync::atomic::Ordering::Relaxed)
                );
            }
            // pass-through
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
