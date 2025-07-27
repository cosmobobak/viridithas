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
        if $a > $b {
            $a
        } else {
            $b
        }
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
        if $a < $b {
            $a
        } else {
            $b
        }
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
            static TOTAL: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
            static COUNT: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
            TOTAL.fetch_add($v as i64, std::sync::atomic::Ordering::Relaxed);
            let count = COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if count % 2048 == 0 {
                let total = TOTAL.load(std::sync::atomic::Ordering::Relaxed);
                let avg = total as f64 / count as f64;
                println!("average value of {}: {}", stringify!($v), avg);
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
