#![allow(clippy::redundant_pub_crate)]

macro_rules! inconceivable {
    () => {{
        #[cfg(debug_assertions)]
        {
            panic!("That word you use - I do not think it means what you think it means.");
        }
        #[allow(unreachable_code)]
        {
            std::hint::unreachable_unchecked();
        }
    }};
}

macro_rules! max {
    ($a:expr, $b:expr) => {
        if $a > $b {
            $a
        } else {
            $b
        }
    };
}

pub(crate) use inconceivable;
