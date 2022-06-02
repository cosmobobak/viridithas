#![allow(clippy::redundant_pub_crate)]

macro_rules! impossible {
    () => {{
        #[cfg(debug_assertions)]
        {
            panic!("Unreachable code!");
        }
        #[allow(unreachable_code)]
        {
            std::hint::unreachable_unchecked();
        }
    }};
}

pub(crate) use impossible;
