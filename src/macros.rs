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

pub(crate) use inconceivable;
