use std::sync::atomic::AtomicBool;

pub mod board;
pub mod chessmove;
pub mod piece;
pub mod piecelayout;
pub mod squareset;
pub mod types;

pub static CHESS960: AtomicBool = AtomicBool::new(false);
