use std::sync::atomic::AtomicBool;

pub mod board;
pub mod chessmove;
pub mod fen;
mod magic;
pub mod piece;
pub mod piecelayout;
pub mod quick;
pub mod squareset;
pub mod types;

pub static CHESS960: AtomicBool = AtomicBool::new(false);
