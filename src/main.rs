#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(dead_code, unused_imports, clippy::missing_const_for_fn)]
use rand::Rng;

/// Rust chess engine built using the VICE video series.
/// 
/// 

use crate::{definitions::Square64, lookups::PIECE_KEYS};
use crate::lookups::SIDE_KEY;
// #[macro_use]
// extern crate lazy_static;

mod bitboard;
mod definitions;
mod board;
mod lookups;

fn main() {
    let mut board = board::Board::new();
    board.set_from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    println!("{:?}", board);
    board.check_validity();
    println!("test passed");
}
