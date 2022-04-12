#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(dead_code, unused_imports, clippy::missing_const_for_fn)]
use rand::Rng;

/// Rust chess engine built using the VICE video series.
/// 
/// 

use crate::definitions::{print_bb, Square64};
use crate::lookups::SIDE_KEY;
// #[macro_use]
// extern crate lazy_static;

mod definitions;
mod board;
mod lookups;

fn main() {
    let mut board = board::Board::new();
    println!("{}", board);
    board.set_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    println!("{}", board);
    board.set_from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
    println!("{}", board);
    board.set_from_fen("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2");
    println!("{}", board);
    board.set_from_fen("rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2");
    println!("{}", board);
}
