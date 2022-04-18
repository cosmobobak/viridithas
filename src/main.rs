#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(dead_code, unused_imports, clippy::missing_const_for_fn)]
use board::Board;
use chessmove::Move;
use definitions::Piece;
use rand::Rng;

/// Rust chess engine built using the VICE video series.
/// 
/// 

use crate::{definitions::Square64, lookups::{PIECE_KEYS, filerank_to_square}};
use crate::lookups::SIDE_KEY;
// #[macro_use]
// extern crate lazy_static;

mod chessmove;
mod bitboard;
mod definitions;
mod board;
mod lookups;
mod attack;

fn sq_attack_by_side(side: u8, board: &Board) {
    println!("Squares attacked by side {}:", side);
    for rank in (0..8).rev() {
        for file in 0..8 {
            let sq = filerank_to_square(file, rank);
            if board.sq_attacked(sq as usize, side) {
                print!("X ");
            } else {
                print!("- ");
            }
        }
        println!();
    }
    println!();
}

fn main() {
    let m = Move::new(6, 12, Piece::WR as u8, Piece::BR as u8, false, false, false);
    println!("{0} {0:x} {0:b}", m.data);
    println!("{:?}", m);
    println!("{}", m);
}
