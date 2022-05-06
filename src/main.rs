#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(
    dead_code,
    clippy::if_not_else
)]

mod errors;
mod pvtable;
mod attack;
mod bitboard;
mod board;
mod chessmove;
mod definitions;
mod evaluation;
mod lookups;
mod makemove;
mod movegen;
mod perft;
mod validate;
mod searchinfo;
mod search;
mod piecesquaretable;
mod uci;

fn main() {
    uci::main_loop();
}