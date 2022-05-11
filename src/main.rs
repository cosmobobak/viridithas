#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(dead_code, clippy::if_not_else)]

mod attack;
mod bitboard;
mod board;
mod chessmove;
mod definitions;
mod errors;
mod evaluation;
mod lookups;
mod makemove;
mod movegen;
mod perft;
mod piecelist;
mod piecesquaretable;
mod pvtable;
mod search;
mod searchinfo;
mod transpositiontable;
mod uci;
mod validate;

fn main() {
    uci::main_loop();
}
