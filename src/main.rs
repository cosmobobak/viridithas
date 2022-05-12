#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::if_not_else)]

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
mod search;
mod searchinfo;
mod transpositiontable;
mod uci;
mod validate;

fn main() {
    #[cfg(debug_assertions)]
    std::env::set_var("RUST_BACKTRACE", "1");

    println!("btyes: {}", transpositiontable::TT_ENTRY_SIZE);
    println!("default size: {}", transpositiontable::DEFAULT_TABLE_SIZE);

    uci::main_loop();
}
