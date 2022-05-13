#![warn(clippy::all, clippy::pedantic, clippy::nursery)]

mod attack;
mod bitboard;
mod board;
mod chessmove;
mod definitions;
mod errors;
mod lookups;
mod makemove;
mod perft;
mod piecelist;
mod piecesquaretable;
mod search;
mod searchinfo;
mod transpositiontable;
mod uci;
mod validate;

pub const NAME: &str = "Viridithas";

fn main() {
    #[cfg(debug_assertions)]
    std::env::set_var("RUST_BACKTRACE", "1");

    uci::main_loop();
}
