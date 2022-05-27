#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::if_not_else)]

mod attack;
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
mod tuning;
mod uci;
mod validate;

pub const NAME: &str = "Viridithas 2.1.0dev";

fn main() {
    #[cfg(debug_assertions)]
    std::env::set_var("RUST_BACKTRACE", "1");

    uci::main_loop();

    // tuning::annotate_positions("chessData.csv", "TRAINING_DATA.csv", 1_000_000, 4);
    // perft::gamut();
}
