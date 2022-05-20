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
mod tuning;

pub const NAME: &str = "Viridithas";

fn main() {
    #[cfg(debug_assertions)]
    std::env::set_var("RUST_BACKTRACE", "1");

    // tuning::annotate_positions("chessData.csv", "TRAINING_DATA.csv", 1_000_000, 4);
    uci::main_loop();
}
