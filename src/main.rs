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
mod transpositiontable;
mod vecset;
mod tuning;

fn main() {
    // let tfen = "5r1k/6bP/4p3/2P1P1q1/PnP1R3/6B1/4Q2P/3NK3 b - - 7 45";
    // if let Ok(v) = tuning::eval_vec_for_fen(tfen, &mut board::Board::new()) { 
    //     println!("evec: {v:?}");
    // }
    // println!("eval: {}", board::Board::from_fen(tfen).unwrap().evaluate());
    // tuning::annotate_positions("SF_EVALS.csv", "TRAINING_DATA.csv");
    uci::main_loop();
}