#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(
    dead_code,
    clippy::if_not_else
)]

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

fn main() {
    perft::run_test(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        6,
    );
}
