#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]
#![deny(missing_docs)]

//! Viridithas, a UCI chess engine written in Rust.

use crate::{nnue::network, search::parameters::Config};

#[macro_use]
mod macros;

mod bench;
mod board;
mod chessmove;
mod cli;
mod errors;
mod historytable;
mod image;
mod lookups;
mod magic;
mod makemove;
mod nnue;
mod perft;
mod piece;
mod rng;
mod search;
mod searchinfo;
mod squareset;
mod tablebases;
mod threadlocal;
mod timemgmt;
mod transpositiontable;
mod uci;
mod util;

mod datagen;

/// The name of the engine.
pub static NAME: &str = "Viridithas";
/// The version of the engine.
pub static VERSION: &str = env!("CARGO_PKG_VERSION");

fn main() {
    #[cfg(debug_assertions)]
    std::env::set_var("RUST_BACKTRACE", "1");

    if std::env::args_os().len() == 1 {
        // fast path to UCI:
        return uci::main_loop(false);
    }

    let cli = <cli::Cli as clap::Parser>::parse();

    match cli.subcommand {
        Some(cli::Subcommands::Perft) => perft::gamut(),
        Some(cli::Subcommands::VisNNUE) => network::visualise_nnue(),
        Some(cli::Subcommands::Analyse { input }) => datagen::dataset_stats(&input),
        Some(cli::Subcommands::CountPositions { input }) => datagen::dataset_count(&input),
        Some(cli::Subcommands::Spsa { json }) => {
            if json {
                println!("{}", Config::default().emit_json_for_spsa());
            } else {
                println!("{}", Config::default().emit_csv_for_spsa());
            }
        }
        Some(cli::Subcommands::Splat { input, marlinformat, pgn, output, limit }) => {
            if pgn {
                datagen::run_topgn(&input, &output, limit);
            } else {
                datagen::run_splat(&input, &output, true, marlinformat, limit);
            };
        }
        Some(cli::Subcommands::Datagen { games, threads, tbs, depth_limit, dfrc }) => {
            datagen::gen_data_main(datagen::DataGenOptionsBuilder {
                num_games: games,
                num_threads: threads,
                tablebases_path: tbs,
                use_depth: depth_limit,
                generate_dfrc: dfrc,
            });
        }
        Some(cli::Subcommands::Bench) => uci::main_loop(true),
        None => uci::main_loop(false),
    }
}
