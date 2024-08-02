// #![feature(stdarch_x86_avx512)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![deny(missing_docs, clippy::undocumented_unsafe_blocks)]

//! Viridithas, a UCI chess engine written in Rust.

#[macro_use]
mod macros;

#[cfg(feature = "datagen")]
mod datagen;

mod bench;
mod board;
mod chessmove;
mod cli;
mod cuckoo;
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
mod stack;
mod tablebases;
mod term;
mod threadlocal;
mod timemgmt;
mod transpositiontable;
mod uci;
mod util;

use cli::Subcommands::{Bench, Perft, Spsa, VisNNUE};
#[cfg(feature = "datagen")]
use cli::Subcommands::{Analyse, CountPositions, Datagen, Splat};

/// The name of the engine.
pub static NAME: &str = "Viridithas";
/// The version of the engine.
pub static VERSION: &str = env!("CARGO_PKG_VERSION");

fn main() -> anyhow::Result<()> {
    #[cfg(debug_assertions)]
    std::env::set_var("RUST_BACKTRACE", "1");

    if std::env::args_os().len() == 1 {
        // fast path to UCI:
        return uci::main_loop(false);
    }

    let cli = <cli::Cli as clap::Parser>::parse();

    match cli.subcommand {
        Some(Perft) => perft::gamut(),
        Some(VisNNUE) => nnue::network::visualise_nnue(),
        #[cfg(feature = "datagen")]
        Some(Analyse { input }) => datagen::dataset_stats(&input),
        #[cfg(feature = "datagen")]
        Some(CountPositions { input }) => datagen::dataset_count(&input),
        Some(Spsa { json }) => {
            if json {
                println!("{}", search::parameters::Config::default().emit_json_for_spsa());
            } else {
                println!("{}", search::parameters::Config::default().emit_csv_for_spsa());
            }
            Ok(())
        }
        #[cfg(feature = "datagen")]
        Some(Splat { input, marlinformat, pgn, output, limit, cfg_path }) => {
            if pgn {
                datagen::run_topgn(&input, &output, limit)
            } else {
                datagen::run_splat(&input, &output, cfg_path.as_deref(), marlinformat, limit)
            }
        }
        #[cfg(feature = "datagen")]
        Some(Datagen { games, threads, tbs, depth_limit, dfrc }) => {
            datagen::gen_data_main(datagen::DataGenOptionsBuilder {
                num_games: games,
                num_threads: threads,
                tablebases_path: tbs,
                use_depth: depth_limit,
                generate_dfrc: dfrc,
            })
        }
        Some(Bench) => uci::main_loop(true),
        None => uci::main_loop(false),
    }
}
