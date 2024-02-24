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

    if let Some(config) = cli.datagen {
        #[cfg(feature = "datagen")]
        return datagen::gen_data_main(config.as_deref());
        #[cfg(not(feature = "datagen"))]
        {
            std::mem::drop(config);
            println!("datagen feature not enabled (compile with --features datagen)");
            return;
        }
    }

    if let Some(input) = cli.splat {
        let Some(output) = cli.output else {
            println!("Output path required for splatting (use --output)");
            return;
        };
        return datagen::run_splat(&input, &output, true, cli.marlinformat, cli.limit);
    }

    if let Some(input) = cli.topgn {
        let Some(output) = cli.output else {
            println!("Output path required for PGN conversion (use --output)");
            return;
        };
        return datagen::run_topgn(&input, &output, cli.limit);
    }

    if let Some(data_path) = cli.dataset_stats {
        return datagen::dataset_stats(&data_path);
    }

    if cli.perfttest {
        return perft::gamut();
    }

    if cli.spsajson {
        return println!("{}", Config::default().emit_json_for_spsa());
    }

    if cli.visnnue {
        return network::visualise_nnue();
    }

    uci::main_loop(cli.bench.is_some());
}
