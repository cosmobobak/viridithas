// #![feature(stdarch_x86_avx512)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::missing_const_for_fn)]
#![deny(missing_docs, clippy::undocumented_unsafe_blocks)]

//! Viridithas, a UCI chess engine written in Rust.

#[macro_use]
mod macros;

#[cfg(feature = "datagen")]
mod datagen;

mod bench;
mod chess;
mod cli;
mod cuckoo;
mod errors;
mod evaluation;
mod history;
mod historytable;
mod image;
mod lookups;
mod movepicker;
mod nnue;
mod perft;
mod rng;
mod search;
mod searchinfo;
mod stack;
mod tablebases;
mod term;
mod threadlocal;
mod timemgmt;
mod transpositiontable;
mod uci;
mod util;

#[cfg(feature = "datagen")]
use cli::Subcommands::{Analyse, CountPositions, Datagen, Splat};
use cli::Subcommands::{Bench, Merge, NNUEDryRun, Perft, Quantise, Spsa, Verbatim, VisNNUE};

/// The name of the engine.
pub static NAME: &str = "Viridithas";
/// The version of the engine.
pub static VERSION: &str = env!("CARGO_PKG_VERSION");

fn main() -> anyhow::Result<()> {
    #[cfg(debug_assertions)]
    std::env::set_var("RUST_BACKTRACE", "1");

    if std::env::args_os().len() == 1 {
        // fast path to UCI:
        return uci::main_loop();
    }

    let cli = <cli::Cli as clap::Parser>::parse();

    match cli.subcommand {
        Some(Perft) => perft::gamut(),
        Some(VisNNUE) => nnue::network::visualise_nnue(),
        Some(NNUEDryRun) => nnue::network::dry_run(),
        Some(Quantise { input, output }) => nnue::network::quantise(&input, &output),
        Some(Merge { input, output }) => nnue::network::merge(&input, &output),
        Some(Verbatim { output }) => nnue::network::dump_verbatim(&output),
        #[cfg(feature = "datagen")]
        Some(Analyse { input }) => datagen::dataset_stats(&input),
        #[cfg(feature = "datagen")]
        Some(CountPositions { input }) => datagen::dataset_count(&input),
        Some(Spsa { json }) => {
            if json {
                println!(
                    "{}",
                    search::parameters::Config::default().emit_json_for_spsa()
                );
            } else {
                println!(
                    "{}",
                    search::parameters::Config::default().emit_csv_for_spsa()
                );
            }
            Ok(())
        }
        #[cfg(feature = "datagen")]
        Some(Splat {
            input,
            marlinformat,
            pgn,
            output,
            limit,
            cfg_path,
        }) => {
            if pgn {
                datagen::run_topgn(&input, &output, limit)
            } else {
                datagen::run_splat(&input, &output, cfg_path.as_deref(), marlinformat, limit)
            }
        }
        #[cfg(feature = "datagen")]
        Some(Datagen {
            games,
            threads,
            tbs,
            book,
            depth_limit,
            dfrc,
        }) => datagen::gen_data_main(datagen::DataGenOptionsBuilder {
            games,
            threads,
            tbs,
            book,
            depth_limit,
            dfrc,
        }),
        Some(Bench { depth }) => {
            let nnue_params = nnue::network::NNUEParams::decompress_and_alloc()?;
            let stopped = std::sync::atomic::AtomicBool::new(false);
            let nodes = std::sync::atomic::AtomicU64::new(0);
            let info = searchinfo::SearchInfo::new(&stopped, &nodes);
            uci::bench("openbench", &info.conf, nnue_params, depth)?;
            Ok(())
        }
        None => uci::main_loop(),
    }
}
