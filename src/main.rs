#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]
#![deny(missing_docs)]

//! Viridithas, a UCI chess engine written in Rust.

use crate::{
    board::evaluation::parameters::EvalParams,
    nnue::{convert, network},
    search::parameters::SearchParams,
};

#[macro_use]
mod macros;

mod bench;
mod board;
mod chessmove;
mod cli;
mod definitions;
mod epd;
mod errors;
mod historytable;
mod image;
mod lookups;
mod magic;
mod makemove;
mod nnue;
mod perft;
mod piece;
mod piecesquaretable;
mod rng;
mod search;
mod searchinfo;
mod tablebases;
mod texel;
mod threadlocal;
mod timemgmt;
mod transpositiontable;
mod uci;
mod squareset;

#[cfg(feature = "datagen")]
mod datagen;

/// The name of the engine.
pub static NAME: &str = "Viridithas";
/// The version of the engine.
pub static VERSION: &str = env!("CARGO_PKG_VERSION");

fn main() {
    #[cfg(debug_assertions)]
    std::env::set_var("RUST_BACKTRACE", "1");

    // takes about 3ms to generate the attack tables on boot
    magic::initialise();

    if std::env::args_os().len() == 1 {
        // fast path to UCI:
        return uci::main_loop(EvalParams::default(), false);
    }

    let cli = <cli::Cli as clap::Parser>::parse();

    if let Some(config) = cli.datagen {
        #[cfg(feature = "datagen")]
        return datagen::gen_data_main(config.as_deref());
        #[cfg(not(feature = "datagen"))]
        {
            std::mem::drop(config);
            println!("datagen feature not enabled");
            return;
        }
    }

    let eparams = cli.eparams.as_deref().map_or_else(EvalParams::default, |p| {
        EvalParams::from_file(p).expect("failed to load evaluation parameters")
    });

    assert!([0, 2].contains(&cli.merge.len()), "merge requires exactly two paths");

    if cli.gensource {
        return piecesquaretable::tables::printout_pst_source(&eparams.piece_square_tables);
    }

    if cli.perfttest {
        return perft::gamut();
    }

    if let Some(path) = cli.tune {
        return texel::tune(cli.resume, cli.examples, &eparams, cli.limitparams.as_deref(), path);
    }

    if let Some(input_file) = cli.nnueconversionpath {
        let output_file = cli.output.unwrap_or_else(|| {
            let mut path = input_file.clone();
            path.set_extension("nnuedata");
            path
        });
        return convert::evaluate_fens(
            input_file,
            output_file,
            convert::Format::OurTexel,
            cli.nnuedepth,
            true,
            cli.nnuefornnue,
        )
        .unwrap();
    } else if let Some(path) = cli.nnuereanalysepath {
        let output_path = cli.output.unwrap_or_else(|| {
            let mut path = path.clone();
            path.set_extension("nnuedata");
            path
        });
        return convert::evaluate_fens(
            path,
            output_path,
            convert::Format::Marlinflow,
            cli.nnuedepth,
            true,
            cli.nnuefornnue,
        )
        .unwrap();
    } else if let Some(path) = cli.dedup {
        let output_path = cli.output.unwrap_or_else(|| {
            let mut path = path.clone();
            path.set_extension("nnuedata");
            path
        });
        return convert::dedup(path, output_path).unwrap();
    } else if let [path_1, path_2] = cli.merge.as_slice() {
        let output_path = cli.output.unwrap_or_else(|| {
            // create merged.nnuedata in the current directory
            let mut path = std::path::PathBuf::from(".");
            path.push("merged.nnuedata");
            path
        });
        return convert::merge(path_1, path_2, output_path).unwrap();
    }

    if cli.info {
        return lookups::info_dump();
    }

    if cli.visparams {
        println!("{eparams}");
        println!("{}", SearchParams::default());
        return;
    }

    if cli.vispsqt {
        return piecesquaretable::render_pst_table(&eparams.piece_square_tables);
    }

    if let Some(epd_path) = cli.epdpath.as_deref() {
        return epd::gamut(epd_path, &eparams, &cli);
    }

    if cli.visnnue {
        return network::visualise_nnue();
    }

    uci::main_loop(eparams, cli.bench.is_some());
}
