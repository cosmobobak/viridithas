#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![deny(missing_docs)]

//! Viridithas II, a UCI chess engine written in Rust.

#[macro_use]
mod macros;

mod board;
mod chessmove;
mod cli;
mod definitions;
mod epd;
mod errors;
mod historytable;
mod lookups;
mod magic;
mod makemove;
mod nnue;
mod perft;
mod piecelist;
mod piecesquaretable;
mod rng;
mod search;
mod searchinfo;
mod texel;
mod threadlocal;
mod transpositiontable;
mod uci;
mod validate;

/// The name of the engine.
pub static NAME: &str = "Viridithas";
/// The version of the engine.
pub static VERSION: &str = env!("CARGO_PKG_VERSION");

fn main() {
    #[cfg(debug_assertions)]
    std::env::set_var("RUST_BACKTRACE", "1");

    // takes about 3ms to generate the attack tables on boot
    magic::initialise();

    let cli = <cli::Cli as clap::Parser>::parse();

    let eparams =
        cli.eparams.map_or_else(board::evaluation::parameters::EvalParams::default, |p| {
            board::evaluation::parameters::EvalParams::from_file(p).unwrap()
        });

    assert_eq!(cli.merge.len(), 2, "merge requires exactly two paths");
    assert_eq!(cli.jsontobin.len(), 2, "jsontobin requires exactly two paths");

    if cli.gensource {
        return piecesquaretable::tables::printout_pst_source(&eparams.piece_square_tables);
    }

    if cli.perfttest {
        return perft::gamut();
    }

    if let Some(path) = cli.tune {
        return texel::tune(cli.resume, cli.examples, &eparams, cli.limitparams.as_deref(), path);
    }

    if let Some(path) = cli.nnueconversionpath {
        let output_path = cli.output.unwrap_or_else(|| {
            let mut path = path.clone();
            path.set_extension("nnuedata");
            path
        });
        return nnue::convert::evaluate_fens(
            path,
            output_path,
            nnue::convert::Format::OurTexel,
            cli.nnuedepth,
            true,
            true,
        )
        .unwrap();
    } else if let Some(path) = cli.nnuereanalysepath {
        let output_path = cli.output.unwrap_or_else(|| {
            let mut path = path.clone();
            path.set_extension("nnuedata");
            path
        });
        return nnue::convert::evaluate_fens(
            path,
            output_path,
            nnue::convert::Format::Marlinflow,
            cli.nnuedepth,
            true,
            true,
        )
        .unwrap();
    } else if let Some(path) = cli.dedup {
        let output_path = cli.output.unwrap_or_else(|| {
            let mut path = path.clone();
            path.set_extension("nnuedata");
            path
        });
        return nnue::convert::dedup(path, output_path).unwrap();
    } else if let [path_1, path_2] = cli.merge.as_slice() {
        let output_path = cli.output.unwrap_or_else(|| {
            // create merged.nnuedata in the current directory
            let mut path = std::path::PathBuf::from(".");
            path.push("merged.nnuedata");
            path
        });
        return nnue::convert::merge(path_1, path_2, output_path).unwrap();
    }

    if cli.info {
        println!("name: {NAME}");
        println!("version: {VERSION}");
        println!(
            "number of evaluation parameters: {}",
            board::evaluation::parameters::EvalParams::default().vectorise().len()
        );
        println!(
            "size of a transposition table entry: {}",
            std::mem::size_of::<transpositiontable::TTEntry>()
        );
        return;
    }

    if cli.visparams {
        println!("{eparams}");
        println!("{}", crate::search::parameters::SearchParams::default());
        return;
    }

    if cli.vispsqt {
        return piecesquaretable::render_pst_table(&eparams.piece_square_tables);
    }

    if let Some(epd_path) = cli.epdpath {
        return epd::gamut(epd_path, eparams, cli.epdtime);
    }

    if let [json_path, bin_path] = cli.jsontobin.as_slice() {
        return nnue::convert_json_to_binary(json_path, bin_path);
    }

    uci::main_loop(eparams);
}
