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
mod perft;
mod piecelist;
mod piecesquaretable;
mod rng;
mod search;
mod searchinfo;
mod texel;
mod transpositiontable;
mod uci;
mod validate;
mod nnue;
mod threadlocal;

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
        return nnue::convert::wdl_to_nnue(path, output_path).unwrap();
    }

    if cli.info {
        println!("name: {NAME}");
        println!("version: {VERSION}");
        println!(
            "number of evaluation parameters: {}",
            board::evaluation::parameters::EvalParams::default().vectorise().len()
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

    if cli.nnuejsonconversion {
        let mut json_path = String::new();
        print!("Enter the path to the JSON file: ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        std::io::stdin().read_line(&mut json_path).unwrap();
        let json_path = json_path.trim();
        let mut nnue_path = String::new();
        print!("Enter the path for the NNUE binary file: ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        std::io::stdin().read_line(&mut nnue_path).unwrap();
        let nnue_path = nnue_path.trim();
        return nnue::convert_json_to_binary(json_path, nnue_path);
    }

    uci::main_loop(eparams);
}
