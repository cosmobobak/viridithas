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

    let eparams = cli
        .eparams
        .map_or_else(board::evaluation::parameters::EvalParams::default, |p| {
            board::evaluation::parameters::EvalParams::from_file(p).unwrap()
        });

    if cli.gensource {
        println!("PSQT source code:");
        piecesquaretable::tables::printout_pst_source(&eparams.piece_square_tables);
        return;
    }

    if cli.perfttest {
        perft::gamut();
        return;
    }

    if cli.tune {
        texel::tune(cli.resume, cli.examples, &eparams, cli.limitparams.as_deref());
        return;
    }

    if cli.info {
        println!("name: {NAME}");
        println!("version: {VERSION}");
        println!(
            "number of evaluation parameters: {}",
            board::evaluation::parameters::EvalParams::default()
                .vectorise()
                .len()
        );
        return;
    }

    if cli.visparams {
        println!("{eparams}");
        return;
    }

    if cli.vispsqt {
        piecesquaretable::render_pst_table(&eparams.piece_square_tables);
        return;
    }

    if let Some(epd_path) = cli.epdpath {
        epd::gamut(epd_path, eparams, cli.epdtime);
        return;
    }

    uci::main_loop(eparams);
}
