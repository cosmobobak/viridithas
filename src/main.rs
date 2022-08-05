#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![deny(missing_docs)]

//! Viridithas II, a UCI chess engine written in Rust.

#[macro_use]
mod macros;

mod board;
mod chessmove;
mod cli;
mod definitions;
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
pub const NAME: &str = "Viridithas 2.3.0";

fn main() {
    #[cfg(debug_assertions)]
    std::env::set_var("RUST_BACKTRACE", "1");

    // takes about 3ms to generate the attack tables on boot
    magic::initialise();

    let cli = <cli::Cli as clap::Parser>::parse();

    let params = cli
        .params
        .map_or_else(board::evaluation::parameters::Parameters::default, |p| {
            board::evaluation::parameters::Parameters::from_file(p).unwrap()
        });

    if cli.gensource {
        println!("PSQT source code:");
        piecesquaretable::tables::printout_pst_source(&params.piece_square_tables);
        return;
    }

    if cli.perfttest {
        perft::gamut();
        return;
    }

    if cli.tune {
        texel::tune();
        return;
    }

    if cli.info {
        println!("{NAME}");
        println!(
            "evaluation parameters: {}",
            board::evaluation::parameters::Parameters::default()
                .vectorise()
                .len()
        );
        println!("TT buckets: {}", transpositiontable::DEFAULT_TABLE_SIZE);
        println!(
            "TT size (kb): {}",
            std::mem::size_of::<transpositiontable::TTEntry>()
                * transpositiontable::DEFAULT_TABLE_SIZE
                / 1024
        );
        return;
    }

    if cli.visparams {
        println!("{params}");
        return;
    }

    if cli.vispsqt {
        piecesquaretable::render_pst_table(&params.piece_square_tables);
        return;
    }

    uci::main_loop(params);
}
