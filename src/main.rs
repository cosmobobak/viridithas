#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::if_not_else)]

use board::evaluation::Parameters;

#[macro_use]
mod macros;

mod board;
mod chessmove;
mod definitions;
mod errors;
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

pub const NAME: &str = "Viridithas 2.1.0dev";

fn main() {
    #[cfg(debug_assertions)]
    std::env::set_var("RUST_BACKTRACE", "1");

    let args = std::env::args().collect::<Vec<_>>();

    // takes about 3ms to generate the attack tables on boot
    magic::initialise();

    match args.get(1).map(String::as_str) {
        None | Some("uci") => {
            let evaluation_parameters = args.get(2).map_or_else(Parameters::default, |path| {
                Parameters::from_file(path).unwrap()
            });
            uci::main_loop(evaluation_parameters);
        }
        Some("perfttest") => perft::gamut(),
        Some("tune") => texel::tune(),
        Some("info") => {
            println!("{NAME}");
            println!(
                "evaluation parameters: {}",
                Parameters::default().vectorise().len()
            );
            println!("TT buckets: {}", transpositiontable::DEFAULT_TABLE_SIZE);
            println!(
                "TT size (kb): {}",
                std::mem::size_of::<transpositiontable::Bucket>()
                    * transpositiontable::DEFAULT_TABLE_SIZE
                    / 1024
            );
        }
        Some("visparams") => {
            let path = args.get(2);
            let params = path.map_or_else(Parameters::default, |path| {
                Parameters::from_file(path).unwrap()
            });
            println!("{params}");
        }
        Some("vispst") => {
            let path = args.get(2);
            let params = path.map_or_else(Parameters::default, |path| {
                Parameters::from_file(path).unwrap()
            });
            piecesquaretable::render_pst_table(&params.piece_square_tables);
        }
        Some("gensource") => {
            let path = args.get(2);
            let params = path.map_or_else(Parameters::default, |path| {
                Parameters::from_file(path).unwrap()
            });
            println!("PSQT source code:");
            piecesquaretable::tables::printout_pst_source(&params.piece_square_tables);
        }
        Some(unknown) => {
            if unknown != "help" {
                println!("Unknown command: {unknown}");
            }
            println!("Available CLI args:");
            println!(" - uci (default) : run the Universal Chess Interface");
            println!(" - perfttest     : run the perft test suite");
            println!(" - tune          : use texel's tuning method to optimise the evaluation parameters");
            println!(" - info          : miscellaneous information about the engine");
            println!(" - visparams     : visualise the evaluation parameters");
            println!(" - vispst        : visualise the piece square tables");
            println!(" - gensource     : generate the source code for the piece square tables");
        }
    }
}
