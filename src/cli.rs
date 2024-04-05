use std::path::PathBuf;

use clap::Parser;

#[derive(Parser)]
#[clap(author, version, about)]
#[allow(clippy::struct_excessive_bools, clippy::option_option)]
pub struct Cli {
    /// All sub-commands that viri supports.
    #[clap(subcommand)]
    pub subcommand: Option<Subcommands>,
}

#[derive(Parser)]
pub enum Subcommands {
    /// Output node benchmark for OpenBench
    Bench,
    /// Run the perft suite.
    Perft,
    /// Generate graphical visualisations of the NNUE weights.
    VisNNUE,
    /// Count the number of positions contained within one or more packed game records.
    CountPositions {
        /// Path to input packed game record, or directory containing only packed game records.
        input: PathBuf,
    },
    /// Analyse a packed game record
    Analyse {
        /// Path to input packed game record.
        input: PathBuf,
    },
    /// Emit configuration for SPSA
    Spsa {
        /// Emit configuration in JSON format instead of OpenBench format
        json: bool,
    },
    /// Splat a packed game record into bulletformat records (or other format)
    Splat {
        /// Path to input packed game record.
        input: PathBuf,
        /// Output path.
        output: PathBuf,
        /// Splat into marlinformat instead of bulletformat.
        #[clap(long)]
        marlinformat: bool,
        /// Splat into PGN instead of bulletformat.
        #[clap(long)]
        pgn: bool,
        /// Limit the number of games to convert.
        #[clap(long, value_name = "N")]
        limit: Option<usize>,
    },
    /// Generate self-play data
    Datagen {
        /// Number of games to play
        #[clap(long, value_name = "N")]
        games: usize,
        /// Number of threads to parallelise datagen across
        #[clap(long, value_name = "N")]
        threads: usize,
        /// Path to a tablebases folder
        #[clap(long, value_name = "PATH")]
        tbs: Option<PathBuf>,
        /// Limit by depth instead of nodes
        #[clap(long)]
        depth_limit: bool,
        // Whether to generate DFRC data.
        #[clap(long)]
        dfrc: bool,
    },
}
