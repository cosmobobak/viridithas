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
    /// Output node benchmark for openbench
    Bench {
        /// Optionally specify the depth at which to run the benchmark.
        depth: Option<usize>,
        /// Optionally specify the number of threads to use.
        threads: Option<usize>,
    },
    /// Run the perft suite.
    Perft,
    /// Quantise a network parameter file.
    Quantise {
        /// Path to input network parameter file.
        input: PathBuf,
        /// Path to output network parameter file.
        output: PathBuf,
    },
    /// Merge the factorisers in a network parameter file.
    Merge {
        /// Path to input network parameter file.
        input: PathBuf,
        /// Path to output network parameter file.
        output: PathBuf,
    },
    /// Dump the verbatim network to a file.
    Verbatim {
        /// Path to output verbatim network file.
        output: PathBuf,
    },
    /// Generate graphical visualisations of the NNUE weights.
    VisNNUE,
    /// Dry-run the NNUE inference.
    NNUEDryRun,
    /// Emit configuration for SPSA
    Spsa {
        /// Emit configuration in JSON format instead of openbench format
        #[clap(long)]
        json: bool,
    },
    /// Compute statistics about the static evaluation across an EPD file.
    EvalStats {
        /// Path to input EPD file.
        input: PathBuf,
        /// Optional path to output histogram data.
        #[clap(short, long)]
        output: Option<PathBuf>,
    },
    /// Count the number of positions contained within one or more packed game records.
    #[cfg(feature = "datagen")]
    CountPositions {
        /// Path to input packed game record, or directory containing only packed game records.
        input: PathBuf,
    },
    /// Analyse a packed game record
    #[cfg(feature = "datagen")]
    Analyse {
        /// Path to input packed game record.
        input: PathBuf,
    },
    /// Rescale evaluations in a packed game record
    #[cfg(feature = "datagen")]
    Rescale {
        /// Scaling factor to apply to evaluations
        #[clap(long)]
        scale: f64,
        /// Path to input packed game record.
        input: PathBuf,
        /// Path to output packed game record.
        output: PathBuf,
    },
    /// Splat a packed game record into bulletformat records (or other format)
    #[cfg(feature = "datagen")]
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
        /// Override the filter settings with a TOML configuration file.
        /// Example file format:
        /// ```toml
        /// min_ply = 16
        /// min_pieces = 4
        /// max_eval = 10000
        /// filter_tactical = true
        /// filter_check = true
        /// filter_castling = false
        /// max_eval_incorrectness = 4294967295
        /// ```
        #[clap(long, verbatim_doc_comment)]
        cfg_path: Option<PathBuf>,
    },
    /// Generate self-play data
    #[cfg(feature = "datagen")]
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
        /// Path to a book file to use for starting positions
        #[clap(long, value_name = "PATH")]
        book: Option<PathBuf>,
        /// Number of nodes to search per position.
        #[clap(long)]
        nodes: u64,
        // Whether to generate DFRC data.
        #[clap(long)]
        dfrc: bool,
    },
}
