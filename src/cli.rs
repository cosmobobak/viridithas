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
    },
    /// Run the perft suite.
    Perft,
    /// Quantise a network parameter file.
    Quantise {
        /// Path to input network parameter file.
        input: std::path::PathBuf,
        /// Path to output network parameter file.
        output: std::path::PathBuf,
    },
    /// Merge the factorisers in a network parameter file.
    Merge {
        /// Path to input network parameter file.
        input: std::path::PathBuf,
        /// Path to output network parameter file.
        output: std::path::PathBuf,
    },
    /// Generate graphical visualisations of the NNUE weights.
    VisNNUE,
    /// Count the number of positions contained within one or more packed game records.
    #[cfg(feature = "datagen")]
    CountPositions {
        /// Path to input packed game record, or directory containing only packed game records.
        input: std::path::PathBuf,
    },
    /// Analyse a packed game record
    #[cfg(feature = "datagen")]
    Analyse {
        /// Path to input packed game record.
        input: std::path::PathBuf,
    },
    /// Emit configuration for SPSA
    Spsa {
        /// Emit configuration in JSON format instead of openbench format
        json: bool,
    },
    /// Splat a packed game record into bulletformat records (or other format)
    #[cfg(feature = "datagen")]
    Splat {
        /// Path to input packed game record.
        input: std::path::PathBuf,
        /// Output path.
        output: std::path::PathBuf,
        /// Splat into marlinformat instead of bulletformat.
        #[clap(long)]
        marlinformat: bool,
        /// Splat into PGN instead of bulletformat.
        #[clap(long)]
        pgn: bool,
        /// Limit the number of games to convert.
        #[clap(long, value_name = "N")]
        limit: Option<usize>,
        /// Override the filter settings.
        #[clap(long)]
        cfg_path: Option<std::path::PathBuf>,
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
        tbs: Option<std::path::PathBuf>,
        /// Path to a book file to use for starting positions
        #[clap(long, value_name = "PATH")]
        book: Option<std::path::PathBuf>,
        /// Limit by depth instead of nodes
        #[clap(long)]
        depth_limit: bool,
        // Whether to generate DFRC data.
        #[clap(long)]
        dfrc: bool,
    },
}
