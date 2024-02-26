use clap::Parser;

#[derive(Parser)]
#[clap(author, version, about)]
#[allow(clippy::struct_excessive_bools, clippy::option_option)]
pub struct Cli {
    /// Run the perft test suite
    #[clap(long)]
    pub perfttest: bool,
    /// Scan a packed data record and report statistics
    #[clap(short, long)]
    pub dataset_stats: Option<std::path::PathBuf>,
    /// emit JSON for SPSA
    #[clap(long)]
    pub spsajson: bool,
    /// emit OB-format for SPSA
    #[clap(long)]
    pub spsaob: bool,
    /// Output path.
    #[clap(short, long, value_name = "PATH")]
    pub output: Option<std::path::PathBuf>,
    /// Visualise the NNUE.
    #[clap(long)]
    pub visnnue: bool,
    /// Generate training data for the NNUE.
    #[clap(long)]
    pub datagen: Option<Option<String>>,
    /// Splat a binary game record into binary records.
    #[clap(long)]
    pub splat: Option<std::path::PathBuf>,
    /// Splat into marlinformat instead of bulletformat.
    /// Only valid with --splat.
    #[clap(long)]
    pub marlinformat: bool,
    /// Convert a binary game record into PGN.
    #[clap(long)]
    pub topgn: Option<std::path::PathBuf>,
    /// Limit the number of games to convert.
    #[clap(long, value_name = "N")]
    pub limit: Option<usize>,
    /// Output node benchmark for OpenBench.
    /// Implemented as a subcommand because that's what OpenBench expects.
    #[clap(subcommand)]
    pub bench: Option<Bench>,
}

#[derive(Parser)]
pub enum Bench {
    /// Output node benchmark for OpenBench.
    Bench,
}
