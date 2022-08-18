use clap::Parser;

#[derive(Parser)]
#[clap(author, version, about)]
#[allow(clippy::struct_excessive_bools)]
pub struct Cli {
    /// A path to the evaluation parameters for the engine - if omitted, the default parameters will be used
    #[clap(long, value_parser, value_name = "PATH")]
    pub params: Option<std::path::PathBuf>,
    /// Generate source code for PSQTs based on the evaluation parameters
    #[clap(long)]
    pub gensource: bool,
    /// Run the perft test suite
    #[clap(long)]
    pub perfttest: bool,
    /// Run the texel tuner
    #[clap(long)]
    pub tune: bool,
    /// Pick up texel tuning from halfway through the tuning process
    #[clap(long)]
    pub resume: bool,
    /// Display misc. information about the engine
    #[clap(short, long)]
    pub info: bool,
    /// Visualise the evaluation parameters
    #[clap(long)]
    pub visparams: bool,
    /// Visualise the Piece-Square Tables
    #[clap(long)]
    pub vispsqt: bool,
    /// Path to an Extended Position Description file to run as a test suite.
    #[clap(long, value_name = "PATH")]
    pub epdpath: Option<std::path::PathBuf>,
    /// Time in milliseconds to search for each move when doing an epd test suite.
    #[clap(long, value_name = "MS", default_value = "3000")]
    pub epdtime: u64,
}
