use clap::Parser;

#[derive(Parser)]
#[clap(author, version, about)]
#[allow(clippy::struct_excessive_bools)]
pub struct Cli {
    /// A path to the evaluation parameters for the engine - if omitted, the default parameters will be used
    #[clap(long, value_parser, value_name = "PATH")]
    pub params: Option<std::path::PathBuf>,
    /// Generate source code for PSQTs based on the evaluation parameters
    #[clap(long, value_parser, value_name = "PATH")]
    pub gensource: bool,
    /// Run the perft test suite
    #[clap(long)]
    pub perfttest: bool,
    /// Run the texel tuner
    #[clap(long)]
    pub tune: bool,
    /// Display misc. information about the engine
    #[clap(short, long)]
    pub info: bool,
    /// Visualise the evaluation parameters
    #[clap(long)]
    pub visparams: bool,
    /// Visualise the Piece-Square Tables
    #[clap(long)]
    pub vispsqt: bool,
}