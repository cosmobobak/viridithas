use clap::Parser;

#[derive(Parser)]
#[clap(author, version, about)]
#[allow(clippy::struct_excessive_bools)]
pub struct Cli {
    /// A path to the evaluation parameters for the engine - if omitted, the default parameters will be used
    #[clap(long, value_parser, value_name = "PATH")]
    pub eparams: Option<std::path::PathBuf>,
    /// A path to the search parameters for the engine - if omitted, the default parameters will be used
    #[clap(long, value_parser, value_name = "PATH")]
    pub sparams: Option<std::path::PathBuf>,
    /// Generate source code for PSQTs based on the evaluation parameters
    #[clap(long)]
    pub gensource: bool,
    /// Run the perft test suite
    #[clap(long)]
    pub perfttest: bool,
    /// Run the texel tuner on a WDL data file.
    #[clap(long, value_parser, value_name = "PATH")]
    pub tune: Option<std::path::PathBuf>,
    /// Limit tuning to a parameter. Can be passed multiple times to limit tuning to multiple parameters.
    #[clap(short, long, value_parser, value_name = "PARAMETERS")]
    pub limitparams: Option<Vec<usize>>,
    /// Pick up texel tuning from halfway through the tuning process
    #[clap(long)]
    pub resume: bool,
    /// Number of examples to use for tuning
    #[clap(long, value_name = "N_EXAMPLES", default_value = "16000000")]
    pub examples: usize,
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
    /// Hash size in MB to use when doing an epd test suite.
    #[clap(long, value_name = "MB", default_value = "4")]
    pub epdhash: usize,
    /// Path to a WDL data file to evaluate for NNUE data.
    #[clap(long, value_name = "PATH")]
    pub nnueconversionpath: Option<std::path::PathBuf>,
    /// Path to an NNUE data file to reanalyse with the current evaluation parameters.
    #[clap(long, value_name = "PATH")]
    pub nnuereanalysepath: Option<std::path::PathBuf>,
    /// Whether to use NNUE for generating NNUE training data.
    #[clap(long)]
    pub nnuefornnue: bool,
    /// Depth at which to do NNUE data generation.
    #[clap(long, value_name = "DEPTH", default_value = "8")]
    pub nnuedepth: i32,
    /// Output path.
    #[clap(short, long, value_name = "PATH")]
    pub output: Option<std::path::PathBuf>,
    /// Do conversion from NNUE json to NNUE binary. Arg1 is the input path, arg2 is the output path. Due to clap's idiosyncrasies, this must be passed as two arguments, e.g. --jsontobin network.json --jsontobin binfolder
    #[clap(long)]
    pub jsontobin: Vec<std::path::PathBuf>,
    /// Deduplicate an NNUE data file by removing duplicate positions.
    #[clap(long, value_name = "PATH")]
    pub dedup: Option<std::path::PathBuf>,
    /// Merge and deduplicate two NNUE data files
    #[clap(long, value_name = "PATH")]
    pub merge: Vec<std::path::PathBuf>,
    /// Visualise the NNUE.
    #[clap(long)]
    pub visnnue: bool,
}
