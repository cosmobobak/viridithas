#![deny(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::todo,
    clippy::unimplemented
)]

use std::{
    error::Error,
    fmt::{self, Display},
    io::Write,
    num::{ParseFloatError, ParseIntError},
    str::{FromStr, ParseBoolError},
    sync::{
        Mutex, Once,
        atomic::{self, AtomicBool, AtomicI32, AtomicU8, AtomicU64, AtomicUsize, Ordering},
        mpsc,
    },
    time::Instant,
};

use anyhow::{Context, anyhow, bail};

use crate::{
    NAME, VERSION,
    bench::BENCH_POSITIONS,
    chess::{
        CHESS960,
        board::{
            Board,
            movegen::{self, MoveList},
        },
        piece::Colour,
    },
    cuckoo,
    errors::{FenParseError, MoveParseError},
    evaluation::{MATE_SCORE, TB_WIN_SCORE, evaluate, is_decisive, is_mate_score},
    nnue::{
        self,
        network::{self, NNUEParams},
    },
    perft,
    search::{LMTable, adj_shuffle, parameters::Config, search_position},
    searchinfo::SearchInfo,
    tablebases, term,
    threadlocal::{ThreadData, make_thread_data},
    threadpool,
    timemgmt::SearchLimit,
    transpositiontable::TT,
    util::{MAX_DEPTH, MEGABYTE},
};

#[cfg(feature = "nnz-counts")]
use crate::nnue::network::layers::{NNZ_COUNT, NNZ_DENOM};

const UCI_DEFAULT_HASH_MEGABYTES: usize = 16;
const UCI_MAX_HASH_MEGABYTES: usize = 1_048_576;
const UCI_MAX_THREADS: usize = 512;

static STDIN_READER_THREAD_KEEP_RUNNING: AtomicBool = AtomicBool::new(true);
pub static QUIT: AtomicBool = AtomicBool::new(false);
pub static GO_MATE_MAX_DEPTH: AtomicUsize = AtomicUsize::new(MAX_DEPTH);
pub static PRETTY_PRINT: AtomicBool = AtomicBool::new(true);
pub static SYZYGY_PROBE_LIMIT: AtomicU8 = AtomicU8::new(6);
pub static SYZYGY_PROBE_DEPTH: AtomicI32 = AtomicI32::new(1);
pub static SYZYGY_PATH: Mutex<String> = Mutex::new(String::new());
pub static SYZYGY_ENABLED: AtomicBool = AtomicBool::new(false);
pub static CONTEMPT: AtomicI32 = AtomicI32::new(0);

#[derive(Debug, PartialEq, Eq)]
enum UciError {
    ParseOption(String),
    ParseFen(FenParseError),
    ParseMove(MoveParseError),
    UnexpectedCommandTermination(String),
    InvalidFormat(String),
    UnknownCommand(String),
    InternalError(String),
    IllegalValue(String),
}

impl From<MoveParseError> for UciError {
    fn from(err: MoveParseError) -> Self {
        Self::ParseMove(err)
    }
}

impl From<FenParseError> for UciError {
    fn from(err: FenParseError) -> Self {
        Self::ParseFen(err)
    }
}

impl From<ParseFloatError> for UciError {
    fn from(pfe: ParseFloatError) -> Self {
        Self::ParseOption(pfe.to_string())
    }
}

impl From<ParseIntError> for UciError {
    fn from(pie: ParseIntError) -> Self {
        Self::ParseOption(pie.to_string())
    }
}

impl From<ParseBoolError> for UciError {
    fn from(pbe: ParseBoolError) -> Self {
        Self::ParseOption(pbe.to_string())
    }
}

impl Display for UciError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ParseOption(s) => write!(f, "ParseOption: {s}"),
            Self::ParseFen(s) => write!(f, "ParseFen: {s}"),
            Self::ParseMove(s) => write!(f, "ParseMove: {s}"),
            Self::UnexpectedCommandTermination(s) => {
                write!(f, "UnexpectedCommandTermination: {s}")
            }
            Self::InvalidFormat(s) => write!(f, "InvalidFormat: {s}"),
            Self::UnknownCommand(s) => write!(f, "UnknownCommand: {s}"),
            Self::InternalError(s) => write!(f, "InternalError: {s}"),
            Self::IllegalValue(s) => write!(f, "IllegalValue: {s}"),
        }
    }
}

// position fen
// position startpos
// ... moves e2e4 e7e5 b7b8q
fn parse_position(text: &str, pos: &mut Board) -> anyhow::Result<()> {
    let mut parts = text.split_ascii_whitespace();
    let command = parts.next().with_context(|| {
        UciError::UnexpectedCommandTermination("No command in parse_position".into())
    })?;
    if command != "position" {
        bail!(UciError::InvalidFormat("Expected 'position'".into()));
    }
    let determiner = parts.next().with_context(|| {
        UciError::UnexpectedCommandTermination("No determiner after \"position\"".into())
    })?;
    if determiner == "startpos" {
        pos.set_startpos();
        let moves = parts.next(); // skip "moves"
        if !(matches!(moves, Some("moves") | None)) {
            bail!(UciError::InvalidFormat(
                "Expected either \"moves\" or no content to follow \"startpos\".".into(),
            ));
        }
    } else if determiner == "frc" {
        let Some(index) = parts.next() else {
            bail!("Expected an index value to follow \"frc\"");
        };
        let index = index
            .parse()
            .with_context(|| format!("Failed to parse {index} as FRC index"))?;
        anyhow::ensure!(index < 960, "FRC index can be at most 959 but got {index}");
        pos.set_frc_idx(index);
    } else if determiner == "dfrc" {
        let Some(index) = parts.next() else {
            bail!("Expected an index value to follow \"dfrc\"");
        };
        let index = index
            .parse()
            .with_context(|| format!("Failed to parse {index} as DFRC index"))?;
        anyhow::ensure!(
            index < 960 * 960,
            "DFRC index can be at most 921599 but got {index}"
        );
        pos.set_dfrc_idx(index);
    } else {
        if determiner != "fen" {
            bail!(UciError::InvalidFormat(format!(
                "Unknown term after \"position\": {determiner}"
            )));
        }
        let mut fen = String::new();
        for part in &mut parts {
            if part == "moves" {
                break;
            }
            fen.push_str(part);
            fen.push(' ');
        }
        pos.set_from_fen(&fen)
            .with_context(|| format!("Failed to set fen {fen}"))?;
    }
    for san in parts {
        pos.zero_height(); // stuff breaks really hard without this lmao
        let m = pos.parse_uci(san)?;
        pos.make_move_simple(m);
    }
    pos.zero_height();
    Ok(())
}

fn parse_go(text: &str, stm: Colour) -> anyhow::Result<SearchLimit> {
    #![allow(clippy::too_many_lines)]

    let mut depth: Option<usize> = None;
    let mut moves_to_go: Option<u64> = None;
    let mut movetime: Option<u64> = None;
    let mut clocks: [Option<i64>; 2] = [None, None];
    let mut incs: [Option<i64>; 2] = [None, None];
    let mut nodes: Option<u64> = None;
    let mut limit = SearchLimit::Infinite;
    let mut ponder = false;

    let mut parts = text.split_ascii_whitespace();
    let command = parts
        .next()
        .with_context(|| UciError::UnexpectedCommandTermination("No command in parse_go".into()))?;
    if command != "go" {
        bail!(UciError::InvalidFormat("Expected \"go\"".into()));
    }

    while let Some(part) = parts.next() {
        match part {
            "depth" => {
                depth = Some(
                    part_parse("depth", parts.next())
                        .with_context(|| "Failed to parse depth part.")?,
                );
            }
            "movestogo" => moves_to_go = Some(part_parse("movestogo", parts.next())?),
            "movetime" => movetime = Some(part_parse("movetime", parts.next())?),
            "wtime" => clocks[stm] = Some(part_parse("wtime", parts.next())?),
            "btime" => clocks[stm.flip()] = Some(part_parse("btime", parts.next())?),
            "winc" => incs[stm] = Some(part_parse("winc", parts.next())?),
            "binc" => incs[stm.flip()] = Some(part_parse("binc", parts.next())?),
            "infinite" => limit = SearchLimit::Infinite,
            "mate" => {
                let mate_distance: usize = part_parse("mate", parts.next())?;
                let ply = mate_distance * 2; // gives padding when we're giving mate, but whatever
                GO_MATE_MAX_DEPTH.store(ply, Ordering::SeqCst);
                limit = SearchLimit::Mate { ply };
            }
            "nodes" => nodes = Some(part_parse("nodes", parts.next())?),
            "ponder" => ponder = true,
            other => bail!(UciError::InvalidFormat(format!("Unknown term: {other}"))),
        }
    }
    if !matches!(limit, SearchLimit::Mate { .. }) {
        GO_MATE_MAX_DEPTH.store(MAX_DEPTH, Ordering::SeqCst);
    }

    if let Some(movetime) = movetime {
        limit = SearchLimit::Time(movetime);
    }
    if let Some(depth) = depth {
        limit = SearchLimit::Depth(depth);
    }

    if let [Some(our_clock), Some(their_clock)] = clocks {
        let [our_inc, their_inc] = [incs[0].unwrap_or(0), incs[1].unwrap_or(0)];
        let our_clock: u64 = our_clock.try_into().unwrap_or(0);
        let their_clock: u64 = their_clock.try_into().unwrap_or(0);
        let our_inc: u64 = our_inc.try_into().unwrap_or(0);
        let their_inc: u64 = their_inc.try_into().unwrap_or(0);
        limit = SearchLimit::Dynamic {
            our_clock,
            their_clock,
            our_inc,
            their_inc,
            moves_to_go,
        };
    } else if clocks.iter().chain(incs.iter()).any(Option::is_some) {
        bail!(UciError::InvalidFormat(
            "at least one of [wtime, btime, winc, binc] provided, but not all.".into(),
        ));
    }

    if let Some(nodes) = nodes {
        limit = SearchLimit::Nodes(nodes);
    }

    if ponder {
        limit = limit.to_pondering();
    }

    Ok(limit)
}

fn part_parse<T>(target: &str, next_part: Option<&str>) -> anyhow::Result<T>
where
    T: FromStr,
    <T as FromStr>::Err: Display + Send + Sync + Error + 'static,
{
    let next_part = next_part
        .with_context(|| UciError::InvalidFormat(format!("nothing after \"{target}\"")))?;
    let value = next_part.parse();
    value.with_context(|| {
        UciError::InvalidFormat(format!(
            "value for {target} is not a number, tried to parse {next_part}"
        ))
    })
}

struct SetOptions {
    pub search_config: Config,
    pub hash_mb: usize,
    pub threads: usize,
}

#[allow(clippy::too_many_lines)]
fn parse_setoption(text: &str, pre_config: SetOptions) -> anyhow::Result<SetOptions> {
    use UciError::UnexpectedCommandTermination;
    let mut parts = text.split_ascii_whitespace();
    let Some(_) = parts.next() else {
        bail!(UnexpectedCommandTermination(
            "no \"setoption\" found".into()
        ));
    };
    let Some(name_part) = parts.next() else {
        bail!(UciError::InvalidFormat(
            "no \"name\" after \"setoption\"".into()
        ));
    };
    if name_part != "name" {
        bail!(UciError::InvalidFormat(format!(
            "unexpected character after \"setoption\", expected \"name\", got \"{name_part}\". Did you mean \"setoption name {name_part}\"?"
        )));
    }
    let opt_name = parts.next().with_context(|| {
        UnexpectedCommandTermination("no option name given after \"setoption name\"".into())
    })?;
    let Some(value_part) = parts.next() else {
        bail!(UciError::InvalidFormat(format!(
            "no \"value\" after \"setoption name {opt_name}\""
        )));
    };
    if value_part != "value" {
        bail!(UciError::InvalidFormat(format!(
            "unexpected character after \"setoption name {opt_name}\", expected \"value\", got \"{value_part}\". Did you mean \"setoption name {opt_name} value {value_part}\"?"
        )));
    }
    let opt_value = parts.next().with_context(|| {
        UnexpectedCommandTermination(format!(
            "no option value given after \"setoption name {opt_name} value\""
        ))
    })?;
    let mut out = pre_config;
    let id_parser_pairs = out.search_config.ids_with_parsers();
    let mut found_match = false;
    for (param_name, mut parser) in id_parser_pairs {
        if param_name == opt_name {
            let res = parser(opt_value);
            if let Err(e) = res {
                bail!(UciError::InvalidFormat(e.to_string()));
            }
            found_match = true;
            break;
        }
    }
    if found_match {
        return Ok(out);
    }
    match opt_name {
        "Hash" => {
            let value: usize = opt_value.parse()?;
            if !(value > 0 && value <= UCI_MAX_HASH_MEGABYTES) {
                // "Hash value must be between 1 and {UCI_MAX_HASH_MEGABYTES}"
                bail!(UciError::IllegalValue(format!(
                    "Hash value must be between 1 and {UCI_MAX_HASH_MEGABYTES}"
                )));
            }
            out.hash_mb = value;
        }
        "Threads" => {
            let value: usize = opt_value.parse()?;
            if !(value > 0 && value <= UCI_MAX_THREADS) {
                // "Threads value must be between 1 and {UCI_MAX_THREADS}"
                bail!(UciError::IllegalValue(format!(
                    "Threads value must be between 1 and {UCI_MAX_THREADS}"
                )));
            }
            out.threads = value;
        }
        "PrettyPrint" => {
            let value: bool = opt_value.parse()?;
            PRETTY_PRINT.store(value, Ordering::SeqCst);
        }
        "SyzygyPath" => {
            let path = opt_value.to_string();
            tablebases::probe::init(&path);
            if let Ok(mut lock) = SYZYGY_PATH.lock() {
                *lock = path;
                SYZYGY_ENABLED.store(true, Ordering::SeqCst);
            } else {
                bail!(UciError::InternalError(
                    "failed to take lock on SyzygyPath".into()
                ));
            }
        }
        "SyzygyProbeLimit" => {
            let value: u8 = opt_value.parse()?;
            if value > 6 {
                bail!(UciError::IllegalValue(
                    "SyzygyProbeLimit value must be between 0 and 6".to_string()
                ));
            }
            SYZYGY_PROBE_LIMIT.store(value, Ordering::SeqCst);
        }
        "SyzygyProbeDepth" => {
            let value: i32 = opt_value.parse()?;
            if !(1..=100).contains(&value) {
                bail!(UciError::IllegalValue(
                    "SyzygyProbeDepth value must be between 0 and 100".to_string()
                ));
            }
            SYZYGY_PROBE_DEPTH.store(value, Ordering::SeqCst);
        }
        "Contempt" => {
            let value: i32 = opt_value.parse()?;
            if !(-10000..=10000).contains(&value) {
                bail!(UciError::IllegalValue(
                    "Contempt value must be between -10000 and 10000".to_string()
                ));
            }
            CONTEMPT.store(value, Ordering::SeqCst);
        }
        "UCI_Chess960" => {
            let val = opt_value.parse()?;
            CHESS960.store(val, Ordering::SeqCst);
        }
        _ => {
            eprintln!("info string ignoring option {opt_name}, type \"uci\" for a list of options");
        }
    }
    Ok(out)
}

fn stdin_reader() -> anyhow::Result<(
    mpsc::Receiver<String>,
    std::thread::JoinHandle<anyhow::Result<()>>,
)> {
    let (sender, receiver) = mpsc::channel();
    let handle = std::thread::Builder::new()
        .name("stdin-reader".into())
        .spawn(|| stdin_reader_worker(sender))
        .with_context(|| "Couldn't start stdin reader worker thread")?;
    Ok((receiver, handle))
}

fn stdin_reader_worker(sender: mpsc::Sender<String>) -> anyhow::Result<()> {
    let mut linebuf = String::with_capacity(128);
    while let Ok(bytes) = std::io::stdin().read_line(&mut linebuf) {
        if bytes == 0 {
            // EOF
            sender
                .send("quit".into())
                .with_context(|| "couldn't send quit command to main thread")?;
            QUIT.store(true, Ordering::SeqCst);
            break;
        }
        let cmd = linebuf.trim();
        if cmd.is_empty() {
            linebuf.clear();
            continue;
        }
        if let Err(e) = sender.send(cmd.to_owned()) {
            bail!("info string error sending command to main thread: {e}");
        }
        if !STDIN_READER_THREAD_KEEP_RUNNING.load(atomic::Ordering::SeqCst) {
            break;
        }
        linebuf.clear();
    }

    std::mem::drop(sender);

    Ok(())
}

pub struct ScoreFormatWrapper(i32);
impl Display for ScoreFormatWrapper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if is_mate_score(self.0) {
            let plies_to_mate = MATE_SCORE - self.0.abs();
            let moves_to_mate = (plies_to_mate + 1) / 2;
            if self.0 > 0 {
                write!(f, "mate {moves_to_mate}")
            } else {
                write!(f, "mate -{moves_to_mate}")
            }
        } else if is_decisive(self.0) {
            write!(f, "cp {}", self.0)
        } else {
            write!(f, "cp {}", self.0 * 100 / NORMALISE_TO_PAWN_VALUE)
        }
    }
}
pub const fn format_score(score: i32) -> ScoreFormatWrapper {
    ScoreFormatWrapper(score)
}
pub struct PrettyScoreFormatWrapper(i32, Colour);
impl Display for PrettyScoreFormatWrapper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            -20..=20 => write!(f, "\u{001b}[0m")?, // drawish, no colour.
            21..=100 => write!(f, "\u{001b}[38;5;10m")?, // slightly better for us, light green.
            -100..=-21 => write!(f, "\u{001b}[38;5;9m")?, // slightly better for them, light red.
            101..=500 => write!(f, "\u{001b}[38;5;2m")?, // clearly better for us, green.
            -10000..=-101 => write!(f, "\u{001b}[38;5;1m")?, // clearly/much better for them, red.
            501..=10000 => write!(f, "\u{001b}[38;5;4m")?, // much better for us, blue.
            _ => write!(f, "\u{001b}[38;5;219m")?, // probably a mate score, pink.
        }
        let white_pov = if self.1 == Colour::White {
            self.0
        } else {
            -self.0
        };
        if is_mate_score(white_pov) {
            let plies_to_mate = MATE_SCORE - white_pov.abs();
            let moves_to_mate = (plies_to_mate + 1) / 2;
            if white_pov > 0 {
                write!(f, "   #{moves_to_mate:<2}")?;
            } else {
                write!(f, "  #-{moves_to_mate:<2}")?;
            }
        } else if is_decisive(white_pov) {
            let plies_to_tb = TB_WIN_SCORE - white_pov.abs();
            if white_pov > 0 {
                write!(f, " +TB{plies_to_tb:<2}")?;
            } else {
                write!(f, " -TB{plies_to_tb:<2}")?;
            }
        } else {
            let white_pov = white_pov * 100 / NORMALISE_TO_PAWN_VALUE;
            let white_pov = white_pov.clamp(-9999, 9999);
            if white_pov == 0 {
                // same as below, but with no sign
                write!(f, "{:6.2}", f64::from(white_pov) / 100.0)?;
            } else {
                // six chars wide: one for the sign, two for the pawn values,
                // one for the decimal point, and two for the centipawn values
                write!(f, "{:+6.2}", f64::from(white_pov) / 100.0)?;
            }
        }
        write!(f, "\u{001b}[0m") // reset
    }
}
pub const fn pretty_format_score(v: i32, c: Colour) -> PrettyScoreFormatWrapper {
    PrettyScoreFormatWrapper(v, c)
}

pub struct HumanTimeFormatWrapper {
    millis: u128,
}
impl Display for HumanTimeFormatWrapper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let millis = self.millis;
        let seconds = millis / 1000;
        let minutes = seconds / 60;
        let hours = minutes / 60;
        let days = hours / 24;
        if days > 0 {
            write!(f, "{days:2}d{r_hours:02}h", r_hours = hours % 24)
        } else if hours > 0 {
            write!(f, "{hours:2}h{r_minutes:02}m", r_minutes = minutes % 60)
        } else if minutes > 0 {
            write!(f, "{minutes:2}m{r_seconds:02}s", r_seconds = seconds % 60)
        } else if seconds > 0 {
            write!(
                f,
                "{seconds:2}.{r_millis:02}s",
                r_millis = millis % 1000 / 10
            )
        } else {
            write!(f, "{millis:4}ms")
        }
    }
}
pub const fn format_time(millis: u128) -> HumanTimeFormatWrapper {
    HumanTimeFormatWrapper { millis }
}

fn print_uci_response(info: &SearchInfo, full: bool) {
    let version_extension = if cfg!(feature = "final-release") {
        ""
    } else {
        "-dev"
    };
    println!("id name {NAME} {VERSION}{version_extension}");
    println!("id author Cosmo");
    println!(
        "option name Hash type spin default {UCI_DEFAULT_HASH_MEGABYTES} min 1 max {UCI_MAX_HASH_MEGABYTES}"
    );
    println!("option name Threads type spin default 1 min 1 max 512");
    println!("option name PrettyPrint type check default false");
    println!("option name SyzygyPath type string default <empty>");
    println!("option name SyzygyProbeLimit type spin default 6 min 0 max 6");
    println!("option name SyzygyProbeDepth type spin default 1 min 1 max 100");
    println!("option name Contempt type spin default 0 min -10000 max 10000");
    println!("option name Ponder type check default false");
    println!("option name UCI_Chess960 type check default false");
    if full {
        for (id, default, min, max, _) in info.conf.base_config() {
            println!("option name {id} type spin default {default} min {min} max {max}");
        }
    }
    println!("uciok");
}

static SET_TERM: Once = Once::new();

#[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
pub fn main_loop() -> anyhow::Result<()> {
    let version_extension = if cfg!(feature = "final-release") {
        ""
    } else {
        "-dev"
    };
    println!("{NAME} {VERSION}{version_extension} by Cosmo");

    let mut worker_threads = threadpool::make_worker_threads(1);

    let mut tt = TT::new();
    tt.resize(UCI_DEFAULT_HASH_MEGABYTES * MEGABYTE, &worker_threads); // default hash size

    let nnue_params = NNUEParams::decompress_and_alloc()?;

    let (stdin, stdin_reader_handle) = stdin_reader()?;
    let stdin = Mutex::new(stdin);
    let stopped = AtomicBool::new(false);
    let nodes = AtomicU64::new(0);
    let mut thread_data = make_thread_data(
        &Board::default(),
        tt.view(),
        nnue_params,
        &stopped,
        &nodes,
        &worker_threads,
    )?;
    thread_data[0].info.set_stdin(&stdin);

    loop {
        std::io::stdout()
            .flush()
            .with_context(|| "couldn't flush stdout")?;
        let Ok(line) = stdin
            .lock()
            .map_err(|_| anyhow!("failed to take lock on stdin"))?
            .recv()
        else {
            break;
        };
        let input = line.trim();

        let res = match input {
            "\n" => continue,
            "uci" => {
                #[cfg(feature = "tuning")]
                print_uci_response(&thread_data[0].info, true);
                #[cfg(not(feature = "tuning"))]
                print_uci_response(&thread_data[0].info, false);
                PRETTY_PRINT.store(false, Ordering::SeqCst);
                Ok(())
            }
            "ucifull" => {
                print_uci_response(&thread_data[0].info, true);
                PRETTY_PRINT.store(false, Ordering::SeqCst);
                Ok(())
            }
            arg @ ("ucidump" | "ucidumpfull") => {
                // dump the values of the current UCI options
                println!("Hash: {}", tt.size() / MEGABYTE);
                println!("Threads: {}", thread_data.len());
                println!("PrettyPrint: {}", PRETTY_PRINT.load(Ordering::SeqCst));
                println!(
                    "SyzygyPath: {}",
                    SYZYGY_PATH
                        .lock()
                        .map_err(|_| anyhow!("failed to lock syzygy path"))?
                );
                println!(
                    "SyzygyProbeLimit: {}",
                    SYZYGY_PROBE_LIMIT.load(Ordering::SeqCst)
                );
                println!(
                    "SyzygyProbeDepth: {}",
                    SYZYGY_PROBE_DEPTH.load(Ordering::SeqCst)
                );
                println!("Contempt: {}", CONTEMPT.load(Ordering::SeqCst));
                if arg == "ucidumpfull" {
                    for (id, default) in Config::default().ids_with_values() {
                        println!("{id}: {default}");
                    }
                }
                Ok(())
            }
            "isready" => {
                println!("readyok");
                Ok(())
            }
            "quit" => {
                QUIT.store(true, Ordering::SeqCst);
                break;
            }
            "ucinewgame" => do_newgame(&tt, &mut thread_data, &worker_threads),
            "eval" => {
                let t = thread_data
                    .first_mut()
                    .with_context(|| "the thread headers are empty.")?;
                let eval = if t.board.in_check() {
                    0
                } else {
                    let eval = evaluate(t, 0);
                    adj_shuffle(t, eval, t.board.fifty_move_counter())
                };
                println!("{eval}");
                Ok(())
            }
            "raweval" => {
                let t = thread_data
                    .first_mut()
                    .with_context(|| "the thread headers are empty.")?;
                let eval = if t.board.in_check() {
                    0
                } else {
                    t.nnue.evaluate(
                        t.nnue_params,
                        t.board.turn(),
                        network::output_bucket(&t.board),
                    )
                };
                println!("{eval}");
                Ok(())
            }
            "show" => {
                let t = thread_data
                    .first_mut()
                    .with_context(|| "the thread headers are empty.")?;
                println!("{:X}", t.board);
                Ok(())
            }
            "nnuebench" => {
                nnue::network::inference_benchmark(
                    &thread_data[0].nnue,
                    thread_data[0].nnue_params,
                );
                Ok(())
            }
            "gobench" => go_benchmark(nnue_params),
            "initcuckoo" => cuckoo::init(),
            "initattacks" => movegen::init_sliders_attacks(),
            input if input.starts_with("setoption") => {
                let pre_config = SetOptions {
                    search_config: thread_data[0].info.conf.clone(),
                    hash_mb: tt.size() / MEGABYTE,
                    threads: thread_data.len(),
                };
                let threads_before = thread_data.len();
                let res = parse_setoption(input, pre_config);
                match res {
                    Ok(conf) => {
                        if threads_before != conf.threads {
                            println!(
                                "info string changing threads from {threads_before} to {}",
                                conf.threads
                            );
                            worker_threads
                                .into_iter()
                                .for_each(threadpool::WorkerThread::join);
                            worker_threads = threadpool::make_worker_threads(conf.threads);
                        }
                        let new_size = conf.hash_mb * MEGABYTE;
                        let pos = thread_data[0].board.clone();
                        // drop all the thread_data, as they are borrowing the old tt
                        std::mem::drop(thread_data);
                        tt.resize(new_size, &worker_threads);
                        // recreate the thread_data with the new tt
                        thread_data = make_thread_data(
                            &pos,
                            tt.view(),
                            nnue_params,
                            &stopped,
                            &nodes,
                            &worker_threads,
                        )?;

                        for t in &mut thread_data {
                            t.info.conf = conf.search_config.clone();
                            t.info.lm_table = LMTable::new(&t.info.conf);
                            t.info.set_stdin(&stdin);
                        }

                        Ok(())
                    }
                    Err(err) => Err(err),
                }
            }
            input if input.starts_with("position") => (|| {
                for t in &mut thread_data {
                    parse_position(input, &mut t.board)?;
                    t.nnue.reinit_from(&t.board, t.nnue_params);
                }
                Ok(())
            })(),
            input if input.starts_with("go perft") || input.starts_with("perft") => {
                let t = thread_data
                    .first_mut()
                    .with_context(|| "the thread headers are empty.")?;
                let tail = input
                    .trim_start_matches("go perft ")
                    .trim_start_matches("perft ");
                match tail.split_whitespace().next() {
                    Some("divide" | "split") => {
                        let depth = tail
                            .trim_start_matches("divide ")
                            .trim_start_matches("split ");
                        depth
                            .parse::<usize>()
                            .with_context(|| {
                                UciError::InvalidFormat(format!(
                                    "cannot parse \"{depth}\" as usize"
                                ))
                            })
                            .map(|depth| divide_perft(depth, &mut t.board))
                    }
                    Some(depth) => depth
                        .parse::<usize>()
                        .with_context(|| {
                            UciError::InvalidFormat(format!("cannot parse \"{depth}\" as usize"))
                        })
                        .map(|depth| block_perft(depth, &mut t.board)),
                    None => Err(anyhow!(UciError::InvalidFormat(
                        "expected a depth after 'go perft'".to_string()
                    ))),
                }
            }
            input if input.starts_with("go") => {
                // start the clock *immediately*
                thread_data[0].info.clock.start();

                // if we're in pretty-printing mode, set the terminal properly:
                if PRETTY_PRINT.load(Ordering::SeqCst) {
                    SET_TERM.call_once(|| {
                        term::set_mode_uci();
                    });
                }

                let res = parse_go(input, thread_data[0].board.turn());
                if let Ok(search_limit) = res {
                    thread_data[0].info.clock.set_limit(search_limit);
                    tt.increase_age();
                    search_position(&worker_threads, &mut thread_data);
                    Ok(())
                } else {
                    res.map(|_| ())
                }
            }
            "ponderhit" => {
                println!("info error ponderhit given while not searching.");
                Ok(())
            }
            benchcmd @ ("bench" | "benchfull") => {
                bench(benchcmd, &thread_data[0].info.conf, nnue_params, None)
            }
            _ => Err(anyhow!(UciError::UnknownCommand(input.to_string()))),
        };

        if let Err(e) = res {
            eprintln!("info string {e}");
        }

        if QUIT.load(Ordering::SeqCst) {
            // quit can be set true in parse_go
            break;
        }
    }
    STDIN_READER_THREAD_KEEP_RUNNING.store(false, atomic::Ordering::SeqCst);
    if stdin_reader_handle.is_finished() {
        stdin_reader_handle
            .join()
            .map_err(|_| anyhow!("Thread panicked!"))??;
    }
    Ok(())
}

const BENCH_DEPTH: usize = 14;
const BENCH_THREADS: usize = 1;
pub fn bench(
    benchcmd: &str,
    search_params: &Config,
    nnue_params: &'static NNUEParams,
    depth: Option<usize>,
) -> anyhow::Result<()> {
    let bench_string = format!("go depth {}\n", depth.unwrap_or(BENCH_DEPTH));
    let stopped = AtomicBool::new(false);
    let nodes = AtomicU64::new(0);
    let pool = threadpool::make_worker_threads(BENCH_THREADS);
    let mut tt = TT::new();
    tt.resize(16 * MEGABYTE, &pool);
    let mut thread_data = make_thread_data(
        &Board::default(),
        tt.view(),
        nnue_params,
        &stopped,
        &nodes,
        &pool,
    )?;
    thread_data[0].info.conf = search_params.clone();
    thread_data[0].info.print_to_stdout = false;
    let mut node_sum = 0u64;
    let start = Instant::now();
    let max_fen_len = BENCH_POSITIONS
        .iter()
        .map(|s| s.len())
        .max()
        .with_context(|| "this array is nonempty.")?;
    for fen in BENCH_POSITIONS {
        let res = do_newgame(&tt, &mut thread_data, &pool);
        if let Err(e) = res {
            thread_data[0].info.print_to_stdout = true;
            return Err(e);
        }
        for t in &mut thread_data {
            let res = parse_position(&format!("position fen {fen}\n"), &mut t.board);
            if let Err(e) = res {
                thread_data[0].info.print_to_stdout = true;
                return Err(e);
            }
            t.nnue.reinit_from(&t.board, nnue_params);
        }
        thread_data[0].info.clock.start();
        let res = parse_go(&bench_string, thread_data[0].board.turn());
        match res {
            Ok(limit) => thread_data[0].info.clock.set_limit(limit),
            Err(e) => {
                thread_data[0].info.print_to_stdout = true;
                return Err(e);
            }
        }
        tt.increase_age();
        search_position(&pool, &mut thread_data);
        node_sum += thread_data[0].info.nodes.get_global();
        if matches!(benchcmd, "benchfull" | "openbench") {
            println!(
                "{fen:<max_fen_len$} | {:>7} nodes",
                thread_data[0].info.nodes.get_global()
            );
        }
    }
    let time = start.elapsed();
    #[allow(clippy::cast_precision_loss)]
    let nps = node_sum as f64 / time.as_secs_f64();
    if benchcmd == "openbench" {
        println!("{node_sum} nodes {nps:.0} nps");
    } else {
        println!(
            "{node_sum} nodes in {time:.3}s ({nps:.0} nps)",
            time = time.as_secs_f64()
        );
    }
    thread_data[0].info.print_to_stdout = true;

    // logging for permutation
    #[cfg(feature = "nnz-counts")]
    std::fs::write(
        "correlations.txt",
        format!(
            "{:?}",
            network::layers::NNZ_COUNTS
                .iter()
                .map(|c| c
                    .iter()
                    .map(|c| c.load(Ordering::Relaxed))
                    .collect::<Vec<u64>>())
                .collect::<Vec<Vec<u64>>>()
        ),
    )?;
    #[cfg(feature = "nnz-counts")]
    {
        let count = NNZ_COUNT.load(Ordering::Relaxed);
        let denom = NNZ_DENOM.load(Ordering::Relaxed);
        #[allow(clippy::cast_precision_loss)]
        let ratio = count as f64 / denom as f64 * 100.0;
        println!("NNZ COUNT: {count}");
        println!("NNZ DENOM: {denom}");
        println!("NNZ RATIO: {ratio:.2}%");
    }

    Ok(())
}

/// Benchmark the go UCI command.
pub fn go_benchmark(nnue_params: &'static NNUEParams) -> anyhow::Result<()> {
    #![allow(clippy::cast_precision_loss)]
    const COUNT: usize = 1000;
    const THREADS: usize = 250;
    let stopped = AtomicBool::new(false);
    let nodes = AtomicU64::new(0);
    let pool = threadpool::make_worker_threads(THREADS);
    let mut tt = TT::new();
    tt.resize(16 * MEGABYTE, &pool);
    let mut thread_data = make_thread_data(
        &Board::default(),
        tt.view(),
        nnue_params,
        &stopped,
        &nodes,
        &pool,
    )?;
    thread_data[0].info.print_to_stdout = false;
    let start = std::time::Instant::now();
    for _ in 0..COUNT {
        thread_data[0].info.clock.start();
        let limit = parse_go(
            std::hint::black_box("go wtime 0 btime 0 winc 0 binc 0"),
            thread_data[0].board.turn(),
        )?;
        thread_data[0].info.clock.set_limit(limit);
        tt.increase_age();
        std::hint::black_box(search_position(&pool, &mut thread_data));
    }
    let elapsed = start.elapsed();
    let micros = elapsed.as_secs_f64() * (1_000_000.0 / COUNT as f64);
    println!("{micros} us per parse_go");
    Ok(())
}

fn block_perft(depth: usize, pos: &mut Board) {
    #![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    let start_time = Instant::now();
    let nodes = perft::perft(pos, depth);
    let elapsed = start_time.elapsed();
    let nps = nodes as f64 / elapsed.as_secs_f64();
    println!(
        "info depth {depth} nodes {nodes} time {elapsed} nps {nps:.0}",
        elapsed = elapsed.as_millis()
    );
}

fn divide_perft(depth: usize, pos: &mut Board) {
    #![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    let start_time = Instant::now();
    let mut nodes = 0;
    let mut ml = MoveList::new();
    pos.generate_moves(&mut ml);
    for &m in ml.iter_moves() {
        if !pos.is_legal(m) {
            continue;
        }
        pos.make_move_simple(m);
        let arm_nodes = perft::perft(pos, depth - 1);
        nodes += arm_nodes;
        println!(
            "{}: {arm_nodes}",
            m.display(CHESS960.load(Ordering::Relaxed))
        );
        pos.unmake_move_base();
    }
    let elapsed = start_time.elapsed();
    println!(
        "info depth {depth} nodes {nodes} time {elapsed} nps {nps:.0}",
        elapsed = elapsed.as_millis(),
        nps = nodes as f64 / elapsed.as_secs_f64()
    );
}

fn do_newgame(
    tt: &TT,
    thread_data: &mut [Box<ThreadData>],
    pool: &[threadpool::WorkerThread],
) -> anyhow::Result<()> {
    tt.clear(pool);
    for t in thread_data {
        parse_position("position startpos\n", &mut t.board)
            .with_context(|| "Failed to set startpos")?;
        t.clear_tables();
    }
    Ok(())
}

/// Normalizes the internal value as reported by evaluate or search
/// to the UCI centipawn result used in output. This value is derived from
/// [the WLD model](https://github.com/vondele/WLD_model) such that Viridithas
/// outputs an advantage of 100 centipawns for a position if the engine has a
/// 50% probability to win from this position in selfplay at 16s+0.16s time control.
const NORMALISE_TO_PAWN_VALUE: i32 = 229;

fn wdl_model(eval: i32, ply: usize) -> (i32, i32, i32) {
    #![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    const AS: [f64; 4] = [6.871_558_62, -39.652_263_91, 90.684_603_52, 170.669_963_64];
    const BS: [f64; 4] = [
        -7.198_907_10,
        56.139_471_85,
        -139.910_911_83,
        182.810_074_27,
    ];
    debug_assert_eq!(
        NORMALISE_TO_PAWN_VALUE,
        AS.iter().sum::<f64>().round() as i32,
        "AS sum should be {NORMALISE_TO_PAWN_VALUE} but is {:.2}",
        AS.iter().sum::<f64>()
    );

    let m = std::cmp::min(240, ply) as f64 / 64.0;

    let a = AS[0].mul_add(m, AS[1]).mul_add(m, AS[2]).mul_add(m, AS[3]);
    let b = BS[0].mul_add(m, BS[1]).mul_add(m, BS[2]).mul_add(m, BS[3]);

    let x = f64::clamp(
        f64::from(100 * eval) / f64::from(NORMALISE_TO_PAWN_VALUE),
        -2000.0,
        2000.0,
    );
    let win = 1.0 / (1.0 + f64::exp((a - x) / b));
    let loss = 1.0 / (1.0 + f64::exp((a + x) / b));
    let draw = 1.0 - win - loss;

    // Round to the nearest integer
    (
        (1000.0 * win).round() as i32,
        (1000.0 * draw).round() as i32,
        (1000.0 * loss).round() as i32,
    )
}

struct UciWdlFormat {
    eval: i32,
    ply: usize,
}
impl Display for UciWdlFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (wdl_w, wdl_d, wdl_l) = wdl_model(self.eval, self.ply);
        write!(f, "{wdl_w} {wdl_d} {wdl_l}")
    }
}

struct PrettyUciWdlFormat {
    eval: i32,
    ply: usize,
}
impl Display for PrettyUciWdlFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        #![allow(clippy::cast_possible_truncation)]
        let (wdl_w, wdl_d, wdl_l) = wdl_model(self.eval, self.ply);
        let wdl_w = (f64::from(wdl_w) / 10.0).round() as i32;
        let wdl_d = (f64::from(wdl_d) / 10.0).round() as i32;
        let wdl_l = (f64::from(wdl_l) / 10.0).round() as i32;
        write!(
            f,
            "\u{001b}[38;5;243m{wdl_w:3.0}W {wdl_d:3.0}D {wdl_l:3.0}L\u{001b}[0m",
        )
    }
}

pub fn format_wdl(eval: i32, ply: usize) -> impl Display {
    UciWdlFormat { eval, ply }
}
pub fn pretty_format_wdl(eval: i32, ply: usize) -> impl Display {
    PrettyUciWdlFormat { eval, ply }
}

struct PrettyCounterFormat(u64);
impl Display for PrettyCounterFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        #![allow(clippy::match_overlapping_arm)]
        match self.0 {
            ..1_000 => write!(f, "{:>4}", self.0),
            ..1_000_000 => write!(f, "{:>3}K", self.0 / 1_000),
            ..1_000_000_000 => write!(f, "{:>3}M", self.0 / 1_000_000),
            ..1_000_000_000_000 => write!(f, "{:>3}G", self.0 / 1_000_000_000),
            _ => write!(f, "{:>3}T", self.0 / 1_000_000_000_000),
        }
    }
}
pub const fn pretty_format_counter(v: u64) -> impl Display {
    PrettyCounterFormat(v)
}
