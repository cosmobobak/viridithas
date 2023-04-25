#![deny(clippy::panic, clippy::unwrap_used, clippy::todo, clippy::unimplemented)]

use std::{
    fmt::{self, Display},
    io::Write,
    num::{ParseFloatError, ParseIntError},
    str::{FromStr, ParseBoolError},
    sync::{
        atomic::{self, AtomicBool, AtomicI32, AtomicU8, AtomicUsize, Ordering},
        mpsc, Mutex,
    },
    time::Instant,
};

use crate::{
    bench::BENCH_POSITIONS,
    board::{
        evaluation::{
            is_game_theoretic_score, is_mate_score, parameters::EvalParams, MATE_SCORE,
            TB_WIN_SCORE,
        },
        movegen::MoveList,
        Board,
    },
    definitions::{MAX_DEPTH, MEGABYTE},
    errors::{FenParseError, MoveParseError},
    nnue, perft,
    piece::Colour,
    search::{parameters::SearchParams, LMTable},
    searchinfo::SearchInfo,
    timemgmt::SearchLimit,
    tablebases,
    threadlocal::ThreadData,
    transpositiontable::TT,
    NAME, VERSION,
};

const UCI_DEFAULT_HASH_MEGABYTES: usize = 16;
const UCI_MAX_HASH_MEGABYTES: usize = 1_048_576;
const UCI_MAX_THREADS: usize = 512;
const UCI_MAX_MULTIPV: usize = 500;

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
fn parse_position(text: &str, pos: &mut Board) -> Result<(), UciError> {
    let mut parts = text.split_ascii_whitespace();
    let command = parts.next().ok_or_else(|| {
        UciError::UnexpectedCommandTermination("No command in parse_position".into())
    })?;
    if command != "position" {
        return Err(UciError::InvalidFormat("Expected 'position'".into()));
    }
    let determiner = parts.next().ok_or_else(|| {
        UciError::UnexpectedCommandTermination("No determiner after \"position\"".into())
    })?;
    if determiner == "startpos" {
        pos.set_startpos();
        let moves = parts.next(); // skip "moves"
        if !(matches!(moves, Some("moves") | None)) {
            return Err(UciError::InvalidFormat(
                "Expected either \"moves\" or no content to follow \"startpos\".".into(),
            ));
        }
    } else {
        if determiner != "fen" {
            return Err(UciError::InvalidFormat(format!(
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
        pos.set_from_fen(&fen)?;
    }
    for san in parts {
        pos.zero_height(); // stuff breaks really hard without this lmao
        let m = pos.parse_uci(san)?;
        pos.make_move_base(m);
    }
    pos.zero_height();
    Ok(())
}

pub static GO_MATE_MAX_DEPTH: AtomicUsize = AtomicUsize::new(MAX_DEPTH.ply_to_horizon());
fn parse_go(text: &str, info: &mut SearchInfo, pos: &mut Board) -> Result<(), UciError> {
    #![allow(clippy::too_many_lines)]
    let mut depth: Option<i32> = None;
    let mut moves_to_go: Option<u64> = None;
    let mut movetime: Option<u64> = None;
    let mut clocks: [Option<i64>; 2] = [None, None];
    let mut incs: [Option<i64>; 2] = [None, None];
    let mut nodes: Option<u64> = None;

    let mut parts = text.split_ascii_whitespace();
    let command = parts
        .next()
        .ok_or_else(|| UciError::UnexpectedCommandTermination("No command in parse_go".into()))?;
    if command != "go" {
        return Err(UciError::InvalidFormat("Expected \"go\"".into()));
    }

    while let Some(part) = parts.next() {
        match part {
            "depth" => depth = Some(part_parse("depth", parts.next())?),
            "movestogo" => moves_to_go = Some(part_parse("movestogo", parts.next())?),
            "movetime" => movetime = Some(part_parse("movetime", parts.next())?),
            "wtime" => clocks[pos.turn().index()] = Some(part_parse("wtime", parts.next())?),
            "btime" => clocks[pos.turn().flip().index()] = Some(part_parse("btime", parts.next())?),
            "winc" => incs[pos.turn().index()] = Some(part_parse("winc", parts.next())?),
            "binc" => incs[pos.turn().flip().index()] = Some(part_parse("binc", parts.next())?),
            "infinite" => info.time_manager.limit = SearchLimit::Infinite,
            "mate" => {
                let mate_distance: usize = part_parse("mate", parts.next())?;
                let ply = mate_distance * 2; // gives padding when we're giving mate, but whatever
                GO_MATE_MAX_DEPTH.store(ply, Ordering::SeqCst);
                info.time_manager.limit = SearchLimit::Mate { ply };
            }
            "nodes" => nodes = Some(part_parse("nodes", parts.next())?),
            other => return Err(UciError::InvalidFormat(format!("Unknown term: {other}"))),
        }
    }
    if !matches!(info.time_manager.limit, SearchLimit::Mate { .. }) {
        GO_MATE_MAX_DEPTH.store(MAX_DEPTH.ply_to_horizon(), Ordering::SeqCst);
    }

    if let Some(movetime) = movetime {
        info.time_manager.limit = SearchLimit::Time(movetime);
    }
    if let Some(depth) = depth {
        info.time_manager.limit = SearchLimit::Depth(depth.into());
    }

    if let [Some(our_clock), Some(their_clock)] = clocks {
        let [our_inc, their_inc] = [incs[0].unwrap_or(0), incs[1].unwrap_or(0)];
        let our_clock: u64 = our_clock.try_into().unwrap_or(0);
        let their_clock: u64 = their_clock.try_into().unwrap_or(0);
        let our_inc: u64 = our_inc.try_into().unwrap_or(0);
        let their_inc: u64 = their_inc.try_into().unwrap_or(0);
        // let moves_to_go = moves_to_go.unwrap_or_else(|| pos.predicted_moves_left());
        let (time_window, max_time_window) =
            SearchLimit::compute_time_windows(our_clock, moves_to_go, our_inc, &info.search_params);
        info.time_manager.limit = SearchLimit::Dynamic {
            our_clock,
            their_clock,
            our_inc,
            their_inc,
            moves_to_go: moves_to_go.unwrap_or_else(|| pos.predicted_moves_left()),
            max_time_window,
            time_window,
        };
    } else if clocks.iter().chain(incs.iter()).any(Option::is_some) {
        return Err(UciError::InvalidFormat(
            "at least one of [wtime, btime, winc, binc] provided, but not all.".into(),
        ));
    }

    if let Some(nodes) = nodes {
        info.time_manager.limit = SearchLimit::Nodes(nodes);
    }

    info.time_manager.start_time = Instant::now();

    Ok(())
}

fn part_parse<T>(target: &str, next_part: Option<&str>) -> Result<T, UciError>
where
    T: FromStr,
    <T as FromStr>::Err: Display,
{
    let next_part =
        next_part.ok_or_else(|| UciError::InvalidFormat(format!("nothing after \"{target}\"")))?;
    let value = next_part.parse();
    value.map_err(|e| {
        UciError::InvalidFormat(format!(
            "value for {target} is not a number: {e}, tried to parse {next_part}"
        ))
    })
}

struct SetOptions {
    pub search_config: SearchParams,
    pub hash_mb: Option<usize>,
    pub threads: Option<usize>,
}

#[allow(clippy::too_many_lines)]
fn parse_setoption(
    text: &str,
    _info: &mut SearchInfo,
    pre_config: SetOptions,
) -> Result<SetOptions, UciError> {
    use UciError::UnexpectedCommandTermination;
    let mut parts = text.split_ascii_whitespace();
    let Some(_) = parts.next() else {
        return Err(UnexpectedCommandTermination("no \"setoption\" found".into()));
    };
    let Some(name_part) = parts.next() else {
        return Err(UciError::InvalidFormat("no \"name\" after \"setoption\"".into()));
    };
    if name_part != "name" {
        return Err(UciError::InvalidFormat(format!(
            "unexpected character after \"setoption\", expected \"name\", got \"{name_part}\". Did you mean \"setoption name {name_part}\"?"
        )));
    }
    let opt_name = parts.next().ok_or_else(|| {
        UnexpectedCommandTermination("no option name given after \"setoption name\"".into())
    })?;
    let Some(value_part) = parts.next() else {
        return Err(UciError::InvalidFormat("no \"value\" after \"setoption name {opt_name}\"".into()));
    };
    if value_part != "value" {
        return Err(UciError::InvalidFormat(format!(
            "unexpected character after \"setoption name {opt_name}\", expected \"value\", got \"{value_part}\". Did you mean \"setoption name {opt_name} value {value_part}\"?"
        )));
    }
    let opt_value = parts.next().ok_or_else(|| {
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
                return Err(UciError::InvalidFormat(e.to_string()));
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
                return Err(UciError::IllegalValue(format!(
                    "Hash value must be between 1 and {UCI_MAX_HASH_MEGABYTES}"
                )));
            }
            out.hash_mb = Some(value);
        }
        "Threads" => {
            let value: usize = opt_value.parse()?;
            if !(value > 0 && value <= UCI_MAX_THREADS) {
                // "Threads value must be between 1 and {UCI_MAX_THREADS}"
                return Err(UciError::IllegalValue(format!(
                    "Threads value must be between 1 and {UCI_MAX_THREADS}"
                )));
            }
            out.threads = Some(value);
        }
        "MultiPV" => {
            let value: usize = opt_value.parse()?;
            if !(value > 0 && value <= UCI_MAX_MULTIPV) {
                // "MultiPV value must be between 1 and {UCI_MAX_MULTIPV}"
                return Err(UciError::IllegalValue(format!(
                    "MultiPV value must be between 1 and {UCI_MAX_MULTIPV}"
                )));
            }
            MULTI_PV.store(value, Ordering::SeqCst);
        }
        "PrettyPrint" => {
            let value: bool = opt_value.parse()?;
            PRETTY_PRINT.store(value, Ordering::SeqCst);
        }
        "UseNNUE" => {
            let value: bool = opt_value.parse()?;
            USE_NNUE.store(value, Ordering::SeqCst);
        }
        "SyzygyPath" => {
            let path = opt_value.to_string();
            tablebases::probe::init(&path);
            if let Ok(mut lock) = SYZYGY_PATH.lock() {
                *lock = path;
                SYZYGY_ENABLED.store(true, Ordering::SeqCst);
            } else {
                return Err(UciError::InternalError("failed to take lock on SyzygyPath".into()));
            }
        }
        "SyzygyProbeLimit" => {
            let value: u8 = opt_value.parse()?;
            if value > 6 {
                return Err(UciError::IllegalValue(
                    "SyzygyProbeLimit value must be between 0 and 6".to_string(),
                ));
            }
            SYZYGY_PROBE_LIMIT.store(value, Ordering::SeqCst);
        }
        "SyzygyProbeDepth" => {
            let value: i32 = opt_value.parse()?;
            if !(1..=100).contains(&value) {
                return Err(UciError::IllegalValue(
                    "SyzygyProbeDepth value must be between 0 and 100".to_string(),
                ));
            }
            SYZYGY_PROBE_DEPTH.store(value, Ordering::SeqCst);
        }
        _ => {
            eprintln!("info string ignoring option {opt_name}, type \"uci\" for a list of options");
        }
    }
    Ok(out)
}

static KEEP_RUNNING: AtomicBool = AtomicBool::new(true);

fn stdin_reader() -> mpsc::Receiver<String> {
    let (sender, receiver) = mpsc::channel();
    std::thread::Builder::new()
        .name("stdin-reader".into())
        .spawn(|| stdin_reader_worker(sender))
        .expect("Couldn't start stdin reader worker thread");
    receiver
}

fn stdin_reader_worker(sender: mpsc::Sender<String>) {
    let mut linebuf = String::with_capacity(128);
    while std::io::stdin().read_line(&mut linebuf).is_ok() {
        let cmd = linebuf.trim();
        if cmd.is_empty() {
            linebuf.clear();
            continue;
        }
        if sender.send(cmd.to_owned()).is_err() {
            break;
        }
        if !KEEP_RUNNING.load(atomic::Ordering::SeqCst) {
            break;
        }
        linebuf.clear();
    }
    std::mem::drop(sender);
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
        } else if is_game_theoretic_score(self.0) {
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
        let white_pov = if self.1 == Colour::WHITE { self.0 } else { -self.0 };
        if is_mate_score(white_pov) {
            let plies_to_mate = MATE_SCORE - white_pov.abs();
            let moves_to_mate = (plies_to_mate + 1) / 2;
            if white_pov > 0 {
                write!(f, "   #{moves_to_mate:<2}")?;
            } else {
                write!(f, "  #-{moves_to_mate:<2}")?;
            }
        } else if is_game_theoretic_score(white_pov) {
            let plies_to_tb = TB_WIN_SCORE - white_pov.abs();
            if white_pov > 0 {
                write!(f, " +TB{plies_to_tb:<2}")?;
            } else {
                write!(f, " -TB{plies_to_tb:<2}")?;
            }
        } else {
            let white_pov = white_pov * 100 / NORMALISE_TO_PAWN_VALUE;
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
            write!(f, "{seconds:2}.{r_millis:02}s", r_millis = millis % 1000 / 10)
        } else {
            write!(f, "{millis:4}ms")
        }
    }
}
pub const fn format_time(millis: u128) -> HumanTimeFormatWrapper {
    HumanTimeFormatWrapper { millis }
}

fn print_uci_response(full: bool) {
    println!("id name {NAME} {VERSION}");
    println!("id author Cosmo");
    println!("option name Hash type spin default {UCI_DEFAULT_HASH_MEGABYTES} min 1 max {UCI_MAX_HASH_MEGABYTES}");
    println!("option name Threads type spin default 1 min 1 max 512");
    println!("option name PrettyPrint type check default false");
    println!("option name UseNNUE type check default true");
    println!("option name SyzygyPath type string default <empty>");
    println!("option name SyzygyProbeLimit type spin default 6 min 0 max 6");
    println!("option name SyzygyProbeDepth type spin default 1 min 1 max 100");
    // println!("option name MultiPV type spin default 1 min 1 max 500");
    if full {
        for (id, default) in SearchParams::default().ids_with_values() {
            println!("option name {id} type spin default {default} min -999999 max 999999");
        }
    }
    println!("uciok");
}

pub static PRETTY_PRINT: AtomicBool = AtomicBool::new(true);
pub static USE_NNUE: AtomicBool = AtomicBool::new(true);
pub static SYZYGY_PROBE_LIMIT: AtomicU8 = AtomicU8::new(6);
pub static SYZYGY_PROBE_DEPTH: AtomicI32 = AtomicI32::new(1);
pub static SYZYGY_PATH: Mutex<String> = Mutex::new(String::new());
pub static SYZYGY_ENABLED: AtomicBool = AtomicBool::new(false);
pub static MULTI_PV: AtomicUsize = AtomicUsize::new(1);
pub fn is_multipv() -> bool {
    MULTI_PV.load(Ordering::SeqCst) > 1
}

#[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
pub fn main_loop(params: EvalParams, global_bench: bool) {
    let mut pos = Board::default();

    let mut tt = TT::new();
    tt.resize(UCI_DEFAULT_HASH_MEGABYTES * MEGABYTE); // default hash size

    let stopped = AtomicBool::new(false);
    let stdin = Mutex::new(stdin_reader());
    let mut info = SearchInfo::new(&stopped);
    info.set_stdin(&stdin);
    info.eval_params = params;

    let mut thread_data = vec![ThreadData::new(0, &pos)];
    pos.refresh_psqt(&info);

    println!("{NAME} {VERSION} by Cosmo");

    if global_bench {
        bench(&mut info, &mut pos, &mut tt, &mut thread_data, "openbench").expect("bench failed");
        return;
    }

    loop {
        std::io::stdout().flush().expect("couldn't flush stdout");
        let line = stdin
            .lock()
            .expect("failed to take lock on stdin")
            .recv()
            .expect("couldn't receive from stdin");
        let input = line.trim();

        let res = match input {
            "\n" => continue,
            "uci" => {
                #[cfg(feature = "tuning")]
                print_uci_response(true);
                #[cfg(not(feature = "tuning"))]
                print_uci_response(false);
                PRETTY_PRINT.store(false, Ordering::SeqCst);
                Ok(())
            }
            "ucifull" => {
                print_uci_response(true);
                PRETTY_PRINT.store(false, Ordering::SeqCst);
                Ok(())
            }
            arg @ ("ucidump" | "ucidumpfull") => {
                // dump the values of the current UCI options
                println!("Hash: {}", tt.size() / MEGABYTE);
                println!("Threads: {}", thread_data.len());
                println!("PrettyPrint: {}", PRETTY_PRINT.load(Ordering::SeqCst));
                println!("UseNNUE: {}", USE_NNUE.load(Ordering::SeqCst));
                println!("SyzygyPath: {}", SYZYGY_PATH.lock().expect("failed to lock syzygy path"));
                println!("SyzygyProbeLimit: {}", SYZYGY_PROBE_LIMIT.load(Ordering::SeqCst));
                println!("SyzygyProbeDepth: {}", SYZYGY_PROBE_DEPTH.load(Ordering::SeqCst));
                // println!("MultiPV: {}", MULTI_PV.load(Ordering::SeqCst));
                if arg == "ucidumpfull" {
                    for (id, default) in SearchParams::default().ids_with_values() {
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
                info.quit = true;
                break;
            }
            "ucinewgame" => do_newgame(&mut pos, &tt),
            "eval" => {
                let eval = if pos.in_check::<{ Board::US }>() {
                    0
                } else {
                    pos.evaluate::<true>(
                        &info,
                        thread_data.first_mut().expect("the thread headers are empty."),
                        0,
                    )
                };
                println!("{eval}");
                Ok(())
            }
            "show" => {
                println!("{pos}");
                Ok(())
            }
            "nnuebench" => {
                nnue::network::inference_benchmark(&thread_data[0].nnue);
                Ok(())
            }
            input if input.starts_with("setoption") => {
                let pre_config = SetOptions {
                    search_config: info.search_params.clone(),
                    hash_mb: None,
                    threads: None,
                };
                let res = parse_setoption(input, &mut info, pre_config);
                match res {
                    Ok(conf) => {
                        info.search_params = conf.search_config;
                        info.lm_table = LMTable::new(&info.search_params);
                        if let Some(hash_mb) = conf.hash_mb {
                            let new_size = hash_mb * MEGABYTE;
                            tt.resize(new_size);
                        }
                        if let Some(threads) = conf.threads {
                            thread_data = (0..threads)
                                .zip(std::iter::repeat(&pos))
                                .map(|(i, p)| ThreadData::new(i, p))
                                .collect();
                        }
                        Ok(())
                    }
                    Err(err) => Err(err),
                }
            }
            input if input.starts_with("position") => {
                let res = parse_position(input, &mut pos);
                if res.is_ok() {
                    for t in &mut thread_data {
                        t.nnue.refresh_acc(&pos);
                    }
                    pos.refresh_psqt(&info);
                }
                res
            }
            input if input.starts_with("go perft") || input.starts_with("perft") => {
                let tail = input.trim_start_matches("go perft ").trim_start_matches("perft ");
                match tail.split_whitespace().next() {
                    Some("divide" | "split") => {
                        let depth = tail.trim_start_matches("divide ").trim_start_matches("split ");
                        depth
                            .parse::<usize>()
                            .map_err(|_| {
                                UciError::InvalidFormat(format!(
                                    "cannot parse \"{depth}\" as usize"
                                ))
                            })
                            .map(|depth| divide_perft(depth, &mut pos))
                    }
                    Some(depth) => depth
                        .parse::<usize>()
                        .map_err(|_| {
                            UciError::InvalidFormat(format!("cannot parse \"{depth}\" as usize"))
                        })
                        .map(|depth| block_perft(depth, &mut pos)),
                    None => Err(UciError::InvalidFormat(
                        "expected a depth after 'go perft'".to_string(),
                    )),
                }
            }
            input if input.starts_with("go") => {
                let res = parse_go(input, &mut info, &mut pos);
                if res.is_ok() {
                    tt.increase_age();
                    if USE_NNUE.load(Ordering::SeqCst) {
                        pos.search_position::<true>(&mut info, &mut thread_data, tt.view());
                    } else {
                        pos.search_position::<false>(&mut info, &mut thread_data, tt.view());
                    }
                }
                res
            }
            benchcmd @ ("bench" | "benchfull") => {
                bench(&mut info, &mut pos, &mut tt, &mut thread_data, benchcmd)
            }
            _ => Err(UciError::UnknownCommand(input.to_string())),
        };

        if let Err(e) = res {
            println!("info string {e}");
        }

        if info.quit {
            // quit can be set true in parse_go
            break;
        }
    }
    KEEP_RUNNING.store(false, atomic::Ordering::SeqCst);
}

fn bench(
    info: &mut SearchInfo,
    pos: &mut Board,
    tt: &mut TT,
    thread_data: &mut [ThreadData],
    benchcmd: &str,
) -> Result<(), UciError> {
    info.print_to_stdout = false;
    let mut node_sum = 0u64;
    let start = Instant::now();
    for fen in BENCH_POSITIONS {
        let res = do_newgame(pos, tt);
        if let Err(e) = res {
            info.print_to_stdout = true;
            return Err(e);
        }
        let res = parse_position(&format!("position fen {fen}\n"), pos);
        if let Err(e) = res {
            info.print_to_stdout = true;
            return Err(e);
        }
        for t in thread_data.iter_mut() {
            t.nnue.refresh_acc(pos);
        }
        pos.refresh_psqt(&*info);
        let res = parse_go("go depth 12\n", info, pos);
        if let Err(e) = res {
            info.print_to_stdout = true;
            return Err(e);
        }
        tt.increase_age();
        if USE_NNUE.load(Ordering::SeqCst) {
            pos.search_position::<true>(info, thread_data, tt.view());
        } else {
            pos.search_position::<false>(info, thread_data, tt.view());
        }
        node_sum += info.nodes;
        if matches!(benchcmd, "benchfull" | "openbench") {
            println!("{fen} has {} nodes", info.nodes);
        }
    }
    let time = start.elapsed();
    #[allow(clippy::cast_precision_loss)]
    let nps = node_sum as f64 / time.as_secs_f64();
    if benchcmd == "openbench" {
        println!("{node_sum} nodes {nps:.0} nps");
    } else {
        println!("{node_sum} nodes in {time:.3}s ({nps:.0} nps)", time = time.as_secs_f64());
    }
    info.print_to_stdout = true;
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
    #![allow(clippy::cast_possible_truncation)]
    let start_time = Instant::now();
    let mut nodes = 0;
    let mut ml = MoveList::new();
    pos.generate_moves(&mut ml);
    for &m in ml.iter() {
        if !pos.make_move_base(m) {
            continue;
        }
        let arm_nodes = perft::perft(pos, depth - 1);
        nodes += arm_nodes;
        println!("{m}: {arm_nodes}");
        pos.unmake_move_base();
    }
    let elapsed = start_time.elapsed();
    println!(
        "info depth {depth} nodes {nodes} time {elapsed} nps {nps}",
        elapsed = elapsed.as_millis(),
        nps = nodes * 1000 / elapsed.as_millis() as u64
    );
}

fn do_newgame(pos: &mut Board, tt: &TT) -> Result<(), UciError> {
    let res = parse_position("position startpos\n", pos);
    tt.clear();
    res
}

/// Normalizes the internal value as reported by evaluate or search
/// to the UCI centipawn result used in output. This value is derived from
/// the `win_rate_model` such that Viridithas outputs an advantage of
/// "100 centipawns" for a position if the engine has a 50% probability to win
/// from this position in selfplay at 8s+0.08s time control.
const NORMALISE_TO_PAWN_VALUE: i32 = 269;
fn win_rate_model(eval: i32, ply: usize) -> i32 {
    #![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    const AS: [f64; 4] = [-26.994_571_68, 207.590_501_28, -214.904_939_28, 303.735_316_05];
    const BS: [f64; 4] = [-19.224_614_27, 141.390_056_35, -283.690_790_60, 378.025_261_86];
    let m = min!(240.0, ply as f64) / 64.0;
    debug_assert_eq!(NORMALISE_TO_PAWN_VALUE, AS.iter().sum::<f64>() as i32);
    let a = AS[0].mul_add(m, AS[1]).mul_add(m, AS[2]).mul_add(m, AS[3]);
    let b = BS[0].mul_add(m, BS[1]).mul_add(m, BS[2]).mul_add(m, BS[3]);

    // Transform the eval to centipawns with limited range
    let x = f64::from(eval.clamp(-4000, 4000));

    // Return the win rate in per mille units rounded to the nearest value
    (0.5 + 1000.0 / (1.0 + f64::exp((a - x) / b))) as i32
}

struct UciWdlFormat {
    eval: i32,
    ply: usize,
}
impl Display for UciWdlFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let wdl_w = win_rate_model(self.eval, self.ply);
        let wdl_l = win_rate_model(-self.eval, self.ply);
        let wdl_d = 1000 - wdl_w - wdl_l;
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
        let wdl_w = win_rate_model(self.eval, self.ply);
        let wdl_l = win_rate_model(-self.eval, self.ply);
        let wdl_d = 1000 - wdl_w - wdl_l;
        let wdl_w = (f64::from(wdl_w) / 10.0).round() as i32;
        let wdl_d = (f64::from(wdl_d) / 10.0).round() as i32;
        let wdl_l = (f64::from(wdl_l) / 10.0).round() as i32;
        write!(f, "\u{001b}[38;5;243m{wdl_w:3.0}%W {wdl_d:3.0}%D {wdl_l:3.0}%L\u{001b}[0m",)
    }
}

pub fn format_wdl(eval: i32, ply: usize) -> impl Display {
    UciWdlFormat { eval, ply }
}
pub fn pretty_format_wdl(eval: i32, ply: usize) -> impl Display {
    PrettyUciWdlFormat { eval, ply }
}
