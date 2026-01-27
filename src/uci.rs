#![deny(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::todo,
    clippy::unimplemented
)]

pub mod fmt;

use std::{
    io::Write as _,
    sync::{
        Mutex, Once,
        atomic::{self, AtomicBool, AtomicI32, AtomicU8, AtomicU64, AtomicUsize, Ordering},
        mpsc,
    },
    time::Instant,
};

use crate::{
    NAME, VERSION,
    bench::BENCH_POSITIONS,
    chess::{
        CHESS960,
        board::{
            Board,
            movegen::{self, MoveList},
        },
        fen::Fen,
        piece::Colour,
        quick::Quick,
    },
    cuckoo,
    errors::{GoParseError, PerftParseError, PositionParseError, SetOptionParseError, UciError},
    evaluation::evaluate,
    nnue::{self, network::NNUEParams},
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
const BENCH_DEPTH: usize = 14;
const BENCH_THREADS: usize = 1;

static SET_TERM: Once = Once::new();
static STDIN_READER_THREAD_KEEP_RUNNING: AtomicBool = AtomicBool::new(true);
pub static QUIT: AtomicBool = AtomicBool::new(false);
pub static GO_MATE_MAX_DEPTH: AtomicUsize = AtomicUsize::new(MAX_DEPTH);
pub static PRETTY_PRINT: AtomicBool = AtomicBool::new(true);
pub static SYZYGY_PROBE_LIMIT: AtomicU8 = AtomicU8::new(7);
pub static SYZYGY_PROBE_DEPTH: AtomicI32 = AtomicI32::new(1);
pub static CONTEMPT: AtomicI32 = AtomicI32::new(0);

#[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
pub fn main_loop() -> Result<(), UciError> {
    let version_extension = if cfg!(feature = "final-release") {
        ""
    } else {
        "-dev"
    };
    println!("{NAME} {VERSION}{version_extension} by Cosmo");

    let mut worker_threads = threadpool::make_worker_threads(1);

    let mut tt = TT::new();
    tt.resize(UCI_DEFAULT_HASH_MEGABYTES * MEGABYTE, &worker_threads); // default hash size

    let nnue_params =
        NNUEParams::decompress_and_alloc().map_err(|e| UciError::NnueInit(e.to_string()))?;

    let (stdin, stdin_reader_handle) = stdin_reader()?;
    let stdin = Mutex::new(stdin);
    let stopped = AtomicBool::new(false);
    let nodes = AtomicU64::new(0);
    let tbhits = AtomicU64::new(0);
    let mut thread_data = make_thread_data(
        &Board::default(),
        tt.view(),
        nnue_params,
        &stopped,
        &nodes,
        &tbhits,
        &worker_threads,
    )
    .map_err(|e| UciError::NnueInit(e.to_string()))?;
    thread_data[0].info.set_stdin(&stdin);

    loop {
        std::io::stdout().flush()?;
        let Ok(line) = stdin
            .lock()
            .map_err(|_| UciError::Internal("failed to take lock on stdin"))?
            .recv()
        else {
            break;
        };
        let input = line.trim();

        let res: Result<(), UciError> = match input {
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
                let t = thread_data.first_mut();
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
                let t = thread_data.first_mut();
                let eval = if t.board.in_check() {
                    0
                } else {
                    t.nnue.evaluate(t.nnue_params, &t.board)
                };
                println!("{eval}");
                Ok(())
            }
            "show" => {
                let t = thread_data.first_mut();
                println!("{:X}", t.board);
                Ok(())
            }
            "d" | "debug" => {
                let t = thread_data.first_mut();
                println!("{:?}", t.board);
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
            "initcuckoo" => Ok(cuckoo::init()?),
            "initattacks" => Ok(movegen::init_sliders_attacks()?),
            input if is_cmd(input, "setoption") => {
                let pre_config = SetOptions {
                    search_config: thread_data[0].info.conf.clone(),
                    hash_mb: tt.size() / MEGABYTE,
                    threads: thread_data.len(),
                };
                let threads_before = thread_data.len();
                match parse_setoption(input, pre_config) {
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
                            &tbhits,
                            &worker_threads,
                        )
                        .map_err(|e| UciError::NnueInit(e.to_string()))?;

                        for t in &mut thread_data {
                            t.info.conf = conf.search_config.clone();
                            t.info.lm_table = LMTable::new(&t.info.conf);
                            t.info.set_stdin(&stdin);
                        }

                        Ok(())
                    }
                    Err(e) => Err(e.into()),
                }
            }
            input if is_cmd(input, "position") => thread_data
                .iter_mut()
                .try_for_each(|t| {
                    parse_position(input, &mut t.board)?;
                    t.nnue.reinit_from(&t.board, t.nnue_params);
                    Ok::<_, PositionParseError>(())
                })
                .map_err(Into::into),
            input if is_cmd(input, "go perft") || is_cmd(input, "perft") => {
                parse_perft(thread_data.first_mut(), input)
            }
            input if is_cmd(input, "go") => {
                // start the clock *immediately*
                thread_data[0].info.clock.start();

                // if we're in pretty-printing mode, set the terminal properly:
                if PRETTY_PRINT.load(Ordering::SeqCst) {
                    SET_TERM.call_once(|| {
                        term::set_mode_uci();
                    });
                }

                match parse_go(input, thread_data[0].board.turn()) {
                    Ok(search_limit) => {
                        thread_data[0].info.clock.set_limit(search_limit);
                        tt.increase_age();
                        search_position(&worker_threads, &mut thread_data);
                        Ok(())
                    }
                    Err(e) => Err(e.into()),
                }
            }
            "ponderhit" => {
                println!("info error ponderhit given while not searching.");
                Ok(())
            }
            benchcmd @ ("bench" | "benchfull") => {
                bench(benchcmd, &thread_data[0].info.conf, nnue_params, None, None)
            }
            command => {
                // before failing outright, try to parse as a fen:
                if command.contains('/')
                    && let Ok(fen) = Fen::parse_relaxed(command)
                {
                    for t in &mut thread_data {
                        t.board.set_from_fen(&fen);
                        for tok in command.split_whitespace() {
                            if let Ok(mv) =
                                t.board.parse_uci(tok).or_else(|_| t.board.parse_san(tok))
                            {
                                t.board.make_move_simple(mv);
                                t.board.zero_height();
                            }
                        }
                        t.board.zero_height();
                        t.nnue.reinit_from(&t.board, t.nnue_params);
                    }
                    Ok(())
                // then try to quick-ly parse
                } else if let Some(first_40) = command.get(..command.len().min(40))
                    && let Ok(quick) = Quick::parse(first_40)
                {
                    for t in &mut thread_data {
                        t.board.set_from_quick(&quick);
                        for tok in command.split_whitespace() {
                            if let Ok(mv) =
                                t.board.parse_uci(tok).or_else(|_| t.board.parse_san(tok))
                            {
                                t.board.make_move_simple(mv);
                                t.board.zero_height();
                            }
                        }
                        t.board.zero_height();
                        t.nnue.reinit_from(&t.board, t.nnue_params);
                    }
                    Ok(())
                // lastly, attempt to find some legal moves
                // this is a tad iffy, and comes the closest to
                // silently accepting keysmashes
                } else if command.split_whitespace().any(|tok| {
                    thread_data[0]
                        .board
                        .parse_uci(tok)
                        .or_else(|_| thread_data[0].board.parse_san(tok))
                        .is_ok()
                }) {
                    for t in &mut thread_data {
                        for tok in command.split_whitespace() {
                            if let Ok(mv) =
                                t.board.parse_uci(tok).or_else(|_| t.board.parse_san(tok))
                            {
                                t.board.make_move_simple(mv);
                                t.board.zero_height();
                            }
                        }
                        t.board.zero_height();
                        t.nnue.reinit_from(&t.board, t.nnue_params);
                    }
                    Ok(())
                } else {
                    Err(UciError::UnknownCommand(input.to_string()))
                }
            }
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
            .map_err(|_| UciError::Thread("stdin reader thread panicked".to_string()))??;
    }

    #[cfg(feature = "stats")]
    crate::stats::dump_and_plot();

    Ok(())
}

/// Check if `input` is the command `cmd` itself, or starts with `cmd` followed by a space.
fn is_cmd(input: &str, cmd: &str) -> bool {
    input == cmd || (input.starts_with(cmd) && input.as_bytes().get(cmd.len()) == Some(&b' '))
}

// position fen
// position startpos
// ... moves e2e4 e7e5 b7b8q
fn parse_position(text: &str, pos: &mut Board) -> Result<(), PositionParseError> {
    let mut parts = text.split_ascii_whitespace();
    let command = parts.next();
    debug_assert_eq!(
        command,
        Some("position"),
        "parse_position called with non-position command"
    );
    let determiner = parts
        .next()
        .ok_or(PositionParseError::MissingPositionSpecifier)?;
    if determiner == "startpos" {
        pos.set_startpos();
        let moves = parts.next(); // skip "moves"
        if let Some(moves) = moves
            && moves != "moves"
        {
            return Err(PositionParseError::InvalidStartposSuffix(moves.into()));
        }
    } else if determiner == "frc" {
        let index_str = parts.next().ok_or(PositionParseError::MissingFrcIndex)?;
        let index: usize = index_str
            .parse()
            .map_err(|e| PositionParseError::InvalidFrcIndex {
                text: index_str.to_string(),
                source: e,
            })?;
        if index >= 960 {
            #[expect(clippy::cast_possible_truncation)]
            return Err(PositionParseError::FrcIndexOutOfRange(index as u32));
        }
        pos.set_frc_idx(index);
    } else if determiner == "dfrc" {
        let index_str = parts.next().ok_or(PositionParseError::MissingDfrcIndex)?;
        let index: usize = index_str
            .parse()
            .map_err(|e| PositionParseError::InvalidDfrcIndex {
                text: index_str.to_string(),
                source: e,
            })?;
        if index >= 960 * 960 {
            #[expect(clippy::cast_possible_truncation)]
            return Err(PositionParseError::DfrcIndexOutOfRange(index as u32));
        }
        pos.set_dfrc_idx(index);
    } else if determiner == "fen" {
        let mut fen_str = String::new();
        for part in &mut parts {
            if part == "moves" {
                break;
            }
            fen_str.push_str(part);
            fen_str.push(' ');
        }
        let fen = Fen::parse(&fen_str)?;
        pos.set_from_fen(&fen);
    } else {
        return Err(PositionParseError::UnknownPositionSpecifier(
            determiner.to_string(),
        ));
    }
    for san in parts {
        pos.zero_height(); // stuff breaks really hard without this lmao
        let m = pos.parse_uci(san)?;
        pos.make_move_simple(m);
    }
    pos.zero_height();
    Ok(())
}

fn parse_go(text: &str, stm: Colour) -> Result<SearchLimit, GoParseError> {
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
    let command = parts.next().ok_or(GoParseError::EmptyCommand)?;
    debug_assert_eq!(command, "go", "parse_go called with non-go command");

    while let Some(part) = parts.next() {
        match part {
            "depth" => depth = Some(go_part_parse("depth", parts.next())?),
            "movestogo" => moves_to_go = Some(go_part_parse("movestogo", parts.next())?),
            "movetime" => movetime = Some(go_part_parse("movetime", parts.next())?),
            "wtime" => clocks[stm] = Some(go_part_parse("wtime", parts.next())?),
            "btime" => clocks[stm.flip()] = Some(go_part_parse("btime", parts.next())?),
            "winc" => incs[stm] = Some(go_part_parse("winc", parts.next())?),
            "binc" => incs[stm.flip()] = Some(go_part_parse("binc", parts.next())?),
            "infinite" => limit = SearchLimit::Infinite,
            "mate" => {
                let mate_distance: usize = go_part_parse("mate", parts.next())?;
                let ply = mate_distance * 2; // gives padding when we're giving mate, but whatever
                GO_MATE_MAX_DEPTH.store(ply, Ordering::SeqCst);
                limit = SearchLimit::Mate { ply };
            }
            "nodes" => nodes = Some(go_part_parse("nodes", parts.next())?),
            "ponder" => ponder = true,
            other => return Err(GoParseError::UnknownSubcommand(other.to_string())),
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
        return Err(GoParseError::IncompleteTimeControl);
    }

    if let Some(nodes) = nodes {
        limit = SearchLimit::Nodes(nodes);
    }

    if ponder {
        limit = limit.to_pondering();
    }

    Ok(limit)
}

fn go_part_parse<T>(param: &'static str, next_part: Option<&str>) -> Result<T, GoParseError>
where
    T: std::str::FromStr<Err = std::num::ParseIntError>,
{
    let next_part = next_part.ok_or(GoParseError::MissingValue(param))?;
    next_part
        .parse()
        .map_err(|e| GoParseError::InvalidValue { param, source: e })
}

fn parse_perft(t: &mut ThreadData<'_>, input: &str) -> Result<(), UciError> {
    let tail = input
        .strip_prefix("go perft")
        .or_else(|| input.strip_prefix("perft"))
        .unwrap_or("")
        .trim_start();
    match tail.split_whitespace().next() {
        Some("divide" | "split") => {
            let depth_str = tail
                .strip_prefix("divide")
                .or_else(|| tail.strip_prefix("split"))
                .unwrap_or("")
                .trim_start();
            if depth_str.is_empty() {
                return Err(PerftParseError::MissingDepth.into());
            }
            let depth: usize = depth_str
                .parse()
                .map_err(|e| PerftParseError::InvalidDepth {
                    text: depth_str.to_string(),
                    source: e,
                })?;
            divide_perft(depth, &mut t.board);
            Ok(())
        }
        Some(depth_str) => {
            let depth: usize = depth_str
                .parse()
                .map_err(|e| PerftParseError::InvalidDepth {
                    text: depth_str.to_string(),
                    source: e,
                })?;
            block_perft(depth, &mut t.board);
            Ok(())
        }
        None => Err(PerftParseError::MissingDepth.into()),
    }
}

struct SetOptions {
    pub search_config: Config,
    pub hash_mb: usize,
    pub threads: usize,
}

#[allow(clippy::too_many_lines)]
fn parse_setoption(text: &str, pre_config: SetOptions) -> Result<SetOptions, SetOptionParseError> {
    let mut parts = text.split_ascii_whitespace();
    // Skip "setoption"
    let _ = parts.next();
    let name_part = parts
        .next()
        .ok_or(SetOptionParseError::MissingNameKeyword)?;
    if name_part != "name" {
        return Err(SetOptionParseError::ExpectedNameKeyword(
            name_part.to_string(),
        ));
    }
    let opt_name = parts.next().ok_or(SetOptionParseError::MissingOptionName)?;
    let value_part = parts
        .next()
        .ok_or_else(|| SetOptionParseError::ExpectedValueKeyword(String::new()))?;
    if value_part != "value" {
        return Err(SetOptionParseError::ExpectedValueKeyword(
            value_part.to_string(),
        ));
    }
    let opt_value = parts
        .next()
        .ok_or_else(|| SetOptionParseError::MissingOptionValue(opt_name.to_string()))?;
    let mut out = pre_config;
    let id_parser_pairs = out.search_config.ids_with_parsers();
    let mut found_match = false;
    for (param_name, mut parser) in id_parser_pairs {
        if param_name == opt_name {
            let res = parser(opt_value);
            if let Err(e) = res {
                return Err(SetOptionParseError::InvalidTuningParam {
                    name: opt_name.to_string(),
                    message: e.to_string(),
                });
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
            let value: usize =
                opt_value
                    .parse()
                    .map_err(|e| SetOptionParseError::InvalidIntValue {
                        name: "Hash".to_string(),
                        source: e,
                    })?;
            if !(value > 0 && value <= UCI_MAX_HASH_MEGABYTES) {
                return Err(SetOptionParseError::ValueOutOfRange {
                    name: "Hash".to_string(),
                    lo: 1,
                    #[expect(clippy::cast_possible_wrap)]
                    hi: UCI_MAX_HASH_MEGABYTES as i64,
                    #[expect(clippy::cast_possible_wrap)]
                    got: value as i64,
                });
            }
            out.hash_mb = value;
        }
        "Threads" => {
            let value: usize =
                opt_value
                    .parse()
                    .map_err(|e| SetOptionParseError::InvalidIntValue {
                        name: "Threads".to_string(),
                        source: e,
                    })?;
            if !(value > 0 && value <= UCI_MAX_THREADS) {
                return Err(SetOptionParseError::ValueOutOfRange {
                    name: "Threads".to_string(),
                    lo: 1,
                    #[expect(clippy::cast_possible_wrap)]
                    hi: UCI_MAX_THREADS as i64,
                    #[expect(clippy::cast_possible_wrap)]
                    got: value as i64,
                });
            }
            out.threads = value;
        }
        "PrettyPrint" => {
            let value: bool =
                opt_value
                    .parse()
                    .map_err(|e| SetOptionParseError::InvalidBoolValue {
                        name: "PrettyPrint".to_string(),
                        source: e,
                    })?;
            PRETTY_PRINT.store(value, Ordering::SeqCst);
        }
        "SyzygyPath" => {
            let path = opt_value.to_string();
            tablebases::probe::init(&path);
        }
        "SyzygyProbeLimit" => {
            let value: u8 =
                opt_value
                    .parse()
                    .map_err(|e| SetOptionParseError::InvalidIntValue {
                        name: "SyzygyProbeLimit".to_string(),
                        source: e,
                    })?;
            if value > 7 {
                return Err(SetOptionParseError::ValueOutOfRange {
                    name: "SyzygyProbeLimit".to_string(),
                    lo: 0,
                    hi: 7,
                    got: i64::from(value),
                });
            }
            SYZYGY_PROBE_LIMIT.store(value, Ordering::SeqCst);
        }
        "SyzygyProbeDepth" => {
            let value: i32 =
                opt_value
                    .parse()
                    .map_err(|e| SetOptionParseError::InvalidIntValue {
                        name: "SyzygyProbeDepth".to_string(),
                        source: e,
                    })?;
            if !(1..=100).contains(&value) {
                return Err(SetOptionParseError::ValueOutOfRange {
                    name: "SyzygyProbeDepth".to_string(),
                    lo: 1,
                    hi: 100,
                    got: i64::from(value),
                });
            }
            SYZYGY_PROBE_DEPTH.store(value, Ordering::SeqCst);
        }
        "Contempt" => {
            let value: i32 =
                opt_value
                    .parse()
                    .map_err(|e| SetOptionParseError::InvalidIntValue {
                        name: "Contempt".to_string(),
                        source: e,
                    })?;
            if !(-10000..=10000).contains(&value) {
                return Err(SetOptionParseError::ValueOutOfRange {
                    name: "Contempt".to_string(),
                    lo: -10000,
                    hi: 10000,
                    got: i64::from(value),
                });
            }
            CONTEMPT.store(value, Ordering::SeqCst);
        }
        "UCI_Chess960" => {
            let val: bool =
                opt_value
                    .parse()
                    .map_err(|e| SetOptionParseError::InvalidBoolValue {
                        name: "UCI_Chess960".to_string(),
                        source: e,
                    })?;
            CHESS960.store(val, Ordering::SeqCst);
        }
        _ => {
            eprintln!("info string ignoring option {opt_name}, type \"uci\" for a list of options");
        }
    }
    Ok(out)
}

type StdinReader = (
    mpsc::Receiver<String>,
    std::thread::JoinHandle<Result<(), UciError>>,
);

fn stdin_reader() -> Result<StdinReader, std::io::Error> {
    let (sender, receiver) = mpsc::channel();
    let handle = std::thread::Builder::new()
        .name("stdin-reader".into())
        .spawn(|| stdin_reader_worker(sender))?;
    Ok((receiver, handle))
}

fn stdin_reader_worker(sender: mpsc::Sender<String>) -> Result<(), UciError> {
    let mut linebuf = String::with_capacity(128);
    while let Ok(bytes) = std::io::stdin().read_line(&mut linebuf) {
        if bytes == 0 {
            // EOF
            sender
                .send("quit".into())
                .map_err(|e| UciError::Thread(e.to_string()))?;
            QUIT.store(true, Ordering::SeqCst);
            break;
        }
        let cmd = linebuf.trim();
        if cmd.is_empty() {
            linebuf.clear();
            continue;
        }
        sender
            .send(cmd.to_owned())
            .map_err(|e| UciError::Thread(e.to_string()))?;
        if !STDIN_READER_THREAD_KEEP_RUNNING.load(atomic::Ordering::SeqCst) {
            break;
        }
        linebuf.clear();
    }

    std::mem::drop(sender);

    Ok(())
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
    println!("option name SyzygyProbeLimit type spin default 7 min 0 max 7");
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

pub fn bench(
    benchcmd: &str,
    search_params: &Config,
    nnue_params: &'static NNUEParams,
    depth: Option<usize>,
    threads: Option<usize>,
) -> Result<(), UciError> {
    let bench_string = format!("go depth {}\n", depth.unwrap_or(BENCH_DEPTH));
    let stopped = AtomicBool::new(false);
    let nodes = AtomicU64::new(0);
    let tbhits = AtomicU64::new(0);
    let pool = threadpool::make_worker_threads(threads.unwrap_or(BENCH_THREADS));
    let mut tt = TT::new();
    tt.resize(16 * MEGABYTE, &pool);
    let mut thread_data = make_thread_data(
        &Board::default(),
        tt.view(),
        nnue_params,
        &stopped,
        &nodes,
        &tbhits,
        &pool,
    )
    .map_err(|e| UciError::NnueInit(e.to_string()))?;
    thread_data[0].info.conf = search_params.clone();
    thread_data[0].info.print_to_stdout = false;
    let mut node_sum = 0u64;
    let start = Instant::now();
    // BENCH_POSITIONS is nonempty, so unwrap is safe
    let max_fen_len = BENCH_POSITIONS.iter().map(|s| s.len()).max().unwrap_or(0);
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
                return Err(e.into());
            }
            t.nnue.reinit_from(&t.board, nnue_params);
        }
        thread_data[0].info.clock.start();
        let res = parse_go(&bench_string, thread_data[0].board.turn());
        match res {
            Ok(limit) => thread_data[0].info.clock.set_limit(limit),
            Err(e) => {
                thread_data[0].info.print_to_stdout = true;
                return Err(e.into());
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

    #[cfg(feature = "stats")]
    crate::stats::dump_and_plot();

    // logging for permutation
    #[cfg(feature = "nnz-counts")]
    std::fs::write(
        "correlations.txt",
        format!(
            "{:?}",
            nnue::network::layers::NNZ_COUNTS
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
pub fn go_benchmark(nnue_params: &'static NNUEParams) -> Result<(), UciError> {
    #![allow(clippy::cast_precision_loss)]
    const COUNT: usize = 1000;
    const THREADS: usize = 250;
    let stopped = AtomicBool::new(false);
    let nodes = AtomicU64::new(0);
    let tbhits = AtomicU64::new(0);
    let pool = threadpool::make_worker_threads(THREADS);
    let mut tt = TT::new();
    tt.resize(16 * MEGABYTE, &pool);
    let mut thread_data = make_thread_data(
        &Board::default(),
        tt.view(),
        nnue_params,
        &stopped,
        &nodes,
        &tbhits,
        &pool,
    )
    .map_err(|e| UciError::NnueInit(e.to_string()))?;
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
) -> Result<(), UciError> {
    tt.clear(pool);
    for t in thread_data {
        parse_position("position startpos\n", &mut t.board)?;
        t.clear_tables();
    }
    Ok(())
}
