use std::{
    fmt::Display,
    io::Write,
    num::{ParseFloatError, ParseIntError},
    sync::{
        atomic::{self, AtomicBool},
        mpsc,
    }, time::Duration,
};

use crate::{
    board::{
        evaluation::{is_mate_score, parameters::EvalParams, MATE_SCORE},
        Board,
    },
    definitions::{BLACK, MAX_DEPTH, WHITE},
    errors::{FenParseError, MoveParseError},
    search::parameters::SearchParams,
    searchinfo::SearchInfo,
    NAME, VERSION,
};

enum UciError {
    ParseGo(String),
    ParseOption(String),
    ParseFen(FenParseError),
    ParseMove(MoveParseError),
    UnexpectedCommandTermination(String),
    InvalidFormat(String),
    UnknownCommand(String),
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

impl Display for UciError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ParseGo(s) => write!(f, "ParseGo: {}", s),
            Self::ParseOption(s) => write!(f, "ParseOption: {}", s),
            Self::ParseFen(s) => write!(f, "ParseFen: {}", s),
            Self::ParseMove(s) => write!(f, "ParseMove: {}", s),
            Self::UnexpectedCommandTermination(s) => {
                write!(f, "UnexpectedCommandTermination: {}", s)
            }
            Self::InvalidFormat(s) => write!(f, "InvalidFormat: {}", s),
            Self::UnknownCommand(s) => write!(f, "UnknownCommand: {}", s),
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
        pos.make_move(m);
    }
    pos.zero_height();
    // eprintln!("{}", pos);
    Ok(())
}

fn parse_go(text: &str, info: &mut SearchInfo, pos: &mut Board) -> Result<(), UciError> {
    #![allow(clippy::too_many_lines)]
    let mut depth: Option<i32> = None;
    let mut moves_to_go: Option<u64> = None;
    let mut movetime: Option<u64> = None;
    let mut time: Option<u64> = None;
    let mut inc: Option<u64> = None;
    info.time_set = false;

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
            "wtime" if pos.turn() == WHITE => time = Some(part_parse("wtime", parts.next())?),
            "btime" if pos.turn() == BLACK => time = Some(part_parse("btime", parts.next())?),
            "winc" if pos.turn() == WHITE => inc = Some(part_parse("winc", parts.next())?),
            "binc" if pos.turn() == BLACK => inc = Some(part_parse("binc", parts.next())?),
            "infinite" => info.infinite = true,
            _ => (), //eprintln!("ignoring term in parse_go: {}", part),
        }
    }

    let search_time_window = match movetime {
        Some(movetime) => {
            info.time_set = true;
            time = Some(movetime);
            movetime
        }
        None => {
            if let Some(t) = time {
                info.time_set = true;
                let time = t / moves_to_go.unwrap_or(30) + inc.unwrap_or(0);
                let time = time.saturating_sub(30);
                time.min(t)
            } else {
                info.time_set = false;
                0
            }
        }
    };
    let max_time_window = movetime.map_or_else(
        || Duration::from_millis((search_time_window * 2).min(time.unwrap_or(0))), 
        Duration::from_millis
    );
    info.max_time_window = max_time_window;

    let is_computed_time_window_valid =
        !info.time_set || search_time_window <= time.unwrap() as u64;
    if !is_computed_time_window_valid {
        let time = time.unwrap();
        return Err(UciError::ParseGo(format!(
            "search window was {search_time_window}, but time was {time}"
        )));
    }

    info.set_time_window(search_time_window);

    if let Some(depth) = depth {
        info.depth = depth.into();
    } else {
        info.depth = MAX_DEPTH;
    }

    // println!(
    //     "time: {}, depth: {}, timeset: {}",
    //     info.stop_time.duration_since(info.start_time).as_millis(),
    //     info.depth.n_ply(),
    //     info.time_set
    // );

    Ok(())
}

fn part_parse<T>(target: &str, next_part: Option<&str>) -> Result<T, UciError>
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    next_part
        .ok_or_else(|| UciError::InvalidFormat(format!("nothing after \"{target}\"")))?
        .parse()
        .map_err(|e| UciError::InvalidFormat(format!("value for {target} is not a number: {e}")))
}

struct SetOptions {
    pub search_config: SearchParams,
    pub hash_mb: Option<usize>,
}

fn parse_setoption(
    text: &str,
    _info: &mut SearchInfo,
    pre_config: SetOptions,
) -> Result<SetOptions, UciError> {
    use UciError::UnexpectedCommandTermination;
    let mut parts = text.split_ascii_whitespace();
    parts.next().unwrap();
    parts
        .next()
        .map(|s| {
            assert!(
                s == "name",
                "unexpected character after \"setoption\", expected \"name\", got {}",
                s
            );
        })
        .ok_or_else(|| UnexpectedCommandTermination("no name after setoption".into()))?;
    let opt_name = parts.next().ok_or_else(|| {
        UnexpectedCommandTermination("no option name given after \"setoption name\"".into())
    })?;
    parts.next().map_or_else(|| panic!("no value after \"setoption name {opt_name}\""), |s| assert!(s == "value", "unexpected character after \"setoption name {opt_name}\", expected \"value\", got {}", s));
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
        "Hash" => out.hash_mb = Some(opt_value.parse()?),
        _ => eprintln!("ignoring option {opt_name}"),
    }
    Ok(out)
}

static KEEP_RUNNING: AtomicBool = AtomicBool::new(true);

fn stdin_reader() -> mpsc::Receiver<String> {
    let (sender, reciever) = mpsc::channel();
    std::thread::Builder::new()
        .name("stdin-reader".into())
        .spawn(|| stdin_reader_worker(sender))
        .expect("Couldn't start stdin reader worker thread");
    reciever
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

pub fn format_score(score: i32, turn: u8) -> String {
    assert!(turn == WHITE || turn == BLACK);
    if is_mate_score(score) {
        let plies_to_mate = MATE_SCORE - score.abs();
        let moves_to_mate = (plies_to_mate + 1) / 2;
        if score > 0 {
            format!("mate {moves_to_mate}")
        } else {
            format!("mate -{moves_to_mate}")
        }
    } else {
        format!("cp {score}")
    }
}

fn print_uci_response() {
    println!("id name {NAME} {VERSION}");
    println!("id author Cosmo");
    println!("option name Hash type spin default 4 min 1 max 1024");
    for (id, default) in SearchParams::default().ids_with_values() {
        println!("option name {id} type spin default {default} min -999999 max 999999");
    }
    println!("uciok");
}

pub fn main_loop(params: EvalParams) {
    print_uci_response();

    let mut pos = Board::new();

    pos.alloc_tables();

    let mut info = SearchInfo::default();

    pos.set_eval_params(params);

    let stdin = stdin_reader();

    info.set_stdin(&stdin);

    loop {
        std::io::stdout().flush().unwrap();
        let line = stdin.recv().expect("Couldn't read from stdin");
        let input = line.trim();

        let res = match input {
            "\n" => continue,
            "uci" => {
                print_uci_response();
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
            "ucinewgame" => {
                let res = parse_position("position startpos\n", &mut pos);
                pos.clear_tt();
                res
            }
            "eval" => {
                println!("{}", pos.evaluate(0));
                Ok(())
            }
            input if input.starts_with("setoption") => parse_setoption(
                input,
                &mut info,
                SetOptions { search_config: pos.sparams.clone(), hash_mb: None },
            )
            .map(|config| {
                pos.set_search_params(config.search_config);
                if let Some(hash_mb) = config.hash_mb {
                    pos.set_hash_size(hash_mb);
                }
            }),
            input if input.starts_with("position") => parse_position(input, &mut pos),
            input if input.starts_with("go") => {
                let res = parse_go(input, &mut info, &mut pos);
                if res.is_ok() {
                    pos.search_position(&mut info);
                }
                res
            }
            _ => Err(UciError::UnknownCommand(input.to_string())),
        };

        if let Err(e) = res {
            eprintln!("Error: {e}");
        }

        if info.quit {
            // quit can be set true in parse_go
            break;
        }
    }
    KEEP_RUNNING.store(false, atomic::Ordering::SeqCst);
}
