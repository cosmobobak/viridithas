use std::{
    fmt::Display,
    io::Write,
    num::{ParseFloatError, ParseIntError},
    sync::{
        atomic::{self, AtomicBool},
        mpsc,
    },
};

use crate::{
    board::{
        evaluation::{is_mate_score, parameters::Parameters, MATE_SCORE},
        Board,
    },
    definitions::{BLACK, MAX_DEPTH, WHITE},
    errors::{FenParseError, MoveParseError},
    search::{LMRGRADIENT, LMRMAXDEPTH, LMRMIDPOINT, LOGCONSTANT, LOGSCALEFACTOR},
    searchinfo::SearchInfo,
    NAME,
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
            UciError::ParseGo(s) => write!(f, "ParseGo: {}", s),
            UciError::ParseOption(s) => write!(f, "ParseOption: {}", s),
            UciError::ParseFen(s) => write!(f, "ParseFen: {}", s),
            UciError::ParseMove(s) => write!(f, "ParseMove: {}", s),
            UciError::UnexpectedCommandTermination(s) => {
                write!(f, "UnexpectedCommandTermination: {}", s)
            }
            UciError::InvalidFormat(s) => write!(f, "InvalidFormat: {}", s),
            UciError::UnknownCommand(s) => write!(f, "UnknownCommand: {}", s),
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
        let m = pos.parse_san(san)?;
        pos.make_move(m);
    }
    pos.zero_ply();
    eprintln!("{}", pos);
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
            "depth" => {
                depth = Some(
                    parts
                        .next()
                        .ok_or_else(|| UciError::InvalidFormat("nothing after \"depth\"".into()))?
                        .parse()
                        .map_err(|e| {
                            UciError::InvalidFormat(format!("value for depth is not a number: {e}"))
                        })?,
                );
            }
            "movestogo" => {
                moves_to_go = Some(
                    parts
                        .next()
                        .ok_or_else(|| {
                            UciError::InvalidFormat("nothing after \"movestogo\"".into())
                        })?
                        .parse()
                        .map_err(|e| {
                            UciError::InvalidFormat(format!(
                                "value for movestogo is not a number: {e}"
                            ))
                        })?,
                );
            }
            "movetime" => {
                movetime = Some(
                    parts
                        .next()
                        .ok_or_else(|| {
                            UciError::InvalidFormat("nothing after \"movetime\"".into())
                        })?
                        .parse()
                        .map_err(|e| {
                            UciError::InvalidFormat(format!(
                                "value for movetime is not a number: {e}"
                            ))
                        })?,
                );
            }
            "wtime" if pos.turn() == WHITE => {
                time = Some(
                    parts
                        .next()
                        .ok_or_else(|| UciError::InvalidFormat("nothing after \"wtime\"".into()))?
                        .parse()
                        .map_err(|e| {
                            UciError::InvalidFormat(format!("value for wtime is not a number: {e}"))
                        })?,
                );
            }
            "btime" if pos.turn() == BLACK => {
                time = Some(
                    parts
                        .next()
                        .ok_or_else(|| UciError::InvalidFormat("nothing after \"btime\"".into()))?
                        .parse()
                        .map_err(|e| {
                            UciError::InvalidFormat(format!("value for btime is not a number: {e}"))
                        })?,
                );
            }
            "winc" if pos.turn() == WHITE => {
                inc = Some(
                    parts
                        .next()
                        .ok_or_else(|| UciError::InvalidFormat("nothing after \"winc\"".into()))?
                        .parse()
                        .map_err(|e| {
                            UciError::InvalidFormat(format!("value for winc is not a number: {e}"))
                        })?,
                );
            }
            "binc" if pos.turn() == BLACK => {
                inc = Some(
                    parts
                        .next()
                        .ok_or_else(|| UciError::InvalidFormat("nothing after \"binc\"".into()))?
                        .parse()
                        .map_err(|e| {
                            UciError::InvalidFormat(format!("value for binc is not a number: {e}"))
                        })?,
                );
            }
            "infinite" => info.infinite = true,
            _ => eprintln!("ignoring term in parse_go: {}", part),
        }
    }

    let search_time_window = match movetime {
        Some(movetime) => {
            info.time_set = true;
            time = Some(movetime);
            movetime
        }
        None => match time {
            Some(t) => {
                info.time_set = true;
                let time = t / moves_to_go.unwrap_or(30) + inc.unwrap_or(0);
                let time = time.saturating_sub(30);
                time.min(t)
            }
            None => {
                info.time_set = false;
                0
            }
        },
    };

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

    println!(
        "time: {}, depth: {}, timeset: {}",
        info.stop_time.duration_since(info.start_time).as_millis(),
        info.depth.n_ply(),
        info.time_set
    );

    Ok(())
}

fn parse_setoption(text: &str, _info: &mut SearchInfo) -> Result<(), UciError> {
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
    match opt_name {
        "LMRGRADIENT" => unsafe {
            LMRGRADIENT = opt_value.parse()?;
        },
        "LMRMIDPOINT" => unsafe {
            LMRMIDPOINT = opt_value.parse()?;
        },
        "LMRMAXDEPTH" => unsafe {
            LMRMAXDEPTH = opt_value.parse()?;
        },
        "LOGSCALEFACTOR" => unsafe {
            LOGSCALEFACTOR = opt_value.parse()?;
        },
        "LOGCONSTANT" => unsafe {
            LOGCONSTANT = opt_value.parse()?;
        },
        _ => eprintln!("ignoring option {}", opt_name),
    }
    Ok(())
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
            format!("mate {}", moves_to_mate)
        } else {
            format!("mate -{}", moves_to_mate)
        }
    } else {
        format!("cp {}", score)
    }
}

pub fn main_loop(evaluation_parameters: Parameters) {
    println!("id name {NAME}");
    println!("id author Cosmo");
    println!("uciok");

    let mut pos = Board::new();
    let mut info = SearchInfo::default();

    pos.set_eval_params(evaluation_parameters);

    let stdin = stdin_reader();

    info.set_stdin(&stdin);

    loop {
        std::io::stdout().flush().unwrap();
        let line = stdin.recv().expect("Couldn't read from stdin");
        let input = line.trim();

        let res = match input {
            "\n" => continue,
            "uci" => {
                println!("id name {NAME}");
                println!("id author Cosmo");
                println!("uciok");
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
            input if input.starts_with("setoption") => parse_setoption(input, &mut info),
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
            println!("Error: {}", e);
        }

        if info.quit {
            // quit can be set true in parse_go
            break;
        }
    }
    KEEP_RUNNING.store(false, atomic::Ordering::SeqCst);
}
