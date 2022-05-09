use std::{io::Write, sync::{mpsc, atomic}};

use crate::{board::Board, searchinfo::SearchInfo, definitions::{WHITE, BLACK, MAX_DEPTH}, evaluation::{IS_MATE_SCORE, MATE_SCORE}};

// position fen
// position startpos
// ... moves e2e4 e7e5 b7b8q
fn parse_position(text: &str, pos: &mut Board) {
    let mut parts = text.split_ascii_whitespace();
    let command = parts.next().expect("No command in parse_position");
    assert_eq!(command, "position");
    let determiner = parts.next().expect("No determiner after \"position\"");
    if determiner == "startpos" {
        pos.set_startpos();
        let moves = parts.next(); // skip "moves"
        assert!(matches!(moves, Some("moves") | None));
    } else {
        assert_eq!(determiner, "fen", "Unknown term after \"position\": {}", determiner); 
        let mut fen = String::new();
        for part in &mut parts {
            if part == "moves" {
                break;
            }
            fen.push_str(part);
            fen.push(' ');
        }
        pos.set_from_fen(&fen).unwrap();
    }
    for san in parts {
        let m = pos.parse_san(san);
        let m = m.expect("Invalid move passed in uci");
        pos.make_move(m);
    }
    pos.zero_ply();
    eprintln!("{}", pos);
}

fn parse_go(text: &str, info: &mut SearchInfo, pos: &mut Board) {
    let mut depth: Option<usize> = None;
    let mut moves_to_go: Option<usize> = None;
    let mut movetime: Option<usize> = None;
    let mut time: Option<usize> = None;
    let mut inc: Option<usize> = None;
    info.time_set = false;

    let mut parts = text.split_ascii_whitespace();
    let command = parts.next().expect("No command in parse_go");
    assert_eq!(command, "go");

    while let Some(part) = parts.next() {
        match part {
            "depth" => depth = Some(parts.next().expect("nothing after \"depth\"").parse().expect("depth not a number")),
            "movestogo" => moves_to_go = Some(parts.next().expect("nothing after \"movestogo\"").parse().expect("movestogo not a number")),
            "movetime" => movetime = Some(parts.next().expect("nothing after \"movetime\"").parse().expect("movetime not a number")),
            "wtime" if pos.turn() == WHITE => time = Some(parts.next().expect("nothing after \"wtime\"").parse().expect("wtime not a number")),
            "btime" if pos.turn() == BLACK => time = Some(parts.next().expect("nothing after \"btime\"").parse().expect("btime not a number")),
            "winc" if pos.turn() == WHITE => inc = Some(parts.next().expect("nothing after \"winc\"").parse().expect("winc not a number")),
            "binc" if pos.turn() == BLACK => inc = Some(parts.next().expect("nothing after \"binc\"").parse().expect("binc not a number")),
            "infinite" => info.infinite = true,
            _ => eprintln!("ignoring term in parse_go: {}", part),
        }
    }

    if movetime.is_some() {
        time = movetime;
        moves_to_go = Some(1);
    }

    info.start_time = std::time::Instant::now();

    if let Some(time) = time {
        info.time_set = true;
        let time = time as u64 / moves_to_go.unwrap_or(30) as u64 + inc.unwrap_or(0) as u64;
        let time = time.checked_sub(50).unwrap_or(1);
        info.stop_time = info.start_time + std::time::Duration::from_millis(time);
    }

    if let Some(depth) = depth {
        info.depth = depth;
    } else {
        info.depth = MAX_DEPTH;
    }

    println!(
        "time: {}, depth: {}, timeset: {}", 
        info.stop_time.duration_since(info.start_time).as_millis(), info.depth, info.time_set
    );
}

static KEEP_RUNNING: atomic::AtomicBool = atomic::AtomicBool::new(true);

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

pub fn format_score(score: i32) -> String {
    if score.abs() > IS_MATE_SCORE {
        let plies_to_mate = MATE_SCORE - score.abs();
        let moves_to_mate = (plies_to_mate + 1) / 2;
        format!("mate {}", moves_to_mate)
    } else {
        format!("cp {}", score / 10)
    }
}

pub fn main_loop() {
    println!("id name Viridithas");
    println!("id author Cosmo");
    println!("uciok");

    let mut pos = Board::new();
    let mut info = SearchInfo::default();

    let stdin = stdin_reader();

    info.set_stdin(&stdin);

    loop {
        std::io::stdout().flush().unwrap();
        let line = stdin.recv().expect("Couldn't read from stdin");
        let input = line.trim();

        match input {
            "\n" => continue,
            "uci" => {
                println!("id name Viridithas");
                println!("id author Cosmo");
                println!("uciok");
            }
            "isready" => println!("readyok"),
            "quit" => {
                info.quit = true;
                break;
            }
            "ucinewgame" => parse_position("position startpos\n", &mut pos),
            input if input.starts_with("position") => parse_position(input, &mut pos),
            input if input.starts_with("go") => { 
                parse_go(input, &mut info, &mut pos);
                pos.search_position(&mut info);
            },
            _ => println!("Unknown command: {}", input),
        }

        if info.quit { // quit can be set true in parse_go
            break;
        }
    }
    KEEP_RUNNING.store(false, atomic::Ordering::SeqCst);
}
