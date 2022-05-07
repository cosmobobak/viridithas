use std::io::Write;

use crate::{board::Board, searchinfo::SearchInfo, definitions::{WHITE, BLACK, MAX_DEPTH}};

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

const fn input_waiting() -> bool {
    false
}

pub fn read_input(info: &mut SearchInfo) {
    if input_waiting() {
        info.stopped = true;
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();
        if input == "quit" {
            info.quit = true;
        }
    }
}

pub fn main_loop() {
    println!("id name Viridithas");
    println!("id author Cosmo");
    println!("uciok");
    // std::io::stdout().flush().unwrap();
    let mut line = String::new();

    let mut pos = Board::new();
    let mut info = SearchInfo::default();

    loop {
        line.clear();
        std::io::stdout().flush().unwrap();
        std::io::stdin().read_line(&mut line).unwrap();
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
}
