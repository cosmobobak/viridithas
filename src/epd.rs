use std::{path::Path, io::BufRead};

use crate::{board::{evaluation::parameters::Parameters, Board}, searchinfo::SearchInfo, chessmove::Move};

const CONTROL_GREEN: &str = "\u{001b}[32m";
const CONTROL_RED: &str = "\u{001b}[31m";
const CONTROL_RESET: &str = "\u{001b}[0m";

struct Position {
    fen: String,
    best_moves: Vec<Move>,
    id: String,
}

pub fn gamut(epd_path: impl AsRef<Path>, params: Parameters, time: u64) {
    let mut board = Board::new();
    board.reset_tables();
    board.set_eval_params(params);
    let file = std::fs::File::open(epd_path).unwrap();
    let mut reader = std::io::BufReader::new(file);

    let mut line = String::new();
    let mut positions = Vec::new();

    while reader.read_line(&mut line).expect("Got invalid UTF-8") > 0 {
        let fen = line.split_whitespace().take(4).chain(Some("1 1")).collect::<Vec<_>>();
        let fen = fen.join(" ");
        board.set_from_fen(&fen).unwrap();
        let fen_out = board.fen();
        assert_eq!(fen, fen_out);
        let best_move_idx = line.find("bm").unwrap_or_else(|| panic!("no bestmove found in {line}"));
        let best_moves = &line[best_move_idx + 3..];
        let end_of_best_moves = best_moves.find(';').unwrap_or_else(|| panic!("no end of bestmove found in {line}"));
        let best_moves = &best_moves[..end_of_best_moves].split(' ').collect::<Vec<_>>();
        let best_moves = best_moves.iter().map(|best_move| {
            board.parse_san(best_move).unwrap_or_else(|err| panic!("invalid bestmove: {best_move}, {err}"))
        }).collect::<Vec<_>>();
        let id_idx = line.find("id").unwrap_or_else(|| panic!("no id found in {line}"));
        let id = line[id_idx + 4..].split(|c| c == '"').next().unwrap_or_else(|| panic!("no id found in {line}")).to_string();
        positions.push(Position { fen, best_moves, id });
        line.clear();
    }

    let n_positions = positions.len();
    println!("successfully parsed {n_positions} positions!");

    let mut failed_positions = vec![];
    let mut successes = 0;
    for position in positions {
        let Position { fen, best_moves, id } = &position;
        board.set_from_fen(fen).unwrap();
        board.reset_tables();
        let now = std::time::Instant::now();
        let mut info = SearchInfo { 
            print_to_stdout: false, 
            time_set: true,
            start_time: now,
            stop_time: now + std::time::Duration::from_millis(time), 
            ..SearchInfo::default()
        };
        let (_, bm) = board.search_position(&mut info);
        let passed = best_moves.contains(&bm);
        let color = if passed {
            CONTROL_GREEN
        } else {
            CONTROL_RED
        };
        let failinfo = if passed {
            "".into()
        } else {
            format!(", program chose {bm}")
        };
        println!("{id} {color}{}{CONTROL_RESET}{fen} {best_moves:?}{failinfo}", if passed { "PASS " } else { "FAIL " });
        if passed {
            successes += 1;
        } else {
            failed_positions.push(position);
        }
    }

    println!("{}/{} passed", successes, n_positions);

    if !failed_positions.is_empty() {
        println!("failed positions:");
        for position in failed_positions {
            println!("{:?} in {}", position.best_moves, position.fen);
        }
    }
}