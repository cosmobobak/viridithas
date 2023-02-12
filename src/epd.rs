use std::{path::Path, time::Instant, sync::atomic::AtomicBool};

use crate::{
    board::{
        evaluation::{parameters::EvalParams, set_eval_params},
        Board,
    },
    chessmove::Move,
    definitions::MEGABYTE,
    searchinfo::{SearchInfo, SearchLimit},
    threadlocal::ThreadData,
    transpositiontable::TT,
};

const CONTROL_GREY: &str = "\u{001b}[38;5;243m";
const CONTROL_GREEN: &str = "\u{001b}[32m";
const CONTROL_RED: &str = "\u{001b}[31m";
const CONTROL_RESET: &str = "\u{001b}[0m";

struct EpdPosition {
    fen: String,
    best_moves: Vec<Move>,
    id: String,
}

pub fn gamut(epd_path: impl AsRef<Path>, params: EvalParams, time: u64, hash: usize, threads: usize) {
    let mut board = Board::new();
    unsafe {
        set_eval_params(params);
    }
    let raw_text = std::fs::read_to_string(epd_path).unwrap();
    let text = raw_text.trim();

    let mut positions = Vec::new();

    for line in text.lines() {
        positions.push(parse_epd(line, &mut board));
    }

    let n_positions = positions.len();
    println!("successfully parsed {n_positions} positions!");

    let successes = run_on_positions(positions, board, time, hash, threads);

    println!("{successes}/{n_positions} passed");
}

fn parse_epd(line: &str, board: &mut Board) -> EpdPosition {
    let fen = line.split_whitespace().take(4).chain(Some("1 1")).collect::<Vec<_>>().join(" ");
    board.set_from_fen(&fen).unwrap_or_else(|err| panic!("Invalid FEN: {fen}\n - {err}"));
    let fen_out = board.fen();
    assert_eq!(fen, fen_out);
    let best_move_idx = line.find("bm").unwrap_or_else(|| panic!("no bestmove found in {line}"));
    let best_moves = &line[best_move_idx + 3..];
    let end_of_best_moves =
        best_moves.find(';').unwrap_or_else(|| panic!("no end of bestmove found in {line}"));
    let best_moves = &best_moves[..end_of_best_moves].split(' ').collect::<Vec<_>>();
    let best_moves = best_moves
        .iter()
        .map(|best_move| {
            board
                .parse_san(best_move)
                .unwrap_or_else(|err| panic!("invalid bestmove: {best_move}, {err}"))
        })
        .collect::<Vec<_>>();
    let id_idx = line.find("id").unwrap_or_else(|| panic!("no id found in {line}"));
    let id = line[id_idx + 4..]
        .split(|c| c == '"')
        .next()
        .unwrap_or_else(|| panic!("no id found in {line}"))
        .to_string();
    EpdPosition { fen, best_moves, id }
}

fn run_on_positions(positions: Vec<EpdPosition>, mut board: Board, time: u64, hash: usize, threads: usize) -> i32 {
    let mut tt = TT::new();
    tt.resize(hash * MEGABYTE);
    let mut thread_data = (0..threads).map(ThreadData::new).collect::<Vec<_>>();
    let mut successes = 0;
    let maxfenlen = positions.iter().map(|pos| pos.fen.len()).max().unwrap();
    let maxidlen = positions.iter().map(|pos| pos.id.len()).max().unwrap();
    let n = positions.len();
    let start_time = std::time::Instant::now();
    for EpdPosition { fen, best_moves, id } in positions {
        board.set_from_fen(&fen).unwrap();
        tt.clear();
        for t in &mut thread_data {
            t.nnue.refresh_acc(&board);
            t.alloc_tables();
        }
        let stopped = AtomicBool::new(false);
        let mut info = SearchInfo {
            print_to_stdout: false,
            limit: SearchLimit::TimeOrCorrectMoves(time, best_moves.clone()),
            start_time: Instant::now(),
            ..SearchInfo::new(&stopped)
        };
        let (_, bm) = board.search_position::<true>(&mut info, &mut thread_data, tt.view());
        let passed = best_moves.contains(&bm);
        let colour = if passed { CONTROL_GREEN } else { CONTROL_RED };
        let failinfo = if passed {
            format!(" {CONTROL_GREY}{:.1}s{CONTROL_RESET}", info.start_time.elapsed().as_secs_f64())
        } else {
            format!(", program chose {CONTROL_RED}{bm}{CONTROL_RESET}")
        };
        let move_strings =
            best_moves
                .iter()
                .map(|&m| {
                    if m == bm {
                        format!("{m}")
                    } else {
                        format!("{CONTROL_GREY}{m}{CONTROL_RESET}")
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");
        println!(
            "{CONTROL_GREY}{id:midl$}{CONTROL_RESET} {fen:mfl$} {colour}{}{CONTROL_RESET} [{move_strings}]{failinfo}",
            if passed { "PASS" } else { "FAIL" },
            midl = maxidlen,
            mfl = maxfenlen,
        );
        if passed {
            successes += 1;
        }
    }
    let elapsed = start_time.elapsed();
    println!("{n} positions in {}.{:03}s", elapsed.as_secs(), elapsed.subsec_millis());
    successes
}
