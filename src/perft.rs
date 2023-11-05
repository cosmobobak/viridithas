#![allow(clippy::module_name_repetitions)]

use std::{
    fs::File,
    io::{BufRead, BufReader},
    sync::atomic::Ordering,
};

use crate::{
    board::{movegen::MoveList, Board},
    uci::CHESS960,
};
#[cfg(test)]
use crate::{searchinfo::SearchInfo, threadlocal::ThreadData};

pub fn perft(pos: &mut Board, depth: usize) -> u64 {
    #[cfg(debug_assertions)]
    pos.check_validity().unwrap();

    if depth == 0 {
        return 1;
    }

    let mut ml = MoveList::new();
    pos.generate_moves(&mut ml);

    let mut count = 0;
    for &m in ml.iter() {
        if !pos.make_move_base(m) {
            continue;
        }
        count += perft(pos, depth - 1);
        pos.unmake_move_base();
    }

    count
}

#[cfg(test)]
pub fn hce_perft(pos: &mut Board, info: &SearchInfo, depth: usize) -> u64 {
    #[cfg(debug_assertions)]
    pos.check_validity().unwrap();
    debug_assert!(pos.check_hce_coherency(info), "{pos}");

    if depth == 0 {
        return 1;
    }

    let mut ml = MoveList::new();
    pos.generate_moves(&mut ml);

    let mut count = 0;
    for &m in ml.iter() {
        if !pos.make_move_hce(m, info) {
            continue;
        }
        count += hce_perft(pos, info, depth - 1);
        pos.unmake_move_hce(info);
    }

    count
}

#[cfg(test)]
pub fn nnue_perft(pos: &mut Board, t: &mut ThreadData, depth: usize) -> u64 {
    #[cfg(debug_assertions)]
    pos.check_validity().unwrap();
    debug_assert!(pos.check_nnue_coherency(&t.nnue));

    if depth == 0 {
        return 1;
    }

    let mut ml = MoveList::new();
    pos.generate_moves(&mut ml);

    let mut count = 0;
    for &m in ml.iter() {
        if !pos.make_move_nnue(m, t) {
            continue;
        }
        count += nnue_perft(pos, t, depth - 1);
        pos.unmake_move_nnue(t);
    }

    count
}

pub fn gamut() {
    #[cfg(debug_assertions)]
    const NODES_LIMIT: u64 = 60_000;
    #[cfg(not(debug_assertions))]
    const NODES_LIMIT: u64 = 60_000_000;
    // open perftsuite.epd
    println!("running perft on perftsuite.epd");
    let f = File::open("epds/perftsuite.epd").unwrap();
    let mut pos = Board::new();
    for line in BufReader::new(f).lines() {
        let line = line.unwrap();
        let mut parts = line.split(';');
        let fen = parts.next().unwrap().trim();
        pos.set_from_fen(fen).unwrap();
        for depth_part in parts {
            let depth_part = depth_part.trim();
            let (d, nodes) = depth_part.split_once(' ').unwrap();
            let d = d.chars().nth(1).unwrap().to_digit(10).unwrap();
            let nodes = nodes.parse::<u64>().unwrap();
            if nodes > NODES_LIMIT {
                println!("Skipping...");
                break;
            }
            let perft_nodes = perft(&mut pos, d as usize);
            if perft_nodes == nodes {
                println!("PASS: fen {fen}, depth {d}");
            } else {
                println!("FAIL: fen {fen}, depth {d}: expected {nodes}, got {perft_nodes}");
                panic!("perft failed");
            }
        }
    }
    // open frcperftsuite.epd
    println!("running perft on frcperftsuite.epd");
    CHESS960.store(true, Ordering::SeqCst);
    let f = File::open("epds/frcperftsuite.epd").unwrap();
    let mut pos = Board::new();
    for line in BufReader::new(f).lines() {
        let line = line.unwrap();
        let mut parts = line.split(';');
        let fen = parts.next().unwrap().trim();
        pos.set_from_fen(fen).unwrap();
        for depth_part in parts {
            let depth_part = depth_part.trim();
            let (d, nodes) = depth_part.split_once(' ').unwrap();
            let d = d.chars().nth(1).unwrap().to_digit(10).unwrap();
            let nodes = nodes.parse::<u64>().unwrap();
            if nodes > NODES_LIMIT {
                println!("Skipping...");
                break;
            }
            let perft_nodes = perft(&mut pos, d as usize);
            if perft_nodes == nodes {
                println!("PASS: fen {fen}, depth {d}");
            } else {
                println!("FAIL: fen {fen}, depth {d}: expected {nodes}, got {perft_nodes}");
                panic!("perft failed");
            }
        }
    }
    CHESS960.store(false, Ordering::SeqCst);
}

mod tests {
    #![allow(unused_imports)]
    use std::sync::atomic::{AtomicBool, AtomicU64};

    use crate::{chessmove::Move, piece::PieceType, util::Square};

    #[test]
    fn perft_hard_position() {
        use super::*;
        const TEST_FEN: &str =
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

        std::env::set_var("RUST_BACKTRACE", "1");
        let mut pos = Board::new();
        pos.set_from_fen(TEST_FEN).unwrap();
        assert_eq!(perft(&mut pos, 1), 48, "got {}", {
            pos.legal_moves().into_iter().map(|m| m.to_string()).collect::<Vec<_>>().join(", ")
        });
        assert_eq!(perft(&mut pos, 2), 2_039);
        // assert_eq!(perft(&mut pos, 3), 97_862);
        // assert_eq!(perft(&mut pos, 4), 4_085_603);
    }

    #[test]
    fn perft_start_position() {
        use super::*;

        let mut pos = Board::new();
        std::env::set_var("RUST_BACKTRACE", "1");
        pos.set_startpos();
        assert_eq!(perft(&mut pos, 1), 20, "got {}", {
            pos.legal_moves().into_iter().map(|m| m.to_string()).collect::<Vec<_>>().join(", ")
        });
        assert_eq!(perft(&mut pos, 2), 400);
        assert_eq!(perft(&mut pos, 3), 8_902);
        // assert_eq!(perft(&mut pos, 4), 197_281);
    }

    #[test]
    fn perft_hce_start_position() {
        use super::*;

        let mut pos = Board::new();
        let stopped = AtomicBool::new(false);
        let nodes = AtomicU64::new(0);
        let info = SearchInfo::new(&stopped, &nodes);
        pos.set_startpos();
        pos.refresh_psqt(&info);
        assert_eq!(hce_perft(&mut pos, &info, 1), 20, "got {}", {
            pos.legal_moves().into_iter().map(|m| m.to_string()).collect::<Vec<_>>().join(", ")
        });
        assert_eq!(hce_perft(&mut pos, &info, 2), 400);
        assert_eq!(hce_perft(&mut pos, &info, 3), 8_902);
        // assert_eq!(hce_perft(&mut pos, &info, 4), 197_281);
    }

    #[test]
    fn perft_nnue_start_position() {
        use super::*;

        let mut pos = Board::default();
        let mut t = ThreadData::new(0, &pos);
        assert_eq!(nnue_perft(&mut pos, &mut t, 1), 20, "got {}", {
            pos.legal_moves().into_iter().map(|m| m.to_string()).collect::<Vec<_>>().join(", ")
        });
        assert_eq!(nnue_perft(&mut pos, &mut t, 2), 400);
        assert_eq!(nnue_perft(&mut pos, &mut t, 3), 8_902);
        // assert_eq!(nnue_perft(&mut pos, &mut t, 4), 197_281);
    }

    #[test]
    fn perft_krk() {
        use super::*;

        let mut pos = Board::new();
        pos.set_from_fen("8/8/8/8/8/8/1k6/R2K4 b - - 1 1").unwrap();
        assert_eq!(perft(&mut pos, 1), 3, "got {}", {
            pos.legal_moves().into_iter().map(|m| m.to_string()).collect::<Vec<_>>().join(", ")
        });
    }

    #[test]
    fn simple_move_undoability() {
        use super::*;

        let mut pos = Board::new();
        pos.set_startpos();
        let e4 = Move::new(Square::E2, Square::E4);
        let bitboard_before = pos.pieces;
        println!("{bitboard_before}");
        let hashkey_before = pos.hashkey();
        pos.make_move_base(e4);
        assert_ne!(pos.pieces, bitboard_before);
        println!("{bb_after}", bb_after = pos.pieces);
        assert_ne!(pos.hashkey(), hashkey_before);
        pos.unmake_move_base();
        assert_eq!(pos.pieces, bitboard_before);
        println!("{bb_returned}", bb_returned = pos.pieces);
        assert_eq!(pos.hashkey(), hashkey_before);
    }

    #[test]
    fn simple_capture_undoability() {
        use super::*;

        let mut pos = Board::new();
        pos.set_from_fen("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2").unwrap();
        let exd5 = Move::new(Square::E4, Square::D5);
        let bitboard_before = pos.pieces;
        println!("{bitboard_before}");
        let hashkey_before = pos.hashkey();
        pos.make_move_base(exd5);
        assert_ne!(pos.pieces, bitboard_before);
        println!("{bb_after}", bb_after = pos.pieces);
        assert_ne!(pos.hashkey(), hashkey_before);
        pos.unmake_move_base();
        assert_eq!(pos.pieces, bitboard_before);
        println!("{bb_returned}", bb_returned = pos.pieces);
        assert_eq!(pos.hashkey(), hashkey_before);
    }
}
