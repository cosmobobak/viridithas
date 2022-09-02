use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use crate::board::{movegen::MoveList, Board};

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
        if !pos.make_move(m) {
            continue;
        }
        count += perft(pos, depth - 1);
        pos.unmake_move();
    }

    count
}

pub fn gamut() {
    // open perftsuite.epd
    let f = File::open("perftsuite.epd").unwrap();
    let mut pos = Board::new();
    pos.reset_tables();
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
            if nodes > 60_000_000 {
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
}

mod tests {

    #[test]
    fn perft_hard_position() {
        use super::*;
        use crate::board::movegen::MoveVecWrapper;
        const TEST_FEN: &str =
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        crate::magic::initialise();
        std::env::set_var("RUST_BACKTRACE", "1");
        let mut pos = Board::new();
        pos.reset_tables();
        pos.set_from_fen(TEST_FEN).unwrap();
        assert_eq!(perft(&mut pos, 1), 48, "got {}", {
            let mut ml = MoveList::new();
            pos.generate_moves(&mut ml);
            let mut legal = vec![];
            for &m in ml.iter() {
                if pos.make_move(m) {
                    legal.push(m);
                    pos.unmake_move();
                }
            }
            MoveVecWrapper(legal)
        });
        assert_eq!(perft(&mut pos, 2), 2_039);
        // assert_eq!(perft(&mut pos, 3), 97_862);
        // assert_eq!(perft(&mut pos, 4), 4_085_603);
    }

    #[test]
    fn perft_start_position() {
        use super::*;
        use crate::board::movegen::MoveVecWrapper;
        crate::magic::initialise();
        let mut pos = Board::new();
        pos.reset_tables();
        std::env::set_var("RUST_BACKTRACE", "1");
        pos.set_startpos();
        assert_eq!(perft(&mut pos, 1), 20, "got {}", {
            let mut ml = MoveList::new();
            pos.generate_moves(&mut ml);
            let mut legal = vec![];
            for &m in ml.iter() {
                if pos.make_move(m) {
                    legal.push(m);
                    pos.unmake_move();
                }
            }
            MoveVecWrapper(legal)
        });
        assert_eq!(perft(&mut pos, 2), 400);
        assert_eq!(perft(&mut pos, 3), 8_902);
        // assert_eq!(perft(&mut pos, 4), 197_281);
    }

    #[test]
    fn perft_krk() {
        use super::*;
        use crate::board::movegen::MoveVecWrapper;
        crate::magic::initialise();
        let mut pos = Board::new();
        pos.reset_tables();
        pos.set_from_fen("8/8/8/8/8/8/1k6/R2K4 b - - 1 1").unwrap();
        assert_eq!(perft(&mut pos, 1), 3, "got {}", {
            let mut ml = MoveList::new();
            pos.generate_moves(&mut ml);
            let mut legal = vec![];
            for &m in ml.iter() {
                if pos.make_move(m) {
                    legal.push(m);
                    pos.unmake_move();
                }
            }
            MoveVecWrapper(legal)
        });
    }
}
