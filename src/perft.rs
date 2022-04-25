use std::{
    fs::File,
    io::{BufRead, BufReader},
    time::Instant,
};

use crate::{board::Board, movegen::MoveList};

fn perft(pos: &mut Board, depth: usize) -> u64 {
    #[cfg(debug_assertions)]
    pos.check_validity();

    if depth == 0 {
        return 1;
    }

    let mut ml = MoveList::new();
    pos.generate_all_moves(&mut ml);

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

pub fn run_test(fen: &str, depth: usize) {
    let mut pos = Board::from_fen(fen);
    #[cfg(debug_assertions)]
    pos.check_validity();

    println!("{}", pos);
    println!("Starting perft test to depth {}", depth);
    let mut leafnodes = 0;

    let mut ml = MoveList::new();
    pos.generate_all_moves(&mut ml);
    let start_time = Instant::now();
    for (i, &m) in ml.iter().enumerate() {
        if !pos.make_move(m) {
            continue;
        }
        let nodes = perft(&mut pos, depth - 1);
        leafnodes += nodes;
        println!("move {} ({}): {} nodes", i, m, nodes);
        pos.unmake_move();
    }
    let elapsed = start_time.elapsed();

    println!("Test complete, {} nodes visited.", leafnodes);
    println!("Time taken: {}ms", elapsed.as_millis());
    #[allow(clippy::cast_precision_loss)]
    let nodesf64 = leafnodes as f64;
    println!("kNPS: {:.0}", nodesf64 / elapsed.as_secs_f64() / 1000.0);
}

pub fn gamut() {
    // open perftsuite.epd
    let f = File::open("perftsuite.epd").unwrap();
    let mut pos = Board::new();
    for line in BufReader::new(f).lines() {
        let line = line.unwrap();
        let mut parts = line.split(';');
        let fen = parts.next().unwrap().trim();
        pos.set_from_fen(fen);
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
            if perft_nodes != nodes {
                println!("FAIL: fen {fen}, depth {d}: expected {nodes}, got {perft_nodes}");
                panic!("perft failed");
            } else {
                println!("PASS: fen {fen}, depth {d}");
            }
        }
    }
}

fn perft_to_depth(fen: &str, depth: usize) -> u64 {
    let mut pos = Board::from_fen(fen);
    perft(&mut pos, depth)
}

mod tests {
    #[test]
    fn test_perft() {
        use super::*;
        use crate::definitions::STARTING_FEN;
        const TEST_FEN: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        let mut pos = Board::new();
        pos.set_from_fen(STARTING_FEN);
        assert_eq!(perft(&mut pos, 1), 20);
        assert_eq!(perft(&mut pos, 2), 400);
        assert_eq!(perft(&mut pos, 3), 8_902);
        assert_eq!(perft(&mut pos, 4), 197_281);
        pos.set_from_fen(TEST_FEN);
        assert_eq!(perft(&mut pos, 1), 48);
        assert_eq!(perft(&mut pos, 2), 2_039);
        assert_eq!(perft(&mut pos, 3), 97_862);
        // assert_eq!(perft(&mut pos, 4), 4_085_603);
    }
}
