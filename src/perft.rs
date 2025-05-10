#![allow(clippy::module_name_repetitions)]

use std::{
    fs::File,
    io::{BufRead, BufReader},
    sync::atomic::Ordering,
};

use anyhow::{bail, Context};

use crate::{
    chess::board::{movegen::MoveList, Board},
    chess::CHESS960,
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
    for &m in ml.iter_moves() {
        if !pos.make_move_simple(m) {
            continue;
        }
        count += perft(pos, depth - 1);
        pos.unmake_move_base();
    }

    count
}

#[cfg(test)]
pub fn nnue_perft(pos: &mut Board, t: &mut ThreadData, depth: usize) -> u64 {
    #[cfg(debug_assertions)]
    pos.check_validity().unwrap();
    // debug_assert!(pos.check_nnue_coherency(&t.nnue));

    if depth == 0 {
        return 1;
    }

    let mut ml = MoveList::new();
    pos.generate_moves(&mut ml);

    let mut count = 0;
    for &m in ml.iter_moves() {
        if !pos.make_move_nnue(m, t) {
            continue;
        }
        count += nnue_perft(pos, t, depth - 1);
        pos.unmake_move_nnue(t);
    }

    count
}

#[cfg(test)]
pub fn movepicker_perft(
    pos: &mut Board,
    t: &mut ThreadData,
    info: &SearchInfo,
    depth: usize,
) -> u64 {
    use crate::movepicker::MovePicker;

    #[cfg(debug_assertions)]
    pos.check_validity().unwrap();
    // debug_assert!(pos.check_nnue_coherency(&t.nnue));

    if depth == 0 {
        return 1;
    }

    let mut ml = MovePicker::new(None, None, 0);

    let mut count = 0;
    while let Some(m) = ml.next(pos, t, info) {
        if !pos.make_move(m, t) {
            continue;
        }
        count += movepicker_perft(pos, t, info, depth - 1);
        pos.unmake_move(t);
    }

    count
}

pub fn gamut() -> anyhow::Result<()> {
    #[cfg(debug_assertions)]
    const NODES_LIMIT: u64 = 60_000;
    #[cfg(not(debug_assertions))]
    const NODES_LIMIT: u64 = 60_000_000;
    // open perftsuite.epd
    println!("running perft on perftsuite.epd");
    let f =
        File::open("epds/perftsuite.epd").with_context(|| "Failed to open epds/perftsuite.epd")?;
    let mut pos = Board::new();
    for line in BufReader::new(f).lines() {
        let line = line?;
        let mut parts = line.split(';');
        let fen = parts
            .next()
            .with_context(|| "Failed to find fen in line.")?
            .trim();
        pos.set_from_fen(fen)?;
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
                bail!("FAIL: fen {fen}, depth {d}: expected {nodes}, got {perft_nodes}");
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
                bail!("FAIL: fen {fen}, depth {d}: expected {nodes}, got {perft_nodes}");
            }
        }
    }
    CHESS960.store(false, Ordering::SeqCst);
    Ok(())
}

mod tests {
    #![allow(unused_imports)]
    use std::sync::atomic::{AtomicBool, AtomicU64};

    use crate::{
        chess::{chessmove::Move, piece::PieceType, types::Square, CHESS960},
        nnue::network::NNUEParams,
        transpositiontable::TT,
        util::MEGABYTE,
    };

    #[test]
    fn perft_hard_position() {
        use super::*;
        const TEST_FEN: &str =
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

        std::env::set_var("RUST_BACKTRACE", "1");
        let mut pos = Board::new();
        pos.set_from_fen(TEST_FEN).unwrap();
        assert_eq!(perft(&mut pos, 1), 48, "got {}", {
            pos.legal_moves()
                .into_iter()
                .map(|m| m.display(CHESS960.load(Ordering::Relaxed)).to_string())
                .collect::<Vec<_>>()
                .join(", ")
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
            pos.legal_moves()
                .into_iter()
                .map(|m| m.display(CHESS960.load(Ordering::Relaxed)).to_string())
                .collect::<Vec<_>>()
                .join(", ")
        });
        assert_eq!(perft(&mut pos, 2), 400);
        assert_eq!(perft(&mut pos, 3), 8_902);
        // assert_eq!(perft(&mut pos, 4), 197_281);
    }

    #[test]
    fn perft_nnue_start_position() {
        use super::*;

        let mut pos = Board::default();
        let mut tt = TT::new();
        tt.resize(MEGABYTE * 16, 1);
        let nnue_params = NNUEParams::decompress_and_alloc().unwrap();
        let mut t = ThreadData::new(0, &pos, tt.view(), nnue_params);
        assert_eq!(nnue_perft(&mut pos, &mut t, 1), 20, "got {}", {
            pos.legal_moves()
                .into_iter()
                .map(|m| m.display(CHESS960.load(Ordering::Relaxed)).to_string())
                .collect::<Vec<_>>()
                .join(", ")
        });
        assert_eq!(nnue_perft(&mut pos, &mut t, 2), 400);
        assert_eq!(nnue_perft(&mut pos, &mut t, 3), 8_902);
        // assert_eq!(nnue_perft(&mut pos, &mut t, 4), 197_281);
    }

    #[test]
    fn perft_movepicker_start_position() {
        use super::*;

        let mut pos = Board::default();
        let mut tt = TT::new();
        tt.resize(MEGABYTE * 16, 1);
        let nnue_params = NNUEParams::decompress_and_alloc().unwrap();
        let mut t = ThreadData::new(0, &pos, tt.view(), nnue_params);
        let stopped = AtomicBool::new(false);
        let nodes = AtomicU64::new(0);
        let info = SearchInfo::new(&stopped, &nodes);
        assert_eq!(
            movepicker_perft(&mut pos, &mut t, &info, 1),
            20,
            "got {}",
            {
                pos.legal_moves()
                    .into_iter()
                    .map(|m| m.display(CHESS960.load(Ordering::Relaxed)).to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        );
        assert_eq!(movepicker_perft(&mut pos, &mut t, &info, 2), 400);
        assert_eq!(movepicker_perft(&mut pos, &mut t, &info, 3), 8_902);
        // assert_eq!(movepicker_perft(&mut pos, &mut t, &info, 4), 197_281);
    }

    #[test]
    fn perft_movepicker_hard_position() {
        use super::*;
        const TEST_FEN: &str =
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

        std::env::set_var("RUST_BACKTRACE", "1");
        let mut pos = Board::new();
        pos.set_from_fen(TEST_FEN).unwrap();
        let mut tt = TT::new();
        tt.resize(MEGABYTE * 16, 1);
        let nnue_params = NNUEParams::decompress_and_alloc().unwrap();
        let mut t = ThreadData::new(0, &pos, tt.view(), nnue_params);
        let stopped = AtomicBool::new(false);
        let nodes = AtomicU64::new(0);
        let info = SearchInfo::new(&stopped, &nodes);
        assert_eq!(
            movepicker_perft(&mut pos, &mut t, &info, 1),
            48,
            "got {}",
            {
                pos.legal_moves()
                    .into_iter()
                    .map(|m| m.display(CHESS960.load(Ordering::Relaxed)).to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        );
        assert_eq!(movepicker_perft(&mut pos, &mut t, &info, 2), 2_039);
        // assert_eq!(movepicker_perft(&mut pos, &mut t, &info, 3), 97_862);
        // assert_eq!(movepicker_perft(&mut pos, &mut t, &info, 4), 4_085_603);
    }

    #[test]
    fn perft_movepicker_forward_promo_evasion() {
        use super::*;
        const TEST_FEN: &str = "r7/P2r4/7R/8/5p2/5K2/3p2P1/R5k1 b - - 0 1";

        std::env::set_var("RUST_BACKTRACE", "1");
        let mut pos = Board::new();
        pos.set_from_fen(TEST_FEN).unwrap();
        let mut tt = TT::new();
        tt.resize(MEGABYTE * 16, 1);
        let nnue_params = NNUEParams::decompress_and_alloc().unwrap();
        let mut t = ThreadData::new(0, &pos, tt.view(), nnue_params);
        let stopped = AtomicBool::new(false);
        let nodes = AtomicU64::new(0);
        let info = SearchInfo::new(&stopped, &nodes);
        assert_eq!(movepicker_perft(&mut pos, &mut t, &info, 1), 4, "got {}", {
            pos.legal_moves()
                .into_iter()
                .map(|m| m.display(CHESS960.load(Ordering::Relaxed)).to_string())
                .collect::<Vec<_>>()
                .join(", ")
        });
        assert_eq!(movepicker_perft(&mut pos, &mut t, &info, 2), 62);
        assert_eq!(movepicker_perft(&mut pos, &mut t, &info, 3), 1474);
    }

    #[test]
    fn perft_movepicker_forward_promo_evasion_and_capture() {
        use super::*;
        const TEST_FEN: &str = "r7/P2r4/7R/8/5p2/5K2/3p2P1/2R3k1 b - - 0 1";

        std::env::set_var("RUST_BACKTRACE", "1");
        let mut pos = Board::new();
        pos.set_from_fen(TEST_FEN).unwrap();
        let mut tt = TT::new();
        tt.resize(MEGABYTE * 16, 1);
        let nnue_params = NNUEParams::decompress_and_alloc().unwrap();
        let mut t = ThreadData::new(0, &pos, tt.view(), nnue_params);
        let stopped = AtomicBool::new(false);
        let nodes = AtomicU64::new(0);
        let info = SearchInfo::new(&stopped, &nodes);
        assert_eq!(movepicker_perft(&mut pos, &mut t, &info, 1), 8, "got {}", {
            pos.legal_moves()
                .into_iter()
                .map(|m| m.display(CHESS960.load(Ordering::Relaxed)).to_string())
                .collect::<Vec<_>>()
                .join(", ")
        });
        assert_eq!(movepicker_perft(&mut pos, &mut t, &info, 2), 143);
        assert_eq!(movepicker_perft(&mut pos, &mut t, &info, 3), 3954);
    }

    #[test]
    fn perft_krk() {
        use super::*;

        let mut pos = Board::new();
        pos.set_from_fen("8/8/8/8/8/8/1k6/R2K4 b - - 1 1").unwrap();
        assert_eq!(perft(&mut pos, 1), 3, "got {}", {
            pos.legal_moves()
                .into_iter()
                .map(|m| m.display(CHESS960.load(Ordering::Relaxed)).to_string())
                .collect::<Vec<_>>()
                .join(", ")
        });
    }

    #[test]
    fn simple_move_undoability() {
        use super::*;

        let mut pos = Board::new();
        pos.set_startpos();
        let e4 = Move::new(Square::E2, Square::E4);
        let piece_layout_before = pos.state.bbs;
        println!("{piece_layout_before}");
        let hashkey_before = pos.state.keys.zobrist;
        pos.make_move_simple(e4);
        assert_ne!(pos.state.bbs, piece_layout_before);
        println!("{bb_after}", bb_after = pos.state.bbs);
        assert_ne!(pos.state.keys.zobrist, hashkey_before);
        pos.unmake_move_base();
        assert_eq!(pos.state.bbs, piece_layout_before);
        println!("{bb_returned}", bb_returned = pos.state.bbs);
        assert_eq!(pos.state.keys.zobrist, hashkey_before);
    }

    #[test]
    fn simple_capture_undoability() {
        use super::*;

        let mut pos = Board::new();
        pos.set_from_fen("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
            .unwrap();
        let exd5 = Move::new(Square::E4, Square::D5);
        let piece_layout_before = pos.state.bbs;
        println!("{piece_layout_before}");
        let hashkey_before = pos.state.keys.zobrist;
        pos.make_move_simple(exd5);
        assert_ne!(pos.state.bbs, piece_layout_before);
        println!("{bb_after}", bb_after = pos.state.bbs);
        assert_ne!(pos.state.keys.zobrist, hashkey_before);
        pos.unmake_move_base();
        assert_eq!(pos.state.bbs, piece_layout_before);
        println!("{bb_returned}", bb_returned = pos.state.bbs);
        assert_eq!(pos.state.keys.zobrist, hashkey_before);
    }
}
