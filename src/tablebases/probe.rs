#![allow(
    dead_code,
    unused_imports,
    unused_variables,
    clippy::missing_const_for_fn
)]

use crate::{
    chess::{
        board::{Board, movegen::MoveList},
        chessmove::Move,
        piece::{Colour, PieceType},
        types::{CastlingRights, Square},
    },
    evaluation::TB_WIN_SCORE,
    tablebases::bindings::{
        PYRRHIC_FLAG_BPROMO as TB_PROMOTES_BISHOP, PYRRHIC_FLAG_NPROMO as TB_PROMOTES_KNIGHT,
        PYRRHIC_FLAG_QPROMO as TB_PROMOTES_QUEEN, PYRRHIC_FLAG_RPROMO as TB_PROMOTES_ROOK,
        TB_BLESSED_LOSS, TB_CURSED_WIN, TB_DRAW, TB_LARGEST, TB_LOSS, TB_RESULT_DTZ_MASK,
        TB_RESULT_DTZ_SHIFT, TB_RESULT_FAILED, TB_RESULT_FROM_MASK, TB_RESULT_FROM_SHIFT,
        TB_RESULT_PROMOTES_MASK, TB_RESULT_PROMOTES_SHIFT, TB_RESULT_TO_MASK, TB_RESULT_TO_SHIFT,
        TB_RESULT_WDL_MASK, TB_RESULT_WDL_SHIFT, TB_WIN, tb_init, tb_probe_root, tb_probe_wdl,
    },
    uci,
};
use std::{ffi::CString, sync::atomic::AtomicBool};

pub static SYZYGY_ENABLED: AtomicBool = AtomicBool::new(false);

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub enum WDL {
    Win,
    Loss,
    Draw,
}
pub struct WdlDtzResult {
    wdl: WDL,
    dtz: u32,
    best_move: Move,
}

/// Loads Syzygy tablebases stored in `syzygy_path` location.
pub fn init(syzygy_path: &str) {
    // SAFETY: Not much.
    #[cfg(feature = "syzygy")]
    unsafe {
        let path = CString::new(syzygy_path).unwrap();
        let res = tb_init(path.as_ptr());
        assert!(res, "Failed to load Syzygy tablebases from {syzygy_path}");
        SYZYGY_ENABLED.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Gets maximal pieces count supported by loaded Syzygy tablebases. Returns 0 if the feature is disabled.
pub fn get_max_pieces_count() -> u8 {
    #![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    #[cfg(feature = "syzygy")]
    {
        let user_limit = uci::SYZYGY_PROBE_LIMIT.load(std::sync::atomic::Ordering::Relaxed);
        // SAFETY: Not much.
        let hard_limit = unsafe { TB_LARGEST as u8 };
        std::cmp::min(user_limit, hard_limit)
    }
    #[cfg(not(feature = "syzygy"))]
    0
}

/// Gets WDL (Win-Draw-Loss) for the position specified in `board`. Returns [None] if data couldn't be obtained or the feature is disabled.
pub fn get_wdl(board: &Board) -> Option<WDL> {
    const WHITE: bool = true;
    const BLACK: bool = false;

    // guards for invalid positions
    if board.castling_rights() != CastlingRights::default() || board.fifty_move_counter() != 0 {
        return None;
    }

    // SAFETY: Not much.
    #[cfg(feature = "syzygy")]
    unsafe {
        let b = &board.state.bbs;
        let ep = board.ep_sq().map_or(0, |sq| sq as u32);
        let wdl = tb_probe_wdl(
            b.colours[Colour::White].inner(),
            b.colours[Colour::Black].inner(),
            b.pieces[PieceType::King].inner(),
            b.pieces[PieceType::Queen].inner(),
            b.pieces[PieceType::Rook].inner(),
            b.pieces[PieceType::Bishop].inner(),
            b.pieces[PieceType::Knight].inner(),
            b.pieces[PieceType::Pawn].inner(),
            ep,
            board.turn() == Colour::White,
        );

        match wdl {
            TB_WIN => Some(WDL::Win),
            TB_LOSS => Some(WDL::Loss),
            TB_DRAW | TB_CURSED_WIN | TB_BLESSED_LOSS => Some(WDL::Draw),
            _ => None,
        }
    }
    #[cfg(not(feature = "syzygy"))]
    None
}

/// Gets WDL (Win-Draw-Loss), DTZ (Distance To Zeroing) and the best move for the position specified in `board`.
/// Returns [None] if data couldn't be obtained or the feature is disabled.
pub fn get_root_wdl_dtz(board: &Board) -> Option<WdlDtzResult> {
    const WHITE: bool = true;
    const BLACK: bool = false;

    // SAFETY: Not much.
    #[cfg(feature = "syzygy")]
    unsafe {
        let b = &board.state.bbs;
        let ep = board.ep_sq().map_or(0, |sq| sq as u32);
        let result = tb_probe_root(
            b.colours[Colour::White].inner(),
            b.colours[Colour::Black].inner(),
            b.pieces[PieceType::King].inner(),
            b.pieces[PieceType::Queen].inner(),
            b.pieces[PieceType::Rook].inner(),
            b.pieces[PieceType::Bishop].inner(),
            b.pieces[PieceType::Knight].inner(),
            b.pieces[PieceType::Pawn].inner(),
            u32::from(board.fifty_move_counter()),
            ep,
            board.turn() == Colour::White,
            std::ptr::null_mut(),
        );

        let wdl = (result & TB_RESULT_WDL_MASK) >> TB_RESULT_WDL_SHIFT;
        let wdl = match wdl {
            TB_WIN => WDL::Win,
            TB_LOSS => WDL::Loss,
            _ => WDL::Draw,
        };
        let dtz = (result & TB_RESULT_DTZ_MASK) >> TB_RESULT_DTZ_SHIFT;

        if result == TB_RESULT_FAILED {
            return None;
        }

        let mut moves = MoveList::new();
        board.generate_moves(&mut moves);

        let from =
            Square::new(((result & TB_RESULT_FROM_MASK) >> TB_RESULT_FROM_SHIFT) as u8).unwrap();
        let to = Square::new(((result & TB_RESULT_TO_MASK) >> TB_RESULT_TO_SHIFT) as u8).unwrap();
        let promotion = (result & TB_RESULT_PROMOTES_MASK) >> TB_RESULT_PROMOTES_SHIFT;

        let promo_piece_type = match promotion {
            TB_PROMOTES_QUEEN => Some(PieceType::Queen),
            TB_PROMOTES_ROOK => Some(PieceType::Rook),
            TB_PROMOTES_BISHOP => Some(PieceType::Bishop),
            TB_PROMOTES_KNIGHT => Some(PieceType::Knight),
            _ => None,
        };

        for &m in moves.iter_moves() {
            if m.from() == from
                && m.to() == to
                && (promotion == 0 || m.promotion_type() == promo_piece_type)
            {
                return Some(WdlDtzResult {
                    wdl,
                    dtz,
                    best_move: m,
                });
            }
        }

        None
    }
    #[cfg(not(feature = "syzygy"))]
    None
}

/// Checks if there's a tablebase move and returns it as [Some], otherwise [None].
pub fn get_tablebase_move(board: &Board) -> Option<(Move, i32)> {
    if board.state.bbs.occupied().count() > u32::from(get_max_pieces_count()) {
        return None;
    }

    let result = get_root_wdl_dtz(board)?;

    let score = match result.wdl {
        WDL::Win => TB_WIN_SCORE,
        WDL::Draw => 0,
        WDL::Loss => -TB_WIN_SCORE,
    };

    Some((result.best_move, score))
}

/// Gets the WDL of the position from the perspective of White.
/// Returns [None] if data couldn't be obtained or the feature is disabled.
pub fn get_wdl_white(board: &Board) -> Option<WDL> {
    let probe_result = get_root_wdl_dtz(board)?;

    let stm = board.turn() == Colour::White;

    match probe_result.wdl {
        WDL::Win => Some(if stm { WDL::Win } else { WDL::Loss }),
        WDL::Draw => Some(WDL::Draw),
        WDL::Loss => Some(if stm { WDL::Loss } else { WDL::Win }),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        LazyLock,
        atomic::{AtomicBool, AtomicU64},
    };

    use crate::{
        evaluation::MINIMUM_TB_WIN_SCORE,
        nnue::network::NNUEParams,
        search::search_position,
        threadlocal::ThreadData,
        threadpool,
        timemgmt::{SearchLimit, TimeManager},
        transpositiontable::TT,
        util::MEGABYTE,
    };

    use super::*;

    static INIT_SYZYGY: std::sync::LazyLock<()> = std::sync::LazyLock::new(|| {
        init(
            std::env::var("SYZYGY_PATH")
                .unwrap_or_else(|_| "/syzygy".to_string())
                .as_str(),
        );
    });

    #[test]
    fn test_syzygy_wdl() {
        LazyLock::force(&INIT_SYZYGY);

        let win_3man = "4Q3/8/8/8/8/8/7k/K7 w - - 0 1";
        let draw_3man = "4N3/8/8/8/8/4K3/8/4k3 w - - 0 1";
        let loss_3man = "8/8/7K/8/8/8/8/k6q w - - 0 1";

        let win_7man = "3k4/4p3/8/1r6/6P1/5P2/4RK2/8 w - - 0 1";

        if get_max_pieces_count() >= 3 {
            let board = Board::from_fen(win_3man).unwrap();
            assert_eq!(get_wdl(&board), Some(WDL::Win));

            let board = Board::from_fen(draw_3man).unwrap();
            assert_eq!(get_wdl(&board), Some(WDL::Draw));

            let board = Board::from_fen(loss_3man).unwrap();
            assert_eq!(get_wdl(&board), Some(WDL::Loss));
        }

        if get_max_pieces_count() >= 7 {
            let board = Board::from_fen(win_7man).unwrap();
            assert_eq!(get_wdl(&board), Some(WDL::Win));
        }
    }

    #[test]
    fn solve_7man() {
        LazyLock::force(&INIT_SYZYGY);

        if get_max_pieces_count() < 7 {
            return;
        }

        let position = Board::from_fen("3k4/4p3/1r6/Qq6/6P1/5P2/4RK2/8 w - - 0 1").unwrap();
        let stopped = AtomicBool::new(false);
        let nodes = AtomicU64::new(0);
        let tbhits = AtomicU64::new(0);
        let pool = threadpool::make_worker_threads(1);
        let mut tt = TT::new();
        tt.resize(MEGABYTE, &pool);
        let nnue_params = NNUEParams::decompress_and_alloc().unwrap();
        let mut t = Box::new(ThreadData::new(
            0,
            position,
            tt.view(),
            nnue_params,
            &stopped,
            &nodes,
            &tbhits,
        ));
        t.info.clock = TimeManager::default_with_limit(SearchLimit::Depth(16));
        let (value, mov) = search_position(&pool, std::array::from_mut(&mut t));

        assert!(matches!(
            t.board
                .san(mov.unwrap())
                .as_ref()
                .map(ToString::to_string)
                .as_deref(),
            Some("Qxb5")
        ));
        assert!(value >= MINIMUM_TB_WIN_SCORE);
    }

    #[test]
    fn solve_5man() {
        LazyLock::force(&INIT_SYZYGY);

        if get_max_pieces_count() < 5 {
            return;
        }

        let position = Board::from_fen("8/Q7/8/5N2/1p6/q7/8/2NK3k w - - 0 1").unwrap();
        let stopped = AtomicBool::new(false);
        let nodes = AtomicU64::new(0);
        let tbhits = AtomicU64::new(0);
        let pool = threadpool::make_worker_threads(1);
        let mut tt = TT::new();
        tt.resize(MEGABYTE, &pool);
        let nnue_params = NNUEParams::decompress_and_alloc().unwrap();
        let mut t = Box::new(ThreadData::new(
            0,
            position,
            tt.view(),
            nnue_params,
            &stopped,
            &nodes,
            &tbhits,
        ));
        t.info.clock = TimeManager::default_with_limit(SearchLimit::Depth(10));
        let (value, mov) = search_position(&pool, std::array::from_mut(&mut t));

        assert!(matches!(
            t.board
                .san(mov.unwrap())
                .as_ref()
                .map(ToString::to_string)
                .as_deref(),
            Some("Qxa3")
        ));
        assert!(value >= MINIMUM_TB_WIN_SCORE);
    }
}
