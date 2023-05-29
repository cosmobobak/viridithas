#![allow(dead_code, unused_imports, unused_variables, clippy::missing_const_for_fn)]

use crate::{
    board::{evaluation::TB_WIN_SCORE, movegen::MoveList, Board},
    chessmove::Move,
    definitions::{CastlingRights, Square},
    piece::{Colour, PieceType},
    tablebases::bindings::{
        tb_init, tb_probe_root, tb_probe_wdl, TB_BLESSED_LOSS, TB_CURSED_WIN, TB_DRAW, TB_LARGEST,
        TB_LOSS, TB_PROMOTES_BISHOP, TB_PROMOTES_KNIGHT, TB_PROMOTES_QUEEN, TB_PROMOTES_ROOK,
        TB_RESULT_DTZ_MASK, TB_RESULT_DTZ_SHIFT, TB_RESULT_FAILED, TB_RESULT_FROM_MASK,
        TB_RESULT_FROM_SHIFT, TB_RESULT_PROMOTES_MASK, TB_RESULT_PROMOTES_SHIFT, TB_RESULT_TO_MASK,
        TB_RESULT_TO_SHIFT, TB_RESULT_WDL_MASK, TB_RESULT_WDL_SHIFT, TB_WIN,
    },
    uci,
};
use std::ffi::CString;
use std::ptr;

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
    #[cfg(feature = "syzygy")]
    unsafe {
        let path = CString::new(syzygy_path).unwrap();
        let res = tb_init(path.as_ptr());
        assert!(res, "Failed to load Syzygy tablebases from {syzygy_path}");
    }
}

/// Gets maximal pieces count supported by loaded Syzygy tablebases. Returns 0 if the feature is disabled.
pub fn get_max_pieces_count() -> u8 {
    #![allow(clippy::cast_possible_truncation)]
    #[cfg(feature = "syzygy")]
    {
        let user_limit = uci::SYZYGY_PROBE_LIMIT.load(std::sync::atomic::Ordering::SeqCst);
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
    if board.castling_rights() != CastlingRights::NONE || board.fifty_move_counter() != 0 {
        return None;
    }

    #[cfg(feature = "syzygy")]
    unsafe {
        let wdl = tb_probe_wdl(
            board.pieces.occupied_co(Colour::WHITE),
            board.pieces.occupied_co(Colour::BLACK),
            board.pieces.all_kings(),
            board.pieces.all_queens(),
            board.pieces.all_rooks(),
            board.pieces.all_bishops(),
            board.pieces.all_knights(),
            board.pieces.all_pawns(),
            0,
            0,
            0,
            board.turn() == Colour::WHITE,
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
    #[cfg(feature = "syzygy")]
    unsafe {
        let result = tb_probe_root(
            board.pieces.occupied_co(Colour::WHITE),
            board.pieces.occupied_co(Colour::BLACK),
            board.pieces.all_kings(),
            board.pieces.all_queens(),
            board.pieces.all_rooks(),
            board.pieces.all_bishops(),
            board.pieces.all_knights(),
            board.pieces.all_pawns(),
            u32::from(board.fifty_move_counter()),
            0,
            0,
            board.turn() == Colour::WHITE,
            ptr::null_mut(),
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

        let from = Square::new(((result & TB_RESULT_FROM_MASK) >> TB_RESULT_FROM_SHIFT) as u8);
        let to = Square::new(((result & TB_RESULT_TO_MASK) >> TB_RESULT_TO_SHIFT) as u8);
        let promotion = (result & TB_RESULT_PROMOTES_MASK) >> TB_RESULT_PROMOTES_SHIFT;

        let promo_piece_type = match promotion {
            TB_PROMOTES_QUEEN => PieceType::QUEEN,
            TB_PROMOTES_ROOK => PieceType::ROOK,
            TB_PROMOTES_BISHOP => PieceType::BISHOP,
            TB_PROMOTES_KNIGHT => PieceType::KNIGHT,
            _ => PieceType::NONE,
        };

        for &m in moves.iter() {
            if m.from() == from
                && m.to() == to
                && (promotion == 0 || m.safe_promotion_type() == promo_piece_type)
            {
                return Some(WdlDtzResult { wdl, dtz, best_move: m });
            }
        }

        None
    }
    #[cfg(not(feature = "syzygy"))]
    None
}

/// Checks if there's a tablebase move and returns it as [Some], otherwise [None].
pub fn get_tablebase_move(board: &Board) -> Option<(Move, i32)> {
    if board.n_men() > get_max_pieces_count() {
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

    let stm = board.turn() == Colour::WHITE;

    match probe_result.wdl {
        WDL::Win => Some(if stm { WDL::Win } else { WDL::Loss }),
        WDL::Draw => Some(WDL::Draw),
        WDL::Loss => Some(if stm { WDL::Loss } else { WDL::Win }),
    }
}
