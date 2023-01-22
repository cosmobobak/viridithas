#![allow(dead_code)]

use crate::{tablebases::bindings::*, uci, board::{Board, movegen::MoveList}, piece::{Colour, PieceType}, chessmove::Move, definitions::Square};
use std::ffi::CString;
use std::ptr;

pub enum WDL { Win, Loss, Draw }
pub struct WdlDtzResult { wdl: WDL, dtz: u32, best_move: Move }

/// Loads Syzygy tablebases stored in `syzygy_path` location.
pub fn init(syzygy_path: &str) {
    #[cfg(feature = "syzygy")]
    unsafe {
        let path = CString::new(syzygy_path).unwrap();
        let res = tb_init(path.as_ptr());
        if !res {
            panic!("Failed to load Syzygy tablebases from {syzygy_path}");
        }
    }
}

/// Gets maximal pieces count supported by loaded Syzygy tablebases. Returns 0 if the feature is disabled.
pub fn get_max_pieces_count() -> u8 {
    #[cfg(feature = "syzygy")]
    {
        let user_limit = uci::SYZYGY_PROBE_LIMIT.load(std::sync::atomic::Ordering::SeqCst);
        let hard_limit = unsafe { TB_LARGEST as u8 };
        #[cfg(debug_assertions)]
        println!("Syzygy probe limit: {} (user limit: {})", hard_limit, user_limit);
        return std::cmp::min(user_limit, hard_limit);
    }
    #[cfg(not(feature = "syzygy"))]
    0
}

/// Gets WDL (Win-Draw-Loss) for the position specified in `board`. Returns [None] if data couldn't be obtained or the feature is disabled.
pub fn get_wdl(board: &Board) -> Option<WDL> {
    const WHITE: bool = true;
    const BLACK: bool = false;

    // guards for invalid positions
    if board.castling_rights() != 0 || board.fifty_move_counter() != 0 {
        return None;
    }

    #[cfg(feature = "syzygy")]
    unsafe {
        let wdl = tb_probe_wdl(
            board.pieces.occupied_co(Colour::WHITE),
            board.pieces.occupied_co(Colour::BLACK),
            board.pieces.king::<WHITE>() | board.pieces.king::<BLACK>(),
            board.pieces.queens::<WHITE>() | board.pieces.queens::<BLACK>(),
            board.pieces.rooks::<WHITE>() | board.pieces.rooks::<BLACK>(),
            board.pieces.bishops::<WHITE>() | board.pieces.bishops::<BLACK>(),
            board.pieces.knights::<WHITE>() | board.pieces.knights::<BLACK>(),
            board.pieces.pawns::<WHITE>() | board.pieces.pawns::<BLACK>(),
            0,
            0,
            0,
            board.turn() == Colour::WHITE,
        );

        return match wdl {
            TB_WIN => Some(WDL::Win),
            TB_LOSS => Some(WDL::Loss),
            TB_DRAW | TB_CURSED_WIN | TB_BLESSED_LOSS => Some(WDL::Draw),
            _ => None,
        };
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
            board.pieces.king::<WHITE>() | board.pieces.king::<BLACK>(),
            board.pieces.queens::<WHITE>() | board.pieces.queens::<BLACK>(),
            board.pieces.rooks::<WHITE>() | board.pieces.rooks::<BLACK>(),
            board.pieces.bishops::<WHITE>() | board.pieces.bishops::<BLACK>(),
            board.pieces.knights::<WHITE>() | board.pieces.knights::<BLACK>(),
            board.pieces.pawns::<WHITE>() | board.pieces.pawns::<BLACK>(),
            board.fifty_move_counter() as u32,
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
            _ => PieceType::NO_PIECE_TYPE,
        };

        for &m in moves.iter() {
            if m.from() == from && m.to() == to {
                if promotion == 0 || m.safe_promotion_type() == promo_piece_type {
                    return Some(WdlDtzResult { wdl, dtz, best_move: m });
                }
            }
        }

        return None;
    }
    #[cfg(not(feature = "syzygy"))]
    None
}