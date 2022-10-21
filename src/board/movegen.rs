pub mod bitboards;
pub mod movepicker;

pub use self::bitboards::{BitLoop, BB_NONE};
use self::{
    bitboards::{lsb, BitShiftExt, BB_RANK_2, BB_RANK_7},
    movepicker::MovePicker,
};

use super::Board;

use std::{
    fmt::{Display, Formatter},
    ops::Index,
};

use crate::{
    chessmove::Move,
    definitions::{
        Square::{B1, B8, C1, C8, D1, D8, E1, E8, F1, F8, G1, G8, NO_SQUARE},
        BB, BISHOP, BKCA, BN, BQ, BQCA, BR, KING, KNIGHT, PAWN, PIECE_EMPTY, ROOK, WB, WHITE, WKCA,
        WN, WQ, WQCA, WR,
    },
    lookups::get_mvv_lva_score,
    magic::MAGICS_READY,
    validate::piece_valid,
};

pub const TT_MOVE_SCORE: i32 = 20_000_000;
const FIRST_ORDER_KILLER_SCORE: i32 = 9_000_000;
const SECOND_ORDER_KILLER_SCORE: i32 = 8_000_000;
const COUNTER_MOVE_SCORE: i32 = 2_000_000;
const THIRD_ORDER_KILLER_SCORE: i32 = 1_000_000;
const WINNING_CAPTURE_SCORE: i32 = 10_000_000;
const MOVEGEN_SEE_THRESHOLD: i32 = 0;

const MAX_POSITION_MOVES: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoveListEntry {
    pub entry: Move,
    pub score: i32,
}

#[derive(Clone)]
pub struct MoveList {
    moves: [MoveListEntry; MAX_POSITION_MOVES],
    count: usize,
}

impl MoveList {
    pub const fn new() -> Self {
        const DEFAULT: MoveListEntry = MoveListEntry { entry: Move { data: 0 }, score: 0 };
        Self { moves: [DEFAULT; MAX_POSITION_MOVES], count: 0 }
    }

    pub fn lookup_by_move(&mut self, m: Move) -> Option<&mut MoveListEntry> {
        unsafe { self.moves.get_unchecked_mut(..self.count).iter_mut().find(|e| e.entry == m) }
    }

    pub fn push(&mut self, m: Move, score: i32) {
        // it's quite dangerous to do this,
        // but this function is very much in the
        // hot path.
        debug_assert!(self.count < MAX_POSITION_MOVES);
        unsafe {
            *self.moves.get_unchecked_mut(self.count) = MoveListEntry { entry: m, score };
        }
        self.count += 1;
    }

    pub fn iter(&self) -> impl Iterator<Item = &Move> {
        self.moves[..self.count].iter().map(|e| &e.entry)
    }

    pub fn init_movepicker(&mut self) -> MovePicker {
        MovePicker::new(&mut self.moves, self.count)
    }
}

impl Index<usize> for MoveList {
    type Output = Move;

    fn index(&self, index: usize) -> &Self::Output {
        &self.moves[..self.count][index].entry
    }
}

impl Display for MoveList {
    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
        if self.count == 0 {
            return write!(f, "MoveList: (0) []");
        }
        writeln!(f, "MoveList: ({}) [", self.count)?;
        for m in &self.moves[0..self.count - 1] {
            writeln!(f, "  {} ${}, ", m.entry, m.score)?;
        }
        writeln!(
            f,
            "  {} ${}",
            self.moves[self.count - 1].entry,
            self.moves[self.count - 1].score
        )?;
        write!(f, "]")
    }
}

impl Board {
    fn add_promo_move(&self, m: Move, move_list: &mut MoveList) {
        let mut score = get_mvv_lva_score(m.promotion(), PAWN);
        if self.static_exchange_eval(m, MOVEGEN_SEE_THRESHOLD) {
            score += WINNING_CAPTURE_SCORE;
        }

        move_list.push(m, score);
    }

    fn add_quiet_move(&self, m: Move, move_list: &mut MoveList) {
        let killer_entry = self.killer_move_table[self.height];

        let score = if killer_entry[0] == m {
            FIRST_ORDER_KILLER_SCORE
        } else if killer_entry[1] == m {
            SECOND_ORDER_KILLER_SCORE
        } else if self.is_countermove(m) {
            // move that refuted the previous move
            COUNTER_MOVE_SCORE
        } else if self.is_third_order_killer(m) {
            // killer from two moves ago
            THIRD_ORDER_KILLER_SCORE
        } else {
            let history = self.history_score(m);
            let followup_history = self.followup_history_score(m);
            history + 2 * followup_history
        };

        move_list.push(m, score);
    }

    fn add_capture_move(&self, m: Move, move_list: &mut MoveList) {
        let mut score = get_mvv_lva_score(m.capture(), self.piece_at(m.from()));
        if self.static_exchange_eval(m, MOVEGEN_SEE_THRESHOLD) {
            score += WINNING_CAPTURE_SCORE;
        }
        move_list.push(m, score);
    }

    fn add_ep_move(&self, m: Move, move_list: &mut MoveList) {
        let mut score = 1050; // the score for PxP in MVVLVA
        if self.static_exchange_eval(m, MOVEGEN_SEE_THRESHOLD) {
            score += WINNING_CAPTURE_SCORE;
        }
        move_list.push(m, score);
    }

    #[allow(clippy::cognitive_complexity)]
    fn generate_pawn_caps<const IS_WHITE: bool>(&self, move_list: &mut MoveList) {
        let our_pawns = self.pieces.pawns::<IS_WHITE>();
        let their_pieces = self.pieces.their_pieces::<IS_WHITE>();
        // to determine which pawns can capture, we shift the opponent's pieces backwards and find the intersection
        let attacks_west = if IS_WHITE {
            their_pieces.south_east_one() & our_pawns
        } else {
            their_pieces.north_east_one() & our_pawns
        };
        let attacks_east = if IS_WHITE {
            their_pieces.south_west_one() & our_pawns
        } else {
            their_pieces.north_west_one() & our_pawns
        };
        let promo_rank = if IS_WHITE { BB_RANK_7 } else { BB_RANK_2 };
        for from in BitLoop::new(attacks_west & !promo_rank) {
            let to = if IS_WHITE { from + 7 } else { from - 9 };
            let cap = self.piece_at(to);
            debug_assert!(piece_valid(cap));
            self.add_capture_move(Move::new(from, to, cap, PIECE_EMPTY, 0), move_list);
        }
        for from in BitLoop::new(attacks_east & !promo_rank) {
            let to = if IS_WHITE { from + 9 } else { from - 7 };
            let cap = self.piece_at(to);
            debug_assert!(piece_valid(cap));
            self.add_capture_move(Move::new(from, to, cap, PIECE_EMPTY, 0), move_list);
        }
        for from in BitLoop::new(attacks_west & promo_rank) {
            let to = if IS_WHITE { from + 7 } else { from - 9 };
            let cap = self.piece_at(to);
            debug_assert!(piece_valid(cap));
            if IS_WHITE {
                for &promo in &[WQ, WN, WR, WB] {
                    self.add_capture_move(Move::new(from, to, cap, promo, 0), move_list);
                }
            } else {
                for &promo in &[BQ, BN, BR, BB] {
                    self.add_capture_move(Move::new(from, to, cap, promo, 0), move_list);
                }
            }
        }
        for from in BitLoop::new(attacks_east & promo_rank) {
            let to = if IS_WHITE { from + 9 } else { from - 7 };
            let cap = self.piece_at(to);
            debug_assert!(piece_valid(cap));
            if IS_WHITE {
                for &promo in &[WQ, WN, WR, WB] {
                    self.add_capture_move(Move::new(from, to, cap, promo, 0), move_list);
                }
            } else {
                for &promo in &[BQ, BN, BR, BB] {
                    self.add_capture_move(Move::new(from, to, cap, promo, 0), move_list);
                }
            }
        }
    }

    fn generate_ep<const IS_WHITE: bool>(&self, move_list: &mut MoveList) {
        #![allow(clippy::cast_possible_truncation)]
        if self.ep_sq == NO_SQUARE {
            return;
        }
        let ep_bb = 1 << self.ep_sq;
        let our_pawns = self.pieces.pawns::<IS_WHITE>();
        let attacks_west = if IS_WHITE {
            ep_bb.south_east_one() & our_pawns
        } else {
            ep_bb.north_east_one() & our_pawns
        };
        let attacks_east = if IS_WHITE {
            ep_bb.south_west_one() & our_pawns
        } else {
            ep_bb.north_west_one() & our_pawns
        };

        if attacks_west != 0 {
            let from_sq = lsb(attacks_west) as u8;
            self.add_ep_move(
                Move::new(from_sq, self.ep_sq, PIECE_EMPTY, PIECE_EMPTY, Move::EP_MASK),
                move_list,
            );
        }
        if attacks_east != 0 {
            let from_sq = lsb(attacks_east) as u8;
            self.add_ep_move(
                Move::new(from_sq, self.ep_sq, PIECE_EMPTY, PIECE_EMPTY, Move::EP_MASK),
                move_list,
            );
        }
    }

    fn generate_pawn_forward<const IS_WHITE: bool>(&self, move_list: &mut MoveList) {
        let start_rank = if IS_WHITE { BB_RANK_2 } else { BB_RANK_7 };
        let promo_rank = if IS_WHITE { BB_RANK_7 } else { BB_RANK_2 };
        let shifted_empty_squares =
            if IS_WHITE { self.pieces.empty() >> 8 } else { self.pieces.empty() << 8 };
        let double_shifted_empty_squares =
            if IS_WHITE { self.pieces.empty() >> 16 } else { self.pieces.empty() << 16 };
        let our_pawns = self.pieces.pawns::<IS_WHITE>();
        let pushable_pawns = our_pawns & shifted_empty_squares;
        let double_pushable_pawns = pushable_pawns & double_shifted_empty_squares & start_rank;
        let promoting_pawns = pushable_pawns & promo_rank;
        for sq in BitLoop::new(pushable_pawns & !promoting_pawns) {
            let to = if IS_WHITE { sq + 8 } else { sq - 8 };
            self.add_quiet_move(Move::new(sq, to, PIECE_EMPTY, PIECE_EMPTY, 0), move_list);
        }
        for sq in BitLoop::new(double_pushable_pawns) {
            let to = if IS_WHITE { sq + 16 } else { sq - 16 };
            self.add_quiet_move(
                Move::new(sq, to, PIECE_EMPTY, PIECE_EMPTY, Move::PAWN_START_MASK),
                move_list,
            );
        }
        for sq in BitLoop::new(promoting_pawns) {
            let to = if IS_WHITE { sq + 8 } else { sq - 8 };
            if IS_WHITE {
                for &promo in &[WQ, WN, WR, WB] {
                    self.add_promo_move(Move::new(sq, to, PIECE_EMPTY, promo, 0), move_list);
                }
            } else {
                for &promo in &[BQ, BN, BR, BB] {
                    self.add_promo_move(Move::new(sq, to, PIECE_EMPTY, promo, 0), move_list);
                }
            }
        }
    }

    fn generate_forward_promos<const IS_WHITE: bool>(&self, move_list: &mut MoveList) {
        let promo_rank = if IS_WHITE { BB_RANK_7 } else { BB_RANK_2 };
        let shifted_empty_squares =
            if IS_WHITE { self.pieces.empty() >> 8 } else { self.pieces.empty() << 8 };
        let our_pawns = self.pieces.pawns::<IS_WHITE>();
        let pushable_pawns = our_pawns & shifted_empty_squares;
        let promoting_pawns = pushable_pawns & promo_rank;
        for sq in BitLoop::new(promoting_pawns) {
            let to = if IS_WHITE { sq + 8 } else { sq - 8 };
            if IS_WHITE {
                for &promo in &[WQ, WN, WR, WB] {
                    self.add_promo_move(Move::new(sq, to, PIECE_EMPTY, promo, 0), move_list);
                }
            } else {
                for &promo in &[BQ, BN, BR, BB] {
                    self.add_promo_move(Move::new(sq, to, PIECE_EMPTY, promo, 0), move_list);
                }
            }
        }
    }

    pub fn generate_moves(&self, move_list: &mut MoveList) {
        debug_assert!(self.movegen_ready);
        debug_assert!(MAGICS_READY.load(std::sync::atomic::Ordering::SeqCst));
        if self.side == WHITE {
            self.generate_moves_for::<true>(move_list);
        } else {
            self.generate_moves_for::<false>(move_list);
        }
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    pub fn generate_moves_for<const IS_WHITE: bool>(&self, move_list: &mut MoveList) {
        #[cfg(debug_assertions)]
        self.check_validity().unwrap();

        self.generate_pawn_forward::<IS_WHITE>(move_list);
        self.generate_pawn_caps::<IS_WHITE>(move_list);
        self.generate_ep::<IS_WHITE>(move_list);

        // knights
        let our_knights = self.pieces.knights::<IS_WHITE>();
        let their_pieces = self.pieces.their_pieces::<IS_WHITE>();
        let freespace = self.pieces.empty();
        for sq in BitLoop::new(our_knights) {
            let moves = bitboards::attacks::<KNIGHT>(sq, BB_NONE);
            for to in BitLoop::new(moves & their_pieces) {
                self.add_capture_move(
                    Move::new(sq, to, self.piece_at(to), PIECE_EMPTY, 0),
                    move_list,
                );
            }
            for to in BitLoop::new(moves & freespace) {
                self.add_quiet_move(Move::new(sq, to, PIECE_EMPTY, PIECE_EMPTY, 0), move_list);
            }
        }

        // kings
        let our_king = self.pieces.king::<IS_WHITE>();
        for sq in BitLoop::new(our_king) {
            let moves = bitboards::attacks::<KING>(sq, BB_NONE);
            for to in BitLoop::new(moves & their_pieces) {
                self.add_capture_move(
                    Move::new(sq, to, self.piece_at(to), PIECE_EMPTY, 0),
                    move_list,
                );
            }
            for to in BitLoop::new(moves & freespace) {
                self.add_quiet_move(Move::new(sq, to, PIECE_EMPTY, PIECE_EMPTY, 0), move_list);
            }
        }

        // bishops and queens
        let our_diagonal_sliders = self.pieces.bishopqueen::<IS_WHITE>();
        let blockers = self.pieces.occupied();
        for sq in BitLoop::new(our_diagonal_sliders) {
            let moves = bitboards::attacks::<BISHOP>(sq, blockers);
            for to in BitLoop::new(moves & their_pieces) {
                self.add_capture_move(
                    Move::new(sq, to, self.piece_at(to), PIECE_EMPTY, 0),
                    move_list,
                );
            }
            for to in BitLoop::new(moves & freespace) {
                self.add_quiet_move(Move::new(sq, to, PIECE_EMPTY, PIECE_EMPTY, 0), move_list);
            }
        }

        // rooks and queens
        let our_orthogonal_sliders = self.pieces.rookqueen::<IS_WHITE>();
        for sq in BitLoop::new(our_orthogonal_sliders) {
            let moves = bitboards::attacks::<ROOK>(sq, blockers);
            for to in BitLoop::new(moves & their_pieces) {
                self.add_capture_move(
                    Move::new(sq, to, self.piece_at(to), PIECE_EMPTY, 0),
                    move_list,
                );
            }
            for to in BitLoop::new(moves & freespace) {
                self.add_quiet_move(Move::new(sq, to, PIECE_EMPTY, PIECE_EMPTY, 0), move_list);
            }
        }

        self.generate_castling_moves_for::<IS_WHITE>(move_list);
    }

    pub fn generate_captures(&self, move_list: &mut MoveList) {
        if self.side == WHITE {
            self.generate_captures_for::<true>(move_list);
        } else {
            self.generate_captures_for::<false>(move_list);
        }
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    pub fn generate_captures_for<const IS_WHITE: bool>(&self, move_list: &mut MoveList) {
        #[cfg(debug_assertions)]
        self.check_validity().unwrap();

        // promotions
        self.generate_forward_promos::<IS_WHITE>(move_list);

        // pawn captures and capture promos
        self.generate_pawn_caps::<IS_WHITE>(move_list);
        self.generate_ep::<IS_WHITE>(move_list);

        // knights
        let our_knights = self.pieces.knights::<IS_WHITE>();
        let their_pieces = self.pieces.their_pieces::<IS_WHITE>();
        for sq in BitLoop::new(our_knights) {
            let moves = bitboards::attacks::<KNIGHT>(sq, BB_NONE);
            for to in BitLoop::new(moves & their_pieces) {
                self.add_capture_move(
                    Move::new(sq, to, self.piece_at(to), PIECE_EMPTY, 0),
                    move_list,
                );
            }
        }

        // kings
        let our_king = self.pieces.king::<IS_WHITE>();
        for sq in BitLoop::new(our_king) {
            let moves = bitboards::attacks::<KING>(sq, BB_NONE);
            for to in BitLoop::new(moves & their_pieces) {
                self.add_capture_move(
                    Move::new(sq, to, self.piece_at(to), PIECE_EMPTY, 0),
                    move_list,
                );
            }
        }

        // bishops and queens
        let our_diagonal_sliders = self.pieces.bishopqueen::<IS_WHITE>();
        let blockers = self.pieces.occupied();
        for sq in BitLoop::new(our_diagonal_sliders) {
            let moves = bitboards::attacks::<BISHOP>(sq, blockers);
            for to in BitLoop::new(moves & their_pieces) {
                self.add_capture_move(
                    Move::new(sq, to, self.piece_at(to), PIECE_EMPTY, 0),
                    move_list,
                );
            }
        }

        // rooks and queens
        let our_orthogonal_sliders = self.pieces.rookqueen::<IS_WHITE>();
        let blockers = self.pieces.occupied();
        for sq in BitLoop::new(our_orthogonal_sliders) {
            let moves = bitboards::attacks::<ROOK>(sq, blockers);
            for to in BitLoop::new(moves & their_pieces) {
                self.add_capture_move(
                    Move::new(sq, to, self.piece_at(to), PIECE_EMPTY, 0),
                    move_list,
                );
            }
        }
    }

    pub fn generate_castling_moves(&self, move_list: &mut MoveList) {
        if self.side == WHITE {
            self.generate_castling_moves_for::<true>(move_list);
        } else {
            self.generate_castling_moves_for::<false>(move_list);
        }
    }

    pub fn generate_castling_moves_for<const IS_WHITE: bool>(&self, move_list: &mut MoveList) {
        const WK_FREESPACE: u64 = 1 << F1 | 1 << G1;
        const WQ_FREESPACE: u64 = 1 << B1 | 1 << C1 | 1 << D1;
        const BK_FREESPACE: u64 = 1 << F8 | 1 << G8;
        const BQ_FREESPACE: u64 = 1 << B8 | 1 << C8 | 1 << D8;
        let occupied = self.pieces.occupied();
        if IS_WHITE {
            if self.castle_perm & WKCA != 0
                && occupied & WK_FREESPACE == 0
                && !self.sq_attacked_by::<false>(E1)
                && !self.sq_attacked_by::<false>(F1)
            {
                self.add_quiet_move(
                    Move::new(E1, G1, PIECE_EMPTY, PIECE_EMPTY, Move::CASTLE_MASK),
                    move_list,
                );
            }

            if self.castle_perm & WQCA != 0
                && occupied & WQ_FREESPACE == 0
                && !self.sq_attacked_by::<false>(E1)
                && !self.sq_attacked_by::<false>(D1)
            {
                self.add_quiet_move(
                    Move::new(E1, C1, PIECE_EMPTY, PIECE_EMPTY, Move::CASTLE_MASK),
                    move_list,
                );
            }
        } else {
            if self.castle_perm & BKCA != 0
                && occupied & BK_FREESPACE == 0
                && !self.sq_attacked_by::<true>(E8)
                && !self.sq_attacked_by::<true>(F8)
            {
                self.add_quiet_move(
                    Move::new(E8, G8, PIECE_EMPTY, PIECE_EMPTY, Move::CASTLE_MASK),
                    move_list,
                );
            }

            if self.castle_perm & BQCA != 0
                && occupied & BQ_FREESPACE == 0
                && !self.sq_attacked_by::<true>(E8)
                && !self.sq_attacked_by::<true>(D8)
            {
                self.add_quiet_move(
                    Move::new(E8, C8, PIECE_EMPTY, PIECE_EMPTY, Move::CASTLE_MASK),
                    move_list,
                );
            }
        }
    }

    pub fn _attackers_mask(&self, sq: u8, side: u8, blockers: u64) -> u64 {
        let mut attackers = 0;
        if side == WHITE {
            let our_pawns = self.pieces.pawns::<true>();
            let west_attacks = our_pawns.north_west_one();
            let east_attacks = our_pawns.north_east_one();
            let pawn_attacks = west_attacks | east_attacks;
            attackers |= pawn_attacks;
        } else {
            let our_pawns = self.pieces.pawns::<false>();
            let west_attacks = our_pawns.south_west_one();
            let east_attacks = our_pawns.south_east_one();
            let pawn_attacks = west_attacks | east_attacks;
            attackers |= pawn_attacks;
        }

        let our_knights = if side == WHITE {
            self.pieces.knights::<true>()
        } else {
            self.pieces.knights::<false>()
        };

        let knight_attacks = bitboards::attacks::<KNIGHT>(sq, BB_NONE) & our_knights;
        attackers |= knight_attacks;

        let our_diag_pieces = if side == WHITE {
            self.pieces.bishopqueen::<true>()
        } else {
            self.pieces.bishopqueen::<false>()
        };

        let diag_attacks = bitboards::attacks::<BISHOP>(sq, blockers) & our_diag_pieces;
        attackers |= diag_attacks;

        let our_orth_pieces = if side == WHITE {
            self.pieces.rookqueen::<true>()
        } else {
            self.pieces.rookqueen::<false>()
        };

        let orth_attacks = bitboards::attacks::<ROOK>(sq, blockers) & our_orth_pieces;
        attackers |= orth_attacks;

        let our_king =
            if side == WHITE { self.pieces.king::<true>() } else { self.pieces.king::<false>() };

        let king_attacks = bitboards::attacks::<KING>(sq, BB_NONE) & our_king;
        attackers |= king_attacks;

        attackers
    }
}
