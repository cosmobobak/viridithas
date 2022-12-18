pub mod bitboards;
pub mod movepicker;

use self::bitboards::{first_square, BitShiftExt, BB_RANK_2, BB_RANK_7};
pub use self::bitboards::{BitLoop, BB_NONE};

use super::Board;

use std::{
    fmt::{Display, Formatter},
    ops::Index,
};

use crate::{
    chessmove::Move,
    definitions::{
        Square, BISHOP, BKCA, BQCA, KING, KNIGHT, PIECE_EMPTY, QUEEN, ROOK, WHITE, WKCA, WQCA,
    },
    magic::MAGICS_READY,
    validate::piece_valid,
};

pub const MAX_POSITION_MOVES: usize = 218;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoveListEntry {
    pub mov: Move,
    pub score: i32,
}

impl MoveListEntry {
    pub const TACTICAL_SENTINEL: i32 = 0x7FFF_FFFF;
    pub const QUIET_SENTINEL: i32 = 0x7FFF_FFFE;
}

#[derive(Clone)]
pub struct MoveList {
    moves: [MoveListEntry; MAX_POSITION_MOVES],
    count: usize,
}

impl MoveList {
    pub const fn new() -> Self {
        const DEFAULT: MoveListEntry = MoveListEntry { mov: Move { data: 0 }, score: 0 };
        Self { moves: [DEFAULT; MAX_POSITION_MOVES], count: 0 }
    }

    pub fn push<const TACTICAL: bool>(&mut self, m: Move) {
        // it's quite dangerous to do this,
        // but this function is very much in the
        // hot path.
        debug_assert!(self.count < MAX_POSITION_MOVES);
        let score =
            if TACTICAL { MoveListEntry::TACTICAL_SENTINEL } else { MoveListEntry::QUIET_SENTINEL };
        unsafe {
            *self.moves.get_unchecked_mut(self.count) = MoveListEntry { mov: m, score };
        }
        self.count += 1;
    }

    pub fn iter(&self) -> impl Iterator<Item = &Move> {
        self.moves[..self.count].iter().map(|e| &e.mov)
    }
}

impl Index<usize> for MoveList {
    type Output = Move;

    fn index(&self, index: usize) -> &Self::Output {
        &self.moves[..self.count][index].mov
    }
}

impl Display for MoveList {
    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
        if self.count == 0 {
            return write!(f, "MoveList: (0) []");
        }
        writeln!(f, "MoveList: ({}) [", self.count)?;
        for m in &self.moves[0..self.count - 1] {
            writeln!(f, "  {} ${}, ", m.mov, m.score)?;
        }
        writeln!(f, "  {} ${}", self.moves[self.count - 1].mov, self.moves[self.count - 1].score)?;
        write!(f, "]")
    }
}

impl Board {
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
            let to = if IS_WHITE { from.add(7) } else { from.sub(9) };
            let cap = self.piece_at(to);
            debug_assert!(piece_valid(cap));
            move_list.push::<true>(Move::new(from, to, PIECE_EMPTY, 0));
        }
        for from in BitLoop::new(attacks_east & !promo_rank) {
            let to = if IS_WHITE { from.add(9) } else { from.sub(7) };
            let cap = self.piece_at(to);
            debug_assert!(piece_valid(cap));
            move_list.push::<true>(Move::new(from, to, PIECE_EMPTY, 0));
        }
        for from in BitLoop::new(attacks_west & promo_rank) {
            let to = if IS_WHITE { from.add(7) } else { from.sub(9) };
            let cap = self.piece_at(to);
            debug_assert!(piece_valid(cap));
            for &promo in &[QUEEN, KNIGHT, ROOK, BISHOP] {
                move_list.push::<true>(Move::new(from, to, promo, Move::PROMO_FLAG));
            }
        }
        for from in BitLoop::new(attacks_east & promo_rank) {
            let to = if IS_WHITE { from.add(9) } else { from.sub(7) };
            let cap = self.piece_at(to);
            debug_assert!(piece_valid(cap));
            for &promo in &[QUEEN, KNIGHT, ROOK, BISHOP] {
                move_list.push::<true>(Move::new(from, to, promo, Move::PROMO_FLAG));
            }
        }
    }

    fn generate_ep<const IS_WHITE: bool>(&self, move_list: &mut MoveList) {
        #![allow(clippy::cast_possible_truncation)]
        if self.ep_sq == Square::NO_SQUARE {
            return;
        }
        let ep_bb = self.ep_sq.bitboard();
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
            let from_sq = first_square(attacks_west);
            move_list.push::<true>(Move::new(from_sq, self.ep_sq, PIECE_EMPTY, Move::EP_FLAG));
        }
        if attacks_east != 0 {
            let from_sq = first_square(attacks_east);
            move_list.push::<true>(Move::new(from_sq, self.ep_sq, PIECE_EMPTY, Move::EP_FLAG));
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
            let to = if IS_WHITE { sq.add(8) } else { sq.sub(8) };
            move_list.push::<false>(Move::new(sq, to, PIECE_EMPTY, 0));
        }
        for sq in BitLoop::new(double_pushable_pawns) {
            let to = if IS_WHITE { sq.add(16) } else { sq.sub(16) };
            move_list.push::<false>(Move::new(sq, to, PIECE_EMPTY, 0));
        }
        for sq in BitLoop::new(promoting_pawns) {
            let to = if IS_WHITE { sq.add(8) } else { sq.sub(8) };
            for &promo in &[QUEEN, KNIGHT, ROOK, BISHOP] {
                move_list.push::<true>(Move::new(sq, to, promo, Move::PROMO_FLAG));
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
            let to = if IS_WHITE { sq.add(8) } else { sq.sub(8) };
            for &promo in &[QUEEN, KNIGHT, ROOK, BISHOP] {
                move_list.push::<true>(Move::new(sq, to, promo, Move::PROMO_FLAG));
            }
        }
    }

    pub fn generate_moves(&self, move_list: &mut MoveList) {
        debug_assert!(MAGICS_READY.load(std::sync::atomic::Ordering::SeqCst));
        if self.side == WHITE {
            self.generate_moves_for::<true>(move_list);
        } else {
            self.generate_moves_for::<false>(move_list);
        }
        debug_assert!(move_list.iter().all(|m| m.is_valid()));
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
                move_list.push::<true>(Move::new(sq, to, PIECE_EMPTY, 0));
            }
            for to in BitLoop::new(moves & freespace) {
                move_list.push::<false>(Move::new(sq, to, PIECE_EMPTY, 0));
            }
        }

        // kings
        let our_king = self.pieces.king::<IS_WHITE>();
        for sq in BitLoop::new(our_king) {
            let moves = bitboards::attacks::<KING>(sq, BB_NONE);
            for to in BitLoop::new(moves & their_pieces) {
                move_list.push::<true>(Move::new(sq, to, PIECE_EMPTY, 0));
            }
            for to in BitLoop::new(moves & freespace) {
                move_list.push::<false>(Move::new(sq, to, PIECE_EMPTY, 0));
            }
        }

        // bishops and queens
        let our_diagonal_sliders = self.pieces.bishopqueen::<IS_WHITE>();
        let blockers = self.pieces.occupied();
        for sq in BitLoop::new(our_diagonal_sliders) {
            let moves = bitboards::attacks::<BISHOP>(sq, blockers);
            for to in BitLoop::new(moves & their_pieces) {
                move_list.push::<true>(Move::new(sq, to, PIECE_EMPTY, 0));
            }
            for to in BitLoop::new(moves & freespace) {
                move_list.push::<false>(Move::new(sq, to, PIECE_EMPTY, 0));
            }
        }

        // rooks and queens
        let our_orthogonal_sliders = self.pieces.rookqueen::<IS_WHITE>();
        for sq in BitLoop::new(our_orthogonal_sliders) {
            let moves = bitboards::attacks::<ROOK>(sq, blockers);
            for to in BitLoop::new(moves & their_pieces) {
                move_list.push::<true>(Move::new(sq, to, PIECE_EMPTY, 0));
            }
            for to in BitLoop::new(moves & freespace) {
                move_list.push::<false>(Move::new(sq, to, PIECE_EMPTY, 0));
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
                move_list.push::<true>(Move::new(sq, to, PIECE_EMPTY, 0));
            }
        }

        // kings
        let our_king = self.pieces.king::<IS_WHITE>();
        for sq in BitLoop::new(our_king) {
            let moves = bitboards::attacks::<KING>(sq, BB_NONE);
            for to in BitLoop::new(moves & their_pieces) {
                move_list.push::<true>(Move::new(sq, to, PIECE_EMPTY, 0));
            }
        }

        // bishops and queens
        let our_diagonal_sliders = self.pieces.bishopqueen::<IS_WHITE>();
        let blockers = self.pieces.occupied();
        for sq in BitLoop::new(our_diagonal_sliders) {
            let moves = bitboards::attacks::<BISHOP>(sq, blockers);
            for to in BitLoop::new(moves & their_pieces) {
                move_list.push::<true>(Move::new(sq, to, PIECE_EMPTY, 0));
            }
        }

        // rooks and queens
        let our_orthogonal_sliders = self.pieces.rookqueen::<IS_WHITE>();
        let blockers = self.pieces.occupied();
        for sq in BitLoop::new(our_orthogonal_sliders) {
            let moves = bitboards::attacks::<ROOK>(sq, blockers);
            for to in BitLoop::new(moves & their_pieces) {
                move_list.push::<true>(Move::new(sq, to, PIECE_EMPTY, 0));
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
        const WK_FREESPACE: u64 = Square::F1.bitboard() | Square::G1.bitboard();
        const WQ_FREESPACE: u64 =
            Square::B1.bitboard() | Square::C1.bitboard() | Square::D1.bitboard();
        const BK_FREESPACE: u64 = Square::F8.bitboard() | Square::G8.bitboard();
        const BQ_FREESPACE: u64 =
            Square::B8.bitboard() | Square::C8.bitboard() | Square::D8.bitboard();
        let occupied = self.pieces.occupied();
        if IS_WHITE {
            if self.castle_perm & WKCA != 0
                && occupied & WK_FREESPACE == 0
                && !self.sq_attacked_by::<false>(Square::E1)
                && !self.sq_attacked_by::<false>(Square::F1)
            {
                move_list.push::<false>(Move::new(
                    Square::E1,
                    Square::G1,
                    PIECE_EMPTY,
                    Move::CASTLE_FLAG,
                ));
            }

            if self.castle_perm & WQCA != 0
                && occupied & WQ_FREESPACE == 0
                && !self.sq_attacked_by::<false>(Square::E1)
                && !self.sq_attacked_by::<false>(Square::D1)
            {
                move_list.push::<false>(Move::new(
                    Square::E1,
                    Square::C1,
                    PIECE_EMPTY,
                    Move::CASTLE_FLAG,
                ));
            }
        } else {
            if self.castle_perm & BKCA != 0
                && occupied & BK_FREESPACE == 0
                && !self.sq_attacked_by::<true>(Square::E8)
                && !self.sq_attacked_by::<true>(Square::F8)
            {
                move_list.push::<false>(Move::new(
                    Square::E8,
                    Square::G8,
                    PIECE_EMPTY,
                    Move::CASTLE_FLAG,
                ));
            }

            if self.castle_perm & BQCA != 0
                && occupied & BQ_FREESPACE == 0
                && !self.sq_attacked_by::<true>(Square::E8)
                && !self.sq_attacked_by::<true>(Square::D8)
            {
                move_list.push::<false>(Move::new(
                    Square::E8,
                    Square::C8,
                    PIECE_EMPTY,
                    Move::CASTLE_FLAG,
                ));
            }
        }
    }

    pub fn _attackers_mask(&self, sq: Square, side: u8, blockers: u64) -> u64 {
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
