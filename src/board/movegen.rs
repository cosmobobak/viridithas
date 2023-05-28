pub mod bitboards;
pub mod movepicker;

use self::bitboards::{first_square, BitHackExt, BB_RANK_2, BB_RANK_7};
pub use self::bitboards::{BitLoop, BB_NONE};

use super::Board;

use std::{
    fmt::{Display, Formatter},
    ops::Index,
};

use crate::{
    chessmove::Move,
    definitions::{Square, ORTHO_RAY_BETWEEN},
    magic::MAGICS_READY,
    piece::{Colour, PieceType},
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
        const DEFAULT: MoveListEntry = MoveListEntry { mov: Move::NULL, score: 0 };
        Self { moves: [DEFAULT; MAX_POSITION_MOVES], count: 0 }
    }

    fn push<const TACTICAL: bool>(&mut self, m: Move) {
        debug_assert!(self.count < MAX_POSITION_MOVES, "overflowed {self}");
        let score =
            if TACTICAL { MoveListEntry::TACTICAL_SENTINEL } else { MoveListEntry::QUIET_SENTINEL };

        self.moves[self.count] = MoveListEntry { mov: m, score };
        self.count += 1;
    }

    pub fn iter(&self) -> impl Iterator<Item = &Move> {
        self.moves[..self.count].iter().map(|e| &e.mov)
    }

    pub fn as_slice(&self) -> &[MoveListEntry] {
        &self.moves[..self.count]
    }

    pub fn as_slice_mut(&mut self) -> &mut [MoveListEntry] {
        &mut self.moves[..self.count]
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
    fn generate_pawn_caps<const IS_WHITE: bool, const QS: bool>(&self, move_list: &mut MoveList) {
        let our_pawns = self.pieces.pawns::<IS_WHITE>();
        let their_pieces = self.pieces.their_pieces::<IS_WHITE>();
        // to determine which pawns can capture, we shift the opponent's pieces backwards and find the intersection
        let attacking_west = if IS_WHITE {
            their_pieces.south_east_one() & our_pawns
        } else {
            their_pieces.north_east_one() & our_pawns
        };
        let attacking_east = if IS_WHITE {
            their_pieces.south_west_one() & our_pawns
        } else {
            their_pieces.north_west_one() & our_pawns
        };
        let promo_rank = if IS_WHITE { BB_RANK_7 } else { BB_RANK_2 };
        for from in BitLoop::new(attacking_west & !promo_rank) {
            let to = if IS_WHITE { from.add(7) } else { from.sub(9) };
            move_list.push::<true>(Move::new(from, to));
        }
        for from in BitLoop::new(attacking_east & !promo_rank) {
            let to = if IS_WHITE { from.add(9) } else { from.sub(7) };
            move_list.push::<true>(Move::new(from, to));
        }
        if QS {
            // in quiescence search, we only generate promotions to queen.
            for from in BitLoop::new(attacking_west & promo_rank) {
                let to = if IS_WHITE { from.add(7) } else { from.sub(9) };
                move_list.push::<true>(Move::new_with_promo(from, to, PieceType::QUEEN));
            }
            for from in BitLoop::new(attacking_east & promo_rank) {
                let to = if IS_WHITE { from.add(9) } else { from.sub(7) };
                move_list.push::<true>(Move::new_with_promo(from, to, PieceType::QUEEN));
            }
        } else {
            for from in BitLoop::new(attacking_west & promo_rank) {
                let to = if IS_WHITE { from.add(7) } else { from.sub(9) };
                for promo in
                    [PieceType::QUEEN, PieceType::ROOK, PieceType::BISHOP, PieceType::KNIGHT]
                {
                    move_list.push::<true>(Move::new_with_promo(from, to, promo));
                }
            }
            for from in BitLoop::new(attacking_east & promo_rank) {
                let to = if IS_WHITE { from.add(9) } else { from.sub(7) };
                for promo in
                    [PieceType::QUEEN, PieceType::ROOK, PieceType::BISHOP, PieceType::KNIGHT]
                {
                    move_list.push::<true>(Move::new_with_promo(from, to, promo));
                }
            }
        }
    }

    fn generate_ep<const IS_WHITE: bool>(&self, move_list: &mut MoveList) {
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
            move_list.push::<true>(Move::new_with_flags(from_sq, self.ep_sq, Move::EP_FLAG));
        }
        if attacks_east != 0 {
            let from_sq = first_square(attacks_east);
            move_list.push::<true>(Move::new_with_flags(from_sq, self.ep_sq, Move::EP_FLAG));
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
            move_list.push::<false>(Move::new(sq, to));
        }
        for sq in BitLoop::new(double_pushable_pawns) {
            let to = if IS_WHITE { sq.add(16) } else { sq.sub(16) };
            move_list.push::<false>(Move::new(sq, to));
        }
        for sq in BitLoop::new(promoting_pawns) {
            let to = if IS_WHITE { sq.add(8) } else { sq.sub(8) };
            for promo in [PieceType::QUEEN, PieceType::KNIGHT, PieceType::ROOK, PieceType::BISHOP] {
                move_list.push::<true>(Move::new_with_promo(sq, to, promo));
            }
        }
    }

    fn generate_forward_promos<const IS_WHITE: bool, const QS: bool>(
        &self,
        move_list: &mut MoveList,
    ) {
        let promo_rank = if IS_WHITE { BB_RANK_7 } else { BB_RANK_2 };
        let shifted_empty_squares =
            if IS_WHITE { self.pieces.empty() >> 8 } else { self.pieces.empty() << 8 };
        let our_pawns = self.pieces.pawns::<IS_WHITE>();
        let pushable_pawns = our_pawns & shifted_empty_squares;
        let promoting_pawns = pushable_pawns & promo_rank;
        for sq in BitLoop::new(promoting_pawns) {
            let to = if IS_WHITE { sq.add(8) } else { sq.sub(8) };
            if QS {
                // in quiescence search, we only generate promotions to queen.
                move_list.push::<true>(Move::new_with_promo(sq, to, PieceType::QUEEN));
            } else {
                for promo in
                    [PieceType::QUEEN, PieceType::KNIGHT, PieceType::ROOK, PieceType::BISHOP]
                {
                    move_list.push::<true>(Move::new_with_promo(sq, to, promo));
                }
            }
        }
    }

    pub fn generate_moves(&self, move_list: &mut MoveList) {
        debug_assert!(MAGICS_READY.load(std::sync::atomic::Ordering::SeqCst));
        move_list.count = 0; // VERY IMPORTANT FOR UPHOLDING INVARIANTS.
        if self.side == Colour::WHITE {
            self.generate_moves_for::<true>(move_list);
        } else {
            self.generate_moves_for::<false>(move_list);
        }
        debug_assert!(move_list.iter().all(|m| m.is_valid()));
    }

    fn generate_moves_for<const IS_WHITE: bool>(&self, move_list: &mut MoveList) {
        #[cfg(debug_assertions)]
        self.check_validity().unwrap();

        self.generate_pawn_forward::<IS_WHITE>(move_list);
        self.generate_pawn_caps::<IS_WHITE, false>(move_list);
        self.generate_ep::<IS_WHITE>(move_list);

        // knights
        let our_knights = self.pieces.knights::<IS_WHITE>();
        let their_pieces = self.pieces.their_pieces::<IS_WHITE>();
        let freespace = self.pieces.empty();
        for sq in BitLoop::new(our_knights) {
            let moves = bitboards::attacks::<{ PieceType::KNIGHT.inner() }>(sq, BB_NONE);
            for to in BitLoop::new(moves & their_pieces) {
                move_list.push::<true>(Move::new(sq, to));
            }
            for to in BitLoop::new(moves & freespace) {
                move_list.push::<false>(Move::new(sq, to));
            }
        }

        // kings
        let our_king = self.pieces.king::<IS_WHITE>();
        for sq in BitLoop::new(our_king) {
            let moves = bitboards::attacks::<{ PieceType::KING.inner() }>(sq, BB_NONE);
            for to in BitLoop::new(moves & their_pieces) {
                move_list.push::<true>(Move::new(sq, to));
            }
            for to in BitLoop::new(moves & freespace) {
                move_list.push::<false>(Move::new(sq, to));
            }
        }

        // bishops and queens
        let our_diagonal_sliders = self.pieces.bishopqueen::<IS_WHITE>();
        let blockers = self.pieces.occupied();
        for sq in BitLoop::new(our_diagonal_sliders) {
            let moves = bitboards::attacks::<{ PieceType::BISHOP.inner() }>(sq, blockers);
            for to in BitLoop::new(moves & their_pieces) {
                move_list.push::<true>(Move::new(sq, to));
            }
            for to in BitLoop::new(moves & freespace) {
                move_list.push::<false>(Move::new(sq, to));
            }
        }

        // rooks and queens
        let our_orthogonal_sliders = self.pieces.rookqueen::<IS_WHITE>();
        for sq in BitLoop::new(our_orthogonal_sliders) {
            let moves = bitboards::attacks::<{ PieceType::ROOK.inner() }>(sq, blockers);
            for to in BitLoop::new(moves & their_pieces) {
                move_list.push::<true>(Move::new(sq, to));
            }
            for to in BitLoop::new(moves & freespace) {
                move_list.push::<false>(Move::new(sq, to));
            }
        }

        self.generate_castling_moves_for::<IS_WHITE>(move_list);
    }

    pub fn generate_captures<const QS: bool>(&self, move_list: &mut MoveList) {
        debug_assert!(MAGICS_READY.load(std::sync::atomic::Ordering::SeqCst));
        move_list.count = 0; // VERY IMPORTANT FOR UPHOLDING INVARIANTS.
        if self.side == Colour::WHITE {
            self.generate_captures_for::<true, QS>(move_list);
        } else {
            self.generate_captures_for::<false, QS>(move_list);
        }
        debug_assert!(move_list.iter().all(|m| m.is_valid()));
    }

    fn generate_captures_for<const IS_WHITE: bool, const QS: bool>(
        &self,
        move_list: &mut MoveList,
    ) {
        #[cfg(debug_assertions)]
        self.check_validity().unwrap();

        // promotions
        self.generate_forward_promos::<IS_WHITE, QS>(move_list);

        // pawn captures and capture promos
        self.generate_pawn_caps::<IS_WHITE, QS>(move_list);
        self.generate_ep::<IS_WHITE>(move_list);

        // knights
        let our_knights = self.pieces.knights::<IS_WHITE>();
        let their_pieces = self.pieces.their_pieces::<IS_WHITE>();
        for sq in BitLoop::new(our_knights) {
            let moves = bitboards::attacks::<{ PieceType::KNIGHT.inner() }>(sq, BB_NONE);
            for to in BitLoop::new(moves & their_pieces) {
                move_list.push::<true>(Move::new(sq, to));
            }
        }

        // kings
        let our_king = self.pieces.king::<IS_WHITE>();
        for sq in BitLoop::new(our_king) {
            let moves = bitboards::attacks::<{ PieceType::KING.inner() }>(sq, BB_NONE);
            for to in BitLoop::new(moves & their_pieces) {
                move_list.push::<true>(Move::new(sq, to));
            }
        }

        // bishops and queens
        let our_diagonal_sliders = self.pieces.bishopqueen::<IS_WHITE>();
        let blockers = self.pieces.occupied();
        for sq in BitLoop::new(our_diagonal_sliders) {
            let moves = bitboards::attacks::<{ PieceType::BISHOP.inner() }>(sq, blockers);
            for to in BitLoop::new(moves & their_pieces) {
                move_list.push::<true>(Move::new(sq, to));
            }
        }

        // rooks and queens
        let our_orthogonal_sliders = self.pieces.rookqueen::<IS_WHITE>();
        let blockers = self.pieces.occupied();
        for sq in BitLoop::new(our_orthogonal_sliders) {
            let moves = bitboards::attacks::<{ PieceType::ROOK.inner() }>(sq, blockers);
            for to in BitLoop::new(moves & their_pieces) {
                move_list.push::<true>(Move::new(sq, to));
            }
        }
    }

    pub fn generate_castling_moves(&self, move_list: &mut MoveList) {
        debug_assert!(MAGICS_READY.load(std::sync::atomic::Ordering::SeqCst));
        move_list.count = 0; // VERY IMPORTANT FOR UPHOLDING INVARIANTS.
        if self.side == Colour::WHITE {
            self.generate_castling_moves_for::<true>(move_list);
        } else {
            self.generate_castling_moves_for::<false>(move_list);
        }
        debug_assert!(move_list.iter().all(|m| m.is_valid()));
    }

    fn generate_castling_moves_for<const IS_WHITE: bool>(&self, move_list: &mut MoveList) {
        let occupied = self.pieces.occupied();
        if IS_WHITE {
            let king_sq = self.king_sq(Colour::WHITE);
            if self.castle_perm.wk != Square::NO_SQUARE {
                assert_eq!(self.castle_perm.wk, Square::H1);
                let king_dst = Square::G1;
                let king_side_path = ORTHO_RAY_BETWEEN[king_sq.index()][king_dst.index()];
                let king_side_path_to_rook =
                    ORTHO_RAY_BETWEEN[king_sq.index()][self.castle_perm.wk.index()];
                if occupied & (king_side_path | king_side_path_to_rook) == 0
                    && !self.any_attacked(king_side_path | king_sq.bitboard(), Colour::BLACK)
                {
                    move_list.push::<false>(Move::new_with_flags(
                        king_sq,
                        self.castle_perm.wk,
                        Move::CASTLE_FLAG,
                    ));
                }
            }

            if self.castle_perm.wq != Square::NO_SQUARE {
                let king_dst = Square::C1;
                let queen_side_path = ORTHO_RAY_BETWEEN[king_sq.index()][king_dst.index()];
                let queen_side_path_to_rook =
                    ORTHO_RAY_BETWEEN[king_sq.index()][self.castle_perm.wq.index()];
                if occupied & (queen_side_path | queen_side_path_to_rook) == 0
                    && !self.any_attacked(queen_side_path | king_sq.bitboard(), Colour::BLACK)
                {
                    move_list.push::<false>(Move::new_with_flags(
                        king_sq,
                        self.castle_perm.wq,
                        Move::CASTLE_FLAG,
                    ));
                }
            }
        } else {
            let king_sq = self.king_sq(Colour::BLACK);
            if self.castle_perm.bk != Square::NO_SQUARE {
                let king_dst = Square::G8;
                let king_side_path = ORTHO_RAY_BETWEEN[king_sq.index()][king_dst.index()];
                let king_side_path_to_rook =
                    ORTHO_RAY_BETWEEN[king_sq.index()][self.castle_perm.bk.index()];
                if occupied & (king_side_path | king_side_path_to_rook) == 0
                    && !self.any_attacked(king_side_path | king_sq.bitboard(), Colour::WHITE)
                {
                    move_list.push::<false>(Move::new_with_flags(
                        king_sq,
                        self.castle_perm.bk,
                        Move::CASTLE_FLAG,
                    ));
                }
            }

            if self.castle_perm.bq != Square::NO_SQUARE {
                let king_dst = Square::C8;
                let queen_side_path = ORTHO_RAY_BETWEEN[king_sq.index()][king_dst.index()];
                let queen_side_path_to_rook =
                    ORTHO_RAY_BETWEEN[king_sq.index()][self.castle_perm.bq.index()];
                if occupied & (queen_side_path | queen_side_path_to_rook) == 0
                    && !self.any_attacked(queen_side_path | king_sq.bitboard(), Colour::WHITE)
                {
                    move_list.push::<false>(Move::new_with_flags(
                        king_sq,
                        self.castle_perm.bq,
                        Move::CASTLE_FLAG,
                    ));
                }
            }
        }
    }

    pub fn generate_quiets(&self, move_list: &mut MoveList) {
        debug_assert!(MAGICS_READY.load(std::sync::atomic::Ordering::SeqCst));
        // we don't need to clear the move list here because we're only adding to it.
        if self.side == Colour::WHITE {
            self.generate_quiets_for::<true>(move_list);
        } else {
            self.generate_quiets_for::<false>(move_list);
        }
        debug_assert!(move_list.iter().all(|m| m.is_valid()));
    }

    fn generate_pawn_quiet<const IS_WHITE: bool>(&self, move_list: &mut MoveList) {
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
            move_list.push::<false>(Move::new(sq, to));
        }
        for sq in BitLoop::new(double_pushable_pawns) {
            let to = if IS_WHITE { sq.add(16) } else { sq.sub(16) };
            move_list.push::<false>(Move::new(sq, to));
        }
    }

    fn generate_quiets_for<const IS_WHITE: bool>(&self, move_list: &mut MoveList) {
        // pawns
        self.generate_pawn_quiet::<IS_WHITE>(move_list);

        // knights
        let our_knights = self.pieces.knights::<IS_WHITE>();
        let blockers = self.pieces.occupied();
        for sq in BitLoop::new(our_knights) {
            let moves = bitboards::attacks::<{ PieceType::KNIGHT.inner() }>(sq, BB_NONE);
            for to in BitLoop::new(moves & !blockers) {
                move_list.push::<false>(Move::new(sq, to));
            }
        }

        // kings
        let our_king = self.pieces.king::<IS_WHITE>();
        let blockers = self.pieces.occupied();
        for sq in BitLoop::new(our_king) {
            let moves = bitboards::attacks::<{ PieceType::KING.inner() }>(sq, BB_NONE);
            for to in BitLoop::new(moves & !blockers) {
                move_list.push::<false>(Move::new(sq, to));
            }
        }

        // bishops and queens
        let our_diagonal_sliders = self.pieces.bishopqueen::<IS_WHITE>();
        let blockers = self.pieces.occupied();
        for sq in BitLoop::new(our_diagonal_sliders) {
            let moves = bitboards::attacks::<{ PieceType::BISHOP.inner() }>(sq, blockers);
            for to in BitLoop::new(moves & !blockers) {
                move_list.push::<false>(Move::new(sq, to));
            }
        }

        // rooks and queens
        let our_orthogonal_sliders = self.pieces.rookqueen::<IS_WHITE>();
        let blockers = self.pieces.occupied();
        for sq in BitLoop::new(our_orthogonal_sliders) {
            let moves = bitboards::attacks::<{ PieceType::ROOK.inner() }>(sq, blockers);
            for to in BitLoop::new(moves & !blockers) {
                move_list.push::<false>(Move::new(sq, to));
            }
        }

        // castling
        self.generate_castling_moves_for::<IS_WHITE>(move_list);
    }
}

mod tests {
    #[test]
    fn staged_matches_full() {
        use super::*;
        use crate::bench;
        fn synced_perft(pos: &mut Board, depth: usize) -> u64 {
            pos.check_validity().unwrap();

            if depth == 0 {
                return 1;
            }

            let mut ml = MoveList::new();
            pos.generate_moves(&mut ml);
            let mut ml_staged = MoveList::new();
            pos.generate_captures::<false>(&mut ml_staged);
            pos.generate_quiets(&mut ml_staged);

            let mut full_moves_vec = ml.as_slice().to_vec();
            let mut staged_moves_vec = ml_staged.as_slice().to_vec();
            full_moves_vec.sort_unstable_by_key(|m| m.mov);
            staged_moves_vec.sort_unstable_by_key(|m| m.mov);
            let eq = full_moves_vec == staged_moves_vec;
            assert!(eq, "full and staged move lists differ in {}, \nfull list: \n[{}], \nstaged list: \n[{}]", pos.fen(), {
                let mut mvs = Vec::new();
                for m in full_moves_vec {
                    mvs.push(format!("{}{}", pos.san(m.mov).unwrap(), if m.score == MoveListEntry::TACTICAL_SENTINEL { "T" } else { "Q" }));
                }
                mvs.join(", ")
            }, {
                let mut mvs = Vec::new();
                for m in staged_moves_vec {
                    mvs.push(format!("{}{}", pos.san(m.mov).unwrap(), if m.score == MoveListEntry::TACTICAL_SENTINEL { "T" } else { "Q" }));
                }
                mvs.join(", ")
            });

            let mut count = 0;
            for &m in ml.iter() {
                if !pos.make_move_base(m) {
                    continue;
                }
                count += synced_perft(pos, depth - 1);
                pos.unmake_move_base();
            }

            count
        }
        crate::magic::initialise();

        let mut pos = Board::default();

        for fen in bench::BENCH_POSITIONS {
            pos.set_from_fen(fen).unwrap();
            synced_perft(&mut pos, 2);
        }
    }
}
