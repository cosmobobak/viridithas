pub mod bitboards;
pub mod movepicker;

use arrayvec::ArrayVec;

pub use self::bitboards::BitLoop;

use super::Board;

use std::{
    fmt::{Display, Formatter},
    ops::{Deref, DerefMut},
    sync::atomic::Ordering,
};

use crate::{
    chessmove::Move,
    piece::{Black, Col, Colour, PieceType, White},
    squareset::SquareSet,
    uci::CHESS960,
    util::{Square, RAY_BETWEEN},
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
    // moves: [MoveListEntry; MAX_POSITION_MOVES],
    // count: usize,
    inner: ArrayVec<MoveListEntry, MAX_POSITION_MOVES>,
}

impl MoveList {
    pub fn new() -> Self {
        Self { inner: ArrayVec::new() }
    }

    fn push<const TACTICAL: bool>(&mut self, m: Move) {
        // debug_assert!(self.count < MAX_POSITION_MOVES, "overflowed {self}");
        let score =
            if TACTICAL { MoveListEntry::TACTICAL_SENTINEL } else { MoveListEntry::QUIET_SENTINEL };

        self.inner.push(MoveListEntry { mov: m, score });
    }

    pub fn iter_moves(&self) -> impl Iterator<Item = &Move> {
        self.inner.iter().map(|e| &e.mov)
    }

    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

impl Deref for MoveList {
    type Target = [MoveListEntry];

    fn deref(&self) -> &[MoveListEntry] {
        &self.inner
    }
}

impl DerefMut for MoveList {
    fn deref_mut(&mut self) -> &mut [MoveListEntry] {
        &mut self.inner
    }
}

impl Display for MoveList {
    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
        if self.inner.is_empty() {
            return write!(f, "MoveList: (0) []");
        }
        writeln!(f, "MoveList: ({}) [", self.inner.len())?;
        for m in &self.inner[0..self.inner.len() - 1] {
            writeln!(f, "  {} ${}, ", m.mov, m.score)?;
        }
        writeln!(
            f,
            "  {} ${}",
            self.inner[self.inner.len() - 1].mov,
            self.inner[self.inner.len() - 1].score
        )?;
        write!(f, "]")
    }
}

impl Board {
    fn generate_pawn_caps<C: Col, const QS: bool>(
        &self,
        move_list: &mut MoveList,
        valid_target_squares: SquareSet,
    ) {
        let our_pawns = self.pieces.pawns::<C>();
        let their_pieces = self.pieces.their_pieces::<C>();
        // to determine which pawns can capture, we shift the opponent's pieces backwards and find the intersection
        let attacking_west = if C::WHITE {
            their_pieces.south_east_one() & our_pawns
        } else {
            their_pieces.north_east_one() & our_pawns
        };
        let attacking_east = if C::WHITE {
            their_pieces.south_west_one() & our_pawns
        } else {
            their_pieces.north_west_one() & our_pawns
        };
        let valid_west = if C::WHITE {
            valid_target_squares.south_east_one()
        } else {
            valid_target_squares.north_east_one()
        };
        let valid_east = if C::WHITE {
            valid_target_squares.south_west_one()
        } else {
            valid_target_squares.north_west_one()
        };
        let promo_rank = if C::WHITE { SquareSet::RANK_7 } else { SquareSet::RANK_2 };
        for from in attacking_west & !promo_rank & valid_west {
            let to = if C::WHITE { from.add(7) } else { from.sub(9) };
            move_list.push::<true>(Move::new(from, to));
        }
        for from in attacking_east & !promo_rank & valid_east {
            let to = if C::WHITE { from.add(9) } else { from.sub(7) };
            move_list.push::<true>(Move::new(from, to));
        }
        if QS {
            // in quiescence search, we only generate promotions to queen.
            for from in attacking_west & promo_rank & valid_west {
                let to = if C::WHITE { from.add(7) } else { from.sub(9) };
                move_list.push::<true>(Move::new_with_promo(from, to, PieceType::QUEEN));
            }
            for from in attacking_east & promo_rank & valid_east {
                let to = if C::WHITE { from.add(9) } else { from.sub(7) };
                move_list.push::<true>(Move::new_with_promo(from, to, PieceType::QUEEN));
            }
        } else {
            for from in attacking_west & promo_rank & valid_west {
                let to = if C::WHITE { from.add(7) } else { from.sub(9) };
                for promo in
                    [PieceType::QUEEN, PieceType::ROOK, PieceType::BISHOP, PieceType::KNIGHT]
                {
                    move_list.push::<true>(Move::new_with_promo(from, to, promo));
                }
            }
            for from in attacking_east & promo_rank & valid_east {
                let to = if C::WHITE { from.add(9) } else { from.sub(7) };
                for promo in
                    [PieceType::QUEEN, PieceType::ROOK, PieceType::BISHOP, PieceType::KNIGHT]
                {
                    move_list.push::<true>(Move::new_with_promo(from, to, promo));
                }
            }
        }
    }

    fn generate_ep<C: Col>(&self, move_list: &mut MoveList) {
        if self.ep_sq == Square::NO_SQUARE {
            return;
        }
        let ep_bb = self.ep_sq.as_set();
        let our_pawns = self.pieces.pawns::<C>();
        let attacks_west = if C::WHITE {
            ep_bb.south_east_one() & our_pawns
        } else {
            ep_bb.north_east_one() & our_pawns
        };
        let attacks_east = if C::WHITE {
            ep_bb.south_west_one() & our_pawns
        } else {
            ep_bb.north_west_one() & our_pawns
        };

        if attacks_west.non_empty() {
            let from_sq = attacks_west.first();
            move_list.push::<true>(Move::new_with_flags(from_sq, self.ep_sq, Move::EP_FLAG));
        }
        if attacks_east.non_empty() {
            let from_sq = attacks_east.first();
            move_list.push::<true>(Move::new_with_flags(from_sq, self.ep_sq, Move::EP_FLAG));
        }
    }

    fn generate_pawn_forward<C: Col>(
        &self,
        move_list: &mut MoveList,
        valid_target_squares: SquareSet,
    ) {
        let start_rank = if C::WHITE { SquareSet::RANK_2 } else { SquareSet::RANK_7 };
        let promo_rank = if C::WHITE { SquareSet::RANK_7 } else { SquareSet::RANK_2 };
        let shifted_empty_squares =
            if C::WHITE { self.pieces.empty() >> 8 } else { self.pieces.empty() << 8 };
        let double_shifted_empty_squares =
            if C::WHITE { self.pieces.empty() >> 16 } else { self.pieces.empty() << 16 };
        let shifted_valid_squares =
            if C::WHITE { valid_target_squares >> 8 } else { valid_target_squares << 8 };
        let double_shifted_valid_squares =
            if C::WHITE { valid_target_squares >> 16 } else { valid_target_squares << 16 };
        let our_pawns = self.pieces.pawns::<C>();
        let pushable_pawns = our_pawns & shifted_empty_squares;
        let double_pushable_pawns = pushable_pawns & double_shifted_empty_squares & start_rank;
        let promoting_pawns = pushable_pawns & promo_rank;
        for sq in pushable_pawns & !promoting_pawns & shifted_valid_squares {
            let to = if C::WHITE { sq.add(8) } else { sq.sub(8) };
            move_list.push::<false>(Move::new(sq, to));
        }
        for sq in double_pushable_pawns & double_shifted_valid_squares {
            let to = if C::WHITE { sq.add(16) } else { sq.sub(16) };
            move_list.push::<false>(Move::new(sq, to));
        }
        for sq in promoting_pawns & shifted_valid_squares {
            let to = if C::WHITE { sq.add(8) } else { sq.sub(8) };
            for promo in [PieceType::QUEEN, PieceType::KNIGHT, PieceType::ROOK, PieceType::BISHOP] {
                move_list.push::<true>(Move::new_with_promo(sq, to, promo));
            }
        }
    }

    fn generate_forward_promos<C: Col, const QS: bool>(
        &self,
        move_list: &mut MoveList,
        valid_target_squares: SquareSet,
    ) {
        let promo_rank = if C::WHITE { SquareSet::RANK_7 } else { SquareSet::RANK_2 };
        let shifted_empty_squares =
            if C::WHITE { self.pieces.empty() >> 8 } else { self.pieces.empty() << 8 };
        let shifted_valid_squares =
            if C::WHITE { valid_target_squares >> 8 } else { valid_target_squares << 8 };
        let our_pawns = self.pieces.pawns::<C>();
        let pushable_pawns = our_pawns & shifted_empty_squares;
        let promoting_pawns = pushable_pawns & promo_rank;
        for sq in promoting_pawns & shifted_valid_squares {
            let to = if C::WHITE { sq.add(8) } else { sq.sub(8) };
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
        move_list.clear();
        if self.side == Colour::WHITE {
            self.generate_moves_for::<White>(move_list);
        } else {
            self.generate_moves_for::<Black>(move_list);
        }
        debug_assert!(move_list.iter_moves().all(|m| m.is_valid()));
    }

    fn generate_moves_for<C: Col>(&self, move_list: &mut MoveList) {
        #[cfg(debug_assertions)]
        self.check_validity().unwrap();

        let their_pieces = self.pieces.their_pieces::<C>();
        let freespace = self.pieces.empty();
        let our_king_sq = self.pieces.king::<C>().first();

        if self.threats.checkers.count() > 1 {
            // we're in double-check, so we can only move the king.
            let moves = bitboards::king_attacks(our_king_sq) & !self.threats.all;
            for to in moves & their_pieces {
                move_list.push::<true>(Move::new(our_king_sq, to));
            }
            for to in moves & freespace {
                move_list.push::<false>(Move::new(our_king_sq, to));
            }
            return;
        }

        let valid_target_squares = if self.in_check() {
            RAY_BETWEEN[our_king_sq.index()][self.threats.checkers.first().index()]
                | self.threats.checkers
        } else {
            SquareSet::FULL
        };

        self.generate_pawn_forward::<C>(move_list, valid_target_squares);
        self.generate_pawn_caps::<C, false>(move_list, valid_target_squares);
        self.generate_ep::<C>(move_list);

        // knights
        let our_knights = self.pieces.knights::<C>();
        for sq in our_knights {
            let moves = bitboards::knight_attacks(sq) & valid_target_squares;
            for to in moves & their_pieces {
                move_list.push::<true>(Move::new(sq, to));
            }
            for to in moves & freespace {
                move_list.push::<false>(Move::new(sq, to));
            }
        }

        // kings
        let moves = bitboards::king_attacks(our_king_sq) & !self.threats.all;
        for to in moves & their_pieces {
            move_list.push::<true>(Move::new(our_king_sq, to));
        }
        for to in moves & freespace {
            move_list.push::<false>(Move::new(our_king_sq, to));
        }

        // bishops and queens
        let our_diagonal_sliders = self.pieces.diags::<C>();
        let blockers = self.pieces.occupied();
        for sq in our_diagonal_sliders {
            let moves = bitboards::bishop_attacks(sq, blockers) & valid_target_squares;
            for to in moves & their_pieces {
                move_list.push::<true>(Move::new(sq, to));
            }
            for to in moves & freespace {
                move_list.push::<false>(Move::new(sq, to));
            }
        }

        // rooks and queens
        let our_orthogonal_sliders = self.pieces.orthos::<C>();
        for sq in our_orthogonal_sliders {
            let moves = bitboards::rook_attacks(sq, blockers) & valid_target_squares;
            for to in moves & their_pieces {
                move_list.push::<true>(Move::new(sq, to));
            }
            for to in moves & freespace {
                move_list.push::<false>(Move::new(sq, to));
            }
        }

        if !self.in_check() {
            self.generate_castling_moves_for::<C>(move_list);
        }
    }

    pub fn generate_captures<const QS: bool>(&self, move_list: &mut MoveList) {
        move_list.clear();
        if self.side == Colour::WHITE {
            self.generate_captures_for::<White, QS>(move_list);
        } else {
            self.generate_captures_for::<Black, QS>(move_list);
        }
        debug_assert!(move_list.iter_moves().all(|m| m.is_valid()));
    }

    fn generate_captures_for<C: Col, const QS: bool>(&self, move_list: &mut MoveList) {
        #[cfg(debug_assertions)]
        self.check_validity().unwrap();

        let their_pieces = self.pieces.their_pieces::<C>();
        let our_king_sq = self.pieces.king::<C>().first();

        if self.threats.checkers.count() > 1 {
            // we're in double-check, so we can only move the king.
            let moves = bitboards::king_attacks(our_king_sq) & !self.threats.all;
            for to in moves & their_pieces {
                move_list.push::<true>(Move::new(our_king_sq, to));
            }
            return;
        }

        let valid_target_squares = if self.in_check() {
            RAY_BETWEEN[our_king_sq.index()][self.threats.checkers.first().index()]
                | self.threats.checkers
        } else {
            SquareSet::FULL
        };

        // promotions
        self.generate_forward_promos::<C, QS>(move_list, valid_target_squares);

        // pawn captures and capture promos
        self.generate_pawn_caps::<C, QS>(move_list, valid_target_squares);
        self.generate_ep::<C>(move_list);

        // knights
        let our_knights = self.pieces.knights::<C>();
        let their_pieces = self.pieces.their_pieces::<C>();
        for sq in our_knights {
            let moves = bitboards::knight_attacks(sq) & valid_target_squares;
            for to in moves & their_pieces {
                move_list.push::<true>(Move::new(sq, to));
            }
        }

        // kings
        let moves = bitboards::king_attacks(our_king_sq) & !self.threats.all;
        for to in moves & their_pieces {
            move_list.push::<true>(Move::new(our_king_sq, to));
        }

        // bishops and queens
        let our_diagonal_sliders = self.pieces.diags::<C>();
        let blockers = self.pieces.occupied();
        for sq in our_diagonal_sliders {
            let moves = bitboards::bishop_attacks(sq, blockers) & valid_target_squares;
            for to in moves & their_pieces {
                move_list.push::<true>(Move::new(sq, to));
            }
        }

        // rooks and queens
        let our_orthogonal_sliders = self.pieces.orthos::<C>();
        for sq in our_orthogonal_sliders {
            let moves = bitboards::rook_attacks(sq, blockers) & valid_target_squares;
            for to in moves & their_pieces {
                move_list.push::<true>(Move::new(sq, to));
            }
        }
    }

    fn generate_castling_moves_for<C: Col>(&self, move_list: &mut MoveList) {
        let occupied = self.pieces.occupied();

        if CHESS960.load(Ordering::Relaxed) {
            let king_sq = self.king_sq(C::COLOUR);
            if self.sq_attacked(king_sq, C::Opposite::COLOUR) {
                return;
            }

            let castling_kingside = self.castle_perm.kingside(C::COLOUR);
            if castling_kingside != Square::NO_SQUARE {
                let king_dst = Square::G1.relative_to(C::COLOUR);
                let rook_dst = Square::F1.relative_to(C::COLOUR);
                self.try_generate_frc_castling::<C>(
                    king_sq,
                    castling_kingside,
                    king_dst,
                    rook_dst,
                    occupied,
                    move_list,
                );
            }

            let castling_queenside = self.castle_perm.queenside(C::COLOUR);
            if castling_queenside != Square::NO_SQUARE {
                let king_dst = Square::C1.relative_to(C::COLOUR);
                let rook_dst = Square::D1.relative_to(C::COLOUR);
                self.try_generate_frc_castling::<C>(
                    king_sq,
                    castling_queenside,
                    king_dst,
                    rook_dst,
                    occupied,
                    move_list,
                );
            }
        } else {
            const WK_FREESPACE: SquareSet = Square::F1.as_set().add_square(Square::G1);
            const WQ_FREESPACE: SquareSet =
                Square::B1.as_set().add_square(Square::C1).add_square(Square::D1);
            const BK_FREESPACE: SquareSet = Square::F8.as_set().add_square(Square::G8);
            const BQ_FREESPACE: SquareSet =
                Square::B8.as_set().add_square(Square::C8).add_square(Square::D8);

            let k_freespace = if C::WHITE { WK_FREESPACE } else { BK_FREESPACE };
            let q_freespace = if C::WHITE { WQ_FREESPACE } else { BQ_FREESPACE };
            let from = Square::E1.relative_to(C::COLOUR);
            let k_to = Square::H1.relative_to(C::COLOUR);
            let q_to = Square::A1.relative_to(C::COLOUR);
            let k_thru = Square::F1.relative_to(C::COLOUR);
            let q_thru = Square::D1.relative_to(C::COLOUR);
            let k_perm = self.castle_perm.kingside(C::COLOUR);
            let q_perm = self.castle_perm.queenside(C::COLOUR);

            // stupid hack to avoid redoing or eagerly doing hard work.
            let mut cache = None;

            if k_perm != Square::NO_SQUARE
                && (occupied & k_freespace).is_empty()
                && {
                    let got_attacked_king = self.sq_attacked_by::<C::Opposite>(from);
                    cache = Some(got_attacked_king);
                    !got_attacked_king
                }
                && !self.sq_attacked_by::<C::Opposite>(k_thru)
            {
                move_list.push::<false>(Move::new_with_flags(from, k_to, Move::CASTLE_FLAG));
            }

            if q_perm != Square::NO_SQUARE
                && (occupied & q_freespace).is_empty()
                && !cache.unwrap_or_else(|| self.sq_attacked_by::<C::Opposite>(from))
                && !self.sq_attacked_by::<C::Opposite>(q_thru)
            {
                move_list.push::<false>(Move::new_with_flags(from, q_to, Move::CASTLE_FLAG));
            }
        }
    }

    fn try_generate_frc_castling<C: Col>(
        &self,
        king_sq: Square,
        castling_sq: Square,
        king_dst: Square,
        rook_dst: Square,
        occupied: SquareSet,
        move_list: &mut MoveList,
    ) {
        let king_path = RAY_BETWEEN[king_sq.index()][king_dst.index()];
        let rook_path = RAY_BETWEEN[king_sq.index()][castling_sq.index()];
        let relevant_occupied = occupied ^ king_sq.as_set() ^ castling_sq.as_set();
        if (relevant_occupied & (king_path | rook_path | king_dst.as_set() | rook_dst.as_set()))
            .is_empty()
            && !self.any_attacked(king_path, C::Opposite::COLOUR)
        {
            move_list.push::<false>(Move::new_with_flags(king_sq, castling_sq, Move::CASTLE_FLAG));
        }
    }

    pub fn generate_quiets(&self, move_list: &mut MoveList) {
        // we don't need to clear the move list here because we're only adding to it.
        if self.side == Colour::WHITE {
            self.generate_quiets_for::<White>(move_list);
        } else {
            self.generate_quiets_for::<Black>(move_list);
        }
        debug_assert!(move_list.iter_moves().all(|m| m.is_valid()));
    }

    fn generate_pawn_quiet<C: Col>(
        &self,
        move_list: &mut MoveList,
        valid_target_squares: SquareSet,
    ) {
        let start_rank = if C::WHITE { SquareSet::RANK_2 } else { SquareSet::RANK_7 };
        let promo_rank = if C::WHITE { SquareSet::RANK_7 } else { SquareSet::RANK_2 };
        let shifted_empty_squares =
            if C::WHITE { self.pieces.empty() >> 8 } else { self.pieces.empty() << 8 };
        let double_shifted_empty_squares =
            if C::WHITE { self.pieces.empty() >> 16 } else { self.pieces.empty() << 16 };
        let shifted_valid_squares =
            if C::WHITE { valid_target_squares >> 8 } else { valid_target_squares << 8 };
        let double_shifted_valid_squares =
            if C::WHITE { valid_target_squares >> 16 } else { valid_target_squares << 16 };
        let our_pawns = self.pieces.pawns::<C>();
        let pushable_pawns = our_pawns & shifted_empty_squares;
        let double_pushable_pawns = pushable_pawns & double_shifted_empty_squares & start_rank;
        let promoting_pawns = pushable_pawns & promo_rank;
        for sq in pushable_pawns & !promoting_pawns & shifted_valid_squares {
            let to = if C::WHITE { sq.add(8) } else { sq.sub(8) };
            move_list.push::<false>(Move::new(sq, to));
        }
        for sq in double_pushable_pawns & double_shifted_valid_squares {
            let to = if C::WHITE { sq.add(16) } else { sq.sub(16) };
            move_list.push::<false>(Move::new(sq, to));
        }
    }

    fn generate_quiets_for<C: Col>(&self, move_list: &mut MoveList) {
        let freespace = self.pieces.empty();
        let our_king_sq = self.pieces.king::<C>().first();
        let blockers = self.pieces.occupied();

        if self.threats.checkers.count() > 1 {
            // we're in double-check, so we can only move the king.
            let moves = bitboards::king_attacks(our_king_sq) & !self.threats.all;
            for to in moves & freespace {
                move_list.push::<false>(Move::new(our_king_sq, to));
            }
            return;
        }

        let valid_target_squares = if self.in_check() {
            RAY_BETWEEN[our_king_sq.index()][self.threats.checkers.first().index()]
                | self.threats.checkers
        } else {
            SquareSet::FULL
        };

        // pawns
        self.generate_pawn_quiet::<C>(move_list, valid_target_squares);

        // knights
        let our_knights = self.pieces.knights::<C>();
        for sq in our_knights {
            let moves = bitboards::knight_attacks(sq) & valid_target_squares;
            for to in moves & !blockers {
                move_list.push::<false>(Move::new(sq, to));
            }
        }

        // kings
        let moves = bitboards::king_attacks(our_king_sq) & !self.threats.all;
        for to in moves & !blockers {
            move_list.push::<false>(Move::new(our_king_sq, to));
        }

        // bishops and queens
        let our_diagonal_sliders = self.pieces.diags::<C>();
        for sq in our_diagonal_sliders {
            let moves = bitboards::bishop_attacks(sq, blockers) & valid_target_squares;
            for to in moves & !blockers {
                move_list.push::<false>(Move::new(sq, to));
            }
        }

        // rooks and queens
        let our_orthogonal_sliders = self.pieces.orthos::<C>();
        for sq in our_orthogonal_sliders {
            let moves = bitboards::rook_attacks(sq, blockers) & valid_target_squares;
            for to in moves & !blockers {
                move_list.push::<false>(Move::new(sq, to));
            }
        }

        // castling
        if !self.in_check() {
            self.generate_castling_moves_for::<C>(move_list);
        }
    }
}

#[cfg(test)]
pub fn synced_perft(pos: &mut Board, depth: usize) -> u64 {
    #[cfg(debug_assertions)]
    pos.check_validity().unwrap();

    if depth == 0 {
        return 1;
    }

    let mut ml = MoveList::new();
    pos.generate_moves(&mut ml);
    let mut ml_staged = MoveList::new();
    pos.generate_captures::<false>(&mut ml_staged);
    pos.generate_quiets(&mut ml_staged);

    let mut full_moves_vec = ml.to_vec();
    let mut staged_moves_vec = ml_staged.to_vec();
    full_moves_vec.sort_unstable_by_key(|m| m.mov);
    staged_moves_vec.sort_unstable_by_key(|m| m.mov);
    let eq = full_moves_vec == staged_moves_vec;
    assert!(
        eq,
        "full and staged move lists differ in {}, \nfull list: \n[{}], \nstaged list: \n[{}]",
        pos.fen(),
        {
            let mut mvs = Vec::new();
            for m in full_moves_vec {
                mvs.push(format!(
                    "{}{}",
                    pos.san(m.mov).unwrap(),
                    if m.score == MoveListEntry::TACTICAL_SENTINEL { "T" } else { "Q" }
                ));
            }
            mvs.join(", ")
        },
        {
            let mut mvs = Vec::new();
            for m in staged_moves_vec {
                mvs.push(format!(
                    "{}{}",
                    pos.san(m.mov).unwrap(),
                    if m.score == MoveListEntry::TACTICAL_SENTINEL { "T" } else { "Q" }
                ));
            }
            mvs.join(", ")
        }
    );

    let mut count = 0;
    for &m in ml.iter_moves() {
        if !pos.make_move_simple(m) {
            continue;
        }
        count += synced_perft(pos, depth - 1);
        pos.unmake_move_base();
    }

    count
}

mod tests {
    #[test]
    fn staged_matches_full() {
        use super::*;
        use crate::bench;

        let mut pos = Board::default();

        for fen in bench::BENCH_POSITIONS {
            pos.set_from_fen(fen).unwrap();
            synced_perft(&mut pos, 2);
        }
    }
}
