pub mod bitboards;

use self::bitboards::{
    lsb, north_east_one, north_west_one, south_east_one, south_west_one, BB_RANK_2, BB_RANK_7,
};
pub use self::bitboards::{BitLoop, BB_NONE};

use super::Board;

use std::{
    fmt::{Display, Formatter},
    ops::Index,
};

use crate::{
    chessmove::Move,
    definitions::{
        Square::{B1, B8, C1, C8, D1, D8, E1, E8, F1, F8, G1, G8, NO_SQUARE},
        BB, BISHOP, BKCA, BLACK, BN, BQ, BQCA, BR, KING, KNIGHT, PIECE_EMPTY, ROOK, WB, WHITE,
        WKCA, WN, WQ, WQCA, WR,
    },
    lookups::MVV_LVA_SCORE,
    macros,
    validate::{piece_valid, square_on_board}, magic::MAGICS_READY,
};

pub const TT_MOVE_SCORE: i32 = 20_000_000;
const FIRST_ORDER_KILLER_SCORE: i32 = 9_000_000;
const SECOND_ORDER_KILLER_SCORE: i32 = 8_000_000;
const COUNTER_MOVE_SCORE: i32 = 2_000_000;
const THIRD_ORDER_KILLER_SCORE: i32 = 1_000_000;

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

pub struct MoveListIter {
    moves: [MoveListEntry; MAX_POSITION_MOVES],
    count: usize,
    index: usize,
}

impl MoveList {
    pub const fn new() -> Self {
        const DEFAULT: MoveListEntry = MoveListEntry {
            entry: Move { data: 0 },
            score: 0,
        };
        Self {
            moves: [DEFAULT; MAX_POSITION_MOVES],
            count: 0,
        }
    }

    pub fn lookup_by_move(&mut self, m: Move) -> Option<&mut MoveListEntry> {
        unsafe {
            self.moves
                .get_unchecked_mut(..self.count)
                .iter_mut()
                .find(|e| e.entry == m)
        }
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
}

impl IntoIterator for MoveList {
    type Item = Move;
    type IntoIter = MoveListIter;

    fn into_iter(self) -> Self::IntoIter {
        MoveListIter {
            moves: self.moves,
            count: self.count,
            index: 0,
        }
    }
}

impl Iterator for MoveListIter {
    type Item = Move;

    fn next(&mut self) -> Option<Move> {
        if self.index == self.count {
            return None;
        }
        let mut best_score = 0;
        let mut best_num = self.index;

        for index in self.index..self.count {
            let score = unsafe { self.moves.get_unchecked(index).score };
            if score > best_score {
                best_score = score;
                best_num = index;
            }
        }

        debug_assert!(self.index < self.count);
        debug_assert!(best_num < self.count);
        debug_assert!(best_num >= self.index);

        let m = unsafe { self.moves.get_unchecked(best_num).entry };

        unsafe {
            *self.moves.get_unchecked_mut(best_num) = *self.moves.get_unchecked(self.index);
        }

        self.index += 1;

        Some(m)
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

pub struct MoveVecWrapper(pub Vec<Move>);
impl Display for MoveVecWrapper {
    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
        if self.0.is_empty() {
            return write!(f, "MoveList: (0) []");
        }
        writeln!(f, "MoveList: ({}) [", self.0.len())?;
        for m in &self.0[..self.0.len() - 1] {
            writeln!(f, "  {},", m)?;
        }
        writeln!(f, "  {}", self.0.last().unwrap())?;
        write!(f, "]")
    }
}

impl Board {
    fn add_quiet_move(&self, m: Move, move_list: &mut MoveList) {
        debug_assert!(square_on_board(m.from()));
        debug_assert!(square_on_board(m.to()));

        let killer_entry = self.killer_move_table[self.height];

        let score = if killer_entry[0] == m {
            FIRST_ORDER_KILLER_SCORE
        } else if killer_entry[1] == m {
            SECOND_ORDER_KILLER_SCORE
        } else if self.is_countermove(m) { // move that refuted the previous move
            COUNTER_MOVE_SCORE
        } else if self.is_third_order_killer(m) { // killer from two moves ago
            THIRD_ORDER_KILLER_SCORE
        } else {
            let history = self.history_score(m);
            let followup_history = self.followup_history_score(m);
            history + 2 * followup_history
        };

        move_list.push(m, score);
    }

    fn add_capture_move(&self, m: Move, move_list: &mut MoveList) {
        debug_assert!(square_on_board(m.from()));
        debug_assert!(square_on_board(m.to()));
        debug_assert!(piece_valid(m.capture()), "piece: {}", m);

        let capture = m.capture() as usize;
        let piece_moved = self.piece_at(m.from()) as usize;
        let mmvlva = unsafe {
            *MVV_LVA_SCORE
                .get_unchecked(capture)
                .get_unchecked(piece_moved)
        };

        let score = mmvlva + 10_000_000;
        move_list.push(m, score);
    }

    fn add_ep_move(m: Move, move_list: &mut MoveList) {
        move_list.push(m, 1050 + 10_000_000);
    }

    #[allow(clippy::cognitive_complexity)]
    fn generate_pawn_caps<const SIDE: u8>(&self, move_list: &mut MoveList) {
        let our_pawns = if SIDE == WHITE {
            self.pieces.pawns::<true>()
        } else {
            self.pieces.pawns::<false>()
        };
        let their_pieces = if SIDE == WHITE {
            self.pieces.their_pieces::<true>()
        } else {
            self.pieces.their_pieces::<false>()
        };
        // to determine which pawns can capture, we shift the opponent's pieces backwards and find the intersection
        let attacks_west = if SIDE == WHITE {
            south_east_one(their_pieces) & our_pawns
        } else {
            north_east_one(their_pieces) & our_pawns
        };
        let attacks_east = if SIDE == WHITE {
            south_west_one(their_pieces) & our_pawns
        } else {
            north_west_one(their_pieces) & our_pawns
        };
        let promo_rank = if SIDE == WHITE { BB_RANK_7 } else { BB_RANK_2 };
        for from in BitLoop::new(attacks_west & !promo_rank) {
            let to = if SIDE == WHITE { from + 7 } else { from - 9 };
            let cap = self.piece_at(to);
            debug_assert!(piece_valid(cap));
            self.add_capture_move(Move::new(from, to, cap, PIECE_EMPTY, 0), move_list);
        }
        for from in BitLoop::new(attacks_east & !promo_rank) {
            let to = if SIDE == WHITE { from + 9 } else { from - 7 };
            let cap = self.piece_at(to);
            debug_assert!(piece_valid(cap));
            self.add_capture_move(Move::new(from, to, cap, PIECE_EMPTY, 0), move_list);
        }
        for from in BitLoop::new(attacks_west & promo_rank) {
            let to = if SIDE == WHITE { from + 7 } else { from - 9 };
            let cap = self.piece_at(to);
            debug_assert!(piece_valid(cap));
            if SIDE == WHITE {
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
            let to = if SIDE == WHITE { from + 9 } else { from - 7 };
            let cap = self.piece_at(to);
            debug_assert!(piece_valid(cap));
            if SIDE == WHITE {
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

    fn generate_ep<const SIDE: u8>(&self, move_list: &mut MoveList) {
        #![allow(clippy::cast_possible_truncation)]
        if self.ep_sq == NO_SQUARE {
            return;
        }
        let ep_bb = 1 << self.ep_sq;
        let our_pawns = if SIDE == WHITE {
            self.pieces.pawns::<true>()
        } else {
            self.pieces.pawns::<false>()
        };
        let attacks_west = if SIDE == WHITE {
            south_east_one(ep_bb) & our_pawns
        } else {
            north_east_one(ep_bb) & our_pawns
        };
        let attacks_east = if SIDE == WHITE {
            south_west_one(ep_bb) & our_pawns
        } else {
            north_west_one(ep_bb) & our_pawns
        };

        if attacks_west != 0 {
            let from_sq = lsb(attacks_west) as u8;
            Self::add_ep_move(
                Move::new(from_sq, self.ep_sq, PIECE_EMPTY, PIECE_EMPTY, Move::EP_MASK),
                move_list,
            );
        }
        if attacks_east != 0 {
            let from_sq = lsb(attacks_east) as u8;
            Self::add_ep_move(
                Move::new(from_sq, self.ep_sq, PIECE_EMPTY, PIECE_EMPTY, Move::EP_MASK),
                move_list,
            );
        }
    }

    fn generate_pawn_forward<const SIDE: u8>(&self, move_list: &mut MoveList) {
        let start_rank = if SIDE == WHITE { BB_RANK_2 } else { BB_RANK_7 };
        let promo_rank = if SIDE == WHITE { BB_RANK_7 } else { BB_RANK_2 };
        let shifted_empty_squares = if SIDE == WHITE {
            self.pieces.empty() >> 8
        } else {
            self.pieces.empty() << 8
        };
        let double_shifted_empty_squares = if SIDE == WHITE {
            self.pieces.empty() >> 16
        } else {
            self.pieces.empty() << 16
        };
        let our_pawns = if SIDE == WHITE {
            self.pieces.pawns::<true>()
        } else {
            self.pieces.pawns::<false>()
        };
        let pushable_pawns = our_pawns & shifted_empty_squares;
        let double_pushable_pawns = pushable_pawns & double_shifted_empty_squares & start_rank;
        let promoting_pawns = pushable_pawns & promo_rank;
        for sq in BitLoop::new(pushable_pawns & !promoting_pawns) {
            let to = if SIDE == WHITE { sq + 8 } else { sq - 8 };
            self.add_quiet_move(Move::new(sq, to, PIECE_EMPTY, PIECE_EMPTY, 0), move_list);
        }
        for sq in BitLoop::new(double_pushable_pawns) {
            let to = if SIDE == WHITE { sq + 16 } else { sq - 16 };
            self.add_quiet_move(
                Move::new(sq, to, PIECE_EMPTY, PIECE_EMPTY, Move::PAWN_START_MASK),
                move_list,
            );
        }
        for sq in BitLoop::new(promoting_pawns) {
            let to = if SIDE == WHITE { sq + 8 } else { sq - 8 };
            if SIDE == WHITE {
                for &promo in &[WQ, WN, WR, WB] {
                    self.add_quiet_move(Move::new(sq, to, PIECE_EMPTY, promo, 0), move_list);
                }
            } else {
                for &promo in &[BQ, BN, BR, BB] {
                    self.add_quiet_move(Move::new(sq, to, PIECE_EMPTY, promo, 0), move_list);
                }
            }
        }
    }

    pub fn generate_moves(&self, move_list: &mut MoveList) {
        debug_assert!(self.movegen_ready);
        debug_assert!(MAGICS_READY.load(std::sync::atomic::Ordering::SeqCst));
        if self.side == WHITE {
            self.generate_moves_for::<WHITE>(move_list);
        } else {
            self.generate_moves_for::<BLACK>(move_list);
        }
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    #[inline(never)]
    pub fn generate_moves_for<const SIDE: u8>(&self, move_list: &mut MoveList) {
        #[cfg(debug_assertions)]
        self.check_validity().unwrap();

        if SIDE != WHITE && SIDE != BLACK {
            unsafe {
                macros::inconceivable!();
            }
        }

        if SIDE == WHITE {
            self.generate_pawn_forward::<WHITE>(move_list);
            self.generate_pawn_caps::<WHITE>(move_list);
            self.generate_ep::<WHITE>(move_list);
        } else {
            self.generate_pawn_forward::<BLACK>(move_list);
            self.generate_pawn_caps::<BLACK>(move_list);
            self.generate_ep::<BLACK>(move_list);
        }

        // knights
        let our_knights = if SIDE == WHITE {
            self.pieces.knights::<true>()
        } else {
            self.pieces.knights::<false>()
        };
        let their_pieces = if SIDE == WHITE {
            self.pieces.their_pieces::<true>()
        } else {
            self.pieces.their_pieces::<false>()
        };
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
        let our_king = if SIDE == WHITE {
            self.pieces.king::<true>()
        } else {
            self.pieces.king::<false>()
        };
        let their_pieces = if SIDE == WHITE {
            self.pieces.their_pieces::<true>()
        } else {
            self.pieces.their_pieces::<false>()
        };
        let freespace = self.pieces.empty();
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
        let our_diagonal_sliders = if SIDE == WHITE {
            self.pieces.bishopqueen::<true>()
        } else {
            self.pieces.bishopqueen::<false>()
        };
        let their_pieces = if SIDE == WHITE {
            self.pieces.their_pieces::<true>()
        } else {
            self.pieces.their_pieces::<false>()
        };
        let freespace = self.pieces.empty();
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
        let our_orthogonal_sliders = if SIDE == WHITE {
            self.pieces.rookqueen::<true>()
        } else {
            self.pieces.rookqueen::<false>()
        };
        let their_pieces = if SIDE == WHITE {
            self.pieces.their_pieces::<true>()
        } else {
            self.pieces.their_pieces::<false>()
        };
        let freespace = self.pieces.empty();
        let blockers = self.pieces.occupied();
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

        self.generate_castling_moves::<SIDE>(move_list);
    }

    pub fn generate_captures(&self, move_list: &mut MoveList) {
        if self.side == WHITE {
            self.generate_captures_comptime::<WHITE>(move_list);
        } else {
            self.generate_captures_comptime::<BLACK>(move_list);
        }
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    pub fn generate_captures_comptime<const SIDE: u8>(&self, move_list: &mut MoveList) {
        #[cfg(debug_assertions)]
        self.check_validity().unwrap();

        if SIDE != WHITE && SIDE != BLACK {
            unsafe {
                macros::inconceivable!();
            }
        }

        // both pawn moves and captures
        if SIDE == WHITE {
            self.generate_pawn_caps::<WHITE>(move_list);
            self.generate_ep::<WHITE>(move_list);
        } else {
            self.generate_pawn_caps::<BLACK>(move_list);
            self.generate_ep::<BLACK>(move_list);
        }

        // knights
        let our_knights = if SIDE == WHITE {
            self.pieces.knights::<true>()
        } else {
            self.pieces.knights::<false>()
        };
        let their_pieces = if SIDE == WHITE {
            self.pieces.their_pieces::<true>()
        } else {
            self.pieces.their_pieces::<false>()
        };
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
        let our_king = if SIDE == WHITE {
            self.pieces.king::<true>()
        } else {
            self.pieces.king::<false>()
        };
        let their_pieces = if SIDE == WHITE {
            self.pieces.their_pieces::<true>()
        } else {
            self.pieces.their_pieces::<false>()
        };
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
        let our_diagonal_sliders = if SIDE == WHITE {
            self.pieces.bishopqueen::<true>()
        } else {
            self.pieces.bishopqueen::<false>()
        };
        let their_pieces = if SIDE == WHITE {
            self.pieces.their_pieces::<true>()
        } else {
            self.pieces.their_pieces::<false>()
        };
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
        let our_orthogonal_sliders = if SIDE == WHITE {
            self.pieces.rookqueen::<true>()
        } else {
            self.pieces.rookqueen::<false>()
        };
        let their_pieces = if SIDE == WHITE {
            self.pieces.their_pieces::<true>()
        } else {
            self.pieces.their_pieces::<false>()
        };
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

    fn generate_castling_moves<const SIDE: u8>(&self, move_list: &mut MoveList) {
        if SIDE == WHITE {
            if (self.castle_perm & WKCA) != 0
                && self.piece_at(F1) == PIECE_EMPTY
                && self.piece_at(G1) == PIECE_EMPTY
                && !self.sq_attacked(E1, BLACK)
                && !self.sq_attacked(F1, BLACK)
            {
                self.add_quiet_move(
                    Move::new(E1, G1, PIECE_EMPTY, PIECE_EMPTY, Move::CASTLE_MASK),
                    move_list,
                );
            }

            if (self.castle_perm & WQCA) != 0
                && self.piece_at(D1) == PIECE_EMPTY
                && self.piece_at(C1) == PIECE_EMPTY
                && self.piece_at(B1) == PIECE_EMPTY
                && !self.sq_attacked(E1, BLACK)
                && !self.sq_attacked(D1, BLACK)
            {
                self.add_quiet_move(
                    Move::new(E1, C1, PIECE_EMPTY, PIECE_EMPTY, Move::CASTLE_MASK),
                    move_list,
                );
            }
        } else {
            if (self.castle_perm & BKCA) != 0
                && self.piece_at(F8) == PIECE_EMPTY
                && self.piece_at(G8) == PIECE_EMPTY
                && !self.sq_attacked(E8, WHITE)
                && !self.sq_attacked(F8, WHITE)
            {
                self.add_quiet_move(
                    Move::new(E8, G8, PIECE_EMPTY, PIECE_EMPTY, Move::CASTLE_MASK),
                    move_list,
                );
            }

            if (self.castle_perm & BQCA) != 0
                && self.piece_at(D8) == PIECE_EMPTY
                && self.piece_at(C8) == PIECE_EMPTY
                && self.piece_at(B8) == PIECE_EMPTY
                && !self.sq_attacked(E8, WHITE)
                && !self.sq_attacked(D8, WHITE)
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
            let west_attacks = north_west_one(our_pawns);
            let east_attacks = north_east_one(our_pawns);
            let pawn_attacks = west_attacks | east_attacks;
            attackers |= pawn_attacks;
        } else {
            let our_pawns = self.pieces.pawns::<false>();
            let west_attacks = south_west_one(our_pawns);
            let east_attacks = south_east_one(our_pawns);
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

        let our_king = if side == WHITE {
            self.pieces.king::<true>()
        } else {
            self.pieces.king::<false>()
        };

        let king_attacks = bitboards::attacks::<KING>(sq, BB_NONE) & our_king;
        attackers |= king_attacks;

        attackers
    }
}
