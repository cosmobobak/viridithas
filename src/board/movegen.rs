use std::{
    fmt::{Display, Formatter},
    ops::Index,
};

use crate::{
    attack::{
        BLACK_JUMPERS, BLACK_SLIDERS, B_DIR, K_DIRS, N_DIRS, Q_DIR, R_DIR, WHITE_JUMPERS,
        WHITE_SLIDERS,
    },
    chessmove::Move,
    definitions::{
        Castling, Rank, Square120, BB, BLACK, BN, BP, BQ, BR, FIRST_ORDER_KILLER_SCORE, NO_SQUARE,
        PIECE_EMPTY, SECOND_ORDER_KILLER_SCORE, WB, WHITE, WN, WP, WQ, WR,
    },
    lookups::{FILES_BOARD, MVV_LVA_SCORE, PIECE_COL, RANKS_BOARD},
    validate::{piece_valid, piece_valid_empty, square_on_board},
};

use super::Board;

pub trait MoveConsumer {
    const DO_PAWN_MOVEGEN: bool;
    fn push(&mut self, m: Move, score: i32);
}

const MAX_POSITION_MOVES: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoveListEntry {
    pub entry: Move,
    pub score: i32,
}

pub struct MoveList {
    moves: [MoveListEntry; MAX_POSITION_MOVES],
    count: usize,
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

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &Move> {
        unsafe {
            self.moves
                .get_unchecked(..self.count)
                .iter()
                .map(|e| &e.entry)
        }
    }

    #[inline]
    pub fn sort(&mut self) {
        // reversed, as we want to sort from highest to lowest
        unsafe {
            self.moves
                .get_unchecked_mut(..self.count)
                .sort_unstable_by(|a, b| b.score.cmp(&a.score));
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
}

impl MoveConsumer for MoveList {
    const DO_PAWN_MOVEGEN: bool = true;

    #[inline]
    fn push(&mut self, m: Move, score: i32) {
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

#[inline]
pub fn offset_square_offboard(offset_sq: i8) -> bool {
    debug_assert!((0..120).contains(&offset_sq));
    let idx: usize = unsafe { offset_sq.try_into().unwrap_unchecked() };
    let value = unsafe { *FILES_BOARD.get_unchecked(idx) };
    value == Square120::OffBoard as u8
}

impl Board {
    #[inline]
    fn add_quiet_move(&self, m: Move, move_list: &mut impl MoveConsumer) {
        let _ = self;
        debug_assert!(square_on_board(m.from()));
        debug_assert!(square_on_board(m.to()));

        let killer_entry = unsafe { self.killer_move_table.get_unchecked(self.ply) };

        let score = if killer_entry[0] == m {
            FIRST_ORDER_KILLER_SCORE
        } else if killer_entry[1] == m {
            SECOND_ORDER_KILLER_SCORE
        } else {
            let from = m.from() as usize;
            let to = m.to() as usize;
            let piece_moved = unsafe { *self.pieces.get_unchecked(from) as usize };
            unsafe {
                *self
                    .history_table
                    .get_unchecked(piece_moved)
                    .get_unchecked(to)
            }
        };

        move_list.push(m, score);
    }

    #[inline]
    fn add_capture_move(&self, m: Move, move_list: &mut impl MoveConsumer) {
        debug_assert!(square_on_board(m.from()));
        debug_assert!(square_on_board(m.to()));
        debug_assert!(piece_valid(m.capture()));

        let capture = m.capture() as usize;
        let from = m.from() as usize;
        let piece_moved = unsafe { *self.pieces.get_unchecked(from) as usize };
        let mmvlva = unsafe {
            *MVV_LVA_SCORE
                .get_unchecked(capture)
                .get_unchecked(piece_moved)
        };

        let score = mmvlva + 10_000_000;
        move_list.push(m, score);
    }

    fn add_ep_move(&self, m: Move, move_list: &mut impl MoveConsumer) {
        let _ = self;
        move_list.push(m, 1050 + 10_000_000);
    }

    fn add_pawn_cap_move<const SIDE: u8, MC: MoveConsumer>(
        &self,
        from: u8,
        to: u8,
        cap: u8,
        move_list: &mut MC,
    ) {
        debug_assert!(piece_valid_empty(cap));
        debug_assert!(square_on_board(from));
        debug_assert!(square_on_board(to));
        let promo_rank = if SIDE == WHITE {
            Rank::Rank7 as u8
        } else {
            Rank::Rank2 as u8
        };
        if RANKS_BOARD[from as usize] == promo_rank {
            if SIDE == WHITE {
                for &promo in &[WQ, WN, WR, WB] {
                    self.add_capture_move(Move::new(from, to, cap, promo, 0), move_list);
                }
            } else {
                for &promo in &[BQ, BN, BR, BB] {
                    self.add_capture_move(Move::new(from, to, cap, promo, 0), move_list);
                }
            };
        } else {
            self.add_capture_move(Move::new(from, to, cap, PIECE_EMPTY, 0), move_list);
        }
    }

    fn add_pawn_move<const SIDE: u8, MC: MoveConsumer>(
        &self,
        from: u8,
        to: u8,
        move_list: &mut MC,
    ) {
        debug_assert!(square_on_board(from));
        debug_assert!(square_on_board(to));
        let promo_rank = if SIDE == WHITE {
            Rank::Rank7 as u8
        } else {
            Rank::Rank2 as u8
        };
        if RANKS_BOARD[from as usize] == promo_rank {
            if SIDE == WHITE {
                for &promo in &[WQ, WN, WR, WB] {
                    self.add_quiet_move(Move::new(from, to, PIECE_EMPTY, promo, 0), move_list);
                }
            } else {
                for &promo in &[BQ, BN, BR, BB] {
                    self.add_quiet_move(Move::new(from, to, PIECE_EMPTY, promo, 0), move_list);
                }
            };
        } else {
            self.add_quiet_move(Move::new(from, to, PIECE_EMPTY, PIECE_EMPTY, 0), move_list);
        }
    }

    fn generate_pawn_caps<const SIDE: u8, MC: MoveConsumer>(&self, sq: u8, move_list: &mut MC) {
        let left_sq = if SIDE == WHITE { sq + 9 } else { sq - 9 };
        let right_sq = if SIDE == WHITE { sq + 11 } else { sq - 11 };
        if square_on_board(left_sq) && PIECE_COL[self.piece_at(left_sq) as usize] as u8 == SIDE ^ 1
        {
            self.add_pawn_cap_move::<SIDE, MC>(sq, left_sq, self.piece_at(left_sq), move_list);
        }
        if square_on_board(right_sq)
            && PIECE_COL[self.piece_at(right_sq) as usize] as u8 == SIDE ^ 1
        {
            self.add_pawn_cap_move::<SIDE, MC>(sq, right_sq, self.piece_at(right_sq), move_list);
        }
    }

    fn generate_ep<const SIDE: u8, MC: MoveConsumer>(&self, sq: u8, move_list: &mut MC) {
        if self.ep_sq == NO_SQUARE {
            return;
        }
        let left_sq = if SIDE == WHITE { sq + 9 } else { sq - 9 };
        let right_sq = if SIDE == WHITE { sq + 11 } else { sq - 11 };
        if left_sq == self.ep_sq {
            self.add_ep_move(
                Move::new(sq, left_sq, PIECE_EMPTY, PIECE_EMPTY, Move::EP_MASK),
                move_list,
            );
        }
        if right_sq == self.ep_sq {
            self.add_ep_move(
                Move::new(sq, right_sq, PIECE_EMPTY, PIECE_EMPTY, Move::EP_MASK),
                move_list,
            );
        }
    }

    fn generate_pawn_forward<const SIDE: u8, MC: MoveConsumer>(&self, sq: u8, move_list: &mut MC) {
        let start_rank = if SIDE == WHITE {
            Rank::Rank2 as u8
        } else {
            Rank::Rank7 as u8
        };
        let offset_sq = if SIDE == WHITE { sq + 10 } else { sq - 10 };
        if self.piece_at(offset_sq) == PIECE_EMPTY {
            self.add_pawn_move::<SIDE, MC>(sq, offset_sq, move_list);
            let double_sq = if SIDE == WHITE { sq + 20 } else { sq - 20 };
            if RANKS_BOARD[sq as usize] == start_rank && self.piece_at(double_sq) == PIECE_EMPTY {
                self.add_quiet_move(
                    Move::new(
                        sq,
                        double_sq,
                        PIECE_EMPTY,
                        PIECE_EMPTY,
                        Move::PAWN_START_MASK,
                    ),
                    move_list,
                );
            }
        }
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    pub fn generate_moves<MC: MoveConsumer>(&self, move_list: &mut MC) {
        #[cfg(debug_assertions)]
        self.check_validity().unwrap();

        if MC::DO_PAWN_MOVEGEN {
            if self.side == WHITE {
                for &sq in self.piece_lists[WP as usize].iter() {
                    debug_assert!(square_on_board(sq));
                    self.generate_pawn_forward::<{ WHITE }, MC>(sq, move_list);
                    self.generate_pawn_caps::<{ WHITE }, MC>(sq, move_list);
                    self.generate_ep::<{ WHITE }, MC>(sq, move_list);
                }
            } else {
                for &sq in self.piece_lists[BP as usize].iter() {
                    debug_assert!(square_on_board(sq));
                    self.generate_pawn_forward::<{ BLACK }, MC>(sq, move_list);
                    self.generate_pawn_caps::<{ BLACK }, MC>(sq, move_list);
                    self.generate_ep::<{ BLACK }, MC>(sq, move_list);
                }
            }
        }

        let jumpers = if self.side == WHITE {
            &WHITE_JUMPERS
        } else {
            &BLACK_JUMPERS
        };
        for &piece in jumpers {
            let dirs = if piece == WN || piece == BN {
                &N_DIRS
            } else {
                &K_DIRS
            };
            for &sq in self.piece_lists[piece as usize].iter() {
                debug_assert!(square_on_board(sq));
                for &offset in dirs {
                    let t_sq = sq as i8 + offset;
                    if offset_square_offboard(t_sq) {
                        continue;
                    }

                    // now safe to convert to u8
                    // as offset_square_offboard() is false
                    let t_sq: u8 = unsafe { t_sq.try_into().unwrap_unchecked() };

                    if self.piece_at(t_sq) == PIECE_EMPTY {
                        self.add_quiet_move(
                            Move::new(sq, t_sq, PIECE_EMPTY, PIECE_EMPTY, 0),
                            move_list,
                        );
                    } else {
                        if PIECE_COL[self.piece_at(t_sq) as usize] as u8 == self.side ^ 1 {
                            self.add_capture_move(
                                Move::new(sq, t_sq, self.piece_at(t_sq), PIECE_EMPTY, 0),
                                move_list,
                            );
                        }
                    }
                }
            }
        }

        let sliders = if self.side == WHITE {
            &WHITE_SLIDERS
        } else {
            &BLACK_SLIDERS
        };
        for &piece in sliders {
            debug_assert!(piece_valid(piece));
            let dirs: &[i8] = match piece {
                WB | BB => &B_DIR,
                WR | BR => &R_DIR,
                WQ | BQ => &Q_DIR,
                _ => unsafe { std::hint::unreachable_unchecked() },
            };
            for &sq in self.piece_lists[piece as usize].iter() {
                debug_assert!(square_on_board(sq));

                for &dir in dirs {
                    let mut slider = sq as i8 + dir;
                    while !offset_square_offboard(slider) {
                        // now safe to convert to u8
                        // as offset_square_offboard() is false
                        let t_sq: u8 = unsafe { slider.try_into().unwrap_unchecked() };

                        if self.piece_at(t_sq) != PIECE_EMPTY {
                            if PIECE_COL[self.piece_at(t_sq) as usize] as u8 == self.side ^ 1 {
                                self.add_capture_move(
                                    Move::new(sq, t_sq, self.piece_at(t_sq), PIECE_EMPTY, 0),
                                    move_list,
                                );
                            }
                            break;
                        }
                        self.add_quiet_move(
                            Move::new(sq, t_sq, PIECE_EMPTY, PIECE_EMPTY, 0),
                            move_list,
                        );
                        slider += dir;
                    }
                }
            }
        }

        self.generate_castling_moves(move_list);
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    pub fn generate_captures<MC: MoveConsumer>(&self, move_list: &mut MC) {
        #[cfg(debug_assertions)]
        self.check_validity().unwrap();

        // both pawn moves and captures
        if self.side == WHITE {
            for &sq in self.piece_lists[WP as usize].iter() {
                debug_assert!(square_on_board(sq));
                self.generate_pawn_caps::<{ WHITE }, MC>(sq, move_list);
                self.generate_ep::<{ WHITE }, MC>(sq, move_list);
            }
        } else {
            for &sq in self.piece_lists[BP as usize].iter() {
                debug_assert!(square_on_board(sq));
                self.generate_pawn_caps::<{ BLACK }, MC>(sq, move_list);
                self.generate_ep::<{ BLACK }, MC>(sq, move_list);
            }
        }

        let jumpers = if self.side == WHITE {
            &WHITE_JUMPERS
        } else {
            &BLACK_JUMPERS
        };
        for &piece in jumpers {
            let dirs = if piece == WN || piece == BN {
                &N_DIRS
            } else {
                &K_DIRS
            };
            for &sq in self.piece_lists[piece as usize].iter() {
                debug_assert!(square_on_board(sq));
                for &offset in dirs {
                    let t_sq = sq as i8 + offset;
                    if offset_square_offboard(t_sq) {
                        continue;
                    }

                    // now safe to convert to u8
                    // as offset_square_offboard() is false
                    let t_sq: u8 = unsafe { t_sq.try_into().unwrap_unchecked() };

                    if self.piece_at(t_sq) != PIECE_EMPTY
                        && PIECE_COL[self.piece_at(t_sq) as usize] as u8 == self.side ^ 1
                    {
                        self.add_capture_move(
                            Move::new(sq, t_sq, self.piece_at(t_sq), PIECE_EMPTY, 0),
                            move_list,
                        );
                    }
                }
            }
        }

        let sliders = if self.side == WHITE {
            &WHITE_SLIDERS
        } else {
            &BLACK_SLIDERS
        };
        for &piece in sliders {
            debug_assert!(piece_valid(piece));
            let dirs: &[i8] = match piece {
                WB | BB => &B_DIR,
                WR | BR => &R_DIR,
                WQ | BQ => &Q_DIR,
                _ => unsafe { std::hint::unreachable_unchecked() },
            };
            for &sq in self.piece_lists[piece as usize].iter() {
                debug_assert!(square_on_board(sq));

                for &dir in dirs {
                    let mut slider = sq as i8 + dir;
                    while !offset_square_offboard(slider) {
                        // now safe to convert to u8
                        // as offset_square_offboard() is false
                        let t_sq: u8 = unsafe { slider.try_into().unwrap_unchecked() };

                        if self.piece_at(t_sq) != PIECE_EMPTY {
                            if PIECE_COL[self.piece_at(t_sq) as usize] as u8 == self.side ^ 1 {
                                self.add_capture_move(
                                    Move::new(sq, t_sq, self.piece_at(t_sq), PIECE_EMPTY, 0),
                                    move_list,
                                );
                            }
                            break;
                        }
                        slider += dir;
                    }
                }
            }
        }
    }

    fn generate_castling_moves(&self, move_list: &mut impl MoveConsumer) {
        if self.side == WHITE {
            if (self.castle_perm & Castling::WK as u8) != 0
                && self.piece_at(Square120::F1 as u8) == PIECE_EMPTY
                && self.piece_at(Square120::G1 as u8) == PIECE_EMPTY
                && !self.sq_attacked(Square120::E1 as u8, BLACK)
                && !self.sq_attacked(Square120::F1 as u8, BLACK)
            {
                self.add_quiet_move(
                    Move::new(
                        Square120::E1 as u8,
                        Square120::G1 as u8,
                        PIECE_EMPTY,
                        PIECE_EMPTY,
                        Move::CASTLE_MASK,
                    ),
                    move_list,
                );
            }

            if (self.castle_perm & Castling::WQ as u8) != 0
                && self.piece_at(Square120::D1 as u8) == PIECE_EMPTY
                && self.piece_at(Square120::C1 as u8) == PIECE_EMPTY
                && self.piece_at(Square120::B1 as u8) == PIECE_EMPTY
                && !self.sq_attacked(Square120::E1 as u8, BLACK)
                && !self.sq_attacked(Square120::D1 as u8, BLACK)
            {
                self.add_quiet_move(
                    Move::new(
                        Square120::E1 as u8,
                        Square120::C1 as u8,
                        PIECE_EMPTY,
                        PIECE_EMPTY,
                        Move::CASTLE_MASK,
                    ),
                    move_list,
                );
            }
        } else {
            if (self.castle_perm & Castling::BK as u8) != 0
                && self.piece_at(Square120::F8 as u8) == PIECE_EMPTY
                && self.piece_at(Square120::G8 as u8) == PIECE_EMPTY
                && !self.sq_attacked(Square120::E8 as u8, WHITE)
                && !self.sq_attacked(Square120::F8 as u8, WHITE)
            {
                self.add_quiet_move(
                    Move::new(
                        Square120::E8 as u8,
                        Square120::G8 as u8,
                        PIECE_EMPTY,
                        PIECE_EMPTY,
                        Move::CASTLE_MASK,
                    ),
                    move_list,
                );
            }

            if (self.castle_perm & Castling::BQ as u8) != 0
                && self.piece_at(Square120::D8 as u8) == PIECE_EMPTY
                && self.piece_at(Square120::C8 as u8) == PIECE_EMPTY
                && self.piece_at(Square120::B8 as u8) == PIECE_EMPTY
                && !self.sq_attacked(Square120::E8 as u8, WHITE)
                && !self.sq_attacked(Square120::D8 as u8, WHITE)
            {
                self.add_quiet_move(
                    Move::new(
                        Square120::E8 as u8,
                        Square120::C8 as u8,
                        PIECE_EMPTY,
                        PIECE_EMPTY,
                        Move::CASTLE_MASK,
                    ),
                    move_list,
                );
            }
        }
    }
}
