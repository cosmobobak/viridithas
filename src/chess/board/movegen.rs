use anyhow::Context;
use arrayvec::ArrayVec;

use std::{
    fmt::{Display, Formatter},
    ops::{Deref, DerefMut},
    sync::atomic::Ordering,
};

use crate::{
    cfor,
    chess::{
        board::Board,
        chessmove::{Move, MoveFlags},
        magic::{
            bishop_attacks_on_the_fly, rook_attacks_on_the_fly, set_occupancy, BISHOP_ATTACKS, BISHOP_REL_BITS, BISHOP_TABLE, ROOK_ATTACKS, ROOK_REL_BITS, ROOK_TABLE
        },
        piece::{Black, Col, Colour, PieceType, White},
        squareset::SquareSet,
        types::{Rank, Square},
        CHESS960,
    },
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

#[derive(Clone, Debug)]
pub struct MoveList {
    // moves: [MoveListEntry; MAX_POSITION_MOVES],
    // count: usize,
    inner: ArrayVec<MoveListEntry, MAX_POSITION_MOVES>,
}

impl MoveList {
    pub fn new() -> Self {
        Self {
            inner: ArrayVec::new(),
        }
    }

    fn push<const TACTICAL: bool>(&mut self, m: Move) {
        // debug_assert!(self.count < MAX_POSITION_MOVES, "overflowed {self}");
        let score = if TACTICAL {
            MoveListEntry::TACTICAL_SENTINEL
        } else {
            MoveListEntry::QUIET_SENTINEL
        };

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
            writeln!(f, "  {} ${}, ", m.mov.display(false), m.score)?;
        }
        writeln!(
            f,
            "  {} ${}",
            self.inner[self.inner.len() - 1].mov.display(false),
            self.inner[self.inner.len() - 1].score
        )?;
        write!(f, "]")
    }
}

const fn in_between(sq1: Square, sq2: Square) -> SquareSet {
    const M1: u64 = 0xFFFF_FFFF_FFFF_FFFF;
    const A2A7: u64 = 0x0001_0101_0101_0100;
    const B2G7: u64 = 0x0040_2010_0804_0200;
    const H1B7: u64 = 0x0002_0408_1020_4080;
    let sq1 = sq1.index();
    let sq2 = sq2.index();
    let btwn = (M1 << sq1) ^ (M1 << sq2);
    let file = ((sq2 & 7).wrapping_add((sq1 & 7).wrapping_neg())) as u64;
    let rank = (((sq2 | 7).wrapping_sub(sq1)) >> 3) as u64;
    let mut line = ((file & 7).wrapping_sub(1)) & A2A7;
    line += 2 * ((rank & 7).wrapping_sub(1) >> 58);
    line += ((rank.wrapping_sub(file) & 15).wrapping_sub(1)) & B2G7;
    line += ((rank.wrapping_add(file) & 15).wrapping_sub(1)) & H1B7;
    line = line.wrapping_mul(btwn & btwn.wrapping_neg());
    SquareSet::from_inner(line & btwn)
}

pub static RAY_BETWEEN: [[SquareSet; 64]; 64] = {
    let mut res = [[SquareSet::EMPTY; 64]; 64];
    let mut from = Square::A1;
    loop {
        let mut to = Square::A1;
        loop {
            res[from.index()][to.index()] = in_between(from, to);
            let Some(next) = to.add(1) else {
                break;
            };
            to = next;
        }
        let Some(next) = from.add(1) else {
            break;
        };
        from = next;
    }
    res
};

const fn init_jumping_attacks<const IS_KNIGHT: bool>() -> [SquareSet; 64] {
    let mut attacks = [SquareSet::EMPTY; 64];
    let deltas = if IS_KNIGHT {
        &[17, 15, 10, 6, -17, -15, -10, -6]
    } else {
        &[9, 8, 7, 1, -9, -8, -7, -1]
    };

    cfor!(let mut sq = Square::A1; true; sq = sq.saturating_add(1); {
        let mut attacks_bb = 0;
        cfor!(let mut idx = 0; idx < 8; idx += 1; {
            let delta = deltas[idx];
            let attacked_sq = sq.signed_inner() + delta;
            #[allow(clippy::cast_sign_loss)]
            if 0 <= attacked_sq && attacked_sq < 64 && Square::distance(
                sq,
                Square::new_clamped(attacked_sq as u8)) <= 2 {
                attacks_bb |= 1 << attacked_sq;
            }
        });
        attacks[sq.index()] = SquareSet::from_inner(attacks_bb);
        if matches!(sq, Square::H8) {
            break;
        }
    });

    attacks
}

pub fn init_sliders_attacks() -> anyhow::Result<()> {
    #![allow(clippy::large_stack_arrays)]
    let mut bishop_attacks = vec![[SquareSet::EMPTY; 512]; 64];
    for sq in Square::all() {
        let entry = &BISHOP_TABLE[sq];
        // init the current mask
        let mask = entry.mask;
        // count attack mask bits
        let bit_count = mask.count();
        // occupancy var count
        let occupancy_variations = 1 << bit_count;
        // loop over all occupancy variations
        for count in 0..occupancy_variations {
            // init occupancies
            let occupancy = set_occupancy(count, bit_count.into(), mask);
            let magic_index: usize = ((occupancy.inner().wrapping_mul(entry.magic))
                >> (64 - BISHOP_REL_BITS))
                .try_into()
                .unwrap();
            bishop_attacks[sq as usize][magic_index] = bishop_attacks_on_the_fly(sq, occupancy);
        }
    }
    let mut rook_attacks = vec![[SquareSet::EMPTY; 4096]; 64];
    for sq in Square::all() {
        let entry = &ROOK_TABLE[sq];
        // init the current mask
        let mask = entry.mask;
        // count attack mask bits
        let bit_count = mask.count();
        // occupancy var count
        let occupancy_variations = 1 << bit_count;
        // loop over all occupancy variations
        for count in 0..occupancy_variations {
            // init occupancies
            let occupancy = set_occupancy(count, bit_count.into(), mask);
            let magic_index: usize = ((occupancy.inner().wrapping_mul(entry.magic))
                >> (64 - ROOK_REL_BITS))
                .try_into()
                .unwrap();
            rook_attacks[sq as usize][magic_index] = rook_attacks_on_the_fly(sq, occupancy);
        }
    }

    // SAFETY: SquareSet is POD.
    let bishop_bytes = unsafe { bishop_attacks.align_to::<u8>().1 };
    // SAFETY: SquareSet is POD.
    let rook_bytes = unsafe { rook_attacks.align_to::<u8>().1 };

    std::fs::write("embeds/diagonal_attacks.bin", bishop_bytes)
        .context("failed to write embeds/diagonal_attacks.bin")?;
    std::fs::write("embeds/orthogonal_attacks.bin", rook_bytes)
        .context("failed to write embeds/orthogonal_attacks.bin")?;

    Ok(())
}

#[allow(clippy::cast_possible_truncation)]
pub fn bishop_attacks(sq: Square, blockers: SquareSet) -> SquareSet {
    let entry = &BISHOP_TABLE[sq];
    let relevant_blockers = blockers & entry.mask;
    let data = relevant_blockers.inner().wrapping_mul(entry.magic);
    // BISHOP_REL_BITS is 9, so this shift is by 55.
    let idx = (data >> (64 - BISHOP_REL_BITS)) as usize;
    const {
        assert!(1 << BISHOP_REL_BITS == BISHOP_ATTACKS[0].len());
    }
    // SAFETY: The largest value we can obtain from (data >> 55)
    // is u64::MAX >> 55, which is 511 (0x1FF). BISHOP_ATTACKS[sq]
    // is 512 elements long, so this is always in bounds.
    unsafe { *BISHOP_ATTACKS[sq].get_unchecked(idx) }
}
#[allow(clippy::cast_possible_truncation)]
pub fn rook_attacks(sq: Square, blockers: SquareSet) -> SquareSet {
    let entry = &ROOK_TABLE[sq];
    let relevant_blockers = blockers & entry.mask;
    let data = relevant_blockers.inner().wrapping_mul(entry.magic);
    // ROOK_REL_BITS is 12, so this shift is by 52.
    let idx = (data >> (64 - ROOK_REL_BITS)) as usize;
    const {
        assert!(1 << ROOK_REL_BITS == ROOK_ATTACKS[0].len());
    }
    // SAFETY: The largest value we can obtain from (data >> 52)
    // is u64::MAX >> 52, which is 4095 (0xFFF). ROOK_ATTACKS[sq]
    // is 4096 elements long, so this is always in bounds.
    unsafe { *ROOK_ATTACKS[sq].get_unchecked(idx) }
}
pub fn knight_attacks(sq: Square) -> SquareSet {
    static KNIGHT_ATTACKS: [SquareSet; 64] = init_jumping_attacks::<true>();
    KNIGHT_ATTACKS[sq]
}
pub fn king_attacks(sq: Square) -> SquareSet {
    static KING_ATTACKS: [SquareSet; 64] = init_jumping_attacks::<false>();
    KING_ATTACKS[sq]
}
pub fn pawn_attacks<C: Col>(bb: SquareSet) -> SquareSet {
    if C::WHITE {
        bb.north_east_one() | bb.north_west_one()
    } else {
        bb.south_east_one() | bb.south_west_one()
    }
}

pub fn attacks_by_type(pt: PieceType, sq: Square, blockers: SquareSet) -> SquareSet {
    match pt {
        PieceType::Pawn => {
            debug_assert!(false, "Invalid piece type: {pt:?}");
            SquareSet::EMPTY
        }
        PieceType::Knight => knight_attacks(sq),
        PieceType::Bishop => bishop_attacks(sq, blockers),
        PieceType::Rook => rook_attacks(sq, blockers),
        PieceType::Queen => bishop_attacks(sq, blockers) | rook_attacks(sq, blockers),
        PieceType::King => king_attacks(sq),
    }
}

pub trait MoveGenMode {
    const SKIP_QUIETS: bool;
}

pub struct SkipQuiets;
impl MoveGenMode for SkipQuiets {
    const SKIP_QUIETS: bool = true;
}
pub struct AllMoves;
impl MoveGenMode for AllMoves {
    const SKIP_QUIETS: bool = false;
}

impl Board {
    fn generate_pawn_caps<C: Col, Mode: MoveGenMode>(
        &self,
        move_list: &mut MoveList,
        valid_target_squares: SquareSet,
    ) {
        let our_pawns = self.state.piece_layout.pawns::<C>();
        let their_pieces = self.state.piece_layout.their_pieces::<C>();
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
        let promo_rank = if C::WHITE {
            SquareSet::RANK_7
        } else {
            SquareSet::RANK_2
        };
        let from_mask = attacking_west & !promo_rank & valid_west;
        let to_mask = if C::WHITE {
            from_mask.north_west_one()
        } else {
            from_mask.south_west_one()
        };
        for (from, to) in from_mask.into_iter().zip(to_mask) {
            move_list.push::<true>(Move::new(from, to));
        }
        let from_mask = attacking_east & !promo_rank & valid_east;
        let to_mask = if C::WHITE {
            from_mask.north_east_one()
        } else {
            from_mask.south_east_one()
        };
        for (from, to) in from_mask.into_iter().zip(to_mask) {
            move_list.push::<true>(Move::new(from, to));
        }
        let from_mask = attacking_west & promo_rank & valid_west;
        let to_mask = if C::WHITE {
            from_mask.north_west_one()
        } else {
            from_mask.south_west_one()
        };
        for (from, to) in from_mask.into_iter().zip(to_mask) {
            // in quiescence search, we only generate promotions to queen.
            if Mode::SKIP_QUIETS {
                move_list.push::<true>(Move::new_with_promo(from, to, PieceType::Queen));
            } else {
                for promo in [
                    PieceType::Queen,
                    PieceType::Rook,
                    PieceType::Bishop,
                    PieceType::Knight,
                ] {
                    move_list.push::<true>(Move::new_with_promo(from, to, promo));
                }
            }
        }
        let from_mask = attacking_east & promo_rank & valid_east;
        let to_mask = if C::WHITE {
            from_mask.north_east_one()
        } else {
            from_mask.south_east_one()
        };
        for (from, to) in from_mask.into_iter().zip(to_mask) {
            // in quiescence search, we only generate promotions to queen.
            if Mode::SKIP_QUIETS {
                move_list.push::<true>(Move::new_with_promo(from, to, PieceType::Queen));
            } else {
                for promo in [
                    PieceType::Queen,
                    PieceType::Rook,
                    PieceType::Bishop,
                    PieceType::Knight,
                ] {
                    move_list.push::<true>(Move::new_with_promo(from, to, promo));
                }
            }
        }
    }

    fn generate_ep<C: Col>(&self, move_list: &mut MoveList) {
        let Some(ep_sq) = self.state.ep_square else {
            return;
        };
        let ep_bb = ep_sq.as_set();
        let our_pawns = self.state.piece_layout.pawns::<C>();
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
            move_list.push::<true>(Move::new_with_flags(from_sq, ep_sq, MoveFlags::EnPassant));
        }
        if attacks_east.non_empty() {
            let from_sq = attacks_east.first();
            move_list.push::<true>(Move::new_with_flags(from_sq, ep_sq, MoveFlags::EnPassant));
        }
    }

    fn generate_pawn_forward<C: Col>(
        &self,
        move_list: &mut MoveList,
        valid_target_squares: SquareSet,
    ) {
        let start_rank = if C::WHITE {
            SquareSet::RANK_2
        } else {
            SquareSet::RANK_7
        };
        let promo_rank = if C::WHITE {
            SquareSet::RANK_7
        } else {
            SquareSet::RANK_2
        };
        let shifted_empty_squares = if C::WHITE {
            self.state.piece_layout.empty() >> 8
        } else {
            self.state.piece_layout.empty() << 8
        };
        let double_shifted_empty_squares = if C::WHITE {
            self.state.piece_layout.empty() >> 16
        } else {
            self.state.piece_layout.empty() << 16
        };
        let shifted_valid_squares = if C::WHITE {
            valid_target_squares >> 8
        } else {
            valid_target_squares << 8
        };
        let double_shifted_valid_squares = if C::WHITE {
            valid_target_squares >> 16
        } else {
            valid_target_squares << 16
        };
        let our_pawns = self.state.piece_layout.pawns::<C>();
        let pushable_pawns = our_pawns & shifted_empty_squares;
        let double_pushable_pawns = pushable_pawns & double_shifted_empty_squares & start_rank;
        let promoting_pawns = pushable_pawns & promo_rank;

        let from_mask = pushable_pawns & !promoting_pawns & shifted_valid_squares;
        let to_mask = if C::WHITE {
            from_mask.north_one()
        } else {
            from_mask.south_one()
        };
        for (from, to) in from_mask.into_iter().zip(to_mask) {
            move_list.push::<false>(Move::new(from, to));
        }
        let from_mask = double_pushable_pawns & double_shifted_valid_squares;
        let to_mask = if C::WHITE {
            from_mask.north_one().north_one()
        } else {
            from_mask.south_one().south_one()
        };
        for (from, to) in from_mask.into_iter().zip(to_mask) {
            move_list.push::<false>(Move::new(from, to));
        }
        let from_mask = promoting_pawns & shifted_valid_squares;
        let to_mask = if C::WHITE {
            from_mask.north_one()
        } else {
            from_mask.south_one()
        };
        for (from, to) in from_mask.into_iter().zip(to_mask) {
            for promo in [
                PieceType::Queen,
                PieceType::Knight,
                PieceType::Rook,
                PieceType::Bishop,
            ] {
                move_list.push::<true>(Move::new_with_promo(from, to, promo));
            }
        }
    }

    fn generate_forward_promos<C: Col, Mode: MoveGenMode>(
        &self,
        move_list: &mut MoveList,
        valid_target_squares: SquareSet,
    ) {
        let promo_rank = if C::WHITE {
            SquareSet::RANK_7
        } else {
            SquareSet::RANK_2
        };
        let shifted_empty_squares = if C::WHITE {
            self.state.piece_layout.empty() >> 8
        } else {
            self.state.piece_layout.empty() << 8
        };
        let shifted_valid_squares = if C::WHITE {
            valid_target_squares >> 8
        } else {
            valid_target_squares << 8
        };
        let our_pawns = self.state.piece_layout.pawns::<C>();
        let pushable_pawns = our_pawns & shifted_empty_squares;
        let promoting_pawns = pushable_pawns & promo_rank;

        let from_mask = promoting_pawns & shifted_valid_squares;
        let to_mask = if C::WHITE {
            from_mask.north_one()
        } else {
            from_mask.south_one()
        };
        for (from, to) in from_mask.into_iter().zip(to_mask) {
            if Mode::SKIP_QUIETS {
                // in quiescence search, we only generate promotions to queen.
                move_list.push::<true>(Move::new_with_promo(from, to, PieceType::Queen));
            } else {
                for promo in [
                    PieceType::Queen,
                    PieceType::Knight,
                    PieceType::Rook,
                    PieceType::Bishop,
                ] {
                    move_list.push::<true>(Move::new_with_promo(from, to, promo));
                }
            }
        }
    }

    pub fn generate_moves(&self, move_list: &mut MoveList) {
        move_list.clear();
        if self.side == Colour::White {
            self.generate_moves_for::<White>(move_list);
        } else {
            self.generate_moves_for::<Black>(move_list);
        }
        debug_assert!(move_list.iter_moves().all(|m| m.is_valid()));
    }

    fn generate_moves_for<C: Col>(&self, move_list: &mut MoveList) {
        #[cfg(debug_assertions)]
        self.check_validity().unwrap();

        let their_pieces = self.state.piece_layout.their_pieces::<C>();
        let freespace = self.state.piece_layout.empty();
        let our_king_sq = self.state.piece_layout.king::<C>().first();

        if self.state.threats.checkers.count() > 1 {
            // we're in double-check, so we can only move the king.
            let moves = king_attacks(our_king_sq) & !self.state.threats.all;
            for to in moves & their_pieces {
                move_list.push::<true>(Move::new(our_king_sq, to));
            }
            for to in moves & freespace {
                move_list.push::<false>(Move::new(our_king_sq, to));
            }
            return;
        }

        let valid_target_squares = if self.in_check() {
            RAY_BETWEEN[our_king_sq][self.state.threats.checkers.first()]
                | self.state.threats.checkers
        } else {
            SquareSet::FULL
        };

        self.generate_pawn_forward::<C>(move_list, valid_target_squares);
        self.generate_pawn_caps::<C, AllMoves>(move_list, valid_target_squares);
        self.generate_ep::<C>(move_list);

        // knights
        let our_knights = self.state.piece_layout.knights::<C>();
        for sq in our_knights {
            let moves = knight_attacks(sq) & valid_target_squares;
            for to in moves & their_pieces {
                move_list.push::<true>(Move::new(sq, to));
            }
            for to in moves & freespace {
                move_list.push::<false>(Move::new(sq, to));
            }
        }

        // kings
        let moves = king_attacks(our_king_sq) & !self.state.threats.all;
        for to in moves & their_pieces {
            move_list.push::<true>(Move::new(our_king_sq, to));
        }
        for to in moves & freespace {
            move_list.push::<false>(Move::new(our_king_sq, to));
        }

        // bishops and queens
        let our_diagonal_sliders = self.state.piece_layout.diags::<C>();
        let blockers = self.state.piece_layout.occupied();
        for sq in our_diagonal_sliders {
            let moves = bishop_attacks(sq, blockers) & valid_target_squares;
            for to in moves & their_pieces {
                move_list.push::<true>(Move::new(sq, to));
            }
            for to in moves & freespace {
                move_list.push::<false>(Move::new(sq, to));
            }
        }

        // rooks and queens
        let our_orthogonal_sliders = self.state.piece_layout.orthos::<C>();
        for sq in our_orthogonal_sliders {
            let moves = rook_attacks(sq, blockers) & valid_target_squares;
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

    pub fn generate_captures<Mode: MoveGenMode>(&self, move_list: &mut MoveList) {
        move_list.clear();
        if self.side == Colour::White {
            self.generate_captures_for::<White, Mode>(move_list);
        } else {
            self.generate_captures_for::<Black, Mode>(move_list);
        }
        debug_assert!(move_list.iter_moves().all(|m| m.is_valid()));
    }

    fn generate_captures_for<C: Col, Mode: MoveGenMode>(&self, move_list: &mut MoveList) {
        #[cfg(debug_assertions)]
        self.check_validity().unwrap();

        let their_pieces = self.state.piece_layout.their_pieces::<C>();
        let our_king_sq = self.state.piece_layout.king::<C>().first();

        if self.state.threats.checkers.count() > 1 {
            // we're in double-check, so we can only move the king.
            let moves = king_attacks(our_king_sq) & !self.state.threats.all;
            for to in moves & their_pieces {
                move_list.push::<true>(Move::new(our_king_sq, to));
            }
            return;
        }

        let valid_target_squares = if self.in_check() {
            RAY_BETWEEN[our_king_sq][self.state.threats.checkers.first()]
                | self.state.threats.checkers
        } else {
            SquareSet::FULL
        };

        // promotions
        self.generate_forward_promos::<C, Mode>(move_list, valid_target_squares);

        // pawn captures and capture promos
        self.generate_pawn_caps::<C, Mode>(move_list, valid_target_squares);
        self.generate_ep::<C>(move_list);

        // knights
        let our_knights = self.state.piece_layout.knights::<C>();
        let their_pieces = self.state.piece_layout.their_pieces::<C>();
        for sq in our_knights {
            let moves = knight_attacks(sq) & valid_target_squares;
            for to in moves & their_pieces {
                move_list.push::<true>(Move::new(sq, to));
            }
        }

        // kings
        let moves = king_attacks(our_king_sq) & !self.state.threats.all;
        for to in moves & their_pieces {
            move_list.push::<true>(Move::new(our_king_sq, to));
        }

        // bishops and queens
        let our_diagonal_sliders = self.state.piece_layout.diags::<C>();
        let blockers = self.state.piece_layout.occupied();
        for sq in our_diagonal_sliders {
            let moves = bishop_attacks(sq, blockers) & valid_target_squares;
            for to in moves & their_pieces {
                move_list.push::<true>(Move::new(sq, to));
            }
        }

        // rooks and queens
        let our_orthogonal_sliders = self.state.piece_layout.orthos::<C>();
        for sq in our_orthogonal_sliders {
            let moves = rook_attacks(sq, blockers) & valid_target_squares;
            for to in moves & their_pieces {
                move_list.push::<true>(Move::new(sq, to));
            }
        }
    }

    fn generate_castling_moves_for<C: Col>(&self, move_list: &mut MoveList) {
        let occupied = self.state.piece_layout.occupied();

        if CHESS960.load(Ordering::Relaxed) {
            let king_sq = self.king_sq(C::COLOUR);
            if self.sq_attacked(king_sq, C::Opposite::COLOUR) {
                return;
            }

            let castling_kingside = self.state.castle_perm.kingside(C::COLOUR);
            if let Some(castling_kingside) = castling_kingside {
                let king_dst = Square::G1.relative_to(C::COLOUR);
                let rook_dst = Square::F1.relative_to(C::COLOUR);
                let castling_sq = castling_kingside.with(match C::COLOUR {
                    Colour::White => Rank::One,
                    Colour::Black => Rank::Eight,
                });
                self.try_generate_frc_castling::<C>(
                    king_sq,
                    castling_sq,
                    king_dst,
                    rook_dst,
                    occupied,
                    move_list,
                );
            }

            let castling_queenside = self.state.castle_perm.queenside(C::COLOUR);
            if let Some(castling_queenside) = castling_queenside {
                let king_dst = Square::C1.relative_to(C::COLOUR);
                let rook_dst = Square::D1.relative_to(C::COLOUR);
                let castling_sq = castling_queenside.with(match C::COLOUR {
                    Colour::White => Rank::One,
                    Colour::Black => Rank::Eight,
                });
                self.try_generate_frc_castling::<C>(
                    king_sq,
                    castling_sq,
                    king_dst,
                    rook_dst,
                    occupied,
                    move_list,
                );
            }
        } else {
            const WK_FREESPACE: SquareSet = Square::F1.as_set().add_square(Square::G1);
            const WQ_FREESPACE: SquareSet = Square::B1
                .as_set()
                .add_square(Square::C1)
                .add_square(Square::D1);
            const BK_FREESPACE: SquareSet = Square::F8.as_set().add_square(Square::G8);
            const BQ_FREESPACE: SquareSet = Square::B8
                .as_set()
                .add_square(Square::C8)
                .add_square(Square::D8);

            let k_freespace = if C::WHITE { WK_FREESPACE } else { BK_FREESPACE };
            let q_freespace = if C::WHITE { WQ_FREESPACE } else { BQ_FREESPACE };
            let from = Square::E1.relative_to(C::COLOUR);
            let k_to = Square::H1.relative_to(C::COLOUR);
            let q_to = Square::A1.relative_to(C::COLOUR);
            let k_thru = Square::F1.relative_to(C::COLOUR);
            let q_thru = Square::D1.relative_to(C::COLOUR);
            let k_perm = self.state.castle_perm.kingside(C::COLOUR);
            let q_perm = self.state.castle_perm.queenside(C::COLOUR);

            // stupid hack to avoid redoing or eagerly doing hard work.
            let mut cache = None;

            if k_perm.is_some()
                && (occupied & k_freespace).is_empty()
                && {
                    let got_attacked_king = self.sq_attacked_by::<C::Opposite>(from);
                    cache = Some(got_attacked_king);
                    !got_attacked_king
                }
                && !self.sq_attacked_by::<C::Opposite>(k_thru)
            {
                move_list.push::<false>(Move::new_with_flags(from, k_to, MoveFlags::Castle));
            }

            if q_perm.is_some()
                && (occupied & q_freespace).is_empty()
                && !cache.unwrap_or_else(|| self.sq_attacked_by::<C::Opposite>(from))
                && !self.sq_attacked_by::<C::Opposite>(q_thru)
            {
                move_list.push::<false>(Move::new_with_flags(from, q_to, MoveFlags::Castle));
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
        let king_path = RAY_BETWEEN[king_sq][king_dst];
        let rook_path = RAY_BETWEEN[king_sq][castling_sq];
        let relevant_occupied = occupied ^ king_sq.as_set() ^ castling_sq.as_set();
        if (relevant_occupied & (king_path | rook_path | king_dst.as_set() | rook_dst.as_set()))
            .is_empty()
            && !self.any_attacked(king_path, C::Opposite::COLOUR)
        {
            move_list.push::<false>(Move::new_with_flags(
                king_sq,
                castling_sq,
                MoveFlags::Castle,
            ));
        }
    }

    pub fn generate_quiets(&self, move_list: &mut MoveList) {
        // we don't need to clear the move list here because we're only adding to it.
        if self.side == Colour::White {
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
        let start_rank = if C::WHITE {
            SquareSet::RANK_2
        } else {
            SquareSet::RANK_7
        };
        let promo_rank = if C::WHITE {
            SquareSet::RANK_7
        } else {
            SquareSet::RANK_2
        };
        let shifted_empty_squares = if C::WHITE {
            self.state.piece_layout.empty() >> 8
        } else {
            self.state.piece_layout.empty() << 8
        };
        let double_shifted_empty_squares = if C::WHITE {
            self.state.piece_layout.empty() >> 16
        } else {
            self.state.piece_layout.empty() << 16
        };
        let shifted_valid_squares = if C::WHITE {
            valid_target_squares >> 8
        } else {
            valid_target_squares << 8
        };
        let double_shifted_valid_squares = if C::WHITE {
            valid_target_squares >> 16
        } else {
            valid_target_squares << 16
        };
        let our_pawns = self.state.piece_layout.pawns::<C>();
        let pushable_pawns = our_pawns & shifted_empty_squares;
        let double_pushable_pawns = pushable_pawns & double_shifted_empty_squares & start_rank;
        let promoting_pawns = pushable_pawns & promo_rank;

        let from_mask = pushable_pawns & !promoting_pawns & shifted_valid_squares;
        let to_mask = if C::WHITE {
            from_mask.north_one()
        } else {
            from_mask.south_one()
        };
        for (from, to) in from_mask.into_iter().zip(to_mask) {
            move_list.push::<false>(Move::new(from, to));
        }
        let from_mask = double_pushable_pawns & double_shifted_valid_squares;
        let to_mask = if C::WHITE {
            from_mask.north_one().north_one()
        } else {
            from_mask.south_one().south_one()
        };
        for (from, to) in from_mask.into_iter().zip(to_mask) {
            move_list.push::<false>(Move::new(from, to));
        }
    }

    fn generate_quiets_for<C: Col>(&self, move_list: &mut MoveList) {
        let freespace = self.state.piece_layout.empty();
        let our_king_sq = self.state.piece_layout.king::<C>().first();
        let blockers = self.state.piece_layout.occupied();

        if self.state.threats.checkers.count() > 1 {
            // we're in double-check, so we can only move the king.
            let moves = king_attacks(our_king_sq) & !self.state.threats.all;
            for to in moves & freespace {
                move_list.push::<false>(Move::new(our_king_sq, to));
            }
            return;
        }

        let valid_target_squares = if self.in_check() {
            RAY_BETWEEN[our_king_sq][self.state.threats.checkers.first()]
                | self.state.threats.checkers
        } else {
            SquareSet::FULL
        };

        // pawns
        self.generate_pawn_quiet::<C>(move_list, valid_target_squares);

        // knights
        let our_knights = self.state.piece_layout.knights::<C>();
        for sq in our_knights {
            let moves = knight_attacks(sq) & valid_target_squares;
            for to in moves & !blockers {
                move_list.push::<false>(Move::new(sq, to));
            }
        }

        // kings
        let moves = king_attacks(our_king_sq) & !self.state.threats.all;
        for to in moves & !blockers {
            move_list.push::<false>(Move::new(our_king_sq, to));
        }

        // bishops and queens
        let our_diagonal_sliders = self.state.piece_layout.diags::<C>();
        for sq in our_diagonal_sliders {
            let moves = bishop_attacks(sq, blockers) & valid_target_squares;
            for to in moves & !blockers {
                move_list.push::<false>(Move::new(sq, to));
            }
        }

        // rooks and queens
        let our_orthogonal_sliders = self.state.piece_layout.orthos::<C>();
        for sq in our_orthogonal_sliders {
            let moves = rook_attacks(sq, blockers) & valid_target_squares;
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
    #![allow(clippy::to_string_in_format_args)]
    #[cfg(debug_assertions)]
    pos.check_validity().unwrap();

    if depth == 0 {
        return 1;
    }

    let mut ml = MoveList::new();
    pos.generate_moves(&mut ml);
    let mut ml_staged = MoveList::new();
    pos.generate_captures::<AllMoves>(&mut ml_staged);
    pos.generate_quiets(&mut ml_staged);

    let mut full_moves_vec = ml.to_vec();
    let mut staged_moves_vec = ml_staged.to_vec();
    full_moves_vec.sort_unstable_by_key(|m| m.mov);
    staged_moves_vec.sort_unstable_by_key(|m| m.mov);
    let eq = full_moves_vec == staged_moves_vec;
    assert!(
        eq,
        "full and staged move lists differ in {}, \nfull list: \n[{}], \nstaged list: \n[{}]",
        pos.to_string(),
        {
            let mut mvs = Vec::new();
            for m in full_moves_vec {
                mvs.push(format!(
                    "{}{}",
                    pos.san(m.mov).unwrap(),
                    if m.score == MoveListEntry::TACTICAL_SENTINEL {
                        "T"
                    } else {
                        "Q"
                    }
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
                    if m.score == MoveListEntry::TACTICAL_SENTINEL {
                        "T"
                    } else {
                        "Q"
                    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        bench,
        chess::{
            board::movegen::{king_attacks, knight_attacks},
            magic::{bishop_attacks_on_the_fly, rook_attacks_on_the_fly},
            squareset::SquareSet,
            types::Square,
        },
    };

    #[test]
    fn staged_matches_full() {
        let mut pos = Board::default();

        for fen in bench::BENCH_POSITIONS {
            pos.set_from_fen(fen).unwrap();
            synced_perft(&mut pos, 2);
        }
    }

    #[test]
    fn python_chess_validation() {
        // testing that the attack squaresets match the ones in the python-chess library,
        // which are known to be correct.
        assert_eq!(
            knight_attacks(Square::new(0).unwrap()),
            SquareSet::from_inner(132_096)
        );
        assert_eq!(
            knight_attacks(Square::new(63).unwrap()),
            SquareSet::from_inner(9_077_567_998_918_656)
        );

        assert_eq!(
            king_attacks(Square::new(0).unwrap()),
            SquareSet::from_inner(770)
        );
        assert_eq!(
            king_attacks(Square::new(63).unwrap()),
            SquareSet::from_inner(4_665_729_213_955_833_856)
        );
    }

    #[test]
    fn rook_attacks_basic() {
        let sq = Square::E4;
        let mask = ROOK_TABLE[sq].mask;
        let mut subset = SquareSet::EMPTY;
        loop {
            let attacks_naive = rook_attacks_on_the_fly(sq, subset);
            let attacks_fast = rook_attacks(sq, subset);
            assert_eq!(
                attacks_naive, attacks_fast,
                "naive:\n{attacks_naive}\nfast:\n{attacks_fast}\nblockers were\n{subset}"
            );
            subset = SquareSet::from_inner(subset.inner().wrapping_sub(mask.inner())) & mask;
            if subset.is_empty() {
                break;
            }
        }
    }

    #[test]
    fn bishop_attacks_basic() {
        let sq = Square::E4;
        let mask = BISHOP_TABLE[sq].mask;
        let mut subset = SquareSet::EMPTY;
        loop {
            let attacks_naive = bishop_attacks_on_the_fly(sq, subset);
            let attacks_fast = bishop_attacks(sq, subset);
            assert_eq!(
                attacks_naive, attacks_fast,
                "naive:\n{attacks_naive}\nfast:\n{attacks_fast}\nblockers were\n{subset}"
            );
            subset = SquareSet::from_inner(subset.inner().wrapping_sub(mask.inner())) & mask;
            if subset.is_empty() {
                break;
            }
        }
    }

    #[test]
    fn ray_test() {
        use super::{Square, RAY_BETWEEN};
        use crate::chess::squareset::SquareSet;
        assert_eq!(RAY_BETWEEN[Square::A1][Square::A1], SquareSet::EMPTY);
        assert_eq!(RAY_BETWEEN[Square::A1][Square::B1], SquareSet::EMPTY);
        assert_eq!(RAY_BETWEEN[Square::A1][Square::C1], Square::B1.as_set());
        assert_eq!(
            RAY_BETWEEN[Square::A1][Square::D1],
            Square::B1.as_set() | Square::C1.as_set()
        );
        assert_eq!(RAY_BETWEEN[Square::B1][Square::D1], Square::C1.as_set());
        assert_eq!(RAY_BETWEEN[Square::D1][Square::B1], Square::C1.as_set());

        for from in Square::all() {
            for to in Square::all() {
                assert_eq!(RAY_BETWEEN[from][to], RAY_BETWEEN[to][from]);
            }
        }
    }

    #[test]
    fn ray_diag_test() {
        use super::{Square, RAY_BETWEEN};
        let ray = RAY_BETWEEN[Square::B5][Square::E8];
        assert_eq!(ray, Square::C6.as_set() | Square::D7.as_set());
    }
}
