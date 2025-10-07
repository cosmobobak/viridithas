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
        CHESS960,
        board::Board,
        chessmove::{Move, MoveFlags},
        magic::{
            BISHOP_ATTACKS, BISHOP_REL_BITS, BISHOP_TABLE, ROOK_ATTACKS, ROOK_REL_BITS, ROOK_TABLE,
            bishop_attacks_on_the_fly, rook_attacks_on_the_fly, set_occupancy,
        },
        piece::{Black, Col, Colour, PieceType, White},
        squareset::SquareSet,
        types::{Rank, Square},
    },
};

pub const MAX_POSITION_MOVES: usize = 218;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoveListEntry {
    pub score: i32,
    pub mov: Move,
}

#[derive(Clone, Debug)]
pub struct MoveList {
    inner: ArrayVec<MoveListEntry, MAX_POSITION_MOVES>,
}

impl MoveList {
    pub fn new() -> Self {
        Self {
            inner: ArrayVec::new(),
        }
    }

    fn push(&mut self, m: Move) {
        self.inner.push(MoveListEntry { mov: m, score: 0 });
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

pub static RAY_INTERSECTING: [[SquareSet; 64]; 64] = {
    let mut res = [[SquareSet::EMPTY; 64]; 64];
    let mut from = Square::A1;
    loop {
        let mut to = Square::A1;
        loop {
            res[from.index()][to.index()] =
                in_between(from, to).union(from.as_set()).union(to.as_set());
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

pub static RAY_FULL: [[SquareSet; 64]; 64] = {
    // cache these to accelerate consteval
    let mut rook_table = [SquareSet::EMPTY; 64];
    let mut bishop_table = [SquareSet::EMPTY; 64];
    let mut from = Square::A1;
    loop {
        rook_table[from as usize] = rook_attacks_on_the_fly(from, SquareSet::EMPTY);
        bishop_table[from as usize] = bishop_attacks_on_the_fly(from, SquareSet::EMPTY);
        let Some(next) = from.add(1) else {
            break;
        };
        from = next;
    }

    let mut res = [[SquareSet::EMPTY; 64]; 64];
    let mut from = Square::A1;

    loop {
        let from_mask = from.as_set();
        let rook_attacks = rook_table[from as usize];
        let bishop_attacks = bishop_table[from as usize];

        let mut to = Square::A1;
        loop {
            let to_mask = to.as_set();
            if from as usize == to as usize {
                // do nothing
            } else if rook_attacks.contains_square(to) {
                res[from as usize][to as usize] = SquareSet::intersection(
                    rook_table[from as usize].union(from_mask),
                    rook_table[to as usize].union(to_mask),
                );
            } else if bishop_attacks.contains_square(to) {
                res[from as usize][to as usize] = SquareSet::intersection(
                    bishop_table[from as usize].union(from_mask),
                    bishop_table[to as usize].union(to_mask),
                );
            }

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
    // const _INDEX_LEGAL: () = assert!(1 << BISHOP_REL_BITS == BISHOP_ATTACKS[0].len());
    let entry = &BISHOP_TABLE[sq];
    let relevant_blockers = blockers & entry.mask;
    let data = relevant_blockers.inner().wrapping_mul(entry.magic);
    // BISHOP_REL_BITS is 9, so this shift is by 55.
    let idx = (data >> (64 - BISHOP_REL_BITS)) as usize;
    // SAFETY: The largest value we can obtain from (data >> 55)
    // is u64::MAX >> 55, which is 511 (0x1FF). BISHOP_ATTACKS[sq]
    // is 512 elements long, so this is always in bounds.
    unsafe { *BISHOP_ATTACKS[sq].get_unchecked(idx) }
}
#[allow(clippy::cast_possible_truncation)]
pub fn rook_attacks(sq: Square, blockers: SquareSet) -> SquareSet {
    // const _INDEX_LEGAL: () = assert!(1 << ROOK_REL_BITS == ROOK_ATTACKS[0].len());
    let entry = &ROOK_TABLE[sq];
    let relevant_blockers = blockers & entry.mask;
    let data = relevant_blockers.inner().wrapping_mul(entry.magic);
    // ROOK_REL_BITS is 12, so this shift is by 52.
    let idx = (data >> (64 - ROOK_REL_BITS)) as usize;
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
pub fn pawn_attacks_by(bb: SquareSet, colour: Colour) -> SquareSet {
    if colour == Colour::White {
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
        #![allow(clippy::useless_let_if_seq)]

        use PieceType::{Bishop, Knight, Pawn, Queen, Rook};

        let bbs = &self.state.bbs;
        let our_pawns = bbs.pieces[Pawn] & bbs.colours[C::COLOUR];
        let valid_targets = bbs.colours[!C::COLOUR] & valid_target_squares;
        let promo_rank = [SquareSet::RANK_7, SquareSet::RANK_2][C::COLOUR];

        let attacking_west;
        let attacking_east;

        // to determine which pawns can capture,
        // we shift the opponent's pieces backwards and find the intersection.
        if C::WHITE {
            attacking_west = valid_targets.south_east_one() & our_pawns;
            attacking_east = valid_targets.south_west_one() & our_pawns;
        } else {
            attacking_west = valid_targets.north_east_one() & our_pawns;
            attacking_east = valid_targets.north_west_one() & our_pawns;
        }

        for from in attacking_west & !promo_rank {
            // SAFETY: masking guarantees a valid square
            let to = unsafe { from.add_unchecked(C::PAWN_LEFT_OFFSET) };
            move_list.push(Move::new(from, to));
        }

        for from in attacking_east & !promo_rank {
            // SAFETY: masking guarantees a valid square
            let to = unsafe { from.add_unchecked(C::PAWN_RIGHT_OFFSET) };
            move_list.push(Move::new(from, to));
        }

        for from in attacking_west & promo_rank {
            // SAFETY: masking guarantees a valid square
            let to = unsafe { from.add_unchecked(C::PAWN_LEFT_OFFSET) };
            // in quiescence search, we only generate promotions to queen.
            if Mode::SKIP_QUIETS {
                move_list.push(Move::new_with_promo(from, to, Queen));
            } else {
                for promo in [Queen, Rook, Bishop, Knight] {
                    move_list.push(Move::new_with_promo(from, to, promo));
                }
            }
        }

        for from in attacking_east & promo_rank {
            // SAFETY: masking guarantees a valid square
            let to = unsafe { from.add_unchecked(C::PAWN_RIGHT_OFFSET) };
            // in quiescence search, we only generate promotions to queen.
            if Mode::SKIP_QUIETS {
                move_list.push(Move::new_with_promo(from, to, Queen));
            } else {
                for promo in [Queen, Rook, Bishop, Knight] {
                    move_list.push(Move::new_with_promo(from, to, promo));
                }
            }
        }
    }

    fn generate_ep<C: Col>(&self, move_list: &mut MoveList) {
        let Some(ep_sq) = self.state.ep_square else {
            return;
        };

        let ep_bb = ep_sq.as_set();
        let our_pawns = self.state.bbs.pieces[PieceType::Pawn] & self.state.bbs.colours[C::COLOUR];
        let attacks = if C::WHITE {
            ep_bb.south_east_one() | ep_bb.south_west_one()
        } else {
            ep_bb.north_east_one() | ep_bb.north_west_one()
        } & our_pawns;

        for from_sq in attacks {
            move_list.push(Move::new_with_flags(from_sq, ep_sq, MoveFlags::EnPassant));
        }
    }

    fn generate_pawn_forward<C: Col>(
        &self,
        move_list: &mut MoveList,
        valid_target_squares: SquareSet,
    ) {
        use PieceType::{Bishop, Knight, Pawn, Queen, Rook};
        let bbs = &self.state.bbs;
        let promo_rank = [SquareSet::RANK_7, SquareSet::RANK_2][C::COLOUR];
        let start_rank = [SquareSet::RANK_2, SquareSet::RANK_7][C::COLOUR];
        let our_pawns = bbs.pieces[Pawn] & bbs.colours[C::COLOUR];
        let empty = bbs.empty();

        if C::WHITE {
            let shifted_valid_squares = valid_target_squares.south_one();
            let pushable_pawns = our_pawns & empty.south_one();
            let double_pushable_pawns = pushable_pawns & empty.south_one().south_one() & start_rank;
            let promoting_pawns = pushable_pawns & promo_rank;

            let from_mask = pushable_pawns & !promoting_pawns & shifted_valid_squares;
            let to_mask = from_mask.north_one();
            for (from, to) in from_mask.into_iter().zip(to_mask) {
                move_list.push(Move::new(from, to));
            }
            let from_mask = double_pushable_pawns & valid_target_squares.south_one().south_one();
            let to_mask = from_mask.north_one().north_one();
            for (from, to) in from_mask.into_iter().zip(to_mask) {
                move_list.push(Move::new(from, to));
            }
            let from_mask = promoting_pawns & shifted_valid_squares;
            let to_mask = from_mask.north_one();
            for (from, to) in from_mask.into_iter().zip(to_mask) {
                for promo in [Queen, Knight, Rook, Bishop] {
                    move_list.push(Move::new_with_promo(from, to, promo));
                }
            }
        } else {
            let shifted_valid_squares = valid_target_squares.north_one();
            let pushable_pawns = our_pawns & empty.north_one();
            let double_pushable_pawns = pushable_pawns & empty.north_one().north_one() & start_rank;
            let promoting_pawns = pushable_pawns & promo_rank;

            let from_mask = pushable_pawns & !promoting_pawns & shifted_valid_squares;
            let to_mask = from_mask.south_one();
            for (from, to) in from_mask.into_iter().zip(to_mask) {
                move_list.push(Move::new(from, to));
            }
            let from_mask = double_pushable_pawns & valid_target_squares.north_one().north_one();
            let to_mask = from_mask.south_one().south_one();
            for (from, to) in from_mask.into_iter().zip(to_mask) {
                move_list.push(Move::new(from, to));
            }
            let from_mask = promoting_pawns & shifted_valid_squares;
            let to_mask = from_mask.south_one();
            for (from, to) in from_mask.into_iter().zip(to_mask) {
                for promo in [Queen, Knight, Rook, Bishop] {
                    move_list.push(Move::new_with_promo(from, to, promo));
                }
            }
        }
    }

    fn generate_forward_promos<C: Col, Mode: MoveGenMode>(
        &self,
        move_list: &mut MoveList,
        valid_target_squares: SquareSet,
    ) {
        use PieceType::{Bishop, Knight, Pawn, Queen, Rook};
        let bbs = &self.state.bbs;
        let promo_rank = if C::WHITE {
            SquareSet::RANK_7
        } else {
            SquareSet::RANK_2
        };
        let shifted_empty_squares = if C::WHITE {
            bbs.empty() >> 8
        } else {
            bbs.empty() << 8
        };
        let shifted_valid_squares = if C::WHITE {
            valid_target_squares >> 8
        } else {
            valid_target_squares << 8
        };
        let our_pawns = bbs.pieces[Pawn] & bbs.colours[C::COLOUR];
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
                move_list.push(Move::new_with_promo(from, to, Queen));
            } else {
                for promo in [Queen, Knight, Rook, Bishop] {
                    move_list.push(Move::new_with_promo(from, to, promo));
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
        use PieceType::{Bishop, King, Knight, Queen, Rook};
        #[cfg(debug_assertions)]
        self.check_validity().unwrap();

        let bbs = &self.state.bbs;
        let our_pieces = bbs.colours[C::COLOUR];
        let their_pieces = bbs.colours[!C::COLOUR];
        let freespace = !(our_pieces | their_pieces);
        let our_king = bbs.pieces[King] & our_pieces;
        debug_assert_eq!(our_king.count(), 1);
        let our_king_sq = our_king.first().unwrap();

        if self.state.threats.checkers.count() > 1 {
            // we're in double-check, so we can only move the king.
            let moves = king_attacks(our_king_sq) & !self.state.threats.all;
            for to in moves & (their_pieces | freespace) {
                move_list.push(Move::new(our_king_sq, to));
            }
            return;
        }

        let valid_target_squares = if self.in_check() {
            RAY_INTERSECTING[our_king_sq][self.state.threats.checkers.first().unwrap()]
        } else {
            SquareSet::FULL
        };

        self.generate_pawn_forward::<C>(move_list, valid_target_squares);
        self.generate_pawn_caps::<C, AllMoves>(move_list, valid_target_squares);
        self.generate_ep::<C>(move_list);

        // knights
        let our_knights = bbs.pieces[Knight] & our_pieces;
        for sq in our_knights {
            let moves = knight_attacks(sq) & valid_target_squares;
            for to in moves & (their_pieces | freespace) {
                move_list.push(Move::new(sq, to));
            }
        }

        // kings
        let moves = king_attacks(our_king_sq) & !self.state.threats.all;
        for to in moves & (their_pieces | freespace) {
            move_list.push(Move::new(our_king_sq, to));
        }

        // bishops and queens
        let our_diagonal_sliders = (bbs.pieces[Queen] | bbs.pieces[Bishop]) & our_pieces;
        let blockers = bbs.occupied();
        for sq in our_diagonal_sliders {
            let moves = bishop_attacks(sq, blockers) & valid_target_squares;
            for to in moves & (their_pieces | freespace) {
                move_list.push(Move::new(sq, to));
            }
        }

        // rooks and queens
        let our_orthogonal_sliders = (bbs.pieces[Queen] | bbs.pieces[Rook]) & our_pieces;
        for sq in our_orthogonal_sliders {
            let moves = rook_attacks(sq, blockers) & valid_target_squares;
            for to in moves & (their_pieces | freespace) {
                move_list.push(Move::new(sq, to));
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
        use PieceType::{Bishop, King, Knight, Queen, Rook};
        #[cfg(debug_assertions)]
        self.check_validity().unwrap();

        let bbs = &self.state.bbs;
        let our_pieces = bbs.colours[C::COLOUR];
        let their_pieces = bbs.colours[!C::COLOUR];
        let our_king = bbs.pieces[King] & our_pieces;
        debug_assert_eq!(our_king.count(), 1);
        let our_king_sq = our_king.first().unwrap();

        if self.state.threats.checkers.count() > 1 {
            // we're in double-check, so we can only move the king.
            let moves = king_attacks(our_king_sq) & !self.state.threats.all;
            for to in moves & their_pieces {
                move_list.push(Move::new(our_king_sq, to));
            }
            return;
        }

        let valid_target_squares = if self.in_check() {
            RAY_INTERSECTING[our_king_sq][self.state.threats.checkers.first().unwrap()]
        } else {
            SquareSet::FULL
        };

        // promotions
        self.generate_forward_promos::<C, Mode>(move_list, valid_target_squares);

        // pawn captures and capture promos
        self.generate_pawn_caps::<C, Mode>(move_list, valid_target_squares);
        self.generate_ep::<C>(move_list);

        // knights
        let our_knights = bbs.pieces[Knight] & our_pieces;
        for sq in our_knights {
            let moves = knight_attacks(sq) & valid_target_squares;
            for to in moves & their_pieces {
                move_list.push(Move::new(sq, to));
            }
        }

        // kings
        let moves = king_attacks(our_king_sq) & !self.state.threats.all;
        for to in moves & their_pieces {
            move_list.push(Move::new(our_king_sq, to));
        }

        // bishops and queens
        let our_diagonal_sliders = (bbs.pieces[Queen] | bbs.pieces[Bishop]) & our_pieces;
        let blockers = bbs.occupied();
        for sq in our_diagonal_sliders {
            let moves = bishop_attacks(sq, blockers) & valid_target_squares;
            for to in moves & their_pieces {
                move_list.push(Move::new(sq, to));
            }
        }

        // rooks and queens
        let our_orthogonal_sliders = (bbs.pieces[Queen] | bbs.pieces[Rook]) & our_pieces;
        for sq in our_orthogonal_sliders {
            let moves = rook_attacks(sq, blockers) & valid_target_squares;
            for to in moves & their_pieces {
                move_list.push(Move::new(sq, to));
            }
        }
    }

    fn generate_castling_moves_for<C: Col>(&self, move_list: &mut MoveList) {
        let occupied = self.state.bbs.occupied();

        if CHESS960.load(Ordering::Relaxed) {
            let king_sq = self.state.bbs.king_sq(C::COLOUR);
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
                && occupied & k_freespace == SquareSet::EMPTY
                && {
                    let got_attacked_king = self.sq_attacked_by::<C::Opposite>(from);
                    cache = Some(got_attacked_king);
                    !got_attacked_king
                }
                && !self.sq_attacked_by::<C::Opposite>(k_thru)
            {
                move_list.push(Move::new_with_flags(from, k_to, MoveFlags::Castle));
            }

            if q_perm.is_some()
                && occupied & q_freespace == SquareSet::EMPTY
                && !cache.unwrap_or_else(|| self.sq_attacked_by::<C::Opposite>(from))
                && !self.sq_attacked_by::<C::Opposite>(q_thru)
            {
                move_list.push(Move::new_with_flags(from, q_to, MoveFlags::Castle));
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
        let intersection =
            relevant_occupied & (king_path | rook_path | king_dst.as_set() | rook_dst.as_set());
        if intersection == SquareSet::EMPTY && !self.any_attacked(king_path, C::Opposite::COLOUR) {
            move_list.push(Move::new_with_flags(
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
        use PieceType::Pawn;
        let bbs = &self.state.bbs;
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
            bbs.empty() >> 8
        } else {
            bbs.empty() << 8
        };
        let double_shifted_empty_squares = if C::WHITE {
            bbs.empty() >> 16
        } else {
            bbs.empty() << 16
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
        let our_pawns = bbs.pieces[Pawn] & bbs.colours[C::COLOUR];
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
            move_list.push(Move::new(from, to));
        }
        let from_mask = double_pushable_pawns & double_shifted_valid_squares;
        let to_mask = if C::WHITE {
            from_mask.north_one().north_one()
        } else {
            from_mask.south_one().south_one()
        };
        for (from, to) in from_mask.into_iter().zip(to_mask) {
            move_list.push(Move::new(from, to));
        }
    }

    fn generate_quiets_for<C: Col>(&self, move_list: &mut MoveList) {
        use PieceType::{Bishop, Knight, Queen, Rook};

        let bbs = &self.state.bbs;
        let our_pieces = bbs.colours[C::COLOUR];
        let their_pieces = bbs.colours[!C::COLOUR];
        let blockers = our_pieces | their_pieces;
        let freespace = !blockers;
        let our_king = bbs.pieces[PieceType::King] & our_pieces;
        debug_assert_eq!(our_king.count(), 1);
        let our_king_sq = our_king.first().unwrap();

        if self.state.threats.checkers.count() > 1 {
            // we're in double-check, so we can only move the king.
            let moves = king_attacks(our_king_sq) & !self.state.threats.all;
            for to in moves & freespace {
                move_list.push(Move::new(our_king_sq, to));
            }
            return;
        }

        let valid_target_squares = if self.in_check() {
            RAY_BETWEEN[our_king_sq][self.state.threats.checkers.first().unwrap()]
        } else {
            SquareSet::FULL
        };

        // pawns
        self.generate_pawn_quiet::<C>(move_list, valid_target_squares);

        // knights
        let our_knights = bbs.pieces[Knight] & our_pieces;
        for sq in our_knights {
            let moves = knight_attacks(sq) & valid_target_squares;
            for to in moves & !blockers {
                move_list.push(Move::new(sq, to));
            }
        }

        // kings
        let moves = king_attacks(our_king_sq) & !self.state.threats.all;
        for to in moves & !blockers {
            move_list.push(Move::new(our_king_sq, to));
        }

        // bishops and queens
        let our_diagonal_sliders = (bbs.pieces[Queen] | bbs.pieces[Bishop]) & our_pieces;
        for sq in our_diagonal_sliders {
            let moves = bishop_attacks(sq, blockers) & valid_target_squares;
            for to in moves & !blockers {
                move_list.push(Move::new(sq, to));
            }
        }

        // rooks and queens
        let our_orthogonal_sliders = (bbs.pieces[Queen] | bbs.pieces[Rook]) & our_pieces;
        for sq in our_orthogonal_sliders {
            let moves = rook_attacks(sq, blockers) & valid_target_squares;
            for to in moves & !blockers {
                move_list.push(Move::new(sq, to));
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
                mvs.push(pos.san(m.mov).unwrap());
            }
            mvs.join(", ")
        },
        {
            let mut mvs = Vec::new();
            for m in staged_moves_vec {
                mvs.push(pos.san(m.mov).unwrap());
            }
            mvs.join(", ")
        }
    );

    let mut count = 0;
    for &m in ml.iter_moves() {
        if !pos.is_legal(m) {
            continue;
        }
        pos.make_move_simple(m);
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
            piece::Piece,
            squareset::SquareSet,
            types::Square,
        },
    };

    #[test]
    fn staged_matches_full() {
        let mut pos = Board::default();

        let positions = bench::BENCH_POSITIONS
            .into_iter()
            .chain(["r4rk1/2pb1ppQ/2pp1q2/p1n5/2P1B3/PP2P3/3N1PPP/R4RK1 b - - 0 17"]);

        for fen in positions {
            pos.set_from_fen(fen).unwrap();
            synced_perft(&mut pos, 2);
        }
    }

    #[test]
    fn no_king_into_check() {
        let pos = Board::from_fen("r4rk1/2pb1ppQ/2pp1q2/p1n5/2P1B3/PP2P3/3N1PPP/R4RK1 b - - 0 17")
            .unwrap();

        assert_eq!(
            pos.state.threats.checkers,
            SquareSet::from_square(Square::H7)
        );

        assert!(pos.state.threats.all.contains_square(Square::H8));
        assert!(pos.state.threats.all.contains_square(Square::H5));

        let mut ml = MoveList::new();

        pos.generate_captures::<AllMoves>(&mut ml);
        pos.generate_quiets(&mut ml);

        for m in ml.iter_moves() {
            let Some(Piece::WK) = pos.state.mailbox[m.from()] else {
                continue;
            };
            assert!(!pos.state.threats.all.contains_square(m.to()));
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
            if subset == SquareSet::EMPTY {
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
            if subset == SquareSet::EMPTY {
                break;
            }
        }
    }

    #[test]
    fn ray_test() {
        use super::{RAY_BETWEEN, Square};
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
        use super::{RAY_BETWEEN, Square};
        let ray = RAY_BETWEEN[Square::B5][Square::E8];
        assert_eq!(ray, Square::C6.as_set() | Square::D7.as_set());
    }
}
