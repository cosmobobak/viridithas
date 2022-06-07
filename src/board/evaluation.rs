// The granularity of evaluation in this engine is going to be thousandths of a pawn.

use std::{
    iter::Sum,
    ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign},
};

use crate::{
    board::Board,
    definitions::{
        BB, BISHOP, BLACK, BN, BP, BQ, BR, KNIGHT, MAX_DEPTH, QUEEN, ROOK, WB, WHITE, WN, WP, WQ,
        WR,
    },
    lookups::{file, init_eval_masks, init_passed_isolated_bb, rank},
};

use super::movegen::{bitboards::attacks, BitLoop, BB_NONE};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct S(pub i32, pub i32);

impl Add for S {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0, self.1 + rhs.1)
    }
}
impl Sub for S {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0, self.1 - rhs.1)
    }
}
impl AddAssign for S {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
        self.1 += rhs.1;
    }
}
impl SubAssign for S {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
        self.1 -= rhs.1;
    }
}
impl Neg for S {
    type Output = Self;

    fn neg(self) -> Self {
        Self(-self.0, -self.1)
    }
}
impl Mul<i32> for S {
    type Output = Self;

    fn mul(self, rhs: i32) -> Self {
        Self(self.0 * rhs, self.1 * rhs)
    }
}
impl Sum for S {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self(0, 0), |acc, x| acc + x)
    }
}

impl S {
    pub fn value(self, phase: i32) -> i32 {
        lerp(self.0, self.1, phase)
    }
}

// These piece values are taken from PeSTO (which in turn took them from RofChade 1.0).
pub const PAWN_VALUE: S = S(82, 94);
pub const KNIGHT_VALUE: S = S(337, 281);
pub const BISHOP_VALUE: S = S(365, 297);
pub const ROOK_VALUE: S = S(477, 512);
pub const QUEEN_VALUE: S = S(1025, 936);

pub const ONE_PAWN: i32 = 100;

/// The value of checkmate.
/// To recover depth-to-mate, we subtract depth (ply) from this value.
/// e.g. if white has a mate in two ply, the output from a depth-5 search will be
/// `3_000_000 - 2 = 2_999_998`.
pub const MATE_SCORE: i32 = 3_000_000;

/// A threshold over which scores must be mate.
#[allow(clippy::cast_possible_truncation)]
pub const IS_MATE_SCORE: i32 = MATE_SCORE - MAX_DEPTH as i32;

/// The value of a draw.
pub const DRAW_SCORE: i32 = 0;

#[rustfmt::skip]
pub static PIECE_VALUES: [S; 13] = [
    S(0, 0),
    PAWN_VALUE, KNIGHT_VALUE, BISHOP_VALUE, ROOK_VALUE, QUEEN_VALUE, S(0, 0),
    PAWN_VALUE, KNIGHT_VALUE, BISHOP_VALUE, ROOK_VALUE, QUEEN_VALUE, S(0, 0),
];

/// The malus applied when a pawn has no pawns of its own colour to the left or right.
pub const ISOLATED_PAWN_MALUS: S = S(ONE_PAWN / 4, ONE_PAWN / 3);

/// The malus applied when two (or more) pawns of a colour are on the same file.
pub const DOUBLED_PAWN_MALUS: S = S(3 * ONE_PAWN / 8, ONE_PAWN / 2);

/// The bonus granted for having two bishops.
pub const BISHOP_PAIR_BONUS: S = S(ONE_PAWN / 4, ONE_PAWN / 3);

/// The bonus for having a rook on an open file.
pub const ROOK_OPEN_FILE_BONUS: S = S(ONE_PAWN / 10, 0);
/// The bonus for having a rook on a semi-open file.
pub const ROOK_HALF_OPEN_FILE_BONUS: S = S(ONE_PAWN / 20, 0);
/// The bonus for having a queen on an open file.
pub const QUEEN_OPEN_FILE_BONUS: S = S(ONE_PAWN / 20, 0);
/// The bonus for having a queen on a semi-open file.
pub const QUEEN_HALF_OPEN_FILE_BONUS: S = S(ONE_PAWN / 40, 0);

/// The bonus granted for having more pawns when you have knights on the board.
// pub const KNIGHT_PAWN_BONUS: i32 = PAWN_VALUE / 15;

/// The bonus for having IDX pawns in front of the king.
pub static PAWN_SHIELD_BONUS: [S; 4] = [S(0, 0), S(5, 6), S(17, 10), S(20, 15)];

/// The bonus for xraying the king with a piece.
static XRAY_ATTACKERS_BONUS: [S; 20] = [
    S(5, 5),
    S(20, 20),
    S(45, 45),
    S(70, 70),
    S(90, 90),
    S(200, 200),
    S(200, 200),
    S(200, 200),
    S(200, 200),
    S(200, 200),
    S(200, 200),
    S(200, 200),
    S(200, 200),
    S(200, 200),
    S(200, 200),
    S(200, 200),
    S(200, 200),
    S(200, 200),
    S(200, 200),
    S(200, 200),
];

// MADCHESS nonlinear mobility eval tables.
// (10 * x Pow 0.5) - 15;
// (20 * x Pow 0.5) - 30;
static KNIGHT_MOBILITY: [S; 9] = [
    S(-15, -30),
    S(-5, -10),
    S(-1, -2),
    S(2, 4),
    S(5, 10),
    S(7, 14),
    S(9, 18),
    S(11, 22),
    S(13, 26),
];

// (14 * x Pow 0.5) - 25;
// (28 * x Pow 0.5) - 50;
static BISHOP_MOBILITY: [S; 14] = [
    S(-25, -50),
    S(-11, -22),
    S(-6, -11),
    S(-1, -2),
    S(3, 6),
    S(6, 12),
    S(9, 18),
    S(12, 24),
    S(14, 29),
    S(17, 34),
    S(19, 38),
    S(21, 42),
    S(23, 46),
    S(25, 50),
];

// (6 * x Pow 0.5) - 10;
// (28 * x Pow 0.5) - 50;
static ROOK_MOBILITY: [S; 15] = [
    S(-10, -50),
    S(-4, -22),
    S(-2, -11),
    S(0, -2),
    S(2, 6),
    S(3, 12),
    S(4, 18),
    S(5, 24),
    S(6, 29),
    S(8, 34),
    S(8, 38),
    S(9, 42),
    S(10, 46),
    S(11, 50),
    S(12, 54),
];

// (4 * x Pow 0.5) - 10;
// (20 * x Pow 0.5) - 50;
static QUEEN_MOBILITY: [S; 28] = [
    S(-10, -50),
    S(-6, -30),
    S(-5, -22),
    S(-4, -16),
    S(-2, -10),
    S(-2, -6),
    S(-1, -2),
    S(0, 2),
    S(1, 6),
    S(2, 10),
    S(2, 13),
    S(3, 16),
    S(3, 19),
    S(4, 22),
    S(4, 24),
    S(5, 27),
    S(6, 30),
    S(6, 32),
    S(6, 34),
    S(7, 37),
    S(7, 39),
    S(8, 41),
    S(8, 43),
    S(9, 45),
    S(9, 47),
    S(10, 50),
    S(10, 51),
    S(10, 53),
];

/// A threshold over which we will not bother evaluating more than material and PSTs.
pub const LAZY_THRESHOLD_1: i32 = 14_00;

const PAWN_PHASE: i32 = 1;
const KNIGHT_PHASE: i32 = 10;
const BISHOP_PHASE: i32 = 10;
const ROOK_PHASE: i32 = 20;
const QUEEN_PHASE: i32 = 40;
const TOTAL_PHASE: i32 =
    16 * PAWN_PHASE + 4 * KNIGHT_PHASE + 4 * BISHOP_PHASE + 4 * ROOK_PHASE + 2 * QUEEN_PHASE;

#[allow(dead_code)]
pub static RANK_BB: [u64; 8] = init_eval_masks().0;
pub static FILE_BB: [u64; 8] = init_eval_masks().1;

pub static WHITE_PASSED_BB: [u64; 64] = init_passed_isolated_bb().0;
pub static BLACK_PASSED_BB: [u64; 64] = init_passed_isolated_bb().1;

pub static ISOLATED_BB: [u64; 64] = init_passed_isolated_bb().2;

/// The bonus applied when a pawn has no pawns of the opposite colour ahead of it, or to the left or right, scaled by the rank that the pawn is on.
/// values from VICE.
pub static PASSED_PAWN_BONUS: [S; 8] = [
    S(0, 0), // illegal
    S(5, 5),
    S(10, 10),
    S(20, 20),
    S(35, 35),
    S(60, 60),
    S(100, 100),
    S(0, 0), // illegal
];

/// `game_phase` computes a number between 0 and 256, which is the phase of the game.
/// 0 is the opening, 256 is the endgame.
#[allow(clippy::many_single_char_names)]
pub const fn game_phase(p: u8, n: u8, b: u8, r: u8, q: u8) -> i32 {
    let mut phase = TOTAL_PHASE;
    phase -= PAWN_PHASE * p as i32;
    phase -= KNIGHT_PHASE * n as i32;
    phase -= BISHOP_PHASE * b as i32;
    phase -= ROOK_PHASE * r as i32;
    phase -= QUEEN_PHASE * q as i32;
    phase
}

/// `lerp` linearly interpolates between `a` and `b` by `t`.
/// `t` is between 0 and 256.
pub fn lerp(mg: i32, eg: i32, t: i32) -> i32 {
    // debug_assert!((0..=256).contains(&t));
    let t = t.min(256);
    mg * (256 - t) / 256 + eg * t / 256
}

pub const fn is_mate_score(score: i32) -> bool {
    score.abs() >= IS_MATE_SCORE
}

/// A struct that holds all the terms in the evaluation function, intended to be used by the
/// tuner for optimising the evaluation function.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvalVector {
    /// Whether this position is valid to use for tuning (positions should be quiescent, amongst other considerations).
    pub valid: bool,
    /// The aggregated material and PST term, as we do not plan to tune this.
    pub material_pst: i32,
    /// The relative bishop pair count. (can only be -1, 0 or 1)
    pub bishop_pair: i32,
    /// The relative number of passed pawns by rank.
    pub passed_pawns_by_rank: [i32; 8],
    /// The relative number of isolated pawns.
    pub isolated_pawns: i32,
    /// The relative number of doubled pawns.
    pub doubled_pawns: i32,
    /// The relative knight mobility count.
    pub knight_mobility: i32,
    /// The relative bishop mobility count.
    pub bishop_mobility: i32,
    /// The relative rook mobility count.
    pub rook_mobility: i32,
    /// The relative queen mobility count.
    pub queen_mobility: i32,
    /// The relative king mobility count.
    pub king_mobility: i32,
    /// The relative shield count.
    pub pawn_shield: i32,
    /// The relative number of rooks on open files.
    pub open_rooks: i32,
    /// The relative number of rooks on semi-open files.
    pub half_open_rooks: i32,
    /// The turn (1 or -1)
    pub turn: i32,
}

#[allow(dead_code)]
impl EvalVector {
    pub fn csvify(&self) -> String {
        let csv = format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            self.bishop_pair,
            self.passed_pawns_by_rank[1],
            self.passed_pawns_by_rank[2],
            self.passed_pawns_by_rank[3],
            self.passed_pawns_by_rank[4],
            self.passed_pawns_by_rank[5],
            self.passed_pawns_by_rank[6],
            self.isolated_pawns,
            self.doubled_pawns,
            self.knight_mobility,
            self.bishop_mobility,
            self.rook_mobility,
            self.queen_mobility,
            self.king_mobility,
            self.pawn_shield,
            self.open_rooks,
            self.half_open_rooks,
            self.turn
        );
        assert_eq!(
            csv.chars().filter(|&c| c == ',').count(),
            Self::header().chars().filter(|&c| c == ',').count()
        );
        csv
    }

    pub const fn header() -> &'static str {
        "bpair,ppr2,ppr3,ppr4,ppr5,ppr6,ppr7,isolated,doubled,n_mob,b_mob,r_mob,q_mob,k_mob,p_shield,turn"
    }
}

impl Board {
    /// Computes a score for the position, from the point of view of the side to move.
    /// This function should strive to be as cheap to call as possible, relying on
    /// incremental updates in make-unmake to avoid recomputation.
    pub fn evaluate(&mut self) -> i32 {
        #![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

        if !self.pieces.any_pawns() && self.is_material_draw() {
            return if self.side == WHITE {
                DRAW_SCORE
            } else {
                -DRAW_SCORE
            };
        }
        let material = self.material[WHITE as usize] - self.material[BLACK as usize];
        let pst = self.pst_vals;

        let mut score = material + pst;

        // if the score is already outwith a threshold, we can stop here.
        if score.0.abs() > LAZY_THRESHOLD_1 {
            // score = self.clamp_score(score);
            if self.side == WHITE {
                return score.value(self.phase());
            }
            return -score.value(self.phase());
        }

        let pawn_val = self.pawn_structure_term(); // INCREMENTAL UPDATE.
        let bishop_pair_val = self.bishop_pair_term();
        let mobility_val = self.mobility();
        let king_safety_val = self.pawn_shield_term();
        let rook_open_file_val = self.rook_open_file_term();
        let queen_open_file_val = self.queen_open_file_term();
        let king_xray_val = self.king_xray_term();

        score += pawn_val;
        score += bishop_pair_val;
        score += mobility_val;
        score += king_safety_val;
        score += rook_open_file_val;
        score += queen_open_file_val;
        score += king_xray_val;

        let score = score.value(self.phase());

        let score = self.clamp_score(score);

        if self.side == WHITE {
            score
        } else {
            -score
        }
    }

    fn unwinnable_for<const SIDE: u8>(&self) -> bool {
        assert!(
            SIDE == WHITE || SIDE == BLACK,
            "unwinnable_for called with invalid side"
        );

        if SIDE == WHITE {
            if self.major_piece_counts[WHITE as usize] != 0 {
                return false;
            }
            if self.minor_piece_counts[WHITE as usize] > 1 {
                return false;
            }
            if self.num(WP) != 0 {
                return false;
            }
            true
        } else {
            if self.major_piece_counts[BLACK as usize] != 0 {
                return false;
            }
            if self.minor_piece_counts[BLACK as usize] > 1 {
                return false;
            }
            if self.num(BP) != 0 {
                return false;
            }
            true
        }
    }

    const fn is_material_draw(&self) -> bool {
        if self.num(WR) == 0 && self.num(BR) == 0 && self.num(WQ) == 0 && self.num(BQ) == 0 {
            if self.num(WB) == 0 && self.num(BB) == 0 {
                if self.num(WN) < 3 && self.num(BN) < 3 {
                    return true;
                }
            } else if (self.num(WN) == 0
                && self.num(BN) == 0
                && self.num(WB).abs_diff(self.num(BB)) < 2)
                || (self.num(WB) + self.num(WN) == 1 && self.num(BB) + self.num(BN) == 1)
            {
                return true;
            }
        } else if self.num(WQ) == 0 && self.num(BQ) == 0 {
            if self.num(WR) == 1 && self.num(BR) == 1 {
                if (self.num(WN) + self.num(WB)) < 2 && (self.num(BN) + self.num(BB)) < 2 {
                    return true;
                }
            } else if self.num(WR) == 1 && self.num(BR) == 0 {
                if (self.num(WN) + self.num(WB)) == 0
                    && ((self.num(BN) + self.num(BB)) == 1 || (self.num(BN) + self.num(BB)) == 2)
                {
                    return true;
                }
            } else if self.num(WR) == 0
                && self.num(BR) == 1
                && (self.num(BN) + self.num(BB)) == 0
                && ((self.num(WN) + self.num(WB)) == 1 || (self.num(WN) + self.num(WB)) == 2)
            {
                return true;
            }
        }
        false
    }

    fn clamp_score(&mut self, score: i32) -> i32 {
        // if we can't win with our material, we clamp the eval to zero.
        if score > 0 && self.unwinnable_for::<{ WHITE }>()
            || score < 0 && self.unwinnable_for::<{ BLACK }>()
        {
            0
        } else {
            score
        }
    }

    pub const fn zugzwang_unlikely(&self) -> bool {
        const ENDGAME_PHASE: i32 = game_phase(3, 0, 0, 2, 0);
        self.big_piece_counts[self.side as usize] > 0 && self.phase() < ENDGAME_PHASE
    }

    fn bishop_pair_term(&self) -> S {
        let w_count = self.num(WB);
        let b_count = self.num(BB);
        if w_count == b_count {
            return S(0, 0);
        }
        if w_count >= 2 {
            return BISHOP_PAIR_BONUS;
        }
        if b_count >= 2 {
            return -BISHOP_PAIR_BONUS;
        }
        S(0, 0)
    }

    fn pawn_shield_term(&self) -> S {
        #![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

        let white_kingloc = self.king_sq(WHITE);
        let black_kingloc = self.king_sq(BLACK);

        let mut white_shield = 0;
        for loc in white_kingloc + 7..=white_kingloc + 9 {
            if self.piece_at(loc) == WP {
                white_shield += 1;
            }
        }

        let mut black_shield = 0;
        for loc in black_kingloc - 9..=black_kingloc - 7 {
            if self.piece_at(loc) == BP {
                black_shield += 1;
            }
        }

        PAWN_SHIELD_BONUS[white_shield] - PAWN_SHIELD_BONUS[black_shield]
    }

    fn pawn_structure_term(&self) -> S {
        static DOUBLED_PAWN_MAPPING: [i32; 7] = [0, 0, 1, 2, 3, 4, 5];
        let mut w_score = S(0, 0);
        let (white_pawns, black_pawns) =
            (self.pieces.pawns::<true>(), self.pieces.pawns::<false>());
        for &white_pawn_loc in self.piece_lists[WP as usize].iter() {
            if unsafe { *ISOLATED_BB.get_unchecked(white_pawn_loc as usize) } & white_pawns == 0 {
                w_score -= ISOLATED_PAWN_MALUS;
            }

            if unsafe { *WHITE_PASSED_BB.get_unchecked(white_pawn_loc as usize) } & black_pawns == 0
            {
                let rank = rank(white_pawn_loc) as usize;
                w_score += unsafe { *PASSED_PAWN_BONUS.get_unchecked(rank) };
            }
        }

        let mut b_score = S(0, 0);
        for &black_pawn_loc in self.piece_lists[BP as usize].iter() {
            if unsafe { *ISOLATED_BB.get_unchecked(black_pawn_loc as usize) } & black_pawns == 0 {
                b_score -= ISOLATED_PAWN_MALUS;
            }

            if unsafe { *BLACK_PASSED_BB.get_unchecked(black_pawn_loc as usize) } & white_pawns == 0
            {
                let rank = rank(black_pawn_loc) as usize;
                b_score += unsafe { *PASSED_PAWN_BONUS.get_unchecked(7 - rank) };
            }
        }

        for &file_mask in &FILE_BB {
            let pawns_in_file = (file_mask & white_pawns).count_ones() as usize;
            let multiplier = unsafe { *DOUBLED_PAWN_MAPPING.get_unchecked(pawns_in_file) };
            w_score -= DOUBLED_PAWN_MALUS * multiplier;
            let pawns_in_file = (file_mask & black_pawns).count_ones() as usize;
            let multiplier = unsafe { *DOUBLED_PAWN_MAPPING.get_unchecked(pawns_in_file) };
            b_score -= DOUBLED_PAWN_MALUS * multiplier;
        }

        w_score - b_score
    }

    fn is_file_open(&self, file: u8) -> bool {
        let mask = FILE_BB[file as usize];
        let pawns = self.pieces.pawns::<true>() | self.pieces.pawns::<false>();
        (mask & pawns) == 0
    }

    fn is_file_halfopen<const SIDE: u8>(&self, file: u8) -> bool {
        let mask = FILE_BB[file as usize];
        let pawns = if SIDE == WHITE {
            self.pieces.pawns::<true>()
        } else {
            self.pieces.pawns::<false>()
        };
        (mask & pawns) == 0
    }

    fn rook_open_file_term(&self) -> S {
        let mut score = S(0, 0);
        for &rook_sq in self.piece_lists[WR as usize].iter() {
            let file = file(rook_sq);
            if self.is_file_open(file) {
                score += ROOK_OPEN_FILE_BONUS;
            } else if self.is_file_halfopen::<WHITE>(file) {
                score += ROOK_HALF_OPEN_FILE_BONUS;
            }
        }
        for &rook_sq in self.piece_lists[BR as usize].iter() {
            let file = file(rook_sq);
            if self.is_file_open(file) {
                score -= ROOK_OPEN_FILE_BONUS;
            } else if self.is_file_halfopen::<BLACK>(file) {
                score -= ROOK_HALF_OPEN_FILE_BONUS;
            }
        }
        score
    }

    fn queen_open_file_term(&self) -> S {
        let mut score = S(0, 0);
        for &queen_sq in self.piece_lists[WQ as usize].iter() {
            let file = file(queen_sq);
            if self.is_file_open(file) {
                score += QUEEN_OPEN_FILE_BONUS;
            } else if self.is_file_halfopen::<WHITE>(file) {
                score += QUEEN_HALF_OPEN_FILE_BONUS;
            }
        }
        for &queen_sq in self.piece_lists[BQ as usize].iter() {
            let file = file(queen_sq);
            if self.is_file_open(file) {
                score -= QUEEN_OPEN_FILE_BONUS;
            } else if self.is_file_halfopen::<BLACK>(file) {
                score -= QUEEN_HALF_OPEN_FILE_BONUS;
            }
        }
        score
    }

    /// `phase` computes a number between 0 and 256, which is the phase of the game. 0 is the opening, 256 is the endgame.
    pub const fn phase(&self) -> i32 {
        // todo: this can be incrementally updated.
        let pawns = self.num(WP) + self.num(BP);
        let knights = self.num(WN) + self.num(BN);
        let bishops = self.num(WB) + self.num(BB);
        let rooks = self.num(WR) + self.num(BR);
        let queens = self.num(WQ) + self.num(BQ);
        game_phase(pawns, knights, bishops, rooks, queens)
    }

    fn mobility(&mut self) -> S {
        let mut mob_score = S(0, 0);
        let white_knight_moves: S = BitLoop::<u8>::new(self.pieces.knights::<true>())
            .map(|sq| KNIGHT_MOBILITY[attacks::<KNIGHT>(sq, BB_NONE).count_ones() as usize])
            .sum();
        let black_knight_moves: S = BitLoop::<u8>::new(self.pieces.knights::<false>())
            .map(|sq| KNIGHT_MOBILITY[attacks::<KNIGHT>(sq, BB_NONE).count_ones() as usize])
            .sum();
        mob_score += white_knight_moves - black_knight_moves;
        let white_bishop_moves: S = BitLoop::<u8>::new(self.pieces.bishops::<true>())
            .map(|sq| {
                BISHOP_MOBILITY[attacks::<BISHOP>(sq, self.pieces.occupied()).count_ones() as usize]
            })
            .sum();
        let black_bishop_moves: S = BitLoop::<u8>::new(self.pieces.bishops::<false>())
            .map(|sq| {
                BISHOP_MOBILITY[attacks::<BISHOP>(sq, self.pieces.occupied()).count_ones() as usize]
            })
            .sum();
        mob_score += white_bishop_moves - black_bishop_moves;
        let white_rook_moves: S = BitLoop::<u8>::new(self.pieces.rooks::<true>())
            .map(|sq| {
                ROOK_MOBILITY[attacks::<ROOK>(sq, self.pieces.occupied()).count_ones() as usize]
            })
            .sum();
        let black_rook_moves: S = BitLoop::<u8>::new(self.pieces.rooks::<false>())
            .map(|sq| {
                ROOK_MOBILITY[attacks::<ROOK>(sq, self.pieces.occupied()).count_ones() as usize]
            })
            .sum();
        mob_score += white_rook_moves - black_rook_moves;
        let white_queen_moves: S = BitLoop::<u8>::new(self.pieces.queens::<true>())
            .map(|sq| {
                QUEEN_MOBILITY[attacks::<QUEEN>(sq, self.pieces.occupied()).count_ones() as usize]
            })
            .sum();
        let black_queen_moves: S = BitLoop::<u8>::new(self.pieces.queens::<false>())
            .map(|sq| {
                QUEEN_MOBILITY[attacks::<QUEEN>(sq, self.pieces.occupied()).count_ones() as usize]
            })
            .sum();
        mob_score += white_queen_moves - black_queen_moves;

        if self.side == WHITE {
            mob_score
        } else {
            -mob_score
        }
    }

    pub fn king_xray_term(&self) -> S {
        let black_king = self.king_sq(BLACK);
        let white_king = self.king_sq(WHITE);
        let white_xray_attackers = self.attackers_mask(black_king, WHITE, BB_NONE);
        let black_xray_attackers = self.attackers_mask(white_king, BLACK, BB_NONE);
        let white_xray_attackers_count = white_xray_attackers.count_ones() as usize;
        let black_xray_attackers_count = black_xray_attackers.count_ones() as usize;
        let white_xray_attackers_count_term = XRAY_ATTACKERS_BONUS[white_xray_attackers_count];
        let black_xray_attackers_count_term = XRAY_ATTACKERS_BONUS[black_xray_attackers_count];
        white_xray_attackers_count_term - black_xray_attackers_count_term
    }
}

mod tests {

    #[test]
    fn unwinnable() {
        const FEN: &str = "8/8/8/8/2K2k2/2n2P2/8/8 b - - 1 1";
        crate::magic::initialise();
        let mut board = super::Board::from_fen(FEN).unwrap();
        let eval = board.evaluate();
        assert!(
            eval.abs() == 0,
            "eval is not a draw score ({eval}cp != 0cp) in a position unwinnable for both sides."
        );
    }

    #[test]
    fn turn_equality() {
        const FEN1: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        const FEN2: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1";
        crate::magic::initialise();
        let mut board1 = super::Board::from_fen(FEN1).unwrap();
        let mut board2 = super::Board::from_fen(FEN2).unwrap();
        let eval1 = board1.evaluate();
        let eval2 = board2.evaluate();
        assert_eq!(eval1, -eval2);
    }

    #[test]
    fn startpos_mobility_equality() {
        use crate::board::evaluation::S;
        let mut board = super::Board::default();
        assert_eq!(board.mobility(), S(0, 0));
    }

    #[test]
    fn startpos_eval_equality() {
        let mut board = super::Board::default();
        assert_eq!(board.evaluate(), 0);
    }

    #[test]
    fn startpos_pawn_structure_equality() {
        use crate::board::evaluation::S;
        let board = super::Board::default();
        assert_eq!(board.pawn_structure_term(), S(0, 0));
    }

    #[test]
    fn startpos_pawn_shield_equality() {
        use crate::board::evaluation::S;
        let board = super::Board::default();
        assert_eq!(board.pawn_shield_term(), S(0, 0));
    }

    #[test]
    fn startpos_open_file_equality() {
        use crate::board::evaluation::S;
        let board = super::Board::default();
        let rook_points = board.rook_open_file_term();
        let queen_points = board.queen_open_file_term();
        assert_eq!(rook_points + queen_points, S(0, 0));
    }

    #[test]
    fn double_pawn_eval() {
        use crate::board::evaluation::DOUBLED_PAWN_MALUS;
        let board =
            super::Board::from_fen("rnbqkbnr/pppppppp/8/8/8/5P2/PPPP1PPP/RNBQKBNR w KQkq - 0 1")
                .unwrap();
        let pawn_eval = board.pawn_structure_term();
        assert_eq!(pawn_eval, -DOUBLED_PAWN_MALUS);
        let board =
            super::Board::from_fen("rnbqkbnr/pppppppp/8/8/8/2P2P2/PPP2PPP/RNBQKBNR b KQkq - 0 1")
                .unwrap();
        let pawn_eval = board.pawn_structure_term();
        assert_eq!(pawn_eval, -DOUBLED_PAWN_MALUS * 2);
    }
}
