// The granularity of evaluation in this engine is going to be thousandths of a pawn.

use std::{
    error::Error,
    fmt::Display,
    iter::Sum,
    ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign},
};

use crate::{
    board::Board,
    definitions::{
        BB, BISHOP, BLACK, BN, BP, BQ, BR, KNIGHT, MAX_DEPTH, QUEEN, ROOK, WB, WHITE, WN, WP, WQ,
        WR,
    },
    lookups::{file, init_eval_masks, init_passed_isolated_bb, rank}, piecesquaretable::PieceSquareTable,
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
    pub const NULL: Self = Self(0, 0);

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
pub const IS_MATE_SCORE: i32 = MATE_SCORE - MAX_DEPTH.n_ply() as i32;

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

// Stockfish nonlinear mobility eval tables.
#[rustfmt::skip]
const KNIGHT_MOBILITY_BONUS: [S; 9] = 
    [ S(-62,-79), S(-53,-57), S(-12,-31), S( -3,-17), S(  3,  7), S( 12, 13), // Knight
      S( 21, 16), S( 28, 21), S( 37, 26) ];
#[rustfmt::skip]
const BISHOP_MOBILITY_BONUS: [S; 14] = [ S(-47,-59), S(-20,-25), S( 14, -8), S( 29, 12), S( 39, 21), S( 53, 40), // Bishop
      S( 53, 56), S( 60, 58), S( 62, 65), S( 69, 72), S( 78, 78), S( 83, 87),
      S( 91, 88), S( 96, 98) ];
#[rustfmt::skip]
const ROOK_MOBILITY_BONUS: [S; 15] = [ S(-60,-82), S(-24,-15), S(  0, 17) ,S(  3, 43), S(  4, 72), S( 14,100), // Rook
      S( 20,102), S( 30,122), S( 41,133), S(41 ,139), S( 41,153), S( 45,160),
      S( 57,165), S( 58,170), S( 67,175) ];
#[rustfmt::skip]
const QUEEN_MOBILITY_BONUS: [S; 28] = [ S(-29,-49), S(-16,-29), S( -8, -8), S( -8, 17), S( 18, 39), S( 25, 54), // Queen
      S( 23, 59), S( 37, 73), S( 41, 76), S( 54, 95), S( 65, 95) ,S( 68,101),
      S( 69,124), S( 70,128), S( 70,132), S( 70,133) ,S( 71,136), S( 72,140),
      S( 74,147), S( 76,149), S( 90,153), S(104,169), S(105,171), S(106,171),
      S(112,178), S(114,185), S(114,187), S(119,221) ];

/// The bonus applied when a pawn has no pawns of the opposite colour ahead of it, or to the left or right, scaled by the rank that the pawn is on.
/// values from VICE.
pub static PASSED_PAWN_BONUS: [S; 6] = [
    S(5, 5),
    S(10, 10),
    S(20, 20),
    S(35, 35),
    S(60, 60),
    S(100, 100),
];

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Parameters {
    pub piece_values: [S; 13],
    pub isolated_pawn_malus: S,
    pub doubled_pawn_malus: S,
    pub bishop_pair_bonus: S,
    pub rook_open_file_bonus: S,
    pub rook_half_open_file_bonus: S,
    pub queen_open_file_bonus: S,
    pub queen_half_open_file_bonus: S,
    pub knight_mobility_bonus: [S; 9],
    pub bishop_mobility_bonus: [S; 14],
    pub rook_mobility_bonus: [S; 15],
    pub queen_mobility_bonus: [S; 28],
    pub passed_pawn_bonus: [S; 6],
    pub piece_square_tables: [[S; 64]; 13],
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            piece_values: PIECE_VALUES,
            isolated_pawn_malus: ISOLATED_PAWN_MALUS,
            doubled_pawn_malus: DOUBLED_PAWN_MALUS,
            bishop_pair_bonus: BISHOP_PAIR_BONUS,
            rook_open_file_bonus: ROOK_OPEN_FILE_BONUS,
            rook_half_open_file_bonus: ROOK_HALF_OPEN_FILE_BONUS,
            queen_open_file_bonus: QUEEN_OPEN_FILE_BONUS,
            queen_half_open_file_bonus: QUEEN_HALF_OPEN_FILE_BONUS,
            knight_mobility_bonus: KNIGHT_MOBILITY_BONUS,
            bishop_mobility_bonus: BISHOP_MOBILITY_BONUS,
            rook_mobility_bonus: ROOK_MOBILITY_BONUS,
            queen_mobility_bonus: QUEEN_MOBILITY_BONUS,
            passed_pawn_bonus: PASSED_PAWN_BONUS,
            piece_square_tables: crate::piecesquaretable::sftables::construct_sf_pst(),
        }
    }
}

impl Display for Parameters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        writeln!(f, "Parameters {{")?;
        writeln!(f, "    piece_values: {:?},", &self.piece_values[1..6])?;
        writeln!(
            f,
            "    isolated_pawn_malus: {:?},",
            self.isolated_pawn_malus
        )?;
        writeln!(f, "    doubled_pawn_malus: {:?},", self.doubled_pawn_malus)?;
        writeln!(f, "    bishop_pair_bonus: {:?},", self.bishop_pair_bonus)?;
        writeln!(
            f,
            "    rook_open_file_bonus: {:?},",
            self.rook_open_file_bonus
        )?;
        writeln!(
            f,
            "    rook_half_open_file_bonus: {:?},",
            self.rook_half_open_file_bonus
        )?;
        writeln!(
            f,
            "    queen_open_file_bonus: {:?},",
            self.queen_open_file_bonus
        )?;
        writeln!(
            f,
            "    queen_half_open_file_bonus: {:?},",
            self.queen_half_open_file_bonus
        )?;
        writeln!(
            f,
            "    knight_mobility_bonus: {:?},",
            self.knight_mobility_bonus
        )?;
        writeln!(
            f,
            "    bishop_mobility_bonus: {:?},",
            self.bishop_mobility_bonus
        )?;
        writeln!(
            f,
            "    rook_mobility_bonus: {:?},",
            self.rook_mobility_bonus
        )?;
        writeln!(
            f,
            "    queen_mobility_bonus: {:?},",
            self.queen_mobility_bonus
        )?;
        writeln!(f, "    passed_pawn_bonus: {:?},", self.passed_pawn_bonus)?;
        writeln!(
            f,
            "    piece_square_tables: {:?},",
            &self.piece_square_tables[1..7]
        )?;
        write!(f, "}}")?;
        Ok(())
    }
}

impl Parameters {
    pub const NULL: Self = Self {
        piece_values: [S::NULL; 13],
        isolated_pawn_malus: S::NULL,
        doubled_pawn_malus: S::NULL,
        bishop_pair_bonus: S::NULL,
        rook_open_file_bonus: S::NULL,
        rook_half_open_file_bonus: S::NULL,
        queen_open_file_bonus: S::NULL,
        queen_half_open_file_bonus: S::NULL,
        knight_mobility_bonus: [S::NULL; 9],
        bishop_mobility_bonus: [S::NULL; 14],
        rook_mobility_bonus: [S::NULL; 15],
        queen_mobility_bonus: [S::NULL; 28],
        passed_pawn_bonus: [S::NULL; 6],
        piece_square_tables: [[S::NULL; 64]; 13],
    };

    pub fn vectorise(&self) -> Vec<i32> {
        let ss = self.piece_values[1..6] // pawn to queen
            .iter()
            .copied()
            .chain(Some(self.isolated_pawn_malus))
            .chain(Some(self.doubled_pawn_malus))
            .chain(Some(self.bishop_pair_bonus))
            .chain(Some(S(
                self.rook_open_file_bonus.0,
                self.rook_half_open_file_bonus.0,
            )))
            .chain(Some(S(
                self.queen_open_file_bonus.0,
                self.queen_half_open_file_bonus.0,
            )))
            .chain(self.knight_mobility_bonus.into_iter())
            .chain(self.bishop_mobility_bonus.into_iter())
            .chain(self.rook_mobility_bonus.into_iter())
            .chain(self.queen_mobility_bonus.into_iter())
            .chain(self.passed_pawn_bonus.into_iter())
            .chain(self.piece_square_tables[1..7].iter().flatten().copied());
        ss.flat_map(|s| [s.0, s.1].into_iter()).collect()
    }

    pub fn devectorise(data: &[i32]) -> Self {
        let mut out = Self::NULL;
        let mut data = data.chunks(2).map(|x| S(x[0], x[1]));
        for p in 1..6 {
            let val = data
                .next()
                .expect("failed to read piece_value term from vector");
            out.piece_values[p] = val;
            out.piece_values[p + 6] = val;
        }
        out.isolated_pawn_malus = data
            .next()
            .expect("failed to read isolated_pawn_malus term from vector");
        out.doubled_pawn_malus = data
            .next()
            .expect("failed to read doubled_pawn_malus term from vector");
        out.bishop_pair_bonus = data
            .next()
            .expect("failed to read bishop_pair_bonus term from vector");
        let rook_file_bonus = data
            .next()
            .expect("failed to read rook_file_bonus term from vector");
        out.rook_open_file_bonus = S(rook_file_bonus.0, 0);
        out.rook_half_open_file_bonus = S(rook_file_bonus.1, 0);
        let queen_file_bonus = data
            .next()
            .expect("failed to read queen_file_bonus term from vector");
        out.queen_open_file_bonus = S(queen_file_bonus.0, 0);
        out.queen_half_open_file_bonus = S(queen_file_bonus.1, 0);
        for knight_mobility_bonus in &mut out.knight_mobility_bonus {
            *knight_mobility_bonus = data
                .next()
                .expect("failed to read knight_mobility_bonus term from vector");
        }
        for bishop_mobility_bonus in &mut out.bishop_mobility_bonus {
            *bishop_mobility_bonus = data
                .next()
                .expect("failed to read bishop_mobility_bonus term from vector");
        }
        for rook_mobility_bonus in &mut out.rook_mobility_bonus {
            *rook_mobility_bonus = data
                .next()
                .expect("failed to read rook_mobility_bonus term from vector");
        }
        for queen_mobility_bonus in &mut out.queen_mobility_bonus {
            *queen_mobility_bonus = data
                .next()
                .expect("failed to read queen_mobility_bonus term from vector");
        }
        for passed_pawn_bonus in &mut out.passed_pawn_bonus {
            *passed_pawn_bonus = data
                .next()
                .expect("failed to read passed_pawn_bonus term from vector");
        }
        for pst in 1..7 {
            // pawn to king
            for sq in 0..64 {
                let val = data
                    .next()
                    .expect("failed to read piece_square_table term from vector");
                out.piece_square_tables[pst][sq] = val;
                out.piece_square_tables[pst + 6][sq ^ 56] = -val;
            }
        }
        assert!(
            data.next().is_none(),
            "reading data from a vector of wrong size (too big)"
        );
        out
    }

    pub fn save_param_vec(param_vec: &[i32], path: &str) {
        let mut output = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
        for param in param_vec {
            std::io::Write::write_all(&mut output, format!("{param},").as_bytes()).unwrap();
        }
        std::io::Write::flush(&mut output).unwrap();
    }

    pub fn load_param_vec(path: &str) -> Result<Vec<i32>, Box<dyn Error>> {
        let mut params = Vec::new();
        let input = std::io::BufReader::new(std::fs::File::open(path)?);
        for param in std::io::BufRead::split(input, b',') {
            let param = param?;
            let param = String::from_utf8(param)?;
            let param: i32 = param.parse()?;
            params.push(param);
        }
        Ok(params)
    }

    pub fn from_file(path: &str) -> Result<Self, Box<dyn Error>> {
        let vec = Self::load_param_vec(path)?;
        Ok(Self::devectorise(&vec))
    }
}

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

impl Board {
    pub fn set_eval_params(&mut self, params: Parameters) {
        self.eval_params = params;
    }

    pub const fn get_pst(&self) -> &PieceSquareTable {
        &self.eval_params.piece_square_tables
    }

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

        let pawn_val = self.pawn_structure_term(); // INCREMENTAL UPDATE.
        let bishop_pair_val = self.bishop_pair_term();
        let mobility_val = self.mobility();
        let rook_open_file_val = self.rook_open_file_term();
        let queen_open_file_val = self.queen_open_file_term();

        score += pawn_val;
        score += bishop_pair_val;
        score += mobility_val;
        score += rook_open_file_val;
        score += queen_open_file_val;

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
            return self.eval_params.bishop_pair_bonus;
        }
        if b_count >= 2 {
            return -self.eval_params.bishop_pair_bonus;
        }
        S(0, 0)
    }

    fn pawn_structure_term(&self) -> S {
        static DOUBLED_PAWN_MAPPING: [i32; 7] = [0, 0, 1, 2, 3, 4, 5]; // not a tunable parameter, just how "number of pawns in a file" is mapped to "amount of doubled pawn-ness"
        let mut w_score = S(0, 0);
        let (white_pawns, black_pawns) =
            (self.pieces.pawns::<true>(), self.pieces.pawns::<false>());
        for &white_pawn_loc in self.piece_lists[WP as usize].iter() {
            if unsafe { *ISOLATED_BB.get_unchecked(white_pawn_loc as usize) } & white_pawns == 0 {
                w_score -= self.eval_params.isolated_pawn_malus;
            }

            if unsafe { *WHITE_PASSED_BB.get_unchecked(white_pawn_loc as usize) } & black_pawns == 0
            {
                let rank = rank(white_pawn_loc) as usize;
                w_score += self.eval_params.passed_pawn_bonus[rank - 1];
            }
        }

        let mut b_score = S(0, 0);
        for &black_pawn_loc in self.piece_lists[BP as usize].iter() {
            if unsafe { *ISOLATED_BB.get_unchecked(black_pawn_loc as usize) } & black_pawns == 0 {
                b_score -= self.eval_params.isolated_pawn_malus;
            }

            if unsafe { *BLACK_PASSED_BB.get_unchecked(black_pawn_loc as usize) } & white_pawns == 0
            {
                let rank = rank(black_pawn_loc) as usize;
                b_score += self.eval_params.passed_pawn_bonus[7 - rank - 1];
            }
        }

        for &file_mask in &FILE_BB {
            let pawns_in_file = (file_mask & white_pawns).count_ones() as usize;
            let multiplier = unsafe { *DOUBLED_PAWN_MAPPING.get_unchecked(pawns_in_file) };
            w_score -= self.eval_params.doubled_pawn_malus * multiplier;
            let pawns_in_file = (file_mask & black_pawns).count_ones() as usize;
            let multiplier = unsafe { *DOUBLED_PAWN_MAPPING.get_unchecked(pawns_in_file) };
            b_score -= self.eval_params.doubled_pawn_malus * multiplier;
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
                score += self.eval_params.rook_open_file_bonus;
            } else if self.is_file_halfopen::<WHITE>(file) {
                score += self.eval_params.rook_half_open_file_bonus;
            }
        }
        for &rook_sq in self.piece_lists[BR as usize].iter() {
            let file = file(rook_sq);
            if self.is_file_open(file) {
                score -= self.eval_params.rook_open_file_bonus;
            } else if self.is_file_halfopen::<BLACK>(file) {
                score -= self.eval_params.rook_half_open_file_bonus;
            }
        }
        score
    }

    fn queen_open_file_term(&self) -> S {
        let mut score = S(0, 0);
        for &queen_sq in self.piece_lists[WQ as usize].iter() {
            let file = file(queen_sq);
            if self.is_file_open(file) {
                score += self.eval_params.queen_open_file_bonus;
            } else if self.is_file_halfopen::<WHITE>(file) {
                score += self.eval_params.queen_half_open_file_bonus;
            }
        }
        for &queen_sq in self.piece_lists[BQ as usize].iter() {
            let file = file(queen_sq);
            if self.is_file_open(file) {
                score -= self.eval_params.queen_open_file_bonus;
            } else if self.is_file_halfopen::<BLACK>(file) {
                score -= self.eval_params.queen_half_open_file_bonus;
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
        let safe_white_moves = !self.pieces.pawn_attacks::<false>();
        let safe_black_moves = !self.pieces.pawn_attacks::<true>();
        let blockers = self.pieces.occupied();
        for knight_sq in BitLoop::new(self.pieces.knights::<true>()) {
            let attacks = attacks::<KNIGHT>(knight_sq, BB_NONE);
            let attacks = attacks & safe_white_moves;
            let attacks = attacks.count_ones() as usize;
            mob_score += self.eval_params.knight_mobility_bonus[attacks];
        }
        for knight_sq in BitLoop::new(self.pieces.knights::<false>()) {
            let attacks = attacks::<KNIGHT>(knight_sq, BB_NONE);
            let attacks = attacks & safe_black_moves;
            let attacks = attacks.count_ones() as usize;
            mob_score -= self.eval_params.knight_mobility_bonus[attacks];
        }
        for bishop_sq in BitLoop::new(self.pieces.bishops::<true>()) {
            let attacks = attacks::<BISHOP>(bishop_sq, blockers);
            let attacks = attacks & safe_white_moves;
            let attacks = attacks.count_ones() as usize;
            mob_score += self.eval_params.bishop_mobility_bonus[attacks];
        }
        for bishop_sq in BitLoop::new(self.pieces.bishops::<false>()) {
            let attacks = attacks::<BISHOP>(bishop_sq, blockers);
            let attacks = attacks & safe_black_moves;
            let attacks = attacks.count_ones() as usize;
            mob_score -= self.eval_params.bishop_mobility_bonus[attacks];
        }
        for rook_sq in BitLoop::new(self.pieces.rooks::<true>()) {
            let attacks = attacks::<ROOK>(rook_sq, blockers);
            let attacks = attacks & safe_white_moves;
            let attacks = attacks.count_ones() as usize;
            mob_score += self.eval_params.rook_mobility_bonus[attacks];
        }
        for rook_sq in BitLoop::new(self.pieces.rooks::<false>()) {
            let attacks = attacks::<ROOK>(rook_sq, blockers);
            let attacks = attacks & safe_black_moves;
            let attacks = attacks.count_ones() as usize;
            mob_score -= self.eval_params.rook_mobility_bonus[attacks];
        }
        for queen_sq in BitLoop::new(self.pieces.queens::<true>()) {
            let attacks = attacks::<QUEEN>(queen_sq, blockers);
            let attacks = attacks & safe_white_moves;
            let attacks = attacks.count_ones() as usize;
            mob_score += self.eval_params.queen_mobility_bonus[attacks];
        }
        for queen_sq in BitLoop::new(self.pieces.queens::<false>()) {
            let attacks = attacks::<QUEEN>(queen_sq, blockers);
            let attacks = attacks & safe_black_moves;
            let attacks = attacks.count_ones() as usize;
            mob_score -= self.eval_params.queen_mobility_bonus[attacks];
        }
        mob_score
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

    #[test]
    fn params_round_trip() {
        let params = crate::board::evaluation::Parameters::default();
        let vec = params.vectorise();
        let params2 = crate::board::evaluation::Parameters::devectorise(&vec);
        assert_eq!(params, params2);

        let n_params = vec.len();
        for _ in 0..100 {
            let vec = (0..n_params)
                .map(|_| rand::random::<i32>())
                .collect::<Vec<_>>();
            let params = crate::board::evaluation::Parameters::devectorise(&vec);
            let vec2 = params.vectorise();
            assert_eq!(vec, vec2);
        }
    }

    #[test]
    fn passers_should_be_pushed() {
        let mut starting_rank_passer =
            super::Board::from_fen("8/k7/8/8/8/8/K6P/8 w - - 0 1").unwrap();
        let mut end_rank_passer = super::Board::from_fen("8/k6P/8/8/8/8/K7/8 w - - 0 1").unwrap();

        let starting_rank_eval = starting_rank_passer.evaluate();
        let end_rank_eval = end_rank_passer.evaluate();

        // is should be better to have a passer that is more advanced.
        assert!(
            end_rank_eval > starting_rank_eval,
            "end_rank_eval: {end_rank_eval}, starting_rank_eval: {starting_rank_eval}"
        );
    }
}
