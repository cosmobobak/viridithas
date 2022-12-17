// The granularity of evaluation in this engine is in centipawns.

pub mod parameters;
pub mod score;

use parameters::EvalParams;
use score::S;

use crate::{
    board::Board,
    chessmove::Move,
    definitions::{
        type_of, BB, BISHOP, BLACK, BN, BP, BQ, BR, KING, KNIGHT, MAX_DEPTH, PAWN, QUEEN, ROOK, WB,
        WHITE, WN, WP, WQ, WR, Square,
    },
    lookups::{init_eval_masks, init_passed_isolated_bb},
    search::draw_score,
    threadlocal::ThreadData,
};

use super::movegen::{
    bitboards::{attacks, BitShiftExt, DARK_SQUARE, LIGHT_SQUARE},
    BitLoop, BB_NONE,
};

pub const PAWN_VALUE: S = S(126, 196);
pub const KNIGHT_VALUE: S = S(437, 455);
pub const BISHOP_VALUE: S = S(437, 491);
pub const ROOK_VALUE: S = S(604, 805);
pub const QUEEN_VALUE: S = S(1369, 1274);

pub fn get_see_value(piece: u8) -> i32 {
    static SEE_PIECE_VALUES: [i32; 7] = [
        0,
        PAWN_VALUE.value(128),
        KNIGHT_VALUE.value(128),
        BISHOP_VALUE.value(128),
        ROOK_VALUE.value(128),
        QUEEN_VALUE.value(128),
        0,
    ];
    SEE_PIECE_VALUES[piece as usize]
}

/// The value of checkmate.
/// To recover depth-to-mate, we subtract depth (ply) from this value.
/// e.g. if white has a mate in two ply, the output from a depth-5 search will be
/// two less than `MATE_SCORE`.
pub const MATE_SCORE: i32 = i16::MAX as i32 - 300;
pub const fn mate_in(ply: i32) -> i32 {
    MATE_SCORE - ply
}
pub const fn mated_in(ply: i32) -> i32 {
    -MATE_SCORE + ply
}

/// A threshold over which scores must be mate.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub const MINIMUM_MATE_SCORE: i32 = MATE_SCORE - MAX_DEPTH.ply_to_horizon() as i32;

#[rustfmt::skip]
pub static PIECE_VALUES: [S; 13] = [
    S(0, 0),
    PAWN_VALUE, KNIGHT_VALUE, BISHOP_VALUE, ROOK_VALUE, QUEEN_VALUE, S(0, 0),
    PAWN_VALUE, KNIGHT_VALUE, BISHOP_VALUE, ROOK_VALUE, QUEEN_VALUE, S(0, 0),
];

/// The malus applied when a pawn has no pawns of its own colour to the left or right.
pub const ISOLATED_PAWN_MALUS: S = S(21, 19);

/// The malus applied when two (or more) pawns of a colour are on the same file.
pub const DOUBLED_PAWN_MALUS: S = S(23, 34);

/// The bonus granted for having two bishops.
pub const BISHOP_PAIR_BONUS: S = S(41, 129);

/// The bonus for having a rook on an open file.
pub const ROOK_OPEN_FILE_BONUS: S = S(72, 0);
/// The bonus for having a rook on a semi-open file.
pub const ROOK_HALF_OPEN_FILE_BONUS: S = S(42, 0);
/// The bonus for having a queen on an open file.
pub const QUEEN_OPEN_FILE_BONUS: S = S(-4, 0);
/// The bonus for having a queen on a semi-open file.
pub const QUEEN_HALF_OPEN_FILE_BONUS: S = S(21, 0);

// nonlinear mobility eval tables.
#[rustfmt::skip]
static KNIGHT_MOBILITY_BONUS: [S; 9] = [S(-143, -214), S(-16, -13), S(10, 50), S(29, 105), S(47, 128), S(53, 162), S(73, 161), S(97, 161), S(128, 129)];
#[rustfmt::skip]
static BISHOP_MOBILITY_BONUS: [S; 14] = [S(-17, -115), S(-20, -80), S(30, -2), S(45, 74), S(67, 95), S(80, 115), S(86, 138), S(93, 157), S(101, 159), S(107, 167), S(112, 158), S(120, 154), S(200, 123), S(175, 150)];
#[rustfmt::skip]
static ROOK_MOBILITY_BONUS: [S; 15] = [S(-9, -2), S(-1, 103), S(23, 136), S(33, 184), S(41, 225), S(48, 237), S(47, 262), S(54, 269), S(63, 262), S(71, 276), S(72, 285), S(84, 284), S(96, 277), S(116, 263), S(104, 265)];
#[rustfmt::skip]
static QUEEN_MOBILITY_BONUS: [S; 28] = [S(-29, -49), S(-185, -204), S(-116, -296), S(184, 160), S(139, 64), S(218, 101), S(220, 143), S(218, 224), S(217, 299), S(217, 336), S(219, 373), S(225, 393), S(231, 416), S(237, 426), S(242, 442), S(243, 456), S(245, 465), S(255, 455), S(256, 461), S(277, 438), S(285, 432), S(334, 387), S(365, 375), S(401, 330), S(423, 304), S(362, 293), S(361, 299), S(264, 225)];

/// The bonus applied when a pawn has no pawns of the opposite colour ahead of it, or to the left or right, scaled by the rank that the pawn is on.
pub static PASSED_PAWN_BONUS: [S; 6] =
    [S(-8, 32), S(-26, 43), S(-28, 83), S(13, 124), S(37, 227), S(101, 264)];

pub const TEMPO_BONUS: S = S(12, 0);

pub const PAWN_THREAT_ON_MINOR: S = S(80, 79);
pub const PAWN_THREAT_ON_MAJOR: S = S(74, 55);
pub const MINOR_THREAT_ON_MAJOR: S = S(75, 53);

pub static KING_DANGER_COEFFS: [i32; 3] = [36, 165, -719];
pub static KING_DANGER_PIECE_WEIGHTS: [i32; 8] = [40, 20, 40, 20, 60, 20, 100, 40];
const KINGDANGER_DESCALE: i32 = 20;

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
pub const fn lerp(mg: i32, eg: i32, t: i32) -> i32 {
    // debug_assert!((0..=256).contains(&t));
    let t = if t > 256 { 256 } else { t };
    mg * (256 - t) / 256 + eg * t / 256
}

pub const fn is_mate_score(score: i32) -> bool {
    score.abs() >= MINIMUM_MATE_SCORE
}

impl Board {
    pub fn set_eval_params(&mut self, params: EvalParams) {
        self.eparams = params;
    }

    /// Computes a score for the position, from the point of view of the side to move.
    /// This function should strive to be as cheap to call as possible, relying on
    /// incremental updates in make-unmake to avoid recomputation.
    pub fn evaluate_classical(&self, nodes: u64) -> i32 {
        if !self.pieces.any_pawns() && self.is_material_draw() {
            return if self.side == WHITE { draw_score(nodes) } else { -draw_score(nodes) };
        }

        let material = self.material();
        let pst = self.pst_vals;

        let pawn_structure = self.pawn_structure_term();
        let bishop_pair = self.bishop_pair_term();
        let rook_files = self.rook_open_file_term();
        let queen_files = self.queen_open_file_term();
        let (mobility, threats, danger_info) = self.mobility_threats_kingdanger();
        let king_danger = self.score_kingdanger(danger_info);
        let tempo = if self.turn() == WHITE { self.eparams.tempo } else { -self.eparams.tempo };

        let mut score = material;
        score += pst;
        score += pawn_structure;
        score += bishop_pair;
        score += rook_files;
        score += queen_files;
        score += mobility;
        score += threats;
        score += king_danger;
        score += tempo;

        let score = score.value(self.phase());

        let score = self.preprocess_drawish_scores(score, nodes);

        if self.side == WHITE {
            score
        } else {
            -score
        }
    }

    fn material(&self) -> S {
        self.material[WHITE as usize] - self.material[BLACK as usize]
    }

    fn simple_evaluation(&self) -> i32 {
        (self.pst_vals + self.material()).value(self.phase())
    }

    pub fn evaluate_nnue(&self, t: &mut ThreadData, nodes: u64) -> i32 {
        if !self.pieces.any_pawns() && self.is_material_draw() {
            return if self.side == WHITE { draw_score(nodes) } else { -draw_score(nodes) };
        }

        let v = t.nnue.evaluate(self.side);
        let simple = self.simple_evaluation();
        let simple = if self.side == WHITE {
            simple
        } else {
            -simple
        };
        let complexity = (v - simple).abs();
        
        v + complexity / 20
    }

    pub fn evaluate<const USE_NNUE: bool>(&self, t: &mut ThreadData, nodes: u64) -> i32 {
        if USE_NNUE {
            self.evaluate_nnue(t, nodes)
        } else {
            self.evaluate_classical(nodes)
        }
    }

    const fn unwinnable_for<const IS_WHITE: bool>(&self) -> bool {
        if IS_WHITE {
            if self.major_piece_counts[WHITE as usize] != 0 {
                return false;
            }
            if self.minor_piece_counts[WHITE as usize] > 1 {
                return false;
            }
            if self.num_ct::<WP>() != 0 {
                return false;
            }
        } else {
            if self.major_piece_counts[BLACK as usize] != 0 {
                return false;
            }
            if self.minor_piece_counts[BLACK as usize] > 1 {
                return false;
            }
            if self.num_ct::<BP>() != 0 {
                return false;
            }
        }

        true
    }

    const fn is_material_draw(&self) -> bool {
        if self.num_pt_ct::<ROOK>() == 0 && self.num_pt_ct::<QUEEN>() == 0 {
            if self.num_pt_ct::<BISHOP>() == 0 {
                if self.num_ct::<WN>() < 3 && self.num_ct::<BN>() < 3 {
                    return true;
                }
            } else if (self.num_pt_ct::<KNIGHT>() == 0
                && self.num_ct::<WB>().abs_diff(self.num_ct::<BB>()) < 2)
                || (self.num_ct::<WB>() + self.num_ct::<WN>() == 1
                    && self.num_ct::<BB>() + self.num_ct::<BN>() == 1)
            {
                return true;
            }
        } else if self.num_pt_ct::<QUEEN>() == 0 {
            if self.num_ct::<WR>() == 1 && self.num_ct::<BR>() == 1 {
                if (self.num_ct::<WN>() + self.num_ct::<WB>()) < 2
                    && (self.num_ct::<BN>() + self.num_ct::<BB>()) < 2
                {
                    return true;
                }
            } else if self.num_ct::<WR>() == 1 && self.num_ct::<BR>() == 0 {
                if (self.num_ct::<WN>() + self.num_ct::<WB>()) == 0
                    && ((self.num_ct::<BN>() + self.num_ct::<BB>()) == 1
                        || (self.num_ct::<BN>() + self.num_ct::<BB>()) == 2)
                {
                    return true;
                }
            } else if self.num_ct::<WR>() == 0
                && self.num_ct::<BR>() == 1
                && (self.num_ct::<BN>() + self.num_ct::<BB>()) == 0
                && ((self.num_ct::<WN>() + self.num_ct::<WB>()) == 1
                    || (self.num_ct::<WN>() + self.num_ct::<WB>()) == 2)
            {
                return true;
            }
        }
        false
    }

    const fn preprocess_drawish_scores(&self, score: i32, nodes: u64) -> i32 {
        // if we can't win with our material, we clamp the eval to zero.
        let drawscore = draw_score(nodes);
        if score > drawscore && self.unwinnable_for::<true>()
            || score < drawscore && self.unwinnable_for::<false>()
        {
            drawscore
        } else {
            score
        }
    }

    pub const fn zugzwang_unlikely(&self) -> bool {
        const ENDGAME_PHASE: i32 = game_phase(3, 0, 0, 2, 0);
        self.big_piece_counts[self.side as usize] > 0 && self.phase() < ENDGAME_PHASE
    }

    fn bishop_pair_term(&self) -> S {
        let white_pair = self.pieces.bishops_sqco::<true, LIGHT_SQUARE>() != 0
            && self.pieces.bishops_sqco::<true, DARK_SQUARE>() != 0;
        let black_pair = self.pieces.bishops_sqco::<false, LIGHT_SQUARE>() != 0
            && self.pieces.bishops_sqco::<false, DARK_SQUARE>() != 0;
        let multiplier = i32::from(white_pair) - i32::from(black_pair);
        self.eparams.bishop_pair_bonus * multiplier
    }

    fn pawn_structure_term(&self) -> S {
        #![allow(clippy::cast_possible_wrap)]
        /// not a tunable parameter, just how "number of pawns in a file" is mapped to "amount of doubled pawn-ness"
        static DOUBLED_PAWN_MAPPING: [i32; 7] = [0, 0, 1, 2, 3, 4, 5];
        let mut w_score = S(0, 0);
        let mut b_score = S(0, 0);
        let white_pawns = self.pieces.pawns::<true>();
        let black_pawns = self.pieces.pawns::<false>();

        for white_pawn_loc in BitLoop::new(white_pawns) {
            if ISOLATED_BB[white_pawn_loc.index()] & white_pawns == 0 {
                w_score -= self.eparams.isolated_pawn_malus;
            }

            if WHITE_PASSED_BB[white_pawn_loc.index()] & black_pawns == 0 {
                let rank = white_pawn_loc.rank() as usize;
                w_score += self.eparams.passed_pawn_bonus[rank - 1];
            }
        }

        for black_pawn_loc in BitLoop::new(black_pawns) {
            if ISOLATED_BB[black_pawn_loc.index()] & black_pawns == 0 {
                b_score -= self.eparams.isolated_pawn_malus;
            }

            if BLACK_PASSED_BB[black_pawn_loc.index()] & white_pawns == 0 {
                let rank = black_pawn_loc.rank() as usize;
                b_score += self.eparams.passed_pawn_bonus[7 - rank - 1];
            }
        }

        for &file_mask in &FILE_BB {
            let pawns_in_file = (file_mask & white_pawns).count_ones() as usize;
            let multiplier = DOUBLED_PAWN_MAPPING[pawns_in_file];
            w_score -= self.eparams.doubled_pawn_malus * multiplier;
            let pawns_in_file = (file_mask & black_pawns).count_ones() as usize;
            let multiplier = DOUBLED_PAWN_MAPPING[pawns_in_file];
            b_score -= self.eparams.doubled_pawn_malus * multiplier;
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
        let pawns =
            if SIDE == WHITE { self.pieces.pawns::<true>() } else { self.pieces.pawns::<false>() };
        (mask & pawns) == 0
    }

    fn rook_open_file_term(&self) -> S {
        let mut score = S(0, 0);
        for rook_sq in BitLoop::new(self.pieces.rooks::<true>()) {
            let file = rook_sq.file();
            if self.is_file_open(file) {
                score += self.eparams.rook_open_file_bonus;
            } else if self.is_file_halfopen::<WHITE>(file) {
                score += self.eparams.rook_half_open_file_bonus;
            }
        }
        for rook_sq in BitLoop::new(self.pieces.rooks::<false>()) {
            let file = rook_sq.file();
            if self.is_file_open(file) {
                score -= self.eparams.rook_open_file_bonus;
            } else if self.is_file_halfopen::<BLACK>(file) {
                score -= self.eparams.rook_half_open_file_bonus;
            }
        }
        score
    }

    fn queen_open_file_term(&self) -> S {
        let mut score = S(0, 0);
        for queen_sq in BitLoop::new(self.pieces.queens::<true>()) {
            let file = queen_sq.file();
            if self.is_file_open(file) {
                score += self.eparams.queen_open_file_bonus;
            } else if self.is_file_halfopen::<WHITE>(file) {
                score += self.eparams.queen_half_open_file_bonus;
            }
        }
        for queen_sq in BitLoop::new(self.pieces.queens::<false>()) {
            let file = queen_sq.file();
            if self.is_file_open(file) {
                score -= self.eparams.queen_open_file_bonus;
            } else if self.is_file_halfopen::<BLACK>(file) {
                score -= self.eparams.queen_half_open_file_bonus;
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

    #[allow(clippy::too_many_lines)]
    fn mobility_threats_kingdanger(&self) -> (S, S, KingDangerInfo) {
        #![allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)] // for count_ones, which can return at most 64.
        let mut king_danger_info =
            KingDangerInfo { attack_units_on_white: 0, attack_units_on_black: 0 };
        let ptmul = &self.eparams.king_danger_piece_weights;
        let mut mob_score = S(0, 0);
        let mut threat_score = S(0, 0);
        let white_king_area = king_area::<true>(self.king_sq(WHITE));
        let black_king_area = king_area::<false>(self.king_sq(BLACK));
        let white_pawn_attacks = self.pieces.pawn_attacks::<true>();
        let black_pawn_attacks = self.pieces.pawn_attacks::<false>();
        let white_minor = self.pieces.minors::<true>();
        let black_minor = self.pieces.minors::<false>();
        let white_major = self.pieces.majors::<true>();
        let black_major = self.pieces.majors::<false>();
        threat_score += self.eparams.pawn_threat_on_minor
            * (black_minor & white_pawn_attacks).count_ones() as i32;
        threat_score -= self.eparams.pawn_threat_on_minor
            * (white_minor & black_pawn_attacks).count_ones() as i32;
        threat_score += self.eparams.pawn_threat_on_major
            * (black_major & white_pawn_attacks).count_ones() as i32;
        threat_score -= self.eparams.pawn_threat_on_major
            * (white_major & black_pawn_attacks).count_ones() as i32;
        let safe_white_moves = !black_pawn_attacks;
        let safe_black_moves = !white_pawn_attacks;
        let blockers = self.pieces.occupied();
        for knight_sq in BitLoop::new(self.pieces.knights::<true>()) {
            let attacks = attacks::<KNIGHT>(knight_sq, BB_NONE);
            // kingsafety
            let attacks_on_black_king = attacks & black_king_area;
            let defense_of_white_king = attacks & white_king_area;
            king_danger_info.attack_units_on_black +=
                attacks_on_black_king.count_ones() as i32 * ptmul[0];
            king_danger_info.attack_units_on_white -=
                defense_of_white_king.count_ones() as i32 * ptmul[1];
            // threats
            let attacks_on_majors = attacks & black_major;
            threat_score +=
                self.eparams.minor_threat_on_major * attacks_on_majors.count_ones() as i32;
            // mobility
            let attacks = attacks & safe_white_moves;
            let attacks = attacks.count_ones() as usize;
            mob_score += self.eparams.knight_mobility_bonus[attacks];
        }
        for knight_sq in BitLoop::new(self.pieces.knights::<false>()) {
            let attacks = attacks::<KNIGHT>(knight_sq, BB_NONE);
            // kingsafety
            let attacks_on_white_king = attacks & white_king_area;
            let defense_of_black_king = attacks & black_king_area;
            king_danger_info.attack_units_on_white +=
                attacks_on_white_king.count_ones() as i32 * ptmul[0];
            king_danger_info.attack_units_on_black -=
                defense_of_black_king.count_ones() as i32 * ptmul[1];
            // threats
            let attacks_on_majors = attacks & white_major;
            threat_score -=
                self.eparams.minor_threat_on_major * attacks_on_majors.count_ones() as i32;
            // mobility
            let attacks = attacks & safe_black_moves;
            let attacks = attacks.count_ones() as usize;
            mob_score -= self.eparams.knight_mobility_bonus[attacks];
        }
        for bishop_sq in BitLoop::new(self.pieces.bishops::<true>()) {
            let attacks = attacks::<BISHOP>(bishop_sq, blockers);
            // kingsafety
            let attacks_on_black_king = attacks & black_king_area;
            let defense_of_white_king = attacks & white_king_area;
            king_danger_info.attack_units_on_black +=
                attacks_on_black_king.count_ones() as i32 * ptmul[2];
            king_danger_info.attack_units_on_white -=
                defense_of_white_king.count_ones() as i32 * ptmul[3];
            // threats
            let attacks_on_majors = attacks & black_major;
            threat_score +=
                self.eparams.minor_threat_on_major * attacks_on_majors.count_ones() as i32;
            // mobility
            let attacks = attacks & safe_white_moves;
            let attacks = attacks.count_ones() as usize;
            mob_score += self.eparams.bishop_mobility_bonus[attacks];
        }
        for bishop_sq in BitLoop::new(self.pieces.bishops::<false>()) {
            let attacks = attacks::<BISHOP>(bishop_sq, blockers);
            // kingsafety
            let attacks_on_white_king = attacks & white_king_area;
            let defense_of_black_king = attacks & black_king_area;
            king_danger_info.attack_units_on_white +=
                attacks_on_white_king.count_ones() as i32 * ptmul[2];
            king_danger_info.attack_units_on_black -=
                defense_of_black_king.count_ones() as i32 * ptmul[3];
            // threats
            let attacks_on_majors = attacks & white_major;
            threat_score -=
                self.eparams.minor_threat_on_major * attacks_on_majors.count_ones() as i32;
            // mobility
            let attacks = attacks & safe_black_moves;
            let attacks = attacks.count_ones() as usize;
            mob_score -= self.eparams.bishop_mobility_bonus[attacks];
        }
        for rook_sq in BitLoop::new(self.pieces.rooks::<true>()) {
            let attacks = attacks::<ROOK>(rook_sq, blockers);
            // kingsafety
            let attacks_on_black_king = attacks & black_king_area;
            let defense_of_white_king = attacks & white_king_area;
            king_danger_info.attack_units_on_black +=
                attacks_on_black_king.count_ones() as i32 * ptmul[4];
            king_danger_info.attack_units_on_white -=
                defense_of_white_king.count_ones() as i32 * ptmul[5];
            // mobility
            let attacks = attacks & safe_white_moves;
            let attacks = attacks.count_ones() as usize;
            mob_score += self.eparams.rook_mobility_bonus[attacks];
        }
        for rook_sq in BitLoop::new(self.pieces.rooks::<false>()) {
            let attacks = attacks::<ROOK>(rook_sq, blockers);
            // kingsafety
            let attacks_on_white_king = attacks & white_king_area;
            let defense_of_black_king = attacks & black_king_area;
            king_danger_info.attack_units_on_white +=
                attacks_on_white_king.count_ones() as i32 * ptmul[4];
            king_danger_info.attack_units_on_black -=
                defense_of_black_king.count_ones() as i32 * ptmul[5];
            // mobility
            let attacks = attacks & safe_black_moves;
            let attacks = attacks.count_ones() as usize;
            mob_score -= self.eparams.rook_mobility_bonus[attacks];
        }
        for queen_sq in BitLoop::new(self.pieces.queens::<true>()) {
            let attacks = attacks::<QUEEN>(queen_sq, blockers);
            // kingsafety
            let attacks_on_black_king = attacks & black_king_area;
            let defense_of_white_king = attacks & white_king_area;
            king_danger_info.attack_units_on_black +=
                attacks_on_black_king.count_ones() as i32 * ptmul[6];
            king_danger_info.attack_units_on_white -=
                defense_of_white_king.count_ones() as i32 * ptmul[7];
            // mobility
            let attacks = attacks & safe_white_moves;
            let attacks = attacks.count_ones() as usize;
            mob_score += self.eparams.queen_mobility_bonus[attacks];
        }
        for queen_sq in BitLoop::new(self.pieces.queens::<false>()) {
            let attacks = attacks::<QUEEN>(queen_sq, blockers);
            // kingsafety
            let attacks_on_white_king = attacks & white_king_area;
            let defense_of_black_king = attacks & black_king_area;
            king_danger_info.attack_units_on_white +=
                attacks_on_white_king.count_ones() as i32 * ptmul[6];
            king_danger_info.attack_units_on_black -=
                defense_of_black_king.count_ones() as i32 * ptmul[7];
            // mobility
            let attacks = attacks & safe_black_moves;
            let attacks = attacks.count_ones() as usize;
            mob_score -= self.eparams.queen_mobility_bonus[attacks];
        }
        king_danger_info.attack_units_on_white /= KINGDANGER_DESCALE;
        king_danger_info.attack_units_on_black /= KINGDANGER_DESCALE;
        (mob_score, threat_score, king_danger_info)
    }

    fn score_kingdanger(&self, kd: KingDangerInfo) -> S {
        let [a, b, c] = self.eparams.king_danger_coeffs;
        let kd_formula = |au| (a * au * au + b * au + c) / 100;

        let white_attack_strength = kd_formula(kd.attack_units_on_black.clamp(0, 99)).min(500);
        let black_attack_strength = kd_formula(kd.attack_units_on_white.clamp(0, 99)).min(500);
        let relscore = white_attack_strength - black_attack_strength;
        S(relscore, relscore / 2)
    }

    pub fn estimated_see(&self, m: Move) -> i32 {
        // initially take the value of the thing on the target square
        let mut value = get_see_value(type_of(self.piece_at(m.to())));

        if m.is_promo() {
            // if it's a promo, swap a pawn for the promoted piece type
            value += get_see_value(m.promotion_type()) - get_see_value(PAWN);
        } else if m.is_ep() {
            // for e.p. we will miss a pawn because the target square is empty
            value = get_see_value(PAWN);
        }

        value
    }
}

pub fn king_area<const IS_WHITE: bool>(king_sq: Square) -> u64 {
    let king_attacks = attacks::<KING>(king_sq, BB_NONE);
    let forward_area = if IS_WHITE { king_attacks.north_one() } else { king_attacks.south_one() };
    king_attacks | forward_area
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct KingDangerInfo {
    attack_units_on_white: i32,
    attack_units_on_black: i32,
}

mod tests {
    #[test]
    fn unwinnable() {
        const FEN: &str = "8/8/8/8/2K2k2/2n2P2/8/8 b - - 1 1";
        crate::magic::initialise();
        let board = super::Board::from_fen(FEN).unwrap();
        let eval = board.evaluate_classical(0);
        assert!(
            (-2..=2).contains(&(eval.abs())),
            "eval is not a draw score in a position unwinnable for both sides."
        );
    }

    #[test]
    fn turn_equality() {
        use crate::board::evaluation::parameters::EvalParams;
        const FEN1: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        const FEN2: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1";
        crate::magic::initialise();
        let tempo = EvalParams::default().tempo.0;
        let board1 = super::Board::from_fen(FEN1).unwrap();
        let board2 = super::Board::from_fen(FEN2).unwrap();
        let eval1 = board1.evaluate_classical(0);
        let eval2 = board2.evaluate_classical(0);
        assert_eq!(eval1, -eval2 + 2 * tempo);
    }

    #[test]
    fn startpos_mobility_equality() {
        use crate::board::evaluation::S;
        crate::magic::initialise();
        let board = super::Board::default();
        assert_eq!(board.mobility_threats_kingdanger().0, S(0, 0));
    }

    #[test]
    fn startpos_eval_equality() {
        use crate::board::evaluation::parameters::EvalParams;
        crate::magic::initialise();
        let tempo = EvalParams::default().tempo.0;
        let board = super::Board::default();
        assert_eq!(board.evaluate_classical(0), tempo);
    }

    #[test]
    fn startpos_bits_equality() {
        use crate::board::evaluation::score::S;

        crate::magic::initialise();

        let board = super::Board::default();

        let material = board.material[crate::definitions::WHITE as usize]
            - board.material[crate::definitions::BLACK as usize];
        let pst = board.pst_vals;
        let pawn_val = board.pawn_structure_term();
        let bishop_pair_val = board.bishop_pair_term();
        let mobility_val = board.mobility_threats_kingdanger().0;
        let rook_open_file_val = board.rook_open_file_term();
        let queen_open_file_val = board.queen_open_file_term();

        assert_eq!(material, S(0, 0));
        assert_eq!(pst, S(0, 0));
        assert_eq!(pawn_val, S(0, 0));
        assert_eq!(bishop_pair_val, S(0, 0));
        assert_eq!(mobility_val, S(0, 0));
        assert_eq!(rook_open_file_val, S(0, 0));
        assert_eq!(queen_open_file_val, S(0, 0));
    }

    #[test]
    fn startpos_pawn_structure_equality() {
        use crate::board::evaluation::S;
        crate::magic::initialise();
        let board = super::Board::default();
        assert_eq!(board.pawn_structure_term(), S(0, 0));
    }

    #[test]
    fn startpos_open_file_equality() {
        use crate::board::evaluation::S;
        crate::magic::initialise();
        let board = super::Board::default();
        let rook_points = board.rook_open_file_term();
        let queen_points = board.queen_open_file_term();
        assert_eq!(rook_points + queen_points, S(0, 0));
    }

    #[test]
    fn double_pawn_eval() {
        use super::Board;
        use crate::board::evaluation::DOUBLED_PAWN_MALUS;

        let board =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/8/5P2/PPPP1PPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let pawn_eval = board.pawn_structure_term();
        assert_eq!(pawn_eval, -DOUBLED_PAWN_MALUS);
        let board =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/8/2P2P2/PPP2PPP/RNBQKBNR b KQkq - 0 1").unwrap();
        let pawn_eval = board.pawn_structure_term();
        assert_eq!(pawn_eval, -DOUBLED_PAWN_MALUS * 2);
    }

    #[test]
    fn passers_should_be_pushed() {
        use super::Board;

        let starting_rank_passer = Board::from_fen("8/k7/8/8/8/8/K6P/8 w - - 0 1").unwrap();
        let end_rank_passer = Board::from_fen("8/k6P/8/8/8/8/K7/8 w - - 0 1").unwrap();

        let starting_rank_eval = starting_rank_passer.evaluate_classical(0);
        let end_rank_eval = end_rank_passer.evaluate_classical(0);

        // is should be better to have a passer that is more advanced.
        assert!(
            end_rank_eval > starting_rank_eval,
            "end_rank_eval: {end_rank_eval}, starting_rank_eval: {starting_rank_eval}"
        );
    }
}
