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
        WHITE, WN, WP, WQ, WR,
    },
    lookups::{file, init_eval_masks, init_passed_isolated_bb, rank},
    search::draw_score,
};

use super::movegen::{
    bitboards::{attacks, pawn_attacks, BitShiftExt, DARK_SQUARE, LIGHT_SQUARE},
    BitLoop, BB_NONE,
};

pub const PAWN_VALUE: S = S(116, 194);
pub const KNIGHT_VALUE: S = S(415, 416);
pub const BISHOP_VALUE: S = S(407, 448);
pub const ROOK_VALUE: S = S(572, 742);
pub const QUEEN_VALUE: S = S(1303, 1208);

pub static SEE_PIECE_VALUES: [i32; 7] = [
    0,
    PAWN_VALUE.value(128),
    KNIGHT_VALUE.value(128),
    BISHOP_VALUE.value(128),
    ROOK_VALUE.value(128),
    QUEEN_VALUE.value(128),
    0,
];

/// The value of checkmate.
/// To recover depth-to-mate, we subtract depth (ply) from this value.
/// e.g. if white has a mate in two ply, the output from a depth-5 search will be
/// `3_000_000 - 2 = 2_999_998`.
pub const MATE_SCORE: i32 = 3_000_000;

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
pub const ISOLATED_PAWN_MALUS: S = S(22, 11);

/// The malus applied when two (or more) pawns of a colour are on the same file.
pub const DOUBLED_PAWN_MALUS: S = S(18, 42);

/// The bonus granted for having two bishops.
pub const BISHOP_PAIR_BONUS: S = S(46, 116);

/// The bonus for having a rook on an open file.
pub const ROOK_OPEN_FILE_BONUS: S = S(70, 0);
/// The bonus for having a rook on a semi-open file.
pub const ROOK_HALF_OPEN_FILE_BONUS: S = S(38, 0);
/// The bonus for having a queen on an open file.
pub const QUEEN_OPEN_FILE_BONUS: S = S(-5, 0);
/// The bonus for having a queen on a semi-open file.
pub const QUEEN_HALF_OPEN_FILE_BONUS: S = S(22, 0);

// nonlinear mobility eval tables.
#[rustfmt::skip]
static KNIGHT_MOBILITY_BONUS: [S; 9] = [S(-188, -265), S(-10, 11), S(10, 29), S(24, 84), S(43, 98), S(49, 131), S(69, 135), S(92, 136), S(119, 107)];
#[rustfmt::skip]
static BISHOP_MOBILITY_BONUS: [S; 14] = [S(-12, -147), S(-10, -92), S(28, -5), S(43, 74), S(62, 94), S(76, 108), S(85, 131), S(92, 148), S(99, 147), S(102, 155), S(107, 146), S(109, 149), S(206, 110), S(178, 131)];
#[rustfmt::skip]
static ROOK_MOBILITY_BONUS: [S; 15] = [S(32, 34), S(41, 115), S(22, 137), S(35, 183), S(41, 212), S(48, 226), S(46, 252), S(52, 257), S(61, 248), S(65, 265), S(73, 270), S(80, 273), S(84, 268), S(98, 255), S(83, 254)];
#[rustfmt::skip]
static QUEEN_MOBILITY_BONUS: [S; 28] = [S(-29, -49), S(-161, -187), S(-172, -277), S(152, 176), S(93, 34), S(192, 95), S(205, 97), S(197, 173), S(194, 246), S(193, 293), S(198, 321), S(204, 342), S(209, 362), S(218, 366), S(221, 384), S(223, 400), S(223, 404), S(234, 397), S(244, 402), S(275, 372), S(271, 376), S(314, 352), S(329, 350), S(358, 316), S(375, 305), S(300, 315), S(308, 291), S(199, 201)];

/// The bonus applied when a pawn has no pawns of the opposite colour ahead of it, or to the left or right, scaled by the rank that the pawn is on.
pub static PASSED_PAWN_BONUS: [S; 6] =
    [S(1, 19), S(-23, 35), S(-26, 75), S(12, 112), S(35, 225), S(100, 311)];

pub const TEMPO_BONUS: S = S(12, 5);

pub const PAWN_THREAT_ON_MINOR: S = S(73, 69);
pub const PAWN_THREAT_ON_MAJOR: S = S(78, 67);
pub const MINOR_THREAT_ON_MAJOR: S = S(70, 60);

const KING_DANGER_COEFFS: [i32; 3] = [36, 181, -729];

pub const CONTEMPT: i32 = 10;

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
        self.eval_params = params;
    }

    /// Computes a score for the position, from the point of view of the side to move.
    /// This function should strive to be as cheap to call as possible, relying on
    /// incremental updates in make-unmake to avoid recomputation.
    pub fn evaluate(&mut self, nodes: u64) -> i32 {
        if !self.pieces.any_pawns() && self.is_material_draw() {
            return if self.side == WHITE { draw_score(nodes) } else { -draw_score(nodes) };
        }

        let material = self.material[WHITE as usize] - self.material[BLACK as usize];
        let pst = self.pst_vals;

        let pawn_structure = self.pawn_structure_term();
        let bishop_pair = self.bishop_pair_term();
        let rook_files = self.rook_open_file_term();
        let queen_files = self.queen_open_file_term();
        let (mobility, threats, danger_info) = self.mobility_threats_kingdanger();
        let king_danger = self.score_kingdanger(danger_info);
        let tempo =
            if self.turn() == WHITE { self.eval_params.tempo } else { -self.eval_params.tempo };

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

    fn preprocess_drawish_scores(&mut self, score: i32, nodes: u64) -> i32 {
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
        if white_pair && black_pair {
            return S(0, 0);
        }
        if white_pair {
            return self.eval_params.bishop_pair_bonus;
        }
        if black_pair {
            return -self.eval_params.bishop_pair_bonus;
        }
        S(0, 0)
    }

    #[rustfmt::skip]
    fn backward_pawns<const IS_WHITE: bool>(&self) -> u64 {
        // this function will be so much nicer with const operations.
        let our_pawns = self.pieces.pawns::<IS_WHITE>();
        let their_pawns = if IS_WHITE { self.pieces.pawns::<false>() } else { self.pieces.pawns::<true>() };
        let our_stops = if IS_WHITE { our_pawns.north_one() } else { our_pawns.south_one() };
        let our_pawn_attacks = pawn_attacks::<IS_WHITE>(our_pawns);
        let their_pawn_attacks = if IS_WHITE { pawn_attacks::<false>(their_pawns) } else { pawn_attacks::<true>(their_pawns) };
        let blocked_squares = our_stops & their_pawn_attacks & !our_pawn_attacks;
        if IS_WHITE { blocked_squares.south_one() } else { blocked_squares.north_one() }
    }

    fn pawn_structure_term(&self) -> S {
        /// not a tunable parameter, just how "number of pawns in a file" is mapped to "amount of doubled pawn-ness"
        static DOUBLED_PAWN_MAPPING: [i32; 7] = [0, 0, 1, 2, 3, 4, 5];
        let mut w_score = S(0, 0);
        let mut b_score = S(0, 0);
        let white_pawns = self.pieces.pawns::<true>();
        let black_pawns = self.pieces.pawns::<false>();

        for &white_pawn_loc in self.piece_lists[WP as usize].iter() {
            if ISOLATED_BB[white_pawn_loc as usize] & white_pawns == 0 {
                w_score -= self.eval_params.isolated_pawn_malus;
            }

            if WHITE_PASSED_BB[white_pawn_loc as usize] & black_pawns == 0 {
                let rank = rank(white_pawn_loc) as usize;
                w_score += self.eval_params.passed_pawn_bonus[rank - 1];
            }
        }

        for &black_pawn_loc in self.piece_lists[BP as usize].iter() {
            if ISOLATED_BB[black_pawn_loc as usize] & black_pawns == 0 {
                b_score -= self.eval_params.isolated_pawn_malus;
            }

            if BLACK_PASSED_BB[black_pawn_loc as usize] & white_pawns == 0 {
                let rank = rank(black_pawn_loc) as usize;
                b_score += self.eval_params.passed_pawn_bonus[7 - rank - 1];
            }
        }

        for &file_mask in &FILE_BB {
            let pawns_in_file = (file_mask & white_pawns).count_ones() as usize;
            let multiplier = DOUBLED_PAWN_MAPPING[pawns_in_file];
            w_score -= self.eval_params.doubled_pawn_malus * multiplier;
            let pawns_in_file = (file_mask & black_pawns).count_ones() as usize;
            let multiplier = DOUBLED_PAWN_MAPPING[pawns_in_file];
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
        let pawns =
            if SIDE == WHITE { self.pieces.pawns::<true>() } else { self.pieces.pawns::<false>() };
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

    #[allow(clippy::too_many_lines)]
    fn mobility_threats_kingdanger(&mut self) -> (S, S, KingDangerInfo) {
        #![allow(clippy::cast_possible_wrap)] // for count_ones, which can return at most 64.
        let mut king_danger_info =
            KingDangerInfo { attack_units_on_white: 0, attack_units_on_black: 0 };
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
        threat_score += self.eval_params.pawn_threat_on_minor
            * (black_minor & white_pawn_attacks).count_ones() as i32;
        threat_score -= self.eval_params.pawn_threat_on_minor
            * (white_minor & black_pawn_attacks).count_ones() as i32;
        threat_score += self.eval_params.pawn_threat_on_major
            * (black_major & white_pawn_attacks).count_ones() as i32;
        threat_score -= self.eval_params.pawn_threat_on_major
            * (white_major & black_pawn_attacks).count_ones() as i32;
        let safe_white_moves = !black_pawn_attacks;
        let safe_black_moves = !white_pawn_attacks;
        let blockers = self.pieces.occupied();
        for knight_sq in BitLoop::new(self.pieces.knights::<true>()) {
            let attacks = attacks::<KNIGHT>(knight_sq, BB_NONE);
            // kingsafety
            let attacks_on_black_king = attacks & black_king_area;
            let defense_of_white_king = attacks & white_king_area;
            king_danger_info.attack_units_on_black += attacks_on_black_king.count_ones() as i32 * 2;
            king_danger_info.attack_units_on_white -= defense_of_white_king.count_ones() as i32;
            // threats
            let attacks_on_majors = attacks & black_major;
            threat_score +=
                self.eval_params.minor_threat_on_major * attacks_on_majors.count_ones() as i32;
            // mobility
            let attacks = attacks & safe_white_moves;
            let attacks = attacks.count_ones() as usize;
            mob_score += self.eval_params.knight_mobility_bonus[attacks];
        }
        for knight_sq in BitLoop::new(self.pieces.knights::<false>()) {
            let attacks = attacks::<KNIGHT>(knight_sq, BB_NONE);
            // kingsafety
            let attacks_on_white_king = attacks & white_king_area;
            let defense_of_black_king = attacks & black_king_area;
            king_danger_info.attack_units_on_white += attacks_on_white_king.count_ones() as i32 * 2;
            king_danger_info.attack_units_on_black -= defense_of_black_king.count_ones() as i32;
            // threats
            let attacks_on_majors = attacks & white_major;
            threat_score -=
                self.eval_params.minor_threat_on_major * attacks_on_majors.count_ones() as i32;
            // mobility
            let attacks = attacks & safe_black_moves;
            let attacks = attacks.count_ones() as usize;
            mob_score -= self.eval_params.knight_mobility_bonus[attacks];
        }
        for bishop_sq in BitLoop::new(self.pieces.bishops::<true>()) {
            let attacks = attacks::<BISHOP>(bishop_sq, blockers);
            // kingsafety
            let attacks_on_black_king = attacks & black_king_area;
            let defense_of_white_king = attacks & white_king_area;
            king_danger_info.attack_units_on_black += attacks_on_black_king.count_ones() as i32 * 2;
            king_danger_info.attack_units_on_white -= defense_of_white_king.count_ones() as i32;
            // threats
            let attacks_on_majors = attacks & black_major;
            threat_score +=
                self.eval_params.minor_threat_on_major * attacks_on_majors.count_ones() as i32;
            // mobility
            let attacks = attacks & safe_white_moves;
            let attacks = attacks.count_ones() as usize;
            mob_score += self.eval_params.bishop_mobility_bonus[attacks];
        }
        for bishop_sq in BitLoop::new(self.pieces.bishops::<false>()) {
            let attacks = attacks::<BISHOP>(bishop_sq, blockers);
            // kingsafety
            let attacks_on_white_king = attacks & white_king_area;
            let defense_of_black_king = attacks & black_king_area;
            king_danger_info.attack_units_on_white += attacks_on_white_king.count_ones() as i32 * 2;
            king_danger_info.attack_units_on_black -= defense_of_black_king.count_ones() as i32;
            // threats
            let attacks_on_majors = attacks & white_major;
            threat_score -=
                self.eval_params.minor_threat_on_major * attacks_on_majors.count_ones() as i32;
            // mobility
            let attacks = attacks & safe_black_moves;
            let attacks = attacks.count_ones() as usize;
            mob_score -= self.eval_params.bishop_mobility_bonus[attacks];
        }
        for rook_sq in BitLoop::new(self.pieces.rooks::<true>()) {
            let attacks = attacks::<ROOK>(rook_sq, blockers);
            // kingsafety
            let attacks_on_black_king = attacks & black_king_area;
            let defense_of_white_king = attacks & white_king_area;
            king_danger_info.attack_units_on_black += attacks_on_black_king.count_ones() as i32 * 3;
            king_danger_info.attack_units_on_white -= defense_of_white_king.count_ones() as i32;
            // mobility
            let attacks = attacks & safe_white_moves;
            let attacks = attacks.count_ones() as usize;
            mob_score += self.eval_params.rook_mobility_bonus[attacks];
        }
        for rook_sq in BitLoop::new(self.pieces.rooks::<false>()) {
            let attacks = attacks::<ROOK>(rook_sq, blockers);
            // kingsafety
            let attacks_on_white_king = attacks & white_king_area;
            let defense_of_black_king = attacks & black_king_area;
            king_danger_info.attack_units_on_white += attacks_on_white_king.count_ones() as i32 * 3;
            king_danger_info.attack_units_on_black -= defense_of_black_king.count_ones() as i32;
            // mobility
            let attacks = attacks & safe_black_moves;
            let attacks = attacks.count_ones() as usize;
            mob_score -= self.eval_params.rook_mobility_bonus[attacks];
        }
        for queen_sq in BitLoop::new(self.pieces.queens::<true>()) {
            let attacks = attacks::<QUEEN>(queen_sq, blockers);
            // kingsafety
            let attacks_on_black_king = attacks & black_king_area;
            let defense_of_white_king = attacks & white_king_area;
            king_danger_info.attack_units_on_black += attacks_on_black_king.count_ones() as i32 * 5;
            king_danger_info.attack_units_on_white -= defense_of_white_king.count_ones() as i32 * 2;
            // mobility
            let attacks = attacks & safe_white_moves;
            let attacks = attacks.count_ones() as usize;
            mob_score += self.eval_params.queen_mobility_bonus[attacks];
        }
        for queen_sq in BitLoop::new(self.pieces.queens::<false>()) {
            let attacks = attacks::<QUEEN>(queen_sq, blockers);
            // kingsafety
            let attacks_on_white_king = attacks & white_king_area;
            let defense_of_black_king = attacks & black_king_area;
            king_danger_info.attack_units_on_white += attacks_on_white_king.count_ones() as i32 * 5;
            king_danger_info.attack_units_on_black -= defense_of_black_king.count_ones() as i32 * 2;
            // mobility
            let attacks = attacks & safe_black_moves;
            let attacks = attacks.count_ones() as usize;
            mob_score -= self.eval_params.queen_mobility_bonus[attacks];
        }
        (mob_score, threat_score, king_danger_info)
    }

    fn score_kingdanger(&self, kd: KingDangerInfo) -> S {
        #![allow(clippy::unused_self)]
        let [a, b, c] = self.eval_params.king_danger_coeffs;
        let kd_formula = |au| (a * au * au + b * au + c) / 100;

        let white_attack_strength = kd_formula(kd.attack_units_on_black.clamp(0, 99)).min(500);
        let black_attack_strength = kd_formula(kd.attack_units_on_white.clamp(0, 99)).min(500);
        let relscore = white_attack_strength - black_attack_strength;
        S(relscore, relscore / 2)
    }

    pub fn estimated_see(&self, m: Move) -> i32 {
        // initially take the value of the thing on the target square
        let mut value = SEE_PIECE_VALUES[type_of(self.piece_at(m.to())) as usize];

        if m.is_promo() {
            // if it's a promo, swap a pawn for the promoted piece type
            value +=
                SEE_PIECE_VALUES[type_of(m.promotion()) as usize] - SEE_PIECE_VALUES[PAWN as usize];
        } else if m.is_ep() {
            // for e.p. we will miss a pawn because the target square is empty
            value = SEE_PIECE_VALUES[PAWN as usize];
        }

        value
    }
}

pub fn king_area<const IS_WHITE: bool>(king_sq: u8) -> u64 {
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
        use crate::board::evaluation::CONTEMPT;
        const FEN: &str = "8/8/8/8/2K2k2/2n2P2/8/8 b - - 1 1";
        crate::magic::initialise();
        let mut board = super::Board::from_fen(FEN).unwrap();
        let eval = board.evaluate(0);
        assert!(
            (-2..=2).contains(&(eval.abs() - CONTEMPT)),
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
        let mut board1 = super::Board::from_fen(FEN1).unwrap();
        let mut board2 = super::Board::from_fen(FEN2).unwrap();
        let eval1 = board1.evaluate(0);
        let eval2 = board2.evaluate(0);
        assert_eq!(eval1, -eval2 + 2 * tempo);
    }

    #[test]
    fn startpos_mobility_equality() {
        use crate::board::evaluation::S;
        crate::magic::initialise();
        let mut board = super::Board::default();
        assert_eq!(board.mobility_threats_kingdanger().0, S(0, 0));
    }

    #[test]
    fn startpos_eval_equality() {
        use crate::board::evaluation::parameters::EvalParams;
        crate::magic::initialise();
        let tempo = EvalParams::default().tempo.0;
        let mut board = super::Board::default();
        assert_eq!(board.evaluate(0), tempo);
    }

    #[test]
    fn startpos_bits_equality() {
        use crate::board::evaluation::score::S;

        crate::magic::initialise();

        let mut board = super::Board::default();

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

        let mut starting_rank_passer = Board::from_fen("8/k7/8/8/8/8/K6P/8 w - - 0 1").unwrap();
        let mut end_rank_passer = Board::from_fen("8/k6P/8/8/8/8/K7/8 w - - 0 1").unwrap();

        let starting_rank_eval = starting_rank_passer.evaluate(0);
        let end_rank_eval = end_rank_passer.evaluate(0);

        // is should be better to have a passer that is more advanced.
        assert!(
            end_rank_eval > starting_rank_eval,
            "end_rank_eval: {end_rank_eval}, starting_rank_eval: {starting_rank_eval}"
        );
    }
}
