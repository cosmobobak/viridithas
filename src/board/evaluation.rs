// The granularity of evaluation in this engine is in centipawns.

pub mod parameters;
pub mod score;

use score::S;

use crate::{
    board::Board,
    chessmove::Move,
    lookups::{init_eval_masks, init_passed_isolated_bb},
    piece::{Colour, Piece, PieceType},
    search::draw_score,
    searchinfo::SearchInfo,
    squareset::SquareSet,
    threadlocal::ThreadData,
    util::{Square, MAX_DEPTH},
};

use super::movegen::bitboards::{self, DARK_SQUARE, LIGHT_SQUARE};

pub const PAWN_VALUE: S = S(126, 196);
pub const KNIGHT_VALUE: S = S(437, 455);
pub const BISHOP_VALUE: S = S(437, 491);
pub const ROOK_VALUE: S = S(604, 805);
pub const QUEEN_VALUE: S = S(1369, 1274);

/// The value of checkmate.
/// To recover depth-to-mate, we subtract depth (ply) from this value.
/// e.g. if white has a mate in two ply, the output from a depth-5 search will be
/// two less than `MATE_SCORE`.
pub const MATE_SCORE: i32 = i16::MAX as i32 - 300;
pub const fn mate_in(ply: usize) -> i32 {
    #![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    debug_assert!(ply <= MAX_DEPTH.ply_to_horizon());
    MATE_SCORE - ply as i32
}
pub const fn mated_in(ply: usize) -> i32 {
    #![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    debug_assert!(ply <= MAX_DEPTH.ply_to_horizon());
    -MATE_SCORE + ply as i32
}
pub const TB_WIN_SCORE: i32 = MATE_SCORE - 1000;
pub const fn tb_win_in(ply: usize) -> i32 {
    #![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    debug_assert!(ply <= MAX_DEPTH.ply_to_horizon());
    TB_WIN_SCORE - ply as i32
}
pub const fn tb_loss_in(ply: usize) -> i32 {
    #![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    debug_assert!(ply <= MAX_DEPTH.ply_to_horizon());
    -TB_WIN_SCORE + ply as i32
}

/// A threshold over which scores must be mate.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub const MINIMUM_MATE_SCORE: i32 = MATE_SCORE - MAX_DEPTH.ply_to_horizon() as i32;
/// A threshold over which scores must be a TB win (or mate).
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub const MINIMUM_TB_WIN_SCORE: i32 = TB_WIN_SCORE - MAX_DEPTH.ply_to_horizon() as i32;

#[rustfmt::skip]
pub const PIECE_VALUES: [S; 13] = [
    PAWN_VALUE, KNIGHT_VALUE, BISHOP_VALUE, ROOK_VALUE, QUEEN_VALUE, S(0, 0),
    PAWN_VALUE, KNIGHT_VALUE, BISHOP_VALUE, ROOK_VALUE, QUEEN_VALUE, S(0, 0),
    S(0, 0),
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
const KNIGHT_MOBILITY_BONUS: [S; 9] = [S(-143, -214), S(-16, -13), S(10, 50), S(29, 105), S(47, 128), S(53, 162), S(73, 161), S(97, 161), S(128, 129)];
#[rustfmt::skip]
const BISHOP_MOBILITY_BONUS: [S; 14] = [S(-17, -115), S(-20, -80), S(30, -2), S(45, 74), S(67, 95), S(80, 115), S(86, 138), S(93, 157), S(101, 159), S(107, 167), S(112, 158), S(120, 154), S(200, 123), S(175, 150)];
#[rustfmt::skip]
const ROOK_MOBILITY_BONUS: [S; 15] = [S(-9, -2), S(-1, 103), S(23, 136), S(33, 184), S(41, 225), S(48, 237), S(47, 262), S(54, 269), S(63, 262), S(71, 276), S(72, 285), S(84, 284), S(96, 277), S(116, 263), S(104, 265)];
#[rustfmt::skip]
const QUEEN_MOBILITY_BONUS: [S; 28] = [S(-29, -49), S(-185, -204), S(-116, -296), S(184, 160), S(139, 64), S(218, 101), S(220, 143), S(218, 224), S(217, 299), S(217, 336), S(219, 373), S(225, 393), S(231, 416), S(237, 426), S(242, 442), S(243, 456), S(245, 465), S(255, 455), S(256, 461), S(277, 438), S(285, 432), S(334, 387), S(365, 375), S(401, 330), S(423, 304), S(362, 293), S(361, 299), S(264, 225)];

/// The bonus applied when a pawn has no pawns of the opposite colour ahead of it, or to the left or right, scaled by the rank that the pawn is on.
pub const PASSED_PAWN_BONUS: [S; 6] =
    [S(-8, 32), S(-26, 43), S(-28, 83), S(13, 124), S(37, 227), S(101, 264)];

pub const TEMPO_BONUS: S = S(12, 0);

pub const PAWN_THREAT_ON_MINOR: S = S(80, 79);
pub const PAWN_THREAT_ON_MAJOR: S = S(74, 55);
pub const MINOR_THREAT_ON_MAJOR: S = S(75, 53);

pub const KING_DANGER_COEFFS: [i32; 3] = [36, 165, -719];
pub const KING_DANGER_PIECE_WEIGHTS: [i32; 8] = [40, 20, 40, 20, 60, 20, 100, 40];
const KINGDANGER_DESCALE: i32 = 20;

const PAWN_PHASE: i32 = 1;
const KNIGHT_PHASE: i32 = 10;
const BISHOP_PHASE: i32 = 10;
const ROOK_PHASE: i32 = 20;
const QUEEN_PHASE: i32 = 40;
const TOTAL_PHASE: i32 =
    16 * PAWN_PHASE + 4 * KNIGHT_PHASE + 4 * BISHOP_PHASE + 4 * ROOK_PHASE + 2 * QUEEN_PHASE;

pub static FILE_BB: [SquareSet; 8] = init_eval_masks().1;

pub static WHITE_PASSED_BB: [SquareSet; 64] = init_passed_isolated_bb().0;
pub static BLACK_PASSED_BB: [SquareSet; 64] = init_passed_isolated_bb().1;

pub static ISOLATED_BB: [SquareSet; 64] = init_passed_isolated_bb().2;

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
pub const fn is_game_theoretic_score(score: i32) -> bool {
    score.abs() >= MINIMUM_TB_WIN_SCORE
}

impl Board {
    /// Computes a score for the position, from the point of view of the side to move.
    /// This function should strive to be as cheap to call as possible, relying on
    /// incremental updates in make-unmake to avoid recomputation.
    pub fn evaluate_classical(&self, t: &ThreadData, i: &SearchInfo, nodes: u64) -> i32 {
        if !self.pieces.any_pawns() && self.is_material_draw() {
            return if self.side == Colour::WHITE {
                draw_score(t, nodes, self.turn())
            } else {
                -draw_score(t, nodes, self.turn())
            };
        }

        let material = self.material();
        let pst = self.pst_vals;

        let pawn_structure = self.pawn_structure_term(i);
        let bishop_pair = self.bishop_pair_term(i);
        let rook_files = self.rook_open_file_term(i);
        let queen_files = self.queen_open_file_term(i);
        let (mobility, threats, danger_info) = self.mobility_threats_kingdanger(i);
        let king_danger = Self::score_kingdanger(danger_info, i);
        let tempo = i.eval_params.tempo;
        let tempo = if self.turn() == Colour::WHITE { tempo } else { -tempo };

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

        let score = self.preprocess_drawish_scores(t, score, nodes);

        if self.side == Colour::WHITE {
            score
        } else {
            -score
        }
    }

    fn material(&self) -> S {
        self.material[Colour::WHITE.index()] - self.material[Colour::BLACK.index()]
    }

    const fn material_scale(&self) -> i32 {
        #![allow(clippy::cast_possible_wrap)]
        700 + (PieceType::KNIGHT.see_value() * self.pieces.all_knights().count() as i32
            + PieceType::BISHOP.see_value() * self.pieces.all_bishops().count() as i32
            + PieceType::ROOK.see_value() * self.pieces.all_rooks().count() as i32
            + PieceType::QUEEN.see_value() * self.pieces.all_queens().count() as i32)
            / 32
    }

    pub fn evaluate_nnue(&self, t: &ThreadData, nodes: u64) -> i32 {
        if !self.pieces.any_pawns() && self.is_material_draw() {
            return if self.side == Colour::WHITE {
                draw_score(t, nodes, self.turn())
            } else {
                -draw_score(t, nodes, self.turn())
            };
        }

        let v = t.nnue.evaluate(self.side);
        let v = v * self.material_scale() / 1024;

        v * (200 - i32::from(self.fifty_move_counter)) / 200
    }

    pub fn evaluate<const USE_NNUE: bool>(
        &self,
        i: &SearchInfo,
        t: &ThreadData,
        nodes: u64,
    ) -> i32 {
        if USE_NNUE {
            self.evaluate_nnue(t, nodes)
        } else {
            self.evaluate_classical(t, i, nodes)
        }
    }

    fn unwinnable_for<const IS_WHITE: bool>(&self) -> bool {
        if IS_WHITE {
            if self.major_piece_counts[Colour::WHITE.index()] != 0 {
                return false;
            }
            if self.minor_piece_counts[Colour::WHITE.index()] > 1 {
                return false;
            }
            if self.num(Piece::WP) != 0 {
                return false;
            }
        } else {
            if self.major_piece_counts[Colour::BLACK.index()] != 0 {
                return false;
            }
            if self.minor_piece_counts[Colour::BLACK.index()] > 1 {
                return false;
            }
            if self.num(Piece::BP) != 0 {
                return false;
            }
        }

        true
    }

    fn is_material_draw(&self) -> bool {
        if self.num_pt(PieceType::ROOK) == 0 && self.num_pt(PieceType::QUEEN) == 0 {
            if self.num_pt(PieceType::BISHOP) == 0 {
                if self.num(Piece::WN) < 3 && self.num(Piece::BN) < 3 {
                    return true;
                }
            } else if (self.num_pt(PieceType::KNIGHT) == 0
                && self.num(Piece::WB).abs_diff(self.num(Piece::BB)) < 2)
                || (self.num(Piece::WB) + self.num(Piece::WN) == 1
                    && self.num(Piece::BB) + self.num(Piece::BN) == 1)
            {
                return true;
            }
        } else if self.num_pt(PieceType::QUEEN) == 0 {
            if self.num(Piece::WR) == 1 && self.num(Piece::BR) == 1 {
                if (self.num(Piece::WN) + self.num(Piece::WB)) < 2
                    && (self.num(Piece::BN) + self.num(Piece::BB)) < 2
                {
                    return true;
                }
            } else if self.num(Piece::WR) == 1 && self.num(Piece::BR) == 0 {
                if (self.num(Piece::WN) + self.num(Piece::WB)) == 0
                    && ((self.num(Piece::BN) + self.num(Piece::BB)) == 1
                        || (self.num(Piece::BN) + self.num(Piece::BB)) == 2)
                {
                    return true;
                }
            } else if self.num(Piece::WR) == 0
                && self.num(Piece::BR) == 1
                && (self.num(Piece::BN) + self.num(Piece::BB)) == 0
                && ((self.num(Piece::WN) + self.num(Piece::WB)) == 1
                    || (self.num(Piece::WN) + self.num(Piece::WB)) == 2)
            {
                return true;
            }
        }
        false
    }

    fn preprocess_drawish_scores(&self, t: &ThreadData, score: i32, nodes: u64) -> i32 {
        // if we can't win with our material, we clamp the eval to zero.
        let drawscore = draw_score(t, nodes, self.turn());
        if score > drawscore && self.unwinnable_for::<true>()
            || score < drawscore && self.unwinnable_for::<false>()
        {
            drawscore
        } else {
            score
        }
    }

    pub fn zugzwang_unlikely(&self) -> bool {
        let stm = self.turn();
        let us = self.pieces.occupied_co(stm);
        let kings = self.pieces.all_kings();
        let pawns = self.pieces.all_pawns();
        (us & (kings | pawns)) != us
    }

    fn bishop_pair_term(&self, i: &SearchInfo) -> S {
        let white_pair = self.pieces.bishops_sqco::<true, LIGHT_SQUARE>().non_empty()
            && self.pieces.bishops_sqco::<true, DARK_SQUARE>().non_empty();
        let black_pair = self.pieces.bishops_sqco::<false, LIGHT_SQUARE>().non_empty()
            && self.pieces.bishops_sqco::<false, DARK_SQUARE>().non_empty();
        let multiplier = i32::from(white_pair) - i32::from(black_pair);
        i.eval_params.bishop_pair_bonus * multiplier
    }

    fn pawn_structure_term(&self, i: &SearchInfo) -> S {
        #![allow(clippy::cast_possible_wrap)]
        /// not a tunable parameter, just how "number of pawns in a file" is mapped to "amount of doubled pawn-ness"
        static DOUBLED_PAWN_MAPPING: [i32; 7] = [0, 0, 1, 2, 3, 4, 5];
        let mut w_score = S(0, 0);
        let mut b_score = S(0, 0);
        let white_pawns = self.pieces.pawns::<true>();
        let black_pawns = self.pieces.pawns::<false>();

        for white_pawn_loc in white_pawns.iter() {
            if (ISOLATED_BB[white_pawn_loc.index()] & white_pawns).is_empty() {
                w_score -= i.eval_params.isolated_pawn_malus;
            }

            if (WHITE_PASSED_BB[white_pawn_loc.index()] & black_pawns).is_empty() {
                let rank = white_pawn_loc.rank() as usize;
                w_score += i.eval_params.passed_pawn_bonus[rank - 1];
            }
        }

        for black_pawn_loc in black_pawns.iter() {
            if (ISOLATED_BB[black_pawn_loc.index()] & black_pawns).is_empty() {
                b_score -= i.eval_params.isolated_pawn_malus;
            }

            if (BLACK_PASSED_BB[black_pawn_loc.index()] & white_pawns).is_empty() {
                let rank = black_pawn_loc.rank() as usize;
                b_score += i.eval_params.passed_pawn_bonus[7 - rank - 1];
            }
        }

        for &file_mask in &FILE_BB {
            let pawns_in_file = (file_mask & white_pawns).count() as usize;
            let multiplier = DOUBLED_PAWN_MAPPING[pawns_in_file];
            w_score -= i.eval_params.doubled_pawn_malus * multiplier;
            let pawns_in_file = (file_mask & black_pawns).count() as usize;
            let multiplier = DOUBLED_PAWN_MAPPING[pawns_in_file];
            b_score -= i.eval_params.doubled_pawn_malus * multiplier;
        }

        w_score - b_score
    }

    fn is_file_open(&self, file: u8) -> bool {
        let mask = FILE_BB[file as usize];
        let pawns = self.pieces.all_pawns();
        (mask & pawns).is_empty()
    }

    fn is_file_halfopen<const SIDE: u8>(&self, file: u8) -> bool {
        let mask = FILE_BB[file as usize];
        let pawns = if Colour::new(SIDE) == Colour::WHITE {
            self.pieces.pawns::<true>()
        } else {
            self.pieces.pawns::<false>()
        };
        (mask & pawns).is_empty()
    }

    fn rook_open_file_term(&self, i: &SearchInfo) -> S {
        let mut score = S(0, 0);
        for rook_sq in self.pieces.rooks::<true>().iter() {
            let file = rook_sq.file();
            if self.is_file_open(file) {
                score += i.eval_params.rook_open_file_bonus;
            } else if self.is_file_halfopen::<{ Colour::WHITE.inner() }>(file) {
                score += i.eval_params.rook_half_open_file_bonus;
            }
        }
        for rook_sq in self.pieces.rooks::<false>().iter() {
            let file = rook_sq.file();
            if self.is_file_open(file) {
                score -= i.eval_params.rook_open_file_bonus;
            } else if self.is_file_halfopen::<{ Colour::BLACK.inner() }>(file) {
                score -= i.eval_params.rook_half_open_file_bonus;
            }
        }
        score
    }

    fn queen_open_file_term(&self, i: &SearchInfo) -> S {
        let mut score = S(0, 0);
        for queen_sq in self.pieces.queens::<true>().iter() {
            let file = queen_sq.file();
            if self.is_file_open(file) {
                score += i.eval_params.queen_open_file_bonus;
            } else if self.is_file_halfopen::<{ Colour::WHITE.inner() }>(file) {
                score += i.eval_params.queen_half_open_file_bonus;
            }
        }
        for queen_sq in self.pieces.queens::<false>().iter() {
            let file = queen_sq.file();
            if self.is_file_open(file) {
                score -= i.eval_params.queen_open_file_bonus;
            } else if self.is_file_halfopen::<{ Colour::BLACK.inner() }>(file) {
                score -= i.eval_params.queen_half_open_file_bonus;
            }
        }
        score
    }

    /// `phase` computes a number between 0 and 256, which is the phase of the game. 0 is the opening, 256 is the endgame.
    pub fn phase(&self) -> i32 {
        // todo: this can be incrementally updated.
        let pawns = self.num(Piece::WP) + self.num(Piece::BP);
        let knights = self.num(Piece::WN) + self.num(Piece::BN);
        let bishops = self.num(Piece::WB) + self.num(Piece::BB);
        let rooks = self.num(Piece::WR) + self.num(Piece::BR);
        let queens = self.num(Piece::WQ) + self.num(Piece::BQ);
        game_phase(pawns, knights, bishops, rooks, queens)
    }

    #[allow(clippy::too_many_lines)]
    fn mobility_threats_kingdanger(&self, i: &SearchInfo) -> (S, S, KingDangerInfo) {
        #![allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)] // for count, which can return at most 64.
        let mut king_danger_info =
            KingDangerInfo { attack_units_on_white: 0, attack_units_on_black: 0 };
        let ptmul = &i.eval_params.king_danger_piece_weights;
        let mut mob_score = S(0, 0);
        let mut threat_score = S(0, 0);
        let white_king_area = king_area::<true>(self.king_sq(Colour::WHITE));
        let black_king_area = king_area::<false>(self.king_sq(Colour::BLACK));
        let white_pawn_attacks = self.pieces.pawn_attacks::<true>();
        let black_pawn_attacks = self.pieces.pawn_attacks::<false>();
        let white_minor = self.pieces.minors::<true>();
        let black_minor = self.pieces.minors::<false>();
        let white_major = self.pieces.majors::<true>();
        let black_major = self.pieces.majors::<false>();
        threat_score +=
            i.eval_params.pawn_threat_on_minor * (black_minor & white_pawn_attacks).count() as i32;
        threat_score -=
            i.eval_params.pawn_threat_on_minor * (white_minor & black_pawn_attacks).count() as i32;
        threat_score +=
            i.eval_params.pawn_threat_on_major * (black_major & white_pawn_attacks).count() as i32;
        threat_score -=
            i.eval_params.pawn_threat_on_major * (white_major & black_pawn_attacks).count() as i32;
        let safe_white_moves = !black_pawn_attacks;
        let safe_black_moves = !white_pawn_attacks;
        let blockers = self.pieces.occupied();
        for knight_sq in self.pieces.knights::<true>().iter() {
            let attacks = bitboards::knight_attacks(knight_sq);
            // kingsafety
            let attacks_on_black_king = attacks & black_king_area;
            let defense_of_white_king = attacks & white_king_area;
            king_danger_info.attack_units_on_black +=
                attacks_on_black_king.count() as i32 * ptmul[0];
            king_danger_info.attack_units_on_white -=
                defense_of_white_king.count() as i32 * ptmul[1];
            // threats
            let attacks_on_majors = attacks & black_major;
            threat_score += i.eval_params.minor_threat_on_major * attacks_on_majors.count() as i32;
            // mobility
            let attacks = attacks & safe_white_moves;
            let attacks = attacks.count() as usize;
            mob_score += i.eval_params.knight_mobility_bonus[attacks];
        }
        for knight_sq in self.pieces.knights::<false>().iter() {
            let attacks = bitboards::knight_attacks(knight_sq);
            // kingsafety
            let attacks_on_white_king = attacks & white_king_area;
            let defense_of_black_king = attacks & black_king_area;
            king_danger_info.attack_units_on_white +=
                attacks_on_white_king.count() as i32 * ptmul[0];
            king_danger_info.attack_units_on_black -=
                defense_of_black_king.count() as i32 * ptmul[1];
            // threats
            let attacks_on_majors = attacks & white_major;
            threat_score -= i.eval_params.minor_threat_on_major * attacks_on_majors.count() as i32;
            // mobility
            let attacks = attacks & safe_black_moves;
            let attacks = attacks.count() as usize;
            mob_score -= i.eval_params.knight_mobility_bonus[attacks];
        }
        for bishop_sq in self.pieces.bishops::<true>().iter() {
            let attacks = bitboards::bishop_attacks(bishop_sq, blockers);
            // kingsafety
            let attacks_on_black_king = attacks & black_king_area;
            let defense_of_white_king = attacks & white_king_area;
            king_danger_info.attack_units_on_black +=
                attacks_on_black_king.count() as i32 * ptmul[2];
            king_danger_info.attack_units_on_white -=
                defense_of_white_king.count() as i32 * ptmul[3];
            // threats
            let attacks_on_majors = attacks & black_major;
            threat_score += i.eval_params.minor_threat_on_major * attacks_on_majors.count() as i32;
            // mobility
            let attacks = attacks & safe_white_moves;
            let attacks = attacks.count() as usize;
            mob_score += i.eval_params.bishop_mobility_bonus[attacks];
        }
        for bishop_sq in self.pieces.bishops::<false>().iter() {
            let attacks = bitboards::bishop_attacks(bishop_sq, blockers);
            // kingsafety
            let attacks_on_white_king = attacks & white_king_area;
            let defense_of_black_king = attacks & black_king_area;
            king_danger_info.attack_units_on_white +=
                attacks_on_white_king.count() as i32 * ptmul[2];
            king_danger_info.attack_units_on_black -=
                defense_of_black_king.count() as i32 * ptmul[3];
            // threats
            let attacks_on_majors = attacks & white_major;
            threat_score -= i.eval_params.minor_threat_on_major * attacks_on_majors.count() as i32;
            // mobility
            let attacks = attacks & safe_black_moves;
            let attacks = attacks.count() as usize;
            mob_score -= i.eval_params.bishop_mobility_bonus[attacks];
        }
        for rook_sq in self.pieces.rooks::<true>().iter() {
            let attacks = bitboards::rook_attacks(rook_sq, blockers);
            // kingsafety
            let attacks_on_black_king = attacks & black_king_area;
            let defense_of_white_king = attacks & white_king_area;
            king_danger_info.attack_units_on_black +=
                attacks_on_black_king.count() as i32 * ptmul[4];
            king_danger_info.attack_units_on_white -=
                defense_of_white_king.count() as i32 * ptmul[5];
            // mobility
            let attacks = attacks & safe_white_moves;
            let attacks = attacks.count() as usize;
            mob_score += i.eval_params.rook_mobility_bonus[attacks];
        }
        for rook_sq in self.pieces.rooks::<false>().iter() {
            let attacks = bitboards::rook_attacks(rook_sq, blockers);
            // kingsafety
            let attacks_on_white_king = attacks & white_king_area;
            let defense_of_black_king = attacks & black_king_area;
            king_danger_info.attack_units_on_white +=
                attacks_on_white_king.count() as i32 * ptmul[4];
            king_danger_info.attack_units_on_black -=
                defense_of_black_king.count() as i32 * ptmul[5];
            // mobility
            let attacks = attacks & safe_black_moves;
            let attacks = attacks.count() as usize;
            mob_score -= i.eval_params.rook_mobility_bonus[attacks];
        }
        for queen_sq in self.pieces.queens::<true>().iter() {
            let attacks = bitboards::queen_attacks(queen_sq, blockers);
            // kingsafety
            let attacks_on_black_king = attacks & black_king_area;
            let defense_of_white_king = attacks & white_king_area;
            king_danger_info.attack_units_on_black +=
                attacks_on_black_king.count() as i32 * ptmul[6];
            king_danger_info.attack_units_on_white -=
                defense_of_white_king.count() as i32 * ptmul[7];
            // mobility
            let attacks = attacks & safe_white_moves;
            let attacks = attacks.count() as usize;
            mob_score += i.eval_params.queen_mobility_bonus[attacks];
        }
        for queen_sq in self.pieces.queens::<false>().iter() {
            let attacks = bitboards::queen_attacks(queen_sq, blockers);
            // kingsafety
            let attacks_on_white_king = attacks & white_king_area;
            let defense_of_black_king = attacks & black_king_area;
            king_danger_info.attack_units_on_white +=
                attacks_on_white_king.count() as i32 * ptmul[6];
            king_danger_info.attack_units_on_black -=
                defense_of_black_king.count() as i32 * ptmul[7];
            // mobility
            let attacks = attacks & safe_black_moves;
            let attacks = attacks.count() as usize;
            mob_score -= i.eval_params.queen_mobility_bonus[attacks];
        }
        king_danger_info.attack_units_on_white /= KINGDANGER_DESCALE;
        king_danger_info.attack_units_on_black /= KINGDANGER_DESCALE;
        (mob_score, threat_score, king_danger_info)
    }

    fn score_kingdanger(kd: KingDangerInfo, i: &SearchInfo) -> S {
        let [a, b, c] = i.eval_params.king_danger_coeffs;
        let kd_formula = |au| (a * au * au + b * au + c) / 100;

        let white_attack_strength = kd_formula(kd.attack_units_on_black.clamp(0, 99)).min(500);
        let black_attack_strength = kd_formula(kd.attack_units_on_white.clamp(0, 99)).min(500);
        let relscore = white_attack_strength - black_attack_strength;
        S(relscore, relscore / 2)
    }

    pub fn estimated_see(&self, m: Move) -> i32 {
        // initially take the value of the thing on the target square
        let mut value = self.piece_at(m.to()).piece_type().see_value();

        if m.is_promo() {
            // if it's a promo, swap a pawn for the promoted piece type
            value += m.promotion_type().see_value() - PieceType::PAWN.see_value();
        } else if m.is_ep() {
            // for e.p. we will miss a pawn because the target square is empty
            value = PieceType::PAWN.see_value();
        }

        value
    }
}

pub fn king_area<const IS_WHITE: bool>(king_sq: Square) -> SquareSet {
    let king_attacks = bitboards::king_attacks(king_sq);
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
        use crate::threadlocal::ThreadData;
        const FEN: &str = "8/8/8/8/2K2k2/2n2P2/8/8 b - - 1 1";

        let board = super::Board::from_fen(FEN).unwrap();
        let stopped = std::sync::atomic::AtomicBool::new(false);
        let nodes = std::sync::atomic::AtomicU64::new(0);
        let info = crate::searchinfo::SearchInfo::new(&stopped, &nodes);
        let thread = ThreadData::new(0, &board);
        let eval = board.evaluate_classical(&thread, &info, 0);
        assert!(
            (-2..=2).contains(&(eval.abs())),
            "eval is not a draw score in a position unwinnable for both sides."
        );
    }

    #[test]
    fn turn_equality() {
        use crate::board::evaluation::parameters::EvalParams;
        use crate::threadlocal::ThreadData;
        const FEN1: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        const FEN2: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1";

        let tempo = EvalParams::default().tempo.0;
        let board1 = super::Board::from_fen(FEN1).unwrap();
        let board2 = super::Board::from_fen(FEN2).unwrap();
        let stopped = std::sync::atomic::AtomicBool::new(false);
        let nodes = std::sync::atomic::AtomicU64::new(0);
        let info = crate::searchinfo::SearchInfo::new(&stopped, &nodes);
        let thread1 = ThreadData::new(0, &board1);
        let thread2 = ThreadData::new(0, &board2);
        let eval1 = board1.evaluate_classical(&thread1, &info, 0);
        let eval2 = board2.evaluate_classical(&thread2, &info, 0);
        assert_eq!(eval1, -eval2 + 2 * tempo);
    }

    #[test]
    fn startpos_mobility_equality() {
        use crate::board::evaluation::S;

        let board = super::Board::default();
        let stopped = std::sync::atomic::AtomicBool::new(false);
        let nodes = std::sync::atomic::AtomicU64::new(0);
        let info = crate::searchinfo::SearchInfo::new(&stopped, &nodes);
        assert_eq!(board.mobility_threats_kingdanger(&info).0, S(0, 0));
    }

    #[test]
    fn startpos_eval_equality() {
        use crate::board::evaluation::parameters::EvalParams;
        use crate::threadlocal::ThreadData;

        let tempo = EvalParams::default().tempo.0;
        let board = super::Board::default();
        let stopped = std::sync::atomic::AtomicBool::new(false);
        let nodes = std::sync::atomic::AtomicU64::new(0);
        let info = crate::searchinfo::SearchInfo::new(&stopped, &nodes);
        let thread = ThreadData::new(0, &board);
        assert_eq!(board.evaluate_classical(&thread, &info, 0), tempo);
    }

    #[test]
    fn startpos_bits_equality() {
        use crate::board::evaluation::score::S;
        use crate::piece::Colour;

        let board = super::Board::default();
        let stopped = std::sync::atomic::AtomicBool::new(false);
        let nodes = std::sync::atomic::AtomicU64::new(0);
        let info = crate::searchinfo::SearchInfo::new(&stopped, &nodes);

        let material =
            board.material[Colour::WHITE.index()] - board.material[Colour::BLACK.index()];
        let pst = board.pst_vals;
        let pawn_val = board.pawn_structure_term(&info);
        let bishop_pair_val = board.bishop_pair_term(&info);
        let mobility_val = board.mobility_threats_kingdanger(&info).0;
        let rook_open_file_val = board.rook_open_file_term(&info);
        let queen_open_file_val = board.queen_open_file_term(&info);

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

        let board = super::Board::default();
        let stopped = std::sync::atomic::AtomicBool::new(false);
        let nodes = std::sync::atomic::AtomicU64::new(0);
        let info = crate::searchinfo::SearchInfo::new(&stopped, &nodes);
        assert_eq!(board.pawn_structure_term(&info), S(0, 0));
    }

    #[test]
    fn startpos_open_file_equality() {
        use crate::board::evaluation::S;

        let board = super::Board::default();
        let stopped = std::sync::atomic::AtomicBool::new(false);
        let nodes = std::sync::atomic::AtomicU64::new(0);
        let info = crate::searchinfo::SearchInfo::new(&stopped, &nodes);
        let rook_points = board.rook_open_file_term(&info);
        let queen_points = board.queen_open_file_term(&info);
        assert_eq!(rook_points + queen_points, S(0, 0));
    }

    #[test]
    fn double_pawn_eval() {
        use super::Board;
        use crate::board::evaluation::DOUBLED_PAWN_MALUS;
        let stopped = std::sync::atomic::AtomicBool::new(false);
        let nodes = std::sync::atomic::AtomicU64::new(0);
        let info = crate::searchinfo::SearchInfo::new(&stopped, &nodes);

        let board =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/8/5P2/PPPP1PPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let pawn_eval = board.pawn_structure_term(&info);
        assert_eq!(pawn_eval, -DOUBLED_PAWN_MALUS);
        let board =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/8/2P2P2/PPP2PPP/RNBQKBNR b KQkq - 0 1").unwrap();
        let pawn_eval = board.pawn_structure_term(&info);
        assert_eq!(pawn_eval, -DOUBLED_PAWN_MALUS * 2);
    }

    #[test]
    fn passers_should_be_pushed() {
        use super::Board;
        use crate::threadlocal::ThreadData;

        let starting_rank_passer = Board::from_fen("8/k7/8/8/8/8/K6P/8 w - - 0 1").unwrap();
        let end_rank_passer = Board::from_fen("8/k6P/8/8/8/8/K7/8 w - - 0 1").unwrap();

        let stopped = std::sync::atomic::AtomicBool::new(false);
        let nodes = std::sync::atomic::AtomicU64::new(0);
        let info = crate::searchinfo::SearchInfo::new(&stopped, &nodes);

        let thread1 = ThreadData::new(0, &starting_rank_passer);
        let thread2 = ThreadData::new(0, &end_rank_passer);

        let starting_rank_eval = starting_rank_passer.evaluate_classical(&thread1, &info, 0);
        let end_rank_eval = end_rank_passer.evaluate_classical(&thread2, &info, 0);

        // is should be better to have a passer that is more advanced.
        assert!(
            end_rank_eval > starting_rank_eval,
            "end_rank_eval: {end_rank_eval}, starting_rank_eval: {starting_rank_eval}"
        );
    }
}
