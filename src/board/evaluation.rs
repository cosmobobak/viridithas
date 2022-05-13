// The granularity of evaluation in this engine is going to be thousandths of a pawn.

use crate::{
    board::Board,
    chessmove::Move,
    definitions::{BB, BK, BLACK, BN, BOTH, BP, BQ, BR, WB, WHITE, WK, WN, WP, WQ, WR},
    lookups::{init_eval_masks, init_passed_isolated_bb, FILES_BOARD, RANKS_BOARD, SQ120_TO_SQ64},
    board::movegen::MoveConsumer,
    piecesquaretable::{ENDGAME_PST, MIDGAME_PST},
};

// These piece values are taken from PeSTO (which in turn took them from RofChade 1.0).
pub const MG_PAWN_VALUE: i32 = 82;
pub const MG_KNIGHT_VALUE: i32 = 337;
pub const MG_BISHOP_VALUE: i32 = 365;
pub const MG_ROOK_VALUE: i32 = 477;
pub const MG_QUEEN_VALUE: i32 = 1025;

pub const EG_PAWN_VALUE: i32 = 94;
pub const EG_KNIGHT_VALUE: i32 = 281;
pub const EG_BISHOP_VALUE: i32 = 297;
pub const EG_ROOK_VALUE: i32 = 512;
pub const EG_QUEEN_VALUE: i32 = 936;

pub const ONE_PAWN: i32 = 100;

/// The value of checkmate.
/// To recover depth-to-mate, we subtract depth (ply) from this value.
/// e.g. if white has a mate in two ply, the output from a depth-5 search will be
/// `3_000_000 - 2 = 2_999_998`.
pub const MATE_SCORE: i32 = 3_000_000;

/// A threshold over which scores must be mate.
pub const IS_MATE_SCORE: i32 = MATE_SCORE - 300;

/// The value of a draw.
pub const DRAW_SCORE: i32 = 0;

#[rustfmt::skip]
pub static MG_PIECE_VALUES: [i32; 13] = [
    0,
    MG_PAWN_VALUE, MG_KNIGHT_VALUE, MG_BISHOP_VALUE, MG_ROOK_VALUE, MG_QUEEN_VALUE, 0,
    MG_PAWN_VALUE, MG_KNIGHT_VALUE, MG_BISHOP_VALUE, MG_ROOK_VALUE, MG_QUEEN_VALUE, 0,
];

#[rustfmt::skip]
pub static EG_PIECE_VALUES: [i32; 13] = [
    0,
    EG_PAWN_VALUE, EG_KNIGHT_VALUE, EG_BISHOP_VALUE, EG_ROOK_VALUE, EG_QUEEN_VALUE, 0,
    EG_PAWN_VALUE, EG_KNIGHT_VALUE, EG_BISHOP_VALUE, EG_ROOK_VALUE, EG_QUEEN_VALUE, 0,
];

/// The malus applied when a pawn has no pawns of its own colour to the left or right.
pub const ISOLATED_PAWN_MALUS: i32 = ONE_PAWN / 3;

/// The malus applied when two (or more) pawns of a colour are on the same file.
pub const DOUBLED_PAWN_MALUS: i32 = 2 * ONE_PAWN / 5;

/// The bonus granted for having two bishops.
pub const BISHOP_PAIR_BONUS: i32 = ONE_PAWN / 5;

/// The bonus granted for having more pawns when you have knights on the board.
// pub const KNIGHT_PAWN_BONUS: i32 = PAWN_VALUE / 15;

// The multipliers applied to mobility scores.
pub const KNIGHT_MOBILITY_MULTIPLIER: i32 = 4;
pub const BISHOP_MOBILITY_MULTIPLIER: i32 = 5;
pub const ROOK_MOBILITY_MULTIPLIER: i32 = 2;
pub const QUEEN_MOBILITY_MULTIPLIER: i32 = 1;
pub const KING_MOBILITY_MULTIPLIER: i32 = 1;

const PAWN_DANGER: i32 = 40;
const KNIGHT_DANGER: i32 = 80;
const BISHOP_DANGER: i32 = 30;
const ROOK_DANGER: i32 = 90;
const QUEEN_DANGER: i32 = 190;

#[rustfmt::skip]
pub static PIECE_DANGER_VALUES: [i32; 13] = [
    0,
    PAWN_DANGER, KNIGHT_DANGER, BISHOP_DANGER, ROOK_DANGER, QUEEN_DANGER, 0,
    PAWN_DANGER, KNIGHT_DANGER, BISHOP_DANGER, ROOK_DANGER, QUEEN_DANGER, 0,
];

/// The bonus for having IDX pawns in front of the king.
pub static SHIELD_BONUS: [i32; 4] = [0, 5, 17, 20];

/// A threshold over which we will not bother evaluating more than material and PSTs.
pub const LAZY_THRESHOLD_1: i32 = 14_00;

const PAWN_PHASE: f32 = 0.1;
const KNIGHT_PHASE: f32 = 1.0;
const BISHOP_PHASE: f32 = 1.0;
const ROOK_PHASE: f32 = 2.0;
const QUEEN_PHASE: f32 = 4.0;
const TOTAL_PHASE: f32 = 16.0 * PAWN_PHASE
    + 4.0 * KNIGHT_PHASE
    + 4.0 * BISHOP_PHASE
    + 4.0 * ROOK_PHASE
    + 2.0 * QUEEN_PHASE;

#[allow(dead_code)]
pub static RANK_BB: [u64; 8] = init_eval_masks().0;
pub static FILE_BB: [u64; 8] = init_eval_masks().1;

pub static WHITE_PASSED_BB: [u64; 64] = init_passed_isolated_bb().0;
pub static BLACK_PASSED_BB: [u64; 64] = init_passed_isolated_bb().1;

pub static ISOLATED_BB: [u64; 64] = init_passed_isolated_bb().2;

/// The bonus applied when a pawn has no pawns of the opposite colour ahead of it, or to the left or right, scaled by the rank that the pawn is on.
pub static PASSED_PAWN_BONUS: [i32; 8] = [
    0, // illegal
    30, 40, 50, 70, 110, 250, 0, // illegal
];

/// `game_phase` computes a number between 0.0 and 1.0, which is the phase of the game.
/// 0.0 is the opening, 1.0 is the endgame.
#[allow(clippy::many_single_char_names)]
pub fn game_phase(p: u8, n: u8, b: u8, r: u8, q: u8) -> f32 {
    let mut phase = TOTAL_PHASE;
    phase -= PAWN_PHASE * f32::from(p);
    phase -= KNIGHT_PHASE * f32::from(n);
    phase -= BISHOP_PHASE * f32::from(b);
    phase -= ROOK_PHASE * f32::from(r);
    phase -= QUEEN_PHASE * f32::from(q);
    phase / TOTAL_PHASE
}

/// A struct that holds all the terms in the evaluation function, intended to be used by the
/// tuner for optimising the evaluation function.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvalVector {
    /// Whether this position is valid to use for tuning (positions should be quiescent, amongst other considerations).
    pub valid: bool,
    /// The relative pawn count
    pub pawns: i32,
    /// The relative knight count
    pub knights: i32,
    /// The relative bishop count
    pub bishops: i32,
    /// The relative rook count
    pub rooks: i32,
    /// The relative queen count
    pub queens: i32,
    /// The bishop pair score. (can only be -1, 0 or 1)
    pub bishop_pair: i32,
    /// The relative number of passed pawns by rank.
    pub passed_pawns_by_rank: [i32; 8],
    /// The relative number of isolated pawns.
    pub isolated_pawns: i32,
    /// The relative number of doubled pawns.
    pub doubled_pawns: i32,
    /// The relative pst score, before scaling.
    pub pst: i32,
    /// The relative pawn mobility count.
    pub pawn_mobility: i32,
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
    /// The turn (1 or -1)
    pub turn: i32,
}

#[allow(dead_code)]
impl EvalVector {
    pub const fn new() -> Self {
        Self {
            valid: true,
            pawns: 0,
            knights: 0,
            bishops: 0,
            rooks: 0,
            queens: 0,
            bishop_pair: 0,
            passed_pawns_by_rank: [0; 8],
            isolated_pawns: 0,
            doubled_pawns: 0,
            pst: 0,
            pawn_mobility: 0,
            knight_mobility: 0,
            bishop_mobility: 0,
            rook_mobility: 0,
            queen_mobility: 0,
            king_mobility: 0,
            pawn_shield: 0,
            turn: 0,
        }
    }

    pub fn csvify(&self) -> String {
        let csv = format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            self.pawns,
            self.knights,
            self.bishops,
            self.rooks,
            self.queens,
            self.bishop_pair,
            self.passed_pawns_by_rank[0],
            self.passed_pawns_by_rank[1],
            self.passed_pawns_by_rank[2],
            self.passed_pawns_by_rank[3],
            self.passed_pawns_by_rank[4],
            self.passed_pawns_by_rank[5],
            self.passed_pawns_by_rank[6],
            self.passed_pawns_by_rank[7],
            self.isolated_pawns,
            self.doubled_pawns,
            self.pst,
            self.pawn_mobility,
            self.knight_mobility,
            self.bishop_mobility,
            self.rook_mobility,
            self.queen_mobility,
            self.king_mobility,
            self.pawn_shield,
            self.turn
        );
        assert!(
            csv.chars().filter(|&c| c == ',').count()
                == Self::header().chars().filter(|&c| c == ',').count()
        );
        csv
    }

    pub const fn header() -> &'static str {
        "p,n,b,r,q,bpair,ppr0,ppr1,ppr2,ppr3,ppr4,ppr5,ppr6,ppr7,isolated,doubled,pst,p_mob,n_mob,b_mob,r_mob,q_mob,k_mob,p_shield,turn"
    }
}

pub struct MoveCounter<'a> {
    counters: [i32; 6],
    board: &'a Board,
}

impl<'a> MoveCounter<'a> {
    pub const fn new(board: &'a Board) -> Self {
        Self {
            counters: [0; 6],
            board,
        }
    }

    pub const fn score(&self) -> i32 {
        let knights = self.counters[1] * KNIGHT_MOBILITY_MULTIPLIER;
        let bishops = self.counters[2] * BISHOP_MOBILITY_MULTIPLIER;
        let rooks = self.counters[3] * ROOK_MOBILITY_MULTIPLIER;
        let queens = self.counters[4] * QUEEN_MOBILITY_MULTIPLIER;
        let kings = self.counters[5] * KING_MOBILITY_MULTIPLIER;
        knights + bishops + rooks + queens + kings
    }
}

impl<'a> MoveConsumer for MoveCounter<'a> {
    const DO_PAWN_MOVEGEN: bool = false;

    fn push(&mut self, m: Move, _score: i32) {
        let moved_piece = self.board.moved_piece(m);
        let idx = (moved_piece - 1) % 6;
        unsafe {
            *self.counters.get_unchecked_mut(idx as usize) += 1;
        }
    }
}

impl Board {
    /// Computes a score for the position, from the point of view of the side to move.
    /// This function should strive to be as cheap to call as possible, relying on
    /// incremental updates in make-unmake to avoid recomputation.
    pub fn evaluate(&mut self) -> i32 {
        #![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

        if self.pawns[BOTH as usize] == 0 && self.is_material_draw() {
            return if self.side == WHITE {
                DRAW_SCORE
            } else {
                -DRAW_SCORE
            };
        }

        let game_phase = self.phase();
        let midgame_material = self.mg_material[WHITE as usize] - self.mg_material[BLACK as usize];
        let endgame_material = self.eg_material[WHITE as usize] - self.eg_material[BLACK as usize];
        let material = (midgame_material as f32 * (1.0 - game_phase)) as i32
            + (endgame_material as f32 * game_phase) as i32;
        let pst_val = self.pst_value(game_phase);

        let mut score = material + pst_val;

        // if the score is already outwith a threshold, we can stop here.
        if score.abs() > LAZY_THRESHOLD_1 {
            // score = self.clamp_score(score);
            if self.side == WHITE {
                return score;
            }
            return -score;
        }

        let pawn_val = self.pawn_structure_term(); // INCREMENTAL UPDATE.
        let bishop_pair_val = self.bishop_pair_term();
        // let mobility_val = self.mobility();
        let king_safety_val = self.pawn_shield_term(game_phase);

        score += pawn_val;
        score += bishop_pair_val;
        // score += mobility_val;
        score += king_safety_val;

        // println!(
        //     "material: {} pst: {} pawn: {} bishop pair: {} mobility: {}",
        //     score, pst_val, pawn_val, bishop_pair_val, mobility_val
        // );

        score = self.clamp_score(score);

        if self.side == WHITE {
            score
        } else {
            -score
        }
    }

    #[inline]
    fn clamp_score(&mut self, score: i32) -> i32 {
        // if we can't win with our material, we clamp the eval to be at best one millipawn.
        if score > 0 && self.unwinnable_for::<{ WHITE }>() {
            1
        } else if score < 0 && self.unwinnable_for::<{ BLACK }>() {
            -1
        } else {
            score
        }
    }

    fn pst_value(&self, phase: f32) -> i32 {
        #![allow(
            clippy::cast_possible_truncation,
            clippy::cast_precision_loss,
            clippy::similar_names
        )]
        let mg_val = self.pst_vals[0] as f32;
        let eg_val = self.pst_vals[1] as f32;
        eg_val.mul_add(phase, (1.0 - phase) * mg_val) as i32
    }

    const fn bishop_pair_term(&self) -> i32 {
        let w_count = self.num(WB);
        let b_count = self.num(BB);
        if w_count == b_count {
            return 0;
        }
        if w_count >= 2 {
            return BISHOP_PAIR_BONUS;
        }
        if b_count >= 2 {
            return -BISHOP_PAIR_BONUS;
        }
        0
    }

    fn pawn_shield_term(&self, phase: f32) -> i32 {
        #![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

        let white_kingloc = self.king_sq[WHITE as usize];
        let black_kingloc = self.king_sq[BLACK as usize];

        let mut white_shield = 0;
        for loc in white_kingloc + 9..=white_kingloc + 11 {
            if self.piece_at(loc) == WP {
                white_shield += 1;
            }
        }

        let mut black_shield = 0;
        for loc in black_kingloc - 11..=black_kingloc - 9 {
            if self.piece_at(loc) == BP {
                black_shield += 1;
            }
        }

        (((SHIELD_BONUS[white_shield] - SHIELD_BONUS[black_shield]) as f32) * (1.0 - phase)) as i32
    }

    fn king_tropism_term(&self) -> i32 {
        #![allow(clippy::similar_names)]
        let white_king_square = self.king_sq[WHITE as usize] as usize;
        let (wkr, wkf) = (
            RANKS_BOARD[white_king_square],
            FILES_BOARD[white_king_square],
        );
        let black_king_square = self.king_sq[BLACK as usize] as usize;
        let (bkr, bkf) = (
            RANKS_BOARD[black_king_square],
            FILES_BOARD[black_king_square],
        );
        let mut score = 0;
        for piece_type in BP..=BQ {
            let piece_type = piece_type as usize;
            let danger = PIECE_DANGER_VALUES[piece_type];
            for &sq in self.piece_lists[piece_type].iter() {
                let rank = RANKS_BOARD[sq as usize];
                let file = FILES_BOARD[sq as usize];
                let dist = i32::from(wkr.abs_diff(rank) + wkf.abs_diff(file));
                score -= danger / dist;
            }
        }
        for piece_type in WP..=WQ {
            let piece_type = piece_type as usize;
            let danger = PIECE_DANGER_VALUES[piece_type];
            for &sq in self.piece_lists[piece_type].iter() {
                let rank = RANKS_BOARD[sq as usize];
                let file = FILES_BOARD[sq as usize];
                let dist = i32::from(bkr.abs_diff(rank) + bkf.abs_diff(file));
                score += danger / dist;
            }
        }

        score
    }

    fn pawn_structure_term(&self) -> i32 {
        static DOUBLED_PAWN_MAPPING: [i32; 7] = [0, 0, 1, 2, 3, 4, 5];
        let mut w_score = 0;
        let (white_pawns, black_pawns) = (self.pawns[WHITE as usize], self.pawns[BLACK as usize]);
        for &white_pawn_loc in self.piece_lists[WP as usize].iter() {
            let sq64 = unsafe { *SQ120_TO_SQ64.get_unchecked(white_pawn_loc as usize) as usize };
            if unsafe { *ISOLATED_BB.get_unchecked(sq64) } & white_pawns == 0 {
                w_score -= ISOLATED_PAWN_MALUS;
            }

            if unsafe { *WHITE_PASSED_BB.get_unchecked(sq64) } & black_pawns == 0 {
                let rank = unsafe { *RANKS_BOARD.get_unchecked(white_pawn_loc as usize) } as usize;
                w_score += unsafe { *PASSED_PAWN_BONUS.get_unchecked(rank) };
            }
        }

        let mut b_score = 0;
        for &black_pawn_loc in self.piece_lists[BP as usize].iter() {
            let sq64 = unsafe { *SQ120_TO_SQ64.get_unchecked(black_pawn_loc as usize) } as usize;
            if unsafe { *ISOLATED_BB.get_unchecked(sq64) } & black_pawns == 0 {
                b_score -= ISOLATED_PAWN_MALUS;
            }

            if unsafe { *BLACK_PASSED_BB.get_unchecked(sq64) } & white_pawns == 0 {
                let rank = unsafe { *RANKS_BOARD.get_unchecked(black_pawn_loc as usize) } as usize;
                b_score += unsafe { *PASSED_PAWN_BONUS.get_unchecked(7 - rank) };
            }
        }

        for &file_mask in &FILE_BB {
            let pawns_in_file = (file_mask & white_pawns).count_ones() as usize;
            let multiplier = unsafe { *DOUBLED_PAWN_MAPPING.get_unchecked(pawns_in_file) };
            w_score -= multiplier * DOUBLED_PAWN_MALUS;
            let pawns_in_file = (file_mask & black_pawns).count_ones() as usize;
            let multiplier = unsafe { *DOUBLED_PAWN_MAPPING.get_unchecked(pawns_in_file) };
            b_score -= multiplier * DOUBLED_PAWN_MALUS;
        }

        w_score - b_score
    }

    /// `phase` computes a number between 0.0 and 1.0, which is the phase of the game. 0.0 is the opening, 1.0 is the endgame.
    fn phase(&self) -> f32 {
        let pawns = self.num(WP) + self.num(BP);
        let knights = self.num(WN) + self.num(BN);
        let bishops = self.num(WB) + self.num(BB);
        let rooks = self.num(WR) + self.num(BR);
        let queens = self.num(WQ) + self.num(BQ);
        game_phase(pawns, knights, bishops, rooks, queens)
    }

    pub fn generate_pst_value(&self) -> (i32, i32) {
        #![allow(clippy::needless_range_loop, clippy::similar_names)]
        let mut mg_pst_val = 0;
        let mut eg_pst_val = 0;
        for piece in (WP as usize)..=(WK as usize) {
            for &sq in self.piece_lists[piece].iter() {
                let mg = MIDGAME_PST[piece][sq as usize];
                let eg = ENDGAME_PST[piece][sq as usize];
                // println!("adding {} for {} at {}", mg, PIECE_NAMES[piece], SQUARE_NAMES[SQ120_TO_SQ64[sq as usize] as usize]);
                mg_pst_val += mg;
                eg_pst_val += eg;
            }
        }

        for piece in (BP as usize)..=(BK as usize) {
            for &sq in self.piece_lists[piece].iter() {
                let mg = MIDGAME_PST[piece][sq as usize];
                let eg = ENDGAME_PST[piece][sq as usize];
                // println!("adding {} for {} at {}", mg, PIECE_NAMES[piece], SQUARE_NAMES[SQ120_TO_SQ64[sq as usize] as usize]);
                mg_pst_val += mg;
                eg_pst_val += eg;
            }
        }
        (mg_pst_val, eg_pst_val)
    }

    fn mobility(&mut self) -> i32 {
        #![allow(clippy::cast_possible_truncation)]
        let is_check = self.in_check::<{ Self::US }>();
        if is_check {
            match self.side {
                WHITE => return -50,
                BLACK => return 50,
                _ => unsafe { std::hint::unreachable_unchecked() },
            }
        }

        let mut list = MoveCounter::new(self);
        self.generate_moves(&mut list);

        let our_pseudo_legal_moves = list.score();

        self.make_nullmove();
        let mut list = MoveCounter::new(self);
        self.generate_moves(&mut list);
        let their_pseudo_legal_moves = list.score();
        self.unmake_nullmove();

        if self.side == WHITE {
            our_pseudo_legal_moves as i32 - their_pseudo_legal_moves as i32
        } else {
            their_pseudo_legal_moves as i32 - our_pseudo_legal_moves as i32
        }
    }
}
