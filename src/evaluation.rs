// The granularity of evaluation in this engine is going to be thousandths of a pawn.

use crate::{
    board::Board,
    chessmove::Move,
    lookups::{init_eval_masks, init_passed_isolated_bb},
    movegen::MoveConsumer, definitions::{PIECE_EMPTY, WK, WQ, WR, WB, WN, WP, BP, BN, BB, BR, BQ, BK},
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
pub const ISOLATED_PAWN_MALUS: i32 = MG_PAWN_VALUE / 3;

/// The malus applied when two (or more) pawns of a colour are on the same file.
pub const DOUBLED_PAWN_MALUS: i32 = 2 * MG_PAWN_VALUE / 5;

/// The bonus granted for having two bishops.
pub const BISHOP_PAIR_BONUS: i32 = MG_PAWN_VALUE / 5;

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
/// A threshold over which we will not bother evaluating more than pawns and mobility.
pub const LAZY_THRESHOLD_2: i32 = 8_00;

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

    pub fn get_mobility_of(&self, piece: u8) -> i32 {
        match piece {
            WP | BP => self.counters[0],
            WN | BN => self.counters[1],
            WB | BB => self.counters[2],
            WR | BR => self.counters[3],
            WQ | BQ => self.counters[4],
            WK | BK => self.counters[5],
            PIECE_EMPTY => panic!("Tried to get mobility of empty piece"),
            _ => panic!("Tried to get mobility of invalid piece"),
        }
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
