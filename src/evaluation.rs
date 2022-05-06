// The granularity of evaluation in this engine is going to be thousandths of a pawn.

pub const PAWN_VALUE: i32 = 1_000;
pub const KNIGHT_VALUE: i32 = 3_250;
pub const BISHOP_VALUE: i32 = 3_330;
pub const ROOK_VALUE: i32 = 5_500;
pub const QUEEN_VALUE: i32 = 10_000;
pub const KING_VALUE: i32 = 500_000;

/// The value of checkmate.
/// To recover depth-to-mate, we subtract depth (ply) from this value.
/// e.g. if white has a mate in two ply, the output from a depth-5 search will be
/// `3_000_000 - 2 = 2_999_998`.
pub const MATE_SCORE: i32 = 3_000_000;

/// The value of a draw.
pub const DRAW_SCORE: i32 = 0;

#[rustfmt::skip]
pub static PIECE_VALUES: [i32; 13] = [
    0,
    PAWN_VALUE, KNIGHT_VALUE, BISHOP_VALUE, ROOK_VALUE, QUEEN_VALUE, KING_VALUE,
    PAWN_VALUE, KNIGHT_VALUE, BISHOP_VALUE, ROOK_VALUE, QUEEN_VALUE, KING_VALUE,
];

/// The malus applied when a pawn has no pawns of its own colour to the left or right.
pub const ISOLATED_PAWN_MALUS: i32 = PAWN_VALUE / 2;

/// The malus applied when two (or more) pawns of a colour are on the same file.
pub const DOUBLED_PAWN_MALUS: i32 = PAWN_VALUE / 2 + 50;

/// The bonus applied when a pawn has no pawns of the opposite colour ahead of it, or to the left or right.
pub const PASSED_PAWN_BONUS: i32 = PAWN_VALUE / 2 + 100;

/// The bonus granted for having two bishops.
pub const BISHOP_PAIR_BONUS: i32 = PAWN_VALUE / 4;

/// The bonus granted for having more pawns when you have knights on the board.
pub const KNIGHT_PAWN_BONUS: i32 = PAWN_VALUE / 15;

/// The multiplier applied to mobility scores.
pub const MOBILITY_MULTIPLIER: i32 = 15;

const PAWN_DANGER: i32 = 200;
const KNIGHT_DANGER: i32 = 300;
const BISHOP_DANGER: i32 = 100;
const ROOK_DANGER: i32 = 400;
const QUEEN_DANGER: i32 = 500;

#[rustfmt::skip]
pub static PIECE_DANGER_VALUES: [i32; 13] = [
    0,
    PAWN_DANGER, KNIGHT_DANGER, BISHOP_DANGER, ROOK_DANGER, QUEEN_DANGER, 0,
    PAWN_DANGER, KNIGHT_DANGER, BISHOP_DANGER, ROOK_DANGER, QUEEN_DANGER, 0,
];

const PAWN_PHASE: f32 = 0.1;
const KNIGHT_PHASE: f32 = 1.0;
const BISHOP_PHASE: f32 = 1.0;
const ROOK_PHASE: f32 = 2.0;
const QUEEN_PHASE: f32 = 4.0;
const TOTAL_PHASE: f32 = 16.0 * PAWN_PHASE + 4.0 * KNIGHT_PHASE + 4.0 * BISHOP_PHASE + 4.0 * ROOK_PHASE + 2.0 * QUEEN_PHASE;

#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
pub fn game_phase(p: usize, n: usize, b: usize, r: usize, q: usize) -> f32 {
    let mut phase = TOTAL_PHASE;
    phase -= PAWN_PHASE * p as f32;
    phase -= KNIGHT_PHASE * n as f32;
    phase -= BISHOP_PHASE * b as f32;
    phase -= ROOK_PHASE * r as f32;
    phase -= QUEEN_PHASE * q as f32;
    phase / TOTAL_PHASE
}

/// A struct that holds fp values (0.0->1.0) for each piece type, scaled to the game phase.
/// For example, if there is a white pawn on square a1, and the game phase is 0.7, then
/// `mid[WP][0] == 0.7` and `end[WP][0] == 0.3`. This allows a PST tuner to adjust the value of
/// `MIDGAME_PST`[WP][0] and `ENDGAME_PST`[WP][0] to optimise eval.
pub struct PstCounters {
    mid: [[f32; 64]; 13],
    end: [[f32; 64]; 13],
}

/// A struct that holds all the terms in the evaluation function, intended to be used by the
/// tuner for optimising the evaluation function.
pub struct EvalTerms {
    /// The relative material score.
    pub material: i32,
    /// The relative mobility score.
    pub mobility: i32,
    /// The relative king tropism score.
    pub kingsafety: i32,
    /// The bishop pair score. (can only be -1, 0 or 1)
    pub bishop_pair: i32,
    /// The relative number of passed pawns.
    pub passed_pawns: i32,
    /// The relative number of isolated pawns.
    pub isolated_pawns: i32,
    /// The relative number of doubled pawns.
    pub doubled_pawns: i32,
    /// The pst counters.
    pub counters: PstCounters,
}