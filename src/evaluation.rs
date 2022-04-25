// The granularity of evaluation in this engine is going to be thousandths of a pawn.

pub const PAWN_VALUE: i32 = 1_000;
pub const KNIGHT_VALUE: i32 = 3_250;
pub const BISHOP_VALUE: i32 = 3_250;
pub const ROOK_VALUE: i32 = 5_500;
pub const QUEEN_VALUE: i32 = 10_000;
pub const KING_VALUE: i32 = 500_000;

/// The value of checkmate.
/// To recover depth-to-mate, we add (true) depth to this value.
/// e.g. if white has a mate in two ply, the output from a depth-5 search will be
/// `3_000_000 + (5 - 2) = 3_000_003`
pub const MATE_SCORE: i32 = 3_000_000;

#[rustfmt::skip]
pub static PIECE_VALUES: [i32; 13] = [
    0,
    PAWN_VALUE, KNIGHT_VALUE, BISHOP_VALUE, ROOK_VALUE, QUEEN_VALUE, KING_VALUE,
    PAWN_VALUE, KNIGHT_VALUE, BISHOP_VALUE, ROOK_VALUE, QUEEN_VALUE, KING_VALUE,
];
