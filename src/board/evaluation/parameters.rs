use super::{
    score::S, BISHOP_MOBILITY_BONUS, BISHOP_PAIR_BONUS, DOUBLED_PAWN_MALUS, ISOLATED_PAWN_MALUS,
    KING_DANGER_COEFFS, KING_DANGER_PIECE_WEIGHTS, KNIGHT_MOBILITY_BONUS, MINOR_THREAT_ON_MAJOR,
    PASSED_PAWN_BONUS, PAWN_THREAT_ON_MAJOR, PAWN_THREAT_ON_MINOR, PIECE_VALUES,
    QUEEN_HALF_OPEN_FILE_BONUS, QUEEN_MOBILITY_BONUS, QUEEN_OPEN_FILE_BONUS,
    ROOK_HALF_OPEN_FILE_BONUS, ROOK_MOBILITY_BONUS, ROOK_OPEN_FILE_BONUS, TEMPO_BONUS,
};

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct EvalParams {
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
    pub tempo: S,
    pub pawn_threat_on_minor: S,
    pub pawn_threat_on_major: S,
    pub minor_threat_on_major: S,
    pub king_danger_coeffs: [i32; 3],
    pub king_danger_piece_weights: [i32; 8],
}

impl EvalParams {
    pub const fn default() -> Self {
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
            piece_square_tables: crate::piecesquaretable::tables::construct_piece_square_table(),
            tempo: TEMPO_BONUS,
            pawn_threat_on_minor: PAWN_THREAT_ON_MINOR,
            pawn_threat_on_major: PAWN_THREAT_ON_MAJOR,
            minor_threat_on_major: MINOR_THREAT_ON_MAJOR,
            king_danger_coeffs: KING_DANGER_COEFFS,
            king_danger_piece_weights: KING_DANGER_PIECE_WEIGHTS,
        }
    }
}
