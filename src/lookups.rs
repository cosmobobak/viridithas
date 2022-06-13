#![allow(clippy::cast_possible_truncation)]

use crate::{
    definitions::{square_distance, BK, FILE_A, FILE_H, KING, KNIGHT, RANK_1, RANK_8, WP},
    rng::XorShiftState,
};

macro_rules! cfor {
    ($init: stmt; $cond: expr; $step: expr; $body: block) => {
        {
            $init
            while $cond {
                $body;

                $step;
            }
        }
    }
}

pub const fn filerank_to_square(f: u8, r: u8) -> u8 {
    f + r * 8
}

const fn init_hash_keys() -> ([[u64; 64]; 13], [u64; 16], u64) {
    let mut state = XorShiftState::new();
    let mut piece_keys = [[0; 64]; 13];
    cfor!(let mut index = 0; index < 13; index += 1; {
        cfor!(let mut sq = 0; sq < 64; sq += 1; {
            let key;
            (key, state) = state.next_self();
            piece_keys[index][sq] = key;
        });
    });
    let mut castle_keys = [0; 16];
    cfor!(let mut index = 0; index < 16; index += 1; {
        let key;
        (key, state) = state.next_self();
        castle_keys[index] = key;
    });
    let key;
    (key, _) = state.next_self();
    let side_key = key;
    (piece_keys, castle_keys, side_key)
}

pub const fn init_eval_masks() -> ([u64; 8], [u64; 8]) {
    let mut rank_masks = [0; 8];
    let mut file_masks = [0; 8];

    let mut r = RANK_8;
    loop {
        let mut f = FILE_A;
        while f <= FILE_H {
            let sq = r * 8 + f;
            file_masks[f as usize] |= 1 << sq;
            rank_masks[r as usize] |= 1 << sq;
            f += 1;
        }
        if r == RANK_1 {
            break;
        }
        r -= 1;
    }

    (rank_masks, file_masks)
}

pub const fn init_passed_isolated_bb() -> ([u64; 64], [u64; 64], [u64; 64]) {
    #![allow(clippy::cast_possible_wrap)]
    const _FILE_BB: [u64; 8] = init_eval_masks().1;
    let mut white_passed_bb = [0; 64];
    let mut black_passed_bb = [0; 64];
    let mut isolated_bb = [0; 64];

    let mut sq = 0;
    while sq < 64 {
        let mut t_sq = sq as isize + 8;
        while t_sq < 64 {
            white_passed_bb[sq] |= 1 << t_sq;
            t_sq += 8;
        }

        t_sq = sq as isize - 8;
        while t_sq >= 0 {
            black_passed_bb[sq] |= 1 << t_sq;
            t_sq -= 8;
        }

        if file(sq as u8) > FILE_A {
            isolated_bb[sq] |= _FILE_BB[file(sq as u8) as usize - 1];

            t_sq = sq as isize + 7;
            while t_sq < 64 {
                white_passed_bb[sq] |= 1 << t_sq;
                t_sq += 8;
            }

            t_sq = sq as isize - 9;
            while t_sq >= 0 {
                black_passed_bb[sq] |= 1 << t_sq;
                t_sq -= 8;
            }
        }

        if file(sq as u8) < FILE_H {
            isolated_bb[sq] |= _FILE_BB[file(sq as u8) as usize + 1];

            t_sq = sq as isize + 9;
            while t_sq < 64 {
                white_passed_bb[sq] |= 1 << t_sq;
                t_sq += 8;
            }

            t_sq = sq as isize - 7;
            while t_sq >= 0 {
                black_passed_bb[sq] |= 1 << t_sq;
                t_sq -= 8;
            }
        }

        sq += 1;
    }

    (white_passed_bb, black_passed_bb, isolated_bb)
}

pub static PIECE_KEYS: [[u64; 64]; 13] = init_hash_keys().0;
pub static CASTLE_KEYS: [u64; 16] = init_hash_keys().1;
pub const SIDE_KEY: u64 = init_hash_keys().2;

/// knights, bishops, rooks, and queens.
pub static PIECE_BIG: [bool; 13] = [
    false, false, true, true, true, true, false, false, true, true, true, true, false,
];
/// rooks and queens.
pub static PIECE_MAJ: [bool; 13] = [
    false, false, false, false, true, true, false, false, false, false, true, true, false,
];
/// knights and bishops.
#[allow(dead_code)]
pub static PIECE_MIN: [bool; 13] = [
    false, false, true, true, false, false, false, false, true, true, false, false, false,
];

/// The file that this square is on.
pub const fn file(sq: u8) -> u8 {
    sq % 8
}
/// The rank that this square is on.
pub const fn rank(sq: u8) -> u8 {
    sq / 8
}

/// The name of this 64-indexed square.
pub static SQUARE_NAMES: [&str; 64] = [
    "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1", "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
    "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3", "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
    "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5", "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
    "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7", "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
];

/// The name of this piece.
#[allow(dead_code)]
static PIECE_NAMES: [&str; 13] = [
    "NO_PIECE", "pawn", "knight", "bishop", "rook", "queen", "king", "pawn", "knight", "bishop",
    "rook", "queen", "king",
];

#[allow(dead_code)]
pub fn piece_name(piece: u8) -> Option<&'static str> {
    PIECE_NAMES.get(piece as usize).copied()
}

static PIECE_CHARS: [u8; 13] = *b".PNBRQKpnbrqk";
pub static PROMO_CHAR_LOOKUP: [u8; 13] = *b"XXnbrqXXnbrqX";

pub fn piece_char(piece: u8) -> Option<char> {
    PIECE_CHARS.get(piece as usize).map(|&c| c as char)
}

/// The score of this piece, for MVV/LVA move ordering.
const VICTIM_SCORE: [i32; 13] = [
    0, 1000, 2000, 3000, 4000, 5000, 6000, 1000, 2000, 3000, 4000, 5000, 6000,
];

const fn mvvlva_init() -> [[i32; 13]; 13] {
    let mut mvvlva = [[0; 13]; 13];
    let mut attacker = WP as usize;
    while attacker <= BK as usize {
        let mut victim = WP as usize;
        while victim <= BK as usize {
            mvvlva[victim][attacker] = VICTIM_SCORE[victim] + 60 - VICTIM_SCORE[attacker] / 100;
            victim += 1;
        }
        attacker += 1;
    }
    mvvlva
}

/// The score of this pair of pieces, for MVV/LVA move ordering.
pub static MVV_LVA_SCORE: [[i32; 13]; 13] = mvvlva_init();

const fn init_jumping_attacks<const IS_KNIGHT: bool>() -> [u64; 64] {
    let mut attacks = [0; 64];
    let deltas = if IS_KNIGHT {
        &[17, 15, 10, 6, -17, -15, -10, -6]
    } else {
        &[9, 8, 7, 1, -9, -8, -7, -1]
    };
    cfor!(let mut sq = 0; sq < 64; sq += 1; {
        let mut attacks_bb = 0;
        cfor!(let mut idx = 0; idx < 8; idx += 1; {
            let delta = deltas[idx];
            #[allow(clippy::cast_possible_wrap)]
            let attacked_sq = sq as i32 + delta;
            #[allow(clippy::cast_sign_loss)]
            if attacked_sq >= 0 && attacked_sq < 64 && square_distance(sq as u8, attacked_sq as u8) <= 2 {
                attacks_bb |= 1 << attacked_sq;
            }
        });
        attacks[sq] = attacks_bb;
    });
    attacks
}

static JUMPING_ATTACKS: [[u64; 64]; 7] = [
    [0; 64],                         // no_piece
    [0; 64],                         // pawn
    init_jumping_attacks::<true>(),  // knight
    [0; 64],                         // bishop
    [0; 64],                         // rook
    [0; 64],                         // queen
    init_jumping_attacks::<false>(), // king
];

pub fn get_jumping_piece_attack<const PIECE: u8>(sq: u8) -> u64 {
    debug_assert!(PIECE < 7);
    debug_assert!(sq < 64);
    debug_assert!(PIECE == KNIGHT || PIECE == KING);
    unsafe {
        *JUMPING_ATTACKS
            .get_unchecked(PIECE as usize)
            .get_unchecked(sq as usize)
    }
}

#[allow(dead_code)]
pub const SQRT: [f32; 512] = [
    0.0, 1.0, 1.414, 1.732, 2.0, 2.236, 2.449, 2.646, 2.828, 3.0, 3.162, 3.317, 3.464, 3.606,
    3.742, 3.873, 4.0, 4.123, 4.243, 4.359, 4.472, 4.583, 4.69, 4.796, 4.899, 5.0, 5.099, 5.196,
    5.292, 5.385, 5.477, 5.568, 5.657, 5.745, 5.831, 5.916, 6.0, 6.083, 6.164, 6.245, 6.325, 6.403,
    6.481, 6.557, 6.633, 6.708, 6.782, 6.856, 6.928, 7.0, 7.071, 7.141, 7.211, 7.28, 7.348, 7.416,
    7.483, 7.55, 7.616, 7.681, 7.746, 7.81, 7.874, 7.937, 8.0, 8.062, 8.124, 8.185, 8.246, 8.307,
    8.367, 8.426, 8.485, 8.544, 8.602, 8.66, 8.718, 8.775, 8.832, 8.888, 8.944, 9.0, 9.055, 9.11,
    9.165, 9.22, 9.274, 9.327, 9.381, 9.434, 9.487, 9.539, 9.592, 9.644, 9.695, 9.747, 9.798,
    9.849, 9.899, 9.95, 10.0, 10.05, 10.1, 10.149, 10.198, 10.247, 10.296, 10.344, 10.392, 10.44,
    10.488, 10.536, 10.583, 10.63, 10.677, 10.724, 10.77, 10.817, 10.863, 10.909, 10.954, 11.0,
    11.045, 11.091, 11.136, 11.18, 11.225, 11.269, 11.314, 11.358, 11.402, 11.446, 11.489, 11.533,
    11.576, 11.619, 11.662, 11.705, 11.747, 11.79, 11.832, 11.874, 11.916, 11.958, 12.0, 12.042,
    12.083, 12.124, 12.166, 12.207, 12.247, 12.288, 12.329, 12.369, 12.41, 12.45, 12.49, 12.53,
    12.57, 12.61, 12.649, 12.689, 12.728, 12.767, 12.806, 12.845, 12.884, 12.923, 12.961, 13.0,
    13.038, 13.077, 13.115, 13.153, 13.191, 13.229, 13.266, 13.304, 13.342, 13.379, 13.416, 13.454,
    13.491, 13.528, 13.565, 13.601, 13.638, 13.675, 13.711, 13.748, 13.784, 13.82, 13.856, 13.892,
    13.928, 13.964, 14.0, 14.036, 14.071, 14.107, 14.142, 14.177, 14.213, 14.248, 14.283, 14.318,
    14.353, 14.387, 14.422, 14.457, 14.491, 14.526, 14.56, 14.595, 14.629, 14.663, 14.697, 14.731,
    14.765, 14.799, 14.832, 14.866, 14.9, 14.933, 14.967, 15.0, 15.033, 15.067, 15.1, 15.133,
    15.166, 15.199, 15.232, 15.264, 15.297, 15.33, 15.362, 15.395, 15.427, 15.46, 15.492, 15.524,
    15.556, 15.588, 15.62, 15.652, 15.684, 15.716, 15.748, 15.78, 15.811, 15.843, 15.875, 15.906,
    15.937, 15.969, 16.0, 16.031, 16.062, 16.093, 16.125, 16.155, 16.186, 16.217, 16.248, 16.279,
    16.31, 16.34, 16.371, 16.401, 16.432, 16.462, 16.492, 16.523, 16.553, 16.583, 16.613, 16.643,
    16.673, 16.703, 16.733, 16.763, 16.793, 16.823, 16.852, 16.882, 16.912, 16.941, 16.971, 17.0,
    17.029, 17.059, 17.088, 17.117, 17.146, 17.176, 17.205, 17.234, 17.263, 17.292, 17.321, 17.349,
    17.378, 17.407, 17.436, 17.464, 17.493, 17.521, 17.55, 17.578, 17.607, 17.635, 17.664, 17.692,
    17.72, 17.748, 17.776, 17.804, 17.833, 17.861, 17.889, 17.916, 17.944, 17.972, 18.0, 18.028,
    18.055, 18.083, 18.111, 18.138, 18.166, 18.193, 18.221, 18.248, 18.276, 18.303, 18.33, 18.358,
    18.385, 18.412, 18.439, 18.466, 18.493, 18.52, 18.547, 18.574, 18.601, 18.628, 18.655, 18.682,
    18.708, 18.735, 18.762, 18.788, 18.815, 18.841, 18.868, 18.894, 18.921, 18.947, 18.974, 19.0,
    19.026, 19.053, 19.079, 19.105, 19.131, 19.157, 19.183, 19.209, 19.235, 19.261, 19.287, 19.313,
    19.339, 19.365, 19.391, 19.416, 19.442, 19.468, 19.494, 19.519, 19.545, 19.57, 19.596, 19.621,
    19.647, 19.672, 19.698, 19.723, 19.748, 19.774, 19.799, 19.824, 19.849, 19.875, 19.9, 19.925,
    19.95, 19.975, 20.0, 20.025, 20.05, 20.075, 20.1, 20.125, 20.149, 20.174, 20.199, 20.224,
    20.248, 20.273, 20.298, 20.322, 20.347, 20.372, 20.396, 20.421, 20.445, 20.469, 20.494, 20.518,
    20.543, 20.567, 20.591, 20.616, 20.64, 20.664, 20.688, 20.712, 20.736, 20.761, 20.785, 20.809,
    20.833, 20.857, 20.881, 20.905, 20.928, 20.952, 20.976, 21.0, 21.024, 21.048, 21.071, 21.095,
    21.119, 21.142, 21.166, 21.19, 21.213, 21.237, 21.26, 21.284, 21.307, 21.331, 21.354, 21.378,
    21.401, 21.424, 21.448, 21.471, 21.494, 21.517, 21.541, 21.564, 21.587, 21.61, 21.633, 21.656,
    21.679, 21.703, 21.726, 21.749, 21.772, 21.794, 21.817, 21.84, 21.863, 21.886, 21.909, 21.932,
    21.954, 21.977, 22.0, 22.023, 22.045, 22.068, 22.091, 22.113, 22.136, 22.159, 22.181, 22.204,
    22.226, 22.249, 22.271, 22.293, 22.316, 22.338, 22.361, 22.383, 22.405, 22.428, 22.45, 22.472,
    22.494, 22.517, 22.539, 22.561, 22.583, 22.605,
];

#[allow(clippy::approx_constant)]
pub const LOG: [f32; 512] = [
    0.0, 0.0, 0.69, 1.1, 1.39, 1.61, 1.79, 1.95, 2.08, 2.2, 2.3, 2.4, 2.48, 2.56, 2.64, 2.71, 2.77,
    2.83, 2.89, 2.94, 3.0, 3.04, 3.09, 3.14, 3.18, 3.22, 3.26, 3.3, 3.33, 3.37, 3.4, 3.43, 3.47,
    3.5, 3.53, 3.56, 3.58, 3.61, 3.64, 3.66, 3.69, 3.71, 3.74, 3.76, 3.78, 3.81, 3.83, 3.85, 3.87,
    3.89, 3.91, 3.93, 3.95, 3.97, 3.99, 4.01, 4.03, 4.04, 4.06, 4.08, 4.09, 4.11, 4.13, 4.14, 4.16,
    4.17, 4.19, 4.2, 4.22, 4.23, 4.25, 4.26, 4.28, 4.29, 4.3, 4.32, 4.33, 4.34, 4.36, 4.37, 4.38,
    4.39, 4.41, 4.42, 4.43, 4.44, 4.45, 4.47, 4.48, 4.49, 4.5, 4.51, 4.52, 4.53, 4.54, 4.55, 4.56,
    4.57, 4.58, 4.6, 4.61, 4.62, 4.62, 4.63, 4.64, 4.65, 4.66, 4.67, 4.68, 4.69, 4.7, 4.71, 4.72,
    4.73, 4.74, 4.74, 4.75, 4.76, 4.77, 4.78, 4.79, 4.8, 4.8, 4.81, 4.82, 4.83, 4.84, 4.84, 4.85,
    4.86, 4.87, 4.88, 4.88, 4.89, 4.9, 4.91, 4.91, 4.92, 4.93, 4.93, 4.94, 4.95, 4.96, 4.96, 4.97,
    4.98, 4.98, 4.99, 5.0, 5.0, 5.01, 5.02, 5.02, 5.03, 5.04, 5.04, 5.05, 5.06, 5.06, 5.07, 5.08,
    5.08, 5.09, 5.09, 5.1, 5.11, 5.11, 5.12, 5.12, 5.13, 5.14, 5.14, 5.15, 5.15, 5.16, 5.16, 5.17,
    5.18, 5.18, 5.19, 5.19, 5.2, 5.2, 5.21, 5.21, 5.22, 5.23, 5.23, 5.24, 5.24, 5.25, 5.25, 5.26,
    5.26, 5.27, 5.27, 5.28, 5.28, 5.29, 5.29, 5.3, 5.3, 5.31, 5.31, 5.32, 5.32, 5.33, 5.33, 5.34,
    5.34, 5.35, 5.35, 5.36, 5.36, 5.37, 5.37, 5.38, 5.38, 5.38, 5.39, 5.39, 5.4, 5.4, 5.41, 5.41,
    5.42, 5.42, 5.42, 5.43, 5.43, 5.44, 5.44, 5.45, 5.45, 5.46, 5.46, 5.46, 5.47, 5.47, 5.48, 5.48,
    5.48, 5.49, 5.49, 5.5, 5.5, 5.51, 5.51, 5.51, 5.52, 5.52, 5.53, 5.53, 5.53, 5.54, 5.54, 5.55,
    5.55, 5.55, 5.56, 5.56, 5.56, 5.57, 5.57, 5.58, 5.58, 5.58, 5.59, 5.59, 5.59, 5.6, 5.6, 5.61,
    5.61, 5.61, 5.62, 5.62, 5.62, 5.63, 5.63, 5.63, 5.64, 5.64, 5.65, 5.65, 5.65, 5.66, 5.66, 5.66,
    5.67, 5.67, 5.67, 5.68, 5.68, 5.68, 5.69, 5.69, 5.69, 5.7, 5.7, 5.7, 5.71, 5.71, 5.71, 5.72,
    5.72, 5.72, 5.73, 5.73, 5.73, 5.74, 5.74, 5.74, 5.75, 5.75, 5.75, 5.76, 5.76, 5.76, 5.77, 5.77,
    5.77, 5.77, 5.78, 5.78, 5.78, 5.79, 5.79, 5.79, 5.8, 5.8, 5.8, 5.81, 5.81, 5.81, 5.81, 5.82,
    5.82, 5.82, 5.83, 5.83, 5.83, 5.83, 5.84, 5.84, 5.84, 5.85, 5.85, 5.85, 5.86, 5.86, 5.86, 5.86,
    5.87, 5.87, 5.87, 5.87, 5.88, 5.88, 5.88, 5.89, 5.89, 5.89, 5.89, 5.9, 5.9, 5.9, 5.91, 5.91,
    5.91, 5.91, 5.92, 5.92, 5.92, 5.92, 5.93, 5.93, 5.93, 5.93, 5.94, 5.94, 5.94, 5.95, 5.95, 5.95,
    5.95, 5.96, 5.96, 5.96, 5.96, 5.97, 5.97, 5.97, 5.97, 5.98, 5.98, 5.98, 5.98, 5.99, 5.99, 5.99,
    5.99, 6.0, 6.0, 6.0, 6.0, 6.01, 6.01, 6.01, 6.01, 6.02, 6.02, 6.02, 6.02, 6.03, 6.03, 6.03,
    6.03, 6.04, 6.04, 6.04, 6.04, 6.05, 6.05, 6.05, 6.05, 6.05, 6.06, 6.06, 6.06, 6.06, 6.07, 6.07,
    6.07, 6.07, 6.08, 6.08, 6.08, 6.08, 6.08, 6.09, 6.09, 6.09, 6.09, 6.1, 6.1, 6.1, 6.1, 6.1,
    6.11, 6.11, 6.11, 6.11, 6.12, 6.12, 6.12, 6.12, 6.12, 6.13, 6.13, 6.13, 6.13, 6.14, 6.14, 6.14,
    6.14, 6.14, 6.15, 6.15, 6.15, 6.15, 6.15, 6.16, 6.16, 6.16, 6.16, 6.17, 6.17, 6.17, 6.17, 6.17,
    6.18, 6.18, 6.18, 6.18, 6.18, 6.19, 6.19, 6.19, 6.19, 6.19, 6.2, 6.2, 6.2, 6.2, 6.2, 6.21,
    6.21, 6.21, 6.21, 6.21, 6.22, 6.22, 6.22, 6.22, 6.22, 6.23, 6.23, 6.23, 6.23, 6.23, 6.24,
];

mod tests {
    #[test]
    fn all_piece_keys_different() {
        use crate::lookups::PIECE_KEYS;
        let mut hashkeys = PIECE_KEYS.iter().flat_map(|&k| k).collect::<Vec<u64>>();
        hashkeys.sort_unstable();
        let len_before = hashkeys.len();
        hashkeys.dedup();
        let len_after = hashkeys.len();
        assert_eq!(len_before, len_after);
    }

    #[test]
    fn all_castle_keys_different() {
        use crate::lookups::CASTLE_KEYS;
        let mut hashkeys = CASTLE_KEYS.to_vec();
        hashkeys.sort_unstable();
        let len_before = hashkeys.len();
        hashkeys.dedup();
        let len_after = hashkeys.len();
        assert_eq!(len_before, len_after);
    }

    #[test]
    fn python_chess_validation() {
        use crate::definitions::{KING, KNIGHT};
        use crate::lookups::get_jumping_piece_attack;
        // testing that the attack bitboards match the ones in the python-chess library,
        // which are known to be correct.
        assert_eq!(get_jumping_piece_attack::<KNIGHT>(0), 132_096);
        assert_eq!(
            get_jumping_piece_attack::<KNIGHT>(63),
            9_077_567_998_918_656
        );

        assert_eq!(get_jumping_piece_attack::<KING>(0), 770);
        assert_eq!(
            get_jumping_piece_attack::<KING>(63),
            4_665_729_213_955_833_856
        );
    }
}
