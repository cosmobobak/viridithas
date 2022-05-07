use crate::definitions::{BLACK, WHITE};

struct S(i32, i32);

// 'Bonus' contains Piece-Square parameters.
// Scores are explicit for files A to D, implicitly mirrored for E to H.
#[rustfmt::skip]
const BONUS: [[[S; 4]; 8]; 6] = [
    [
        // None
        [S(0, 0), S(0, 0), S(0, 0), S(0, 0)],
        [S(0, 0), S(0, 0), S(0, 0), S(0, 0)],
        [S(0, 0), S(0, 0), S(0, 0), S(0, 0)],
        [S(0, 0), S(0, 0), S(0, 0), S(0, 0)],
        [S(0, 0), S(0, 0), S(0, 0), S(0, 0)],
        [S(0, 0), S(0, 0), S(0, 0), S(0, 0)],
        [S(0, 0), S(0, 0), S(0, 0), S(0, 0)],
        [S(0, 0), S(0, 0), S(0, 0), S(0, 0)],
    ],
    [
        // Knight
        [S(-175, -96), S(-92, -65), S(-74, -49), S(-73, -21)],
        [S(-77, -67), S(-41, -54), S(-27, -18), S(-15, 8)],
        [S(-61, -40), S(-17, -27), S(6, -8), S(12, 29)],
        [S(-35, -35), S(8, -2), S(40, 13), S(49, 28)],
        [S(-34, -45), S(13, -16), S(44, 9), S(51, 39)],
        [S(-9, -51), S(22, -44), S(58, -16), S(53, 17)],
        [S(-67, -69), S(-27, -50), S(4, -51), S(37, 12)],
        [S(-201, -100), S(-83, -88), S(-56, -56), S(-26, -17)],
    ],
    [
        // Bishop
        [S(-37, -40), S(-4, -21), S(-6, -26), S(-16, -8)],
        [S(-11, -26), S(6, -9), S(13, -12), S(3, 1)],
        [S(-5, -11), S(15, -1), S(-4, -1), S(12, 7)],
        [S(-4, -14), S(8, -4), S(18, 0), S(27, 12)],
        [S(-8, -12), S(20, -1), S(15, -10), S(22, 11)],
        [S(-11, -21), S(4, 4), S(1, 3), S(8, 4)],
        [S(-12, -22), S(-10, -14), S(4, -1), S(0, 1)],
        [S(-34, -32), S(1, -29), S(-10, -26), S(-16, -17)],
    ],
    [
        // Rook
        [S(-31, -9), S(-20, -13), S(-14, -10), S(-5, -9)],
        [S(-21, -12), S(-13, -9), S(-8, -1), S(6, -2)],
        [S(-25, 6), S(-11, -8), S(-1, -2), S(3, -6)],
        [S(-13, -6), S(-5, 1), S(-4, -9), S(-6, 7)],
        [S(-27, -5), S(-15, 8), S(-4, 7), S(3, -6)],
        [S(-22, 6), S(-2, 1), S(6, -7), S(12, 10)],
        [S(-2, 4), S(12, 5), S(16, 20), S(18, -5)],
        [S(-17, 18), S(-19, 0), S(-1, 19), S(9, 13)],
    ],
    [
        // Queen
        [S(3, -69), S(-5, -57), S(-5, -47), S(4, -26)],
        [S(-3, -54), S(5, -31), S(8, -22), S(12, -4)],
        [S(-3, -39), S(6, -18), S(13, -9), S(7, 3)],
        [S(4, -23), S(5, -3), S(9, 13), S(8, 24)],
        [S(0, -29), S(14, -6), S(12, 9), S(5, 21)],
        [S(-4, -38), S(10, -18), S(6, -11), S(8, 1)],
        [S(-5, -50), S(6, -27), S(10, -24), S(8, -8)],
        [S(-2, -74), S(-2, -52), S(1, -43), S(-2, -34)],
    ],
    [
        // King
        [S(271, 1), S(327, 45), S(271, 85), S(198, 76)],
        [S(278, 53), S(303, 100), S(234, 133), S(179, 135)],
        [S(195, 88), S(258, 130), S(169, 169), S(120, 175)],
        [S(164, 103), S(190, 156), S(138, 172), S(98, 172)],
        [S(154, 96), S(179, 166), S(105, 199), S(70, 199)],
        [S(123, 92), S(145, 172), S(81, 184), S(31, 191)],
        [S(88, 47), S(120, 121), S(65, 116), S(33, 131)],
        [S(59, 11), S(89, 59), S(45, 73), S(-1, 78)],
    ],
];

#[rustfmt::skip]
const P_BONUS: [[S; 8]; 8] = [
    // Pawn (asymmetric distribution)
    [ S(0, 0),   S(0, 0),    S(0, 0),    S(0, 0),    S(0, 0),    S(0, 0),    S(0, 0),    S(0, 0) ],
    [ S(2, -8),  S(4, -6),   S(11, 9),   S(18, 5),   S(16, 16),  S(21, 6),   S(9, -6),   S(-3, -18) ],
    [ S(-9, -9), S(-15, -7), S(11, -10), S(15, 5),   S(31, 2),   S(23, 3),   S(6, -8),   S(-20, -5) ],
    [ S(-3, 7),  S(-20, 1),  S(8, -8),   S(19, -2),  S(39, -14), S(17, -13), S(2, -11),  S(-5, -6) ],
    [ S(11, 12), S(-4, 6),   S(-11, 2),  S(2, -6),   S(11, -5),  S(0, -4),   S(-12, 14), S(5, 9) ],
    [ S(3, 27),  S(-11, 18), S(-6, 19),  S(22, 29),  S(-8, 30),  S(-5, 9),   S(-14, 8),  S(-11, 14) ],
    [ S(-7, -1), S(6, -14),  S(-2, 13),  S(-11, 22), S(4, 24),   S(-14, 17), S(10, 7),   S(-9, 7) ],
    [ S(0, 0),   S(0, 0),    S(0, 0),    S(0, 0),    S(0, 0),    S(0, 0),    S(0, 0),    S(0, 0) ],
];

const MIDGAME: bool = false;
const ENDGAME: bool = true;

const SQ120: [u8; 64] = crate::lookups::init_sq120_to_sq64().1;
const INDEX_MAPPING: [usize; 8] = [0, 1, 2, 3, 3, 2, 1, 0];

const fn generate_pst<const MID_OR_END: bool>() -> [[i32; 120]; 13] {
    let mut out = [[0; 120]; 13];
    let mut colour = WHITE;
    loop {
        let offset = if colour == BLACK { 7 } else { 1 };
        let multiplier = if colour == BLACK { -1 } else { 1 };
        let mut pieces_idx = 0;
        while pieces_idx < BONUS.len() {
            let mut square64 = 0;
            while square64 < 64 {
                let rank = if colour == WHITE {
                    square64 / 8
                } else {
                    7 - (square64 / 8)
                };
                let file = square64 % 8;
                if pieces_idx == 0 {
                    if rank == 0 || rank == 7 {
                        square64 += 1;
                        continue;
                    }
                    let square120 = SQ120[square64];
                    if MID_OR_END == MIDGAME {
                        out[pieces_idx + offset][square120 as usize] =
                            multiplier * P_BONUS[rank][file].0 * 10;
                    } else {
                        out[pieces_idx + offset][square120 as usize] =
                            multiplier * P_BONUS[rank][file].1 * 10;
                    }
                } else {
                    let access_file = INDEX_MAPPING[file];
                    let square120 = SQ120[square64];
                    if MID_OR_END == MIDGAME {
                        out[pieces_idx + offset][square120 as usize] =
                            multiplier * BONUS[pieces_idx][rank][access_file].0 * 10;
                    } else {
                        out[pieces_idx + offset][square120 as usize] =
                            multiplier * BONUS[pieces_idx][rank][access_file].1 * 10;
                    }
                }
                square64 += 1;
            }
            pieces_idx += 1;
        }
        if colour == BLACK {
            break;
        }
        colour = BLACK;
    }
    out
}

pub static MIDGAME_PST: [[i32; 120]; 13] = generate_pst::<MIDGAME>();
pub static ENDGAME_PST: [[i32; 120]; 13] = generate_pst::<ENDGAME>();
