use crate::{
    board::evaluation::score::S,
    definitions::{
        flip_rank,
        File::{FILE_A, FILE_D, FILE_H},
        Rank::{RANK_1, RANK_2, RANK_7, RANK_8},
        BLACK, WHITE, WK, WN, WP,
    },
    lookups::{file, filerank_to_square, rank},
};

use super::PieceSquareTable;

// Scores are explicit for files A to D, implicitly mirrored for E to H.
static BONUS: [[[S; 4]; 8]; 7] = [
    [[S::NULL; 4]; 8],
    [[S::NULL; 4]; 8],
    [
        // Knight
        [S(-79, 45), S(3, -11), S(0, -8), S(10, 1)],
        [S(19, 18), S(-6, 23), S(-3, -2), S(17, 11)],
        [S(-9, -4), S(11, 10), S(-9, 24), S(17, 52)],
        [S(16, 32), S(49, 38), S(50, 59), S(27, 79)],
        [S(62, 45), S(54, 35), S(78, 66), S(81, 69)],
        [S(57, 27), S(96, 33), S(126, 51), S(145, 44)],
        [S(49, 23), S(17, 43), S(151, 11), S(190, 13)],
        [S(-95, 25), S(17, 38), S(-10, 49), S(128, 19)],
    ],
    [
        // Bishop
        [S(-3, 47), S(32, 18), S(3, 22), S(-41, 51)],
        [S(62, 21), S(35, 27), S(48, 19), S(9, 37)],
        [S(35, 24), S(41, 32), S(29, 41), S(21, 67)],
        [S(46, 25), S(6, 36), S(9, 64), S(36, 69)],
        [S(1, 71), S(28, 52), S(55, 49), S(62, 62)],
        [S(54, 58), S(89, 39), S(94, 49), S(72, 36)],
        [S(-31, 79), S(-31, 69), S(11, 53), S(10, 49)],
        [S(-43, 102), S(-42, 101), S(-131, 108), S(-78, 59)],
    ],
    [
        // Rook
        [S(5, 65), S(17, 73), S(16, 83), S(26, 75)],
        [S(-22, 53), S(3, 29), S(4, 44), S(-4, 43)],
        [S(-35, 81), S(-2, 66), S(-32, 70), S(-31, 64)],
        [S(-12, 102), S(9, 101), S(-24, 101), S(-7, 81)],
        [S(11, 134), S(20, 121), S(36, 114), S(53, 89)],
        [S(18, 148), S(54, 130), S(50, 128), S(83, 104)],
        [S(61, 143), S(31, 149), S(88, 132), S(104, 127)],
        [S(75, 124), S(73, 138), S(60, 153), S(58, 139)],
    ],
    [
        // Queen
        [S(79, 76), S(113, 24), S(118, -23), S(125, 63)],
        [S(117, 53), S(103, 11), S(119, 27), S(127, 50)],
        [S(103, 81), S(100, 120), S(92, 141), S(79, 148)],
        [S(105, 128), S(103, 152), S(89, 159), S(63, 221)],
        [S(85, 165), S(88, 183), S(71, 203), S(74, 251)],
        [S(101, 163), S(145, 178), S(111, 191), S(97, 205)],
        [S(135, 185), S(72, 211), S(136, 200), S(110, 214)],
        [S(208, 117), S(229, 149), S(227, 174), S(163, 164)],
    ],
    [
        // King
        [S(289, -19), S(303, 34), S(205, 60), S(221, 30)],
        [S(264, 59), S(258, 102), S(172, 138), S(130, 144)],
        [S(144, 101), S(263, 111), S(197, 153), S(184, 168)],
        [S(117, 98), S(269, 132), S(258, 159), S(190, 185)],
        [S(176, 123), S(312, 156), S(228, 188), S(153, 191)],
        [S(148, 147), S(116, 210), S(133, 206), S(180, 181)],
        [S(191, 97), S(163, 178), S(187, 173), S(175, 154)],
        [S(200, -72), S(277, 120), S(204, 137), S(116, 128)],
    ],
];

#[rustfmt::skip]
static P_BONUS: [[S; 8]; 8] = [
    // Pawn (asymmetric distribution)
    [ S::NULL; 8 ],
    [ S(-19, 42),  S(-27, 32), S(-17, 23), S(2, -17),  S(-3, 23),  S(58, 21), S(47, 15), S(-14, 0),    ],
    [ S(-26, 22),  S(-25, 22), S(-16, 14), S(17, -14), S(9, -5),   S(7, 14),  S(45, -4), S(0, 4),      ],
    [ S(-20, 36),  S(-31, 53), S(3, 12),   S(26, -15), S(14, -5),  S(36, 8),  S(-1, 23), S(-6, 11),    ],
    [ S(-11, 82),  S(-19, 64), S(5, 36),   S(28, -9),  S(28, 11),  S(47, 9),  S(-3, 48), S(-10, 38),   ],
    [ S(33, 82),   S(16, 105), S(41, 42),  S(63, -30), S(97, -38), S(159, 9), S(99, 46), S(90, 37),    ],
    [ S(111, 143), S(66, 183), S(15, 188), S(73, 81),  S(46, 116), S(4, 162), S(5, 235), S(-138, 191), ],
    [ S::NULL; 8 ],
];

pub fn printout_pst_source(pst: &PieceSquareTable) {
    #[rustfmt::skip]
    println!(
"static BONUS: [[[S; 4]; 8]; 7] = [
    [[S::NULL; 4]; 8],
    [[S::NULL; 4]; 8],"
    );
    let names = ["NULL", "Pawn", "Knight", "Bishop", "Rook", "Queen", "King"];
    for piece in WN..=WK {
        println!("    [");
        println!("        // {}", names[piece as usize]);
        for rank in RANK_1..=RANK_8 {
            print!("        [");
            for file in FILE_A..=FILE_D {
                let sq = filerank_to_square(file, rank);
                let val = pst[piece as usize][sq as usize];
                print!("{}, ", val);
            }
            println!("],");
        }
        println!("    ],");
    }
    println!("];");
    println!();
    #[rustfmt::skip]
    println!(
"#[rustfmt::skip]
static P_BONUS: [[S; 8]; 8] = [
    // Pawn (asymmetric distribution)
    [ S::NULL; 8 ],"
    );
    for rank in RANK_2..=RANK_7 {
        print!("    [ ");
        for file in FILE_A..=FILE_H {
            let sq = filerank_to_square(file, rank);
            let val = pst[WP as usize][sq as usize];
            print!("{}, ", val);
        }
        println!("],");
    }
    println!("    [ S::NULL; 8 ],");
    println!("];");
}

pub fn construct_piece_square_table() -> PieceSquareTable {
    let mut pst = [[S::NULL; 64]; 13];
    let mut colour = WHITE;
    loop {
        let offset = if colour == BLACK { 7 } else { 1 };
        let multiplier = if colour == BLACK { -1 } else { 1 };
        let mut pieces_idx = 0;
        while pieces_idx < 6 {
            let mut pst_idx = 0;
            while pst_idx < 64 {
                let sq = if colour == WHITE { pst_idx } else { flip_rank(pst_idx) };
                let r = rank(pst_idx) as usize;
                let f = file(pst_idx) as usize;
                let value = if pieces_idx == 0 {
                    P_BONUS[r][f]
                } else {
                    let f = if f >= 4 { 7 - f } else { f };
                    BONUS[pieces_idx + 1][r][f]
                };
                let S(mg, eg) = value;
                pst[pieces_idx + offset][sq as usize] = S(mg * multiplier, eg * multiplier);
                pst_idx += 1;
            }
            pieces_idx += 1;
        }
        if colour == BLACK {
            break;
        }
        colour = BLACK;
    }
    pst
}
