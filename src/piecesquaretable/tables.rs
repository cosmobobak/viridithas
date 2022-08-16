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
const BONUS: [[[S; 4]; 8]; 7] = [
    [[S::NULL; 4]; 8],
    [[S::NULL; 4]; 8],
    [
        // Knight
        [S(-113, -34), S(-30, -36), S(-42, -33), S(-54, 14)],
        [S(-37, -5), S(-71, 2), S(-40, -18), S(-20, -23)],
        [S(-46, -19), S(-20, -21), S(-36, -12), S(-7, 25)],
        [S(-10, 0), S(9, 20), S(4, 41), S(-12, 53)],
        [S(28, 17), S(19, 11), S(35, 43), S(27, 49)],
        [S(53, 7), S(36, 10), S(77, 36), S(95, 9)],
        [S(-5, -7), S(22, 11), S(66, 3), S(87, 24)],
        [S(-139, -38), S(-21, -26), S(2, 6), S(35, 34)],
    ],
    [
        // Bishop
        [S(-12, -3), S(38, -8), S(-15, 5), S(-36, 5)],
        [S(35, -29), S(1, -14), S(5, -11), S(-31, 1)],
        [S(-18, -14), S(4, -9), S(-29, 5), S(-14, 21)],
        [S(18, -28), S(-8, -20), S(-22, 11), S(7, 19)],
        [S(-24, 25), S(-5, 3), S(30, -9), S(24, 15)],
        [S(31, 31), S(65, -11), S(57, -3), S(45, -21)],
        [S(22, 12), S(-39, 32), S(10, -2), S(-8, 25)],
        [S(-50, 29), S(-31, 15), S(-51, 32), S(-62, 18)],
    ],
    [
        // Rook
        [S(-27, -6), S(-12, -17), S(-12, -1), S(-12, -11)],
        [S(-57, -2), S(-10, -28), S(-26, -28), S(-22, -24)],
        [S(-58, 25), S(-13, -17), S(-36, -8), S(-41, -24)],
        [S(-37, 47), S(-16, 5), S(-21, 4), S(-28, 12)],
        [S(21, 31), S(25, 14), S(-4, 29), S(14, 27)],
        [S(13, 39), S(26, 23), S(18, 32), S(15, 32)],
        [S(-10, 56), S(-2, 49), S(-3, 58), S(43, 45)],
        [S(21, 42), S(10, 32), S(37, 54), S(37, 36)],
    ],
    [
        // Queen
        [S(-10, -47), S(-13, -57), S(-13, -106), S(-4, -31)],
        [S(22, -92), S(1, -82), S(-10, -49), S(-6, -46)],
        [S(6, -87), S(3, -33), S(-28, 8), S(-29, 2)],
        [S(8, -48), S(2, -13), S(-10, -9), S(-39, 60)],
        [S(-3, -7), S(-11, 7), S(-17, 30), S(-26, 77)],
        [S(17, -13), S(28, 8), S(29, 22), S(7, 30)],
        [S(57, 9), S(-17, 35), S(10, 29), S(1, 51)],
        [S(59, -59), S(60, -27), S(62, 5), S(60, 15)],
    ],
    [
        // King
        [S(276, -20), S(275, 49), S(245, 45), S(253, 42)],
        [S(248, 63), S(251, 87), S(185, 126), S(167, 124)],
        [S(142, 81), S(198, 104), S(172, 140), S(168, 155)],
        [S(109, 115), S(159, 144), S(124, 178), S(154, 194)],
        [S(158, 121), S(173, 166), S(162, 203), S(128, 210)],
        [S(179, 125), S(206, 210), S(141, 222), S(93, 225)],
        [S(139, 109), S(180, 183), S(127, 178), S(95, 193)],
        [S(121, 43), S(136, 103), S(107, 133), S(61, 140)],
    ],
];

#[rustfmt::skip]
const P_BONUS: [[S; 8]; 8] = [
    // Pawn (asymmetric distribution)
    [ S::NULL; 8 ],
    [ S(-28, 24), S(-24, 26), S(-8, 32), S(14, 48), S(17, 26), S(43, 25), S(35, 3), S(0, -6), ],
    [ S(-42, 12), S(-21, 19), S(-23, 18), S(1, 15), S(15, 20), S(10, 32), S(28, -4), S(-7, -3), ],
    [ S(-36, 27), S(-33, 32), S(-5, 13), S(2, -3), S(4, -3), S(21, 16), S(16, 15), S(3, 0), ],
    [ S(-14, 65), S(-19, 65), S(-7, 29), S(8, -23), S(25, 5), S(35, 19), S(33, 33), S(26, 18), ],
    [ S(7, 88), S(-12, 80), S(30, 49), S(34, 1), S(52, 35), S(57, 57), S(48, 54), S(51, 66), ],
    [ S(55, 61), S(67, 48), S(41, 75), S(48, 76), S(6, 85), S(46, 79), S(50, 69), S(-38, 63), ],
    [ S::NULL; 8 ],
];

pub fn printout_pst_source(pst: &PieceSquareTable) {
    #[rustfmt::skip]
    println!(
"const BONUS: [[[S; 4]; 8]; 7] = [
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
const P_BONUS: [[S; 8]; 8] = [
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

pub const fn construct_piece_square_table() -> PieceSquareTable {
    let mut pst = [[S::NULL; 64]; 13];
    let mut colour = WHITE;
    loop {
        let offset = if colour == BLACK { 7 } else { 1 };
        let multiplier = if colour == BLACK { -1 } else { 1 };
        let mut pieces_idx = 0;
        while pieces_idx < 6 {
            let mut pst_idx = 0;
            while pst_idx < 64 {
                let sq = if colour == WHITE {
                    pst_idx
                } else {
                    flip_rank(pst_idx)
                };
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
