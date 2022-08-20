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
        [S(-30, 52), S(4, -24), S(-7, -33), S(5, -7), ],
        [S(33, 6), S(-6, 12), S(3, -14), S(19, -5), ],
        [S(-12, -2), S(6, -1), S(-9, 23), S(21, 48), ],
        [S(38, 19), S(68, 20), S(49, 61), S(43, 67), ],
        [S(75, 22), S(55, 35), S(93, 65), S(99, 69), ],
        [S(54, 28), S(111, 20), S(144, 44), S(171, 39), ],
        [S(63, 9), S(26, 37), S(154, -3), S(170, 11), ],
        [S(-64, 46), S(50, 26), S(29, 40), S(122, 20), ],
    ],
    [
        // Bishop
        [S(2, 68), S(43, 29), S(1, 33), S(-20, 32), ],
        [S(59, 26), S(40, 27), S(53, 3), S(4, 31), ],
        [S(31, 31), S(48, 19), S(20, 43), S(19, 58), ],
        [S(44, 21), S(22, 24), S(4, 60), S(42, 56), ],
        [S(0, 62), S(20, 52), S(61, 40), S(58, 68), ],
        [S(51, 63), S(77, 41), S(108, 41), S(91, 27), ],
        [S(-11, 75), S(-25, 77), S(11, 48), S(20, 52), ],
        [S(-32, 100), S(-32, 98), S(-110, 95), S(-51, 60), ],
    ],
    [
        // Rook
        [S(10, 61), S(14, 68), S(21, 74), S(32, 54), ],
        [S(-38, 49), S(-4, 35), S(-7, 32), S(-1, 32), ],
        [S(-43, 80), S(-1, 55), S(-34, 63), S(-30, 52), ],
        [S(-38, 113), S(-4, 99), S(-14, 95), S(-11, 82), ],
        [S(24, 117), S(33, 114), S(37, 109), S(41, 100), ],
        [S(28, 137), S(77, 115), S(79, 115), S(74, 104), ],
        [S(56, 135), S(60, 137), S(77, 139), S(107, 129), ],
        [S(108, 117), S(101, 124), S(63, 148), S(76, 131), ],
    ],
    [
        // Queen
        [S(33, 30), S(54, -27), S(60, -75), S(68, 19), ],
        [S(72, 2), S(55, -25), S(67, -11), S(77, 0), ],
        [S(60, 26), S(56, 74), S(40, 95), S(35, 96), ],
        [S(61, 69), S(65, 100), S(44, 101), S(19, 164), ],
        [S(58, 110), S(40, 124), S(43, 145), S(28, 192), ],
        [S(80, 104), S(102, 119), S(78, 132), S(72, 146), ],
        [S(116, 126), S(47, 152), S(86, 141), S(52, 155), ],
        [S(176, 58), S(175, 90), S(176, 122), S(104, 105), ],
    ],
    [
        // King
        [S(303, -31), S(318, 28), S(212, 52), S(210, 32), ],
        [S(283, 42), S(270, 94), S(179, 131), S(131, 138), ],
        [S(154, 72), S(245, 106), S(193, 141), S(164, 161), ],
        [S(97, 101), S(251, 127), S(216, 168), S(198, 182), ],
        [S(142, 121), S(265, 156), S(206, 186), S(159, 189), ],
        [S(175, 143), S(111, 204), S(92, 211), S(158, 188), ],
        [S(205, 111), S(197, 206), S(183, 187), S(193, 165), ],
        [S(144, -13), S(240, 152), S(186, 193), S(162, 187), ],
    ],
];

#[rustfmt::skip]
const P_BONUS: [[S; 8]; 8] = [
    // Pawn (asymmetric distribution)
    [ S::NULL; 8 ],
    [ S(-16, 42), S(-24, 31), S(-14, 16), S(-9, -7), S(-4, 16), S(50, 14), S(33, 15), S(-4, 3), ],
    [ S(-17, 25), S(-19, 16), S(-20, 5), S(10, -21), S(10, -7), S(6, 10), S(38, -7), S(8, 0), ],
    [ S(-17, 37), S(-25, 44), S(-3, 6), S(26, -24), S(5, -19), S(22, 0), S(-4, 22), S(-4, 5), ],
    [ S(-4, 78), S(-25, 75), S(2, 29), S(35, -15), S(35, -1), S(46, 6), S(-8, 48), S(11, 37), ],
    [ S(23, 100), S(10, 93), S(49, 40), S(63, -24), S(77, -18), S(167, 11), S(95, 50), S(79, 55), ],
    [ S(110, 158), S(94, 163), S(49, 181), S(16, 119), S(-1, 146), S(32, 162), S(53, 186), S(-83, 174), ],
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
