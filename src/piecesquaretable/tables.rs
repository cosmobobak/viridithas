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
        [S(-70, 0), S(-12, -23), S(-14, -53), S(-12, -21), ],
        [S(7, 3), S(-28, -1), S(-11, -26), S(10, -13), ],
        [S(-26, -17), S(-5, -11), S(-15, 12), S(14, 30), ],
        [S(22, 2), S(50, 10), S(37, 47), S(31, 52), ],
        [S(59, 9), S(44, 23), S(77, 51), S(71, 65), ],
        [S(37, 20), S(79, 16), S(119, 38), S(139, 38), ],
        [S(36, 7), S(29, 22), S(110, 4), S(127, 16), ],
        [S(-95, 6), S(23, 18), S(37, 27), S(72, 21), ],
    ],
    [
        // Bishop
        [S(7, 31), S(32, 14), S(-2, 22), S(-25, 17), ],
        [S(50, 13), S(33, 19), S(43, -3), S(-3, 24), ],
        [S(25, 19), S(38, 11), S(11, 32), S(11, 52), ],
        [S(39, 11), S(15, 15), S(-3, 52), S(34, 52), ],
        [S(-12, 55), S(13, 43), S(54, 31), S(52, 59), ],
        [S(37, 51), S(67, 31), S(90, 38), S(84, 22), ],
        [S(-2, 50), S(-21, 64), S(6, 39), S(4, 51), ],
        [S(-7, 72), S(3, 59), S(-53, 71), S(-43, 52), ],
    ],
    [
        // Rook
        [S(-8, 38), S(0, 27), S(10, 43), S(18, 24), ],
        [S(-51, 13), S(-8, -4), S(-18, -2), S(-13, -5), ],
        [S(-58, 48), S(-7, 17), S(-43, 26), S(-42, 15), ],
        [S(-39, 71), S(-7, 49), S(-18, 48), S(-21, 46), ],
        [S(21, 75), S(31, 58), S(28, 73), S(28, 71), ],
        [S(43, 83), S(70, 67), S(62, 76), S(58, 76), ],
        [S(34, 100), S(42, 93), S(41, 102), S(87, 89), ],
        [S(65, 86), S(54, 76), S(72, 98), S(79, 80), ],
    ],
    [
        // Queen
        [S(-16, -34), S(0, -70), S(18, -119), S(30, -27), ],
        [S(35, -63), S(13, -68), S(23, -52), S(33, -44), ],
        [S(20, -43), S(17, 11), S(-3, 45), S(-6, 34), ],
        [S(25, -4), S(24, 31), S(4, 35), S(-14, 104), ],
        [S(14, 37), S(0, 51), S(13, 74), S(-9, 121), ],
        [S(35, 31), S(59, 51), S(47, 65), S(30, 74), ],
        [S(80, 53), S(8, 79), S(49, 73), S(33, 94), ],
        [S(103, -15), S(104, 17), S(106, 49), S(96, 59), ],
    ],
    [
        // King
        [S(292, -21), S(312, 34), S(212, 51), S(214, 34), ],
        [S(275, 46), S(269, 92), S(183, 126), S(137, 131), ],
        [S(150, 70), S(235, 101), S(193, 134), S(171, 154), ],
        [S(95, 97), S(202, 133), S(168, 171), S(176, 181), ],
        [S(125, 119), S(215, 161), S(173, 188), S(137, 190), ],
        [S(173, 139), S(162, 192), S(101, 208), S(107, 195), ],
        [S(173, 124), S(189, 202), S(152, 191), S(137, 179), ],
        [S(136, 49), S(180, 147), S(151, 177), S(105, 184), ],
    ],
];

#[rustfmt::skip]
const P_BONUS: [[S; 8]; 8] = [
    // Pawn (asymmetric distribution)
    [ S::NULL; 8 ],
    [ S(-13, 52), S(-17, 37), S(-14, 17), S(-15, 4), S(-7, 12), S(40, 19), S(29, 22), S(-8, 10), ],
    [ S(-12, 28), S(-13, 16), S(-21, 5), S(6, -21), S(5, -11), S(-2, 11), S(34, -4), S(7, 1), ],
    [ S(-13, 36), S(-18, 44), S(-6, 6), S(26, -26), S(0, -18), S(15, -2), S(-7, 24), S(-6, 4), ],
    [ S(-1, 76), S(-21, 73), S(0, 29), S(33, -14), S(31, 0), S(42, 5), S(-9, 45), S(9, 35), ],
    [ S(18, 109), S(-3, 108), S(28, 56), S(40, 7), S(49, 12), S(101, 49), S(76, 69), S(64, 66), ],
    [ S(99, 105), S(107, 92), S(77, 119), S(50, 109), S(34, 129), S(71, 123), S(88, 113), S(-14, 107), ],
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
