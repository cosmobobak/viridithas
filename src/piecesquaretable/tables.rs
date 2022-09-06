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
        [S(-56, 41), S(1, -13), S(-1, 1), S(5, 13), ],
        [S(23, 18), S(-9, 36), S(4, 7), S(17, 14), ],
        [S(-7, 6), S(13, 14), S(-8, 32), S(20, 58), ],
        [S(28, 40), S(58, 37), S(50, 67), S(44, 80), ],
        [S(74, 45), S(57, 43), S(89, 72), S(85, 77), ],
        [S(51, 32), S(88, 34), S(120, 55), S(136, 46), ],
        [S(33, 20), S(9, 45), S(127, -2), S(179, 8), ],
        [S(-99, 13), S(0, 40), S(-25, 39), S(119, 18), ],
    ],
    [
        // Bishop
        [S(-3, 52), S(34, 15), S(3, 28), S(-30, 52), ],
        [S(52, 6), S(36, 30), S(51, 19), S(13, 40), ],
        [S(32, 32), S(41, 37), S(28, 49), S(28, 75), ],
        [S(50, 28), S(4, 46), S(15, 67), S(47, 69), ],
        [S(7, 60), S(33, 55), S(59, 52), S(65, 74), ],
        [S(53, 57), S(84, 49), S(83, 59), S(74, 37), ],
        [S(-29, 77), S(-47, 75), S(13, 53), S(5, 57), ],
        [S(-49, 121), S(-58, 107), S(-126, 116), S(-80, 69), ],
    ],
    [
        // Rook
        [S(5, 65), S(16, 78), S(18, 86), S(26, 78), ],
        [S(-36, 67), S(13, 37), S(6, 54), S(4, 49), ],
        [S(-36, 90), S(3, 74), S(-27, 82), S(-27, 76), ],
        [S(-15, 114), S(-1, 115), S(-15, 113), S(-1, 95), ],
        [S(1, 135), S(25, 121), S(43, 116), S(67, 95), ],
        [S(22, 150), S(58, 130), S(62, 127), S(93, 105), ],
        [S(63, 146), S(43, 152), S(100, 134), S(121, 126), ],
        [S(86, 126), S(78, 139), S(55, 157), S(74, 141), ],
    ],
    [
        // Queen
        [S(92, 78), S(128, 17), S(123, -24), S(131, 68), ],
        [S(118, 70), S(103, 29), S(130, 20), S(134, 51), ],
        [S(101, 102), S(109, 138), S(97, 154), S(87, 149), ],
        [S(112, 152), S(120, 157), S(96, 181), S(76, 235), ],
        [S(96, 182), S(101, 203), S(81, 226), S(83, 272), ],
        [S(114, 187), S(145, 202), S(120, 218), S(118, 230), ],
        [S(135, 210), S(77, 236), S(145, 227), S(135, 240), ],
        [S(198, 140), S(228, 146), S(231, 185), S(189, 191), ],
    ],
    [
        // King
        [S(289, -20), S(303, 34), S(196, 63), S(216, 20), ],
        [S(263, 59), S(250, 107), S(165, 138), S(118, 144), ],
        [S(142, 100), S(250, 118), S(186, 157), S(179, 169), ],
        [S(118, 102), S(268, 131), S(264, 161), S(184, 187), ],
        [S(165, 122), S(326, 155), S(223, 186), S(140, 194), ],
        [S(170, 155), S(134, 215), S(150, 207), S(188, 182), ],
        [S(210, 100), S(189, 192), S(209, 182), S(190, 157), ],
        [S(206, -99), S(303, 125), S(231, 152), S(142, 117), ],
    ],
];

#[rustfmt::skip]
static P_BONUS: [[S; 8]; 8] = [
    // Pawn (asymmetric distribution)
    [ S::NULL; 8 ],
    [ S(-19, 39), S(-27, 29), S(-13, 28), S(-3, -23), S(-3, 36), S(63, 23), S(55, 9), S(-4, -2), ],
    [ S(-23, 23), S(-25, 22), S(-16, 16), S(15, -3), S(16, 6), S(9, 17), S(52, -4), S(7, 4), ],
    [ S(-16, 39), S(-30, 50), S(-2, 15), S(26, -7), S(12, -3), S(35, 11), S(0, 23), S(-2, 14), ],
    [ S(-12, 88), S(-26, 71), S(2, 38), S(18, -7), S(27, 10), S(50, 7), S(0, 48), S(-6, 38), ],
    [ S(36, 93), S(6, 105), S(43, 43), S(59, -20), S(77, -27), S(181, 5), S(118, 36), S(87, 40), ],
    [ S(96, 118), S(47, 158), S(6, 161), S(59, 55), S(33, 89), S(-13, 137), S(-19, 209), S(-156, 164), ],
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
