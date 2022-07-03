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
        [S(-148, -69), S(-65, -41), S(-53, -42), S(-61, -2)],
        [S(-56, -40), S(-68, -28), S(-34, -19), S(-13, -19)],
        [S(-55, -30), S(-11, -21), S(-21, -13), S(4, 25)],
        [S(-21, -10), S(22, 23), S(18, 39), S(22, 39)],
        [S(-7, -18), S(36, 11), S(56, 36), S(41, 57)],
        [S(18, -24), S(49, -17), S(85, 11), S(80, 35)],
        [S(-40, -42), S(0, -23), S(31, -24), S(64, 39)],
        [S(-174, -73), S(-56, -61), S(-29, -29), S(1, 10)],
    ],
    [
        // Bishop
        [S(-19, -25), S(23, 1), S(-11, -5), S(-35, -8)],
        [S(16, -18), S(9, -23), S(13, -20), S(-24, -8)],
        [S(-12, -22), S(12, -18), S(-26, -2), S(-7, 16)],
        [S(23, -29), S(-2, -27), S(-9, 0), S(17, 10)],
        [S(-6, 12), S(0, 1), S(34, -15), S(34, 8)],
        [S(16, 6), S(31, 10), S(28, 15), S(35, -14)],
        [S(15, 5), S(-15, 13), S(21, -10), S(0, 12)],
        [S(-26, -5), S(1, -4), S(-16, 1), S(-32, 4)],
    ],
    [
        // Rook
        [S(-9, -31), S(-2, -39), S(-24, 3), S(-20, -16)],
        [S(-48, -28), S(-20, -29), S(-30, -28), S(-21, -29)],
        [S(-34, 0), S(5, -30), S(-28, -14), S(-24, -33)],
        [S(-12, 21), S(-1, -4), S(-21, 4), S(-33, 11)],
        [S(0, 22), S(12, 18), S(-3, 31), S(25, 21)],
        [S(4, 30), S(25, 17), S(33, 20), S(27, 32)],
        [S(12, 31), S(16, 32), S(15, 45), S(45, 22)],
        [S(10, 25), S(8, 0), S(24, 35), S(33, 23)],
    ],
    [
        // Queen
        [S(-13, -75), S(-25, -76), S(-32, -74), S(-10, -51)],
        [S(7, -81), S(-10, -58), S(-14, -49), S(-15, -31)],
        [S(1, -64), S(-4, -38), S(-14, -18), S(-20, -24)],
        [S(8, -31), S(11, -2), S(-3, -2), S(-19, 29)],
        [S(0, -2), S(14, 17), S(16, 33), S(-3, 48)],
        [S(23, -11), S(37, 9), S(33, 16), S(35, 28)],
        [S(22, -23), S(-21, 0), S(36, 3), S(30, 18)],
        [S(25, -60), S(25, -31), S(28, -16), S(25, -7)],
    ],
    [
        // King
        [S(296, 2), S(300, 49), S(254, 58), S(225, 77)],
        [S(268, 67), S(276, 81), S(207, 112), S(168, 119)],
        [S(168, 81), S(231, 103), S(142, 142), S(136, 152)],
        [S(144, 115), S(166, 142), S(123, 172), S(122, 189)],
        [S(179, 123), S(181, 164), S(131, 205), S(97, 210)],
        [S(150, 119), S(172, 199), S(108, 211), S(58, 218)],
        [S(115, 74), S(147, 148), S(92, 143), S(60, 158)],
        [S(86, 38), S(116, 86), S(72, 100), S(26, 105)],
    ],
];

#[rustfmt::skip]
const P_BONUS: [[S; 8]; 8] = [
    // Pawn (asymmetric distribution)
    [ S::NULL; 8 ],
    [ S(-18, 14), S(-22, 21), S(-7, 35), S(20, 31), S(22, 26), S(44, 30), S(34, 9), S(2, 0), ],
    [ S(-26, -4), S(-19, 13), S(-12, 9), S(6, 12), S(21, 16), S(11, 29), S(22, 1), S(-4, 0), ],
    [ S(-28, 18), S(-31, 26), S(0, 6), S(7, -6), S(12, -5), S(22, 14), S(18, 16), S(10, 3), ],
    [ S(0, 39), S(-13, 33), S(-4, 28), S(8, -21), S(28, 8), S(0, 23), S(15, 41), S(32, 23), ],
    [ S(19, 54), S(6, 45), S(19, 46), S(20, 15), S(19, 54), S(22, 36), S(13, 35), S(16, 41), ],
    [ S(20, 26), S(33, 13), S(25, 40), S(16, 49), S(31, 51), S(13, 44), S(37, 34), S(-3, 34), ],
    [ S::NULL; 8 ],
];

pub fn printout_pst_source(pst: &PieceSquareTable) {
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
