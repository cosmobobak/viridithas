use crate::{
    board::evaluation::score::S,
    cfor,
    definitions::{
        File::{FILE_A, FILE_D, FILE_H},
        Rank::{RANK_1, RANK_2, RANK_7, RANK_8},
        Square,
    },
    piece::{Colour, Piece},
};

use super::PieceSquareTable;

// Scores are explicit for files A to D, implicitly mirrored for E to H.
const BONUS: [[[S; 4]; 8]; 7] = [
    [[S::NULL; 4]; 8],
    [[S::NULL; 4]; 8],
    [
        // Knight
        [S(-61, 30), S(4, -8), S(-2, -4), S(7, 11)],
        [S(29, 16), S(-2, 39), S(9, -6), S(20, 13)],
        [S(-1, -3), S(13, 16), S(-7, 40), S(23, 58)],
        [S(32, 44), S(60, 32), S(54, 69), S(48, 85)],
        [S(78, 40), S(54, 54), S(94, 74), S(92, 83)],
        [S(45, 39), S(84, 36), S(117, 67), S(143, 47)],
        [S(38, 30), S(5, 50), S(112, 4), S(167, 23)],
        [S(-89, 15), S(3, 44), S(-16, 53), S(121, 29)],
    ],
    [
        // Bishop
        [S(-1, 62), S(49, 15), S(5, 33), S(-22, 53)],
        [S(59, 14), S(40, 37), S(54, 20), S(13, 46)],
        [S(31, 47), S(44, 44), S(28, 55), S(28, 78)],
        [S(49, 34), S(11, 50), S(15, 75), S(53, 71)],
        [S(-2, 71), S(28, 63), S(57, 60), S(72, 76)],
        [S(47, 68), S(73, 61), S(89, 62), S(71, 46)],
        [S(-26, 95), S(-63, 90), S(1, 61), S(-9, 67)],
        [S(-44, 131), S(-58, 127), S(-132, 118), S(-66, 88)],
    ],
    [
        // Rook
        [S(6, 70), S(13, 84), S(21, 86), S(26, 81)],
        [S(-38, 78), S(-1, 53), S(10, 57), S(0, 54)],
        [S(-34, 91), S(14, 83), S(-18, 82), S(-22, 82)],
        [S(-14, 120), S(-1, 125), S(-15, 123), S(0, 103)],
        [S(9, 140), S(39, 126), S(58, 122), S(64, 106)],
        [S(31, 156), S(73, 135), S(80, 129), S(95, 116)],
        [S(62, 152), S(36, 167), S(112, 141), S(132, 133)],
        [S(103, 130), S(99, 144), S(71, 162), S(89, 148)],
    ],
    [
        // Queen
        [S(109, 71), S(140, 14), S(135, -14), S(139, 82)],
        [S(123, 73), S(117, 41), S(139, 43), S(142, 62)],
        [S(112, 119), S(118, 144), S(108, 171), S(97, 170)],
        [S(120, 174), S(131, 175), S(107, 200), S(87, 257)],
        [S(102, 206), S(109, 230), S(87, 248), S(92, 300)],
        [S(116, 219), S(152, 231), S(123, 254), S(115, 268)],
        [S(134, 243), S(74, 270), S(146, 262), S(131, 277)],
        [S(207, 169), S(239, 171), S(249, 199), S(216, 223)],
    ],
    [
        // King
        [S(289, -26), S(303, 31), S(194, 64), S(215, 19)],
        [S(263, 58), S(253, 107), S(167, 137), S(121, 142)],
        [S(152, 98), S(248, 119), S(185, 157), S(181, 169)],
        [S(129, 104), S(279, 131), S(267, 159), S(182, 188)],
        [S(181, 122), S(341, 155), S(212, 190), S(126, 198)],
        [S(187, 154), S(142, 218), S(170, 211), S(164, 183)],
        [S(241, 99), S(200, 202), S(231, 181), S(200, 160)],
        [S(243, -126), S(340, 121), S(268, 158), S(180, 125)],
    ],
];

#[rustfmt::skip]
const P_BONUS: [[S; 8]; 8] = [
    // Pawn (asymmetric distribution)
    [ S::NULL; 8 ],
    [ S(-18, 46), S(-27, 39), S(-15, 31), S(-5, -10), S(-7, 41), S(68, 25), S(63, 11), S(2, 2), ],
    [ S(-23, 31), S(-25, 29), S(-16, 19), S(12, 2), S(16, 10), S(10, 21), S(57, -4), S(12, 5), ],
    [ S(-15, 47), S(-34, 57), S(-5, 21), S(27, -5), S(8, -1), S(34, 12), S(5, 24), S(0, 14), ],
    [ S(-10, 92), S(-27, 78), S(4, 44), S(19, -4), S(28, 11), S(50, 11), S(2, 44), S(-2, 39), ],
    [ S(36, 103), S(9, 110), S(42, 52), S(66, -20), S(80, -34), S(193, -2), S(128, 36), S(91, 39), ],
    [ S(119, 117), S(66, 146), S(27, 149), S(91, 50), S(35, 62), S(2, 117), S(-34, 193), S(-129, 150), ],
    [ S::NULL; 8 ],
];

pub fn printout_pst_source(pst: &PieceSquareTable) {
    println!("PSQT source code:");
    #[rustfmt::skip]
    println!(
"static BONUS: [[[S; 4]; 8]; 7] = [
    [[S::NULL; 4]; 8],
    [[S::NULL; 4]; 8],"
    );
    let names = ["NULL", "Pawn", "Knight", "Bishop", "Rook", "Queen", "King"];
    for piece in Piece::all().skip(1).take(5) {
        // white knight to white king
        println!("    [");
        println!("        // {}", names[piece.index()]);
        for rank in RANK_1..=RANK_8 {
            print!("        [");
            for file in FILE_A..=FILE_D {
                let sq = Square::from_rank_file(rank, file);
                let val = pst[piece.index()][sq.index()];
                print!("{val}, ");
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
            let sq = Square::from_rank_file(rank, file);
            let val = pst[Piece::WP.index()][sq.index()];
            print!("{val}, ");
        }
        println!("],");
    }
    println!("    [ S::NULL; 8 ],");
    println!("];");
}

pub const fn construct_piece_square_table() -> PieceSquareTable {
    let mut pst = [[S::NULL; 64]; 13];
    cfor!(let mut colour = 0; colour < 2; colour += 1; {
        let offset = if colour == Colour::BLACK.inner() { 7 } else { 1 };
        let multiplier = if colour == Colour::BLACK.inner() { -1 } else { 1 };
        cfor!(let mut pieces_idx = 0; pieces_idx < 6; pieces_idx += 1; {
            cfor!(let mut pst_idx = 0; pst_idx < 64; pst_idx += 1; {
                let pst_idx = Square::new(pst_idx);
                let sq = if colour == Colour::WHITE.inner() { pst_idx } else { pst_idx.flip_rank() };
                let r = pst_idx.rank() as usize;
                let f = pst_idx.file() as usize;
                let value = if pieces_idx == 0 {
                    P_BONUS[r][f]
                } else {
                    let f = if f >= 4 { 7 - f } else { f };
                    BONUS[pieces_idx + 1][r][f]
                };
                let S(mg, eg) = value;
                pst[pieces_idx + offset][sq.index()] = S(mg * multiplier, eg * multiplier);
            });
        });
    });
    pst
}
