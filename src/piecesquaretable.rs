pub mod tables;

use crate::{board::evaluation::score::S, definitions::Square::A1, lookups::piece_name};

pub type PieceSquareTable = [[S; 64]; 13];

pub fn pst_value(piece: u8, sq: u8, pst: &PieceSquareTable) -> S {
    debug_assert!(crate::validate::piece_valid(piece));
    debug_assert!(crate::validate::square_on_board(sq));
    unsafe { *pst.get_unchecked(piece as usize).get_unchecked(sq as usize) }
}

pub fn render_pst_table(pst: &PieceSquareTable) {
    #![allow(clippy::needless_range_loop, clippy::cast_possible_truncation)]
    for piece in 0..13 {
        println!("{}", piece_name(piece as u8).unwrap());
        println!("mg eval on a1 (bottom left) {}", pst[piece][A1 as usize].0);
        for row in (0..8).rev() {
            print!("RANK {}: ", row + 1);
            for col in 0..8 {
                let sq = row * 8 + col;
                let pst_val = pst[piece][sq].0;
                print!("{:>5}", pst_val);
            }
            println!();
        }
        println!("eg eval on a1 (bottom left) {}", pst[piece][A1 as usize].1);
        for row in (0..8).rev() {
            print!("RANK {}: ", row + 1);
            for col in 0..8 {
                let sq = row * 8 + col;
                let pst_val = pst[piece][sq].1;
                print!("{:>5}", pst_val);
            }
            println!();
        }
    }
}

mod tests {
    #[test]
    fn psts_are_mirrored_properly() {
        #![allow(clippy::similar_names, clippy::cast_possible_truncation)]
        use super::*;
        use crate::definitions::square_name;
        let psts = super::tables::construct_piece_square_table();
        for white_piece in 1..7 {
            let white_pst = &psts[white_piece];
            let black_pst = &psts[white_piece + 6];
            for sq in 0..64 {
                assert_eq!(
                    white_pst[sq as usize],
                    -black_pst[crate::definitions::flip_rank(sq) as usize],
                    "pst mirroring failed on square {} for piece {}",
                    square_name(sq as u8).unwrap(),
                    piece_name(white_piece as u8).unwrap()
                );
            }
        }
    }
}
