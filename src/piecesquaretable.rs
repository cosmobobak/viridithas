pub mod tables;

use crate::{board::evaluation::score::S, definitions::Square, piece::Piece};

pub type PieceSquareTable = [[S; 64]; 13];

pub fn pst_value(piece: Piece, sq: Square, pst: &PieceSquareTable) -> S {
    debug_assert!(sq.on_board());
    unsafe { *pst.get_unchecked(piece.index()).get_unchecked(sq.index()) }
}

pub fn render_pst_table(pst: &PieceSquareTable) {
    #![allow(clippy::needless_range_loop, clippy::cast_possible_truncation)]
    for piece in Piece::all() {
        println!("{piece}");
        let piece = piece.index();
        println!("mg eval on a1 (bottom left) {}", pst[piece][Square::A1.index()].0);
        for row in (0..8).rev() {
            print!("RANK {}: ", row + 1);
            for col in 0..8 {
                let sq = row * 8 + col;
                let pst_val = pst[piece][sq].0;
                print!("{pst_val:>5}");
            }
            println!();
        }
        println!("eg eval on a1 (bottom left) {}", pst[piece][Square::A1.index()].1);
        for row in (0..8).rev() {
            print!("RANK {}: ", row + 1);
            for col in 0..8 {
                let sq = row * 8 + col;
                let pst_val = pst[piece][sq].1;
                print!("{pst_val:>5}");
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
        let psts = super::tables::construct_piece_square_table();
        for white_piece in Piece::all().take(6) {
            let idx = white_piece.index();
            let white_pst = &psts[idx];
            let black_pst = &psts[idx + 6];
            for sq in 0..64 {
                let sq = Square::new(sq);
                assert_eq!(
                    white_pst[sq.index()],
                    -black_pst[sq.flip_rank().index()],
                    "pst mirroring failed on square {sq} for piece {}",
                    white_piece
                );
            }
        }
    }
}
