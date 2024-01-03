pub mod tables;

use crate::{board::evaluation::score::S, piece::Piece, util::Square};

pub type PieceSquareTable = [[S; 64]; 13];

pub fn pst_value(piece: Piece, sq: Square, pst: &PieceSquareTable) -> S {
    debug_assert!(sq.on_board());
    pst[piece.index()][sq.index()]
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
                    "pst mirroring failed on square {sq} for piece {white_piece}"
                );
            }
        }
    }
}
