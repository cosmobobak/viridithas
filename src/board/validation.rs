use crate::{
    board::evaluation::score::S,
    definitions::{
        colour_of, square_name,
        Rank::{RANK_3, RANK_6},
        Square::NO_SQUARE,
        BB, BISHOP, BK, BLACK, BN, BP, BQ, BR, KING, KNIGHT, PAWN, PIECE_EMPTY, QUEEN, ROOK, WB,
        WHITE, WK, WN, WP, WQ, WR,
    },
    errors::PositionValidityError,
    lookups::{piece_char, rank, PIECE_BIG, PIECE_MAJ, PIECE_MIN},
    piecelist::PieceList,
};

use super::{movegen::bitboards::BitLoop, Board};

impl Board {
    #[allow(clippy::cognitive_complexity, clippy::too_many_lines, dead_code)]
    pub fn check_validity(&self) -> Result<(), PositionValidityError> {
        #![allow(clippy::similar_names, clippy::cast_possible_truncation)]
        let mut piece_num = [0u8; 13];
        let mut big_pce = [0, 0];
        let mut maj_pce = [0, 0];
        let mut min_pce = [0, 0];
        let mut material = [S(0, 0), S(0, 0)];

        // check piece lists
        for piece in WP..=BK {
            for &sq in self.piece_lists[piece as usize].iter() {
                if self.piece_at(sq) != piece {
                    return Err(format!(
                        "piece list corrupt: expected square {} to be '{}' but was '{}'",
                        square_name(sq).unwrap_or(&format!("offboard: {}", sq)),
                        piece_char(piece)
                            .map(|c| c.to_string())
                            .unwrap_or(format!("unknown piece: {}", piece)),
                        piece_char(self.piece_at(sq))
                            .map(|c| c.to_string())
                            .unwrap_or(format!("unknown piece: {}", self.piece_at(sq)))
                    ));
                }
            }
        }

        // check turn
        if self.side != WHITE && self.side != BLACK {
            return Err(format!("invalid side: {}", self.side));
        }

        // check piece count and other counters
        for sq in 0..64 {
            let piece = self.piece_at(sq);
            if piece == PIECE_EMPTY {
                continue;
            }
            piece_num[piece as usize] += 1;
            let colour = colour_of(piece);
            if PIECE_BIG[piece as usize] {
                big_pce[colour as usize] += 1;
            }
            if PIECE_MAJ[piece as usize] {
                maj_pce[colour as usize] += 1;
            }
            if PIECE_MIN[piece as usize] {
                min_pce[colour as usize] += 1;
            }
            material[colour as usize] += self.eval_params.piece_values[piece as usize];
        }

        if piece_num[1..].to_vec()
            != self.piece_lists[1..]
                .iter()
                .map(PieceList::len)
                .collect::<Vec<_>>()
        {
            return Err(format!(
                "piece counts are corrupt: expected {:?}, got {:?}",
                &piece_num[1..],
                &self.piece_lists[1..]
                    .iter()
                    .map(PieceList::len)
                    .collect::<Vec<_>>()
            ));
        }

        // check bitboard / piece array coherency
        for piece in WP..=BK {
            let bb = self.pieces.piece_bb(piece);
            for sq in BitLoop::new(bb) {
                if self.piece_at(sq) != piece {
                    return Err(format!(
                        "bitboard / piece array coherency corrupt: expected square {} to be '{}' but was '{}'",
                        square_name(sq).unwrap_or(&format!("offboard: {}", sq)),
                        piece_char(piece).map(|c| c.to_string()).unwrap_or(format!("unknown piece: {}", piece)),
                        piece_char(self.piece_at(sq)).map(|c| c.to_string()).unwrap_or(format!("unknown piece: {}", self.piece_at(sq)))
                    ));
                }
            }
        }

        if material[WHITE as usize].0 != self.material[WHITE as usize].0 {
            return Err(format!(
                "white midgame material is corrupt: expected {:?}, got {:?}",
                material[WHITE as usize].0, self.material[WHITE as usize].0
            ));
        }
        if material[WHITE as usize].1 != self.material[WHITE as usize].1 {
            return Err(format!(
                "white endgame material is corrupt: expected {:?}, got {:?}",
                material[WHITE as usize].1, self.material[WHITE as usize].1
            ));
        }
        if material[BLACK as usize].0 != self.material[BLACK as usize].0 {
            return Err(format!(
                "black midgame material is corrupt: expected {:?}, got {:?}",
                material[BLACK as usize].0, self.material[BLACK as usize].0
            ));
        }
        if material[BLACK as usize].1 != self.material[BLACK as usize].1 {
            return Err(format!(
                "black endgame material is corrupt: expected {:?}, got {:?}",
                material[BLACK as usize].1, self.material[BLACK as usize].1
            ));
        }
        if min_pce[WHITE as usize] != self.minor_piece_counts[WHITE as usize] {
            return Err(format!(
                "white minor piece count is corrupt: expected {:?}, got {:?}",
                min_pce[WHITE as usize], self.minor_piece_counts[WHITE as usize]
            ));
        }
        if min_pce[BLACK as usize] != self.minor_piece_counts[BLACK as usize] {
            return Err(format!(
                "black minor piece count is corrupt: expected {:?}, got {:?}",
                min_pce[BLACK as usize], self.minor_piece_counts[BLACK as usize]
            ));
        }
        if maj_pce[WHITE as usize] != self.major_piece_counts[WHITE as usize] {
            return Err(format!(
                "white major piece count is corrupt: expected {:?}, got {:?}",
                maj_pce[WHITE as usize], self.major_piece_counts[WHITE as usize]
            ));
        }
        if maj_pce[BLACK as usize] != self.major_piece_counts[BLACK as usize] {
            return Err(format!(
                "black major piece count is corrupt: expected {:?}, got {:?}",
                maj_pce[BLACK as usize], self.major_piece_counts[BLACK as usize]
            ));
        }
        if big_pce[WHITE as usize] != self.big_piece_counts[WHITE as usize] {
            return Err(format!(
                "white big piece count is corrupt: expected {:?}, got {:?}",
                big_pce[WHITE as usize], self.big_piece_counts[WHITE as usize]
            ));
        }
        if big_pce[BLACK as usize] != self.big_piece_counts[BLACK as usize] {
            return Err(format!(
                "black big piece count is corrupt: expected {:?}, got {:?}",
                big_pce[BLACK as usize], self.big_piece_counts[BLACK as usize]
            ));
        }

        if !(self.side == WHITE || self.side == BLACK) {
            return Err(format!(
                "side is corrupt: expected WHITE or BLACK, got {:?}",
                self.side
            ));
        }
        if self.generate_pos_key() != self.key {
            return Err(format!(
                "key is corrupt: expected {:?}, got {:?}",
                self.generate_pos_key(),
                self.key
            ));
        }

        if !(self.ep_sq == NO_SQUARE
            || (rank(self.ep_sq) == RANK_6 && self.side == WHITE)
            || (rank(self.ep_sq) == RANK_3 && self.side == BLACK))
        {
            return Err(format!("en passant square is corrupt: expected square to be {} (NoSquare) or to be on ranks 6 or 3, got {} (Rank {})", NO_SQUARE, self.ep_sq, rank(self.ep_sq)));
        }

        if self.fifty_move_counter >= 100 {
            return Err(format!(
                "fifty move counter is corrupt: expected 0-99, got {}",
                self.fifty_move_counter
            ));
        }

        // check there are the correct number of kings for each side
        if self.num(WK) != 1 {
            return Err(format!(
                "white king count is corrupt: expected 1, got {}",
                self.num(WK)
            ));
        }
        if self.num(BK) != 1 {
            return Err(format!(
                "black king count is corrupt: expected 1, got {}",
                self.num(BK)
            ));
        }
        let comptime_consistency = self.num(WP) == self.num_ct::<WP>()
            && self.num(BP) == self.num_ct::<BP>()
            && self.num(WN) == self.num_ct::<WN>()
            && self.num(BN) == self.num_ct::<BN>()
            && self.num(WB) == self.num_ct::<WB>()
            && self.num(BB) == self.num_ct::<BB>()
            && self.num(WR) == self.num_ct::<WR>()
            && self.num(BR) == self.num_ct::<BR>()
            && self.num(WQ) == self.num_ct::<WQ>()
            && self.num(BQ) == self.num_ct::<BQ>()
            && self.num(WK) == self.num_ct::<WK>()
            && self.num(BK) == self.num_ct::<BK>();
        if !comptime_consistency {
            return Err(
                "comptime consistency for Board::num is corrupt: expected true, got false".into(),
            );
        }
        let comptime_consistency = self.num_pt(PAWN) == self.num_pt_ct::<PAWN>()
            && self.num_pt(KNIGHT) == self.num_pt_ct::<KNIGHT>()
            && self.num_pt(BISHOP) == self.num_pt_ct::<BISHOP>()
            && self.num_pt(ROOK) == self.num_pt_ct::<ROOK>()
            && self.num_pt(QUEEN) == self.num_pt_ct::<QUEEN>()
            && self.num_pt(KING) == self.num_pt_ct::<KING>();
        if !comptime_consistency {
            return Err(
                "comptime consistency for Board::num_pt is corrupt: expected true, got false"
                    .into(),
            );
        }

        if self.piece_at(self.king_sq(WHITE)) != WK {
            return Err(format!(
                "white king square is corrupt: expected white king, got {:?}",
                self.piece_at(self.king_sq(WHITE))
            ));
        }
        if self.piece_at(self.king_sq(BLACK)) != BK {
            return Err(format!(
                "black king square is corrupt: expected black king, got {:?}",
                self.piece_at(self.king_sq(BLACK))
            ));
        }

        Ok(())
    }
}
