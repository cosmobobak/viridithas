use crate::{piece::{PieceType, Colour, Piece}, board::{Board, GameOutcome}, definitions::{Rank, Square}, squareset::SquareSet};

const UNMOVED_ROOK: u8 = PieceType::NONE.inner();

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct PackedBoard {
    occupancy: util::U64Le,
    pieces: util::U4Array32,
    stm_ep_square: u8,
    halfmove_clock: u8,
    fullmove_number: util::U16Le,
    eval: util::I16Le,
    wdl: u8,
    extra: u8,
}

impl PackedBoard {
    pub fn set_outcome(&mut self, outcome: GameOutcome) {
        self.wdl = outcome.as_packed_u8();
    }

    pub fn pack(board: &Board, eval: i16, wdl: u8, extra: u8) -> Self {
        let occupancy = board.pieces.occupied();

        let mut pieces = util::U4Array32::default();
        for (i, sq) in occupancy.iter().enumerate() {
            let piece = board.piece_at(sq);
            let piece_type = piece.piece_type();
            let colour = piece.colour();

            let mut piece_code = piece_type.inner();
            let rank1 = if colour == Colour::WHITE {
                Rank::RANK_1
            } else {
                Rank::RANK_8
            };
            if piece_type == PieceType::ROOK && sq.rank() == rank1 {
                let castling_sq = match board.king_sq(colour) < sq {
                    true => board.castling_rights().kingside(colour),
                    false => board.castling_rights().queenside(colour),
                };
                let castling_file = if castling_sq != Square::NO_SQUARE {
                    Some(castling_sq.file())
                } else {
                    None
                };
                if Some(sq.file()) == castling_file {
                    piece_code = UNMOVED_ROOK;
                }
            }

            debug_assert_eq!(piece_code & 0b0111, piece_code);
            debug_assert_ne!(piece_code, 0b0111, "we are not using the 0b0111 piece code");
            pieces.set(i, piece_code | (colour.inner()) << 3);
        }

        Self {
            occupancy: util::U64Le::new(occupancy.inner()),
            pieces,
            stm_ep_square: (board.turn().inner()) << 7
                | board.ep_sq().inner(),
            halfmove_clock: board.fifty_move_counter(),
            fullmove_number: util::U16Le::new(board.full_move_number() as u16),
            wdl,
            eval: util::I16Le::new(eval),
            extra,
        }
    }

    pub fn unpack(&self) -> Option<(Board, i16, u8, u8)> {
        let mut builder = Board::new();

        let mut seen_king = [false; 2];
        for (i, sq) in SquareSet::from_inner(self.occupancy.get()).iter().enumerate() {
            let colour = Colour::new((self.pieces.get(i) as usize >> 3) as u8);
            let piece_code = self.pieces.get(i) & 0b0111;
            let piece_type = match piece_code {
                UNMOVED_ROOK => {
                    if seen_king[colour.index()] {
                        *builder.castling_rights_mut().kingside_mut(colour) = sq;
                    } else {
                        *builder.castling_rights_mut().queenside_mut(colour) = sq;
                    }
                    PieceType::ROOK
                }
                _ => PieceType::new(piece_code),
            };
            if piece_type == PieceType::KING {
                seen_king[colour.index()] = true;
            }
            builder.add_piece(sq, Piece::new(colour, piece_type));
        }

        *builder.ep_sq_mut() = Square::new((self.stm_ep_square as usize & 0b0111_1111) as u8);
        *builder.turn_mut() = Colour::new((self.stm_ep_square as usize >> 7) as u8);
        *builder.halfmove_clock_mut() = self.halfmove_clock;
        builder.set_fullmove_clock(self.fullmove_number.get());

        builder.regenerate_zobrist();

        Some((builder, self.eval.get(), self.wdl, self.extra))
    }
}

impl Board {
    pub fn pack(&self, eval: i16, wdl: u8, extra: u8) -> PackedBoard {
        PackedBoard::pack(self, eval, wdl, extra)
    }

    pub fn unpack(packed: &PackedBoard) -> Option<(Self, i16, u8, u8)> {
        packed.unpack()
    }
}

mod util {
    #[derive(Copy, Clone, Debug, Default)]
    #[repr(transparent)]
    pub struct U64Le(u64);

    impl U64Le {
        pub fn new(v: u64) -> Self {
            Self(v.to_le())
        }

        pub fn get(self) -> u64 {
            u64::from_le(self.0)
        }
    }

    #[derive(Copy, Clone, Debug, Default)]
    #[repr(transparent)]
    pub struct U16Le(u16);

    impl U16Le {
        pub fn new(v: u16) -> Self {
            Self(v.to_le())
        }

        pub fn get(self) -> u16 {
            u16::from_le(self.0)
        }
    }

    #[derive(Copy, Clone, Debug, Default)]
    #[repr(transparent)]
    pub struct I16Le(i16);

    impl I16Le {
        pub fn new(v: i16) -> Self {
            Self(v.to_le())
        }

        pub fn get(self) -> i16 {
            i16::from_le(self.0)
        }
    }

    #[derive(Copy, Clone, Debug, Default)]
    #[repr(transparent)]
    pub struct U4Array32([u8; 16]);

    impl U4Array32 {
        pub fn get(&self, i: usize) -> u8 {
            (self.0[i / 2] >> ((i % 2) * 4)) & 0xF
        }

        pub fn set(&mut self, i: usize, v: u8) {
            debug_assert!(v < 0x10);
            self.0[i / 2] |= v << ((i % 2) * 4);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::uci::CHESS960;

    use super::*;

    #[test]
    fn roundtrip() {
        CHESS960.store(true, std::sync::atomic::Ordering::SeqCst);
        // Grab `valid.sfens` from `cozy-chess` to run test
        for sfen in include_str!("valid.sfens").lines() {
            let board = Board::from_fen(sfen).unwrap();
            let packed = PackedBoard::pack(&board, 0, 0, 0);
            let (unpacked, _, _, _) = packed
                .unpack()
                .unwrap_or_else(|| panic!("Failed to unpack {}. {:#X?}", sfen, packed));
            crate::board::check_eq(&board, &unpacked, sfen);
        }
    }
}
