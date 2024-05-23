use crate::piece::Piece;

use crate::squareset::SquareSet;

use crate::util::Square;

use crate::util::Rank;

use crate::piece::Colour;

use crate::board::Board;

use crate::board::GameOutcome;

use crate::piece::PieceType;

const UNMOVED_ROOK: u8 = 6; // one higher than the max piecetype enum value

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct PackedBoard {
    pub(crate) occupancy: util::U64Le,
    pub(crate) pieces: util::U4Array32,
    pub(crate) stm_ep_square: u8,
    pub(crate) halfmove_clock: u8,
    pub(crate) fullmove_number: util::U16Le,
    pub(crate) eval: util::I16Le,
    pub(crate) wdl: u8,
    pub(crate) extra: u8,
}

impl PackedBoard {
    pub const WDL_WIN: u8 = 2;
    pub const WDL_DRAW: u8 = 1;
    pub const WDL_LOSS: u8 = 0;

    pub fn set_outcome(&mut self, outcome: GameOutcome) {
        self.wdl = outcome.as_packed_u8();
    }

    pub fn pack(board: &Board, eval: i16, wdl: u8, extra: u8) -> Self {
        let occupancy = board.pieces.occupied();

        let mut pieces = util::U4Array32::default();
        for (i, sq) in occupancy.iter().enumerate() {
            let piece = board.piece_at(sq).unwrap();
            let piece_type = piece.piece_type();
            let colour = piece.colour();

            let mut piece_code = piece_type.inner();
            let rank1 = if colour == Colour::WHITE { Rank::RANK_1 } else { Rank::RANK_8 };
            if piece_type == PieceType::Rook && sq.rank() == rank1 {
                let castling_sq = if board.king_sq(colour) < sq {
                    board.castling_rights().kingside(colour)
                } else {
                    board.castling_rights().queenside(colour)
                };
                let castling_file = if castling_sq == Square::NO_SQUARE { None } else { Some(castling_sq.file()) };
                if Some(sq.file()) == castling_file {
                    piece_code = UNMOVED_ROOK;
                }
            }

            debug_assert_eq!(piece_code & 0b0111, piece_code);
            debug_assert_ne!(piece_code, 0b0111, "we are not using the 0b0111 piece code");
            pieces.set(i, piece_code | u8::from(colour.inner()) << 3);
        }

        Self {
            occupancy: util::U64Le::new(occupancy.inner()),
            pieces,
            stm_ep_square: u8::from(board.turn().inner()) << 7 | board.ep_sq().inner(),
            halfmove_clock: board.fifty_move_counter(),
            fullmove_number: util::U16Le::new(board.full_move_number().try_into().unwrap()),
            wdl,
            eval: util::I16Le::new(eval),
            extra,
        }
    }

    pub fn unpack(&self) -> (Board, i16, u8, u8) {
        let mut builder = Board::new();

        let mut seen_king = [false; 2];
        for (i, sq) in SquareSet::from_inner(self.occupancy.get()).iter().enumerate() {
            let colour = Colour::new(self.pieces.get(i) >> 3 != 0);
            let piece_code = self.pieces.get(i) & 0b0111;
            let piece_type = match piece_code {
                UNMOVED_ROOK => {
                    if seen_king[colour] {
                        *builder.castling_rights_mut().kingside_mut(colour) = sq;
                    } else {
                        *builder.castling_rights_mut().queenside_mut(colour) = sq;
                    }
                    PieceType::Rook
                }
                _ => PieceType::new(piece_code).unwrap(),
            };
            if piece_type == PieceType::King {
                seen_king[colour] = true;
            }
            builder.add_piece(sq, Piece::new(colour, piece_type));
        }

        *builder.ep_sq_mut() = Square::new_checked(self.stm_ep_square & 0b0111_1111).unwrap_or(Square::A1);
        *builder.turn_mut() = Colour::new(self.stm_ep_square >> 7 != 0);
        *builder.halfmove_clock_mut() = self.halfmove_clock;
        builder.set_fullmove_clock(self.fullmove_number.get());

        builder.regenerate_zobrist();
        builder.regenerate_threats();

        (builder, self.eval.get(), self.wdl, self.extra)
    }

    pub const fn as_bytes(self) -> [u8; std::mem::size_of::<Self>()] {
        unsafe { std::mem::transmute(self) }
    }

    pub const fn from_bytes(bytes: [u8; std::mem::size_of::<Self>()]) -> Self {
        unsafe { std::mem::transmute(bytes) }
    }
}

impl Board {
    pub fn pack(&self, eval: i16, wdl: u8, extra: u8) -> PackedBoard {
        PackedBoard::pack(self, eval, wdl, extra)
    }

    pub fn unpack(packed: &PackedBoard) -> (Self, i16, u8, u8) {
        packed.unpack()
    }
}

pub mod util {
    #[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
    #[repr(transparent)]
    pub struct U64Le(u64);

    impl U64Le {
        pub const fn new(v: u64) -> Self {
            Self(v.to_le())
        }

        pub const fn get(self) -> u64 {
            u64::from_le(self.0)
        }
    }

    #[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
    #[repr(transparent)]
    pub struct U16Le(u16);

    impl U16Le {
        pub const fn new(v: u16) -> Self {
            Self(v.to_le())
        }

        pub const fn get(self) -> u16 {
            u16::from_le(self.0)
        }
    }

    #[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
    #[repr(transparent)]
    pub struct I16Le(i16);

    impl I16Le {
        pub const fn new(v: i16) -> Self {
            Self(v.to_le())
        }

        pub const fn get(self) -> i16 {
            i16::from_le(self.0)
        }
    }

    #[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
    #[repr(transparent)]
    pub struct U4Array32([u8; 16]);

    impl U4Array32 {
        pub const fn get(&self, i: usize) -> u8 {
            (self.0[i / 2] >> ((i % 2) * 4)) & 0xF
        }

        pub fn set(&mut self, i: usize, v: u8) {
            debug_assert!(v < 0x10);
            self.0[i / 2] |= v << ((i % 2) * 4);
        }
    }
}
