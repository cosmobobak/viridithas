use crate::{
    board::{Board, GameOutcome},
    chessmove::Move,
    piece::{Colour, Piece, PieceType},
    squareset::SquareSet,
    util::{Rank, Square},
};

use self::util::I16Le;

const UNMOVED_ROOK: u8 = PieceType::NONE.inner();

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
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
            let piece = board.piece_at(sq);
            let piece_type = piece.piece_type();
            let colour = piece.colour();

            let mut piece_code = piece_type.inner();
            let rank1 = if colour == Colour::WHITE { Rank::RANK_1 } else { Rank::RANK_8 };
            if piece_type == PieceType::ROOK && sq.rank() == rank1 {
                let castling_sq = if board.king_sq(colour) < sq {
                    board.castling_rights().kingside(colour)
                } else {
                    board.castling_rights().queenside(colour)
                };
                let castling_file =
                    if castling_sq == Square::NO_SQUARE { None } else { Some(castling_sq.file()) };
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
            stm_ep_square: (board.turn().inner()) << 7 | board.ep_sq().inner(),
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
            let colour = Colour::new(self.pieces.get(i) >> 3);
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

        *builder.ep_sq_mut() = Square::new(self.stm_ep_square & 0b0111_1111);
        *builder.turn_mut() = Colour::new(self.stm_ep_square >> 7);
        *builder.halfmove_clock_mut() = self.halfmove_clock;
        builder.set_fullmove_clock(self.fullmove_number.get());

        builder.regenerate_zobrist();
        builder.regenerate_threats();

        (builder, self.eval.get(), self.wdl, self.extra)
    }

    pub const fn as_bytes(&self) -> &[u8; std::mem::size_of::<Self>()] {
        unsafe { &*(self as *const Self).cast::<[u8; std::mem::size_of::<Self>()]>() }
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

mod util {
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

pub struct Game {
    /// The initial position of the self-play game.
    initial_position: PackedBoard,
    /// The moves played in the self-play game, along with the evaluation of the position in which they were played.
    moves: Vec<(Move, util::I16Le)>,
}

const SEQUENCE_ELEM_SIZE: usize = std::mem::size_of::<Move>() + std::mem::size_of::<util::I16Le>();
const NULL_TERMINATOR: [u8; SEQUENCE_ELEM_SIZE] = [0; SEQUENCE_ELEM_SIZE];

impl Game {
    pub fn new(initial_position: &Board) -> Self {
        Self { initial_position: initial_position.pack(0, 0, 0), moves: Vec::new() }
    }

    pub fn set_outcome(&mut self, outcome: GameOutcome) {
        self.initial_position.set_outcome(outcome);
    }

    pub fn add_move(&mut self, mv: Move, eval: i16) {
        self.moves.push((mv, util::I16Le::new(eval)));
    }

    pub fn len(&self) -> usize {
        self.moves.len()
    }

    /// Serialises the game into a byte stream.
    pub fn serialise_into(&self, writer: &mut impl std::io::Write) -> std::io::Result<()> {
        let starting_position = unsafe {
            std::slice::from_raw_parts(
                std::ptr::addr_of!(self.initial_position).cast::<u8>(),
                std::mem::size_of::<PackedBoard>(),
            )
        };
        writer.write_all(starting_position)?;
        for (mv, eval) in &self.moves {
            let mv = unsafe {
                std::slice::from_raw_parts(
                    (mv as *const Move).cast::<u8>(),
                    std::mem::size_of::<Move>(),
                )
            };
            writer.write_all(mv)?;
            let eval = unsafe {
                std::slice::from_raw_parts(
                    (eval as *const util::I16Le).cast::<u8>(),
                    std::mem::size_of::<util::I16Le>(),
                )
            };
            writer.write_all(eval)?;
        }
        writer.write_all(&NULL_TERMINATOR)?;
        Ok(())
    }

    /// Deserialises a game from a byte stream.
    pub fn deserialise_from(reader: &mut impl std::io::BufRead, buffer: Vec<(Move, I16Le)>) -> std::io::Result<Self> {
        let mut initial_position = [0; std::mem::size_of::<PackedBoard>()];
        reader.read_exact(&mut initial_position)?;
        let initial_position = unsafe { std::mem::transmute::<_, PackedBoard>(initial_position) };
        // we allow the caller to give us a pre-allocated buffer as an optimisation
        let mut moves = buffer;
        moves.clear();
        loop {
            let mut buf = [0; SEQUENCE_ELEM_SIZE];
            reader.read_exact(&mut buf)?;
            if buf == NULL_TERMINATOR {
                break;
            }
            let mv = [buf[0], buf[1]];
            let mv = unsafe { std::mem::transmute::<_, Move>(mv) };
            let eval = [buf[2], buf[3]];
            let eval = unsafe { std::mem::transmute::<_, util::I16Le>(eval) };
            moves.push((mv, eval));
        }
        Ok(Self { initial_position, moves })
    }

    /// Converts the game into a sequence of `PackedBoard` objects, yielding only those positions that pass the filter.
    pub fn splat(&self, mut callback: impl FnMut(PackedBoard), filter: impl Fn(Move, i32, &Board) -> bool) {
        let (mut board, _, wdl, _) = self.initial_position.unpack();
        for (mv, eval) in &self.moves {
            let eval = eval.get();
            if filter(*mv, i32::from(eval), &board) {
                callback(board.pack(eval, wdl, 0));
            }
            board.make_move_simple(*mv);
        }
    }

    /// Efficiency method that allows us to recover the move vector without allocating a new vector.
    pub fn into_move_buffer(self) -> Vec<(Move, I16Le)> {
        self.moves
    }
}

#[cfg(test)]
mod tests {
    use crate::uci::CHESS960;

    use super::*;

    #[test]
    #[ignore]
    fn roundtrip() {
        CHESS960.store(true, std::sync::atomic::Ordering::SeqCst);
        // Grab `valid.sfens` from `cozy-chess` to run test
        for sfen in include_str!("valid.sfens").lines() {
            let board = Board::from_fen(sfen).unwrap();
            let packed = PackedBoard::pack(&board, 0, 0, 0);
            let (unpacked, _, _, _) = packed.unpack();
            crate::board::check_eq(&board, &unpacked, sfen);
        }
    }

    #[test]
    fn game_roundtrip() {
        let mut game = Game::new(&Board::default());
        game.add_move(Move::new(Square::E2, Square::E4), 0);
        game.add_move(Move::new(Square::E7, Square::E5), -314);
        game.add_move(Move::new(Square::G1, Square::F3), 200);

        let mut buf = Vec::new();
        game.serialise_into(&mut buf).unwrap();
        let game2 = Game::deserialise_from(&mut buf.as_slice(), Vec::new()).unwrap();
        assert_eq!(game.initial_position, game2.initial_position);
        assert_eq!(game.moves, game2.moves);
    }

    #[test]
    fn splat() {
        let mut game = Game::new(&Board::default());
        game.add_move(Move::new(Square::E2, Square::E4), 3);
        game.add_move(Move::new(Square::E7, Square::E5), -314);
        game.add_move(Move::new(Square::G1, Square::F3), 200);

        let mut boards = Vec::new();
        game.splat(|board| boards.push(board), |_, _, _| true);
        assert_eq!(boards.len(), 3);
        let mut check_board = Board::default();
        assert_eq!(boards[0].unpack().0.fen(), check_board.fen());
        assert_eq!(boards[0].unpack().1, 3);
        assert!(check_board.make_move_simple(Move::new(Square::E2, Square::E4)));
        assert_eq!(boards[1].unpack().0.fen(), check_board.fen());
        assert_eq!(boards[1].unpack().1, -314);
        assert!(check_board.make_move_simple(Move::new(Square::E7, Square::E5)));
        assert_eq!(boards[2].unpack().0.fen(), check_board.fen());
        assert_eq!(boards[2].unpack().1, 200);
    }
}
