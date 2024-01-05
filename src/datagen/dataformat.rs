use crate::{
    board::{Board, GameOutcome},
    chessmove::Move,
    piece::{Colour, PieceType},
};

use self::marlinformat::{util::I16Le, PackedBoard};

mod marlinformat;

pub struct Game {
    /// The initial position of the self-play game.
    initial_position: marlinformat::PackedBoard,
    /// The moves played in the self-play game, along with the evaluation of the position in which they were played.
    moves: Vec<(Move, marlinformat::util::I16Le)>,
}

const SEQUENCE_ELEM_SIZE: usize =
    std::mem::size_of::<Move>() + std::mem::size_of::<marlinformat::util::I16Le>();
const NULL_TERMINATOR: [u8; SEQUENCE_ELEM_SIZE] = [0; SEQUENCE_ELEM_SIZE];

impl Game {
    pub fn new(initial_position: &Board) -> Self {
        Self { initial_position: initial_position.pack(0, 0, 0), moves: Vec::new() }
    }

    pub fn set_outcome(&mut self, outcome: GameOutcome) {
        self.initial_position.set_outcome(outcome);
    }

    pub fn add_move(&mut self, mv: Move, eval: i16) {
        self.moves.push((mv, marlinformat::util::I16Le::new(eval)));
    }

    pub fn len(&self) -> usize {
        self.moves.len()
    }

    /// Serialises the game into a byte stream.
    pub fn serialise_into(&self, writer: &mut impl std::io::Write) -> std::io::Result<()> {
        writer.write_all(&self.initial_position.as_bytes())?;
        for (mv, eval) in &self.moves {
            writer.write_all(&mv.inner().to_le_bytes())?;
            writer.write_all(&eval.get().to_le_bytes())?;
        }
        writer.write_all(&NULL_TERMINATOR)?;
        Ok(())
    }

    /// Deserialises a game from a byte stream.
    pub fn deserialise_from(
        reader: &mut impl std::io::BufRead,
        buffer: Vec<(Move, marlinformat::util::I16Le)>,
    ) -> std::io::Result<Self> {
        let mut initial_position = [0; std::mem::size_of::<marlinformat::PackedBoard>()];
        reader.read_exact(&mut initial_position)?;
        let initial_position = PackedBoard::from_bytes(initial_position);
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
            if !mv.is_valid() || mv.from() == mv.to() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("invalid move: {mv:?}"),
                ));
            }
            let eval = [buf[2], buf[3]];
            let eval = i16::from_le_bytes(eval);
            let eval = I16Le::new(eval);
            moves.push((mv, eval));
        }
        Ok(Self { initial_position, moves })
    }

    /// Converts the game into a sequence of marlinformat `PackedBoard` objects, yielding only those positions that pass the filter.
    pub fn splat_to_marlinformat(
        &self,
        mut callback: impl FnMut(marlinformat::PackedBoard),
        filter: impl Fn(Move, i32, &Board) -> bool,
    ) {
        let (mut board, _, wdl, _) = self.initial_position.unpack();
        for (mv, eval) in &self.moves {
            let eval = eval.get();
            if filter(*mv, i32::from(eval), &board) {
                callback(board.pack(eval, wdl, 0));
            }
            board.make_move_simple(*mv);
        }
    }

    /// Converts the game into a sequence of bulletformat `ChessBoard` objects, yielding only those positions that pass the filter.
    pub fn splat_to_bulletformat(
        &self,
        mut callback: impl FnMut(bulletformat::ChessBoard),
        filter: impl Fn(Move, i32, &Board) -> bool,
    ) {
        let (mut board, _, wdl, _) = self.initial_position.unpack();
        for (mv, eval) in &self.moves {
            let eval = eval.get();
            if filter(*mv, i32::from(eval), &board) {
                let mut bbs = [0; 8];
                let bitboard = &board.pieces;
                bbs[0] = bitboard.occupied_co(Colour::WHITE).inner();
                bbs[1] = bitboard.occupied_co(Colour::BLACK).inner();
                bbs[2] = bitboard.of_type(PieceType::PAWN).inner();
                bbs[3] = bitboard.of_type(PieceType::KNIGHT).inner();
                bbs[4] = bitboard.of_type(PieceType::BISHOP).inner();
                bbs[5] = bitboard.of_type(PieceType::ROOK).inner();
                bbs[6] = bitboard.of_type(PieceType::QUEEN).inner();
                bbs[7] = bitboard.of_type(PieceType::KING).inner();
                callback(
                    bulletformat::ChessBoard::from_raw(
                        bbs,
                        (board.turn() != Colour::WHITE).into(),
                        eval,
                        f32::from(wdl) / 2.0,
                    )
                    .unwrap(),
                );
            }
            board.make_move_simple(*mv);
        }
    }

    /// Efficiency method that allows us to recover the move vector without allocating a new vector.
    pub fn into_move_buffer(self) -> Vec<(Move, marlinformat::util::I16Le)> {
        self.moves
    }
}

#[cfg(test)]
mod tests {
    use crate::uci::CHESS960;

    use super::*;

    use crate::util::Square;

    #[test]
    #[ignore]
    fn roundtrip() {
        CHESS960.store(true, std::sync::atomic::Ordering::SeqCst);
        // Grab `valid.sfens` from `cozy-chess` to run test
        for sfen in include_str!("valid.sfens").lines() {
            let board = Board::from_fen(sfen).unwrap();
            let packed = marlinformat::PackedBoard::pack(&board, 0, 0, 0);
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
        game.splat_to_marlinformat(|board| boards.push(board), |_, _, _| true);
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
