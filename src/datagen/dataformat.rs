use std::path::Path;

use crate::{
    chess::{
        board::{Board, GameOutcome},
        chessmove::Move,
        piece::{Colour, PieceType},
    },
    tablebases::probe::WDL,
};

use self::marlinformat::{PackedBoard, util::I16Le};
use anyhow::{Context, anyhow};
use serde::{Deserialize, Serialize};

mod marlinformat;

/// The configuration for a filter that can be applied to a game during unpacking.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[allow(clippy::struct_field_names)]
#[serde(default)]
pub struct Filter {
    /// Filter out positions that have a ply count less than this value.
    min_ply: u32,
    /// Filter out positions that have fewer pieces on the board than this value.
    min_pieces: u32,
    /// Filter out positions that have an absolute evaluation above this value.
    max_eval: u32,
    /// Filter out games that have an absolute evaluation above this value in the starting position.
    max_opening_eval: u32,
    /// Filter out positions where a tactical move was made.
    filter_tactical: bool,
    /// Filter out positions that are in check.
    filter_check: bool,
    /// Filter out positions where a castling move was made.
    filter_castling: bool,
    /// Filter out positions where eval diverges from WDL by more than this value.
    max_eval_incorrectness: u32,
}

impl Default for Filter {
    fn default() -> Self {
        Self {
            min_ply: 16,
            min_pieces: 4,
            max_eval: 20_000,
            max_opening_eval: u32::MAX,
            filter_tactical: true,
            filter_check: true,
            filter_castling: false,
            max_eval_incorrectness: u32::MAX,
        }
    }
}

impl Filter {
    const UNRESTRICTED: Self = Self {
        min_ply: 0,
        min_pieces: 0,
        max_eval: u32::MAX,
        max_opening_eval: u32::MAX,
        filter_tactical: false,
        filter_check: false,
        filter_castling: false,
        max_eval_incorrectness: u32::MAX,
    };

    pub fn should_filter(&self, mv: Move, eval: i32, board: &Board, wdl: WDL) -> bool {
        if board.ply() < self.min_ply as usize {
            return true;
        }
        if eval.unsigned_abs() >= self.max_eval {
            return true;
        }
        if board.state.bbs.occupied().count() < self.min_pieces {
            return true;
        }
        if self.filter_tactical && board.is_tactical(mv) {
            return true;
        }
        if self.filter_check && board.in_check() {
            return true;
        }
        if self.filter_castling && mv.is_castle() {
            return true;
        }
        if self.max_eval_incorrectness != u32::MAX {
            // if the game was a draw, prune evals that are too far away from a draw.
            if wdl == WDL::Draw && eval.unsigned_abs() > self.max_eval_incorrectness {
                return true;
            }
            // otherwise, if the winner's eval drops too low, prune.
            let winner_pov_eval = if wdl == WDL::Win {
                // if white won, get white's eval.
                eval
            } else {
                // if black won, get black's eval.
                -eval
            };
            // clamp winner_pov_eval down to 0, check size.
            if winner_pov_eval.min(0).unsigned_abs() > self.max_eval_incorrectness {
                // too high for the losing side.
                return true;
            }
        }
        false
    }

    pub fn from_path(path: &Path) -> Result<Self, anyhow::Error> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read filter config file at {}", path.display()))?;
        toml::from_str(&text).with_context(|| {
            let default = toml::to_string_pretty(&Self::default()).unwrap();
            format!("Failed to parse filter config file at {} \nNote: the config file must be in TOML format. The default config looks like this: \n```\n{default}```", path.display())
        })
    }
}

/// A game annotated with evaluations starting from a potentially custom position, with support for efficent binary serialisation and deserialisation.
pub struct Game {
    /// The initial position of the self-play game.
    initial_position: marlinformat::PackedBoard,
    /// The moves played in the self-play game, along with the evaluation of the position in which they were played.
    moves: Vec<(Move, marlinformat::util::I16Le)>,
}

const SEQUENCE_ELEM_SIZE: usize =
    std::mem::size_of::<Move>() + std::mem::size_of::<marlinformat::util::I16Le>();
const NULL_TERMINATOR: [u8; SEQUENCE_ELEM_SIZE] = [0; SEQUENCE_ELEM_SIZE];

impl WDL {
    pub fn from_packed(packed: u8) -> Self {
        match packed {
            2 => Self::Win,
            1 => Self::Draw,
            0 => Self::Loss,
            _ => panic!("invalid WDL, expected 0, 1, or 2, got {packed}"),
        }
    }
}

impl Game {
    pub const MAX_SPLATTABLE_GAME_SIZE: usize = 512;

    pub fn new(initial_position: &Board) -> Self {
        Self {
            initial_position: initial_position.pack(0, 0, 0),
            moves: Vec::new(),
        }
    }

    pub fn initial_position(&self) -> Board {
        self.initial_position.unpack().0
    }

    pub fn moves(&self) -> impl Iterator<Item = Move> + '_ {
        self.moves.iter().map(|(mv, _)| *mv)
    }

    pub fn buffer(&self) -> &[(Move, marlinformat::util::I16Le)] {
        &self.moves
    }

    pub fn buffer_mut(&mut self) -> &mut [(Move, marlinformat::util::I16Le)] {
        &mut self.moves
    }

    pub fn set_outcome(&mut self, outcome: GameOutcome) {
        self.initial_position.set_outcome(outcome);
    }

    pub fn outcome(&self) -> WDL {
        let (_, _, wdl, _) = self.initial_position.unpack();
        WDL::from_packed(wdl)
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
        #[cfg(debug_assertions)]
        let (mut real_board, _, _, _) = initial_position.unpack();
        #[cfg(debug_assertions)]
        real_board.check_validity();
        // we allow the caller to give us a pre-allocated buffer as an optimisation
        let mut moves = buffer;
        moves.clear();
        loop {
            let mut buf = [0; SEQUENCE_ELEM_SIZE];
            reader.read_exact(&mut buf)?;
            if buf == NULL_TERMINATOR {
                break;
            }
            let mv = Move::from_raw(u16::from_le_bytes([buf[0], buf[1]]));
            let Some(mv) = mv else {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "parsed invalid move - move was null (the all-zeroes bitpattern)".to_string(),
                ));
            };
            if !mv.is_valid() || mv.from() == mv.to() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("parsed invalid move: {mv:?}"),
                ));
            }
            #[cfg(debug_assertions)]
            if !real_board.legal_moves().contains(&mv) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("parsed illegal move: {mv:?}"),
                ));
            }
            let eval = I16Le::new(i16::from_le_bytes([buf[2], buf[3]]));
            moves.push((mv, eval));
            #[cfg(debug_assertions)]
            real_board.make_move_simple(mv);
        }
        Ok(Self {
            initial_position,
            moves,
        })
    }

    /// Exposes a reference to each position and associated evaluation in the game sequentially, via a callback.
    pub fn visit_positions(&self, mut callback: impl FnMut(&Board, i32)) {
        let (mut board, _, _, _) = self.initial_position.unpack();
        for (mv, eval) in &self.moves {
            let eval = eval.get();
            callback(&board, i32::from(eval));
            board.make_move_simple(*mv);
        }
    }

    /// Internally counts how many positions would pass the filter in this game.
    pub fn filter_pass_count(&self, filter: &Filter) -> u64 {
        let mut cnt = 0;
        let (mut board, _, wdl, _) = self.initial_position.unpack();
        let outcome = WDL::from_packed(wdl);
        if let Some(opening_eval) = self.moves.first().map(|(_, e)| e.get())
            && u32::from(opening_eval.unsigned_abs()) > filter.max_opening_eval
        {
            return 0;
        }
        for (mv, eval) in &self.moves {
            let eval = eval.get();
            if !filter.should_filter(*mv, i32::from(eval), &board, outcome) {
                cnt += 1;
            }
            board.make_move_simple(*mv);
        }

        cnt
    }

    /// Converts the game into a sequence of marlinformat `PackedBoard` objects, yielding only those positions that pass the filter.
    pub fn splat_to_marlinformat(
        &self,
        mut callback: impl FnMut(marlinformat::PackedBoard) -> anyhow::Result<()>,
        filter: &Filter,
    ) -> anyhow::Result<()> {
        let (mut board, _, wdl, _) = self.initial_position.unpack();
        let outcome = WDL::from_packed(wdl);

        if let Some(opening_eval) = self.moves.first().map(|(_, e)| e.get())
            && u32::from(opening_eval.unsigned_abs()) > filter.max_opening_eval
        {
            return Ok(());
        }
        // record all the positions that pass the filter.
        for (mv, eval) in &self.moves {
            let eval = eval.get();
            if !filter.should_filter(*mv, i32::from(eval), &board, outcome) {
                callback(board.pack(eval, wdl, 0))?;
            }
            board.make_move_simple(*mv);
        }

        Ok(())
    }

    /// Converts the game into a sequence of bulletformat `ChessBoard` objects, yielding only those positions that pass the filter.
    pub fn splat_to_bulletformat(
        &self,
        mut callback: impl FnMut(bulletformat::ChessBoard) -> anyhow::Result<()>,
        filter: &Filter,
    ) -> anyhow::Result<()> {
        let (mut board, _, wdl, _) = self.initial_position.unpack();
        let outcome = WDL::from_packed(wdl);

        if let Some(opening_eval) = self.moves.first().map(|(_, e)| e.get())
            && u32::from(opening_eval.unsigned_abs()) > filter.max_opening_eval
        {
            return Ok(());
        }
        // record all the positions that pass the filter.
        for (mv, eval) in &self.moves {
            let eval = eval.get();
            if !filter.should_filter(*mv, i32::from(eval), &board, outcome) {
                let mut bbs = [0; 8];
                let piece_layout = &board.state.bbs;
                bbs[0] = piece_layout.colours[Colour::White].inner();
                bbs[1] = piece_layout.colours[Colour::Black].inner();
                bbs[2] = piece_layout.pieces[PieceType::Pawn].inner();
                bbs[3] = piece_layout.pieces[PieceType::Knight].inner();
                bbs[4] = piece_layout.pieces[PieceType::Bishop].inner();
                bbs[5] = piece_layout.pieces[PieceType::Rook].inner();
                bbs[6] = piece_layout.pieces[PieceType::Queen].inner();
                bbs[7] = piece_layout.pieces[PieceType::King].inner();
                callback(
                    bulletformat::ChessBoard::from_raw(
                        bbs,
                        (board.turn() != Colour::White).into(),
                        eval,
                        f32::from(wdl) / 2.0,
                    )
                    .map_err(|e| anyhow!(e))
                    .with_context(
                        || "Failed to convert raw components into bulletformat::ChessBoard.",
                    )?,
                )?;
            }
            board.make_move_simple(*mv);
        }

        Ok(())
    }

    /// Efficiency method that allows us to recover the move vector without allocating a new vector.
    pub fn into_move_buffer(self) -> Vec<(Move, marlinformat::util::I16Le)> {
        self.moves
    }
}

#[cfg(test)]
mod tests {
    use crate::chess::CHESS960;

    use super::*;

    use crate::chess::types::Square;

    #[test]
    #[ignore = "this test is slow, it checks all valid SFENs for roundtrip correctness"]
    fn roundtrip() {
        fn check_eq(lhs: &Board, rhs: &Board, msg: &str) {
            assert_eq!(
                lhs.state.bbs.pieces[PieceType::Pawn],
                rhs.state.bbs.pieces[PieceType::Pawn],
                "pawn square-sets {msg}"
            );
            assert_eq!(
                lhs.state.bbs.pieces[PieceType::Knight],
                rhs.state.bbs.pieces[PieceType::Knight],
                "knight square-sets {msg}"
            );
            assert_eq!(
                lhs.state.bbs.pieces[PieceType::Bishop],
                rhs.state.bbs.pieces[PieceType::Bishop],
                "bishop square-sets {msg}"
            );
            assert_eq!(
                lhs.state.bbs.pieces[PieceType::Rook],
                rhs.state.bbs.pieces[PieceType::Rook],
                "rook square-sets {msg}"
            );
            assert_eq!(
                lhs.state.bbs.pieces[PieceType::Queen],
                rhs.state.bbs.pieces[PieceType::Queen],
                "queen square-sets {msg}"
            );
            assert_eq!(
                lhs.state.bbs.pieces[PieceType::King],
                rhs.state.bbs.pieces[PieceType::King],
                "king square-sets {msg}"
            );
            assert_eq!(
                lhs.state.bbs.colours[Colour::White],
                rhs.state.bbs.colours[Colour::White],
                "white square-sets {msg}"
            );
            assert_eq!(
                lhs.state.bbs.colours[Colour::Black],
                rhs.state.bbs.colours[Colour::Black],
                "black square-sets {msg}"
            );
            for sq in Square::all() {
                assert_eq!(
                    lhs.state.mailbox[sq], rhs.state.mailbox[sq],
                    ".state.mailbox[{sq:?}] {msg}"
                );
            }
            assert_eq!(lhs.turn(), rhs.turn(), "side {msg}");
            assert_eq!(lhs.ep_sq(), rhs.ep_sq(), "ep_sq {msg}");
            assert_eq!(
                lhs.castling_rights(),
                rhs.castling_rights(),
                "castle_perm {msg}"
            );
            assert_eq!(
                lhs.fifty_move_counter(),
                rhs.fifty_move_counter(),
                "fifty_move_counter {msg}"
            );
            assert_eq!(lhs.ply(), rhs.ply(), "ply {msg}");
            assert_eq!(lhs.state.keys, rhs.state.keys, "key {msg}");
            assert_eq!(lhs.state.threats, rhs.state.threats, "threats {msg}");
            assert_eq!(lhs.height(), rhs.height(), "height {msg}");
        }
        CHESS960.store(true, std::sync::atomic::Ordering::SeqCst);
        // Grab `valid.sfens` from `cozy-chess` to run test
        for sfen in include_str!("valid.sfens").lines() {
            let board = Board::from_fen(sfen).unwrap();
            let packed = marlinformat::PackedBoard::pack(&board, 0, 0, 0);
            let (unpacked, _, _, _) = packed.unpack();
            check_eq(&board, &unpacked, sfen);
        }
    }

    #[test]
    fn game_roundtrip() {
        let mut game = Game::new(&Board::startpos());
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
        let mut game = Game::new(&Board::startpos());
        game.add_move(Move::new(Square::E2, Square::E4), 3);
        game.add_move(Move::new(Square::E7, Square::E5), -314);
        game.add_move(Move::new(Square::G1, Square::F3), 200);

        let mut boards = Vec::new();
        let filter = Filter::UNRESTRICTED;
        game.splat_to_marlinformat(
            |board| {
                boards.push(board);
                Ok(())
            },
            &filter,
        )
        .unwrap();
        assert_eq!(boards.len(), 3);
        let mut check_board = Board::startpos();
        assert_eq!(boards[0].unpack().0.to_string(), check_board.to_string());
        assert_eq!(boards[0].unpack().1, 3);
        let e4 = Move::new(Square::E2, Square::E4);
        assert!(check_board.is_pseudo_legal(e4));
        assert!(check_board.is_legal(e4));
        check_board.make_move_simple(e4);
        assert_eq!(boards[1].unpack().0.to_string(), check_board.to_string());
        assert_eq!(boards[1].unpack().1, -314);
        let e5 = Move::new(Square::E7, Square::E5);
        assert!(check_board.is_pseudo_legal(e5));
        assert!(check_board.is_legal(e5));
        check_board.make_move_simple(e5);
        assert_eq!(boards[2].unpack().0.to_string(), check_board.to_string());
        assert_eq!(boards[2].unpack().1, 200);
    }
}
