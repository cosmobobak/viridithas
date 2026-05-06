use std::{num::NonZeroUsize, str::SplitWhitespace};

use arrayvec::ArrayVec;

use crate::{
    chess::{
        piece::{Colour, Piece, PieceType},
        piecelayout::PieceLayout,
        squareset::SquareSet,
        types::{CastlingRights, File, Rank, Square},
    },
    errors::FenParseError,
};

/// A parsed FEN representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fen {
    pub board: PieceLayout,
    pub turn: Colour,
    pub castling: CastlingRights,
    pub ep: Option<Square>,
    pub halfmove: u8,
    pub fullmove: NonZeroUsize,
}

impl Fen {
    const DEFAULT_FULL_MOVE: NonZeroUsize = NonZeroUsize::new(1).unwrap();

    /// Parse a FEN string in strict mode.
    /// All 6 fields must be present, and no extra tokens are allowed.
    pub fn parse(fen: &str) -> Result<Self, FenParseError> {
        let mut tokens = fen.split_whitespace();
        let result = Self::parse_inner(&mut tokens, true)?;
        // In strict mode, no extra tokens allowed.
        if tokens.next().is_some() {
            return Err(FenParseError::ExtraTokens);
        }
        Ok(result)
    }

    /// Parse a FEN string in relaxed mode.
    /// Missing fields after the board are defaulted to: w, -, -, 0, 1
    /// Extra tokens after the fullmove counter are permitted but ignored.
    pub fn parse_relaxed(fen: &str) -> Result<Self, FenParseError> {
        let mut tokens = fen.split_whitespace();
        Self::parse_inner(&mut tokens, false)
    }

    fn parse_inner(tokens: &mut SplitWhitespace<'_>, strict: bool) -> Result<Self, FenParseError> {
        // Field #1: Piece placement
        let board_str = tokens.next().ok_or(FenParseError::MissingBoard)?;
        let board = Self::parse_board(board_str)?;

        // Field #2: Active colour
        let turn = match tokens.next() {
            Some(s) if strict => Self::parse_turn(s)?,
            Some(s) => Self::parse_turn(s).unwrap_or(Colour::White),
            None if strict => return Err(FenParseError::MissingSide),
            None => Colour::White,
        };

        // At this point we can test if we're illegally checking:
        if board.sq_attacked(board.king_sq(!turn), turn) {
            return Err(FenParseError::WaitingInCheck);
        }

        // Field #3: Castling availability
        let castling = match tokens.next() {
            Some(s) if strict => Self::parse_castling(s, &board)?,
            Some(s) => Self::parse_castling(s, &board).unwrap_or_default(),
            None if strict => return Err(FenParseError::MissingCastling),
            None => CastlingRights::default(),
        };

        // Field #4: En passant target square
        let ep = match tokens.next() {
            Some(s) if strict => Self::parse_ep(s, turn)?,
            Some(s) => Self::parse_ep(s, turn).unwrap_or(None),
            None if strict => return Err(FenParseError::MissingEnPassant),
            None => None,
        };

        // Field #5: Halfmove clock
        let halfmove = match tokens.next() {
            Some(s) if strict => Self::parse_halfmove(s)?,
            Some(s) => Self::parse_halfmove(s).unwrap_or(0),
            None if strict => return Err(FenParseError::MissingHalfmoveClock),
            None => 0,
        };

        // Field #6: Fullmove number
        let fullmove = match tokens.next() {
            Some(s) if strict => Self::parse_fullmove(s)?,
            Some(s) => Self::parse_fullmove(s).unwrap_or(Self::DEFAULT_FULL_MOVE),
            None if strict => return Err(FenParseError::MissingFullmoveNumber),
            None => Self::DEFAULT_FULL_MOVE,
        };

        Ok(Self {
            board,
            turn,
            castling,
            ep,
            halfmove,
            fullmove,
        })
    }

    fn parse_board(board_str: &str) -> Result<PieceLayout, FenParseError> {
        let mut layout = PieceLayout::default();
        let mut rank = Rank::Eight;

        let mut ranks = ArrayVec::<&str, 8>::new();

        let mut board_parts = board_str.split('/');

        while let Some(rank) = board_parts.next() {
            if ranks.try_push(rank).is_err() {
                // 8 successfully parse, plus one now, plus the rest.
                return Err(FenParseError::BoardSegments(8 + 1 + board_parts.count()));
            }
        }

        if ranks.len() != 8 {
            return Err(FenParseError::BoardSegments(ranks.len()));
        }

        for (rank_idx, rank_str) in ranks.iter().enumerate() {
            let mut file = File::A;
            let mut squares_in_rank = 0;
            let mut prev_was_digit = false;

            for c in rank_str.chars() {
                match c {
                    '1'..='8' => {
                        if prev_was_digit {
                            return Err(FenParseError::AdjacentDigits);
                        }
                        prev_was_digit = true;
                        let count = c as u8 - b'0';
                        squares_in_rank += count;
                        if squares_in_rank > 8 {
                            return Err(FenParseError::BadSquaresInSegment);
                        }
                        // Advance file by count
                        for _ in 0..count {
                            file = file.add(1).unwrap_or(File::A);
                        }
                    }
                    'P' | 'R' | 'N' | 'B' | 'Q' | 'K' | 'p' | 'r' | 'n' | 'b' | 'q' | 'k' => {
                        prev_was_digit = false;
                        squares_in_rank += 1;
                        if squares_in_rank > 8 {
                            return Err(FenParseError::BadSquaresInSegment);
                        }
                        let piece = Self::char_to_piece(c);
                        let sq = Square::from_rank_file(rank, file);
                        layout.set_piece_at(sq, piece);
                        file = file.add(1).unwrap_or(File::A);
                    }
                    _ => return Err(FenParseError::UnexpectedCharacter(c)),
                }
            }

            if squares_in_rank != 8 {
                return Err(FenParseError::BadSquaresInSegment);
            }

            // Move to next rank (going from 8 down to 1)
            if rank_idx < 7 {
                rank = rank.sub(1).ok_or(FenParseError::BadSquaresInSegment)?;
            }
        }

        // general correctness validation - misses a lot, but does some nice things.

        // pawns are on sensible squares
        if layout.pieces[PieceType::Pawn] & SquareSet::BACK_RANKS != SquareSet::EMPTY {
            return Err(FenParseError::PawnsOnBackranks);
        }

        // check king counts
        for colour in Colour::all() {
            match (layout.pieces[PieceType::King] & layout.colours[colour]).count() {
                0 => return Err(FenParseError::MissingKing { colour }),
                2.. => return Err(FenParseError::DuplicateKings { colour }),
                1 => (),
            }
        }

        Ok(layout)
    }

    fn char_to_piece(c: char) -> Piece {
        match c {
            'P' => Piece::WP,
            'R' => Piece::WR,
            'N' => Piece::WN,
            'B' => Piece::WB,
            'Q' => Piece::WQ,
            'K' => Piece::WK,
            'p' => Piece::BP,
            'r' => Piece::BR,
            'n' => Piece::BN,
            'b' => Piece::BB,
            'q' => Piece::BQ,
            'k' => Piece::BK,
            _ => panic!("char_to_piece called with invalid char"),
        }
    }

    fn parse_turn(s: &str) -> Result<Colour, FenParseError> {
        match s {
            "w" => Ok(Colour::White),
            "b" => Ok(Colour::Black),
            _ => Err(FenParseError::InvalidSide(s.to_string())),
        }
    }

    fn parse_castling(s: &str, board: &PieceLayout) -> Result<CastlingRights, FenParseError> {
        if s == "-" {
            return Ok(CastlingRights::default());
        }

        let mut rights = CastlingRights::default();

        // Find king positions
        let kings = board.pieces[PieceType::King];
        let white_king_sq = (kings & board.colours[Colour::White]).first().unwrap();
        let black_king_sq = (kings & board.colours[Colour::Black]).first().unwrap();

        for c in s.chars() {
            match c {
                // Standard notation (assumes rooks on A/H files)
                'K' => {
                    rights.set_kingside(Colour::White, File::H);
                }
                'Q' => {
                    rights.set_queenside(Colour::White, File::A);
                }
                'k' => {
                    rights.set_kingside(Colour::Black, File::H);
                }
                'q' => {
                    rights.set_queenside(Colour::Black, File::A);
                }
                // X-FEN / Shredder-FEN: uppercase file letter for white
                'A'..='H' => {
                    let file = File::from_index(c as u8 - b'A')
                        .ok_or_else(|| FenParseError::InvalidCastling(s.to_string()))?;

                    let king_sq = white_king_sq;

                    if king_sq.rank() != Rank::One {
                        return Err(FenParseError::KingNotOnBackRank {
                            colour: "white",
                            castling: s.to_string(),
                        });
                    }

                    let king_file = king_sq.file();
                    if file == king_file {
                        return Err(FenParseError::KingOnCastlingFile {
                            colour: "white",
                            file: format!("{king_file:?}"),
                            castling: s.to_string(),
                        });
                    }

                    if file > king_file {
                        rights.set_kingside(Colour::White, file);
                    } else {
                        rights.set_queenside(Colour::White, file);
                    }
                }
                // X-FEN / Shredder-FEN: lowercase file letter for black
                'a'..='h' => {
                    let file = File::from_index(c as u8 - b'a')
                        .ok_or_else(|| FenParseError::InvalidCastling(s.to_string()))?;

                    let king_sq = black_king_sq;

                    if king_sq.rank() != Rank::Eight {
                        return Err(FenParseError::KingNotOnBackRank {
                            colour: "black",
                            castling: s.to_string(),
                        });
                    }

                    let king_file = king_sq.file();
                    if file == king_file {
                        return Err(FenParseError::KingOnCastlingFile {
                            colour: "black",
                            file: format!("{king_file:?}"),
                            castling: s.to_string(),
                        });
                    }

                    if file > king_file {
                        rights.set_kingside(Colour::Black, file);
                    } else {
                        rights.set_queenside(Colour::Black, file);
                    }
                }
                _ => return Err(FenParseError::InvalidCastling(s.to_string())),
            }
        }

        Ok(rights)
    }

    fn parse_ep(s: &str, turn: Colour) -> Result<Option<Square>, FenParseError> {
        if s == "-" {
            return Ok(None);
        }

        if s.len() != 2 {
            return Err(FenParseError::InvalidEnPassant(s.to_string()));
        }

        let mut chars = s.chars();
        let file_char = chars.next().unwrap();
        let rank_char = chars.next().unwrap();

        // File must be lowercase a-h
        if !file_char.is_ascii_lowercase() {
            return Err(FenParseError::InvalidEnPassant(s.to_string()));
        }

        let file = File::from_index(file_char as u8 - b'a')
            .ok_or_else(|| FenParseError::InvalidEnPassant(s.to_string()))?;

        let rank = Rank::from_index(rank_char as u8 - b'1')
            .ok_or_else(|| FenParseError::InvalidEnPassant(s.to_string()))?;

        // Validate rank based on side to move:
        // If white to move, ep square must be on rank 6 (black pawn just moved)
        // If black to move, ep square must be on rank 3 (white pawn just moved)
        let expected_rank = match turn {
            Colour::White => Rank::Six,
            Colour::Black => Rank::Three,
        };

        if rank != expected_rank {
            return Err(FenParseError::InvalidEnPassantRank {
                square: s.to_string(),
                expected: expected_rank,
                got: rank,
            });
        }

        Ok(Some(Square::from_rank_file(rank, file)))
    }

    fn parse_halfmove(s: &str) -> Result<u8, FenParseError> {
        let value: u8 = s
            .parse()
            .map_err(|_| FenParseError::InvalidHalfmoveClock(s.to_string()))?;

        if value > 100 {
            return Err(FenParseError::HalfmoveClockTooLarge(value));
        }

        Ok(value)
    }

    fn parse_fullmove(s: &str) -> Result<NonZeroUsize, FenParseError> {
        let value: usize = s
            .parse()
            .map_err(|_| FenParseError::InvalidFullmoveNumber(s.to_string()))?;

        NonZeroUsize::new(value).ok_or(FenParseError::FullmoveNumberZero)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    #[test]
    fn parse_startpos() {
        let fen = Fen::parse(STARTPOS).unwrap();
        assert_eq!(fen.turn, Colour::White);
        assert_eq!(fen.halfmove, 0);
        assert_eq!(fen.fullmove.get(), 1);
        assert!(fen.ep.is_none());
    }

    #[test]
    fn parse_relaxed_board_only() {
        let fen = Fen::parse_relaxed("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR").unwrap();
        assert_eq!(fen.turn, Colour::White);
        assert_eq!(fen.halfmove, 0);
        assert_eq!(fen.fullmove.get(), 1);
    }

    #[test]
    fn parse_bad_segments() {
        let err = Fen::parse_relaxed("rnbqkbnr/pppppppp/8/8/8/8").unwrap_err();
        assert_eq!(err, FenParseError::BoardSegments(6));
        let err =
            Fen::parse_relaxed("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR/PPPPPPPP/RNBQKBNR")
                .unwrap_err();
        assert_eq!(err, FenParseError::BoardSegments(10));
        let err =
            Fen::parse_relaxed("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR/PPPPPPPP/RNBQKBNR/8")
                .unwrap_err();
        assert_eq!(err, FenParseError::BoardSegments(11));
    }

    #[test]
    fn reject_adjacent_digits() {
        let result = Fen::parse("rnbqkbnr/pppppppp/44/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        assert!(matches!(result, Err(FenParseError::AdjacentDigits)));
    }

    #[test]
    fn reject_uppercase_side() {
        let result = Fen::parse("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR W KQkq - 0 1");
        assert!(matches!(result, Err(FenParseError::InvalidSide(_))));
    }

    #[test]
    fn reject_invalid_ep_rank() {
        // e4 is not a valid ep square (should be e3 or e6)
        let result = Fen::parse("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e4 0 1");
        assert!(matches!(
            result,
            Err(FenParseError::InvalidEnPassantRank { .. })
        ));
    }

    #[test]
    fn accept_valid_ep_square() {
        let fen =
            Fen::parse("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1").unwrap();
        assert_eq!(fen.ep, Some(Square::from_rank_file(Rank::Three, File::E)));
    }

    #[test]
    fn reject_halfmove_over_100() {
        let result = Fen::parse("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 101 1");
        assert!(matches!(
            result,
            Err(FenParseError::HalfmoveClockTooLarge(101))
        ));
    }

    #[test]
    fn reject_fullmove_zero() {
        let result = Fen::parse("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 0");
        assert!(matches!(result, Err(FenParseError::FullmoveNumberZero)));
    }

    #[test]
    fn reject_extra_tokens_strict() {
        let result = Fen::parse("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 extra");
        assert!(matches!(result, Err(FenParseError::ExtraTokens)));
    }

    #[test]
    fn allow_extra_tokens_relaxed() {
        let fen = Fen::parse_relaxed(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 extra tokens",
        )
        .unwrap();
        assert_eq!(fen.fullmove.get(), 1);
    }

    #[test]
    fn parse_chess960_shredder_fen() {
        // Chess960 position with rooks on B and G files
        let fen = Fen::parse("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w BGbg - 0 1");
        // This should fail because standard position has king on E, and B < E, G > E
        // so B is queenside, G is kingside
        assert!(fen.is_ok());
    }

    #[test]
    fn relaxed_defaults_invalid_tokens() {
        // Invalid tokens for castling, ep, halfmove - should all default
        let fen = Fen::parse_relaxed("4k3/8/8/8/8/8/8/4K3 b blah foo bar").unwrap();
        assert_eq!(fen.turn, Colour::Black);
        assert_eq!(fen.castling, CastlingRights::default()); // defaulted from "blah"
        assert!(fen.ep.is_none()); // defaulted from "foo"
        assert_eq!(fen.halfmove, 0); // defaulted from "bar"
        assert_eq!(fen.fullmove.get(), 1); // defaulted (missing)
    }

    #[test]
    fn relaxed_partial_valid_tokens() {
        // Valid side, invalid castling, valid ep would fail (wrong rank for white),
        // so it defaults, then valid halfmove
        let fen = Fen::parse_relaxed("4k3/8/8/8/4P3/8/8/4K3 w invalid e3 5").unwrap();
        assert_eq!(fen.turn, Colour::White);
        assert_eq!(fen.castling, CastlingRights::default()); // defaulted from "invalid"
        // "e3" is invalid ep for white-to-move (should be rank 6), so defaults to None
        assert!(fen.ep.is_none());
        // "5" parsed as halfmove
        assert_eq!(fen.halfmove, 5);
        assert_eq!(fen.fullmove.get(), 1); // defaulted (missing)
    }

    /// Helper to parse all positions from an EPD file.
    /// EPD format: FEN fields followed by operations (bm, id, etc.)
    /// We use `parse_relaxed` since EPDs have trailing content.
    fn parse_epd_file(path: &str) -> (usize, Vec<String>) {
        let content = std::fs::read_to_string(path).unwrap();
        let mut success = 0;
        let mut failures = Vec::new();

        for (line_num, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            match Fen::parse_relaxed(line) {
                Ok(_) => success += 1,
                Err(e) => failures.push(format!("line {}: {} -- {}", line_num + 1, e, line)),
            }
        }

        (success, failures)
    }

    #[test]
    fn epd_perftsuite() {
        let (success, failures) = parse_epd_file("assets/epds/perftsuite.epd");
        assert!(failures.is_empty(), "failures:\n{}", failures.join("\n"));
        assert_eq!(success, 126);
    }

    #[test]
    fn epd_frcperftsuite() {
        let (success, failures) = parse_epd_file("assets/epds/frcperftsuite.epd");
        assert!(failures.is_empty(), "failures:\n{}", failures.join("\n"));
        assert_eq!(success, 960);
    }

    #[test]
    fn epd_wac() {
        let (success, failures) = parse_epd_file("assets/epds/wac.epd");
        assert!(failures.is_empty(), "failures:\n{}", failures.join("\n"));
        assert_eq!(success, 300);
    }

    #[test]
    fn epd_arasan21() {
        let (success, failures) = parse_epd_file("assets/epds/arasan21.epd");
        assert!(failures.is_empty(), "failures:\n{}", failures.join("\n"));
        assert_eq!(success, 200);
    }

    #[test]
    fn epd_tbtest() {
        let (success, failures) = parse_epd_file("assets/epds/tbtest.epd");
        assert!(failures.is_empty(), "failures:\n{}", failures.join("\n"));
        assert_eq!(success, 434);
    }

    #[test]
    fn epd_all_files() {
        // Stress test: parse all EPD files
        let epd_files = [
            "assets/epds/arasan21.epd",
            "assets/epds/bt2630.epd",
            "assets/epds/ecmgcp.epd",
            "assets/epds/eet.epd",
            "assets/epds/frcperftsuite.epd",
            "assets/epds/hardzugs.epd",
            "assets/epds/iq4.epd",
            "assets/epds/lapuce2.epd",
            "assets/epds/perftsuite.epd",
            "assets/epds/pet.epd",
            "assets/epds/prof.epd",
            "assets/epds/tbtest.epd",
            "assets/epds/wac.epd",
            "assets/epds/zugts.epd",
        ];

        let mut total_success = 0;
        let mut all_failures = Vec::new();

        for path in epd_files {
            let (success, failures) = parse_epd_file(path);
            total_success += success;
            for f in failures {
                all_failures.push(format!("{path}: {f}"));
            }
        }

        assert!(
            all_failures.is_empty(),
            "{} failures:\n{}",
            all_failures.len(),
            all_failures.join("\n")
        );
        assert_eq!(total_success, 2640, "expected 2640 positions parsed");
    }
}
