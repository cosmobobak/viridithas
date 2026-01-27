use std::fmt::Display;

use crate::{
    chess::{
        board::{
            Board,
            movegen::{self, MoveList},
        },
        chessmove::{Move, MoveFlags},
        piece::{Piece, PieceType},
        squareset::SquareSet,
        types::{CheckState, File, Rank, Square},
    },
    errors::SanError,
};

#[derive(Clone, Copy)]
pub struct SanThunk<'a> {
    board: &'a Board,
    m: Move,
}

impl Display for SanThunk<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self { board, m } = *self;
        let check_char = match board.gives(m) {
            CheckState::None => "",
            CheckState::Check => "+",
            CheckState::Checkmate => "#",
        };
        if m.is_castle() {
            match () {
                () if m.to() > m.from() => return write!(f, "O-O{check_char}"),
                () if m.to() < m.from() => return write!(f, "O-O-O{check_char}"),
                () => unreachable!(),
            }
        }
        let to_sq = m.to();
        let moved_piece = board.state.mailbox[m.from()].unwrap();
        let is_capture = board.is_capture(m)
            || (moved_piece.piece_type() == PieceType::Pawn
                && Some(to_sq) == board.state.ep_square);
        let piece_prefix = match moved_piece.piece_type() {
            PieceType::Pawn if !is_capture => "",
            PieceType::Pawn => &"abcdefgh"[m.from().file() as usize..=m.from().file() as usize],
            PieceType::Knight => "N",
            PieceType::Bishop => "B",
            PieceType::Rook => "R",
            PieceType::Queen => "Q",
            PieceType::King => "K",
        };
        let possible_ambiguous_attackers = if moved_piece.piece_type() == PieceType::Pawn {
            SquareSet::EMPTY
        } else {
            movegen::attacks_by_type(moved_piece.piece_type(), to_sq, board.state.bbs.occupied())
                & board.state.bbs.piece_bb(moved_piece)
        };
        let needs_disambiguation =
            possible_ambiguous_attackers.count() > 1 && moved_piece.piece_type() != PieceType::Pawn;
        let from_file = SquareSet::FILES[m.from().file()];
        let from_rank = SquareSet::RANKS[m.from().rank()];
        let can_be_disambiguated_by_file = (possible_ambiguous_attackers & from_file).count() == 1;
        let can_be_disambiguated_by_rank = (possible_ambiguous_attackers & from_rank).count() == 1;
        let needs_both = !can_be_disambiguated_by_file && !can_be_disambiguated_by_rank;
        let must_be_disambiguated_by_file = needs_both || can_be_disambiguated_by_file;
        let must_be_disambiguated_by_rank =
            needs_both || (can_be_disambiguated_by_rank && !can_be_disambiguated_by_file);
        let disambiguator1 = if needs_disambiguation && must_be_disambiguated_by_file {
            &"abcdefgh"[m.from().file() as usize..=m.from().file() as usize]
        } else {
            ""
        };
        let disambiguator2 = if needs_disambiguation && must_be_disambiguated_by_rank {
            &"12345678"[m.from().rank() as usize..=m.from().rank() as usize]
        } else {
            ""
        };
        let capture_sigil = if is_capture { "x" } else { "" };
        let promo_str = match m.promotion_type() {
            Some(PieceType::Knight) => "=N",
            Some(PieceType::Bishop) => "=B",
            Some(PieceType::Rook) => "=R",
            Some(PieceType::Queen) => "=Q",
            None => "",
            _ => unreachable!(),
        };

        write!(
            f,
            "{piece_prefix}{disambiguator1}{disambiguator2}{capture_sigil}{to_sq}{promo_str}{check_char}"
        )
    }
}

impl Board {
    pub fn san(&self, m: Move) -> Option<SanThunk<'_>> {
        if !self.is_pseudo_legal(m) || !self.is_legal(m) {
            return None;
        }

        Some(SanThunk { board: self, m })
    }

    /// Parses a move in Standard Algebraic Notation (SAN) and returns the corresponding move.
    ///
    /// Uses the current position as context to parse the move. Ambiguous moves are rejected.
    /// Overspecified moves (including long algebraic notation) are accepted. Common syntactical
    /// variations are also accepted.
    ///
    /// The returned move is guaranteed to be legal.
    #[expect(clippy::too_many_lines)]
    pub fn parse_san(&self, san: &str) -> Result<Move, SanError> {
        let san = san.trim();

        // Handle castling
        match san {
            "O-O" | "O-O+" | "O-O#" | "0-0" | "0-0+" | "0-0#" => {
                return self
                    .find_castling_move(true)
                    .ok_or_else(|| SanError::IllegalMove(san.to_string()));
            }
            "O-O-O" | "O-O-O+" | "O-O-O#" | "0-0-0" | "0-0-0+" | "0-0-0#" => {
                return self
                    .find_castling_move(false)
                    .ok_or_else(|| SanError::IllegalMove(san.to_string()));
            }
            _ => (),
        }

        // Strip check/mate indicators for parsing
        let trim = san.trim_end_matches(['+', '#']);
        let bytes = trim.as_bytes();

        if bytes.is_empty() {
            return Err(SanError::InvalidSan(san.to_string()));
        }

        // Parse piece type, disambiguation, capture, target square, and promotion
        let mut idx = 0;

        // Determine piece type
        let piece_type = match bytes[idx] {
            b'N' => PieceType::Knight,
            b'B' => PieceType::Bishop,
            b'R' => PieceType::Rook,
            b'Q' => PieceType::Queen,
            b'K' => PieceType::King,
            _ => PieceType::Pawn,
        };

        // Pawn moves aren't specified, but otherwise we need to advance
        // to read the square.
        if piece_type != PieceType::Pawn {
            idx += 1;
        }

        // Parse optional disambiguation and capture, then find target square
        // Format can be: [file][rank][x]<target> or [file][x]<target> or [rank][x]<target> or [x]<target> or <target>
        let mut from_file: Option<File> = None;
        let mut from_rank: Option<Rank> = None;

        // Scan backwards from end to find target square (last two chars before optional promotion)
        let mut promo_idx = bytes.len();
        let promotion = if bytes.len() >= 2 {
            let maybe_promo = bytes[bytes.len() - 1];
            let maybe_eq = bytes[bytes.len() - 2];
            if maybe_eq == b'=' {
                promo_idx = bytes.len() - 2;
                match maybe_promo {
                    b'N' | b'n' => Some(PieceType::Knight),
                    b'B' | b'b' => Some(PieceType::Bishop),
                    b'R' | b'r' => Some(PieceType::Rook),
                    b'Q' | b'q' => Some(PieceType::Queen),
                    _ => return Err(SanError::InvalidSan(san.to_string())),
                }
            } else if let Some(b'1' | b'8') = bytes.get(bytes.len() - 2) {
                // Also handle promotion without '=' (e.g. "e8Q")
                promo_idx = bytes.len() - 1;
                match maybe_promo {
                    b'N' | b'n' => Some(PieceType::Knight),
                    b'B' | b'b' => Some(PieceType::Bishop),
                    b'R' | b'r' => Some(PieceType::Rook),
                    b'Q' | b'q' => Some(PieceType::Queen),
                    _ => None,
                }
            } else {
                None
            }
        } else {
            None
        };

        // Target square should be the last two characters before promotion
        if promo_idx < idx + 2 {
            return Err(SanError::InvalidSan(san.to_string()));
        }

        let to_file_char = bytes[promo_idx - 2];
        let to_rank_char = bytes[promo_idx - 1];

        let to_file = match to_file_char {
            b'a'..=b'h' => File::from_index(to_file_char - b'a').unwrap(),
            _ => return Err(SanError::InvalidSan(san.to_string())),
        };
        let to_rank = match to_rank_char {
            b'1'..=b'8' => Rank::from_index(to_rank_char - b'1').unwrap(),
            _ => return Err(SanError::InvalidSan(san.to_string())),
        };
        let to_square = Square::from_rank_file(to_rank, to_file);

        // Parse disambiguation and capture marker between piece and target
        let middle = &bytes[idx..promo_idx - 2];
        for &ch in middle {
            match ch {
                b'a'..=b'h' => {
                    from_file = File::from_index(ch - b'a');
                }
                b'1'..=b'8' => {
                    from_rank = Rank::from_index(ch - b'1');
                }
                b'x' | b':' | b'-' => {
                    // Capture marker or move separator, ignore
                }
                _ => return Err(SanError::InvalidSan(san.to_string())),
            }
        }

        // Build masks for filtering source squares
        let mut from_mask = SquareSet::FULL;
        if let Some(file) = from_file {
            from_mask &= SquareSet::FILES[file];
        }
        if let Some(rank) = from_rank {
            from_mask &= SquareSet::RANKS[rank];
        }

        // If fully specified (both file and rank), try to find exact move
        if let (Some(file), Some(rank)) = (from_file, from_rank) {
            let from_square = Square::from_rank_file(rank, file);
            if from_square == to_square {
                return Err(SanError::IllegalMove(san.to_string()));
            }

            let m = promotion.map_or_else(
                || Move::new(from_square, to_square),
                |promotion| Move::new_with_promo(from_square, to_square, promotion),
            );

            if SquareSet::BACK_RANKS.contains_square(to_square)
                && let Some(p) = self.state.mailbox[from_square]
                && p.piece_type() == PieceType::Pawn
                && promotion.is_none()
            {
                return Err(SanError::MissingPromotion(san.to_string()));
            }

            if !self.is_pseudo_legal(m) || !self.is_legal(m) {
                return Err(SanError::IllegalMove(san.to_string()));
            }

            return Ok(m);
        }

        // Filter by piece type
        let our_pieces = self.state.bbs.colours[self.turn()];
        from_mask = from_mask
            & our_pieces
            & if piece_type == PieceType::Pawn {
                self.state.bbs.pieces[PieceType::Pawn]
            } else {
                self.state.bbs.pieces[piece_type]
            };

        // For pawn moves without file disambiguation, require same file as target (non-captures)
        if piece_type == PieceType::Pawn && from_file.is_none() {
            from_mask &= SquareSet::FILES[to_file];
        }

        // Find matching legal moves
        let mut matched_move: Option<Move> = None;

        let mut move_buffer = MoveList::new();
        self.generate_moves(&mut move_buffer);

        for &m in move_buffer.iter_moves() {
            // Check target square
            if m.to() != to_square {
                continue;
            }

            // Check source square is in our mask
            if !from_mask.contains_square(m.from()) {
                continue;
            }

            // Check piece
            if self.state.mailbox[m.from()] != Some(Piece::new(self.side, piece_type)) {
                continue;
            }

            // Check promotion matches
            if m.promotion_type() != promotion {
                continue;
            }

            if !self.is_legal(m) {
                continue;
            }

            if matched_move.is_some() {
                return Err(SanError::AmbiguousMove(san.to_string()));
            }

            matched_move = Some(m);
        }

        // Check for promotion requirement
        if matched_move.is_none() && piece_type == PieceType::Pawn && promotion.is_none() {
            let is_promo_rank = to_rank == Rank::One || to_rank == Rank::Eight;
            if is_promo_rank {
                return Err(SanError::MissingPromotion(san.to_string()));
            }
        }

        matched_move.ok_or_else(|| SanError::IllegalMove(san.to_string()))
    }

    fn find_castling_move(&self, kingside: bool) -> Option<Move> {
        let king_sq = self.state.bbs.king_sq(self.turn());
        let castle_perm = if kingside {
            self.state.castle_perm.kingside(self.turn())
        } else {
            self.state.castle_perm.queenside(self.turn())
        };

        let rook_file = castle_perm?;

        let rook_sq = Square::from_rank_file(king_sq.rank(), rook_file);
        let castle_move = Move::new_with_flags(king_sq, rook_sq, MoveFlags::Castle);

        if self.is_pseudo_legal(castle_move) && self.is_legal(castle_move) {
            Some(castle_move)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn board_from_fen(fen: &str) -> Board {
        use crate::chess::fen::Fen;
        let mut board = Board::empty();
        board.set_from_fen(&Fen::parse(fen).unwrap());
        board
    }

    #[test]
    fn simple_pawn_move() {
        let board = Board::default();
        let m = board.parse_san("e4").unwrap();
        assert_eq!(m.from(), Square::E2);
        assert_eq!(m.to(), Square::E4);
    }

    #[test]
    fn pawn_single_push() {
        let board = Board::default();
        let m = board.parse_san("e3").unwrap();
        assert_eq!(m.from(), Square::E2);
        assert_eq!(m.to(), Square::E3);
    }

    #[test]
    fn knight_move() {
        let board = Board::default();
        let m = board.parse_san("Nf3").unwrap();
        assert_eq!(m.from(), Square::G1);
        assert_eq!(m.to(), Square::F3);
    }

    #[test]
    fn pawn_capture() {
        let board = board_from_fen("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2");
        let m = board.parse_san("exd5").unwrap();
        assert_eq!(m.from(), Square::E4);
        assert_eq!(m.to(), Square::D5);
    }

    #[test]
    fn disambiguation_knight() {
        let board =
            board_from_fen("r1bqkbnr/pppp1ppp/2n5/1N2p3/4P3/5N2/PPPP1PPP/R1BQKB1R w KQ - 14 9");
        let e = board.parse_san("Nd4").unwrap_err();
        assert_eq!(e, SanError::AmbiguousMove("Nd4".into()));
        let m = board.parse_san("Nfd4").unwrap();
        assert_eq!(m.from(), Square::F3);
        assert_eq!(m.to(), Square::D4);
        let m = board.parse_san("N3d4").unwrap();
        assert_eq!(m.from(), Square::F3);
        assert_eq!(m.to(), Square::D4);
        let m = board.parse_san("Nbd4").unwrap();
        assert_eq!(m.from(), Square::B5);
        assert_eq!(m.to(), Square::D4);
        let m = board.parse_san("N5d4").unwrap();
        assert_eq!(m.from(), Square::B5);
        assert_eq!(m.to(), Square::D4);
    }

    #[test]
    fn disambiguation_pin() {
        let board =
            board_from_fen("r1b1k1nr/ppppbppp/2n5/1N2p3/2Q1P1q1/5N2/PPPP1PPP/R1BK1B1R w - - 20 12");
        let e = board.parse_san("Nfd4").unwrap_err();
        assert_eq!(e, SanError::IllegalMove("Nfd4".into()));
        let e = board.parse_san("N3d4").unwrap_err();
        assert_eq!(e, SanError::IllegalMove("N3d4".into()));
        let m = board.parse_san("Nd4").unwrap();
        assert_eq!(m.from(), Square::B5);
        assert_eq!(m.to(), Square::D4);
        let m = board.parse_san("Nbd4").unwrap();
        assert_eq!(m.from(), Square::B5);
        assert_eq!(m.to(), Square::D4);
        let m = board.parse_san("N5d4").unwrap();
        assert_eq!(m.from(), Square::B5);
        assert_eq!(m.to(), Square::D4);
    }

    #[test]
    fn disambiguation_rank() {
        let board = board_from_fen("4k3/8/8/R7/8/8/8/R3K3 w Q - 0 1");
        let e = board.parse_san("Ra3").unwrap_err();
        assert_eq!(e, SanError::AmbiguousMove("Ra3".into()));
        let m = board.parse_san("R1a3").unwrap();
        assert_eq!(m.from(), Square::A1);
        assert_eq!(m.to(), Square::A3);
    }

    #[test]
    fn castling_kingside() {
        let board = board_from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1");
        let m = board.parse_san("O-O").unwrap();
        assert!(m.is_castle());
        assert_eq!(m.from(), Square::E1);
        assert_eq!(m.to(), Square::H1);
    }

    #[test]
    fn castling_queenside() {
        let board = board_from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1");
        let m = board.parse_san("O-O-O").unwrap();
        assert!(m.is_castle());
        assert_eq!(m.from(), Square::E1);
        assert_eq!(m.to(), Square::A1);
    }

    #[test]
    fn castling_zeros() {
        let board = board_from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1");
        let m = board.parse_san("0-0").unwrap();
        assert!(m.is_castle());
        let m2 = board.parse_san("0-0-0").unwrap();
        assert!(m2.is_castle());
    }

    #[test]
    fn promotion() {
        let board = board_from_fen("8/P7/8/8/8/8/8/4K2k w - - 0 1");
        let m = board.parse_san("a8=Q").unwrap();
        assert_eq!(m.from(), Square::A7);
        assert_eq!(m.to(), Square::A8);
        assert_eq!(m.promotion_type(), Some(PieceType::Queen));
    }

    #[test]
    fn missing_promotion() {
        let board = board_from_fen("8/P7/8/8/8/8/8/4K2k w - - 0 1");
        let e = board.parse_san("a8").unwrap_err();
        assert_eq!(e, SanError::MissingPromotion("a8".into()));
        let e = board.parse_san("a7a8").unwrap_err();
        assert_eq!(e, SanError::MissingPromotion("a7a8".into()));
    }

    #[test]
    fn terse_promotion() {
        let board = board_from_fen("8/P7/8/8/8/8/8/4K2k w - - 0 1");
        let m = board.parse_san("a8q").unwrap();
        assert_eq!(m.from(), Square::A7);
        assert_eq!(m.to(), Square::A8);
        assert_eq!(m.promotion_type(), Some(PieceType::Queen));
        let m = board.parse_san("a8n").unwrap();
        assert_eq!(m.from(), Square::A7);
        assert_eq!(m.to(), Square::A8);
        assert_eq!(m.promotion_type(), Some(PieceType::Knight));
    }

    #[test]
    fn promotion_knight() {
        let board = board_from_fen("8/P7/8/8/8/8/8/4K2k w - - 0 1");
        let m = board.parse_san("a8=N").unwrap();
        assert_eq!(m.promotion_type(), Some(PieceType::Knight));
    }

    #[test]
    fn promotion_without_equals() {
        let board = board_from_fen("8/P7/8/8/8/8/8/4K2k w - - 0 1");
        let m = board.parse_san("a8Q").unwrap();
        assert_eq!(m.promotion_type(), Some(PieceType::Queen));
    }

    #[test]
    fn with_check_marker() {
        let board = board_from_fen("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2");
        let m = board.parse_san("Qh4+").unwrap();
        assert_eq!(m.from(), Square::D8);
        assert_eq!(m.to(), Square::H4);
    }

    #[test]
    fn capture_with_x() {
        let board = board_from_fen("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2");
        let m = board.parse_san("exd5").unwrap();
        assert_eq!(m.to(), Square::D5);
    }

    #[test]
    fn capture_without_x() {
        let board = board_from_fen("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2");
        let m = board.parse_san("ed5").unwrap();
        assert_eq!(m.to(), Square::D5);
    }

    #[test]
    fn invalid() {
        let board = Board::default();
        assert!(board.parse_san("Ze4").is_err());
        assert!(board.parse_san("").is_err());
        assert!(board.parse_san("xxxx").is_err());
    }

    #[test]
    fn illegal_move() {
        let board = Board::default();
        // Can't move pawn to e5 in one move from starting position
        assert!(matches!(
            board.parse_san("e5"),
            Err(SanError::IllegalMove(_))
        ));
    }

    #[test]
    fn roundtrip() {
        // Test that san() and parse_san() are inverses
        let positions = [
            Board::STARTING_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            "rn3r2/pbppq1p1/1p2pN2/8/3P2NP/6P1/PPP1BP1R/R3K1k1 w Q - 5 18",
        ];
        for p in positions {
            let board = board_from_fen(p);
            for legal_move in board.legal_moves() {
                let san_str = board.san(legal_move).unwrap().to_string();
                let parsed = board.parse_san(&san_str).unwrap();
                assert_eq!(
                    legal_move, parsed,
                    "Roundtrip failed for {san_str}: got {parsed:?}"
                );
            }
        }
    }

    #[test]
    fn fully_specified_move() {
        // Test fully specified notation like "Ng1f3"
        let board = Board::default();
        let m = board.parse_san("Ng1f3").unwrap();
        assert_eq!(m.from(), Square::G1);
        assert_eq!(m.to(), Square::F3);
    }

    #[test]
    fn en_passant() {
        let board = board_from_fen("rnbqkbnr/pppp1ppp/8/4pP2/8/8/PPPPP1PP/RNBQKBNR w KQkq e6 0 3");
        let m = board.parse_san("fxe6").unwrap();
        assert!(m.is_ep());
        assert_eq!(m.to(), Square::E6);
        let m = board.parse_san("fe6").unwrap();
        assert!(m.is_ep());
        assert_eq!(m.to(), Square::E6);
    }
}
