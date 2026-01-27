use std::num::ParseIntError;
use std::str::ParseBoolError;

use thiserror::Error;

use crate::chess::piece::Colour;
use crate::chess::types::{Rank, Square};

/// Errors that can occur when parsing SAN (Standard Algebraic Notation) moves.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SanError {
    #[error("invalid san: {0:?}")]
    InvalidSan(String),
    #[error("illegal san: {0:?}")]
    IllegalMove(String),
    #[error("ambiguous san: {0:?}")]
    AmbiguousMove(String),
    #[error("missing promotion piece type: {0:?}")]
    MissingPromotion(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum MoveParseError {
    #[error("invalid move length {0}")]
    InvalidLength(usize),
    #[error("invalid from-square file {0}")]
    InvalidFromSquareFile(char),
    #[error("invalid from-square rank {0}")]
    InvalidFromSquareRank(char),
    #[error("invalid to-square file {0}")]
    InvalidToSquareFile(char),
    #[error("invalid to-square rank {0}")]
    InvalidToSquareRank(char),
    #[error("invalid promotion piece {0}")]
    InvalidPromotionPiece(char),
    #[error("illegal move {0}")]
    IllegalMove(String),
    #[error("unknown error")]
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum QuickParseError {
    #[error("invalid length {0}")]
    InvalidLength(usize),
    #[error("invalid annotation")]
    InvalidAnnotation,
    #[error("invalid outcome")]
    InvalidOutcome,
    #[error("invalid miscellania")]
    InvalidMiscellania,
    #[error("collision at square {0}")]
    DuplicateLocation(Square),
    #[error("invalid variable piece")]
    InvalidVariable,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum FenParseError {
    #[error("FEN string is missing board part")]
    MissingBoard,
    #[error("board part of FEN has {0} segments, expected 8")]
    BoardSegments(usize),
    #[error("wrong number of squares in board segment")]
    BadSquaresInSegment,
    #[error("adjacent digits in board segment are not allowed")]
    AdjacentDigits,
    #[error("unexpected character in piece placement: '{0}'")]
    UnexpectedCharacter(char),
    #[error("expected side to be 'w' or 'b', got \"{0}\"")]
    InvalidSide(String),
    #[error("expected side part")]
    MissingSide,
    #[error("expected castling part")]
    MissingCastling,
    #[error("invalid castling format: \"{0}\"")]
    InvalidCastling(String),
    #[error("{} king is missing", if *colour == Colour::White { "white "} else { "black" })]
    MissingKing { colour: Colour },
    #[error("more than one {} king", if *colour == Colour::White { "white "} else { "black" })]
    DuplicateKings { colour: Colour },
    #[error("pawns present on backranks")]
    PawnsOnBackranks,
    #[error("waiting player's king in check")]
    WaitingInCheck,
    #[error(
        "{colour} king is not on the back rank, but castling rights \"{castling}\" imply present castling rights"
    )]
    KingNotOnBackRank {
        colour: &'static str,
        castling: String,
    },
    #[error(
        "{colour} king is on file {file}, but got castling rights on that file: \"{castling}\""
    )]
    KingOnCastlingFile {
        colour: &'static str,
        file: String,
        castling: String,
    },
    #[error("expected en passant part")]
    MissingEnPassant,
    #[error("invalid en passant square: \"{0}\"")]
    InvalidEnPassant(String),
    #[error("invalid en passant rank for square \"{square}\": expected {expected:?}, got {got:?}")]
    InvalidEnPassantRank {
        square: String,
        expected: Rank,
        got: Rank,
    },
    #[error("expected halfmove clock part")]
    MissingHalfmoveClock,
    #[error("invalid halfmove clock: \"{0}\"")]
    InvalidHalfmoveClock(String),
    #[error("halfmove clock {0} exceeds maximum of 100")]
    HalfmoveClockTooLarge(u8),
    #[error("expected fullmove number part")]
    MissingFullmoveNumber,
    #[error("invalid fullmove number: \"{0}\"")]
    InvalidFullmoveNumber(String),
    #[error("fullmove number must be at least 1")]
    FullmoveNumberZero,
    #[error("unexpected extra tokens after fullmove number")]
    ExtraTokens,
}

/// Errors that can occur when parsing the `position` command.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PositionParseError {
    #[error("failed to parse FEN: {0}")]
    Fen(#[from] FenParseError),
    #[error("failed to parse move: {0}")]
    Move(#[from] MoveParseError),
    #[error("`position` command requires a position specifier (fen, startpos, frc, or dfrc)")]
    MissingPositionSpecifier,
    #[error("`position startpos` must be followed by `moves` or nothing, got \"{0}\"")]
    InvalidStartposSuffix(String),
    #[error("unknown position specifier \"{0}\", expected fen, startpos, frc, or dfrc")]
    UnknownPositionSpecifier(String),
    #[error("`position frc` requires an index (0-959)")]
    MissingFrcIndex,
    #[error("failed to parse FRC index \"{text}\": {source}")]
    InvalidFrcIndex { text: String, source: ParseIntError },
    #[error("FRC index {0} out of range, must be 0-959")]
    FrcIndexOutOfRange(u32),
    #[error("`position dfrc` requires an index (0-921599)")]
    MissingDfrcIndex,
    #[error("failed to parse DFRC index \"{text}\": {source}")]
    InvalidDfrcIndex { text: String, source: ParseIntError },
    #[error("DFRC index {0} out of range, must be 0-921599")]
    DfrcIndexOutOfRange(u32),
}

/// Errors that can occur when parsing the `go` command.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum GoParseError {
    #[error("`go` command is empty")]
    EmptyCommand,
    #[error("unknown go subcommand \"{0}\"")]
    UnknownSubcommand(String),
    #[error("missing value after `{0}`")]
    MissingValue(&'static str),
    #[error("failed to parse value for `{param}`: {source}")]
    InvalidValue {
        param: &'static str,
        source: ParseIntError,
    },
    #[error("incomplete time control: got some of wtime/btime/winc/binc but not both clocks")]
    IncompleteTimeControl,
}

/// Errors that can occur when parsing the `setoption` command.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SetOptionParseError {
    #[error("`setoption` must be followed by `name`")]
    MissingNameKeyword,
    #[error("expected `name` after `setoption`, got \"{0}\"")]
    ExpectedNameKeyword(String),
    #[error("missing option name after `setoption name`")]
    MissingOptionName,
    #[error("expected `value` after option name, got \"{0}\"")]
    ExpectedValueKeyword(String),
    #[error("missing value after `setoption name {0} value`")]
    MissingOptionValue(String),
    #[error("invalid integer value for option `{name}`: {source}")]
    InvalidIntValue { name: String, source: ParseIntError },
    #[error("invalid boolean value for option `{name}`: {source}")]
    InvalidBoolValue {
        name: String,
        source: ParseBoolError,
    },
    #[error("value {got} out of range for option `{name}`, expected {lo}..={hi}")]
    ValueOutOfRange {
        name: String,
        lo: i64,
        hi: i64,
        got: i64,
    },
    #[error("invalid value for tuning parameter `{name}`: {message}")]
    InvalidTuningParam { name: String, message: String },
}

/// Errors that can occur when parsing a `go perft` command.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PerftParseError {
    #[error("missing depth after `perft`")]
    MissingDepth,
    #[error("failed to parse perft depth \"{text}\": {source}")]
    InvalidDepth { text: String, source: ParseIntError },
}

/// Top-level UCI errors.
#[derive(Debug, Error)]
pub enum UciError {
    #[error("unknown command: {0}")]
    UnknownCommand(String),
    #[error("{0}")]
    Position(#[from] PositionParseError),
    #[error("{0}")]
    Go(#[from] GoParseError),
    #[error("{0}")]
    SetOption(#[from] SetOptionParseError),
    #[error("{0}")]
    Perft(#[from] PerftParseError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("internal error: {0}")]
    Internal(&'static str),
    // TODO: Convert to non-anyhow, proper inner error.
    #[error("thread error: {0}")]
    Thread(String),
    // TODO: Convert to non-anyhow, proper inner error.
    #[error("NNUE initialization failed: {0}")]
    NnueInit(String),
}
