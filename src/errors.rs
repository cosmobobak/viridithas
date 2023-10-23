use std::fmt::Display;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MoveParseError {
    InvalidLength(usize),
    InvalidFromSquareFile(char),
    InvalidFromSquareRank(char),
    InvalidToSquareFile(char),
    InvalidToSquareRank(char),
    InvalidPromotionPiece(char),
    IllegalMove(String),
}
impl Display for MoveParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::InvalidLength(len) => write!(f, "Invalid move length {len}"),
            Self::InvalidFromSquareFile(file) => write!(f, "Invalid from-square file {file}"),
            Self::InvalidFromSquareRank(rank) => write!(f, "Invalid from-square rank {rank}"),
            Self::InvalidToSquareFile(file) => write!(f, "Invalid to-square file {file}"),
            Self::InvalidToSquareRank(rank) => write!(f, "Invalid to-square rank {rank}"),
            Self::InvalidPromotionPiece(piece) => write!(f, "Invalid promotion piece {piece}"),
            Self::IllegalMove(m) => write!(f, "Illegal move {m}"),
        }
    }
}

#[cfg(debug_assertions)]
pub type PositionValidityError = String;

pub type FenParseError = String;
