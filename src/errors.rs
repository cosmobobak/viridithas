
pub type FenParseError = String;
#[derive(Debug)]
pub enum MoveParseError {
    InvalidLength,
    InvalidFromSquareFile,
    InvalidFromSquareRank,
    InvalidToSquareFile,
    InvalidToSquareRank,
    InvalidPromotionPiece,
    IllegalMove,
}
pub type PositionValidityError = String;