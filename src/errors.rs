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
#[allow(dead_code)]
pub type PositionValidityError = String;
