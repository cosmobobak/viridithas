use crate::{
    definitions::{Piece, Square120},
    lookups::FILES_BOARD,
};

#[inline]
pub fn square_on_board(sq: u8) -> bool {
    FILES_BOARD[sq as usize] != Square120::OffBoard as usize
}

pub fn side_valid(side: u8) -> bool {
    side == 0 || side == 1
}

pub fn file_rank_valid(fr: u8) -> bool {
    (0..=7).contains(&fr)
}

pub fn piece_valid_empty(pc: u8) -> bool {
    pc >= Piece::Empty as u8 && pc <= Piece::BK as u8
}

pub fn piece_valid(pc: u8) -> bool {
    pc >= Piece::WP as u8 && pc <= Piece::BK as u8
}
