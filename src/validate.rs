use crate::{
    definitions::{Square120, BK, PIECE_EMPTY, WP},
    lookups::FILES_BOARD,
};

pub fn square_on_board(sq: u8) -> bool {
    FILES_BOARD[sq as usize] != Square120::OffBoard as u8
}

pub const fn side_valid(side: u8) -> bool {
    side == 0 || side == 1
}

pub const fn piece_valid_empty(pc: u8) -> bool {
    pc >= PIECE_EMPTY as u8 && pc <= BK
}

pub const fn piece_valid(pc: u8) -> bool {
    pc >= WP && pc <= BK
}
