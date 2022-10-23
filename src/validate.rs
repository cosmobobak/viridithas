use crate::definitions::{BK, KING, PAWN, WP};

pub const fn side_valid(side: u8) -> bool {
    side == 0 || side == 1
}

pub const fn piece_valid(pc: u8) -> bool {
    pc >= WP && pc <= BK
}

pub const fn piece_type_valid(pc: u8) -> bool {
    pc >= PAWN && pc <= KING
}
