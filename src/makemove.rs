// move making doesn't actually happen here,
// it happens in board.rs, but there are
// utility functions here that are used in
// the Board::make_move() function.

use crate::{
    definitions::Square,
    lookups::{CASTLE_KEYS, PIECE_KEYS, SIDE_KEY}, piece::Piece,
};

pub fn hash_castling(key: &mut u64, castle_perm: u8) {
    debug_assert!((castle_perm as usize) < CASTLE_KEYS.len());
    let castle_key = unsafe { *CASTLE_KEYS.get_unchecked(castle_perm as usize) };
    *key ^= castle_key;
}

pub fn hash_piece(key: &mut u64, piece: Piece, sq: Square) {
    debug_assert!((piece.index()) < PIECE_KEYS.len());
    debug_assert!(sq.on_board());
    let piece_key = unsafe { *PIECE_KEYS.get_unchecked(piece.index()).get_unchecked(sq.index()) };
    *key ^= piece_key;
}

pub fn hash_side(key: &mut u64) {
    *key ^= SIDE_KEY;
}

pub fn hash_ep(key: &mut u64, ep_sq: Square) {
    debug_assert!(ep_sq.on_board());
    let ep_key =
        unsafe { *PIECE_KEYS.get_unchecked(Piece::EMPTY.index()).get_unchecked(ep_sq.index()) };
    *key ^= ep_key;
}

#[rustfmt::skip]
pub static CASTLE_PERM_MASKS: [u8; 64] = [
    13, 15, 15, 15, 12, 15, 15, 14, // 0b1101, 0b1100, 0b1110
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
     7, 15, 15, 15,  3, 15, 15, 11, // 0b0111, 0b0011, 0b1011
];
