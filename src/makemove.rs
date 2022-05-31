// move making doesn't actually happen here,
// it happens in board.rs, but there are
// utility functions here that are used in
// the Board::make_move() function.

use crate::{
    definitions::PIECE_EMPTY,
    lookups::{CASTLE_KEYS, PIECE_KEYS, SIDE_KEY},
};

pub fn hash_castling(key: &mut u64, castle_perm: u8) {
    debug_assert!((castle_perm as usize) < CASTLE_KEYS.len());
    let castle_key = unsafe { *CASTLE_KEYS.get_unchecked(castle_perm as usize) };
    *key ^= castle_key;
}

pub fn hash_piece(key: &mut u64, piece: u8, sq: u8) {
    debug_assert!((piece as usize) < PIECE_KEYS.len());
    debug_assert!((sq as usize) < 120);
    let piece_key = unsafe {
        *PIECE_KEYS
            .get_unchecked(piece as usize)
            .get_unchecked(sq as usize)
    };
    *key ^= piece_key;
}

pub fn hash_side(key: &mut u64) {
    *key ^= SIDE_KEY;
}

pub fn hash_ep(key: &mut u64, ep_sq: u8) {
    debug_assert!((ep_sq as usize) < 120);
    let ep_key = unsafe {
        *PIECE_KEYS
            .get_unchecked(PIECE_EMPTY as usize)
            .get_unchecked(ep_sq as usize)
    };
    *key ^= ep_key;
}

#[rustfmt::skip]
pub static CASTLE_PERM_MASKS: [u8; 120] = [
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 13, 15, 15, 15, 12, 15, 15, 14, 15, // 0b1101, 0b1100, 0b1110
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15,  7, 15, 15, 15,  3, 15, 15, 11, 15, // 0b0111, 0b0011, 0b1011
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
];
