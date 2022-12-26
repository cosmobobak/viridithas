// move making doesn't actually happen here,
// it happens in board.rs, but there are
// utility functions here that are used in
// the Board::make_move() function.

use crate::{
    definitions::{Square, PIECE_EMPTY},
    lookups::{CASTLE_KEYS, PIECE_KEYS, SIDE_KEY, FIFTY_MOVE_KEYS},
};

pub fn hash_castling(key: &mut u64, castle_perm: u8) {
    debug_assert!((castle_perm as usize) < CASTLE_KEYS.len());
    let castle_key = unsafe { *CASTLE_KEYS.get_unchecked(castle_perm as usize) };
    *key ^= castle_key;
}

pub fn hash_piece(key: &mut u64, piece: u8, sq: Square) {
    debug_assert!((piece as usize) < PIECE_KEYS.len());
    debug_assert!(sq.on_board());
    let piece_key = unsafe { *PIECE_KEYS.get_unchecked(piece as usize).get_unchecked(sq.index()) };
    *key ^= piece_key;
}

pub fn hash_side(key: &mut u64) {
    *key ^= SIDE_KEY;
}

pub fn hash_ep(key: &mut u64, ep_sq: Square) {
    debug_assert!(ep_sq.on_board());
    let ep_key =
        unsafe { *PIECE_KEYS.get_unchecked(PIECE_EMPTY as usize).get_unchecked(ep_sq.index()) };
    *key ^= ep_key;
}

pub fn hash_fiftymove(key: &mut u64, before: u8, after: u8) {
    // fiftymove is in the range 0..100:
    debug_assert!(before < 100);
    debug_assert!(after < 100);
    // we only change the hashkey in blocks of eight - if the fiftymove counter
    // is not crossing an eight-move boundary, we don't update the hashkey.
    let before_generation = before / 8;
    let after_generation = after / 8;
    if before_generation == after_generation {
        return;
    }
    // get the two keys
    let before_key = FIFTY_MOVE_KEYS[before_generation as usize];
    let after_key = FIFTY_MOVE_KEYS[after_generation as usize];
    // xor before in:
    *key ^= before_key;
    // xor after in:
    *key ^= after_key;
}

pub fn hash_in_fiftymove(key: &mut u64, fifty_move_counter: u8) {
    debug_assert!(fifty_move_counter < 100);
    let generation = fifty_move_counter / 8;
    let fiftymo_key = FIFTY_MOVE_KEYS[generation as usize];
    *key ^= fiftymo_key;
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
