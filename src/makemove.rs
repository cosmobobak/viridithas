// move making doesn't actually happen here,
// it happens in board.rs, but there are
// utility functions here that are used in
// the Board::make_move() function.

use crate::{
    lookups::{CASTLE_KEYS, EP_KEYS, PIECE_KEYS, SIDE_KEY},
    chess::piece::Piece,
    util::{CastlingRights, Square},
};

pub fn hash_castling(key: &mut u64, castle_perm: CastlingRights) {
    let castle_key = CASTLE_KEYS[castle_perm.hashkey_index()];
    *key ^= castle_key;
}

pub fn hash_piece(key: &mut u64, piece: Piece, sq: Square) {
    let piece_key = PIECE_KEYS[piece][sq];
    *key ^= piece_key;
}

pub fn hash_side(key: &mut u64) {
    *key ^= SIDE_KEY;
}

pub fn hash_ep(key: &mut u64, ep_sq: Square) {
    let ep_key = EP_KEYS[ep_sq];
    *key ^= ep_key;
}
