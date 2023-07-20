// move making doesn't actually happen here,
// it happens in board.rs, but there are
// utility functions here that are used in
// the Board::make_move() function.

use crate::{
    lookups::{CASTLE_KEYS, PIECE_KEYS, SIDE_KEY},
    piece::Piece,
    util::{CastlingRights, Square},
};

pub fn hash_castling(key: &mut u64, castle_perm: CastlingRights) {
    let castle_key = CASTLE_KEYS[castle_perm.hashkey_index()];
    *key ^= castle_key;
}

pub fn hash_piece(key: &mut u64, piece: Piece, sq: Square) {
    debug_assert!((piece.index()) < PIECE_KEYS.len());
    debug_assert!(sq.on_board());
    let piece_key = PIECE_KEYS[piece.index()][sq.index()];
    *key ^= piece_key;
}

pub fn hash_side(key: &mut u64) {
    *key ^= SIDE_KEY;
}

pub fn hash_ep(key: &mut u64, ep_sq: Square) {
    debug_assert!(ep_sq.on_board());
    let ep_key = PIECE_KEYS[Piece::EMPTY.index()][ep_sq.index()];
    *key ^= ep_key;
}
