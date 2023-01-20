#![allow(clippy::cast_possible_truncation)]

use crate::{
    definitions::{
        File::{FILE_A, FILE_H},
        Rank::{RANK_1, RANK_8},
        Square,
    },
    rng::XorShiftState, piece::PieceType,
};

/// Implements a C-style for loop, for use in const fn.
#[macro_export]
macro_rules! cfor {
    ($init: stmt; $cond: expr; $step: expr; $body: block) => {
        {
            $init
            #[allow(while_true)]
            while $cond {
                $body;

                $step;
            }
        }
    }
}

const fn init_hash_keys() -> ([[u64; 64]; 13], [u64; 16], u64) {
    let mut state = XorShiftState::new();
    let mut piece_keys = [[0; 64]; 13];
    cfor!(let mut index = 0; index < 13; index += 1; {
        cfor!(let mut sq = 0; sq < 64; sq += 1; {
            let key;
            (key, state) = state.next_self();
            piece_keys[index][sq] = key;
        });
    });
    let mut castle_keys = [0; 16];
    cfor!(let mut index = 0; index < 16; index += 1; {
        let key;
        (key, state) = state.next_self();
        castle_keys[index] = key;
    });
    let key;
    (key, _) = state.next_self();
    let side_key = key;
    (piece_keys, castle_keys, side_key)
}

pub const fn init_eval_masks() -> ([u64; 8], [u64; 8]) {
    let mut rank_masks = [0; 8];
    let mut file_masks = [0; 8];

    let mut r = RANK_8;
    loop {
        let mut f = FILE_A;
        while f <= FILE_H {
            let sq = r * 8 + f;
            file_masks[f as usize] |= 1 << sq;
            rank_masks[r as usize] |= 1 << sq;
            f += 1;
        }
        if r == RANK_1 {
            break;
        }
        r -= 1;
    }

    (rank_masks, file_masks)
}

pub const fn init_passed_isolated_bb() -> ([u64; 64], [u64; 64], [u64; 64]) {
    #![allow(clippy::cast_possible_wrap)]
    const _FILE_BB: [u64; 8] = init_eval_masks().1;
    let mut white_passed_bb = [0; 64];
    let mut black_passed_bb = [0; 64];
    let mut isolated_bb = [0; 64];

    let mut sq = Square::A1;
    loop {
        let mut t_sq = sq.signed_inner() + 8;
        while t_sq < 64 {
            white_passed_bb[sq.index()] |= 1 << t_sq;
            t_sq += 8;
        }

        t_sq = sq.signed_inner() - 8;
        while t_sq >= 0 {
            black_passed_bb[sq.index()] |= 1 << t_sq;
            t_sq -= 8;
        }

        if sq.file() > FILE_A {
            isolated_bb[sq.index()] |= _FILE_BB[sq.file() as usize - 1];

            t_sq = sq.signed_inner() + 7;
            while t_sq < 64 {
                white_passed_bb[sq.index()] |= 1 << t_sq;
                t_sq += 8;
            }

            t_sq = sq.signed_inner() - 9;
            while t_sq >= 0 {
                black_passed_bb[sq.index()] |= 1 << t_sq;
                t_sq -= 8;
            }
        }

        if sq.file() < FILE_H {
            isolated_bb[sq.index()] |= _FILE_BB[sq.file() as usize + 1];

            t_sq = sq.signed_inner() + 9;
            while t_sq < 64 {
                white_passed_bb[sq.index()] |= 1 << t_sq;
                t_sq += 8;
            }

            t_sq = sq.signed_inner() - 7;
            while t_sq >= 0 {
                black_passed_bb[sq.index()] |= 1 << t_sq;
                t_sq -= 8;
            }
        }

        if matches!(sq, Square::H8) {
            break;
        }
        sq = sq.add(1);
    }

    (white_passed_bb, black_passed_bb, isolated_bb)
}

pub static PIECE_KEYS: [[u64; 64]; 13] = init_hash_keys().0;
pub static CASTLE_KEYS: [u64; 16] = init_hash_keys().1;
pub const SIDE_KEY: u64 = init_hash_keys().2;

/// knights, bishops, rooks, and queens.
pub static PIECE_BIG: [bool; 13] =
    [false, false, true, true, true, true, false, false, true, true, true, true, false];
/// rooks and queens.
pub static PIECE_MAJ: [bool; 13] =
    [false, false, false, false, true, true, false, false, false, false, true, true, false];
/// knights and bishops.
#[cfg(debug_assertions)]
pub static PIECE_MIN: [bool; 13] =
    [false, false, true, true, false, false, false, false, true, true, false, false, false];

fn victim_score(piece: PieceType) -> i32 {
    i32::from(piece.inner()) * 1000 // pawn = 1000, knight = 2000, bishop = 3000, etc.
}

/// The score of this pair of pieces, for MVV/LVA move ordering.
pub fn get_mvv_lva_score(victim: PieceType, attacker: PieceType) -> i32 {
    victim_score(victim) + 60 - victim_score(attacker) / 100
}

const fn init_jumping_attacks<const IS_KNIGHT: bool>() -> [u64; 64] {
    let mut attacks = [0; 64];
    let deltas =
        if IS_KNIGHT { &[17, 15, 10, 6, -17, -15, -10, -6] } else { &[9, 8, 7, 1, -9, -8, -7, -1] };
    cfor!(let mut sq = Square::A1; true; sq = sq.add(1); {
        let mut attacks_bb = 0;
        cfor!(let mut idx = 0; idx < 8; idx += 1; {
            let delta = deltas[idx];
            let attacked_sq = sq.signed_inner() + delta;
            #[allow(clippy::cast_sign_loss)]
            if 0 <= attacked_sq && attacked_sq < 64 && Square::distance(sq, Square::new(attacked_sq as u8)) <= 2 {
                attacks_bb |= 1 << attacked_sq;
            }
        });
        attacks[sq.index()] = attacks_bb;
        if matches!(sq, Square::H8) {
            break;
        }
    });
    attacks
}

pub fn get_jumping_piece_attack<const PIECE_TYPE: u8>(sq: Square) -> u64 {
    static KNIGHT_ATTACKS: [u64; 64] = init_jumping_attacks::<true>();
    static KING_ATTACKS: [u64; 64] = init_jumping_attacks::<false>();
    debug_assert!(PIECE_TYPE < 7);
    debug_assert!(sq.on_board());
    debug_assert!(PIECE_TYPE == PieceType::KNIGHT.inner() || PIECE_TYPE == PieceType::KING.inner());
    if PIECE_TYPE == PieceType::KNIGHT.inner() {
        unsafe { *KNIGHT_ATTACKS.get_unchecked(sq.index()) }
    } else {
        unsafe { *KING_ATTACKS.get_unchecked(sq.index()) }
    }
}

mod tests {
    #[test]
    fn all_piece_keys_different() {
        use crate::lookups::PIECE_KEYS;
        let mut hashkeys = PIECE_KEYS.iter().flat_map(|&k| k).collect::<Vec<u64>>();
        hashkeys.sort_unstable();
        let len_before = hashkeys.len();
        hashkeys.dedup();
        let len_after = hashkeys.len();
        assert_eq!(len_before, len_after);
    }

    #[test]
    fn all_castle_keys_different() {
        use crate::lookups::CASTLE_KEYS;
        let mut hashkeys = CASTLE_KEYS.to_vec();
        hashkeys.sort_unstable();
        let len_before = hashkeys.len();
        hashkeys.dedup();
        let len_after = hashkeys.len();
        assert_eq!(len_before, len_after);
    }

    #[test]
    fn python_chess_validation() {
        use crate::definitions::Square;
        use crate::lookups::get_jumping_piece_attack;
        use crate::piece::PieceType;
        // testing that the attack bitboards match the ones in the python-chess library,
        // which are known to be correct.
        assert_eq!(get_jumping_piece_attack::<{ PieceType::KNIGHT.inner() }>(Square::new(0)), 132_096);
        assert_eq!(get_jumping_piece_attack::<{ PieceType::KNIGHT.inner() }>(Square::new(63)), 9_077_567_998_918_656);

        assert_eq!(get_jumping_piece_attack::<{ PieceType::KING.inner() }>(Square::new(0)), 770);
        assert_eq!(get_jumping_piece_attack::<{ PieceType::KING.inner() }>(Square::new(63)), 4_665_729_213_955_833_856);
    }
}
