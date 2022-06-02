#![allow(clippy::cast_possible_truncation)]

use crate::{definitions::{
    BK, FILE_A, FILE_H, RANK_1, RANK_8, WP, KNIGHT, KING, square_distance,
}, rng::XorShiftState};

macro_rules! cfor {
    ($init: stmt; $cond: expr; $step: expr; $body: block) => {
        {
            $init
            while $cond {
                $body;

                $step;
            }
        }
    }
}

pub const fn filerank_to_square(f: u8, r: u8) -> u8 {
    f + r * 8
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

    let mut sq = 0;
    while sq < 64 {
        let mut t_sq = sq as isize + 8;
        while t_sq < 64 {
            white_passed_bb[sq] |= 1 << t_sq;
            t_sq += 8;
        }

        t_sq = sq as isize - 8;
        while t_sq >= 0 {
            black_passed_bb[sq] |= 1 << t_sq;
            t_sq -= 8;
        }

        if file(sq as u8) > FILE_A {
            isolated_bb[sq] |= _FILE_BB[file(sq as u8) as usize - 1];

            t_sq = sq as isize + 7;
            while t_sq < 64 {
                white_passed_bb[sq] |= 1 << t_sq;
                t_sq += 8;
            }

            t_sq = sq as isize - 9;
            while t_sq >= 0 {
                black_passed_bb[sq] |= 1 << t_sq;
                t_sq -= 8;
            }
        }

        if file(sq as u8) < FILE_H {
            isolated_bb[sq] |= _FILE_BB[file(sq as u8) as usize + 1];

            t_sq = sq as isize + 9;
            while t_sq < 64 {
                white_passed_bb[sq] |= 1 << t_sq;
                t_sq += 8;
            }

            t_sq = sq as isize - 7;
            while t_sq >= 0 {
                black_passed_bb[sq] |= 1 << t_sq;
                t_sq -= 8;
            }
        }

        sq += 1;
    }

    (white_passed_bb, black_passed_bb, isolated_bb)
}

pub static PIECE_KEYS: [[u64; 64]; 13] = init_hash_keys().0;
pub static CASTLE_KEYS: [u64; 16] = init_hash_keys().1;
pub const SIDE_KEY: u64 = init_hash_keys().2;

/// knights, bishops, rooks, and queens.
pub static PIECE_BIG: [bool; 13] = [
    false, false, true, true, true, true, false, false, true, true, true, true, false,
];
/// rooks and queens.
pub static PIECE_MAJ: [bool; 13] = [
    false, false, false, false, true, true, false, false, false, false, true, true, false,
];
/// knights and bishops.
#[allow(dead_code)]
pub static PIECE_MIN: [bool; 13] = [
    false, false, true, true, false, false, false, false, true, true, false, false, false,
];

/// The file that this square is on.
pub const fn file(sq: u8) -> u8 { sq % 8 }
/// The rank that this square is on.
pub const fn rank(sq: u8) -> u8 { sq / 8 }

/// The name of this 64-indexed square.
pub static SQUARE_NAMES: [&str; 64] = [
    "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1", "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
    "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3", "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
    "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5", "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
    "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7", "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
];

/// The name of this piece.
#[allow(dead_code)]
static PIECE_NAMES: [&str; 13] = [
    "NO_PIECE", "pawn", "knight", "bishop", "rook", "queen", "king", "pawn", "knight", "bishop",
    "rook", "queen", "king",
];

#[allow(dead_code)]
pub fn piece_name(piece: u8) -> Option<&'static str> {
    PIECE_NAMES.get(piece as usize).copied()
}

static PIECE_CHARS: [u8; 13] = *b".PNBRQKpnbrqk";
pub static PROMO_CHAR_LOOKUP: [u8; 13] = *b"XXnbrqXXnbrqX";

pub fn piece_char(piece: u8) -> Option<char> {
    PIECE_CHARS.get(piece as usize).map(|&c| c as char)
}

/// The score of this piece, for MVV/LVA move ordering.
const VICTIM_SCORE: [i32; 13] = [
    0, 1000, 2000, 3000, 4000, 5000, 6000, 1000, 2000, 3000, 4000, 5000, 6000,
];

const fn mvvlva_init() -> [[i32; 13]; 13] {
    let mut mvvlva = [[0; 13]; 13];
    let mut attacker = WP as usize;
    while attacker <= BK as usize {
        let mut victim = WP as usize;
        while victim <= BK as usize {
            mvvlva[victim][attacker] = VICTIM_SCORE[victim] + 60 - VICTIM_SCORE[attacker] / 100;
            victim += 1;
        }
        attacker += 1;
    }
    mvvlva
}

/// The score of this pair of pieces, for MVV/LVA move ordering.
pub static MVV_LVA_SCORE: [[i32; 13]; 13] = mvvlva_init();

const fn init_jumping_attacks<const IS_KNIGHT: bool>() -> [u64; 64] {
    let mut attacks = [0; 64];
    let deltas = if IS_KNIGHT {
        &[17, 15, 10, 6, -17, -15, -10, -6]
    } else {
        &[9, 8, 7, 1, -9, -8, -7, -1]
    };
    cfor!(let mut sq = 0; sq < 64; sq += 1; {
        let mut attacks_bb = 0;
        cfor!(let mut idx = 0; idx < 8; idx += 1; {
            let delta = deltas[idx];
            #[allow(clippy::cast_possible_wrap)]
            let attacked_sq = sq as i32 + delta;
            #[allow(clippy::cast_sign_loss)]
            if attacked_sq >= 0 && attacked_sq < 64 && square_distance(sq as u8, attacked_sq as u8) <= 2 {
                attacks_bb |= 1 << attacked_sq;
            }
        });
        attacks[sq] = attacks_bb;
    });
    attacks
}

static JUMPING_ATTACKS: [[u64; 64]; 7] = [
    [0; 64], // no_piece
    [0; 64], // pawn
    init_jumping_attacks::<true>(), // knight
    [0; 64], // bishop
    [0; 64], // rook
    [0; 64], // queen
    init_jumping_attacks::<false>(), // king
];

pub fn get_jumping_piece_attack(sq: u8, piece: u8) -> u64 {
    debug_assert!(piece < 7);
    debug_assert!(sq < 64);
    debug_assert!(piece == KNIGHT || piece == KING);
    unsafe {
        *JUMPING_ATTACKS
            .get_unchecked(piece as usize)
            .get_unchecked(sq as usize)
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
        use crate::lookups::get_jumping_piece_attack;
        use crate::definitions::{KING, KNIGHT}; 
        // testing that the attack bitboards match the ones in the python-chess library,
        // which are known to be correct.
        assert_eq!(get_jumping_piece_attack(0, KNIGHT), 132_096);
        assert_eq!(get_jumping_piece_attack(63, KNIGHT), 9_077_567_998_918_656);

        assert_eq!(get_jumping_piece_attack(0, KING), 770);
        assert_eq!(get_jumping_piece_attack(63, KING), 4_665_729_213_955_833_856);
    }
}
