#![allow(clippy::cast_possible_truncation)]

use crate::{definitions::{
    Colour, File, Rank, Square120, BK, BOARD_N_SQUARES, FILE_A, FILE_H, RANK_1, RANK_8, WP,
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
    21 + f + (r * 10)
}

pub const fn init_sq120_to_sq64() -> ([u8; BOARD_N_SQUARES], [u8; 64]) {
    let mut sq120_to_sq64 = [0; BOARD_N_SQUARES];
    let mut index = 0;
    while index < BOARD_N_SQUARES {
        sq120_to_sq64[index] = 65;
        index += 1;
    }
    let mut sq64_to_sq120 = [0; 64];
    let mut index = 0;
    while index < 64 {
        sq64_to_sq120[index] = 120;
        index += 1;
    }
    let mut sq64 = 0;
    let mut rank = Rank::Rank1 as u8;
    while rank <= Rank::Rank8 as u8 {
        let mut file = File::FileA as u8;
        while file <= File::FileH as u8 {
            let sq = filerank_to_square(file, rank);
            sq64_to_sq120[sq64] = sq as u8;
            sq120_to_sq64[sq as usize] = sq64 as u8;
            sq64 += 1;
            file += 1;
        }
        rank += 1;
    }
    (sq120_to_sq64, sq64_to_sq120)
}

const fn init_hash_keys() -> ([[u64; 120]; 13], [u64; 16], u64) {
    let mut state = XorShiftState::new();
    let mut piece_keys = [[0; 120]; 13];
    cfor!(let mut index = 0; index < 13; index += 1; {
        cfor!(let mut sq = 0; sq < 120; sq += 1; {
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
    const _FILES_BOARD: [u8; BOARD_N_SQUARES] = files_ranks().0;
    const _SQ64_TO_SQ120: [u8; 64] = init_sq120_to_sq64().1;
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

        if _FILES_BOARD[_SQ64_TO_SQ120[sq] as usize] > FILE_A {
            isolated_bb[sq] |= _FILE_BB[_FILES_BOARD[_SQ64_TO_SQ120[sq] as usize] as usize - 1];

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

        if _FILES_BOARD[_SQ64_TO_SQ120[sq] as usize] < FILE_H {
            isolated_bb[sq] |= _FILE_BB[_FILES_BOARD[_SQ64_TO_SQ120[sq] as usize] as usize + 1];

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

pub const fn files_ranks() -> ([u8; BOARD_N_SQUARES], [u8; BOARD_N_SQUARES]) {
    let mut files = [0; BOARD_N_SQUARES];
    let mut ranks = [0; BOARD_N_SQUARES];
    cfor!(let mut index = 0; index < BOARD_N_SQUARES; index += 1; {
        files[index] = Square120::OffBoard as u8;
        ranks[index] = Square120::OffBoard as u8;
    });
    cfor!(let mut rank = Rank::Rank1 as u8; rank <= Rank::Rank8 as u8; rank += 1; {
        cfor!(let mut file = File::FileA as u8; file <= File::FileH as u8; file += 1; {
            let sq = filerank_to_square(file, rank);
            files[sq as usize] = file;
            ranks[sq as usize] = rank;
        });
    });
    (files, ranks)
}

pub static SQ120_TO_SQ64: [u8; BOARD_N_SQUARES] = init_sq120_to_sq64().0;
pub static SQ64_TO_SQ120: [u8; 64] = init_sq120_to_sq64().1;

pub static PIECE_KEYS: [[u64; 120]; 13] = init_hash_keys().0;
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
pub static PIECE_MIN: [bool; 13] = [
    false, false, true, true, false, false, false, false, true, true, false, false, false,
];

/// The colour of a piece.
pub static PIECE_COL: [Colour; 13] = [
    Colour::Both,
    Colour::White,
    Colour::White,
    Colour::White,
    Colour::White,
    Colour::White,
    Colour::White,
    Colour::Black,
    Colour::Black,
    Colour::Black,
    Colour::Black,
    Colour::Black,
    Colour::Black,
];

/// The file that this 120-indexed square is on.
pub static FILES_BOARD: [u8; BOARD_N_SQUARES] = files_ranks().0;
/// The rank that this 120-indexed square is on.
pub static RANKS_BOARD: [u8; BOARD_N_SQUARES] = files_ranks().1;

/// The name of this 64-indexed square.
pub static SQUARE_NAMES: [&str; 64] = [
    "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1", "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
    "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3", "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
    "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5", "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
    "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7", "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
];

/// The name of this piece.
#[allow(dead_code)]
pub static PIECE_NAMES: [&str; 13] = [
    "NO_PIECE", "pawn", "knight", "bishop", "rook", "queen", "king", "pawn", "knight", "bishop",
    "rook", "queen", "king",
];

pub static PIECE_CHARS: [u8; 13] = *b".PNBRQKpnbrqk";
pub static PROMO_CHAR_LOOKUP: [u8; 13] = *b"XXnbrqXXnbrqX";

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
}
