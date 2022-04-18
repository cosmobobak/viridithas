#![allow(clippy::cast_possible_truncation)]

use crate::definitions::{File, Rank, BOARD_N_SQUARES, Colour, Square120};

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

const fn init_sq120_to_sq64() -> ([u8; BOARD_N_SQUARES], [u8; 64]) {
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

const fn init_bit_masks() -> ([u64; 64], [u64; 64]) {
    let mut setmask = [0; 64];
    let mut clearmask = [0; 64];

    cfor!(let mut index = 0; index < 64; index += 1; {
        setmask[index] = 1 << index;
        clearmask[index] = !setmask[index];
    });

    (setmask, clearmask)
}

struct XorShiftState {
    pub state: u128
}

const fn rand_u64(mut xs: XorShiftState) -> (u64, XorShiftState) {
    let mut x = xs.state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    xs.state = x;
    (x as u64, xs)
}

const fn init_hash_keys() -> ([[u64; 120]; 13], [u64; 16], u64) {
    let mut state = XorShiftState { state: const_random::const_random!(u128) };
    let mut piece_keys = [[0; 120]; 13];
    cfor!(let mut index = 0; index < 13; index += 1; {
        cfor!(let mut sq = 0; sq < 120; sq += 1; {
            let key;
            (key, state) = rand_u64(state);
            piece_keys[index][sq] = key;
        });
    });
    let mut castle_keys = [0; 16];
    cfor!(let mut index = 0; index < 16; index += 1; {
        let key;
        (key, state) = rand_u64(state);
        castle_keys[index] = key;
    });
    let key;
    (key, _) = rand_u64(state);
    let side_key = key;
    (piece_keys, castle_keys, side_key)
}

const fn files_ranks() -> ([usize; BOARD_N_SQUARES], [usize; BOARD_N_SQUARES]) {
    let mut files = [0; BOARD_N_SQUARES];
    let mut ranks = [0; BOARD_N_SQUARES];
    cfor!(let mut index = 0; index < BOARD_N_SQUARES; index += 1; {
        files[index] = Square120::OffBoard as usize;
        ranks[index] = Square120::OffBoard as usize;
    });
    cfor!(let mut rank = Rank::Rank1 as usize; rank <= Rank::Rank8 as usize; rank += 1; {
        cfor!(let mut file = File::FileA as usize; file <= File::FileH as usize; file += 1; {
            let sq = filerank_to_square(file as u8, rank as u8);
            files[sq as usize] = file;
            ranks[sq as usize] = rank;
        });
    });
    (files, ranks)
}

pub static SQ120_TO_SQ64: [u8; BOARD_N_SQUARES] = init_sq120_to_sq64().0;
pub static SQ64_TO_SQ120: [u8; 64] = init_sq120_to_sq64().1;

pub static SET_MASK: [u64; 64] = init_bit_masks().0;
pub static CLEAR_MASK: [u64; 64] = init_bit_masks().1;

pub static PIECE_KEYS: [[u64; 120]; 13] = init_hash_keys().0;
pub static CASTLE_KEYS: [u64; 16] = init_hash_keys().1;
pub const SIDE_KEY: u64 = init_hash_keys().2;

pub static PIECE_BIG: [bool; 13] = [false, false, true, true, true, true, true, false, true, true, true, true, true];
pub static PIECE_MAJ: [bool; 13] = [false, false, false, false, true, true, true, false, false, false, true, true, true];
pub static PIECE_MIN: [bool; 13] = [false, false, true, true, false, false, false, false, true, true, false, false, false];
pub static PIECE_VAL: [i32; 13] = [0, 100, 325, 325, 550, 1000, 50000, 100, 325, 325, 550, 1000, 50000];

pub static PIECE_COL: [Colour; 13] = [
    Colour::Both, Colour::White, Colour::White, Colour::White, Colour::White, Colour::White, Colour::White, Colour::Black, Colour::Black, Colour::Black, Colour::Black, Colour::Black, Colour::Black,
];

pub static FILES_BOARD: [usize; BOARD_N_SQUARES] = files_ranks().0;
pub static RANKS_BOARD: [usize; BOARD_N_SQUARES] = files_ranks().1;

pub static SQUARE_NAMES: [&str; 64] = [
    "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1", "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
    "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3", "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
    "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5", "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
    "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7", "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
];

mod tests {
    use crate::lookups::{PIECE_KEYS, CASTLE_KEYS};

    #[test]
    fn all_piece_keys_different() {
        let mut hashkeys = PIECE_KEYS.iter().flat_map(|&k| k).collect::<Vec<u64>>();
        hashkeys.sort_unstable();
        let len_before = hashkeys.len();
        hashkeys.dedup();
        let len_after = hashkeys.len();
        assert_eq!(len_before, len_after);
    }

    #[test]
    fn all_castle_keys_different() {
        let mut hashkeys = CASTLE_KEYS.to_vec();
        hashkeys.sort_unstable();
        let len_before = hashkeys.len();
        hashkeys.dedup();
        let len_after = hashkeys.len();
        assert_eq!(len_before, len_after);
    }
}