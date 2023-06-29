#![allow(clippy::cast_possible_truncation)]

use crate::{
    board,
    definitions::{File, Rank, Square},
    nnue,
    rng::XorShiftState,
    squareset::SquareSet,
    transpositiontable,
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

pub const fn init_eval_masks() -> ([SquareSet; 8], [SquareSet; 8]) {
    let mut rank_masks = [SquareSet::EMPTY; 8];
    let mut file_masks = [SquareSet::EMPTY; 8];

    let mut r = Rank::RANK_8;
    loop {
        let mut f = File::FILE_A;
        while f <= File::FILE_H {
            let sq = Square::from_rank_file(r, f);
            file_masks[f as usize] = file_masks[f as usize].add_square(sq);
            rank_masks[r as usize] = rank_masks[r as usize].add_square(sq);
            f += 1;
        }
        if r == Rank::RANK_1 {
            break;
        }
        r -= 1;
    }

    (rank_masks, file_masks)
}

pub const fn init_passed_isolated_bb() -> ([SquareSet; 64], [SquareSet; 64], [SquareSet; 64]) {
    #![allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    const _FILE_BB: [SquareSet; 8] = init_eval_masks().1;
    let mut white_passed_bb = [SquareSet::EMPTY; 64];
    let mut black_passed_bb = [SquareSet::EMPTY; 64];
    let mut isolated_bb = [SquareSet::EMPTY; 64];

    let mut sq = Square::A1;
    loop {
        let mut t_sq = sq.signed_inner() + 8;
        while t_sq < 64 {
            let add = Square::new(t_sq as u8);
            white_passed_bb[sq.index()] = white_passed_bb[sq.index()].add_square(add);
            t_sq += 8;
        }

        t_sq = sq.signed_inner() - 8;
        while t_sq >= 0 {
            let add = Square::new(t_sq as u8);
            black_passed_bb[sq.index()] = black_passed_bb[sq.index()].add_square(add);
            t_sq -= 8;
        }

        if sq.file() > File::FILE_A {
            isolated_bb[sq.index()] =
                isolated_bb[sq.index()].union(_FILE_BB[sq.file() as usize - 1]);

            t_sq = sq.signed_inner() + 7;
            while t_sq < 64 {
                let add = Square::new(t_sq as u8);
                white_passed_bb[sq.index()] = white_passed_bb[sq.index()].add_square(add);
                t_sq += 8;
            }

            t_sq = sq.signed_inner() - 9;
            while t_sq >= 0 {
                let add = Square::new(t_sq as u8);
                black_passed_bb[sq.index()] = black_passed_bb[sq.index()].add_square(add);
                t_sq -= 8;
            }
        }

        if sq.file() < File::FILE_H {
            isolated_bb[sq.index()] =
                isolated_bb[sq.index()].union(_FILE_BB[sq.file() as usize + 1]);

            t_sq = sq.signed_inner() + 9;
            while t_sq < 64 {
                let add = Square::new(t_sq as u8);
                white_passed_bb[sq.index()] = white_passed_bb[sq.index()].add_square(add);
                t_sq += 8;
            }

            t_sq = sq.signed_inner() - 7;
            while t_sq >= 0 {
                let add = Square::new(t_sq as u8);
                black_passed_bb[sq.index()] = black_passed_bb[sq.index()].add_square(add);
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
    [false, true, true, true, true, false, false, true, true, true, true, false, false];
/// rooks and queens.
pub static PIECE_MAJ: [bool; 13] =
    [false, false, false, true, true, false, false, false, false, true, true, false, false];
/// knights and bishops.
#[cfg(debug_assertions)]
pub static PIECE_MIN: [bool; 13] =
    [false, true, true, false, false, false, false, true, true, false, false, false, false];

const fn init_jumping_attacks<const IS_KNIGHT: bool>() -> [SquareSet; 64] {
    let mut attacks = [SquareSet::EMPTY; 64];
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
        attacks[sq.index()] = SquareSet::from_inner(attacks_bb);
        if matches!(sq, Square::H8) {
            break;
        }
    });
    attacks
}

pub fn get_knight_attacks(sq: Square) -> SquareSet {
    static KNIGHT_ATTACKS: [SquareSet; 64] = init_jumping_attacks::<true>();
    debug_assert!(sq.on_board());
    KNIGHT_ATTACKS[sq.index()]
}
pub fn get_king_attacks(sq: Square) -> SquareSet {
    static KING_ATTACKS: [SquareSet; 64] = init_jumping_attacks::<false>();
    debug_assert!(sq.on_board());
    KING_ATTACKS[sq.index()]
}

pub fn info_dump() {
    use crate::{NAME, VERSION};
    let version_extension = if cfg!(feature = "final-release") { "" } else { "-dev" };
    println!("{NAME} {VERSION}{version_extension}");
    println!("Compiled with architecture: {}", std::env::consts::ARCH);
    println!("Compiled for OS: {}", std::env::consts::OS);
    println!(
        "Number of HCE parameters: {}",
        board::evaluation::parameters::EvalParams::default().vectorise().len()
    );
    println!("Number of NNUE parameters: {}", nnue::network::NNUEParams::num_params());
    println!("Size of NNUE network: {} bytes", std::mem::size_of::<nnue::network::NNUEParams>());
    println!(
        "Size of a transposition table entry: {} bytes",
        std::mem::size_of::<transpositiontable::TTEntry>()
    );
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
        use crate::lookups::{get_king_attacks, get_knight_attacks};
        use crate::squareset::SquareSet;
        // testing that the attack bitboards match the ones in the python-chess library,
        // which are known to be correct.
        assert_eq!(get_knight_attacks(Square::new(0)), SquareSet::from_inner(132_096));
        assert_eq!(
            get_knight_attacks(Square::new(63)),
            SquareSet::from_inner(9_077_567_998_918_656)
        );

        assert_eq!(get_king_attacks(Square::new(0)), SquareSet::from_inner(770));
        assert_eq!(
            get_king_attacks(Square::new(63)),
            SquareSet::from_inner(4_665_729_213_955_833_856)
        );
    }
}
