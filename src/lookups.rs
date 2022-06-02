#![allow(clippy::cast_possible_truncation)]

use crate::{
    definitions::{
        square_distance, BK, FILE_A, FILE_H, KING, KNIGHT, MAX_DEPTH, RANK_1, RANK_8, WP,
    },
    rng::XorShiftState,
};

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
pub const fn file(sq: u8) -> u8 {
    sq % 8
}
/// The rank that this square is on.
pub const fn rank(sq: u8) -> u8 {
    sq / 8
}

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

/// each slot is 1024 * (1 + log(index)).
pub static REDUCTIONS: [i32; MAX_DEPTH as usize] = [
    0, 1024, 1733, 2148, 2443, 2672, 2858, 3016, 3153, 3273, 3381, 3479, 3568, 3650, 3726, 3797,
    3863, 3925, 3983, 4039, 4091, 4141, 4189, 4234, 4278, 4320, 4360, 4398, 4436, 4472, 4506, 4540,
    4572, 4604, 4634, 4664, 4693, 4721, 4748, 4775, 4801, 4826, 4851, 4875, 4899, 4922, 4944, 4966,
    4988, 5009, 5029, 5050, 5070, 5089, 5108, 5127, 5145, 5164, 5181, 5199, 5216, 5233, 5250, 5266,
    5282, 5298, 5314, 5329, 5344, 5359, 5374, 5388, 5403, 5417, 5431, 5445, 5458, 5472, 5485, 5498,
    5511, 5523, 5536, 5548, 5561, 5573, 5585, 5597, 5608, 5620, 5631, 5643, 5654, 5665, 5676, 5687,
    5697, 5708, 5719, 5729, 5739, 5749, 5759, 5769, 5779, 5789, 5799, 5808, 5818, 5827, 5837, 5846,
    5855, 5864, 5873, 5882, 5891, 5900, 5909, 5917, 5926, 5934, 5943, 5951, 5959, 5968, 5976, 5984,
    5992, 6000, 6008, 6016, 6023, 6031, 6039, 6047, 6054, 6062, 6069, 6076, 6084, 6091, 6098, 6105,
    6113, 6120, 6127, 6134, 6141, 6148, 6154, 6161, 6168, 6175, 6181, 6188, 6195, 6201, 6208, 6214,
    6220, 6227, 6233, 6240, 6246, 6252, 6258, 6264, 6270, 6277, 6283, 6289, 6295, 6300, 6306, 6312,
    6318, 6324, 6330, 6335, 6341, 6347, 6352, 6358, 6364, 6369, 6375, 6380, 6386, 6391, 6396, 6402,
    6407, 6412, 6418, 6423, 6428, 6434, 6439, 6444, 6449, 6454, 6459, 6464, 6469, 6474, 6479, 6484,
    6489, 6494, 6499, 6504, 6509, 6513, 6518, 6523, 6528, 6533, 6537, 6542, 6547, 6551, 6556, 6560,
    6565, 6570, 6574, 6579, 6583, 6588, 6592, 6597, 6601, 6605, 6610, 6614, 6618, 6623, 6627, 6631,
    6636, 6640, 6644, 6648, 6653, 6657, 6661, 6665, 6669, 6673, 6677, 6682, 6686, 6690, 6694, 6698,
    6702, 6706, 6710, 6714, 6718, 6722, 6725, 6729, 6733, 6737, 6741, 6745, 6749, 6752, 6756, 6760,
    6764, 6768, 6771, 6775, 6779, 6782, 6786, 6790, 6794, 6797, 6801, 6804, 6808, 6812, 6815, 6819,
    6822, 6826, 6829, 6833, 6836, 6840, 6843, 6847, 6850, 6854, 6857, 6861, 6864, 6868, 6871, 6874,
    6878, 6881, 6884, 6888, 6891, 6894, 6898, 6901, 6904, 6908, 6911, 6914, 6917, 6921, 6924, 6927,
    6930, 6933, 6937, 6940, 6943, 6946, 6949, 6952, 6956, 6959, 6962, 6965, 6968, 6971, 6974, 6977,
    6980, 6983, 6986, 6989, 6992, 6995, 6998, 7001, 7004, 7007, 7010, 7013, 7016, 7019, 7022, 7025,
    7028, 7031, 7034, 7037, 7039, 7042, 7045, 7048, 7051, 7054, 7057, 7059, 7062, 7065, 7068, 7071,
    7073, 7076, 7079, 7082, 7084, 7087, 7090, 7093, 7095, 7098, 7101, 7104, 7106, 7109, 7112, 7114,
    7117, 7120, 7122, 7125, 7128, 7130, 7133, 7135, 7138, 7141, 7143, 7146, 7148, 7151, 7154, 7156,
    7159, 7161, 7164, 7166, 7169, 7171, 7174, 7177, 7179, 7182, 7184, 7187, 7189, 7192, 7194, 7196,
    7199, 7201, 7204, 7206, 7209, 7211, 7214, 7216, 7218, 7221, 7223, 7226, 7228, 7230, 7233, 7235,
    7238, 7240, 7242, 7245, 7247, 7249, 7252, 7254, 7256, 7259, 7261, 7263, 7266, 7268, 7270, 7273,
    7275, 7277, 7279, 7282, 7284, 7286, 7288, 7291, 7293, 7295, 7297, 7300, 7302, 7304, 7306, 7309,
    7311, 7313, 7315, 7317, 7320, 7322, 7324, 7326, 7328, 7330, 7333, 7335, 7337, 7339, 7341, 7343,
    7345, 7348, 7350, 7352, 7354, 7356, 7358, 7360, 7362, 7364, 7367, 7369, 7371, 7373, 7375, 7377,
    7379, 7381, 7383, 7385, 7387, 7389, 7391, 7393, 7395, 7397, 7399, 7401, 7404, 7406, 7408, 7410,
];

static JUMPING_ATTACKS: [[u64; 64]; 7] = [
    [0; 64],                         // no_piece
    [0; 64],                         // pawn
    init_jumping_attacks::<true>(),  // knight
    [0; 64],                         // bishop
    [0; 64],                         // rook
    [0; 64],                         // queen
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
        use crate::definitions::{KING, KNIGHT};
        use crate::lookups::get_jumping_piece_attack;
        // testing that the attack bitboards match the ones in the python-chess library,
        // which are known to be correct.
        assert_eq!(get_jumping_piece_attack(0, KNIGHT), 132_096);
        assert_eq!(get_jumping_piece_attack(63, KNIGHT), 9_077_567_998_918_656);

        assert_eq!(get_jumping_piece_attack(0, KING), 770);
        assert_eq!(
            get_jumping_piece_attack(63, KING),
            4_665_729_213_955_833_856
        );
    }
}
