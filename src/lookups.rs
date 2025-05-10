#![allow(clippy::cast_possible_truncation)]

use crate::rng::XorShiftState;

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

#[allow(clippy::type_complexity)]
const fn init_hash_keys() -> ([[u64; 64]; 12], [u64; 64], [u64; 16], u64, [u64; 256]) {
    let mut state = XorShiftState::new();
    let mut piece_keys = [[0; 64]; 12];
    cfor!(let mut index = 0; index < 12; index += 1; {
        cfor!(let mut sq = 0; sq < 64; sq += 1; {
            let key;
            (key, state) = state.next_self();
            piece_keys[index][sq] = key;
        });
    });
    let mut ep_keys = [0; 64];
    cfor!(let mut sq = 0; sq < 64; sq += 1; {
        let key;
        (key, state) = state.next_self();
        ep_keys[sq] = key;
    });
    let mut castle_keys = [0; 16];
    cfor!(let mut index = 0; index < 16; index += 1; {
        let key;
        (key, state) = state.next_self();
        castle_keys[index] = key;
    });
    let key;
    (key, state) = state.next_self();
    let side_key = key;
    let mut halfmove_clock_keys = [0; 256];
    cfor!(let mut i = 14; i < 256 - 8; i += 8; {
        let key;
        (key, state) = state.next_self();
        cfor!(let mut j = 0; j < 8; j += 1; {
            halfmove_clock_keys[i + j] = key;
        });
    });
    (
        piece_keys,
        ep_keys,
        castle_keys,
        side_key,
        halfmove_clock_keys,
    )
}

pub static PIECE_KEYS: [[u64; 64]; 12] = init_hash_keys().0;
pub static EP_KEYS: [u64; 64] = init_hash_keys().1;
pub static CASTLE_KEYS: [u64; 16] = init_hash_keys().2;
pub const SIDE_KEY: u64 = init_hash_keys().3;
pub static HM_CLOCK_KEYS: [u64; 256] = init_hash_keys().4;

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
