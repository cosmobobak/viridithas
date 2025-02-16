use crate::{
    chess::{
        board::movegen::attacks_by_type,
        chessmove::Move,
        piece::{Piece, PieceType},
        squareset::SquareSet,
        types::Square,
    },
    lookups::{PIECE_KEYS, SIDE_KEY},
};

pub static KEYS: [u32; 8192] =
    unsafe { std::mem::transmute(*include_bytes!("../embeds/cuckoo_keys.bin")) };
pub static MOVES: [Option<Move>; 8192] =
    unsafe { std::mem::transmute(*include_bytes!("../embeds/cuckoo_moves.bin")) };

pub const fn h1(key: u32) -> usize {
    (key & 0x1FFF) as usize
}

pub const fn h2(key: u32) -> usize {
    ((key >> 16) & 0x1FFF) as usize
}

pub fn init() -> anyhow::Result<()> {
    println!("Initialising cuckoo-hash tables.");
    // keep a tally of the table entries to sanity-check the initialisation process.
    let mut count = 0;

    let mut keys = vec![0; 8192];
    let mut cuckoo_moves = vec![None; 8192];

    for piece in Piece::all() {
        if piece.piece_type() == PieceType::Pawn {
            continue;
        }
        for square0 in Square::all() {
            for square1 in Square::all().filter(|&s1| s1 > square0) {
                // check if a piece of this type standing on square0 could attack square1
                let attack_overlap = attacks_by_type(piece.piece_type(), square0, SquareSet::EMPTY)
                    & square1.as_set();
                if attack_overlap == SquareSet::EMPTY {
                    continue;
                }

                let mut mv = Some(Move::new(square0, square1));
                #[allow(clippy::cast_possible_truncation)]
                let mut key =
                    (PIECE_KEYS[piece][square0] ^ PIECE_KEYS[piece][square1] ^ SIDE_KEY) as u32;
                let mut slot = h1(key);
                loop {
                    std::mem::swap(&mut keys[slot], &mut key);
                    std::mem::swap(&mut cuckoo_moves[slot], &mut mv);

                    if mv.is_none() {
                        break;
                    }

                    slot = if slot == h1(key) { h2(key) } else { h1(key) };
                }
                count += 1;
            }
        }
    }
    assert_eq!(count, 3668);
    // SAFETY: u64 is fine to view as bytes.
    let key_bytes = unsafe { keys.align_to::<u8>().1 };
    // SAFETY: Option<Move> is repr-equivalent to u16.
    let move_bytes = unsafe { cuckoo_moves.align_to::<u8>().1 };
    std::fs::write("embeds/cuckoo_keys.bin", key_bytes)?;
    std::fs::write("embeds/cuckoo_moves.bin", move_bytes)?;
    println!("Wrote cuckoo-hash tables.");
    Ok(())
}
