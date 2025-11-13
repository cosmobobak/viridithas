#![expect(clippy::cast_possible_truncation)]

//! FFI bridge for Pyrrhic tablebase probing.
//! exposes movegen functions to C code.

use crate::chess::{
    board::movegen::{bishop_attacks, king_attacks, knight_attacks, rook_attacks},
    squareset::SquareSet,
    types::Square,
};

#[unsafe(no_mangle)]
pub extern "C" fn viridithas_popcount(bb: u64) -> u32 {
    bb.count_ones()
}

#[unsafe(no_mangle)]
pub extern "C" fn viridithas_lsb(bb: u64) -> u32 {
    bb.trailing_zeros()
}

#[unsafe(no_mangle)]
pub extern "C" fn viridithas_poplsb(bb: *mut u64) -> u64 {
    // SAFETY: We assume the C code passes a valid pointer
    unsafe {
        let value = *bb;
        let lsb = value.trailing_zeros();
        *bb = value & (value - 1); // Clear the LSB
        u64::from(lsb)
    }
}

/// colour: true = white (1), false = black (0)
#[unsafe(no_mangle)]
pub extern "C" fn viridithas_pawn_attacks(sq: u32, colour: bool) -> u64 {
    let Some(square) = Square::new(sq as u8) else {
        return 0;
    };

    let bb = SquareSet::from_square(square);

    let attacks = if colour {
        // White pawns attack northeast and northwest
        bb.north_east_one() | bb.north_west_one()
    } else {
        // Black pawns attack southeast and southwest
        bb.south_east_one() | bb.south_west_one()
    };

    attacks.inner()
}

#[unsafe(no_mangle)]
pub extern "C" fn viridithas_knight_attacks(sq: u32) -> u64 {
    let Some(square) = Square::new(sq as u8) else {
        return 0;
    };

    knight_attacks(square).inner()
}

#[unsafe(no_mangle)]
pub extern "C" fn viridithas_bishop_attacks(sq: u32, occupied: u64) -> u64 {
    let Some(square) = Square::new(sq as u8) else {
        return 0;
    };

    let blockers = SquareSet::from_inner(occupied);
    bishop_attacks(square, blockers).inner()
}

#[unsafe(no_mangle)]
pub extern "C" fn viridithas_rook_attacks(sq: u32, occupied: u64) -> u64 {
    let Some(square) = Square::new(sq as u8) else {
        return 0;
    };

    let blockers = SquareSet::from_inner(occupied);
    rook_attacks(square, blockers).inner()
}

#[unsafe(no_mangle)]
pub extern "C" fn viridithas_queen_attacks(sq: u32, occupied: u64) -> u64 {
    let Some(square) = Square::new(sq as u8) else {
        return 0;
    };

    let blockers = SquareSet::from_inner(occupied);
    let bishop = bishop_attacks(square, blockers);
    let rook = rook_attacks(square, blockers);
    (bishop | rook).inner()
}

#[unsafe(no_mangle)]
pub extern "C" fn viridithas_king_attacks(sq: u32) -> u64 {
    let Some(square) = Square::new(sq as u8) else {
        return 0;
    };

    king_attacks(square).inner()
}
