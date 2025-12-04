/*
 * FFI bridge between Pyrrhic and Viridithas
 * Provides attack generation and bitboard manipulation functions
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Bitboard manipulation functions
uint32_t viridithas_popcount(uint64_t bb);
uint32_t viridithas_lsb(uint64_t bb);
uint64_t viridithas_poplsb(uint64_t* bb);

// Attack generation functions
uint64_t viridithas_pawn_attacks(uint32_t sq, bool colour);
uint64_t viridithas_knight_attacks(uint32_t sq);
uint64_t viridithas_bishop_attacks(uint32_t sq, uint64_t occupied);
uint64_t viridithas_rook_attacks(uint32_t sq, uint64_t occupied);
uint64_t viridithas_queen_attacks(uint32_t sq, uint64_t occupied);
uint64_t viridithas_king_attacks(uint32_t sq);

#ifdef __cplusplus
}
#endif
