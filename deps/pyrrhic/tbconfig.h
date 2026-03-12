/*
 * (c) 2015 basil, all rights reserved,
 * Modifications Copyright (c) 2016-2019 by Jon Dart
 * Modifications Copyright (c) 2020-2024 by Andrew Grant
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

/*
 * Configuration for Pyrrhic to work with Viridithas
 *
 * This uses FFI to call back into Rust for attack generation
 * and bitboard manipulation functions.
 */

#include "viridithas_bridge.h"

#define PYRRHIC_POPCOUNT(x) (viridithas_popcount(x))
#define PYRRHIC_LSB(x) (viridithas_lsb(x))
#define PYRRHIC_POPLSB(x) (viridithas_poplsb(x))

#define PYRRHIC_PAWN_ATTACKS(sq, c) (viridithas_pawn_attacks(sq, c))
#define PYRRHIC_KNIGHT_ATTACKS(sq) (viridithas_knight_attacks(sq))
#define PYRRHIC_BISHOP_ATTACKS(sq, occ) (viridithas_bishop_attacks(sq, occ))
#define PYRRHIC_ROOK_ATTACKS(sq, occ) (viridithas_rook_attacks(sq, occ))
#define PYRRHIC_QUEEN_ATTACKS(sq, occ) (viridithas_queen_attacks(sq, occ))
#define PYRRHIC_KING_ATTACKS(sq) (viridithas_king_attacks(sq))
