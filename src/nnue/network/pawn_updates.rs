// SPDX-License-Identifier: AGPL-3.0-only

use crate::chess::squareset::SquareSet;

/// For a pawn on a given file, which other pawns are
/// included in its “vision” for pawn - pawn features?
#[rustfmt::skip]
pub static PAWN_PAWN_MASKS: [SquareSet; 8] = [
                             (SquareSet::FILE_A).union(SquareSet::FILE_B),
    (SquareSet::FILE_A).union(SquareSet::FILE_B).union(SquareSet::FILE_C),
    (SquareSet::FILE_B).union(SquareSet::FILE_C).union(SquareSet::FILE_D),
    (SquareSet::FILE_C).union(SquareSet::FILE_D).union(SquareSet::FILE_E),
    (SquareSet::FILE_D).union(SquareSet::FILE_E).union(SquareSet::FILE_F),
    (SquareSet::FILE_E).union(SquareSet::FILE_F).union(SquareSet::FILE_G),
    (SquareSet::FILE_F).union(SquareSet::FILE_G).union(SquareSet::FILE_H),
    (SquareSet::FILE_G).union(SquareSet::FILE_H),
];
