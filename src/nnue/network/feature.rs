use crate::{piece::Col, util::File};

use super::FeatureUpdate;

use crate::util::Square;

use super::INPUT;

/// wrapper to enforce bounds.
#[allow(clippy::module_name_repetitions)]
#[derive(Clone, Copy, Debug)]
pub struct FeatureIndex(usize);

impl FeatureIndex {
    /// Invariant: the result of this function is less than the number of NNUE input features (768),
    /// so it can be used to index a row of the feature-transformer matrix without bounds checking.
    #[allow(clippy::inline_always)]
    #[must_use]
    #[inline(always)]
    pub const fn index(self) -> usize {
        self.0
    }
}

pub fn indices(white_king: Square, black_king: Square, f: FeatureUpdate) -> [FeatureIndex; 2] {
    const COLOUR_STRIDE: usize = 64 * 6;
    const PIECE_STRIDE: usize = 64;

    let white_sq = if white_king.file() >= File::E {
        f.sq.flip_file()
    } else {
        f.sq
    };
    let black_sq = if black_king.file() >= File::E {
        f.sq.flip_file()
    } else {
        f.sq
    };

    let piece_type = f.piece.piece_type().index();
    let colour = f.piece.colour().index();

    let white_idx = colour * COLOUR_STRIDE + piece_type * PIECE_STRIDE + white_sq.index();
    let black_idx =
        (1 ^ colour) * COLOUR_STRIDE + piece_type * PIECE_STRIDE + black_sq.flip_rank().index();

    // SAFETY: important invariant being upheld here!!
    assert!(
        white_idx < INPUT && black_idx < INPUT,
        "attempt to construct illegal FeatureIndex."
    );
    [FeatureIndex(white_idx), FeatureIndex(black_idx)]
}

pub fn index<C: Col>(king: Square, f: FeatureUpdate) -> FeatureIndex {
    const COLOUR_STRIDE: usize = 64 * 6;
    const PIECE_STRIDE: usize = 64;

    let sq = if king.file() >= File::E {
        f.sq.flip_file()
    } else {
        f.sq
    }
    .relative_to(C::COLOUR);

    let piece_type = f.piece.piece_type().index();
    let colour = f.piece.colour().index() ^ C::COLOUR.index();

    let idx = colour * COLOUR_STRIDE + piece_type * PIECE_STRIDE + sq.index();

    // SAFETY: important invariant being upheld here!!
    assert!(idx < INPUT, "attempt to construct illegal FeatureIndex.");
    FeatureIndex(idx)
}

#[cfg(test)]
mod tests {
    use crate::piece::{Black, Piece, White};

    use super::*;

    #[test]
    fn partial_equals_full() {
        for sq in Square::all() {
            for piece in Piece::all() {
                for wk in Square::all() {
                    for bk in Square::all() {
                        let f = FeatureUpdate { sq, piece };
                        let [i1, i2] = indices(wk, bk, f);
                        assert_eq!(
                            i1.index(),
                            index::<White>(wk, f).index(),
                            "Failure when trying WHITE ({f:?}, {wk:?}, {bk:?}) - {i1:?} != {i2:?}"
                        );
                        assert_eq!(
                            i2.index(),
                            index::<Black>(bk, f).index(),
                            "Failure when trying BLACK ({f:?}, {wk:?}, {bk:?}) - {i1:?} != {i2:?}"
                        );
                    }
                }
            }
        }
    }
}
