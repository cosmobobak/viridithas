use crate::{
    chess::{
        piece::{Colour, PieceType},
        types::{File, Square},
    },
    nnue::network::{FeatureUpdate, INPUT},
};

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

pub fn index(colour: Colour, king: Square, f: FeatureUpdate) -> FeatureIndex {
    const COLOUR_STRIDE: usize = 64 * 6;
    const PIECE_STRIDE: usize = 64;

    let sq = if king.file() >= File::E {
        f.sq.flip_file()
    } else {
        f.sq
    }
    .relative_to(colour);

    let piece_type = f.piece.piece_type().index();
    let colour = (f.piece.colour().index() ^ colour.index())
        * usize::from(f.piece.piece_type() != PieceType::King);

    let idx = colour * COLOUR_STRIDE + piece_type * PIECE_STRIDE + sq.index();

    // SAFETY: important invariant being upheld here!!
    assert!(idx < INPUT, "attempt to construct illegal FeatureIndex.");
    FeatureIndex(idx)
}

/// For non-merged king planes.
pub fn index_full(colour: Colour, king: Square, f: FeatureUpdate) -> usize {
    const COLOUR_STRIDE: usize = 64 * 6;
    const PIECE_STRIDE: usize = 64;

    let sq = if king.file() >= File::E {
        f.sq.flip_file()
    } else {
        f.sq
    }
    .relative_to(colour);

    let piece_type = f.piece.piece_type().index();
    let colour = f.piece.colour().index() ^ colour.index();

    let idx = colour * COLOUR_STRIDE + piece_type * PIECE_STRIDE + sq.index();

    assert!(idx < 12 * 64);
    idx
}
