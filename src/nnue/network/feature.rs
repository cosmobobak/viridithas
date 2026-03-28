use crate::{
    cfor,
    chess::{
        board::movegen::{attacks_by_type, attacks_by_type_slow},
        piece::{Colour, Piece, PieceType},
        squareset::SquareSet,
        types::{File, Rank, Square},
    },
    nnue::network::{MERGE_KING_PLANES, PSQT_FEATURES, PsqtFeatureUpdate, THREAT_FEATURES},
};

/// wrapper to enforce bounds.
#[allow(clippy::module_name_repetitions)]
#[derive(Clone, Copy, Debug)]
pub struct PsqtFeatureIndex(usize);

impl PsqtFeatureIndex {
    /// Invariant: the result of this function is less than the number of NNUE input features (704),
    /// so it can be used to index a row of the feature-transformer matrix without bounds checking.
    #[allow(clippy::inline_always)]
    #[must_use]
    #[inline(always)]
    pub const fn index(self) -> usize {
        self.0
    }
}

pub fn psqt_index(colour: Colour, king: Square, f: PsqtFeatureUpdate) -> PsqtFeatureIndex {
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
        * usize::from(!MERGE_KING_PLANES || f.piece.piece_type() != PieceType::King);

    let idx = colour * COLOUR_STRIDE + piece_type * PIECE_STRIDE + sq.index();

    // SAFETY: important invariant being upheld here!!
    assert!(
        idx < PSQT_FEATURES,
        "attempt to construct illegal FeatureIndex."
    );
    PsqtFeatureIndex(idx)
}

/// For non-merged king planes.
pub fn psqt_index_full(colour: Colour, king: Square, f: PsqtFeatureUpdate) -> usize {
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

/// wrapper to enforce bounds.
#[allow(clippy::module_name_repetitions)]
#[derive(Clone, Copy, Debug)]
pub struct ThreatFeatureIndex(u32);

impl ThreatFeatureIndex {
    /// Invariant: the result of this function is less than the number of NNUE threat features (60144),
    /// so it can be used to index a row of the feature-transformer matrix without bounds checking.
    #[allow(clippy::inline_always)]
    #[must_use]
    #[inline(always)]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// `PIECE_TARGET_MAP[i][j]` is –1 if the interaction between piece-types
/// `i` and `j` is fully excluded from the feature-set. At time of writing,
/// all interactions involving kings are fully-excluded, as are
/// 
/// - PAWN   → BISHOP
/// - PAWN   → QUEEN
/// - BISHOP → QUEEN
/// - ROOK   → QUEEN
/// 
/// One might imagine that the diagonal (sans pawns) ought to be
/// fully-excluded too, since if symmetric-piece A attacks same-type B,
/// then B attacks A, and so the feature is redundant – but setting the diagonal to
/// –1 in this instance would remove *both* the forward and backward features.
/// Instead, we use “semi-exclusion” later in the feature-indexing process,
/// whereby we use the direction of the threat to conditionally filter it.
/// (c.f. documentation of “forwards” and “backwards” threats in `threat_index`).
/// 
/// If `PIECE_TARGET_MAP[i][j]` is non-negative, it is an index TKTK
#[rustfmt::skip]
const PIECE_TARGET_MAP: [[i32; 6]; 6] = [
    [ 0,  1, -1,  2, -1, -1],
    [ 0,  1,  2,  3,  4, -1],
    [ 0,  1,  2,  3, -1, -1],
    [ 0,  1,  2,  3, -1, -1],
    [ 0,  1,  2,  3,  4, -1],
    [-1, -1, -1, -1, -1, -1],
];

/// `PIECE_TARGET_COUNT[i]` is the number of pieces that piece-type `i` can threaten,
/// according to `PIECE_TARGET_MAP`. This is the number of `Piece`s, not `PieceType`s,
/// so it can exceed 6.
const PIECE_TARGET_COUNT: [i32; 6] = {
    let mut count = [0; 6];
    cfor!(let mut i = 0; i < 6; i += 1; {
        let mut c = 0;
        cfor!(let mut j = 0; j < 6; j += 1; {
            if PIECE_TARGET_MAP[i][j] != -1 {
                c += 1;
            }
        });
        count[i] = 2 * c;
    });
    count
};

/// Given a piece, generate the table mapping a from-square and a to-square to
/// the number of squares attacked by that piece from the from-square that are
/// “backwards” from the to-square.
const fn generate_piece_index(piece: Piece) -> [[u8; 64]; 64] {
    #![expect(clippy::cast_possible_truncation)]

    let mut table = [[0; 64]; 64];

    cfor!(let mut from_index = 0; from_index < 64; from_index += 1; {
        let from = Square::new_clamped(from_index);
        // Create an attack-mask of this piece, standing on an empty board.
        let pseudo_attacks = attacks_by_type_slow(piece, from, SquareSet::EMPTY);
        cfor!(let mut to_index = 0; to_index < 64; to_index += 1; {
            // We then create a mask of all the places this piece attacks
            // that intersect with the squares *prior* to the second square.
            // e.g. if we have a queen in the center of the board, on D5:
            // \ . . | . . /
            // . \ . | . / .
            // . . \ | / . .
            // — — — Q — — —
            // . . / | \ . .
            // . / . | . \ .
            // / . . | . . \
            // . . . | . . .
            // and we select D4 as our `to_index`, we get this mask:
            // D4 = 27, D4.as_set() = 0x8000000, and subtracting one,
            // we get the mask 0x7ffffff, which looks like this:
            // . . . . . . .
            // . . . . . . .
            // . . . . . . .
            // X X X . . . .
            // X X X X X X X
            // X X X X X X X
            // X X X X X X X
            // as such, for [QUEEN][D5][D4], we have this:
            // . . . . . . .
            // . . . . . . .
            // . . . . . . .
            // . . . . . . .
            // . . / . . . .
            // . / . | . \ .
            // / . . | . . \
            // . . . | . . .
            let mask = pseudo_attacks.inner() & ((1u64 << to_index) - 1);
            // We then store the number of active squares, which in
            // the case of our example, is 3 + 3 + 2 = 8.
            table[from.index()][to_index] = mask.count_ones() as u8;
        });
    });

    table
}

/// This table stores, given a piece, a from-square, and a to-square,
/// the number of squares that piece attacks from the from-square that
/// are “backward” from the to-square.
static PIECE_INDEX: [[[u8; 64]; 64]; 12] = {
    let mut table = [[[0; 64]; 64]; 12];

    cfor!(let mut piece = 0; piece < 12; piece += 1; {
        // As all pieces bar pawns are symmetric, we
        // could reduce our calls to generate_piece_index
        // from 12 to 7, but this is perfectly neat.
        table[piece as usize] = generate_piece_index(Piece::from_index(piece).unwrap());
    });

    table
};

pub struct Offset {
    pub indices: [(i32, i32); 12],
    pub offsets: [[u32; 64]; 12],
}

static OFFSET: Offset = {
    let mut dst = Offset {
        indices: [(0, 0); 12],
        offsets: [[0; 64]; 12],
    };

    let mut offset = 0;

    cfor!(let mut colour = 0; colour < 2; colour += 1; {
        let colour = Colour::new(colour != 0);
        cfor!(let mut pt_idx = 0; pt_idx < 6; pt_idx += 1; {
            let piece_type = PieceType::new(pt_idx).unwrap();
            let piece = Piece::new(colour, piece_type);
            let mut piece_offset = 0;
            cfor!(let mut sq_idx = 0; sq_idx < 64; sq_idx += 1; {
                let sq = Square::new_clamped(sq_idx);
                dst.offsets[piece as usize][sq as usize] = piece_offset;
                if !matches!((piece_type, sq.rank()), (PieceType::Pawn, Rank::One | Rank::Eight)) {
                    let attacks = attacks_by_type_slow(piece, sq, SquareSet::EMPTY);
                    piece_offset += attacks.count();
                }
            });
            dst.indices[piece as usize] = (piece_offset as i32, offset);
            offset += PIECE_TARGET_COUNT[piece_type as usize] * piece_offset as i32;
        });
    });

    dst
};

static ATTACK_INDEX: [[[u32; 2]; 12]; 12] = {
    #[expect(clippy::cast_possible_truncation)]
    const FEATURES: u32 = THREAT_FEATURES as u32;

    let mut dst = [[[0; 2]; 12]; 12];

    cfor!(let mut attacker_idx = 0; attacker_idx < 12; attacker_idx += 1; {
        let attacker = Piece::from_index(attacker_idx).unwrap();
        cfor!(let mut victim_idx = 0; victim_idx < 12; victim_idx += 1; {
            let victim = Piece::from_index(victim_idx).unwrap();

            let opposed = attacker.colour() as u8 != victim.colour() as u8;
            let map = PIECE_TARGET_MAP[attacker.piece_type() as usize][victim.piece_type() as usize];

            let semi_excluded = attacker.piece_type() as u8 == victim.piece_type() as u8
                && (opposed || attacker.piece_type() as u8 != PieceType::Pawn as u8);
            let full_excluded = map == -1;

            let (piece_offset, offset) = OFFSET.indices[attacker as usize];

            let x = victim.colour().flip() as i32 * (PIECE_TARGET_COUNT[attacker.piece_type() as usize] / 2);
            assert!(x >= 0);
            let feature = offset + (x + map) * piece_offset;
            // assert!(feature >= 0); // failing for some reason
            let feature = feature as u32;

            dst[attacker as usize][victim as usize][0] = if full_excluded { FEATURES } else { feature };
            dst[attacker as usize][victim as usize][1] = if full_excluded || semi_excluded { FEATURES } else { feature };
        });
    });

    dst
};

/// Compute an index from 0 to 60143 representing the given threat, for use in the NNUE feature transformer.
pub fn threat_index(
    colour: Colour,
    // The king’s position is relevant for horizontal mirroring, but is not part of the feature index itself.
    king: Square,
    // The piece giving the threat.
    mut attacker: Piece,
    // The piece being threatened.
    mut victim: Piece,
    // The square upon which the attacker stands.
    mut from: Square,
    // The square being attacked (i.e. the victim’s square).
    mut to: Square,
) -> ThreatFeatureIndex {
    // All threat indices are reversed for black.
    if colour == Colour::Black {
        attacker = attacker.flip_colour();
        victim = victim.flip_colour();

        from = from.flip_rank();
        to = to.flip_rank();
    }

    // All features are mirrored when the king is on the right half of the board.
    if king.file() >= File::E {
        from = from.flip_file();
        to = to.flip_file();
    }

    // If two pieces of the same type are threatening each other,
    // this fact is deducible from the fact that the pieces are
    // of the same type, and one is attacking t’other. As such,
    // some feature indices are eliminable, and whether the attack
    // is “forwards” on the board is used to spot this kind of situation.
    // Obviously this does not hold for pawns.
    let forwards = from.index() < to.index();

    let attack_index = ATTACK_INDEX[attacker][victim][usize::from(forwards)];
    let offset = OFFSET.offsets[attacker][from];
    let piece_index = u32::from(PIECE_INDEX[attacker][from][to]);

    // SAFETY: important invariant being upheld here!!
    assert!(
        attack_index + offset + piece_index < 60144,
        "attempt to construct illegal ThreatFeatureIndex."
    );
    ThreatFeatureIndex(attack_index + offset + piece_index)
}
