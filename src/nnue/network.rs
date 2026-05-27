use std::{
    fmt::{Debug, Display},
    fs::{File, OpenOptions},
    hash::Hasher,
    io::{BufReader, BufWriter, Write},
    mem::size_of,
    ops::Deref,
    path::Path,
    sync::{Mutex, OnceLock},
    time::Duration,
};

use anyhow::Context;
use arrayvec::ArrayVec;
use memmap2::Mmap;

use crate::{
    chess::{
        board::{Board, movegen::attacks_by_type},
        piece::{Black, Col, Colour, Piece, PieceType, White},
        piecelayout::PieceLayout,
        types::Square,
    },
    image::{self, Image},
    nnue::{
        self,
        network::feature::{ThreatFeatureIndex, threat_index},
    },
    util::{Align, MAX_DEPTH},
};

use super::accumulator::{self, Accumulator};

pub mod feature;
pub mod layers;
pub mod threat_updates;

/// The embedded neural network parameters.
pub static EMBEDDED_NNUE: &[u8] = include_bytes_aligned!("../../viridithas.nnue.zst");

/// Whether the embedded network can be used verbatim.
pub const EMBEDDED_NNUE_VERBATIM: bool = false;
// Assertion for correctness of the embedded network:
const _: () = assert!(!EMBEDDED_NNUE_VERBATIM || EMBEDDED_NNUE.len() == size_of::<NNUEParams>());
/// Whether to perform the king-plane merging optimisation.
pub const MERGE_KING_PLANES: bool = true;
/// Whether the unquantised network has a feature factoriser.
pub const UNQUANTISED_HAS_FACTORISER: bool = true;
/// The number of features present in PSQT part of the input.
pub const PSQT_FEATURES: usize = (12 - MERGE_KING_PLANES as usize) * 64;
/// The number of features present in the threat part of the input.
pub const THREAT_FEATURES: usize = 60144;
/// The amount to scale the output of the network by.
/// This is to allow for the sigmoid activation to differentiate positions with
/// a small difference in evaluation.
pub const SCALE: i32 = 240;
/// The size of one-half of the hidden layer of the network.
pub const L1_SIZE: usize = 1024;
/// The size of the second layer of the network.
pub const L2_SIZE: usize = 32;
/// The size of the third layer of the network.
pub const L3_SIZE: usize = 32;
/// The number of output heads.
pub const HEADS: usize = 1;
/// The quantisation factor for the feature transformer weights.
const QA: i16 = 255;
/// The quantisation factor for the L1 weights.
const QB: i16 = 64;
/// Chunking constant for l1
pub const L1_CHUNK_PER_32: usize = size_of::<i32>() / size_of::<i8>();
/// The structure of the king-buckets.
#[rustfmt::skip]
const HALF_BUCKET_MAP: [usize; 32] = [
     0,  1,  2,  3,
     4,  5,  6,  7,
     8,  9, 10, 11,
     8,  9, 10, 11,
    12, 12, 13, 13,
    12, 12, 13, 13,
    14, 14, 15, 15,
    14, 14, 15, 15,
];
/// The number of buckets in the feature transformer.
pub const BUCKETS: usize = max!(HALF_BUCKET_MAP) + 1;
/// The number of output buckets
pub const OUTPUT_BUCKETS: usize = 8;
/// The mapping from square to bucket.
const BUCKET_MAP: [usize; 64] = {
    let mut map = [0; 64];
    let mut row = 0;
    while row < 8 {
        let mut col = 0;
        while col < 4 {
            let mirrored = 7 - col;
            map[row * 8 + col] = HALF_BUCKET_MAP[row * 4 + col];
            map[row * 8 + mirrored] = HALF_BUCKET_MAP[row * 4 + col] + BUCKETS;
            col += 1;
        }
        row += 1;
    }
    map
};

/// Get index into the output layer given a board state.
pub fn output_bucket(pos: &Board) -> usize {
    #![allow(clippy::cast_possible_truncation)]
    const DIVISOR: usize = usize::div_ceil(32, OUTPUT_BUCKETS);
    (pos.state.bbs.occupied().count() as usize - 2) / DIVISOR
}

pub fn nnue_checksum() -> u64 {
    let mut hasher = fxhash::FxHasher::default();
    hasher.write(EMBEDDED_NNUE);
    for index in REPERMUTE_INDICES {
        hasher.write_usize(index);
    }
    hasher.finish()
}

/// Struct representing the floating-point parameter file emitted by bullet.
#[rustfmt::skip]
#[repr(C)]
struct UnquantisedNetwork {
    l0_threat:     [f32; THREAT_FEATURES * L1_SIZE],
    // extra bucket for the feature-factoriser.
    l0_weights:    [f32; 12 * 64 * L1_SIZE * (BUCKETS + UNQUANTISED_HAS_FACTORISER as usize)],
    l0_biases:     [f32; L1_SIZE],
    l1_weights:  [[[f32; L2_SIZE]; OUTPUT_BUCKETS]; L1_SIZE],
    l1_biases:    [[f32; L2_SIZE]; OUTPUT_BUCKETS],
    l2x_weights: [[[f32; L3_SIZE * 2]; OUTPUT_BUCKETS]; L2_SIZE],
    l2f_weights:  [[f32; L3_SIZE * 2]; L2_SIZE],
    l2x_biases:   [[f32; L3_SIZE * 2]; OUTPUT_BUCKETS],
    l2f_biases:    [f32; L3_SIZE * 2],
    l3x_weights: [[[f32;   HEADS]; OUTPUT_BUCKETS]; L3_SIZE],
    l3f_weights:  [[f32;   HEADS]; L3_SIZE],
    l3x_biases:   [[f32;   HEADS]; OUTPUT_BUCKETS],
    l3f_biases:    [f32;   HEADS],
}

/// The floating-point parameters of the network, after de-factorisation.
#[rustfmt::skip]
#[repr(C)]
struct MergedNetwork {
    l0_threat:    [f32; THREAT_FEATURES * L1_SIZE],
    l0_weights:   [f32; 12 * 64 * L1_SIZE * BUCKETS],
    l0_biases:    [f32; L1_SIZE],
    l1_weights: [[[f32; L2_SIZE]; OUTPUT_BUCKETS]; L1_SIZE],
    l1_biases:   [[f32; L2_SIZE]; OUTPUT_BUCKETS],
    l2_weights: [[[f32; L3_SIZE * 2]; OUTPUT_BUCKETS]; L2_SIZE],
    l2_biases:   [[f32; L3_SIZE * 2]; OUTPUT_BUCKETS],
    l3_weights: [[[f32;   HEADS]; OUTPUT_BUCKETS]; L3_SIZE],
    l3_biases:   [[f32;   HEADS]; OUTPUT_BUCKETS],
}

/// A quantised network file, for compressed embedding.
#[rustfmt::skip]
#[repr(C)]
#[derive(PartialEq, Debug)]
struct QuantisedNetwork {
    l0_threat:    [ i8; THREAT_FEATURES * L1_SIZE],
    l0_weights:   [i16; PSQT_FEATURES * L1_SIZE * BUCKETS],
    l0_biases:    [i16; L1_SIZE],
    l1_weights: [[[ i8; L2_SIZE]; OUTPUT_BUCKETS]; L1_SIZE],
    l1_biases:   [[f32; L2_SIZE]; OUTPUT_BUCKETS],
    l2_weights: [[[f32; L3_SIZE * 2]; OUTPUT_BUCKETS]; L2_SIZE],
    l2_biases:   [[f32; L3_SIZE * 2]; OUTPUT_BUCKETS],
    l3_weights: [[[f32;   HEADS]; OUTPUT_BUCKETS]; L3_SIZE],
    l3_biases:   [[f32;   HEADS]; OUTPUT_BUCKETS],
}

/// The parameters of viri's neural network, quantised and permuted
/// for efficient SIMD inference.
#[rustfmt::skip]
#[repr(C)]
pub struct NNUEParams {
    pub l0_threat:    Align<[ i8; THREAT_FEATURES * L1_SIZE]>,
    pub l0_weights:   Align<[i16; PSQT_FEATURES * L1_SIZE * BUCKETS]>,
    pub l0_biases:    Align<[i16; L1_SIZE]>,
    pub l1_weights:  [Align<[ i8; L1_SIZE * L2_SIZE]>; OUTPUT_BUCKETS],
    pub l1_bias:     [Align<[f32; L2_SIZE]>; OUTPUT_BUCKETS],
    pub l2_weights:  [Align<[f32; L2_SIZE * L3_SIZE * 2]>; OUTPUT_BUCKETS],
    pub l2_bias:     [Align<[f32; L3_SIZE * 2]>; OUTPUT_BUCKETS],
    pub l3_weights: [[Align<[f32; L3_SIZE]>; HEADS]; OUTPUT_BUCKETS],
    pub l3_bias:             [[f32; HEADS]; OUTPUT_BUCKETS],
}

// const REPERMUTE_INDICES: [usize; L1_SIZE / 2] = {
//     let mut indices = [0; L1_SIZE / 2];
//     let mut i = 0;
//     while i < L1_SIZE / 2 {
//         indices[i] = i;
//         i += 1;
//     }
//     indices
// };

const REPERMUTE_INDICES: [usize; L1_SIZE / 2] = [
    154, 419, 117, 373, 57, 12, 446, 40, 505, 300, 476, 331, 428, 240, 268, 369, 41, 325, 313, 222,
    94, 6, 308, 125, 132, 388, 205, 186, 381, 442, 129, 430, 385, 200, 262, 291, 35, 199, 413, 157,
    139, 336, 206, 49, 250, 119, 252, 82, 318, 62, 367, 111, 483, 364, 169, 425, 108, 227, 195, 79,
    18, 146, 431, 488, 382, 481, 352, 141, 50, 239, 21, 201, 55, 326, 311, 70, 114, 370, 179, 66,
    435, 324, 292, 355, 255, 254, 456, 510, 174, 91, 259, 3, 193, 409, 153, 89, 345, 461, 151, 277,
    407, 116, 400, 107, 289, 33, 20, 467, 273, 176, 90, 104, 130, 228, 103, 109, 471, 135, 436,
    219, 264, 8, 112, 283, 485, 229, 339, 281, 25, 83, 118, 31, 22, 334, 427, 258, 164, 37, 295,
    242, 226, 63, 28, 166, 77, 349, 246, 74, 322, 67, 4, 177, 315, 178, 34, 185, 187, 95, 127, 113,
    297, 383, 80, 302, 46, 511, 478, 350, 69, 275, 172, 496, 75, 249, 296, 220, 272, 417, 375, 444,
    44, 338, 188, 19, 354, 98, 190, 508, 76, 140, 396, 68, 332, 341, 85, 347, 270, 14, 86, 450, 99,
    1, 245, 457, 406, 499, 390, 210, 466, 138, 150, 243, 5, 233, 72, 149, 52, 152, 399, 96, 506,
    225, 212, 232, 274, 58, 126, 92, 395, 489, 455, 134, 405, 194, 257, 192, 328, 468, 261, 42, 87,
    459, 59, 509, 279, 429, 445, 115, 310, 408, 319, 372, 298, 238, 122, 378, 294, 38, 386, 365,
    155, 215, 448, 368, 180, 15, 490, 303, 342, 64, 434, 290, 495, 441, 501, 216, 415, 7, 30, 286,
    423, 161, 263, 159, 16, 472, 36, 271, 121, 320, 47, 234, 475, 253, 173, 189, 371, 54, 235, 65,
    491, 321, 453, 282, 26, 217, 183, 473, 348, 439, 265, 477, 301, 361, 61, 389, 380, 307, 440,
    487, 458, 379, 247, 503, 327, 241, 454, 56, 424, 362, 81, 136, 32, 24, 337, 23, 280, 203, 288,
    392, 197, 17, 432, 346, 360, 312, 498, 168, 497, 437, 218, 363, 414, 43, 244, 421, 165, 102,
    358, 224, 480, 73, 162, 71, 329, 418, 317, 175, 397, 402, 181, 398, 182, 330, 391, 204, 460,
    299, 110, 493, 84, 45, 156, 393, 256, 128, 377, 462, 27, 167, 394, 492, 142, 470, 438, 276,
    211, 314, 29, 500, 494, 309, 251, 160, 209, 465, 416, 507, 267, 266, 469, 452, 10, 404, 353,
    148, 97, 196, 106, 11, 213, 120, 198, 145, 131, 479, 230, 486, 223, 144, 401, 474, 376, 359,
    484, 433, 387, 260, 158, 447, 366, 133, 143, 105, 124, 184, 51, 231, 123, 202, 53, 482, 285,
    284, 422, 93, 333, 502, 411, 0, 344, 88, 384, 214, 323, 237, 412, 137, 340, 207, 306, 463, 191,
    343, 60, 170, 356, 449, 464, 504, 403, 304, 269, 13, 147, 48, 248, 208, 221, 236, 426, 305,
    100, 316, 443, 9, 374, 351, 293, 420, 39, 2, 171, 287, 335, 451, 78, 278, 410, 163, 101, 357,
];

impl UnquantisedNetwork {
    /// Convert a parameter file generated by bullet into a merged parameter set,
    /// for further processing or for resuming training in a more efficient format.
    #[expect(clippy::too_many_lines)]
    fn merge(&self) -> Box<MergedNetwork> {
        #![allow(clippy::similar_names)]

        let mut net = MergedNetwork::zeroed();
        let mut buckets = self.l0_weights.chunks_exact(12 * 64 * L1_SIZE);
        let factoriser;
        let alternate_buffer;
        if UNQUANTISED_HAS_FACTORISER {
            factoriser = buckets.next().unwrap();
        } else {
            alternate_buffer = vec![0.0; 12 * 64 * L1_SIZE];
            factoriser = &alternate_buffer;
        }
        for (src_bucket, tgt_bucket) in
            buckets.zip(net.l0_weights.chunks_exact_mut(12 * 64 * L1_SIZE))
        {
            for piece in Piece::all() {
                for sq in Square::all() {
                    let i = feature::psqt_index_full(
                        Colour::White,
                        Square::A1,
                        PsqtFeatureUpdate { sq, piece },
                    );
                    let j = feature::psqt_index_full(
                        Colour::White,
                        Square::A1,
                        PsqtFeatureUpdate { sq, piece },
                    );
                    let src = &src_bucket[i * L1_SIZE..i * L1_SIZE + L1_SIZE];
                    let fac_src = &factoriser[i * L1_SIZE..i * L1_SIZE + L1_SIZE];
                    let tgt = &mut tgt_bucket[j * L1_SIZE..j * L1_SIZE + L1_SIZE];
                    for ((src, fac_src), tgt) in src.iter().zip(fac_src).zip(tgt) {
                        *tgt = *src + *fac_src;
                    }
                }
            }
        }

        // copy the threat weights
        for i in 0..THREAT_FEATURES * L1_SIZE {
            net.l0_threat[i] = self.l0_threat[i];
        }

        // copy the biases
        net.l0_biases.copy_from_slice(&self.l0_biases);
        // copy the L1 weights
        for i in 0..L1_SIZE {
            for bucket in 0..OUTPUT_BUCKETS {
                for j in 0..L2_SIZE {
                    net.l1_weights[i][bucket][j] = self.l1_weights[i][bucket][j];
                }
            }
        }
        // copy the L1 biases
        for i in 0..L2_SIZE {
            for bucket in 0..OUTPUT_BUCKETS {
                net.l1_biases[bucket][i] = self.l1_biases[bucket][i];
            }
        }
        // copy the L2 weights
        for i in 0..L2_SIZE {
            for bucket in 0..OUTPUT_BUCKETS {
                for j in 0..L3_SIZE * 2 {
                    net.l2_weights[i][bucket][j] =
                        self.l2x_weights[i][bucket][j] + self.l2f_weights[i][j];
                }
            }
        }
        // copy the L2 biases
        for i in 0..L3_SIZE * 2 {
            for bucket in 0..OUTPUT_BUCKETS {
                net.l2_biases[bucket][i] = self.l2x_biases[bucket][i] + self.l2f_biases[i];
            }
        }
        // copy the L3 weights
        for i in 0..L3_SIZE {
            for bucket in 0..OUTPUT_BUCKETS {
                for head in 0..HEADS {
                    net.l3_weights[i][bucket][head] =
                        self.l3x_weights[i][bucket][head] + self.l3f_weights[i][head];
                }
            }
        }
        // copy the L3 biases
        for head in 0..HEADS {
            for i in 0..OUTPUT_BUCKETS {
                net.l3_biases[i][head] = self.l3x_biases[i][head] + self.l3f_biases[head];
            }
        }

        let range = |slice: &[f32]| {
            let init = (f32::INFINITY, f32::NEG_INFINITY);
            slice
                .iter()
                .copied()
                .fold(init, |(min, max), v| (min.min(v), max.max(v)))
        };

        let (l0w_min, l0w_max) = range(&net.l0_weights);
        let (l0b_min, l0b_max) = range(&net.l0_biases);
        println!("L0 weight range: [{l0w_min}, {l0w_max}]");
        println!("L0 bias range: [{l0b_min}, {l0b_max}]");

        let l1_weights_flat = net.l1_weights.as_flattened().as_flattened();
        let l1_biases_flat = net.l1_biases.as_flattened();
        let (l1w_min, l1w_max) = range(l1_weights_flat);
        let (l1b_min, l1b_max) = range(l1_biases_flat);
        println!("L1 weight range: [{l1w_min}, {l1w_max}]");
        println!("L1 bias range: [{l1b_min}, {l1b_max}]");

        let l2_weights_flat = net.l2_weights.as_flattened().as_flattened();
        let l2_biases_flat = net.l2_biases.as_flattened();
        let (l2w_min, l2w_max) = range(l2_weights_flat);
        let (l2b_min, l2b_max) = range(l2_biases_flat);
        println!("L2 weight range: [{l2w_min}, {l2w_max}]");
        println!("L2 bias range: [{l2b_min}, {l2b_max}]");

        let l3_weights_flat = net.l3_weights.as_flattened().as_flattened();
        let l3_biases_flat = net.l3_biases.as_flattened();
        let (l3w_min, l3w_max) = range(l3_weights_flat);
        let (l3b_min, l3b_max) = range(l3_biases_flat);
        println!("L3 weight range: [{l3w_min}, {l3w_max}]");
        println!("L3 bias range: [{l3b_min}, {l3b_max}]");

        net
    }

    fn zeroed() -> Box<Self> {
        // SAFETY: UnquantisedNetwork can be zeroed.
        unsafe {
            let layout = std::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc_zeroed(layout);
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Box::from_raw(ptr.cast())
        }
    }

    fn read(reader: &mut impl std::io::Read) -> anyhow::Result<Box<Self>> {
        // SAFETY: NNUEParams can be zeroed.
        unsafe {
            let mut net = Self::zeroed();
            let mem = std::slice::from_raw_parts_mut(
                std::ptr::from_mut(net.as_mut()).cast::<u8>(),
                size_of::<Self>(),
            );
            reader.read_exact(mem)?;
            Ok(net)
        }
    }
}

impl MergedNetwork {
    fn zeroed() -> Box<Self> {
        // SAFETY: NNUEParams can be zeroed.
        unsafe {
            let layout = std::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc_zeroed(layout);
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Box::from_raw(ptr.cast())
        }
    }

    fn write_bullet(&self, writer: &mut impl std::io::Write) -> anyhow::Result<()> {
        macro_rules! dump_layer {
            ($name:expr, $field:expr) => {
                writeln!(writer, $name)?;
                let len = size_of_val(&$field) / size_of::<f32>();
                // SAFETY: the field is a contiguous array of f32, so casting
                // to *const f32 is valid, and `len` is size_of_val / size_of::<f32>().
                let slice = unsafe {
                    let ptr = $field.as_ptr().cast::<f32>();
                    std::slice::from_raw_parts(ptr, len)
                };
                writer.write_all(&usize::to_le_bytes(len))?;
                for val in slice {
                    writer.write_all(&val.to_le_bytes())?;
                }
            };
        }

        dump_layer!("l0w", self.l0_weights);
        dump_layer!("l0b", self.l0_biases);
        dump_layer!("l1w", self.l1_weights);
        dump_layer!("l1b", self.l1_biases);
        dump_layer!("l2w", self.l2_weights);
        dump_layer!("l2b", self.l2_biases);
        dump_layer!("l3w", self.l3_weights);
        dump_layer!("l3b", self.l3_biases);

        Ok(())
    }

    #[allow(clippy::cast_possible_truncation, clippy::assertions_on_constants)]
    fn quantise(&self) -> Box<QuantisedNetwork> {
        const QA_BOUND: f32 = 1.98 * QA as f32;
        const QB_BOUND: f32 = 1.98 * QB as f32;

        let mut net = QuantisedNetwork::zeroed();
        // quantise the feature transformer weights.
        let buckets = self.l0_weights.chunks_exact(12 * 64 * L1_SIZE);

        for (bucket_idx, (src_bucket, tgt_bucket)) in buckets
            .zip(net.l0_weights.chunks_exact_mut(PSQT_FEATURES * L1_SIZE))
            .enumerate()
        {
            // for repermuting the weights.
            let mut things_written = 0;
            for piece in Piece::all() {
                for sq in Square::all() {
                    // don't write black king data into the white king's slots
                    let in_bucket = BUCKET_MAP[sq] == bucket_idx;
                    if MERGE_KING_PLANES && in_bucket && piece == Piece::BK {
                        continue;
                    }
                    // don't write white king data into the black king's slots
                    if MERGE_KING_PLANES && !in_bucket && piece == Piece::WK {
                        continue;
                    }
                    let i = feature::psqt_index_full(
                        Colour::White,
                        Square::A1,
                        PsqtFeatureUpdate { sq, piece },
                    );
                    let j = feature::psqt_index(
                        Colour::White,
                        Square::A1,
                        PsqtFeatureUpdate { sq, piece },
                    )
                    .index();
                    assert!(
                        MERGE_KING_PLANES || i == j,
                        "if not merging the king planes, indices should match"
                    );
                    let src = &src_bucket[i * L1_SIZE..i * L1_SIZE + L1_SIZE];
                    let tgt = &mut tgt_bucket[j * L1_SIZE..j * L1_SIZE + L1_SIZE];
                    for (src, tgt) in src.iter().zip(tgt) {
                        // extra clamp in case bucket + factoriser goes out of the clipping bounds
                        let scaled = f32::clamp(*src, -1.98, 1.98) * f32::from(QA);
                        *tgt = scaled.round() as i16;
                    }
                    things_written += 1;
                }
            }
            assert_eq!(PSQT_FEATURES, things_written);
        }

        // transfer the threat plane weights:
        for (src, tgt) in self.l0_threat.iter().zip(net.l0_threat.iter_mut()) {
            let scaled = *src * f32::from(QA);
            if scaled.abs() > QA_BOUND {
                eprintln!("threat plane weight {scaled} is too large (max = {QA_BOUND})");
            }
            // directly hard-quantised to i8.
            *tgt = scaled.clamp(f32::from(i8::MIN), f32::from(i8::MAX)).round() as i8;
        }

        // quantise the FT biases
        for (src, tgt) in self.l0_biases.iter().zip(net.l0_biases.iter_mut()) {
            let scaled = *src * f32::from(QA);
            if scaled.abs() > QA_BOUND {
                eprintln!("feature transformer bias {scaled} is too large (max = {QA_BOUND})");
            }
            *tgt = scaled.clamp(-QA_BOUND, QA_BOUND).round() as i16;
        }

        // quantise the l1 weights
        for i in 0..L1_SIZE {
            for bucket in 0..OUTPUT_BUCKETS {
                for j in 0..L2_SIZE {
                    let v = self.l1_weights[i][bucket][j] * f32::from(QB);
                    if v.abs() > QB_BOUND {
                        eprintln!("L1 weight {v} is too large (max = {QB_BOUND})");
                    }
                    let v = v.clamp(-QB_BOUND, QB_BOUND).round() as i8;
                    net.l1_weights[i][bucket][j] = v;
                }
            }
        }

        // transfer the f32 components of the network
        net.l1_biases = self.l1_biases;
        net.l2_weights = self.l2_weights;
        net.l2_biases = self.l2_biases;
        net.l3_weights = self.l3_weights;
        net.l3_biases = self.l3_biases;

        net
    }
}

impl QuantisedNetwork {
    /// Convert the network parameters into a format optimal for inference.
    #[allow(
        clippy::cognitive_complexity,
        clippy::needless_range_loop,
        clippy::too_many_lines
    )]
    fn permute(&self, use_simd: bool) -> Box<NNUEParams> {
        let mut net = NNUEParams::zeroed();
        // permute the feature transformer weights
        let src_buckets = self.l0_weights.chunks_exact(PSQT_FEATURES * L1_SIZE);
        let tgt_buckets = net.l0_weights.chunks_exact_mut(PSQT_FEATURES * L1_SIZE);
        for (src_bucket, tgt_bucket) in src_buckets.zip(tgt_buckets) {
            repermute_ft_bucket(tgt_bucket, src_bucket);
        }

        // permute the feature transformer biases
        repermute_ft_bias(&mut net.l0_biases, &self.l0_biases);

        // repermute the threat plane weights
        repermute_threat_weights(&mut net.l0_threat, &self.l0_threat);

        // transpose FT weights and biases so that packus transposes it back to the intended order
        if use_simd {
            type PermChunk<I> = [I; 8];
            // reinterpret as data of size __m128i
            let mut weights: Vec<&mut PermChunk<i16>> = net
                .l0_weights
                .chunks_exact_mut(8)
                .map(|a| a.try_into().unwrap())
                .collect();
            let mut biases: Vec<&mut PermChunk<i16>> = net
                .l0_biases
                .chunks_exact_mut(8)
                .map(|a| a.try_into().unwrap())
                .collect();
            let num_chunks = size_of::<PermChunk<i16>>() / size_of::<i16>();

            #[cfg(target_feature = "avx512f")]
            let num_regs = 8;
            #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
            let num_regs = 4;
            #[cfg(all(
                target_arch = "x86_64",
                not(target_feature = "avx2"),
                not(target_feature = "avx512f")
            ))]
            let num_regs = 2;
            #[cfg(target_feature = "neon")]
            let num_regs = 2;
            #[cfg(not(any(target_arch = "x86_64", target_feature = "neon")))]
            let num_regs = 1;
            #[cfg(target_feature = "avx512f")]
            let order = [0, 2, 4, 6, 1, 3, 5, 7];
            #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
            let order = [0, 2, 1, 3];
            #[cfg(all(
                target_arch = "x86_64",
                not(target_feature = "avx2"),
                not(target_feature = "avx512f")
            ))]
            let order = [0, 1];
            #[cfg(target_feature = "neon")]
            let order = [0, 1];
            #[cfg(not(any(target_arch = "x86_64", target_feature = "neon")))]
            let order = [0];

            let mut regs = vec![[0i16; 8]; num_regs];

            // transpose weights
            for i in (0..PSQT_FEATURES * L1_SIZE * BUCKETS / num_chunks).step_by(num_regs) {
                for j in 0..num_regs {
                    regs[j] = *weights[i + j];
                }

                for j in 0..num_regs {
                    *weights[i + j] = regs[order[j]];
                }
            }

            // transpose biases
            for i in (0..L1_SIZE / num_chunks).step_by(num_regs) {
                for j in 0..num_regs {
                    regs[j] = *biases[i + j];
                }

                for j in 0..num_regs {
                    *biases[i + j] = regs[order[j]];
                }
            }

            let mut i8_regs = vec![[0i8; 8]; num_regs];

            // now the same for the threat plane weights
            let mut threat_weights: Vec<&mut PermChunk<i8>> =
                net.l0_threat.as_chunks_mut::<8>().0.iter_mut().collect();
            for i in (0..THREAT_FEATURES * L1_SIZE / num_chunks).step_by(num_regs) {
                for j in 0..num_regs {
                    i8_regs[j] = *threat_weights[i + j];
                }

                for j in 0..num_regs {
                    *threat_weights[i + j] = i8_regs[order[j]];
                }
            }
        }

        // transpose the L{1,2,3} weights and biases
        let mut sorted = vec![[[0i8; L2_SIZE]; OUTPUT_BUCKETS]; L1_SIZE];
        repermute_l1_weights(&mut sorted, &self.l1_weights);
        for bucket in 0..OUTPUT_BUCKETS {
            // quant the L1 weights
            if use_simd {
                for i in 0..L1_SIZE / L1_CHUNK_PER_32 {
                    for j in 0..L2_SIZE {
                        for k in 0..L1_CHUNK_PER_32 {
                            net.l1_weights[bucket]
                                [i * L1_CHUNK_PER_32 * L2_SIZE + j * L1_CHUNK_PER_32 + k] =
                                sorted[i * L1_CHUNK_PER_32 + k][bucket][j];
                        }
                    }
                }
            } else {
                for i in 0..L1_SIZE {
                    for j in 0..L2_SIZE {
                        net.l1_weights[bucket][j * L1_SIZE + i] = sorted[i][bucket][j];
                    }
                }
            }

            // transfer the L1 biases
            for i in 0..L2_SIZE {
                net.l1_bias[bucket][i] = self.l1_biases[bucket][i];
            }

            // transpose the L2 weights
            for i in 0..L2_SIZE {
                for j in 0..L3_SIZE * 2 {
                    net.l2_weights[bucket][i * L3_SIZE * 2 + j] = self.l2_weights[i][bucket][j];
                }
            }

            // transfer the L2 biases
            for i in 0..L3_SIZE * 2 {
                net.l2_bias[bucket][i] = self.l2_biases[bucket][i];
            }

            // transfer the L3 weights
            for i in 0..L3_SIZE {
                for head in 0..HEADS {
                    net.l3_weights[bucket][head][i] = self.l3_weights[i][bucket][head];
                }
            }

            // transfer the L3 biases
            net.l3_bias[bucket] = self.l3_biases[bucket];
        }

        net
    }

    fn zeroed() -> Box<Self> {
        // SAFETY: NNUEParams can be zeroed.
        unsafe {
            let layout = std::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc_zeroed(layout);
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Box::from_raw(ptr.cast())
        }
    }

    fn write(&self, writer: &mut impl std::io::Write) -> anyhow::Result<()> {
        let ptr = std::ptr::from_ref::<Self>(self).cast::<u8>();
        let len = size_of::<Self>();
        // SAFETY: We're writing a slice of bytes, and we know that the slice is valid.
        writer.write_all(unsafe { std::slice::from_raw_parts(ptr, len) })?;
        Ok(())
    }
}

fn repermute_l1_weights(
    sorted: &mut [[[i8; L2_SIZE]; OUTPUT_BUCKETS]],
    l1_weights: &[[[i8; L2_SIZE]; OUTPUT_BUCKETS]; L1_SIZE],
) {
    for (tgt_index, src_index) in REPERMUTE_INDICES.iter().copied().enumerate() {
        sorted[tgt_index] = l1_weights[src_index];
    }
    for (tgt_index, src_index) in REPERMUTE_INDICES.iter().copied().enumerate() {
        sorted[tgt_index + L1_SIZE / 2] = l1_weights[src_index + L1_SIZE / 2];
    }
}

fn repermute_ft_bias(feature_bias: &mut [i16; L1_SIZE], unsorted: &[i16]) {
    for (tgt_index, src_index) in REPERMUTE_INDICES.iter().copied().enumerate() {
        feature_bias[tgt_index] = unsorted[src_index];
    }
    for (tgt_index, src_index) in REPERMUTE_INDICES.iter().copied().enumerate() {
        feature_bias[tgt_index + L1_SIZE / 2] = unsorted[src_index + L1_SIZE / 2];
    }
}

fn repermute_ft_bucket(tgt_bucket: &mut [i16], unsorted: &[i16]) {
    // for each input feature,
    for i in 0..PSQT_FEATURES {
        // for each neuron in the layer,
        for (tgt_index, src_index) in REPERMUTE_INDICES.iter().copied().enumerate() {
            // get the neuron's corresponding weight in the unsorted bucket,
            // and write it to the same feature (but the new position) in the target bucket.
            let feature = i * L1_SIZE;
            tgt_bucket[feature + tgt_index] = unsorted[feature + src_index];
        }
        for (tgt_index, src_index) in REPERMUTE_INDICES.iter().copied().enumerate() {
            let tgt_index = tgt_index + L1_SIZE / 2;
            let src_index = src_index + L1_SIZE / 2;
            // get the neuron's corresponding weight in the unsorted bucket,
            // and write it to the same feature (but the new position) in the target bucket.
            let feature = i * L1_SIZE;
            tgt_bucket[feature + tgt_index] = unsorted[feature + src_index];
        }
    }
}

fn repermute_threat_weights(
    tgt: &mut Align<[i8; THREAT_FEATURES * L1_SIZE]>,
    unsorted: &[i8; THREAT_FEATURES * L1_SIZE],
) {
    for i in 0..THREAT_FEATURES {
        for (tgt_index, src_index) in REPERMUTE_INDICES.iter().copied().enumerate() {
            let feature = i * L1_SIZE;
            tgt[feature + tgt_index] = unsorted[feature + src_index];
        }
        for (tgt_index, src_index) in REPERMUTE_INDICES.iter().copied().enumerate() {
            let tgt_index = tgt_index + L1_SIZE / 2;
            let src_index = src_index + L1_SIZE / 2;
            let feature = i * L1_SIZE;
            tgt[feature + tgt_index] = unsorted[feature + src_index];
        }
    }
}

impl NNUEParams {
    #[allow(clippy::too_many_lines)]
    pub fn decompress_and_alloc() -> anyhow::Result<&'static Self> {
        #[cfg(not(feature = "zstd"))]
        type ZstdDecoder<R, D> = ruzstd::decoding::StreamingDecoder<R, D>;
        #[cfg(feature = "zstd")]
        type ZstdDecoder<'a, R> = zstd::stream::Decoder<'a, R>;

        // this function is not particularly happy about running in parallel.
        static LOCK: Mutex<()> = Mutex::new(());
        // additionally, we'd quite like to cache the results of this function.
        static CACHED: OnceLock<Mmap> = OnceLock::new();

        if EMBEDDED_NNUE_VERBATIM {
            // if we're using the verbatim network, we don't need to decompress anything.
            // just return a reference to the static network.
            // SAFETY: The static network is valid for the lifetime of the program,
            // and is the same size as the NNUEParams struct.
            #[allow(clippy::cast_ptr_alignment)]
            unsafe {
                let ptr = EMBEDDED_NNUE.as_ptr();
                assert_eq!(
                    size_of::<Self>(),
                    EMBEDDED_NNUE.len(),
                    "Verbatim NNUE is not the right size, expected {} bytes, got {} bytes",
                    size_of::<Self>(),
                    EMBEDDED_NNUE.len()
                );
                assert_eq!(
                    ptr.align_offset(64),
                    0,
                    "Embedded NNUE is not aligned to 64 bytes, ptr is {ptr:p}"
                );
                // SAFETY: We know that the pointer is valid and aligned.
                return Ok(&*ptr.cast::<Self>());
            }
        }

        let _guard = LOCK.lock().unwrap();
        // check if we've already loaded the weights
        if let Some(cached) = CACHED.get() {
            // cast the mmap to a NNUEParams
            // SAFETY: We check that the mmap is the right size and alignment.
            #[allow(clippy::cast_ptr_alignment)]
            let params: &'static Self = unsafe { &*cached.as_ptr().cast::<Self>() };

            return Ok(params);
        }

        let weights_file_name = format!(
            "viridithas-shared-network-weights-{}-{}-{}-{:X}.bin",
            std::env::consts::ARCH,
            std::env::consts::OS,
            // target cpu
            nnue::simd::ARCH,
            // avoid clashing with other versions
            nnue_checksum(),
        );

        let temp_dir = std::env::temp_dir();
        let weights_path = temp_dir.join(&weights_file_name);

        // Try to open existing weights file
        let exists = weights_path
            .try_exists()
            .with_context(|| format!("Could not check existence of {}", weights_path.display()))?;

        if exists {
            let mmap = Self::map_weight_file(&weights_path).with_context(|| {
                format!(
                    "Failed while attempting to load pre-existing weight file at {}",
                    weights_path.display()
                )
            })?;

            // store the mmap in the cache
            CACHED.set(mmap).unwrap();

            // cast the mmap to a NNUEParams
            // SAFETY: We check that the mmap is the right size and alignment.
            #[allow(clippy::cast_ptr_alignment)]
            let params: &'static Self = unsafe { &*CACHED.get().unwrap().as_ptr().cast::<Self>() };

            return Ok(params);
        }

        let mut net = QuantisedNetwork::zeroed();
        // SAFETY: QN is POD and we only write to it.
        let mut mem = unsafe {
            std::slice::from_raw_parts_mut(
                std::ptr::from_mut(net.as_mut()).cast::<u8>(),
                size_of::<QuantisedNetwork>(),
            )
        };
        let expected_bytes = mem.len() as u64;
        let decoding_start = std::time::Instant::now();
        let mut decoder = ZstdDecoder::new(EMBEDDED_NNUE)
            .with_context(|| "Failed to construct zstd decoder for NNUE weights.")?;
        let bytes_written = std::io::copy(&mut decoder, &mut mem)
            .with_context(|| "Failed to decompress NNUE weights.")?;
        let decoding_time = decoding_start.elapsed();
        println!(
            "info string decompressed NNUE weights in {}us",
            decoding_time.as_micros()
        );
        anyhow::ensure!(
            bytes_written == expected_bytes,
            "encountered issue while decompressing NNUE weights, expected {expected_bytes} bytes, but got {bytes_written}"
        );
        let use_simd = cfg!(any(target_arch = "x86_64", target_feature = "neon"));
        let net = net.permute(use_simd);

        // create a temporary file to store the weights
        // uses a path unique to our process to avoid
        // a race condition where one process is quicker at
        // writing the file than another.
        let temp_path = weights_path.with_extension(format!("tmp.{}", std::process::id()));

        // If we get here, we need to create and populate the weights file
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            // use a temporary path to avoid race conditions
            .open(&temp_path)
            .with_context(|| format!("Failed to open temporary file at {}", temp_path.display()))?;

        // Allocate the file to the right size
        let size = size_of::<Self>();
        file.set_len(size as u64).with_context(|| {
            format!(
                "Failed to set length of file at {} to {size}",
                temp_path.display()
            )
        })?;

        // SAFETY: This file must not be modified while we have a reference to it.
        // we avoid doing this ourselves, but we can't defend against other processes.
        let mut mmap = unsafe {
            memmap2::MmapOptions::new()
                .map_mut(&file)
                .with_context(|| format!("Failed to map temp file at {}", temp_path.display()))?
        };

        // Verify that the pointer is aligned to 64 bytes
        anyhow::ensure!(
            mmap.as_ptr().align_offset(64) == 0,
            "Temporary file mmap pointer is not aligned to 64 bytes"
        );

        // write the NNUEParams to the mmap
        #[allow(clippy::cast_ptr_alignment)]
        let ptr = mmap.as_mut_ptr().cast::<Self>();
        // SAFETY: We just allocated the mmap, and we know that the pointer is aligned to 64 bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(net.as_ref(), ptr, 1);
        }

        // sync the file to disk
        mmap.flush().with_context(|| {
            format!(
                "Failed to flush mmaped temporary file at {}",
                temp_path.display()
            )
        })?;

        // move the file to the correct path
        let rename_result = std::fs::rename(&temp_path, &weights_path);

        // if the file now exists, either we succeeded or got beaten to the punch:
        let exists = weights_path
            .try_exists()
            .with_context(|| format!("Could not check existence of {}", weights_path.display()))?;

        if !exists {
            let tfile = temp_path.file_name().unwrap_or_else(|| "<empty>".as_ref());
            let wfile = weights_path
                .file_name()
                .unwrap_or_else(|| "<empty>".as_ref());

            rename_result.with_context(|| {
                format!(
                    "Failed to rename temp file from {} to {} in {}",
                    tfile.display(),
                    wfile.display(),
                    temp_dir.display()
                )
            })?;

            panic!("Somehow rename succeeded but the file doesn't exist!");
        }

        #[cfg(debug_assertions)]
        {
            // log that we've created the file freshly
            println!(
                "Created NNUE weights file at {} from decompressed data",
                weights_path.display()
            );
        }

        // file created, return the mapped weights
        let mmap = Self::map_weight_file(&weights_path).with_context(|| {
            format!(
                "Failed while attempting to load just-created weight file at {}",
                weights_path.display()
            )
        })?;

        // store the mmap in the cache
        CACHED.set(mmap).unwrap();

        // cast the mmap to a NNUEParams
        // SAFETY: We check that the mmap is the right size and alignment.
        #[allow(clippy::cast_ptr_alignment)]
        let params: &'static Self = unsafe { &*CACHED.get().unwrap().as_ptr().cast::<Self>() };

        Ok(params)
    }

    fn map_weight_file(weights_path: &Path) -> anyhow::Result<Mmap> {
        let without_full_ext = weights_path.with_extension("tmp");
        let without_full_ext = without_full_ext.as_os_str().to_string_lossy();

        // wait until there are no temporary files left
        //
        // this is a bit of a hack, but it's the best way to ensure that the file is
        // fully written before we try to use it.
        let temp_dir_path = weights_path.parent().with_context(|| {
            format!(
                "Weights path ({}) is not in a directory!",
                weights_path.display()
            )
        })?;
        while std::fs::read_dir(temp_dir_path)
            .with_context(|| {
                format!(
                    "Failed to read temporary directory at {}",
                    temp_dir_path.display()
                )
            })?
            .filter_map(Result::ok)
            .any(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .contains(&*without_full_ext)
            })
        {
            std::thread::sleep(Duration::from_millis(100));
        }

        let file = File::open(weights_path).with_context(|| {
            format!("Failed to open weights file at {}", weights_path.display())
        })?;
        // SAFETY: This file must not be modified while we have a reference to it.
        // we avoid doing this ourselves, but we can't defend against other processes.
        let mmap = unsafe {
            memmap2::MmapOptions::new().map(&file).with_context(|| {
                format!("Failed to map weights file at {}", weights_path.display())
            })?
        };

        anyhow::ensure!(
            mmap.len() == size_of::<Self>(),
            "Wrong number of bytes: expected {}, got {}",
            size_of::<Self>(),
            mmap.len()
        );

        anyhow::ensure!(
            mmap.as_ptr().align_offset(64) == 0,
            "Pointer is not aligned to 64 bytes"
        );

        #[cfg(debug_assertions)]
        {
            // log the address of the mmap with pointer formatting
            println!(
                "Loaded NNUE weights from mmap at {:p} from file {}",
                mmap.as_ptr(),
                weights_path.display()
            );
        }

        Ok(mmap)
    }

    fn zeroed() -> Box<Self> {
        // SAFETY: NNUEParams can be zeroed.
        unsafe {
            let layout = std::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc_zeroed(layout);
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Box::from_raw(ptr.cast())
        }
    }

    pub fn select_feature_weights(&self, bucket: usize) -> &Align<[i16; PSQT_FEATURES * L1_SIZE]> {
        // handle mirroring
        let bucket = bucket % BUCKETS;
        let start = bucket * PSQT_FEATURES * L1_SIZE;
        let end = start + PSQT_FEATURES * L1_SIZE;
        let slice = &self.l0_weights[start..end];
        // SAFETY: The resulting slice is indeed INPUT * LAYER_1_SIZE long,
        // and we check that the slice is aligned to 64 bytes.
        // additionally, we're generating the reference from our own data,
        // so we know that the lifetime is valid.
        unsafe {
            // don't immediately cast to Align64, as we want to check the alignment first.
            let ptr = slice.as_ptr();
            assert_eq!(ptr.align_offset(64), 0);
            // alignments are sensible, so we can safely cast.
            #[allow(clippy::cast_ptr_alignment)]
            &*ptr.cast()
        }
    }
}

pub fn quantise(input: &std::path::Path, output: &std::path::Path) -> anyhow::Result<()> {
    let input_file =
        File::open(input).with_context(|| format!("Failed to open file at {}", input.display()))?;
    let mut reader = BufReader::new(input_file);
    let mut writer = File::create(output)
        .with_context(|| format!("Failed to create file at {}", output.display()))?;
    let unquantised_net = UnquantisedNetwork::read(&mut reader)?;
    let net = unquantised_net.merge().quantise();
    net.write(&mut writer)?;
    Ok(())
}

pub fn merge(input: &std::path::Path, output: &std::path::Path) -> anyhow::Result<()> {
    let input_file =
        File::open(input).with_context(|| format!("Failed to open file at {}", input.display()))?;
    let mut reader = BufReader::new(input_file);
    let output_file = File::create(output)
        .with_context(|| format!("Failed to create file at {}", output.display()))?;
    let mut writer = BufWriter::new(output_file);
    let unquantised_net = UnquantisedNetwork::read(&mut reader)?;
    let net = unquantised_net.merge();
    net.write_bullet(&mut writer)?;
    Ok(())
}

pub fn dump_verbatim(output: &std::path::Path) -> anyhow::Result<()> {
    let output_file = File::create(output)
        .with_context(|| format!("Failed to create file at {}", output.display()))?;
    let mut writer = BufWriter::new(output_file);
    let network = NNUEParams::decompress_and_alloc()
        .with_context(|| "Failed to decompress and allocate NNUEParams")?;
    // SAFETY: look,
    let slice = unsafe {
        std::slice::from_raw_parts(
            std::ptr::from_ref::<NNUEParams>(network).cast::<u8>(),
            size_of::<NNUEParams>(),
        )
    };
    writer.write_all(slice)?;
    Ok(())
}

enum BoxedOrStatic<T: 'static> {
    Boxed(Box<T>),
    Static(&'static T),
}

impl<T> Deref for BoxedOrStatic<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Boxed(b) => b.as_ref(),
            Self::Static(s) => s,
        }
    }
}

pub fn dry_run() -> anyhow::Result<()> {
    use BoxedOrStatic::{Boxed, Static};
    if !EMBEDDED_NNUE_VERBATIM {
        println!("[#] Embedded NNUE is compressed, dry-run must operate on zeroed network.");
    }
    println!("[#] Constructing Board");
    let start_pos = Board::startpos();
    println!("[#] Generating network parameters");
    let nnue_params = if EMBEDDED_NNUE_VERBATIM {
        Static(NNUEParams::decompress_and_alloc()?)
    } else {
        // create a zeroed network
        Boxed(NNUEParams::zeroed())
    };
    println!("[#] Generating network state");
    let state = NNUEState::new(&start_pos, &nnue_params);
    println!("[#] Running forward pass");
    let eval = state.evaluate(&nnue_params, &start_pos);
    std::hint::black_box(eval);
    Ok(())
}

/// The size of the stack used to store the activations of the hidden layer.
const ACC_STACK_SIZE: usize = MAX_DEPTH + 1;

/// Struct representing some unmaterialised feature update made as part of a move.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct PsqtFeatureUpdate {
    pub sq: Square,
    pub piece: Piece,
}

/// Struct representing some unmaterialised threat update made as part of a move.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(C)]
pub struct ThreatFeatureUpdate {
    pub attacker: Piece,
    pub from: Square,
    pub victim: Piece,
    pub to: Square,
}

impl ThreatFeatureUpdate {
    pub fn index(self, colour: Colour, king: Square) -> Option<ThreatFeatureIndex> {
        feature::threat_index(colour, king, self.attacker, self.victim, self.from, self.to)
    }
}

impl Display for PsqtFeatureUpdate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{piece} on {sq}", piece = self.piece, sq = self.sq)
    }
}

#[derive(PartialEq, Eq, Clone, Debug, Default)]
pub struct PsqtUpdateBuffer {
    add: ArrayVec<PsqtFeatureUpdate, 2>,
    sub: ArrayVec<PsqtFeatureUpdate, 2>,
}

impl PsqtUpdateBuffer {
    pub fn move_piece(&mut self, from: Square, to: Square, piece: Piece) {
        self.add.push(PsqtFeatureUpdate { sq: to, piece });
        self.sub.push(PsqtFeatureUpdate { sq: from, piece });
    }

    pub fn clear_piece(&mut self, sq: Square, piece: Piece) {
        self.sub.push(PsqtFeatureUpdate { sq, piece });
    }

    pub fn add_piece(&mut self, sq: Square, piece: Piece) {
        self.add.push(PsqtFeatureUpdate { sq, piece });
    }

    pub fn adds(&self) -> &[PsqtFeatureUpdate] {
        &self.add[..]
    }

    pub fn subs(&self) -> &[PsqtFeatureUpdate] {
        &self.sub[..]
    }

    pub fn clear(&mut self) {
        self.add.clear();
        self.sub.clear();
    }
}

#[derive(PartialEq, Eq, Clone, Debug, Default)]
pub struct ThreatUpdateBuffer {
    pub add: ArrayVec<ThreatFeatureUpdate, 128>,
    pub sub: ArrayVec<ThreatFeatureUpdate, 128>,
}

impl ThreatUpdateBuffer {
    pub fn adds(&self) -> &[ThreatFeatureUpdate] {
        &self.add[..]
    }

    pub fn subs(&self) -> &[ThreatFeatureUpdate] {
        &self.sub[..]
    }

    pub fn clear(&mut self) {
        self.add.clear();
        self.sub.clear();
    }
}

/// Combined PSQT + threat update buffer, filled during move-making.
#[derive(PartialEq, Eq, Clone, Debug, Default)]
pub struct UpdateBuffer {
    pub psqt: PsqtUpdateBuffer,
    pub threat: ThreatUpdateBuffer,
}

impl UpdateBuffer {
    pub fn clear(&mut self) {
        self.psqt.clear();
        self.threat.clear();
    }
}

/// Stores last-seen accumulators for each bucket, so that we can hopefully avoid
/// having to completely recompute the accumulator for a position, instead
/// partially reconstructing it from the last-seen accumulator.
pub struct BucketAccumulatorCache {
    // both of these are BUCKETS * 2, rather than just BUCKETS,
    // because we use a horizontally-mirrored architecture.
    accs: [[Align<[i16; L1_SIZE]>; 2]; BUCKETS * 2],
    board_states: [[PieceLayout; BUCKETS * 2]; 2],
}

impl BucketAccumulatorCache {
    #[allow(clippy::too_many_lines)]
    pub fn load_accumulator_for_position(
        &mut self,
        nnue_params: &NNUEParams,
        board_state: PieceLayout,
        colour: Colour,
        acc: &mut Accumulator,
    ) {
        let king = (board_state.pieces[PieceType::King] & board_state.colours[colour])
            .first()
            .unwrap();
        let bucket = BUCKET_MAP[king.relative_to(colour)];
        let cache_acc = &mut self.accs[bucket][colour];

        let mut adds = ArrayVec::<_, 32>::new();
        let mut subs = ArrayVec::<_, 32>::new();
        self.board_states[colour][bucket].update_iter(board_state, |sq, piece, is_add| {
            let index = feature::psqt_index(colour, king, PsqtFeatureUpdate { sq, piece });
            if is_add {
                adds.push(index);
            } else {
                subs.push(index);
            }
        });

        let weights = nnue_params.select_feature_weights(bucket);

        accumulator::vector_update_inplace_psqt(cache_acc, weights, &adds, &subs);

        acc.halves[colour] = cache_acc.clone();

        self.board_states[colour][bucket] = board_state;
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MovedPiece {
    pub from: Square,
    pub to: Square,
    pub piece: Piece,
}

trait AccUpdateType {
    const PSQT: bool;
}

struct PsqtUpdate;
impl AccUpdateType for PsqtUpdate {
    const PSQT: bool = true;
}

struct ThreatUpdate;
impl AccUpdateType for ThreatUpdate {
    const PSQT: bool = false;
}

/// State of the partial activations of the NNUE network.
#[allow(clippy::upper_case_acronyms)]
pub struct NNUEState {
    /// Board-state accumulators for the first layer.
    pub psqt_accumulators: [Accumulator; ACC_STACK_SIZE],
    /// “dirty” flags for the PSQT accumulators.
    pub psqt_correct: [[bool; 2]; ACC_STACK_SIZE],

    /// Threat-state accumulators for the first layer.
    pub threat_accumulators: [Accumulator; ACC_STACK_SIZE],
    /// “dirty” flags for the threat accumulators.
    pub threat_correct: [[bool; 2]; ACC_STACK_SIZE],

    /// Diffs for the updates.
    pub updates: [UpdateBuffer; ACC_STACK_SIZE],

    /// Moves made for update computation.
    pub moves: [MovedPiece; ACC_STACK_SIZE],
    /// Index of the current accumulator.
    pub current_acc: usize,

    /// Cache of last-seen accumulators for each bucket.
    pub bucket_cache: BucketAccumulatorCache,
}

impl NNUEState {
    /// Create a new `NNUEState`.
    pub fn new(board: &Board, nnue_params: &NNUEParams) -> Box<Self> {
        // SAFETY: NNUEState has four fields:
        // {white,black}_pov, which are just arrays of ints, for whom the all-zeroes bitpattern is valid.
        // current_acc, which is just an int, so the all-zeroes bitpattern is valid.
        // accumulators, which is an array of Accumulator<SIZE>.
        //     Accumulator is a struct containing a pair of arrays of ints, so this field is safe for zeroing too.
        // As all fields can be safely initialised to all zeroes, the following code is sound.
        let mut net: Box<Self> = unsafe {
            let layout = std::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc_zeroed(layout);
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Box::from_raw(ptr.cast())
        };

        net.reïnit_from(board, nnue_params);

        net
    }

    /// reïnitialise the state from a board.
    pub fn reïnit_from(&mut self, board: &Board, nnue_params: &NNUEParams) {
        // set the current accumulator to the first one
        self.current_acc = 0;

        // initalise all the accumulators in the bucket cache to the bias
        for acc in &mut self.bucket_cache.accs {
            acc[Colour::White] = nnue_params.l0_biases.clone();
            acc[Colour::Black] = nnue_params.l0_biases.clone();
        }
        // initialise all the board states in the bucket cache to the empty board
        for board_state in self.bucket_cache.board_states.iter_mut().flatten() {
            *board_state = PieceLayout::default();
        }

        // refresh the first accumulator
        for colour in Colour::all() {
            // PSQT half:
            self.bucket_cache.load_accumulator_for_position(
                nnue_params,
                board.state.bbs,
                colour,
                &mut self.psqt_accumulators[0],
            );
            self.psqt_correct[0][colour] = true;

            // threat half:
            Self::refresh_threats(nnue_params, board, colour, &mut self.threat_accumulators[0]);
            self.threat_correct[0][colour] = true;
        }
    }

    pub fn refresh_threats(
        nnue_params: &NNUEParams,
        board: &Board,
        colour: Colour,
        acc: &mut Accumulator,
    ) {
        let acc = &mut acc.halves[colour];

        acc.fill(0); // clear the accumulator, we'll rebuild it from scratch

        let bbs = &board.state.bbs;
        let occ = bbs.occupied();
        let king = board.state.bbs.king_sq(colour);
        let bb = occ & !bbs.pieces[PieceType::King];

        // todo: think about whether this would be better as
        // a per-type loop — we might do better if we knew
        // what sort of piece we were moving, or we could merge
        // queen directions with ortho/diagonal moves, &c &c
        for from in bb {
            let attacker = board.state.mailbox[from].unwrap();
            let threats = occ & attacks_by_type(attacker, from, occ) & !bbs.pieces[PieceType::King];
            for to in threats {
                let victim = board.state.mailbox[to].unwrap();
                let Some(feature) = threat_index(colour, king, attacker, victim, from, to) else {
                    continue;
                };
                let start = feature.index() * L1_SIZE;
                let row = &nnue_params.l0_threat[start..start + L1_SIZE];
                for i in 0..L1_SIZE {
                    acc[i] += i16::from(row[i]);
                }
            }
        }
    }

    fn requires_refresh<A: AccUpdateType>(piece: Piece, from: Square, to: Square) -> bool {
        if piece.piece_type() != PieceType::King {
            return false;
        }

        // Threat features are not king-bucketed:
        if A::PSQT {
            BUCKET_MAP[from] != BUCKET_MAP[to]
        } else {
            // do we cross the mid-line?
            // [0,1,2,3,4,5,6,7] ⇒ [0,0,0,0,1,1,1,1]
            from.file() as u8 / 4 != to.file() as u8 / 4
        }
    }

    fn can_efficiently_update<A: AccUpdateType>(&self, colour: Colour) -> bool {
        let correct_table = if A::PSQT {
            &self.psqt_correct
        } else {
            &self.threat_correct
        };
        let mut curr_idx = self.current_acc;
        loop {
            curr_idx -= 1;

            let mv = self.moves[curr_idx];
            let from = mv.from.relative_to(colour);
            let to = mv.to.relative_to(colour);
            let piece = mv.piece;

            if piece.colour() == colour && Self::requires_refresh::<A>(piece, from, to) {
                return false;
            }
            if correct_table[curr_idx][colour] {
                return true;
            }
        }
    }

    fn apply_lazy_updates<A: AccUpdateType>(
        &mut self,
        nnue_params: &NNUEParams,
        board: &Board,
        colour: Colour,
    ) {
        let stack = if A::PSQT {
            &mut self.psqt_accumulators
        } else {
            &mut self.threat_accumulators
        };

        let correct = if A::PSQT {
            &mut self.psqt_correct
        } else {
            &mut self.threat_correct
        };

        let mut curr_index = self.current_acc;
        loop {
            curr_index -= 1;

            if correct[curr_index][colour] {
                break;
            }
        }

        let king = board.state.bbs.king_sq(colour);

        loop {
            let (front, back) = stack.split_at_mut(curr_index + 1);
            let src_acc = front.last().unwrap();
            let tgt_acc = back.first_mut().unwrap();

            if A::PSQT {
                Self::materialise_new_psqt_acc_from(
                    src_acc,
                    tgt_acc,
                    &self.updates[curr_index].psqt,
                    king,
                    colour,
                    nnue_params,
                );
            } else {
                Self::materialise_new_threat_acc_from(
                    src_acc,
                    tgt_acc,
                    &self.updates[curr_index].threat,
                    king,
                    colour,
                    nnue_params,
                );
            }

            correct[curr_index + 1][colour] = true;

            curr_index += 1;
            if curr_index == self.current_acc {
                break;
            }
        }
    }

    /// Apply all in-flight updates, generating all the accumulators up to the current one.
    ///
    /// When we do this, we update the piece-square and threat features separately,
    /// as threat features are almost-always updatable efficiently, as they are not
    /// bucketed (though they do mirror when the king crosses the center-line).
    pub fn force(&mut self, board: &Board, nnue_params: &NNUEParams) {
        for colour in Colour::all() {
            if !self.psqt_correct[self.current_acc][colour] {
                if self.can_efficiently_update::<PsqtUpdate>(colour) {
                    self.apply_lazy_updates::<PsqtUpdate>(nnue_params, board, colour);
                } else {
                    self.bucket_cache.load_accumulator_for_position(
                        nnue_params,
                        board.state.bbs,
                        colour,
                        &mut self.psqt_accumulators[self.current_acc],
                    );
                    self.psqt_correct[self.current_acc][colour] = true;
                }
            }

            if !self.threat_correct[self.current_acc][colour] {
                if self.can_efficiently_update::<ThreatUpdate>(colour) {
                    self.apply_lazy_updates::<ThreatUpdate>(nnue_params, board, colour);
                } else {
                    Self::refresh_threats(
                        nnue_params,
                        board,
                        colour,
                        &mut self.threat_accumulators[self.current_acc],
                    );
                    self.threat_correct[self.current_acc][colour] = true;
                }
            }
        }
    }

    pub fn hint_common_access(&mut self, pos: &Board, nnue_params: &NNUEParams) {
        self.hint_common_access_for_perspective::<White>(pos, nnue_params);
        self.hint_common_access_for_perspective::<Black>(pos, nnue_params);
    }

    fn hint_common_access_for_perspective<C: Col>(
        &mut self,
        pos: &Board,
        nnue_params: &NNUEParams,
    ) {
        if self.psqt_correct[self.current_acc][C::COLOUR] {
            return;
        }

        let oldest = self.try_find_computed_accumulator::<C>(pos);

        if let Some(source) = oldest {
            assert!(self.psqt_correct[source][C::COLOUR]);
            // directly construct the top accumulator from the last-known-good one
            let mut curr_index = source;
            let king = pos.state.bbs.king_sq(C::COLOUR);
            let bucket = BUCKET_MAP[king.relative_to(C::COLOUR)];
            let weights = nnue_params.select_feature_weights(bucket);
            let mut adds = ArrayVec::<_, 32>::new();
            let mut subs = ArrayVec::<_, 32>::new();

            loop {
                for &add in self.updates[curr_index].psqt.adds() {
                    adds.push(feature::psqt_index(C::COLOUR, king, add));
                }
                for &sub in self.updates[curr_index].psqt.subs() {
                    subs.push(feature::psqt_index(C::COLOUR, king, sub));
                }

                curr_index += 1;

                if curr_index == self.current_acc {
                    break;
                }
            }

            self.psqt_accumulators[self.current_acc].halves[C::COLOUR] =
                self.psqt_accumulators[source].halves[C::COLOUR].clone();
            accumulator::vector_update_inplace_psqt(
                &mut self.psqt_accumulators[self.current_acc].halves[C::COLOUR],
                weights,
                &adds,
                &subs,
            );
        } else {
            self.bucket_cache.load_accumulator_for_position(
                nnue_params,
                pos.state.bbs,
                C::COLOUR,
                &mut self.psqt_accumulators[self.current_acc],
            );
        }

        self.psqt_correct[self.current_acc][C::COLOUR] = true;
    }

    /// Find the index of the first materialised accumulator, or nothing
    /// if moving back that far would be too costly.
    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap
    )]
    fn try_find_computed_accumulator<C: Col>(&self, pos: &Board) -> Option<usize> {
        let mut idx = self.current_acc;
        let mut budget = pos.state.bbs.occupied().count() as i32;
        while idx > 0 && !self.psqt_correct[idx][C::COLOUR] {
            let mv = self.moves[idx - 1];
            let psqt_updates = &self.updates[idx - 1].psqt;
            if mv.piece.colour() == C::COLOUR
                // wrote PsqtUpdate to fix lint, as-yet unsure of correctness
                && Self::requires_refresh::<PsqtUpdate>(
                    mv.piece,
                    mv.from.relative_to(C::COLOUR),
                    mv.to.relative_to(C::COLOUR),
                )
            {
                break;
            }
            let adds = psqt_updates.adds().len() as i32;
            let subs = psqt_updates.subs().len() as i32;
            budget -= adds + subs + 1;
            if budget < 0 {
                break;
            }
            idx -= 1;
        }
        if self.psqt_correct[idx][C::COLOUR] {
            Some(idx)
        } else {
            None
        }
    }

    pub fn materialise_new_psqt_acc_from(
        src_acc: &Accumulator,
        tgt_acc: &mut Accumulator,
        updates: &PsqtUpdateBuffer,
        king: Square,
        colour: Colour,
        nnue_params: &NNUEParams,
    ) {
        let bucket = BUCKET_MAP[king.relative_to(colour)];

        let bucket = nnue_params.select_feature_weights(bucket);

        let src = &src_acc.halves[colour];
        let tgt = &mut tgt_acc.halves[colour];

        match (updates.adds(), updates.subs()) {
            // quiet or promotion
            (&[add], &[sub]) => {
                let add = feature::psqt_index(colour, king, add);
                let sub = feature::psqt_index(colour, king, sub);
                accumulator::vector_add_sub_psqt(src, tgt, bucket, add, sub);
            }
            // capture
            (&[add], &[sub1, sub2]) => {
                let add = feature::psqt_index(colour, king, add);
                let sub1 = feature::psqt_index(colour, king, sub1);
                let sub2 = feature::psqt_index(colour, king, sub2);
                accumulator::vector_add_sub2_psqt(src, tgt, bucket, add, sub1, sub2);
            }
            // castling
            (&[add1, add2], &[sub1, sub2]) => {
                let add1 = feature::psqt_index(colour, king, add1);
                let add2 = feature::psqt_index(colour, king, add2);
                let sub1 = feature::psqt_index(colour, king, sub1);
                let sub2 = feature::psqt_index(colour, king, sub2);
                accumulator::vector_add2_sub2_psqt(src, tgt, bucket, add1, add2, sub1, sub2);
            }
            (_, _) => panic!("invalid update buffer: {updates:?}"),
        }
    }

    pub fn materialise_new_threat_acc_from(
        src_acc: &Accumulator,
        tgt_acc: &mut Accumulator,
        updates: &ThreatUpdateBuffer,
        king: Square,
        colour: Colour,
        nnue_params: &NNUEParams,
    ) {
        let src = &src_acc.halves[colour];
        let tgt = &mut tgt_acc.halves[colour];

        // todo: don’t double-buffer.
        let mut adds = ArrayVec::<ThreatFeatureIndex, 128>::new();
        let mut subs = ArrayVec::<ThreatFeatureIndex, 128>::new();

        for &add in updates.adds() {
            if let Some(index) = add.index(colour, king) {
                adds.push(index);
            }
        }
        for &sub in updates.subs() {
            if let Some(index) = sub.index(colour, king) {
                subs.push(index);
            }
        }

        accumulator::vector_update_threats(src, tgt, &nnue_params.l0_threat, &adds, &subs);
    }

    /// Evaluate the final layer on the partial activations.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn evaluate(&self, nn: &NNUEParams, board: &Board) -> i32 {
        const K: f32 = SCALE as f32;

        debug_assert!(
            self.psqt_correct[self.current_acc][0] && self.psqt_correct[self.current_acc][1]
        );

        debug_assert!(
            self.threat_correct[self.current_acc][0] && self.threat_correct[self.current_acc][1]
        );

        let stm = board.turn();
        let out = output_bucket(board);

        let psqt_acc = &self.psqt_accumulators[self.current_acc];
        let thrt_acc = &self.threat_accumulators[self.current_acc];

        let [stm_psqt, ntm_psqt] = if stm == Colour::White {
            psqt_acc.halves.each_ref()
        } else {
            [
                &psqt_acc.halves[Colour::Black],
                &psqt_acc.halves[Colour::White],
            ]
        };
        let [stm_thrt, ntm_thrt] = if stm == Colour::White {
            thrt_acc.halves.each_ref()
        } else {
            [
                &thrt_acc.halves[Colour::Black],
                &thrt_acc.halves[Colour::White],
            ]
        };

        let mut l1_outputs = Align([0.0; L2_SIZE]);
        let mut l2_outputs = Align([0.0; L3_SIZE]);

        layers::activate_ft_and_propagate_l1(
            stm_psqt,
            ntm_psqt,
            stm_thrt,
            ntm_thrt,
            &nn.l1_weights[out],
            &nn.l1_bias[out],
            &mut l1_outputs,
        );
        layers::propagate_l2(
            &l1_outputs,
            &nn.l2_weights[out],
            &nn.l2_bias[out],
            &mut l2_outputs,
        );

        if HEADS == 1 {
            let mut l3_output = 0.0;

            layers::propagate_l3(
                &l2_outputs,
                &nn.l3_weights[out][0],
                nn.l3_bias[out][0],
                &mut l3_output,
            );

            (l3_output * SCALE as f32) as i32
        } else if HEADS == 3 {
            let mut l3_output_logits = [0.0; 3];

            for ((w, b), o) in nn.l3_weights[out]
                .iter()
                .zip(nn.l3_bias[out])
                .zip(&mut l3_output_logits)
            {
                layers::propagate_l3(&l2_outputs, w, b, o);
            }

            // softmax
            let mut win = l3_output_logits[2];
            let mut draw = l3_output_logits[1];
            let mut loss = l3_output_logits[0];

            let max = win.max(draw).max(loss);

            win = (win - max).exp();
            draw = (draw - max).exp();
            loss = (loss - max).exp();

            let sum = win + draw + loss;

            win /= sum;
            draw /= sum;
            // loss /= sum;

            let score = draw.mul_add(0.5, win).clamp(0.0, 1.0);

            (-K * (1.0 / score - 1.0).ln()) as i32
        } else {
            panic!("Unsupported number of heads: {HEADS}");
        }
    }
}

/// Benchmark the inference portion of the NNUE evaluation.
/// (everything after the feature extraction)
pub fn inference_benchmark(state: &NNUEState, nnue_params: &NNUEParams) {
    let start = std::time::Instant::now();
    let board = Board::startpos();
    for _ in 0..1_000_000 {
        std::hint::black_box(std::hint::black_box(state).evaluate(
            std::hint::black_box(nnue_params),
            std::hint::black_box(&board),
        ));
    }
    let elapsed = start.elapsed();
    let nanos = elapsed.as_nanos();
    let ns_per_eval = nanos / 1_000_000;
    println!("{ns_per_eval} ns per evaluation");
}

pub fn visualise_nnue() -> anyhow::Result<()> {
    let nnue_params = NNUEParams::decompress_and_alloc()?;
    // create folder for the images
    let path = std::path::PathBuf::from("nnue-visualisations");
    std::fs::create_dir_all(&path)
        .with_context(|| "Failed to create NNUE visualisations folder.")?;
    for neuron in 0..crate::nnue::network::L1_SIZE {
        nnue_params.visualise_neuron(neuron, &path);
    }
    nnue_params.composite_neurons(&path);
    let (min, max) = nnue_params.min_max_feature_weight();
    println!("Min / Max FT values: {min} / {max}");
    Ok(())
}

const IMAGE_SPACING: usize = 0;

impl NNUEParams {
    pub fn visualise_neuron(&self, neuron: usize, path: &std::path::Path) {
        let image = self.neuron_image(neuron);
        let path = path.join(format!("neuron_{neuron}.tga"));
        image.save_as_tga(path);
    }

    fn neuron_image(&self, neuron: usize) -> Image {
        #![allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        // remap pieces to keep opposite colours together
        static PIECE_REMAPPING: [usize; 12] = [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11];
        assert!(neuron < L1_SIZE);
        let starting_idx = neuron;
        let mut slice = Vec::with_capacity(768);
        for colour in Colour::all() {
            for piece_type in PieceType::all() {
                for square in Square::all() {
                    let white_king = Square::H1;
                    let f = PsqtFeatureUpdate {
                        sq: square,
                        piece: Piece::new(colour, piece_type),
                    };
                    let feature_index = feature::psqt_index(Colour::White, white_king, f);
                    let index = feature_index.index() * L1_SIZE + starting_idx;
                    slice.push(self.l0_weights[index]);
                }
            }
        }

        let max_abs = slice.iter().copied().map(i16::unsigned_abs).max().unwrap();
        let weight_to_colour = |weight: i16| -> u32 {
            if max_abs == 0 {
                return image::inferno_colour_map(0);
            }
            let magnitude = f32::from(weight.unsigned_abs()) / f32::from(max_abs);
            let idx = (magnitude * 255.0).round() as u8;
            if weight >= 0 {
                image::inferno_colour_map(idx)
            } else {
                image::cool_inferno_colour_map(idx)
            }
        };

        let mut image = Image::zeroed(8 * 6 + IMAGE_SPACING * 5, 8 * 2 + IMAGE_SPACING);

        for (piece, chunk) in slice.chunks(64).enumerate() {
            let piece = PIECE_REMAPPING[piece];
            let piece_colour = piece % 2;
            let piece_type = piece / 2;
            for (square, &weight) in chunk.iter().enumerate() {
                let row = square / 8;
                let col = square % 8;
                let colour = if (row == 0 || row == 7) && piece_type == 0 {
                    0 // pawns on first and last rank are always 0
                } else {
                    weight_to_colour(weight)
                };
                image.set(
                    col + piece_type * (8 + IMAGE_SPACING),
                    row + piece_colour * (8 + IMAGE_SPACING),
                    colour,
                );
            }
        }

        image
    }

    pub fn composite_neurons(&self, path: &Path) {
        const TILE_W: usize = 8 * 6 + IMAGE_SPACING * 5;
        const TILE_H: usize = 8 * 2 + IMAGE_SPACING;

        // aiming for a 16:9 aspect ratio
        let cols = (1..=L1_SIZE)
            .min_by_key(|&c| {
                let rows = L1_SIZE.div_ceil(c);
                let w = c * (TILE_W + IMAGE_SPACING);
                let h = rows * (TILE_H + IMAGE_SPACING);
                // minimise |w/h - 16/9|, i.e. |9w - 16h|
                (9 * w).abs_diff(16 * h)
            })
            .unwrap();
        let rows = L1_SIZE.div_ceil(cols);
        let img_w = cols * TILE_W + (cols - 1) * IMAGE_SPACING;
        let img_h = rows * TILE_H + (rows - 1) * IMAGE_SPACING;

        #[expect(clippy::cast_possible_truncation)]
        let neuron_order = (0..L1_SIZE as u16).collect::<ArrayVec<u16, L1_SIZE>>();
        // the sorting is nice, but doesn’t look quite so pretty.
        // neuron_order.sort_by_key(|&n| {
        //     let start = n;
        //     let mean_abs: u64 = (0..INPUT)
        //         .map(|i| u64::from(self.l0_weights[i * L1_SIZE + start as usize].unsigned_abs()))
        //         .sum();
        //     mean_abs
        // });
        let mut composite = Image::zeroed(img_w, img_h);
        for (loc, &neuron) in neuron_order.iter().enumerate() {
            let col = loc % cols;
            let row = loc / cols;
            let ox = col * (TILE_W + IMAGE_SPACING);
            let oy = row * (TILE_H + IMAGE_SPACING);
            let tile = self.neuron_image(neuron as usize);
            for ty in 0..TILE_H {
                for tx in 0..TILE_W {
                    composite.set(ox + tx, oy + ty, tile.pixel(tx, ty));
                }
            }
        }

        let path = path.join("composite.tga");

        composite.save_as_tga(path);
    }

    pub fn min_max_feature_weight(&self) -> (i16, i16) {
        let mut min = i16::MAX;
        let mut max = i16::MIN;
        for &f in &self.l0_weights.0 {
            if f < min {
                min = f;
            }
            if f > max {
                max = f;
            }
        }
        (min, max)
    }
}
