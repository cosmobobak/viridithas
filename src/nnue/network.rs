use std::{
    fmt::{Debug, Display},
    fs::{File, OpenOptions},
    hash::Hasher,
    io::{BufReader, BufWriter},
    ops::{Deref, DerefMut},
    path::Path,
    sync::{Mutex, OnceLock},
    time::Duration,
};

use anyhow::Context;
use arrayvec::ArrayVec;
use memmap2::Mmap;

use crate::{
    chess::{
        board::Board,
        piece::{Black, Col, Colour, Piece, PieceType, White},
        piecelayout::PieceLayout,
        squareset::SquareSet,
        types::Square,
    },
    image::{self, Image},
    nnue,
    util::{self, MAX_PLY},
};

use super::accumulator::{self, Accumulator};

pub mod feature;
pub mod layers;

/// Whether to perform the king-plane merging optimisation.
pub const MERGE_KING_PLANES: bool = true;
/// Whether the unquantised network has a feature factoriser.
pub const UNQUANTISED_HAS_FACTORISER: bool = true;
/// The size of the input layer of the network.
pub const INPUT: usize = (12 - MERGE_KING_PLANES as usize) * 64;
/// The amount to scale the output of the network by.
/// This is to allow for the sigmoid activation to differentiate positions with
/// a small difference in evaluation.
const SCALE: i32 = 400;
/// The size of one-half of the hidden layer of the network.
pub const L1_SIZE: usize = 2048;
/// The size of the second layer of the network.
pub const L2_SIZE: usize = 16;
/// The size of the third layer of the network.
pub const L3_SIZE: usize = 32;
/// chunking constant for l1
pub const L1_CHUNK_PER_32: usize = std::mem::size_of::<i32>() / std::mem::size_of::<i8>();
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

/// The number of output buckets
pub const OUTPUT_BUCKETS: usize = 8;
/// Get index into the output layer given a board state.
pub fn output_bucket(pos: &Board) -> usize {
    #![allow(clippy::cast_possible_truncation)]
    const DIVISOR: usize = usize::div_ceil(32, OUTPUT_BUCKETS);
    (pos.state.bbs.occupied().count() as usize - 2) / DIVISOR
}

const QA: i16 = 255;
const QB: i16 = 64;

// read in the binary file containing the network parameters
// have to do some path manipulation to get relative paths to work
pub static COMPRESSED_NNUE: &[u8] = include_bytes!("../../viridithas.nnue.zst");

pub fn nnue_checksum() -> u64 {
    let mut hasher = fxhash::FxHasher::default();
    hasher.write(COMPRESSED_NNUE);
    for index in REPERMUTE_INDICES {
        hasher.write_usize(index);
    }
    hasher.finish()
}

/// Struct representing the floating-point parameter file emitted by bullet.
#[rustfmt::skip]
#[repr(C)]
struct UnquantisedNetwork {
    // extra bucket for the feature-factoriser.
    ft_weights:    [f32; 12 * 64 * L1_SIZE * (BUCKETS + UNQUANTISED_HAS_FACTORISER as usize)],
    ft_biases:     [f32; L1_SIZE],
    l1x_weights: [[[f32; L2_SIZE]; OUTPUT_BUCKETS]; L1_SIZE],
    l1f_weights:  [[f32; L2_SIZE]; L1_SIZE],
    l1x_biases:   [[f32; L2_SIZE]; OUTPUT_BUCKETS],
    l1f_biases:    [f32; L2_SIZE],
    l2x_weights: [[[f32; L3_SIZE]; OUTPUT_BUCKETS]; L2_SIZE],
    l2f_weights:  [[f32; L3_SIZE]; L2_SIZE],
    l2x_biases:   [[f32; L3_SIZE]; OUTPUT_BUCKETS],
    l2f_biases:    [f32; L3_SIZE],
    l3x_weights:  [[f32; OUTPUT_BUCKETS]; L3_SIZE],
    l3f_weights:   [f32; L3_SIZE],
    l3x_biases:    [f32; OUTPUT_BUCKETS],
    l3f_biases:    [f32; 1],
}

/// The floating-point parameters of the network, after de-factorisation.
#[rustfmt::skip]
#[repr(C)]
struct MergedNetwork {
    ft_weights:   [f32; 12 * 64 * L1_SIZE * BUCKETS],
    ft_biases:    [f32; L1_SIZE],
    l1_weights: [[[f32; L2_SIZE]; OUTPUT_BUCKETS]; L1_SIZE],
    l1_biases:   [[f32; L2_SIZE]; OUTPUT_BUCKETS],
    l2_weights: [[[f32; L3_SIZE]; OUTPUT_BUCKETS]; L2_SIZE],
    l2_biases:   [[f32; L3_SIZE]; OUTPUT_BUCKETS],
    l3_weights:  [[f32; OUTPUT_BUCKETS]; L3_SIZE],
    l3_biases:    [f32; OUTPUT_BUCKETS],
}

/// A quantised network file, for compressed embedding.
#[rustfmt::skip]
#[repr(C)]
#[derive(PartialEq, Debug)]
struct QuantisedNetwork {
    ft_weights:   [i16; INPUT * L1_SIZE * BUCKETS],
    ft_biases:    [i16; L1_SIZE],
    l1_weights: [[[ i8; L2_SIZE]; OUTPUT_BUCKETS]; L1_SIZE],
    l1_biases:   [[f32; L2_SIZE]; OUTPUT_BUCKETS],
    l2_weights: [[[f32; L3_SIZE]; OUTPUT_BUCKETS]; L2_SIZE],
    l2_biases:   [[f32; L3_SIZE]; OUTPUT_BUCKETS],
    l3_weights:  [[f32; OUTPUT_BUCKETS]; L3_SIZE],
    l3_biases:    [f32; OUTPUT_BUCKETS],
}

/// The parameters of viri's neural network, quantised and permuted
/// for efficient SIMD inference.
#[rustfmt::skip]
#[repr(C)]
pub struct NNUEParams {
    pub feature_weights: Align64<[i16; INPUT * L1_SIZE * BUCKETS]>,
    pub feature_bias:    Align64<[i16; L1_SIZE]>,
    pub l1_weights:     [Align64<[ i8; L1_SIZE * L2_SIZE]>; OUTPUT_BUCKETS],
    pub l1_bias:        [Align64<[f32; L2_SIZE]>; OUTPUT_BUCKETS],
    pub l2_weights:     [Align64<[f32; L2_SIZE * L3_SIZE]>; OUTPUT_BUCKETS],
    pub l2_bias:        [Align64<[f32; L3_SIZE]>; OUTPUT_BUCKETS],
    pub l3_weights:     [Align64<[f32; L3_SIZE]>; OUTPUT_BUCKETS],
    pub l3_bias:        [f32; OUTPUT_BUCKETS],
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
    342, 273, 402, 934, 704, 995, 391, 474, 504, 738, 245, 618, 100, 502, 350, 212, 13, 326, 370,
    284, 130, 324, 58, 980, 580, 125, 875, 247, 268, 932, 93, 208, 573, 959, 297, 735, 984, 381,
    538, 253, 413, 129, 679, 699, 883, 319, 961, 28, 999, 276, 976, 809, 363, 613, 467, 972, 734,
    118, 285, 436, 775, 434, 209, 311, 263, 217, 776, 626, 201, 332, 970, 26, 747, 661, 537, 97,
    69, 581, 43, 873, 904, 988, 490, 328, 382, 857, 458, 619, 536, 748, 296, 23, 321, 449, 630,
    902, 849, 15, 109, 417, 89, 700, 804, 741, 204, 855, 399, 944, 558, 460, 660, 133, 55, 880,
    385, 147, 259, 510, 945, 82, 506, 375, 559, 1007, 236, 224, 685, 442, 269, 605, 1022, 726, 930,
    85, 723, 240, 858, 876, 389, 828, 145, 303, 659, 656, 867, 315, 1006, 344, 115, 879, 777, 729,
    596, 277, 40, 170, 1023, 646, 994, 638, 897, 648, 394, 231, 223, 670, 697, 957, 969, 293, 528,
    574, 907, 942, 122, 918, 46, 117, 386, 819, 1019, 310, 489, 503, 470, 103, 255, 35, 642, 430,
    198, 662, 86, 19, 175, 779, 428, 710, 952, 570, 51, 870, 304, 521, 887, 771, 491, 369, 603,
    639, 615, 900, 728, 557, 238, 123, 812, 14, 647, 356, 891, 732, 96, 443, 627, 993, 784, 466,
    851, 578, 602, 1021, 48, 61, 632, 200, 367, 267, 203, 717, 756, 862, 609, 318, 563, 383, 595,
    186, 672, 545, 0, 54, 37, 199, 70, 366, 594, 669, 151, 280, 361, 1015, 78, 643, 599, 882, 608,
    494, 718, 853, 307, 288, 793, 909, 546, 516, 50, 714, 696, 390, 986, 691, 910, 690, 127, 682,
    625, 744, 457, 1013, 633, 966, 438, 424, 52, 517, 131, 347, 556, 218, 334, 893, 469, 166, 675,
    412, 107, 79, 929, 68, 913, 692, 365, 684, 113, 554, 593, 111, 20, 279, 357, 351, 281, 822, 17,
    527, 928, 863, 237, 997, 977, 432, 266, 6, 406, 91, 658, 144, 378, 885, 456, 196, 148, 128,
    871, 444, 338, 63, 622, 77, 64, 496, 387, 908, 745, 7, 119, 955, 512, 445, 772, 722, 922, 774,
    71, 65, 214, 939, 143, 451, 441, 141, 4, 32, 215, 788, 751, 564, 92, 499, 778, 526, 998, 529,
    746, 629, 401, 34, 181, 454, 866, 856, 983, 289, 852, 813, 577, 841, 824, 827, 716, 213, 459,
    992, 482, 345, 57, 572, 762, 884, 11, 567, 524, 475, 733, 649, 120, 9, 501, 820, 116, 368, 508,
    607, 242, 126, 105, 264, 158, 678, 996, 586, 421, 515, 933, 192, 654, 916, 409, 354, 31, 323,
    24, 481, 162, 850, 796, 8, 548, 429, 431, 2, 346, 171, 206, 920, 637, 241, 892, 588, 301, 222,
    410, 94, 76, 210, 495, 392, 962, 963, 561, 1020, 663, 693, 569, 724, 974, 509, 251, 657, 869,
    552, 810, 919, 225, 518, 33, 427, 645, 667, 830, 220, 41, 798, 786, 66, 562, 362, 308, 814,
    617, 592, 845, 3, 551, 312, 550, 260, 316, 353, 56, 299, 598, 425, 982, 184, 492, 305, 437,
    1016, 135, 760, 479, 960, 243, 261, 823, 335, 641, 202, 655, 698, 5, 653, 400, 102, 585, 374,
    730, 668, 979, 360, 270, 83, 376, 761, 42, 865, 306, 752, 805, 486, 246, 720, 185, 803, 846,
    393, 258, 1017, 480, 36, 257, 840, 705, 549, 989, 968, 703, 403, 978, 47, 926, 894, 901, 861,
    327, 829, 372, 807, 139, 547, 965, 29, 797, 764, 606, 250, 232, 951, 189, 520, 132, 941, 317,
    98, 99, 967, 815, 168, 453, 388, 755, 792, 806, 719, 721, 177, 278, 584, 146, 836, 711, 600,
    80, 249, 917, 352, 839, 110, 74, 216, 575, 543, 680, 44, 789, 838, 1001, 579, 906, 636, 505,
    121, 329, 826, 801, 896, 948, 149, 737, 473, 936, 140, 612, 420, 628, 843, 169, 182, 433, 1014,
    860, 911, 591, 742, 464, 759, 487, 440, 943, 597, 712, 325, 817, 541, 958, 84, 899, 446, 254,
    373, 108, 794, 650, 462, 542, 18, 530, 160, 985, 949, 1002, 286, 379, 519, 256, 67, 38, 30,
    535, 750, 205, 248, 905, 219, 233, 616, 450, 587, 589, 898, 359, 621, 534, 702, 821, 673, 195,
    749, 644, 514, 415, 971, 298, 890, 665, 340, 782, 808, 159, 953, 114, 187, 938, 833, 767, 10,
    397, 230, 404, 681, 523, 343, 477, 488, 921, 583, 539, 739, 156, 349, 947, 507, 859, 1000, 272,
    419, 163, 1012, 731, 371, 380, 981, 687, 816, 227, 610, 414, 525, 553, 765, 604, 455, 244, 448,
    956, 611, 176, 197, 925, 164, 333, 25, 193, 235, 478, 652, 497, 1008, 868, 165, 881, 191, 1003,
    275, 707, 540, 1004, 651, 152, 666, 874, 313, 183, 211, 640, 799, 452, 620, 154, 877, 950, 940,
    12, 188, 912, 112, 927, 337, 834, 990, 137, 854, 576, 601, 634, 16, 701, 142, 1011, 709, 493,
    781, 727, 773, 207, 757, 991, 53, 106, 358, 124, 283, 27, 150, 408, 398, 635, 331, 835, 39,
    842, 22, 743, 770, 341, 287, 676, 1018, 511, 1005, 544, 937, 157, 886, 706, 62, 582, 848, 405,
    49, 377, 671, 533, 954, 239, 75, 153, 161, 787, 21, 396, 73, 471, 314, 1010, 194, 339, 753,
    476, 915, 695, 173, 426, 785, 190, 291, 708, 423, 522, 134, 964, 300, 864, 923, 348, 355, 975,
    987, 725, 364, 631, 811, 180, 292, 566, 155, 295, 485, 465, 435, 416, 903, 60, 623, 590, 825,
    531, 447, 179, 686, 395, 889, 780, 498, 484, 178, 461, 226, 294, 336, 736, 689, 418, 252, 872,
    795, 768, 766, 769, 783, 228, 571, 688, 320, 72, 472, 407, 290, 914, 302, 754, 384, 713, 818,
    624, 532, 221, 330, 715, 1009, 844, 101, 104, 740, 411, 683, 802, 174, 439, 565, 468, 422, 322,
    763, 483, 888, 229, 931, 513, 924, 463, 895, 45, 614, 95, 790, 831, 791, 282, 309, 88, 832,
    677, 172, 274, 664, 555, 837, 847, 262, 234, 1, 694, 935, 674, 560, 136, 946, 758, 500, 265,
    800, 167, 878, 87, 138, 90, 59, 271, 973, 568, 81,
];

impl UnquantisedNetwork {
    /// Convert a parameter file generated by bullet into a merged parameter set,
    /// for further processing or for resuming training in a more efficient format.
    fn merge(&self) -> Box<MergedNetwork> {
        let mut net = MergedNetwork::zeroed();
        let mut buckets = self.ft_weights.chunks_exact(12 * 64 * L1_SIZE);
        let factoriser;
        let alternate_buffer;
        if UNQUANTISED_HAS_FACTORISER {
            factoriser = buckets.next().unwrap();
        } else {
            alternate_buffer = vec![0.0; 12 * 64 * L1_SIZE];
            factoriser = &alternate_buffer;
        }
        for (src_bucket, tgt_bucket) in
            buckets.zip(net.ft_weights.chunks_exact_mut(12 * 64 * L1_SIZE))
        {
            for piece in Piece::all() {
                for sq in Square::all() {
                    let i =
                        feature::index_full(Colour::White, Square::A1, FeatureUpdate { sq, piece });
                    let j =
                        feature::index_full(Colour::White, Square::A1, FeatureUpdate { sq, piece });
                    let src = &src_bucket[i * L1_SIZE..i * L1_SIZE + L1_SIZE];
                    let fac_src = &factoriser[i * L1_SIZE..i * L1_SIZE + L1_SIZE];
                    let tgt = &mut tgt_bucket[j * L1_SIZE..j * L1_SIZE + L1_SIZE];
                    for ((src, fac_src), tgt) in src.iter().zip(fac_src).zip(tgt) {
                        *tgt = *src + *fac_src;
                    }
                }
            }
        }

        // copy the biases
        net.ft_biases.copy_from_slice(&self.ft_biases);
        // copy the L1 weights
        for i in 0..L1_SIZE {
            for bucket in 0..OUTPUT_BUCKETS {
                for j in 0..L2_SIZE {
                    net.l1_weights[i][bucket][j] =
                        self.l1x_weights[i][bucket][j] + self.l1f_weights[i][j];
                }
            }
        }
        // copy the L1 biases
        for i in 0..L2_SIZE {
            for bucket in 0..OUTPUT_BUCKETS {
                net.l1_biases[bucket][i] = self.l1x_biases[bucket][i] + self.l1f_biases[i];
            }
        }
        // copy the L2 weights
        for i in 0..L2_SIZE {
            for bucket in 0..OUTPUT_BUCKETS {
                for j in 0..L3_SIZE {
                    net.l2_weights[i][bucket][j] =
                        self.l2x_weights[i][bucket][j] + self.l2f_weights[i][j];
                }
            }
        }
        // copy the L2 biases
        for i in 0..L3_SIZE {
            for bucket in 0..OUTPUT_BUCKETS {
                net.l2_biases[bucket][i] = self.l2x_biases[bucket][i] + self.l2f_biases[i];
            }
        }
        // copy the L3 weights
        for i in 0..L3_SIZE {
            for bucket in 0..OUTPUT_BUCKETS {
                net.l3_weights[i][bucket] = self.l3x_weights[i][bucket] + self.l3f_weights[i];
            }
        }
        // copy the L3 biases
        for i in 0..OUTPUT_BUCKETS {
            net.l3_biases[i] = self.l3x_biases[i] + self.l3f_biases[0];
        }

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
                util::from_mut(net.as_mut()).cast::<u8>(),
                std::mem::size_of::<Self>(),
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
                let len = std::mem::size_of_val(&$field) / std::mem::size_of::<f32>();
                // SAFETY: lol
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

        dump_layer!("l0w", self.ft_weights);
        dump_layer!("l0b", self.ft_biases);
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
        let buckets = self.ft_weights.chunks_exact(12 * 64 * L1_SIZE);

        for (bucket_idx, (src_bucket, tgt_bucket)) in buckets
            .zip(net.ft_weights.chunks_exact_mut(INPUT * L1_SIZE))
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
                    let i =
                        feature::index_full(Colour::White, Square::A1, FeatureUpdate { sq, piece });
                    let j = feature::index(Colour::White, Square::A1, FeatureUpdate { sq, piece })
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
            assert_eq!(INPUT, things_written);
        }

        // quantise the FT biases
        for (src, tgt) in self.ft_biases.iter().zip(net.ft_biases.iter_mut()) {
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
        let src_buckets = self.ft_weights.chunks_exact(INPUT * L1_SIZE);
        let tgt_buckets = net.feature_weights.chunks_exact_mut(INPUT * L1_SIZE);
        for (src_bucket, tgt_bucket) in src_buckets.zip(tgt_buckets) {
            repermute_ft_bucket(tgt_bucket, src_bucket);
        }

        // permute the feature transformer biases
        repermute_ft_bias(&mut net.feature_bias, &self.ft_biases);

        // transpose FT weights and biases so that packus transposes it back to the intended order
        if use_simd {
            type PermChunk = [i16; 8];
            // reinterpret as data of size __m128i
            let mut weights: Vec<&mut PermChunk> = net
                .feature_weights
                .chunks_exact_mut(8)
                .map(|a| a.try_into().unwrap())
                .collect();
            let mut biases: Vec<&mut PermChunk> = net
                .feature_bias
                .chunks_exact_mut(8)
                .map(|a| a.try_into().unwrap())
                .collect();
            let num_chunks = std::mem::size_of::<PermChunk>() / std::mem::size_of::<i16>();

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
            #[cfg(not(target_arch = "x86_64"))]
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
            #[cfg(not(target_arch = "x86_64"))]
            let order = [0];

            let mut regs = vec![[0i16; 8]; num_regs];

            // transpose weights
            for i in (0..INPUT * L1_SIZE * BUCKETS / num_chunks).step_by(num_regs) {
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
                for j in 0..L3_SIZE {
                    net.l2_weights[bucket][i * L3_SIZE + j] = self.l2_weights[i][bucket][j];
                }
            }

            // transfer the L2 biases
            for i in 0..L3_SIZE {
                net.l2_bias[bucket][i] = self.l2_biases[bucket][i];
            }

            // transfer the L3 weights
            for i in 0..L3_SIZE {
                net.l3_weights[bucket][i] = self.l3_weights[i][bucket];
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
        let ptr = util::from_ref::<Self>(self).cast::<u8>();
        let len = std::mem::size_of::<Self>();
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
    for i in 0..INPUT {
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

impl NNUEParams {
    #[allow(clippy::too_many_lines)]
    pub fn decompress_and_alloc() -> anyhow::Result<&'static Self> {
        #[cfg(not(feature = "zstd"))]
        type ZstdDecoder<R, D> = ruzstd::StreamingDecoder<R, D>;
        #[cfg(feature = "zstd")]
        type ZstdDecoder<'a, R> = zstd::stream::Decoder<'a, R>;

        // this function is not particularly happy about running in parallel.
        static LOCK: Mutex<()> = Mutex::new(());
        // additionally, we'd quite like to cache the results of this function.
        static CACHED: OnceLock<Mmap> = OnceLock::new();
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
                util::from_mut(net.as_mut()).cast::<u8>(),
                std::mem::size_of::<QuantisedNetwork>(),
            )
        };
        let expected_bytes = mem.len() as u64;
        let mut decoder = ZstdDecoder::new(COMPRESSED_NNUE)
            .with_context(|| "Failed to construct zstd decoder for NNUE weights.")?;
        let bytes_written = std::io::copy(&mut decoder, &mut mem)
            .with_context(|| "Failed to decompress NNUE weights.")?;
        anyhow::ensure!(bytes_written == expected_bytes, "encountered issue while decompressing NNUE weights, expected {expected_bytes} bytes, but got {bytes_written}");
        let use_simd = cfg!(target_arch = "x86_64");
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
        let size = std::mem::size_of::<Self>();
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
                    "Failed to rename temp file from {tfile:#?} to {wfile:#?} in {}",
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
            mmap.len() == std::mem::size_of::<Self>(),
            "Wrong number of bytes: expected {}, got {}",
            std::mem::size_of::<Self>(),
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

    pub fn select_feature_weights(&self, bucket: usize) -> &Align64<[i16; INPUT * L1_SIZE]> {
        // handle mirroring
        let bucket = bucket % BUCKETS;
        let start = bucket * INPUT * L1_SIZE;
        let end = start + INPUT * L1_SIZE;
        let slice = &self.feature_weights[start..end];
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

/// The size of the stack used to store the activations of the hidden layer.
const ACC_STACK_SIZE: usize = MAX_PLY + 1;

/// Struct representing some unmaterialised feature update made as part of a move.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FeatureUpdate {
    pub sq: Square,
    pub piece: Piece,
}

impl Display for FeatureUpdate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{piece} on {sq}", piece = self.piece, sq = self.sq)
    }
}

#[derive(PartialEq, Eq, Clone, Debug, Default)]
pub struct UpdateBuffer {
    add: ArrayVec<FeatureUpdate, 2>,
    sub: ArrayVec<FeatureUpdate, 2>,
}

impl UpdateBuffer {
    pub fn move_piece(&mut self, from: Square, to: Square, piece: Piece) {
        self.add.push(FeatureUpdate { sq: to, piece });
        self.sub.push(FeatureUpdate { sq: from, piece });
    }

    pub fn clear_piece(&mut self, sq: Square, piece: Piece) {
        self.sub.push(FeatureUpdate { sq, piece });
    }

    pub fn add_piece(&mut self, sq: Square, piece: Piece) {
        self.add.push(FeatureUpdate { sq, piece });
    }

    pub fn adds(&self) -> &[FeatureUpdate] {
        &self.add[..]
    }

    pub fn subs(&self) -> &[FeatureUpdate] {
        &self.sub[..]
    }
}

/// Stores last-seen accumulators for each bucket, so that we can hopefully avoid
/// having to completely recompute the accumulator for a position, instead
/// partially reconstructing it from the last-seen accumulator.
pub struct BucketAccumulatorCache {
    // both of these are BUCKETS * 2, rather than just BUCKETS,
    // because we use a horizontally-mirrored architecture.
    accs: [Accumulator; BUCKETS * 2],
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
        let king =
            SquareSet::first(board_state.pieces[PieceType::King] & board_state.colours[colour]);
        let bucket = BUCKET_MAP[king.relative_to(colour)];
        let cache_acc = self.accs[bucket].select_mut(colour);

        let mut adds = ArrayVec::<_, 32>::new();
        let mut subs = ArrayVec::<_, 32>::new();
        self.board_states[colour][bucket].update_iter(board_state, |sq, piece, is_add| {
            let index = feature::index(colour, king, FeatureUpdate { sq, piece });
            if is_add {
                adds.push(index);
            } else {
                subs.push(index);
            }
        });

        let weights = nnue_params.select_feature_weights(bucket);

        accumulator::vector_update_inplace(cache_acc, weights, &adds, &subs);

        *acc.select_mut(colour) = cache_acc.clone();
        acc.correct[colour] = true;

        self.board_states[colour][bucket] = board_state;
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[repr(C, align(64))]
pub struct Align64<T>(pub T);

impl<T, const SIZE: usize> Deref for Align64<[T; SIZE]> {
    type Target = [T; SIZE];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<T, const SIZE: usize> DerefMut for Align64<[T; SIZE]> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MovedPiece {
    pub from: Square,
    pub to: Square,
    pub piece: Piece,
}

/// State of the partial activations of the NNUE network.
#[allow(clippy::upper_case_acronyms)]
pub struct NNUEState {
    /// Accumulators for the first layer.
    pub accumulators: [Accumulator; ACC_STACK_SIZE],
    /// Index of the current accumulator.
    pub current_acc: usize,
    /// Cache of last-seen accumulators for each bucket.
    pub bucket_cache: BucketAccumulatorCache,
}

impl NNUEState {
    /// Create a new `NNUEState`.
    #[allow(clippy::unnecessary_box_returns)]
    pub fn new(board: &Board, nnue_params: &NNUEParams) -> Box<Self> {
        #![allow(clippy::cast_ptr_alignment)]
        // NNUEState is INPUT * 2 * 2 + LAYER_1_SIZE * ACC_STACK_SIZE * 2 * 2 + 8 bytes
        // at time of writing, this adds up to 396,296 bytes.
        // Putting this on the stack will almost certainly blow it, so we box it.
        // Unfortunately, in debug mode `Box::new(Self::new())` will allocate on the stack
        // and then memcpy it to the heap, so we have to do this manually.

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

        net.reinit_from(board, nnue_params);

        net
    }

    /// Reinitialise the state from a board.
    pub fn reinit_from(&mut self, board: &Board, nnue_params: &NNUEParams) {
        // set the current accumulator to the first one
        self.current_acc = 0;

        // initalise all the accumulators in the bucket cache to the bias
        for acc in &mut self.bucket_cache.accs {
            acc.white = nnue_params.feature_bias.clone();
            acc.black = nnue_params.feature_bias.clone();
        }
        // initialise all the board states in the bucket cache to the empty board
        for board_state in self.bucket_cache.board_states.iter_mut().flatten() {
            *board_state = PieceLayout::default();
        }

        // refresh the first accumulator
        for colour in Colour::all() {
            self.bucket_cache.load_accumulator_for_position(
                nnue_params,
                board.state.bbs,
                colour,
                &mut self.accumulators[0],
            );
        }
    }

    fn requires_refresh(piece: Piece, from: Square, to: Square) -> bool {
        if piece.piece_type() != PieceType::King {
            return false;
        }

        BUCKET_MAP[from] != BUCKET_MAP[to]
    }

    fn can_efficiently_update(&self, colour: Colour) -> bool {
        let mut curr_idx = self.current_acc;
        loop {
            curr_idx -= 1;
            let curr = &self.accumulators[curr_idx];

            let mv = curr.mv;
            let from = mv.from.relative_to(colour);
            let to = mv.to.relative_to(colour);
            let piece = mv.piece;

            if piece.colour() == colour && Self::requires_refresh(piece, from, to) {
                return false;
            }
            if curr.correct[colour] {
                return true;
            }
        }
    }

    fn apply_lazy_updates(&mut self, nnue_params: &NNUEParams, board: &Board, colour: Colour) {
        let mut curr_index = self.current_acc;
        loop {
            curr_index -= 1;

            if self.accumulators[curr_index].correct[colour] {
                break;
            }
        }

        let king = board.state.bbs.king_sq(colour);

        loop {
            self.materialise_new_acc_from(king, colour, curr_index + 1, nnue_params);

            self.accumulators[curr_index + 1].correct[colour] = true;

            curr_index += 1;
            if curr_index == self.current_acc {
                break;
            }
        }
    }

    /// Apply all in-flight updates, generating all the accumulators up to the current one.
    pub fn force(&mut self, board: &Board, nnue_params: &NNUEParams) {
        for colour in Colour::all() {
            if !self.accumulators[self.current_acc].correct[colour] {
                if self.can_efficiently_update(colour) {
                    self.apply_lazy_updates(nnue_params, board, colour);
                } else {
                    self.bucket_cache.load_accumulator_for_position(
                        nnue_params,
                        board.state.bbs,
                        colour,
                        &mut self.accumulators[self.current_acc],
                    );
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
        if self.accumulators[self.current_acc].correct[C::COLOUR] {
            return;
        }

        let oldest = self.try_find_computed_accumulator::<C>(pos);

        if let Some(source) = oldest {
            assert!(self.accumulators[source].correct[C::COLOUR]);
            // directly construct the top accumulator from the last-known-good one
            let mut curr_index = source;
            let king = pos.state.bbs.king_sq(C::COLOUR);
            let bucket = BUCKET_MAP[king.relative_to(C::COLOUR)];
            let weights = nnue_params.select_feature_weights(bucket);
            let mut adds = ArrayVec::<_, 32>::new();
            let mut subs = ArrayVec::<_, 32>::new();

            loop {
                for &add in self.accumulators[curr_index].update_buffer.adds() {
                    adds.push(feature::index(C::COLOUR, king, add));
                }
                for &sub in self.accumulators[curr_index].update_buffer.subs() {
                    subs.push(feature::index(C::COLOUR, king, sub));
                }

                curr_index += 1;

                if curr_index == self.current_acc {
                    break;
                }
            }

            *self.accumulators[self.current_acc].select_mut(C::COLOUR) =
                self.accumulators[source].select_mut(C::COLOUR).clone();
            accumulator::vector_update_inplace(
                self.accumulators[self.current_acc].select_mut(C::COLOUR),
                weights,
                &adds,
                &subs,
            );
            self.accumulators[self.current_acc].correct[C::COLOUR] = true;
        } else {
            self.bucket_cache.load_accumulator_for_position(
                nnue_params,
                pos.state.bbs,
                C::COLOUR,
                &mut self.accumulators[self.current_acc],
            );
        }
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
        while idx > 0 && !self.accumulators[idx].correct[C::COLOUR] {
            let curr = &self.accumulators[idx - 1];
            if curr.mv.piece.colour() == C::COLOUR
                && Self::requires_refresh(
                    curr.mv.piece,
                    curr.mv.from.relative_to(C::COLOUR),
                    curr.mv.to.relative_to(C::COLOUR),
                )
            {
                break;
            }
            let adds = curr.update_buffer.adds().len() as i32;
            let subs = curr.update_buffer.subs().len() as i32;
            budget -= adds + subs + 1;
            if budget < 0 {
                break;
            }
            idx -= 1;
        }
        if self.accumulators[idx].correct[C::COLOUR] {
            Some(idx)
        } else {
            None
        }
    }

    pub fn materialise_new_acc_from(
        &mut self,
        king: Square,
        colour: Colour,
        create_at_idx: usize,
        nnue_params: &NNUEParams,
    ) {
        let (front, back) = self.accumulators.split_at_mut(create_at_idx);
        let src_acc = front.last().unwrap();
        let tgt_acc = back.first_mut().unwrap();

        let bucket = BUCKET_MAP[king.relative_to(colour)];

        let bucket = nnue_params.select_feature_weights(bucket);

        let src = src_acc.select(colour);
        let tgt = tgt_acc.select_mut(colour);

        match (src_acc.update_buffer.adds(), src_acc.update_buffer.subs()) {
            // quiet or promotion
            (&[add], &[sub]) => {
                let add = feature::index(colour, king, add);
                let sub = feature::index(colour, king, sub);
                accumulator::vector_add_sub(src, tgt, bucket, add, sub);
            }
            // capture
            (&[add], &[sub1, sub2]) => {
                let add = feature::index(colour, king, add);
                let sub1 = feature::index(colour, king, sub1);
                let sub2 = feature::index(colour, king, sub2);
                accumulator::vector_add_sub2(src, tgt, bucket, add, sub1, sub2);
            }
            // castling
            (&[add1, add2], &[sub1, sub2]) => {
                let add1 = feature::index(colour, king, add1);
                let add2 = feature::index(colour, king, add2);
                let sub1 = feature::index(colour, king, sub1);
                let sub2 = feature::index(colour, king, sub2);
                accumulator::vector_add2_sub2(src, tgt, bucket, add1, add2, sub1, sub2);
            }
            (_, _) => panic!("invalid update buffer: {:?}", src_acc.update_buffer),
        }
    }

    /// Evaluate the final layer on the partial activations.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn evaluate(&self, nn: &NNUEParams, stm: Colour, out: usize) -> i32 {
        let acc = &self.accumulators[self.current_acc];

        debug_assert!(acc.correct[0] && acc.correct[1]);

        let (us, them) = if stm == Colour::White {
            (&acc.white, &acc.black)
        } else {
            (&acc.black, &acc.white)
        };

        let mut l1_outputs = Align64([0.0; L2_SIZE]);
        let mut l2_outputs = Align64([0.0; L3_SIZE]);
        let mut l3_output = 0.0;

        layers::activate_ft_and_propagate_l1(
            us,
            them,
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
        layers::propagate_l3(
            &l2_outputs,
            &nn.l3_weights[out],
            nn.l3_bias[out],
            &mut l3_output,
        );

        (l3_output * SCALE as f32) as i32
    }
}

/// Benchmark the inference portion of the NNUE evaluation.
/// (everything after the feature extraction)
pub fn inference_benchmark(state: &NNUEState, nnue_params: &NNUEParams) {
    let start = std::time::Instant::now();
    for _ in 0..1_000_000 {
        std::hint::black_box(std::hint::black_box(state).evaluate(
            std::hint::black_box(nnue_params),
            std::hint::black_box(Colour::White),
            std::hint::black_box(0),
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
    let (min, max) = nnue_params.min_max_feature_weight();
    println!("Min / Max FT values: {min} / {max}");
    Ok(())
}

impl NNUEParams {
    pub fn visualise_neuron(&self, neuron: usize, path: &std::path::Path) {
        #![allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        // remap pieces to keep opposite colours together
        static PIECE_REMAPPING: [usize; 12] = [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11];
        assert!(neuron < L1_SIZE);
        let starting_idx = neuron;
        let mut slice = Vec::with_capacity(768);
        for colour in Colour::all() {
            for piece_type in PieceType::all() {
                for square in Square::all() {
                    let feature_indices = {
                        let white_king = Square::H1;
                        let black_king = Square::H8;
                        let f = FeatureUpdate {
                            sq: square,
                            piece: Piece::new(colour, piece_type),
                        };
                        [
                            feature::index(Colour::White, white_king, f),
                            feature::index(Colour::Black, black_king, f),
                        ]
                    };
                    let index = feature_indices[Colour::White].index() * L1_SIZE + starting_idx;
                    slice.push(self.feature_weights[index]);
                }
            }
        }

        let mut image = Image::zeroed(8 * 6 + 5, 8 * 2 + 1); // + for inter-piece spacing

        for (piece, chunk) in slice.chunks(64).enumerate() {
            let piece = PIECE_REMAPPING[piece];
            let piece_colour = piece % 2;
            let piece_type = piece / 2;
            let (max, min) = if piece_type == 0 {
                let chunk = &chunk[8..56]; // first and last rank are always 0 for pawns
                (*chunk.iter().max().unwrap(), *chunk.iter().min().unwrap())
            } else {
                (*chunk.iter().max().unwrap(), *chunk.iter().min().unwrap())
            };
            let weight_to_colour = |weight: i16| -> u32 {
                let intensity = f32::from(weight - min) / f32::from(max - min);
                let idx = (intensity * 255.0).round() as u8;
                image::inferno_colour_map(idx)
            };
            for (square, &weight) in chunk.iter().enumerate() {
                let row = square / 8;
                let col = square % 8;
                let colour = if (row == 0 || row == 7) && piece_type == 0 {
                    0 // pawns on first and last rank are always 0
                } else {
                    weight_to_colour(weight)
                };
                image.set(
                    col + piece_type * 8 + piece_type,
                    row + piece_colour * 9,
                    colour,
                );
            }
        }

        let path = path.join(format!("neuron_{neuron}.tga"));
        image.save_as_tga(path);
    }

    pub fn min_max_feature_weight(&self) -> (i16, i16) {
        let mut min = i16::MAX;
        let mut max = i16::MIN;
        for &f in &self.feature_weights.0 {
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
