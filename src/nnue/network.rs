use std::{
    fmt::{Debug, Display},
    fs::File,
    io::BufReader,
    ops::{Deref, DerefMut},
};

use anyhow::Context;
use arrayvec::ArrayVec;

use crate::{
    board::{movegen::piecelayout::PieceLayout, Board},
    image::{self, Image},
    piece::{Black, Col, Colour, Piece, PieceType, White},
    util::{self, Square, MAX_DEPTH},
};

use super::accumulator::{self, Accumulator};

pub mod feature;
pub mod layers;

/// The size of the input layer of the network.
pub const INPUT: usize = 768;
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
/// Get indices into the feature transformer given king positions.
pub fn get_bucket_indices(white_king: Square, black_king: Square) -> [usize; 2] {
    let white_bucket = BUCKET_MAP[white_king];
    let black_bucket = BUCKET_MAP[black_king.flip_rank()];
    [white_bucket, black_bucket]
}
/// The number of output buckets
pub const OUTPUT_BUCKETS: usize = 8;
/// Get index into the output layer given a board state.
pub fn output_bucket(pos: &Board) -> usize {
    #![allow(clippy::cast_possible_truncation)]
    const DIVISOR: usize = (32 + OUTPUT_BUCKETS - 1) / OUTPUT_BUCKETS;
    (pos.n_men() as usize - 2) / DIVISOR
}

const QA: i16 = 255;
const QB: i16 = 64;

// read in the binary file containing the network parameters
// have to do some path manipulation to get relative paths to work
pub static COMPRESSED_NNUE: &[u8] = include_bytes!("../../viridithas.nnue.zst");

/// Struct representing the floating-point parameter file emmitted by bullet.
#[rustfmt::skip]
#[repr(C)]
struct UnquantisedNetwork {
    // extra bucket for the feature-factoriser.
    ft_weights:   [f32; INPUT * L1_SIZE * (BUCKETS + 1)],
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

const REPERMUTE_INDICES: [usize; L1_SIZE / 2] = [948, 151, 580, 671, 346, 625, 832, 44, 798, 783, 551, 355, 1009, 499, 896, 16, 805, 323, 555, 735, 767, 844, 808, 351, 255, 905, 161, 440, 308, 178, 674, 158, 285, 276, 594, 278, 809, 531, 658, 926, 96, 732, 589, 129, 406, 708, 205, 693, 1007, 812, 15, 543, 388, 889, 296, 742, 979, 274, 11, 51, 294, 787, 847, 888, 772, 599, 517, 442, 57, 68, 893, 788, 918, 137, 414, 235, 7, 250, 1003, 712, 412, 77, 132, 365, 554, 174, 113, 93, 62, 607, 47, 950, 697, 971, 1011, 76, 496, 786, 746, 737, 332, 128, 806, 247, 958, 819, 145, 343, 524, 14, 731, 491, 978, 24, 101, 845, 325, 723, 387, 861, 364, 3, 633, 635, 916, 192, 781, 204, 408, 380, 260, 972, 218, 797, 45, 486, 147, 863, 532, 748, 84, 227, 448, 1004, 656, 960, 272, 430, 683, 302, 774, 244, 289, 738, 65, 817, 615, 155, 144, 638, 238, 964, 490, 722, 804, 800, 359, 705, 528, 209, 747, 78, 9, 493, 617, 225, 403, 959, 191, 397, 461, 264, 659, 415, 628, 30, 377, 450, 901, 181, 67, 150, 887, 327, 336, 987, 868, 715, 837, 417, 760, 915, 156, 287, 546, 741, 880, 385, 386, 822, 153, 770, 938, 391, 165, 866, 200, 483, 443, 670, 573, 394, 229, 962, 526, 82, 383, 999, 38, 312, 506, 159, 500, 98, 333, 422, 775, 949, 133, 557, 578, 529, 719, 460, 586, 424, 848, 197, 273, 116, 610, 821, 95, 99, 756, 825, 378, 299, 505, 842, 538, 718, 927, 237, 329, 480, 838, 776, 269, 851, 347, 799, 242, 189, 74, 190, 36, 503, 49, 1021, 655, 631, 425, 186, 795, 375, 562, 876, 390, 859, 32, 634, 42, 124, 10, 41, 463, 645, 574, 253, 300, 982, 881, 72, 231, 230, 803, 587, 914, 970, 988, 257, 4, 429, 407, 955, 677, 432, 713, 536, 447, 481, 71, 286, 765, 66, 810, 358, 163, 243, 689, 251, 894, 878, 35, 241, 413, 758, 103, 966, 676, 233, 785, 434, 665, 492, 539, 427, 796, 125, 792, 865, 662, 768, 202, 618, 495, 348, 470, 942, 444, 27, 80, 855, 466, 476, 692, 530, 616, 641, 8, 1017, 730, 571, 445, 862, 657, 519, 26, 198, 703, 455, 922, 304, 477, 395, 326, 22, 897, 1006, 836, 597, 870, 270, 419, 508, 572, 1019, 750, 620, 567, 680, 221, 283, 931, 468, 627, 475, 816, 941, 762, 219, 591, 522, 1008, 85, 983, 707, 488, 664, 973, 515, 552, 18, 112, 83, 991, 162, 1001, 714, 733, 410, 282, 755, 884, 376, 248, 70, 214, 903, 612, 828, 256, 471, 910, 789, 340, 640, 702, 423, 489, 222, 605, 967, 857, 535, 171, 104, 940, 501, 341, 575, 990, 590, 449, 458, 169, 140, 63, 60, 928, 389, 514, 354, 537, 899, 545, 920, 217, 384, 280, 989, 744, 265, 814, 79, 534, 12, 224, 494, 548, 92, 134, 293, 87, 34, 542, 420, 652, 453, 995, 5, 729, 175, 934, 811, 860, 498, 523, 311, 773, 435, 157, 761, 533, 342, 711, 215, 108, 328, 912, 827, 284, 611, 303, 69, 757, 570, 885, 947, 114, 102, 956, 513, 433, 701, 864, 588, 261, 924, 187, 975, 188, 148, 933, 220, 405, 17, 726, 646, 1, 858, 484, 382, 600, 310, 569, 778, 917, 997, 815, 1018, 479, 841, 109, 563, 195, 642, 663, 891, 911, 974, 622, 840, 516, 472, 436, 485, 601, 199, 525, 1000, 698, 469, 309, 875, 268, 710, 1002, 142, 784, 751, 951, 240, 980, 139, 780, 301, 613, 606, 54, 576, 194, 565, 929, 20, 630, 152, 306, 650, 462, 428, 849, 431, 644, 141, 474, 402, 1020, 138, 986, 371, 977, 43, 882, 143, 170, 207, 521, 88, 593, 307, 392, 298, 86, 540, 820, 558, 833, 963, 369, 361, 900, 709, 399, 624, 97, 824, 123, 892, 368, 596, 177, 585, 210, 830, 53, 879, 118, 146, 1005, 497, 985, 122, 324, 263, 216, 598, 381, 614, 668, 954, 21, 908, 232, 75, 73, 418, 507, 944, 749, 473, 725, 367, 727, 675, 577, 647, 943, 766, 603, 362, 19, 40, 404, 1013, 935, 745, 59, 106, 874, 314, 932, 105, 28, 639, 366, 734, 592, 288, 629, 211, 164, 520, 437, 196, 994, 291, 409, 94, 441, 252, 313, 791, 172, 925, 131, 678, 854, 236, 185, 452, 684, 239, 136, 322, 316, 360, 130, 930, 794, 451, 512, 653, 206, 886, 64, 266, 271, 39, 898, 50, 695, 992, 834, 961, 179, 632, 258, 37, 890, 249, 2, 464, 566, 511, 454, 793, 318, 769, 81, 752, 319, 753, 945, 984, 334, 184, 439, 823, 350, 740, 691, 167, 416, 764, 337, 779, 720, 1016, 180, 651, 1012, 1010, 895, 281, 1023, 23, 58, 953, 913, 835, 246, 338, 679, 1022, 704, 672, 541, 374, 717, 396, 560, 518, 154, 623, 349, 564, 1015, 208, 801, 699, 119, 969, 13, 482, 621, 739, 681, 379, 0, 182, 509, 52, 687, 550, 846, 649, 790, 110, 976, 400, 465, 315, 111, 353, 277, 547, 909, 923, 33, 31, 121, 690, 149, 487, 317, 724, 6, 826, 90, 335, 648, 544, 608, 581, 813, 100, 673, 331, 582, 871, 279, 583, 609, 339, 993, 716, 902, 173, 320, 743, 852, 636, 457, 127, 906, 877, 579, 968, 626, 981, 29, 921, 295, 549, 654, 856, 637, 321, 759, 754, 213, 669, 667, 643, 120, 46, 700, 345, 559, 115, 363, 504, 706, 176, 946, 456, 688, 682, 467, 193, 721, 919, 292, 763, 619, 459, 438, 939, 201, 904, 356, 561, 839, 160, 510, 135, 883, 344, 426, 595, 685, 952, 254, 998, 853, 234, 398, 869, 183, 259, 411, 55, 936, 305, 728, 25, 203, 604, 275, 212, 850, 290, 166, 873, 736, 370, 226, 352, 61, 297, 782, 502, 56, 996, 1014, 357, 228, 245, 117, 957, 527, 907, 168, 584, 330, 771, 556, 373, 831, 262, 937, 660, 401, 818, 568, 777, 421, 872, 372, 867, 126, 91, 48, 807, 107, 829, 694, 446, 89, 267, 802, 393, 661, 696, 666, 686, 478, 553, 843, 223, 602, 965];

impl UnquantisedNetwork {
    /// Convert a parameter file generated by bullet into a quantised parameter set,
    /// for embedding into viri as a zstd-compressed archive. We do one processing
    /// step other than quantisation, namely merging the feature factoriser with the
    /// main king buckets.
    #[allow(clippy::cast_possible_truncation)]
    fn quantise(&self) -> Box<QuantisedNetwork> {
        const QA_BOUND: f32 = 1.98 * QA as f32;
        const QB_BOUND: f32 = 1.98 * QB as f32;

        let mut net = QuantisedNetwork::zeroed();
        // quantise the feature transformer weights, and merge the feature factoriser in.
        let mut buckets = self.ft_weights.chunks_exact(INPUT * L1_SIZE);
        let factoriser = buckets.next().unwrap();
        for (src_bucket, tgt_bucket) in buckets.zip(net.ft_weights.chunks_exact_mut(INPUT * L1_SIZE)) {
            // for repermuting the weights.
            for ((src, fac_src), tgt) in src_bucket.iter().zip(factoriser.iter()).zip(tgt_bucket.iter_mut()) {
                // extra clamp in case bucket + factoriser goes out of the clipping bounds
                let scaled = f32::clamp(*src + *fac_src, -1.98, 1.98) * f32::from(QA);
                *tgt = scaled.round() as i16;
            }
        }

        // quantise the feature transformer biases
        for (src, tgt) in self.ft_biases.iter().zip(net.ft_biases.iter_mut()) {
            let scaled = *src * f32::from(QA);
            assert!(scaled.abs() <= QA_BOUND, "feature transformer bias {scaled} is too large (max = {QA_BOUND})");
            *tgt = scaled.round() as i16;
        }

        // quantise (or not) later layers
        for i in 0..L1_SIZE {
            for bucket in 0..OUTPUT_BUCKETS {
                for j in 0..L2_SIZE {
                    let v = self.l1_weights[i][bucket][j] * f32::from(QB);
                    assert!(v.abs() <= QB_BOUND, "L1 weight {v} is too large (max = {QB_BOUND})");
                    let v = v.round() as i8;
                    net.l1_weights[i][bucket][j] = v;
                }
            }
        }

        // transfer the L1 biases
        net.l1_biases = self.l1_biases;
        net.l2_weights = self.l2_weights;
        net.l2_biases = self.l2_biases;
        net.l3_weights = self.l3_weights;
        net.l3_biases = self.l3_biases;

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
            let mem =
                std::slice::from_raw_parts_mut(util::from_mut(net.as_mut()).cast::<u8>(), std::mem::size_of::<Self>());
            reader.read_exact(mem)?;
            Ok(net)
        }
    }
}

impl QuantisedNetwork {
    /// Convert the network parameters into a format optimal for inference.
    #[allow(clippy::cognitive_complexity, clippy::needless_range_loop)]
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
            let mut weights: Vec<&mut PermChunk> =
                net.feature_weights.chunks_exact_mut(8).map(|a| a.try_into().unwrap()).collect();
            let mut biases: Vec<&mut PermChunk> =
                net.feature_bias.chunks_exact_mut(8).map(|a| a.try_into().unwrap()).collect();
            let num_chunks = std::mem::size_of::<PermChunk>() / std::mem::size_of::<i16>();

            #[cfg(target_feature = "avx512f")]
            let num_regs = 8;
            #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
            let num_regs = 4;
            #[cfg(all(target_feature = "ssse3", not(target_feature = "avx2"), not(target_feature = "avx512f")))]
            let num_regs = 2;
            #[cfg(not(any(target_feature = "ssse3", target_feature = "avx2", target_feature = "avx512f")))]
            let num_regs = 1;
            #[cfg(target_feature = "avx512f")]
            let order = [0, 2, 4, 6, 1, 3, 5, 7];
            #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
            let order = [0, 2, 1, 3];
            #[cfg(all(target_feature = "ssse3", not(target_feature = "avx2"), not(target_feature = "avx512f")))]
            let order = [0, 1];
            #[cfg(not(any(target_feature = "ssse3", target_feature = "avx2", target_feature = "avx512f")))]
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
                            net.l1_weights[bucket][i * L1_CHUNK_PER_32 * L2_SIZE + j * L1_CHUNK_PER_32 + k] =
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

fn repermute_l1_weights(sorted: &mut [[[i8; L2_SIZE]; OUTPUT_BUCKETS]], l1_weights: &[[[i8; L2_SIZE]; OUTPUT_BUCKETS]; L1_SIZE]) {
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
    pub fn decompress_and_alloc() -> anyhow::Result<Box<Self>> {
        #[cfg(not(feature = "zstd"))]
        type ZstdDecoder<R, D> = ruzstd::StreamingDecoder<R, D>;
        #[cfg(feature = "zstd")]
        type ZstdDecoder<'a, R> = zstd::stream::Decoder<'a, R>;

        // SAFETY: NNUEParams is composed entiredly of POD types, so we can
        // reinterpret it as bytes and write into it. We don't need to worry
        // about padding bytes, because the boxed NNUEParams is zeroed.
        unsafe {
            let mut net = QuantisedNetwork::zeroed();
            let mut mem = std::slice::from_raw_parts_mut(
                util::from_mut(net.as_mut()).cast::<u8>(),
                std::mem::size_of::<QuantisedNetwork>(),
            );
            let expected_bytes = mem.len() as u64;
            let mut decoder = ZstdDecoder::new(COMPRESSED_NNUE)
                .with_context(|| "Failed to construct zstd decoder for NNUE weights.")?;
            let bytes_written =
                std::io::copy(&mut decoder, &mut mem).with_context(|| "Failed to decompress NNUE weights.")?;
            anyhow::ensure!(bytes_written == expected_bytes, "encountered issue while decompressing NNUE weights, expected {expected_bytes} bytes, but got {bytes_written}");
            let use_simd = cfg!(target_feature = "ssse3");
            let net = net.permute(use_simd);
            Ok(net)
        }
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
    let mut reader = BufReader::new(File::open(input)?);
    let mut writer = File::create(output)?;
    let unquantised_net = UnquantisedNetwork::read(&mut reader)?;
    let net = unquantised_net.quantise();
    net.write(&mut writer)?;
    Ok(())
}

/// The size of the stack used to store the activations of the hidden layer.
const ACC_STACK_SIZE: usize = MAX_DEPTH.ply_to_horizon() + 1;

#[derive(Debug, Copy, Clone)]
pub struct PovUpdate {
    pub white: bool,
    pub black: bool,
}
impl PovUpdate {
    pub const BOTH: Self = Self { white: true, black: true };

    pub const fn colour(colour: Colour) -> Self {
        match colour {
            Colour::White => Self { white: true, black: false },
            Colour::Black => Self { white: false, black: true },
        }
    }
}

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
#[allow(clippy::large_stack_frames)]
#[derive(Clone)]
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
        pov_update: PovUpdate,
        acc: &mut Accumulator,
    ) {
        let side_we_care_about = if pov_update.white { Colour::White } else { Colour::Black };
        let wk = board_state.piece_bb(Piece::WK).first();
        let bk = board_state.piece_bb(Piece::BK).first();
        let [white_bucket, black_bucket] = get_bucket_indices(wk, bk);
        let bucket = if side_we_care_about == Colour::White { white_bucket } else { black_bucket };
        let cache_acc = self.accs[bucket].select_mut(side_we_care_about);

        let mut adds = ArrayVec::<_, 32>::new();
        let mut subs = ArrayVec::<_, 32>::new();
        self.board_states[side_we_care_about][bucket].update_iter(board_state, |f, is_add| {
            let [white_idx, black_idx] = feature::indices(wk, bk, f);
            let index = if side_we_care_about == Colour::White { white_idx } else { black_idx };
            if is_add {
                adds.push(index);
            } else {
                subs.push(index);
            }
        });

        let weights = nnue_params.select_feature_weights(bucket);

        accumulator::vector_update_inplace(cache_acc, weights, &adds, &subs);

        *acc.select_mut(side_we_care_about) = cache_acc.clone();
        acc.correct[side_we_care_about] = true;

        self.board_states[side_we_care_about][bucket] = board_state;
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
#[allow(clippy::upper_case_acronyms, clippy::large_stack_frames)]
#[derive(Clone)]
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
            acc.init(&nnue_params.feature_bias, PovUpdate::BOTH);
        }
        // initialise all the board states in the bucket cache to the empty board
        for board_state in self.bucket_cache.board_states.iter_mut().flatten() {
            *board_state = PieceLayout::NULL;
        }

        // refresh the first accumulator
        self.bucket_cache.load_accumulator_for_position(
            nnue_params,
            board.pieces,
            PovUpdate { white: true, black: false },
            &mut self.accumulators[0],
        );
        self.bucket_cache.load_accumulator_for_position(
            nnue_params,
            board.pieces,
            PovUpdate { white: false, black: true },
            &mut self.accumulators[0],
        );
    }

    fn requires_refresh(piece: Piece, from: Square, to: Square) -> bool {
        if piece.piece_type() != PieceType::King {
            return false;
        }

        BUCKET_MAP[from] != BUCKET_MAP[to]
    }

    fn can_efficiently_update(&self, view: Colour) -> bool {
        let mut curr_idx = self.current_acc;
        loop {
            curr_idx -= 1;
            let curr = &self.accumulators[curr_idx];

            let mv = curr.mv;
            let from = mv.from.relative_to(view);
            let to = mv.to.relative_to(view);
            let piece = mv.piece;

            if piece.colour() == view && Self::requires_refresh(piece, from, to) {
                return false;
            }
            if curr.correct[view] {
                return true;
            }
        }
    }

    fn apply_lazy_updates(&mut self, nnue_params: &NNUEParams, board: &Board, view: Colour) {
        let mut curr_index = self.current_acc;
        loop {
            curr_index -= 1;

            if self.accumulators[curr_index].correct[view] {
                break;
            }
        }

        let pov_update = PovUpdate::colour(view);
        let white_king = board.king_sq(Colour::White);
        let black_king = board.king_sq(Colour::Black);

        loop {
            self.materialise_new_acc_from(white_king, black_king, pov_update, curr_index + 1, nnue_params);

            self.accumulators[curr_index + 1].correct[view] = true;

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
                        board.pieces,
                        PovUpdate::colour(colour),
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

    fn hint_common_access_for_perspective<C: Col>(&mut self, pos: &Board, nnue_params: &NNUEParams) {
        if self.accumulators[self.current_acc].correct[C::COLOUR] {
            return;
        }

        let oldest = self.try_find_computed_accumulator::<C>(pos);

        if let Some(source) = oldest {
            assert!(self.accumulators[source].correct[C::COLOUR]);
            // directly construct the top accumulator from the last-known-good one
            let mut curr_index = source;
            let wk = pos.king_sq(Colour::White);
            let bk = pos.king_sq(Colour::Black);
            let [white_bucket, black_bucket] = get_bucket_indices(wk, bk);
            let bucket = if C::COLOUR == Colour::White { white_bucket } else { black_bucket };
            let weights = nnue_params.select_feature_weights(bucket);
            let mut adds = ArrayVec::<_, 32>::new();
            let mut subs = ArrayVec::<_, 32>::new();

            loop {
                for &add in self.accumulators[curr_index].update_buffer.adds() {
                    let add = feature::indices(wk, bk, add)[C::COLOUR];
                    adds.push(add);
                }
                for &sub in self.accumulators[curr_index].update_buffer.subs() {
                    let sub = feature::indices(wk, bk, sub)[C::COLOUR];
                    subs.push(sub);
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
                pos.pieces,
                PovUpdate::colour(C::COLOUR),
                &mut self.accumulators[self.current_acc],
            );
        }
    }

    /// Find the index of the first materialised accumulator, or nothing
    /// if moving back that far would be too costly.
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn try_find_computed_accumulator<C: Col>(&self, pos: &Board) -> Option<usize> {
        let mut idx = self.current_acc;
        let mut budget = pos.pieces.occupied().count() as i32;
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
        white_king: Square,
        black_king: Square,
        pov_update: PovUpdate,
        create_at_idx: usize,
        nnue_params: &NNUEParams,
    ) {
        let (front, back) = self.accumulators.split_at_mut(create_at_idx);
        let src = front.last().unwrap();
        let tgt = back.first_mut().unwrap();

        match (src.update_buffer.adds(), src.update_buffer.subs()) {
            // quiet or promotion
            (&[add], &[sub]) => {
                Self::apply_quiet(white_king, black_king, add, sub, pov_update, src, tgt, nnue_params);
            }
            // capture
            (&[add], &[sub1, sub2]) => {
                Self::apply_capture(white_king, black_king, add, sub1, sub2, pov_update, src, tgt, nnue_params);
            }
            // castling
            (&[add1, add2], &[sub1, sub2]) => {
                Self::apply_castling(white_king, black_king, add1, add2, sub1, sub2, pov_update, src, tgt, nnue_params);
            }
            (_, _) => panic!("invalid update buffer: {:?}", src.update_buffer),
        }
    }

    /// Move a single piece on the board.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_quiet(
        white_king: Square,
        black_king: Square,
        add: FeatureUpdate,
        sub: FeatureUpdate,
        update: PovUpdate,
        src: &Accumulator,
        tgt: &mut Accumulator,
        nnue_params: &NNUEParams,
    ) {
        let [w_add, b_add] = feature::indices(white_king, black_king, add);
        let [w_sub, b_sub] = feature::indices(white_king, black_king, sub);

        let [white_bucket, black_bucket] = get_bucket_indices(white_king, black_king);

        let w_bucket = nnue_params.select_feature_weights(white_bucket);
        let b_bucket = nnue_params.select_feature_weights(black_bucket);

        if update.white {
            accumulator::vector_add_sub(&src.white, &mut tgt.white, w_bucket, w_add, w_sub);
        }
        if update.black {
            accumulator::vector_add_sub(&src.black, &mut tgt.black, b_bucket, b_add, b_sub);
        }
    }

    /// Make a capture on the board.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_capture(
        white_king: Square,
        black_king: Square,
        add: FeatureUpdate,
        sub1: FeatureUpdate,
        sub2: FeatureUpdate,
        update: PovUpdate,
        src: &Accumulator,
        tgt: &mut Accumulator,
        nnue_params: &NNUEParams,
    ) {
        let [white_add, black_add] = feature::indices(white_king, black_king, add);
        let [white_sub1, black_sub1] = feature::indices(white_king, black_king, sub1);
        let [white_sub2, black_sub2] = feature::indices(white_king, black_king, sub2);

        let [white_bucket, black_bucket] = get_bucket_indices(white_king, black_king);

        let white_bucket = nnue_params.select_feature_weights(white_bucket);
        let black_bucket = nnue_params.select_feature_weights(black_bucket);

        if update.white {
            accumulator::vector_add_sub2(&src.white, &mut tgt.white, white_bucket, white_add, white_sub1, white_sub2);
        }
        if update.black {
            accumulator::vector_add_sub2(&src.black, &mut tgt.black, black_bucket, black_add, black_sub1, black_sub2);
        }
    }

    /// Make a castling move on the board.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_castling(
        white_king: Square,
        black_king: Square,
        add1: FeatureUpdate,
        add2: FeatureUpdate,
        sub1: FeatureUpdate,
        sub2: FeatureUpdate,
        update: PovUpdate,
        src: &Accumulator,
        tgt: &mut Accumulator,
        nnue_params: &NNUEParams,
    ) {
        let [white_add1, black_add1] = feature::indices(white_king, black_king, add1);
        let [white_add2, black_add2] = feature::indices(white_king, black_king, add2);
        let [white_sub1, black_sub1] = feature::indices(white_king, black_king, sub1);
        let [white_sub2, black_sub2] = feature::indices(white_king, black_king, sub2);

        let [white_bucket, black_bucket] = get_bucket_indices(white_king, black_king);

        let white_bucket = nnue_params.select_feature_weights(white_bucket);
        let black_bucket = nnue_params.select_feature_weights(black_bucket);

        if update.white {
            accumulator::vector_add2_sub2(
                &src.white,
                &mut tgt.white,
                white_bucket,
                white_add1,
                white_add2,
                white_sub1,
                white_sub2,
            );
        }
        if update.black {
            accumulator::vector_add2_sub2(
                &src.black,
                &mut tgt.black,
                black_bucket,
                black_add1,
                black_add2,
                black_sub1,
                black_sub2,
            );
        }
    }

    /// Evaluate the final layer on the partial activations.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn evaluate(&self, nn: &NNUEParams, stm: Colour, out: usize) -> i32 {
        let acc = &self.accumulators[self.current_acc];

        debug_assert!(acc.correct[0] && acc.correct[1]);

        let (us, them) = if stm == Colour::White { (&acc.white, &acc.black) } else { (&acc.black, &acc.white) };

        let mut l1_outputs = Align64([0.0; L2_SIZE]);
        let mut l2_outputs = Align64([0.0; L3_SIZE]);
        let mut l3_output = 0.0;

        layers::activate_ft_and_propagate_l1(us, them, &nn.l1_weights[out], &nn.l1_bias[out], &mut l1_outputs);
        layers::propagate_l2(&l1_outputs, &nn.l2_weights[out], &nn.l2_bias[out], &mut l2_outputs);
        layers::propagate_l3(&l2_outputs, &nn.l3_weights[out], nn.l3_bias[out], &mut l3_output);

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
    std::fs::create_dir_all(&path).with_context(|| "Failed to create NNUE visualisations folder.")?;
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
                    let feature_indices = feature::indices(
                        Square::H1,
                        Square::H8,
                        FeatureUpdate { sq: square, piece: Piece::new(colour, piece_type) },
                    );
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
                image.set(col + piece_type * 8 + piece_type, row + piece_colour * 9, colour);
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
