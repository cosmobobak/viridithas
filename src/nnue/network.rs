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

const REPERMUTE_INDICES: [usize; L1_SIZE / 2] = {
    let mut indices = [0; L1_SIZE / 2];
    let mut i = 0;
    while i < L1_SIZE / 2 {
        indices[i] = i;
        i += 1;
    }
    indices
};

// const REPERMUTE_INDICES: [usize; L1_SIZE / 2] = [840, 838, 168, 364, 27, 147, 350, 469, 825, 343, 279, 759, 480, 78, 284, 483, 153, 80, 685, 872, 623, 436, 844, 1020, 824, 478, 694, 774, 686, 987, 337, 87, 495, 597, 487, 524, 120, 88, 360, 456, 702, 766, 744, 789, 1, 239, 803, 417, 333, 608, 368, 204, 301, 426, 385, 963, 843, 262, 526, 970, 883, 109, 401, 915, 36, 708, 948, 375, 118, 528, 905, 16, 717, 444, 243, 218, 180, 197, 43, 1001, 636, 199, 299, 796, 395, 21, 442, 511, 681, 158, 679, 598, 895, 520, 585, 1018, 509, 25, 131, 400, 277, 125, 182, 484, 688, 91, 479, 128, 216, 170, 581, 918, 268, 431, 863, 746, 238, 938, 105, 228, 293, 34, 414, 376, 576, 157, 741, 852, 492, 797, 622, 54, 297, 129, 728, 595, 537, 571, 854, 256, 315, 943, 831, 543, 639, 610, 745, 920, 485, 633, 267, 501, 737, 917, 514, 410, 990, 669, 453, 331, 196, 880, 813, 47, 723, 641, 248, 278, 121, 529, 516, 566, 448, 866, 348, 184, 772, 306, 402, 758, 981, 302, 835, 452, 15, 629, 229, 127, 496, 504, 270, 468, 86, 853, 934, 491, 734, 553, 942, 290, 369, 670, 664, 202, 213, 22, 579, 482, 236, 193, 477, 253, 380, 510, 906, 856, 49, 650, 443, 409, 931, 682, 321, 457, 503, 951, 1008, 276, 313, 163, 921, 995, 668, 827, 455, 308, 40, 877, 94, 933, 396, 894, 422, 223, 660, 538, 865, 704, 411, 142, 188, 465, 955, 830, 433, 140, 51, 397, 563, 66, 527, 839, 116, 782, 698, 439, 980, 882, 617, 750, 548, 275, 793, 381, 1016, 490, 474, 881, 99, 257, 135, 365, 837, 265, 644, 817, 513, 535, 8, 359, 982, 507, 338, 713, 10, 189, 435, 727, 370, 440, 146, 190, 62, 176, 523, 152, 155, 144, 540, 240, 795, 602, 65, 926, 74, 319, 773, 940, 45, 967, 873, 280, 994, 33, 407, 493, 604, 263, 783, 584, 663, 710, 525, 977, 35, 20, 464, 192, 826, 48, 1003, 821, 899, 577, 221, 73, 106, 373, 71, 209, 935, 649, 269, 859, 232, 536, 292, 769, 434, 72, 324, 778, 570, 361, 371, 619, 247, 757, 771, 342, 412, 60, 421, 711, 946, 736, 960, 953, 845, 654, 897, 547, 868, 884, 291, 901, 386, 219, 612, 645, 986, 947, 472, 964, 64, 388, 791, 675, 420, 601, 954, 600, 200, 707, 134, 310, 808, 258, 1015, 150, 89, 790, 181, 984, 779, 183, 855, 377, 505, 389, 683, 716, 862, 458, 731, 763, 286, 266, 676, 374, 497, 167, 929, 672, 642, 599, 285, 578, 560, 353, 210, 517, 283, 327, 287, 893, 356, 902, 973, 334, 903, 220, 760, 691, 542, 288, 703, 810, 767, 50, 557, 848, 857, 591, 822, 462, 701, 620, 515, 61, 384, 936, 624, 486, 635, 768, 889, 841, 860, 237, 349, 1017, 809, 44, 816, 419, 637, 178, 244, 326, 447, 110, 177, 316, 264, 989, 101, 508, 222, 621, 460, 561, 476, 945, 721, 30, 58, 646, 574, 205, 83, 273, 325, 345, 787, 961, 607, 531, 38, 37, 31, 587, 339, 415, 53, 613, 466, 569, 697, 939, 274, 871, 956, 42, 594, 32, 394, 628, 282, 770, 850, 1011, 425, 117, 993, 56, 14, 661, 296, 449, 226, 2, 648, 133, 7, 186, 798, 5, 161, 534, 784, 1012, 700, 39, 678, 530, 558, 786, 590, 46, 634, 846, 818, 781, 807, 556, 705, 937, 154, 544, 445, 861, 801, 693, 108, 502, 379, 59, 550, 405, 564, 347, 765, 609, 988, 966, 156, 847, 991, 596, 743, 169, 68, 1002, 870, 90, 565, 195, 294, 473, 217, 235, 305, 4, 304, 114, 317, 1000, 864, 699, 588, 241, 811, 834, 115, 191, 362, 552, 910, 932, 272, 726, 307, 67, 792, 533, 488, 13, 950, 254, 923, 311, 975, 430, 999, 885, 573, 748, 733, 165, 589, 665, 692, 398, 126, 927, 521, 996, 390, 626, 112, 81, 271, 76, 833, 175, 974, 11, 355, 555, 896, 351, 888, 618, 876, 832, 467, 408, 225, 605, 1009, 892, 77, 687, 667, 187, 0, 689, 559, 79, 215, 95, 814, 69, 742, 997, 489, 1019, 998, 102, 233, 211, 340, 17, 706, 928, 869, 652, 546, 715, 298, 113, 98, 84, 806, 393, 958, 3, 122, 52, 751, 250, 673, 625, 729, 214, 320, 968, 829, 657, 57, 104, 674, 851, 777, 886, 512, 70, 423, 799, 722, 451, 735, 1023, 461, 925, 231, 965, 738, 842, 913, 446, 143, 432, 363, 309, 914, 725, 185, 658, 93, 1022, 891, 261, 194, 922, 75, 138, 336, 404, 100, 399, 413, 494, 139, 224, 162, 712, 709, 611, 651, 874, 898, 383, 638, 323, 985, 575, 295, 976, 97, 437, 145, 630, 819, 500, 230, 992, 788, 655, 441, 580, 438, 959, 592, 983, 459, 136, 322, 366, 506, 92, 762, 85, 1004, 631, 541, 690, 780, 568, 603, 632, 662, 907, 732, 329, 828, 804, 406, 151, 206, 242, 812, 227, 159, 761, 656, 29, 18, 119, 344, 160, 260, 303, 251, 234, 593, 972, 312, 522, 1005, 137, 289, 328, 367, 900, 941, 532, 198, 815, 103, 203, 63, 26, 427, 392, 696, 201, 754, 372, 952, 858, 666, 640, 719, 281, 107, 149, 382, 416, 912, 130, 908, 23, 671, 919, 358, 179, 1021, 714, 470, 957, 429, 909, 55, 148, 518, 428, 615, 314, 387, 357, 28, 172, 208, 677, 582, 653, 539, 164, 1007, 24, 207, 606, 499, 684, 911, 823, 391, 740, 171, 904, 124, 756, 680, 794, 752, 519, 878, 971, 916, 424, 551, 300, 627, 747, 330, 481, 41, 249, 567, 805, 471, 836, 879, 875, 775, 764, 616, 659, 820, 586, 1014, 332, 132, 785, 96, 9, 141, 111, 867, 463, 890, 724, 583, 255, 695, 498, 245, 647, 730, 979, 403, 944, 166, 978, 352, 318, 718, 418, 6, 123, 450, 720, 572, 949, 1010, 174, 212, 962, 802, 454, 887, 749, 82, 354, 562, 346, 755, 614, 753, 545, 549, 969, 378, 475, 259, 12, 776, 800, 924, 930, 341, 246, 1013, 643, 335, 1006, 252, 849, 554, 739, 19, 173];

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

fn repermute_l1_weights(sorted: &mut [[[i8; 16]; 8]], l1_weights: &[[[i8; 16]; 8]; 2048]) {
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
