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
    piece::{Colour, Piece, PieceType},
    util::{Square, MAX_DEPTH},
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
pub fn get_bucket_indices(white_king: Square, black_king: Square) -> (usize, usize) {
    let white_bucket = BUCKET_MAP[white_king];
    let black_bucket = BUCKET_MAP[black_king.flip_rank()];
    (white_bucket, black_bucket)
}
/// The number of output buckets
pub const OUTPUT_BUCKETS: usize = 8;
/// Get index into the output layer given a board state.
pub fn output_bucket(pos: &Board) -> usize {
    #![allow(clippy::cast_possible_truncation)]
    const DIVISOR: usize = (32 + OUTPUT_BUCKETS - 1) / OUTPUT_BUCKETS;
    (pos.n_men() as usize - 2) / DIVISOR
}

const QA: i32 = 255;
const QB: i32 = 64;

// read in the binary file containing the network parameters
// have to do some path manipulation to get relative paths to work
pub static COMPRESSED_NNUE: &[u8] = include_bytes!("../../viridithas.nnue.zst");

#[repr(C)]
struct UnquantisedNetwork {
    // extra bucket for the feature-factoriser.
    ft_weights: [f32; INPUT * L1_SIZE * (BUCKETS + 1)],
    ft_biases: [f32; L1_SIZE],
    l1_weights: [[[f32; L2_SIZE]; OUTPUT_BUCKETS]; L1_SIZE],
    l1_biases: [[f32; L2_SIZE]; OUTPUT_BUCKETS],
    l2_weights: [[[f32; L3_SIZE]; OUTPUT_BUCKETS]; L2_SIZE],
    l2_biases: [[f32; L3_SIZE]; OUTPUT_BUCKETS],
    l3_weights: [[f32; OUTPUT_BUCKETS]; L3_SIZE],
    l3_biases: [f32; OUTPUT_BUCKETS],
}

#[repr(C)]
pub struct NNUEParams {
    pub feature_weights: Align64<[i16; INPUT * L1_SIZE * BUCKETS]>,
    pub feature_bias: Align64<[i16; L1_SIZE]>,
    pub l1_weights: [Align64<[i8; L1_SIZE * L2_SIZE]>; OUTPUT_BUCKETS],
    pub l1_bias: [Align64<[f32; L2_SIZE]>; OUTPUT_BUCKETS],
    pub l2_weights: [Align64<[f32; L2_SIZE * L3_SIZE]>; OUTPUT_BUCKETS],
    pub l2_bias: [Align64<[f32; L3_SIZE]>; OUTPUT_BUCKETS],
    pub l3_weights: [Align64<[f32; L3_SIZE]>; OUTPUT_BUCKETS],
    pub l3_bias: [f32; OUTPUT_BUCKETS],
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

// const REPERMUTE_INDICES: [usize; L1_SIZE / 2] = [840, 168, 838, 364, 27, 147, 350, 469, 825, 480, 343, 78, 759, 685, 153, 279, 284, 483, 623, 436, 80, 872, 824, 844, 1020, 774, 478, 337, 495, 120, 686, 694, 987, 597, 87, 789, 1, 702, 524, 456, 375, 766, 88, 487, 239, 803, 360, 262, 243, 417, 157, 963, 526, 608, 197, 109, 883, 333, 204, 843, 492, 744, 717, 25, 401, 796, 36, 905, 170, 301, 520, 368, 118, 1001, 199, 385, 509, 708, 576, 442, 128, 970, 105, 598, 895, 268, 158, 400, 681, 636, 915, 511, 477, 938, 948, 431, 15, 21, 863, 180, 182, 131, 581, 679, 1018, 688, 444, 484, 395, 43, 16, 622, 34, 315, 267, 238, 746, 990, 585, 125, 453, 852, 479, 918, 299, 795, 331, 297, 45, 414, 248, 633, 682, 797, 741, 277, 216, 202, 854, 369, 426, 537, 376, 943, 504, 121, 853, 293, 934, 54, 196, 410, 501, 129, 831, 917, 737, 641, 443, 402, 866, 595, 528, 184, 380, 745, 1008, 256, 880, 566, 47, 723, 981, 253, 229, 49, 629, 553, 835, 91, 813, 228, 278, 452, 514, 306, 704, 529, 1016, 86, 951, 409, 457, 510, 763, 396, 728, 485, 639, 516, 827, 276, 669, 942, 275, 830, 455, 877, 734, 906, 213, 348, 218, 650, 921, 270, 448, 468, 142, 980, 610, 188, 127, 758, 571, 920, 313, 548, 540, 664, 579, 668, 240, 503, 933, 302, 491, 40, 193, 308, 22, 435, 236, 782, 523, 290, 873, 223, 995, 8, 496, 135, 116, 670, 257, 543, 465, 74, 698, 982, 856, 955, 513, 736, 882, 89, 422, 527, 826, 773, 644, 440, 94, 577, 946, 563, 474, 359, 411, 324, 772, 66, 51, 649, 321, 286, 750, 189, 265, 839, 280, 73, 10, 490, 482, 507, 433, 365, 155, 727, 663, 926, 894, 269, 617, 472, 62, 793, 163, 373, 381, 33, 421, 176, 538, 144, 967, 99, 285, 783, 397, 977, 940, 953, 602, 1003, 106, 20, 881, 964, 899, 192, 535, 660, 986, 604, 710, 247, 146, 1015, 338, 263, 931, 713, 767, 370, 83, 48, 778, 35, 711, 808, 619, 707, 508, 291, 71, 947, 65, 356, 150, 319, 377, 791, 72, 140, 412, 264, 342, 837, 525, 570, 64, 167, 884, 757, 578, 654, 769, 675, 536, 388, 845, 560, 865, 642, 635, 821, 760, 221, 493, 584, 219, 389, 716, 190, 439, 497, 310, 672, 361, 960, 929, 232, 768, 464, 407, 935, 486, 69, 210, 37, 771, 897, 434, 178, 779, 42, 183, 829, 134, 850, 347, 505, 600, 371, 244, 517, 287, 973, 420, 209, 862, 591, 848, 612, 984, 44, 889, 60, 258, 531, 334, 386, 50, 32, 601, 374, 200, 274, 645, 419, 394, 31, 637, 220, 683, 515, 30, 770, 703, 599, 288, 810, 868, 186, 859, 902, 110, 594, 550, 97, 458, 1017, 701, 556, 266, 353, 1012, 857, 349, 607, 994, 841, 787, 462, 860, 993, 273, 855, 327, 613, 547, 316, 181, 822, 700, 937, 574, 226, 705, 624, 449, 445, 14, 425, 325, 108, 61, 871, 542, 621, 901, 292, 691, 790, 846, 53, 59, 816, 217, 587, 936, 557, 801, 205, 561, 954, 235, 222, 798, 765, 784, 5, 989, 154, 460, 283, 282, 530, 237, 596, 676, 903, 366, 996, 558, 620, 476, 721, 46, 117, 648, 152, 564, 781, 817, 786, 939, 661, 628, 384, 807, 304, 58, 177, 398, 339, 956, 569, 38, 317, 945, 101, 966, 502, 345, 195, 254, 544, 415, 326, 7, 56, 697, 294, 67, 133, 405, 447, 731, 809, 68, 13, 743, 161, 678, 114, 590, 609, 885, 665, 975, 693, 552, 847, 811, 90, 729, 169, 307, 305, 115, 296, 194, 1000, 818, 892, 861, 634, 311, 393, 473, 888, 156, 362, 559, 927, 1002, 832, 0, 699, 814, 4, 834, 928, 534, 191, 11, 588, 748, 974, 241, 573, 733, 466, 81, 233, 886, 864, 991, 272, 298, 2, 950, 489, 546, 187, 910, 726, 165, 211, 687, 870, 667, 225, 76, 521, 689, 320, 390, 39, 923, 896, 833, 383, 79, 533, 692, 351, 77, 732, 706, 709, 340, 932, 175, 988, 893, 126, 1019, 673, 143, 104, 751, 379, 589, 541, 735, 423, 657, 52, 842, 958, 968, 224, 461, 806, 3, 70, 112, 113, 215, 605, 618, 98, 432, 792, 75, 625, 869, 876, 57, 122, 430, 100, 712, 780, 413, 799, 84, 999, 404, 925, 93, 95, 271, 914, 494, 626, 715, 1011, 575, 851, 102, 1009, 214, 998, 295, 568, 738, 961, 565, 309, 555, 985, 162, 632, 874, 17, 658, 506, 891, 725, 261, 363, 488, 762, 913, 185, 646, 446, 145, 662, 1022, 965, 250, 922, 139, 159, 638, 18, 997, 630, 138, 459, 230, 85, 690, 652, 674, 234, 242, 231, 336, 408, 522, 441, 1004, 959, 1023, 898, 92, 512, 322, 909, 451, 580, 651, 467, 399, 119, 1005, 160, 592, 136, 655, 151, 437, 367, 722, 992, 907, 593, 719, 819, 329, 912, 804, 416, 858, 29, 438, 198, 684, 788, 500, 328, 312, 631, 582, 983, 777, 603, 355, 281, 611, 752, 332, 344, 260, 656, 137, 919, 900, 387, 714, 358, 372, 941, 201, 314, 972, 323, 828, 812, 761, 179, 532, 382, 63, 904, 303, 357, 227, 55, 424, 107, 289, 976, 406, 754, 567, 666, 251, 463, 103, 794, 203, 28, 742, 653, 908, 916, 164, 392, 428, 805, 206, 130, 659, 952, 677, 1021, 26, 123, 208, 671, 836, 785, 470, 971, 499, 640, 724, 481, 820, 172, 429, 23, 519, 539, 747, 391, 957, 551, 756, 606, 1007, 647, 696, 207, 615, 979, 171, 815, 124, 148, 740, 41, 627, 471, 330, 720, 911, 300, 149, 586, 9, 878, 427, 24, 245, 249, 255, 875, 730, 695, 583, 132, 96, 680, 111, 949, 141, 764, 403, 823, 318, 518, 879, 944, 775, 616, 572, 498, 867, 352, 475, 418, 1014, 166, 890, 978, 82, 718, 802, 354, 450, 454, 212, 755, 6, 1010, 887, 753, 562, 545, 549, 969, 174, 614, 962, 346, 749, 378, 12, 776, 259, 800, 1013, 173, 930, 1006, 19, 849, 335, 739, 554, 246, 252, 643, 924, 341];

impl UnquantisedNetwork {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cognitive_complexity,
        clippy::needless_range_loop
    )]
    fn process(&self, use_simd: bool) -> Box<NNUEParams> {
        const QA_BOUND: f32 = 1.98 * QA as f32;
        const QB_BOUND: f32 = 1.98 * QB as f32;

        let mut net = NNUEParams::zeroed();
        // quantise the feature transformer weights
        let mut buckets = self.ft_weights.chunks_exact(INPUT * L1_SIZE);
        let factoriser = buckets.next().unwrap();
        for (src_bucket, tgt_bucket) in buckets.zip(net.feature_weights.chunks_exact_mut(INPUT * L1_SIZE)) {
            // for repermuting the weights.
            let mut unsorted = vec![0i16; INPUT * L1_SIZE];
            for ((src, fac_src), tgt) in src_bucket.iter().zip(factoriser.iter()).zip(unsorted.iter_mut()) {
                let scaled = f32::clamp(*src + *fac_src, -1.98, 1.98) * QA as f32;
                *tgt = scaled.round() as i16;
            }
            repermute_ft_bucket(tgt_bucket, &unsorted);
        }

        // quantise the feature transformer biases
        let mut unsorted = vec![0i16; L1_SIZE];
        for (src, tgt) in self.ft_biases.iter().zip(unsorted.iter_mut()) {
            let scaled = *src * QA as f32;
            assert!(scaled.abs() <= QA_BOUND, "feature transformer bias {scaled} is too large (max = {QA_BOUND})");
            *tgt = scaled.round() as i16;
        }
        repermute_ft_bias(&mut net.feature_bias, &unsorted);

        // transpose the L{1,2,3} weights and biases
        let mut sorted = vec![[[0f32; L2_SIZE]; OUTPUT_BUCKETS]; L1_SIZE];
        let l1_weights = &self.l1_weights;
        repermute_l1_weights(&mut sorted, l1_weights);
        for bucket in 0..OUTPUT_BUCKETS {
            // quant the L1 weights
            if use_simd {
                for i in 0..L1_SIZE / L1_CHUNK_PER_32 {
                    for j in 0..L2_SIZE {
                        for k in 0..L1_CHUNK_PER_32 {
                            let v = sorted[i * L1_CHUNK_PER_32 + k][bucket][j] * QB as f32;
                            assert!(v.abs() <= QB_BOUND, "L1 weight {v} is too large (max = {QB_BOUND})");
                            let v = v.round() as i8;
                            net.l1_weights[bucket][i * L1_CHUNK_PER_32 * L2_SIZE + j * L1_CHUNK_PER_32 + k] = v;
                        }
                    }
                }
            } else {
                for i in 0..L1_SIZE {
                    for j in 0..L2_SIZE {
                        let v = sorted[i][bucket][j] * QB as f32;
                        assert!(v.abs() <= QB_BOUND, "L1 weight {v} is too large (max = {QB_BOUND})");
                        let v = v.round() as i8;
                        net.l1_weights[bucket][j * L1_SIZE + i] = v;
                    }
                }
            }

            // transfer the L1 biases
            for i in 0..L2_SIZE {
                net.l1_bias[bucket][i] = self.l1_biases[bucket][i];
            }

            // transpose the L2 weights
            if use_simd {
                for i in 0..L2_SIZE {
                    for j in 0..L3_SIZE {
                        net.l2_weights[bucket][i * L3_SIZE + j] = self.l2_weights[i][bucket][j];
                    }
                }
            } else {
                for i in 0..L2_SIZE {
                    for j in 0..L3_SIZE {
                        net.l2_weights[bucket][j * L2_SIZE + i] = self.l2_weights[i][bucket][j];
                    }
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

    fn read(reader: &mut impl std::io::Read) -> anyhow::Result<Box<Self>> {
        // SAFETY: NNUEParams can be zeroed.
        unsafe {
            let layout = std::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc_zeroed(layout);
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            #[allow(clippy::cast_ptr_alignment)]
            let mut net = Box::from_raw(ptr.cast::<Self>());
            let mem = std::slice::from_raw_parts_mut(std::ptr::from_mut(net.as_mut()).cast::<u8>(), layout.size());
            reader.read_exact(mem)?;
            Ok(net)
        }
    }
}

fn repermute_l1_weights(sorted: &mut [[[f32; 16]; 8]], l1_weights: &[[[f32; 16]; 8]; 2048]) {
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
        // SAFETY: NNUEParams can be zeroed.
        unsafe {
            let layout = std::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc_zeroed(layout);
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            #[allow(clippy::cast_ptr_alignment)]
            let mut net = Box::from_raw(ptr.cast::<Self>());
            let mut mem = std::slice::from_raw_parts_mut(std::ptr::from_mut(net.as_mut()).cast::<u8>(), layout.size());
            let expected_bytes = mem.len() as u64;
            let mut decoder = ruzstd::StreamingDecoder::new(COMPRESSED_NNUE)
                .with_context(|| "Failed to construct zstd decoder for NNUE weights.")?;
            let bytes_written =
                std::io::copy(&mut decoder, &mut mem).with_context(|| "Failed to decompress NNUE weights.")?;
            anyhow::ensure!(bytes_written == expected_bytes, "encountered issue while decompressing NNUE weights, expected {expected_bytes} bytes, but got {bytes_written}");
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

    fn write(&self, writer: &mut impl std::io::Write) -> anyhow::Result<()> {
        let ptr = std::ptr::from_ref::<Self>(self).cast::<u8>();
        let len = std::mem::size_of::<Self>();
        // SAFETY: We're writing a slice of bytes, and we know that the slice is valid.
        writer.write_all(unsafe { std::slice::from_raw_parts(ptr, len) })?;
        Ok(())
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
    let net = unquantised_net.process(true);
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
        let (white_bucket, black_bucket) = get_bucket_indices(wk, bk);
        let bucket = if side_we_care_about == Colour::White { white_bucket } else { black_bucket };
        let cache_acc = self.accs[bucket].select_mut(side_we_care_about);

        let mut adds = ArrayVec::<_, 32>::new();
        let mut subs = ArrayVec::<_, 32>::new();
        self.board_states[side_we_care_about][bucket].update_iter(board_state, |f, is_add| {
            let (white_idx, black_idx) = feature::indices(wk, bk, f);
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
        source_acc: &Accumulator,
        target_acc: &mut Accumulator,
        nnue_params: &NNUEParams,
    ) {
        let (white_add, black_add) = feature::indices(white_king, black_king, add);
        let (white_sub, black_sub) = feature::indices(white_king, black_king, sub);

        let (white_bucket, black_bucket) = get_bucket_indices(white_king, black_king);

        let white_bucket = nnue_params.select_feature_weights(white_bucket);
        let black_bucket = nnue_params.select_feature_weights(black_bucket);

        if update.white {
            accumulator::vector_add_sub(&source_acc.white, &mut target_acc.white, white_bucket, white_add, white_sub);
        }
        if update.black {
            accumulator::vector_add_sub(&source_acc.black, &mut target_acc.black, black_bucket, black_add, black_sub);
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
        source_acc: &Accumulator,
        target_acc: &mut Accumulator,
        nnue_params: &NNUEParams,
    ) {
        let (white_add, black_add) = feature::indices(white_king, black_king, add);
        let (white_sub1, black_sub1) = feature::indices(white_king, black_king, sub1);
        let (white_sub2, black_sub2) = feature::indices(white_king, black_king, sub2);

        let (white_bucket, black_bucket) = get_bucket_indices(white_king, black_king);

        let white_bucket = nnue_params.select_feature_weights(white_bucket);
        let black_bucket = nnue_params.select_feature_weights(black_bucket);

        if update.white {
            accumulator::vector_add_sub2(
                &source_acc.white,
                &mut target_acc.white,
                white_bucket,
                white_add,
                white_sub1,
                white_sub2,
            );
        }
        if update.black {
            accumulator::vector_add_sub2(
                &source_acc.black,
                &mut target_acc.black,
                black_bucket,
                black_add,
                black_sub1,
                black_sub2,
            );
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
        source_acc: &Accumulator,
        target_acc: &mut Accumulator,
        nnue_params: &NNUEParams,
    ) {
        let (white_add1, black_add1) = feature::indices(white_king, black_king, add1);
        let (white_add2, black_add2) = feature::indices(white_king, black_king, add2);
        let (white_sub1, black_sub1) = feature::indices(white_king, black_king, sub1);
        let (white_sub2, black_sub2) = feature::indices(white_king, black_king, sub2);

        let (white_bucket, black_bucket) = get_bucket_indices(white_king, black_king);

        let white_bucket = nnue_params.select_feature_weights(white_bucket);
        let black_bucket = nnue_params.select_feature_weights(black_bucket);

        if update.white {
            accumulator::vector_add2_sub2(
                &source_acc.white,
                &mut target_acc.white,
                white_bucket,
                white_add1,
                white_add2,
                white_sub1,
                white_sub2,
            );
        }
        if update.black {
            accumulator::vector_add2_sub2(
                &source_acc.black,
                &mut target_acc.black,
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
        std::hint::black_box(state.evaluate(nnue_params, Colour::White, 0));
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
                    let index = feature_indices.0.index() * L1_SIZE + starting_idx;
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
