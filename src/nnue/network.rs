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
        board::Board,
        piece::{Black, Col, Colour, Piece, PieceType, White},
        piecelayout::PieceLayout,
        types::Square,
    },
    image::{self, Image},
    nnue,
    util::{self, Align64, MAX_DEPTH},
};

use super::accumulator::{self, Accumulator};

pub mod feature;
pub mod layers;

/// The embedded neural network parameters.
pub static EMBEDDED_NNUE: &[u8] = include_bytes_aligned!("../../viridithas.nnue.zst");

/// Whether the embedded network can be used verbatim.
pub const EMBEDDED_NNUE_VERBATIM: bool = false;
// Assertion for correctness of the embedded network:
const _: () = assert!(!EMBEDDED_NNUE_VERBATIM || EMBEDDED_NNUE.len() == size_of::<NNUEParams>());
/// Whether to perform the king-plane merging optimisation.
pub const MERGE_KING_PLANES: bool = true;
/// Whether the unquantised network has a feature factoriser.
pub const UNQUANTISED_HAS_FACTORISER: bool = false;
/// The size of the input layer of the network.
pub const INPUT: usize = (12 - MERGE_KING_PLANES as usize) * 64;
/// The amount to scale the output of the network by.
/// This is to allow for the sigmoid activation to differentiate positions with
/// a small difference in evaluation.
pub const SCALE: i32 = 210;
/// The size of one-half of the hidden layer of the network.
pub const L1_SIZE: usize = 2560;
/// The size of the second layer of the network.
pub const L2_SIZE: usize = 16;
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
    // extra bucket for the feature-factoriser.
    l0_weights:    [f32; 12 * 64 * L1_SIZE * (BUCKETS + UNQUANTISED_HAS_FACTORISER as usize)],
    l0_biases:     [f32; L1_SIZE],
    l1_weights:  [[[f32; L2_SIZE]; OUTPUT_BUCKETS]; L1_SIZE],
    l1_biases:    [[f32; L2_SIZE]; OUTPUT_BUCKETS],
    l2x_weights: [[[f32; L3_SIZE]; OUTPUT_BUCKETS]; L2_SIZE],
    l2f_weights:  [[f32; L3_SIZE]; L2_SIZE],
    l2x_biases:   [[f32; L3_SIZE]; OUTPUT_BUCKETS],
    l2f_biases:    [f32; L3_SIZE],
    l3x_weights: [[[f32;   HEADS]; OUTPUT_BUCKETS]; L3_SIZE],
    l3f_weights:  [[f32;   HEADS]; L3_SIZE],
    l3x_biases:   [[f32;   HEADS]; OUTPUT_BUCKETS],
    l3f_biases:    [f32;   HEADS],
}

/// The floating-point parameters of the network, after de-factorisation.
#[rustfmt::skip]
#[repr(C)]
struct MergedNetwork {
    l0_weights:   [f32; 12 * 64 * L1_SIZE * BUCKETS],
    l0_biases:    [f32; L1_SIZE],
    l1_weights: [[[f32; L2_SIZE]; OUTPUT_BUCKETS]; L1_SIZE],
    l1_biases:   [[f32; L2_SIZE]; OUTPUT_BUCKETS],
    l2_weights: [[[f32; L3_SIZE]; OUTPUT_BUCKETS]; L2_SIZE],
    l2_biases:   [[f32; L3_SIZE]; OUTPUT_BUCKETS],
    l3_weights: [[[f32;   HEADS]; OUTPUT_BUCKETS]; L3_SIZE],
    l3_biases:   [[f32;   HEADS]; OUTPUT_BUCKETS],
}

/// A quantised network file, for compressed embedding.
#[rustfmt::skip]
#[repr(C)]
#[derive(PartialEq, Debug)]
struct QuantisedNetwork {
    l0_weights:   [i16; INPUT * L1_SIZE * BUCKETS],
    l0_biases:    [i16; L1_SIZE],
    l1_weights: [[[ i8; L2_SIZE]; OUTPUT_BUCKETS]; L1_SIZE],
    l1_biases:   [[f32; L2_SIZE]; OUTPUT_BUCKETS],
    l2_weights: [[[f32; L3_SIZE]; OUTPUT_BUCKETS]; L2_SIZE],
    l2_biases:   [[f32; L3_SIZE]; OUTPUT_BUCKETS],
    l3_weights: [[[f32;   HEADS]; OUTPUT_BUCKETS]; L3_SIZE],
    l3_biases:   [[f32;   HEADS]; OUTPUT_BUCKETS],
}

/// The parameters of viri's neural network, quantised and permuted
/// for efficient SIMD inference.
#[rustfmt::skip]
#[repr(C)]
pub struct NNUEParams {
    pub l0_weights:   Align64<[i16; INPUT * L1_SIZE * BUCKETS]>,
    pub l0_biases:    Align64<[i16; L1_SIZE]>,
    pub l1_weights:  [Align64<[ i8; L1_SIZE * L2_SIZE]>; OUTPUT_BUCKETS],
    pub l1_bias:     [Align64<[f32; L2_SIZE]>; OUTPUT_BUCKETS],
    pub l2_weights:  [Align64<[f32; L2_SIZE * L3_SIZE]>; OUTPUT_BUCKETS],
    pub l2_bias:     [Align64<[f32; L3_SIZE]>; OUTPUT_BUCKETS],
    pub l3_weights: [[Align64<[f32; L3_SIZE]>; HEADS]; OUTPUT_BUCKETS],
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
    335, 370, 65, 107, 363, 361, 585, 492, 126, 626, 472, 385, 526, 420, 390, 208, 273, 17, 331,
    497, 464, 164, 304, 1012, 317, 114, 168, 782, 20, 134, 326, 342, 86, 96, 893, 102, 346, 337,
    399, 402, 602, 1008, 303, 196, 183, 201, 118, 59, 400, 501, 143, 94, 91, 474, 405, 741, 637,
    179, 748, 593, 642, 162, 514, 875, 641, 251, 226, 1004, 48, 728, 788, 864, 833, 627, 482, 507,
    792, 310, 1009, 109, 656, 616, 750, 270, 172, 14, 470, 70, 360, 755, 371, 365, 898, 984, 429,
    789, 148, 291, 777, 412, 54, 444, 393, 379, 468, 591, 876, 159, 873, 951, 19, 561, 935, 244,
    762, 129, 82, 774, 1007, 271, 174, 407, 917, 460, 185, 508, 355, 510, 100, 276, 721, 81, 892,
    931, 874, 237, 202, 252, 799, 401, 449, 598, 31, 45, 55, 374, 471, 330, 327, 534, 165, 934,
    184, 533, 210, 308, 936, 43, 225, 504, 962, 290, 34, 588, 568, 494, 960, 633, 986, 624, 880,
    428, 817, 724, 256, 994, 191, 136, 214, 28, 253, 27, 77, 130, 315, 116, 386, 489, 74, 726, 137,
    518, 421, 21, 452, 215, 404, 255, 511, 146, 652, 358, 245, 594, 979, 770, 124, 455, 424, 10,
    457, 240, 1011, 587, 24, 894, 836, 425, 295, 478, 1006, 506, 582, 334, 382, 108, 354, 397, 141,
    440, 694, 285, 660, 396, 536, 679, 738, 16, 493, 246, 322, 480, 302, 343, 367, 415, 275, 316,
    901, 857, 537, 923, 57, 392, 284, 138, 439, 495, 505, 423, 286, 18, 649, 451, 678, 606, 199,
    12, 426, 886, 997, 47, 301, 853, 250, 233, 794, 481, 192, 95, 713, 571, 509, 409, 4, 299, 865,
    406, 307, 706, 248, 7, 194, 112, 98, 52, 249, 195, 655, 695, 268, 854, 670, 170, 859, 953, 60,
    42, 829, 669, 977, 312, 198, 658, 119, 375, 498, 754, 242, 313, 562, 601, 778, 863, 834, 384,
    790, 176, 954, 825, 241, 104, 259, 529, 784, 153, 454, 969, 611, 155, 965, 320, 190, 685, 676,
    862, 163, 260, 254, 72, 691, 476, 111, 661, 952, 266, 380, 167, 75, 212, 66, 30, 224, 46, 521,
    615, 64, 736, 709, 899, 145, 76, 166, 737, 682, 711, 502, 36, 749, 347, 180, 309, 948, 53, 56,
    87, 1005, 413, 983, 93, 513, 341, 300, 152, 186, 761, 231, 101, 918, 1, 357, 767, 353, 512,
    442, 41, 62, 305, 961, 805, 902, 11, 684, 946, 171, 1019, 644, 527, 657, 937, 519, 964, 843,
    366, 975, 535, 651, 920, 830, 887, 680, 797, 911, 690, 907, 548, 515, 943, 282, 828, 531, 985,
    791, 819, 577, 563, 826, 747, 1044, 1116, 1129, 1148, 1150, 1080, 1045, 1271, 1200, 1189, 1187,
    1025, 1229, 1052, 1142, 1090, 1192, 1273, 1195, 1057, 1128, 1242, 1111, 1213, 1228, 1249, 1084,
    1029, 1265, 1274, 1263, 1141, 1104, 1160, 1183, 1236, 1145, 1243, 1233, 1206, 1138, 1279, 1086,
    1253, 1169, 1244, 1209, 1162, 1048, 1176, 1095, 1210, 1164, 1035, 1032, 1056, 1131, 1203, 1040,
    1251, 1212, 1277, 1225, 1245, 1036, 1254, 1155, 1038, 1124, 1239, 1196, 1197, 1107, 1252, 1226,
    1077, 1109, 1272, 1177, 1207, 1100, 1103, 1088, 1171, 1066, 1117, 1194, 1149, 1053, 1115, 1114,
    1256, 1037, 1059, 1220, 1034, 1202, 1237, 1153, 1140, 1072, 1027, 1069, 1073, 1156, 1214, 1186,
    1151, 1182, 1261, 1105, 1227, 1247, 1092, 1139, 1113, 1174, 1278, 1146, 1172, 1260, 1241, 1083,
    1055, 1232, 1089, 1074, 1152, 1061, 1190, 1276, 1093, 1270, 1135, 1218, 1235, 1159, 1112, 1070,
    1240, 1041, 1257, 1068, 1163, 1234, 1259, 1264, 1230, 1219, 1184, 1030, 1170, 1119, 1033, 1238,
    1154, 1082, 1175, 1028, 1191, 1143, 1165, 1110, 1266, 1091, 1121, 1208, 1076, 1193, 1085, 1248,
    1179, 1039, 1078, 1125, 1096, 1157, 1099, 1216, 1123, 1168, 1199, 1101, 1067, 1097, 1062, 1161,
    1144, 1108, 1081, 1258, 1063, 1071, 1026, 1051, 1043, 1173, 1126, 1106, 1188, 1047, 1060, 1042,
    1134, 1250, 1102, 1217, 1204, 1058, 1262, 1136, 1079, 1231, 1049, 1211, 1137, 1147, 1075, 1269,
    1222, 1267, 1087, 1178, 1215, 1221, 1127, 1065, 1224, 1185, 1275, 1181, 1198, 1167, 1024, 1158,
    1133, 1031, 1120, 1201, 1166, 1050, 1205, 1054, 1118, 1180, 1246, 1046, 1098, 1223, 1094, 1255,
    1122, 1064, 1268, 1132, 1130, 1003, 989, 766, 603, 671, 987, 584, 613, 913, 855, 619, 798, 570,
    554, 807, 847, 904, 662, 718, 786, 607, 681, 702, 665, 663, 632, 835, 483, 844, 845, 289, 921,
    614, 806, 139, 765, 808, 769, 929, 522, 69, 92, 25, 243, 673, 720, 900, 800, 600, 565, 719, 32,
    763, 158, 867, 646, 178, 576, 956, 177, 348, 813, 90, 896, 699, 29, 462, 564, 206, 523, 37,
    779, 631, 941, 63, 999, 414, 324, 229, 2, 272, 125, 381, 485, 557, 115, 878, 727, 572, 221,
    546, 372, 349, 1023, 742, 612, 772, 265, 488, 643, 712, 674, 700, 940, 759, 717, 503, 49, 142,
    398, 132, 388, 461, 263, 609, 746, 258, 704, 541, 686, 824, 1002, 294, 38, 787, 942, 871, 395,
    949, 945, 823, 620, 810, 524, 1013, 958, 846, 776, 733, 858, 677, 636, 856, 547, 567, 919, 579,
    469, 83, 528, 39, 866, 314, 933, 667, 783, 73, 851, 714, 821, 592, 837, 556, 217, 666, 648,
    434, 211, 387, 50, 909, 517, 860, 764, 232, 992, 653, 629, 516, 287, 1017, 459, 732, 431, 369,
    735, 247, 664, 545, 716, 883, 161, 889, 775, 599, 110, 416, 267, 729, 450, 861, 891, 635, 963,
    144, 683, 840, 630, 1020, 566, 621, 200, 885, 915, 991, 910, 968, 586, 639, 938, 383, 654, 966,
    622, 532, 976, 189, 973, 972, 1016, 745, 378, 207, 623, 604, 106, 897, 912, 884, 235, 926, 222,
    484, 610, 914, 473, 181, 870, 569, 456, 490, 771, 705, 430, 1022, 879, 596, 261, 122, 319, 422,
    277, 281, 731, 67, 228, 205, 993, 325, 293, 274, 542, 831, 220, 905, 793, 818, 499, 540, 432,
    698, 795, 697, 433, 71, 458, 151, 751, 597, 328, 872, 438, 812, 269, 578, 394, 925, 618, 408,
    234, 352, 802, 634, 278, 417, 768, 877, 120, 376, 981, 982, 213, 687, 157, 280, 638, 364, 640,
    525, 608, 971, 448, 820, 453, 628, 832, 816, 849, 675, 650, 288, 410, 175, 996, 869, 688, 740,
    944, 1015, 988, 595, 725, 544, 815, 781, 848, 773, 672, 888, 121, 1021, 550, 543, 703, 715, 26,
    1001, 553, 809, 589, 839, 227, 88, 970, 131, 22, 939, 8, 219, 708, 264, 625, 133, 23, 296, 487,
    560, 135, 491, 990, 445, 345, 903, 204, 441, 980, 127, 927, 906, 443, 351, 1010, 78, 298, 922,
    467, 292, 15, 580, 796, 257, 150, 539, 722, 475, 668, 230, 890, 140, 538, 950, 51, 85, 377,
    692, 44, 479, 1018, 97, 437, 583, 419, 223, 530, 707, 895, 13, 297, 209, 822, 605, 58, 558,
    306, 842, 99, 389, 647, 743, 279, 957, 520, 552, 391, 574, 974, 427, 339, 710, 549, 160, 575,
    581, 340, 447, 318, 998, 760, 154, 356, 411, 203, 193, 852, 84, 967, 804, 693, 757, 689, 617,
    344, 33, 739, 803, 103, 465, 924, 362, 814, 40, 928, 1000, 496, 955, 197, 995, 659, 696, 801,
    463, 182, 947, 359, 5, 780, 128, 559, 477, 323, 500, 701, 785, 645, 466, 311, 436, 868, 744,
    850, 752, 187, 959, 368, 147, 333, 930, 156, 0, 89, 555, 932, 446, 841, 908, 238, 332, 68, 811,
    329, 573, 551, 169, 336, 6, 734, 239, 173, 403, 758, 9, 827, 123, 3, 80, 79, 218, 916, 838,
    321, 756, 61, 35, 113, 117, 881, 373, 435, 723, 262, 1014, 882, 730, 188, 338, 978, 149, 486,
    283, 350, 590, 418, 753, 105, 236, 216,
];

impl UnquantisedNetwork {
    /// Convert a parameter file generated by bullet into a merged parameter set,
    /// for further processing or for resuming training in a more efficient format.
    fn merge(&self) -> Box<MergedNetwork> {
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
            .zip(net.l0_weights.chunks_exact_mut(INPUT * L1_SIZE))
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
        let src_buckets = self.l0_weights.chunks_exact(INPUT * L1_SIZE);
        let tgt_buckets = net.l0_weights.chunks_exact_mut(INPUT * L1_SIZE);
        for (src_bucket, tgt_bucket) in src_buckets.zip(tgt_buckets) {
            repermute_ft_bucket(tgt_bucket, src_bucket);
        }

        // permute the feature transformer biases
        repermute_ft_bias(&mut net.l0_biases, &self.l0_biases);

        // transpose FT weights and biases so that packus transposes it back to the intended order
        if use_simd {
            type PermChunk = [i16; 8];
            // reinterpret as data of size __m128i
            let mut weights: Vec<&mut PermChunk> = net
                .l0_weights
                .chunks_exact_mut(8)
                .map(|a| a.try_into().unwrap())
                .collect();
            let mut biases: Vec<&mut PermChunk> = net
                .l0_biases
                .chunks_exact_mut(8)
                .map(|a| a.try_into().unwrap())
                .collect();
            let num_chunks = size_of::<PermChunk>() / size_of::<i16>();

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
        let ptr = util::from_ref::<Self>(self).cast::<u8>();
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
                util::from_mut(net.as_mut()).cast::<u8>(),
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

    pub fn select_feature_weights(&self, bucket: usize) -> &Align64<[i16; INPUT * L1_SIZE]> {
        // handle mirroring
        let bucket = bucket % BUCKETS;
        let start = bucket * INPUT * L1_SIZE;
        let end = start + INPUT * L1_SIZE;
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
            util::from_ref::<NNUEParams>(network).cast::<u8>(),
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
    let start_pos = Board::default();
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
        let king = (board_state.pieces[PieceType::King] & board_state.colours[colour])
            .first()
            .unwrap();
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
            acc.white = nnue_params.l0_biases.clone();
            acc.black = nnue_params.l0_biases.clone();
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
    pub fn evaluate(&self, nn: &NNUEParams, board: &Board) -> i32 {
        const K: f32 = SCALE as f32;

        let stm = board.turn();
        let out = output_bucket(board);

        let acc = &self.accumulators[self.current_acc];

        debug_assert!(acc.correct[0] && acc.correct[1]);

        let (us, them) = if stm == Colour::White {
            (&acc.white, &acc.black)
        } else {
            (&acc.black, &acc.white)
        };

        let mut l1_outputs = Align64([0.0; L2_SIZE]);
        let mut l2_outputs = Align64([0.0; L3_SIZE]);

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
    let board = Board::default();
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
                    slice.push(self.l0_weights[index]);
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
