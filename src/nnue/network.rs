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
pub const SCALE: i32 = 240;
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
    l0_weights:   [i16; INPUT * L1_SIZE * BUCKETS],
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
    pub l0_weights:   Align64<[i16; INPUT * L1_SIZE * BUCKETS]>,
    pub l0_biases:    Align64<[i16; L1_SIZE]>,
    pub l1_weights:  [Align64<[ i8; L1_SIZE * L2_SIZE]>; OUTPUT_BUCKETS],
    pub l1_bias:     [Align64<[f32; L2_SIZE]>; OUTPUT_BUCKETS],
    pub l2_weights:  [Align64<[f32; L2_SIZE * L3_SIZE * 2]>; OUTPUT_BUCKETS],
    pub l2_bias:     [Align64<[f32; L3_SIZE * 2]>; OUTPUT_BUCKETS],
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
    447, 865, 97, 758, 746, 250, 1014, 921, 264, 400, 292, 235, 229, 483, 46, 330, 159, 491, 216,
    730, 284, 21, 191, 277, 474, 315, 812, 28, 472, 665, 520, 302, 415, 74, 614, 188, 106, 301,
    343, 646, 1002, 723, 717, 399, 747, 261, 332, 189, 559, 143, 406, 517, 421, 365, 150, 461, 123,
    826, 445, 181, 790, 169, 419, 451, 390, 68, 401, 683, 407, 366, 482, 869, 629, 815, 554, 1016,
    20, 742, 888, 430, 233, 252, 489, 508, 501, 245, 386, 458, 200, 977, 258, 329, 422, 391, 503,
    166, 91, 18, 325, 824, 385, 729, 56, 581, 714, 913, 319, 822, 600, 610, 349, 269, 139, 537,
    725, 745, 591, 992, 749, 814, 523, 369, 418, 172, 835, 1001, 598, 526, 988, 938, 211, 811, 395,
    674, 704, 423, 115, 129, 911, 469, 907, 121, 413, 556, 266, 932, 1010, 796, 854, 604, 639, 701,
    616, 856, 872, 900, 15, 667, 971, 947, 788, 813, 647, 630, 806, 644, 361, 551, 657, 738, 899,
    928, 676, 648, 339, 718, 525, 643, 575, 794, 963, 558, 975, 603, 595, 841, 547, 776, 750, 733,
    456, 583, 3, 585, 576, 929, 634, 791, 803, 358, 295, 678, 26, 427, 737, 560, 86, 571, 799, 976,
    2, 543, 493, 764, 635, 175, 183, 989, 155, 981, 213, 896, 640, 626, 432, 193, 204, 962, 471,
    459, 708, 162, 589, 135, 840, 681, 825, 532, 442, 1022, 1019, 251, 29, 197, 679, 19, 504, 845,
    524, 104, 128, 24, 540, 530, 360, 142, 987, 167, 533, 842, 649, 786, 313, 225, 242, 23, 337,
    137, 510, 990, 254, 843, 653, 974, 710, 882, 821, 82, 160, 416, 964, 889, 740, 996, 633, 564,
    171, 760, 280, 972, 779, 965, 378, 107, 164, 12, 218, 380, 656, 661, 351, 910, 304, 237, 659,
    548, 208, 490, 706, 592, 290, 1008, 147, 71, 982, 353, 373, 403, 234, 240, 455, 870, 954, 688,
    25, 53, 891, 62, 248, 754, 903, 476, 565, 371, 1, 417, 628, 463, 392, 618, 377, 364, 995, 936,
    697, 875, 572, 132, 1009, 946, 66, 728, 904, 927, 57, 846, 810, 85, 654, 623, 836, 184, 850,
    271, 529, 808, 6, 230, 957, 895, 682, 1007, 263, 195, 202, 154, 187, 93, 1011, 405, 439, 767,
    612, 783, 298, 736, 837, 699, 457, 70, 651, 772, 584, 108, 4, 580, 475, 884, 998, 199, 853,
    420, 1021, 327, 34, 752, 141, 289, 221, 937, 178, 209, 705, 146, 98, 538, 338, 467, 933, 450,
    238, 494, 336, 518, 262, 805, 586, 539, 309, 17, 650, 700, 80, 182, 770, 149, 941, 412, 156,
    334, 217, 124, 622, 14, 436, 498, 352, 515, 310, 546, 372, 685, 223, 362, 281, 348, 993, 318,
    219, 122, 33, 721, 804, 441, 408, 978, 314, 793, 879, 246, 215, 1018, 588, 931, 296, 868, 440,
    76, 148, 205, 126, 569, 766, 470, 393, 496, 294, 429, 92, 38, 32, 153, 552, 370, 660, 724, 48,
    521, 99, 67, 549, 689, 321, 680, 79, 596, 49, 545, 1003, 594, 282, 486, 844, 830, 997, 382,
    487, 77, 367, 743, 878, 286, 627, 460, 113, 574, 983, 763, 507, 203, 675, 40, 275, 797, 930,
    664, 288, 801, 871, 267, 945, 852, 720, 206, 84, 404, 276, 265, 293, 312, 145, 468, 55, 374,
    497, 95, 944, 464, 303, 257, 561, 88, 50, 331, 677, 324, 270, 299, 356, 283, 426, 344, 506,
    513, 305, 119, 60, 227, 37, 212, 311, 116, 118, 163, 381, 222, 125, 765, 214, 151, 170, 368,
    613, 0, 848, 226, 939, 1005, 357, 140, 817, 89, 453, 27, 394, 138, 893, 13, 452, 63, 190, 30,
    65, 328, 694, 562, 72, 924, 268, 912, 307, 986, 397, 961, 839, 102, 176, 726, 120, 224, 819,
    833, 480, 527, 179, 354, 180, 955, 519, 668, 573, 168, 273, 61, 134, 297, 111, 58, 915, 359,
    534, 858, 902, 96, 444, 424, 317, 587, 438, 45, 9, 956, 514, 75, 78, 158, 775, 47, 847, 784,
    908, 857, 346, 484, 152, 228, 363, 333, 300, 917, 509, 607, 157, 550, 702, 777, 192, 42, 695,
    866, 198, 505, 402, 90, 109, 389, 434, 449, 165, 795, 691, 81, 823, 435, 849, 567, 919, 744,
    433, 798, 133, 973, 599, 979, 874, 249, 851, 922, 210, 696, 185, 611, 827, 379, 860, 41, 949,
    920, 774, 396, 621, 980, 916, 220, 769, 11, 739, 173, 968, 112, 256, 709, 637, 781, 831, 196,
    820, 446, 308, 1012, 69, 748, 22, 641, 388, 828, 759, 105, 535, 756, 481, 617, 966, 886, 555,
    557, 807, 832, 994, 873, 761, 859, 8, 544, 887, 570, 320, 541, 880, 528, 755, 335, 578, 1004,
    662, 566, 863, 495, 958, 609, 253, 925, 341, 999, 522, 809, 239, 876, 326, 7, 601, 652, 201,
    10, 636, 454, 669, 881, 376, 731, 103, 477, 279, 909, 593, 926, 243, 291, 83, 443, 692, 1020,
    322, 751, 892, 959, 960, 991, 323, 867, 278, 52, 347, 577, 398, 834, 780, 512, 934, 818, 488,
    663, 864, 542, 177, 466, 690, 625, 632, 898, 619, 712, 1017, 161, 838, 64, 703, 800, 670, 768,
    970, 861, 789, 51, 73, 136, 35, 207, 130, 1000, 890, 732, 631, 722, 605, 101, 894, 620, 582,
    862, 1013, 131, 590, 855, 174, 658, 984, 384, 431, 411, 969, 285, 194, 260, 479, 711, 918, 771,
    448, 707, 39, 606, 1006, 345, 638, 985, 698, 655, 687, 897, 753, 952, 757, 608, 1241, 1126,
    1102, 1110, 1176, 1157, 1137, 1111, 1262, 1129, 1239, 1074, 1276, 1100, 1057, 1059, 1190, 1114,
    1234, 1193, 1256, 1084, 1166, 1152, 1135, 1037, 1219, 1164, 1096, 1142, 1132, 1108, 1248, 1045,
    1072, 1255, 1034, 1156, 1211, 1195, 1243, 1267, 1106, 1183, 1090, 1184, 1206, 1065, 1272, 1258,
    1215, 1089, 1150, 1232, 1127, 1047, 1062, 1208, 1131, 1197, 1200, 1052, 1105, 1080, 1257, 1260,
    1038, 1079, 1073, 1264, 1174, 1259, 1054, 1196, 1160, 1172, 1244, 1026, 1220, 1213, 1115, 1134,
    1228, 1194, 1147, 1139, 1230, 1130, 1266, 1148, 1188, 1123, 1265, 1078, 1025, 1203, 1233, 1271,
    1060, 1140, 1223, 1124, 1028, 1198, 1263, 1254, 1027, 1154, 1128, 1277, 1168, 1247, 1179, 1042,
    1222, 1159, 1212, 1180, 1053, 1116, 1071, 1201, 1169, 1112, 1218, 1119, 1024, 1046, 1120, 1240,
    1068, 1231, 1224, 1221, 1186, 1153, 1242, 1167, 1205, 1088, 1253, 1036, 1067, 1158, 1191, 1041,
    1050, 1058, 1177, 1237, 1151, 1252, 1178, 1225, 1121, 1204, 1144, 1146, 1246, 1171, 1162, 1238,
    1033, 1092, 1040, 1117, 1093, 1274, 1095, 1181, 1187, 1155, 1145, 1055, 1279, 1056, 1136, 1099,
    1269, 1214, 1268, 1098, 1143, 1249, 1210, 1192, 1029, 1250, 1133, 1227, 1077, 1141, 1199, 1235,
    1261, 1175, 1270, 1251, 1165, 1103, 1138, 1173, 1118, 1049, 1109, 1094, 1189, 1039, 1278, 1069,
    1070, 1229, 1101, 1113, 1032, 1245, 1043, 1226, 1275, 1085, 1087, 1083, 1030, 1149, 1048, 1107,
    1082, 1182, 1081, 1061, 1091, 1075, 1217, 1122, 1236, 1035, 1051, 1064, 1097, 1044, 1031, 1207,
    1163, 1125, 1185, 1216, 1063, 1104, 1170, 1076, 1202, 1273, 1209, 1066, 1161, 1086, 782, 1023,
    536, 943, 901, 693, 735, 387, 713, 579, 719, 914, 940, 516, 923, 715, 232, 948, 953, 645, 673,
    437, 672, 792, 967, 951, 905, 597, 671, 642, 462, 802, 716, 734, 727, 87, 342, 935, 94, 883,
    684, 350, 499, 885, 428, 741, 531, 1015, 259, 54, 231, 816, 762, 877, 778, 563, 553, 255, 110,
    829, 5, 950, 785, 425, 787, 602, 355, 906, 247, 666, 316, 375, 615, 410, 241, 287, 244, 127,
    236, 31, 36, 485, 502, 144, 100, 511, 306, 117, 43, 16, 473, 478, 414, 624, 274, 272, 568, 340,
    186, 492, 465, 773, 383, 114, 500, 409, 59, 686, 44, 942,
];

impl UnquantisedNetwork {
    /// Convert a parameter file generated by bullet into a merged parameter set,
    /// for further processing or for resuming training in a more efficient format.
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
