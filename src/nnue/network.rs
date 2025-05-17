use std::{
    fmt::{Debug, Display},
    fs::{File, OpenOptions},
    hash::Hasher,
    io::BufReader,
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
    hasher.finish()
}

/// Struct representing the floating-point parameter file emmitted by bullet.
#[rustfmt::skip]
#[repr(C)]
struct UnquantisedNetwork {
    // extra bucket for the feature-factoriser.
    ft_weights:   [f32; 12 * 64 * L1_SIZE * (BUCKETS + UNQUANTISED_HAS_FACTORISER as usize)],
    ft_biases:    [f32; L1_SIZE],
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

static REPERMUTE_INDICES: [u16; L1_SIZE / 2] = {
    let mut indices = [0; L1_SIZE / 2];
    let mut i = 0;
    #[allow(clippy::cast_possible_truncation)]
    while i < L1_SIZE as u16 / 2 {
        indices[i as usize] = i;
        i += 1;
    }
    indices
};

// static REPERMUTE_INDICES: [u16; L1_SIZE / 2] = [
//     846, 382, 423, 1443, 1192, 660, 52, 502, 421, 658, 1293, 329, 98, 874, 247, 759, 820, 29, 146,
//     1261, 802, 139, 1150, 1347, 919, 429, 888, 202, 1308, 1331, 733, 267, 875, 1358, 643, 601,
//     1070, 592, 1399, 522, 1034, 261, 1376, 1429, 588, 998, 1269, 766, 704, 1450, 795, 1203, 787,
//     538, 1463, 1278, 1418, 230, 1488, 1026, 961, 786, 1146, 1339, 6, 245, 378, 342, 668, 215, 612,
//     1449, 633, 1167, 645, 823, 1084, 405, 837, 1014, 447, 446, 1287, 840, 1378, 1241, 602, 1210,
//     445, 635, 1280, 89, 1251, 952, 808, 235, 1182, 1281, 1004, 1094, 859, 561, 492, 520, 140, 556,
//     1396, 203, 694, 1021, 680, 286, 53, 580, 58, 1058, 572, 428, 256, 749, 439, 1452, 403, 614,
//     1456, 755, 1500, 348, 1485, 735, 899, 184, 1130, 834, 213, 1159, 299, 113, 172, 279, 499, 842,
//     1404, 798, 950, 661, 1073, 216, 1468, 333, 1201, 690, 540, 349, 1016, 1257, 1128, 1481, 1098,
//     310, 500, 149, 829, 362, 1416, 1133, 1425, 1292, 913, 455, 1119, 1152, 1031, 1252, 573, 1515,
//     74, 1371, 624, 949, 16, 164, 1323, 314, 1065, 777, 1397, 1103, 1097, 1381, 797, 526, 1527, 427,
//     528, 1338, 606, 774, 537, 148, 881, 489, 927, 981, 15, 471, 865, 991, 782, 1038, 909, 36, 671,
//     902, 461, 404, 1209, 1170, 354, 485, 729, 124, 1491, 134, 161, 258, 524, 596, 1355, 1431, 557,
//     1492, 1373, 12, 659, 19, 992, 1060, 275, 1244, 85, 1041, 449, 1316, 1295, 791, 493, 448, 1349,
//     424, 1136, 266, 545, 530, 1309, 928, 1250, 1039, 359, 980, 1458, 812, 582, 718, 1513, 937, 217,
//     475, 925, 836, 1230, 801, 1105, 1001, 277, 1148, 1228, 527, 1432, 1379, 280, 576, 1127, 1028,
//     1082, 133, 294, 232, 335, 1409, 1363, 964, 377, 724, 88, 740, 1340, 1000, 1320, 804, 1327,
//     1217, 1154, 8, 611, 650, 1286, 809, 1068, 435, 1247, 326, 131, 1076, 860, 1291, 794, 1054,
//     1077, 181, 69, 935, 1275, 681, 683, 241, 1419, 983, 1503, 944, 967, 1352, 187, 1424, 197, 1086,
//     1526, 1326, 575, 60, 817, 551, 1045, 1024, 1114, 376, 564, 462, 1204, 410, 1056, 26, 1366,
//     1059, 102, 9, 1095, 64, 1189, 145, 369, 123, 1157, 1211, 141, 472, 1495, 1444, 788, 604, 395,
//     667, 298, 518, 137, 838, 827, 413, 250, 1267, 905, 706, 685, 533, 464, 370, 605, 1300, 882,
//     1303, 34, 1018, 178, 1108, 568, 384, 1325, 1036, 234, 819, 1236, 1032, 398, 934, 565, 425, 651,
//     770, 908, 4, 1382, 1304, 986, 463, 1161, 636, 176, 283, 355, 709, 328, 674, 191, 110, 39, 218,
//     1135, 994, 513, 895, 756, 758, 768, 302, 1142, 227, 799, 1164, 1200, 679, 1025, 1319, 689, 77,
//     1002, 323, 209, 1390, 1408, 243, 1384, 873, 1477, 320, 591, 385, 201, 361, 153, 1147, 1179,
//     1013, 616, 1274, 1246, 1497, 1040, 852, 494, 814, 989, 1400, 406, 240, 57, 626, 816, 1205, 649,
//     972, 1354, 702, 1490, 620, 228, 1361, 408, 999, 254, 1420, 1125, 1516, 132, 1334, 316, 177,
//     562, 505, 698, 1322, 996, 206, 547, 687, 127, 1517, 1243, 974, 1061, 1172, 577, 412, 62, 210,
//     174, 1460, 1155, 1391, 1344, 1180, 942, 1264, 1207, 28, 1175, 1120, 760, 815, 332, 751, 318,
//     1297, 627, 75, 481, 763, 1138, 1423, 264, 1479, 1483, 894, 1010, 76, 631, 1235, 304, 1227, 748,
//     669, 1216, 451, 970, 121, 503, 1089, 2, 721, 1143, 848, 1294, 897, 644, 1283, 757, 911, 1006,
//     850, 1134, 1351, 1052, 550, 705, 1072, 107, 1407, 1474, 142, 990, 971, 233, 554, 1268, 1156,
//     391, 1359, 61, 330, 339, 1284, 1151, 1141, 388, 219, 129, 458, 985, 960, 514, 532, 1126, 517,
//     693, 862, 47, 274, 1233, 762, 779, 400, 399, 229, 1440, 821, 670, 37, 337, 1426, 155, 1208,
//     450, 1057, 112, 324, 1113, 1470, 912, 599, 296, 265, 1523, 750, 188, 1528, 130, 1437, 1023,
//     1046, 1083, 407, 238, 496, 457, 87, 775, 1178, 1030, 959, 932, 246, 271, 160, 1467, 66, 357,
//     1272, 723, 1401, 1144, 1532, 438, 692, 613, 910, 1345, 409, 1315, 1457, 1169, 1176, 136, 566,
//     730, 1137, 1174, 1336, 122, 531, 741, 306, 784, 242, 473, 880, 1499, 27, 863, 1296, 589, 1455,
//     169, 1393, 738, 1049, 515, 419, 1009, 42, 59, 1277, 41, 387, 1330, 392, 56, 876, 31, 477, 549,
//     312, 731, 114, 516, 79, 1478, 1173, 761, 126, 10, 833, 1184, 1035, 975, 1117, 1109, 632, 207,
//     1224, 1447, 946, 374, 1289, 293, 553, 67, 1153, 1190, 1402, 1163, 82, 936, 1464, 914, 1165,
//     135, 868, 338, 199, 487, 1414, 48, 567, 555, 805, 544, 542, 95, 1529, 1329, 106, 1403, 790, 30,
//     1484, 519, 156, 1357, 885, 168, 1177, 1353, 1453, 301, 525, 300, 585, 839, 535, 597, 386, 711,
//     444, 1476, 1288, 1451, 13, 1332, 1015, 1507, 1231, 159, 534, 1388, 742, 93, 1075, 1506, 1074,
//     977, 810, 647, 781, 45, 598, 1434, 452, 1321, 7, 331, 822, 1311, 954, 1441, 25, 1051, 617,
//     1494, 1048, 158, 50, 252, 1271, 807, 1202, 903, 1166, 877, 1263, 120, 1310, 282, 1249, 890,
//     866, 1370, 1265, 719, 581, 103, 17, 856, 46, 871, 212, 14, 1367, 225, 830, 506, 486, 1183, 105,
//     906, 955, 570, 190, 1395, 1254, 657, 360, 699, 68, 654, 593, 1389, 976, 78, 796, 454, 701, 5,
//     163, 1225, 1465, 484, 22, 1160, 1, 1422, 1374, 1282, 1186, 529, 1099, 673, 610, 297, 867, 1480,
//     1394, 1328, 1380, 747, 965, 1100, 678, 49, 1405, 811, 886, 1438, 478, 186, 195, 474, 154, 366,
//     1033, 327, 586, 1435, 1255, 712, 655, 979, 1206, 1122, 1106, 858, 1232, 491, 80, 104, 1199,
//     1219, 675, 857, 715, 0, 418, 1011, 263, 1008, 1003, 608, 460, 1298, 244, 1369, 381, 825, 1486,
//     891, 167, 916, 841, 828, 51, 951, 402, 956, 476, 417, 1368, 708, 1081, 697, 319, 615, 1279,
//     543, 1121, 1245, 336, 1445, 237, 390, 1430, 383, 953, 663, 1107, 1168, 152, 1043, 194, 24, 393,
//     270, 771, 1324, 926, 432, 767, 1188, 920, 982, 1459, 1337, 308, 918, 1069, 1521, 508, 1412,
//     743, 347, 1017, 656, 629, 1410, 358, 1498, 941, 1064, 700, 1273, 732, 222, 1131, 483, 179, 165,
//     1496, 933, 211, 1118, 609, 921, 900, 33, 273, 917, 40, 166, 628, 465, 546, 44, 931, 623, 291,
//     844, 806, 116, 346, 1348, 861, 143, 63, 173, 548, 1505, 54, 507, 1472, 1129, 344, 305, 249,
//     1139, 434, 480, 115, 621, 285, 1524, 634, 284, 334, 325, 1096, 622, 1050, 1226, 901, 1493, 584,
//     18, 1398, 948, 426, 646, 1102, 204, 1197, 488, 1427, 1124, 929, 1510, 21, 997, 1482, 365, 1044,
//     803, 1071, 1290, 111, 189, 315, 1191, 1365, 303, 224, 714, 765, 1181, 55, 1428, 32, 1020, 23,
//     987, 340, 578, 433, 1535, 255, 872, 780, 1080, 196, 666, 722, 109, 170, 379, 1385, 1436, 257,
//     292, 892, 397, 1132, 1185, 1270, 125, 1198, 193, 924, 221, 1489, 251, 1212, 1005, 1406, 579,
//     467, 295, 703, 1218, 607, 288, 558, 728, 870, 726, 1508, 259, 287, 1112, 896, 1087, 208, 440,
//     993, 1362, 938, 380, 1518, 171, 984, 638, 1318, 552, 642, 1364, 978, 466, 1471, 1442, 752,
//     1461, 364, 1317, 443, 414, 1259, 778, 879, 587, 1262, 720, 1229, 1306, 1335, 214, 1222, 459,
//     1377, 1116, 157, 907, 764, 1469, 1387, 350, 1473, 1027, 200, 1301, 290, 737, 411, 753, 618,
//     945, 739, 497, 101, 1375, 1509, 915, 800, 495, 272, 1520, 1504, 672, 1276, 1417, 1088, 70,
//     1123, 1454, 278, 727, 469, 1299, 1066, 847, 855, 1221, 1439, 940, 1053, 1063, 1078, 1392, 415,
//     1356, 574, 1248, 943, 236, 923, 521, 1111, 1085, 1101, 183, 1092, 430, 835, 1220, 968, 769,
//     501, 988, 1514, 707, 468, 1511, 691, 198, 11, 1531, 436, 1415, 783, 922, 995, 3, 94, 962, 824,
//     343, 182, 684, 99, 845, 745, 773, 185, 1285, 1193, 1333, 1079, 97, 479, 341, 220, 322, 1240,
//     1341, 253, 498, 373, 696, 144, 262, 939, 317, 276, 313, 1213, 422, 792, 1019, 81, 453, 849,
//     652, 375, 640, 307, 90, 1253, 223, 269, 1260, 1383, 65, 118, 641, 239, 91, 345, 619, 571, 504,
//     541, 1215, 1037, 594, 1501, 372, 793, 1162, 1110, 175, 686, 1525, 1433, 431, 442, 353, 930,
//     1234, 889, 420, 231, 559, 590, 289, 394, 1093, 192, 1223, 887, 92, 1312, 884, 441, 1307, 963,
//     832, 772, 96, 509, 973, 639, 1238, 898, 512, 583, 482, 831, 653, 595, 523, 511, 1042, 352, 637,
//     1421, 1195, 665, 1029, 1194, 1104, 226, 717, 1012, 878, 1022, 560, 1145, 490, 818, 351, 1258,
//     1475, 1448, 1239, 1522, 35, 248, 117, 677, 1067, 536, 1413, 1237, 367, 716, 260, 1196, 1242,
//     83, 789, 746, 1411, 1115, 128, 119, 1372, 180, 736, 1305, 563, 695, 321, 630, 456, 648, 72,
//     1512, 966, 470, 1055, 1158, 1487, 100, 947, 437, 826, 401, 869, 1313, 147, 1360, 744, 86, 1214,
//     600, 1187, 73, 1350, 1343, 43, 662, 843, 363, 958, 71, 281, 1266, 1342, 1256, 1091, 151, 969,
//     725, 510, 368, 1534, 205, 1446, 138, 754, 682, 1386, 603, 356, 854, 1171, 389, 864, 664, 1062,
//     853, 734, 710, 539, 1302, 1090, 108, 162, 676, 150, 1047, 813, 1346, 1007, 893, 1314, 883, 311,
//     309, 569, 851, 713, 688, 1149, 625, 416, 1140, 371, 785, 957, 776, 84, 1466, 1462, 396, 268,
//     904, 1502, 38, 1519, 1530, 20, 1533,
// ];

impl UnquantisedNetwork {
    /// Convert a parameter file generated by bullet into a quantised parameter set,
    /// for embedding into viri as a zstd-compressed archive. We do one processing
    /// step other than quantisation, namely merging the feature factoriser with the
    /// main king buckets.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::assertions_on_constants,
        clippy::too_many_lines,
        clippy::cognitive_complexity
    )]
    fn quantise(&self) -> Box<QuantisedNetwork> {
        const QA_BOUND: f32 = 1.98 * QA as f32;
        const QB_BOUND: f32 = 1.98 * QB as f32;

        let mut net = QuantisedNetwork::zeroed();
        // quantise the feature transformer weights, and merge the feature factoriser in.
        let mut buckets = self.ft_weights.chunks_exact(12 * 64 * L1_SIZE);
        let factoriser;
        let alternate_buffer;
        if UNQUANTISED_HAS_FACTORISER {
            factoriser = buckets.next().unwrap();
        } else {
            alternate_buffer = vec![0.0; 12 * 64 * L1_SIZE];
            factoriser = &alternate_buffer;
        }
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
                    let fac_src = &factoriser[i * L1_SIZE..i * L1_SIZE + L1_SIZE];
                    let tgt = &mut tgt_bucket[j * L1_SIZE..j * L1_SIZE + L1_SIZE];
                    for ((src, fac_src), tgt) in src.iter().zip(fac_src).zip(tgt) {
                        // extra clamp in case bucket + factoriser goes out of the clipping bounds
                        let scaled = f32::clamp(*src + *fac_src, -1.98, 1.98) * f32::from(QA);
                        *tgt = scaled.round() as i16;
                    }
                    things_written += 1;
                }
            }
            assert_eq!(INPUT, things_written);
        }

        // quantise the feature transformer biases
        let mut max_seen = 0.0f32;
        for (src, tgt) in self.ft_biases.iter().zip(net.ft_biases.iter_mut()) {
            let scaled = *src * f32::from(QA);
            max_seen = max_seen.max(scaled.abs());
            if scaled.abs() > QA_BOUND {
                eprintln!("feature transformer bias {scaled} is too large (max = {QA_BOUND})");
            }
            *tgt = scaled.clamp(-QA_BOUND, QA_BOUND).round() as i16;
        }
        let needed = QA_BOUND / max_seen * f32::from(QA);
        if max_seen > QA_BOUND {
            eprintln!("Worst offender was {max_seen}");
            eprintln!(
                "In order to fit these weights, QA would need to be {needed} instead of {QA}"
            );
        } else {
            eprintln!("Quantisation of FT weights was successful");
            eprintln!("QA could be increased to {needed} instead of {QA}");
        }

        // quantise (or not) later layers
        let mut max_seen = 0.0f32;
        for i in 0..L1_SIZE {
            for bucket in 0..OUTPUT_BUCKETS {
                for j in 0..L2_SIZE {
                    let v =
                        (self.l1x_weights[i][bucket][j] + self.l1f_weights[i][j]) * f32::from(QB);
                    max_seen = max_seen.max(v.abs());
                    if v.abs() > QB_BOUND {
                        eprintln!("L1 weight {v} is too large (max = {QB_BOUND})");
                    }
                    let v = v.clamp(-QB_BOUND, QB_BOUND).round() as i8;
                    net.l1_weights[i][bucket][j] = v;
                }
            }
        }
        let needed = QB_BOUND / max_seen * f32::from(QB);
        if max_seen > QB_BOUND {
            eprintln!("Worst offender was {max_seen}");
            eprintln!(
                "In order to fit these weights, QB would need to be {needed} instead of {QB}"
            );
        } else {
            eprintln!("Quantisation of L1 weights was successful");
            eprintln!("QB could be increased to {needed} instead of {QB}");
        }

        // transfer the L1 biases
        net.l1_biases = self.l1x_biases;
        for bucket in 0..OUTPUT_BUCKETS {
            for i in 0..L2_SIZE {
                net.l1_biases[bucket][i] += self.l1f_biases[i];
            }
        }
        net.l2_weights = self.l2x_weights;
        for bucket in 0..OUTPUT_BUCKETS {
            for i in 0..L3_SIZE {
                for j in 0..L2_SIZE {
                    net.l2_weights[j][bucket][i] += self.l2f_weights[j][i];
                }
            }
        }
        net.l2_biases = self.l2x_biases;
        for bucket in 0..OUTPUT_BUCKETS {
            for i in 0..L3_SIZE {
                net.l2_biases[bucket][i] += self.l2f_biases[i];
            }
        }
        net.l3_weights = self.l3x_weights;
        for bucket in 0..OUTPUT_BUCKETS {
            for i in 0..L3_SIZE {
                net.l3_weights[i][bucket] += self.l3f_weights[i];
            }
        }
        net.l3_biases = self.l3x_biases;
        for bucket in 0..OUTPUT_BUCKETS {
            net.l3_biases[bucket] += self.l3f_biases[0];
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
        sorted[tgt_index] = l1_weights[src_index as usize];
    }
    for (tgt_index, src_index) in REPERMUTE_INDICES.iter().copied().enumerate() {
        sorted[tgt_index + L1_SIZE / 2] = l1_weights[src_index as usize + L1_SIZE / 2];
    }
}

fn repermute_ft_bias(feature_bias: &mut [i16; L1_SIZE], unsorted: &[i16]) {
    for (tgt_index, src_index) in REPERMUTE_INDICES.iter().copied().enumerate() {
        feature_bias[tgt_index] = unsorted[src_index as usize];
    }
    for (tgt_index, src_index) in REPERMUTE_INDICES.iter().copied().enumerate() {
        feature_bias[tgt_index + L1_SIZE / 2] = unsorted[src_index as usize + L1_SIZE / 2];
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
            tgt_bucket[feature + tgt_index] = unsorted[feature + src_index as usize];
        }
        for (tgt_index, src_index) in REPERMUTE_INDICES.iter().copied().enumerate() {
            let tgt_index = tgt_index + L1_SIZE / 2;
            let src_index = src_index as usize + L1_SIZE / 2;
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
    let mut reader = BufReader::new(File::open(input)?);
    let mut writer = File::create(output)?;
    let unquantised_net = UnquantisedNetwork::read(&mut reader)?;
    let net = unquantised_net.quantise();
    net.write(&mut writer)?;
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
