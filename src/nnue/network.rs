// SPDX-License-Identifier: AGPL-3.0-only

use std::{
    fmt::{Debug, Display},
    fs::{File, OpenOptions},
    hash::Hasher,
    io::{BufReader, BufWriter, Write},
    mem::size_of,
    ops::Deref,
    path::Path,
    sync::{LazyLock, Mutex, OnceLock},
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
    util::{Align, ArrayChunks, MAX_DEPTH, MEGABYTE},
};

use super::accumulator::{self, Accumulator};

pub mod feature;
pub mod layers;
pub mod pawn_updates;
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
/// The number of features for pawn-pawn relations.
pub const PAWN_TUPLE_FEATURES: usize = 96 * 95 / 2;
/// The number of features for threats.
pub const THREAT_FEATURES: usize = 59808;
/// The number of features present in the non-psqt part of the input.
pub const AUX_FEATURES: usize = THREAT_FEATURES + PAWN_TUPLE_FEATURES;

pub const L0_OUT: usize = 1024 * 4;
pub const ACC_LEN: usize = L0_OUT * 2 / 2;
pub const L1_IN: usize = ACC_LEN * 2 / 2;
pub const L1_OUT: usize = 32 * 4;
pub const L2_IN: usize = 32 * 4;
pub const L2_OUT: usize = 32 * 4;
pub const L3_IN: usize = 32 * 4;

/// The amount to scale the output of the network by.
/// This is to allow for the sigmoid activation to differentiate positions with
/// a small difference in evaluation.
pub const SCALE: i32 = 240;
/// The number of output heads.
pub const HEADS: usize = 1;
/// The quantisation factor for the feature transformer weights.
const QA: i16 = 255;
/// The quantisation factor for the L1 weights.
const QB: i16 = 64;
/// Chunking constant for l1
pub const L1_CHUNK_PER_32: usize = size_of::<i32>() / size_of::<i8>();

#[cfg(target_feature = "avx512f")]
pub const PACK_REGS: usize = 8;
#[cfg(target_feature = "neon")]
pub const PACK_REGS: usize = 2;
#[cfg(not(any(target_feature = "avx512f", target_feature = "neon")))]
pub const PACK_REGS: usize = 4;

const _: () = assert!(PACK_REGS * 8 == nnue::simd::I16_CHUNK * 2);

pub const PACK_ORDER: [usize; PACK_REGS] = {
    let mut order = [0; PACK_REGS];
    let half = PACK_REGS / 2;
    let mut i = 0;
    while i < half {
        order[i] = 2 * i;
        order[half + i] = 2 * i + 1;
        i += 1;
    }
    order
};

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
    for index in PACK_ORDER {
        hasher.write_usize(index);
    }
    hasher.finish()
}

/// Struct representing the floating-point parameter file emitted by bullet.
#[rustfmt::skip]
#[repr(C)]
struct UnquantisedNetwork {
    l0_aux:        [f32; AUX_FEATURES * L0_OUT],
    // extra bucket for the feature-factoriser.
    l0_weights:    [f32; 12 * 64 * L0_OUT * (BUCKETS + UNQUANTISED_HAS_FACTORISER as usize)],
    l0_biases:     [f32; L0_OUT],
    l1_weights:  [[[f32; L1_OUT]; OUTPUT_BUCKETS]; L1_IN],
    l1_biases:    [[f32; L1_OUT]; OUTPUT_BUCKETS],
    l2x_weights: [[[f32; L2_OUT * 2]; OUTPUT_BUCKETS]; L2_IN],
    l2f_weights:  [[f32; L2_OUT * 2]; L2_IN],
    l2x_biases:   [[f32; L2_OUT * 2]; OUTPUT_BUCKETS],
    l2f_biases:    [f32; L2_OUT * 2],
    l3x_weights: [[[f32; HEADS]; OUTPUT_BUCKETS]; L3_IN],
    l3f_weights:  [[f32; HEADS]; L3_IN],
    l3x_biases:   [[f32; HEADS]; OUTPUT_BUCKETS],
    l3f_biases:    [f32; HEADS],
}

/// The floating-point parameters of the network, after de-factorisation.
#[rustfmt::skip]
#[repr(C)]
struct MergedNetwork {
    l0_aux:       [f32; AUX_FEATURES * L0_OUT],
    l0_weights:   [f32; 12 * 64 * L0_OUT * BUCKETS],
    l0_biases:    [f32; L0_OUT],
    l1_weights: [[[f32; L1_OUT]; OUTPUT_BUCKETS]; L1_IN],
    l1_biases:   [[f32; L1_OUT]; OUTPUT_BUCKETS],
    l2_weights: [[[f32; L2_OUT * 2]; OUTPUT_BUCKETS]; L2_IN],
    l2_biases:   [[f32; L2_OUT * 2]; OUTPUT_BUCKETS],
    l3_weights: [[[f32; HEADS]; OUTPUT_BUCKETS]; L3_IN],
    l3_biases:   [[f32; HEADS]; OUTPUT_BUCKETS],
}

/// A quantised network file, for compressed embedding.
#[rustfmt::skip]
#[repr(C)]
#[derive(PartialEq, Debug)]
struct QuantisedNetwork {
    l0_aux:       [ i8; AUX_FEATURES * L0_OUT],
    l0_weights:   [i16; PSQT_FEATURES * L0_OUT * BUCKETS],
    l0_biases:    [i16; L0_OUT],
    l1_weights: [[[ i8; L1_OUT]; OUTPUT_BUCKETS]; L1_IN],
    l1_biases:   [[f32; L1_OUT]; OUTPUT_BUCKETS],
    l2_weights: [[[f32; L2_OUT * 2]; OUTPUT_BUCKETS]; L2_IN],
    l2_biases:   [[f32; L2_OUT * 2]; OUTPUT_BUCKETS],
    l3_weights: [[[f32; HEADS]; OUTPUT_BUCKETS]; L3_IN],
    l3_biases:   [[f32; HEADS]; OUTPUT_BUCKETS],
}

/// The parameters of viri's neural network, quantised and permuted
/// for efficient SIMD inference.
#[rustfmt::skip]
#[repr(C)]
pub struct NNUEParams {
    pub l0_aux:       Align<[ i8; AUX_FEATURES * L0_OUT]>,
    pub l0_weights:   Align<[i16; PSQT_FEATURES * L0_OUT * BUCKETS]>,
    pub l0_biases:    Align<[i16; L0_OUT]>,
    pub l1_weights:  [Align<[ i8; L1_IN * L1_OUT]>; OUTPUT_BUCKETS],
    pub l1_bias:     [Align<[f32; L1_OUT]>; OUTPUT_BUCKETS],
    pub l2_weights:  [Align<[f32; L2_IN * (L2_OUT * 2)]>; OUTPUT_BUCKETS],
    pub l2_bias:     [Align<[f32; L2_OUT * 2]>; OUTPUT_BUCKETS],
    pub l3_weights: [[Align<[f32; L3_IN]>; HEADS]; OUTPUT_BUCKETS],
    pub l3_bias:           [[f32; HEADS]; OUTPUT_BUCKETS],
}

// const REPERMUTE_INDICES: [usize; L1_IN / 2] = {
//     let mut indices = [0; L1_IN / 2];
//     let mut i = 0;
//     while i < L1_IN / 2 {
//         indices[i] = i;
//         i += 1;
//     }
//     indices
// };

const REPERMUTE_INDICES: [usize; L1_IN / 2] = [
    1817, 1376, 1911, 1799, 1954, 230, 1041, 1530, 1975, 508, 1153, 900, 1995, 580, 1181, 1472,
    1621, 739, 982, 1668, 1334, 1257, 1595, 1807, 1374, 367, 112, 35, 1612, 497, 259, 307, 913,
    273, 2042, 1116, 1211, 1945, 1555, 1925, 813, 955, 1953, 1348, 1610, 1714, 125, 482, 1814,
    1543, 327, 404, 358, 310, 467, 131, 638, 549, 676, 1968, 830, 1940, 71, 1964, 881, 940, 1661,
    824, 507, 458, 50, 438, 1132, 1255, 463, 861, 152, 1235, 1499, 1095, 118, 1999, 1651, 828, 658,
    594, 1919, 1159, 247, 1089, 593, 751, 1026, 459, 1939, 661, 538, 858, 1072, 1963, 831, 1385,
    1277, 722, 922, 1665, 1774, 141, 1042, 81, 473, 1027, 649, 2004, 1169, 165, 1878, 966, 548,
    166, 213, 1624, 1751, 1686, 33, 1171, 104, 1764, 420, 808, 2007, 1497, 1287, 720, 1052, 1736,
    272, 777, 1639, 1765, 1536, 1822, 973, 1115, 99, 1298, 528, 517, 1505, 1302, 893, 1559, 1726,
    1088, 77, 2001, 1019, 299, 1508, 1579, 1744, 1186, 161, 114, 1454, 1387, 2034, 1554, 385, 852,
    542, 1476, 1846, 652, 860, 365, 441, 1475, 1535, 1500, 47, 1165, 1246, 59, 1794, 1761, 1498,
    1415, 1282, 1836, 588, 1253, 699, 522, 570, 1228, 321, 931, 2031, 1110, 842, 1484, 1631, 974,
    1448, 100, 2014, 607, 703, 1388, 540, 1174, 1571, 1192, 1310, 1083, 1429, 1323, 1576, 934, 38,
    1787, 73, 265, 904, 985, 1702, 1966, 1527, 1896, 805, 1539, 1204, 30, 685, 1030, 1677, 1401,
    83, 157, 121, 270, 437, 841, 1495, 223, 1215, 1000, 1782, 1642, 1845, 1012, 1216, 330, 478,
    1036, 958, 628, 1248, 408, 781, 764, 240, 554, 221, 1433, 456, 1521, 66, 1177, 1666, 1013,
    1270, 425, 1634, 572, 2043, 877, 754, 770, 1860, 717, 200, 474, 8, 17, 740, 1191, 1902, 1811,
    148, 1386, 1127, 1538, 1195, 615, 844, 1941, 1771, 336, 760, 1423, 2006, 1096, 1431, 1757,
    1670, 271, 1557, 162, 997, 768, 948, 144, 1687, 1487, 1446, 1080, 242, 1491, 1640, 1378, 1534,
    1149, 1150, 488, 133, 1320, 850, 1589, 812, 502, 302, 160, 1344, 1766, 1391, 1004, 5, 1156,
    1234, 526, 156, 1103, 825, 1731, 641, 935, 529, 648, 1461, 905, 186, 117, 1572, 609, 2016,
    1081, 275, 1245, 86, 614, 1206, 1082, 1998, 918, 1501, 773, 1886, 1459, 266, 637, 439, 389,
    1511, 867, 1645, 914, 932, 27, 216, 1364, 167, 1400, 1607, 293, 1600, 220, 826, 789, 1575,
    1694, 1056, 994, 24, 2010, 1326, 1985, 1154, 1064, 315, 1075, 1073, 776, 978, 669, 541, 1934,
    172, 952, 1611, 1471, 154, 69, 1430, 402, 363, 1490, 950, 1750, 1203, 101, 1268, 744, 195,
    1675, 119, 865, 1170, 341, 1755, 1236, 1355, 1732, 204, 1106, 1124, 1946, 356, 129, 1188, 656,
    1341, 1703, 285, 714, 250, 80, 102, 619, 1526, 1135, 1520, 62, 484, 1273, 1776, 1815, 684, 407,
    1604, 401, 1932, 1570, 1759, 158, 984, 489, 1643, 346, 1058, 1331, 1738, 1396, 1283, 1969,
    1989, 386, 1327, 61, 1888, 26, 1409, 2008, 1509, 1251, 446, 1847, 181, 949, 1151, 122, 1709,
    31, 1569, 337, 595, 261, 155, 1453, 1483, 1086, 514, 908, 399, 838, 1894, 1562, 1121, 245, 109,
    1157, 1609, 1584, 840, 942, 39, 1883, 716, 1178, 802, 2032, 1993, 771, 835, 85, 398, 545, 145,
    1161, 757, 1618, 509, 84, 519, 1412, 681, 1367, 886, 924, 988, 471, 706, 1730, 1328, 765, 537,
    187, 1692, 492, 1803, 912, 604, 1778, 613, 798, 1269, 1781, 951, 1140, 380, 338, 1413, 655,
    1252, 2021, 1713, 1247, 1626, 1222, 728, 1528, 1147, 176, 859, 1725, 2025, 120, 971, 362, 193,
    970, 16, 49, 1658, 1585, 686, 68, 214, 1594, 1739, 617, 1588, 602, 1201, 2027, 1813, 1711, 759,
    1333, 34, 238, 975, 1370, 724, 930, 219, 1418, 523, 123, 192, 1457, 784, 695, 832, 598, 1805,
    1143, 329, 1650, 1363, 1990, 1785, 1225, 1743, 57, 1979, 1864, 1094, 814, 1861, 1202, 1737,
    1097, 1444, 624, 1768, 1419, 281, 1613, 1627, 1405, 1424, 870, 412, 1549, 933, 1167, 1131, 451,
    1381, 440, 1673, 878, 544, 947, 673, 381, 1371, 1458, 1098, 601, 516, 633, 820, 707, 426, 1685,
    1882, 1398, 189, 889, 1581, 1220, 968, 1100, 110, 1482, 551, 603, 180, 1137, 552, 626, 1353,
    136, 803, 11, 1885, 746, 1825, 515, 1721, 1723, 674, 1908, 252, 304, 1628, 1227, 1318, 1117,
    1417, 1935, 151, 369, 1129, 206, 52, 63, 124, 454, 498, 351, 1120, 1158, 505, 1262, 1275, 1516,
    1884, 1869, 3, 2020, 461, 1142, 1982, 1118, 1037, 1652, 360, 1680, 1578, 1615, 1601, 2030, 503,
    1789, 475, 1959, 115, 139, 1752, 1582, 485, 1403, 1804, 577, 873, 709, 1031, 762, 1249, 2012,
    1655, 1066, 394, 1452, 290, 639, 1893, 435, 1564, 45, 871, 291, 1319, 153, 472, 1605, 530, 659,
    1973, 1622, 1936, 977, 1209, 605, 693, 954, 1357, 1267, 98, 761, 1548, 1790, 902, 589, 1168,
    627, 1608, 872, 1436, 1957, 236, 796, 415, 521, 917, 1113, 543, 1775, 67, 450, 1040, 1055,
    1988, 597, 267, 558, 487, 1076, 894, 2036, 147, 1426, 1913, 432, 486, 334, 429, 772, 149, 1695,
    1067, 1187, 1077, 322, 876, 1728, 1909, 862, 929, 1183, 774, 815, 1155, 683, 1492, 179, 163,
    303, 89, 480, 1054, 1657, 246, 1697, 1667, 499, 116, 1479, 1947, 928, 314, 1696, 300, 895, 783,
    1358, 1023, 231, 1960, 294, 128, 1876, 1297, 1294, 1630, 1190, 469, 405, 277, 849, 864, 357,
    1770, 513, 4, 1218, 786, 512, 1271, 606, 1849, 1970, 792, 382, 546, 1840, 1881, 244, 178, 1014,
    1800, 1683, 494, 1028, 1952, 1114, 2039, 778, 1017, 1070, 1184, 1897, 42, 1109, 1905, 1566,
    264, 1496, 1265, 1797, 535, 54, 758, 557, 94, 1259, 677, 1705, 1648, 1545, 2033, 226, 460, 713,
    1389, 1838, 1349, 1518, 433, 1286, 1769, 13, 12, 87, 1917, 1531, 1260, 368, 373, 578, 1343,
    464, 257, 827, 1194, 253, 1656, 1144, 1451, 836, 795, 1565, 1889, 82, 1863, 694, 1474, 1278,
    1108, 1119, 1196, 1603, 1016, 2019, 1032, 710, 1942, 1974, 874, 1852, 296, 1981, 1646, 1243,
    350, 1783, 92, 1205, 1586, 734, 839, 452, 666, 1898, 1212, 256, 670, 1547, 1305, 1010, 326, 72,
    249, 1663, 1951, 856, 591, 142, 705, 1801, 1510, 428, 1719, 298, 1965, 1688, 2013, 1312, 963,
    317, 1502, 1068, 1525, 183, 2045, 1924, 663, 701, 1679, 449, 1568, 477, 943, 1955, 1291, 9,
    1005, 1057, 821, 962, 1633, 361, 1377, 866, 427, 1020, 319, 1636, 1493, 1978, 75, 851, 642,
    610, 106, 689, 95, 1910, 1835, 2037, 1051, 388, 1166, 1923, 833, 1851, 1895, 347, 738, 730,
    647, 1112, 696, 1928, 1104, 126, 1039, 1592, 288, 297, 301, 718, 715, 785, 504, 1373, 1316,
    644, 218, 1223, 748, 816, 287, 1469, 481, 53, 662, 1392, 431, 378, 1360, 64, 1207, 370, 1806,
    1949, 583, 682, 555, 668, 1879, 2024, 1237, 97, 1254, 1532, 822, 417, 562, 539, 1189, 1274,
    880, 793, 182, 320, 1420, 1226, 234, 1899, 443, 804, 18, 1494, 857, 653, 1272, 953, 1488, 173,
    1907, 1290, 582, 422, 1984, 560, 1587, 1439, 1240, 1734, 1280, 1160, 749, 756, 775, 843, 1477,
    823, 608, 1875, 1996, 43, 1332, 1339, 46, 1093, 1130, 343, 1061, 2044, 998, 41, 1465, 1244,
    393, 262, 818, 911, 2046, 1368, 1994, 799, 1930, 817, 712, 1045, 612, 897, 1101, 1551, 1107,
    632, 657, 921, 1111, 1025, 564, 1092, 868, 1406, 630, 936, 1006, 335, 175, 747, 1834, 1208,
    1707, 416, 1071, 1306, 585, 1380, 1241, 923, 1456, 237, 801, 107, 1175, 1880, 1281, 137, 349,
    1356, 691, 229, 1802, 159, 1231, 525, 1411, 283, 925, 1533, 787, 1414, 782, 55, 185, 430, 1308,
    898, 1859, 700, 901, 1455, 65, 769, 991, 1421, 1848, 353, 1292, 1008, 1383, 211, 448, 6, 698,
    318, 1856, 79, 1122, 1877, 834, 419, 395, 1991, 1523, 1678, 1198, 1701, 1173, 1024, 1329, 40,
    1350, 1862, 1793, 1574, 22, 1529, 1791, 282, 423, 1078, 1866, 1372, 1540, 987, 664, 1335, 837,
    847, 807, 1038, 1463, 1324, 1048, 899, 1544, 227, 1197, 1105, 384, 587, 1503, 855, 1773, 1307,
    74, 1590, 1870, 1018, 135, 1379, 1340, 550, 20, 1059, 1145, 1873, 91, 553, 2015, 132, 1404,
    1704, 708, 1918, 1763, 1172, 150, 2038, 806, 1843, 891, 972, 1669, 956, 566, 996, 1623, 354,
    1322, 225, 316, 1250, 1314, 1138, 1029, 1330, 725, 1741, 442, 1927, 374, 1837, 697, 1854, 466,
    366, 1693, 1706, 794, 1915, 2003, 465, 721, 1700, 1912, 1179, 1550, 568, 752, 403, 1745, 496,
    1063, 2, 1826, 1079, 1336, 1485, 210, 1660, 1003, 28, 1397, 495, 1148, 937, 527, 19, 1890,
    1659, 70, 1715, 957, 737, 567, 258, 960, 755, 1125, 1002, 198, 1285, 15, 212, 500, 1716, 576,
    1629, 993, 1065, 1926, 1758, 1185, 177, 1395, 1009, 1558, 1313, 1035, 10, 945, 391, 1901, 687,
    1263, 635, 1992, 1182, 1916, 556, 1662, 1232, 887, 345, 1556, 1818, 1219, 1735, 1762, 1577,
    397, 1740, 196, 1362, 1593, 491, 60, 1337, 455, 1654, 1447, 1133, 946, 1473, 387, 1239, 203,
    286, 1833, 215, 732, 424, 1351, 274, 164, 2022, 611, 1972, 1891, 202, 194, 1541, 174, 741, 199,
    1238, 999, 2029, 1986, 88, 235, 1644, 406, 704, 468, 325, 650, 1676, 25, 1481, 1937, 233, 1514,
    1217, 1720, 511, 1311, 1438, 103, 1193, 1904, 1264, 1090, 643, 309, 675, 305, 586, 243, 600,
    462, 766, 1513, 848, 260, 875, 197, 723, 1258, 1047, 1460, 1342, 1408, 726, 134, 1141, 1289,
    409, 1961, 308, 1015, 579, 892, 622, 1242, 400, 1506, 323, 56, 1440, 339, 944, 1295, 1892,
    1976, 78, 640, 1462, 1950, 190, 964, 490, 1427, 171, 352, 992, 1684, 853, 883, 791, 276, 574,
    1053, 348, 306, 976, 1830, 1980, 170, 434, 1754, 621, 1011, 645, 44, 76, 241, 1489, 1962, 735,
    1779, 14, 1874, 1422, 654, 1583, 854, 1839, 1346, 1449, 1756, 1466, 1347, 191, 884, 169, 93,
    436, 959, 1084, 146, 846, 1515, 631, 1442, 1717, 483, 863, 1375, 888, 1001, 2026, 1724, 1480,
    1136, 983, 1922, 1858, 1486, 1635, 476, 1384, 2011, 634, 2002, 1614, 1467, 1619, 1689, 819,
    1293, 581, 209, 569, 667, 1266, 1517, 520, 331, 767, 312, 201, 1354, 1620, 32, 1617, 1567,
    1060, 1288, 590, 672, 359, 596, 1780, 1903, 457, 1043, 1606, 1221, 896, 1230, 692, 410, 1394,
    222, 885, 1369, 1933, 2047, 284, 1987, 263, 1213, 1727, 1200, 736, 108, 979, 986, 1102, 342,
    1682, 344, 1309, 127, 1671, 646, 1786, 1653, 1261, 980, 1967, 1616, 731, 1747, 636, 333, 1428,
    1407, 702, 1867, 1504, 1553, 1792, 1956, 1199, 1674, 111, 1710, 1034, 1300, 1748, 763, 105,
    565, 1470, 1139, 1507, 278, 1632, 1820, 1522, 939, 1664, 534, 1816, 1561, 383, 1742, 1920,
    1099, 1641, 1729, 113, 1279, 1091, 414, 941, 599, 1128, 255, 780, 1432, 1625, 1126, 1560, 1450,
    660, 445, 1938, 21, 1649, 1777, 23, 1602, 376, 1857, 48, 444, 184, 1512, 575, 1046, 1819, 479,
    969, 396, 2000, 332, 1708, 1366, 239, 1944, 1321, 1468, 1690, 665, 51, 1855, 208, 811, 1546,
    909, 750, 96, 1296, 36, 829, 1733, 1823, 1044, 797, 1842, 1906, 623, 228, 1, 845, 1146, 810,
    1519, 2009, 1437, 927, 1069, 1276, 961, 1914, 313, 729, 1393, 1315, 506, 727, 224, 733, 254,
    800, 1868, 217, 1050, 671, 1464, 561, 547, 745, 2023, 618, 907, 2041, 1865, 390, 1152, 1361,
    411, 311, 1434, 1647, 711, 563, 1597, 690, 616, 1390, 1303, 1691, 1637, 1163, 340, 651, 7,
    1085, 1598, 268, 779, 29, 1798, 531, 571, 1441, 1591, 1573, 879, 625, 1033, 1983, 1943, 371,
    679, 1352, 1760, 1425, 2017, 1831, 743, 1359, 916, 289, 1074, 584, 1134, 981, 620, 1007, 1958,
    328, 882, 379, 1062, 1596, 1746, 688, 510, 1552, 1580, 1345, 2028, 910, 1850, 919, 130, 2040,
    292, 1681, 2035, 1445, 967, 1542, 1338, 1638, 1749, 1788, 938, 518, 1977, 493, 1224, 990, 188,
    1049, 447, 143, 678, 90, 355, 248, 965, 629, 920, 418, 1997, 1210, 742, 1832, 1021, 903, 413,
    788, 1772, 533, 1921, 1599, 1164, 1712, 1931, 140, 592, 279, 138, 251, 1087, 1844, 1718, 168,
    809, 377, 501, 269, 1416, 1853, 364, 753, 1284, 1795, 1399, 1753, 1301, 1365, 532, 2005, 1829,
    906, 869, 232, 1872, 1767, 421, 1410, 1824, 1524, 392, 1971, 1929, 1948, 995, 524, 1317, 926,
    1325, 1698, 790, 1841, 1233, 0, 1478, 470, 1809, 890, 280, 324, 1821, 1672, 37, 1402, 559,
    1299, 536, 1176, 1796, 1443, 989, 1304, 372, 1537, 1022, 1887, 2018, 1229, 1722, 1162, 58,
    1808, 1699, 1180, 1784, 453, 1123, 205, 1382, 1812, 207, 1871, 1435, 1827, 680, 1563, 375,
    1810, 719, 1828, 1214, 1256, 1900, 295, 915, 573,
];

impl UnquantisedNetwork {
    /// Convert a parameter file generated by bullet into a merged parameter set,
    /// for further processing or for resuming training in a more efficient format.
    #[expect(clippy::too_many_lines)]
    fn merge(&self) -> Box<MergedNetwork> {
        #![allow(clippy::similar_names)]

        let mut net = MergedNetwork::zeroed();
        let buckets = self.l0_weights.chunks_exact_const::<{ 12 * 64 * L0_OUT }>();

        let alternate_buffer;
        let (factoriser, buckets) = if UNQUANTISED_HAS_FACTORISER {
            buckets.split_first().unwrap()
        } else {
            alternate_buffer = vec![0.0; 12 * 64 * L0_OUT];
            (alternate_buffer.as_array().unwrap(), buckets)
        };
        let target_buckets = net
            .l0_weights
            .chunks_exact_const_mut::<{ 12 * 64 * L0_OUT }>();
        for (src_bucket, tgt_bucket) in buckets.iter().zip(target_buckets) {
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
                    let src = &src_bucket[i * L0_OUT..i * L0_OUT + L0_OUT];
                    let fac_src = &factoriser[i * L0_OUT..i * L0_OUT + L0_OUT];
                    let tgt = &mut tgt_bucket[j * L0_OUT..j * L0_OUT + L0_OUT];
                    for ((src, fac_src), tgt) in src.iter().zip(fac_src).zip(tgt) {
                        *tgt = *src + *fac_src;
                    }
                }
            }
        }

        // copy the threat weights
        for i in 0..AUX_FEATURES * L0_OUT {
            net.l0_aux[i] = self.l0_aux[i];
        }

        // copy the biases
        net.l0_biases.copy_from_slice(&self.l0_biases);
        // copy the L1 weights
        for i in 0..L1_IN {
            for bucket in 0..OUTPUT_BUCKETS {
                for j in 0..L1_OUT {
                    net.l1_weights[i][bucket][j] = self.l1_weights[i][bucket][j];
                }
            }
        }
        // copy the L1 biases
        for i in 0..L1_OUT {
            for bucket in 0..OUTPUT_BUCKETS {
                net.l1_biases[bucket][i] = self.l1_biases[bucket][i];
            }
        }
        // copy the L2 weights
        for i in 0..L2_IN {
            for bucket in 0..OUTPUT_BUCKETS {
                for j in 0..L2_OUT * 2 {
                    net.l2_weights[i][bucket][j] =
                        self.l2x_weights[i][bucket][j] + self.l2f_weights[i][j];
                }
            }
        }
        // copy the L2 biases
        for i in 0..L2_OUT * 2 {
            for bucket in 0..OUTPUT_BUCKETS {
                net.l2_biases[bucket][i] = self.l2x_biases[bucket][i] + self.l2f_biases[i];
            }
        }
        // copy the L3 weights
        for i in 0..L3_IN {
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
        let buckets = self.l0_weights.chunks_exact_const::<{ 12 * 64 * L0_OUT }>();
        let target_buckets = net
            .l0_weights
            .chunks_exact_const_mut::<{ PSQT_FEATURES * L0_OUT }>();

        for (bucket_idx, (src_bucket, tgt_bucket)) in buckets.iter().zip(target_buckets).enumerate()
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
                    let src = &src_bucket[i * L0_OUT..i * L0_OUT + L0_OUT];
                    let tgt = &mut tgt_bucket[j * L0_OUT..j * L0_OUT + L0_OUT];
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
        for (src, tgt) in self.l0_aux.iter().zip(net.l0_aux.iter_mut()) {
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
        for i in 0..L1_IN {
            for bucket in 0..OUTPUT_BUCKETS {
                for j in 0..L1_OUT {
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
        let src_buckets = self
            .l0_weights
            .chunks_exact_const::<{ PSQT_FEATURES * L0_OUT }>();
        let tgt_buckets = net
            .l0_weights
            .chunks_exact_const_mut::<{ PSQT_FEATURES * L0_OUT }>();
        for (src_bucket, tgt_bucket) in src_buckets.iter().zip(tgt_buckets) {
            repermute_l0_psqt_bucket(tgt_bucket, src_bucket);
        }

        // permute the feature transformer biases
        repermute_l0_bias(&mut net.l0_biases, &self.l0_biases);

        // repermute the threat plane weights
        repermute_l0_aux(&mut net.l0_aux, &self.l0_aux);

        // transpose FT weights and biases so that packus transposes it back to the intended order
        if use_simd {
            type PermChunk<I> = [I; 8];
            // reinterpret as data of size __m128i
            let mut weights: Vec<&mut PermChunk<i16>> = net
                .l0_weights
                .chunks_exact_const_mut::<8>()
                .iter_mut()
                .collect();
            let mut biases: Vec<&mut PermChunk<i16>> = net
                .l0_biases
                .chunks_exact_const_mut::<8>()
                .iter_mut()
                .collect();
            let num_chunks = size_of::<PermChunk<i16>>() / size_of::<i16>();

            let num_regs = PACK_REGS;
            let order = PACK_ORDER;

            let mut regs = vec![[0i16; 8]; num_regs];

            // transpose weights
            for row in 0..PSQT_FEATURES * BUCKETS {
                let base = row * L0_OUT / num_chunks;
                for i in (0..L1_IN / num_chunks).step_by(num_regs) {
                    for j in 0..num_regs {
                        regs[j] = *weights[base + i + j];
                    }

                    for j in 0..num_regs {
                        *weights[base + i + j] = regs[order[j]];
                    }
                }
            }

            // transpose biases
            for i in (0..L1_IN / num_chunks).step_by(num_regs) {
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
                net.l0_aux.chunks_exact_const_mut().iter_mut().collect();
            for row in 0..AUX_FEATURES {
                let base = row * L0_OUT / num_chunks;
                for i in (0..L1_IN / num_chunks).step_by(num_regs) {
                    for j in 0..num_regs {
                        i8_regs[j] = *threat_weights[base + i + j];
                    }

                    for j in 0..num_regs {
                        *threat_weights[base + i + j] = i8_regs[order[j]];
                    }
                }
            }
        }

        // transpose the L{1,2,3} weights and biases
        let mut sorted = vec![[[0i8; L1_OUT]; OUTPUT_BUCKETS]; L1_IN];
        repermute_l1_weights(sorted.as_mut_array().unwrap(), &self.l1_weights);
        for bucket in 0..OUTPUT_BUCKETS {
            // quant the L1 weights
            if use_simd {
                for i in 0..L1_IN / L1_CHUNK_PER_32 {
                    for j in 0..L1_OUT {
                        for k in 0..L1_CHUNK_PER_32 {
                            net.l1_weights[bucket]
                                [i * L1_CHUNK_PER_32 * L1_OUT + j * L1_CHUNK_PER_32 + k] =
                                sorted[i * L1_CHUNK_PER_32 + k][bucket][j];
                        }
                    }
                }
            } else {
                for i in 0..L1_IN {
                    for j in 0..L1_OUT {
                        net.l1_weights[bucket][j * L1_IN + i] = sorted[i][bucket][j];
                    }
                }
            }

            // transfer the L1 biases
            for i in 0..L1_OUT {
                net.l1_bias[bucket][i] = self.l1_biases[bucket][i];
            }

            // transpose the L2 weights
            for i in 0..L2_IN {
                for j in 0..L2_OUT * 2 {
                    net.l2_weights[bucket][i * L2_OUT * 2 + j] = self.l2_weights[i][bucket][j];
                }
            }

            // transfer the L2 biases
            for i in 0..L2_OUT * 2 {
                net.l2_bias[bucket][i] = self.l2_biases[bucket][i];
            }

            // transfer the L3 weights
            for i in 0..L3_IN {
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
    sorted: &mut [[[i8; L1_OUT]; OUTPUT_BUCKETS]; L1_IN],
    l1_weights: &[[[i8; L1_OUT]; OUTPUT_BUCKETS]; L1_IN],
) {
    for (tgt_index, src_index) in REPERMUTE_INDICES.iter().copied().enumerate() {
        sorted[tgt_index] = l1_weights[src_index];
    }
    for (tgt_index, src_index) in REPERMUTE_INDICES.iter().copied().enumerate() {
        sorted[tgt_index + L1_IN / 2] = l1_weights[src_index + L1_IN / 2];
    }
}

fn repermute_l0_bias(feature_bias: &mut [i16; L0_OUT], unsorted: &[i16; L0_OUT]) {
    for (tgt_index, src_index) in REPERMUTE_INDICES.iter().copied().enumerate() {
        feature_bias[tgt_index] = unsorted[src_index];
    }
    for (tgt_index, src_index) in REPERMUTE_INDICES.iter().copied().enumerate() {
        feature_bias[tgt_index + L1_IN / 2] = unsorted[src_index + L1_IN / 2];
    }
}

fn repermute_l0_psqt_bucket(
    tgt_bucket: &mut [i16; PSQT_FEATURES * L0_OUT],
    unsorted: &[i16; PSQT_FEATURES * L0_OUT],
) {
    // for each input feature,
    for i in 0..PSQT_FEATURES {
        let feature = i * L0_OUT;
        // for each neuron in the layer,
        for (tgt_index, src_index) in REPERMUTE_INDICES.iter().copied().enumerate() {
            // get the neuron's corresponding weight in the unsorted bucket,
            // and write it to the same feature (but the new position) in the target bucket.
            tgt_bucket[feature + tgt_index] = unsorted[feature + src_index];
        }
        for (tgt_index, src_index) in REPERMUTE_INDICES.iter().copied().enumerate() {
            let tgt_index = tgt_index + L1_IN / 2;
            let src_index = src_index + L1_IN / 2;
            // get the neuron's corresponding weight in the unsorted bucket,
            // and write it to the same feature (but the new position) in the target bucket.
            tgt_bucket[feature + tgt_index] = unsorted[feature + src_index];
        }
    }
}

fn repermute_l0_aux(
    tgt: &mut Align<[i8; AUX_FEATURES * L0_OUT]>,
    unsorted: &[i8; AUX_FEATURES * L0_OUT],
) {
    for i in 0..AUX_FEATURES {
        let feature = i * L0_OUT;
        for (tgt_index, src_index) in REPERMUTE_INDICES.iter().copied().enumerate() {
            tgt[feature + tgt_index] = unsorted[feature + src_index];
        }
        for (tgt_index, src_index) in REPERMUTE_INDICES.iter().copied().enumerate() {
            let tgt_index = tgt_index + L1_IN / 2;
            let src_index = src_index + L1_IN / 2;
            tgt[feature + tgt_index] = unsorted[feature + src_index];
        }
    }
}

impl NNUEParams {
    /// Size of the mapped network, padded to 2MB.
    const WEIGHT_FILE_LEN: usize = size_of::<Self>().next_multiple_of(2 * MEGABYTE);

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

        // If we’re under MIRI, this function is way too slow.
        if cfg!(miri) {
            static MIRI_NULL_NETWORK: LazyLock<Box<NNUEParams>> = LazyLock::new(|| {
                // Safety: All bitpatterns of Self are valid.
                unsafe { Box::new_zeroed().assume_init() }
            });
            return Ok(&*MIRI_NULL_NETWORK);
        }

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
            "viridithas-shared-network-weights-v2-{}-{}-{}-{:X}.bin",
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
        file.set_len(Self::WEIGHT_FILE_LEN as u64)
            .with_context(|| {
                format!(
                    "Failed to set length of file at {} to {}",
                    temp_path.display(),
                    Self::WEIGHT_FILE_LEN,
                )
            })?;

        // SAFETY: This file must not be modified while we have a reference to it.
        // we avoid doing this ourselves, but we can't defend against other processes.
        let mut mmap = unsafe {
            memmap2::MmapOptions::new()
                .map_mut(&file)
                .with_context(|| format!("Failed to map temp file at {}", temp_path.display()))?
        };

        #[cfg(target_os = "linux")]
        let _ = mmap.advise(memmap2::Advice::HugePage);

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

        #[cfg(target_os = "linux")]
        {
            // https://man7.org/linux/man-pages/man2/madvise.2.html
            // MADV_COLLAPSE (since Linux 6.1)
            // > Perform a best-effort synchronous collapse of the native
            // > pages mapped by the memory range into Transparent Huge
            // > Pages (THPs).  MADV_COLLAPSE operates on the current state
            // > of memory of the calling process and makes no persistent
            // > changes or guarantees on how pages will be mapped,
            // > constructed, or faulted in the future.
            const MADV_COLLAPSE: libc::c_int = 25;
            let _ = mmap.advise(memmap2::Advice::HugePage);
            // SAFETY: AIUI it is hard to misuse madvise().
            //         We pass a valid range of memory.
            unsafe { libc::madvise(mmap.as_ptr().cast_mut().cast(), mmap.len(), MADV_COLLAPSE) };
        }

        anyhow::ensure!(
            mmap.len() == Self::WEIGHT_FILE_LEN,
            "Wrong number of bytes: expected {}, got {}",
            Self::WEIGHT_FILE_LEN,
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

    pub fn select_feature_weights(&self, bucket: usize) -> &Align<[i16; PSQT_FEATURES * L0_OUT]> {
        // handle mirroring
        let bucket = bucket % BUCKETS;
        let start = bucket * PSQT_FEATURES * L0_OUT;
        let end = start + PSQT_FEATURES * L0_OUT;
        let slice = &self.l0_weights[start..end];
        // SAFETY: The resulting slice is indeed INPUT × L0_OUT long,
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
    pub fn index(self, colour: Colour, king: Square) -> (bool, u32) {
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
pub struct AuxUpdateBuffer {
    pub add: ArrayVec<ThreatFeatureUpdate, 128>,
    pub sub: ArrayVec<ThreatFeatureUpdate, 128>,
    pub afore: [SquareSet; 2],
    pub after: [SquareSet; 2],
}

impl AuxUpdateBuffer {
    pub fn clear(&mut self) {
        self.add.clear();
        self.sub.clear();
        #[cfg(debug_assertions)]
        {
            self.afore = [SquareSet::FULL; 2];
            self.after = [SquareSet::FULL; 2];
        }
    }
}

/// Combined PSQT + threat update buffer, filled during move-making.
#[derive(PartialEq, Eq, Clone, Debug, Default)]
pub struct UpdateBuffer {
    pub psqt: PsqtUpdateBuffer,
    pub aux: AuxUpdateBuffer,
}

impl UpdateBuffer {
    pub fn clear(&mut self) {
        self.psqt.clear();
        self.aux.clear();
    }
}

/// Stores last-seen accumulators for each bucket, so that we can hopefully avoid
/// having to completely recompute the accumulator for a position, instead
/// partially reconstructing it from the last-seen accumulator.
pub struct BucketAccumulatorCache {
    // both of these are BUCKETS * 2, rather than just BUCKETS,
    // because we use a horizontally-mirrored architecture.
    accs: [[Align<[i16; ACC_LEN]>; 2]; BUCKETS * 2],
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
            accumulator::refresh_aux(
                &nnue_params.l0_aux,
                &mut self.threat_accumulators[0].halves[colour],
                board,
                colour,
            );
            self.threat_correct[0][colour] = true;
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
                Self::materialise_new_aux_acc_from(
                    src_acc,
                    tgt_acc,
                    &self.updates[curr_index].aux,
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
                    accumulator::refresh_aux(
                        &nnue_params.l0_aux,
                        &mut self.threat_accumulators[self.current_acc].halves[colour],
                        board,
                        colour,
                    );
                    self.threat_correct[self.current_acc][colour] = true;
                }
            }
        }
    }

    /// Shunt the top of the acc stack to the bottom.
    #[cfg(feature = "datagen")]
    pub fn collapse_stack(&mut self) {
        assert_eq!(self.psqt_correct[self.current_acc], [true; 2]);
        assert_eq!(self.threat_correct[self.current_acc], [true; 2]);

        if self.current_acc == 0 {
            return;
        }

        let (bottom, top) = self.psqt_accumulators.split_at_mut(self.current_acc);
        bottom[0].halves.clone_from(&top[0].halves);
        let (bottom, top) = self.threat_accumulators.split_at_mut(self.current_acc);
        bottom[0].halves.clone_from(&top[0].halves);
        self.psqt_correct[0] = [true; 2];
        self.threat_correct[0] = [true; 2];
        self.current_acc = 0;
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

    pub fn materialise_new_aux_acc_from(
        src_acc: &Accumulator,
        tgt_acc: &mut Accumulator,
        updates: &AuxUpdateBuffer,
        king: Square,
        colour: Colour,
        nnue_params: &NNUEParams,
    ) {
        let src = &src_acc.halves[colour];
        let tgt = &mut tgt_acc.halves[colour];

        accumulator::vector_update_aux(src, tgt, &nnue_params.l0_aux, updates, king, colour);
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

        let mut l2_inputs = Align([0.0; L2_IN]);
        let mut l3_inputs = Align([0.0; L3_IN]);

        layers::activate_ft_and_propagate_l1(
            stm_psqt,
            ntm_psqt,
            stm_thrt,
            ntm_thrt,
            &nn.l1_weights[out],
            &nn.l1_bias[out],
            &mut l2_inputs,
        );
        layers::propagate_l2(
            &l2_inputs,
            &nn.l2_weights[out],
            &nn.l2_bias[out],
            &mut l3_inputs,
        );

        if HEADS == 1 {
            let mut l3_output = 0.0;

            layers::propagate_l3(
                &l3_inputs,
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
                layers::propagate_l3(&l3_inputs, w, b, o);
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
    for neuron in 0..L0_OUT {
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
        assert!(neuron < L0_OUT);
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
                    let index = feature_index.index() * L0_OUT + starting_idx;
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
        let cols = (1..=L0_OUT)
            .min_by_key(|&c| {
                let rows = L0_OUT.div_ceil(c);
                let w = c * (TILE_W + IMAGE_SPACING);
                let h = rows * (TILE_H + IMAGE_SPACING);
                // minimise |w/h - 16/9|, i.e. |9w - 16h|
                (9 * w).abs_diff(16 * h)
            })
            .unwrap();
        let rows = L0_OUT.div_ceil(cols);
        let img_w = cols * TILE_W + (cols - 1) * IMAGE_SPACING;
        let img_h = rows * TILE_H + (rows - 1) * IMAGE_SPACING;

        #[expect(clippy::cast_possible_truncation)]
        let neuron_order = (0..L0_OUT as u16).collect::<ArrayVec<u16, L0_OUT>>();
        // the sorting is nice, but doesn’t look quite so pretty.
        // neuron_order.sort_by_key(|&n| {
        //     let start = n;
        //     let mean_abs: u64 = (0..INPUT)
        //         .map(|i| u64::from(self.l0_weights[i * L0_OUT + start as usize].unsigned_abs()))
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
