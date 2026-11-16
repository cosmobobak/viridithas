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
    1900, 1376, 1911, 1799, 230, 1975, 1995, 580, 1181, 1493, 1621, 1472, 739, 982, 615, 1153,
    1668, 1334, 1374, 1595, 1257, 112, 205, 1784, 1123, 1020, 502, 302, 160, 1344, 1636, 788, 427,
    1810, 903, 1021, 851, 479, 742, 1886, 637, 1149, 27, 1511, 1645, 914, 389, 216, 439, 332, 1468,
    2000, 2034, 1554, 1080, 852, 1476, 1487, 1435, 842, 1846, 1446, 900, 1041, 1687, 508, 1812,
    144, 985, 1530, 768, 77, 1557, 162, 997, 1030, 685, 30, 1204, 1539, 1896, 1527, 1702, 1954,
    904, 1538, 1941, 844, 1195, 1505, 242, 1786, 1782, 1378, 1807, 1845, 1012, 330, 805, 1036, 628,
    408, 1248, 764, 781, 958, 1305, 240, 1783, 350, 560, 1439, 33, 194, 1171, 2007, 612, 1400, 569,
    1212, 1347, 428, 1898, 666, 1200, 1213, 839, 986, 936, 474, 630, 1406, 740, 1405, 1115, 99,
    528, 1298, 1423, 517, 1302, 1771, 1249, 1726, 760, 1627, 1424, 870, 336, 1096, 271, 893, 1559,
    1088, 980, 1612, 367, 1744, 1126, 660, 1560, 113, 1432, 1640, 1128, 255, 780, 1287, 599, 1708,
    396, 429, 486, 149, 720, 772, 1456, 776, 1427, 1118, 1652, 1144, 190, 964, 490, 1142, 993, 53,
    1656, 1104, 827, 1553, 1674, 378, 565, 779, 417, 1254, 1879, 2024, 1340, 1237, 1314, 177, 491,
    613, 1926, 1778, 604, 912, 1160, 492, 440, 1551, 836, 187, 1765, 973, 1536, 1822, 272, 777,
    1387, 296, 1052, 161, 1607, 1819, 373, 444, 1937, 575, 1836, 244, 1451, 795, 1267, 570, 1876,
    1933, 1775, 1881, 969, 1014, 924, 706, 287, 1223, 1245, 916, 584, 1074, 1134, 328, 981, 1433,
    163, 379, 1155, 1492, 815, 683, 828, 1523, 732, 1991, 280, 324, 1809, 890, 959, 1478, 1620,
    1489, 1615, 1998, 1206, 918, 1501, 1459, 773, 266, 614, 1082, 118, 1243, 1981, 678, 1949, 1806,
    1207, 1366, 1564, 435, 45, 665, 1893, 208, 811, 1546, 639, 1452, 96, 133, 2001, 1589, 1320,
    239, 365, 860, 1279, 1860, 414, 1616, 1639, 1534, 1382, 1491, 223, 437, 1019, 1508, 299, 1386,
    59, 1475, 1165, 1415, 1282, 1127, 1761, 1228, 450, 522, 699, 1022, 588, 233, 1367, 109, 245,
    921, 1889, 1121, 1770, 1454, 810, 874, 1498, 845, 1, 1146, 1519, 228, 680, 12, 385, 1827, 2032,
    207, 802, 1993, 607, 2014, 786, 606, 694, 1448, 513, 623, 974, 1192, 321, 681, 512, 540, 84,
    703, 2031, 1110, 100, 1484, 265, 1631, 1787, 73, 38, 1576, 934, 1323, 1429, 1083, 1571, 931,
    1174, 1388, 542, 1883, 39, 942, 75, 1625, 1584, 1157, 1565, 1609, 840, 1863, 1218, 4, 82, 1178,
    1310, 1215, 1828, 1256, 375, 1214, 719, 1563, 1437, 1796, 2009, 927, 1069, 1613, 1794, 1768,
    1419, 624, 1444, 1097, 1202, 1737, 67, 1425, 1920, 625, 1792, 1352, 1943, 1944, 371, 920, 1573,
    629, 965, 1441, 464, 90, 1266, 1913, 274, 447, 1164, 1260, 1033, 248, 66, 355, 1596, 738, 647,
    1746, 730, 1844, 823, 597, 938, 1869, 2020, 1416, 1516, 1884, 1379, 1275, 1262, 1918, 550, 553,
    978, 132, 1637, 1303, 316, 340, 1085, 1598, 1163, 7, 891, 668, 682, 1159, 1919, 858, 999, 1939,
    459, 168, 695, 751, 809, 800, 1669, 1758, 972, 1065, 1843, 2038, 806, 150, 956, 411, 1980, 669,
    1647, 1086, 838, 399, 990, 1562, 592, 1499, 832, 1942, 598, 1805, 86, 329, 1617, 1107, 587,
    1923, 1619, 24, 819, 1293, 1354, 1614, 32, 201, 1749, 1964, 746, 1721, 1825, 1602, 515, 419,
    834, 1122, 1877, 400, 470, 987, 1728, 0, 1233, 790, 472, 153, 291, 1319, 871, 1988, 1316, 360,
    907, 139, 1865, 1528, 618, 2041, 390, 1718, 2023, 253, 745, 547, 1464, 2025, 561, 671, 971,
    859, 380, 951, 338, 655, 1252, 1413, 347, 1781, 1247, 1725, 2021, 1622, 581, 456, 667, 209,
    331, 1194, 520, 688, 1742, 1093, 653, 1332, 1339, 43, 1692, 2045, 658, 164, 1399, 661, 608,
    269, 1853, 1552, 62, 753, 1365, 875, 102, 724, 1370, 219, 1418, 123, 523, 192, 930, 1892, 1462,
    1453, 975, 574, 992, 883, 276, 763, 651, 29, 268, 514, 571, 182, 662, 531, 1392, 481, 504,
    1373, 644, 218, 1543, 1578, 404, 1968, 830, 676, 549, 638, 131, 467, 358, 310, 2006, 290, 917,
    627, 796, 872, 1436, 236, 1957, 415, 521, 1608, 717, 8, 148, 1191, 17, 1579, 1811, 1902, 1757,
    1670, 2043, 770, 1066, 877, 754, 572, 1634, 1925, 1102, 47, 733, 254, 935, 1593, 1154, 293,
    1600, 2010, 1056, 220, 826, 789, 1642, 1575, 771, 85, 398, 835, 545, 145, 519, 1161, 1618, 757,
    509, 1905, 1109, 778, 2039, 1184, 546, 1897, 42, 1566, 494, 264, 1265, 1797, 1496, 1917, 535,
    68, 54, 758, 460, 2033, 713, 226, 650, 994, 1531, 368, 87, 325, 1286, 1349, 1838, 13, 433,
    1518, 1769, 1858, 217, 1676, 352, 279, 1075, 171, 315, 1064, 1294, 506, 1586, 1985, 1062, 2004,
    247, 92, 734, 1234, 1646, 452, 1852, 1696, 300, 783, 895, 1023, 1960, 1967, 128, 294, 167, 702,
    1671, 333, 1907, 1747, 1428, 913, 761, 281, 1168, 1113, 1804, 485, 1987, 1857, 817, 1231, 881,
    159, 1802, 48, 229, 1880, 1788, 530, 1894, 634, 1225, 1689, 564, 868, 1025, 1467, 2002, 1635,
    476, 1384, 1208, 2011, 1220, 189, 1977, 1581, 1517, 493, 1224, 188, 126, 696, 359, 596, 1928,
    297, 1592, 1039, 138, 443, 1300, 1112, 1973, 105, 977, 543, 1984, 267, 1426, 1055, 558, 147,
    1720, 1285, 1076, 1542, 511, 1217, 1481, 1311, 382, 25, 2036, 1486, 894, 1904, 1193, 1090, 432,
    1077, 152, 463, 787, 1132, 438, 334, 1661, 767, 1067, 1465, 820, 1061, 41, 516, 812, 998, 1098,
    1385, 850, 679, 1004, 1983, 5, 1731, 1391, 1049, 1766, 1143, 393, 1999, 1582, 1959, 1601, 475,
    1363, 482, 57, 1785, 1094, 1743, 1990, 701, 1891, 1650, 1100, 202, 1547, 1238, 1644, 736, 199,
    108, 2029, 174, 1986, 235, 88, 1974, 406, 313, 1315, 468, 40, 727, 691, 1342, 1329, 407, 275,
    1701, 1173, 1815, 227, 1716, 172, 1629, 1035, 865, 1395, 1313, 1558, 1350, 1185, 292, 449,
    1604, 1776, 2016, 1755, 401, 1198, 1308, 1396, 135, 1482, 984, 311, 1992, 1570, 1197, 1934,
    130, 1445, 1681, 2040, 2035, 967, 1338, 1638, 1753, 919, 910, 2028, 1948, 1345, 1850, 1513,
    377, 1152, 224, 566, 1623, 1759, 1236, 996, 1273, 923, 1763, 1009, 193, 1172, 120, 91, 225,
    2015, 1250, 1927, 1326, 1018, 60, 1473, 946, 1337, 1654, 1447, 1133, 455, 1239, 387, 1935, 525,
    952, 1420, 552, 1932, 1873, 1526, 1520, 1471, 1611, 69, 154, 1961, 619, 80, 364, 1135, 1129,
    362, 16, 1390, 1594, 970, 1585, 697, 374, 659, 1854, 1837, 1693, 616, 1372, 690, 1264, 925,
    1866, 1540, 282, 1078, 1791, 22, 423, 409, 1529, 782, 1533, 134, 766, 1969, 243, 600, 586, 345,
    675, 1219, 462, 416, 818, 262, 1801, 1175, 1244, 2047, 1369, 284, 263, 1040, 1309, 1092, 98,
    1450, 1790, 1548, 1679, 1994, 1587, 889, 968, 1187, 911, 609, 322, 2046, 621, 1011, 2005, 1355,
    1821, 238, 34, 170, 44, 1754, 1331, 949, 341, 1170, 119, 312, 1643, 1628, 1434, 158, 1124,
    1404, 708, 1704, 489, 714, 541, 285, 250, 176, 349, 1750, 1477, 756, 749, 503, 843, 424, 611,
    1572, 1651, 1972, 1081, 963, 2022, 376, 477, 1510, 799, 1071, 1707, 1780, 1903, 457, 1043,
    1606, 1221, 2042, 896, 1230, 692, 410, 1394, 222, 1276, 885, 704, 1914, 961, 1393, 729, 1461,
    2018, 117, 536, 1299, 1176, 1402, 37, 559, 1672, 81, 1042, 979, 642, 866, 1633, 1377, 142, 95,
    106, 1910, 343, 1130, 1719, 1734, 1864, 1068, 183, 1502, 317, 1525, 1861, 2037, 1211, 326, 388,
    1166, 833, 1051, 1895, 716, 141, 1665, 1774, 922, 1277, 722, 831, 3, 2044, 107, 1072, 1963,
    538, 649, 165, 1169, 657, 1027, 966, 1878, 548, 213, 166, 200, 1380, 1624, 1686, 1751, 430,
    185, 55, 1414, 1756, 392, 1875, 1996, 1824, 1610, 1946, 1353, 656, 356, 1188, 1703, 1106, 129,
    1341, 1675, 995, 1698, 124, 1971, 898, 110, 1862, 232, 1872, 1829, 308, 869, 1826, 1015, 1408,
    674, 707, 426, 824, 633, 1685, 1398, 1882, 991, 631, 863, 1717, 146, 846, 1515, 1375, 1442,
    483, 1001, 1283, 888, 563, 1327, 155, 1709, 595, 337, 31, 1569, 261, 122, 711, 1151, 1847, 181,
    26, 524, 446, 1409, 2008, 1888, 1251, 794, 1706, 434, 602, 2027, 1201, 1588, 617, 1739, 1438,
    686, 15, 1890, 791, 19, 1037, 1840, 1328, 765, 1730, 537, 1411, 1148, 897, 283, 712, 1045, 178,
    721, 1003, 576, 1590, 1700, 1550, 28, 1485, 568, 626, 10, 945, 391, 687, 899, 837, 1263, 635,
    1901, 1705, 49, 1284, 1735, 677, 643, 1740, 1389, 196, 1762, 1577, 1362, 1658, 1818, 1503,
    1105, 1544, 384, 431, 855, 1351, 1346, 94, 1259, 1545, 557, 1773, 471, 74, 70, 1514, 988, 1114,
    1800, 1952, 1028, 212, 1683, 500, 960, 198, 1070, 755, 303, 1648, 1017, 1002, 1125, 465, 1509,
    103, 1915, 1182, 556, 1916, 1232, 1662, 1137, 887, 1556, 848, 397, 260, 395, 723, 197, 14, 305,
    309, 203, 1322, 484, 1183, 1330, 1348, 955, 1138, 1953, 813, 1360, 1488, 1240, 173, 1290, 422,
    582, 1729, 1186, 1099, 221, 841, 1403, 1867, 748, 357, 1190, 1013, 849, 1297, 902, 344, 1979,
    127, 342, 412, 1031, 273, 1851, 35, 497, 1312, 361, 962, 856, 1057, 821, 9, 249, 1005, 1835,
    1688, 2013, 298, 1965, 814, 577, 2017, 743, 1359, 591, 445, 319, 1760, 1364, 451, 1381, 18,
    1131, 1682, 1549, 933, 1641, 307, 1167, 1091, 941, 1924, 1727, 1006, 1111, 1834, 335, 1103,
    175, 747, 115, 420, 1306, 585, 1235, 50, 1241, 1255, 104, 861, 458, 1541, 507, 741, 579, 551,
    1301, 1989, 180, 1179, 369, 52, 206, 63, 304, 351, 1158, 505, 603, 1120, 1583, 1307, 1659, 527,
    461, 1982, 957, 567, 737, 258, 1397, 1660, 1870, 1463, 1324, 1325, 1455, 926, 1841, 403, 466,
    1715, 1745, 937, 2, 2026, 1724, 2003, 1136, 1480, 983, 1922, 214, 1885, 1908, 518, 1723, 252,
    71, 940, 1695, 1227, 590, 1060, 672, 1567, 1288, 594, 286, 501, 215, 495, 1833, 801, 1795, 684,
    1361, 1597, 454, 1738, 346, 1813, 498, 735, 1145, 20, 1087, 251, 1059, 97, 473, 816, 301, 288,
    1222, 1626, 728, 1710, 1147, 1034, 774, 234, 929, 1054, 1657, 480, 179, 89, 1457, 1073, 792,
    1691, 1970, 1271, 1849, 1412, 1694, 1343, 886, 632, 354, 578, 1868, 1101, 1156, 46, 1752, 689,
    1649, 21, 23, 1777, 670, 610, 1095, 705, 1291, 663, 1955, 943, 1007, 1568, 1026, 882, 976,
    1830, 348, 1874, 306, 1368, 1048, 1912, 775, 1930, 1038, 6, 1856, 1490, 807, 847, 664, 1335,
    1962, 1779, 402, 1203, 950, 101, 1289, 195, 1268, 1336, 1574, 1848, 1767, 1793, 1410, 421, 79,
    448, 61, 892, 622, 1524, 1242, 1506, 645, 1079, 76, 241, 1063, 1440, 323, 1141, 56, 339, 744,
    1317, 386, 1732, 1929, 726, 1711, 1333, 759, 1318, 151, 1417, 1117, 366, 363, 496, 1460, 65,
    211, 1258, 1047, 318, 353, 1383, 1292, 1008, 698, 752, 532, 136, 1421, 204, 803, 210, 769, 11,
    1058, 908, 78, 640, 1950, 1976, 1839, 854, 436, 1084, 257, 1430, 1280, 1483, 93, 191, 884,
    1422, 654, 1466, 1029, 1741, 64, 725, 804, 370, 857, 1899, 156, 1356, 1591, 137, 1497, 1205,
    1599, 825, 648, 1736, 905, 529, 1764, 186, 641, 808, 1712, 1458, 140, 1931, 1512, 184, 1772,
    1921, 533, 413, 1832, 1162, 1722, 58, 1180, 1808, 1699, 1119, 1997, 1474, 1108, 554, 441, 1278,
    947, 1887, 1229, 143, 1150, 418, 488, 453, 1210, 1500, 1535, 1663, 1246, 114, 1046, 1938, 932,
    1978, 1253, 1010, 72, 1951, 256, 989, 372, 1443, 1537, 1304, 601, 1958, 259, 1116, 1431, 873,
    762, 709, 715, 589, 327, 526, 1945, 1140, 1521, 1469, 2012, 1504, 1655, 394, 750, 909, 1816,
    1677, 1000, 1653, 948, 1261, 646, 1855, 51, 1494, 1690, 1407, 1321, 381, 636, 544, 878, 731,
    1673, 1357, 383, 1555, 1226, 693, 954, 673, 1209, 605, 862, 1936, 289, 700, 876, 1605, 1909,
    901, 487, 1678, 1024, 1859, 1680, 906, 2030, 1789, 718, 1449, 1814, 320, 1532, 1798, 793, 822,
    562, 1189, 539, 1274, 880, 785, 1281, 169, 442, 1295, 237, 620, 944, 1053, 510, 1580, 583, 125,
    1714, 853, 1272, 953, 1831, 1561, 879, 1748, 1470, 1507, 1820, 1664, 1632, 939, 278, 1269,
    1139, 1089, 469, 798, 1630, 425, 555, 1050, 1270, 593, 405, 1713, 277, 1522, 534, 111, 1199,
    1940, 829, 1956, 1371, 1177, 1196, 1603, 1016, 1032, 2019, 784, 928, 1479, 314, 1803, 1947,
    1684, 864, 116, 499, 1667, 797, 1697, 246, 1358, 231, 1666, 652, 1296, 36, 1733, 867, 1906,
    1044, 1842, 710, 1823, 1401, 1216, 478, 83, 270, 157, 1495, 121, 1966, 1871, 1817, 295, 915,
    573,
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
