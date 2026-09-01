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
    util::{Align, MAX_DEPTH},
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
    1900, 1376, 1911, 1799, 1995, 230, 580, 1041, 615, 900, 1812, 1153, 508, 242, 914, 1080, 1446,
    1382, 1687, 144, 985, 768, 1530, 1527, 1896, 30, 1539, 1204, 685, 805, 83, 162, 1088, 77, 997,
    1508, 1019, 299, 1210, 1616, 980, 1663, 1500, 1559, 1535, 1386, 441, 1579, 1902, 1811, 1191,
    17, 740, 8, 474, 717, 1608, 521, 627, 1957, 236, 872, 1436, 796, 917, 290, 1988, 435, 45, 871,
    291, 1319, 153, 790, 472, 0, 1605, 1455, 1728, 154, 1698, 400, 926, 1317, 1325, 1841, 10, 1421,
    1550, 136, 752, 210, 1485, 1336, 318, 1826, 991, 426, 1882, 1398, 633, 1685, 824, 1227, 707,
    1318, 930, 1908, 1628, 192, 395, 1821, 170, 1523, 976, 1980, 1026, 1568, 943, 1955, 1092, 1679,
    1790, 1548, 1450, 98, 1309, 1091, 21, 1074, 1212, 670, 1898, 1347, 428, 256, 1081, 1502, 963,
    1068, 1005, 249, 9, 1835, 821, 1291, 610, 95, 642, 141, 1391, 922, 1774, 722, 1665, 1277, 705,
    5, 142, 1633, 302, 1377, 962, 317, 1057, 1312, 979, 1861, 1688, 298, 2013, 1965, 577, 1488,
    2037, 388, 1166, 1923, 1051, 833, 913, 716, 1777, 1938, 660, 1851, 1682, 1423, 1167, 941, 414,
    1641, 1729, 1786, 383, 127, 646, 1000, 841, 1690, 1428, 1671, 1407, 1867, 405, 105, 277, 1507,
    1664, 1470, 1820, 939, 278, 1632, 1742, 1710, 1269, 358, 504, 565, 526, 1356, 1281, 169, 691,
    137, 436, 1872, 1617, 1601, 1363, 1990, 110, 1891, 202, 1100, 393, 1999, 1783, 1766, 1234,
    1246, 160, 440, 239, 1099, 582, 422, 1907, 1290, 173, 1748, 953, 1272, 1561, 1360, 879, 625,
    1199, 1792, 111, 1674, 827, 1034, 53, 1656, 1104, 993, 490, 1927, 516, 820, 41, 998, 1061,
    2044, 649, 1072, 1963, 538, 247, 831, 1385, 1801, 818, 1465, 262, 1244, 1175, 2047, 284, 936,
    1006, 630, 1111, 335, 1834, 254, 175, 733, 224, 608, 1701, 1853, 753, 1482, 495, 1815, 1142,
    172, 527, 1659, 461, 957, 1590, 669, 1018, 1073, 925, 1753, 1399, 1324, 1463, 1795, 807, 847,
    995, 1597, 1283, 1303, 684, 1738, 552, 1273, 1759, 1932, 1124, 158, 119, 129, 744, 1703, 1341,
    1506, 1188, 356, 656, 1610, 1222, 1626, 955, 859, 745, 1348, 390, 120, 401, 1558, 1570, 193,
    1152, 91, 1129, 1873, 2015, 553, 132, 1637, 1918, 1647, 1075, 1331, 1404, 978, 708, 411, 956,
    1172, 150, 806, 1843, 2038, 1350, 891, 1185, 1758, 972, 1065, 809, 1669, 1009, 996, 285, 1567,
    286, 1288, 520, 215, 672, 590, 1049, 1651, 90, 678, 637, 439, 867, 389, 1511, 1645, 216, 27,
    932, 1454, 1186, 1744, 1534, 167, 1612, 948, 2001, 453, 96, 133, 1589, 850, 1320, 502, 856,
    1020, 1951, 319, 1010, 72, 1636, 1253, 35, 860, 365, 624, 1444, 1202, 1737, 1097, 866, 2017,
    342, 445, 1352, 1425, 281, 1920, 1497, 1369, 1205, 92, 666, 754, 770, 1380, 200, 166, 548, 213,
    966, 165, 657, 1169, 921, 1027, 777, 1433, 836, 486, 599, 396, 720, 429, 149, 772, 1695, 334,
    1661, 1060, 456, 815, 1173, 1678, 123, 724, 219, 1370, 1418, 523, 1620, 1998, 1501, 918, 1459,
    773, 266, 1206, 1062, 463, 152, 981, 370, 1880, 1802, 159, 229, 1245, 1755, 1602, 1964, 1825,
    1721, 746, 803, 626, 11, 688, 1268, 1353, 331, 252, 470, 1948, 1106, 124, 211, 1084, 1615, 483,
    1375, 1442, 1515, 631, 1763, 146, 846, 2016, 959, 609, 828, 1478, 787, 280, 890, 324, 1809,
    1991, 515, 419, 834, 1122, 1877, 1342, 1460, 1047, 1258, 766, 723, 197, 496, 403, 1745, 532,
    1724, 2003, 1723, 1480, 1136, 983, 1922, 1888, 1327, 1409, 61, 448, 386, 892, 1242, 1141, 2026,
    1011, 323, 1203, 987, 1024, 1490, 312, 6, 1856, 1930, 1038, 377, 1788, 1681, 292, 1338, 1934,
    1749, 2040, 130, 1489, 1962, 241, 76, 645, 44, 1063, 698, 1675, 1137, 984, 1754, 180, 568, 2,
    1179, 1969, 369, 206, 52, 63, 454, 304, 1379, 346, 135, 1301, 1604, 1396, 498, 551, 260, 397,
    462, 600, 848, 586, 243, 309, 675, 305, 643, 1090, 196, 650, 226, 1518, 1740, 460, 1389, 433,
    1838, 1286, 1349, 13, 12, 1769, 927, 1519, 1437, 1276, 810, 1415, 1214, 2009, 1498, 161, 59,
    114, 223, 1215, 874, 1770, 710, 1016, 554, 1108, 1196, 1278, 947, 1887, 188, 1229, 1722, 1162,
    58, 1808, 1699, 1150, 1180, 1997, 1832, 1784, 1021, 742, 427, 788, 903, 1921, 533, 413, 1772,
    1164, 1712, 1599, 808, 194, 1171, 2007, 420, 1634, 33, 1686, 1624, 430, 1751, 185, 55, 1414,
    782, 1308, 1533, 56, 898, 906, 1008, 1233, 353, 876, 940, 1067, 322, 1909, 1187, 1994, 889,
    968, 1971, 2046, 1220, 1581, 189, 1517, 1638, 1977, 493, 518, 359, 596, 641, 297, 1731, 1928,
    1004, 126, 1039, 1986, 990, 592, 1087, 505, 251, 43, 138, 1844, 1112, 1746, 696, 1300, 647,
    730, 738, 143, 1650, 989, 372, 1304, 1537, 601, 1458, 1098, 488, 1149, 1371, 679, 812, 1708,
    479, 1640, 1287, 259, 1560, 1126, 1432, 1128, 780, 255, 1491, 1860, 1279, 1639, 1178, 1727,
    1822, 973, 1115, 99, 1298, 528, 517, 1302, 1505, 1941, 893, 844, 1195, 271, 1782, 1816, 1677,
    1967, 1653, 1401, 1553, 1823, 1030, 534, 1044, 909, 1522, 750, 1655, 761, 1546, 1504, 415, 394,
    1893, 1452, 639, 208, 811, 51, 1855, 665, 1261, 1468, 1487, 1747, 1378, 1845, 1642, 220, 826,
    789, 1600, 1575, 1056, 2010, 24, 994, 293, 1694, 792, 368, 1271, 546, 1531, 1496, 1970, 1917,
    68, 54, 287, 557, 1105, 1346, 1583, 227, 431, 1048, 1874, 348, 475, 306, 1830, 434, 621, 1859,
    339, 363, 1440, 93, 1295, 1976, 1540, 944, 78, 525, 1462, 776, 640, 964, 190, 1680, 1950, 514,
    952, 880, 1839, 257, 1274, 1223, 218, 816, 1316, 644, 748, 481, 1469, 571, 1591, 1441, 1573,
    629, 965, 920, 248, 371, 1944, 1119, 544, 1321, 636, 878, 381, 1960, 1023, 783, 895, 1358, 300,
    797, 246, 1697, 1733, 829, 1956, 1177, 1666, 1557, 731, 1673, 333, 1357, 66, 355, 469, 425,
    572, 1013, 702, 128, 1297, 928, 357, 864, 1479, 116, 499, 1947, 1667, 314, 1457, 1657, 1054,
    480, 929, 558, 179, 1915, 484, 487, 476, 147, 1076, 2036, 1217, 1542, 1311, 382, 1193, 886, 70,
    924, 706, 471, 1773, 74, 567, 1114, 87, 1514, 988, 1481, 1367, 233, 786, 109, 245, 261, 1121,
    849, 1052, 25, 1607, 2004, 1985, 473, 1270, 1332, 288, 3, 253, 530, 1521, 618, 217, 1696, 1102,
    751, 459, 168, 839, 695, 1803, 1603, 784, 2019, 1032, 832, 1942, 598, 296, 1586, 350, 1646,
    1260, 1243, 1981, 1082, 118, 614, 86, 81, 1672, 1402, 559, 37, 1707, 747, 416, 858, 1103, 1071,
    1780, 1903, 457, 1043, 1606, 1221, 896, 2042, 1230, 692, 326, 410, 1394, 719, 222, 885, 1796,
    1176, 1299, 1443, 536, 1426, 2018, 1461, 117, 905, 186, 1736, 529, 648, 1764, 935, 1116, 67,
    1499, 825, 1592, 48, 71, 1940, 830, 1968, 676, 549, 638, 467, 131, 310, 404, 1578, 139, 2030,
    1582, 589, 715, 2012, 1168, 2006, 1096, 1670, 1431, 1757, 231, 1296, 652, 36, 2000, 2034, 332,
    1228, 1761, 1127, 1282, 113, 1165, 1475, 1794, 148, 1613, 1419, 1768, 760, 336, 1627, 1771,
    1249, 412, 1031, 273, 1525, 873, 762, 1785, 1743, 1945, 1555, 813, 1925, 1953, 604, 800, 798,
    613, 1050, 561, 1464, 671, 374, 1837, 697, 1854, 1372, 366, 151, 1117, 1417, 938, 919, 1829,
    1365, 1961, 308, 1524, 1015, 409, 69, 134, 238, 1335, 975, 1892, 1779, 950, 402, 1946, 101,
    524, 1289, 195, 1643, 1383, 392, 407, 1170, 341, 1732, 1355, 1471, 1850, 574, 1236, 566, 354,
    1322, 1623, 225, 1611, 1250, 1430, 237, 584, 620, 1053, 1007, 328, 1596, 125, 510, 1580, 482,
    1552, 910, 1345, 1875, 2028, 1824, 1996, 1280, 1750, 1477, 749, 756, 164, 1094, 503, 843, 424,
    767, 275, 911, 869, 1329, 40, 102, 1198, 1704, 923, 1241, 801, 1239, 34, 1292, 204, 1434, 769,
    1509, 311, 2008, 1125, 26, 711, 181, 1151, 122, 1569, 31, 595, 337, 1709, 949, 1251, 446, 1058,
    1453, 1870, 1767, 1660, 1077, 1079, 664, 1776, 423, 1866, 1078, 282, 1791, 22, 1529, 1574,
    1793, 1408, 421, 79, 1989, 1410, 232, 622, 1848, 726, 1929, 759, 1333, 1711, 1813, 735, 2027,
    1201, 1588, 602, 1307, 1739, 1594, 16, 1585, 214, 690, 1224, 616, 617, 1693, 2025, 362, 901,
    1526, 1520, 65, 489, 837, 1706, 794, 465, 103, 28, 721, 1700, 1912, 1003, 2035, 937, 1445,
    1890, 765, 19, 1840, 1715, 1037, 1716, 1397, 1926, 1833, 1035, 865, 1629, 1313, 945, 391, 1197,
    687, 899, 1545, 1263, 1901, 1284, 1847, 737, 1705, 1182, 258, 1544, 635, 755, 1017, 1002, 198,
    15, 960, 178, 500, 576, 212, 1683, 1952, 1028, 1800, 244, 494, 607, 1014, 1265, 264, 1566,
    1797, 535, 42, 1109, 1184, 1897, 1905, 778, 2039, 1070, 303, 1648, 713, 2033, 94, 758, 1259,
    855, 1351, 587, 384, 274, 732, 1362, 677, 1735, 1762, 1577, 49, 1818, 1658, 1503, 1219, 345,
    887, 1556, 1232, 556, 1662, 1916, 1992, 1361, 775, 603, 1513, 62, 823, 875, 1135, 714, 619, 80,
    250, 176, 466, 2021, 1725, 970, 882, 971, 659, 543, 1984, 1973, 1147, 977, 667, 1622, 1936,
    605, 862, 673, 954, 693, 443, 294, 234, 804, 1494, 1564, 1366, 1207, 1806, 1949, 156, 583, 14,
    107, 1456, 203, 387, 1935, 541, 1390, 269, 563, 888, 364, 1756, 1001, 597, 1467, 1635, 1689,
    1209, 868, 1225, 1406, 1979, 902, 360, 1040, 1066, 634, 1025, 564, 1536, 1894, 1562, 908, 399,
    838, 155, 1086, 1483, 1865, 1869, 2020, 907, 1884, 1516, 1275, 1262, 1158, 1120, 351, 1145, 20,
    550, 1059, 1340, 97, 1237, 2024, 1254, 1314, 1532, 822, 1420, 1528, 1449, 1139, 1089, 1630,
    1226, 593, 1919, 555, 682, 1159, 1294, 171, 315, 1064, 999, 727, 506, 1315, 1393, 313, 1914,
    1987, 729, 961, 1213, 1200, 1852, 734, 986, 1547, 1972, 1095, 591, 1649, 1033, 1359, 1983, 743,
    1156, 814, 1864, 1714, 709, 347, 1247, 1781, 1713, 1252, 951, 380, 655, 1413, 338, 1140, 327,
    1543, 289, 57, 1814, 1789, 1194, 718, 793, 276, 791, 763, 1684, 992, 883, 1330, 1741, 725,
    1138, 853, 857, 320, 1831, 916, 301, 1804, 1113, 485, 221, 1403, 1143, 240, 958, 781, 1305,
    1248, 408, 628, 1036, 764, 330, 478, 1216, 1871, 157, 1190, 270, 121, 845, 513, 623, 1, 1842,
    1906, 228, 1474, 1146, 1310, 694, 4, 1584, 1849, 1691, 1412, 1778, 1565, 1343, 578, 464, 373,
    376, 575, 444, 1857, 1512, 184, 140, 1931, 1819, 881, 23, 1878, 1266, 560, 1587, 1439, 1240,
    1734, 701, 349, 1160, 1368, 611, 1395, 449, 1572, 2022, 477, 689, 1719, 1910, 106, 1886, 1130,
    1344, 46, 653, 1339, 1211, 1692, 18, 1943, 344, 1899, 1510, 1373, 1466, 64, 442, 884, 854,
    1422, 191, 654, 1029, 378, 1392, 785, 539, 1189, 417, 562, 182, 1798, 662, 531, 779, 29, 268,
    1085, 651, 1598, 340, 7, 316, 1163, 1879, 668, 1154, 491, 177, 1326, 1593, 1473, 946, 1447,
    1133, 1654, 455, 1337, 60, 894, 1486, 1676, 1858, 352, 547, 47, 279, 267, 1144, 1427, 1652,
    967, 1118, 1982, 1718, 2041, 2023, 1416, 1183, 1939, 728, 774, 89, 683, 1492, 163, 1155, 379,
    661, 432, 632, 1868, 1904, 1264, 115, 1255, 1306, 50, 1235, 585, 1132, 104, 438, 861, 458,
    1862, 674, 507, 579, 1885, 1541, 741, 1238, 174, 2029, 199, 108, 736, 1644, 263, 704, 452, 325,
    406, 468, 272, 88, 235, 1974, 1069, 437, 1387, 2043, 877, 1495, 1405, 1424, 1549, 1726, 933,
    870, 307, 1131, 497, 418, 451, 367, 1364, 1810, 851, 1381, 1563, 1978, 75, 1625, 39, 942, 1256,
    1883, 969, 1046, 385, 1554, 1476, 852, 542, 1846, 1123, 1374, 1181, 1493, 1472, 1621, 739,
    1975, 982, 1668, 1257, 1334, 1595, 1807, 205, 361, 112, 1760, 1042, 1805, 183, 1895, 1107, 329,
    1551, 1881, 1765, 1752, 492, 912, 1101, 897, 1045, 712, 612, 343, 817, 1958, 1134, 799, 1959,
    663, 594, 2045, 658, 501, 209, 1924, 581, 819, 1619, 1293, 863, 201, 1354, 32, 1208, 1614,
    1717, 2005, 2002, 700, 1384, 2011, 1231, 283, 1411, 1836, 1148, 537, 1730, 1093, 187, 1328,
    1451, 1775, 795, 570, 1876, 1267, 1863, 1933, 1889, 1400, 569, 447, 1913, 686, 511, 1438, 1285,
    1720, 1055, 450, 1022, 699, 522, 588, 703, 842, 1192, 1110, 82, 931, 1571, 1174, 1218, 1609,
    1157, 840, 84, 509, 757, 1388, 1161, 100, 1484, 1631, 1787, 38, 1576, 934, 1323, 1429, 1083,
    2031, 512, 606, 321, 1618, 1993, 207, 802, 2032, 1828, 771, 85, 398, 835, 545, 145, 519, 2014,
    681, 540, 1937, 680, 1827, 974, 1435, 1448, 265, 73, 904, 375, 1702, 1538, 1954, 1966, 1012,
    1817, 295, 573, 915,
];

impl UnquantisedNetwork {
    /// Convert a parameter file generated by bullet into a merged parameter set,
    /// for further processing or for resuming training in a more efficient format.
    #[expect(clippy::too_many_lines)]
    fn merge(&self) -> Box<MergedNetwork> {
        #![allow(clippy::similar_names)]

        let mut net = MergedNetwork::zeroed();
        let mut buckets = self.l0_weights.chunks_exact(12 * 64 * L0_OUT);
        let factoriser;
        let alternate_buffer;
        if UNQUANTISED_HAS_FACTORISER {
            factoriser = buckets.next().unwrap();
        } else {
            alternate_buffer = vec![0.0; 12 * 64 * L0_OUT];
            factoriser = &alternate_buffer;
        }
        for (src_bucket, tgt_bucket) in
            buckets.zip(net.l0_weights.chunks_exact_mut(12 * 64 * L0_OUT))
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
        let buckets = self.l0_weights.chunks_exact(12 * 64 * L0_OUT);

        for (bucket_idx, (src_bucket, tgt_bucket)) in buckets
            .zip(net.l0_weights.chunks_exact_mut(PSQT_FEATURES * L0_OUT))
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
        let src_buckets = self.l0_weights.chunks_exact(PSQT_FEATURES * L0_OUT);
        let tgt_buckets = net.l0_weights.chunks_exact_mut(PSQT_FEATURES * L0_OUT);
        for (src_bucket, tgt_bucket) in src_buckets.zip(tgt_buckets) {
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
                .chunks_exact_mut(8)
                .map(|a| a.try_into().unwrap())
                .collect();
            let mut biases: Vec<&mut PermChunk<i16>> = net
                .l0_biases
                .chunks_exact_mut(8)
                .map(|a| a.try_into().unwrap())
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
                net.l0_aux.as_chunks_mut::<8>().0.iter_mut().collect();
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

fn repermute_l0_psqt_bucket(tgt_bucket: &mut [i16], unsorted: &[i16]) {
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
