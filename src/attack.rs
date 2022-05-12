use crate::definitions::{WB, WR, WQ, BQ, BR, BB, WN, WK, BK, BN};

pub static N_DIRS: [i8; 8] = [-8, -19, -21, -12, 8, 19, 21, 12];
pub static B_DIR: [i8; 4] = [-9, -11, 11, 9];
pub static R_DIR: [i8; 4] = [-1, -10, 1, 10];
pub static Q_DIR: [i8; 8] = [-1, -10, 1, 10, -9, -11, 11, 9];
pub static K_DIRS: [i8; 8] = [-1, -10, 1, 10, -9, -11, 11, 9];

pub static IS_KNIGHT: [bool; 13] = [
    false, false, true, false, false, false, false, false, true, false, false, false, false,
];
pub static IS_KING: [bool; 13] = [
    false, false, false, false, false, false, true, false, false, false, false, false, true,
];
pub static IS_ROOKQUEEN: [bool; 13] = [
    false, false, false, false, true, true, false, false, false, false, true, true, false,
];
pub static IS_BISHOPQUEEN: [bool; 13] = [
    false, false, false, true, false, true, false, false, false, true, false, true, false,
];
pub static IS_SLIDER: [bool; 13] = [
    false, false, false, true, true, true, false, false, false, true, true, true, false,
];

pub static WHITE_SLIDERS: [u8; 3] = [WB, WR, WQ];

pub static BLACK_SLIDERS: [u8; 3] = [BB, BR, BQ];

pub static WHITE_JUMPERS: [u8; 2] = [WN, WK];

pub static BLACK_JUMPERS: [u8; 2] = [BN, BK];
