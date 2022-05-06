use crate::definitions::Piece;

pub static N_DIRS: [isize; 8] = [-8, -19, -21, -12, 8, 19, 21, 12];
pub static B_DIR: [isize; 4] = [-9, -11, 11, 9];
pub static R_DIR: [isize; 4] = [-1, -10, 1, 10];
pub static Q_DIR: [isize; 8] = [-1, -10, 1, 10, -9, -11, 11, 9];
pub static K_DIRS: [isize; 8] = [-1, -10, 1, 10, -9, -11, 11, 9];

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

pub static WHITE_SLIDERS: [u8; 3] = [Piece::WB as u8, Piece::WR as u8, Piece::WQ as u8];

pub static BLACK_SLIDERS: [u8; 3] = [Piece::BB as u8, Piece::BR as u8, Piece::BQ as u8];

pub static WHITE_JUMPERS: [u8; 2] = [Piece::WN as u8, Piece::WK as u8];

pub static BLACK_JUMPERS: [u8; 2] = [Piece::BN as u8, Piece::BK as u8];
