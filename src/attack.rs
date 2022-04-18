

pub static N_DIR: [isize; 8] = [-8, -19, -21, -12, 8, 19, 21, 12];
pub static R_DIR: [isize; 4] = [-1, -10, 1, 10];
pub static B_DIR: [isize; 4] = [-9, -11, 11, 9];
pub static K_DIR: [isize; 8] = [-1, -10, 1, 10, -9, -11, 11, 9];
pub static Q_DIR: [isize; 8] = [-1, -10, 1, 10, -9, -11, 11, 9];

pub static IS_KNIGHT: [bool; 13] = [false, false, true, false, false, false, false, false, true, false, false, false, false];
pub static IS_KING: [bool; 13] = [false, false, false, false, false, false, true, false, false, false, false, false, true];
pub static IS_ROOKQUEEN: [bool; 13] = [false, false, false, false, true, true, false, false, false, false, true, true, false];
pub static IS_BISHOPQUEEN: [bool; 13] = [false, false, false, true, false, true, false, false, false, true, false, true, false];

