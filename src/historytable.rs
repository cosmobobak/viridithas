use crate::{
    chessmove::Move,
    definitions::{depth::Depth, BOARD_N_SQUARES, PIECE_EMPTY, Square},
    validate::piece_valid,
};

const DO_COLOUR_DIFFERENTIATION: bool = true;
const AGEING_DIVISOR: i32 = 2;

const fn pslots() -> usize {
    if DO_COLOUR_DIFFERENTIATION {
        12
    } else {
        6
    }
}

const fn uncoloured_piece_index(piece: u8) -> u8 {
    (piece - 1) % 6
}

const fn coloured_piece_index(piece: u8) -> u8 {
    piece - 1
}

const fn piece_index(piece: u8) -> u8 {
    debug_assert!(piece_valid(piece) && piece != PIECE_EMPTY);
    if DO_COLOUR_DIFFERENTIATION {
        coloured_piece_index(piece)
    } else {
        uncoloured_piece_index(piece)
    }
}

const fn history_bonus(depth: Depth) -> i32 {
    depth.squared() + depth.round()
}

pub fn update_history<const IS_GOOD: bool>(val: &mut i32, depth: Depth) {
    const HISTORY_DIVISOR: i32 = i16::MAX as i32;
    let delta = if IS_GOOD { history_bonus(depth) } else { -history_bonus(depth) };
    *val += delta - (*val * delta.abs() / HISTORY_DIVISOR);
}

#[derive(Default)]
pub struct HistoryTable {
    table: Box<[[i32; BOARD_N_SQUARES]]>,
}

impl HistoryTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        if self.table.is_empty() {
            self.table = vec![[0; BOARD_N_SQUARES]; pslots()].into_boxed_slice();
        } else {
            self.table.iter_mut().flatten().for_each(|x| *x = 0);
        }
    }

    pub fn age_entries(&mut self) {
        assert!(!self.table.is_empty());
        self.table.iter_mut().flatten().for_each(|x| *x /= AGEING_DIVISOR);
    }

    pub const fn get(&self, piece: u8, sq: Square) -> i32 {
        let pt = piece_index(piece);
        self.table[pt as usize][sq.index()]
    }

    pub fn get_mut(&mut self, piece: u8, sq: Square) -> &mut i32 {
        let pt = piece_index(piece);
        &mut self.table[pt as usize][sq.index()]
    }

    #[allow(dead_code)]
    pub fn print_stats(&self) {
        #![allow(clippy::cast_precision_loss)]
        let sum = self.table.iter().flatten().map(|x| i64::from(*x)).sum::<i64>();
        let mean = sum as f64 / (BOARD_N_SQUARES as f64 * pslots() as f64);
        let stdev = self
            .table
            .iter()
            .flatten()
            .map(|x| i64::from(*x))
            .map(|x| (x as f64 - mean).powi(2))
            .sum::<f64>()
            .sqrt()
            / (BOARD_N_SQUARES as f64 * pslots() as f64);
        println!("mean: {}", mean);
        println!("stdev: {}", stdev);
        println!("max: {}", self.table.iter().flatten().copied().max().unwrap());
        let nonzero = self.table.iter().flatten().copied().filter(|x| *x != 0).collect::<Vec<_>>();
        println!("nonzero: {}", nonzero.len());
        let nz_mean =
            nonzero.iter().map(|x| i64::from(*x)).sum::<i64>() as f64 / (nonzero.len() as f64);
        let nz_stdev = nonzero
            .iter()
            .map(|x| i64::from(*x))
            .map(|x| (x as f64 - nz_mean).powi(2))
            .sum::<f64>()
            .sqrt()
            / (nonzero.len() as f64);
        println!("nz mean: {}", nz_mean);
        println!("nz stdev: {}", nz_stdev);
    }
}

#[derive(Default)]
pub struct DoubleHistoryTable {
    table: Vec<i32>,
}

impl DoubleHistoryTable {
    const I1: usize = BOARD_N_SQUARES * pslots() * BOARD_N_SQUARES;
    const I2: usize = BOARD_N_SQUARES * pslots();
    const I3: usize = BOARD_N_SQUARES;

    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        if self.table.is_empty() {
            self.table.resize(BOARD_N_SQUARES * pslots() * BOARD_N_SQUARES * pslots(), 0);
        } else {
            self.table.fill(0);
        }
    }

    pub fn age_entries(&mut self) {
        assert!(!self.table.is_empty());
        self.table.iter_mut().for_each(|x| *x /= AGEING_DIVISOR);
    }

    pub fn get(&self, piece_1: u8, sq1: Square, piece_2: u8, sq2: Square) -> i32 {
        let pt_1 = piece_index(piece_1) as usize;
        let pt_2 = piece_index(piece_2) as usize;
        let sq1 = sq1.index();
        let sq2 = sq2.index();
        let idx = pt_1 * Self::I1 + pt_2 * Self::I2 + sq1 * Self::I3 + sq2;
        self.table[idx]
    }

    pub fn get_mut(&mut self, piece_1: u8, sq1: Square, piece_2: u8, sq2: Square) -> &mut i32 {
        let pt_1 = piece_index(piece_1) as usize;
        let pt_2 = piece_index(piece_2) as usize;
        let sq1 = sq1.index();
        let sq2 = sq2.index();
        let idx = pt_1 * Self::I1 + pt_2 * Self::I2 + sq1 * Self::I3 + sq2;
        &mut self.table[idx]
    }

    #[allow(dead_code)]
    pub fn print_stats(&self) {
        #![allow(clippy::cast_precision_loss)]
        let sum = self.table.iter().map(|x| i64::from(*x)).sum::<i64>();
        let mean = sum as f64 / (BOARD_N_SQUARES as f64 * pslots() as f64);
        let stdev = self
            .table
            .iter()
            .map(|x| i64::from(*x))
            .map(|x| (x as f64 - mean).powi(2))
            .sum::<f64>()
            .sqrt()
            / (BOARD_N_SQUARES as f64 * pslots() as f64);
        println!("mean: {}", mean);
        println!("stdev: {}", stdev);
        println!("max: {}", self.table.iter().copied().max().unwrap());
        let nonzero = self.table.iter().copied().filter(|x| *x != 0).collect::<Vec<_>>();
        println!("nonzero: {}", nonzero.len());
        let nz_mean =
            nonzero.iter().map(|x| i64::from(*x)).sum::<i64>() as f64 / (nonzero.len() as f64);
        let nz_stdev = nonzero
            .iter()
            .map(|x| i64::from(*x))
            .map(|x| (x as f64 - nz_mean).powi(2))
            .sum::<f64>()
            .sqrt()
            / (nonzero.len() as f64);
        println!("nz mean: {}", nz_mean);
        println!("nz stdev: {}", nz_stdev);
    }
}

pub struct MoveTable {
    table: Vec<Move>,
}

impl MoveTable {
    pub const fn new() -> Self {
        Self { table: Vec::new() }
    }

    pub fn clear(&mut self) {
        if self.table.is_empty() {
            self.table.resize(BOARD_N_SQUARES * pslots(), Move::NULL);
        } else {
            self.table.fill(Move::NULL);
        }
    }

    pub fn add(&mut self, piece: u8, sq: Square, move_: Move) {
        let pt = piece_index(piece) as usize;
        let sq = sq.index();
        self.table[pt * BOARD_N_SQUARES + sq] = move_;
    }

    pub fn get(&self, piece: u8, sq: Square) -> Move {
        let pt = piece_index(piece) as usize;
        let sq = sq.index();
        self.table[pt * BOARD_N_SQUARES + sq]
    }
}
