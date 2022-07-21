use crate::{definitions::BOARD_N_SQUARES, validate::piece_valid};

const DO_COLOUR_DIFFERENTIATION: bool = true;

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
    debug_assert!(piece_valid(piece));
    if DO_COLOUR_DIFFERENTIATION {
        coloured_piece_index(piece)
    } else {
        uncoloured_piece_index(piece)
    }
}

#[derive(Default)]
pub struct HistoryTable {
    table: Box<[[i32; BOARD_N_SQUARES]]>
}

impl HistoryTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        if self.table.len() == 0 {
            self.table = vec![[0; BOARD_N_SQUARES]; pslots()].into_boxed_slice();
        } else {
            self.table
                .iter_mut()
                .flatten()
                .for_each(|x| *x = 0);
        }
    }

    #[allow(clippy::only_used_in_recursion)] // wtf??
    pub fn add(&mut self, piece: u8, sq: u8, score: i32) {
        let pt = piece_index(piece);
        self.table[pt as usize][sq as usize] += score;
    }

    pub const fn get(&self, piece: u8, sq: u8) -> i32 {
        let pt = piece_index(piece);
        self.table[pt as usize][sq as usize]
    }
}

#[derive(Default)]
pub struct DoubleHistoryTable {
    table: Box<[[[[i32; BOARD_N_SQUARES]; pslots()]; BOARD_N_SQUARES]]>
}

impl DoubleHistoryTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        if self.table.len() == 0 {
            self.table = vec![[[[0; BOARD_N_SQUARES]; pslots()]; BOARD_N_SQUARES]; pslots()].into_boxed_slice();
        } else {
            self.table
                .iter_mut()
                .flatten()
                .flatten()
                .flatten()
                .for_each(|x| *x = 0);
        }
    }

    pub fn add(&mut self, piece_1: u8, sq1: u8, piece_2: u8, sq2: u8, score: i32) {
        let pt_1 = piece_index(piece_1);
        let pt_2 = piece_index(piece_2);
        self.table[pt_1 as usize][sq1 as usize][pt_2 as usize][sq2 as usize] += score;
    }

    pub const fn get(&self, piece_1: u8, sq1: u8, piece_2: u8, sq2: u8) -> i32 {
        let pt_1 = piece_index(piece_1);
        let pt_2 = piece_index(piece_2);
        self.table[pt_1 as usize][sq1 as usize][pt_2 as usize][sq2 as usize]
    }
}