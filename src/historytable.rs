use std::array;

use crate::{
    definitions::{depth::Depth, Square, BOARD_N_SQUARES},
    piece::Piece,
};

const AGEING_DIVISOR: i16 = 2;

const fn history_bonus(depth: Depth) -> i32 {
    #![allow(clippy::cast_possible_truncation)]
    // i'm genuinely unsure if this is the same way ethereal does it, operator precedence is a trip.
    let depth = depth.round();
    if depth > 13 {
        32
    } else {
        16 * depth * depth + 128 * max!(depth - 1, 0)
    }
}

pub fn update_history<const IS_GOOD: bool>(val: &mut i16, depth: Depth) {
    #![allow(clippy::cast_possible_truncation)]
    const HISTORY_DIVISOR: i16 = i16::MAX / 2;
    let delta = if IS_GOOD { history_bonus(depth) } else { -history_bonus(depth) };
    *val += delta as i16 - (i32::from(*val) * delta.abs() / i32::from(HISTORY_DIVISOR)) as i16;
}

#[derive(Clone)]
pub struct HistoryTable {
    table: [[i16; BOARD_N_SQUARES]; 12],
}

impl HistoryTable {
    pub const fn new() -> Self {
        Self { table: [[0; BOARD_N_SQUARES]; 12] }
    }

    pub fn clear(&mut self) {
        if self.table.is_empty() {
            self.table = [[0; BOARD_N_SQUARES]; 12];
        } else {
            self.table.iter_mut().flatten().for_each(|x| *x = 0);
        }
    }

    pub fn age_entries(&mut self) {
        assert!(!self.table.is_empty());
        self.table.iter_mut().flatten().for_each(|x| *x /= AGEING_DIVISOR);
    }

    pub const fn get(&self, piece: Piece, sq: Square) -> i16 {
        let pt = piece.hist_table_offset();
        self.table[pt][sq.index()]
    }

    pub fn get_mut(&mut self, piece: Piece, sq: Square) -> &mut i16 {
        let pt = piece.hist_table_offset();
        &mut self.table[pt][sq.index()]
    }
}

#[derive(Clone)]
pub struct DoubleHistoryTable {
    table: [[Box<HistoryTable>; BOARD_N_SQUARES]; 12],
}

impl DoubleHistoryTable {
    pub fn new() -> Self {
        Self { table: array::from_fn(|_| array::from_fn(|_| Box::new(HistoryTable::new()))) }
    }

    pub fn clear(&mut self) {
        self.table.iter_mut().flatten().for_each(|x| x.clear());
    }

    pub fn age_entries(&mut self) {
        assert!(!self.table.is_empty());
        self.table.iter_mut().flatten().for_each(|x| x.age_entries());
    }

    pub const fn get(&self, piece: Piece, sq: Square) -> &HistoryTable {
        let pt = piece.hist_table_offset();
        &self.table[pt][sq.index()]
    }

    pub fn get_mut(&mut self, piece: Piece, sq: Square) -> &mut HistoryTable {
        let pt = piece.hist_table_offset();
        &mut self.table[pt][sq.index()]
    }
}
