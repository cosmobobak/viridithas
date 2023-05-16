use crate::{
    chessmove::Move,
    definitions::{depth::Depth, Square, BOARD_N_SQUARES},
    piece::{Piece, PieceType},
};

const AGEING_DIVISOR: i16 = 2;

const fn history_bonus(depth: Depth) -> i32 {
    let depth = depth.round();
    if depth > 13 {
        32
    } else {
        16 * depth * depth + 128 * max!(depth - 1, 0)
    }
}

pub const MAX_HISTORY: i16 = i16::MAX / 2;

pub fn update_history(val: &mut i16, depth: Depth, is_good: bool) {
    #![allow(clippy::cast_possible_truncation)]
    let delta = if is_good { history_bonus(depth) } else { -history_bonus(depth) };
    *val += delta as i16 - (i32::from(*val) * delta.abs() / i32::from(MAX_HISTORY)) as i16;
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
pub struct CaptureHistoryTable {
    table: [HistoryTable; 6],
}

impl CaptureHistoryTable {
    pub fn boxed() -> Box<Self> {
        #![allow(clippy::cast_ptr_alignment)]
        // SAFETY: we're allocating a zeroed block of memory, and then casting it to a Box<Self>
        // this is fine! because [[HistoryTable; BOARD_N_SQUARES]; 12] is just a bunch of i16s
        // at base, which are fine to zero-out.
        unsafe {
            let layout = std::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc_zeroed(layout);
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Box::from_raw(ptr.cast())
        }
    }

    pub fn clear(&mut self) {
        self.table.iter_mut().for_each(HistoryTable::clear);
    }

    pub fn age_entries(&mut self) {
        assert!(!self.table.is_empty());
        self.table.iter_mut().for_each(HistoryTable::age_entries);
    }

    pub const fn get(&self, piece: Piece, sq: Square, capture: PieceType) -> i16 {
        self.table[capture.index()].get(piece, sq)
    }

    pub fn get_mut(&mut self, piece: Piece, sq: Square, capture: PieceType) -> &mut i16 {
        self.table[capture.index()].get_mut(piece, sq)
    }
}

#[derive(Clone)]
pub struct DoubleHistoryTable {
    table: [[HistoryTable; BOARD_N_SQUARES]; 12],
}

impl DoubleHistoryTable {
    pub fn boxed() -> Box<Self> {
        #![allow(clippy::cast_ptr_alignment)]
        // SAFETY: we're allocating a zeroed block of memory, and then casting it to a Box<Self>
        // this is fine! because [[HistoryTable; BOARD_N_SQUARES]; 12] is just a bunch of i16s
        // at base, which are fine to zero-out.
        unsafe {
            let layout = std::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc_zeroed(layout);
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Box::from_raw(ptr.cast())
        }
    }

    pub fn clear(&mut self) {
        self.table.iter_mut().flatten().for_each(HistoryTable::clear);
    }

    pub fn age_entries(&mut self) {
        assert!(!self.table.is_empty());
        self.table.iter_mut().flatten().for_each(HistoryTable::age_entries);
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

#[derive(Clone)]
pub struct MoveTable {
    table: Vec<Move>,
}

impl MoveTable {
    pub const fn new() -> Self {
        Self { table: Vec::new() }
    }

    pub fn clear(&mut self) {
        if self.table.is_empty() {
            self.table.resize(BOARD_N_SQUARES * 12, Move::NULL);
        } else {
            self.table.fill(Move::NULL);
        }
    }

    pub fn add(&mut self, piece: Piece, sq: Square, m: Move) {
        let pt = piece.hist_table_offset();
        let sq = sq.index();
        self.table[pt * BOARD_N_SQUARES + sq] = m;
    }

    pub fn get(&self, piece: Piece, sq: Square) -> Move {
        let pt = piece.hist_table_offset();
        let sq = sq.index();
        self.table[pt * BOARD_N_SQUARES + sq]
    }
}
