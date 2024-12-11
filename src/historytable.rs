use crate::{
    chessmove::Move,
    piece::{Colour, Piece, PieceType},
    util::{Square, BOARD_N_SQUARES},
};

const AGEING_DIVISOR: i16 = 2;

fn history_bonus(depth: i32) -> i32 {
    i32::min(200 * depth, 1600)
}

pub const MAX_HISTORY: i16 = i16::MAX / 2;
pub const CORRECTION_HISTORY_SIZE: usize = 16_384;
pub const CORRECTION_HISTORY_GRAIN: i32 = 256;
pub const CORRECTION_HISTORY_WEIGHT_SCALE: i32 = 256;
pub const CORRECTION_HISTORY_MAX: i32 = CORRECTION_HISTORY_GRAIN * 32;

pub fn update_history(val: &mut i16, depth: i32, is_good: bool) {
    #![allow(clippy::cast_possible_truncation)]
    const MAX_HISTORY: i32 = crate::historytable::MAX_HISTORY as i32;
    let delta = if is_good {
        history_bonus(depth)
    } else {
        -history_bonus(depth)
    };
    let curr = i32::from(*val);
    *val += delta as i16 - (curr * delta.abs() / MAX_HISTORY) as i16;
}

#[derive(Clone, Copy, Default)]
pub struct AdaptiveHistoryValue {
    score: i32,
    momentum: i32,
    variance: i32,
}

impl AdaptiveHistoryValue {
    const INV_BETA_1: i32 = 9;
    const INV_BETA_2: i32 = 999;

    pub const ZERO: Self = Self {
        score: 0,
        momentum: 0,
        variance: 0,
    };

    pub fn update_history(&mut self, depth: i32, is_good: bool) {
        #![allow(clippy::cast_possible_truncation)]
        const MAX_HISTORY: i32 = crate::historytable::MAX_HISTORY as i32;
        let delta = if is_good {
            history_bonus(depth)
        } else {
            -history_bonus(depth)
        };

        self.momentum = (self.momentum * Self::INV_BETA_1 + delta) / 10;
        self.variance = (self.variance * Self::INV_BETA_2 + delta * delta) / 1000;
        let abs_delta = delta.abs();
        let update = (delta * self.momentum)
            // stupid cast
            .checked_div(f64::from(self.variance).sqrt() as i32)
            .unwrap_or(delta * self.momentum)
            .clamp(-abs_delta * 16, abs_delta * 16);
        self.score += update - (self.score * update.abs() / MAX_HISTORY);
        debug_assert!(TryInto::<i16>::try_into(self.score).is_ok());
    }
}

#[repr(transparent)]
pub struct AdaptiveHistoryTable {
    table: [[AdaptiveHistoryValue; BOARD_N_SQUARES]; 12],
}

impl AdaptiveHistoryTable {
    pub const fn new() -> Self {
        Self {
            table: [[AdaptiveHistoryValue::ZERO; BOARD_N_SQUARES]; 12],
        }
    }

    pub fn clear(&mut self) {
        self.table
            .iter_mut()
            .flatten()
            .for_each(|x| *x = AdaptiveHistoryValue::ZERO);
    }

    pub fn get(&self, piece: Piece, sq: Square) -> AdaptiveHistoryValue {
        self.table[piece][sq]
    }

    pub fn get_mut(&mut self, piece: Piece, sq: Square) -> &mut AdaptiveHistoryValue {
        &mut self.table[piece][sq]
    }
}

#[repr(transparent)]
pub struct HistoryTable {
    table: [[i16; BOARD_N_SQUARES]; 12],
}

impl HistoryTable {
    pub fn clear(&mut self) {
        self.table.iter_mut().flatten().for_each(|x| *x = 0);
    }

    pub fn age_entries(&mut self) {
        self.table
            .iter_mut()
            .flatten()
            .for_each(|x| *x /= AGEING_DIVISOR);
    }

    pub fn get(&self, piece: Piece, sq: Square) -> i16 {
        self.table[piece][sq]
    }

    pub fn get_mut(&mut self, piece: Piece, sq: Square) -> &mut i16 {
        &mut self.table[piece][sq]
    }
}

#[repr(transparent)]
pub struct ThreatsHistoryTable {
    table: [[AdaptiveHistoryTable; 2]; 2],
}

impl ThreatsHistoryTable {
    pub const fn new() -> Self {
        const ELEM: AdaptiveHistoryTable = AdaptiveHistoryTable::new();
        const SLICE: [AdaptiveHistoryTable; 2] = [ELEM; 2];
        const ARRAY: [[AdaptiveHistoryTable; 2]; 2] = [SLICE; 2];
        Self { table: ARRAY }
    }

    pub fn clear(&mut self) {
        self.table
            .iter_mut()
            .flatten()
            .for_each(AdaptiveHistoryTable::clear);
    }

    #[allow(clippy::cast_possible_truncation)]
    pub fn get(&self, piece: Piece, sq: Square, threat_from: bool, threat_to: bool) -> i16 {
        self.table[usize::from(threat_from)][usize::from(threat_to)]
            .get(piece, sq)
            .score as i16
    }

    pub fn get_mut(
        &mut self,
        piece: Piece,
        sq: Square,
        threat_from: bool,
        threat_to: bool,
    ) -> &mut AdaptiveHistoryValue {
        self.table[usize::from(threat_from)][usize::from(threat_to)].get_mut(piece, sq)
    }
}

#[repr(transparent)]
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
        debug_assert!(!self.table.is_empty());
        self.table.iter_mut().for_each(HistoryTable::age_entries);
    }

    pub fn get(&self, piece: Piece, sq: Square, capture: PieceType) -> i16 {
        self.table[capture].get(piece, sq)
    }

    pub fn get_mut(&mut self, piece: Piece, sq: Square, capture: PieceType) -> &mut i16 {
        self.table[capture].get_mut(piece, sq)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContHistIndex {
    pub piece: Piece,
    pub square: Square,
}

#[repr(transparent)]
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
        self.table
            .iter_mut()
            .flatten()
            .for_each(HistoryTable::clear);
    }

    pub fn age_entries(&mut self) {
        debug_assert!(!self.table.is_empty());
        self.table
            .iter_mut()
            .flatten()
            .for_each(HistoryTable::age_entries);
    }

    pub fn get_index_mut(&mut self, index: ContHistIndex) -> &mut HistoryTable {
        &mut self.table[index.piece][index.square]
    }

    pub fn get_index(&self, index: ContHistIndex) -> &HistoryTable {
        &self.table[index.piece][index.square]
    }
}

pub struct MoveTable {
    table: Box<[[Option<Move>; BOARD_N_SQUARES]; 12]>,
}

impl MoveTable {
    pub fn new() -> Self {
        Self {
            table: Box::new([[None; BOARD_N_SQUARES]; 12]),
        }
    }

    pub fn clear(&mut self) {
        self.table.iter_mut().for_each(|t| t.fill(None));
    }

    pub fn add(&mut self, piece: Piece, sq: Square, m: Move) {
        self.table[piece][sq] = Some(m);
    }

    pub fn get(&self, piece: Piece, sq: Square) -> Option<Move> {
        self.table[piece][sq]
    }
}

#[repr(transparent)]
pub struct CorrectionHistoryTable {
    table: [[i32; CORRECTION_HISTORY_SIZE]; 2],
}

impl CorrectionHistoryTable {
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
        self.table.iter_mut().for_each(|t| t.fill(0));
    }

    #[allow(clippy::cast_possible_truncation)]
    pub fn get(&self, side: Colour, key: u64) -> i64 {
        i64::from(self.table[side][(key % CORRECTION_HISTORY_SIZE as u64) as usize])
    }

    #[allow(clippy::cast_possible_truncation)]
    pub fn get_mut(&mut self, side: Colour, key: u64) -> &mut i32 {
        &mut self.table[side][(key % CORRECTION_HISTORY_SIZE as u64) as usize]
    }
}
