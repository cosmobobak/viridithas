use std::ops::{Deref, DerefMut};

use crate::{
    chess::{
        piece::{Colour, Piece},
        types::Square,
    },
    search::parameters::Config,
    util::BOARD_N_SQUARES,
};

pub fn main_history_bonus(conf: &Config, depth: i32) -> i32 {
    i32::min(
        conf.main_history_bonus_mul * depth + conf.main_history_bonus_offset,
        conf.main_history_bonus_max,
    )
}
pub fn main_history_malus(conf: &Config, depth: i32) -> i32 {
    i32::min(
        conf.main_history_malus_mul * depth + conf.main_history_malus_offset,
        conf.main_history_malus_max,
    )
}
pub fn cont1_history_bonus(conf: &Config, depth: i32) -> i32 {
    i32::min(
        conf.cont1_history_bonus_mul * depth + conf.cont1_history_bonus_offset,
        conf.cont1_history_bonus_max,
    )
}
pub fn cont1_history_malus(conf: &Config, depth: i32) -> i32 {
    i32::min(
        conf.cont1_history_malus_mul * depth + conf.cont1_history_malus_offset,
        conf.cont1_history_malus_max,
    )
}
pub fn cont2_history_bonus(conf: &Config, depth: i32) -> i32 {
    i32::min(
        conf.cont2_history_bonus_mul * depth + conf.cont2_history_bonus_offset,
        conf.cont2_history_bonus_max,
    )
}
pub fn cont2_history_malus(conf: &Config, depth: i32) -> i32 {
    i32::min(
        conf.cont2_history_malus_mul * depth + conf.cont2_history_malus_offset,
        conf.cont2_history_malus_max,
    )
}
pub fn tactical_history_bonus(conf: &Config, depth: i32) -> i32 {
    i32::min(
        conf.tactical_history_bonus_mul * depth + conf.tactical_history_bonus_offset,
        conf.tactical_history_bonus_max,
    )
}
pub fn tactical_history_malus(conf: &Config, depth: i32) -> i32 {
    i32::min(
        conf.tactical_history_malus_mul * depth + conf.tactical_history_malus_offset,
        conf.tactical_history_malus_max,
    )
}

pub fn cont_history_bonus(conf: &Config, depth: i32, index: usize) -> i32 {
    match index {
        0 => cont1_history_bonus(conf, depth),
        1 => cont2_history_bonus(conf, depth),
        _ => unreachable!(),
    }
}
pub fn cont_history_malus(conf: &Config, depth: i32, index: usize) -> i32 {
    match index {
        0 => cont1_history_malus(conf, depth),
        1 => cont2_history_malus(conf, depth),
        _ => unreachable!(),
    }
}

pub const MAX_HISTORY: i32 = i16::MAX as i32 / 2;
pub const CORRECTION_HISTORY_SIZE: usize = 16_384;
pub const CORRECTION_HISTORY_MAX: i32 = 1024;

pub fn update_history(val: &mut i16, delta: i32) {
    gravity_update::<MAX_HISTORY>(val, i32::from(*val), delta);
}

pub fn update_correction(val: &mut i16, curr: i32, delta: i32) {
    gravity_update::<CORRECTION_HISTORY_MAX>(val, curr, delta);
}

fn gravity_update<const MAX: i32>(val: &mut i16, curr: i32, delta: i32) {
    #![allow(clippy::cast_possible_truncation)]
    *val += delta as i16 - (curr * delta.abs() / MAX) as i16;
}

#[repr(transparent)]
pub struct HistoryTable {
    table: [[i16; BOARD_N_SQUARES]; 12],
}

impl HistoryTable {
    pub const fn new() -> Self {
        Self {
            table: [[0; BOARD_N_SQUARES]; 12],
        }
    }

    pub fn clear(&mut self) {
        if self.table.is_empty() {
            self.table = [[0; BOARD_N_SQUARES]; 12];
        } else {
            self.table.iter_mut().flatten().for_each(|x| *x = 0);
        }
    }

    pub fn get_mut(&mut self, piece: Piece, sq: Square) -> &mut i16 {
        &mut self[piece][sq]
    }
}

impl Deref for HistoryTable {
    type Target = [[i16; BOARD_N_SQUARES]; 12];

    fn deref(&self) -> &Self::Target {
        &self.table
    }
}

impl DerefMut for HistoryTable {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.table
    }
}

#[repr(transparent)]
pub struct ThreatsHistoryTable {
    table: [[HistoryTable; 2]; 2],
}

impl ThreatsHistoryTable {
    pub const fn new() -> Self {
        const ELEM: HistoryTable = HistoryTable::new();
        const SLICE: [HistoryTable; 2] = [ELEM; 2];
        const ARRAY: [[HistoryTable; 2]; 2] = [SLICE; 2];
        Self { table: ARRAY }
    }

    pub fn clear(&mut self) {
        self.table
            .iter_mut()
            .flatten()
            .for_each(HistoryTable::clear);
    }

    pub fn get_mut(
        &mut self,
        piece: Piece,
        sq: Square,
        threat_from: bool,
        threat_to: bool,
    ) -> &mut i16 {
        &mut self.table[usize::from(threat_from)][usize::from(threat_to)][piece][sq]
    }
}

impl Deref for ThreatsHistoryTable {
    type Target = [[HistoryTable; 2]; 2];

    fn deref(&self) -> &Self::Target {
        &self.table
    }
}

impl DerefMut for ThreatsHistoryTable {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.table
    }
}

#[repr(transparent)]
pub struct CaptureHistoryTable {
    table: [[HistoryTable; 6]; 2],
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
        self.table
            .iter_mut()
            .flatten()
            .for_each(HistoryTable::clear);
    }
}

impl Deref for CaptureHistoryTable {
    type Target = [[HistoryTable; 6]; 2];

    fn deref(&self) -> &Self::Target {
        &self.table
    }
}

impl DerefMut for CaptureHistoryTable {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.table
    }
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
}

impl Deref for DoubleHistoryTable {
    type Target = [[HistoryTable; BOARD_N_SQUARES]; 12];

    fn deref(&self) -> &Self::Target {
        &self.table
    }
}

impl DerefMut for DoubleHistoryTable {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.table
    }
}

#[repr(transparent)]
pub struct CorrectionHistoryTable {
    table: [[i16; 2]; CORRECTION_HISTORY_SIZE],
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
        i64::from(self.table[(key % CORRECTION_HISTORY_SIZE as u64) as usize][side])
    }

    #[allow(clippy::cast_possible_truncation)]
    pub fn get_mut(&mut self, side: Colour, key: u64) -> &mut i16 {
        &mut self.table[(key % CORRECTION_HISTORY_SIZE as u64) as usize][side]
    }
}

impl Deref for CorrectionHistoryTable {
    type Target = [[i16; 2]; CORRECTION_HISTORY_SIZE];

    fn deref(&self) -> &Self::Target {
        &self.table
    }
}

impl DerefMut for CorrectionHistoryTable {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.table
    }
}

#[repr(transparent)]
pub struct ContinuationCorrectionHistoryTable {
    table: [[[[[i16; 2]; 6]; 64]; 6]; 64],
}

impl ContinuationCorrectionHistoryTable {
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
            .flatten()
            .flatten()
            .for_each(|t| t.fill(0));
    }
}

impl Deref for ContinuationCorrectionHistoryTable {
    type Target = [[[[[i16; 2]; 6]; 64]; 6]; 64];

    fn deref(&self) -> &Self::Target {
        &self.table
    }
}

impl DerefMut for ContinuationCorrectionHistoryTable {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.table
    }
}
