use crate::chessmove::Move;

use super::{MAX_POSITION_MOVES, TT_MOVE_SCORE};

use super::MoveListEntry;

pub struct MovePicker<'a> {
    movelist: &'a mut [MoveListEntry; MAX_POSITION_MOVES],
    count: usize,
    index: usize,
    skip_ordering: bool,
}

impl<'a> MovePicker<'a> {
    pub fn new(moves: &'a mut [MoveListEntry; MAX_POSITION_MOVES], count: usize) -> MovePicker<'a> {
        Self { movelist: moves, count, index: 0, skip_ordering: false }
    }

    pub fn moves_made(&self) -> &[MoveListEntry] {
        &self.movelist[..self.index]
    }

    pub fn skip_ordering(&mut self) {
        self.skip_ordering = true;
    }

    pub fn score_by(&mut self, pre_ordered: &[(Move, u64)]) {
        #![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        self.movelist.iter_mut().for_each(|m| {
            if m.score == TT_MOVE_SCORE {
                return;
            }
            let score = pre_ordered
                .iter()
                .position(|p| p.0 == m.entry)
                .map_or(-1_000_000, |idx| 256 - idx as i32);
            m.score = score;
        });
    }
}

impl MovePicker<'_> {
    /// Select the next move to try. Executes one iteration of partial insertion sort.
    pub fn next(&mut self) -> Option<MoveListEntry> {
        // If we have already tried all moves, return None.
        if self.index == self.count {
            return None;
        } else if self.skip_ordering {
            // If we are skipping ordering, just return the next move.
            let &m = unsafe { self.movelist.get_unchecked(self.index) };
            self.index += 1;
            return Some(m);
        }

        // SAFETY: self.index is always in bounds.
        let mut best_score = unsafe { self.movelist.get_unchecked(self.index).score };
        let mut best_num = self.index;

        // find the best move in the unsorted portion of the movelist.
        for index in self.index + 1..self.count {
            // SAFETY: self.count is always less than 256, and self.index is always in bounds.
            let score = unsafe { self.movelist.get_unchecked(index).score };
            if score > best_score {
                best_score = score;
                best_num = index;
            }
        }

        debug_assert!(self.index < self.count);
        debug_assert!(best_num < self.count);
        debug_assert!(best_num >= self.index);

        // SAFETY: best_num is drawn from self.index..self.count, which is always in bounds.
        let &m = unsafe { self.movelist.get_unchecked(best_num) };

        // swap the best move with the first unsorted move.
        // SAFETY: best_num is drawn from self.index..self.count, which is always in bounds.
        // and self.index is always in bounds.
        unsafe {
            *self.movelist.get_unchecked_mut(best_num) = *self.movelist.get_unchecked(self.index);
        }

        self.index += 1;

        Some(m)
    }
}
