use super::MAX_POSITION_MOVES;

use super::MoveListEntry;

pub struct MovePicker {
    pub(crate) moves: [MoveListEntry; MAX_POSITION_MOVES],
    pub(crate) count: usize,
    pub(crate) index: usize,
}

impl MovePicker {
    pub fn moves_made(&self) -> &[MoveListEntry] {
        &self.moves[..self.index]
    }
}

impl MovePicker {
    /// Select the next move to try. Executes one iteration of partial insertion sort.
    pub fn next(&mut self) -> Option<&MoveListEntry> {
        // If we have already tried all moves, return None.
        if self.index == self.count {
            return None;
        }

        let mut best_score = unsafe { self.moves.get_unchecked(self.index).score };
        let mut best_num = self.index;

        // find the best move in the unsorted portion of the movelist.
        for index in self.index + 1..self.count {
            let score = unsafe { self.moves.get_unchecked(index).score };
            if score > best_score {
                best_score = score;
                best_num = index;
            }
        }

        debug_assert!(self.index < self.count);
        debug_assert!(best_num < self.count);
        debug_assert!(best_num >= self.index);

        // swap the best move with the first unsorted move.
        unsafe {
            *self.moves.get_unchecked_mut(best_num) = *self.moves.get_unchecked(self.index);
        }

        let m = unsafe { self.moves.get_unchecked(best_num) };

        self.index += 1;

        Some(m)
    }
}
