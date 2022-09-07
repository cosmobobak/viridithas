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
    pub fn next(&mut self) -> Option<MoveListEntry> {
        if self.index == self.count {
            return None;
        }
        let mut best_score = -i32::MAX;
        let mut best_num = self.index;

        for index in self.index..self.count {
            let score = unsafe { self.moves.get_unchecked(index).score };
            if score > best_score {
                best_score = score;
                best_num = index;
            }
        }

        debug_assert!(self.index < self.count);
        debug_assert!(best_num < self.count);
        debug_assert!(best_num >= self.index);

        let &m = unsafe { self.moves.get_unchecked(best_num) };

        unsafe {
            *self.moves.get_unchecked_mut(best_num) = *self.moves.get_unchecked(self.index);
        }

        self.index += 1;

        Some(m)
    }
}
