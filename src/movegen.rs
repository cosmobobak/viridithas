use std::fmt::{Display, Formatter};

use crate::{
    board::Board,
    chessmove::Move,
    definitions::{Colour, Piece, Square120},
    lookups::{FILES_BOARD, RANKS_BOARD},
    validate::square_on_board,
};

const MAX_POSITION_MOVES: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MoveListEntry {
    pub entry: Move,
    pub score: i32,
}

// Consider using MaybeUninit when you're confident
// that you know how to not break everything.
// Alternatively, add a field to Board so we can save
// on allocating MoveLists.
// Third, struct-of-arrays is supposedly faster than
// array-of-structs. We should try that.
pub struct MoveList {
    moves: [MoveListEntry; MAX_POSITION_MOVES],
    count: usize,
}

impl MoveList {
    pub fn new() -> Self {
        const DEFAULT: MoveListEntry = MoveListEntry {
            entry: Move { data: 0 },
            score: 0,
        };
        Self {
            moves: [DEFAULT; MAX_POSITION_MOVES],
            count: 0,
        }
    }

    pub fn push(&mut self, m: Move, score: i32) {
        // woohoohoohoo this is bad
        debug_assert!(self.count < MAX_POSITION_MOVES);
        unsafe {
            *self.moves.get_unchecked_mut(self.count) = MoveListEntry { entry: m, score };
        }
        self.count += 1;
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn clear(&mut self) {
        self.count = 0;
    }

    pub fn iter(&self) -> impl Iterator<Item = &Move> {
        unsafe {
            self.moves
                .get_unchecked(..self.count)
                .iter()
                .map(|e| &e.entry)
        }
    }

    pub fn sort(&mut self) {
        // reversed, as we want to sort from highest to lowest
        unsafe {
            self.moves
                .get_unchecked_mut(..self.count)
                .sort_unstable_by(|a, b| b.score.cmp(&a.score));
        }
    }
}

impl Display for MoveList {
    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
        if self.count == 0 {
            return write!(f, "MoveList: (0) []");
        }
        writeln!(f, "MoveList: ({}) [", self.count)?;
        for m in &self.moves[0..self.count - 1] {
            writeln!(f, "  {} ${}, ", m.entry, m.score)?;
        }
        writeln!(
            f,
            "  {} ${}",
            self.moves[self.count - 1].entry,
            self.moves[self.count - 1].score
        )?;
        write!(f, "]")
    }
}

#[inline]
pub fn offset_square_offboard(offset_sq: isize) -> bool {
    debug_assert!((0..120).contains(&offset_sq));
    let idx: usize = unsafe { offset_sq.try_into().unwrap_unchecked() };
    let value = unsafe { *FILES_BOARD.get_unchecked(idx) };
    value == Square120::OffBoard as u8
}
