use std::{
    fmt::{Display, Formatter},
    ops::Index,
};

use crate::{chessmove::Move, definitions::Square120, lookups::FILES_BOARD};

pub trait MoveConsumer {
    fn push(&mut self, m: Move, score: i32);
    fn len(&self) -> usize;
}

pub struct MoveCounter {
    count: usize,
}

impl MoveCounter {
    pub const fn new() -> Self {
        Self { count: 0 }
    }
}

impl MoveConsumer for MoveCounter {
    fn push(&mut self, _m: Move, _score: i32) {
        self.count += 1;
    }
    fn len(&self) -> usize {
        self.count
    }
}

const MAX_POSITION_MOVES: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoveListEntry {
    pub entry: Move,
    pub score: i32,
}

pub struct MoveList {
    moves: [MoveListEntry; MAX_POSITION_MOVES],
    count: usize,
}

impl MoveList {
    pub const fn new() -> Self {
        const DEFAULT: MoveListEntry = MoveListEntry {
            entry: Move { data: 0 },
            score: 0,
        };
        Self {
            moves: [DEFAULT; MAX_POSITION_MOVES],
            count: 0,
        }
    }

    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn clear(&mut self) {
        self.count = 0;
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &Move> {
        unsafe {
            self.moves
                .get_unchecked(..self.count)
                .iter()
                .map(|e| &e.entry)
        }
    }

    #[inline]
    pub fn iter_with_scores(&self) -> impl Iterator<Item = &MoveListEntry> {
        unsafe { self.moves.get_unchecked(..self.count).iter() }
    }

    #[inline]
    pub fn sort(&mut self) {
        // reversed, as we want to sort from highest to lowest
        unsafe {
            self.moves
                .get_unchecked_mut(..self.count)
                .sort_unstable_by(|a, b| b.score.cmp(&a.score));
        }
    }

    pub fn lookup_by_move(&mut self, m: Move) -> Option<&mut MoveListEntry> {
        unsafe {
            self.moves
                .get_unchecked_mut(..self.count)
                .iter_mut()
                .find(|e| e.entry == m)
        }
    }
}

impl MoveConsumer for MoveList {
    #[inline]
    fn push(&mut self, m: Move, score: i32) {
        // it's quite dangerous to do this,
        // but this function is very much in the
        // hot path.
        debug_assert!(self.count < MAX_POSITION_MOVES);
        unsafe {
            *self.moves.get_unchecked_mut(self.count) = MoveListEntry { entry: m, score };
        }
        self.count += 1;
    }

    fn len(&self) -> usize {
        self.count
    }
}

impl Index<usize> for MoveList {
    type Output = Move;

    fn index(&self, index: usize) -> &Self::Output {
        &self.moves[..self.count][index].entry
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
