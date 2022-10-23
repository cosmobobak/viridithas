use std::fmt::Display;

use crate::{definitions::Square, macros};

/// A list of squares that a given piece exists on.
#[derive(Debug, Clone, Copy)]
pub struct PieceList {
    data: [Square; 10],
    len: u8,
}

impl PieceList {
    pub const fn new() -> Self {
        Self { data: [Square::NO_SQUARE; 10], len: 0 }
    }

    pub fn first(&self) -> Option<&Square> {
        self.data[..self.len as usize].first()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Square> {
        // SAFETY: self.len is always <= 10, so (..self.len) is always a valid slice range.
        unsafe { self.data.get_unchecked(..self.len as usize).iter() }
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Square> {
        // SAFETY: self.len is always <= 10, so (..self.len) is always a valid slice range.
        unsafe { self.data.get_unchecked_mut(..self.len as usize).iter_mut() }
    }

    pub fn squares(&self) -> &[Square] {
        &self.data[..self.len as usize]
    }

    pub fn insert(&mut self, sq: Square) {
        debug_assert!(self.len < 10, "PieceList is full: {self}");
        debug_assert!(
            !self.squares().contains(&sq),
            "PieceList already contains square {sq}: {self}",
        );
        unsafe {
            *self.data.get_unchecked_mut(self.len as usize) = sq;
        }
        self.len += 1;
    }

    pub fn remove(&mut self, sq: Square) {
        debug_assert!(self.len > 0, "PieceList is empty");
        let mut idx = 0;
        while idx < self.len {
            if unsafe { *self.data.get_unchecked(idx as usize) } == sq {
                self.len -= 1;
                unsafe {
                    *self.data.get_unchecked_mut(idx as usize) =
                        *self.data.get_unchecked(self.len as usize);
                    return;
                }
            }
            idx += 1;
        }
        debug_assert!(
            false,
            "PieceList::remove: piece not found: looking for {sq} in {self}",
        );
        unsafe {
            macros::inconceivable!();
        }
    }

    pub const fn len(&self) -> u8 {
        self.len
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }
}

impl Display for PieceList {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        let repr = self
            .iter()
            .map(|&s| {
                s.to_string()
            })
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "[{}]", repr)
    }
}
