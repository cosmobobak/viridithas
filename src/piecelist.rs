use std::fmt::Display;

use crate::{definitions::square_name, macros};

/// A list of squares that a given piece exists on.
#[derive(Debug, Clone, Copy)]
pub struct PieceList {
    data: [u8; 10],
    len: u8,
}

impl PieceList {
    pub const fn new() -> Self {
        Self {
            data: [0; 10],
            len: 0,
        }
    }

    pub fn first(&self) -> Option<&u8> {
        self.data[..self.len as usize].first()
    }

    pub fn iter(&self) -> impl Iterator<Item = &u8> {
        // SAFETY: self.len is always <= 10, so (..self.len) is always a valid slice range.
        unsafe { self.data.get_unchecked(..self.len as usize).iter() }
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut u8> {
        // SAFETY: self.len is always <= 10, so (..self.len) is always a valid slice range.
        unsafe { self.data.get_unchecked_mut(..self.len as usize).iter_mut() }
    }

    pub fn squares(&self) -> &[u8] {
        &self.data[..self.len as usize]
    }

    pub fn insert(&mut self, sq: u8) {
        debug_assert!(self.len < 10, "PieceList is full: {self}");
        debug_assert!(
            !self.squares().contains(&sq),
            "PieceList already contains square {}: {self}",
            sq_display(sq)
        );
        unsafe {
            *self.data.get_unchecked_mut(self.len as usize) = sq;
        }
        self.len += 1;
    }

    pub fn remove(&mut self, sq: u8) {
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
            "PieceList::remove: piece not found: looking for {} in {self}",
            sq_display(sq)
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

fn sq_display(sq: u8) -> String {
    square_name(sq)
        .map(std::string::ToString::to_string)
        .unwrap_or(format!("offboard: {}", sq))
}

impl Display for PieceList {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        let repr = self
            .iter()
            .map(|&s| {
                square_name(s)
                    .map(std::string::ToString::to_string)
                    .unwrap_or(format!("offboard: {s}"))
            })
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "[{}]", repr)
    }
}
