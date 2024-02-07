use std::fmt::Display;

use crate::util::MAX_PLY;

use crate::chessmove::Move;

#[derive(Clone, Debug)]
pub struct PVariation {
    pub(crate) length: usize,
    pub(crate) score: i32,
    pub(crate) line: [Move; MAX_PLY],
}

impl Default for PVariation {
    fn default() -> Self {
        Self { length: 0, score: 0, line: [Move::NULL; MAX_PLY] }
    }
}

impl PVariation {
    pub fn moves(&self) -> &[Move] {
        &self.line[..self.length]
    }

    pub const fn score(&self) -> i32 {
        self.score
    }

    pub(crate) fn load_from(&mut self, m: Move, rest: &Self) {
        self.line[0] = m;
        self.line[1..=rest.length].copy_from_slice(&rest.line[..rest.length]);
        self.length = rest.length + 1;
    }
}

impl Display for PVariation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.length != 0 {
            write!(f, "pv ")?;
        }
        for &m in self.moves() {
            write!(f, "{m} ")?;
        }
        Ok(())
    }
}
