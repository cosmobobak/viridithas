use std::fmt::Display;

use arrayvec::ArrayVec;

use crate::util::MAX_PLY;

use crate::chessmove::Move;

#[derive(Clone, Debug)]
pub struct PVariation {
    pub(crate) score: i32,
    pub(crate) moves: ArrayVec<Move, MAX_PLY>,
}

impl Default for PVariation {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl PVariation {
    const EMPTY: Self = Self { score: 0, moves: ArrayVec::new_const() };

    pub fn moves(&self) -> &[Move] {
        &self.moves
    }

    pub const fn score(&self) -> i32 {
        self.score
    }

    pub const fn default_const() -> Self {
        Self::EMPTY
    }

    pub(crate) fn load_from(&mut self, m: Move, rest: &Self) {
        self.moves.clear();
        self.moves.push(m);
        self.moves.try_extend_from_slice(&rest.moves).expect("attempted to construct a PV longer than MAX_PLY.");
    }
}

impl Display for PVariation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if !self.moves.is_empty() {
            write!(f, "pv ")?;
        }
        for &m in self.moves() {
            write!(f, "{m} ")?;
        }
        Ok(())
    }
}
