use std::fmt::Display;

use arrayvec::ArrayVec;

use crate::chess::chessmove::Move;
use crate::util::MAX_DEPTH;

#[derive(Clone, Debug)]
pub struct PVariation {
    pub(crate) score: i32,
    pub(crate) moves: ArrayVec<Move, MAX_DEPTH>,
}

impl PVariation {
    pub fn new() -> Self {
        Self {
            score: 0,
            moves: ArrayVec::new(),
        }
    }

    pub fn moves(&self) -> &[Move] {
        &self.moves
    }

    pub const fn score(&self) -> i32 {
        self.score
    }

    pub(crate) fn load_from(&mut self, m: Move, rest: &Self) {
        self.moves.clear();
        self.moves.push(m);
        self.moves
            .try_extend_from_slice(&rest.moves)
            .expect("attempted to construct a PV longer than MAX_PLY.");
    }

    pub const fn display(&self, chess960: bool) -> PVariationDisplay<'_> {
        PVariationDisplay { pv: self, chess960 }
    }
}

pub struct PVariationDisplay<'a> {
    pv: &'a PVariation,
    chess960: bool,
}

impl Display for PVariationDisplay<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if !self.pv.moves.is_empty() {
            write!(f, "pv ")?;
        }
        for &m in self.pv.moves() {
            write!(f, "{} ", m.display(self.chess960))?;
        }
        Ok(())
    }
}

impl Display for PVariation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.display(false).fmt(f)
    }
}
