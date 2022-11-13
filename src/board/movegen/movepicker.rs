use crate::{chessmove::Move, board::Board};

use super::{TT_MOVE_SCORE, MoveList};

use super::MoveListEntry;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    TTMove,
    _Killer1,
    _Killer2,
    GenerateMoves,
    YieldMoves,
}

pub struct MovePicker<const CAPTURES_ONLY: bool, const DO_SEE: bool> {
    movelist: MoveList,
    index: usize,
    skip_ordering: bool,
    stage: Stage,
    tt_move: Move,
    _killers: [Move; 2],
}

impl<const CAPTURES_ONLY: bool, const DO_SEE: bool> MovePicker<CAPTURES_ONLY, DO_SEE> {
    pub const fn new(tt_move: Move, killers: [Move; 2]) -> Self {
        Self { 
            movelist: MoveList::new(), 
            index: 0, 
            skip_ordering: false, 
            stage: Stage::TTMove,
            tt_move,
            _killers: killers,
        }
    }

    pub fn moves_made(&self) -> &[MoveListEntry] {
        &self.movelist.moves[..self.index]
    }

    pub fn skip_ordering(&mut self) {
        self.skip_ordering = true;
    }

    pub fn was_tried_lazily(&self, m: Move) -> bool {
        #![allow(clippy::if_same_then_else, clippy::branches_sharing_code)]
        if CAPTURES_ONLY {
            m == self.tt_move
        } else {
            m == self.tt_move// || m == self.killers[0] || m == self.killers[1]
        }
    }
    
    /// Select the next move to try. Usually executes one iteration of partial insertion sort.
    pub fn next(&mut self, position: &mut Board) -> Option<MoveListEntry> {
        if self.stage == Stage::TTMove {
            self.stage = Stage::GenerateMoves;
            if position.is_pseudo_legal(self.tt_move) {
                return Some(MoveListEntry { entry: self.tt_move, score: TT_MOVE_SCORE });
            }
        }
        // if self.stage == Stage::Killer1 {
        //     self.stage = Stage::Killer2;
        //     let killer = self.killers[0];
        //     if killer != self.tt_move && position.is_pseudo_legal(killer) {
        //         return Some(MoveListEntry { entry: killer, score: FIRST_ORDER_KILLER_SCORE });
        //     }
        // }
        // if self.stage == Stage::Killer2 {
        //     self.stage = Stage::GenerateMoves;
        //     let killer = self.killers[1];
        //     if killer != self.tt_move && position.is_pseudo_legal(killer) {
        //         return Some(MoveListEntry { entry: killer, score: SECOND_ORDER_KILLER_SCORE });
        //     }
        // }
        if self.stage == Stage::GenerateMoves {
            self.stage = Stage::YieldMoves;
            if CAPTURES_ONLY {
                position.generate_captures::<DO_SEE>(&mut self.movelist);
            } else {
                position.generate_moves::<DO_SEE>(&mut self.movelist);
            }
        }
        // If we have already tried all moves, return None.
        if self.index == self.movelist.count {
            return None;
        } else if self.skip_ordering {
            // If we are skipping ordering, just return the next move.
            let &m = unsafe { self.movelist.moves.get_unchecked(self.index) };
            self.index += 1;
            if self.was_tried_lazily(m.entry) {
                return self.next(position);
            }
            return Some(m);
        }

        // SAFETY: self.index is always in bounds.
        let mut best_score = unsafe { self.movelist.moves.get_unchecked(self.index).score };
        let mut best_num = self.index;

        // find the best move in the unsorted portion of the movelist.
        for index in self.index + 1..self.movelist.count {
            // SAFETY: self.count is always less than 256, and self.index is always in bounds.
            let score = unsafe { self.movelist.moves.get_unchecked(index).score };
            if score > best_score {
                best_score = score;
                best_num = index;
            }
        }

        debug_assert!(self.index < self.movelist.count);
        debug_assert!(best_num < self.movelist.count);
        debug_assert!(best_num >= self.index);

        // SAFETY: best_num is drawn from self.index..self.count, which is always in bounds.
        let &m = unsafe { self.movelist.moves.get_unchecked(best_num) };

        // swap the best move with the first unsorted move.
        // SAFETY: best_num is drawn from self.index..self.count, which is always in bounds.
        // and self.index is always in bounds.
        unsafe {
            *self.movelist.moves.get_unchecked_mut(best_num) = *self.movelist.moves.get_unchecked(self.index);
        }

        self.index += 1;

        if self.was_tried_lazily(m.entry) {
            self.next(position)
        } else {
            Some(m)
        }
    }
}
