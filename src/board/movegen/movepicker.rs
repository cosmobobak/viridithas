use crate::{chessmove::Move, board::Board};

use super::{TT_MOVE_SCORE, MoveList};

use super::MoveListEntry;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    TTMove,
    GenerateMoves,
    YieldMoves,
}

pub struct MovePicker<const CAPTURES_ONLY: bool> {
    movelist: MoveList,
    index: usize,
    skip_ordering: bool,
    stage: Stage,
    tt_move: Move,
}

impl<const CAPTURES_ONLY: bool> MovePicker<CAPTURES_ONLY> {
    pub const fn new(tt_move: Move) -> Self {
        Self { 
            movelist: MoveList::new(), 
            index: 0, 
            skip_ordering: false, 
            tt_move, 
            stage: Stage::TTMove,
        }
    }

    pub fn moves_made(&self) -> &[MoveListEntry] {
        &self.movelist.moves[..self.index]
    }

    pub fn skip_ordering(&mut self) {
        self.skip_ordering = true;
    }
    
    /// Select the next move to try. Usually executes one iteration of partial insertion sort.
    pub fn next(&mut self, position: &mut Board) -> Option<MoveListEntry> {
        if self.stage == Stage::TTMove {
            self.stage = Stage::GenerateMoves;
            if position.is_pseudo_legal(self.tt_move) {
                // {
                //     let mut test_movelist = MoveList::new();
                //     position.generate_moves(&mut test_movelist);
                //     let in_movelist = test_movelist.iter().any(
                //         |&m| m == self.tt_move
                //     );
                //     assert!(in_movelist, "TT move was not pseudolegal, but was accepted by is_pseudo_legal\nTT move: {}/{:?} Board: {}", self.tt_move, self.tt_move, position.fen());
                // }
                return Some(MoveListEntry { entry: self.tt_move, score: TT_MOVE_SCORE });
            }
            // {
            //     let mut test_movelist = MoveList::new();
            //     position.generate_moves(&mut test_movelist);
            //     let in_movelist = test_movelist.iter().any(
            //         |&m| m == self.tt_move
            //     );
            //     assert!(!in_movelist, "TT move was pseudolegal, but was rejected by is_pseudo_legal\nTT move: {}/{:?} Board: {}", self.tt_move, self.tt_move, position.fen());
            // }
        }
        if self.stage == Stage::GenerateMoves {
            self.stage = Stage::YieldMoves;
            if CAPTURES_ONLY {
                position.generate_captures(&mut self.movelist);
            } else {
                position.generate_moves(&mut self.movelist);
            }
        }
        // If we have already tried all moves, return None.
        if self.index == self.movelist.count {
            return None;
        } else if self.skip_ordering {
            // If we are skipping ordering, just return the next move.
            let &m = unsafe { self.movelist.moves.get_unchecked(self.index) };
            self.index += 1;
            if m.entry == self.tt_move {
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

        if m.entry == self.tt_move {
            self.next(position)
        } else {
            Some(m)
        }
    }
}
