use crate::{chessmove::Move, definitions::{piece_index, MAX_DEPTH, PIECE_EMPTY}};

use super::Board;

impl Board {
    /// Add a move to the history table.
    pub fn add_history(&mut self, m: Move, score: i32) {
        let piece_moved = self.moved_piece(m) as usize;
        let history_board = unsafe { self.history_table.get_unchecked_mut(piece_moved) };
        let to = m.to() as usize;
        unsafe {
            *history_board.get_unchecked_mut(to) += score;
        }
    }

    /// Get the history score for a move.
    pub(super) fn history_score(&self, m: Move) -> i32 {
        let piece_moved = self.moved_piece(m) as usize;
        let to = m.to() as usize;
        unsafe {
            *self
                .history_table
                .get_unchecked(piece_moved)
                .get_unchecked(to)
        }
    }

    /// Add a move to the countermove history table.
    pub fn add_countermove_history(&mut self, m: Move, score: i32) {
        debug_assert!(self.height < MAX_DEPTH.n_ply());
        let prev_move = if let Some(undo) = self.history.last() {
            undo.m
        } else {
            return;
        };
        if prev_move.is_null() {
            return;
        }
        let prev_to = prev_move.to();
        let prev_piece_t = piece_index(self.piece_at(prev_to));
        let to = m.to();
        let piece_t = piece_index(self.moved_piece(m));
        // only the first indexing op benefits from get_unchecked, 
        // because the size information is erased only from the
        // top-level array type.
        debug_assert!(prev_piece_t < 6);
        self.countermove_history
            .add(prev_piece_t, prev_to, piece_t, to, score);
    }

    /// Get the countermove history score for a move.
    pub(super) fn countermove_history_score(&self, m: Move) -> i32 {
        let prev_move = if let Some(undo) = self.history.last() {
            undo.m
        } else {
            return 0;
        };
        if prev_move == Move::NULL {
            return 0;
        }
        let prev_to = prev_move.to();
        let prev_piece_t = piece_index(self.piece_at(prev_to));
        let to = m.to();
        let piece_t = piece_index(self.moved_piece(m));
        // only the first indexing op benefits from get_unchecked, 
        // because the size information is erased only from the
        // top-level array type.
        debug_assert!(prev_piece_t < 6);
        self.countermove_history
            .get(prev_piece_t, prev_to, piece_t, to)
    }

    /// Add a move to the follow-up history table.
    pub fn add_followup_history(&mut self, m: Move, score: i32) {
        debug_assert!(self.height < MAX_DEPTH.n_ply());
        let two_ply_ago = match self.history.len().checked_sub(2) {
            Some(idx) => idx,
            None => return,
        };
        let move_to_follow_up = self.history[two_ply_ago].m;
        let prev_move = self.history[two_ply_ago + 1].m;
        if move_to_follow_up.is_null() || prev_move.is_null() || prev_move.is_ep() {
            return;
        }
        let tpa_to = move_to_follow_up.to();
        // getting the previous piece type is a little awkward,
        // because follow-up history looks two ply into the past,
        // meaning that the piece on the target square of the move 
        // two ply ago may have been captured.
        let tpa_piece_t = {
            let capture = prev_move.capture();
            // determine where to find the piece_t info:
            // we don't need to worry about ep-captures because
            // we just blanket filter them out with the null checks.
            if capture != PIECE_EMPTY && prev_move.to() == tpa_to {
                // the opponent captured a piece on this square, so we can use the capture.
                piece_index(capture)
            } else {
                // the opponent didn't capture a piece on this square, so it's still on the board.
                piece_index(self.piece_at(tpa_to))
            }
        };
        let to = m.to();
        let piece_t = piece_index(self.moved_piece(m));
        // only the first indexing op benefits from get_unchecked, 
        // because the size information is erased only from the
        // top-level array type.
        debug_assert!(tpa_piece_t < 6);
        self.followup_history
            .add(tpa_piece_t, tpa_to, piece_t, to, score);
    }

    /// Get the follow-up history score for a move.
    pub(super) fn followup_history_score(&self, m: Move) -> i32 {
        let two_ply_ago = match self.history.len().checked_sub(2) {
            Some(idx) => idx,
            None => return 0,
        };
        let move_to_follow_up = self.history[two_ply_ago].m;
        let prev_move = self.history[two_ply_ago + 1].m;
        if move_to_follow_up.is_null() || prev_move.is_null() || prev_move.is_ep() {
            return 0;
        }
        let tpa_to = move_to_follow_up.to();
        // getting the previous piece type is a little awkward,
        // because follow-up history looks two ply into the past,
        // meaning that the piece on the target square of the move 
        // two ply ago may have been captured.
        let tpa_piece_t = {
            let capture = prev_move.capture();
            // determine where to find the piece_t info:
            // we don't need to worry about ep-captures because
            // we just blanket filter them out with the null checks.
            if capture != PIECE_EMPTY && prev_move.to() == tpa_to {
                // the opponent captured a piece on this square, so we can use the capture.
                piece_index(capture)
            } else {
                // the opponent didn't capture a piece on this square, so it's still on the board.
                piece_index(self.piece_at(tpa_to))
            }
        };
        let to = m.to();
        let piece_t = piece_index(self.moved_piece(m));
        // only the first indexing op benefits from get_unchecked, 
        // because the size information is erased only from the
        // top-level array type.
        debug_assert!(tpa_piece_t < 6);
        self.followup_history
            .get(tpa_piece_t, tpa_to, piece_t, to)
    }

    /// Add a killer move.
    pub fn insert_killer(&mut self, m: Move) {
        debug_assert!(self.height < MAX_DEPTH.n_ply());
        let entry = unsafe { self.killer_move_table.get_unchecked_mut(self.height) };
        entry[1] = entry[0];
        entry[0] = m;
    }

    /// Determine if a move is a third-order killer move.
    /// The third-order killer is the first killer from the previous move (two ply ago)
    pub(super) fn is_third_order_killer(&self, m: Move) -> bool {
        self.height > 2 && self.killer_move_table[self.height - 2][0] == m
    }
}