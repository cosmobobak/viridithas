use crate::{
    chessmove::Move,
    definitions::{depth::Depth, MAX_DEPTH, PIECE_EMPTY},
    historytable::update_history,
};

use super::Board;

impl Board {
    /// Add a move to the history table.
    pub fn add_history<const IS_GOOD: bool>(&mut self, m: Move, depth: Depth) {
        let piece_moved = self.moved_piece(m);
        debug_assert!(
            crate::validate::piece_valid(piece_moved) && piece_moved != PIECE_EMPTY,
            "Invalid piece moved by move {m} in position \n{self}"
        );
        let to = m.to();
        let val = self.history_table.get_mut(piece_moved, to);
        update_history::<IS_GOOD>(val, depth);
    }

    /// Get the history score for a move.
    pub(super) fn history_score(&self, m: Move) -> i32 {
        let piece_moved = self.moved_piece(m);
        let to = m.to();
        self.history_table.get(piece_moved, to)
    }

    /// Add a move to the countermove history table.
    pub fn insert_countermove(&mut self, m: Move) {
        debug_assert!(self.height < MAX_DEPTH.ply_to_horizon());
        let prev_move = if let Some(undo) = self.history.last() {
            undo.m
        } else {
            return;
        };
        if prev_move.is_null() {
            return;
        }
        let prev_to = prev_move.to();
        let prev_piece = self.piece_at(prev_to);

        self.counter_move_table.add(prev_piece, prev_to, m);
    }

    /// Get the countermove history score for a move.
    pub(super) fn is_countermove(&self, m: Move) -> bool {
        let prev_move = if let Some(undo) = self.history.last() {
            undo.m
        } else {
            return false;
        };
        if prev_move == Move::NULL {
            return false;
        }
        let prev_to = prev_move.to();
        let prev_piece = self.piece_at(prev_to);

        self.counter_move_table.get(prev_piece, prev_to) == m
    }

    /// Add a move to the follow-up history table.
    pub fn add_followup_history<const IS_GOOD: bool>(&mut self, m: Move, depth: Depth) {
        debug_assert!(self.height < MAX_DEPTH.ply_to_horizon());
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
        let tpa_piece = {
            let capture = prev_move.capture();
            // determine where to find the piece_t info:
            // we don't need to worry about ep-captures because
            // we just blanket filter them out with the null checks.
            if capture != PIECE_EMPTY && prev_move.to() == tpa_to {
                // the opponent captured a piece on this square, so we can use the capture.
                capture
            } else {
                // the opponent didn't capture a piece on this square, so it's still on the board.
                self.piece_at(tpa_to)
            }
        };
        let to = m.to();
        let piece = self.moved_piece(m);

        let val = self.followup_history.get_mut(tpa_piece, tpa_to, piece, to);
        update_history::<IS_GOOD>(val, depth);
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
        let tpa_piece = {
            let capture = prev_move.capture();
            // determine where to find the piece_t info:
            // we don't need to worry about ep-captures because
            // we just blanket filter them out with the null checks.
            if capture != PIECE_EMPTY && prev_move.to() == tpa_to {
                // the opponent captured a piece on this square, so we can use the capture.
                capture
            } else {
                // the opponent didn't capture a piece on this square, so it's still on the board.
                self.piece_at(tpa_to)
            }
        };
        let to = m.to();
        let piece = self.moved_piece(m);

        self.followup_history.get(tpa_piece, tpa_to, piece, to)
    }

    /// Add a killer move.
    pub fn insert_killer(&mut self, m: Move) {
        debug_assert!(self.height < MAX_DEPTH.ply_to_horizon());
        let entry = &mut self.killer_move_table[self.height];
        entry[1] = entry[0];
        entry[0] = m;
    }

    /// Determine if a move is a third-order killer move.
    /// The third-order killer is the first killer from the previous move (two ply ago)
    pub(super) fn is_third_order_killer(&self, m: Move) -> bool {
        self.height > 2 && self.killer_move_table[self.height - 2][0] == m
    }
}
