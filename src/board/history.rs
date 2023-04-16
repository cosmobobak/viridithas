use crate::{
    chessmove::Move,
    definitions::{depth::Depth, Undo, MAX_DEPTH},
    historytable::update_history,
    piece::Piece,
    threadlocal::ThreadData,
};

use super::{Board, movegen::MoveListEntry};

impl ThreadData {
    /// Add a quiet move to the history table.
    pub fn add_history<const IS_GOOD: bool>(&mut self, pos: &Board, m: Move, depth: Depth) {
        let piece_moved = pos.moved_piece(m);
        debug_assert!(
            piece_moved != Piece::EMPTY,
            "Invalid piece moved by move {m} in position \n{pos}"
        );
        let to = m.to();
        let val = self.history_table.get_mut(piece_moved, to);
        update_history::<IS_GOOD>(val, depth);
    }

    /// Get the history score for a quiet move.
    pub(super) fn get_history_scores(&self, pos: &Board, ms: &mut [MoveListEntry]) {
        for m in ms {
            let piece_moved = pos.moved_piece(m.mov);
            let to = m.mov.to();
            m.score += i32::from(self.history_table.get(piece_moved, to));
        }
    }

    /// Add a move to the follow-up history table.
    pub fn add_followup_history<const IS_GOOD: bool>(
        &mut self,
        pos: &Board,
        m: Move,
        depth: Depth,
    ) {
        debug_assert!(pos.height < MAX_DEPTH.ply_to_horizon());
        let Some(two_ply_ago) = pos.history.len().checked_sub(2) else { return };
        let move_to_follow_up = pos.history[two_ply_ago].m;
        let prev_move = pos.history[two_ply_ago + 1].m;
        if move_to_follow_up.is_null() || prev_move.is_null() || prev_move.is_ep() {
            return;
        }
        let tpa_to = move_to_follow_up.to();
        // getting the previous piece type is a little awkward,
        // because follow-up history looks two ply into the past,
        // meaning that the piece on the target square of the move
        // two ply ago may have been captured.
        let tpa_piece = {
            let capture = pos.captured_piece(prev_move);
            // determine where to find the piece_t info:
            // we don't need to worry about ep-captures because
            // we just blanket filter them out with the null checks.
            if capture != Piece::EMPTY && prev_move.to() == tpa_to {
                // the opponent captured a piece on this square, so we can use the capture.
                capture
            } else {
                // the opponent didn't capture a piece on this square, so it's still on the board.
                pos.piece_at(tpa_to)
            }
        };
        let to = m.to();
        let piece = pos.moved_piece(m);

        let val = self.followup_history.get_mut(tpa_piece, tpa_to).get_mut(piece, to);
        update_history::<IS_GOOD>(val, depth);
    }

    /// Get the follow-up history score for a move.
    pub(super) fn get_followup_history_scores(&self, pos: &Board, ms: &mut [MoveListEntry]) {
        let Some(two_ply_ago) = pos.history.len().checked_sub(2) else { return };
        let move_to_follow_up = pos.history[two_ply_ago].m;
        let prev_move = pos.history[two_ply_ago + 1].m;
        if move_to_follow_up.is_null() || prev_move.is_null() || prev_move.is_ep() {
            return;
        }
        let tpa_to = move_to_follow_up.to();
        // getting the previous piece type is a little awkward,
        // because follow-up history looks two ply into the past,
        // meaning that the piece on the target square of the move
        // two ply ago may have been captured.
        let tpa_piece = {
            let capture = pos.captured_piece(prev_move);
            // determine where to find the piece_t info:
            // we don't need to worry about ep-captures because
            // we just blanket filter them out with the null checks.
            if capture != Piece::EMPTY && prev_move.to() == tpa_to {
                // the opponent captured a piece on this square, so we can use the capture.
                capture
            } else {
                // the opponent didn't capture a piece on this square, so it's still on the board.
                pos.piece_at(tpa_to)
            }
        };

        let fuh_block = self.followup_history.get(tpa_piece, tpa_to);
        for m in ms {
            let to = m.mov.to();
            let piece = pos.moved_piece(m.mov);
            m.score += i32::from(fuh_block.get(piece, to));
        }
    }

    /// Add a move to the countermove table.
    pub fn insert_countermove(&mut self, pos: &Board, m: Move) {
        debug_assert!(pos.height < MAX_DEPTH.ply_to_horizon());
        let Some(&Undo { m: prev_move, .. }) = pos.history.last() else {
            return;
        };
        if prev_move.is_null() {
            return;
        }
        let prev_to = prev_move.to();
        let prev_piece = pos.piece_at(prev_to);

        self.counter_move_table.add(prev_piece, prev_to, m);
    }

    /// Returns the counter move for this position.
    pub(super) fn get_counter_move(&self, pos: &Board) -> Move {
        let Some(&Undo { m: prev_move, .. }) = pos.history.last() else {
            return Move::NULL;
        };
        if prev_move == Move::NULL {
            return Move::NULL;
        }
        let prev_to = prev_move.to();
        let prev_piece = pos.piece_at(prev_to);

        self.counter_move_table.get(prev_piece, prev_to)
    }

    /// Add a move to the countermove history table.
    pub fn add_countermove_history<const IS_GOOD: bool>(
        &mut self,
        pos: &Board,
        m: Move,
        depth: Depth,
    ) {
        debug_assert!(pos.height < MAX_DEPTH.ply_to_horizon());
        let Some(&Undo { m: prev_move, .. }) = pos.history.last() else {
            return;
        };
        if prev_move.is_null() {
            return;
        }
        let prev_to = prev_move.to();
        let prev_piece = pos.piece_at(prev_to);
        let to = m.to();
        let piece = pos.moved_piece(m);

        let val = self.counter_move_history.get_mut(prev_piece, prev_to).get_mut(piece, to);
        update_history::<IS_GOOD>(val, depth);
    }

    /// Get the countermove history score for a move.
    pub(super) fn get_counter_move_history_scores(&self, pos: &Board, ms: &mut [MoveListEntry]) {
        let Some(&Undo { m: prev_move, .. }) = pos.history.last() else {
            return;
        };
        if prev_move.is_null() {
            return;
        }
        let prev_to = prev_move.to();
        let prev_piece = pos.piece_at(prev_to);
        let cmh_block = self.counter_move_history.get(prev_piece, prev_to);

        for m in ms {
            let to = m.mov.to();
            let piece = pos.moved_piece(m.mov);
            m.score += i32::from(cmh_block.get(piece, to));
        }
    }

    /// Add a killer move.
    pub fn insert_killer(&mut self, pos: &Board, m: Move) {
        debug_assert!(pos.height < MAX_DEPTH.ply_to_horizon());
        let idx = pos.height;
        self.killer_move_table[idx][1] = self.killer_move_table[idx][0];
        self.killer_move_table[idx][0] = m;
    }
}
