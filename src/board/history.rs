use crate::{
    chessmove::Move,
    definitions::{depth::Depth, Rank, Undo, MAX_DEPTH},
    historytable::update_history,
    piece::{Piece, PieceType},
    threadlocal::ThreadData,
};

use super::{movegen::MoveListEntry, Board};

impl ThreadData {
    /// Update the history counters of a batch of moves.
    pub fn update_history(
        &mut self,
        pos: &Board,
        moves_to_adjust: &[Move],
        best_move: Move,
        depth: Depth,
    ) {
        for &m in moves_to_adjust {
            let piece_moved = pos.moved_piece(m);
            debug_assert!(
                piece_moved != Piece::EMPTY,
                "Invalid piece moved by move {m} in position \n{pos}"
            );
            let to = m.history_to_square();
            let val = self.main_history.get_mut(piece_moved, to);
            update_history(val, depth, m == best_move);
        }
    }

    /// Get the history scores for a batch of moves.
    pub(super) fn get_history_scores(&self, pos: &Board, ms: &mut [MoveListEntry]) {
        for m in ms {
            let piece_moved = pos.moved_piece(m.mov);
            let to = m.mov.history_to_square();
            m.score += i32::from(self.main_history.get(piece_moved, to));
        }
    }

    /// Update the tactical history counters of a batch of moves.
    pub fn update_tactical_history(
        &mut self,
        pos: &Board,
        moves_to_adjust: &[Move],
        best_move: Move,
        depth: Depth,
    ) {
        for &m in moves_to_adjust {
            let piece_moved = pos.moved_piece(m);
            let capture = caphist_piece_type(pos, m);
            debug_assert!(
                piece_moved != Piece::EMPTY,
                "Invalid piece moved by move {m} in position \n{pos}"
            );
            let to = m.to();
            let val = self.tactical_history.get_mut(piece_moved, to, capture);
            update_history(val, depth, m == best_move);
        }
    }

    /// Get the tactical history scores for a batch of moves.
    pub(super) fn get_tactical_history_scores(&self, pos: &Board, ms: &mut [MoveListEntry]) {
        for m in ms {
            let piece_moved = pos.moved_piece(m.mov);
            let capture = caphist_piece_type(pos, m.mov);
            let to = m.mov.to();
            m.score += i32::from(self.tactical_history.get(piece_moved, to, capture));
        }
    }

    /// Update the countermove history counters of a batch of moves.
    pub fn update_countermove_history(
        &mut self,
        pos: &Board,
        moves_to_adjust: &[Move],
        best_move: Move,
        depth: Depth,
    ) {
        debug_assert!(pos.height < MAX_DEPTH.ply_to_horizon());
        let Some(&Undo { m: prev_move, .. }) = pos.history.last() else {
            return;
        };
        if prev_move.is_null() {
            return;
        }
        let prev_to = prev_move.history_to_square();
        let prev_piece = if prev_move.is_castle() {
            Piece::new(pos.turn().flip(), PieceType::KING)
        } else {
            pos.piece_at(prev_to)
        };

        debug_assert_ne!(
            prev_piece,
            Piece::EMPTY,
            "Piece on target square of move to counter has to exist!"
        );
        debug_assert_eq!(
            prev_piece.colour(),
            pos.turn().flip(),
            "Piece on target square of move to counter has to be the opposite colour to us!"
        );

        let cmh_block = self.counter_move_history.get_mut(prev_piece, prev_to);
        for &m in moves_to_adjust {
            let to = m.history_to_square();
            let piece = pos.moved_piece(m);
            update_history(cmh_block.get_mut(piece, to), depth, m == best_move);
        }
    }

    /// Get the countermove history scores for a batch of moves.
    pub(super) fn get_counter_move_history_scores(&self, pos: &Board, ms: &mut [MoveListEntry]) {
        let Some(&Undo { m: prev_move, .. }) = pos.history.last() else {
            return;
        };
        if prev_move.is_null() {
            return;
        }
        let prev_to = prev_move.history_to_square();
        let prev_piece = if prev_move.is_castle() {
            Piece::new(pos.turn().flip(), PieceType::KING)
        } else {
            pos.piece_at(prev_to)
        };

        debug_assert_ne!(
            prev_piece,
            Piece::EMPTY,
            "Piece on target square of move to counter has to exist!"
        );
        debug_assert_eq!(
            prev_piece.colour(),
            pos.turn().flip(),
            "Piece on target square of move to counter has to be the opposite colour to us!"
        );

        let cmh_block = self.counter_move_history.get(prev_piece, prev_to);
        for m in ms {
            let to = m.mov.history_to_square();
            let piece = pos.moved_piece(m.mov);
            m.score += i32::from(cmh_block.get(piece, to));
        }
    }

    /// Update the follow-up history counters of a batch of moves.
    pub fn update_followup_history(
        &mut self,
        pos: &Board,
        moves_to_adjust: &[Move],
        best_move: Move,
        depth: Depth,
    ) {
        debug_assert!(pos.height < MAX_DEPTH.ply_to_horizon());
        let Some(two_ply_ago) = pos.history.len().checked_sub(2) else { return };
        let move_to_follow_up = pos.history[two_ply_ago].m;
        let prev_move = pos.history[two_ply_ago + 1].m;
        if move_to_follow_up.is_null() || prev_move.is_null() {
            return;
        }
        let tpa_to = move_to_follow_up.history_to_square();
        // getting the previous piece type is a little awkward,
        // because follow-up history looks two ply into the past,
        // meaning that the piece on the target square of the move
        // two ply ago may have been captured.
        let tpa_piece = {
            let at_target_square = pos.piece_at(tpa_to);
            debug_assert!(
                at_target_square != Piece::EMPTY || prev_move.is_ep() || move_to_follow_up.is_castle(),
                "Piece on target square of move to follow up on has to exist!"
            );
            if prev_move.is_ep() {
                // if the previous move was an en-passant capture, then the
                // move we're following up must have been a double pawn push.
                // as such, we can just construct a pawn of our colour.
                debug_assert!(
                    move_to_follow_up.to().rank() == Rank::double_pawn_push_rank(pos.turn())
                );
                Piece::new(pos.turn(), PieceType::PAWN)
            } else if at_target_square.colour() == pos.turn() {
                // if the piece on the target square is the same colour as us
                // then nothing happened to our piece, so we can use it directly.
                at_target_square
            } else {
                // otherwise, the most recent move captured our piece, so we
                // look in the undo history to find out what our piece was.
                debug_assert_ne!(
                    pos.history[two_ply_ago + 1].capture,
                    Piece::EMPTY,
                    "Opponent's move has to capture a piece!"
                );
                debug_assert_eq!(prev_move.to(), tpa_to, "Opponent's move has to go to the same square as the move we're following up on!");
                pos.history[two_ply_ago + 1].capture
            }
        };
        debug_assert_ne!(
            tpa_piece,
            Piece::EMPTY,
            "Piece on target square of move to follow up on has to exist!"
        );
        debug_assert_eq!(
            tpa_piece.colour(),
            pos.turn(),
            "Piece on target square of move to follow up on has to be the same colour as us!"
        );

        let fuh_block = self.followup_history.get_mut(tpa_piece, tpa_to);
        for &m in moves_to_adjust {
            let to = m.history_to_square();
            let piece = pos.moved_piece(m);
            update_history(fuh_block.get_mut(piece, to), depth, m == best_move);
        }
    }

    /// Get the follow-up history scores for a batch of moves.
    pub(super) fn get_followup_history_scores(&self, pos: &Board, ms: &mut [MoveListEntry]) {
        let Some(two_ply_ago) = pos.history.len().checked_sub(2) else { return };
        let move_to_follow_up = pos.history[two_ply_ago].m;
        let prev_move = pos.history[two_ply_ago + 1].m;
        if move_to_follow_up.is_null() || prev_move.is_null() {
            return;
        }
        let tpa_to = move_to_follow_up.history_to_square();
        // getting the previous piece type is a little awkward,
        // because follow-up history looks two ply into the past,
        // meaning that the piece on the target square of the move
        // two ply ago may have been captured.
        let tpa_piece = {
            let at_target_square = pos.piece_at(tpa_to);
            debug_assert!(
                at_target_square != Piece::EMPTY || prev_move.is_ep() || move_to_follow_up.is_castle(),
                "Piece on target square of move to follow up on has to exist!"
            );
            if prev_move.is_ep() {
                // if the previous move was an en-passant capture, then the
                // move we're following up must have been a double pawn push.
                // as such, we can just construct a pawn of our colour.
                debug_assert!(
                    move_to_follow_up.to().rank() == Rank::double_pawn_push_rank(pos.turn())
                );
                Piece::new(pos.turn(), PieceType::PAWN)
            } else if at_target_square.colour() == pos.turn() {
                // if the piece on the target square is the same colour as us
                // then nothing happened to our piece, so we can use it directly.
                at_target_square
            } else {
                // otherwise, the most recent move captured our piece, so we
                // look in the undo history to find out what our piece was.
                debug_assert_ne!(
                    pos.history[two_ply_ago + 1].capture,
                    Piece::EMPTY,
                    "Opponent's move has to capture a piece!"
                );
                debug_assert_eq!(prev_move.to(), tpa_to, "Opponent's move has to go to the same square as the move we're following up on!");
                pos.history[two_ply_ago + 1].capture
            }
        };
        debug_assert_ne!(
            tpa_piece,
            Piece::EMPTY,
            "Piece on target square of move to follow up on has to exist!"
        );
        debug_assert_eq!(
            tpa_piece.colour(),
            pos.turn(),
            "Piece on target square of move to follow up on has to be the same colour as us!"
        );

        let fuh_block = self.followup_history.get(tpa_piece, tpa_to);
        for m in ms {
            let to = m.mov.history_to_square();
            let piece = pos.moved_piece(m.mov);
            m.score += i32::from(fuh_block.get(piece, to));
        }
    }

    /// Add a killer move.
    pub fn insert_killer(&mut self, pos: &Board, m: Move) {
        debug_assert!(pos.height < MAX_DEPTH.ply_to_horizon());
        let idx = pos.height;
        if self.killer_move_table[idx][0] == m {
            return;
        }
        self.killer_move_table[idx][1] = self.killer_move_table[idx][0];
        self.killer_move_table[idx][0] = m;
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

        let prev_to = prev_move.history_to_square();
        let prev_piece = pos.piece_at(prev_to);

        self.counter_move_table.add(prev_piece, prev_to, m);
    }

    /// Returns the counter move for this position.
    pub fn get_counter_move(&self, pos: &Board) -> Move {
        let Some(&Undo { m: prev_move, .. }) = pos.history.last() else {
            return Move::NULL;
        };
        if prev_move == Move::NULL {
            return Move::NULL;
        }

        let prev_to = prev_move.history_to_square();
        let prev_piece = pos.piece_at(prev_to);

        self.counter_move_table.get(prev_piece, prev_to)
    }
}

pub fn caphist_piece_type(pos: &Board, mv: Move) -> PieceType {
    if mv.is_ep() || mv.is_promo() {
        // it's fine to make all promos of type PAWN,
        // because you'd never usually capture pawns on
        // the back ranks, so these slots are free in
        // the capture history table.
        PieceType::PAWN
    } else {
        pos.captured_piece(mv).piece_type()
    }
}
