use crate::{
    chessmove::Move,
    historytable::{update_history, CORRECTION_HISTORY_GRAIN, CORRECTION_HISTORY_MAX, CORRECTION_HISTORY_WEIGHT_SCALE},
    piece::PieceType,
    threadlocal::ThreadData,
    util::{depth::Depth, Undo, MAX_DEPTH},
};

use super::{movegen::MoveListEntry, Board};

impl ThreadData<'_> {
    /// Update the history counters of a batch of moves.
    pub fn update_history(&mut self, pos: &Board, moves_to_adjust: &[Move], best_move: Move, depth: Depth) {
        for &m in moves_to_adjust {
            let piece_moved = pos.moved_piece(m);
            debug_assert!(piece_moved.is_some(), "Invalid piece moved by move {m} in position \n{pos}");
            let from = m.from();
            let to = m.history_to_square();
            let val = self.main_history.get_mut(
                piece_moved.unwrap(),
                to,
                pos.threats.all.contains_square(from),
                pos.threats.all.contains_square(to),
            );
            update_history(val, depth, m == best_move);
        }
    }

    /// Update the history counters for a single move.
    pub fn update_history_single(&mut self, pos: &Board, m: Move, depth: Depth) {
        let piece_moved = pos.moved_piece(m);
        debug_assert!(piece_moved.is_some(), "Invalid piece moved by move {m} in position \n{pos}");
        let from = m.from();
        let to = m.history_to_square();
        let val = self.main_history.get_mut(
            piece_moved.unwrap(),
            to,
            pos.threats.all.contains_square(from),
            pos.threats.all.contains_square(to),
        );
        update_history(val, depth, true);
    }

    /// Get the history scores for a batch of moves.
    pub(super) fn get_history_scores(&self, pos: &Board, ms: &mut [MoveListEntry]) {
        for m in ms {
            let piece_moved = pos.moved_piece(m.mov);
            let from = m.mov.from();
            let to = m.mov.history_to_square();
            m.score += i32::from(self.main_history.get(
                piece_moved.unwrap(),
                to,
                pos.threats.all.contains_square(from),
                pos.threats.all.contains_square(to),
            ));
        }
    }

    /// Get the history score for a single move.
    pub fn get_history_score(&self, pos: &Board, m: Move) -> i32 {
        let piece_moved = pos.moved_piece(m);
        let from = m.from();
        let to = m.history_to_square();
        i32::from(self.main_history.get(
            piece_moved.unwrap(),
            to,
            pos.threats.all.contains_square(from),
            pos.threats.all.contains_square(to),
        ))
    }

    /// Update the tactical history counters of a batch of moves.
    pub fn update_tactical_history(&mut self, pos: &Board, moves_to_adjust: &[Move], best_move: Move, depth: Depth) {
        for &m in moves_to_adjust {
            let piece_moved = pos.moved_piece(m);
            let capture = caphist_piece_type(pos, m);
            debug_assert!(piece_moved.is_some(), "Invalid piece moved by move {m} in position \n{pos}");
            let to = m.to();
            let val = self.tactical_history.get_mut(piece_moved.unwrap(), to, capture);
            update_history(val, depth, m == best_move);
        }
    }

    /// Get the tactical history scores for a batch of moves.
    pub(super) fn get_tactical_history_scores(&self, pos: &Board, ms: &mut [MoveListEntry]) {
        for m in ms {
            let piece_moved = pos.moved_piece(m.mov);
            let capture = caphist_piece_type(pos, m.mov);
            let to = m.mov.to();
            m.score += i32::from(self.tactical_history.get(piece_moved.unwrap(), to, capture));
        }
    }

    /// Get the tactical history score for a single move.
    #[allow(dead_code)]
    pub fn get_tactical_history_score(&self, pos: &Board, m: Move) -> i32 {
        let piece_moved = pos.moved_piece(m);
        let capture = caphist_piece_type(pos, m);
        let to = m.to();
        i32::from(self.tactical_history.get(piece_moved.unwrap(), to, capture))
    }

    /// Update the continuation history counters of a batch of moves.
    pub fn update_continuation_history(
        &mut self,
        pos: &Board,
        moves_to_adjust: &[Move],
        best_move: Move,
        depth: Depth,
        index: usize,
    ) {
        // get the index'th from the back of the conthist history, and make sure the entry is valid.
        if let Some(Undo { cont_hist_index: None, .. }) = pos.history.last() {
            return;
        }
        let conthist_index = match pos.history.len().checked_sub(index + 1).and_then(|i| pos.history.get(i)) {
            Some(Undo { cont_hist_index: Some(cont_hist_index), .. }) => *cont_hist_index,
            _ => return,
        };
        let cmh_block = self.continuation_history.get_index_mut(conthist_index);
        for &m in moves_to_adjust {
            let to = m.history_to_square();
            let piece = pos.moved_piece(m).unwrap();
            update_history(cmh_block.get_mut(piece, to), depth, m == best_move);
        }
    }

    /// Update the continuation history counter for a single move.
    pub fn update_continuation_history_single(&mut self, pos: &Board, m: Move, depth: Depth, index: usize) {
        // get the index'th from the back of the conthist history, and make sure the entry is valid.
        if let Some(Undo { cont_hist_index: None, .. }) = pos.history.last() {
            return;
        }
        let conthist_index = match pos.history.len().checked_sub(index + 1).and_then(|i| pos.history.get(i)) {
            Some(Undo { cont_hist_index: Some(cont_hist_index), .. }) => *cont_hist_index,
            _ => return,
        };
        let cmh_block = self.continuation_history.get_index_mut(conthist_index);

        let to = m.history_to_square();
        let piece = pos.moved_piece(m).unwrap();
        update_history(cmh_block.get_mut(piece, to), depth, true);
    }

    /// Get the continuation history scores for a batch of moves.
    pub(super) fn get_continuation_history_scores(&self, pos: &Board, ms: &mut [MoveListEntry], index: usize) {
        // get the index'th from the back of the conthist history, and make sure the entry is valid.
        if let Some(Undo { cont_hist_index: None, .. }) = pos.history.last() {
            return;
        }
        let conthist_index = match pos.history.len().checked_sub(index + 1).and_then(|i| pos.history.get(i)) {
            Some(Undo { cont_hist_index: Some(cont_hist_index), .. }) => *cont_hist_index,
            _ => return,
        };
        let cmh_block = self.continuation_history.get_index(conthist_index);
        for m in ms {
            let to = m.mov.history_to_square();
            let piece = pos.moved_piece(m.mov).unwrap();
            m.score += i32::from(cmh_block.get(piece, to));
        }
    }

    /// Get the continuation history score for a single move.
    pub fn get_continuation_history_score(&self, pos: &Board, m: Move, index: usize) -> i32 {
        // get the index'th from the back of the conthist history, and make sure the entry is valid.
        if let Some(Undo { cont_hist_index: None, .. }) = pos.history.last() {
            return 0;
        }
        let conthist_index = match pos.history.len().checked_sub(index + 1).and_then(|i| pos.history.get(i)) {
            Some(Undo { cont_hist_index: Some(cont_hist_index), .. }) => *cont_hist_index,
            _ => return 0,
        };
        let cmh_block = self.continuation_history.get_index(conthist_index);
        let to = m.history_to_square();
        let piece = pos.moved_piece(m).unwrap();
        i32::from(cmh_block.get(piece, to))
    }

    /// Add a killer move.
    pub fn insert_killer(&mut self, pos: &Board, m: Move) {
        debug_assert!(pos.height < MAX_DEPTH.ply_to_horizon());
        let idx = pos.height;
        if self.killer_move_table[idx][0] == Some(m) {
            return;
        }
        self.killer_move_table[idx][1] = self.killer_move_table[idx][0];
        self.killer_move_table[idx][0] = Some(m);
    }

    /// Add a move to the countermove table.
    pub fn insert_countermove(&mut self, pos: &Board, m: Move) {
        debug_assert!(pos.height < MAX_DEPTH.ply_to_horizon());
        let Some(&Undo { cont_hist_index: Some(cont_hist_index), .. }) = pos.history.last() else {
            return;
        };

        let prev_to = cont_hist_index.square;
        let prev_piece = cont_hist_index.piece;

        self.counter_move_table.add(prev_piece, prev_to, m);
    }

    /// Returns the counter move for this position.
    pub fn get_counter_move(&self, pos: &Board) -> Option<Move> {
        let Some(&Undo { cont_hist_index: Some(cont_hist_index), .. }) = pos.history.last() else {
            return None;
        };

        let prev_to = cont_hist_index.square;
        let prev_piece = cont_hist_index.piece;

        self.counter_move_table.get(prev_piece, prev_to)
    }

    /// Update the correction history for a pawn pattern.
    pub fn update_correction_history(&mut self, pos: &Board, depth: Depth, diff: i32) {
        let entry = self.correction_history.get_mut(pos.turn(), pos.pawn_key());
        let scaled_diff = diff * CORRECTION_HISTORY_GRAIN;
        let new_weight = 16.min(1 + depth.round());
        debug_assert!(new_weight <= CORRECTION_HISTORY_WEIGHT_SCALE);

        let update = *entry * (CORRECTION_HISTORY_WEIGHT_SCALE - new_weight) + scaled_diff * new_weight;
        *entry = i32::clamp(update / CORRECTION_HISTORY_WEIGHT_SCALE, -CORRECTION_HISTORY_MAX, CORRECTION_HISTORY_MAX);
    }

    /// Adjust a raw evaluation using statistics from the correction history.
    pub fn correct_evaluation(&self, pos: &Board, raw_eval: i32) -> i32 {
        let entry = self.correction_history.get(pos.turn(), pos.pawn_key());
        raw_eval + entry / CORRECTION_HISTORY_GRAIN
    }
}

pub fn caphist_piece_type(pos: &Board, mv: Move) -> PieceType {
    if mv.is_ep() || mv.is_promo() {
        // it's fine to make all promos of type PAWN,
        // because you'd never usually capture pawns on
        // the back ranks, so these slots are free in
        // the capture history table.
        PieceType::Pawn
    } else {
        pos.captured_piece(mv).expect("you weren't capturing anything!").piece_type()
    }
}
