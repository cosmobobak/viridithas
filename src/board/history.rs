use crate::{
    chessmove::Move,
    historytable::{update_history, ContHistIndex},
    piece::{Piece, PieceType},
    threadlocal::ThreadData,
    util::{depth::Depth, Square, Undo, MAX_DEPTH},
};

use super::{movegen::MoveListEntry, Board};

impl ThreadData<'_> {
    /// Update the history counters of a batch of moves.
    pub fn update_history(&mut self, pos: &Board, moves_to_adjust: &[Move], best_move: Move, depth: Depth) {
        for &m in moves_to_adjust {
            let piece_moved = pos.moved_piece(m);
            debug_assert!(piece_moved != Piece::EMPTY, "Invalid piece moved by move {m} in position \n{pos}");
            let from = m.from();
            let to = m.history_to_square();
            let val = self.main_history.get_mut(
                piece_moved,
                to,
                pos.threats.all.contains_square(from),
                pos.threats.all.contains_square(to),
            );
            update_history(val, depth, m == best_move);
        }
    }

    /// Get the history scores for a batch of moves.
    pub(super) fn get_history_scores(&self, pos: &Board, ms: &mut [MoveListEntry]) {
        for m in ms {
            let piece_moved = pos.moved_piece(m.mov);
            let from = m.mov.from();
            let to = m.mov.history_to_square();
            m.score += i32::from(self.main_history.get(
                piece_moved,
                to,
                pos.threats.all.contains_square(from),
                pos.threats.all.contains_square(to),
            ));
        }
    }

    /// Get the history score for a single move.
    // pub fn get_history_score(&self, pos: &Board, m: Move) -> i32 {
    //     let piece_moved = pos.moved_piece(m);
    //     let from = m.from();
    //     let to = m.history_to_square();
    //     i32::from(self.main_history.get(
    //         piece_moved,
    //         to,
    //         pos.threats.all.contains_square(from),
    //         pos.threats.all.contains_square(to),
    //     ))
    // }

    /// Update the tactical history counters of a batch of moves.
    pub fn update_tactical_history(&mut self, pos: &Board, moves_to_adjust: &[Move], best_move: Move, depth: Depth) {
        for &m in moves_to_adjust {
            let piece_moved = pos.moved_piece(m);
            let capture = caphist_piece_type(pos, m);
            debug_assert!(piece_moved != Piece::EMPTY, "Invalid piece moved by move {m} in position \n{pos}");
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
        if let Some(Undo { cont_hist_index: ContHistIndex { square: Square::NO_SQUARE, .. }, .. }) = pos.history.last()
        {
            return;
        }
        let conthist_index = match pos.history.len().checked_sub(index + 1).and_then(|i| pos.history.get(i)) {
            None | Some(Undo { cont_hist_index: ContHistIndex { square: Square::NO_SQUARE, .. }, .. }) => return,
            Some(Undo { cont_hist_index, .. }) => *cont_hist_index,
        };
        let table = self.cont_hists[index].as_mut();
        let cmh_block = table.get_index_mut(conthist_index);
        for &m in moves_to_adjust {
            let to = m.history_to_square();
            let piece = pos.moved_piece(m);
            update_history(cmh_block.get_mut(piece, to), depth, m == best_move);
        }
    }

    /// Get the continuation history scores for a batch of moves.
    pub(super) fn get_continuation_history_scores(&self, pos: &Board, ms: &mut [MoveListEntry], index: usize) {
        // get the index'th from the back of the conthist history, and make sure the entry is valid.
        if let Some(Undo { cont_hist_index: ContHistIndex { square: Square::NO_SQUARE, .. }, .. }) = pos.history.last()
        {
            return;
        }
        let conthist_index = match pos.history.len().checked_sub(index + 1).and_then(|i| pos.history.get(i)) {
            None | Some(Undo { cont_hist_index: ContHistIndex { square: Square::NO_SQUARE, .. }, .. }) => return,
            Some(Undo { cont_hist_index, .. }) => *cont_hist_index,
        };
        let table = self.cont_hists[index].as_ref();
        let cmh_block = table.get_index(conthist_index);
        for m in ms {
            let to = m.mov.history_to_square();
            let piece = pos.moved_piece(m.mov);
            m.score += i32::from(cmh_block.get(piece, to));
        }
    }

    /// Get the continuation history score for a single move.
    // pub fn get_continuation_history_score(&self, pos: &Board, m: Move, index: usize) -> i32 {
    //     if pos.height <= index {
    //         return 0;
    //     }
    //     let conthist_index = self.conthist_indices[pos.height - 1 - index];
    //     let table = self.cont_hists[index].as_ref();
    //     let piece_moved = pos.moved_piece(m);
    //     let to = m.history_to_square();
    //     i32::from(table.get_index(conthist_index).get(piece_moved, to))
    // }

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
        let Some(&Undo { cont_hist_index, .. }) = pos.history.last() else {
            return;
        };
        if cont_hist_index.square == Square::NO_SQUARE {
            return;
        }

        let prev_to = cont_hist_index.square;
        let prev_piece = cont_hist_index.piece;

        self.counter_move_table.add(prev_piece, prev_to, m);
    }

    /// Returns the counter move for this position.
    pub fn get_counter_move(&self, pos: &Board) -> Move {
        let Some(&Undo { cont_hist_index, .. }) = pos.history.last() else {
            return Move::NULL;
        };
        if cont_hist_index.square == Square::NO_SQUARE {
            return Move::NULL;
        }

        let prev_to = cont_hist_index.square;
        let prev_piece = cont_hist_index.piece;

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
