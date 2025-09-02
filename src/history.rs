use crate::{
    chess::{
        chessmove::Move,
        piece::{Colour, Piece, PieceType},
        squareset::SquareSet,
        types::Square,
    },
    historytable::{
        cont_history_bonus, cont_history_malus, main_history_bonus, main_history_malus,
        tactical_history_bonus, tactical_history_malus, update_history, CORRECTION_HISTORY_GRAIN,
        CORRECTION_HISTORY_MAX, CORRECTION_HISTORY_WEIGHT_SCALE,
    },
    threadlocal::ThreadData,
    util::MAX_PLY,
};

use crate::chess::board::Board;

impl ThreadData<'_> {
    /// Update the history counters of a batch of moves.
    pub fn update_history(&mut self, moves_to_adjust: &[Move], best_move: Move, depth: i32) {
        let threats = self.board.state.threats.all;
        for &m in moves_to_adjust {
            let from = m.from();
            let piece_moved = self.board.state.mailbox[from];
            let to = m.history_to_square();
            let val = self.main_hist.get_mut(
                piece_moved.unwrap(),
                to,
                threats.contains_square(from),
                threats.contains_square(to),
            );
            let delta = if m == best_move {
                main_history_bonus(&self.info.conf, depth)
            } else {
                -main_history_malus(&self.info.conf, depth)
            };
            update_history(val, delta);
        }
    }

    /// Update the history counters for a single move.
    pub fn update_history_single(
        &mut self,
        from: Square,
        to: Square,
        moved: Piece,
        threats: SquareSet,
        delta: i32,
    ) {
        let val = self.main_hist.get_mut(
            moved,
            to,
            threats.contains_square(from),
            threats.contains_square(to),
        );
        update_history(val, delta);
    }

    /// Update the tactical history counters of a batch of moves.
    pub fn update_tactical_history(
        &mut self,
        moves_to_adjust: &[Move],
        best_move: Move,
        depth: i32,
    ) {
        let threats = self.board.state.threats.all;
        for &m in moves_to_adjust {
            let piece_moved = self.board.state.mailbox[m.from()].unwrap();
            let capture = caphist_piece_type(&self.board, m);
            let to = m.to();
            let to_threat = threats.contains_square(to);
            let val = &mut self.tactical_hist[usize::from(to_threat)][capture][piece_moved][to];
            let delta = if m == best_move {
                tactical_history_bonus(&self.info.conf, depth)
            } else {
                -tactical_history_malus(&self.info.conf, depth)
            };
            update_history(val, delta);
        }
    }

    /// Update the continuation history counters of a batch of moves.
    pub fn update_continuation_history(
        &mut self,
        moves_to_adjust: &[Move],
        best_move: Move,
        depth: i32,
        index: usize,
    ) {
        let height = self.board.height();
        if height <= index {
            return;
        }
        let Some(ss) = self.ss.get(height - index - 1) else {
            return;
        };
        let cmh_block = self.cont_hist.get_index_mut(ss.ch_idx);
        for &m in moves_to_adjust {
            let to = m.history_to_square();
            let piece = self.board.state.mailbox[m.from()].unwrap();

            let delta = if m == best_move {
                cont_history_bonus(&self.info.conf, depth, index)
            } else {
                -cont_history_malus(&self.info.conf, depth, index)
            };
            update_history(cmh_block.get_mut(piece, to), delta);
        }
    }

    /// Update the continuation history counter for a single move.
    pub fn update_continuation_history_single(
        &mut self,
        to: Square,
        moved: Piece,
        delta: i32,
        index: usize,
    ) {
        let height = self.board.height();
        if height <= index {
            return;
        }
        let Some(ss) = self.ss.get(height - index - 1) else {
            return;
        };
        let cmh_block = self.cont_hist.get_index_mut(ss.ch_idx);
        update_history(cmh_block.get_mut(moved, to), delta);
    }

    /// Add a killer move.
    pub fn insert_killer(&mut self, m: Move) {
        debug_assert!(self.board.height() < MAX_PLY);
        let idx = self.board.height();
        self.killer_move_table[idx] = Some(m);
    }

    /// Update the correction history for a pawn pattern.
    pub fn update_correction_history(&mut self, depth: i32, diff: i32) {
        use Colour::{Black, White};
        fn update(entry: &mut i16, new_weight: i32, scaled_diff: i32) {
            #![allow(clippy::cast_possible_truncation)]
            let update = i32::from(*entry) * (CORRECTION_HISTORY_WEIGHT_SCALE - new_weight)
                + scaled_diff * new_weight;
            *entry = i32::clamp(
                update / CORRECTION_HISTORY_WEIGHT_SCALE,
                -CORRECTION_HISTORY_MAX,
                CORRECTION_HISTORY_MAX,
            ) as i16;
        }
        let scaled_diff = diff * CORRECTION_HISTORY_GRAIN;
        let new_weight = 16.min(1 + depth);
        debug_assert!(new_weight <= CORRECTION_HISTORY_WEIGHT_SCALE);
        let us = self.board.turn();

        let keys = &self.board.state.keys;

        update(
            self.pawn_corrhist.get_mut(us, keys.pawn),
            new_weight,
            scaled_diff,
        );
        update(
            self.nonpawn_corrhist[White].get_mut(us, keys.non_pawn[White]),
            new_weight,
            scaled_diff,
        );
        update(
            self.nonpawn_corrhist[Black].get_mut(us, keys.non_pawn[Black]),
            new_weight,
            scaled_diff,
        );
        update(
            self.minor_corrhist.get_mut(us, keys.minor),
            new_weight,
            scaled_diff,
        );
        update(
            self.major_corrhist.get_mut(us, keys.major),
            new_weight,
            scaled_diff,
        );
    }

    /// Adjust a raw evaluation using statistics from the correction history.
    #[allow(clippy::cast_possible_truncation)]
    pub fn correction(&self) -> i32 {
        let keys = &self.board.state.keys;
        let turn = self.board.turn();
        let pawn = self.pawn_corrhist.get(turn, keys.pawn);
        let white = self.nonpawn_corrhist[Colour::White].get(turn, keys.non_pawn[Colour::White]);
        let black = self.nonpawn_corrhist[Colour::Black].get(turn, keys.non_pawn[Colour::Black]);
        let minor = self.minor_corrhist.get(turn, keys.minor);
        let major = self.major_corrhist.get(turn, keys.major);
        let adjustment = pawn * i64::from(self.info.conf.pawn_corrhist_weight)
            + major * i64::from(self.info.conf.major_corrhist_weight)
            + minor * i64::from(self.info.conf.minor_corrhist_weight)
            + (white + black) * i64::from(self.info.conf.nonpawn_corrhist_weight);
        (adjustment / 1024) as i32 / CORRECTION_HISTORY_GRAIN
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
        debug_assert!(!mv.is_castle(), "shouldn't be using caphist for castling.");
        pos.state.mailbox[mv.to()]
            .expect("you weren't capturing anything!")
            .piece_type()
    }
}
