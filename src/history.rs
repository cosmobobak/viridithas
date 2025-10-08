use crate::{
    chess::{
        chessmove::Move,
        piece::{Colour, Piece, PieceType},
        squareset::SquareSet,
        types::Square,
    },
    historytable::{
        CORRECTION_HISTORY_MAX, cont_history_bonus, cont_history_malus, main_history_bonus,
        main_history_malus, tactical_history_bonus, tactical_history_malus, update_correction,
        update_history,
    },
    threadlocal::ThreadData,
    util::MAX_DEPTH,
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

        let cmh_block = &mut self.cont_hist[self.ss[height - index - 1].ch_idx];
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

        let cmh_block = &mut self.cont_hist[self.ss[height - index - 1].ch_idx];
        update_history(cmh_block.get_mut(moved, to), delta);
    }

    /// Add a killer move.
    pub fn insert_killer(&mut self, m: Move) {
        debug_assert!(self.board.height() < MAX_DEPTH);
        let idx = self.board.height();
        self.killer_move_table[idx] = Some(m);
    }

    /// Update the correction history for a pawn pattern.
    pub fn update_correction_history(&mut self, depth: i32, tt_complexity: i32, diff: i32) {
        #![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

        use Colour::{Black, White};

        let us = self.board.turn();
        let height = self.board.height();

        // wow! floating point in a chess engine!
        let tt_complexity_factor =
            ((1.0 + (tt_complexity as f32 + 1.0).log2() / 10.0) * 8.0) as i32;

        let bonus = i32::clamp(
            diff * depth * tt_complexity_factor / 64,
            -CORRECTION_HISTORY_MAX / 4,
            CORRECTION_HISTORY_MAX / 4,
        );

        let update = |entry: &mut i16| {
            update_correction(entry, bonus);
        };

        let keys = &self.board.state.keys;

        let pawn = self.pawn_corrhist.get_mut(us, keys.pawn);
        let [nonpawn_white, nonpawn_black] = &mut self.nonpawn_corrhist;
        let nonpawn_white = nonpawn_white.get_mut(us, keys.non_pawn[White]);
        let nonpawn_black = nonpawn_black.get_mut(us, keys.non_pawn[Black]);
        let minor = self.minor_corrhist.get_mut(us, keys.minor);
        let major = self.major_corrhist.get_mut(us, keys.major);

        update(pawn);
        update(nonpawn_white);
        update(nonpawn_black);
        update(minor);
        update(major);

        if height > 2 {
            let ch1 = self.ss[height - 1].ch_idx;
            let ch2 = self.ss[height - 2].ch_idx;
            let pt1 = ch1.piece.piece_type();
            let pt2 = ch2.piece.piece_type();
            update(&mut self.continuation_corrhist[ch1.to][pt1][ch2.to][pt2][us]);
        }
    }

    /// Adjust a raw evaluation using statistics from the correction history.
    #[allow(clippy::cast_possible_truncation)]
    pub fn correction(&self) -> i32 {
        use Colour::{Black, White};

        let keys = &self.board.state.keys;
        let us = self.board.turn();
        let height = self.board.height();

        let pawn = self.pawn_corrhist.get(us, keys.pawn);
        let [white, black] = &self.nonpawn_corrhist;
        let white = white.get(us, keys.non_pawn[White]);
        let black = black.get(us, keys.non_pawn[Black]);
        let minor = self.minor_corrhist.get(us, keys.minor);
        let major = self.major_corrhist.get(us, keys.major);

        let cont = if height > 2 {
            let ch1 = self.ss[height - 1].ch_idx;
            let ch2 = self.ss[height - 2].ch_idx;
            let pt1 = ch1.piece.piece_type();
            let pt2 = ch2.piece.piece_type();
            i64::from(self.continuation_corrhist[ch1.to][pt1][ch2.to][pt2][us])
        } else {
            0
        };

        let adjustment = pawn * i64::from(self.info.conf.pawn_corrhist_weight)
            + major * i64::from(self.info.conf.major_corrhist_weight)
            + minor * i64::from(self.info.conf.minor_corrhist_weight)
            + (white + black) * i64::from(self.info.conf.nonpawn_corrhist_weight)
            + cont * i64::from(self.info.conf.continuation_corrhist_weight);

        (adjustment * 12 / 0x40000) as i32
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
