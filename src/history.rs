use crate::{
    chess::{
        chessmove::Move,
        piece::{Colour, Piece, PieceType},
        squareset::SquareSet,
        types::Square,
    },
    historytable::{
        CORRECTION_HISTORY_MAX, HASH_HISTORY_SIZE, history_bonus, history_malus,
        update_cont_history, update_correction, update_history,
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
                history_bonus(&self.info.conf.main_history, depth)
            } else {
                -history_malus(&self.info.conf.main_history, depth)
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
        depth: i32,
        good: bool,
    ) {
        let delta = if good {
            history_bonus(&self.info.conf.main_history, depth)
        } else {
            -history_malus(&self.info.conf.main_history, depth)
        };
        let val = self.main_hist.get_mut(
            moved,
            to,
            threats.contains_square(from),
            threats.contains_square(to),
        );
        update_history(val, delta);
    }

    /// Update the pawn-structure history counters of a batch of moves.
    pub fn update_pawn_history(&mut self, moves_to_adjust: &[Move], best_move: Move, depth: i32) {
        let pawn_index = self.board.state.keys.pawn % HASH_HISTORY_SIZE as u64;
        for &m in moves_to_adjust {
            let piece_moved = self.board.state.mailbox[m.from()].unwrap();
            let to = m.history_to_square();
            #[expect(clippy::cast_possible_truncation)]
            let val = &mut self.pawn_hist[pawn_index as usize][piece_moved][to];
            let delta = if m == best_move {
                history_bonus(&self.info.conf.pawn_history, depth)
            } else {
                -history_malus(&self.info.conf.pawn_history, depth)
            };
            update_history(val, delta);
        }
    }

    /// Update the pawn-structure history counter for a single move.
    pub fn update_pawn_history_single(&mut self, to: Square, moved: Piece, depth: i32, good: bool) {
        let delta = if good {
            history_bonus(&self.info.conf.pawn_history, depth)
        } else {
            -history_malus(&self.info.conf.pawn_history, depth)
        };
        let pawn_index = self.board.state.keys.pawn % HASH_HISTORY_SIZE as u64;
        #[expect(clippy::cast_possible_truncation)]
        let val = &mut self.pawn_hist[pawn_index as usize][moved][to];
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
                history_bonus(&self.info.conf.tactical_history, depth)
            } else {
                -history_malus(&self.info.conf.tactical_history, depth)
            };
            update_history(val, delta);
        }
    }

    /// Update the continuation history counters of a batch of moves.
    pub fn update_cont_hist(&mut self, moves_to_adjust: &[Move], best_move: Move, depth: i32) {
        let height = self.board.height();

        for &m in moves_to_adjust {
            let good = m == best_move;
            let to = m.history_to_square();
            let piece = self.board.state.mailbox[m.from()].unwrap();
            self.update_cont_hist_single(to, piece, depth, height, good);
        }
    }

    pub fn update_cont_hist_single(
        &mut self,
        to: Square,
        piece: Piece,
        depth: i32,
        height: usize,
        good: bool,
    ) {
        let indexed_multipliers = [
            (1, self.info.conf.cont1_stat_score_mul),
            (2, self.info.conf.cont2_stat_score_mul),
            (3, self.info.conf.cont3_stat_score_mul),
            (4, self.info.conf.cont4_stat_score_mul),
            (5, self.info.conf.cont5_stat_score_mul),
            (6, self.info.conf.cont6_stat_score_mul),
        ];

        let boni = [
            history_bonus(&self.info.conf.cont1_history, depth),
            history_bonus(&self.info.conf.cont2_history, depth),
            history_bonus(&self.info.conf.cont3_history, depth),
            history_bonus(&self.info.conf.cont4_history, depth),
            history_bonus(&self.info.conf.cont5_history, depth),
            history_bonus(&self.info.conf.cont6_history, depth),
        ];

        let mali = [
            -history_malus(&self.info.conf.cont1_history, depth),
            -history_malus(&self.info.conf.cont2_history, depth),
            -history_malus(&self.info.conf.cont3_history, depth),
            -history_malus(&self.info.conf.cont4_history, depth),
            -history_malus(&self.info.conf.cont5_history, depth),
            -history_malus(&self.info.conf.cont6_history, depth),
        ];

        let adjustments = if good { boni } else { mali };

        let mut sum = 0;
        for (index, mul) in indexed_multipliers {
            if height < index {
                break;
            }
            sum += mul * i32::from(self.cont_hist[self.ss[height - index].ch_idx][piece][to]);
        }
        sum /= 32;

        for ((index, _), delta) in indexed_multipliers.into_iter().zip(adjustments) {
            if height < index {
                break;
            }
            let val = &mut self.cont_hist[self.ss[height - index].ch_idx][piece][to];
            update_cont_history(val, sum, delta);
        }
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

        let keys = &self.board.state.keys;

        let pawn = self.pawn_corrhist.get_mut(us, keys.pawn);
        let [nonpawn_white, nonpawn_black] = &mut self.nonpawn_corrhist;
        let nonpawn_white = nonpawn_white.get_mut(us, keys.non_pawn[White]);
        let nonpawn_black = nonpawn_black.get_mut(us, keys.non_pawn[Black]);
        let minor = self.minor_corrhist.get_mut(us, keys.minor);
        let major = self.major_corrhist.get_mut(us, keys.major);

        let update = move |entry: &mut i16| {
            update_correction(entry, bonus);
        };

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
