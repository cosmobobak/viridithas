use crate::{
    chess::{
        board::Board,
        chessmove::Move,
        piece::{Piece, PieceType},
        squareset::SquareSet,
        types::Square,
    },
    historytable::{
        HASH_HISTORY_SIZE, cont_history_bonus, cont_history_malus, main_history_bonus,
        main_history_malus, pawn_history_bonus, pawn_history_malus, tactical_history_bonus,
        tactical_history_malus, update_cont_history, update_history,
    },
    threadlocal::ThreadData,
    util::MAX_DEPTH,
};

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
        depth: i32,
        good: bool,
    ) {
        let delta = if good {
            main_history_bonus(&self.info.conf, depth)
        } else {
            -main_history_malus(&self.info.conf, depth)
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
                pawn_history_bonus(&self.info.conf, depth)
            } else {
                -pawn_history_malus(&self.info.conf, depth)
            };
            update_history(val, delta);
        }
    }

    /// Update the pawn-structure history counter for a single move.
    pub fn update_pawn_history_single(&mut self, to: Square, moved: Piece, depth: i32, good: bool) {
        let delta = if good {
            pawn_history_bonus(&self.info.conf, depth)
        } else {
            -pawn_history_malus(&self.info.conf, depth)
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
                tactical_history_bonus(&self.info.conf, depth)
            } else {
                -tactical_history_malus(&self.info.conf, depth)
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
        let list = [
            (1, self.info.conf.cont1_stat_score_mul),
            (2, self.info.conf.cont2_stat_score_mul),
            (4, self.info.conf.cont4_stat_score_mul),
        ];

        let mut sum = 0;
        for (index, mul) in list {
            if height < index {
                break;
            }
            sum += mul * i32::from(self.cont_hist[self.ss[height - index].ch_idx][piece][to]);
        }
        sum /= 32;

        for (index, _) in list {
            if height < index {
                break;
            }
            let delta = if good {
                cont_history_bonus(&self.info.conf, depth, index)
            } else {
                -cont_history_malus(&self.info.conf, depth, index)
            };
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

    /// Update the correction history for a position.
    pub fn update_correction_history(&self, depth: i32, tt_complexity: i32, diff: i32) {
        let height = self.board.height();
        let cont_indices = if height > 2 {
            Some((self.ss[height - 1].ch_idx, self.ss[height - 2].ch_idx))
        } else {
            None
        };
        self.corrhists.update(
            &self.board.state.keys,
            self.board.turn(),
            cont_indices,
            depth,
            tt_complexity,
            diff,
        );
    }

    /// Compute the correction history adjustment for a position.
    pub fn correction(&self) -> i32 {
        let height = self.board.height();
        let cont_indices = if height > 2 {
            Some((self.ss[height - 1].ch_idx, self.ss[height - 2].ch_idx))
        } else {
            None
        };
        self.corrhists.correction(
            &self.board.state.keys,
            self.board.turn(),
            cont_indices,
            &self.info.conf,
        )
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
