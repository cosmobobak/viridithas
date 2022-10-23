pub mod parameters;

use std::time::Duration;

use crate::{
    board::movegen::MoveList,
    board::{
        evaluation::{
            self, get_see_value, is_mate_score, mate_in, mated_in, CONTEMPT, MATE_SCORE,
            MINIMUM_MATE_SCORE,
        },
        movegen::{
            bitboards::{self, lsb},
            MoveListEntry, TT_MOVE_SCORE,
        },
        Board,
    },
    chessmove::Move,
    definitions::{
        depth::Depth, depth::ONE_PLY, depth::ZERO_PLY, type_of, BISHOP, INFINITY, KING, MAX_DEPTH,
        PAWN, QUEEN, ROOK,
    },
    searchinfo::SearchInfo,
    threadlocal::ThreadData,
    transpositiontable::{HFlag, ProbeResult},
};

use self::parameters::SearchParams;

// In alpha-beta search, there are three classes of node to be aware of:
// 1. PV-nodes: nodes that end up being within the alpha-beta window,
// i.e. a call to alpha_beta(PVNODE, a, b) returns a value v where v is within the exclusive window (a, b).
// the score returned is an exact score for the node.
// 2. Cut-nodes: nodes that fail high, i.e. a move is found that leads to a value >= beta.
// in alpha-beta, a call to alpha_beta(CUTNODE, alpha, beta) returns a score >= beta.
// The score returned is a lower bound (might be greater) on the exact score of the node
// 3. All-nodes: nodes that fail low, i.e. no move leads to a value > alpha.
// in alpha-beta, a call to alpha_beta(ALLNODE, alpha, beta) returns a score <= alpha.
// Every move at an All-node is searched, and the score returned is an upper bound, so the exact score might be lower.

pub const ASPIRATION_WINDOW: i32 = 26;
const RFP_MARGIN: i32 = 118;
const RFP_IMPROVING_MARGIN: i32 = 76;
const NMP_IMPROVING_MARGIN: i32 = 76;
const SEE_QUIET_MARGIN: i32 = -59;
const SEE_TACTICAL_MARGIN: i32 = -19;
const LMP_BASE_MOVES: i32 = 2;
const FUTILITY_COEFF_2: i32 = 25;
const FUTILITY_COEFF_1: i32 = 27;
const FUTILITY_COEFF_0: i32 = 103;
const RFP_DEPTH: Depth = Depth::new(8);
const NMP_BASE_REDUCTION: Depth = Depth::new(4);
const LMP_DEPTH: Depth = Depth::new(3);
const TT_REDUCTION_DEPTH: Depth = Depth::new(4);
const FUTILITY_DEPTH: Depth = Depth::new(4);
const SINGULARITY_DEPTH: Depth = Depth::new(8);
const SEE_DEPTH: Depth = Depth::new(9);
const LMR_BASE: f64 = 77.0;
const LMR_DIVISION: f64 = 243.0;

impl Board {
    pub fn quiescence<const USE_NNUE: bool>(
        &mut self,
        info: &mut SearchInfo,
        t: &mut ThreadData,
        mut alpha: i32,
        beta: i32,
    ) -> i32 {
        #[cfg(debug_assertions)]
        self.check_validity().unwrap();

        if info.nodes.trailing_zeros() >= 12 {
            info.check_up();
            if info.stopped {
                return 0;
            }
        }

        let height: i32 = self.height().try_into().unwrap();
        info.seldepth = info.seldepth.max(height.into());

        // check draw
        if self.is_draw() {
            return draw_score(info.nodes);
        }

        // are we too deep?
        if height > (MAX_DEPTH - 1).round() {
            return self.evaluate::<USE_NNUE>(t, info.nodes);
        }

        // probe the TT and see if we get a cutoff.
        if let ProbeResult::Cutoff(s) = self.tt_probe::<false>(alpha, beta, ZERO_PLY) {
            return s;
        }

        let stand_pat = self.evaluate::<USE_NNUE>(t, info.nodes);

        if stand_pat >= beta {
            return beta;
        }

        if stand_pat > alpha {
            alpha = stand_pat;
        }

        let original_alpha = alpha;
        let mut best_move = Move::NULL;
        let mut best_score = -INFINITY;

        let mut move_list = MoveList::new();
        self.generate_captures(&mut move_list);

        let mut move_picker = move_list.init_movepicker();
        while let Some(MoveListEntry { entry: m, score: _ }) = move_picker.next() {
            let worst_case =
                self.estimated_see(m) - get_see_value(type_of(self.piece_at(m.from())));

            if !self.make_move_nnue(m, t) {
                continue;
            }
            info.nodes += 1;

            // low-effort SEE pruning - if the worst case is enough to beat beta, just stop.
            // the worst case for a capture is that we lose the capturing piece immediately.
            // as such, worst_case = (SEE of the capture) - (value of the capturing piece).
            // we have to do this after make_move, because the move has to be legal.
            let at_least = stand_pat + worst_case;
            if at_least > beta && !is_mate_score(at_least * 2) {
                self.unmake_move_nnue(t);
                return beta;
            }

            let score = -self.quiescence::<USE_NNUE>(info, t, -beta, -alpha);
            self.unmake_move_nnue(t);

            if score > best_score {
                best_score = score;
                best_move = m;
                if score > alpha {
                    if score >= beta {
                        self.tt_store(best_move, beta, HFlag::LowerBound, ZERO_PLY);
                        return beta;
                    }
                    alpha = score;
                }
            }
        }

        if alpha == original_alpha {
            // we didn't raise alpha, so this is an all-node
            self.tt_store(best_move, alpha, HFlag::UpperBound, ZERO_PLY);
        } else {
            self.tt_store(best_move, best_score, HFlag::Exact, ZERO_PLY);
        }

        alpha
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    pub fn alpha_beta<const PV: bool, const ROOT: bool, const USE_NNUE: bool>(
        &mut self,
        info: &mut SearchInfo,
        t: &mut ThreadData,
        mut depth: Depth,
        mut alpha: i32,
        beta: i32,
    ) -> i32 {
        #[cfg(debug_assertions)]
        self.check_validity().unwrap();
        debug_assert!(self.check_nnue_coherency(&t.nnue));

        if depth <= ZERO_PLY {
            return self.quiescence::<USE_NNUE>(info, t, alpha, beta);
        }

        if info.nodes.trailing_zeros() >= 12 {
            info.check_up();
            if info.stopped {
                return 0;
            }
        }

        let height: i32 = self.height().try_into().unwrap();

        debug_assert_eq!(height == 0, ROOT);

        info.seldepth = if ROOT { ZERO_PLY } else { info.seldepth.max(height.into()) };

        if !ROOT {
            // check draw
            if self.is_draw() {
                return draw_score(info.nodes);
            }

            // are we too deep?
            if height > MAX_DEPTH.round() - 1 {
                return self.evaluate::<USE_NNUE>(t, info.nodes);
            }

            // mate-distance pruning.
            // doesn't actually add strength, but it makes viri better at solving puzzles.
            // approach taken from Ethereal.
            let r_alpha = alpha.max(mated_in(height));
            let r_beta = beta.min(mate_in(height + 1));
            if r_alpha >= r_beta {
                return r_alpha;
            }
        }

        debug_assert_eq!(PV, alpha + 1 != beta, "PV must be true if the alpha-beta window is larger than 1, but PV was {PV} and alpha-beta window was {alpha}-{beta}");

        let excluded = t.excluded[self.height()];

        let tt_hit = if excluded.is_null() {
            match self.tt_probe::<ROOT>(alpha, beta, depth) {
                ProbeResult::Cutoff(s) => {
                    return s;
                }
                ProbeResult::Hit(tt_hit) => Some(tt_hit),
                ProbeResult::Nothing => {
                    // TT-reduction.
                    if PV && depth >= self.sparams.tt_reduction_depth {
                        depth -= 1;
                    }
                    None
                }
            }
        } else {
            None // do not probe the TT if we're in a singular-verification search.
        };

        // just enforcing immutability here.
        let depth = depth;

        let in_check = self.in_check::<{ Self::US }>();

        let static_eval = if in_check {
            INFINITY // when we're in check, it could be checkmate, so it's unsound to use evaluate().
        } else {
            self.evaluate::<USE_NNUE>(t, info.nodes)
        };

        t.evals[self.height()] = static_eval;

        // "improving" is true when the current position has a better static evaluation than the one from a fullmove ago.
        // if a position is "improving", we can be more aggressive with beta-reductions (eval is too high),
        // but we should be less agressive with alpha-reductions (eval is too low).
        // some engines gain by using improving to increase LMR, but this shouldn't work imo, given that LMR is
        // neutral with regards to the evaluation.
        let improving =
            !in_check && self.height() >= 2 && static_eval >= t.evals[self.height() - 2];

        // beta-pruning. (reverse futility pruning)
        if !PV
            && !in_check
            && excluded.is_null()
            && depth <= self.sparams.rfp_depth
            && static_eval - self.sparams.rfp_margin * depth
                + i32::from(improving) * self.sparams.rfp_improving_margin
                > beta
        {
            return static_eval;
        }

        // null-move pruning.
        if !PV
            && !in_check
            && excluded.is_null()
            && static_eval + i32::from(improving) * self.sparams.nmp_improving_margin >= beta
            && depth >= 3.into()
            && !self.last_move_was_nullmove()
            && self.zugzwang_unlikely()
        {
            let nm_depth = (depth - self.sparams.nmp_base_reduction) - (depth / 3 - 1);
            self.make_nullmove();
            let score = -self.alpha_beta::<PV, false, USE_NNUE>(info, t, nm_depth, -beta, -beta + 1);
            self.unmake_nullmove();
            if info.stopped {
                return 0;
            }
            if score >= beta {
                return beta;
            }
        }

        let mut move_list = MoveList::new();
        self.generate_moves(&mut move_list);

        let original_alpha = alpha;
        let mut best_move = Move::NULL;
        let mut best_score = -INFINITY;
        let mut moves_made = 0;
        let mut quiet_moves_made = 0;

        let imp_2x = 1 + i32::from(improving);
        // number of quiet moves to try before we start pruning
        let lmp_threshold = (self.sparams.lmp_base_moves + depth.squared()) * imp_2x;
        // whether late move pruning is sound in this position.
        let do_lmp = !PV && !ROOT && depth <= self.sparams.lmp_depth && !in_check;
        // whether to skip quiet moves (as they would be futile).
        let do_fut_pruning = self.do_futility_pruning(depth, static_eval, alpha, beta);

        if let Some(tt_hit) = &tt_hit {
            if let Some(movelist_entry) = move_list.lookup_by_move(tt_hit.tt_move) {
                // set the move-ordering score of the TT-move to a very high value.
                movelist_entry.score = TT_MOVE_SCORE;
            }
        }

        let see_table = [
            self.sparams.see_tactical_margin * depth.squared(),
            self.sparams.see_quiet_margin * depth.round(),
        ];

        let mut move_picker = move_list.init_movepicker();
        if ROOT {
            if depth == Depth::new(1) {
                t.root_moves = self.legal_moves().into_iter().map(|m| (m, 0)).collect();
            }
            move_picker.score_by(&t.root_moves);
        }
        let mut root_nodecount_record = Vec::new();
        while let Some(MoveListEntry { entry: m, score: ordering_score }) = move_picker.next() {
            if ordering_score < 0 && depth < Depth::new(5) {
                move_picker.skip_ordering();
            }

            if best_score > -MINIMUM_MATE_SCORE
                && depth <= self.sparams.see_depth
                && !self.static_exchange_eval(m, see_table[usize::from(m.is_quiet())])
            {
                continue;
            }

            if !self.make_move_nnue(m, t) {
                continue;
            }
            info.nodes += 1;
            if excluded == m {
                self.unmake_move_nnue(t);
                continue;
            }
            moves_made += 1;
            if ROOT && info.print_to_stdout && info.time_since_start() > Duration::from_secs(5) {
                println!("info currmove {m} currmovenumber {} nodes {}", moves_made, info.nodes);
            }

            let is_capture = m.is_capture();
            let gives_check = self.in_check::<{ Self::US }>();
            let is_promotion = m.is_promo();

            let is_interesting = is_capture || is_promotion || gives_check || in_check;
            quiet_moves_made += i32::from(!is_interesting);

            if do_lmp && quiet_moves_made >= lmp_threshold {
                self.unmake_move_nnue(t);
                break; // okay to break because captures are ordered first.
            }

            // futility pruning
            // if the static eval is too low, we might just skip the move.
            if !(PV || is_capture || is_promotion || in_check || moves_made <= 1) && do_fut_pruning
            {
                self.unmake_move_nnue(t);
                continue;
            }

            let maybe_singular = tt_hit.as_ref().map_or(false, |tt_hit| {
                !ROOT
                    && depth >= self.sparams.singularity_depth
                    && tt_hit.tt_move == m
                    && excluded.is_null()
                    && tt_hit.tt_depth >= depth - 3
                    && tt_hit.tt_bound == HFlag::LowerBound
            });

            let mut extension = ZERO_PLY;
            if !ROOT && maybe_singular {
                let tt_value = tt_hit.as_ref().unwrap().tt_value;
                let is_singular = self.is_singular::<USE_NNUE>(info, t, m, tt_value, depth);
                extension = Depth::from(is_singular);
            } else if !ROOT {
                extension = Depth::from(gives_check);
            };

            let nodes_before = info.nodes;
            let mut score;
            if moves_made == 1 {
                // first move (presumably the PV-move)
                score =
                    -self.alpha_beta::<PV, false, USE_NNUE>(info, t, depth + extension - 1, -beta, -alpha);
            } else {
                // calculation of LMR stuff
                let r = if extension == ZERO_PLY
                    && m.is_quiet()
                    && depth >= 3.into()
                    && moves_made >= (2 + usize::from(PV))
                {
                    let mut r = self.lmr_table.get(depth, moves_made);
                    r += i32::from(!PV);
                    Depth::new(r).clamp(ONE_PLY, depth - 1)
                } else {
                    ONE_PLY
                };
                // perform a zero-window search
                score = -self.alpha_beta::<false, false, USE_NNUE>(
                    info,
                    t,
                    depth + extension - r,
                    -alpha - 1,
                    -alpha,
                );
                // if we failed, then full window search
                if score > alpha && score < beta {
                    // this is a new best move, so it *is* PV.
                    score = -self.alpha_beta::<PV, false, USE_NNUE>(
                        info,
                        t,
                        depth + extension - 1,
                        -beta,
                        -alpha,
                    );
                }
            }
            self.unmake_move_nnue(t);
            let nodes_used = info.nodes - nodes_before;
            if ROOT {
                root_nodecount_record.push((m, nodes_used));
            }

            if info.stopped {
                return 0;
            }

            if score > best_score {
                best_score = score;
                best_move = m;
                if score > alpha {
                    alpha = score;
                    if score >= beta {
                        // we failed high, so this is a cut-node

                        // record move ordering stats:
                        if moves_made == 1 {
                            info.failhigh_first += 1;
                        }
                        info.failhigh += 1;

                        if best_move.is_quiet() {
                            self.insert_killer(best_move);
                            self.insert_countermove(best_move);
                            self.update_history_metrics::<true>(best_move, depth);

                            // decrease the history of the non-capture moves that came before the cutoff move.
                            let ms = move_picker.moves_made();
                            for e in ms.iter().filter(|e| e.entry.is_quiet()) {
                                self.update_history_metrics::<false>(e.entry, depth);
                            }
                        }

                        if excluded.is_null() {
                            self.tt_store(best_move, beta, HFlag::LowerBound, depth);
                        }

                        if ROOT {
                            t.order_root_moves(&root_nodecount_record);
                        }
                        return beta;
                    }
                }
            }
        }

        if moves_made == 0 {
            if !excluded.is_null() {
                return alpha;
            }
            if in_check {
                return mated_in(height);
            }
            return draw_score(info.nodes);
        }

        if alpha == original_alpha {
            // we didn't raise alpha, so this is an all-node
            if excluded.is_null() {
                self.tt_store(best_move, alpha, HFlag::UpperBound, depth);
            }
        } else {
            // we raised alpha, and didn't raise beta
            // as if we had, we would have returned early,
            // so this is a PV-node
            if best_move.is_quiet() {
                self.insert_killer(best_move);
                self.insert_countermove(best_move);
                self.update_history_metrics::<true>(best_move, depth);

                // decrease the history of the non-capture moves that came before the best move.
                let ms = move_picker.moves_made();
                for e in
                    ms.iter().take_while(|m| m.entry != best_move).filter(|e| e.entry.is_quiet())
                {
                    self.update_history_metrics::<false>(e.entry, depth);
                }
            }

            if excluded.is_null() {
                self.tt_store(best_move, best_score, HFlag::Exact, depth);
            }
        }

        if ROOT {
            t.order_root_moves(&root_nodecount_record);
        }
        alpha
    }

    fn update_history_metrics<const IS_GOOD: bool>(&mut self, m: Move, depth: Depth) {
        self.add_history::<IS_GOOD>(m, depth);
        self.add_followup_history::<IS_GOOD>(m, depth);
    }

    pub fn is_singular<const USE_NNUE: bool>(
        &mut self,
        info: &mut SearchInfo,
        t: &mut ThreadData,
        m: Move,
        tt_value: i32,
        depth: Depth,
    ) -> bool {
        let reduced_beta = (tt_value - 3 * depth.round()).max(-MATE_SCORE);
        let reduced_depth = (depth - 1) / 2;
        // undo the singular move so we can search the position that it exists in.
        self.unmake_move_nnue(t);
        t.excluded[self.height()] = m;
        let value =
            self.alpha_beta::<false, false, USE_NNUE>(info, t, reduced_depth, reduced_beta - 1, reduced_beta);
        t.excluded[self.height()] = Move::NULL;
        // re-make the singular move.
        self.make_move_nnue(m, t);
        value < reduced_beta
    }

    /// function called at root to modify time control if only one move is good.
    pub fn is_forced<const MARGIN: i32>(
        &mut self,
        info: &mut SearchInfo,
        t: &mut ThreadData,
        m: Move,
        value: i32,
        depth: Depth,
    ) -> bool {
        let reduced_beta = (value - MARGIN).max(-MATE_SCORE);
        let reduced_depth = (depth - 1) / 2;
        t.excluded[self.height()] = m;
        let pts_prev = info.print_to_stdout;
        info.print_to_stdout = false;
        let value =
            self.alpha_beta::<false, true, true>(info, t, reduced_depth, reduced_beta - 1, reduced_beta);
        info.print_to_stdout = pts_prev;
        t.excluded[self.height()] = Move::NULL;
        value < reduced_beta
    }

    pub fn static_exchange_eval(&self, m: Move, threshold: i32) -> bool {
        let from = m.from();
        let to = m.to();

        let mut next_victim =
            if m.is_promo() { type_of(m.promotion()) } else { type_of(self.piece_at(from)) };

        let mut balance = self.estimated_see(m) - threshold;

        // if the best case fails, don't bother doing the full search.
        if balance < 0 {
            return false;
        }

        // worst case is losing the piece
        balance -= get_see_value(next_victim);

        // if the worst case passes, we can return true immediately.
        if balance >= 0 {
            return true;
        }

        let diag_sliders = self.pieces.bishopqueen::<true>() | self.pieces.bishopqueen::<false>();
        let orth_sliders = self.pieces.rookqueen::<true>() | self.pieces.rookqueen::<false>();

        // occupied starts with the position after the move `m` is made.
        let mut occupied = (self.pieces.occupied() ^ (1 << from.index())) | (1 << to.index());
        if m.is_ep() {
            occupied ^= 1 << self.ep_sq().index();
        }

        let mut attackers = self.pieces.all_attackers_to_sq(to, occupied) & occupied;

        // after the move, it's the opponent's turn.
        let mut colour = self.turn() ^ 1;

        loop {
            let my_attackers = attackers & self.pieces.occupied_co(colour);
            if my_attackers == 0 {
                break;
            }

            // find cheapest attacker
            for victim in PAWN..=KING {
                next_victim = victim;
                if my_attackers & self.pieces.of_type(victim) != 0 {
                    break;
                }
            }

            occupied ^= 1 << lsb(my_attackers & self.pieces.of_type(next_victim));

            // diagonal moves reveal bishops and queens:
            if next_victim == PAWN || next_victim == BISHOP || next_victim == QUEEN {
                attackers |= bitboards::attacks::<BISHOP>(to, occupied) & diag_sliders;
            }

            // orthogonal moves reveal rooks and queens:
            if next_victim == ROOK || next_victim == QUEEN {
                attackers |= bitboards::attacks::<ROOK>(to, occupied) & orth_sliders;
            }

            attackers &= occupied;

            colour ^= 1;

            balance = -balance - 1 - get_see_value(next_victim);

            if balance >= 0 {
                // from Ethereal:
                // As a slight optimisation for move legality checking, if our last attacking
                // piece is a king, and our opponent still has attackers, then we've
                // lost as the move we followed would be illegal
                if next_victim == KING && attackers & self.pieces.occupied_co(colour) != 0 {
                    colour ^= 1;
                }
                break;
            }
        }

        // the side that is to move after loop exit is the loser.
        self.turn() != colour
    }

    fn do_futility_pruning(&self, depth: Depth, static_eval: i32, a: i32, b: i32) -> bool {
        if depth > self.sparams.futility_depth || is_mate_score(a) || is_mate_score(b) {
            return false;
        }
        let depth = depth.round();
        let margin = depth * depth * self.sparams.futility_coeff_2
            + depth * self.sparams.futility_coeff_1
            + self.sparams.futility_coeff_0;
        static_eval + margin < a
    }
}

pub const fn draw_score(nodes: u64) -> i32 {
    // score fuzzing apparently helps with threefolds.
    -CONTEMPT + 2 - (nodes & 0b111) as i32
}

pub struct LMRTable {
    table: [[i32; 64]; 64],
}

impl LMRTable {
    pub fn new(config: &SearchParams) -> Self {
        #![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        let mut out = Self { table: [[0; 64]; 64] };
        let (base, division) = (config.lmr_base / 100.0, config.lmr_division / 100.0);
        for depth in 1..64 {
            for played in 1..64 {
                let ld = f64::ln(depth as f64);
                let lp = f64::ln(played as f64);
                out.table[depth][played] = (base + ld * lp / division) as i32;
            }
        }
        out
    }

    pub fn get(&self, depth: Depth, moves_made: usize) -> i32 {
        let depth = depth.ply_to_horizon().min(63);
        let played = moves_made.min(63);
        self.table[depth][played]
    }
}

pub struct AspirationWindow {
    pub midpoint: i32,
    pub alpha: i32,
    pub beta: i32,
    pub alpha_fails: i32,
    pub beta_fails: i32,
}

impl AspirationWindow {
    pub const fn new() -> Self {
        Self { alpha: -INFINITY, beta: INFINITY, midpoint: 0, alpha_fails: 0, beta_fails: 0 }
    }

    pub const fn from_last_score(last_score: i32) -> Self {
        if is_mate_score(last_score) {
            Self {
                midpoint: last_score,
                alpha: -INFINITY,
                beta: INFINITY,
                alpha_fails: 0,
                beta_fails: 0,
            }
        } else {
            Self {
                midpoint: last_score,
                alpha: last_score - ASPIRATION_WINDOW,
                beta: last_score + ASPIRATION_WINDOW,
                alpha_fails: 0,
                beta_fails: 0,
            }
        }
    }

    pub fn widen_down(&mut self) {
        let margin = ASPIRATION_WINDOW << (self.alpha_fails + 1);
        if margin > evaluation::QUEEN_VALUE.0 {
            self.alpha = -INFINITY;
            return;
        }
        self.alpha = self.midpoint - margin;
        self.alpha_fails += 1;
    }

    pub fn widen_up(&mut self) {
        let margin = ASPIRATION_WINDOW << (self.beta_fails + 1);
        if margin > evaluation::QUEEN_VALUE.0 {
            self.beta = INFINITY;
            return;
        }
        self.beta = self.midpoint + margin;
        self.beta_fails += 1;
    }
}
