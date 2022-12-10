pub mod parameters;

use std::time::Duration;

use crate::{
    board::{
        evaluation::{
            self, get_see_value, is_mate_score, mate_in, mated_in, MATE_SCORE, MINIMUM_MATE_SCORE,
        },
        movegen::{
            bitboards::{self, lsb},
            movepicker::MovePicker,
            MoveListEntry, MAX_POSITION_MOVES,
        },
        Board,
    },
    chessmove::Move,
    definitions::{
        depth::Depth, depth::ONE_PLY, depth::ZERO_PLY, type_of, BISHOP, INFINITY, KING, MAX_DEPTH,
        PAWN, QUEEN, ROOK, StaticVec,
    },
    searchinfo::SearchInfo,
    threadlocal::ThreadData,
    transpositiontable::{HFlag, ProbeResult}, uci,
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
const RFP_MARGIN: i32 = 80;
const RFP_IMPROVING_MARGIN: i32 = 57;
const NMP_IMPROVING_MARGIN: i32 = 76;
const SEE_QUIET_MARGIN: i32 = -59;
const SEE_TACTICAL_MARGIN: i32 = -19;
const LMP_BASE_MOVES: i32 = 2;
const FUTILITY_COEFF_1: i32 = 90;
const FUTILITY_COEFF_0: i32 = 80;
const RFP_DEPTH: Depth = Depth::new(8);
const NMP_BASE_REDUCTION: Depth = Depth::new(4);
const NMP_VERIFICATION_DEPTH: Depth = Depth::new(8);
const LMP_DEPTH: Depth = Depth::new(8);
const TT_REDUCTION_DEPTH: Depth = Depth::new(4);
const FUTILITY_DEPTH: Depth = Depth::new(6);
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
            // return stand_pat instead of beta, this is fail-soft
            return stand_pat;
        }

        let original_alpha = alpha;
        if stand_pat > alpha {
            alpha = stand_pat;
        }

        let mut best_move = Move::NULL;
        let mut best_score = stand_pat;

        let killers = self.get_killer_set(t);

        let mut move_picker = MovePicker::<true, true>::new(Move::NULL, killers);
        while let Some(MoveListEntry { mov: m, score: _ }) = move_picker.next(self, t) {
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
                // don't bother failing soft, at_least is not really trustworthy.
                return beta;
            }

            let score = -self.quiescence::<USE_NNUE>(info, t, -beta, -alpha);
            self.unmake_move_nnue(t);

            if score > best_score {
                best_score = score;
                best_move = m;
                if score > alpha {
                    alpha = score;
                }
                if alpha >= beta {
                    break; // fail-high
                }
            }
        }

        let flag = if best_score >= beta {
            HFlag::LowerBound
        } else if best_score > original_alpha {
            HFlag::Exact
        } else {
            HFlag::UpperBound
        };

        self.tt_store::<false>(best_move, best_score, flag, ZERO_PLY);

        best_score
    }

    fn get_killer_set(&self, t: &mut ThreadData) -> [Move; 3] {
        let curr_killers = t.killer_move_table[self.height()];
        let prev_killer =
            if self.height() > 2 { t.killer_move_table[self.height() - 2][0] } else { Move::NULL };
        [curr_killers[0], curr_killers[1], prev_killer]
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

        let in_check = self.in_check::<{ Self::US }>();
        if depth <= ZERO_PLY && !in_check {
            return self.quiescence::<USE_NNUE>(info, t, alpha, beta);
        }
        depth = depth.max(ZERO_PLY);

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
                return if in_check { 0 } else { self.evaluate::<USE_NNUE>(t, info.nodes) };
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

        debug_assert_eq!(PV, alpha + 1 != beta, "PV must be true iff the alpha-beta window is larger than 1, but PV was {PV} and alpha-beta window was {alpha}-{beta}");

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

        let last_move_was_null = self.last_move_was_nullmove();

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
            && static_eval - self.rfp_margin(depth, improving) > beta
        {
            return static_eval;
        }

        // null-move pruning.
        if !PV
            && !in_check
            && !last_move_was_null
            && excluded.is_null()
            && static_eval + i32::from(improving) * self.sparams.nmp_improving_margin >= beta
            && depth >= 3.into()
            && self.zugzwang_unlikely()
        {
            let nm_depth = (depth - self.sparams.nmp_base_reduction) - (depth / 3 - 1);
            self.make_nullmove();
            let score =
                -self.alpha_beta::<PV, false, USE_NNUE>(info, t, nm_depth, -beta, -beta + 1);
            self.unmake_nullmove();
            if info.stopped {
                return 0;
            }
            if score >= beta {
                return beta;
            }
        }

        let original_alpha = alpha;
        let mut best_move = Move::NULL;
        let mut best_score = -INFINITY;
        let mut moves_made = 0;

        // number of quiet moves to try before we start pruning
        let lmp_threshold = self.lmr_table.getp(depth, improving);

        let see_table = [
            self.sparams.see_tactical_margin * depth.squared(),
            self.sparams.see_quiet_margin * depth.round(),
        ];

        let killers = self.get_killer_set(t);
        let tt_move = tt_hit.as_ref().map_or(Move::NULL, |hit| hit.tt_move);

        let mut move_picker = MovePicker::<false, true>::new(tt_move, killers);

        let mut quiets_tried = StaticVec::<Move, MAX_POSITION_MOVES>::new_from_default(Move::NULL);
        let mut tacticals_tried = StaticVec::<Move, MAX_POSITION_MOVES>::new_from_default(Move::NULL);
        while let Some(MoveListEntry { mov: m, score: ordering_score }) = move_picker.next(self, t) {
            if ROOT && uci::is_multipv() {
                // handle multi-pv
                if t.multi_pv_excluded.contains(&m) {
                    continue;
                }
            }
            if excluded == m {
                continue;
            }

            let lmr_reduction = self.lmr_table.getr(depth, moves_made);
            let lmr_depth = std::cmp::max(depth - lmr_reduction, ZERO_PLY);
            let is_quiet = !self.is_tactical(m);
            if is_quiet {
                quiets_tried.push(m);
            } else {
                tacticals_tried.push(m);
            }

            // lmp, fp, and hlp.
            if !ROOT && !PV && !in_check && best_score > -MINIMUM_MATE_SCORE {
                // late move pruning
                // if we have made too many moves, we start skipping moves.
                if lmr_depth <= self.sparams.lmp_depth && moves_made >= lmp_threshold {
                    move_picker.skip_quiets = true;
                }

                // futility pruning
                // if the static eval is too low, we start skipping moves.
                let fp_margin = lmr_depth.round() * self.sparams.futility_coeff_1
                    + self.sparams.futility_coeff_0;
                if is_quiet
                    && lmr_depth < self.sparams.futility_depth
                    && static_eval + fp_margin <= alpha
                {
                    move_picker.skip_quiets = true;
                }

                // history leaf pruning
                // if the history score is too low, we skip the move.
                if is_quiet
                    && moves_made > 1
                    && lmr_depth <= ONE_PLY * 2
                    && ordering_score < (-500 * (depth.round() - 1))
                {
                    continue;
                }
            }

            // static exchange evaluation pruning
            // simulate all captures flowing onto the target square, and if we come out badly, we skip the move.
            if !ROOT
                && best_score > -MINIMUM_MATE_SCORE
                && depth <= self.sparams.see_depth
                && !self.static_exchange_eval(m, see_table[usize::from(is_quiet)])
            {
                continue;
            }

            if !self.make_move_nnue(m, t) {
                continue;
            }
            info.nodes += 1;
            moves_made += 1;
            if ROOT && info.print_to_stdout && info.time_since_start() > Duration::from_secs(5) {
                println!("info currmove {m} currmovenumber {moves_made} nodes {}", info.nodes);
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
                let gives_check = self.in_check::<{ Self::US }>();
                extension = Depth::from(gives_check);
            };

            let mut score;
            if moves_made == 1 {
                // first move (presumably the PV-move)
                score = -self.alpha_beta::<PV, false, USE_NNUE>(
                    info,
                    t,
                    depth + extension - 1,
                    -beta,
                    -alpha,
                );
            } else {
                // calculation of LMR stuff
                let r = if extension == ZERO_PLY
                    && is_quiet
                    && depth >= 3.into()
                    && moves_made >= (2 + usize::from(PV))
                {
                    let mut r = self.lmr_table.getr(depth, moves_made);
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

                        if is_quiet {
                            t.insert_killer(self, best_move);
                            t.insert_countermove(self, best_move);
                            self.update_history_metrics::<true>(t, best_move, depth);

                            // decrease the history of the non-capture moves that came before the cutoff move.
                            let qs = quiets_tried.as_slice();
                            let qs = &qs[..qs.len() - 1];
                            for &q in qs {
                                self.update_history_metrics::<false>(t, q, depth);
                            }
                        }

                        if excluded.is_null() {
                            self.tt_store::<ROOT>(best_move, beta, HFlag::LowerBound, depth);
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
                self.tt_store::<ROOT>(best_move, alpha, HFlag::UpperBound, depth);
            }
        } else {
            // we raised alpha, and didn't raise beta
            // as if we had, we would have returned early,
            // so this is a PV-node
            if !self.is_tactical(best_move) {
                t.insert_killer(self, best_move);
                t.insert_countermove(self, best_move);
                self.update_history_metrics::<true>(t, best_move, depth);

                // decrease the history of the non-capture moves that came before the best move.
                let qs = quiets_tried.as_slice();
                let qs = &qs[..qs.len() - 1];
                for &q in qs {
                    self.update_history_metrics::<false>(t, q, depth);
                }
            }

            if excluded.is_null() {
                self.tt_store::<ROOT>(best_move, best_score, HFlag::Exact, depth);
            }
        }

        alpha
    }

    fn rfp_margin(&mut self, depth: Depth, improving: bool) -> i32 {
        self.sparams.rfp_margin * depth - i32::from(improving) * self.sparams.rfp_improving_margin
    }

    fn update_history_metrics<const IS_GOOD: bool>(
        &mut self,
        t: &mut ThreadData,
        m: Move,
        depth: Depth,
    ) {
        t.add_history::<IS_GOOD>(self, m, depth);
        t.add_followup_history::<IS_GOOD>(self, m, depth);
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
        let value = self.alpha_beta::<false, false, USE_NNUE>(
            info,
            t,
            reduced_depth,
            reduced_beta - 1,
            reduced_beta,
        );
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
        let value = self.alpha_beta::<false, true, true>(
            info,
            t,
            reduced_depth,
            reduced_beta - 1,
            reduced_beta,
        );
        info.print_to_stdout = pts_prev;
        t.excluded[self.height()] = Move::NULL;
        value < reduced_beta
    }

    pub fn static_exchange_eval(&self, m: Move, threshold: i32) -> bool {
        let from = m.from();
        let to = m.to();

        let mut next_victim =
            if m.is_promo() { m.promotion_type() } else { type_of(self.piece_at(from)) };

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
        let mut occupied = (self.pieces.occupied() ^ from.bitboard()) | to.bitboard();
        if m.is_ep() {
            occupied ^= self.ep_sq().bitboard();
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
}

pub const fn draw_score(nodes: u64) -> i32 {
    // score fuzzing helps with threefolds.
    (nodes & 0b11) as i32 - 2
}

pub struct LMTable {
    rtable: [[i32; 64]; 64],
    ptable: [[usize; 12]; 2],
}

impl LMTable {
    pub fn new(config: &SearchParams) -> Self {
        #![allow(
            clippy::cast_possible_truncation,
            clippy::cast_precision_loss,
            clippy::cast_sign_loss
        )]
        let mut out = Self { rtable: [[0; 64]; 64], ptable: [[0; 12]; 2] };
        let (base, division) = (config.lmr_base / 100.0, config.lmr_division / 100.0);
        for depth in 1..64 {
            for played in 1..64 {
                let ld = f64::ln(depth as f64);
                let lp = f64::ln(played as f64);
                out.rtable[depth][played] = (base + ld * lp / division) as i32;
            }
        }
        for depth in 1..12 {
            out.ptable[0][depth] = (2.5 + 2.0 * depth as f64 * depth as f64 / 4.5) as usize;
            out.ptable[1][depth] = (4.0 + 4.0 * depth as f64 * depth as f64 / 4.5) as usize;
        }
        out
    }

    pub fn getr(&self, depth: Depth, moves_made: usize) -> i32 {
        let depth = depth.ply_to_horizon().min(63);
        let played = moves_made.min(63);
        self.rtable[depth][played]
    }

    pub fn getp(&self, depth: Depth, improving: bool) -> usize {
        let depth = depth.ply_to_horizon().min(11);
        self.ptable[usize::from(improving)][depth]
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
