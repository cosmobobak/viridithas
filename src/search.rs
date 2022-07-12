use crate::{
    board::movegen::MoveList,
    board::{
        evaluation::{is_mate_score, DRAW_SCORE, MATE_SCORE},
        movegen::TT_MOVE_SCORE,
        Board,
    },
    chessmove::Move,
    definitions::{Depth, INFINITY, MAX_DEPTH},
    searchinfo::SearchInfo,
    transpositiontable::{HFlag, ProbeResult},
};

// In alpha-beta search, there are three classes of node to be aware of:
// 1. PV-nodes: nodes that end up being within the alpha-beta window,
// i.e. a call to alpha_beta(PVNODE, a, b) returns a value v where v is within the window [a, b].
// the score returned is an exact score for the node.
// 2. Cut-nodes: nodes that fail high, i.e. a move is found that leads to a value >= beta.
// in alpha-beta, a call to alpha_beta(CUTNODE, alpha, beta) returns a score >= beta.
// The score returned is a lower bound (might be greater) on the exact score of the node
// 3. All-nodes: nodes that fail low, i.e. no move leads to a value > alpha.
// in alpha-beta, a call to alpha_beta(ALLNODE, alpha, beta) returns a score <= alpha.
// Every move at an All-node is searched, and the score returned is an upper bound, so the exact score might be lower.

impl Board {
    pub fn quiescence(pos: &mut Self, info: &mut SearchInfo, mut alpha: i32, beta: i32) -> i32 {
        #[cfg(debug_assertions)]
        pos.check_validity().unwrap();

        if info.nodes.trailing_zeros() >= 12 {
            info.check_up();
            if info.stopped {
                return 0;
            }
        }

        let height: i32 = pos.height().try_into().unwrap();
        info.nodes += 1;
        info.seldepth = info.seldepth.max(height.into());

        // check draw
        if pos.is_draw() {
            // score fuzzing apparently helps with threefolds.
            return 1 - (info.nodes & 2) as i32;
        }

        // are we too deep?
        if height > (MAX_DEPTH - 1).round() {
            return pos.evaluate();
        }

        let stand_pat = pos.evaluate();

        if stand_pat >= beta {
            return beta;
        }

        if stand_pat > alpha {
            alpha = stand_pat;
        }

        let mut move_list = MoveList::new();
        pos.generate_captures(&mut move_list);

        let mut moves_made = 0;

        for m in move_list {
            if !pos.make_move(m) {
                continue;
            }

            moves_made += 1;
            let score = -Self::quiescence(pos, info, -beta, -alpha);
            pos.unmake_move();

            if score > alpha {
                if score >= beta {
                    if moves_made == 1 {
                        info.failhigh_first += 1.0;
                    }
                    info.failhigh += 1.0;
                    return beta;
                }
                alpha = score;
            }
        }

        alpha
    }

#[rustfmt::skip]
#[allow(clippy::too_many_lines, clippy::cognitive_complexity, clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    pub fn alpha_beta<const PV: bool>(pos: &mut Self, info: &mut SearchInfo, depth: Depth, mut alpha: i32, beta: i32) -> i32 {
    #[cfg(debug_assertions)]
    pos.check_validity().unwrap();

    if depth <= 0.into() {
        return Self::quiescence(pos, info, alpha, beta);
    }

    if info.nodes.trailing_zeros() >= 12 {
        info.check_up();
        if info.stopped {
            return 0;
        }
    }

    let height: i32 = pos.height().try_into().unwrap();

    let root_node = height == 0;

    info.nodes += 1;
    info.seldepth = if root_node { 0.into() } else { info.seldepth.max(height.into()) };

    let static_eval = pos.evaluate();

    if !root_node {
        // check draw
        if pos.is_draw() {
            // score fuzzing apparently helps with threefolds.
            return 1 - (info.nodes & 2) as i32;
        }

        // are we too deep?
        if pos.height() >= MAX_DEPTH.n_ply() - 1 {
            return static_eval;
        }

        // mate-distance pruning.
        // doesn't actually add strength, but it makes viri better at solving puzzles.
        // approach taken from Ethereal.
        let r_alpha = if alpha > -MATE_SCORE + height     { alpha } else { -MATE_SCORE + height };
        let r_beta  = if  beta <  MATE_SCORE - height - 1 {  beta } else {  MATE_SCORE - height - 1 };
        if r_alpha >= r_beta { return r_alpha; }
    }

    let pv_move = match pos.tt_probe(alpha, beta, depth) {
        ProbeResult::Cutoff(s) => {
            return s;
        }
        ProbeResult::BestMove(pv_move) => {
            Some(pv_move)
        }
        ProbeResult::Nothing => {
            None
        }
    };

    let in_check = pos.in_check::<{ Self::US }>();

    if !PV && !in_check && !root_node && depth >= 3.into() && pos.zugzwang_unlikely() {
        pos.make_nullmove();
        let nmr = pos.search_params.null_move_reduction;
        let score = -Self::alpha_beta::<false>(pos, info, depth - nmr, -beta, -alpha);
        pos.unmake_nullmove();
        if info.stopped {
            return 0;
        }
        if score >= beta {
            return beta;
        }
    }

    let mut move_list = MoveList::new();
    pos.generate_moves(&mut move_list);

    let history_score = depth.round() * depth.round();
    let original_alpha = alpha;
    let mut moves_made = 0;
    let mut best_move = Move::NULL;
    let mut best_score = -INFINITY;

    if let Some(pv_move) = pv_move {
        if let Some(movelist_entry) = move_list.lookup_by_move(pv_move) {
            movelist_entry.score = TT_MOVE_SCORE;
        }
    }

    for m in move_list {
        if !pos.make_move(m) {
            continue;
        }
        moves_made += 1;

        let is_capture = m.is_capture();
        let gives_check = pos.in_check::<{ Self::US }>();
        let is_promotion = m.is_promo();

        let is_interesting = is_capture || is_promotion || gives_check || in_check;

        // futility pruning (worth 32 +/- 44 elo)
        if !PV && Self::is_move_futile(depth, moves_made, is_interesting, static_eval, alpha, beta) {
            pos.unmake_move();
            continue;
        }

        let extension: Depth = if gives_check {
            0.8.into()
        } else {
            0.into()
        };

        let mut score;
        if moves_made == 1 {
            // first move (presumably the PV-move)
            score = -Self::alpha_beta::<true>(pos, info, depth + extension - 1, -beta, -alpha);
        } else {
            // nullwindow searches to prove PV.
            // we only do late move reductions when a set of conditions are true:
            // 1. the move we're about to make isn't "interesting" (i.e. it's not a capture, a promotion, or a check)
            // RATIONALE: captures, promotions, and checks are likely to be very important moves to search.
            // 2. we're at a depth >= 3.
            // RATIONALE: depth < 3 is cheap as hell already.
            // 3. we're not already extending the search.
            // RATIONALE: if this search is extended, we explicitly don't want to reduce it.
            // 4. we've tried at least two moves at full depth, or three if we're in a PV-node.
            // RATIONALE: we should be trying at least some moves with full effort, and moves in PV nodes are more important.
            let can_reduce = extension == 0.into()
                && !is_interesting
                && depth >= 3.into()
                && moves_made >= (2 + usize::from(PV));
            let mut r: Depth = 0.into();
            // late move reductions (75 +/- 83 elo)
            if can_reduce {
                r += Self::logistic_lateness_reduction(pos, moves_made, depth).max(0.into());
            }
            // perform a zero-window search, possibly with a reduction
            score = -Self::alpha_beta::<false>(pos, info, depth + extension - r - 1, -alpha - 1, -alpha);
            // if we failed, then full window search
            if score > alpha && score < beta {
                // this is a new best move, so it *is* PV.
                score = -Self::alpha_beta::<true>(pos, info, depth + extension - 1, -beta, -alpha);
            }
        }
        pos.unmake_move();

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
                    if moves_made == 1 {
                        info.failhigh_first += 1.0;
                    }
                    info.failhigh += 1.0;

                    if !is_capture {
                        pos.insert_killer(best_move);
                        pos.update_history_metrics(best_move, history_score);
                    }

                    pos.tt_store(best_move, beta, HFlag::Beta, depth);

                    return beta;
                }
            }
        }
    }

    if moves_made == 0 {
        if in_check {
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            return -MATE_SCORE + height;
        }
        return DRAW_SCORE;
    }

    if alpha == original_alpha {
        // we didn't raise alpha, so this is an all-node
        pos.tt_store(best_move, alpha, HFlag::Alpha, depth);
    } else {
        // we raised alpha, and didn't raise beta
        // as if we had, we would have returned early, 
        // so this is a PV-node
        pos.update_history_metrics(best_move, history_score);
        pos.tt_store(best_move, best_score, HFlag::Exact, depth);
    }

    alpha
}

    fn update_history_metrics(&mut self, best_move: Move, history_score: i32) {
        self.add_history(best_move, history_score);
        self.add_countermove_history(best_move, history_score);
        self.add_followup_history(best_move, history_score);
    }

    fn logistic_lateness_reduction(&self, moves: usize, depth: Depth) -> Depth {
        #![allow(
            clippy::cast_precision_loss,
            clippy::cast_sign_loss,
            clippy::cast_possible_truncation
        )]
        let gradient = self.search_params.lmr_gradient;
        let midpoint = self.search_params.lmr_midpoint;
        let reduction_factor = self.search_params.lmr_max_depth;
        let moves: f32 = moves as f32;
        let depth: f32 = depth.into();
        let numerator = reduction_factor * depth - 1.0;
        let denominator = 1.0 + f32::exp(-gradient * (moves - midpoint));
        ((numerator / denominator + 0.5) as i32).into()
    }

    fn is_move_futile(
        depth: Depth,
        moves_made: usize,
        interesting: bool,
        static_eval: i32,
        a: i32,
        b: i32,
    ) -> bool {
        if !(1.into()..=4.into()).contains(&depth) || interesting || moves_made == 1 {
            return false;
        }
        if is_mate_score(a) || is_mate_score(b) {
            return false;
        }
        #[allow(clippy::cast_sign_loss)]
        let threshold = FUTILITY_PRUNING_MARGINS[depth.n_ply()];
        static_eval + threshold < a
    }
}

// values taken from MadChess.
static FUTILITY_PRUNING_MARGINS: [i32; 5] = [
    100, // 0 moves to the horizon
    150, // 1 move to the horizon
    250, // 2 moves to the horizon
    400, // 3 moves to the horizon
    600, // 4 moves to the horizon
];

#[derive(Debug)]
pub struct Config {
    pub null_move_reduction: Depth,
    pub futility_gradient: i32,
    pub futility_intercept: i32,
    pub lmr_gradient: f32,
    pub lmr_midpoint: f32,
    pub lmr_max_depth: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            null_move_reduction: 3.into(),
            futility_gradient: 41,
            futility_intercept: 51,
            lmr_gradient: 0.12,
            lmr_midpoint: 8.06,
            lmr_max_depth: 0.65,
        }
    }
}
