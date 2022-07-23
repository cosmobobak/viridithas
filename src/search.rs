use crate::{
    board::movegen::MoveList,
    board::{
        evaluation::{DRAW_SCORE, MATE_SCORE, is_mate_score},
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
    pub fn alpha_beta<const PV: bool>(&mut self, info: &mut SearchInfo, depth: Depth, mut alpha: i32, beta: i32) -> i32 {
    #[cfg(debug_assertions)]
    self.check_validity().unwrap();

    if depth <= 0.into() {
        return Self::quiescence(self, info, alpha, beta);
    }

    if info.nodes.trailing_zeros() >= 12 {
        info.check_up();
        if info.stopped {
            return 0;
        }
    }

    let height: i32 = self.height().try_into().unwrap();

    let root_node = height == 0;

    info.nodes += 1;
    info.seldepth = if root_node { 0.into() } else { info.seldepth.max(height.into()) };

    let static_eval = self.evaluate();

    if !root_node {
        // check draw
        if self.is_draw() {
            // score fuzzing apparently helps with threefolds.
            return 1 - (info.nodes & 2) as i32;
        }

        // are we too deep?
        if height > MAX_DEPTH.round() - 1 {
            return static_eval;
        }

        // mate-distance pruning.
        // doesn't actually add strength, but it makes viri better at solving puzzles.
        // approach taken from Ethereal.
        let r_alpha = if alpha > -MATE_SCORE + height     { alpha } else { -MATE_SCORE + height };
        let r_beta  = if  beta <  MATE_SCORE - height - 1 {  beta } else {  MATE_SCORE - height - 1 };
        if r_alpha >= r_beta { return r_alpha; }
    }

    debug_assert_eq!(PV, beta - alpha > 1, "PV must be true if the alpha-beta window is larger than 1");

    let tt_move = match self.tt_probe(alpha, beta, depth) {
        ProbeResult::Cutoff(s) => {
            return s;
        }
        ProbeResult::BestMove(tt_move) => {
            Some(tt_move)
        }
        ProbeResult::Nothing => {
            None
        }
    };

    let in_check = self.in_check::<{ Self::US }>();

    // TEST: pv nullmove pruning
    if !PV && !in_check && !root_node && depth >= 3.into() && self.zugzwang_unlikely() {
        self.make_nullmove();
        let score = -self.alpha_beta::<PV>(info, depth - 3, -beta, -alpha);
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

    let history_score = depth.round() * depth.round();

    let original_alpha = alpha;
    let mut moves_made = 0;
    let mut best_move = Move::NULL;
    let mut best_score = -INFINITY;

    if let Some(tt_move) = tt_move {
        if let Some(movelist_entry) = move_list.lookup_by_move(tt_move) {
            movelist_entry.score = TT_MOVE_SCORE;
        }
    }

    for m in move_list {
        if !self.make_move(m) {
            continue;
        }
        moves_made += 1;

        let is_capture = m.is_capture();
        let gives_check = self.in_check::<{ Self::US }>();
        let is_promotion = m.is_promo();

        let is_interesting = is_capture || is_promotion || gives_check || in_check;

        // futility pruning (worth 32 +/- 44 elo)
        // if the static eval is too low, we might just skip the move.
        if !PV && is_move_futile(depth, moves_made, is_interesting, static_eval, alpha, beta) {
            self.unmake_move();
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
            score = -self.alpha_beta::<PV>(info, depth + extension - 1, -beta, -alpha);
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
            let r = if can_reduce {
                let mut r = self.lmr_table.get(depth, moves_made);
                r += i32::from(!PV);
                Depth::new(r).clamp(1.into(), depth - 1)
            } else {
                1.into()
            };
            // perform a zero-window search
            score = -self.alpha_beta::<false>(info, depth + extension - r, -alpha - 1, -alpha);
            // if we failed, then full window search
            if score > alpha && score < beta {
                // this is a new best move, so it *is* PV.
                score = -self.alpha_beta::<PV>(info, depth + extension - 1, -beta, -alpha);
            }
        }
        self.unmake_move();

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
                        // IDEA (todo): if the cutoff move wasn't the first, i.e.
                        // moves_made != 1, then we should decrease the history
                        // scores of the moves that we've already searched.
                        self.insert_killer(best_move);
                        self.insert_countermove(best_move);
                        self.update_history_metrics(best_move, history_score);
                    }

                    self.tt_store(best_move, beta, HFlag::Beta, depth);

                    return beta;
                }
                assert_eq!(alpha, score);
                assert_eq!(best_move, m);
            }
        }
    }

    if moves_made == 0 {
        if in_check {
            return -MATE_SCORE + height;
        }
        return DRAW_SCORE;
    }

    if alpha == original_alpha {
        // we didn't raise alpha, so this is an all-node
        self.tt_store(best_move, alpha, HFlag::Alpha, depth);
    } else {
        // we raised alpha, and didn't raise beta
        // as if we had, we would have returned early, 
        // so this is a PV-node
        self.insert_killer(best_move);
        self.insert_countermove(best_move);
        self.update_history_metrics(best_move, history_score);
        self.tt_store(best_move, best_score, HFlag::Exact, depth);
    }

    alpha
}

    fn update_history_metrics(&mut self, best_move: Move, history_score: i32) {
        self.add_history(best_move, history_score);
        self.add_followup_history(best_move, history_score);
    }
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
    let threshold = FUTILITY_PRUNING_MARGINS[depth.ply_to_horizon()];
    static_eval + threshold < a
}

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
    pub lmr_base: f64,
    pub lmr_division: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            null_move_reduction: 3.into(),
            futility_gradient: 41,
            futility_intercept: 51,
            lmr_base: 0.75,
            lmr_division: 2.25,
        }
    }
}

pub struct LMRTable {
    table: [[i32; 64]; 64],
}

impl LMRTable {
    pub fn new(config: &Config) -> Self {
        #![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        let mut out = Self {
            table: [[0; 64]; 64],
        };
        for depth in 1..64 {
            for played in 1..64 {
                let ld = f64::ln(depth as f64);
                let lp = f64::ln(played as f64);
                out.table[depth][played] = (config.lmr_base + ld * lp / config.lmr_division) as i32;
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