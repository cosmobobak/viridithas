use crate::{
    board::movegen::MoveList,
    board::{
        evaluation::{DRAW_SCORE, MATE_SCORE, is_mate_score},
        movegen::TT_MOVE_SCORE,
        Board,
    },
    chessmove::Move,
    definitions::{INFINITY, MAX_DEPTH},
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

// values taken from MadChess.
const FUTILITY_PRUNING_MARGINS: [i32; 5] = [
    100, // 0 moves to the horizon
    150, // 1 move to the horizon
    250, // 2 moves to the horizon
    400, // 3 moves to the horizon
    600, // 4 moves to the horizon
];

fn quiescence_search(pos: &mut Board, info: &mut SearchInfo, mut alpha: i32, beta: i32) -> i32 {
    #[cfg(debug_assertions)]
    pos.check_validity().unwrap();

    if info.nodes.trailing_zeros() >= 12 {
        info.check_up();
        if info.stopped {
            return 0;
        }
    }

    let height: i32 = pos.ply().try_into().unwrap();
    info.nodes += 1;
    info.seldepth = info.seldepth.max(height);

    // check draw
    if pos.is_draw() {
        // score fuzzing apparently helps with threefolds.
        return 1 - (info.nodes & 2) as i32;
    }

    // are we too deep?
    if height > MAX_DEPTH - 1 {
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
    let is_check = pos.in_check::<{ Board::US }>();
    if is_check {
        pos.generate_moves(&mut move_list); // if we're in check, the position isn't very quiescent, is it?
    } else {
        pos.generate_captures(&mut move_list);
    }

    let mut moves_made = 0;

    // move_list.sort();

    for m in move_list {
        if !pos.make_move(m) {
            continue;
        }

        moves_made += 1;
        let score = -quiescence_search(pos, info, -beta, -alpha);
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

    if moves_made == 0 && is_check {
        // can't return a draw score when moves_made = 0, as sometimes we only check captures,
        // but we can return mate scores, because we do full movegen when in check.
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        return -MATE_SCORE + pos.ply() as i32;
    }

    alpha
}

fn logistic_lateness_reduction(moves: usize, depth: i32) -> i32 {
    #![allow(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation
    )]
    const GRADIENT: f32 = 0.7;
    const MIDPOINT: f32 = 4.5;
    const REDUCTION_FACTOR: f32 = 2.0 / 5.0;
    let moves = moves as f32;
    let depth = depth as f32;
    let numerator = REDUCTION_FACTOR * depth - 1.0;
    let denominator = 1.0 + f32::exp(-GRADIENT * (moves - MIDPOINT));
    (numerator / denominator + 0.5) as i32
}

fn _fruit_lateness_reduction(moves: usize, depth: i32, in_pv: bool) -> i32 {
    #![allow(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation
    )]
    const PV_BACKOFF: f32 = 2.0 / 3.0;
    let r = f32::sqrt((depth - 1) as f32) + f32::sqrt((moves - 1) as f32);
    if in_pv {
        (r * PV_BACKOFF) as i32
    } else {
        r as i32
    }
}

fn _senpai_lateness_reduction(moves: usize, depth: i32) -> i32 {
    // Senpai reduces by one ply for the first 6 moves and by depth / 3 for remaining moves.
    if moves <= 6 {
        1
    } else {
        (depth / 3).max(1)
    }
}

#[rustfmt::skip]
#[allow(clippy::too_many_lines, clippy::cognitive_complexity, clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub fn alpha_beta(pos: &mut Board, info: &mut SearchInfo, depth: i32, mut alpha: i32, beta: i32) -> i32 {
    #[cfg(debug_assertions)]
    pos.check_validity().unwrap();

    if depth <= 0 {
        return quiescence_search(pos, info, alpha, beta);
    }

    if info.nodes.trailing_zeros() >= 12 {
        info.check_up();
    }

    let height: i32 = pos.ply().try_into().unwrap();

    let root_node = height == 0;

    info.nodes += 1;
    info.seldepth = if root_node { 0 } else { info.seldepth.max(height) };

    let static_eval = pos.evaluate();

    if !root_node {
        // check draw
        if pos.is_draw() {
            // score fuzzing apparently helps with threefolds.
            return 1 - (info.nodes & 2) as i32;
        }

        // are we too deep?
        if height > MAX_DEPTH - 1 {
            return static_eval;
        }

        // mate-distance pruning.
        // doesn't actually add strength, but it makes viri better at solving puzzles.
        // approach taken from Ethereal.
        let r_alpha = if alpha > -MATE_SCORE + height     { alpha } else { -MATE_SCORE + height };
        let r_beta  = if  beta <  MATE_SCORE - height - 1 {  beta } else {  MATE_SCORE - height - 1 };
        if r_alpha >= r_beta {return r_alpha; }
    }

    let in_pv = beta - alpha > 1;

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

    let in_check = pos.in_check::<{ Board::US }>();

    if !in_check && pos.ply() != 0 && depth >= 3 && pos.zugzwang_unlikely() {
        pos.make_nullmove();
        let score = -alpha_beta(pos, info, depth - 3, -beta, -alpha);
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

    let history_score = depth * depth;
    let original_alpha = alpha;
    let mut moves_made = 0;
    let mut best_move = Move::NULL;
    let mut best_score = -INFINITY;

    if let Some(pv_move) = pv_move {
        if let Some(movelist_entry) = move_list.lookup_by_move(pv_move) {
            movelist_entry.score = TT_MOVE_SCORE;
        }
    }

    // move_list.sort();

    for m in move_list {
        if !pos.make_move(m) {
            continue;
        }
        moves_made += 1;

        let is_capture = m.is_capture();
        let gives_check = pos.in_check::<{ Board::US }>();
        let is_promotion = m.is_promo();

        let is_interesting = is_capture || is_promotion || gives_check || in_check;

        // futility pruning (worth 32 +/- 44 elo)
        if is_move_futile(depth, moves_made, is_interesting, static_eval, alpha, beta) {
            pos.unmake_move();
            continue;
        }

        let extension = i32::from(gives_check) + i32::from(is_promotion);

        let mut score;
        if moves_made == 1 {
            // first move (presumably the PV-move)
            score = -alpha_beta(pos, info, depth - 1 + extension, -beta, -alpha);
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
            let can_reduce = extension == 0
                && !is_interesting
                && depth >= 3
                && moves_made >= (2 + usize::from(in_pv));
            let mut r = 0;
            // late move reductions (75.6 +/- 83.3 elo)
            if can_reduce {
                r += logistic_lateness_reduction(moves_made, depth).clamp(1, depth - 2);
            }
            // perform a zero-window search, possibly with a reduction
            score = -alpha_beta(pos, info, depth - 1 + extension - r, -alpha - 1, -alpha);
            // if we failed, then full window search
            if score > alpha && score < beta {
                score = -alpha_beta(pos, info, depth - 1 + extension, -beta, -alpha);
            }
        };
        pos.unmake_move();

        if info.stopped {
            return 0;
        }

        if score > best_score {
            best_score = score;
            best_move = m;
            if score > alpha {
                if score >= beta {
                    // we failed high, so this is a cut-node
                    if moves_made == 1 {
                        info.failhigh_first += 1.0;
                    }
                    info.failhigh += 1.0;

                    if !is_capture {
                        // quiet moves that fail high are killers.
                        pos.insert_killer(m);
                        // // this is a countermove.
                        pos.insert_countermove(m);
                        // // double-strength history heuristic :3
                        pos.add_history(m, 2 * history_score);
                    }

                    pos.tt_store(best_move, beta, HFlag::Beta, depth);

                    return beta;
                }
                alpha = score;
                best_move = m;
                if !is_capture {
                    // quiet moves that improve alpha increment the history table
                    pos.add_history(m, history_score);
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
        pos.tt_store(best_move, best_score, HFlag::Exact, depth);
    }

    alpha
}

fn is_move_futile(depth: i32, moves_made: usize, interesting: bool, static_eval: i32, a: i32, b: i32) -> bool {
    if !(1..=4).contains(&depth) || interesting || moves_made == 1 {
        return false;
    }
    if is_mate_score(a) || is_mate_score(b) {
        return false;
    }
    #[allow(clippy::cast_sign_loss)]
    let threshold = FUTILITY_PRUNING_MARGINS[depth as usize];
    static_eval + threshold < a
}