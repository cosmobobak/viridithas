use crate::{
    board::movegen::MoveList,
    board::{
        evaluation::{DRAW_SCORE, MATE_SCORE, ONE_PAWN},
        Board,
    },
    chessmove::Move,
    definitions::{FUTILITY_MARGIN, INFINITY, MAX_DEPTH},
    searchinfo::SearchInfo,
    transpositiontable::{HFlag, ProbeResult},
};

// In alpha-beta search, there are three classes of node to be aware of:
// 1. PV-nodes: nodes that end up being within the alpha-beta window,
// i.e. a call to alpha_beta(PVNODE, a, b) returns a value v where v is within the window [a, b].
// the score returned is an exact score for the node.
// 2. Cut-nodes: nodes that fail high, i.e. a move is found that leads to a value >= beta.
// in fail-hard alpha-beta, a call to alpha_beta(CUTNODE, a, b) returns b.
// The score returned is a lower bound (might be greater) on the exact score of the node
// 3. All-nodes: nodes that fail low, i.e. no move leads to a value > alpha.
// in fail-hard alpha-beta, a call to alpha_beta(ALLNODE, a, b) returns a.
// Every move at an All-node is searched, and the score returned is an upper bound, so the exact score might be lower.

fn quiescence_search(pos: &mut Board, info: &mut SearchInfo, mut alpha: i32, beta: i32) -> i32 {
    #[cfg(debug_assertions)]
    pos.check_validity().unwrap();

    if info.nodes.trailing_zeros() >= 12 {
        info.check_up();
    }

    info.nodes += 1;

    if pos.is_draw() {
        return DRAW_SCORE;
    }

    if pos.ply() > MAX_DEPTH - 1 {
        return pos.evaluate();
    }

    let score = pos.evaluate();

    if score >= beta {
        return beta;
    }

    if score > alpha {
        alpha = score;
    }

    let mut move_list = MoveList::new();
    let is_check = pos.in_check::<{ Board::US }>();
    if is_check {
        pos.generate_moves(&mut move_list); // if we're in check, the position isn't very quiescent, is it?
    } else {
        pos.generate_captures(&mut move_list);
    }

    let mut moves_made = 0;

    move_list.sort();

    for &m in move_list.iter() {
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

fn lateness_reduction(moves: usize, depth: usize) -> usize {
    #![allow(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation
    )]
    const GRADIENT: f32 = 0.7;
    const MIDPOINT: f32 = 4.5;
    let moves = moves as f32;
    let depth = depth as f32;
    let numerator = 2.0 * depth / 5.0 - 1.0;
    let denominator = 1.0 + f32::exp(-GRADIENT * (moves - MIDPOINT));
    (numerator / denominator + 0.5) as usize
}

#[rustfmt::skip]
#[allow(clippy::too_many_lines)]
pub fn alpha_beta(pos: &mut Board, info: &mut SearchInfo, depth: usize, mut alpha: i32, beta: i32) -> i32 {
    #[cfg(debug_assertions)]
    pos.check_validity().unwrap();

    if depth == 0 {
        return quiescence_search(pos, info, alpha, beta);
    }

    let in_pv_node = beta - alpha > 1;

    if info.nodes.trailing_zeros() >= 12 {
        info.check_up();
    }

    info.nodes += 1;

    if pos.is_draw() {
        return DRAW_SCORE;
    }

    if pos.ply() > MAX_DEPTH - 1 {
        return pos.evaluate();
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

    let we_are_in_check = pos.in_check::<{ Board::US }>();

    if !we_are_in_check && pos.ply() != 0 && pos.zugzwang_unlikely() && depth >= 3 {
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

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let history_score = (depth * depth) as i32;
    let original_alpha = alpha;
    let mut moves_made = 0;
    let mut best_move = Move::null();
    let mut best_score = -INFINITY;

    if let Some(pv_move) = pv_move {
        if let Some(movelist_entry) = move_list.lookup_by_move(pv_move) {
            movelist_entry.score = 20_000_000;
        }
    }

    move_list.sort();

    let futility_pruning_legal = !pos.in_check::<{ Board::US }>() 
        && depth == 1
        && pos.evaluate() + FUTILITY_MARGIN < alpha
        && !in_pv_node;

    for &m in move_list.iter() {
        if !pos.make_move(m) {
            continue;
        }
        moves_made += 1;

        let is_capture = m.is_capture();
        let is_check = pos.in_check::<{ Board::US }>();
        let is_promotion = m.is_promo();

        let is_interesting = is_capture || is_promotion || is_check;

        if futility_pruning_legal && !is_interesting {
            pos.unmake_move();
            continue;
        }

        let extension = usize::from(is_check) + usize::from(is_promotion);

        let score = if moves_made == 1 {
            // first move (presumably the PV-move)
            -alpha_beta(pos, info, depth - 1 + extension, -beta, -alpha)
        } else {
            // nullwindow searches to prove PV.
            // we only reduce when a set of conditions are true:
            // 1. the move we're about to make isn't "interesting" (i.e. it's not a capture, a promotion, or a check)
            // 2. we're at a depth >= 3.
            // 3. we're not already extending the search.
            // 4. we've tried at least 2 moves at full depth.
            // 5. we're not in a pv-node.
            let mut reduction = 0; 
            if !is_interesting
                && depth >= 3
                && extension == 0
                && moves_made >= 2
                && !in_pv_node { 
                reduction += lateness_reduction(moves_made, depth);
            }
            if extension == 0 && !in_pv_node && depth == 2 && reduction < 2 {
                // razoring at the pre-frontier nodes.
                let static_eval = pos.evaluate();
                if static_eval + ONE_PAWN / 2 < alpha {
                    reduction += 1;
                }
            }
            // perform a zero-window search, possibly with a reduction
            let r = -alpha_beta(pos, info, depth - 1 + extension - reduction, -alpha - 1, -alpha);
            if r > alpha && r < beta {
                // if the zero-window search fails, perform a full window search at full depth
                -alpha_beta(pos, info, depth - 1 + extension, -beta, -alpha)
            } else {
                r
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
        if we_are_in_check {
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            return -MATE_SCORE + pos.ply() as i32;
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
