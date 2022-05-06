use crate::{
    board::Board,
    chessmove::Move,
    definitions::{MAX_DEPTH, FUTILITY_MARGIN},
    evaluation::{DRAW_SCORE, MATE_SCORE},
    movegen::MoveList,
    searchinfo::SearchInfo,
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

// I want to try a qsearch-free algorithm, using LMR and
// futility pruning to cleanly interpolate.
fn quiescence_search(pos: &mut Board, info: &mut SearchInfo, mut alpha: i32, beta: i32) -> i32 {
    #[cfg(debug_assertions)]
    pos.check_validity().unwrap();

    if info.nodes.trailing_zeros() >= 11 {
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
    let is_check = pos.is_check();
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
        // can't return a draw score, as sometimes we only check captures,
        // but we can return mate scores, because we do full movegen when in check.
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        return -MATE_SCORE + pos.ply() as i32;
    }

    alpha
}

#[rustfmt::skip]
#[allow(clippy::too_many_lines)]
pub fn alpha_beta(pos: &mut Board, info: &mut SearchInfo, depth: usize, mut alpha: i32, beta: i32) -> i32 {
    #[cfg(debug_assertions)]
    pos.check_validity().unwrap();

    if depth == 0 {
        return quiescence_search(pos, info, alpha, beta);
    }

    if info.nodes.trailing_zeros() >= 11 {
        info.check_up();
    }

    info.nodes += 1;

    if pos.is_draw() {
        return DRAW_SCORE;
    }

    if pos.ply() > MAX_DEPTH - 1 {
        return pos.evaluate();
    }

    let we_are_in_check = pos.is_check();

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

    if let Some(&pv_move) = pos.probe_pv_move() { // this can probably be done in movegen
        if let Some(movelist_entry) = move_list.lookup_by_move(pv_move) {
            movelist_entry.score = 20_000_000;
        }
    }

    move_list.sort();

    let futility_pruning_legal = !pos.is_check() && depth == 1 && pos.evaluate() + FUTILITY_MARGIN < alpha;

    for &m in move_list.iter() {
        if !pos.make_move(m) {
            continue;
        }
        moves_made += 1;

        let is_capture = m.is_capture();
        let is_check = pos.is_check();
        let is_promotion = m.is_promo();

        let is_interesting = is_capture || is_promotion || is_check;

        if futility_pruning_legal && !is_interesting {
            pos.unmake_move();
            continue;
        }

        let check_ext = usize::from(is_check);
        let extension = check_ext;

        let score = if moves_made == 1 {
            // first move (presumably the PV-move)
            -alpha_beta(pos, info, depth - 1 + extension, -beta, -alpha)
        } else {
            // nullwindow searches to prove PV.
            // we only reduce when a set of conditions are true:
            // 1. we're not in a branch of the search that has *already* been reduced higher up
            // 2. the move we're about to make isn't "interesting" (i.e. it's not a capture, a promotion, a check, or a killer)
            // 3. we're at a greater depth than 2.
            let can_reduce = !is_interesting && depth > 2 && extension == 0;
            let reduction = usize::from(can_reduce) * if moves_made >= 7 && depth > 5 { depth / 3 } else { 1 };
            // perform a zero-window search, possibly with a reduction
            let r = -alpha_beta(pos, info, depth - 1 + extension - reduction, -alpha - 1, -alpha);
            if r > alpha && r < beta {
                // if the zero-window search fails high, perform a full window search at full depth
                -alpha_beta(pos, info, depth - 1 + extension, -beta, -alpha)
            } else {
                r
            }
        };
        pos.unmake_move();

        if info.stopped {
            return 0;
        }

        if score > alpha {
            if score >= beta {
                // we failed high, so this is a cut-node
                if moves_made == 1 {
                    info.failhigh_first += 1.0;
                }
                info.failhigh += 1.0;

                if !is_capture {
                    pos.insert_killer(m);
                }

                return beta;
            }
            alpha = score;
            best_move = m;
            if !is_capture {
                pos.insert_history(m, history_score);
            }
        }
    }

    if moves_made == 0 {
        if pos.is_check() {
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            return -MATE_SCORE + pos.ply() as i32;
        }
        return DRAW_SCORE;
    }

    if alpha != original_alpha {
        // we raised alpha, so this is a PV-node
        pos.store_pv_move(best_move);
    } else {
        // we didn't raise alpha, so this is an all-node
    }

    alpha
}
