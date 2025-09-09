#![allow(clippy::too_many_arguments)]

pub mod parameters;
pub mod pv;

use std::{
    sync::atomic::{AtomicU64, Ordering},
    thread,
};

use arrayvec::ArrayVec;

use crate::{
    cfor,
    chess::{
        board::{
            movegen::{self, MAX_POSITION_MOVES, RAY_FULL},
            Board,
        },
        chessmove::Move,
        piece::{Colour, Piece, PieceType},
        squareset::SquareSet,
        types::{ContHistIndex, Square},
        CHESS960,
    },
    evaluation::{
        evaluate, is_game_theoretic_score, mate_in, mated_in, see_value, tb_loss_in, tb_win_in,
        MATE_SCORE, MINIMUM_TB_WIN_SCORE,
    },
    history::caphist_piece_type,
    historytable::{
        cont1_history_bonus, cont1_history_malus, cont2_history_bonus, cont2_history_malus,
        main_history_bonus, main_history_malus,
    },
    lookups::HM_CLOCK_KEYS,
    movepicker::{MovePicker, Stage},
    search::pv::PVariation,
    searchinfo::SearchInfo,
    tablebases::{self, probe::WDL},
    threadlocal::ThreadData,
    threadpool::{self, ScopeExt},
    transpositiontable::{Bound, TTHit},
    uci,
    util::{INFINITY, MAX_DEPTH, VALUE_NONE},
};

use self::parameters::Config;

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

const ASPIRATION_EVAL_DIVISOR: i32 = 29614;
const DELTA_INITIAL: i32 = 12;
const DELTA_BASE_MUL: i32 = 40;
const DELTA_REDUCTION_MUL: i32 = 15;
const RFP_MARGIN: i32 = 84;
const RFP_IMPROVING_MARGIN: i32 = 75;
const NMP_IMPROVING_MARGIN: i32 = 106;
const NMP_DEPTH_MUL: i32 = -8;
const NMP_REDUCTION_EVAL_DIVISOR: i32 = 150;
const SEE_QUIET_MARGIN: i32 = -70;
const SEE_TACTICAL_MARGIN: i32 = -25;
const FUTILITY_COEFF_0: i32 = 110;
const FUTILITY_COEFF_1: i32 = 61;
const RAZORING_COEFF_0: i32 = 63;
const RAZORING_COEFF_1: i32 = 270;
const PROBCUT_MARGIN: i32 = 184;
const PROBCUT_IMPROVING_MARGIN: i32 = 69;
const DOUBLE_EXTENSION_MARGIN: i32 = 12;
const TRIPLE_EXTENSION_MARGIN: i32 = 166;
const LMR_BASE: f64 = 95.0;
const LMR_DIVISION: f64 = 260.0;
const QS_SEE_BOUND: i32 = -50;
const MAIN_SEE_BOUND: i32 = -86;
const DO_DEEPER_BASE_MARGIN: i32 = 45;
const DO_DEEPER_DEPTH_MARGIN: i32 = 14;
const HISTORY_PRUNING_MARGIN: i32 = -4942;
const QS_FUTILITY: i32 = 341;
const SEE_STAT_SCORE_MUL: i32 = 20;

const HISTORY_LMR_DIVISOR: i32 = 17205;
const LMR_REFUTATION_MUL: i32 = 860;
const LMR_NON_PV_MUL: i32 = 1148;
const LMR_TTPV_MUL: i32 = 1432;
const LMR_CUT_NODE_MUL: i32 = 1412;
const LMR_NON_IMPROVING_MUL: i32 = 456;
const LMR_TT_CAPTURE_MUL: i32 = 1301;
const LMR_CHECK_MUL: i32 = 1217;

const MAIN_HISTORY_BONUS_MUL: i32 = 341;
const MAIN_HISTORY_BONUS_OFFSET: i32 = 178;
const MAIN_HISTORY_BONUS_MAX: i32 = 2298;
const MAIN_HISTORY_MALUS_MUL: i32 = 172;
const MAIN_HISTORY_MALUS_OFFSET: i32 = 407;
const MAIN_HISTORY_MALUS_MAX: i32 = 835;
const CONT1_HISTORY_BONUS_MUL: i32 = 187;
const CONT1_HISTORY_BONUS_OFFSET: i32 = 234;
const CONT1_HISTORY_BONUS_MAX: i32 = 3217;
const CONT1_HISTORY_MALUS_MUL: i32 = 323;
const CONT1_HISTORY_MALUS_OFFSET: i32 = 436;
const CONT1_HISTORY_MALUS_MAX: i32 = 1442;
const CONT2_HISTORY_BONUS_MUL: i32 = 177;
const CONT2_HISTORY_BONUS_OFFSET: i32 = 139;
const CONT2_HISTORY_BONUS_MAX: i32 = 1473;
const CONT2_HISTORY_MALUS_MUL: i32 = 218;
const CONT2_HISTORY_MALUS_OFFSET: i32 = 66;
const CONT2_HISTORY_MALUS_MAX: i32 = 1248;
const TACTICAL_HISTORY_BONUS_MUL: i32 = 136;
const TACTICAL_HISTORY_BONUS_OFFSET: i32 = 458;
const TACTICAL_HISTORY_BONUS_MAX: i32 = 2007;
const TACTICAL_HISTORY_MALUS_MUL: i32 = 49;
const TACTICAL_HISTORY_MALUS_OFFSET: i32 = 401;
const TACTICAL_HISTORY_MALUS_MAX: i32 = 1464;

const PAWN_CORRHIST_WEIGHT: i32 = 1583;
const MAJOR_CORRHIST_WEIGHT: i32 = 1495;
const MINOR_CORRHIST_WEIGHT: i32 = 1304;
const NONPAWN_CORRHIST_WEIGHT: i32 = 1813;

const EVAL_POLICY_IMPROVEMENT_SCALE: i32 = 214;
const EVAL_POLICY_OFFSET: i32 = -4;
const EVAL_POLICY_UPDATE_MAX: i32 = 95;

const TIME_MANAGER_UPDATE_MIN_DEPTH: i32 = 4;

const HINDSIGHT_EXT_DEPTH: i32 = 2097;
const HINDSIGHT_RED_DEPTH: i32 = 2421;
const HINDSIGHT_RED_EVAL: i32 = 141;

const OPTIMISM_OFFSET: i32 = 189;
const OPTIMISM_MATERIAL_BASE: i32 = 2320;

static TB_HITS: AtomicU64 = AtomicU64::new(0);

pub trait NodeType {
    /// Whether this node is on the principal variation.
    const PV: bool;
    /// Whether this node is the root of the search tree.
    const ROOT: bool;
    /// The node type that arises from a PV search in this node.
    type Next: NodeType;
}

/// The root node of the search tree.
struct Root;
/// A node with a non-null search window.
struct OnPV;
/// A node with a null window, where we're trying to prove a PV.
struct OffPV;
/// A root node with a null window used for time-management searches.
struct CheckForced;

impl NodeType for Root {
    const PV: bool = true;
    const ROOT: bool = true;
    type Next = OnPV;
}
impl NodeType for OnPV {
    const PV: bool = true;
    const ROOT: bool = false;
    type Next = Self;
}
impl NodeType for OffPV {
    const PV: bool = false;
    const ROOT: bool = false;
    type Next = Self;
}
impl NodeType for CheckForced {
    const PV: bool = false;
    const ROOT: bool = true;
    type Next = OffPV;
}

pub trait SmpThreadType {
    const MAIN_THREAD: bool;
}
pub struct MainThread;
pub struct HelperThread;
impl SmpThreadType for MainThread {
    const MAIN_THREAD: bool = true;
}
impl SmpThreadType for HelperThread {
    const MAIN_THREAD: bool = false;
}

/// Performs the root search. Returns the score of the position, from white's perspective, and the best move.
#[allow(clippy::too_many_lines)]
pub fn search_position(
    pool: &[threadpool::WorkerThread],
    thread_headers: &mut [Box<ThreadData>],
) -> (i32, Option<Move>) {
    for t in &mut *thread_headers {
        t.board.zero_height();
        t.info.set_up_for_search();
        t.set_up_for_search();
    }
    TB_HITS.store(0, Ordering::Relaxed);

    let legal_moves = thread_headers[0].board.legal_moves();
    if legal_moves.is_empty() {
        eprintln!("info string warning search called on a position with no legal moves");
        if thread_headers[0].board.in_check() {
            println!("info depth 0 score mate 0");
        } else {
            println!("info depth 0 score cp 0");
        }
        println!("bestmove (none)");
        return (0, None);
    }
    if legal_moves.len() == 1 {
        thread_headers[0].info.clock.notify_one_legal_move();
    }

    // Probe the tablebases if we're in a TB position and in a game.
    if thread_headers[0].info.clock.is_dynamic() {
        if let Some((best_move, score)) =
            tablebases::probe::get_tablebase_move(&thread_headers[0].board)
        {
            let pv = &mut thread_headers[0].pvs[1];
            pv.load_from(best_move, &PVariation::default());
            pv.score = score;
            TB_HITS.store(1, Ordering::SeqCst);
            thread_headers[0].completed = 1;
            readout_info(
                &thread_headers[0],
                &thread_headers[0].info,
                Bound::Exact,
                1,
                true,
            );
            if thread_headers[0].info.print_to_stdout {
                println!(
                    "bestmove {}",
                    best_move.display(CHESS960.load(Ordering::Relaxed))
                );
            }
            return (score, Some(best_move));
        }
    }

    let global_stopped = thread_headers[0].info.stopped;
    assert!(
        !global_stopped.load(Ordering::SeqCst),
        "global_stopped must be false"
    );

    // start search threads:
    let (t1, rest) = thread_headers.split_first_mut().unwrap();
    let (w1, rest_workers) = pool.split_first().unwrap();
    thread::scope(|s| {
        let mut handles = Vec::with_capacity(pool.len());
        handles.push(s.spawn_into(
            || {
                iterative_deepening::<MainThread>(t1);
                global_stopped.store(true, Ordering::SeqCst);
            },
            w1,
        ));
        for (t, w) in rest.iter_mut().zip(rest_workers) {
            handles.push(s.spawn_into(
                || {
                    iterative_deepening::<HelperThread>(t);
                },
                w,
            ));
        }
        for handle in handles {
            handle.join();
        }
    });

    let best_thread = select_best(thread_headers);
    let pv = best_thread.pv();
    let best_move = pv
        .moves()
        .first()
        .copied()
        .unwrap_or_else(|| default_move(&thread_headers[0]));

    // always give a final info log before ending search
    readout_info(
        best_thread,
        &thread_headers[0].info,
        Bound::Exact,
        thread_headers[0].info.nodes.get_global(),
        true,
    );

    if thread_headers[0].info.print_to_stdout {
        let maybe_ponder = pv.moves().get(1).map_or_else(String::new, |ponder_move| {
            format!(
                " ponder {}",
                ponder_move.display(CHESS960.load(Ordering::Relaxed))
            )
        });
        println!(
            "bestmove {}{maybe_ponder}",
            best_move.display(CHESS960.load(Ordering::Relaxed))
        );
        #[cfg(feature = "stats")]
        {
            thread_headers[0].info.print_stats();
            #[allow(clippy::cast_precision_loss)]
            let branching_factor = (thread_headers[0].info.nodes.get_global() as f64)
                .powf(1.0 / thread_headers[0].completed as f64);
            println!("branching factor: {branching_factor}");
        }
    }

    assert!(
        legal_moves.contains(&best_move),
        "search returned an illegal move."
    );

    thread_headers[0]
        .info
        .stopped
        .store(false, Ordering::Relaxed);

    (
        if thread_headers[0].board.turn() == Colour::White {
            pv.score
        } else {
            -pv.score
        },
        Some(best_move),
    )
}

/// Performs the iterative deepening search.
/// Returns the score of the position, from the side to move's perspective, and the best move.
/// For Lazy SMP, the main thread calls this function with `T0 = true`, and the helper threads with `T0 = false`.
#[allow(clippy::too_many_lines)]
fn iterative_deepening<ThTy: SmpThreadType>(t: &mut ThreadData) {
    assert!(
        !ThTy::MAIN_THREAD || t.thread_id == 0,
        "main thread must have thread_id 0"
    );
    let mut pv = PVariation::default();
    let max_depth = dyn_max_depth(t);
    let starting_depth = 1 + t.thread_id % 10;
    let mut average_value = VALUE_NONE;
    'deepening: for iteration in starting_depth..=max_depth {
        t.iteration = iteration;
        t.depth = i32::try_from(iteration).unwrap();
        t.optimism = [0; 2];

        let min_depth = (t.depth / 2).max(1);

        let mut alpha = -INFINITY;
        let mut beta = INFINITY;

        let mut delta = t.info.conf.delta_initial;
        let mut reduction = 0;

        if t.depth > 1 {
            let us = t.board.turn();
            let offset = t.info.conf.optimism_offset;
            t.optimism[us] = 128 * average_value / (average_value.abs() + offset);
            t.optimism[!us] = -t.optimism[us];

            delta += average_value * average_value / t.info.conf.aspiration_eval_divisor;

            alpha = (average_value - delta).max(-INFINITY);
            beta = (average_value + delta).min(INFINITY);
        }

        // aspiration loop:
        loop {
            let root_draft = (t.depth - reduction).max(min_depth);
            pv.score = alpha_beta::<Root>(&mut pv, t, root_draft, alpha, beta, false);
            if t.info.check_up() {
                break 'deepening; // we've been told to stop searching.
            }

            if pv.score <= alpha {
                if ThTy::MAIN_THREAD {
                    t.pv_mut().score = pv.score;
                    readout_info(t, &t.info, Bound::Upper, t.info.nodes.get_global(), false);
                    t.info
                        .clock
                        .report_aspiration_fail(t.depth, Bound::Upper, &t.info.conf);
                }
                beta = (alpha + beta) / 2;
                alpha = (pv.score - delta).max(-INFINITY);
                reduction = 0;
                // search failed low, so we might have to
                // revert a fail-high pv update
                t.revert_best_line();
            } else if pv.score >= beta {
                t.update_best_line(&pv);
                if ThTy::MAIN_THREAD {
                    readout_info(t, &t.info, Bound::Lower, t.info.nodes.get_global(), false);
                    t.info
                        .clock
                        .report_aspiration_fail(t.depth, Bound::Lower, &t.info.conf);
                }
                beta = (pv.score + delta).min(INFINITY);
                reduction += 1;
                // decrement depth:
                if !is_game_theoretic_score(pv.score) {
                    t.depth = (t.depth - 1).max(min_depth);
                }
            } else {
                t.update_best_line(&pv);
                break;
            }

            delta += delta
                * (t.info.conf.delta_base_mul + t.info.conf.delta_reduction_mul * reduction)
                / 128;
        }

        // if we've made it here, it means we got an exact score.
        let score = pv.score;
        let best_move = pv
            .moves()
            .first()
            .copied()
            .unwrap_or_else(|| default_move(t));

        average_value = if average_value == VALUE_NONE {
            score
        } else {
            (2 * score + average_value) / 3
        };

        if ThTy::MAIN_THREAD {
            readout_info(t, &t.info, Bound::Exact, t.info.nodes.get_global(), false);

            if let Some(margin) = t.info.clock.check_for_forced_move(t.depth) {
                let saved_seldepth = t.info.seldepth;
                let forced = is_forced(margin, t, best_move, score, (t.depth - 1) / 2);
                t.info.seldepth = saved_seldepth;

                if forced {
                    t.info.clock.report_forced_move(t.depth, &t.info.conf);
                }
            }

            if t.depth > TIME_MANAGER_UPDATE_MIN_DEPTH {
                let bm_frac = if t.depth > 8 {
                    let best_move_subtree_size =
                        t.info.root_move_nodes[best_move.from()][best_move.history_to_square()];
                    let tree_size = t.info.nodes.get_local();
                    #[allow(clippy::cast_precision_loss)]
                    Some(best_move_subtree_size as f64 / tree_size as f64)
                } else {
                    None
                };
                t.info.clock.report_completed_depth(
                    t.depth,
                    pv.score,
                    pv.moves[0],
                    bm_frac,
                    &t.info.conf,
                );
            }
        }

        if t.info.check_up() {
            break 'deepening;
        }

        if ThTy::MAIN_THREAD {
            // consider stopping early if we've neatly completed a depth,
            // or if we were told to find a mate and we found one,
            // or if we're on the clock and we've solved a mate.
            if t.info.clock.is_past_opt_time(t.info.nodes.get_global())
                || (iteration > 10
                    && (t.info.clock.solved_breaker(pv.score)
                        || t.info.clock.mate_found_breaker(pv.score)))
            {
                t.info.stopped.store(true, Ordering::SeqCst);
                break 'deepening;
            }
        }
    }
}

fn dyn_max_depth(t: &ThreadData<'_>) -> usize {
    t.info.clock.limit().depth().unwrap_or(MAX_DEPTH - 1)
}

/// Give a legal default move in the case where we don't have enough time to search.
fn default_move(t: &ThreadData) -> Move {
    let tt_move =
        t.tt.probe_move(t.board.state.keys.zobrist)
            .and_then(|e| e.0);

    let mut mp = MovePicker::new(tt_move, t.killer_move_table[t.board.height()], 0);

    std::iter::from_fn(|| mp.next(t))
        .find(|&m| t.board.is_legal(m))
        .expect("Board::default_move called on a position with no legal moves")
}

/// Perform a tactical resolution search, searching only captures and promotions.
#[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
pub fn quiescence<NT: NodeType>(
    pv: &mut PVariation,
    t: &mut ThreadData,
    mut alpha: i32,
    beta: i32,
) -> i32 {
    #[cfg(debug_assertions)]
    t.board.check_validity().unwrap();

    if t.info.nodes.just_ticked_over() && t.info.check_up() {
        return 0;
    }

    let key = t.board.state.keys.zobrist ^ HM_CLOCK_KEYS[t.board.state.fifty_move_counter as usize];

    let mut local_pv = PVariation::default();
    let l_pv = &mut local_pv;

    pv.moves.clear();

    let height = t.board.height();
    t.info.seldepth = t.info.seldepth.max(i32::try_from(height).unwrap());

    // check draw
    if t.board.is_draw() {
        return draw_score(t, t.info.nodes.get_local(), t.board.turn());
    }

    let in_check = t.board.in_check();

    // are we too deep?
    if height > MAX_DEPTH - 1 {
        return if in_check {
            0
        } else {
            evaluate(t, t.info.nodes.get_local())
        };
    }

    // upcoming repetition detection
    if alpha < 0 && t.board.has_game_cycle(height) {
        alpha = 0;
        if alpha >= beta {
            return alpha;
        }
    }

    let clock = t.board.fifty_move_counter();

    // probe the TT and see if we get a cutoff.
    let tt_hit = if let Some(hit) = t.tt.probe(key, height) {
        if !NT::PV
            && clock < 80
            && (hit.bound == Bound::Exact
                || (hit.bound == Bound::Lower && hit.value >= beta)
                || (hit.bound == Bound::Upper && hit.value <= alpha))
        {
            return hit.value;
        }

        Some(hit)
    } else {
        None
    };

    t.ss[height].ttpv = NT::PV || tt_hit.is_some_and(|hit| hit.was_pv);

    let raw_eval;
    let stand_pat;

    if in_check {
        // could be being mated!
        raw_eval = VALUE_NONE;
        stand_pat = -INFINITY;
    } else if let Some(TTHit { eval: tt_eval, .. }) = &tt_hit {
        // if we have a TT hit, check the cached TT eval.
        let v = *tt_eval;
        if v == VALUE_NONE {
            // regenerate the static eval if it's VALUE_NONE.
            raw_eval = evaluate(t, t.info.nodes.get_local());
        } else {
            // if the TT eval is not VALUE_NONE, use it.
            raw_eval = v;
        }
        let adj_eval = adj_shuffle(t, raw_eval, clock) + t.correction();

        // try correcting via search score from TT.
        // notably, this doesn't work for main search for ~reasons.
        let (tt_flag, tt_value) = tt_hit
            .as_ref()
            .map_or((Bound::None, VALUE_NONE), |tte| (tte.bound, tte.value));
        if tt_flag == Bound::Exact
            || tt_flag == Bound::Upper && tt_value < adj_eval
            || tt_flag == Bound::Lower && tt_value > adj_eval
        {
            stand_pat = tt_value;
        } else {
            stand_pat = adj_eval;
        }
    } else {
        // otherwise, use the static evaluation.
        raw_eval = evaluate(t, t.info.nodes.get_local());

        // store the eval into the TT. We know that we won't overwrite anything,
        // because this branch is one where there wasn't a TT-hit.
        t.tt.store(
            key,
            height,
            None,
            VALUE_NONE,
            raw_eval,
            Bound::None,
            0,
            t.ss[height].ttpv,
        );

        stand_pat = adj_shuffle(t, raw_eval, clock) + t.correction();
    }

    if stand_pat >= beta {
        // return stand_pat instead of beta, this is fail-soft
        return stand_pat;
    }

    let original_alpha = alpha;
    if stand_pat > alpha {
        alpha = stand_pat;
    }

    let mut best_move = None;
    let mut best_score = stand_pat;

    let mut moves_made = 0;
    let mut move_picker =
        MovePicker::new(tt_hit.and_then(|e| e.mov), None, t.info.conf.qs_see_bound);
    move_picker.skip_quiets = !in_check;

    let futility = stand_pat + t.info.conf.qs_futility;

    while let Some(m) = move_picker.next(t) {
        t.tt.prefetch(t.board.key_after(m));
        if !t.board.is_legal(m) {
            continue;
        }
        let is_tactical = t.board.is_tactical(m);
        let is_recapture = Some(m.to()) == t.ss[height - 1].searching.map(Move::to);
        if best_score > -MINIMUM_TB_WIN_SCORE
            && is_tactical
            && !in_check
            && futility <= alpha
            && !is_recapture
            && !static_exchange_eval(&t.board, &t.info, m, 1)
        {
            if best_score < futility {
                best_score = futility;
            }
            continue;
        }
        t.ss[height].searching = Some(m);
        t.ss[height].searching_tactical = is_tactical;
        let moved = t.board.state.mailbox[m.from()].unwrap();
        t.ss[height].ch_idx = ContHistIndex {
            piece: moved,
            to: m.history_to_square(),
        };
        t.board.make_move(m, &mut t.nnue);
        // move found, we can start skipping quiets again:
        move_picker.skip_quiets = true;
        t.info.nodes.increment();
        moves_made += 1;

        let score = -quiescence::<NT::Next>(l_pv, t, -beta, -alpha);
        t.board.unmake_move(&mut t.nnue);

        if score > best_score {
            best_score = score;
            if score > alpha {
                best_move = Some(m);
                alpha = score;
                if NT::PV {
                    pv.load_from(m, l_pv);
                }
            }
            if alpha >= beta {
                #[cfg(feature = "stats")]
                t.info.log_fail_high::<true>(moves_made - 1);
                break; // fail-high
            }
        }
    }

    if moves_made == 0 && in_check {
        #[cfg(debug_assertions)]
        t.board.assert_mated();
        return mated_in(height);
    }

    let flag = if best_score >= beta {
        Bound::Lower
    } else if best_score > original_alpha {
        Bound::Exact
    } else {
        Bound::Upper
    };

    t.tt.store(
        key,
        height,
        best_move,
        best_score,
        raw_eval,
        flag,
        0,
        t.ss[height].ttpv,
    );

    best_score
}

/// Perform alpha-beta minimax search.
#[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
pub fn alpha_beta<NT: NodeType>(
    pv: &mut PVariation,
    t: &mut ThreadData,
    mut depth: i32,
    mut alpha: i32,
    mut beta: i32,
    cut_node: bool,
) -> i32 {
    #[cfg(debug_assertions)]
    t.board.check_validity().unwrap();

    let mut local_pv = PVariation::default();
    let l_pv = &mut local_pv;

    let key = t.board.state.keys.zobrist ^ HM_CLOCK_KEYS[t.board.state.fifty_move_counter as usize];

    if depth <= 0 {
        return quiescence::<NT::Next>(pv, t, alpha, beta);
    }

    pv.moves.clear();

    if t.info.nodes.just_ticked_over() && t.info.check_up() {
        return 0;
    }

    let height = t.board.height();

    debug_assert_eq!(height == 0, NT::ROOT);
    debug_assert!(!(NT::PV && cut_node));
    debug_assert!(
        NT::PV || alpha + 1 == beta,
        "Non-PV nodes must be zero-window."
    );

    t.info.seldepth = if NT::ROOT {
        0
    } else {
        t.info.seldepth.max(i32::try_from(height).unwrap())
    };

    let in_check = t.board.in_check();

    if !NT::ROOT {
        // check draw
        if t.board.is_draw() {
            return draw_score(t, t.info.nodes.get_local(), t.board.turn());
        }

        // are we too deep?
        let max_height = MAX_DEPTH.min(uci::GO_MATE_MAX_DEPTH.load(Ordering::SeqCst));
        if height >= max_height {
            return if in_check {
                0
            } else {
                evaluate(t, t.info.nodes.get_local())
            };
        }

        // mate-distance pruning.
        alpha = alpha.max(mated_in(height));
        beta = beta.min(mate_in(height + 1));
        if alpha >= beta {
            return alpha;
        }

        // upcoming repetition detection
        if alpha < 0 && t.board.has_game_cycle(height) {
            alpha = 0;
            if alpha >= beta {
                return alpha;
            }
        }
    }

    let clock = t.board.fifty_move_counter();

    let excluded = t.ss[height].excluded;
    let tt_hit = if excluded.is_none() {
        if let Some(hit) = t.tt.probe(key, height) {
            if !NT::PV
                && hit.depth >= depth + i32::from(hit.value >= beta)
                && clock < 80
                && (hit.bound == Bound::Exact
                    || (hit.bound == Bound::Lower && hit.value >= beta)
                    || (hit.bound == Bound::Upper && hit.value <= alpha))
            {
                if let Some(mov) = hit.mov {
                    // add to the history of a quiet move that fails high here.
                    if hit.value >= beta
                        && !t.board.is_tactical(mov)
                        && t.board.is_pseudo_legal(mov)
                    {
                        let from = mov.from();
                        let to = mov.history_to_square();
                        let moved = t.board.state.mailbox[from].unwrap();
                        let threats = t.board.state.threats.all;
                        update_quiet_history_single::<false>(
                            t, from, to, moved, threats, depth, true,
                        );
                    }
                }

                return hit.value;
            }

            Some(hit)
        } else {
            None
        }
    } else {
        None // do not probe the TT if we're in a singular-verification search.
    };

    if excluded.is_none() {
        t.ss[height].ttpv = NT::PV || tt_hit.is_some_and(|hit| hit.was_pv);
    }

    // Probe the tablebases.
    let (mut syzygy_max, mut syzygy_min) = (MATE_SCORE, -MATE_SCORE);
    let cardinality = u32::from(tablebases::probe::get_max_pieces_count());
    let n_men = t.board.state.bbs.occupied().count();
    if !NT::ROOT
            && excluded.is_none() // do not probe the tablebases if we're in a singular-verification search.
            && uci::SYZYGY_ENABLED.load(Ordering::SeqCst)
            && (depth >= uci::SYZYGY_PROBE_DEPTH.load(Ordering::SeqCst)
                || n_men < cardinality)
            && n_men <= cardinality
    {
        if let Some(wdl) = tablebases::probe::get_wdl(&t.board) {
            TB_HITS.fetch_add(1, Ordering::Relaxed);

            let tb_value = match wdl {
                WDL::Win => tb_win_in(height),
                WDL::Loss => tb_loss_in(height),
                WDL::Draw => draw_score(t, t.info.nodes.get_buffer(), t.board.turn()),
            };

            let tb_bound = match wdl {
                WDL::Win => Bound::Lower,
                WDL::Loss => Bound::Upper,
                WDL::Draw => Bound::Exact,
            };

            if tb_bound == Bound::Exact
                || (tb_bound == Bound::Lower && tb_value >= beta)
                || (tb_bound == Bound::Upper && tb_value <= alpha)
            {
                t.tt.store(
                    key,
                    height,
                    None,
                    tb_value,
                    VALUE_NONE,
                    tb_bound,
                    depth,
                    t.ss[height].ttpv,
                );
                return tb_value;
            }

            if NT::PV && tb_bound == Bound::Lower {
                alpha = alpha.max(tb_value);
                syzygy_min = tb_value;
            }

            if NT::PV && tb_bound == Bound::Upper {
                syzygy_max = tb_value;
            }
        }
    }

    let raw_eval;
    let static_eval;
    let eval;
    let correction;

    if in_check {
        // when we're in check, it could be checkmate, so it's unsound to use evaluate().
        raw_eval = VALUE_NONE;
        static_eval = VALUE_NONE;
        eval = VALUE_NONE;
        correction = 0;
    } else if excluded.is_some() {
        // if we're in a singular-verification search, we already have the static eval.
        // we can set raw_eval to whatever we like, because we're not going to be saving it.
        raw_eval = VALUE_NONE;
        static_eval = t.ss[height].static_eval;
        eval = t.ss[height].eval;
        correction = 0;
        t.nnue.hint_common_access(&t.board, t.nnue_params);
    } else if let Some(tte) = &tt_hit {
        let v = tte.eval; // if we have a TT hit, check the cached TT eval.
        if v == VALUE_NONE {
            // regenerate the static eval if it's VALUE_NONE.
            raw_eval = evaluate(t, t.info.nodes.get_local());
        } else {
            // if the TT eval is not VALUE_NONE, use it.
            raw_eval = v;
            if NT::PV {
                t.nnue.hint_common_access(&t.board, t.nnue_params);
            }
        }
        correction = t.correction();
        static_eval = adj_shuffle(t, raw_eval, clock) + correction;
        if tte.value != VALUE_NONE
            && match tte.bound {
                Bound::Upper => tte.value < static_eval,
                Bound::Lower => tte.value > static_eval,
                Bound::Exact => true,
                Bound::None => false,
            }
        {
            eval = tte.value;
        } else {
            eval = static_eval;
        }
    } else {
        // otherwise, use the static evaluation.
        raw_eval = evaluate(t, t.info.nodes.get_local());

        // store the eval into the TT. We know that we won't overwrite anything,
        // because this branch is one where there wasn't a TT-hit.
        t.tt.store(
            key,
            height,
            None,
            VALUE_NONE,
            raw_eval,
            Bound::None,
            0,
            t.ss[height].ttpv,
        );

        correction = t.correction();
        static_eval = adj_shuffle(t, raw_eval, clock) + correction;
        eval = static_eval;
    }

    t.ss[height].static_eval = static_eval;
    t.ss[height].eval = eval;

    // value-difference based policy update.
    if !NT::ROOT {
        let ss_prev = &t.ss[height - 1];
        if let Some(mov) = ss_prev.searching {
            if ss_prev.static_eval != VALUE_NONE
                && static_eval != VALUE_NONE
                && !ss_prev.searching_tactical
            {
                let from = mov.from();
                let to = mov.history_to_square();
                let moved = t.board.state.mailbox[to].expect("Cannot fail, move has been made.");
                debug_assert_eq!(moved.colour(), !t.board.turn());
                let threats = t.board.history().last().unwrap().threats.all;
                let improvement =
                    -(ss_prev.static_eval + static_eval) + t.info.conf.eval_policy_offset;
                let delta = i32::clamp(
                    improvement * t.info.conf.eval_policy_improvement_scale / 32,
                    -t.info.conf.eval_policy_update_max,
                    t.info.conf.eval_policy_update_max,
                );
                t.update_history_single(from, to, moved, threats, delta);
            }
        }
    }

    // "improving" is true when the current position has a better static evaluation than the one from a fullmove ago.
    // if a position is "improving", we can be more aggressive with beta-reductions (eval is too high),
    // but we should be less aggressive with alpha-reductions (eval is too low).
    // some engines gain by using improving to increase LMR, but this shouldn't work imo, given that LMR is
    // neutral with regards to the evaluation.
    let improving = if in_check {
        false
    } else if height >= 2 && t.ss[height - 2].static_eval != VALUE_NONE {
        static_eval > t.ss[height - 2].static_eval
    } else if height >= 4 && t.ss[height - 4].static_eval != VALUE_NONE {
        static_eval > t.ss[height - 4].static_eval
    } else {
        true
    };

    t.ss[height].dextensions = if NT::ROOT {
        0
    } else {
        t.ss[height - 1].dextensions
    };

    // clear out the next killer move.
    t.killer_move_table[height + 1] = None;

    let tt_move = tt_hit.and_then(|hit| hit.mov);
    let tt_capture = matches!(tt_move, Some(mv) if t.board.is_capture(mv));

    // whole-node techniques:
    if !NT::ROOT && !NT::PV && !in_check && excluded.is_none() {
        if t.ss[height - 1].reduction >= t.info.conf.hindsight_ext_depth
            && static_eval + t.ss[height - 1].static_eval < 0
        {
            depth += 1;
        }

        if depth >= 2
            && t.ss[height - 1].reduction >= t.info.conf.hindsight_red_depth
            && t.ss[height - 1].static_eval != VALUE_NONE
            && static_eval + t.ss[height - 1].static_eval > t.info.conf.hindsight_red_eval
        {
            depth -= 1;
        }

        // razoring.
        // if the static eval is too low, check if qsearch can beat alpha.
        // if it can't, we can prune the node.
        if alpha < 2000
            && eval < alpha - t.info.conf.razoring_coeff_0 - t.info.conf.razoring_coeff_1 * depth
        {
            let v = quiescence::<OffPV>(pv, t, alpha, beta);
            if v <= alpha {
                return v;
            }
        }

        // static null-move pruning, also called beta pruning,
        // reverse futility pruning, and child node futility pruning.
        // if the static eval is too high, we can prune the node.
        // this is a generalisation of stand_pat in quiescence search.
        if !t.ss[height].ttpv
            && depth < 9
            && eval >= beta
            && eval - rfp_margin(&t.board, &t.info, depth, improving, correction) >= beta
            && (tt_move.is_none() || tt_capture)
            && beta > -MINIMUM_TB_WIN_SCORE
        {
            return beta + (eval - beta) / 3;
        }

        // null-move pruning.
        // if we can give the opponent a free move while retaining
        // a score above beta, we can prune the node.
        if cut_node
            && t.ss[height - 1].searching.is_some()
            && depth > 2
            && static_eval
                + i32::from(improving) * t.info.conf.nmp_improving_margin
                + depth * t.info.conf.nmp_depth_mul
                >= beta
            && !t.nmp_banned_for(t.board.turn())
            && t.board.zugzwang_unlikely()
            && !matches!(tt_hit, Some(TTHit { value: v, bound: Bound::Upper, .. }) if v < beta)
        {
            t.tt.prefetch(t.board.key_after_null_move());
            let r = 4
                + depth / 3
                + std::cmp::min(
                    (static_eval - beta) / t.info.conf.nmp_reduction_eval_divisor,
                    4,
                )
                + i32::from(tt_capture);
            let nm_depth = depth - r;
            t.ss[height].searching = None;
            t.ss[height].searching_tactical = false;
            t.ss[height].ch_idx = ContHistIndex {
                piece: Piece::new(t.board.turn(), PieceType::Pawn),
                to: Square::A1,
            };
            t.board.make_nullmove();
            let mut null_score = -alpha_beta::<OffPV>(l_pv, t, nm_depth, -beta, -beta + 1, false);
            t.board.unmake_nullmove();
            if t.info.stopped() {
                return 0;
            }
            if null_score >= beta {
                // don't return game-theoretic scores:
                if is_game_theoretic_score(null_score) {
                    null_score = beta;
                }
                // unconditionally cutoff if we're just too shallow.
                if depth < 12 && !is_game_theoretic_score(beta) {
                    return null_score;
                }
                // verify that it's *actually* fine to prune,
                // by doing a search with NMP disabled.
                // we disallow NMP for the side to move,
                // and if we hit the other side deeper in the tree
                // with sufficient depth, we'll disallow it for them too.
                t.ban_nmp_for(t.board.turn());
                let veri_score = alpha_beta::<OffPV>(l_pv, t, nm_depth, beta - 1, beta, false);
                t.unban_nmp_for(t.board.turn());
                if veri_score >= beta {
                    return null_score;
                }
            }
        }
    }

    // TT-reduction (IIR).
    if NT::PV && !matches!(tt_hit, Some(tte) if tte.depth + 4 > depth) {
        depth -= i32::from(depth >= 4);
    }

    // cutnode-based TT reduction.
    if cut_node
        && excluded.is_none()
        && (tt_move.is_none() || !matches!(tt_hit, Some(tte) if tte.depth + 4 > depth))
    {
        depth -= i32::from(depth >= 8);
    }

    // the margins for static-exchange-evaluation pruning for tactical and quiet moves.
    let see_table = [
        t.info.conf.see_tactical_margin * depth * depth,
        t.info.conf.see_quiet_margin * depth,
    ];

    // probcut:
    let pc_beta = std::cmp::min(
        beta + t.info.conf.probcut_margin
            - i32::from(improving) * t.info.conf.probcut_improving_margin,
        MINIMUM_TB_WIN_SCORE - 1,
    );
    // as usual, don't probcut in PV / check / singular verification / if there are GT truth scores in flight.
    // additionally, if we have a TT hit that's sufficiently deep, we skip trying probcut if the TT value indicates
    // that it's not going to be helpful.
    if !NT::PV
        && !in_check
        && excluded.is_none()
        && depth >= 5
        && !is_game_theoretic_score(beta)
        // don't probcut if we have a tthit with value < pcbeta and depth >= depth - 3:
        && !matches!(tt_hit, Some(TTHit { value: v, depth: d, .. }) if v < pc_beta && d >= depth - 3)
    {
        let tt_move_if_capture = tt_move.filter(|m| t.board.is_tactical(*m));
        let mut move_picker = MovePicker::new(tt_move_if_capture, None, 0);
        move_picker.skip_quiets = true;
        while let Some(m) = move_picker.next(t) {
            t.tt.prefetch(t.board.key_after(m));
            if !t.board.is_legal(m) {
                continue;
            }
            t.ss[height].searching = Some(m);
            t.ss[height].searching_tactical = true;
            let moved = t.board.state.mailbox[m.from()].unwrap();
            t.ss[height].ch_idx = ContHistIndex {
                piece: moved,
                to: m.history_to_square(),
            };
            t.board.make_move(m, &mut t.nnue);

            let mut value = -quiescence::<OffPV>(l_pv, t, -pc_beta, -pc_beta + 1);

            if value >= pc_beta {
                let pc_depth = depth - 3;
                value = -alpha_beta::<OffPV>(l_pv, t, pc_depth, -pc_beta, -pc_beta + 1, !cut_node);
            }

            t.board.unmake_move(&mut t.nnue);

            if value >= pc_beta {
                t.tt.store(
                    key,
                    height,
                    Some(m),
                    value,
                    raw_eval,
                    Bound::Lower,
                    depth - 3,
                    t.ss[height].ttpv,
                );
                return value;
            }
        }

        t.nnue.hint_common_access(&t.board, t.nnue_params);
    }

    let original_alpha = alpha;
    let mut best_move = None;
    let mut best_score = -INFINITY;
    let mut moves_made = 0;

    // number of quiet moves to try before we start pruning
    let lmp_threshold = t.info.lm_table.lmp_movecount(depth, improving);

    let killer = t.killer_move_table[height].filter(|m| !t.board.is_tactical(*m));
    let mut move_picker = MovePicker::new(tt_move, killer, t.info.conf.main_see_bound);

    let mut quiets_tried = ArrayVec::<_, MAX_POSITION_MOVES>::new();
    let mut tacticals_tried = ArrayVec::<_, MAX_POSITION_MOVES>::new();

    let maybe_singular = depth >= 5
        && excluded.is_none()
        && matches!(tt_hit, Some(TTHit { depth: tt_depth, bound: Bound::Lower | Bound::Exact, .. }) if tt_depth >= depth - 3);

    while let Some(m) = move_picker.next(t) {
        if excluded == Some(m) {
            continue;
        }

        t.tt.prefetch(t.board.key_after(m));
        if !t.board.is_legal(m) {
            continue;
        }

        let lmr_reduction = t.info.lm_table.lm_reduction(depth, moves_made);
        let lmr_depth = std::cmp::max(depth - lmr_reduction / 1024, 0);
        let is_quiet = !t.board.is_tactical(m);

        let mut stat_score = 0;

        let from = m.from();
        let hist_to = m.history_to_square();
        let moved = t.board.state.mailbox[from].unwrap();
        let threats = t.board.state.threats.all;
        let from_threat = usize::from(threats.contains_square(from));
        let to_threat = usize::from(threats.contains_square(hist_to));
        if is_quiet {
            stat_score += i32::from(t.main_hist[from_threat][to_threat][moved][hist_to]);
            if height >= 1 {
                let prev = t.ss[height - 1].ch_idx;
                stat_score += i32::from(t.cont_hist[prev.piece][prev.to][moved][hist_to]);
            }
            if height >= 2 {
                let prev = t.ss[height - 2].ch_idx;
                stat_score += i32::from(t.cont_hist[prev.piece][prev.to][moved][hist_to]);
            }
        } else {
            let capture = caphist_piece_type(&t.board, m);
            stat_score += i32::from(t.tactical_hist[to_threat][capture][moved][hist_to]);
        }

        // lmp & fp.
        if !NT::ROOT && !NT::PV && !in_check && best_score > -MINIMUM_TB_WIN_SCORE {
            // late move pruning
            // if we have made too many moves, we start skipping moves.
            if lmr_depth < 9 && moves_made >= lmp_threshold {
                move_picker.skip_quiets = true;
            }

            // history pruning
            // if this move's history score is too low, we start skipping moves.
            if is_quiet
                && (Some(m) != killer)
                && lmr_depth < 7
                && stat_score < t.info.conf.history_pruning_margin * (depth - 1)
            {
                move_picker.skip_quiets = true;
                continue;
            }

            // futility pruning
            // if the static eval is too low, we start skipping moves.
            let fp_margin = lmr_depth * t.info.conf.futility_coeff_1 + t.info.conf.futility_coeff_0;
            if is_quiet && lmr_depth < 6 && static_eval + fp_margin <= alpha {
                move_picker.skip_quiets = true;
            }
        }

        // static exchange evaluation pruning
        // simulate all captures flowing onto the target square, and if we come out badly, we skip the move.
        if !NT::ROOT
            && (!NT::PV || !cfg!(feature = "datagen"))
            && best_score > -MINIMUM_TB_WIN_SCORE
            && depth < 10
            && move_picker.stage > Stage::YieldGoodCaptures
            && t.board.state.threats.all.contains_square(m.to())
            && t.ss[height - 1].searching.is_some()
            && !static_exchange_eval(
                &t.board,
                &t.info,
                m,
                see_table[usize::from(is_quiet)]
                    - stat_score * t.info.conf.see_stat_score_mul / 1024,
            )
        {
            continue;
        }

        if is_quiet {
            quiets_tried.push(m);
        } else {
            tacticals_tried.push(m);
        }

        let nodes_before_search = t.info.nodes.get_local();
        t.info.nodes.increment();
        moves_made += 1;

        let extension;
        if NT::ROOT {
            extension = 0;
        } else if maybe_singular && Some(m) == tt_move {
            let TTHit {
                value: tt_value,
                bound,
                ..
            } = tt_hit.unwrap();
            let r_beta = singularity_margin(tt_value, depth);
            let r_depth = (depth - 1) / 2;
            t.ss[t.board.height()].excluded = Some(m);
            let value = alpha_beta::<OffPV>(
                &mut PVariation::default(),
                t,
                r_depth,
                r_beta - 1,
                r_beta,
                cut_node,
            );
            t.ss[t.board.height()].excluded = None;
            if value >= r_beta && r_beta >= beta {
                // multi-cut: if a move other than the best one beats beta,
                // then we can cut with relatively high confidence.
                return singularity_margin(tt_value, depth);
            }

            if value < r_beta {
                if !NT::PV
                    && t.ss[t.board.height()].dextensions <= 12
                    && value < r_beta - t.info.conf.dext_margin
                {
                    // double-extend if we failed low by a lot
                    extension = 2 + i32::from(is_quiet && value < r_beta - t.info.conf.text_margin);
                } else {
                    // normal singular extension
                    extension = 1;
                }
            } else if cut_node {
                // produce a strong negative extension if we didn't fail low on a cut-node.
                extension = -2;
            } else if tt_value >= beta || tt_value <= alpha {
                // the tt_value >= beta condition is a sort of "light multi-cut"
                // the tt_value <= alpha condition is from Weiss (https://github.com/TerjeKir/weiss/compare/2a7b4ed0...effa8349/).
                extension = -1;
            } else if depth < 8 && !in_check && static_eval < alpha - 25 && bound == Bound::Lower {
                // low-depth singular extension.
                extension = 1;
            } else {
                // no extension.
                extension = 0;
            }
        } else {
            extension = 0;
        }
        if extension >= 2 {
            t.ss[height].dextensions += 1;
        }

        t.ss[height].searching = Some(m);
        t.ss[height].searching_tactical = !is_quiet;
        t.ss[height].ch_idx = ContHistIndex {
            piece: moved,
            to: m.history_to_square(),
        };

        t.board.make_move(m, &mut t.nnue);

        let mut score;
        if moves_made == 1 {
            // first move (presumably the PV-move)
            let new_depth = depth + extension - 1;
            score =
                -alpha_beta::<NT::Next>(l_pv, t, new_depth, -beta, -alpha, !NT::PV && !cut_node);
        } else {
            // calculation of LMR stuff
            let r = if depth > 2 && moves_made > (1 + usize::from(NT::PV)) {
                let mut r = t.info.lm_table.lm_reduction(depth, moves_made);
                // reduce more on non-PV nodes
                r += i32::from(!NT::PV) * t.info.conf.lmr_non_pv_mul;
                r -= i32::from(t.ss[height].ttpv) * t.info.conf.lmr_ttpv_mul;
                // reduce more on cut nodes
                r += i32::from(cut_node) * t.info.conf.lmr_cut_node_mul;
                // extend/reduce using the stat_score of the move
                r -= stat_score * 1024 / t.info.conf.history_lmr_divisor;
                // reduce refutation moves less
                r -= i32::from(Some(m) == killer) * t.info.conf.lmr_refutation_mul;
                // reduce more if not improving
                r += i32::from(!improving) * t.info.conf.lmr_non_improving_mul;
                // reduce more if the move from the transposition table is tactical
                r += i32::from(tt_capture) * t.info.conf.lmr_tt_capture_mul;
                // reduce less if the move gives check
                r -= i32::from(t.board.in_check()) * t.info.conf.lmr_check_mul;
                t.ss[height].reduction = r;
                r / 1024
            } else {
                t.ss[height].reduction = 1024;
                1
            };
            // perform a zero-window search
            let mut new_depth = depth + extension;
            let reduced_depth = (new_depth - r).clamp(0, new_depth);
            score = -alpha_beta::<OffPV>(l_pv, t, reduced_depth, -alpha - 1, -alpha, true);
            // simple reduction for any future searches
            t.ss[height].reduction = 1024;
            // if we beat alpha, and reduced more than one ply,
            // then we do a zero-window search at full depth.
            if score > alpha && r > 1 {
                let do_deeper_search = score
                    > (best_score
                        + t.info.conf.do_deeper_base_margin
                        + t.info.conf.do_deeper_depth_margin * r);
                let do_shallower_search = score < best_score + new_depth;
                // depending on the value that the reduced search kicked out,
                // we might want to do a deeper search, or a shallower search.
                new_depth += i32::from(do_deeper_search) - i32::from(do_shallower_search);
                // check if we're actually going to do a deeper search than before
                // (no point if the re-search is the same as the normal one lol)
                if new_depth - 1 > reduced_depth {
                    score =
                        -alpha_beta::<OffPV>(l_pv, t, new_depth - 1, -alpha - 1, -alpha, !cut_node);
                }

                if is_quiet && (score <= alpha || score >= beta) {
                    let (c1, c2) = if score <= alpha {
                        (
                            -cont1_history_malus(&t.info.conf, new_depth),
                            -cont2_history_malus(&t.info.conf, new_depth),
                        )
                    } else {
                        (
                            cont1_history_bonus(&t.info.conf, new_depth),
                            cont2_history_bonus(&t.info.conf, new_depth),
                        )
                    };
                    t.update_continuation_history_single(hist_to, moved, c1, 1);
                    t.update_continuation_history_single(hist_to, moved, c2, 2);
                }
            } else if score > alpha && score < best_score + 16 {
                new_depth -= 1;
            }
            // if we failed completely, then do full-window search
            if score > alpha && score < beta {
                // this is a new best move, so it *is* PV.
                score = -alpha_beta::<NT::Next>(l_pv, t, new_depth - 1, -beta, -alpha, false);
            }
        }
        t.board.unmake_move(&mut t.nnue);

        // record subtree size for TimeManager
        if NT::ROOT && t.thread_id == 0 {
            let subtree_size = t.info.nodes.get_local() - nodes_before_search;
            t.info.root_move_nodes[from][hist_to] += subtree_size;
        }

        if extension >= 2 {
            t.ss[height].dextensions -= 1;
        }

        if t.info.stopped() {
            return 0;
        }

        if score > best_score {
            best_score = score;
            if score > alpha {
                best_move = Some(m);
                alpha = score;
                if NT::PV {
                    pv.load_from(m, l_pv);
                }
            }
            if alpha >= beta {
                #[cfg(feature = "stats")]
                t.info.log_fail_high::<false>(moves_made - 1);
                break;
            }
        }
    }

    if moves_made == 0 {
        if excluded.is_some() {
            return alpha;
        }
        if in_check {
            #[cfg(debug_assertions)]
            t.board.assert_mated();
            return mated_in(height);
        }
        return draw_score(t, t.info.nodes.get_local(), t.board.turn());
    }

    if best_score >= beta
        && best_score.abs() < MINIMUM_TB_WIN_SCORE
        && alpha.abs() < MINIMUM_TB_WIN_SCORE
    {
        best_score = (best_score * depth + beta) / (depth + 1);
    }

    best_score = best_score.clamp(syzygy_min, syzygy_max);

    let flag = if best_score >= beta {
        Bound::Lower
    } else if best_score > original_alpha {
        Bound::Exact
    } else {
        Bound::Upper
    };

    if alpha != original_alpha {
        // we raised alpha, so this is either a PV-node or a cut-node,
        // so we update history metrics.
        let best_move = best_move.expect("if alpha was raised, we should have a best move.");
        if !t.board.is_tactical(best_move) {
            t.insert_killer(best_move);

            // this heuristic is on the whole unmotivated, beyond mere empiricism.
            // perhaps it's really important to know which quiet moves are good in "bad" positions?
            // note: if in check, static_eval will be VALUE_NONE, but this probably doesn't cause
            // any issues.
            let history_depth_boost = i32::from(static_eval <= alpha);
            update_quiet_history(
                t,
                quiets_tried.as_slice(),
                best_move,
                depth + history_depth_boost,
            );
        }

        // we unconditionally update the tactical history table
        // because tactical moves ought to be good in any position,
        // so it's good to decrease tactical history scores even
        // when the best move was non-tactical.
        update_tactical_history(t, tacticals_tried.as_slice(), best_move, depth);
    }

    if !NT::ROOT && flag == Bound::Upper && (!quiets_tried.is_empty() || depth > 3) {
        // the current node has failed low. this means that the inbound edge to this node
        // will fail high, so we can give a bonus to that edge.
        let ss_prev = &t.ss[height - 1];
        if let Some(mov) = ss_prev.searching {
            if !ss_prev.searching_tactical {
                let from = mov.from();
                let to = mov.history_to_square();
                let moved = t.board.state.mailbox[to].expect("Cannot fail, move has been made.");
                debug_assert_eq!(moved.colour(), !t.board.turn());
                let threats = t.board.history().last().unwrap().threats.all;
                let bonus = main_history_bonus(&t.info.conf, depth);
                t.update_history_single(from, to, moved, threats, bonus);
            }
        }
    }

    if excluded.is_none() {
        debug_assert!(
            alpha != original_alpha || best_move.is_none(),
            "alpha was not raised, but best_move was not null!"
        );
        // if we're not in check, and we don't have a tactical best-move,
        // and the static eval needs moving in a direction, then update corrhist.
        if !(in_check
            || matches!(best_move, Some(m) if t.board.is_tactical(m))
            || flag == Bound::Lower && best_score <= static_eval
            || flag == Bound::Upper && best_score >= static_eval)
        {
            t.update_correction_history(depth, best_score - static_eval);
        }
        t.tt.store(
            key,
            height,
            best_move,
            best_score,
            raw_eval,
            flag,
            depth,
            t.ss[height].ttpv,
        );
    }

    t.ss[height].best_move = best_move;

    best_score
}

/// The margin for Reverse Futility Pruning.
fn rfp_margin(pos: &Board, info: &SearchInfo, depth: i32, improving: bool, correction: i32) -> i32 {
    info.conf.rfp_margin * depth
        - i32::from(improving && !can_win_material(pos)) * info.conf.rfp_improving_margin
        + correction.abs() / 2
}

/// Update the main and continuation history tables for a batch of moves.
fn update_quiet_history(t: &mut ThreadData, moves_to_adjust: &[Move], best_move: Move, depth: i32) {
    t.update_history(moves_to_adjust, best_move, depth);
    t.update_continuation_history(moves_to_adjust, best_move, depth, 0);
    t.update_continuation_history(moves_to_adjust, best_move, depth, 1);
    // t.update_continuation_history(moves_to_adjust, best_move, depth, 3);
}

/// Update the main and continuation history tables for a single move.
#[allow(clippy::identity_op)]
fn update_quiet_history_single<const MADE: bool>(
    t: &mut ThreadData,
    from: Square,
    to: Square,
    moved: Piece,
    threats: SquareSet,
    depth: i32,
    good: bool,
) {
    let (main, cont1, cont2) = if good {
        (
            main_history_bonus(&t.info.conf, depth),
            cont1_history_bonus(&t.info.conf, depth),
            cont2_history_bonus(&t.info.conf, depth),
        )
    } else {
        (
            -main_history_malus(&t.info.conf, depth),
            -cont1_history_malus(&t.info.conf, depth),
            -cont2_history_malus(&t.info.conf, depth),
        )
    };
    t.update_history_single(from, to, moved, threats, main);
    t.update_continuation_history_single(to, moved, cont1, 0 + usize::from(MADE));
    t.update_continuation_history_single(to, moved, cont2, 1 + usize::from(MADE));
    // t.update_continuation_history_single(to, moved, delta, 3 + usize::from(MADE));
}

/// Update the tactical history table.
fn update_tactical_history(
    t: &mut ThreadData,
    moves_to_adjust: &[Move],
    best_move: Move,
    depth: i32,
) {
    t.update_tactical_history(moves_to_adjust, best_move, depth);
}

/// The reduced beta margin for Singular Extension.
fn singularity_margin(tt_value: i32, depth: i32) -> i32 {
    (tt_value - (depth * 3 / 4)).max(-MATE_SCORE)
}

/// Test if a move is *forced* - that is, if it is a move that is
/// significantly better than the rest of the moves in a position,
/// by at least `margin`. (typically ~200cp).
pub fn is_forced(margin: i32, t: &mut ThreadData, m: Move, value: i32, depth: i32) -> bool {
    let r_beta = (value - margin).max(-MATE_SCORE);
    let r_depth = (depth - 1) / 2;
    t.ss[t.board.height()].excluded = Some(m);
    let pts_prev = t.info.print_to_stdout;
    t.info.print_to_stdout = false;
    let value = alpha_beta::<CheckForced>(
        &mut PVariation::default(),
        t,
        r_depth,
        r_beta - 1,
        r_beta,
        false,
    );
    t.info.print_to_stdout = pts_prev;
    t.ss[t.board.height()].excluded = None;
    value < r_beta
}

/// Cheaply estimate whether there's an obvious winning capture to be made
/// somewhere in the position.
pub fn can_win_material(pos: &Board) -> bool {
    let us = pos.state.bbs.colours[pos.turn()];
    let queens = pos.state.bbs.pieces[PieceType::Queen] & us;
    let rooks = pos.state.bbs.pieces[PieceType::Rook] & us;
    let bishops = pos.state.bbs.pieces[PieceType::Bishop] & us;
    let knights = pos.state.bbs.pieces[PieceType::Knight] & us;

    (pos.state.threats.leq_rook & queens) != SquareSet::EMPTY
        || (pos.state.threats.leq_minor & (queens | rooks)) != SquareSet::EMPTY
        || (pos.state.threats.leq_pawn) & (queens | rooks | bishops | knights) != SquareSet::EMPTY
}

/// See if a move looks like it would initiate a winning exchange.
/// This function simulates flowing all moves on to the target square of
/// the given move, from least to most valuable moved piece, and returns
/// true if the exchange comes out with a material advantage of at
/// least `threshold`.
pub fn static_exchange_eval(board: &Board, info: &SearchInfo, m: Move, threshold: i32) -> bool {
    let from = m.from();
    let to = m.to();
    let bbs = &board.state.bbs;

    let mut next_victim = m.promotion_type().map_or_else(
        || board.state.mailbox[from].unwrap().piece_type(),
        |promo| promo,
    );

    let mut balance = board.estimated_see(info, m) - threshold;

    // if the best case fails, don't bother doing the full search.
    if balance < 0 {
        return false;
    }

    // worst case is losing the piece
    balance -= see_value(next_victim, info);

    // if the worst case passes, we can return true immediately.
    if balance >= 0 {
        return true;
    }

    let diag_sliders = bbs.pieces[PieceType::Bishop] | bbs.pieces[PieceType::Queen];
    let orth_sliders = bbs.pieces[PieceType::Rook] | bbs.pieces[PieceType::Queen];

    // occupied starts with the position after the move `m` is made.
    let mut occupied = (bbs.occupied() ^ from.as_set()) | to.as_set();
    if m.is_ep() {
        occupied ^= board.ep_sq().unwrap().as_set();
    }

    // after the move, it's the opponent's turn.
    let mut colour = !board.turn();

    let white_pinned = board.state.pinned[Colour::White];
    let black_pinned = board.state.pinned[Colour::Black];

    let kings = bbs.pieces[PieceType::King];
    let white_king = kings & bbs.colours[Colour::White];
    let black_king = kings & bbs.colours[Colour::Black];

    let white_king_ray = RAY_FULL[to][white_king.first().unwrap()];
    let black_king_ray = RAY_FULL[to][black_king.first().unwrap()];

    let allowed = !(white_pinned | black_pinned)
        | (white_pinned & white_king_ray)
        | (black_pinned & black_king_ray);

    let mut attackers = bbs.all_attackers_to_sq(to, occupied) & allowed;

    loop {
        let my_attackers = attackers & bbs.colours[colour];
        if my_attackers == SquareSet::EMPTY {
            break;
        }

        // find cheapest attacker
        for victim in PieceType::all() {
            next_victim = victim;
            if (my_attackers & bbs.pieces[victim]) != SquareSet::EMPTY {
                break;
            }
        }

        occupied ^= (my_attackers & bbs.pieces[next_victim]).isolate_lsb();

        // diagonal moves reveal bishops and queens:
        if next_victim == PieceType::Pawn
            || next_victim == PieceType::Bishop
            || next_victim == PieceType::Queen
        {
            attackers |= movegen::bishop_attacks(to, occupied) & diag_sliders;
        }

        // orthogonal moves reveal rooks and queens:
        if next_victim == PieceType::Rook || next_victim == PieceType::Queen {
            attackers |= movegen::rook_attacks(to, occupied) & orth_sliders;
        }

        attackers &= occupied;

        colour = !colour;

        balance = -balance - 1 - see_value(next_victim, info);

        if balance >= 0 {
            // from Ethereal:
            // As a slight optimisation for move legality checking, if our last attacking
            // piece is a king, and our opponent still has attackers, then we've
            // lost as the move we followed would be illegal
            if next_victim == PieceType::King
                && (attackers & bbs.colours[colour]) != SquareSet::EMPTY
            {
                colour = !colour;
            }
            break;
        }
    }

    // the side that is to move after loop exit is the loser.
    board.turn() != colour
}

pub fn adj_shuffle(t: &ThreadData, raw_eval: i32, clock: u8) -> i32 {
    // scale down the value estimate when there's not much
    // material left - this will incentivize keeping material
    // on the board if we have winning chances, and trading
    // material off if the position is worse for us.
    let material = t.board.material(&t.info);
    let mat_mul = t.info.conf.material_scale_base + material;
    let opt_mul = t.info.conf.optimism_mat_base + material;
    let raw_eval = (raw_eval * mat_mul + t.optimism[t.board.turn()] * opt_mul / 32) / 1024;

    // scale down the value when the fifty-move counter is high.
    // this goes some way toward making viri realise when he's not
    // making progress in a position.
    raw_eval * (200 - i32::from(clock)) / 200
}

pub fn select_best<'a>(thread_headers: &'a [Box<ThreadData<'a>>]) -> &'a ThreadData<'a> {
    let total_nodes = thread_headers[0].info.nodes.get_global();

    let (mut best_thread, rest) = thread_headers.split_first().unwrap();

    for thread in rest {
        let best_depth = best_thread.completed;
        let best_score = best_thread.pvs[best_depth].score();
        let this_depth = thread.completed;
        let this_score = thread.pvs[this_depth].score();
        if (this_depth == best_depth || this_score >= MINIMUM_TB_WIN_SCORE)
            && this_score > best_score
        {
            best_thread = thread;
        }
        if this_depth > best_depth && (this_score > best_score || best_score < MINIMUM_TB_WIN_SCORE)
        {
            best_thread = thread;
        }
    }

    // if we aren't using the main thread (thread 0) then we need to do
    // an extra uci info line to show the best move/score/pv
    if best_thread.thread_id != 0 {
        readout_info(
            best_thread,
            &thread_headers[0].info,
            Bound::Exact,
            total_nodes,
            false,
        );
    }

    best_thread
}

/// Print the info about an iteration of the search.
fn readout_info(
    t: &ThreadData,
    info: &SearchInfo,
    mut bound: Bound,
    nodes: u64,
    force_print: bool,
) {
    #![allow(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation
    )]
    if !info.print_to_stdout {
        return;
    }
    let ThreadData {
        board,
        iteration,
        tt,
        ..
    } = t;
    let depth = iteration;
    let pv = t.pv();
    // don't print anything if we are in the first 50ms of the search and we are in a game,
    // this helps in ultra-fast time controls where we only have a few ms to think.
    if info.clock.is_dynamic() && info.skip_print() && !force_print {
        return;
    }
    let sstr = uci::format_score(pv.score);
    let normal_uci_output = !uci::PRETTY_PRINT.load(Ordering::SeqCst);
    let nps = (nodes as f64 / info.clock.elapsed().as_secs_f64()) as u64;
    if board.turn() == Colour::Black {
        bound = match bound {
            Bound::Upper => Bound::Lower,
            Bound::Lower => Bound::Upper,
            _ => Bound::Exact,
        };
    }
    let bound_string = match bound {
        Bound::Upper => " upperbound",
        Bound::Lower => " lowerbound",
        _ => "",
    };
    if normal_uci_output {
        println!(
            "info depth {depth} seldepth {} nodes {nodes} time {} nps {nps} hashfull {hashfull} tbhits {tbhits} score {sstr}{bound_string} wdl {wdl} {pv}",
            info.seldepth as usize,
            info.clock.elapsed().as_millis(),
            hashfull = tt.hashfull(),
            tbhits = TB_HITS.load(Ordering::SeqCst),
            wdl = uci::format_wdl(pv.score, board.ply()),
        );
    } else {
        let value = uci::pretty_format_score(pv.score, board.turn());
        let mut pv_string = board.pv_san(pv).unwrap();
        // truncate the pv string if it's too long
        if pv_string.len() > 130 {
            let final_space = pv_string
                .match_indices(' ')
                .filter(|(i, _)| *i < 130)
                .next_back()
                .map_or(0, |(i, _)| i);
            pv_string.truncate(final_space);
            pv_string.push_str("...         ");
        }
        let endchr = if bound == Bound::Exact {
            "\n"
        } else {
            "                                                                   \r"
        };
        eprint!(
            " {depth:2}/{:<2} \u{001b}[38;5;243m{t} {knodes:8}kn\u{001b}[0m {value} ({wdl}) \u{001b}[38;5;243m{knps:5}kn/s\u{001b}[0m {pv_string}{endchr}",
            info.seldepth as usize,
            t = uci::format_time(info.clock.elapsed().as_millis()),
            knps = nps / 1_000,
            knodes = nodes / 1_000,
            wdl = uci::pretty_format_wdl(pv.score, board.ply()),
        );
    }
}

pub fn draw_score(t: &ThreadData, nodes: u64, stm: Colour) -> i32 {
    // score fuzzing helps with threefolds.
    let random_component = (nodes & 0b11) as i32 - 2;
    // higher contempt means we will play on in drawn positions more often,
    // so if we are to play in a drawn position, then we should return the
    // negative of the contempt score.
    let contempt = uci::CONTEMPT.load(Ordering::Relaxed);
    let contempt_component = if stm == t.stm_at_root {
        -contempt
    } else {
        contempt
    };

    random_component + contempt_component
}

#[derive(Clone, Debug)]
pub struct LMTable {
    /// The reduction table. rtable\[depth]\[played] is the base LMR reduction for a move
    lm_reduction_table: [[i32; 64]; 64],
    /// The movecount table. ptable\[played]\[improving] is the movecount at which LMP is triggered.
    lmp_movecount_table: [[usize; 12]; 2],
}

impl LMTable {
    pub const NULL: Self = Self {
        lm_reduction_table: [[0; 64]; 64],
        lmp_movecount_table: [[0; 12]; 2],
    };

    pub fn new(config: &Config) -> Self {
        #![allow(
            clippy::cast_possible_truncation,
            clippy::cast_precision_loss,
            clippy::cast_sign_loss
        )]
        let mut out = Self::NULL;
        let (base, division) = (
            config.lmr_base / 100.0 * 1024.0,
            config.lmr_division / 100.0 / 1024.0,
        );
        cfor!(let mut depth = 1; depth < 64; depth += 1; {
            cfor!(let mut played = 1; played < 64; played += 1; {
                let ld = f64::ln(depth as f64);
                let lp = f64::ln(played as f64);
                out.lm_reduction_table[depth][played] = (base + ld * lp / division) as i32;
            });
        });
        cfor!(let mut depth = 1; depth < 12; depth += 1; {
            out.lmp_movecount_table[0][depth] = (2.5 + 2.0 * depth as f64 * depth as f64 / 4.5) as usize;
            out.lmp_movecount_table[1][depth] = (4.0 + 4.0 * depth as f64 * depth as f64 / 4.5) as usize;
        });
        out
    }

    pub fn lm_reduction(&self, depth: i32, moves_made: usize) -> i32 {
        let depth: usize = depth.clamp(0, 63).try_into().unwrap_or_default();
        let played = moves_made.min(63);
        self.lm_reduction_table[depth][played]
    }

    pub fn lmp_movecount(&self, depth: i32, improving: bool) -> usize {
        let depth: usize = depth.clamp(0, 11).try_into().unwrap_or_default();
        self.lmp_movecount_table[usize::from(improving)][depth]
    }
}

impl Default for LMTable {
    fn default() -> Self {
        Self::new(&Config::default())
    }
}
