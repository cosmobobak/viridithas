#![allow(clippy::too_many_arguments)]

pub mod parameters;
pub mod pv;

use std::{
    ops::ControlFlow,
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
        is_game_theoretic_score, mate_in, mated_in, see_value, tb_loss_in, tb_win_in, MATE_SCORE,
        MINIMUM_TB_WIN_SCORE,
    },
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
    transpositiontable::{Bound, TTHit, TTView},
    uci,
    util::{INFINITY, MAX_DEPTH, MAX_PLY, VALUE_NONE},
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

const ASPIRATION_WINDOW: i32 = 6;
const RFP_MARGIN: i32 = 67;
const RFP_IMPROVING_MARGIN: i32 = 47;
const NMP_IMPROVING_MARGIN: i32 = 74;
const NMP_DEPTH_MUL: i32 = -4;
const NMP_REDUCTION_EVAL_DIVISOR: i32 = 171;
const SEE_QUIET_MARGIN: i32 = -77;
const SEE_TACTICAL_MARGIN: i32 = -27;
const FUTILITY_COEFF_0: i32 = 107;
const FUTILITY_COEFF_1: i32 = 80;
const RAZORING_COEFF_0: i32 = -30;
const RAZORING_COEFF_1: i32 = 255;
const PROBCUT_MARGIN: i32 = 271;
const PROBCUT_IMPROVING_MARGIN: i32 = 65;
const DOUBLE_EXTENSION_MARGIN: i32 = 13;
const TRIPLE_EXTENSION_MARGIN: i32 = 125;
const LMR_BASE: f64 = 89.0;
const LMR_DIVISION: f64 = 245.0;
const QS_SEE_BOUND: i32 = -199;
const MAIN_SEE_BOUND: i32 = -102;
const DO_DEEPER_BASE_MARGIN: i32 = 50;
const DO_DEEPER_DEPTH_MARGIN: i32 = 10;
const HISTORY_PRUNING_MARGIN: i32 = -4322;
const QS_FUTILITY: i32 = 220;
const SEE_STAT_SCORE_MUL: i32 = 26;

const HISTORY_LMR_DIVISOR: i32 = 16962;
const LMR_REFUTATION_MUL: i32 = 931;
const LMR_NON_PV_MUL: i32 = 1034;
const LMR_TTPV_MUL: i32 = 1380;
const LMR_CUT_NODE_MUL: i32 = 1444;
const LMR_NON_IMPROVING_MUL: i32 = 688;
const LMR_TT_CAPTURE_MUL: i32 = 1104;
const LMR_CHECK_MUL: i32 = 1192;

const MAIN_HISTORY_BONUS_MUL: i32 = 303;
const MAIN_HISTORY_BONUS_OFFSET: i32 = 149;
const MAIN_HISTORY_BONUS_MAX: i32 = 2846;

const MAIN_HISTORY_MALUS_MUL: i32 = 199;
const MAIN_HISTORY_MALUS_OFFSET: i32 = 388;
const MAIN_HISTORY_MALUS_MAX: i32 = 1327;

const CONT1_HISTORY_BONUS_MUL: i32 = 319;
const CONT1_HISTORY_BONUS_OFFSET: i32 = 234;
const CONT1_HISTORY_BONUS_MAX: i32 = 2790;

const CONT1_HISTORY_MALUS_MUL: i32 = 220;
const CONT1_HISTORY_MALUS_OFFSET: i32 = 263;
const CONT1_HISTORY_MALUS_MAX: i32 = 1072;

const CONT2_HISTORY_BONUS_MUL: i32 = 264;
const CONT2_HISTORY_BONUS_OFFSET: i32 = 243;
const CONT2_HISTORY_BONUS_MAX: i32 = 2466;

const CONT2_HISTORY_MALUS_MUL: i32 = 275;
const CONT2_HISTORY_MALUS_OFFSET: i32 = 275;
const CONT2_HISTORY_MALUS_MAX: i32 = 1335;

const TACTICAL_HISTORY_BONUS_MUL: i32 = 228;
const TACTICAL_HISTORY_BONUS_OFFSET: i32 = 265;
const TACTICAL_HISTORY_BONUS_MAX: i32 = 2411;

const TACTICAL_HISTORY_MALUS_MUL: i32 = 140;
const TACTICAL_HISTORY_MALUS_OFFSET: i32 = 349;
const TACTICAL_HISTORY_MALUS_MAX: i32 = 1015;

const PAWN_CORRHIST_WEIGHT: i32 = 1333;
const MAJOR_CORRHIST_WEIGHT: i32 = 1518;
const MINOR_CORRHIST_WEIGHT: i32 = 1454;
const NONPAWN_CORRHIST_WEIGHT: i32 = 1611;

const EVAL_POLICY_IMPROVEMENT_SCALE: i32 = 207;
const EVAL_POLICY_OFFSET: i32 = 9;
const EVAL_POLICY_UPDATE_MAX: i32 = 5;
const EVAL_POLICY_UPDATE_MIN: i32 = 42;

const TIME_MANAGER_UPDATE_MIN_DEPTH: i32 = 4;

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
    board: &mut Board,
    info: &mut SearchInfo,
    thread_headers: &mut [ThreadData],
    tt: TTView,
) -> (i32, Option<Move>) {
    board.zero_height();
    info.set_up_for_search();
    TB_HITS.store(0, Ordering::Relaxed);

    let legal_moves = board.legal_moves();
    if legal_moves.is_empty() {
        eprintln!("info string warning search called on a position with no legal moves");
        if board.in_check() {
            println!("info depth 0 score mate 0");
        } else {
            println!("info depth 0 score cp 0");
        }
        println!("bestmove (none)");
        return (0, None);
    }
    if legal_moves.len() == 1 {
        info.time_manager.notify_one_legal_move();
    }

    // Probe the tablebases if we're in a TB position and in a game.
    if info.time_manager.is_dynamic() {
        if let Some((best_move, score)) = tablebases::probe::get_tablebase_move(board) {
            let mut pv = PVariation::default();
            pv.load_from(best_move, &PVariation::default());
            pv.score = score;
            TB_HITS.store(1, Ordering::SeqCst);
            readout_info(board, Bound::Exact, &pv, 0, info, tt, 1, true);
            if info.print_to_stdout {
                println!(
                    "bestmove {}",
                    best_move.display(CHESS960.load(Ordering::Relaxed))
                );
            }
            return (score, Some(best_move));
        }
    }

    let global_stopped = info.stopped;
    assert!(
        !global_stopped.load(Ordering::SeqCst),
        "global_stopped must be false"
    );

    // start search threads:
    let (t1, rest) = thread_headers.split_first_mut().unwrap();
    let bcopy = board.clone();
    let icopy = info.clone();
    thread::scope(|s| {
        s.spawn(|| {
            // copy data into thread
            t1.set_up_for_search(board);
            iterative_deepening::<MainThread>(board, info, t1);
            global_stopped.store(true, Ordering::SeqCst);
        });
        for t in rest.iter_mut() {
            s.spawn(|| {
                // copy data into thread
                let mut board = bcopy.clone();
                let mut info = icopy.clone();
                t.set_up_for_search(&board);
                iterative_deepening::<HelperThread>(&mut board, &mut info, t);
            });
        }
    });

    let best_thread = select_best(board, thread_headers, info, tt, info.nodes.get_global());
    let depth_achieved = best_thread.completed;
    let pv = best_thread.pv();
    let best_move = pv
        .moves()
        .first()
        .copied()
        .unwrap_or_else(|| default_move(board, &thread_headers[0], info));

    if info.print_to_stdout {
        // always give a final info log before ending search
        let nodes = info.nodes.get_global();
        readout_info(
            board,
            Bound::Exact,
            pv,
            depth_achieved,
            info,
            tt,
            nodes,
            true,
        );
    }

    if info.print_to_stdout {
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
        info.print_stats();
        #[cfg(feature = "stats")]
        println!(
            "branching factor: {}",
            (info.nodes.get_global() as f64).powf(1.0 / thread_headers[0].completed as f64)
        );
    }

    assert!(
        legal_moves.contains(&best_move),
        "search returned an illegal move."
    );
    (
        if board.turn() == Colour::White {
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
fn iterative_deepening<ThTy: SmpThreadType>(
    board: &mut Board,
    info: &mut SearchInfo,
    t: &mut ThreadData,
) {
    assert!(
        !ThTy::MAIN_THREAD || t.thread_id == 0,
        "main thread must have thread_id 0"
    );
    let mut aw = AspirationWindow::infinite();
    let mut pv = PVariation::default();
    let max_depth = info
        .time_manager
        .limit()
        .depth()
        .unwrap_or(MAX_DEPTH - 1)
        .try_into()
        .unwrap_or_default();
    let starting_depth = 1 + t.thread_id % 10;
    let mut average_value = VALUE_NONE;
    'deepening: for d in starting_depth..=max_depth {
        t.depth = d;
        if ThTy::MAIN_THREAD {
            // consider stopping early if we've neatly completed a depth:
            if (info.time_manager.is_dynamic() || info.time_manager.is_soft_nodes())
                && info.time_manager.is_past_opt_time(info.nodes.get_global())
            {
                info.stopped.store(true, Ordering::SeqCst);
                break 'deepening;
            }
            if d > info
                .time_manager
                .limit()
                .depth()
                .unwrap_or(MAX_DEPTH - 1)
                .try_into()
                .unwrap_or_default()
            {
                info.stopped.store(true, Ordering::SeqCst);
                break 'deepening;
            }
        }

        t.optimism[board.turn()] = 128 * average_value / (average_value.abs() + 212);
        t.optimism[!board.turn()] = -t.optimism[board.turn()];

        // aspiration loop:
        // (depth can be dynamically modified in the aspiration loop,
        // so we return out the value of depth to the caller)
        let ControlFlow::Continue(depth) =
            aspiration::<ThTy>(board, &mut pv, info, t, &mut aw, d, &mut average_value)
        else {
            break 'deepening;
        };

        if depth > 5 {
            aw = AspirationWindow::around_value(average_value, depth);
        } else {
            aw = AspirationWindow::infinite();
        }

        if ThTy::MAIN_THREAD && depth > TIME_MANAGER_UPDATE_MIN_DEPTH {
            let bm_frac = if d > 8 {
                let best_move = pv.moves[0];
                let best_move_subtree_size = info.root_move_nodes[best_move.from()][best_move.to()];
                let tree_size = info.nodes.get_local();
                #[allow(clippy::cast_precision_loss)]
                Some(best_move_subtree_size as f64 / tree_size as f64)
            } else {
                None
            };
            info.time_manager.report_completed_depth(
                depth,
                pv.score,
                pv.moves[0],
                bm_frac,
                &info.conf,
            );
        }

        if info.check_up() {
            break 'deepening;
        }
    }
}

fn aspiration<ThTy: SmpThreadType>(
    board: &mut Board,
    pv: &mut PVariation,
    info: &mut SearchInfo<'_>,
    t: &mut ThreadData,
    aw: &mut AspirationWindow,
    d: usize,
    average_value: &mut i32,
) -> ControlFlow<(), i32> {
    let mut depth = i32::try_from(d).unwrap();
    let min_depth = (depth / 2).max(1);
    loop {
        pv.score = alpha_beta::<Root>(board, pv, info, t, depth, aw.alpha, aw.beta, false);
        if info.check_up() {
            return ControlFlow::Break(()); // we've been told to stop searching.
        }

        if aw.alpha != -INFINITY && pv.score <= aw.alpha {
            if ThTy::MAIN_THREAD && info.print_to_stdout {
                let nodes = info.nodes.get_global();
                let mut apv = t.pv().clone();
                apv.score = pv.score;
                readout_info(board, Bound::Upper, &apv, d, info, t.tt, nodes, false);
            }
            aw.widen_down(pv.score, depth);
            if ThTy::MAIN_THREAD {
                info.time_manager
                    .report_aspiration_fail(depth, Bound::Upper, &info.conf);
            }
            // search failed low, so we might have to
            // revert a fail-high pv update
            t.revert_best_line();
            continue;
        }
        // search is either exact or fail-high, so we can update the best line.
        t.update_best_line(pv);
        if aw.beta != INFINITY && pv.score >= aw.beta {
            if ThTy::MAIN_THREAD && info.print_to_stdout {
                let nodes = info.nodes.get_global();
                readout_info(board, Bound::Lower, t.pv(), d, info, t.tt, nodes, false);
            }
            aw.widen_up(pv.score, depth);
            if ThTy::MAIN_THREAD {
                info.time_manager
                    .report_aspiration_fail(depth, Bound::Lower, &info.conf);
            }
            // decrement depth:
            if !is_game_theoretic_score(pv.score) {
                depth = (depth - 1).max(min_depth);
            }

            if info.time_manager.solved_breaker::<ThTy>(0, d) == ControlFlow::Break(()) {
                info.stopped.store(true, Ordering::SeqCst);
                return ControlFlow::Break(()); // we've been told to stop searching.
            }

            continue;
        }

        // if we've made it here, it means we got an exact score.
        let score = pv.score;
        let bestmove = t.pvs[t.completed]
            .moves()
            .first()
            .copied()
            .unwrap_or_else(|| default_move(board, t, info));
        *average_value = if *average_value == VALUE_NONE {
            score
        } else {
            (2 * score + *average_value) / 3
        };

        if ThTy::MAIN_THREAD && info.print_to_stdout {
            let total_nodes = info.nodes.get_global();
            readout_info(
                board,
                Bound::Exact,
                t.pv(),
                d,
                info,
                t.tt,
                total_nodes,
                false,
            );
        }

        if info.time_manager.solved_breaker::<ThTy>(pv.score, d) == ControlFlow::Break(()) {
            info.stopped.store(true, Ordering::SeqCst);
            return ControlFlow::Break(());
        }

        if info.time_manager.mate_found_breaker::<ThTy>(pv, depth) == ControlFlow::Break(()) {
            info.stopped.store(true, Ordering::SeqCst);
            return ControlFlow::Break(());
        }

        if ThTy::MAIN_THREAD {
            if let Some(margin) = info.time_manager.check_for_forced_move(depth) {
                let saved_seldepth = info.seldepth;
                let forced = is_forced(board, margin, info, t, bestmove, score, (depth - 1) / 2);
                info.seldepth = saved_seldepth;

                if forced {
                    info.time_manager.report_forced_move(depth, &info.conf);
                }
            }
        }

        if info.stopped() {
            return ControlFlow::Break(());
        }

        break ControlFlow::Continue(depth); // we got an exact score, so we can stop the aspiration loop.
    }
}

/// Give a legal default move in the case where we don't have enough time to search.
fn default_move(board: &mut Board, t: &ThreadData, info: &SearchInfo) -> Move {
    let tt_move =
        t.tt.probe_for_provisional_info(board.state.keys.zobrist)
            .and_then(|e| e.0);
    let mut mp = MovePicker::new(tt_move, t.killer_move_table[board.height()], 0);
    let mut m = None;
    while let Some(mov) = mp.next(board, t, info) {
        if !board.is_legal(mov) {
            continue;
        }
        board.make_move_simple(mov);
        // if we get here, it means the move is legal.
        m = Some(mov);
        board.unmake_move_base();
        break;
    }
    m.expect("Board::default_move called on a position with no legal moves")
}

/// Perform a tactical resolution search, searching only captures and promotions.
#[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
pub fn quiescence<NT: NodeType>(
    board: &mut Board,
    pv: &mut PVariation,
    info: &mut SearchInfo,
    t: &mut ThreadData,
    mut alpha: i32,
    beta: i32,
) -> i32 {
    #[cfg(debug_assertions)]
    board.check_validity().unwrap();

    if info.nodes.just_ticked_over() && info.check_up() {
        return 0;
    }

    let key = board.state.keys.zobrist ^ HM_CLOCK_KEYS[board.state.fifty_move_counter as usize];

    let mut local_pv = PVariation::default();
    let l_pv = &mut local_pv;

    pv.moves.clear();

    let height = board.height();
    info.seldepth = info.seldepth.max(i32::try_from(height).unwrap());

    // check draw
    if board.is_draw() {
        return draw_score(t, info.nodes.get_local(), board.turn());
    }

    let in_check = board.in_check();

    // are we too deep?
    if height > MAX_PLY - 1 {
        return if in_check {
            0
        } else {
            board.evaluate(t, info.nodes.get_local())
        };
    }

    // upcoming repetition detection
    if alpha < 0 && board.has_game_cycle(height) {
        alpha = 0;
        if alpha >= beta {
            return alpha;
        }
    }

    let clock = board.fifty_move_counter();

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
            raw_eval = board.evaluate(t, info.nodes.get_local());
        } else {
            // if the TT eval is not VALUE_NONE, use it.
            raw_eval = v;
        }
        let adj_eval =
            adj_shuffle(board, t, info, raw_eval, clock) + t.correction(&info.conf, board);

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
        raw_eval = board.evaluate(t, info.nodes.get_local());

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

        stand_pat = adj_shuffle(board, t, info, raw_eval, clock) + t.correction(&info.conf, board);
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
    let mut move_picker = MovePicker::new(tt_hit.and_then(|e| e.mov), None, info.conf.qs_see_bound);
    move_picker.skip_quiets = !in_check;

    let futility = stand_pat + info.conf.qs_futility;

    while let Some(m) = move_picker.next(board, t, info) {
        let is_tactical = board.is_tactical(m);
        if best_score > -MINIMUM_TB_WIN_SCORE
            && is_tactical
            && !in_check
            && futility <= alpha
            && !static_exchange_eval(board, info, m, 1)
        {
            if best_score < futility {
                best_score = futility;
            }
            continue;
        }
        t.tt.prefetch(board.key_after(m));
        t.ss[height].searching = Some(m);
        t.ss[height].searching_tactical = is_tactical;
        let moved = board.state.mailbox[m.from()].unwrap();
        t.ss[height].conthist_index = ContHistIndex {
            piece: moved,
            square: m.history_to_square(),
        };
        if !board.is_legal(m) {
            continue;
        }
        board.make_move(m, t);
        // move found, we can start skipping quiets again:
        move_picker.skip_quiets = true;
        info.nodes.increment();
        moves_made += 1;

        let score = -quiescence::<NT::Next>(board, l_pv, info, t, -beta, -alpha);
        board.unmake_move(t);

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
                info.log_fail_high::<true>(moves_made - 1, 0);
                break; // fail-high
            }
        }
    }

    if moves_made == 0 && in_check {
        #[cfg(debug_assertions)]
        board.assert_mated();
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
    board: &mut Board,
    pv: &mut PVariation,
    info: &mut SearchInfo,
    t: &mut ThreadData,
    mut depth: i32,
    mut alpha: i32,
    mut beta: i32,
    cut_node: bool,
) -> i32 {
    #[cfg(debug_assertions)]
    board.check_validity().unwrap();

    let mut local_pv = PVariation::default();
    let l_pv = &mut local_pv;

    let key = board.state.keys.zobrist ^ HM_CLOCK_KEYS[board.state.fifty_move_counter as usize];

    if depth <= 0 {
        return quiescence::<NT::Next>(board, pv, info, t, alpha, beta);
    }

    pv.moves.clear();

    if info.nodes.just_ticked_over() && info.check_up() {
        return 0;
    }

    let height = board.height();

    debug_assert_eq!(height == 0, NT::ROOT);
    debug_assert!(!(NT::PV && cut_node));
    debug_assert_eq!(NT::PV, alpha + 1 != beta, "PV must be true iff the alpha-beta window is larger than 1, but PV was {PV} and alpha-beta window was {alpha}-{beta}", PV = NT::PV);

    info.seldepth = if NT::ROOT {
        0
    } else {
        info.seldepth.max(i32::try_from(height).unwrap())
    };

    let in_check = board.in_check();

    if !NT::ROOT {
        // check draw
        if board.is_draw() {
            return draw_score(t, info.nodes.get_local(), board.turn());
        }

        // are we too deep?
        let max_height = MAX_PLY.min(uci::GO_MATE_MAX_DEPTH.load(Ordering::SeqCst));
        if height >= max_height {
            return if in_check {
                0
            } else {
                board.evaluate(t, info.nodes.get_local())
            };
        }

        // mate-distance pruning.
        alpha = alpha.max(mated_in(height));
        beta = beta.min(mate_in(height + 1));
        if alpha >= beta {
            return alpha;
        }

        // upcoming repetition detection
        if alpha < 0 && board.has_game_cycle(height) {
            alpha = 0;
            if alpha >= beta {
                return alpha;
            }
        }
    }

    let clock = board.fifty_move_counter();

    let excluded = t.ss[height].excluded;
    let tt_hit = if excluded.is_none() {
        if let Some(hit) = t.tt.probe(key, height) {
            if !NT::PV
                && hit.depth >= depth
                && clock < 80
                && (hit.bound == Bound::Exact
                    || (hit.bound == Bound::Lower && hit.value >= beta)
                    || (hit.bound == Bound::Upper && hit.value <= alpha))
            {
                if let Some(mov) = hit.mov {
                    // add to the history of a quiet move that fails high here.
                    if hit.value >= beta && !board.is_tactical(mov) && board.is_pseudo_legal(mov) {
                        let from = mov.from();
                        let to = mov.history_to_square();
                        let moved = board.state.mailbox[from].unwrap();
                        let threats = board.state.threats.all;
                        update_quiet_history_single::<false>(
                            board, t, info, from, to, moved, threats, depth, true,
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
    let n_men = board.state.bbs.occupied().count();
    if !NT::ROOT
            && excluded.is_none() // do not probe the tablebases if we're in a singular-verification search.
            && uci::SYZYGY_ENABLED.load(Ordering::SeqCst)
            && (depth >= uci::SYZYGY_PROBE_DEPTH.load(Ordering::SeqCst)
                || n_men < cardinality)
            && n_men <= cardinality
    {
        if let Some(wdl) = tablebases::probe::get_wdl(board) {
            TB_HITS.fetch_add(1, Ordering::Relaxed);

            let tb_value = match wdl {
                WDL::Win => tb_win_in(height),
                WDL::Loss => tb_loss_in(height),
                WDL::Draw => draw_score(t, info.nodes.get_buffer(), board.turn()),
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
    let correction;

    if in_check {
        // when we're in check, it could be checkmate, so it's unsound to use evaluate().
        raw_eval = VALUE_NONE;
        static_eval = VALUE_NONE;
        correction = 0;
    } else if excluded.is_some() {
        // if we're in a singular-verification search, we already have the static eval.
        // we can set raw_eval to whatever we like, because we're not going to be saving it.
        raw_eval = VALUE_NONE;
        static_eval = t.ss[height].eval;
        correction = 0;
        t.nnue.hint_common_access(board, t.nnue_params);
    } else if let Some(TTHit { eval: tt_eval, .. }) = &tt_hit {
        let v = *tt_eval; // if we have a TT hit, check the cached TT eval.
        if v == VALUE_NONE {
            // regenerate the static eval if it's VALUE_NONE.
            raw_eval = board.evaluate(t, info.nodes.get_local());
        } else {
            // if the TT eval is not VALUE_NONE, use it.
            raw_eval = v;
            if NT::PV {
                t.nnue.hint_common_access(board, t.nnue_params);
            }
        }
        correction = t.correction(&info.conf, board);
        static_eval = adj_shuffle(board, t, info, raw_eval, clock) + correction;
    } else {
        // otherwise, use the static evaluation.
        raw_eval = board.evaluate(t, info.nodes.get_local());

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

        correction = t.correction(&info.conf, board);
        static_eval = adj_shuffle(board, t, info, raw_eval, clock) + correction;
    }

    t.ss[height].eval = static_eval;

    // value-difference based policy update.
    if !NT::ROOT {
        let ss_prev = &t.ss[height - 1];
        if let Some(mov) = ss_prev.searching {
            if ss_prev.eval != VALUE_NONE
                && static_eval != VALUE_NONE
                && !ss_prev.searching_tactical
            {
                let from = mov.from();
                let to = mov.history_to_square();
                let moved = board.state.mailbox[to].expect("Cannot fail, move has been made.");
                debug_assert_eq!(moved.colour(), !board.turn());
                let threats = board.history().last().unwrap().threats.all;
                let improvement = -(ss_prev.eval + static_eval) + info.conf.eval_policy_offset;
                let delta = i32::clamp(
                    improvement * info.conf.eval_policy_improvement_scale / 32,
                    -info.conf.eval_policy_update_min,
                    info.conf.eval_policy_update_max,
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
    } else if height >= 2 && t.ss[height - 2].eval != VALUE_NONE {
        static_eval > t.ss[height - 2].eval
    } else if height >= 4 && t.ss[height - 4].eval != VALUE_NONE {
        static_eval > t.ss[height - 4].eval
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
    let tt_capture = matches!(tt_move, Some(mv) if board.is_capture(mv));

    // whole-node techniques:
    if !NT::ROOT && !NT::PV && !in_check && excluded.is_none() {
        if t.ss[height - 1].reduction >= 4096 && static_eval + t.ss[height - 1].eval < 0 {
            depth += 1;
        }

        if depth >= 2
            && t.ss[height - 1].reduction >= 2048
            && t.ss[height - 1].eval != VALUE_NONE
            && static_eval + t.ss[height - 1].eval > 96
        {
            depth -= 1;
        }

        // razoring.
        // if the static eval is too low, check if qsearch can beat alpha.
        // if it can't, we can prune the node.
        if alpha < 2000
            && static_eval < alpha - info.conf.razoring_coeff_0 - info.conf.razoring_coeff_1 * depth
        {
            let v = quiescence::<OffPV>(board, pv, info, t, alpha, beta);
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
            && static_eval - rfp_margin(info, depth, improving, correction) >= beta
            && (tt_move.is_none() || tt_capture)
            && beta > -MINIMUM_TB_WIN_SCORE
        {
            return beta + (static_eval - beta) / 3;
        }

        // null-move pruning.
        // if we can give the opponent a free move while retaining
        // a score above beta, we can prune the node.
        if t.ss[height - 1].searching.is_some()
            && depth > 2
            && static_eval
                + i32::from(improving) * info.conf.nmp_improving_margin
                + depth * info.conf.nmp_depth_mul
                >= beta
            && !t.nmp_banned_for(board.turn())
            && board.zugzwang_unlikely()
            && !matches!(tt_hit, Some(TTHit { value: v, bound: Bound::Upper, .. }) if v < beta)
        {
            t.tt.prefetch(board.key_after_null_move());
            let r = 4
                + depth / 3
                + std::cmp::min(
                    (static_eval - beta) / info.conf.nmp_reduction_eval_divisor,
                    4,
                )
                + i32::from(tt_capture);
            let nm_depth = depth - r;
            t.ss[height].searching = None;
            t.ss[height].searching_tactical = false;
            t.ss[height].conthist_index = ContHistIndex {
                piece: Piece::new(board.turn(), PieceType::Pawn),
                square: Square::A1,
            };
            board.make_nullmove();
            let mut null_score =
                -alpha_beta::<OffPV>(board, l_pv, info, t, nm_depth, -beta, -beta + 1, !cut_node);
            board.unmake_nullmove();
            if info.stopped() {
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
                t.ban_nmp_for(board.turn());
                let veri_score =
                    alpha_beta::<OffPV>(board, l_pv, info, t, nm_depth, beta - 1, beta, false);
                t.unban_nmp_for(board.turn());
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
        info.conf.see_tactical_margin * depth * depth,
        info.conf.see_quiet_margin * depth,
    ];

    // probcut:
    let pc_beta = std::cmp::min(
        beta + info.conf.probcut_margin - i32::from(improving) * info.conf.probcut_improving_margin,
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
        let tt_move_if_capture = tt_move.filter(|m| board.is_tactical(*m));
        let mut move_picker = MovePicker::new(tt_move_if_capture, None, 0);
        move_picker.skip_quiets = true;
        while let Some(m) = move_picker.next(board, t, info) {
            t.tt.prefetch(board.key_after(m));
            t.ss[height].searching = Some(m);
            t.ss[height].searching_tactical = true;
            let moved = board.state.mailbox[m.from()].unwrap();
            t.ss[height].conthist_index = ContHistIndex {
                piece: moved,
                square: m.history_to_square(),
            };
            if !board.is_legal(m) {
                continue;
            }
            board.make_move(m, t);

            let mut value = -quiescence::<OffPV>(board, l_pv, info, t, -pc_beta, -pc_beta + 1);

            if value >= pc_beta {
                let pc_depth = depth - 3;
                value = -alpha_beta::<OffPV>(
                    board,
                    l_pv,
                    info,
                    t,
                    pc_depth,
                    -pc_beta,
                    -pc_beta + 1,
                    !cut_node,
                );
            }

            board.unmake_move(t);

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

        t.nnue.hint_common_access(board, t.nnue_params);
    }

    let original_alpha = alpha;
    let mut best_move = None;
    let mut best_score = -INFINITY;
    let mut moves_made = 0;

    // number of quiet moves to try before we start pruning
    let lmp_threshold = info.lm_table.lmp_movecount(depth, improving);

    let killer = t.killer_move_table[height].filter(|m| !board.is_tactical(*m));
    let mut move_picker = MovePicker::new(tt_move, killer, info.conf.main_see_bound);

    let mut quiets_tried = ArrayVec::<_, MAX_POSITION_MOVES>::new();
    let mut tacticals_tried = ArrayVec::<_, MAX_POSITION_MOVES>::new();

    while let Some(m) = move_picker.next(board, t, info) {
        if excluded == Some(m) {
            continue;
        }

        let lmr_reduction = info.lm_table.lm_reduction(depth, moves_made);
        let lmr_depth = std::cmp::max(depth - lmr_reduction / 1024, 0);
        let is_quiet = !board.is_tactical(m);

        let mut stat_score = 0;

        if is_quiet {
            stat_score += t.get_history_score(board, m);
            stat_score += t.get_continuation_history_score(board, m, 0);
            stat_score += t.get_continuation_history_score(board, m, 1);
            // stat_score += t.get_continuation_history_score(board, m, 3);
        } else {
            stat_score += t.get_tactical_history_score(board, m);
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
                && stat_score < info.conf.history_pruning_margin * (depth - 1)
            {
                move_picker.skip_quiets = true;
                continue;
            }

            // futility pruning
            // if the static eval is too low, we start skipping moves.
            let fp_margin = lmr_depth * info.conf.futility_coeff_1 + info.conf.futility_coeff_0;
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
            && board.state.threats.all.contains_square(m.to())
            && t.ss[height - 1].searching.is_some()
            && !static_exchange_eval(
                board,
                info,
                m,
                see_table[usize::from(is_quiet)] - stat_score * info.conf.see_stat_score_mul / 1024,
            )
        {
            continue;
        }

        t.tt.prefetch(board.key_after(m));
        t.ss[height].searching = Some(m);
        t.ss[height].searching_tactical = !is_quiet;
        let moved = board.state.mailbox[m.from()].unwrap();
        t.ss[height].conthist_index = ContHistIndex {
            piece: moved,
            square: m.history_to_square(),
        };
        if !board.is_legal(m) {
            continue;
        }

        if is_quiet {
            quiets_tried.push(m);
        } else {
            tacticals_tried.push(m);
        }

        let nodes_before_search = info.nodes.get_local();
        info.nodes.increment();
        moves_made += 1;

        let maybe_singular = depth >= 8
            && excluded.is_none()
            && matches!(tt_hit, Some(TTHit { mov, depth: tt_depth, bound: Bound::Lower | Bound::Exact, .. }) if mov == Some(m) && tt_depth >= depth - 3);

        let extension;
        if NT::ROOT {
            extension = 0;
        } else if maybe_singular {
            let Some(TTHit {
                value: tt_value, ..
            }) = tt_hit
            else {
                unreachable!()
            };
            let r_beta = singularity_margin(tt_value, depth);
            let r_depth = (depth - 1) / 2;
            t.ss[board.height()].excluded = Some(m);
            let value = alpha_beta::<OffPV>(
                board,
                &mut PVariation::default(),
                info,
                t,
                r_depth,
                r_beta - 1,
                r_beta,
                cut_node,
            );
            t.ss[board.height()].excluded = None;
            if value >= r_beta && r_beta >= beta {
                // multi-cut: if a move other than the best one beats beta,
                // then we can cut with relatively high confidence.
                return singularity_margin(tt_value, depth);
            }
            // re-make the singular move.
            t.ss[height].searching = Some(m);
            t.ss[height].searching_tactical = !is_quiet;
            let moved = board.state.mailbox[m.from()].unwrap();
            t.ss[height].conthist_index = ContHistIndex {
                piece: moved,
                square: m.history_to_square(),
            };

            if value < r_beta {
                if !NT::PV
                    && t.ss[board.height()].dextensions <= 12
                    && value < r_beta - info.conf.dext_margin
                {
                    // double-extend if we failed low by a lot
                    extension = 2 + i32::from(is_quiet && value < r_beta - info.conf.text_margin);
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

        board.make_move(m, t);

        let mut score;
        if moves_made == 1 {
            // first move (presumably the PV-move)
            let new_depth = depth + extension - 1;
            score = -alpha_beta::<NT::Next>(board, l_pv, info, t, new_depth, -beta, -alpha, false);
        } else {
            // calculation of LMR stuff
            let r = if depth > 2 && moves_made > (1 + usize::from(NT::PV)) {
                let mut r = info.lm_table.lm_reduction(depth, moves_made);
                // reduce more on non-PV nodes
                r += i32::from(!NT::PV) * info.conf.lmr_non_pv_mul;
                r -= i32::from(t.ss[height].ttpv) * info.conf.lmr_ttpv_mul;
                // reduce more on cut nodes
                r += i32::from(cut_node) * info.conf.lmr_cut_node_mul;
                // extend/reduce using the stat_score of the move
                r -= stat_score * 1024 / info.conf.history_lmr_divisor;
                // reduce refutation moves less
                r -= i32::from(Some(m) == killer) * info.conf.lmr_refutation_mul;
                // reduce more if not improving
                r += i32::from(!improving) * info.conf.lmr_non_improving_mul;
                // reduce more if the move from the transposition table is tactical
                r += i32::from(tt_capture) * info.conf.lmr_tt_capture_mul;
                // reduce less if the move gives check
                r -= i32::from(board.in_check()) * info.conf.lmr_check_mul;
                t.ss[height].reduction = r;
                r / 1024
            } else {
                t.ss[height].reduction = 1024;
                1
            };
            // perform a zero-window search
            let mut new_depth = depth + extension;
            let reduced_depth = (new_depth - r).clamp(0, new_depth);
            score = -alpha_beta::<OffPV>(
                board,
                l_pv,
                info,
                t,
                reduced_depth,
                -alpha - 1,
                -alpha,
                true,
            );
            // simple reduction for any future searches
            t.ss[height].reduction = 1024;
            // if we beat alpha, and reduced more than one ply,
            // then we do a zero-window search at full depth.
            if score > alpha && r > 1 {
                let do_deeper_search = score
                    > (best_score
                        + info.conf.do_deeper_base_margin
                        + info.conf.do_deeper_depth_margin * r);
                let do_shallower_search = score < best_score + new_depth;
                // depending on the value that the reduced search kicked out,
                // we might want to do a deeper search, or a shallower search.
                new_depth += i32::from(do_deeper_search) - i32::from(do_shallower_search);
                // check if we're actually going to do a deeper search than before
                // (no point if the re-search is the same as the normal one lol)
                if new_depth - 1 > reduced_depth {
                    score = -alpha_beta::<OffPV>(
                        board,
                        l_pv,
                        info,
                        t,
                        new_depth - 1,
                        -alpha - 1,
                        -alpha,
                        !cut_node,
                    );
                }
            } else if score > alpha && score < best_score + 16 {
                new_depth -= 1;
            }
            // if we failed completely, then do full-window search
            if score > alpha && score < beta {
                // this is a new best move, so it *is* PV.
                score = -alpha_beta::<NT::Next>(
                    board,
                    l_pv,
                    info,
                    t,
                    new_depth - 1,
                    -beta,
                    -alpha,
                    false,
                );
            }
        }
        board.unmake_move(t);

        // record subtree size for TimeManager
        if NT::ROOT && t.thread_id == 0 {
            let subtree_size = info.nodes.get_local() - nodes_before_search;
            info.root_move_nodes[m.from()][m.to()] += subtree_size;
        }

        if extension >= 2 {
            t.ss[height].dextensions -= 1;
        }

        if info.stopped() {
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
                info.log_fail_high::<false>(moves_made - 1, movepick_score);
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
            board.assert_mated();
            return mated_in(height);
        }
        return draw_score(t, info.nodes.get_local(), board.turn());
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
        if !board.is_tactical(best_move) {
            t.insert_killer(board, best_move);

            // this heuristic is on the whole unmotivated, beyond mere empiricism.
            // perhaps it's really important to know which quiet moves are good in "bad" positions?
            // note: if in check, static_eval will be VALUE_NONE, but this probably doesn't cause
            // any issues.
            let history_depth_boost = i32::from(static_eval <= alpha);
            update_quiet_history(
                board,
                &info.conf,
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
        update_tactical_history(
            board,
            &info.conf,
            t,
            tacticals_tried.as_slice(),
            best_move,
            depth,
        );
    }

    if !NT::ROOT && flag == Bound::Upper {
        // the current node has failed low. this means that the inbound edge to this node
        // will fail high, so we can give a bonus to that edge.
        let ss_prev = &t.ss[height - 1];
        if let Some(mov) = ss_prev.searching {
            if !ss_prev.searching_tactical {
                let from = mov.from();
                let to = mov.history_to_square();
                let moved = board.state.mailbox[to].expect("Cannot fail, move has been made.");
                debug_assert_eq!(moved.colour(), !board.turn());
                let threats = board.history().last().unwrap().threats.all;
                let bonus = main_history_bonus(&info.conf, depth);
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
            || matches!(best_move, Some(m) if board.is_tactical(m))
            || flag == Bound::Lower && best_score <= static_eval
            || flag == Bound::Upper && best_score >= static_eval)
        {
            t.update_correction_history(board, depth, best_score - static_eval);
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
fn rfp_margin(info: &SearchInfo, depth: i32, improving: bool, correction: i32) -> i32 {
    info.conf.rfp_margin * depth - i32::from(improving) * info.conf.rfp_improving_margin
        + correction.abs() / 2
}

/// Update the main and continuation history tables for a batch of moves.
fn update_quiet_history(
    board: &Board,
    conf: &Config,
    t: &mut ThreadData,
    moves_to_adjust: &[Move],
    best_move: Move,
    depth: i32,
) {
    t.update_history(conf, board, moves_to_adjust, best_move, depth);
    t.update_continuation_history(conf, board, moves_to_adjust, best_move, depth, 0);
    t.update_continuation_history(conf, board, moves_to_adjust, best_move, depth, 1);
    // t.update_continuation_history(board, moves_to_adjust, best_move, depth, 3);
}

/// Update the main and continuation history tables for a single move.
#[allow(clippy::identity_op)]
fn update_quiet_history_single<const MADE: bool>(
    board: &Board,
    t: &mut ThreadData,
    info: &SearchInfo,
    from: Square,
    to: Square,
    moved: Piece,
    threats: SquareSet,
    depth: i32,
    good: bool,
) {
    let (main, cont1, cont2) = if good {
        (
            main_history_bonus(&info.conf, depth),
            cont1_history_bonus(&info.conf, depth),
            cont2_history_bonus(&info.conf, depth),
        )
    } else {
        (
            -main_history_malus(&info.conf, depth),
            -cont1_history_malus(&info.conf, depth),
            -cont2_history_malus(&info.conf, depth),
        )
    };
    t.update_history_single(from, to, moved, threats, main);
    t.update_continuation_history_single(board, to, moved, cont1, 0 + usize::from(MADE));
    t.update_continuation_history_single(board, to, moved, cont2, 1 + usize::from(MADE));
    // t.update_continuation_history_single(board, to, moved, delta, 3 + usize::from(MADE));
}

/// Update the tactical history table.
fn update_tactical_history(
    board: &Board,
    conf: &Config,
    t: &mut ThreadData,
    moves_to_adjust: &[Move],
    best_move: Move,
    depth: i32,
) {
    t.update_tactical_history(conf, board, moves_to_adjust, best_move, depth);
}

/// The reduced beta margin for Singular Extension.
fn singularity_margin(tt_value: i32, depth: i32) -> i32 {
    (tt_value - (depth * 3 / 4)).max(-MATE_SCORE)
}

/// Test if a move is *forced* - that is, if it is a move that is
/// significantly better than the rest of the moves in a position,
/// by at least `margin`. (typically ~200cp).
pub fn is_forced(
    board: &mut Board,
    margin: i32,
    info: &mut SearchInfo,
    t: &mut ThreadData,
    m: Move,
    value: i32,
    depth: i32,
) -> bool {
    let r_beta = (value - margin).max(-MATE_SCORE);
    let r_depth = (depth - 1) / 2;
    t.ss[board.height()].excluded = Some(m);
    let pts_prev = info.print_to_stdout;
    info.print_to_stdout = false;
    let value = alpha_beta::<CheckForced>(
        board,
        &mut PVariation::default(),
        info,
        t,
        r_depth,
        r_beta - 1,
        r_beta,
        false,
    );
    info.print_to_stdout = pts_prev;
    t.ss[board.height()].excluded = None;
    value < r_beta
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

    let white_king_ray = RAY_FULL[to][white_king.first()];
    let black_king_ray = RAY_FULL[to][black_king.first()];

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

pub fn adj_shuffle(
    board: &Board,
    t: &ThreadData,
    info: &SearchInfo,
    raw_eval: i32,
    clock: u8,
) -> i32 {
    // scale down the value estimate when there's not much
    // material left - this will incentivize keeping material
    // on the board if we have winning chances, and trading
    // material off if the position is worse for us.
    let material = board.material(info);
    let base = info.conf.material_scale_base;
    let raw_eval =
        (raw_eval * (base + material) + t.optimism[board.turn()] * (2000 + material) / 32) / 1024;

    // scale down the value when the fifty-move counter is high.
    // this goes some way toward making viri realise when he's not
    // making progress in a position.
    raw_eval * (200 - i32::from(clock)) / 200
}

pub fn select_best<'a>(
    board: &mut Board,
    thread_headers: &'a [ThreadData],
    info: &SearchInfo,
    tt: TTView,
    total_nodes: u64,
) -> &'a ThreadData<'a> {
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
    if best_thread.thread_id != 0 && info.print_to_stdout {
        let pv = &best_thread.pvs[best_thread.completed];
        let depth = best_thread.completed;
        readout_info(board, Bound::Exact, pv, depth, info, tt, total_nodes, false);
    }

    best_thread
}

/// Print the info about an iteration of the search.
fn readout_info(
    board: &mut Board,
    mut bound: Bound,
    pv: &PVariation,
    depth: usize,
    info: &SearchInfo,
    tt: TTView,
    nodes: u64,
    force_print: bool,
) {
    #![allow(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation
    )]
    // don't print anything if we are in the first 50ms of the search and we are in a game,
    // this helps in ultra-fast time controls where we only have a few ms to think.
    if info.time_manager.is_dynamic() && info.skip_print() && !force_print {
        return;
    }
    let sstr = uci::format_score(pv.score);
    let normal_uci_output = !uci::PRETTY_PRINT.load(Ordering::SeqCst);
    let nps = (nodes as f64 / info.time_manager.elapsed().as_secs_f64()) as u64;
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
            "info score {sstr}{bound_string} wdl {wdl} depth {depth} seldepth {} nodes {nodes} time {} nps {nps} hashfull {hashfull} tbhits {tbhits} {pv}",
            info.seldepth as usize,
            info.time_manager.elapsed().as_millis(),
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
            t = uci::format_time(info.time_manager.elapsed().as_millis()),
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

pub struct AspirationWindow {
    pub midpoint: i32,
    pub alpha: i32,
    pub beta: i32,
    pub alpha_fails: i32,
    pub beta_fails: i32,
}

pub fn asp_window(depth: i32) -> i32 {
    (ASPIRATION_WINDOW + (50 / depth - 3)).max(10)
}

impl AspirationWindow {
    pub const fn infinite() -> Self {
        Self {
            alpha: -INFINITY,
            beta: INFINITY,
            midpoint: 0,
            alpha_fails: 0,
            beta_fails: 0,
        }
    }

    pub fn around_value(value: i32, depth: i32) -> Self {
        if is_game_theoretic_score(value) {
            // for mates / tbwins we expect a lot of fluctuation, so aspiration
            // windows are not useful.
            Self {
                midpoint: value,
                alpha: -INFINITY,
                beta: INFINITY,
                alpha_fails: 0,
                beta_fails: 0,
            }
        } else {
            Self {
                midpoint: value,
                alpha: value - asp_window(depth),
                beta: value + asp_window(depth),
                alpha_fails: 0,
                beta_fails: 0,
            }
        }
    }

    pub fn widen_down(&mut self, value: i32, depth: i32) {
        self.midpoint = value;
        let margin = asp_window(depth) << (self.alpha_fails + 1);
        if margin > 1369 {
            self.alpha = -INFINITY;
            return;
        }
        self.beta = (self.alpha + self.beta) / 2;
        self.alpha = self.midpoint - margin;
        self.alpha_fails += 1;
    }

    pub fn widen_up(&mut self, value: i32, depth: i32) {
        self.midpoint = value;
        let margin = asp_window(depth) << (self.beta_fails + 1);
        if margin > 1369 {
            self.beta = INFINITY;
            return;
        }
        self.beta = self.midpoint + margin;
        self.beta_fails += 1;
    }
}
