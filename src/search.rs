#![allow(clippy::too_many_arguments)]

pub mod parameters;
pub mod pv;

use std::{
    ops::ControlFlow,
    sync::atomic::{AtomicU64, Ordering},
    thread,
};

use crate::{
    board::{
        evaluation::{
            self, is_game_theoretic_score, mate_in, mated_in, tb_loss_in, tb_win_in, MATE_SCORE,
            MINIMUM_MATE_SCORE, MINIMUM_TB_WIN_SCORE,
        },
        movegen::{
            bitboards::{self, first_square},
            movepicker::{CapturePicker, MainMovePicker, MovePicker, Stage, WINNING_CAPTURE_SCORE},
            MoveListEntry, MAX_POSITION_MOVES,
        },
        Board,
    },
    cfor,
    chessmove::Move,
    definitions::{depth::Depth, depth::ONE_PLY, depth::ZERO_PLY, StackVec, INFINITY, MAX_DEPTH},
    historytable::MAX_HISTORY,
    piece::{Colour, PieceType},
    search::pv::PVariation,
    searchinfo::SearchInfo,
    tablebases::{self, probe::WDL},
    threadlocal::ThreadData,
    transpositiontable::{Bound, ProbeResult, TTHit, TTView},
    uci,
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

const ASPIRATION_WINDOW: i32 = 7;
const ASPIRATION_WINDOW_MIN_DEPTH: Depth = Depth::new(5);
const RFP_MARGIN: i32 = 70;
const RFP_IMPROVING_MARGIN: i32 = 57;
const NMP_IMPROVING_MARGIN: i32 = 76;
const SEE_QUIET_MARGIN: i32 = -59;
const SEE_TACTICAL_MARGIN: i32 = -19;
const LMP_BASE_MOVES: i32 = 2;
const FUTILITY_COEFF_0: i32 = 76;
const FUTILITY_COEFF_1: i32 = 90;
const RAZORING_COEFF_0: i32 = 394;
const RAZORING_COEFF_1: i32 = 290;
const PROBCUT_MARGIN: i32 = 200;
const PROBCUT_IMPROVING_MARGIN: i32 = 50;
const RFP_DEPTH: Depth = Depth::new(8);
const NMP_BASE_REDUCTION: Depth = Depth::new(3);
const NMP_VERIFICATION_DEPTH: Depth = Depth::new(12);
const LMP_DEPTH: Depth = Depth::new(8);
const TT_REDUCTION_DEPTH: Depth = Depth::new(4);
const FUTILITY_DEPTH: Depth = Depth::new(6);
const SINGULARITY_DEPTH: Depth = Depth::new(8);
const SEE_DEPTH: Depth = Depth::new(9);
const PROBCUT_MIN_DEPTH: Depth = Depth::new(5);
const PROBCUT_REDUCTION: Depth = Depth::new(4);
const LMR_BASE: f64 = 77.0;
const LMR_DIVISION: f64 = 236.0;

const TIME_MANAGER_UPDATE_MIN_DEPTH: Depth = Depth::new(4);

static TB_HITS: AtomicU64 = AtomicU64::new(0);

impl Board {
    /// Performs the root search. Returns the score of the position, from white's perspective, and the best move.
    #[allow(clippy::too_many_lines)]
    pub fn search_position<const USE_NNUE: bool>(
        &mut self,
        info: &mut SearchInfo,
        thread_headers: &mut [ThreadData],
        tt: TTView,
    ) -> (i32, Move) {
        set_up_for_search(self, info, thread_headers);

        let legal_moves = self.legal_moves();
        if legal_moves.is_empty() {
            return (0, Move::NULL);
        }
        if legal_moves.len() == 1 {
            info.time_manager.notify_one_legal_move();
        }

        // Probe the tablebases if we're in a TB position.
        // TODO: make this behave nicely if we're in analysis mode.
        if let Some((best_move, score)) = tablebases::probe::get_tablebase_move(self) {
            let mut pv = PVariation::default();
            pv.load_from(best_move, &PVariation::default());
            pv.score = score;
            TB_HITS.store(1, Ordering::SeqCst);
            readout_info(self, Bound::Exact, &pv, 0, info, tt, 1);
            if info.print_to_stdout {
                println!("bestmove {best_move}");
            }
            return (score, best_move);
        }

        let global_stopped = info.stopped;
        // start search threads:
        let (t1, rest) = thread_headers.split_first_mut().unwrap();
        let board_copy = self.clone();
        let info_copy = info.clone();
        let mut board_info_copies =
            rest.iter().map(|_| (board_copy.clone(), info_copy.clone())).collect::<Vec<_>>();
        let total_nodes = AtomicU64::new(0);

        thread::scope(|s| {
            let main_thread_handle = s.spawn(|| {
                self.iterative_deepening::<USE_NNUE, true>(info, tt, t1, &total_nodes);
                global_stopped.store(true, Ordering::SeqCst);
            });
            // we need to eagerly start the threads or nothing will happen
            #[allow(clippy::needless_collect)]
            let helper_handles = rest
                .iter_mut()
                .zip(board_info_copies.iter_mut())
                .map(|(t, (board, info))| {
                    s.spawn(|| {
                        board.iterative_deepening::<USE_NNUE, false>(info, tt, t, &total_nodes);
                    })
                })
                .collect::<Vec<_>>();
            main_thread_handle.join().unwrap();
            for hh in helper_handles {
                hh.join().unwrap();
            }
        });
        global_stopped.store(false, Ordering::SeqCst);

        let d_move = self.default_move(tt, t1);
        let (bestmove, score) =
            select_best(self, thread_headers, info, tt, total_nodes.load(Ordering::SeqCst), d_move);

        if info.print_to_stdout {
            println!("bestmove {bestmove}");
            #[cfg(feature = "stats")]
            info.print_stats();
            #[cfg(feature = "stats")]
            println!(
                "branching factor: {}",
                (info.nodes as f64).powf(1.0 / thread_headers[0].completed as f64)
            );
        }

        assert!(legal_moves.contains(&bestmove), "search returned an illegal move.");
        (if self.turn() == Colour::WHITE { score } else { -score }, bestmove)
    }

    /// Performs the iterative deepening search.
    /// Returns the score of the position, from the side to move's perspective, and the best move.
    /// For Lazy SMP, the main thread calls this function with `MAIN_THREAD = true`, and the helper threads with `MAIN_THREAD = false`.
    #[allow(clippy::too_many_lines)]
    fn iterative_deepening<const USE_NNUE: bool, const MAIN_THREAD: bool>(
        &mut self,
        info: &mut SearchInfo,
        tt: TTView,
        t: &mut ThreadData,
        total_nodes: &AtomicU64,
    ) {
        assert!(!MAIN_THREAD || t.thread_id == 0, "main thread must have thread_id 0");
        let d_move = self.default_move(tt, t);
        let mut aw = AspirationWindow::infinite();
        let mut pv = PVariation::default();
        let max_depth = info.time_manager.limit().depth().unwrap_or(MAX_DEPTH - 1).ply_to_horizon();
        let starting_depth = 1 + t.thread_id % 10;
        'deepening: for d in starting_depth..=max_depth {
            t.depth = d;
            if MAIN_THREAD {
                // consider stopping early if we've neatly completed a depth:
                if info.time_manager.in_game() && info.time_manager.is_past_opt_time() {
                    info.stopped.store(true, Ordering::SeqCst);
                    break 'deepening;
                }
            }
            let mut depth = Depth::new(d.try_into().unwrap());
            let min_depth = (depth / 2).max(ONE_PLY);
            // aspiration loop:
            loop {
                let nodes_before = info.nodes;
                pv.score =
                    self.root_search::<USE_NNUE>(tt, &mut pv, info, t, depth, aw.alpha, aw.beta);
                if info.check_up() {
                    break 'deepening;
                }
                let nodes = info.nodes - nodes_before;
                total_nodes.fetch_add(nodes, Ordering::SeqCst);

                if aw.alpha != -INFINITY && pv.score <= aw.alpha {
                    if MAIN_THREAD && info.print_to_stdout {
                        let total_nodes = total_nodes.load(Ordering::SeqCst);
                        readout_info(self, Bound::Upper, &pv, d, info, tt, total_nodes);
                    }
                    aw.widen_down(pv.score, depth);
                    if MAIN_THREAD {
                        info.time_manager.report_aspiration_fail(depth, Bound::Upper);
                    }
                    // search failed low, so we might have to
                    // revert a fail-high pv update
                    t.revert_best_line();
                    continue;
                }
                // search is either exact or fail-high, so we can update the best line.
                t.update_best_line(&pv);
                if aw.beta != INFINITY && pv.score >= aw.beta {
                    if MAIN_THREAD && info.print_to_stdout {
                        let total_nodes = total_nodes.load(Ordering::SeqCst);
                        readout_info(self, Bound::Lower, &pv, d, info, tt, total_nodes);
                    }
                    aw.widen_up(pv.score, depth);
                    if MAIN_THREAD {
                        info.time_manager.report_aspiration_fail(depth, Bound::Lower);
                    }
                    // decrement depth:
                    if pv.score.abs() < MINIMUM_TB_WIN_SCORE {
                        depth = (depth - 1).max(min_depth);
                    }

                    if let ControlFlow::Break(_) =
                        info.time_manager.solved_breaker::<MAIN_THREAD>(pv.line[0], 0, d)
                    {
                        info.stopped.store(true, Ordering::SeqCst);
                        break 'deepening;
                    }

                    continue;
                }

                // if we've made it here, it means we got an exact score.
                let score = pv.score;
                let bestmove = t.pvs[t.completed].moves().first().copied().unwrap_or(d_move);

                if MAIN_THREAD && info.print_to_stdout {
                    let total_nodes = total_nodes.load(Ordering::SeqCst);
                    readout_info(self, Bound::Exact, &pv, d, info, tt, total_nodes);
                }

                if let ControlFlow::Break(_) =
                    info.time_manager.solved_breaker::<MAIN_THREAD>(bestmove, pv.score, d)
                {
                    info.stopped.store(true, Ordering::SeqCst);
                    break 'deepening;
                }

                if let ControlFlow::Break(_) =
                    info.time_manager.mate_found_breaker::<MAIN_THREAD>(&pv, depth)
                {
                    info.stopped.store(true, Ordering::SeqCst);
                    break 'deepening;
                }

                if MAIN_THREAD {
                    if let Some(margin) = info.time_manager.check_for_forced_move(depth) {
                        let saved_seldepth = info.seldepth;
                        let forced =
                            self.is_forced(margin, tt, info, t, bestmove, score, (depth - 1) / 2);
                        info.seldepth = saved_seldepth;

                        if forced {
                            info.time_manager.report_forced_move(depth);
                        }
                    }
                }

                if info.stopped() {
                    break 'deepening;
                }

                break; // we got an exact score, so we can stop the aspiration loop.
            }

            if depth > ASPIRATION_WINDOW_MIN_DEPTH {
                let score = t.pvs[t.completed].score;
                aw = AspirationWindow::from_last_score(score, depth);
            } else {
                aw = AspirationWindow::infinite();
            }

            if MAIN_THREAD && depth > TIME_MANAGER_UPDATE_MIN_DEPTH {
                let bm_frac = if d > 8 {
                    let best_move = t.pvs[t.completed].line[0];
                    let best_move_subtree_size =
                        info.root_move_nodes[best_move.from().index()][best_move.to().index()];
                    let tree_size = info.nodes;
                    #[allow(clippy::cast_precision_loss)]
                    Some(best_move_subtree_size as f64 / tree_size as f64)
                } else {
                    None
                };
                info.time_manager.report_completed_depth(
                    depth,
                    pv.score,
                    t.pvs[t.completed].line[0],
                    bm_frac,
                );
            }

            if info.check_up() {
                break 'deepening;
            }
        }
    }

    /// Give a legal default move in the case where we don't have enough time to search.
    fn default_move(&mut self, tt: TTView, t: &ThreadData) -> Move {
        let tt_move = tt.probe_for_provisional_info(self.hashkey()).map_or(Move::NULL, |e| e.0);
        let mut mp =
            MovePicker::<false>::new(tt_move, self.get_killer_set(t), t.get_counter_move(self), 0);
        let mut m = Move::NULL;
        while let Some(MoveListEntry { mov, .. }) = mp.next(self, t) {
            if !self.make_move_base(mov) {
                continue;
            }
            // if we get here, it means the move is legal.
            m = mov;
            self.unmake_move_base();
            break;
        }
        m
    }

    /// Perform a tactical resolution search, searching only captures and promotions.
    pub fn quiescence<const PV: bool, const NNUE: bool>(
        &mut self,
        tt: TTView,
        pv: &mut PVariation,
        info: &mut SearchInfo,
        t: &mut ThreadData,
        mut alpha: i32,
        beta: i32,
    ) -> i32 {
        #[cfg(debug_assertions)]
        self.check_validity().unwrap();

        if info.nodes % 1024 == 0 && info.check_up() {
            return 0;
        }

        let key = self.hashkey();

        tt.prefetch(key);

        let mut lpv = PVariation::default();

        pv.length = 0;

        let height = self.height();
        info.seldepth = info.seldepth.max(Depth::from(i32::try_from(height).unwrap()));

        // check draw
        if self.is_draw() {
            return draw_score(t, info.nodes, self.turn());
        }

        let in_check = self.in_check::<{ Self::US }>();

        // are we too deep?
        if height > (MAX_DEPTH - 1).ply_to_horizon() {
            return if in_check { 0 } else { self.evaluate::<NNUE>(info, t, info.nodes) };
        }

        // probe the TT and see if we get a cutoff.
        let fifty_move_rule_near = self.fifty_move_counter() >= 80;
        let do_not_cut = PV || in_check || fifty_move_rule_near;
        let tt_hit = match tt.probe(key, height, alpha, beta, ZERO_PLY, do_not_cut) {
            ProbeResult::Cutoff(tt_hit) => return tt_hit.tt_value,
            ProbeResult::Hit(tt_hit) => Some(tt_hit),
            ProbeResult::Nothing => None,
        };

        let stand_pat = if in_check {
            -INFINITY // could be being mated!
        } else {
            self.evaluate::<NNUE>(info, t, info.nodes)
        };

        if stand_pat >= beta {
            // return stand_pat instead of beta, this is fail-soft
            return stand_pat;
        }

        let original_alpha = alpha;
        if stand_pat > alpha {
            alpha = stand_pat;
        }

        let tt_move = tt_hit.map_or(Move::NULL, |e| e.tt_move);

        let mut best_move = Move::NULL;
        let mut best_score = stand_pat;

        let mut moves_made = 0;
        let mut move_picker = CapturePicker::new(tt_move, [Move::NULL; 2], Move::NULL, -108);
        if !in_check {
            move_picker.skip_quiets = true;
        }
        while let Some(MoveListEntry { mov: m, .. }) = move_picker.next(self, t) {
            if !self.make_move::<NNUE>(m, t, info) {
                continue;
            }
            info.nodes += 1;
            moves_made += 1;

            let score = -self.quiescence::<PV, NNUE>(tt, &mut lpv, info, t, -beta, -alpha);
            self.unmake_move::<NNUE>(t, info);

            if score > best_score {
                best_score = score;
                if score > alpha {
                    best_move = m;
                    alpha = score;
                    pv.load_from(m, &lpv);
                }
                if alpha >= beta {
                    #[cfg(feature = "stats")]
                    info.log_fail_high::<true>(moves_made - 1, 0);
                    break; // fail-high
                }
            }
        }

        if moves_made == 0 && in_check {
            return -5000; // weird but works
        }

        let flag = if best_score >= beta {
            Bound::Lower
        } else if best_score > original_alpha {
            Bound::Exact
        } else {
            Bound::Upper
        };

        tt.store::<false>(key, height, best_move, best_score, flag, ZERO_PLY);

        best_score
    }

    /// Get the two killer moves for this position.
    pub const fn get_killer_set(&self, t: &ThreadData) -> [Move; 2] {
        let ply = self.height();
        t.killer_move_table[ply]
    }

    /// Perform alpha-beta minimax search.
    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    pub fn alpha_beta<const PV: bool, const ROOT: bool, const NNUE: bool>(
        &mut self,
        tt: TTView,
        pv: &mut PVariation,
        info: &mut SearchInfo,
        t: &mut ThreadData,
        mut depth: Depth,
        mut alpha: i32,
        mut beta: i32,
    ) -> i32 {
        #[cfg(debug_assertions)]
        self.check_validity().unwrap();
        if NNUE {
            debug_assert!(self.check_nnue_coherency(&t.nnue));
        } else {
            debug_assert!(self.check_hce_coherency(info));
        }

        let mut local_pv = PVariation::default();
        let l_pv = &mut local_pv;

        let key = self.hashkey();

        tt.prefetch(key);

        let in_check = self.in_check::<{ Self::US }>();
        if depth <= ZERO_PLY && !in_check {
            return self.quiescence::<PV, NNUE>(tt, pv, info, t, alpha, beta);
        }

        depth = depth.max(ZERO_PLY);

        pv.length = 0;

        if info.nodes % 1024 == 0 && info.check_up() {
            return 0;
        }

        let height = self.height();

        debug_assert_eq!(height == 0, ROOT);

        info.seldepth = if ROOT {
            ZERO_PLY
        } else {
            info.seldepth.max(Depth::from(i32::try_from(height).unwrap()))
        };

        if !ROOT {
            // check draw
            if self.is_draw() {
                return draw_score(t, info.nodes, self.turn());
            }

            // are we too deep?
            let max_height =
                MAX_DEPTH.ply_to_horizon().min(uci::GO_MATE_MAX_DEPTH.load(Ordering::SeqCst));
            if height >= max_height {
                return if in_check { 0 } else { self.evaluate::<NNUE>(info, t, info.nodes) };
            }

            // mate-distance pruning.
            alpha = alpha.max(mated_in(height));
            beta = beta.min(mate_in(height + 1));
            if alpha >= beta {
                return alpha;
            }
        }

        debug_assert_eq!(PV, alpha + 1 != beta, "PV must be true iff the alpha-beta window is larger than 1, but PV was {PV} and alpha-beta window was {alpha}-{beta}");

        let excluded = t.excluded[height];
        let fifty_move_rule_near = self.fifty_move_counter() >= 80;
        let do_not_cut = ROOT || PV || fifty_move_rule_near;
        let tt_hit = if excluded.is_null() {
            match tt.probe(key, height, alpha, beta, depth, do_not_cut) {
                ProbeResult::Cutoff(tt_hit) => {
                    return tt_hit.tt_value;
                }
                ProbeResult::Hit(tt_hit) => Some(tt_hit),
                ProbeResult::Nothing => {
                    // TT-reduction (IIR).
                    if PV && depth >= info.search_params.tt_reduction_depth {
                        depth -= 1;
                    }
                    None
                }
            }
        } else {
            None // do not probe the TT if we're in a singular-verification search.
        };

        // Probe the tablebases.
        let (mut syzygy_max, mut syzygy_min) = (MATE_SCORE, -MATE_SCORE);
        let cardinality = tablebases::probe::get_max_pieces_count();
        if !ROOT
            && excluded.is_null() // do not probe the tablebases if we're in a singular-verification search.
            && uci::SYZYGY_ENABLED.load(Ordering::SeqCst)
            && (depth >= Depth::new(uci::SYZYGY_PROBE_DEPTH.load(Ordering::SeqCst))
                || self.n_men() < cardinality)
            && self.n_men() <= cardinality
        {
            if let Some(wdl) = tablebases::probe::get_wdl(self) {
                TB_HITS.fetch_add(1, Ordering::Relaxed);

                let tb_value = match wdl {
                    WDL::Win => tb_win_in(height),
                    WDL::Loss => tb_loss_in(height),
                    WDL::Draw => draw_score(t, info.nodes, self.turn()),
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
                    tt.store::<false>(key, height, Move::NULL, tb_value, tb_bound, depth);
                    return tb_value;
                }

                if PV && tb_bound == Bound::Lower {
                    alpha = alpha.max(tb_value);
                    syzygy_min = tb_value;
                }

                if PV && tb_bound == Bound::Upper {
                    syzygy_max = tb_value;
                }
            }
        }

        let static_eval = if in_check {
            -INFINITY // when we're in check, it could be checkmate, so it's unsound to use evaluate().
        } else if !excluded.is_null() {
            t.evals[height] // if we're in a singular-verification search, we already have the static eval.
        } else {
            self.evaluate::<NNUE>(info, t, info.nodes) // otherwise, use the static evaluation.
        };

        t.evals[height] = static_eval;

        // "improving" is true when the current position has a better static evaluation than the one from a fullmove ago.
        // if a position is "improving", we can be more aggressive with beta-reductions (eval is too high),
        // but we should be less aggressive with alpha-reductions (eval is too low).
        // some engines gain by using improving to increase LMR, but this shouldn't work imo, given that LMR is
        // neutral with regards to the evaluation.
        let improving = !in_check && height >= 2 && static_eval >= t.evals[height - 2];

        t.double_extensions[height] = if ROOT { 0 } else { t.double_extensions[height - 1] };

        // clear out the next set of killer moves.
        t.killer_move_table[height + 1] = [Move::NULL; 2];

        // whole-node pruning techniques:
        if !ROOT && !PV && !in_check && excluded.is_null() {
            // razoring.
            // if the static eval is too low, check if qsearch can beat alpha.
            // if it can't, we can prune the node.
            if static_eval
                < alpha
                    - info.search_params.razoring_coeff_0
                    - info.search_params.razoring_coeff_1 * depth * depth
            {
                let v = self.quiescence::<false, NNUE>(tt, pv, info, t, alpha - 1, alpha);
                if v < alpha {
                    return v;
                }
            }

            // beta-pruning. (reverse futility pruning)
            // if the static eval is too high, we can prune the node.
            // this is a lot like stand_pat in quiescence search.
            if depth <= info.search_params.rfp_depth
                && static_eval - Self::rfp_margin(info, depth, improving) > beta
            {
                return static_eval;
            }

            let last_move_was_null = self.last_move_was_nullmove();

            // null-move pruning.
            // if we can give the opponent a free move while retaining
            // a score above beta, we can prune the node.
            if !last_move_was_null
                && depth >= Depth::new(3)
                && static_eval + i32::from(improving) * info.search_params.nmp_improving_margin
                    >= beta
                && !t.nmp_banned_for(self.turn())
                && self.zugzwang_unlikely()
            {
                let r = info.search_params.nmp_base_reduction
                    + depth / 3
                    + std::cmp::min((static_eval - beta) / 200, 4);
                let nm_depth = depth - r;
                self.make_nullmove();
                let mut null_score =
                    -self.zw_search::<NNUE>(tt, l_pv, info, t, nm_depth, -beta, -beta + 1);
                self.unmake_nullmove();
                if info.stopped() {
                    return 0;
                }
                if null_score >= beta {
                    // don't return game-theoretic scores:
                    if null_score >= MINIMUM_TB_WIN_SCORE {
                        null_score = beta;
                    }
                    // unconditionally cutoff if we're just too shallow.
                    if depth < info.search_params.nmp_verification_depth
                        && !is_game_theoretic_score(beta)
                    {
                        return null_score;
                    }
                    // verify that it's *actually* fine to prune,
                    // by doing a search with NMP disabled.
                    // we disallow NMP for the side to move,
                    // and if we hit the other side deeper in the tree
                    // with sufficient depth, we'll disallow it for them too.
                    t.ban_nmp_for(self.turn());
                    let veri_score =
                        self.zw_search::<NNUE>(tt, l_pv, info, t, nm_depth, beta - 1, beta);
                    t.unban_nmp_for(self.turn());
                    if veri_score >= beta {
                        return null_score;
                    }
                }
            }
        }

        let mut tt_move = tt_hit.map_or(Move::NULL, |hit| hit.tt_move);

        let see_table = [
            info.search_params.see_tactical_margin * depth.squared(),
            info.search_params.see_quiet_margin * depth.round(),
        ];

        // probcut:
        let pc_beta = std::cmp::min(
            beta + PROBCUT_MARGIN - i32::from(improving) * PROBCUT_IMPROVING_MARGIN,
            MINIMUM_TB_WIN_SCORE - 1,
        );
        // as usual, don't probcut in PV / check / singular verification / if there are GT truth scores in flight.
        // additionally, if we have a TT hit that's sufficiently deep, we skip trying probcut if the TT value indicates
        // that it's not going to be helpful.
        if !PV
            && !in_check
            && excluded.is_null()
            && depth >= PROBCUT_MIN_DEPTH
            && beta.abs() < MINIMUM_TB_WIN_SCORE
            // don't probcut if we have a tthit with value < pcbeta and depth >= depth - 3:
            && !matches!(tt_hit, Some(TTHit { tt_value: v, tt_depth: d, .. }) if v < pc_beta && d >= depth - 3)
        {
            let mut move_picker = CapturePicker::new(tt_move, [Move::NULL; 2], Move::NULL, 0);
            while let Some(MoveListEntry { mov: m, score: ordering_score }) =
                move_picker.next(self, t)
            {
                if ordering_score < WINNING_CAPTURE_SCORE {
                    break;
                }

                // skip non-tacticals from the TT:
                if m == tt_move && !self.is_tactical(m) {
                    continue;
                }

                if !self.make_move::<NNUE>(m, t, info) {
                    // illegal move
                    continue;
                }

                let mut value =
                    -self.quiescence::<false, NNUE>(tt, l_pv, info, t, -pc_beta, -pc_beta + 1);

                if value >= pc_beta {
                    let pc_depth = depth - PROBCUT_REDUCTION;
                    value = -self.zw_search::<NNUE>(
                        tt,
                        l_pv,
                        info,
                        t,
                        pc_depth,
                        -pc_beta,
                        -pc_beta + 1,
                    );
                }

                self.unmake_move::<NNUE>(t, info);

                if value >= pc_beta {
                    tt.store::<false>(key, height, m, value, Bound::Lower, depth - 3);
                    return value;
                }
            }
        }

        let original_alpha = alpha;
        let mut best_move = Move::NULL;
        let mut best_score = -INFINITY;
        let mut moves_made = 0;

        // internal iterative deepening -
        // if we didn't get a TT hit, and we're in the PV,
        // then this is going to be a costly search because
        // move ordering will be terrible. To rectify this,
        // we do a shallower search first, to get a bestmove
        // and help along the history tables.
        if PV && depth > Depth::new(3) && tt_hit.is_none() {
            let iid_depth = depth - 2;
            self.alpha_beta::<PV, ROOT, NNUE>(tt, l_pv, info, t, iid_depth, alpha, beta);
            tt_move = t.best_moves[height];
        }

        // number of quiet moves to try before we start pruning
        let lmp_threshold = info.lm_table.lmp_movecount(depth, improving);

        let killers = self.get_killer_set(t);
        let counter_move = t.get_counter_move(self);
        let mut move_picker = MainMovePicker::new(tt_move, killers, counter_move, 0);

        let mut quiets_tried = StackVec::<_, MAX_POSITION_MOVES>::from_default(Move::NULL);
        let mut tacticals_tried = StackVec::<_, MAX_POSITION_MOVES>::from_default(Move::NULL);
        while let Some(MoveListEntry { mov: m, score: ordering_score }) = move_picker.next(self, t)
        {
            debug_assert!(
                !quiets_tried.as_slice().contains(&m) && !tacticals_tried.as_slice().contains(&m)
            );
            if ROOT && uci::is_multipv() {
                // handle multi-pv
                if t.multi_pv_excluded.contains(&m) {
                    continue;
                }
            }
            if excluded == m {
                continue;
            }

            let lmr_reduction = info.lm_table.lm_reduction(depth, moves_made);
            let lmr_depth = std::cmp::max(depth - lmr_reduction, ZERO_PLY);
            let is_quiet = !self.is_tactical(m);
            let is_winning_capture = ordering_score > WINNING_CAPTURE_SCORE;

            // lmp & fp.
            #[cfg(not(feature = "datagen"))]
            if !ROOT && !PV && !in_check && best_score > -MINIMUM_TB_WIN_SCORE {
                // late move pruning
                // if we have made too many moves, we start skipping moves.
                if lmr_depth <= info.search_params.lmp_depth && moves_made >= lmp_threshold {
                    move_picker.skip_quiets = true;
                }

                // futility pruning
                // if the static eval is too low, we start skipping moves.
                let fp_margin = lmr_depth.round() * info.search_params.futility_coeff_1
                    + info.search_params.futility_coeff_0;
                if is_quiet
                    && lmr_depth < info.search_params.futility_depth
                    && static_eval + fp_margin <= alpha
                {
                    move_picker.skip_quiets = true;
                }
            }

            // static exchange evaluation pruning
            // simulate all captures flowing onto the target square, and if we come out badly, we skip the move.
            #[cfg(not(feature = "datagen"))]
            if !ROOT
                && best_score > -MINIMUM_TB_WIN_SCORE
                && depth <= info.search_params.see_depth
                && move_picker.stage > Stage::YieldGoodCaptures
                && !self.static_exchange_eval(m, see_table[usize::from(is_quiet)])
            {
                continue;
            }

            if !self.make_move::<NNUE>(m, t, info) {
                continue;
            }

            if is_quiet {
                quiets_tried.push(m);
            } else {
                tacticals_tried.push(m);
            }

            let nodes_before_search = info.nodes;
            info.nodes += 1;
            moves_made += 1;

            let maybe_singular = depth >= info.search_params.singularity_depth
                && excluded.is_null()
                && matches!(tt_hit, Some(TTHit { tt_move, tt_depth, tt_bound: Bound::Lower | Bound::Exact, .. }) if tt_move == m && tt_depth >= depth - 3);

            let extension;
            if ROOT {
                extension = ZERO_PLY;
            } else if maybe_singular {
                let Some(TTHit { tt_value, .. }) = tt_hit else { unreachable!() };
                extension = self.singularity::<PV, NNUE>(
                    tt,
                    info,
                    t,
                    m,
                    tt_value,
                    alpha,
                    beta,
                    depth,
                    &mut move_picker,
                );

                if move_picker.stage == Stage::Done {
                    // got a multi-cut bubbled up from the singularity search
                    // so we just bail out.
                    return Self::singularity_margin(tt_value, depth);
                }
            } else if self.in_check::<{ Self::US }>() {
                // self.in_check::<{ Self::US }>() determines if the opponent is in check,
                // because we have already made the move.
                extension = Depth::from(is_quiet || is_winning_capture);
            } else {
                extension = ZERO_PLY;
            }
            if extension >= ONE_PLY * 2 {
                t.double_extensions[height] += 1;
            }

            let mut score;
            if moves_made == 1 {
                // first move (presumably the PV-move)
                let new_depth = depth + extension - 1;
                score = -self.full_search::<PV, NNUE>(tt, l_pv, info, t, new_depth, -beta, -alpha);
            } else {
                // calculation of LMR stuff
                let r = if (is_quiet || !is_winning_capture)
                    && depth >= Depth::new(3)
                    && moves_made >= (2 + usize::from(PV))
                {
                    let mut r = info.lm_table.lm_reduction(depth, moves_made);
                    r += i32::from(!PV);
                    if is_quiet {
                        // ordering_score is only robustly a history score
                        // if this is a quiet move. Otherwise, it would be
                        // the MVV/LVA for a capture, plus SEE.
                        // even still, this allows killers and countermoves
                        // which will always have their reduction reduced by one,
                        // as the killer and cm scores are >> MAX_HISTORY.
                        let history = ordering_score;
                        if history > i32::from(MAX_HISTORY) / 2 {
                            r -= 1;
                        } else if history < i32::from(-MAX_HISTORY) / 2 {
                            r += 1;
                        }
                    }
                    Depth::new(r).clamp(ONE_PLY, depth - 1)
                } else {
                    ONE_PLY
                };
                // perform a zero-window search
                let new_depth = depth + extension - r;
                score = -self.zw_search::<NNUE>(tt, l_pv, info, t, new_depth, -alpha - 1, -alpha);
                // if we failed above alpha, and reduced more than one ply,
                // then we do a zero-window search at full depth.
                if score > alpha && r > ONE_PLY {
                    let new_depth = depth + extension - 1;
                    score =
                        -self.zw_search::<NNUE>(tt, l_pv, info, t, new_depth, -alpha - 1, -alpha);
                }
                // if we failed completely, then do full-window search
                if score > alpha && score < beta {
                    // this is a new best move, so it *is* PV.
                    let new_depth = depth + extension - 1;
                    score =
                        -self.full_search::<PV, NNUE>(tt, l_pv, info, t, new_depth, -beta, -alpha);
                }
            }
            self.unmake_move::<NNUE>(t, info);

            // record subtree size for TimeManager
            if ROOT && t.thread_id == 0 {
                let subtree_size = info.nodes - nodes_before_search;
                info.root_move_nodes[m.from().index()][m.to().index()] += subtree_size;
            }

            if extension >= ONE_PLY * 2 {
                t.double_extensions[height] -= 1;
            }

            if info.stopped() {
                return 0;
            }

            if score > best_score {
                best_score = score;
                if score > alpha {
                    best_move = m;
                    alpha = score;
                    pv.load_from(m, l_pv);
                }
                if alpha >= beta {
                    #[cfg(feature = "stats")]
                    info.log_fail_high::<false>(moves_made - 1, ordering_score);
                    break;
                }
            }
        }

        tt.prefetch(key);

        if moves_made == 0 {
            if !excluded.is_null() {
                return alpha;
            }
            if in_check {
                return mated_in(height);
            }
            return draw_score(t, info.nodes, self.turn());
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
            let bm_quiet = !self.is_tactical(best_move);
            if bm_quiet {
                t.insert_killer(self, best_move);
                t.insert_countermove(self, best_move);

                let moves_to_adjust = quiets_tried.as_slice();
                self.update_quiet_history(t, moves_to_adjust, best_move, depth);
            }

            // we unconditionally update the tactical history table
            // because tactical moves ought to be good in any position,
            // so it's good to decrease tactical history scores even
            // when the best move was non-tactical.
            let moves_to_adjust = tacticals_tried.as_slice();
            self.update_tactical_history(t, moves_to_adjust, best_move, depth);
        }

        if excluded.is_null() {
            tt.store::<ROOT>(key, height, best_move, best_score, flag, depth);
        }

        t.best_moves[height] = best_move;

        best_score
    }

    /// The margin for Reverse Futility Pruning.
    fn rfp_margin(info: &SearchInfo, depth: Depth, improving: bool) -> i32 {
        info.search_params.rfp_margin * depth
            - i32::from(improving) * info.search_params.rfp_improving_margin
    }

    /// Update the main, counter-move, and followup history tables.
    fn update_quiet_history(
        &mut self,
        t: &mut ThreadData,
        moves_to_adjust: &[Move],
        best_move: Move,
        depth: Depth,
    ) {
        t.update_history(self, moves_to_adjust, best_move, depth);
        t.update_countermove_history(self, moves_to_adjust, best_move, depth);
        t.update_followup_history(self, moves_to_adjust, best_move, depth);
    }

    /// Update the tactical history table.
    fn update_tactical_history(
        &mut self,
        t: &mut ThreadData,
        moves_to_adjust: &[Move],
        best_move: Move,
        depth: Depth,
    ) {
        t.update_tactical_history(self, moves_to_adjust, best_move, depth);
    }

    /// The reduced beta margin for Singular Extension.
    fn singularity_margin(tt_value: i32, depth: Depth) -> i32 {
        (tt_value - 2 * depth.round()).max(-MATE_SCORE)
    }

    /// Produce extensions when a move is singular - that is, if it is a move that is
    /// significantly better than the rest of the moves in a position.
    pub fn singularity<const PV: bool, const NNUE: bool>(
        &mut self,
        tt: TTView,
        info: &mut SearchInfo,
        t: &mut ThreadData,
        m: Move,
        tt_value: i32,
        alpha: i32,
        beta: i32,
        depth: Depth,
        mp: &mut MainMovePicker,
    ) -> Depth {
        let mut lpv = PVariation::default();
        let r_beta = Self::singularity_margin(tt_value, depth);
        let r_depth = (depth - 1) / 2;
        // undo the singular move so we can search the position that it exists in.
        self.unmake_move::<NNUE>(t, info);
        t.excluded[self.height()] = m;
        let value = self.zw_search::<NNUE>(tt, &mut lpv, info, t, r_depth, r_beta - 1, r_beta);
        t.excluded[self.height()] = Move::NULL;
        if value >= r_beta && r_beta >= beta {
            mp.stage = Stage::Done; // multicut!!
        } else {
            // re-make the singular move.
            self.make_move::<NNUE>(m, t, info);
        }

        let double_extend = !PV && value < r_beta - 15 && t.double_extensions[self.height()] <= 6;

        match () {
            _ if double_extend => ONE_PLY * 2, // double-extend if we failed low by a lot (the move is very singular)
            _ if value < r_beta => ONE_PLY,    // singular extension
            _ if tt_value >= beta => -ONE_PLY, // somewhat multi-cut-y
            _ if tt_value <= alpha => -ONE_PLY, // tt_value <= alpha is from Weiss (https://github.com/TerjeKir/weiss/compare/2a7b4ed0...effa8349/)
            _ => ZERO_PLY,                      // no extension
        }
    }

    /// Test if a move is *forced* - that is, if it is a move that is
    /// significantly better than the rest of the moves in a position,
    /// by at least `margin`. (typically ~200cp).
    pub fn is_forced(
        &mut self,
        margin: i32,
        tt: TTView,
        info: &mut SearchInfo,
        t: &mut ThreadData,
        m: Move,
        value: i32,
        depth: Depth,
    ) -> bool {
        let r_beta = (value - margin).max(-MATE_SCORE);
        let r_depth = (depth - 1) / 2;
        t.excluded[self.height()] = m;
        let pts_prev = info.print_to_stdout;
        info.print_to_stdout = false;
        let value = self.alpha_beta::<false, true, true>(
            tt,
            &mut PVariation::default(),
            info,
            t,
            r_depth,
            r_beta - 1,
            r_beta,
        );
        info.print_to_stdout = pts_prev;
        t.excluded[self.height()] = Move::NULL;
        value < r_beta
    }

    /// See if a move looks like it would initiate a winning exchange.
    /// This function simulates flowing all moves on to the target square of
    /// the given move, from least to most valuable moved piece, and returns
    /// true if the exchange comes out with a material advantage of at
    /// least `threshold`.
    pub fn static_exchange_eval(&self, m: Move, threshold: i32) -> bool {
        let from = m.from();
        let to = m.to();

        let mut next_victim =
            if m.is_promo() { m.promotion_type() } else { self.piece_at(from).piece_type() };

        let mut balance = self.estimated_see(m) - threshold;

        // if the best case fails, don't bother doing the full search.
        if balance < 0 {
            return false;
        }

        // worst case is losing the piece
        balance -= next_victim.see_value();

        // if the worst case passes, we can return true immediately.
        if balance >= 0 {
            return true;
        }

        let diag_sliders = self.pieces.all_bishops() | self.pieces.all_queens();
        let orth_sliders = self.pieces.all_rooks() | self.pieces.all_queens();

        // occupied starts with the position after the move `m` is made.
        let mut occupied = (self.pieces.occupied() ^ from.bitboard()) | to.bitboard();
        if m.is_ep() {
            occupied ^= self.ep_sq().bitboard();
        }

        let mut attackers = self.pieces.all_attackers_to_sq(to, occupied) & occupied;

        // after the move, it's the opponent's turn.
        let mut colour = self.turn().flip();

        loop {
            let my_attackers = attackers & self.pieces.occupied_co(colour);
            if my_attackers == 0 {
                break;
            }

            // find cheapest attacker
            for victim in PieceType::all() {
                next_victim = victim;
                if my_attackers & self.pieces.of_type(victim) != 0 {
                    break;
                }
            }

            occupied ^= first_square(my_attackers & self.pieces.of_type(next_victim)).bitboard();

            // diagonal moves reveal bishops and queens:
            if next_victim == PieceType::PAWN
                || next_victim == PieceType::BISHOP
                || next_victim == PieceType::QUEEN
            {
                attackers |= bitboards::attacks::<{ PieceType::BISHOP.inner() }>(to, occupied)
                    & diag_sliders;
            }

            // orthogonal moves reveal rooks and queens:
            if next_victim == PieceType::ROOK || next_victim == PieceType::QUEEN {
                attackers |=
                    bitboards::attacks::<{ PieceType::ROOK.inner() }>(to, occupied) & orth_sliders;
            }

            attackers &= occupied;

            colour = colour.flip();

            balance = -balance - 1 - next_victim.see_value();

            if balance >= 0 {
                // from Ethereal:
                // As a slight optimisation for move legality checking, if our last attacking
                // piece is a king, and our opponent still has attackers, then we've
                // lost as the move we followed would be illegal
                if next_victim == PieceType::KING
                    && attackers & self.pieces.occupied_co(colour) != 0
                {
                    colour = colour.flip();
                }
                break;
            }
        }

        // the side that is to move after loop exit is the loser.
        self.turn() != colour
    }

    /// root alpha-beta search.
    pub fn root_search<const NNUE: bool>(
        &mut self,
        tt: TTView,
        pv: &mut PVariation,
        info: &mut SearchInfo,
        t: &mut ThreadData,
        depth: Depth,
        alpha: i32,
        beta: i32,
    ) -> i32 {
        self.alpha_beta::<true, true, NNUE>(tt, pv, info, t, depth, alpha, beta)
    }

    /// zero-window alpha-beta search.
    pub fn zw_search<const NNUE: bool>(
        &mut self,
        tt: TTView,
        pv: &mut PVariation,
        info: &mut SearchInfo,
        t: &mut ThreadData,
        depth: Depth,
        alpha: i32,
        beta: i32,
    ) -> i32 {
        self.alpha_beta::<false, false, NNUE>(tt, pv, info, t, depth, alpha, beta)
    }

    /// full-window alpha-beta search.
    pub fn full_search<const PV: bool, const NNUE: bool>(
        &mut self,
        tt: TTView,
        pv: &mut PVariation,
        info: &mut SearchInfo,
        t: &mut ThreadData,
        depth: Depth,
        alpha: i32,
        beta: i32,
    ) -> i32 {
        self.alpha_beta::<PV, false, NNUE>(tt, pv, info, t, depth, alpha, beta)
    }
}

pub fn select_best(
    board: &mut Board,
    thread_headers: &[ThreadData],
    info: &SearchInfo,
    tt: TTView,
    total_nodes: u64,
    default_move: Move,
) -> (Move, i32) {
    let (mut best_thread, rest) = thread_headers.split_first().unwrap();

    if info.time_manager.is_test_suite() {
        // we break early focusing only on the main thread.
        return (best_thread.pv_move().unwrap_or(default_move), best_thread.pv_score());
    }

    for thread in rest {
        let best_depth = best_thread.completed;
        let best_score = best_thread.pvs[best_depth].score();
        let this_depth = thread.completed;
        let this_score = thread.pvs[this_depth].score();
        if (this_depth == best_depth || this_score > MINIMUM_MATE_SCORE) && this_score > best_score
        {
            best_thread = thread;
        }
        if this_depth > best_depth && (this_score > best_score || best_score < MINIMUM_MATE_SCORE) {
            best_thread = thread;
        }
    }

    let best_move = best_thread.pv_move().unwrap_or(default_move);
    let best_score = best_thread.pv_score();

    // if we aren't using the main thread (thread 0) then we need to do
    // an extra uci info line to show the best move/score/pv
    if best_thread.thread_id != 0 && info.print_to_stdout {
        let pv = &best_thread.pvs[best_thread.completed];
        let depth = best_thread.completed;
        readout_info(board, Bound::Exact, pv, depth, info, tt, total_nodes);
    }

    (best_move, best_score)
}

/// Print the info about an iteration of the search.
fn readout_info(
    board: &mut Board,
    mut bound: Bound,
    pv: &PVariation,
    depth: usize,
    info: &SearchInfo,
    tt: TTView,
    total_nodes: u64,
) {
    #![allow(clippy::cast_precision_loss, clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    // don't print anything if we are in the first 50ms of the search and we are in a game,
    // this helps in ultra-fast time controls where we only have a few ms to think.
    if info.time_manager.in_game() && info.time_manager.time_since_start().as_millis() < 50 {
        return;
    }
    let sstr = uci::format_score(pv.score);
    let normal_uci_output = !uci::PRETTY_PRINT.load(Ordering::SeqCst);
    let nps = (total_nodes as f64 / info.time_manager.elapsed().as_secs_f64()) as u64;
    if board.turn() == Colour::BLACK {
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
            "info score {sstr}{bound_string} wdl {wdl} depth {depth} seldepth {} nodes {total_nodes} time {} nps {nps} hashfull {hashfull} tbhits {tbhits} pv {pv}",
            info.seldepth.ply_to_horizon(),
            info.time_manager.elapsed().as_millis(),
            hashfull = tt.hashfull(),
            tbhits = TB_HITS.load(Ordering::SeqCst),
            wdl = uci::format_wdl(pv.score, board.ply()),
        );
    } else {
        let value = uci::pretty_format_score(pv.score, board.turn());
        let pv_string = board.pv_san(pv).unwrap();
        let endchr = if bound == Bound::Exact {
            "\n"
        } else {
            "                                                                   \r"
        };
        eprint!(
            " {depth:2}/{:<2} \u{001b}[38;5;243m{t} {knodes:8}kn\u{001b}[0m {value} ({wdl}) \u{001b}[38;5;243m{knps:5}kn/s\u{001b}[0m {pv_string}{endchr}",
            info.seldepth.ply_to_horizon(),
            t = uci::format_time(info.time_manager.elapsed().as_millis()),
            knps = nps / 1_000,
            knodes = total_nodes / 1_000,
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
    let contempt_component = if stm == t.stm_at_root { -contempt } else { contempt };

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
    pub const NULL: Self =
        Self { lm_reduction_table: [[0; 64]; 64], lmp_movecount_table: [[0; 12]; 2] };

    pub fn new(config: &SearchParams) -> Self {
        #![allow(
            clippy::cast_possible_truncation,
            clippy::cast_precision_loss,
            clippy::cast_sign_loss
        )]
        let mut out = Self::NULL;
        let (base, division) = (config.lmr_base / 100.0, config.lmr_division / 100.0);
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

    pub fn lm_reduction(&self, depth: Depth, moves_made: usize) -> i32 {
        let depth = depth.ply_to_horizon().min(63);
        let played = moves_made.min(63);
        self.lm_reduction_table[depth][played]
    }

    pub fn lmp_movecount(&self, depth: Depth, improving: bool) -> usize {
        let depth = depth.ply_to_horizon().min(11);
        self.lmp_movecount_table[usize::from(improving)][depth]
    }
}

impl Default for LMTable {
    fn default() -> Self {
        Self::new(&SearchParams::default())
    }
}

pub struct AspirationWindow {
    pub midpoint: i32,
    pub alpha: i32,
    pub beta: i32,
    pub alpha_fails: i32,
    pub beta_fails: i32,
}

pub fn asp_window(depth: Depth) -> i32 {
    (ASPIRATION_WINDOW + (50 / depth.round() - 3)).max(10)
}

impl AspirationWindow {
    pub const fn infinite() -> Self {
        Self { alpha: -INFINITY, beta: INFINITY, midpoint: 0, alpha_fails: 0, beta_fails: 0 }
    }

    pub fn from_last_score(last_score: i32, depth: Depth) -> Self {
        if is_game_theoretic_score(last_score) {
            // for mates / tbwins we expect a lot of fluctuation, so aspiration
            // windows are not useful.
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
                alpha: last_score - asp_window(depth),
                beta: last_score + asp_window(depth),
                alpha_fails: 0,
                beta_fails: 0,
            }
        }
    }

    pub fn widen_down(&mut self, value: i32, depth: Depth) {
        self.midpoint = value;
        let margin = asp_window(depth) << (self.alpha_fails + 1);
        if margin > evaluation::QUEEN_VALUE.0 {
            self.alpha = -INFINITY;
            return;
        }
        self.beta = (self.alpha + self.beta) / 2;
        self.alpha = self.midpoint - margin;
        self.alpha_fails += 1;
    }

    pub fn widen_up(&mut self, value: i32, depth: Depth) {
        self.midpoint = value;
        let margin = asp_window(depth) << (self.beta_fails + 1);
        if margin > evaluation::QUEEN_VALUE.0 {
            self.beta = INFINITY;
            return;
        }
        self.beta = self.midpoint + margin;
        self.beta_fails += 1;
    }
}

fn set_up_for_search(board: &mut Board, info: &mut SearchInfo, thread_headers: &mut [ThreadData]) {
    board.zero_height();
    info.set_up_for_search();
    for td in thread_headers.iter_mut() {
        td.set_up_for_search(board);
    }
}
