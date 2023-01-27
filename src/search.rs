#![allow(clippy::too_many_arguments)]

pub mod parameters;

use std::{
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::Duration, fmt::Display, ops::ControlFlow,
};

use crate::{
    board::{
        evaluation::{
            self, get_see_value, is_mate_score, mate_in, mated_in, MATE_SCORE, MINIMUM_TB_WIN_SCORE, tb_win_in, tb_loss_in, is_game_theoretic_score, MINIMUM_MATE_SCORE,
        },
        movegen::{
            bitboards::{self, first_square},
            movepicker::{CapturePicker, MainMovePicker, Stage, WINNING_CAPTURE_SCORE, MovePicker},
            MoveListEntry, MAX_POSITION_MOVES,
        },
        Board,
    },
    cfor,
    chessmove::Move,
    definitions::{
        depth::Depth, depth::ONE_PLY, depth::ZERO_PLY, StaticVec, INFINITY, MAX_DEPTH, MAX_PLY,
    },
    search::parameters::{get_lm_table, get_search_params},
    searchinfo::SearchInfo,
    threadlocal::ThreadData,
    transpositiontable::{Bound, ProbeResult, TTView},
    uci::{self, PRETTY_PRINT}, piece::{PieceType, Colour}, tablebases::{self, probe::WDL},
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

const ASPIRATION_WINDOW: i32 = 20;
const ASPIRATION_WINDOW_MIN_DEPTH: Depth = Depth::new(5);
const RAZORING_MARGIN: i32 = 300;
const RFP_MARGIN: i32 = 70;
const RFP_IMPROVING_MARGIN: i32 = 57;
const NMP_IMPROVING_MARGIN: i32 = 76;
const SEE_QUIET_MARGIN: i32 = -59;
const SEE_TACTICAL_MARGIN: i32 = -19;
const LMP_BASE_MOVES: i32 = 2;
const FUTILITY_COEFF_1: i32 = 90;
const FUTILITY_COEFF_0: i32 = 76;
const RFP_DEPTH: Depth = Depth::new(8);
const NMP_BASE_REDUCTION: Depth = Depth::new(3);
const NMP_VERIFICATION_DEPTH: Depth = Depth::new(12);
const LMP_DEPTH: Depth = Depth::new(8);
const TT_REDUCTION_DEPTH: Depth = Depth::new(4);
const FUTILITY_DEPTH: Depth = Depth::new(6);
const SINGULARITY_DEPTH: Depth = Depth::new(8);
const SEE_DEPTH: Depth = Depth::new(9);
const LMR_BASE: f64 = 77.0;
const LMR_DIVISION: f64 = 236.0;
const SEARCH_TIME_FRACTION: u64 = 26;

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
        TB_HITS.store(0, Ordering::SeqCst);
        info.setup_for_search();
        for td in thread_headers.iter_mut() {
            td.setup_tables_for_search();
        }

        let legal_moves = self.legal_moves();
        if legal_moves.is_empty() {
            return (0, Move::NULL);
        }
        if info.in_game() && legal_moves.len() == 1 {
            info.set_time_window(0);
        }

        // Probe the tablebases if we're in a TB position.
        if let Some((best_move, score)) = tablebases::probe::get_tablebase_move(self) {
            let mut pv = PVariation::default();
            pv.load_from(best_move, &PVariation::default());
            pv.score = score;
            TB_HITS.store(1, Ordering::SeqCst);
            self.readout_info(Bound::Exact, &pv, 0, info, tt, 1);
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
                self.iterative_deepening::<USE_NNUE, true>(
                    info,
                    tt,
                    t1,
                    &total_nodes,
                );
                global_stopped.store(true, Ordering::SeqCst);
            });
            // we need to eagerly start the threads or nothing will happen
            #[allow(clippy::needless_collect)]
            let helper_handles = rest
                .iter_mut()
                .zip(board_info_copies.iter_mut())
                .map(|(t, (board, info))| {
                    s.spawn(|| {
                        board.iterative_deepening::<USE_NNUE, false>(
                            info,
                            tt,
                            t,
                            &total_nodes,
                        );
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
        let (bestmove, score) = self.select_best(thread_headers, info, tt, total_nodes.load(Ordering::SeqCst), d_move);

        if info.print_to_stdout {
            println!("bestmove {bestmove}");
        }

        assert!(legal_moves.contains(&bestmove), "search returned an illegal move.");
        (if self.turn() == Colour::WHITE { score } else { -score }, bestmove)
    }

    /// Performs the iterative deepening search.
    /// Returns the score of the position, from the side to move's perspective, and the best move.
    /// For Lazy SMP, the main thread calls this function with `MAIN_THREAD = true`, and the helper threads with `MAIN_THREAD = false`.
    fn iterative_deepening<const USE_NNUE: bool, const MAIN_THREAD: bool>(
        &mut self,
        info: &mut SearchInfo,
        tt: TTView,
        t: &mut ThreadData,
        total_nodes: &AtomicU64,
    ) {
        let d_move = self.default_move(tt, t);
        let mut aw = AspirationWindow::infinite();
        let mut mate_counter = 0;
        let mut forcing_time_reduction = false;
        let mut fail_increment = false;
        let mut pv = PVariation::default();
        let max_depth = info.limit.depth().unwrap_or(MAX_DEPTH - 1).ply_to_horizon();
        let starting_depth = 1 + t.thread_id % 10;
        'deepening: for d in starting_depth..=max_depth {
            t.depth = d;
            // consider stopping early if we've neatly completed a depth:
            if MAIN_THREAD && d > 8 && info.in_game() && info.is_past_opt_time() {
                break 'deepening;
            }
            let depth = Depth::new(d.try_into().unwrap());
            // aspiration loop:
            loop {
                let nodes_before = info.nodes;
                pv.score = self.root_search::<USE_NNUE>(tt, &mut pv, info, t, depth, aw.alpha, aw.beta);
                if info.check_up() {
                    break 'deepening;
                }
                let nodes = info.nodes - nodes_before;
                total_nodes.fetch_add(nodes, Ordering::SeqCst);

                if aw.alpha != -INFINITY && pv.score <= aw.alpha {
                    if MAIN_THREAD && info.print_to_stdout {
                        let total_nodes = total_nodes.load(Ordering::SeqCst);
                        self.readout_info(Bound::Upper, &pv, d, info, tt, total_nodes);
                    }
                    aw.widen_down();
                    if MAIN_THREAD && !fail_increment && info.in_game() {
                        fail_increment = true;
                        info.multiply_time_window(1.5);
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
                        self.readout_info(Bound::Lower, &pv, d, info, tt, total_nodes);
                    }
                    aw.widen_up();
                    continue;
                }

                let score = pv.score;
                let bestmove = t.pvs[t.completed].moves().first().copied().unwrap_or(d_move);

                if MAIN_THREAD && info.print_to_stdout {
                    let total_nodes = total_nodes.load(Ordering::SeqCst);
                    self.readout_info(Bound::Exact, &pv, d, info, tt, total_nodes);
                }
                if MAIN_THREAD && d > 2 {
                    info.check_if_search_condition_met(bestmove, pv.score, d);
                }
                
                // if we've made it here, it means we got an exact score.
                if let ControlFlow::Break(_) = mate_found_breaker::<MAIN_THREAD>(&pv, d, &mut mate_counter, info) {
                    break 'deepening;
                }

                if let ControlFlow::Break(_) = self.forced_move_breaker::<MAIN_THREAD>(d, &mut forcing_time_reduction, info, tt, t, bestmove, score, depth) {
                    break 'deepening;
                }

                break; // we got an exact score, so we can stop the aspiration loop.
            }

            if depth > ASPIRATION_WINDOW_MIN_DEPTH {
                let score = t.pvs[t.completed].score;
                aw = AspirationWindow::from_last_score(score);
            } else {
                aw = AspirationWindow::infinite();
            }
        }
    }

    fn default_move(&mut self, tt: TTView, t: &ThreadData) -> Move {
        let (tt_move, _) = tt.probe_for_provisional_info(self.hashkey()).unwrap_or((Move::NULL, 0));
        let mut mp = 
            MovePicker::<false, true, true>::new(tt_move, self.get_killer_set(t), 0);
        let mut m = Move::NULL;
        while let Some(MoveListEntry { mov, .. }) = mp.next(self, t) {
            if !self.make_move_hce(mov) {
                continue;
            }
            m = mov;
            self.unmake_move_hce();
            break;
        }
        m
    }

    fn forced_move_breaker<const MAIN_THREAD: bool>(&mut self, d: usize, forcing_time_reduction: &mut bool, info: &mut SearchInfo, tt: TTView, t: &mut ThreadData, bestmove: Move, score: i32, depth: Depth) -> ControlFlow<()> {
        if MAIN_THREAD && d > 8 && !*forcing_time_reduction && info.in_game() {
            let saved_seldepth = info.seldepth;
            let forced = self.is_forced::<200>(tt, info, t, bestmove, score, depth);
            info.seldepth = saved_seldepth;
            if forced {
                *forcing_time_reduction = true;
                info.multiply_time_window(0.25);
            }
            
            if info.check_up() && d > 1 {
                return ControlFlow::Break(());
            }
        }
        ControlFlow::Continue(())
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

        if info.nodes.trailing_zeros() >= 12 && info.check_up() {
            return 0;
        }

        let mut lpv = PVariation::default();

        pv.length = 0;

        let height = self.height();
        let key = self.hashkey();
        info.seldepth = info.seldepth.max(Depth::from(i32::try_from(height).unwrap()));

        // check draw
        if self.is_draw() {
            return draw_score(info.nodes);
        }

        let in_check = self.in_check::<{ Self::US }>();

        // are we too deep?
        if height > (MAX_DEPTH - 1).ply_to_horizon() {
            return if in_check { 0 } else { self.evaluate::<NNUE>(t, info.nodes) };
        }

        // probe the TT and see if we get a cutoff.
        let fifty_move_rule_near = self.fifty_move_counter() >= 80;
        let do_not_cut = PV || in_check || fifty_move_rule_near;
        if !do_not_cut {
            let tt_entry = tt.probe(key, height, alpha, beta, ZERO_PLY, false);
            if let ProbeResult::Cutoff(s) = tt_entry {
                return s;
            }
        }

        let stand_pat = if in_check {
            -INFINITY // could be being mated!
        } else {
            self.evaluate::<NNUE>(t, info.nodes)
        };

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

        let mut moves_made = 0;
        let mut move_picker = CapturePicker::new(Move::NULL, [Move::NULL; 3], 1);
        if !in_check {
            move_picker.skip_quiets = true;
        }
        while let Some(MoveListEntry { mov: m, .. }) = move_picker.next(self, t) {
            let worst_case =
                self.estimated_see(m) - get_see_value(self.piece_at(m.from()).piece_type());

            if !self.make_move::<NNUE>(m, t) {
                continue;
            }
            info.nodes += 1;
            moves_made += 1;

            // low-effort SEE pruning - if the worst case is enough to beat beta, just stop.
            // the worst case for a capture is that we lose the capturing piece immediately.
            // as such, worst_case = (SEE of the capture) - (value of the capturing piece).
            // we have to do this after make_move, because the move has to be legal.
            let at_least = stand_pat + worst_case;
            if at_least > beta && !is_game_theoretic_score(at_least * 2) {
                self.unmake_move::<NNUE>(t);
                pv.length = 1;
                pv.line[0] = m;
                return at_least;
            }

            let score = -self.quiescence::<PV, NNUE>(tt, &mut lpv, info, t, -beta, -alpha);
            self.unmake_move::<NNUE>(t);

            if score > best_score {
                best_score = score;
                best_move = m;
                if score > alpha {
                    alpha = score;
                    pv.load_from(best_move, &lpv);
                }
                if alpha >= beta {
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

    /// Get the two killer moves for this position, and the best killer for the position two ply ago.
    pub const fn get_killer_set(&self, t: &ThreadData) -> [Move; 3] {
        let ply = self.height();
        let curr = t.killer_move_table[ply];
        let prev = if ply > 2 { t.killer_move_table[ply - 2][0] } else { Move::NULL };
        [curr[0], curr[1], prev]
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
        beta: i32,
    ) -> i32 {
        #[cfg(debug_assertions)]
        self.check_validity().unwrap();
        debug_assert!(self.check_nnue_coherency(&t.nnue));

        let mut lpv = PVariation::default();

        let in_check = self.in_check::<{ Self::US }>();
        if depth <= ZERO_PLY && !in_check {
            return self.quiescence::<PV, NNUE>(tt, pv, info, t, alpha, beta);
        }

        pv.length = 0;

        depth = depth.max(ZERO_PLY);

        if info.nodes.trailing_zeros() >= 12 && info.check_up() {
            return 0;
        }

        let key = self.hashkey();
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
                return draw_score(info.nodes);
            }

            // are we too deep?
            let max_height = MAX_DEPTH.ply_to_horizon().min(uci::GO_MATE_MAX_DEPTH.load(Ordering::SeqCst));
            if height >= max_height {
                return if in_check { 0 } else { self.evaluate::<NNUE>(t, info.nodes) };
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

        let excluded = t.excluded[height];
        let fifty_move_rule_near = self.fifty_move_counter() >= 80;
        let do_not_cut = ROOT || PV || fifty_move_rule_near;
        let tt_hit = if excluded.is_null() {
            match tt.probe(key, height, alpha, beta, depth, do_not_cut) {
                ProbeResult::Cutoff(s) => return s,
                ProbeResult::Hit(tt_hit) => Some(tt_hit),
                ProbeResult::Nothing => {
                    // TT-reduction.
                    if PV && depth >= get_search_params().tt_reduction_depth {
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
            && uci::SYZYGY_ENABLED.load(Ordering::SeqCst) 
            && (depth >= Depth::new(uci::SYZYGY_PROBE_DEPTH.load(Ordering::SeqCst)) || self.n_men() < cardinality)
            && self.n_men() <= cardinality {
            if let Some(wdl) = tablebases::probe::get_wdl(self) {
                TB_HITS.fetch_add(1, Ordering::Relaxed);

                let tb_value = match wdl {
                    WDL::Win => tb_win_in(height),
                    WDL::Loss => tb_loss_in(height),
                    WDL::Draw => 0,
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
            INFINITY // when we're in check, it could be checkmate, so it's unsound to use evaluate().
        } else if !excluded.is_null() {
            t.evals[height] // if we're in a singular-verification search, we already have the static eval.
        } else {
            self.evaluate::<NNUE>(t, info.nodes) // otherwise, use the static evaluation.
        };

        t.evals[height] = static_eval;

        // "improving" is true when the current position has a better static evaluation than the one from a fullmove ago.
        // if a position is "improving", we can be more aggressive with beta-reductions (eval is too high),
        // but we should be less agressive with alpha-reductions (eval is too low).
        // some engines gain by using improving to increase LMR, but this shouldn't work imo, given that LMR is
        // neutral with regards to the evaluation.
        let improving = !in_check && height >= 2 && static_eval >= t.evals[height - 2];

        t.double_extensions[height] = if ROOT {
            0
        } else {
            t.double_extensions[height - 1]
        };

        // whole-node pruning techniques:
        if !ROOT && !PV && !in_check && excluded.is_null() {
            // razoring.
            // if the static eval is too low, check if qsearch can beat alpha.
            // if it can't, we can prune the node.
            if static_eval < alpha - 394 - 290 * depth * depth {
                let v = self.quiescence::<false, NNUE>(tt, pv, info, t, alpha - 1, alpha);
                if v < alpha {
                    return v;
                }
            }

            // beta-pruning. (reverse futility pruning)
            // if the static eval is too high, we can prune the node.
            // this is a lot like stand_pat in quiescence search.
            if depth <= get_search_params().rfp_depth
                && static_eval - Self::rfp_margin(depth, improving) > beta
            {
                return static_eval;
            }

            let last_move_was_null = self.last_move_was_nullmove();

            // null-move pruning.
            // if we can give the opponent a free move while retaining
            // a score above beta, we can prune the node.
            if !last_move_was_null
                && depth >= Depth::new(3)
                && static_eval + i32::from(improving) * get_search_params().nmp_improving_margin
                    >= beta
                && !t.nmp_banned_for(self.turn())
                && self.zugzwang_unlikely()
            {
                let nm_depth = depth - get_search_params().nmp_base_reduction - depth / 3;
                self.make_nullmove();
                let null_score =
                    -self.zw_search::<NNUE>(tt, &mut lpv, info, t, nm_depth, -beta, -beta + 1);
                self.unmake_nullmove();
                if info.stopped() {
                    return 0;
                }
                if null_score >= beta {
                    // unconditionally cutoff if we're just too shallow.
                    if depth < get_search_params().nmp_verification_depth && !is_game_theoretic_score(beta) {
                        return null_score;
                    }
                    // verify that it's *actually* fine to prune,
                    // by doing a search with NMP disabled.
                    // we disallow NMP for the side to move, 
                    // and if we hit the other side deeper in the tree
                    // with sufficient depth, we'll disallow it for them too.
                    t.ban_nmp_for(self.turn());
                    let veri_score = self.zw_search::<NNUE>(tt, &mut lpv, info, t, nm_depth, beta - 1, beta);
                    t.unban_nmp_for(self.turn());
                    if veri_score >= beta {
                        return null_score;
                    }
                }
            }
        }

        let original_alpha = alpha;
        let mut tt_move = tt_hit.map_or(Move::NULL, |hit| hit.tt_move);
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
            self.alpha_beta::<PV, ROOT, NNUE>(tt, &mut lpv, info, t, iid_depth, alpha, beta);
            tt_move = t.best_moves[height];
        }

        // number of quiet moves to try before we start pruning
        let lmp_threshold = get_lm_table().getp(depth, improving);

        let see_table = [
            get_search_params().see_tactical_margin * depth.squared(),
            get_search_params().see_quiet_margin * depth.round(),
        ];

        let killers = self.get_killer_set(t);

        let mut move_picker = MainMovePicker::<ROOT>::new(tt_move, killers, 0);

        let mut quiets_tried = StaticVec::<Move, MAX_POSITION_MOVES>::new_from_default(Move::NULL);
        let mut tacticals_tried =
            StaticVec::<Move, MAX_POSITION_MOVES>::new_from_default(Move::NULL);
        while let Some(MoveListEntry { mov: m, score: ordering_score }) = move_picker.next(self, t)
        {
            if ROOT && uci::is_multipv() {
                // handle multi-pv
                if t.multi_pv_excluded.contains(&m) {
                    continue;
                }
            }
            if excluded == m {
                continue;
            }

            let lmr_reduction = get_lm_table().getr(depth, moves_made);
            let lmr_depth = std::cmp::max(depth - lmr_reduction, ZERO_PLY);
            let is_quiet = !self.is_tactical(m);
            let is_winning_capture = ordering_score > WINNING_CAPTURE_SCORE;
            if is_quiet {
                quiets_tried.push(m);
            } else {
                tacticals_tried.push(m);
            }

            // lmp & fp.
            if !ROOT && !PV && !in_check && best_score > -MINIMUM_TB_WIN_SCORE {
                // late move pruning
                // if we have made too many moves, we start skipping moves.
                if lmr_depth <= get_search_params().lmp_depth && moves_made >= lmp_threshold {
                    move_picker.skip_quiets = true;
                }

                // futility pruning
                // if the static eval is too low, we start skipping moves.
                let fp_margin = lmr_depth.round() * get_search_params().futility_coeff_1
                    + get_search_params().futility_coeff_0;
                if is_quiet
                    && lmr_depth < get_search_params().futility_depth
                    && static_eval + fp_margin <= alpha
                {
                    move_picker.skip_quiets = true;
                }
            }

            // static exchange evaluation pruning
            // simulate all captures flowing onto the target square, and if we come out badly, we skip the move.
            if !ROOT
                && best_score > -MINIMUM_TB_WIN_SCORE
                && depth <= get_search_params().see_depth
                && !self.static_exchange_eval(m, see_table[usize::from(is_quiet)])
            {
                continue;
            }

            if !self.make_move::<NNUE>(m, t) {
                continue;
            }
            info.nodes += 1;
            moves_made += 1;
            if ROOT
                && t.thread_id == 0
                && info.print_to_stdout
                && info.time_since_start() > Duration::from_secs(5)
                && !PRETTY_PRINT.load(Ordering::SeqCst)
            {
                println!("info currmove {m} currmovenumber {moves_made:2} nodes {}", info.nodes);
            }

            let maybe_singular = tt_hit.map_or(false, |tt_hit| {
                !ROOT
                    && depth >= get_search_params().singularity_depth
                    && tt_hit.tt_move == m
                    && excluded.is_null()
                    && tt_hit.tt_depth >= depth - 3
                    && matches!(tt_hit.tt_bound, Bound::Lower | Bound::Exact)
            });

            let mut extension = ZERO_PLY;
            if !ROOT && maybe_singular {
                let tt_value = tt_hit.as_ref().unwrap().tt_value;
                extension = self.singularity::<ROOT, PV, NNUE>(
                    tt,
                    info,
                    t,
                    m,
                    tt_value,
                    beta,
                    depth,
                    &mut move_picker,
                );

                if move_picker.stage == Stage::Done {
                    return Self::singularity_margin(tt_value, depth);
                }
            } else if !ROOT && self.in_check::<{ Self::US }>() {
                // here in_check determines if the move gives check
                // extend checks with SEE > 0
                // only captures are automatically SEE'd in the movepicker,
                // so we need to SEE quiets here.
                let is_good_see =
                    if is_quiet { self.static_exchange_eval(m, -1) } else { is_winning_capture };
                extension = Depth::from(is_good_see);
            };
            if extension > ONE_PLY * 2 {
                t.double_extensions[height] += 1;
            }

            let mut score;
            if moves_made == 1 {
                // first move (presumably the PV-move)
                score = -self.fullwindow_search::<PV, NNUE>(
                    tt,
                    &mut lpv,
                    info,
                    t,
                    depth + extension - 1,
                    -beta,
                    -alpha,
                );
            } else {
                // calculation of LMR stuff
                let r = if (is_quiet || !is_winning_capture)
                    && depth >= Depth::new(3)
                    && moves_made >= (2 + usize::from(PV))
                {
                    let mut r = get_lm_table().getr(depth, moves_made);
                    r += i32::from(!PV);
                    Depth::new(r).clamp(ONE_PLY, depth - 1)
                } else {
                    ONE_PLY
                };
                // perform a zero-window search
                score =
                    -self.zw_search::<NNUE>(tt, &mut lpv, info, t, depth + extension - r, -alpha - 1, -alpha);
                // if we failed, then full window search
                if score > alpha && score < beta {
                    // this is a new best move, so it *is* PV.
                    score = -self.fullwindow_search::<PV, NNUE>(
                        tt,
                        &mut lpv,
                        info,
                        t,
                        depth + extension - 1,
                        -beta,
                        -alpha,
                    );
                }
            }
            self.unmake_move::<NNUE>(t);
            if extension > ONE_PLY * 2 {
                t.double_extensions[height] -= 1;
            }

            if info.stopped() {
                return 0;
            }

            if score > best_score {
                best_score = score;
                best_move = m;
                if score > alpha {
                    alpha = score;
                    pv.load_from(best_move, &lpv);
                }
                if alpha >= beta {
                    break;
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
                self.update_history_metrics::<true>(t, best_move, depth);

                // decrease the history of the quiet moves that came before the best move.
                let qs = quiets_tried.as_slice();
                let qs = &qs[..qs.len() - 1];
                for &m in qs {
                    self.update_history_metrics::<false>(t, m, depth);
                }
            }
        }

        if excluded.is_null() {
            tt.store::<ROOT>(key, height, best_move, best_score, flag, depth);
        }

        t.best_moves[height] = best_move;

        best_score
    }

    /// The margin for Reverse Futility Pruning.
    fn rfp_margin(depth: Depth, improving: bool) -> i32 {
        get_search_params().rfp_margin * depth
            - i32::from(improving) * get_search_params().rfp_improving_margin
    }

    /// Update the history and followup history tables.
    fn update_history_metrics<const IS_GOOD: bool>(
        &mut self,
        t: &mut ThreadData,
        m: Move,
        depth: Depth,
    ) {
        t.add_history::<IS_GOOD>(self, m, depth);
        t.add_followup_history::<IS_GOOD>(self, m, depth);
    }

    /// The reduced beta margin for Singular Extension.
    fn singularity_margin(tt_value: i32, depth: Depth) -> i32 {
        (tt_value - 2 * depth.round()).max(-MATE_SCORE)
    }

    /// Produce extensions when a move is singular - that is, if it is a move that is
    /// significantly better than the rest of the moves in a position.
    pub fn singularity<const ROOT: bool, const PV: bool, const NNUE: bool>(
        &mut self,
        tt: TTView,
        info: &mut SearchInfo,
        t: &mut ThreadData,
        m: Move,
        tt_value: i32,
        beta: i32,
        depth: Depth,
        mp: &mut MainMovePicker<ROOT>,
    ) -> Depth {
        let mut lpv = PVariation::default();
        let r_beta = Self::singularity_margin(tt_value, depth);
        let r_depth = (depth - 1) / 2;
        // undo the singular move so we can search the position that it exists in.
        self.unmake_move::<NNUE>(t);
        t.excluded[self.height()] = m;
        let value = self.zw_search::<NNUE>(tt, &mut lpv, info, t, r_depth, r_beta - 1, r_beta);
        t.excluded[self.height()] = Move::NULL;
        if value >= r_beta && r_beta >= beta {
            mp.stage = Stage::Done; // multicut!!
        } else {
            // re-make the singular move.
            self.make_move::<NNUE>(m, t);
        }
        let double_extend = !PV && value < r_beta - 15 && t.double_extensions[self.height()] <= 4;
        if double_extend {
            ONE_PLY * 2 // double-extend if we failed low by a lot (the move is very singular)
        } else if value < r_beta {
            ONE_PLY // singular extension
        } else if tt_value >= beta {
            -ONE_PLY // somewhat multi-cut-y
        } else {
            ZERO_PLY // no extension
        }
    }

    /// Test if a move is *forced* - that is, if it is a move that is
    /// significantly better than the rest of the moves in a position,
    /// by a margin of at least `MARGIN`. (typically ~200cp).
    pub fn is_forced<const MARGIN: i32>(
        &mut self,
        tt: TTView,
        info: &mut SearchInfo,
        t: &mut ThreadData,
        m: Move,
        value: i32,
        depth: Depth,
    ) -> bool {
        let r_beta = (value - MARGIN).max(-MATE_SCORE);
        let r_depth = (depth - 1) / 2;
        t.excluded[self.height()] = m;
        let pts_prev = info.print_to_stdout;
        info.print_to_stdout = false;
        let value = self.alpha_beta::<false, true, true>(tt, &mut PVariation::default(), info, t, r_depth, r_beta - 1, r_beta);
        info.print_to_stdout = pts_prev;
        t.excluded[self.height()] = Move::NULL;
        value < r_beta
    }

    /// See if a move looks like it would initiate a winning exchange.
    /// This function simulates flowing all moves on to the target square of
    /// the given move, from least to most valueable moved piece, and returns
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
            if next_victim == PieceType::PAWN || next_victim == PieceType::BISHOP || next_victim == PieceType::QUEEN {
                attackers |= bitboards::attacks::<{ PieceType::BISHOP.inner() }>(to, occupied) & diag_sliders;
            }

            // orthogonal moves reveal rooks and queens:
            if next_victim == PieceType::ROOK || next_victim == PieceType::QUEEN {
                attackers |= bitboards::attacks::<{ PieceType::ROOK.inner() }>(to, occupied) & orth_sliders;
            }

            attackers &= occupied;

            colour = colour.flip();

            balance = -balance - 1 - get_see_value(next_victim);

            if balance >= 0 {
                // from Ethereal:
                // As a slight optimisation for move legality checking, if our last attacking
                // piece is a king, and our opponent still has attackers, then we've
                // lost as the move we followed would be illegal
                if next_victim == PieceType::KING && attackers & self.pieces.occupied_co(colour) != 0 {
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
    pub fn fullwindow_search<const PV: bool, const NNUE: bool>(
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

    pub fn select_best(
        &mut self,
        thread_headers: &[ThreadData],
        info: &SearchInfo,
        tt: TTView,
        total_nodes: u64,
        default_move: Move,
    ) -> (Move, i32) {
        let (mut best_thread, rest) = thread_headers.split_first().unwrap();
        for thread in rest {
            let best_depth = best_thread.completed;
            let best_score = best_thread.pvs[best_depth].score();
            let this_depth = thread.completed;
            let this_score = thread.pvs[this_depth].score();
            if (this_depth == best_depth || this_score > MINIMUM_MATE_SCORE) && this_score > best_score {
                best_thread = thread;
            }
            if this_depth > best_depth && (this_score > best_score || best_score < MINIMUM_MATE_SCORE) {
                best_thread = thread;
            }
        }

        let best_move = best_thread.pvs[best_thread.completed].moves().first().copied().unwrap_or(default_move);
        let best_score = best_thread.pvs[best_thread.completed].score();

        // if we aren't using the main thread (thread 0) then we need to do
        // an extra uci info line to show the best move/score/pv
        if best_thread.thread_id != 0 {
            let pv = &best_thread.pvs[best_thread.completed];
            let depth = best_thread.completed;
            self.readout_info(Bound::Exact, pv, depth, info, tt, total_nodes);
        }

        (best_move, best_score)
    }

    /// Print the info about an iteration of the search.
    fn readout_info(
        &mut self,
        mut bound: Bound,
        pv: &PVariation,
        depth: usize,
        info: &SearchInfo,
        tt: TTView,
        total_nodes: u64,
    ) {
        #![allow(
            clippy::cast_precision_loss,
            clippy::cast_sign_loss,
            clippy::cast_possible_truncation
        )]
        let sstr = uci::format_score(pv.score);
        let normal_uci_output = !uci::PRETTY_PRINT.load(Ordering::SeqCst);
        let nps = (total_nodes as f64 / info.start_time.elapsed().as_secs_f64()) as u64;
        if self.turn() == Colour::BLACK {
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
                "info score {sstr}{bound_string} depth {depth} seldepth {} nodes {total_nodes} time {} nps {nps} hashfull {hashfull} tbhits {tbhits} pv {pv}",
                info.seldepth.ply_to_horizon(),
                info.start_time.elapsed().as_millis(),
                hashfull = tt.hashfull(),
                tbhits = TB_HITS.load(Ordering::SeqCst),
            );
        } else {
            let value = uci::pretty_format_score(pv.score, self.turn());
            let pv = self.pv_san(pv).unwrap();
            let endchr = if bound == Bound::Exact {
                "\n"
            } else {
                "                                                                   \r"
            };
            eprint!(
                " {depth:2}/{:<2} \u{001b}[38;5;243m{t} {knodes:8}kn\u{001b}[0m {value} \u{001b}[38;5;243m{knps:5}kn/s\u{001b}[0m {pv}{endchr}",
                info.seldepth.ply_to_horizon(),
                t = uci::format_time(info.start_time.elapsed().as_millis()),
                knps = nps / 1000,
                knodes = total_nodes / 1_000,
            );
        }
    }
}

fn mate_found_breaker<const MAIN_THREAD: bool>(pv: &PVariation, d: usize, mate_counter: &mut i32, info: &mut SearchInfo) -> ControlFlow<()> {
    if MAIN_THREAD && is_mate_score(pv.score) && d > 10 {
        *mate_counter += 1;
        if d > 1 && info.in_game() && *mate_counter >= 3 {
            return ControlFlow::Break(());
        }
    } else if MAIN_THREAD {
        *mate_counter = 0;
    }
    ControlFlow::Continue(())
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
    pub const NULL: Self = Self { rtable: [[0; 64]; 64], ptable: [[0; 12]; 2] };

    pub fn new(config: &SearchParams) -> Self {
        #![allow(
            clippy::cast_possible_truncation,
            clippy::cast_precision_loss,
            clippy::cast_sign_loss
        )]
        let mut out = Self { rtable: [[0; 64]; 64], ptable: [[0; 12]; 2] };
        let (base, division) = (config.lmr_base / 100.0, config.lmr_division / 100.0);
        cfor!(let mut depth = 1; depth < 64; depth += 1; {
            cfor!(let mut played = 1; played < 64; played += 1; {
                let ld = f64::ln(depth as f64);
                let lp = f64::ln(played as f64);
                out.rtable[depth][played] = (base + ld * lp / division) as i32;
            });
        });
        cfor!(let mut depth = 1; depth < 12; depth += 1; {
            out.ptable[0][depth] = (2.5 + 2.0 * depth as f64 * depth as f64 / 4.5) as usize;
            out.ptable[1][depth] = (4.0 + 4.0 * depth as f64 * depth as f64 / 4.5) as usize;
        });
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
    pub const fn infinite() -> Self {
        Self { alpha: -INFINITY, beta: INFINITY, midpoint: 0, alpha_fails: 0, beta_fails: 0 }
    }

    pub const fn from_last_score(last_score: i32) -> Self {
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

#[derive(Clone, Debug)]
pub struct PVariation {
    length: usize,
    score: i32,
    line: [Move; MAX_PLY]
}

impl Default for PVariation {
    fn default() -> Self {
        Self { length: 0, score: 0, line: [Move::NULL; MAX_PLY] }
    }
}

impl PVariation {
    pub fn moves(&self) -> &[Move] {
        &self.line[..self.length]
    }

    pub const fn score(&self) -> i32 {
        self.score
    }

    fn load_from(&mut self, m: Move, rest: &Self) {
        self.line[0] = m;
        self.line[1..=rest.length].copy_from_slice(&rest.line[..rest.length]);
        self.length = rest.length + 1;
    }
}

impl Display for PVariation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for &m in self.moves() {
            write!(f, "{m} ")?;
        }
        Ok(())
    }
}