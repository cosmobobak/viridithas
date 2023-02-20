# Mini-Wiki for Viridithas

## General Search Techniques
#### Alpha-Beta
The central algorithm used to determine the values of positions (and by extension, which moves to make) is called alpha-beta pruning. It is a version of the minimax algorithm that maintains bounds on the score to prune parts of the search tree. Alpha-Beta prunes more of the tree when better moves are tried first, so it relies on good move ordering to search efficiently.
```rust
fn alpha_beta(pos: Position, depth: i32, alpha: i32, beta: i32) -> i32 {
    if depth <= 0 {
        return pos.evaluate();
    }
    let mut best_score = -INFINITY;
    for mv in pos.legal_moves() {
        let score = -alpha_beta(pos.make_move(mv), depth - 1, -beta, -alpha);
        best_score = max(best_score, score);
        alpha = max(alpha, score);
        if alpha >= beta {
            break; // cutoff! save time by not searching the rest of the moves.
        }
    }
    best_score
}
```
#### Principal Variation Search
This is an enhancement to alpha-beta search - as we hope that the first move tried is the best one (and it often is), all that must be done for the rest of the moves is to prove that they are worse than the first one. As such, every move after the first is searched with a "zero window" (i.e. beta is brought down to a single centipawn above alpha) in order to prove that it is not the best move. If a move proves to be better, a somewhat costly re-search must be done, but this improves search efficiency on average.
```rust
fn pvs(pos: Position, depth: i32, alpha: i32, beta: i32) -> i32 {
    if depth <= 0 { return pos.evaluate(); }
    let mut best_score = -INFINITY;
    for (i, mv) in pos.legal_moves().enumerate() {
        let mut score; 
        if i == 0 {
            score = -pvs(pos.make_move(mv), depth - 1, -beta, -alpha)
        } else {
            score = -pvs(pos.make_move(mv), depth - 1, -alpha - 1, -alpha);
            if score > alpha && score < beta {
                // nullwindow failed, re-search with full window
                score = -pvs(pos.make_move(mv), depth - 1, -beta, -alpha);
            }
        };
        best_score = max(best_score, score);
        alpha = max(alpha, score);
        if alpha >= beta {
            break;
        }
    }
    best_score
}
```
#### Iterative Deepening
In order to search within a limited time, successively deeper searches are done, instead of searching directly to a specified depth. This sounds like it would cause crippling time wastage, but lower-depth searches contribute to move-ordering for deeper searches, leading counterintuitively to a speedup in time-to-depth.
```rust
fn iterative_deepening(pos: Position, depth: i32) -> i32 {
    let mut best_score = -INFINITY;
    for d in 1..depth {
        best_score = alpha_beta(pos, d, -INFINITY, INFINITY);
    }
    best_score
}
```
#### Aspiration Windows
When the iterative deepening search progresses from depth N to depth N + 1, the search is initially run with a small alpha-beta window around the value returned by the previous search. This provides the search with the ability to prune more aggressively in circumstances where the search is stable. If a window fails, the window is expanded exponentially in the direction of the failiure.
```rust
fn iterative_deepening(pos: Position, depth: i32) -> i32 {
    let mut aw = AspirationWindow::infinite();
    let mut best_score = -INFINITY;
    for d in 1..depth {
        loop {
            let value = alpha_beta(pos, d, aw.lower, aw.upper);
            if aw.alpha != -INFINITY && value <= aw.alpha {
                // fail low, expand window downwards
                aw.widen_down();
                continue;
            }
            if aw.beta != INFINITY && value >= aw.beta {
                // fail high, expand window upwards
                aw.widen_up();
                continue;
            }
            // success!
            break;
        }
    }
    best_score
}
```
#### Transposition Table
Every position searched is hashed into a 64-bit unsigned integer value, or "zobrist hash", which is used to store information about the position in a flat, fixed-size hashtable. This allows information to preserved between searches, and also vastly improves move ordering as best moves from lower depths can be re-tried first.

The code for this is fairly complex, so I won't go into it here. See the source code for more information, particularly `transposition_table.rs` for the implementation of the hashtable itself, and `search.rs` for the code that uses it.

## Move Ordering Techniques
#### MVV/LVA
The Most-Valuable-Victim / Least-Valuable-Attacker heuristic is used to order heuristically winning captures before losing ones in the search, which greatly improves the efficiency of alpha-beta pruning.
```rust
fn victim_score(piece: PieceType) -> i32 {
    i32::from(piece.inner()) * 1000
}

pub fn get_mvv_lva_score(victim: PieceType, attacker: PieceType) -> i32 {
    victim_score(victim) + 60 - victim_score(attacker) / 100
}
```
In the above code, `piece.inner()` returns the index of the piece in the standard piece order (pawn, knight, bishop, rook, queen, king), as we don't have to make any particular effort to use more normal piece-value scores, only to ensure that they are placed in the correct order. The attacker's score is divided by 100 to ensure that the value of the victim always dominates the value of the attacker, as attacking valuable pieces is much more important than using cheap pieces.
#### Killer Moves
Moves that previously proved "so good as to be untenable" (caused a "beta-cutoff") are cached in a table. The search then attempts to use these moves first, as they are more likely to be good.
```rust
fn alpha_beta(pos: Position, depth: i32, alpha: i32, beta: i32) -> i32 {
    if depth <= 0 {
        return pos.evaluate();
    }
    let mut best_score = -INFINITY;
    let mut moves = pos.legal_moves();
    moves.sort(); // this should bring killer moves to the front (after captures, but before quiet moves)
    for mv in moves {
        let score = -alpha_beta(pos.make_move(mv), depth - 1, -beta, -alpha);
        best_score = max(best_score, score);
        alpha = max(alpha, score);
        if alpha >= beta {
            if pos.is_quiet_move(mv) {
                killer_moves.insert(mv);
            }
            break;
        }
    }
    best_score
}
```
The above code omits the implementation of the killer moves array, but it is fairly simple - it is a fixed-size array of moves, indexed by ply from the root of the search. When a killer is found, it is stored into the slot corresponding to the current ply, and similarly when moves are being ordered, the killer moves corresponding to the current ply are tried first out of the quiet moves.
#### Counter Moves
Whenever a move causes a beta-cutoff, it is remembered as a counter-move to the move that was played immediately before it. These counter-moves are tried after the killer moves.

The code for this is virtually identical to the killer moves code, so I won't repeat it here.
#### History Heuristic
Every time a move of a piece to a square results in a beta-cutoff, the score of that pairing of (piece, square) is increased in a table. This allows the search to generalise across subtrees to further improve move ordering. Additionally, if a move that was *not* the first move causes a beta-cutoff, the moves that were tried before it have their history scores decreased.

The code for this is also virtually identical to the killer moves code, so I won't repeat it here.
#### TT-moves
Best moves previously found in the search are cached in the hashtable. When a position is re-searched, the TT-move is tried before all other moves, as it is extremely likely to be the best move.
```rust
fn alpha_beta(pos: Position, depth: i32, alpha: i32, beta: i32) -> i32 {
    if depth <= 0 {
        return pos.evaluate();
    }
    let tt_entry = transposition_table.get(pos.zobrist_hash());
    let mut best_score = -INFINITY;
    let mut moves = pos.legal_moves();
    // this should bring the TT-move to the front, before *all* other moves
    moves.sort(tt_entry.mv);
    for mv in moves {
        let score = -alpha_beta(pos.make_move(mv), depth - 1, -beta, -alpha);
        best_score = max(best_score, score);
        alpha = max(alpha, score);
        if alpha >= beta {
            break;
        }
    }
    best_score
}
```

## Search Reduction and Extension Techniques
### Whole-Node Pruning Techniques
All of these techniques (nullmove, beta-pruning, and razoring) are done before the moves-loop begins.
For all of these tricks, you must ensure that you are not in a root node, that you are not in a PV-node, and that you are not in check. If your program has singular extensions, you also cannot use these heuristics when performing a singular verification search.
#### Null-Move Pruning
The null-move heuristic uses the concept of "giving a side two moves in a row" to prune certain lines of search. If giving the opponent a free move is not sufficient for them to beat beta, then we prune the search tree as our advantage is large enough not to bother searching.
```rust
if !last_move_was_nullmove() // don't let both sides to nullmove at each other
    && depth >= 3
    && static_eval >= beta
    && zugzwang_unlikely() // sufficient material is present that zugzwang is unlikely
{
    let nm_depth = depth - 3;
    make_nullmove();
    let null_score = -alpha_beta(pos, nm_depth, -beta, -beta + 1);
    unmake_nullmove();
    if null_score >= beta {
        return null_score;
    }
}
```
A verification search can be added to ensure that the program does not become blind to some zugzwang positions, but this is not necessary in most cases, and rarely improves the strength of the engine.
#### Reverse Futility Pruning / Beta Pruning
Nodes close to the leaves of the search that have a very high static evaluation score are pruned if the static evaluation beats beta by a depth-dependent margin.
```rust
if depth <= RFP_DEPTH && static_eval - RFP_MARGIN * depth > beta {
    return static_eval;
}
```
### Other Pruning Techniques
#### Late Move Reductions
It is generally assumed during search that the first moves tried are the best ones. As such, the more moves that are tried in a position, the less fruitful a deep search is likely to be. To make use of this insight, moves tried later in a position will be searched to a reduced depth. As with PVS, a re-search must be done if a reduced depth search finds a better score, but this improves search efficiency on average.

In the PVS moves loop:
```rust
let lmr_reduction = LMR_TABLE.get_reduction(depth, moves_made);
let lmr_depth = std::cmp::max(depth - lmr_reduction, 0); // for lmp later
let mut score; 
if i == 0 {
    score = -pvs(pos.make_move(mv), depth - 1, -beta, -alpha)
} else {
    let r = if (is_quiet || !is_winning_capture) && depth >= 3 && moves_made >= (2 + usize::from(PV)) {
        (lmr_reduction + i32::from(!PV)).clamp(1, depth - 1)
    } else {
        1
    };
    score = -pvs(pos.make_move(mv), depth - r, -alpha - 1, -alpha); // reduced null-window search
    if score > alpha && score < beta {
        // nullwindow failed, re-search with full window
        score = -pvs(pos.make_move(mv), depth - 1, -beta, -alpha);
    }
};
best_score = max(best_score, score);
alpha = max(alpha, score);
if alpha >= beta {
    break;
}
```
#### Futility Pruning & Late Move Pruning
These two tricks are done in the same place (the head of the moves-loop) and in the same way, so I will discuss them together.

Futility Pruning: Nodes close to the leaves of the search tree are pruned if they have a very low static evaluation score, assuming that only a few moves will not be enough to recover the position.

LMP: Essentially the same idea as LMR, but at low depth when enough moves have been tried, the rest of the moves are simply skipped.

```rust
// lmp & fp.
if !ROOT && !PV && !in_check && best_score > -MINIMUM_TB_WIN_SCORE {
    // late move pruning
    // if we have made too many moves, we start skipping moves.
    if lmr_depth <= LMP_DEPTH && moves_made >= lmp_threshold {
        move_picker.skip_quiets = true;
    }

    // futility pruning
    // if the static eval is too low, we start skipping moves.
    let fp_margin = lmr_depth.round() * FUTILITY_COEFF_1 + FUTILITY_COEFF_0;
    if is_quiet && lmr_depth < FUTILITY_DEPTH && static_eval + fp_margin <= alpha {
        move_picker.skip_quiets = true;
    }
}
```
#### Transposition Table Reductions
When a position that will be in the principal variation (the most important line of search) is entered, and no best-move is found from the transposition table, the depth is reduced by one ply. This is not done for any tactical reason, but instead because such a search is likely to take a lot of time to complete, so reducing depth is a time-saving measure to avoid search explosion. This is most likely to occur when deep search reveals that the low-depth PV was incorrect, so search must begin on a new PV that has less information saved in the hashtable for it.
```rust
match tt.probe(key, height, alpha, beta, depth, do_not_cut) {
    ProbeResult::Cutoff(s) => return s,
    ProbeResult::Hit(tt_hit) => Some(tt_hit),
    ProbeResult::Nothing => {
        // TT-reduction.
        if PV && depth >= TT_REDUCTION_DEPTH {
            depth -= 1;
        }
        None
    }
}
```
You may also see this technique called IIR (Internal Iterative Reduction) - there is nothing "iterative" about this, but it is named as such because it replaces another technique called Internal Iterative Deepening, which Viridithas also makes use of.
#### Internal Iterative Deepening
When no TT-hit is found on a PV-node, we do a shallower search before proceeding to the main search, in order to improve move ordering and speed things up.
```rust
// internal iterative deepening -
// if we didn't get a TT hit, and we're in the PV,
// then this is going to be a costly search because
// move ordering will be terrible. To rectify this,
// we do a shallower search first, to get a bestmove
// and help along the history tables.
if PV && depth > 3 && tt_hit.is_none() {
    let iid_depth = depth - 2;
    alpha_beta(pos, iid_depth, alpha, beta);
    tt_move = best_moves[ply_from_root];
}
```
#### Check Extensions
If a move is made that gives check, the depth to which that moves is searched is increased by one ply. (This may not be *exactly* one ply, as Viridithas makes use of *fractional depth*.) This often allows Viridithas to see checkmates that are several moves deeper than the nominal search depth.
#### Singular Extensions (and Multi-Cut)
If a move proves to be much better than all the alternatives in a position, the depth used to explore that move is increased by one.

Moves are candidates for singular extensions if they are the best-move from a TT probe, and the TT probe was either an exact-score bound or a lower-bound. The depth of the TT probe must also be at most 3 plies shallower than the current depth.

Multi-Cut occurs when multiple moves in the position appear to beat beta by a large margin. In this case, we just prune the whole node.

This code is essentially verbatim from Viri, so it's a bit more complex than most of the examples in this document so far:
```rust
let maybe_singular = tt_hit.map_or(false, |tt_hit| {
    !ROOT
        && depth >= SINGULARITY_DEPTH
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
        // got a multi-cut bubbled up from the singularity search
        // so we just bail out.
        return Self::singularity_margin(tt_value, depth);
    }
}
```
```rust
/// Produce extensions when a move is singular - that is, if it is a move that is
/// significantly better than the rest of the moves in a position.
pub fn singularity<const ROOT: bool, const PV: bool, const NNUE: bool>(
    &mut self, // (self is the position)
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
```
```rust
/// The reduced beta margin for Singular Extension.
fn singularity_margin(tt_value: i32, depth: Depth) -> i32 {
    (tt_value - 2 * depth.round()).max(-MATE_SCORE)
}
```
#### Quiescence Search SEE Pruning
If a capture is found in the quiescence search that would beat beta even if the capturing piece was immediately lost for nothing, then qsearch terminates early with a beta-cutoff.

Inside the moves loop:
```rust
let worst_case = self.estimated_see(m) - get_see_value(self.piece_at(m.from()).piece_type());

if !self.make_move::<NNUE>(m, t) {
    continue;
}

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
```
```rust
pub fn estimated_see(&self, m: Move) -> i32 {
    // initially take the value of the thing on the target square
    let mut value = get_see_value(self.piece_at(m.to()).piece_type());

    if m.is_promo() {
        // if it's a promo, swap a pawn for the promoted piece type
        value += get_see_value(m.promotion_type()) - get_see_value(PieceType::PAWN);
    } else if m.is_ep() {
        // for e.p. we will miss a pawn because the target square is empty
        value = get_see_value(PieceType::PAWN);
    }

    value
}
```
#### Main Search SEE Pruning
Each time a move is found in main-search, all captures possible on the to-square are tried, cheapest-piece-first, and if the exchange is losing, the move is pruned.

This is done in the moves loop, after FP and LMP.
```rust
// static exchange evaluation pruning
// simulate all captures flowing onto the target square, and if we come out badly, we skip the move.
if !ROOT
    && best_score > -MINIMUM_TB_WIN_SCORE
    && depth <= SEE_DEPTH
    && !self.static_exchange_eval(m, see_table[usize::from(is_quiet)])
{
    continue;
}
```
Before the moves loop, we initialize the see_table:
```rust
let see_table = [SEE_TAC_MARGIN * depth * depth, SEE_QUIET_MARGIN * depth];
```

## Evaluation Techniques
#### Tapered Evaluation
The evaluation function is phased between the opening and the endgame, to account for important differences in the values of certain terms between the two.
#### Piece Values
The values of the pieces broadly conform to the typical 1/3/3/5/9 scheme, but are adjusted to account for changes in relative piece value between the midgame and the endgame.
#### Piece-Square Tables
Lookup tables with heuristic values for piece locations allow the program to understand concepts like "knights are better in the centre" and "your rooks are strong on your opponent's second rank". This term is similarly phased.
#### Pawn Structure
Bonuses and maluses are applied for things like isolated, passed, and doubled pawns.
#### Bishop Pair
A small bonus is given if a side has a pair of bishops.
#### Mobility
The relative mobility of the pieces in a position can be an important factor in the evaluation. Piece mobilities are evaluated with differing weights and with some restrictions on their movement, like ruling that moves to squares that are controlled by enemy pawns are likely not useful.
#### King Safety
The number of attacks on the squares around the king are passed into a nonlinear formula that determines the value in centipawns of the king's safety or danger.
#### Threats
Attacks from pawns on pieces and from minors on majors are given a bonus in the evaluation.
#### Tempo
A small bonus is given for being the side-to-move in a position.
#### Texel Tuning
The weights of the evaluation function are tuned on Viridithas's own self-play games.