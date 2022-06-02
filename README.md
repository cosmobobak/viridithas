# Viridithas II, a UCI chess engine written in Rust.

Viridithas II is a free and open source chess engine and the successor to the original [Viridithas](https://github.com/cosmobobak/viridithas-chess).

Viridithas is not a complete chess program and requires a UCI-compatible graphical user interface (GUI) (e.g. XBoard with PolyGlot, Scid, Cute Chess, eboard, Arena, Sigma Chess, Shredder, Chess Partner or Fritz) in order to be used comfortably. Read the documentation for your GUI of choice for information about how to use Viridithas with it.

# Features

## Search
### Alpha-Beta
The central algorithm used to determine the values of positions (and by extension, which moves to make) is called alpha-beta pruning. It is a version of the minimax algorithm that maintains bounds on the score to prune parts of the search tree. Alpha-Beta prunes more of the tree when better moves are tried first, so it relies on good move ordering to search efficiently.
### Iterative Deepening
In order to search within a limited time, successively deeper searches are done, instead of searching directly to a specified depth.
### MVV/LVA
The Most-Valuable-Victim / Least-Valuable-Attacker score is used to order winning captures before losing ones in the search, which greatly improves the efficiency of alpha-beta pruning.
### Killer Moves
Moves that previously proved "so good as to be untenable" (caused a "beta-cutoff") are cached in a table. The search then attempts to use these moves first, as they are more likely to be good.
### History Heuristic
Every time a move of a piece to a square results in an improvement in score (it "raises alpha"), the score of that pairing of (piece, square) is increased in a table. This allows the search to generalise across subtrees to further improve move ordering.
### Null-Move Pruning
The null-move heuristic uses the concept of "giving a side two moves in a row" to quickly estimate if a position is very good or very bad.
### Futility Pruning
Nodes close to the leaves of the search tree are pruned if they have a very low score, assuming that only a few moves will not be enough to recover the position.
### Principal Variation Search
As we assume that the first move made is the best one, all the must be done for the rest of the moves is to prove that they are worse than the first one. As such, every move after the first is searched with a "zero window" (i.e. alpha = beta) to prove that it is not the best move. If a move proves to be better, a costly re-search must be done, but this improves search efficiency on average.
### Late Move Reductions
It is generally assumed during search that the first moves tried are the best ones. As such, the more moves that are tried in a position, the less fruitful it is to search deeply. To make use of this insight, moves tried later in a position will be searched to a reduced depth. As with PVS, a re-search must be done if a reduced depth search finds a better score, but this improves search efficiency on average.
### Transposition Table
Every position searched is hashed into a 64-bit "zobrist hash", which is used to store information about the position in a bucketless hashtable. This allows information to preserved between searches, and also vastly improves move ordering as best moves from lower depths can be re-tried first.
### Check / Promotion Extensions
If a move is made that gives check or is a promotion, the depth to which that moves is searched is increased by one ply. (If the move is both a promotion and check, the depth is increased by two plies.) This often allows Viridithas to see checkmates that are several moves deeper than the nominal search depth.
### Aspiration Windows
When the iterative deepening search progresses from depth N to depth N + 1, the search is initially run with a small alpha-beta window around the value returned by the previous search. This provides the search with the ability to prune more aggressively in circumstances where the search is stable. If a window fails, a re-search is done with an infinite window, but this does not happen often enough to be a problem.
### Razoring
At pre-frontier nodes (nodes that we intend to search to depth 2), we check if the static evaluation of the position is sufficiently below alpha. If it is, we only search to depth 1. This is probably the most dangerous pruning heuristic in the search, but it is generally reported to improve playing strength.

## Evaluation
### Tapered Evaluation
The evaluation function is phased between the opening and the endgame, to account for important differences in the values of certain terms between the two.
### Piece Values
The values of the pieces broadly conform to the typical 1/3/3/5/9 scheme, but are adjusted to account for changes in relative piece value between the midgame and the endgame.
### Piece-Square Tables
Lookup tables with heuristic values for piece locations allow the program to understand concepts like "knights are better in the centre" and "your rooks are strong on your opponent's second rank". This term is similarly phased.
### Pawn Structure
Bonuses and maluses are applied for things like isolated, passed, and doubled pawns.
### Pawn Shield
A small bonus is given when the king has pawns in front of it, as a heuristic proxy for king safety. This term decreases in absolute value toward the endgame.
### Bishop Pair
A small bonus is given if a side has a pair of bishops.
### Mobility
The relative number of legal moves in a position is the "mobility" of the position. This is determined by generating moves from both sides (irrespective of whose turn it actually is in the position). At the moment, generating moves is too computationally costly, so while it does improve evaluation slightly, these gains are offset by shallower search.