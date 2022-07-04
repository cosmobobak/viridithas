![build](https://github.com/cosmobobak/virtue/actions/workflows/rust.yml/badge.svg)

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
The Most-Valuable-Victim / Least-Valuable-Attacker heuristic is used to order winning captures before losing ones in the search, which greatly improves the efficiency of alpha-beta pruning.
### Killer Moves
Moves that previously proved "so good as to be untenable" (caused a "beta-cutoff") are cached in a table. The search then attempts to use these moves first, as they are more likely to be good.
### History Heuristic
Every time a move of a piece to a square results in a beta-cutoff, the score of that pairing of (piece, square) is increased in a table. This allows the search to generalise across subtrees to further improve move ordering.
### Null-Move Pruning
The null-move heuristic uses the concept of "giving a side two moves in a row" to quickly estimate if a position is very good or very bad.
### Futility Pruning
Nodes close to the leaves of the search tree are pruned if they have a very low static evaluation score, assuming that only a few moves will not be enough to recover the position.
### Principal Variation Search
As we assume that the first move made is the best one, all that must be done for the rest of the moves is to prove that they are worse than the first one. As such, every move after the first is searched with a "zero window" (i.e. alpha = beta) to prove that it is not the best move. If a move proves to be better, a costly re-search must be done, but this improves search efficiency on average.
### Late Move Reductions
It is generally assumed during search that the first moves tried are the best ones. As such, the more moves that are tried in a position, the less fruitful a deep search is likely to be. To make use of this insight, moves tried later in a position will be searched to a reduced depth. As with PVS, a re-search must be done if a reduced depth search finds a better score, but this improves search efficiency on average.
### Transposition Table
Every position searched is hashed into a 64-bit "zobrist hash", which is used to store information about the position in a bucketless hashtable. This allows information to preserved between searches, and also vastly improves move ordering as best moves from lower depths can be re-tried first.
### Check / Promotion Extensions
If a move is made that gives check or is a promotion, the depth to which that moves is searched is increased by one ply. (This may not be *exactly* one ply, as Viridithas makes use of *fractional depth*.) This often allows Viridithas to see checkmates that are several moves deeper than the nominal search depth.
### Aspiration Windows
When the iterative deepening search progresses from depth N to depth N + 1, the search is initially run with a small alpha-beta window around the value returned by the previous search. This provides the search with the ability to prune more aggressively in circumstances where the search is stable. If a window fails, a re-search is done with an infinite window, but this does not happen often enough to be a problem.

## Evaluation
### Tapered Evaluation
The evaluation function is phased between the opening and the endgame, to account for important differences in the values of certain terms between the two.
### Piece Values
The values of the pieces broadly conform to the typical 1/3/3/5/9 scheme, but are adjusted to account for changes in relative piece value between the midgame and the endgame.
### Piece-Square Tables
Lookup tables with heuristic values for piece locations allow the program to understand concepts like "knights are better in the centre" and "your rooks are strong on your opponent's second rank". This term is similarly phased.
### Pawn Structure
Bonuses and maluses are applied for things like isolated, passed, and doubled pawns.
### Bishop Pair
A small bonus is given if a side has a pair of bishops.
### Mobility
The relative mobility of the pieces in a position can be an important factor in the evaluation. Piece mobilities are evaluated with differing weights and with some restrictions on their movement, like ruling that moves to squares that are controlled by enemy pawns are likely not useful.
