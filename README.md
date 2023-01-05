<div align="center">

  ![Visualisation of Neuron 0 in the 21th-gen Viridithas NNUE](logo.gif)
  
  [![Build][build-badge]][build-link]
  [![License][license-badge]][license-link]
  [![Release][release-badge]][release-link]
  [![Commits][commits-badge]][commits-link]
  
</div>

# Viridithas, a UCI chess engine written in Rust.

Viridithas is a free and open source chess engine and the successor to the original [Viridithas](https://github.com/cosmobobak/viridithas-chess).

Viridithas is not a complete chess program and requires a UCI-compatible graphical user interface in order to be used comfortably. Read the documentation for your GUI of choice for information about how to use Viridithas with it.

# Features

## General Search Techniques
#### Alpha-Beta
The central algorithm used to determine the values of positions (and by extension, which moves to make) is called alpha-beta pruning. It is a version of the minimax algorithm that maintains bounds on the score to prune parts of the search tree. Alpha-Beta prunes more of the tree when better moves are tried first, so it relies on good move ordering to search efficiently.
#### Principal Variation Search
As we hope that the first move tried is the best one (and it often is), all that must be done for the rest of the moves is to prove that they are worse than the first one. As such, every move after the first is searched with a "zero window" (i.e. beta is brought down to a single centipawn above alpha) in order to prove that it is not the best move. If a move proves to be better, a somewhat costly re-search must be done, but this improves search efficiency on average.
#### Iterative Deepening
In order to search within a limited time, successively deeper searches are done, instead of searching directly to a specified depth. This sounds like it would cause crippling time wastage, but lower-depth searches contribute to move-ordering for deeper searches, leading counterintuitively to a speedup in time-to-depth.
#### Aspiration Windows
When the iterative deepening search progresses from depth N to depth N + 1, the search is initially run with a small alpha-beta window around the value returned by the previous search. This provides the search with the ability to prune more aggressively in circumstances where the search is stable. If a window fails, the window is expanded exponentially in the direction of the failiure.
#### Transposition Table
Every position searched is hashed into a 64-bit unsigned integer value, or "zobrist hash", which is used to store information about the position in a flat, fixed-size hashtable. This allows information to preserved between searches, and also vastly improves move ordering as best moves from lower depths can be re-tried first.

## Move Ordering Techniques
#### MVV/LVA
The Most-Valuable-Victim / Least-Valuable-Attacker heuristic is used to order heuristically winning captures before losing ones in the search, which greatly improves the efficiency of alpha-beta pruning.
#### Killer Moves
Moves that previously proved "so good as to be untenable" (caused a "beta-cutoff") are cached in a table. The search then attempts to use these moves first, as they are more likely to be good.
#### Counter Moves
Whenever a move causes a beta-cutoff, it is remembered as a counter-move to the move that was played immediately before it. These counter-moves are tried after the killer moves.
#### History Heuristic
Every time a move of a piece to a square results in a beta-cutoff, the score of that pairing of (piece, square) is increased in a table. This allows the search to generalise across subtrees to further improve move ordering. Additionally, if a move that was *not* the first move causes a beta-cutoff, the moves that were tried before it have their history scores decreased.
#### TT-moves
Best moves previously found in the search are cached in the hashtable. When a position is re-searched, the TT-move is tried before all other moves, as it is extremely likely to be the best move.
#### Subtree Size At Root
At the root of the search, the moves are ordered by the number of nodes required to search them in the previous iteration of Iterative Deepening, descending.

## Search Reduction and Extension Techniques
#### Null-Move Pruning
The null-move heuristic uses the concept of "giving a side two moves in a row" to prune certain lines of search. If giving the opponent two moves in a row is not sufficient for them to beat beta, then we prune the search tree as our advantage is large enough not to bother searching.
#### Futility Pruning
Nodes close to the leaves of the search tree are pruned if they have a very low static evaluation score, assuming that only a few moves will not be enough to recover the position.
#### Reverse Futility Pruning / Beta Pruning
Nodes close to the leaves of the search that have a very high static evaluation score are pruned if the static evaluation beats beta by a depth-dependent margin.
#### Late Move Reductions
It is generally assumed during search that the first moves tried are the best ones. As such, the more moves that are tried in a position, the less fruitful a deep search is likely to be. To make use of this insight, moves tried later in a position will be searched to a reduced depth. As with PVS, a re-search must be done if a reduced depth search finds a better score, but this improves search efficiency on average.
#### Late Move Pruning
Essentially the same idea as LMR, but at low depth when enough moves have been tried, the rest of the moves are simply skipped.
#### Transposition Table Reductions
When a position that will be in the principal variation (the most important line of search) is entered, and no best-move is found from the transposition table, the depth is reduced by one ply. This is not done for any tactical reason, but instead because such a search is likely to take a lot of time to complete, so reducing depth is a time-saving measure to avoid search explosion. This is most likely to occur when deep search reveals that the low-depth PV was incorrect, so search must begin on a new PV that has less information saved in the hashtable for it.
#### Check Extensions
If a move is made that gives check, the depth to which that moves is searched is increased by one ply. (This may not be *exactly* one ply, as Viridithas makes use of *fractional depth*.) This often allows Viridithas to see checkmates that are several moves deeper than the nominal search depth.
#### Singular Extensions
If a move proves to be much better than all the alternatives in a position, the depth used to explore that move is increased by one.
#### Quiescence Search SEE Pruning
If a capture is found in the quiescence search that would beat beta even if the capturing piece was immediately lost for nothing, then qsearch terminates early with a beta-cutoff.
#### Main Search SEE Pruning
Each time a move is found in main-search, all captures possible on the to-square are tried, cheapest-piece-first, and if the exchange is losing, the move is pruned.

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

# Evaluation Development History/Originality (HCE/NNUE)
- First evaluation was a simple piece value / psqt table approach, using values from PeSTO. (as far as I remember)
- Early in development, a local-search texel tuning module was developed, and trained on the Lichess Elite dataset, removing the PeSTO values with increased strength.
- Many new evaluation terms were added during development, and were continually re-tuned on Viridithas's self-play games.
- The first NNUE was trained on a dataset of games played by Viridithas 2.7.0, 2.6.0, and 2.5.0, all rescored with a development version of Viridithas 2.7.0 at low depth.
- Subsequent networks were trained on additional self-play games by subsequent versions of Viridithas. The 13th-generation network, and many since, include positions from the Lichess Elite dataset rescored by Viridithas.

In summary, the originality of the data for training Viridithas's NNUE is strong - it is about ~60% games played by Viridithas against himself, 35% positions from human games, and something like ~5% games played in a Viridithas 2.7.0 vs. StockNemo 5.0.0.0 test match I ran on my own hardware. Most importantly, I have never trained on the evaluation output of any engine other than Viridithas.

# Thanks and Acknowledgements
[python-chess](https://github.com/niklasf/python-chess), without which I would never have been able to gain a love for chess programming.

The [VICE](https://www.youtube.com/playlist?list=PLZ1QII7yudbc-Ky058TEaOstZHVbT-2hg) video series, which was invaluable for explaining how various chess engine concepts can be implemented.

Andrew Grant's [Ethereal](https://github.com/AndyGrant/Ethereal), the exceedingly clear code of which helped me realise several horrible flaws in Viridithas.

[Shaheryar Sohail](https://github.com/TheBlackPlague), whose (very strong) StockNemo engine evolved alongside Viridithas, and who helped me significantly with NNUE.

[weather-factory](https://github.com/dsekercioglu/weather-factory), which I used only minimally, but which is still responsible for about ~10 elo in Viridithas.

[marlinflow](https://github.com/dsekercioglu/marlinflow), which is responsible for all neural network training for Viridithas.

[build-badge]:https://img.shields.io/github/actions/workflow/status/cosmobobak/virtue/rust.yml?branch=master&logo=github&style=for-the-badge
[build-link]:https://github.com/cosmobobak/virtue/actions/workflows/rust.yml
[commits-badge]:https://img.shields.io/github/commits-since/cosmobobak/virtue/latest?style=for-the-badge
[commits-link]:https://github.com/cosmobobak/virtue/commits/master
[release-badge]:https://img.shields.io/github/v/release/cosmobobak/virtue?style=for-the-badge&label=official%20release
[release-link]:https://github.com/cosmobobak/virtue/releases/latest
[license-badge]:https://img.shields.io/github/license/cosmobobak/virtue?style=for-the-badge&label=license&color=success
[license-link]:https://github.com/cosmobobak/virtue/blob/master/LICENSE
