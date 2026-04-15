# Contribute to Viridithas

Contributions to Viridithas are highly appreciated!

## If you don't have technical skill

You can easily donate computing hardware that will directly contribute to the enhancement of Viridithas by connecting a worker to the [SweHosting OpenBench Instance](https://chess.swehosting.se/). This will allow you to run testing matches on your computer that allow patches to viri to be evaluated for their effectiveness.

## If you have technical skill

Improvements to Viridithas can be separated into improvements in *code quality*, *speed*, and *intelligence*.

**If you have basic experience with programming**, you could help out with *code quality* by improving the robustness of Viridithas's implementation of the Universal Chess Interface (or other components of Viridithas, like data generation and parsing), making code more robust or integrating more comprehensive error handling.

**If you have experience writing high-performance code**, you can help with *code quality* and *speed* by profiling and identifying bottlenecks in Viridithas's search algorithm, and if you have experience with SIMD you can work on the inference code for the neural network.

**If you have experience in machine learning**, you can help with *intelligence*, and you should do this by going to work on JW's [bullet](https://github.com/jw1912/bullet), which is an incredible CUDA-accelerated neural network trainer specifically designed for the sorts of network Viridithas uses. Enhancements to bullet directly flow into Viridithas, and a number of other open-source chess engines.

**If you have experience with chess programming**, then you can help with all three of *code quality*, *speed*, and *intelligence*, and are suited to directly editing and improving Viridithas's search algorithm.

## Requirements for Patches

Patches to Viridithas are of two sorts - "functional" patches, that change Viridithas's behaviour, and "non-functional" patches that do not. These are distinguished by inspection of the final output of Viridithas's `bench` subcommand, which runs search on a suite of positions and records the number of positions that search considers, summed across all searches.

Example output is as follows:
```
$ viridithas bench
Viridithas [VERSION] by Cosmo
r3k2r/2pb1ppp/2pp1q2/p7/1nP1B3/1P2P3/P2N1PPP/R2QK2R w KQkq a6 0 14      |  416484 nodes
4rrk1/2p1b1p1/p1p3q1/4p3/2P2n1p/1P1NR2P/PB3PP1/3R1QK1 b - - 2 24        |  371350 nodes
r3qbrk/6p1/2b2pPp/p3pP1Q/PpPpP2P/3P1B2/2PB3K/R5R1 w - - 16 42           |   89572 nodes
6k1/1R3p2/6p1/2Bp3p/3P2q1/P7/1P2rQ1K/5R2 b - - 4 44                     |   10498 nodes
...
rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2           |  515303 nodes
2rr2k1/1p4bp/p1q1p1p1/4Pp1n/2PB4/1PN3P1/P3Q2P/2RR2K1 w - f6 0 20        |  436665 nodes
3br1k1/p1pn3p/1p3n2/5pNq/2P1p3/1PN3PP/P2Q1PB1/4R1K1 w - - 0 23          |  302024 nodes
2r2b2/5p2/5k2/p1r1pP2/P2pB3/1P3P2/K1P3R1/7R w - - 23 93                 |   81468 nodes
13355022 nodes 1477371 nps
```

This value preceding "nodes" (13355022) is called the "bench" of this version of Viridithas, and acts as a kind of checksum for the search and evaluation behaviour. Patches that *change* bench from base to patch are called **functional**, and patches that do not are **non-functional**. Note that simple performance optimisations are considered non-functional under this paradigm.

**If you wish to make a functional change to Viridithas, you must test it via SPRT.** You can either get an account on the SweHosting OpenBench instance, or you can request for me to submit your patch on your behalf.

If you wish to make a non-functional change to Viridithas, and it is in hot code, you ought to provide a benchmark that shows it is not a significant slowdown, if it is in cold code, then you can get away with an informal argument that it will not result in pessimisation.

Finally, all final commit messages must have their final line be of the form `Bench: <BENCH>`. This is not important for intermediate commits while you are working on a patch, but is required for any commits that will serve as either BASE or PATCH in a SPRT test, so it is vital that they are included in the final commit you make before submitting a pull request.