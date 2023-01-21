Fathom
======

Fathom is a stand-alone Syzygy tablebase probing tool.  The aims of Fathom
are:

* To make it easy to integrate the Syzygy tablebases into existing chess
  engines;
* To make it easy to create stand-alone applications that use the Syzygy
  tablebases;

Fathom is compilable under either C99 or C++ and supports a variety of
platforms, including at least Windows, Linux, and MacOS.

Programming API
---------------

Fathom provides a simple API. Following are the main function calls:

* `tb_init` initializes the tablebases.
* `tb_free` releases any resources allocated by Fathom.
* `tb_probe_wdl` probes the Win-Draw-Loss (WDL) table for a given position.
* `tb_probe_root` probes the Distance-To-Zero (DTZ) table for the given
   position. It returns a recommended move, and also a list of unsigned
   integers, each one encoding a possible move and its DTZ and WDL values.
* `tb_probe_root_dtz` probes the Distance-To-Zero (DTZ) at the root position.
   It returns a score and a rank for each possible move.
* `tb_probe_root_wdl` probes the Win-Draw-Loss (WDL) at the root position.
   it returns a score and a rank for each possible move.

Fathom does not require the callee to provide any additional functionality
(e.g. move generation). A simple set of chess-related functions including move
generation is provided in file `tbchess.c`. However, chess engines can opt to
replace some of this functionality for better performance (see below).

Chess engines
-------------

Chess engines can use `tb_probe_wdl` to get the WDL value during
search.  This function is thread safe (unless TB_NO_THREADS is
set). The various "probe_root" functions are intended for probing only
at the root node and are not thread-safe.

Chess engines and other clients can modify some features of Fathom and
override some of its internal functions by configuring
`tbconfig.h`. `tbconfig.h` is included in Fathom's code with angle
brackets. This allows a client of Fathom to override tbconfig.h by
placing its own modified copy in its include path before the Fathom
source directory.

One option provided by `tbconfig.h` is to define macros that replace
some aspects of Fathom's functionality, such as calculating piece
attacks, avoiding duplication of functionality.  If doing this,
however, be careful with including typedefs or defines from your own
code into `tbconfig.h`, since these may clash with internal definitions
used by Fathom. I recommend instead interfacing to external
functions via a small module, with an interface something like this:

```
#ifndef _TB_ATTACK_INTERFACE
#define _TB_ATTACK_INTERFACE

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

extern tb_knight_attacks(unsigned square);
extern tb_king_attacks(unsigned square);
extern tb_root_attacks(unsigned square, uint64_t occ);
extern tb_bishop_attacks(unsigned square, uint64_t occ);
extern tb_queen_attacks(unsigned square, uint64_t occ);
extern tb_pawn_attacks(unsigned square, uint64_t occ);

#endif
```

You can add if wanted other function definitions such as a popcnt
function based on the chess engine's native popcnt support.

`tbconfig.h` can then reference these functions safety because the
interface depends only on types defined in standard headers. The
implementation, however, can use any types from the chess engine or
other client that are necessary. (A good optimizer with link-time
optimization will inline the implementation code even though it is not
visible in the interface).

History and Credits
-------------------

The Syzygy tablebases were created by Ronald de Man. The original version of Fathom
(https://github.com/basil00/Fathom) combined probing code from Ronald de Man, originally written for
Stockfish, with chess-related functions and other support code from Basil Falcinelli.
That codebase is no longer being maintained. This repository was originaly a fork of
that codebase, with additional modifications by Jon Dart.

However, the current Fathom code in this repository is no longer
derived directly from the probing code written for Stockfish, but
instead derives from tbprobe.c, which is a component of the Cfish
chess engine (https://github.com/syzygy1/Cfish), a Stockfish
derivative. tbprobe.c includes 7-man tablebase support. It was written
by Ronald de Man and released for unrestricted distribution and use.

This fork of Fathom replaces the Cfish board representation and move
generation code used in tbprobe.c with simpler, MIT-licensed code from the original
Fathom source by Basil. The code has been reorganized so that
`tbchess.c` contains all move generation and most chess-related typedefs
and functions, while `tbprobe.c` contains all the tablebase probing
code. The code replacement and reorganization was done by Jon Dart.

License
-------

(C) 2013-2015 Ronald de Man (original code)
(C) 2015 basil (new modifications)
(C) 2016-2020 Jon Dart (additional modifications)

This version of Fathom is released under the MIT License:

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
