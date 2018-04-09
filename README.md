Finding magic factors for bitboard based move generation
========================================================

Tool to find factors for [Magic bitboards](https://chessprogramming.wikispaces.com/Magic+Bitboards).

Exhaustive search
-----------------

When looking for magic factors `m` with shift `w` it is sufficient to look
in the range `2^{64 - w - lsb(r_max)} <= m < 2^{64 - lsb(r_max)}` where
`r_max` is the maximum relevant occupancy for the square [1].

Tricks for fast testing
-----------------------

1. Avoid zero filling table (like Stockfish).
2. Dynamically reorder relevant occupancies to fail earlier on average.

![Performance comparison](/fig-benchmark-rf8.png)

*Performance based on search of improved rook magics for f8.*

Usage
-----

Select `BISHOP` or `ROOK`, `SQUARE` and `EASY_SHIFT` in `Makefile`, `make` and run `./magics <chunk>`.

References
----------

1. [Disproving the existence of some magics](http://www.talkchess.com/forum/viewtopic.php?t=65187). 2017.
2. [No bishop magics with fixed shift 8](http://www.talkchess.com/forum/viewtopic.php?t=67051). 2018.
